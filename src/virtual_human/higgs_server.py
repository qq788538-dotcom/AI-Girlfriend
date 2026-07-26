from __future__ import annotations

import argparse
import io
import json
import logging
import struct
import threading
import time
import wave
from collections.abc import Iterator
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from virtual_human.voice_profile import (
    LOCKED_TTS_MODEL,
    LOCKED_VOICE_REFERENCE_SHA256,
    verify_locked_voice_reference,
)

logger = logging.getLogger(__name__)


def _wav_bytes(pcm16: bytes, sample_rate: int = 24_000) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16)
    return output.getvalue()


def _streaming_wav_header(sample_rate: int = 24_000) -> bytes:
    """Return a PCM16 WAV header whose data length is finalized by EOF.

    The local realtime adapter only needs the format fields before consuming
    PCM. HTTP/1.0 connection close marks the end of the live stream, while the
    unknown RIFF/data sizes let Higgs start writing before the utterance length
    is known.
    """
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        0xFFFFFFFF,
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        sample_rate,
        sample_rate * 2,
        2,
        16,
        b"data",
        0xFFFFFFFF,
    )


class HiggsEngine:
    """One resident Higgs model and reference voice, serialized on Metal."""

    def __init__(
        self,
        model_path: Path,
        reference_audio: Path,
        reference_text: str,
        *,
        seed: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> None:
        import mlx.core as mx
        import numpy as np
        from mlx_audio.tts import load

        self._mx = mx
        self._np = np
        self._lock = threading.Lock()
        self.model_path = model_path
        self.reference_audio = reference_audio
        self.reference_text = reference_text
        self.seed = seed
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

        started_at = time.perf_counter()
        self._model = load(model_path, lazy=False, model_type="higgs_audio_v3")
        self.load_seconds = time.perf_counter() - started_at
        self._reference_audio_codes: Any | None = None
        encode_reference = getattr(self._model, "encode_reference_audio", None)
        if callable(encode_reference):
            reference_started_at = time.perf_counter()
            self._reference_audio_codes = encode_reference(str(reference_audio))
            logger.info(
                "Cached locked Higgs reference codes in %.3fs",
                time.perf_counter() - reference_started_at,
            )
        logger.info(
            "Higgs model loaded in %.3fs from %s",
            self.load_seconds,
            model_path,
        )

    def _request_parameters(self, request: dict[str, Any]) -> dict[str, Any]:
        text = str(request.get("input") or "").strip()
        if not text:
            raise ValueError("'input' must be a non-empty string")
        if len(text) > 300:
            raise ValueError("'input' must not exceed 300 characters")

        streaming_interval = float(request.get("streaming_interval", 0.5))
        if not 0.2 <= streaming_interval <= 2.0:
            raise ValueError("'streaming_interval' must be between 0.2 and 2.0 seconds")
        return {
            "text": text,
            "seed": int(request.get("seed", self.seed)),
            "temperature": float(request.get("temperature", self.temperature)),
            "top_p": float(request.get("top_p", self.top_p)),
            "top_k": int(request.get("top_k", self.top_k)),
            "max_tokens": int(request.get("max_tokens", 1024)),
            "streaming_interval": streaming_interval,
        }

    def _audio_pcm16(self, audio: Any) -> Any:
        audio = self._np.asarray(audio, dtype=self._np.float32).reshape(-1)
        audio = self._np.nan_to_num(audio)
        peak = float(self._np.max(self._np.abs(audio))) if audio.size else 0.0
        if peak > 1.0:
            audio = audio / peak
        return self._np.round(
            self._np.clip(audio, -1.0, 1.0) * 32767.0
        ).astype(self._np.int16)

    def _result_pcm16(self, result: Any) -> Any:
        return self._audio_pcm16(result.audio)

    def _reference_arguments(self) -> dict[str, Any]:
        if self._reference_audio_codes is not None:
            return {
                "ref_audio_codes": self._reference_audio_codes,
                "ref_text": self.reference_text,
            }
        return {
            "ref_audio": str(self.reference_audio),
            "ref_text": self.reference_text,
        }

    def synthesize(self, request: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
        parameters = self._request_parameters(request)
        with self._lock:
            self._mx.random.seed(parameters["seed"])
            started_at = time.perf_counter()
            result = next(
                self._model.generate(
                    text=parameters["text"],
                    **self._reference_arguments(),
                    temperature=parameters["temperature"],
                    top_p=parameters["top_p"],
                    top_k=parameters["top_k"],
                    max_new_tokens=parameters["max_tokens"],
                )
            )
            generation_seconds = time.perf_counter() - started_at

        pcm16 = self._result_pcm16(result)
        sample_rate = int(result.sample_rate)
        metadata = {
            "seed": parameters["seed"],
            "temperature": parameters["temperature"],
            "top_p": parameters["top_p"],
            "top_k": parameters["top_k"],
            "sample_rate": sample_rate,
            "samples": int(pcm16.size),
            "duration_seconds": (
                round(pcm16.size / sample_rate, 3) if sample_rate else 0.0
            ),
            "generation_seconds": round(generation_seconds, 3),
            "peak_memory_gb": round(float(result.peak_memory_usage), 3),
        }
        return _wav_bytes(pcm16.tobytes(), sample_rate), metadata

    def synthesize_stream(
        self,
        request: dict[str, Any],
    ) -> Iterator[tuple[bytes, dict[str, Any]]]:
        parameters = self._request_parameters(request)
        if request.get("streaming_mode") == "v3_incremental":
            return self._synthesize_v3_incremental(parameters)

        def generate_chunks() -> Iterator[tuple[bytes, dict[str, Any]]]:
            with self._lock:
                self._mx.random.seed(parameters["seed"])
                started_at = time.perf_counter()
                total_samples = 0
                for chunk_index, result in enumerate(
                    self._model.generate(
                        text=parameters["text"],
                        **self._reference_arguments(),
                        temperature=parameters["temperature"],
                        top_p=parameters["top_p"],
                        top_k=parameters["top_k"],
                        max_new_tokens=parameters["max_tokens"],
                        stream=True,
                        streaming_interval=parameters["streaming_interval"],
                        overlap_ms=40.0,
                    )
                ):
                    pcm16 = self._result_pcm16(result)
                    if not pcm16.size:
                        continue
                    total_samples += int(pcm16.size)
                    yield pcm16.tobytes(), {
                        "chunk_index": chunk_index,
                        "samples": int(pcm16.size),
                        "total_samples": total_samples,
                        "sample_rate": int(result.sample_rate),
                        "is_final": bool(getattr(result, "is_final_chunk", False)),
                        "generation_seconds": round(
                            time.perf_counter() - started_at,
                            3,
                        ),
                    }

        return generate_chunks()

    def _synthesize_v3_incremental(
        self,
        parameters: dict[str, Any],
    ) -> Iterator[tuple[bytes, dict[str, Any]]]:
        """Incrementally decode Higgs v3 with a held-tail crossfade.

        mlx-audio's Higgs v3 currently accepts ``stream=True`` but still
        decodes the entire utterance before its single yield. This mirrors the
        model's sampler loop and periodically re-decodes accumulated codec
        rows. The last 40 ms is held back and crossfaded against the next
        decode, avoiding neural-codec edge clicks without splitting text into
        separate TTS takes.
        """
        model = self._model
        required_attributes = (
            "_normalize_references",
            "_build_prompt_embeddings",
            "_audio_logits",
            "_embed_audio_codes",
            "_decode_audio",
            "backbone",
            "config",
        )
        if not all(hasattr(model, name) for name in required_attributes):
            raise ValueError("Loaded Higgs model does not support v3 incremental decoding")

        from mlx_audio.tts.models.higgs_audio_v3.generation import (
            HiggsSamplerState,
            step,
        )
        from mlx_lm.models.cache import make_prompt_cache

        def generate_chunks() -> Iterator[tuple[bytes, dict[str, Any]]]:
            with self._lock:
                self._mx.random.seed(parameters["seed"])
                started_at = time.perf_counter()
                references = model._normalize_references(
                    **self._reference_arguments()
                )
                prompt_embeds, _prompt_tokens = model._build_prompt_embeddings(
                    parameters["text"],
                    references,
                )
                self._mx.eval(prompt_embeds)

                cache = make_prompt_cache(model)
                dummy = self._mx.zeros(
                    (1, prompt_embeds.shape[1]),
                    dtype=self._mx.int32,
                )
                hidden = model.backbone(
                    dummy,
                    cache=cache,
                    input_embeddings=prompt_embeds,
                )
                last_hidden = hidden[:, -1, :]

                state = HiggsSamplerState(
                    num_codebooks=model.config.audio_num_codebooks
                )
                delayed_rows: list[Any] = []
                emit_every_frames = max(
                    1,
                    round(parameters["streaming_interval"] / 0.04),
                )
                sample_rate = int(model.sample_rate)
                overlap_samples = round(sample_rate * 0.04)
                fade_in_samples = round(sample_rate * 0.03)
                fade_out_samples = round(sample_rate * 0.015)
                overlap_tail: Any | None = None
                emitted_samples = 0
                first_decode = True
                chunk_index = 0
                total_samples = 0
                last_decoded_frame_count = 0

                def assemble(decoded: Any, *, final: bool) -> Any | None:
                    nonlocal overlap_tail, emitted_samples, first_decode
                    pcm = self._np.asarray(
                        decoded,
                        dtype=self._np.float32,
                    ).reshape(-1)
                    if not pcm.size:
                        return None
                    pcm = pcm.copy()

                    if first_decode:
                        if pcm.size > fade_in_samples:
                            pcm[:fade_in_samples] *= self._np.linspace(
                                0.0,
                                1.0,
                                fade_in_samples,
                                dtype=self._np.float32,
                            )
                        if final:
                            if pcm.size > fade_out_samples:
                                pcm[-fade_out_samples:] *= self._np.linspace(
                                    1.0,
                                    0.0,
                                    fade_out_samples,
                                    dtype=self._np.float32,
                                )
                            first_decode = False
                            emitted_samples = int(pcm.size)
                            return pcm
                        if pcm.size <= overlap_samples:
                            return None
                        first_decode = False
                        overlap_tail = pcm[-overlap_samples:].copy()
                        emitted_samples = int(pcm.size - overlap_samples)
                        return pcm[:-overlap_samples].copy()

                    if (
                        overlap_tail is None
                        or pcm.size < emitted_samples + overlap_samples
                    ):
                        return None
                    new_overlap = pcm[
                        emitted_samples : emitted_samples + overlap_samples
                    ]
                    fade_out = self._np.linspace(
                        1.0,
                        0.0,
                        overlap_samples,
                        dtype=self._np.float32,
                    )
                    fade_in = self._np.linspace(
                        0.0,
                        1.0,
                        overlap_samples,
                        dtype=self._np.float32,
                    )
                    crossfaded = overlap_tail * fade_out + new_overlap * fade_in
                    if final:
                        tail = pcm[emitted_samples + overlap_samples :].copy()
                        if tail.size > fade_out_samples:
                            tail[-fade_out_samples:] *= self._np.linspace(
                                1.0,
                                0.0,
                                fade_out_samples,
                                dtype=self._np.float32,
                            )
                        emitted_samples = int(pcm.size)
                        overlap_tail = None
                        return self._np.concatenate([crossfaded, tail])

                    middle_end = int(pcm.size - overlap_samples)
                    middle_start = emitted_samples + overlap_samples
                    middle = (
                        pcm[middle_start:middle_end].copy()
                        if middle_end > middle_start
                        else self._np.zeros((0,), dtype=self._np.float32)
                    )
                    overlap_tail = pcm[-overlap_samples:].copy()
                    emitted_samples = middle_end
                    return self._np.concatenate([crossfaded, middle])

                for _ in range(int(parameters["max_tokens"])):
                    logits = model._audio_logits(last_hidden)[0]
                    codes = step(
                        logits,
                        state,
                        temperature=float(parameters["temperature"]),
                        top_p=parameters["top_p"],
                        top_k=parameters["top_k"],
                        boc_id=model.config.audio_boc_token_id,
                        eoc_id=model.config.audio_eoc_token_id,
                    )
                    delayed_rows.append(codes)
                    final = bool(state.generation_done)

                    if (
                        final
                        or len(delayed_rows) - last_decoded_frame_count
                        >= emit_every_frames
                    ):
                        last_decoded_frame_count = len(delayed_rows)
                        decoded = model._decode_audio(delayed_rows)
                        self._mx.eval(decoded)
                        chunk = assemble(decoded, final=final)
                        if chunk is not None and chunk.size:
                            pcm16 = self._audio_pcm16(chunk)
                            total_samples += int(pcm16.size)
                            yield pcm16.tobytes(), {
                                "chunk_index": chunk_index,
                                "samples": int(pcm16.size),
                                "total_samples": total_samples,
                                "sample_rate": sample_rate,
                                "is_final": final,
                                "generation_seconds": round(
                                    time.perf_counter() - started_at,
                                    3,
                                ),
                            }
                            chunk_index += 1
                        if final:
                            return

                    next_embed = model._embed_audio_codes(codes)[None]
                    decode_dummy = self._mx.zeros((1, 1), dtype=self._mx.int32)
                    hidden = model.backbone(
                        decode_dummy,
                        cache=cache,
                        input_embeddings=next_embed,
                    )
                    last_hidden = hidden[:, -1, :]

                if last_decoded_frame_count < len(delayed_rows):
                    decoded = model._decode_audio(delayed_rows)
                    self._mx.eval(decoded)
                    chunk = assemble(decoded, final=True)
                    if chunk is not None and chunk.size:
                        pcm16 = self._audio_pcm16(chunk)
                        total_samples += int(pcm16.size)
                        yield pcm16.tobytes(), {
                            "chunk_index": chunk_index,
                            "samples": int(pcm16.size),
                            "total_samples": total_samples,
                            "sample_rate": sample_rate,
                            "is_final": True,
                            "generation_seconds": round(
                                time.perf_counter() - started_at,
                                3,
                            ),
                        }

        return generate_chunks()


class HiggsHTTPServer(ThreadingHTTPServer):
    engine: HiggsEngine


class HiggsRequestHandler(BaseHTTPRequestHandler):
    server: HiggsHTTPServer

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/healthz":
            self._send_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "model": str(self.server.engine.model_path),
                    "reference_audio": str(self.server.engine.reference_audio),
                    "voice_profile_locked": True,
                    "reference_sha256": LOCKED_VOICE_REFERENCE_SHA256,
                    "load_seconds": round(self.server.engine.load_seconds, 3),
                    "sample_rate": 24_000,
                    "resident": True,
                },
            )
            return
        if self.path.rstrip("/") == "/v1/models":
            self._send_json(
                HTTPStatus.OK,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.server.engine.model_path.name,
                            "object": "model",
                            "owned_by": "local",
                        }
                    ],
                },
            )
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/v1/audio/speech":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > 1_000_000:
                raise ValueError("Invalid Content-Length")
            payload = json.loads(self.rfile.read(content_length))
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object")
            if payload.get("stream"):
                chunks = self.server.engine.synthesize_stream(payload)
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("X-Higgs-Streaming", "overlap-add")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(_streaming_wav_header())
                self.wfile.flush()
                try:
                    for pcm16, _metadata in chunks:
                        self.wfile.write(pcm16)
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    logger.info("Higgs streaming client disconnected")
                self.close_connection = True
                return
            wav, metadata = self.server.engine.synthesize(payload)
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": {"type": "invalid_request", "message": str(error)}},
            )
            return
        except Exception as error:
            logger.exception("Higgs synthesis failed")
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": {"type": "synthesis_error", "message": str(error)}},
            )
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(wav)))
        self.send_header(
            "X-Higgs-Metadata",
            json.dumps(metadata, ensure_ascii=True, separators=(",", ":")),
        )
        self.end_headers()
        self.wfile.write(wav)

    def log_message(self, format: str, *args: object) -> None:
        logger.info("%s - %s", self.address_string(), format % args)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resident local Higgs TTS OpenAI-compatible speech server."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference-audio", type=Path, required=True)
    parser.add_argument("--reference-text", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    for path in (args.model, args.reference_audio):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.model.name != LOCKED_TTS_MODEL:
        parser.error("The production voice is locked to the approved Higgs model")
    try:
        verify_locked_voice_reference(args.reference_audio)
    except ValueError as error:
        parser.error(str(error))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    engine = HiggsEngine(
        args.model,
        args.reference_audio,
        args.reference_text,
        seed=args.seed,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    server = HiggsHTTPServer((args.host, args.port), HiggsRequestHandler)
    server.engine = engine
    logger.info("Higgs TTS listening on http://%s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
