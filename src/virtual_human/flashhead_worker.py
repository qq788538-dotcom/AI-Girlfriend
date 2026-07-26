from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import logging
import math
import os
import secrets
import threading
import wave
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from virtual_human.identity import LOCKED_AVATAR_SHA256
from virtual_human.mp4_stream import ContinuousMP4Muxer, build_video_filter

logger = logging.getLogger(__name__)

FLASHHEAD_WS_PING_TIMEOUT_SECONDS = 600


class FlashHeadSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    host: str = Field("0.0.0.0", alias="VH_FLASHHEAD_HOST")
    port: int = Field(8770, alias="VH_FLASHHEAD_PORT")
    repo_dir: Path = Field(Path("vendor/SoulX-FlashHead"), alias="VH_FLASHHEAD_REPO_DIR")
    checkpoint_dir: Path = Field(
        Path("vendor/SoulX-FlashHead/models/SoulX-FlashHead-1_3B"),
        alias="VH_FLASHHEAD_CKPT_DIR",
    )
    wav2vec_dir: Path = Field(
        Path("vendor/SoulX-FlashHead/models/wav2vec2-base-960h"),
        alias="VH_FLASHHEAD_WAV2VEC_DIR",
    )
    model_type: str = Field("lite", alias="VH_FLASHHEAD_MODEL_TYPE")
    runtime_dir: Path = Field(Path("runtime/flashhead"), alias="VH_FLASHHEAD_RUNTIME_DIR")
    public_base_url: str = Field("", alias="VH_FLASHHEAD_PUBLIC_BASE_URL")
    ffmpeg_bin: str = Field("ffmpeg", alias="VH_FLASHHEAD_FFMPEG")
    seed: int = Field(9999, alias="VH_FLASHHEAD_SEED")
    use_face_crop: bool = Field(False, alias="VH_FLASHHEAD_USE_FACE_CROP")
    fragment_ms: int = Field(480, ge=40, le=2000, alias="VH_FLASHHEAD_FRAGMENT_MS")
    video_crf: int = Field(18, ge=0, le=35, alias="VH_FLASHHEAD_VIDEO_CRF")
    video_sharpen: bool = Field(True, alias="VH_FLASHHEAD_VIDEO_SHARPEN")
    output_size: int = Field(768, ge=512, le=1024, alias="VH_FLASHHEAD_OUTPUT_SIZE")
    audio_motion_scale: float = Field(
        1.0,
        ge=0.25,
        le=1.25,
        alias="VH_FLASHHEAD_AUDIO_MOTION_SCALE",
    )
    cors_origins: str = Field("*", alias="VH_FLASHHEAD_CORS_ORIGINS")
    access_token_file: str = Field("", alias="VH_FLASHHEAD_ACCESS_TOKEN_FILE")
    transport: Literal["segments", "mse", "dual"] = Field("dual", alias="VH_FLASHHEAD_TRANSPORT")
    reference_locked: bool = Field(True, alias="VH_FLASHHEAD_REFERENCE_LOCKED")
    reference_sha256: str = Field(
        LOCKED_AVATAR_SHA256,
        alias="VH_FLASHHEAD_REFERENCE_SHA256",
    )

    @model_validator(mode="after")
    def validate_identity_lock(self) -> "FlashHeadSettings":
        if self.reference_locked and self.reference_sha256 != LOCKED_AVATAR_SHA256:
            raise ValueError("FlashHead avatar lock must use the approved identity digest")
        return self

    def resolved_access_token(self) -> str:
        if not self.access_token_file:
            return ""
        token_path = Path(self.access_token_file).expanduser()
        if not token_path.is_file():
            raise ValueError("FlashHead access token file is missing")
        return token_path.read_text().strip()


@dataclass(slots=True)
class StreamingResponse:
    response_id: str
    sample_rate: int
    expected_chunk_index: int = 0
    pcm16: bytearray = field(default_factory=bytearray)
    finished: bool = False
    canceled: bool = False
    sequence: int = 0
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task[None] | None = None
    muxer: ContinuousMP4Muxer | None = None
    media_sequence: int = 0

    def append(self, message: dict[str, Any]) -> None:
        chunk_index = int(message["chunk_index"])
        if chunk_index != self.expected_chunk_index:
            raise ValueError(f"Expected chunk {self.expected_chunk_index}, received {chunk_index}")
        if int(message["sample_rate"]) != self.sample_rate:
            raise ValueError("Sample rate changed inside one response")
        pcm = base64.b64decode(message["audio"], validate=True)
        if len(pcm) % 2:
            raise ValueError("PCM16 chunk contains a partial sample")
        self.pcm16.extend(pcm)
        self.expected_chunk_index += 1
        self.wake.set()


class FlashHeadEngine:
    """Thin stateful wrapper around the official FlashHead streaming loop."""

    def __init__(self, settings: FlashHeadSettings) -> None:
        self.settings = settings
        self.pipeline: Any | None = None
        self.params: dict[str, Any] = {}
        self.audio_history: deque[float] | None = None
        self._lock = threading.Lock()
        self._np: Any | None = None
        self._librosa: Any | None = None
        self._torch: Any | None = None
        self._get_base_data: Any | None = None
        self._get_audio_embedding: Any | None = None
        self._run_pipeline: Any | None = None
        self._character_digest: str | None = None

    @property
    def loaded(self) -> bool:
        return self.pipeline is not None

    @property
    def fps(self) -> int:
        return int(self.params["tgt_fps"])

    @property
    def width(self) -> int:
        return int(self.params["width"])

    @property
    def height(self) -> int:
        return int(self.params["height"])

    @property
    def model_sample_rate(self) -> int:
        return int(self.params["sample_rate"])

    @property
    def slice_frames(self) -> int:
        return int(self.params["frame_num"] - self.params["motion_frames_num"])

    def input_slice_bytes(self, input_sample_rate: int) -> int:
        samples = round(self.slice_frames * input_sample_rate / self.fps)
        return samples * 2

    def load(self) -> None:
        with self._lock:
            if self.loaded:
                return
            repo_dir = self.settings.repo_dir.resolve()
            checkpoint_dir = self.settings.checkpoint_dir.resolve()
            wav2vec_dir = self.settings.wav2vec_dir.resolve()
            if not repo_dir.is_dir():
                raise RuntimeError(f"FlashHead repository not found: {repo_dir}")
            if not checkpoint_dir.is_dir():
                raise RuntimeError(f"FlashHead checkpoint not found: {checkpoint_dir}")
            if not wav2vec_dir.is_dir():
                raise RuntimeError(f"wav2vec checkpoint not found: {wav2vec_dir}")

            import sys

            import librosa
            import numpy as np
            import torch

            sys.path.insert(0, str(repo_dir))
            previous_cwd = Path.cwd()
            try:
                os.chdir(repo_dir)
                from flash_head.inference import (
                    get_audio_embedding,
                    get_base_data,
                    get_infer_params,
                    get_pipeline,
                    run_pipeline,
                )
                from flash_head.src.pipeline import (
                    flash_head_pipeline as flash_head_pipeline_module,
                )

                # Keep the upstream submodule pinned and clean. Deployment-specific
                # compile controls belong to this integration layer rather than a
                # local edit inside SoulX-FlashHead.
                flash_head_pipeline_module.COMPILE_MODEL = (
                    os.environ.get("FLASHHEAD_COMPILE_MODEL", "1") != "0"
                )
                flash_head_pipeline_module.COMPILE_VAE = (
                    os.environ.get("FLASHHEAD_COMPILE_VAE", "1") != "0"
                )
            finally:
                os.chdir(previous_cwd)

            self.pipeline = get_pipeline(
                world_size=1,
                ckpt_dir=str(checkpoint_dir),
                model_type=self.settings.model_type,
                wav2vec_dir=str(wav2vec_dir),
            )
            self.params = get_infer_params()
            self._np = np
            self._librosa = librosa
            self._torch = torch
            self._get_base_data = get_base_data
            self._get_audio_embedding = get_audio_embedding
            self._run_pipeline = run_pipeline
            history_samples = self.model_sample_rate * int(self.params["cached_audio_duration"])
            self.audio_history = deque([0.0] * history_samples, maxlen=history_samples)

    def prepare_character(self, reference_path: Path) -> None:
        reference_digest = hashlib.sha256(reference_path.read_bytes()).hexdigest()
        if (
            self.settings.reference_locked
            and reference_digest != self.settings.reference_sha256
        ):
            raise ValueError(
                "Avatar identity is locked. FlashHead refused an unapproved reference image."
            )
        self.load()
        assert self._get_base_data is not None
        assert self.audio_history is not None
        with self._lock:
            if reference_digest != self._character_digest:
                self._get_base_data(
                    self.pipeline,
                    cond_image_path_or_dir=str(reference_path),
                    base_seed=self.settings.seed,
                    use_face_crop=self.settings.use_face_crop,
                )
                self._character_digest = reference_digest
                logger.info("Prepared new FlashHead character reference")
            else:
                self.pipeline.reset_person_name()
                logger.info("Reused cached FlashHead character reference")
            self.audio_history.clear()
            self.audio_history.extend(
                [0.0] * (self.model_sample_rate * int(self.params["cached_audio_duration"]))
            )

    def reset_motion(self) -> None:
        with self._lock:
            if self.pipeline is not None:
                self.pipeline.reset_person_name()

    def scale_audio_embedding(self, embedding: Any) -> Any:
        """Dampen audio-driven facial motion without changing the locked identity."""
        if self.settings.audio_motion_scale == 1.0:
            return embedding
        return embedding * self.settings.audio_motion_scale

    def generate_segment(self, pcm16: bytes, input_sample_rate: int) -> Any:
        assert self.loaded
        assert self.audio_history is not None
        assert self._np is not None
        assert self._librosa is not None
        assert self._torch is not None
        assert self._get_audio_embedding is not None
        assert self._run_pipeline is not None

        with self._lock:
            audio = self._np.frombuffer(pcm16, dtype="<i2").astype(self._np.float32) / 32768.0
            resampled = self._librosa.resample(
                audio,
                orig_sr=input_sample_rate,
                target_sr=self.model_sample_rate,
            )
            expected_samples = round(self.slice_frames * self.model_sample_rate / self.fps)
            if len(resampled) < expected_samples:
                resampled = self._np.pad(resampled, (0, expected_samples - len(resampled)))
            elif len(resampled) > expected_samples:
                resampled = resampled[:expected_samples]
            self.audio_history.extend(resampled.tolist())
            audio_array = self._np.asarray(self.audio_history, dtype=self._np.float32)
            audio_end_idx = int(self.params["cached_audio_duration"]) * self.fps
            audio_start_idx = audio_end_idx - int(self.params["frame_num"])
            embedding = self._get_audio_embedding(
                self.pipeline,
                audio_array,
                audio_start_idx,
                audio_end_idx,
            )
            embedding = self.scale_audio_embedding(embedding)
            self._torch.cuda.synchronize()
            video = self._run_pipeline(self.pipeline, embedding)
            frames = video[int(self.params["motion_frames_num"]) :].cpu().numpy()
            return frames.astype(self._np.uint8)


class FlashHeadSession:
    def __init__(
        self,
        settings: FlashHeadSettings,
        engine: FlashHeadEngine,
        websocket: WebSocket,
        base_url: str,
        engine_access: asyncio.Lock,
    ) -> None:
        self.settings = settings
        self.engine = engine
        self.websocket = websocket
        self.base_url = base_url.rstrip("/")
        self.engine_access = engine_access
        self.session_id: str | None = None
        self.sample_rate = 24_000
        self.preroll_ms = 1_000
        self.session_dir: Path | None = None
        self.responses: dict[str, StreamingResponse] = {}
        self._send_lock = asyncio.Lock()
        self._inference_lock = asyncio.Lock()
        self._owns_engine = False

    async def emit(self, event: dict[str, Any]) -> None:
        async with self._send_lock:
            await self.websocket.send_json(event)

    async def handle(self, message: dict[str, Any]) -> None:
        message_type = message.get("type")
        if message_type == "avatar.session.start":
            await self._start(message)
        elif message_type == "avatar.audio.append":
            await self._append(message)
        elif message_type == "avatar.response.finish":
            await self._finish(message.get("response_id"))
        elif message_type == "avatar.response.cancel":
            await self._cancel(message.get("response_id"))
        elif message_type == "avatar.state":
            await self.emit(
                {
                    "type": "avatar.state.ack",
                    "state": message.get("state"),
                    "response_id": message.get("response_id"),
                }
            )
        else:
            raise ValueError(f"Unsupported renderer message: {message_type}")

    async def _start(self, message: dict[str, Any]) -> None:
        await self.engine_access.acquire()
        self._owns_engine = True
        self.session_id = str(message["session_id"])
        self.sample_rate = int(message["sample_rate"])
        self.preroll_ms = max(0, min(5_000, int(message.get("preroll_ms", 1_000))))
        self.session_dir = self.settings.runtime_dir / "sessions" / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        encoded_reference = message.get("reference_image")
        if not encoded_reference:
            raise ValueError("FlashHead worker requires an uploaded reference_image")
        reference_name = Path(message.get("reference_name") or "reference.jpg").name
        reference_path = self.session_dir / reference_name
        reference_path.write_bytes(base64.b64decode(encoded_reference, validate=True))
        # torch.compile initializes process-level compiler state and may fork
        # workers on first use. Running that initialization inside
        # asyncio.to_thread can deadlock on Linux. Cold character preparation
        # intentionally blocks this worker once; segment generation below
        # remains offloaded so steady-state WebSocket handling stays async.
        self.engine.prepare_character(reference_path)
        await self.emit(
            {
                "type": "avatar.session.ready",
                "session_id": self.session_id,
                "backend": f"flashhead-{self.settings.model_type}",
                "fps": self.engine.fps,
                "width": self.settings.output_size,
                "height": self.settings.output_size,
                "audio_chunk_ms": round(self.engine.slice_frames * 1000 / self.engine.fps),
            }
        )

    async def _append(self, message: dict[str, Any]) -> None:
        if self.session_id is None:
            raise ValueError("avatar.session.start is required before audio")
        response_id = str(message["response_id"])
        response = self.responses.setdefault(response_id, StreamingResponse(response_id, self.sample_rate))
        response.append(message)
        if response.task is None:
            response.task = asyncio.create_task(self._process_guard(response))
        await self.emit(
            {
                "type": "avatar.audio.ack",
                "response_id": response_id,
                "chunk_index": int(message["chunk_index"]),
                "buffered_ms": len(response.pcm16) / 2 * 1000 / self.sample_rate,
            }
        )

    async def _finish(self, response_id: str | None) -> None:
        if not response_id:
            return
        response = self.responses.get(response_id)
        if response is None:
            await self.emit(
                {
                    "type": "avatar.render.done",
                    "response_id": response_id,
                    "backend": f"flashhead-{self.settings.model_type}",
                    "segments": 0,
                }
            )
            return
        response.finished = True
        response.wake.set()
        logger.info(
            "FlashHead finish accepted response=%s buffered_bytes=%d segments=%d",
            response.response_id,
            len(response.pcm16),
            response.sequence,
        )
        await self.emit(
            {
                "type": "avatar.render.accepted",
                "response_id": response_id,
                "backend": f"flashhead-{self.settings.model_type}",
            }
        )

    async def _cancel(self, response_id: str | None) -> None:
        if response_id and response_id in self.responses:
            response = self.responses[response_id]
            response.canceled = True
            response.pcm16.clear()
            response.wake.set()
        await self.emit({"type": "avatar.response.cancelled", "response_id": response_id})

    async def _process_guard(self, response: StreamingResponse) -> None:
        try:
            await self._process(response)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.exception("FlashHead streaming response failed")
            if response.muxer:
                with contextlib.suppress(Exception):
                    await response.muxer.cancel()
            with contextlib.suppress(Exception):
                await self.emit(
                    {
                        "type": "avatar.error",
                        "response_id": response.response_id,
                        "message": str(error),
                        "backend": f"flashhead-{self.settings.model_type}",
                    }
                )

    async def _process(self, response: StreamingResponse) -> None:
        slice_bytes = self.engine.input_slice_bytes(response.sample_rate)
        while True:
            while len(response.pcm16) < slice_bytes and not response.finished and not response.canceled:
                response.wake.clear()
                if len(response.pcm16) < slice_bytes and not response.finished and not response.canceled:
                    await response.wake.wait()
            if response.canceled:
                async with self._inference_lock:
                    await asyncio.to_thread(self.engine.reset_motion)
                if response.muxer:
                    await response.muxer.cancel()
                return
            if not response.pcm16:
                break

            take = min(slice_bytes, len(response.pcm16))
            mux_pcm = bytes(response.pcm16[:take])
            del response.pcm16[:take]
            model_pcm = mux_pcm.ljust(slice_bytes, b"\x00")
            sequence = response.sequence
            response.sequence += 1
            async with self._inference_lock:
                frames = await asyncio.to_thread(
                    self.engine.generate_segment,
                    model_pcm,
                    response.sample_rate,
                )
                if response.canceled:
                    await asyncio.to_thread(self.engine.reset_motion)
                    if response.muxer:
                        await response.muxer.cancel()
                    return
            output_frames = frames
            if take < slice_bytes:
                # The model needs a full padded slice, but the muxer must only
                # receive frames covered by the real tail audio. Otherwise
                # FFmpeg waits for a shorter audio stream to catch up with a
                # full video slice and the final write cannot complete.
                frame_count = max(
                    1,
                    math.ceil(take / 2 / response.sample_rate * self.engine.fps),
                )
                output_frames = frames[:frame_count]
            if self.settings.transport in {"mse", "dual"}:
                muxer = await self._ensure_muxer(response)
                await muxer.write(output_frames.tobytes(), mux_pcm)
            if response.canceled:
                if response.muxer:
                    await response.muxer.cancel()
                return
            if self.settings.transport in {"segments", "dual"}:
                segment_path = await self._encode_segment(
                    response,
                    sequence,
                    output_frames,
                    mux_pcm,
                )
                assert self.session_dir is not None
                relative = segment_path.relative_to(self.settings.runtime_dir)
                await self.emit(
                    {
                        "type": "avatar.video.segment",
                        "session_id": self.session_id,
                        "response_id": response.response_id,
                        "sequence": sequence,
                        "url": f"{self.base_url}/runtime/{relative.as_posix()}",
                        "duration_ms": len(mux_pcm) / 2 * 1000 / response.sample_rate,
                        "backend": f"flashhead-{self.settings.model_type}",
                        "clock": "audio-master",
                        "audio_included": True,
                    }
                )
        if response.muxer:
            logger.info(
                "FlashHead muxer finish starting response=%s segments=%d",
                response.response_id,
                response.sequence,
            )
            await response.muxer.finish()
            logger.info("FlashHead muxer finish complete response=%s", response.response_id)
        await self.emit(
            {
                "type": "avatar.render.done",
                "response_id": response.response_id,
                "backend": f"flashhead-{self.settings.model_type}",
                "segments": response.sequence,
            }
        )

    async def _ensure_muxer(self, response: StreamingResponse) -> ContinuousMP4Muxer:
        if response.muxer is not None:
            return response.muxer

        async def emit_fragment(kind: str, payload: bytes) -> None:
            event_type = "avatar.media.init" if kind == "init" else "avatar.media.fragment"
            sequence = response.media_sequence
            if kind == "media":
                response.media_sequence += 1
            await self.emit(
                {
                    "type": event_type,
                    "session_id": self.session_id,
                    "response_id": response.response_id,
                    "sequence": sequence,
                    "data": base64.b64encode(payload).decode("ascii"),
                    "backend": f"flashhead-{self.settings.model_type}",
                    "mime_type": 'video/mp4; codecs="avc1.42E01F, mp4a.40.2"',
                    "clock": "audio-master",
                    "audio_included": True,
                }
            )

        await self.emit(
            {
                "type": "avatar.media.start",
                "session_id": self.session_id,
                "response_id": response.response_id,
                "backend": f"flashhead-{self.settings.model_type}",
                "mime_type": 'video/mp4; codecs="avc1.42E01F, mp4a.40.2"',
                "fallback_segments": self.settings.transport == "dual",
                "preroll_ms": self.preroll_ms,
            }
        )
        response.muxer = ContinuousMP4Muxer(
            ffmpeg_bin=self.settings.ffmpeg_bin,
            width=self.engine.width,
            height=self.engine.height,
            fps=self.engine.fps,
            sample_rate=response.sample_rate,
            gop_size=max(
                1,
                round(self.engine.fps * self.settings.fragment_ms / 1000),
            ),
            emit=emit_fragment,
            crf=self.settings.video_crf,
            sharpen=self.settings.video_sharpen,
            output_width=self.settings.output_size,
            output_height=self.settings.output_size,
        )
        return response.muxer

    async def _encode_segment(
        self,
        response: StreamingResponse,
        sequence: int,
        frames: Any,
        pcm16: bytes,
    ) -> Path:
        assert self.session_dir is not None
        stem = f"{response.response_id}-{sequence:05d}"
        rgb_path = self.session_dir / f"{stem}.rgb"
        wav_path = self.session_dir / f"{stem}.wav"
        video_path = self.session_dir / f"{stem}.mp4"
        rgb_path.write_bytes(frames.tobytes())
        with wave.open(str(wav_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(response.sample_rate)
            wav_file.writeframes(pcm16)
        video_filter = build_video_filter(
            input_width=self.engine.width,
            input_height=self.engine.height,
            output_width=self.settings.output_size,
            output_height=self.settings.output_size,
            sharpen=self.settings.video_sharpen,
        )
        command = [
            self.settings.ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self.engine.width}x{self.engine.height}",
            "-r",
            str(self.engine.fps),
            "-i",
            str(rgb_path),
            "-i",
            str(wav_path),
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            str(self.settings.video_crf),
            *(["-vf", video_filter] if video_filter else []),
            "-tune",
            "zerolatency",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            "-profile:v",
            "baseline",
            "-level:v",
            "3.1",
            "-g",
            str(self.engine.fps),
            "-keyint_min",
            str(self.engine.fps),
            "-sc_threshold",
            "0",
            "-movflags",
            "+frag_keyframe+empty_moov+default_base_moof",
            str(video_path),
        ]
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        rgb_path.unlink(missing_ok=True)
        if process.returncode:
            raise RuntimeError(f"ffmpeg failed: {stderr.decode(errors='replace')}")
        return video_path

    async def close(self) -> None:
        for response in self.responses.values():
            response.canceled = True
            response.wake.set()
        for response in self.responses.values():
            if response.task:
                with contextlib.suppress(asyncio.CancelledError):
                    await response.task
        if self._owns_engine:
            self.engine_access.release()
            self._owns_engine = False


def create_flashhead_app(settings: FlashHeadSettings | None = None) -> FastAPI:
    worker_settings = settings or FlashHeadSettings()
    worker_settings.runtime_dir.mkdir(parents=True, exist_ok=True)
    application = FastAPI(title="Virtual Human FlashHead Streaming Worker", version="0.1.0")
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in worker_settings.cors_origins.split(",") if origin.strip()],
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )
    engine = FlashHeadEngine(worker_settings)
    engine_access = asyncio.Lock()

    @application.get("/healthz")
    async def healthz() -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "backend": f"flashhead-{worker_settings.model_type}",
                "model_loaded": engine.loaded,
                "busy": engine_access.locked(),
                "identity_locked": worker_settings.reference_locked,
                "identity_sha256": (
                    worker_settings.reference_sha256
                    if worker_settings.reference_locked
                    else ""
                ),
                "checkpoint_dir": str(worker_settings.checkpoint_dir),
                "audio_motion_scale": worker_settings.audio_motion_scale,
            }
        )

    @application.websocket("/avatar")
    async def avatar(websocket: WebSocket) -> None:
        expected_token = worker_settings.resolved_access_token()
        authorization = websocket.headers.get("authorization", "")
        supplied_token = (
            authorization.removeprefix("Bearer ").strip()
            if authorization.startswith("Bearer ")
            else websocket.query_params.get("token", "")
        )
        if expected_token and not secrets.compare_digest(
            supplied_token,
            expected_token,
        ):
            await websocket.close(code=1008, reason="Invalid renderer token")
            return
        await websocket.accept()
        scheme = "https" if websocket.url.scheme == "wss" else "http"
        base_url = worker_settings.public_base_url or f"{scheme}://{websocket.url.netloc}"
        session = FlashHeadSession(worker_settings, engine, websocket, base_url, engine_access)
        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    message = json.loads(raw)
                    if not isinstance(message, dict):
                        raise ValueError("Renderer message must be a JSON object")
                    await session.handle(message)
                except (KeyError, ValueError, TypeError) as error:
                    await session.emit({"type": "avatar.error", "message": str(error)})
        except WebSocketDisconnect:
            pass
        finally:
            await session.close()

    application.mount(
        "/runtime",
        StaticFiles(directory=worker_settings.runtime_dir),
        name="runtime",
    )
    return application


app = create_flashhead_app()


def main() -> None:
    settings = FlashHeadSettings()
    uvicorn.run(
        "virtual_human.flashhead_worker:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        ws_ping_interval=20,
        ws_ping_timeout=FLASHHEAD_WS_PING_TIMEOUT_SECONDS,
    )


if __name__ == "__main__":
    main()
