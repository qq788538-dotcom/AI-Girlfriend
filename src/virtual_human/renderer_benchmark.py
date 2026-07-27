from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import json
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

import httpx
import websockets

from virtual_human.av_verify import probe_media


@dataclass(slots=True)
class BenchmarkMetrics:
    renderer_url: str
    session_ready_ms: float
    audio_duration_ms: float
    audio_feed_ms: float
    first_output_ms: float
    render_done_ms: float
    post_audio_tail_ms: float
    end_to_end_realtime_factor: float
    realtime_input: bool
    audio_chunks: int
    media_fragments: int
    fallback_segments: int
    backend: str
    output_path: str | None


def read_pcm16_wav(path: Path) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getnchannels() != 1:
            raise ValueError("Benchmark WAV must be mono")
        if wav_file.getsampwidth() != 2:
            raise ValueError("Benchmark WAV must use PCM16 samples")
        if wav_file.getcomptype() != "NONE":
            raise ValueError("Benchmark WAV must be uncompressed PCM")
        return wav_file.readframes(wav_file.getnframes()), wav_file.getframerate()


def benchmark_violations(
    metrics: BenchmarkMetrics,
    *,
    max_first_output_ms: float | None = None,
    max_post_audio_tail_ms: float | None = None,
    min_realtime_factor: float | None = None,
) -> list[str]:
    violations: list[str] = []
    if max_first_output_ms is not None and metrics.first_output_ms > max_first_output_ms:
        violations.append(
            f"first output {metrics.first_output_ms:.1f} ms exceeds {max_first_output_ms:.1f} ms"
        )
    if (
        max_post_audio_tail_ms is not None
        and metrics.post_audio_tail_ms > max_post_audio_tail_ms
    ):
        violations.append(
            f"post-audio tail {metrics.post_audio_tail_ms:.1f} ms exceeds "
            f"{max_post_audio_tail_ms:.1f} ms"
        )
    if (
        min_realtime_factor is not None
        and metrics.end_to_end_realtime_factor < min_realtime_factor
    ):
        violations.append(
            f"realtime factor {metrics.end_to_end_realtime_factor:.3f} is below "
            f"{min_realtime_factor:.3f}"
        )
    return violations


async def benchmark_renderer(
    *,
    renderer_url: str,
    reference_path: Path,
    audio_path: Path,
    output_path: Path | None,
    chunk_ms: int,
    realtime_input: bool,
    timeout_seconds: float,
    access_token: str = "",
) -> tuple[BenchmarkMetrics, dict]:
    reference = base64.b64encode(reference_path.read_bytes()).decode("ascii")
    pcm16, sample_rate = read_pcm16_wav(audio_path)
    bytes_per_chunk = max(2, round(sample_rate * chunk_ms / 1000) * 2)
    session_id = f"bench_{uuid4().hex[:16]}"
    response_id = f"resp_{uuid4().hex[:16]}"
    connected_at = time.perf_counter()
    ready_at: float | None = None
    audio_started_at: float | None = None
    audio_finished_at: float | None = None
    first_output_at: float | None = None
    done_at: float | None = None
    backend = "unknown"
    media_parts: list[bytes] = []
    media_fragments = 0
    fallback_segments = 0
    final_video_url: str | None = None
    ready_event = asyncio.Event()
    done_event = asyncio.Event()
    error: RuntimeError | None = None

    async with websockets.connect(
        renderer_url,
        additional_headers=(
            {"Authorization": f"Bearer {access_token}"}
            if access_token
            else None
        ),
        max_size=None,
        ping_interval=20,
        ping_timeout=max(20, timeout_seconds),
    ) as websocket:

        async def receive_events() -> None:
            nonlocal ready_at, first_output_at, done_at
            nonlocal backend, media_fragments, fallback_segments, final_video_url, error
            try:
                async for raw in websocket:
                    event = json.loads(raw)
                    event_type = event.get("type")
                    if event.get("backend"):
                        backend = str(event["backend"])
                    if event_type == "avatar.session.ready":
                        ready_at = time.perf_counter()
                        ready_event.set()
                    elif event_type in {"avatar.media.init", "avatar.media.fragment"}:
                        media_parts.append(base64.b64decode(event["data"], validate=True))
                        if event_type == "avatar.media.fragment":
                            media_fragments += 1
                            first_output_at = first_output_at or time.perf_counter()
                    elif event_type == "avatar.video.segment":
                        fallback_segments += 1
                        first_output_at = first_output_at or time.perf_counter()
                    elif event_type == "avatar.video.ready":
                        final_video_url = str(event.get("url") or "") or None
                        first_output_at = first_output_at or time.perf_counter()
                    elif event_type == "avatar.error":
                        error = RuntimeError(
                            str(event.get("message") or "Renderer benchmark failed")
                        )
                        ready_event.set()
                        done_event.set()
                        return
                    elif event_type == "avatar.render.done":
                        done_at = time.perf_counter()
                        done_event.set()
                        return
            except websockets.ConnectionClosed as exception:
                error = RuntimeError(
                    f"Renderer connection closed before completion: {exception}"
                )
                ready_event.set()
                done_event.set()

        receiver = asyncio.create_task(receive_events())
        await websocket.send(
            json.dumps(
                {
                    "type": "avatar.session.start",
                    "session_id": session_id,
                    "backend": "benchmark",
                    "reference": str(reference_path),
                    "reference_name": reference_path.name,
                    "reference_image": reference,
                    "sample_rate": sample_rate,
                    "preroll_ms": 1000,
                    "clock": "audio-master",
                }
            )
        )
        await asyncio.wait_for(ready_event.wait(), timeout=timeout_seconds)
        if error:
            raise error
        audio_started_at = time.perf_counter()
        audio_chunks = 0
        for offset in range(0, len(pcm16), bytes_per_chunk):
            chunk = pcm16[offset : offset + bytes_per_chunk]
            duration_ms = len(chunk) / 2 * 1000 / sample_rate
            await websocket.send(
                json.dumps(
                    {
                        "type": "avatar.audio.append",
                        "response_id": response_id,
                        "chunk_index": audio_chunks,
                        "pts_ms": offset / 2 * 1000 / sample_rate,
                        "duration_ms": duration_ms,
                        "sample_rate": sample_rate,
                        "audio": base64.b64encode(chunk).decode("ascii"),
                    }
                )
            )
            audio_chunks += 1
            if realtime_input:
                await asyncio.sleep(duration_ms / 1000)
        audio_finished_at = time.perf_counter()
        await websocket.send(json.dumps({"type": "avatar.response.finish", "response_id": response_id}))
        await asyncio.wait_for(done_event.wait(), timeout=timeout_seconds)
        if error:
            raise error
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(receiver, timeout=1)
        if not receiver.done():
            receiver.cancel()

    if ready_at is None or audio_started_at is None or audio_finished_at is None or done_at is None:
        raise RuntimeError("Renderer benchmark ended without complete timing events")
    if first_output_at is None:
        first_output_at = done_at

    saved_output: str | None = None
    probe: dict = {}
    if media_parts and output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"".join(media_parts))
        saved_output = str(output_path)
        probe = probe_media(saved_output)
    elif final_video_url and output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.get(final_video_url)
            response.raise_for_status()
        output_path.write_bytes(response.content)
        saved_output = str(output_path)
        probe = probe_media(saved_output)

    metrics = BenchmarkMetrics(
        renderer_url=renderer_url,
        session_ready_ms=(ready_at - connected_at) * 1000,
        audio_duration_ms=len(pcm16) / 2 * 1000 / sample_rate,
        audio_feed_ms=(audio_finished_at - audio_started_at) * 1000,
        first_output_ms=(first_output_at - audio_started_at) * 1000,
        render_done_ms=(done_at - audio_started_at) * 1000,
        post_audio_tail_ms=(done_at - audio_finished_at) * 1000,
        end_to_end_realtime_factor=(
            (len(pcm16) / 2 * 1000 / sample_rate)
            / max((done_at - audio_started_at) * 1000, 0.001)
        ),
        realtime_input=realtime_input,
        audio_chunks=audio_chunks,
        media_fragments=media_fragments,
        fallback_segments=fallback_segments,
        backend=backend,
        output_path=saved_output,
    )
    return metrics, probe


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark an avatar renderer through the production WebSocket protocol."
    )
    parser.add_argument("--url", default="ws://127.0.0.1:8770/avatar")
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("public/avatar-ai-girlfriend-v6.png"),
    )
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runtime/benchmark/renderer-output.mp4"))
    parser.add_argument("--chunk-ms", type=int, default=40)
    parser.add_argument("--realtime-input", action="store_true")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument(
        "--token-file",
        type=Path,
        help="Read a renderer bearer token from this file without exposing it in the URL.",
    )
    parser.add_argument("--max-first-output-ms", type=float)
    parser.add_argument("--max-post-audio-tail-ms", type=float)
    parser.add_argument("--min-realtime-factor", type=float)
    args = parser.parse_args()

    if not args.reference.is_file():
        parser.error(f"Reference image does not exist: {args.reference}")
    if not args.audio.is_file():
        parser.error(f"Audio file does not exist: {args.audio}")
    if args.chunk_ms <= 0:
        parser.error("--chunk-ms must be positive")
    if args.token_file is not None and not args.token_file.is_file():
        parser.error(f"Renderer token file does not exist: {args.token_file}")

    try:
        metrics, probe = asyncio.run(
            benchmark_renderer(
                renderer_url=args.url,
                reference_path=args.reference,
                audio_path=args.audio,
                output_path=args.output,
                chunk_ms=args.chunk_ms,
                realtime_input=args.realtime_input,
                timeout_seconds=args.timeout,
                access_token=(
                    args.token_file.read_text().strip()
                    if args.token_file is not None
                    else ""
                ),
            )
        )
    except Exception as error:
        parser.exit(1, f"Renderer benchmark failed: {error}\n")
    violations = benchmark_violations(
        metrics,
        max_first_output_ms=args.max_first_output_ms,
        max_post_audio_tail_ms=args.max_post_audio_tail_ms,
        min_realtime_factor=args.min_realtime_factor,
    )
    result = {
        "status": "failed" if violations else "ok",
        "metrics": asdict(metrics),
        "probe": probe,
        "violations": violations,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
