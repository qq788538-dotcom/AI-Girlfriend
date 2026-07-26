#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time
import uuid
import wave
from pathlib import Path

import websockets


def read_pcm16(path: Path) -> tuple[int, bytes]:
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2:
            raise ValueError("Input must be mono PCM16 WAV")
        return wav_file.getframerate(), wav_file.readframes(wav_file.getnframes())


async def benchmark(args: argparse.Namespace) -> dict[str, object]:
    sample_rate, pcm16 = read_pcm16(args.audio)
    session_id = f"bench_{uuid.uuid4().hex[:12]}"
    response_id = f"resp_{uuid.uuid4().hex[:12]}"
    reference = base64.b64encode(args.reference.read_bytes()).decode("ascii")
    chunk_bytes = max(2, round(sample_rate * args.chunk_ms / 1000) * 2)
    started = time.perf_counter()
    first_output_at: float | None = None
    fragments: list[bytes] = []
    media_fragments = 0
    rendered_segments = 0

    async with websockets.connect(
        args.url,
        max_size=None,
        ping_interval=20,
        ping_timeout=180,
        close_timeout=10,
        additional_headers=(
            {
                "Authorization": (
                    f"Bearer {args.token_file.read_text().strip()}"
                )
            }
            if args.token_file
            else None
        ),
    ) as websocket:
        await websocket.send(
            json.dumps(
                {
                    "type": "avatar.session.start",
                    "session_id": session_id,
                    "reference_name": args.reference.name,
                    "reference_image": reference,
                    "sample_rate": sample_rate,
                    "preroll_ms": args.preroll_ms,
                }
            )
        )
        while True:
            event = json.loads(await asyncio.wait_for(websocket.recv(), timeout=args.timeout))
            if event.get("type") == "avatar.session.ready":
                break

        for chunk_index, offset in enumerate(range(0, len(pcm16), chunk_bytes)):
            await websocket.send(
                json.dumps(
                    {
                        "type": "avatar.audio.append",
                        "response_id": response_id,
                        "chunk_index": chunk_index,
                        "sample_rate": sample_rate,
                        "audio": base64.b64encode(
                            pcm16[offset : offset + chunk_bytes]
                        ).decode("ascii"),
                    }
                )
            )
        await websocket.send(
            json.dumps(
                {
                    "type": "avatar.response.finish",
                    "response_id": response_id,
                }
            )
        )

        while True:
            event = json.loads(await asyncio.wait_for(websocket.recv(), timeout=args.timeout))
            event_type = event.get("type")
            if event_type in {"avatar.media.init", "avatar.media.fragment"}:
                if first_output_at is None:
                    first_output_at = time.perf_counter()
                fragments.append(base64.b64decode(event["data"], validate=True))
                if event_type == "avatar.media.fragment":
                    media_fragments += 1
            elif event_type == "avatar.video.segment":
                rendered_segments += 1
            elif event_type == "avatar.error":
                raise RuntimeError(str(event.get("message") or "FlashHead worker error"))
            elif event_type == "avatar.render.done":
                rendered_segments = max(rendered_segments, int(event.get("segments") or 0))
                break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(b"".join(fragments))
    finished = time.perf_counter()
    audio_duration = len(pcm16) / 2 / sample_rate
    return {
        "status": "ok",
        "url": args.url,
        "output": str(args.output),
        "audio_duration_s": round(audio_duration, 3),
        "render_s": round(finished - started, 3),
        "first_output_s": (
            round(first_output_at - started, 3) if first_output_at is not None else None
        ),
        "realtime_factor": round(audio_duration / (finished - started), 4),
        "media_fragments": media_fragments,
        "rendered_segments": rendered_segments,
        "bytes": args.output.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a FlashHead worker directly.")
    parser.add_argument("--url", default="ws://127.0.0.1:8771/avatar")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-ms", type=int, default=480)
    parser.add_argument("--preroll-ms", type=int, default=1000)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--token-file", type=Path)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(benchmark(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
