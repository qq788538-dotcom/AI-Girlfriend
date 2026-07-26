#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx

from virtual_human.higgs_server import _wav_bytes
from virtual_human.omlx_realtime import _parse_streaming_wav_header


def benchmark(
    *,
    url: str,
    text: str,
    streaming_mode: str,
    interval: float,
    output: Path | None,
) -> dict[str, object]:
    started_at = time.perf_counter()
    first_audio_ms: float | None = None
    network_chunks = 0
    payload = bytearray()
    pcm_offset: int | None = None

    with httpx.stream(
        "POST",
        url,
        json={
            "input": text,
            "stream": True,
            "streaming_mode": streaming_mode,
            "streaming_interval": interval,
        },
        timeout=180,
    ) as response:
        response.raise_for_status()
        for chunk in response.iter_bytes():
            network_chunks += 1
            payload.extend(chunk)
            if pcm_offset is None:
                parsed = _parse_streaming_wav_header(payload)
                if parsed is not None:
                    pcm_offset = parsed[0]
            if (
                first_audio_ms is None
                and pcm_offset is not None
                and len(payload) > pcm_offset
            ):
                first_audio_ms = (time.perf_counter() - started_at) * 1000

    total_ms = (time.perf_counter() - started_at) * 1000
    if pcm_offset is None:
        raise RuntimeError("Higgs response did not contain a complete WAV header")
    pcm16 = bytes(payload[pcm_offset:])
    if len(pcm16) % 2:
        raise RuntimeError("Higgs response ended with a partial PCM16 sample")
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(_wav_bytes(pcm16))

    return {
        "url": url,
        "streaming_mode": streaming_mode,
        "streaming_interval": interval,
        "first_audio_ms": round(first_audio_ms or total_ms, 3),
        "total_ms": round(total_ms, 3),
        "network_chunks": network_chunks,
        "pcm_bytes": len(pcm16),
        "audio_duration_seconds": round(len(pcm16) / 2 / 24_000, 3),
        "output": str(output) if output else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Higgs PCM time-to-first-byte.")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8010/v1/audio/speech",
    )
    parser.add_argument("--text", required=True)
    parser.add_argument("--streaming-mode", default="v3_incremental")
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = benchmark(
        url=args.url,
        text=args.text,
        streaming_mode=args.streaming_mode,
        interval=args.interval,
        output=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
