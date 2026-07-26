#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
import wave
from array import array
from pathlib import Path


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Generate and validate one reference-conditioned XGC Higgs WAV."
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8010/v1",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=project_root
        / "runtime"
        / "voice-calibration"
        / "reference-female-only-complete-11s.wav",
    )
    parser.add_argument(
        "--reference-text",
        default=os.environ.get("VH_OMLX_TTS_REF_TEXT"),
        help=(
            "Transcript matching the private reference audio. Prefer setting "
            "VH_OMLX_TTS_REF_TEXT in the untracked runtime environment."
        ),
    )
    parser.add_argument(
        "--text",
        default="你回来啦。外面还在下雨吗？先休息一下，我陪你慢慢聊。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "runtime" / "xiangongyun" / "tts-real.wav",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.reference_text:
        raise SystemExit(
            "--reference-text or VH_OMLX_TTS_REF_TEXT is required; "
            "private reference transcripts are intentionally not embedded."
        )
    reference = args.reference.read_bytes()
    payload = {
        "model": "higgs_audio_v3",
        "input": args.text,
        "response_format": "wav",
        "max_new_tokens": 1024,
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 50,
        "seed": 20260817,
        "stream": False,
        "ref_audio": (
            "data:audio/wav;base64,"
            + base64.b64encode(reference).decode("ascii")
        ),
        "ref_text": args.reference_text,
    }
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/audio/speech",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            body = response.read()
            status = response.status
            content_type = response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"TTS request failed with HTTP {error.code}: {detail[:2000]}"
        ) from error
    latency_seconds = time.perf_counter() - started

    with wave.open(io.BytesIO(body), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frame_count = wav.getnframes()
        frames = wav.readframes(frame_count)

    samples = array("h")
    samples.frombytes(frames)
    if sys.byteorder == "big":
        samples.byteswap()
    sample_count = len(samples)
    peak = max((abs(value) for value in samples), default=0)
    rms = math.sqrt(
        sum(value * value for value in samples) / max(sample_count, 1)
    )
    voiced_ratio = (
        sum(abs(value) >= 200 for value in samples) / max(sample_count, 1)
    )
    clipping_ratio = (
        sum(abs(value) >= 32760 for value in samples) / max(sample_count, 1)
    )
    duration_seconds = frame_count / sample_rate

    report = {
        "status": status,
        "content_type": content_type,
        "latency_seconds": round(latency_seconds, 3),
        "bytes": len(body),
        "channels": channels,
        "sample_width": sample_width,
        "sample_rate": sample_rate,
        "frame_count": frame_count,
        "duration_seconds": round(duration_seconds, 3),
        "realtime_factor": round(latency_seconds / duration_seconds, 3),
        "peak": peak,
        "rms": round(rms, 2),
        "voiced_ratio": round(voiced_ratio, 4),
        "clipping_ratio": round(clipping_ratio, 6),
    }

    failures: list[str] = []
    if status != 200:
        failures.append("http_status")
    if channels != 1 or sample_width != 2 or sample_rate != 24000:
        failures.append("wav_format")
    if duration_seconds < 1.0:
        failures.append("audio_too_short")
    if peak < 500 or voiced_ratio < 0.05:
        failures.append("audio_is_silent")
    if clipping_ratio > 0.01:
        failures.append("audio_is_clipped")
    report["failures"] = failures

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(body)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
