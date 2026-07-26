from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class AVMetrics:
    video_codec: str
    audio_codec: str
    width: int
    height: int
    fps: float
    sample_rate: int
    channels: int
    duration_ms: float
    drift_ms: float


def probe_media(source: str, ffprobe_bin: str = "ffprobe") -> dict[str, Any]:
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type,codec_name,width,height,r_frame_rate,sample_rate,channels,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        source,
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def validate_probe(
    probe: dict[str, Any],
    *,
    expected_width: int,
    expected_height: int,
    expected_fps: float,
    expected_sample_rate: int,
    max_drift_ms: float,
) -> AVMetrics:
    streams = probe.get("streams") or []
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    if video is None or audio is None:
        raise ValueError("Both video and audio streams are required")

    width = int(video.get("width", 0))
    height = int(video.get("height", 0))
    fps = float(Fraction(video.get("r_frame_rate", "0/1")))
    sample_rate = int(audio.get("sample_rate", 0))
    video_duration = float(video.get("duration") or probe.get("format", {}).get("duration") or 0)
    audio_duration = float(audio.get("duration") or probe.get("format", {}).get("duration") or 0)
    drift_ms = abs(video_duration - audio_duration) * 1000

    if (width, height) != (expected_width, expected_height):
        raise ValueError(f"Expected {expected_width}x{expected_height}, received {width}x{height}")
    if abs(fps - expected_fps) > 0.01:
        raise ValueError(f"Expected {expected_fps:g} FPS, received {fps:g}")
    if sample_rate != expected_sample_rate:
        raise ValueError(f"Expected {expected_sample_rate} Hz audio, received {sample_rate} Hz")
    if video_duration <= 0 or audio_duration <= 0:
        raise ValueError("Media duration must be positive")
    if drift_ms > max_drift_ms:
        raise ValueError(f"A/V duration drift {drift_ms:.1f} ms exceeds {max_drift_ms:.1f} ms")

    return AVMetrics(
        video_codec=str(video.get("codec_name") or "unknown"),
        audio_codec=str(audio.get("codec_name") or "unknown"),
        width=width,
        height=height,
        fps=fps,
        sample_rate=sample_rate,
        channels=int(audio.get("channels", 0)),
        duration_ms=max(video_duration, audio_duration) * 1000,
        drift_ms=drift_ms,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the generated avatar A/V artifact with ffprobe.")
    parser.add_argument("source", help="Local media path or ffprobe-compatible URL")
    parser.add_argument("--width", type=int, default=416)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=20)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--max-drift-ms", type=float, default=80)
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()

    source = args.source
    if "://" not in source and not Path(source).is_file():
        parser.error(f"Media file does not exist: {source}")

    try:
        metrics = validate_probe(
            probe_media(source, args.ffprobe),
            expected_width=args.width,
            expected_height=args.height,
            expected_fps=args.fps,
            expected_sample_rate=args.sample_rate,
            max_drift_ms=args.max_drift_ms,
        )
    except (OSError, subprocess.CalledProcessError, ValueError, json.JSONDecodeError) as error:
        parser.exit(1, f"A/V verification failed: {error}\n")

    print(json.dumps({"status": "ok", **asdict(metrics)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
