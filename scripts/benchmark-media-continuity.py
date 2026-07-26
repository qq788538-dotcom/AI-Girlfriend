#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def probe_media(path: Path) -> dict[str, Any]:
    completed = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            (
                "format=duration,size,bit_rate:"
                "stream=index,codec_type,codec_name,width,height,"
                "avg_frame_rate,duration,start_time:"
                "frame=best_effort_timestamp_time"
            ),
            "-show_frames",
            "-of",
            "json",
            str(path),
        ]
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or "ffprobe failed")
    return json.loads(completed.stdout)


def filter_log(path: Path, *, media_filter: str, video: bool) -> str:
    command = ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path)]
    command.extend(["-vf" if video else "-af", media_filter])
    command.extend(["-an" if video else "-vn", "-f", "null", "-"])
    completed = run(command)
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or "ffmpeg filter failed")
    # Detection filters log to stderr, while metadata=print:file=- writes to
    # stdout. Preserve both so callers can parse either kind of FFmpeg filter.
    return completed.stdout + "\n" + completed.stderr


def durations(log: str, label: str) -> list[float]:
    pattern = rf"{re.escape(label)}:\s*([0-9.]+)"
    return [float(value) for value in re.findall(pattern, log)]


def metadata_values(log: str, key: str) -> list[float]:
    pattern = rf"^{re.escape(key)}=([-+]?[0-9]*\.?[0-9]+)$"
    return [
        float(match.group(1))
        for line in log.splitlines()
        if (match := re.match(pattern, line.strip()))
    ]


def max_sustained_drop(
    values: list[float],
    *,
    baseline_frames: int,
    window_frames: int,
) -> tuple[float, int | None]:
    if not values:
        return 0.0, None
    baseline_count = min(max(baseline_frames, 1), len(values))
    baseline = sorted(values[:baseline_count])[baseline_count // 2]
    window = min(max(window_frames, 1), len(values))
    window_means = [
        sum(values[start : start + window]) / window
        for start in range(0, len(values) - window + 1)
    ]
    lowest_mean = min(window_means)
    lowest_start = window_means.index(lowest_mean)
    return max(0.0, baseline - lowest_mean), lowest_start


def ratio(value: str) -> float:
    if not value or value == "0/0":
        return 0.0
    return float(Fraction(value))


def analyze(
    path: Path,
    *,
    min_duration: float,
    min_fps: float,
    max_av_drift_ms: float,
    max_frame_gap_ms: float,
    max_silence_s: float,
    max_frame_luma_diff: float,
    max_sustained_entropy_drop: float,
    entropy_window_frames: int,
) -> dict[str, Any]:
    probe = probe_media(path)
    streams = probe.get("streams", [])
    video = next(
        (stream for stream in streams if stream.get("codec_type") == "video"),
        None,
    )
    audio = next(
        (stream for stream in streams if stream.get("codec_type") == "audio"),
        None,
    )
    if video is None or audio is None:
        raise RuntimeError("Media must contain one video and one audio stream")

    decode = run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"])
    black_log = filter_log(
        path,
        media_filter="blackdetect=d=0.08:pix_th=0.10",
        video=True,
    )
    freeze_log = filter_log(
        path,
        media_filter="freezedetect=n=-50dB:d=0.5",
        video=True,
    )
    silence_log = filter_log(
        path,
        media_filter="silencedetect=noise=-45dB:d=0.35",
        video=False,
    )
    signal_log = filter_log(
        path,
        media_filter="signalstats,metadata=print:file=-",
        video=True,
    )
    entropy_log = filter_log(
        path,
        media_filter="entropy,metadata=print:file=-",
        video=True,
    )

    frame_times = [
        float(frame["best_effort_timestamp_time"])
        for frame in probe.get("frames", [])
        if frame.get("media_type") == "video"
        and frame.get("best_effort_timestamp_time") is not None
    ]
    frame_gaps_ms = [
        (right - left) * 1000
        for left, right in zip(frame_times, frame_times[1:])
    ]
    video_duration = float(video.get("duration") or 0)
    audio_duration = float(audio.get("duration") or 0)
    av_drift_ms = abs(video_duration - audio_duration) * 1000
    fps = ratio(str(video.get("avg_frame_rate") or "0/0"))
    black_durations = durations(black_log, "black_duration")
    freeze_durations = durations(freeze_log, "freeze_duration")
    silence_durations = durations(silence_log, "silence_duration")
    frame_luma_diffs = metadata_values(signal_log, "lavfi.signalstats.YDIF")
    largest_luma_diff = max(frame_luma_diffs, default=0.0)
    largest_luma_diff_frame = (
        frame_luma_diffs.index(largest_luma_diff)
        if frame_luma_diffs
        else None
    )
    normalized_luma_entropies = metadata_values(
        entropy_log,
        "lavfi.entropy.normalized_entropy.normal.Y",
    )
    sustained_entropy_drop, entropy_drop_start_frame = max_sustained_drop(
        normalized_luma_entropies,
        baseline_frames=max(math.ceil(fps), 1),
        window_frames=entropy_window_frames,
    )
    failures: list[str] = []

    if decode.returncode:
        failures.append("decode_error")
    if min(video_duration, audio_duration) < min_duration:
        failures.append("too_short")
    if fps < min_fps:
        failures.append("low_frame_rate")
    if av_drift_ms > max_av_drift_ms:
        failures.append("audio_video_duration_mismatch")
    if frame_gaps_ms and max(frame_gaps_ms) > max_frame_gap_ms:
        failures.append("video_timestamp_gap")
    if black_durations:
        failures.append("black_frame_run")
    if freeze_durations:
        failures.append("frozen_frame_run")
    # A large frame delta alone can be a valid hand gesture, camera move, or
    # close-up. Latent/static collapse also causes the image entropy to remain
    # far below its clean startup baseline for several consecutive frames.
    if (
        largest_luma_diff > max_frame_luma_diff
        and sustained_entropy_drop > max_sustained_entropy_drop
    ):
        failures.append("visual_corruption")
    if silence_durations and max(silence_durations) > max_silence_s:
        failures.append("long_audio_silence")

    return {
        "status": "pass" if not failures else "fail",
        "media": str(path),
        "failures": failures,
        "duration_s": {
            "video": round(video_duration, 6),
            "audio": round(audio_duration, 6),
        },
        "av_drift_ms": round(av_drift_ms, 3),
        "video": {
            "codec": video.get("codec_name"),
            "resolution": f"{video.get('width')}x{video.get('height')}",
            "fps": round(fps, 3),
            "frames": len(frame_times),
            "max_timestamp_gap_ms": round(max(frame_gaps_ms, default=0), 3),
            "black_runs_s": black_durations,
            "freeze_runs_s": freeze_durations,
            "max_frame_luma_diff": round(largest_luma_diff, 6),
            "max_frame_luma_diff_frame": largest_luma_diff_frame,
            "max_sustained_entropy_drop": round(sustained_entropy_drop, 6),
            "entropy_drop_start_frame": entropy_drop_start_frame,
            "entropy_window_frames": entropy_window_frames,
        },
        "audio": {
            "codec": audio.get("codec_name"),
            "silence_runs_s": silence_durations,
            "max_silence_s": round(max(silence_durations, default=0), 6),
        },
        "decode_error": decode.stderr.strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect truncated or discontinuous avatar MP4 output."
    )
    parser.add_argument("media", type=Path, nargs="+")
    parser.add_argument("--min-duration", type=float, default=2.0)
    parser.add_argument("--min-fps", type=float, default=24.0)
    parser.add_argument("--max-av-drift-ms", type=float, default=80.0)
    parser.add_argument(
        "--max-frame-gap-ms",
        type=float,
        default=90.0,
        help="Allow the renderer's measured 82.734 ms first-frame startup interval.",
    )
    parser.add_argument("--max-silence-s", type=float, default=1.2)
    parser.add_argument(
        "--max-frame-luma-diff",
        type=float,
        default=12.0,
        help=(
            "Large adjacent-frame YDIF threshold used by the visual-corruption "
            "gate together with sustained entropy loss."
        ),
    )
    parser.add_argument(
        "--max-sustained-entropy-drop",
        type=float,
        default=0.08,
        help=(
            "Maximum allowed drop in normalized luma entropy versus the first "
            "second, averaged across a short frame window."
        ),
    )
    parser.add_argument("--entropy-window-frames", type=int, default=8)
    args = parser.parse_args()

    results = [
        analyze(
            path,
            min_duration=args.min_duration,
            min_fps=args.min_fps,
            max_av_drift_ms=args.max_av_drift_ms,
            max_frame_gap_ms=args.max_frame_gap_ms,
            max_silence_s=args.max_silence_s,
            max_frame_luma_diff=args.max_frame_luma_diff,
            max_sustained_entropy_drop=args.max_sustained_entropy_drop,
            entropy_window_frames=args.entropy_window_frames,
        )
        for path in args.media
    ]
    status = "pass" if all(result["status"] == "pass" for result in results) else "fail"
    print(
        json.dumps(
            {"status": status, "results": results},
            ensure_ascii=False,
            indent=2,
        )
    )
    raise SystemExit(0 if status == "pass" else 1)


if __name__ == "__main__":
    main()
