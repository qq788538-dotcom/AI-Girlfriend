#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import wave
from pathlib import Path

import numpy as np


def decode_gray_crop(
    video_path: Path,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
) -> np.ndarray:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"crop={width}:{height}:{x}:{y},format=gray",
        "-an",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    completed = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
    )
    pixels_per_frame = width * height
    raw = np.frombuffer(completed.stdout, dtype=np.uint8)
    if raw.size % pixels_per_frame:
        raise ValueError("Decoded video ended with a partial frame")
    return raw.reshape(-1, height, width)


def read_pcm16(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2:
            raise ValueError("Audio must be mono PCM16 WAV")
        sample_rate = wav_file.getframerate()
        audio = np.frombuffer(
            wav_file.readframes(wav_file.getnframes()),
            dtype="<i2",
        ).astype(np.float32)
    return audio / 32768.0, sample_rate


def frame_rms(
    audio: np.ndarray,
    *,
    sample_rate: int,
    fps: float,
    frame_count: int,
) -> np.ndarray:
    samples_per_frame = sample_rate / fps
    values = np.zeros(frame_count, dtype=np.float32)
    for index in range(frame_count):
        start = round(index * samples_per_frame)
        end = min(len(audio), round((index + 1) * samples_per_frame))
        if end > start:
            values[index] = float(np.sqrt(np.mean(audio[start:end] ** 2)))
    return values


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 3 or right.size < 3:
        return 0.0
    left = left - np.mean(left)
    right = right - np.mean(right)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def best_lag(
    audio_feature: np.ndarray,
    video_feature: np.ndarray,
    *,
    max_lag_frames: int,
) -> tuple[int, float]:
    result = (0, -1.0)
    for lag in range(-max_lag_frames, max_lag_frames + 1):
        if lag >= 0:
            audio_slice = audio_feature[: len(audio_feature) - lag or None]
            video_slice = video_feature[lag:]
        else:
            audio_slice = audio_feature[-lag:]
            video_slice = video_feature[: len(video_feature) + lag]
        count = min(len(audio_slice), len(video_slice))
        score = correlation(audio_slice[:count], video_slice[:count])
        if score > result[1]:
            result = (lag, score)
    return result


def analyze(
    video_path: Path,
    audio_path: Path,
    *,
    fps: float,
    crop: tuple[int, int, int, int],
) -> dict[str, object]:
    x, y, width, height = crop
    frames = decode_gray_crop(
        video_path,
        x=x,
        y=y,
        width=width,
        height=height,
    )
    audio, sample_rate = read_pcm16(audio_path)
    frame_count = min(len(frames), round(len(audio) / sample_rate * fps))
    frames = frames[:frame_count].astype(np.int16)
    mouth_motion = np.mean(np.abs(np.diff(frames, axis=0)), axis=(1, 2))
    rms = frame_rms(
        audio,
        sample_rate=sample_rate,
        fps=fps,
        frame_count=frame_count,
    )
    audio_change = np.abs(np.diff(rms))
    active_threshold = max(float(np.quantile(rms, 0.2)), 1e-5)
    active = np.maximum(rms[:-1], rms[1:]) >= active_threshold
    active_motion = mouth_motion[active]
    active_audio_change = audio_change[active]
    lag, score = best_lag(
        active_audio_change,
        active_motion,
        max_lag_frames=8,
    )
    jerk = np.abs(np.diff(mouth_motion))
    return {
        "video": str(video_path),
        "audio": str(audio_path),
        "frames_analyzed": frame_count,
        "mouth_crop": {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
        },
        "mouth_motion_mean": round(float(np.mean(active_motion)), 5),
        "mouth_motion_p95": round(float(np.quantile(active_motion, 0.95)), 5),
        "mouth_motion_jerk_mean": round(float(np.mean(jerk)), 5),
        "mouth_motion_jerk_p95": round(float(np.quantile(jerk, 0.95)), 5),
        "zero_lag_correlation": round(
            correlation(active_audio_change, active_motion),
            5,
        ),
        "best_lag_frames": lag,
        "best_lag_ms": round(lag * 1000 / fps, 3),
        "best_lag_correlation": round(score, 5),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure avatar mouth motion timing against an audio track."
    )
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=25.0)
    # The locked 512x512 portrait keeps the lips inside this lower-face box.
    # The previous default started at y=185 and included both eyes and most of
    # the nose, so blinking and small head motions could be misreported as
    # 200+ ms of lip-sync drift.
    parser.add_argument("--crop", default="205,235,110,90")
    parser.add_argument("videos", nargs="+", type=Path)
    args = parser.parse_args()
    crop = tuple(int(value) for value in args.crop.split(","))
    if len(crop) != 4:
        parser.error("--crop must be x,y,width,height")
    results = [
        analyze(
            video,
            args.audio,
            fps=args.fps,
            crop=crop,
        )
        for video in args.videos
    ]
    print(json.dumps({"results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
