#!/usr/bin/env python3
"""Run an isolated Ditto TalkingHead candidate without importing PyTorch."""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import librosa
import numpy as np
from stream_pipeline_offline import StreamSDK

EMOTIONS = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
    "surprise": 6,
    "contempt": 7,
}


class _WriterProbe:
    def __init__(self, writer: Any) -> None:
        self.writer = writer
        self.first_frame_at: float | None = None
        self.frames = 0

    def __call__(self, frame: np.ndarray, *, fmt: str) -> None:
        if self.first_frame_at is None:
            self.first_frame_at = time.perf_counter()
        self.frames += 1
        self.writer(frame, fmt=fmt)

    def close(self) -> None:
        self.writer.close()


def _smooth_speech_energy(
    audio: np.ndarray,
    *,
    sample_rate: int,
    fps: int,
    frame_count: int,
) -> np.ndarray:
    samples_per_frame = sample_rate / fps
    energy = np.zeros(frame_count, dtype=np.float32)
    for frame_index in range(frame_count):
        start = round(frame_index * samples_per_frame)
        end = min(len(audio), round((frame_index + 1) * samples_per_frame))
        if end > start:
            energy[frame_index] = float(
                np.sqrt(np.mean(np.square(audio[start:end], dtype=np.float64)))
            )
    if frame_count >= 9:
        kernel = np.hanning(9).astype(np.float32)
        kernel /= kernel.sum()
        energy = np.convolve(energy, kernel, mode="same")
    active = energy[energy > 1e-5]
    ceiling = float(np.quantile(active, 0.95)) if active.size else 1.0
    return np.clip(energy / max(ceiling, 1e-5), 0, 1)


def _natural_emotion_sequence(
    *,
    emotion: int,
    frame_count: int,
    fps: int,
    speech_energy: np.ndarray,
) -> np.ndarray:
    """Create a slowly changing semantic expression without scaling lip motion."""
    frame = np.arange(frame_count, dtype=np.float32)
    time_s = frame / fps
    edge_frames = max(1, round(0.5 * fps))
    edge = np.ones(frame_count, dtype=np.float32)
    if frame_count > edge_frames * 2:
        ramp = np.sin(np.linspace(0, np.pi / 2, edge_frames)) ** 2
        edge[:edge_frames] = ramp
        edge[-edge_frames:] = ramp[::-1]

    if emotion == EMOTIONS["neutral"]:
        target_weight = 0.08 + 0.06 * speech_energy
    else:
        slow_variation = 0.5 + 0.5 * np.sin(2 * np.pi * time_s / 5.8 - 0.7)
        target_weight = (
            0.18
            + 0.17 * speech_energy
            + 0.10 * slow_variation * speech_energy
        )
    target_weight *= edge
    target_weight = np.clip(target_weight, 0, 0.48)

    sequence = np.zeros((frame_count, len(EMOTIONS)), dtype=np.float32)
    sequence[:, EMOTIONS["neutral"]] = 1 - target_weight
    sequence[:, emotion] += target_weight
    return sequence


def _natural_head_controls(
    *,
    frame_count: int,
    fps: int,
    speech_energy: np.ndarray,
    scale: float,
) -> dict[int, dict[str, float]]:
    """Add continuous sub-degree pose drift while preserving audio lip motion."""
    frame = np.arange(frame_count, dtype=np.float32)
    time_s = frame / fps
    edge = np.sin(np.linspace(0, np.pi, frame_count, dtype=np.float32)) ** 2
    activity = edge * (0.35 + 0.65 * speech_energy)
    yaw = scale * activity * (
        0.85 * np.sin(2 * np.pi * time_s / 6.2)
        + 0.22 * np.sin(2 * np.pi * time_s / 2.7 + 0.8)
    )
    pitch = scale * activity * (
        0.38 * np.sin(2 * np.pi * time_s / 4.9 + 1.2)
        + 0.10 * np.sin(2 * np.pi * time_s / 2.2)
    )
    roll = scale * activity * 0.22 * np.sin(2 * np.pi * time_s / 7.1 - 0.5)
    return {
        frame_index: {
            "delta_yaw": float(yaw[frame_index]),
            "delta_pitch": float(pitch[frame_index]),
            "delta_roll": float(roll[frame_index]),
        }
        for frame_index in range(frame_count)
    }


def run(
    *,
    data_root: Path,
    config: Path,
    source: Path,
    audio_path: Path,
    output: Path,
    emotion: int,
    seed: int,
    blink_open_frames: int,
    expression_scale: float,
    motion_profile: str,
    head_motion_scale: float,
    online: bool,
    realtime_feed: bool,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    audio, _ = librosa.load(str(audio_path), sr=16_000)
    frame_count = math.ceil(len(audio) / 16_000 * 25)
    speech_energy = _smooth_speech_energy(
        audio,
        sample_rate=16_000,
        fps=25,
        frame_count=frame_count,
    )
    emotion_condition: int | np.ndarray = emotion
    ctrl_info: dict[int, dict[str, float]] | None = None
    if motion_profile == "natural":
        emotion_condition = _natural_emotion_sequence(
            emotion=emotion,
            frame_count=frame_count,
            fps=25,
            speech_energy=speech_energy,
        )
        ctrl_info = _natural_head_controls(
            frame_count=frame_count,
            fps=25,
            speech_energy=speech_energy,
            scale=head_motion_scale,
        )

    sdk_class = StreamSDK
    if online:
        from stream_pipeline_online import StreamSDK as OnlineStreamSDK

        sdk_class = OnlineStreamSDK
    model_load_started = time.perf_counter()
    sdk = sdk_class(str(config), str(data_root))
    model_loaded_at = time.perf_counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    setup_kwargs: dict[str, object] = {
        "emo": emotion_condition,
        "online_mode": online,
        "delta_eye_open_n": blink_open_frames,
    }
    if expression_scale != 1:
        setup_kwargs["use_d_keys"] = {
            "exp": expression_scale,
            "pitch": 1,
            "yaw": 1,
            "roll": 1,
            "t": 1,
        }
    sdk.setup(str(source), str(output), **setup_kwargs)
    avatar_ready_at = time.perf_counter()
    writer_probe = _WriterProbe(sdk.writer)
    sdk.writer = writer_probe

    sdk.setup_Nd(N_d=frame_count, ctrl_info=ctrl_info)
    feed_started_at = time.perf_counter()
    if online:
        chunksize = (3, 5, 2)
        padded_audio = np.concatenate(
            [np.zeros((chunksize[0] * 640,), dtype=np.float32), audio]
        )
        split_len = int(sum(chunksize) * 0.04 * 16_000) + 80
        for offset in range(0, len(padded_audio), chunksize[1] * 640):
            audio_chunk = padded_audio[offset : offset + split_len]
            if len(audio_chunk) < split_len:
                audio_chunk = np.pad(
                    audio_chunk,
                    (0, split_len - len(audio_chunk)),
                    mode="constant",
                )
            sdk.run_chunk(audio_chunk, chunksize)
            if realtime_feed:
                time.sleep(chunksize[1] * 0.04)
    else:
        audio_features = sdk.wav2feat.wav2feat(audio)
        sdk.audio2motion_queue.put(audio_features)
    sdk.close()
    render_finished_at = time.perf_counter()

    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(sdk.tmp_output_path),
            "-i",
            str(audio_path),
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            str(output),
        ],
        check=True,
    )
    print(
        json.dumps(
            {
                "model_load_s": round(model_loaded_at - model_load_started, 3),
                "avatar_setup_s": round(avatar_ready_at - model_loaded_at, 3),
                "first_frame_from_feed_s": (
                    round(writer_probe.first_frame_at - feed_started_at, 3)
                    if writer_probe.first_frame_at is not None
                    else None
                ),
                "feed_to_render_done_s": round(
                    render_finished_at - feed_started_at,
                    3,
                ),
                "audio_duration_s": round(len(audio) / 16_000, 3),
                "frames": writer_probe.frames,
                "online": online,
                "realtime_feed": realtime_feed,
            },
            ensure_ascii=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an isolated Ditto emotion-control candidate."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--emotion",
        choices=sorted(EMOTIONS),
        required=True,
    )
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument(
        "--blink-open-frames",
        type=int,
        default=-1,
        help=(
            "Frames to keep the eyes open between generated blinks. "
            "The official default -1 disables blink generation."
        ),
    )
    parser.add_argument(
        "--expression-scale",
        type=float,
        default=1,
        help="Scale expression deltas relative to the first generated frame.",
    )
    parser.add_argument(
        "--motion-profile",
        choices=("none", "natural"),
        default="none",
        help="Add speech-aware semantic expression and subtle pose drift.",
    )
    parser.add_argument(
        "--head-motion-scale",
        type=float,
        default=1,
        help="Scale only the natural profile's sub-degree head motion.",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Use Ditto's streaming HuBERT pipeline and chunked audio input.",
    )
    parser.add_argument(
        "--realtime-feed",
        action="store_true",
        help="Pace online chunks at their 200 ms wall-clock interval.",
    )
    args = parser.parse_args()
    if args.blink_open_frames == 0 or args.blink_open_frames < -1:
        parser.error("--blink-open-frames must be -1 or a positive integer")
    if not 0.5 <= args.expression_scale <= 1.5:
        parser.error("--expression-scale must be between 0.5 and 1.5")
    if not 0 <= args.head_motion_scale <= 2:
        parser.error("--head-motion-scale must be between 0 and 2")
    if args.realtime_feed and not args.online:
        parser.error("--realtime-feed requires --online")

    run(
        data_root=args.data_root,
        config=args.config,
        source=args.source,
        audio_path=args.audio,
        output=args.output,
        emotion=EMOTIONS[args.emotion],
        seed=args.seed,
        blink_open_frames=args.blink_open_frames,
        expression_scale=args.expression_scale,
        motion_profile=args.motion_profile,
        head_motion_scale=args.head_motion_scale,
        online=args.online,
        realtime_feed=args.realtime_feed,
    )


if __name__ == "__main__":
    main()
