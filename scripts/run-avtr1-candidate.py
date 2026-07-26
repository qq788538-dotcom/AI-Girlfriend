#!/usr/bin/env python3
"""Evaluate AVTR-1 with the approved avatar and dual conversational audio.

This is an adapter around the official AVTR-1 ``Pipeline`` API:
https://github.com/avaturn-live/avtr-1
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import soundfile as sf
import soxr
from avtr1_renderer.pipeline import Pipeline
from avtr1_renderer.types import Chunk, RenderOptions

LOCKED_AVATAR_SHA256 = (
    "c8afa1d691711330e525db0048dd21e37ab71c46476a7504d0e75b4ae8fcd162"
)
SAMPLE_RATE = 16_000
FPS = 25


def verify_locked_avatar(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != LOCKED_AVATAR_SHA256:
        raise ValueError(
            "AVTR-1 candidate refused an unapproved avatar image: "
            f"{path} has SHA-256 {digest}"
        )


def load_audio(path: Path | None) -> np.ndarray:
    if path is None:
        return np.zeros(0, dtype=np.float32)
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        mono = soxr.resample(mono, sample_rate, SAMPLE_RATE, quality="HQ")
    return np.asarray(mono, dtype=np.float32)


def align_tracks(
    speech: np.ndarray,
    listen: np.ndarray,
    *,
    duration: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    if duration is not None:
        target_samples = round(duration * SAMPLE_RATE)
    else:
        target_samples = max(len(speech), len(listen))
    if target_samples <= 0:
        raise ValueError("Provide --speech, --listen, or a positive --duration")

    def fit(audio: np.ndarray) -> np.ndarray:
        if len(audio) >= target_samples:
            return audio[:target_samples]
        return np.pad(audio, (0, target_samples - len(audio)))

    return fit(speech), fit(listen)


def audio_chunks(
    audio: np.ndarray,
    *,
    window_samples: int,
    step_samples: int,
) -> list[np.ndarray]:
    count = max(1, math.ceil(len(audio) / step_samples))
    chunks: list[np.ndarray] = []
    for index in range(count):
        start = index * step_samples
        chunk = audio[start : start + window_samples]
        chunks.append(np.pad(chunk, (0, window_samples - len(chunk))))
    return chunks


def run(args: argparse.Namespace) -> None:
    verify_locked_avatar(args.portrait)
    speech, listen = align_tracks(
        load_audio(args.speech),
        load_audio(args.listen),
        duration=args.duration,
    )

    avatar_id = args.portrait.stem
    background = args.background or args.portrait
    pipeline, registry = Pipeline.from_artifacts(
        avatar_ids=[avatar_id],
        portraits_dir=args.portrait.parent,
        background_paths={"locked": background},
        out_size=(args.size, args.size),
    )
    avatar = registry[avatar_id]
    motion_generator = pipeline._motion_generator
    window_samples = (
        motion_generator.chunk_size + motion_generator.future_size
    ) * motion_generator.frame_len + motion_generator.audio_shift
    step_samples = motion_generator.chunk_size * motion_generator.frame_len
    speech_chunks = audio_chunks(
        speech,
        window_samples=window_samples,
        step_samples=step_samples,
    )
    listen_chunks = audio_chunks(
        listen,
        window_samples=window_samples,
        step_samples=step_samples,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(args.output),
        size=(args.size, args.size),
        fps=FPS,
        codec="libx264",
        pix_fmt_in="yuv420p",
        pix_fmt_out="yuv420p",
        quality=8,
        macro_block_size=1,
        audio_path=str(args.speech) if args.speech else None,
        audio_codec="aac" if args.speech else None,
    )
    writer.send(None)
    options = RenderOptions(
        pixel_format="yuv_i420",
        bg_id="locked",
        stream_frames=True,
    )
    state = None
    try:
        for speech_chunk, listen_chunk in zip(
            speech_chunks,
            listen_chunks,
            strict=True,
        ):
            state, frames = pipeline.process_chunk(
                avatar,
                Chunk(
                    audio_speech=speech_chunk,
                    audio_listen=listen_chunk,
                ),
                state,
                options,
            )
            for frame in frames:
                writer.send(frame.data.tobytes())
    finally:
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run AVTR-1 with the locked girlfriend portrait and separate "
            "speaking/listening audio tracks."
        )
    )
    parser.add_argument("--portrait", type=Path, required=True)
    parser.add_argument("--speech", type=Path)
    parser.add_argument("--listen", type=Path)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--background", type=Path)
    parser.add_argument("--size", type=int, default=720)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for path in [
        args.portrait,
        args.speech,
        args.listen,
        args.background,
    ]:
        if path is not None and not path.is_file():
            parser.error(f"Input does not exist: {path}")
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be positive")
    if args.size < 512 or args.size % 2:
        parser.error("--size must be an even integer of at least 512")
    run(args)


if __name__ == "__main__":
    main()
