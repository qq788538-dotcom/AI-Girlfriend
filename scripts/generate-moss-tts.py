#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_audio.tts import load
from scipy.io import wavfile


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a local MOSS-TTS Local v1.5 voice-clone candidate."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--reference-audio", type=Path, required=True)
    parser.add_argument("--reference-text", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("generation", "continuation"),
        default="continuation",
    )
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--audio-temperature", type=float, default=1.7)
    parser.add_argument("--audio-top-p", type=float, default=0.8)
    parser.add_argument("--audio-top-k", type=int, default=25)
    parser.add_argument("--instruction", default="")
    args = parser.parse_args()

    for path in (args.model, args.codec, args.reference_audio):
        if not path.exists():
            raise FileNotFoundError(path)

    mx.random.seed(args.seed)
    started_at = time.perf_counter()
    model = load(args.model, lazy=False)
    loaded_at = time.perf_counter()
    result = next(
        model.generate(
            text=args.text,
            ref_audio=str(args.reference_audio),
            ref_text=args.reference_text,
            mode=args.mode,
            language="Chinese",
            instruction=args.instruction or None,
            max_tokens=args.max_tokens,
            audio_temperature=args.audio_temperature,
            audio_top_p=args.audio_top_p,
            audio_top_k=args.audio_top_k,
            audio_repetition_penalty=1.0,
            audio_tokenizer_source=str(args.codec),
        )
    )
    finished_at = time.perf_counter()

    audio = np.asarray(result.audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.ndim != 2:
        raise ValueError(f"Expected [samples, channels] audio, got {audio.shape}")
    audio = np.nan_to_num(audio)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    pcm16 = np.round(np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(args.output, int(result.sample_rate), pcm16)
    report = {
        "model": str(args.model),
        "codec": str(args.codec),
        "mode": args.mode,
        "seed": args.seed,
        "text": args.text,
        "instruction": args.instruction,
        "reference_audio": str(args.reference_audio),
        "sample_rate": int(result.sample_rate),
        "channels": int(pcm16.shape[1]),
        "samples": int(pcm16.shape[0]),
        "duration_seconds": round(pcm16.shape[0] / float(result.sample_rate), 3),
        "model_load_seconds": round(loaded_at - started_at, 3),
        "generation_seconds": round(finished_at - loaded_at, 3),
        "peak_memory_gb": round(float(result.peak_memory_usage), 3),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
