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
        description="Generate a local Higgs TTS 3 MLX voice-clone candidate."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference-audio", type=Path, required=True)
    parser.add_argument("--reference-text", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--controls", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    for path in (args.model, args.reference_audio):
        if not path.exists():
            raise FileNotFoundError(path)

    mx.random.seed(args.seed)
    started_at = time.perf_counter()
    # The upstream checkpoint reports ``higgs_multimodal_qwen3`` while the
    # MLX-Audio backend is named ``higgs_audio_v3``.  Pass the backend
    # explicitly to avoid an upstream remapping bug for local Path objects.
    model = load(args.model, lazy=False, model_type="higgs_audio_v3")
    loaded_at = time.perf_counter()
    result = next(
        model.generate(
            text=args.controls + args.text,
            ref_audio=str(args.reference_audio),
            ref_text=args.reference_text,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=args.max_tokens,
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
        "seed": args.seed,
        "temperature": args.temperature,
        "controls": args.controls,
        "text": args.text,
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
