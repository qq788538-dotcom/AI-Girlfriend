#!/usr/bin/env python3
"""Generate a small LongCat-AudioDiT voice-clone comparison pack on Apple MLX."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf
from mlx_audio.tts.utils import load

REFERENCE_TEXT = (
    "要是你跟别人聊天，我会吃醋的哦。我都开始胡思乱想了。"
    "我不知道什么是皮老板，只是你不在的时候，我就一直等你。"
)

CASES = (
    ("01-neutral", "你终于回来啦，我还以为你今天又要忙到很晚呢。"),
    (
        "02-playful-jealousy",
        "你刚才跟谁聊得那么开心呀，我可都看见了，好啦，我没有真的生气，就是想让你多陪陪我嘛。",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--steps", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ref_audio, sample_rate = sf.read(args.reference, dtype="float32")
    if sample_rate != 24_000:
        raise ValueError(f"Reference must be 24 kHz, got {sample_rate}")
    if ref_audio.ndim != 1:
        ref_audio = ref_audio.mean(axis=1)

    load_started = time.perf_counter()
    model = load(str(args.model))
    model_load_seconds = time.perf_counter() - load_started

    records = []
    for label, text in CASES:
        started = time.perf_counter()
        result = next(
            model.generate(
                text=text,
                lang_code="zh",
                ref_audio=ref_audio,
                ref_text=REFERENCE_TEXT,
                steps=args.steps,
                cfg_strength=4.0,
                guidance_method="apg",
                seed=args.seed,
            )
        )
        mx.eval(result.audio)
        generation_seconds = time.perf_counter() - started
        output = args.output_dir / f"{label}.wav"
        sf.write(output, np.asarray(result.audio), result.sample_rate)
        records.append(
            {
                "label": label,
                "text": text,
                "output": str(output),
                "sample_rate": result.sample_rate,
                "duration_seconds": len(result.audio) / result.sample_rate,
                "generation_seconds": generation_seconds,
                "peak_memory_gb": mx.get_peak_memory() / 1e9,
            }
        )

    manifest = {
        "model": str(args.model),
        "reference_audio": str(args.reference),
        "reference_text": REFERENCE_TEXT,
        "seed": args.seed,
        "steps": args.steps,
        "guidance_method": "apg",
        "cfg_strength": 4.0,
        "model_load_seconds": model_load_seconds,
        "cases": records,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
