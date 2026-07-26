#!/usr/bin/env python3
"""Generate a small OmniVoice paralinguistic comparison pack on Apple MLX."""

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
    ("02-laughter", "[laughter]你终于回来啦，我还以为你今天又要忙到很晚呢。"),
    (
        "03-question",
        "[question-en]你今天是不是有点累呀，要不要先陪我说会儿话？",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--steps", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    load_started = time.perf_counter()
    model = load(str(args.model))
    model_load_seconds = time.perf_counter() - load_started

    records = []
    for label, text in CASES:
        mx.random.seed(args.seed)
        started = time.perf_counter()
        result = next(
            model.generate(
                text=text,
                language="Chinese",
                ref_audio=str(args.reference),
                ref_text=REFERENCE_TEXT,
                num_steps=args.steps,
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
        "num_steps": args.steps,
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
