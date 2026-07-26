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
        description="Generate multiple local Higgs TTS candidates with one model load."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference-audio", type=Path, required=True)
    parser.add_argument("--reference-text", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    for path in (args.model, args.reference_audio, args.manifest):
        if not path.exists():
            raise FileNotFoundError(path)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Manifest must contain a non-empty 'cases' list")

    batch_started_at = time.perf_counter()
    model = load(args.model, lazy=False, model_type="higgs_audio_v3")
    model_loaded_at = time.perf_counter()
    reports: list[dict[str, object]] = []

    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"Case {index} must be an object")
        label = str(case["label"])
        text = str(case["text"])
        controls = str(case.get("controls", ""))
        output = Path(str(case["output"]))
        seed = int(case.get("seed", 20260816 + index))

        mx.random.seed(seed)
        generation_started_at = time.perf_counter()
        result = next(
            model.generate(
                text=controls + text,
                ref_audio=str(args.reference_audio),
                ref_text=args.reference_text,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=args.max_tokens,
            )
        )
        generation_finished_at = time.perf_counter()

        audio = np.asarray(result.audio, dtype=np.float32)
        if audio.ndim == 1:
            audio = audio[:, None]
        if audio.ndim != 2:
            raise ValueError(
                f"Expected [samples, channels] audio for {label}, got {audio.shape}"
            )
        audio = np.nan_to_num(audio)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1.0:
            audio = audio / peak
        pcm16 = np.round(np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)

        output.parent.mkdir(parents=True, exist_ok=True)
        wavfile.write(output, int(result.sample_rate), pcm16)
        report = {
            "label": label,
            "intent": case.get("intent", label),
            "model": str(args.model),
            "seed": seed,
            "temperature": args.temperature,
            "controls": controls,
            "text": text,
            "reference_audio": str(args.reference_audio),
            "output": str(output),
            "sample_rate": int(result.sample_rate),
            "channels": int(pcm16.shape[1]),
            "samples": int(pcm16.shape[0]),
            "duration_seconds": round(
                pcm16.shape[0] / float(result.sample_rate), 3
            ),
            "generation_seconds": round(
                generation_finished_at - generation_started_at, 3
            ),
            "peak_memory_gb": round(float(result.peak_memory_usage), 3),
        }
        output.with_suffix(".json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        reports.append(report)
        print(json.dumps(report, ensure_ascii=False))

    summary = {
        "model": str(args.model),
        "reference_audio": str(args.reference_audio),
        "model_load_seconds": round(model_loaded_at - batch_started_at, 3),
        "batch_seconds": round(time.perf_counter() - batch_started_at, 3),
        "cases": reports,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
