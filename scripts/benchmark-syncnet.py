#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from types import SimpleNamespace


def metric_violations(
    *,
    offset_frames: int,
    confidence: float,
    max_abs_offset_frames: int | None,
    min_confidence: float | None,
) -> list[str]:
    violations: list[str] = []
    if (
        max_abs_offset_frames is not None
        and abs(offset_frames) > max_abs_offset_frames
    ):
        violations.append(
            f"absolute AV offset {abs(offset_frames)} frames exceeds "
            f"{max_abs_offset_frames} frames"
        )
    if min_confidence is not None and confidence < min_confidence:
        violations.append(
            f"SyncNet confidence {confidence:.3f} is below {min_confidence:.3f}"
        )
    return violations


def benchmark(
    videos: list[Path],
    *,
    model: Path,
    tmp_dir: Path,
    batch_size: int,
    vshift: int,
    max_abs_offset_frames: int | None,
    min_confidence: float | None,
) -> dict[str, object]:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "runtime/syncnet/python"))
    sys.path.insert(0, str(project_root / "vendor/syncnet_python"))

    import numpy as np
    from SyncNetInstance import SyncNetInstance

    logging.getLogger("SyncNetInstance").setLevel(logging.WARNING)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    evaluator = SyncNetInstance(device="cpu")
    evaluator.loadParameters(str(model))
    results: list[dict[str, object]] = []

    for index, video in enumerate(videos):
        reference = re.sub(r"[^A-Za-z0-9_-]+", "-", video.stem).strip("-")
        reference = f"{index:02d}-{reference or 'video'}"
        options = SimpleNamespace(
            tmp_dir=str(tmp_dir),
            reference=reference,
            batch_size=batch_size,
            vshift=vshift,
        )
        offset, confidence, distances = evaluator.evaluate(options, str(video))
        offset_frames = int(offset)
        confidence_value = float(confidence)
        mean_distance = np.mean(distances, axis=0)
        min_distance = float(np.min(mean_distance))
        violations = metric_violations(
            offset_frames=offset_frames,
            confidence=confidence_value,
            max_abs_offset_frames=max_abs_offset_frames,
            min_confidence=min_confidence,
        )
        results.append(
            {
                "video": str(video),
                "offset_frames": offset_frames,
                "offset_ms": offset_frames * 40,
                "min_distance": round(min_distance, 3),
                "confidence": round(confidence_value, 3),
                "violations": violations,
            }
        )

    return {
        "model": str(model),
        "fps": 25,
        "gate": {
            "max_abs_offset_frames": max_abs_offset_frames,
            "min_confidence": min_confidence,
        },
        "status": (
            "ok"
            if all(not result["violations"] for result in results)
            else "failed"
        ),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate tracked face videos with the official Oxford SyncNet."
    )
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("vendor/syncnet_python/data/syncnet_v2.model"),
    )
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=Path("runtime/syncnet/tmp"),
    )
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--vshift", type=int, default=15)
    parser.add_argument("--max-abs-offset-frames", type=int)
    parser.add_argument("--min-confidence", type=float)
    args = parser.parse_args()

    missing = [path for path in [args.model, *args.videos] if not path.is_file()]
    if missing:
        parser.error("Missing file(s): " + ", ".join(str(path) for path in missing))
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.vshift <= 0:
        parser.error("--vshift must be positive")
    if args.max_abs_offset_frames is not None and args.max_abs_offset_frames < 0:
        parser.error("--max-abs-offset-frames must be non-negative")

    result = benchmark(
        args.videos,
        model=args.model,
        tmp_dir=args.tmp_dir,
        batch_size=args.batch_size,
        vshift=args.vshift,
        max_abs_offset_frames=args.max_abs_offset_frames,
        min_confidence=args.min_confidence,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
