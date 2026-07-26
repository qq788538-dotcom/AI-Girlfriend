#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path


def load_analyzer():
    analyzer_path = Path(__file__).with_name("analyze-voice.py")
    spec = importlib.util.spec_from_file_location("voice_analyzer", analyzer_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {analyzer_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.analyze


def semitone_distance(left: float, right: float) -> float:
    if left <= 0 or right <= 0:
        return 12.0
    return abs(12.0 * math.log2(left / right))


def compare(reference: dict[str, object], candidate: dict[str, object]) -> dict[str, object]:
    ref_duration = float(reference["duration_s"])
    duration_error = abs(float(candidate["duration_s"]) - ref_duration) / ref_duration
    median_pitch_error = semitone_distance(
        float(candidate["f0_median_hz"]),
        float(reference["f0_median_hz"]),
    )
    pitch_span_error = abs(
        float(candidate["pitch_span_semitones"])
        - float(reference["pitch_span_semitones"])
    )
    centroid_error = abs(
        float(candidate["spectral_centroid_median_hz"])
        - float(reference["spectral_centroid_median_hz"])
    ) / float(reference["spectral_centroid_median_hz"])
    energy_error = abs(
        float(candidate["rms_active_std_db"])
        - float(reference["rms_active_std_db"])
    )
    speech_ratio_error = abs(
        float(candidate["speech_ratio"]) - float(reference["speech_ratio"])
    )
    pause_overrun_ms = max(0.0, float(candidate["pause_max_ms"]) - 250.0)

    distance = (
        1.3 * duration_error / 0.20
        + 1.0 * speech_ratio_error / 0.15
        + 1.2 * median_pitch_error / 3.0
        + 1.0 * pitch_span_error / 8.0
        + 0.8 * centroid_error / 0.25
        + 0.7 * energy_error / 3.0
        + 0.8 * pause_overrun_ms / 500.0
    )
    reject_reasons = []
    if duration_error > 0.35:
        reject_reasons.append("duration")
    if float(candidate["speech_ratio"]) < 0.75:
        reject_reasons.append("speech_ratio")
    # A long pause is valid when the same-text reference contains it. Reject
    # only an unexplained overrun, not a faithfully cloned natural pause.
    if float(candidate["pause_max_ms"]) > max(
        800.0,
        float(reference["pause_max_ms"]) + 250.0,
    ):
        reject_reasons.append("long_pause")
    if median_pitch_error > 7:
        reject_reasons.append("median_pitch")
    if float(candidate["pitch_span_semitones"]) < 6:
        reject_reasons.append("flat_pitch")

    return {
        **candidate,
        "acoustic_match_score": round(100.0 * math.exp(-distance), 2),
        "duration_error_ratio": round(duration_error, 3),
        "median_pitch_error_semitones": round(median_pitch_error, 2),
        "pitch_span_error_semitones": round(pitch_span_error, 2),
        "spectral_centroid_error_ratio": round(centroid_error, 3),
        "rejected": bool(reject_reasons),
        "reject_reasons": reject_reasons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank TTS candidates against a same-text reference."
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("candidates", nargs="+", type=Path)
    args = parser.parse_args()

    analyze = load_analyzer()
    reference = analyze(args.reference)
    results = [compare(reference, analyze(path)) for path in args.candidates]
    results.sort(
        key=lambda item: (
            bool(item["rejected"]),
            -float(item["acoustic_match_score"]),
        )
    )
    print(
        json.dumps(
            {"reference": reference, "ranking": results},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
