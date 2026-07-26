#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np


def percentile(values: np.ndarray, value: float) -> float:
    return float(np.percentile(values, value)) if values.size else 0.0


def analyze(path: Path) -> dict[str, object]:
    audio, sample_rate = librosa.load(path, sr=24_000, mono=True)
    duration = len(audio) / sample_rate
    frame_length = 2048
    hop_length = 256

    rms = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length,
    )[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    active_rms_db = rms_db[rms_db > -38]

    f0, voiced_flag, _ = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C3"),
        fmax=librosa.note_to_hz("C6"),
        sr=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    voiced_f0 = f0[voiced_flag & np.isfinite(f0)]
    f0_p10 = percentile(voiced_f0, 10)
    f0_p90 = percentile(voiced_f0, 90)
    pitch_span_semitones = (
        float(12 * np.log2(f0_p90 / f0_p10))
        if f0_p10 > 0 and f0_p90 > f0_p10
        else 0.0
    )

    intervals = librosa.effects.split(audio, top_db=38)
    speech_seconds = float(sum((end - start) / sample_rate for start, end in intervals))
    pauses = [
        (right[0] - left[1]) / sample_rate
        for left, right in zip(intervals, intervals[1:])
        if right[0] > left[1]
    ]
    meaningful_pauses = [pause for pause in pauses if pause >= 0.12]

    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sample_rate,
        n_fft=frame_length,
        hop_length=hop_length,
    )[0]

    return {
        "file": str(path),
        "duration_s": round(duration, 3),
        "speech_ratio": round(speech_seconds / duration, 3) if duration else 0.0,
        "pause_count_ge_120ms": len(meaningful_pauses),
        "pause_mean_ms": (
            round(float(np.mean(meaningful_pauses)) * 1000, 1)
            if meaningful_pauses
            else 0.0
        ),
        "pause_max_ms": (
            round(float(np.max(meaningful_pauses)) * 1000, 1)
            if meaningful_pauses
            else 0.0
        ),
        "f0_median_hz": round(float(np.median(voiced_f0)), 1) if voiced_f0.size else 0.0,
        "f0_p10_hz": round(f0_p10, 1),
        "f0_p90_hz": round(f0_p90, 1),
        "pitch_span_semitones": round(pitch_span_semitones, 2),
        "voiced_ratio": round(float(np.mean(voiced_flag)), 3),
        "rms_active_std_db": (
            round(float(np.std(active_rms_db)), 2) if active_rms_db.size else 0.0
        ),
        "rms_active_p10_db": round(percentile(active_rms_db, 10), 2),
        "rms_active_p90_db": round(percentile(active_rms_db, 90), 2),
        "spectral_centroid_median_hz": round(float(np.median(centroid)), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare objective speech prosody metrics.")
    parser.add_argument("audio", nargs="+", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            [analyze(path) for path in args.audio],
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
