#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np

SAMPLE_RATE = 24_000
FRAME_LENGTH = 2_048
HOP_LENGTH = 256


def load_audio(path: Path) -> np.ndarray:
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    return audio / peak if peak > 0 else audio


def active_mask(audio: np.ndarray) -> np.ndarray:
    rms = librosa.feature.rms(
        y=audio,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
    )[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    return rms_db > -38.0


def pitch_track(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f0, voiced, _ = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C3"),
        fmax=librosa.note_to_hz("C6"),
        sr=SAMPLE_RATE,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
    )
    valid = voiced & np.isfinite(f0)
    if not np.any(valid):
        return np.zeros_like(f0), valid
    indices = np.arange(len(f0))
    interpolated = np.interp(indices, indices[valid], f0[valid])
    return 12.0 * np.log2(np.maximum(interpolated, 1e-6)), valid


def dtw_mean(left: np.ndarray, right: np.ndarray, metric: str = "euclidean") -> float:
    cost, path = librosa.sequence.dtw(X=left, Y=right, metric=metric)
    return float(cost[-1, -1] / max(len(path), 1))


def harmonicity_db(audio: np.ndarray, f0: np.ndarray, voiced: np.ndarray) -> float:
    frames = librosa.util.frame(
        np.pad(audio, FRAME_LENGTH // 2),
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
    )
    count = min(frames.shape[1], len(f0))
    values: list[float] = []
    window = np.hanning(FRAME_LENGTH)
    for index in range(count):
        if not voiced[index]:
            continue
        frame = frames[:, index].astype(np.float64)
        frame = (frame - np.mean(frame)) * window
        energy = float(np.dot(frame, frame))
        if energy <= 1e-10:
            continue
        center_lag = int(round(SAMPLE_RATE / (2.0 ** (f0[index] / 12.0))))
        correlations = []
        for lag in range(max(1, center_lag - 2), center_lag + 3):
            if lag >= len(frame):
                continue
            correlations.append(float(np.dot(frame[:-lag], frame[lag:]) / energy))
        if not correlations:
            continue
        periodicity = float(np.clip(max(correlations), 1e-5, 0.99999))
        values.append(10.0 * np.log10(periodicity / (1.0 - periodicity)))
    return float(np.median(values)) if values else 0.0


def features(path: Path) -> dict[str, object]:
    audio = load_audio(path)
    active = active_mask(audio)
    f0_st, voiced = pitch_track(audio)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=20,
        n_fft=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
        n_mels=80,
    )
    rms = librosa.feature.rms(
        y=audio,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
    )[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    flatness = librosa.feature.spectral_flatness(
        y=audio,
        n_fft=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
    )[0]
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
        n_mels=80,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    count = min(
        mfcc.shape[1],
        len(rms_db),
        len(flatness),
        log_mel.shape[1],
        len(active),
    )
    active = active[:count]
    active_indices = np.flatnonzero(active)
    if not active_indices.size:
        active_indices = np.arange(count)

    return {
        "path": str(path),
        "duration_s": len(audio) / SAMPLE_RATE,
        "mfcc": mfcc[1:, :count],
        "rms_db": rms_db[:count][None, :],
        "f0_st": f0_st[None, :],
        "voiced": voiced,
        "mean_log_mel": np.mean(log_mel[:, :count][:, active_indices], axis=1),
        "spectral_flatness_median": float(np.median(flatness[:count][active_indices])),
        "harmonicity_db": harmonicity_db(audio, f0_st, voiced),
    }


def compare(reference: dict[str, object], candidate: dict[str, object]) -> dict[str, object]:
    ref_mfcc = np.asarray(reference["mfcc"])
    cand_mfcc = np.asarray(candidate["mfcc"])
    scale = np.std(ref_mfcc, axis=1, keepdims=True)
    scale = np.maximum(scale, 1.0)
    mfcc_dtw = dtw_mean(ref_mfcc / scale, cand_mfcc / scale)

    ref_f0 = np.asarray(reference["f0_st"])
    cand_f0 = np.asarray(candidate["f0_st"])
    pitch_dtw_cents = 100.0 * dtw_mean(ref_f0, cand_f0)

    ref_rms = np.asarray(reference["rms_db"])
    cand_rms = np.asarray(candidate["rms_db"])
    ref_rms = ref_rms - np.median(ref_rms)
    cand_rms = cand_rms - np.median(cand_rms)
    energy_dtw_db = dtw_mean(ref_rms, cand_rms)

    ref_mel = np.asarray(reference["mean_log_mel"])
    cand_mel = np.asarray(candidate["mean_log_mel"])
    ref_mel = ref_mel - np.mean(ref_mel)
    cand_mel = cand_mel - np.mean(cand_mel)
    mel_cosine = float(
        np.dot(ref_mel, cand_mel)
        / (np.linalg.norm(ref_mel) * np.linalg.norm(cand_mel) + 1e-9)
    )

    return {
        "file": candidate["path"],
        "duration_s": round(float(candidate["duration_s"]), 3),
        "mfcc_dtw": round(mfcc_dtw, 4),
        "pitch_contour_dtw_cents": round(pitch_dtw_cents, 1),
        "energy_envelope_dtw_db": round(energy_dtw_db, 3),
        "mean_log_mel_cosine": round(mel_cosine, 5),
        "harmonicity_db": round(float(candidate["harmonicity_db"]), 3),
        "harmonicity_error_db": round(
            abs(
                float(candidate["harmonicity_db"])
                - float(reference["harmonicity_db"])
            ),
            3,
        ),
        "spectral_flatness_median": round(
            float(candidate["spectral_flatness_median"]),
            7,
        ),
        "spectral_flatness_error": round(
            abs(
                float(candidate["spectral_flatness_median"])
                - float(reference["spectral_flatness_median"])
            ),
            7,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare same-text TTS with detailed timbre and prosody metrics."
    )
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("candidates", nargs="+", type=Path)
    args = parser.parse_args()

    reference = features(args.reference)
    results = [compare(reference, features(path)) for path in args.candidates]
    print(
        json.dumps(
            {
                "reference": {
                    "file": reference["path"],
                    "duration_s": round(float(reference["duration_s"]), 3),
                    "harmonicity_db": round(float(reference["harmonicity_db"]), 3),
                    "spectral_flatness_median": round(
                        float(reference["spectral_flatness_median"]),
                        7,
                    ),
                },
                "candidates": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
