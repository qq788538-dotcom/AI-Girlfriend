from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT_PATH = Path(__file__).parents[1] / "scripts/analyze-face-motion.py"
SPEC = importlib.util.spec_from_file_location("analyze_face_motion", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_detect_blinks_uses_hysteresis_and_ignores_small_eye_motion() -> None:
    scores = np.asarray([0.1, 0.25, 0.55, 0.7, 0.3, 0.1, 0.25])
    timestamps = np.arange(len(scores)) / 25

    assert MODULE.detect_blinks(scores, timestamps) == [
        {"start_s": 0.08, "duration_s": 0.12, "peak": 0.7}
    ]


def test_detect_blinks_does_not_bridge_missing_detection_gap() -> None:
    scores = np.asarray([0.6, 0.7, 0.1])
    timestamps = np.asarray([0.0, 0.04, 0.4])

    assert MODULE.detect_blinks(scores, timestamps) == []


def test_classify_eye_closures_excludes_startup_and_prolonged_closure() -> None:
    blinks, prolonged, startup = MODULE.classify_eye_closures(
        [
            {"start_s": 0.0, "duration_s": 0.2, "peak": 0.7},
            {"start_s": 1.0, "duration_s": 0.16, "peak": 0.8},
            {"start_s": 2.0, "duration_s": 0.8, "peak": 0.9},
        ]
    )

    assert blinks == [{"start_s": 1.0, "duration_s": 0.16, "peak": 0.8}]
    assert prolonged == [{"start_s": 2.0, "duration_s": 0.8, "peak": 0.9}]
    assert startup == [{"start_s": 0.0, "duration_s": 0.2, "peak": 0.7}]


def test_rotation_matrix_to_euler_degrees_identity() -> None:
    result = MODULE.rotation_matrix_to_euler_degrees(np.eye(4))

    np.testing.assert_allclose(result, np.zeros(3), atol=1e-6)


def test_vector_activity_is_zero_for_static_expression() -> None:
    records = [{"browInnerUp": 0.2}] * 4
    timestamps = np.arange(4) / 25

    result = MODULE.vector_activity(
        records,
        timestamps,
        ["browInnerUp"],
        max_gap_s=0.12,
    )

    assert result["variability_rms"] == 0.0
    assert result["velocity_rms_per_s"] == 0.0
