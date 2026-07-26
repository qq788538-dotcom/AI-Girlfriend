#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


def detect_blinks(
    scores: np.ndarray,
    timestamps_s: np.ndarray,
    *,
    onset: float = 0.45,
    release: float = 0.20,
    max_gap_s: float = 0.12,
) -> list[dict[str, float]]:
    events: list[dict[str, float]] = []
    active = False
    started_at = 0.0
    peak = 0.0
    previous_time: float | None = None
    for score, timestamp in zip(scores, timestamps_s):
        score_value = float(score)
        timestamp_value = float(timestamp)
        if (
            previous_time is not None
            and timestamp_value - previous_time > max_gap_s
        ):
            active = False
        if not active and score_value >= onset:
            active = True
            started_at = timestamp_value
            peak = score_value
        elif active:
            peak = max(peak, score_value)
            if score_value <= release:
                events.append(
                    {
                        "start_s": round(started_at, 3),
                        "duration_s": round(timestamp_value - started_at, 3),
                        "peak": round(peak, 4),
                    }
                )
                active = False
        previous_time = timestamp_value
    return events


def classify_eye_closures(
    events: list[dict[str, float]],
) -> tuple[
    list[dict[str, float]],
    list[dict[str, float]],
    list[dict[str, float]],
]:
    blinks = [
        event
        for event in events
        if event["start_s"] >= 0.08 and 0.04 <= event["duration_s"] <= 0.5
    ]
    prolonged = [event for event in events if event["duration_s"] > 0.5]
    startup = [event for event in events if event["start_s"] < 0.08]
    return blinks, prolonged, startup


def rotation_matrix_to_euler_degrees(matrix: np.ndarray) -> np.ndarray:
    rotation = np.asarray(matrix, dtype=np.float64)[:3, :3]
    left, _, right = np.linalg.svd(rotation)
    rotation = left @ right
    sy = math.hypot(rotation[0, 0], rotation[1, 0])
    singular = sy < 1e-6
    if not singular:
        pitch = math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.atan2(-rotation[2, 0], sy)
        roll = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        pitch = math.atan2(-rotation[1, 2], rotation[1, 1])
        yaw = math.atan2(-rotation[2, 0], sy)
        roll = 0.0
    return np.degrees([pitch, yaw, roll])


def smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    if len(values) < window or window <= 1:
        return values.copy()
    padding = window // 2
    padded = np.pad(values, ((padding, padding), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    columns = [
        np.convolve(padded[:, index], kernel, mode="valid")
        for index in range(values.shape[1])
    ]
    return np.column_stack(columns)


def temporal_derivative(
    values: np.ndarray,
    timestamps_s: np.ndarray,
    *,
    max_gap_s: float,
) -> np.ndarray:
    if len(values) < 2:
        return np.empty((0, values.shape[1]), dtype=np.float64)
    delta_t = np.diff(timestamps_s)
    valid = (delta_t > 0) & (delta_t <= max_gap_s)
    if not np.any(valid):
        return np.empty((0, values.shape[1]), dtype=np.float64)
    return np.diff(values, axis=0)[valid] / delta_t[valid, None]


def vector_activity(
    records: list[dict[str, float]],
    timestamps_s: np.ndarray,
    names: list[str],
    *,
    max_gap_s: float,
) -> dict[str, float]:
    if not records or not names:
        return {
            "channels": len(names),
            "variability_rms": 0.0,
            "velocity_rms_per_s": 0.0,
            "velocity_p95_per_s": 0.0,
        }
    values = np.asarray(
        [[record.get(name, 0.0) for name in names] for record in records],
        dtype=np.float64,
    )
    centered = values - np.median(values, axis=0, keepdims=True)
    velocity = temporal_derivative(
        values,
        timestamps_s,
        max_gap_s=max_gap_s,
    )
    velocity_norm = (
        np.sqrt(np.mean(velocity**2, axis=1))
        if velocity.size
        else np.asarray([0.0])
    )
    return {
        "channels": len(names),
        "variability_rms": round(float(np.sqrt(np.mean(centered**2))), 5),
        "velocity_rms_per_s": round(float(np.sqrt(np.mean(velocity**2))), 5)
        if velocity.size
        else 0.0,
        "velocity_p95_per_s": round(float(np.quantile(velocity_norm, 0.95)), 5),
    }


def _blendshape_groups(names: list[str]) -> dict[str, list[str]]:
    return {
        "brow": [name for name in names if name.startswith("brow")],
        "eye_gaze": [name for name in names if name.startswith("eyeLook")],
        "upper_face": [
            name
            for name in names
            if name.startswith(("brow", "cheek", "eye", "noseSneer"))
        ],
        "mouth": [
            name
            for name in names
            if name.startswith(("jaw", "mouth", "tongue"))
        ],
        "non_mouth": [
            name
            for name in names
            if name != "_neutral"
            and not name.startswith(("jaw", "mouth", "tongue"))
        ],
    }


def _distance(left: Any, right: Any) -> float:
    return math.hypot(float(left.x) - float(right.x), float(left.y) - float(right.y))


def _eye_aperture(landmarks: list[Any]) -> tuple[float, float]:
    left_width = max(_distance(landmarks[33], landmarks[133]), 1e-6)
    right_width = max(_distance(landmarks[362], landmarks[263]), 1e-6)
    left = _distance(landmarks[159], landmarks[145]) / left_width
    right = _distance(landmarks[386], landmarks[374]) / right_width
    return left, right


def _pose_metrics(
    poses: np.ndarray,
    timestamps_s: np.ndarray,
    *,
    max_gap_s: float,
) -> dict[str, Any]:
    if not len(poses):
        return {}
    unwrapped = np.degrees(np.unwrap(np.radians(poses), axis=0))
    filtered = smooth(unwrapped, window=5)
    velocity = temporal_derivative(
        filtered,
        timestamps_s,
        max_gap_s=max_gap_s,
    )
    speed = np.linalg.norm(velocity, axis=1) if velocity.size else np.asarray([0.0])
    axes = ("pitch", "yaw", "roll")
    return {
        "std_degrees": {
            axis: round(float(np.std(unwrapped[:, index])), 4)
            for index, axis in enumerate(axes)
        },
        "range_degrees": {
            axis: round(float(np.ptp(unwrapped[:, index])), 4)
            for index, axis in enumerate(axes)
        },
        "angular_speed_median_deg_s": round(float(np.median(speed)), 4),
        "angular_speed_p95_deg_s": round(float(np.quantile(speed, 0.95)), 4),
    }


def analyze_video(
    video_path: Path,
    *,
    model_path: Path,
    confidence: float,
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "runtime/face-motion/python"))

    import cv2
    import mediapipe as mp

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_gap_s = max(0.12, 2.5 / fps)
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(model_path.resolve()),
            delegate=mp.tasks.BaseOptions.Delegate.CPU,
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=confidence,
        min_face_presence_confidence=confidence,
        min_tracking_confidence=confidence,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    timestamps: list[float] = []
    blendshape_records: list[dict[str, float]] = []
    poses: list[np.ndarray] = []
    eye_apertures: list[tuple[float, float]] = []
    frame_index = 0

    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp_ms = round(frame_index * 1000 / fps)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(image, timestamp_ms)
            if result.face_landmarks:
                categories = result.face_blendshapes[0]
                record = {
                    category.category_name: float(category.score)
                    for category in categories
                }
                timestamps.append(frame_index / fps)
                blendshape_records.append(record)
                eye_apertures.append(_eye_aperture(result.face_landmarks[0]))
                if result.facial_transformation_matrixes:
                    poses.append(
                        rotation_matrix_to_euler_degrees(
                            result.facial_transformation_matrixes[0]
                        )
                    )
                else:
                    poses.append(np.zeros(3, dtype=np.float64))
            frame_index += 1
    capture.release()

    detected_frames = len(blendshape_records)
    if not detected_frames:
        return {
            "video": str(video_path),
            "fps": round(fps, 3),
            "total_frames": frame_index or total_frames,
            "detected_frames": 0,
            "detection_coverage": 0.0,
            "error": "No face detected",
        }

    timestamp_array = np.asarray(timestamps, dtype=np.float64)
    names = sorted(
        {name for record in blendshape_records for name in record}
    )
    groups = _blendshape_groups(names)
    left_blink = np.asarray(
        [record.get("eyeBlinkLeft", 0.0) for record in blendshape_records]
    )
    right_blink = np.asarray(
        [record.get("eyeBlinkRight", 0.0) for record in blendshape_records]
    )
    bilateral_blink = (left_blink + right_blink) / 2
    blink_candidates = detect_blinks(
        bilateral_blink,
        timestamp_array,
        max_gap_s=max_gap_s,
    )
    blink_events, prolonged_closures, startup_transitions = classify_eye_closures(
        blink_candidates
    )
    duration_s = frame_index / fps
    apertures = np.asarray(eye_apertures, dtype=np.float64)
    jaw_open = np.asarray(
        [record.get("jawOpen", 0.0) for record in blendshape_records]
    )
    smiles = np.asarray(
        [
            [
                record.get("mouthSmileLeft", 0.0),
                record.get("mouthSmileRight", 0.0),
            ]
            for record in blendshape_records
        ]
    )
    dynamic_shapes = sorted(
        (
            (
                name,
                float(
                    np.std(
                        [record.get(name, 0.0) for record in blendshape_records]
                    )
                ),
            )
            for name in names
            if name != "_neutral"
        ),
        key=lambda item: item[1],
        reverse=True,
    )[:10]

    return {
        "video": str(video_path),
        "fps": round(fps, 3),
        "duration_s": round(duration_s, 3),
        "total_frames": frame_index or total_frames,
        "detected_frames": detected_frames,
        "detection_coverage": round(detected_frames / max(frame_index, 1), 4),
        "blink": {
            "events": len(blink_events),
            "events_per_min": round(len(blink_events) / duration_s * 60, 3),
            "mean_duration_s": round(
                float(np.mean([event["duration_s"] for event in blink_events])),
                4,
            )
            if blink_events
            else 0.0,
            "score_p95": round(float(np.quantile(bilateral_blink, 0.95)), 4),
            "score_max": round(float(np.max(bilateral_blink)), 4),
            "left_right_correlation": round(
                float(np.corrcoef(left_blink, right_blink)[0, 1]),
                4,
            )
            if np.std(left_blink) > 1e-6 and np.std(right_blink) > 1e-6
            else 0.0,
            "events_detail": blink_events,
            "prolonged_eye_closures": prolonged_closures,
            "startup_transitions_ignored": startup_transitions,
        },
        "landmark_eye_aperture": {
            "mean": round(float(np.mean(apertures)), 5),
            "minimum": round(float(np.min(apertures)), 5),
            "p05": round(float(np.quantile(apertures, 0.05)), 5),
        },
        "head_pose": _pose_metrics(
            np.asarray(poses, dtype=np.float64),
            timestamp_array,
            max_gap_s=max_gap_s,
        ),
        "expression_activity": {
            name: vector_activity(
                blendshape_records,
                timestamp_array,
                group_names,
                max_gap_s=max_gap_s,
            )
            for name, group_names in groups.items()
        },
        "jaw_open": {
            "mean": round(float(np.mean(jaw_open)), 4),
            "std": round(float(np.std(jaw_open)), 4),
            "p95": round(float(np.quantile(jaw_open, 0.95)), 4),
        },
        "smile": {
            "mean": round(float(np.mean(smiles)), 4),
            "p95": round(float(np.quantile(smiles, 0.95)), 4),
            "asymmetry_mean": round(
                float(np.mean(np.abs(smiles[:, 0] - smiles[:, 1]))),
                4,
            ),
        },
        "most_dynamic_blendshapes": [
            {"name": name, "std": round(value, 5)}
            for name, value in dynamic_shapes
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Quantify blink, head pose, gaze, brow, and mouth motion with the "
            "official MediaPipe Face Landmarker."
        )
    )
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("runtime/face-motion/models/face_landmarker.task"),
    )
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    missing = [path for path in [args.model, *args.videos] if not path.is_file()]
    if missing:
        parser.error("Missing file(s): " + ", ".join(str(path) for path in missing))
    if not 0 < args.confidence <= 1:
        parser.error("--confidence must be in (0, 1]")

    results = [
        analyze_video(
            video,
            model_path=args.model,
            confidence=args.confidence,
        )
        for video in args.videos
    ]
    report = {
        "model": str(args.model),
        "confidence": args.confidence,
        "results": results,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
