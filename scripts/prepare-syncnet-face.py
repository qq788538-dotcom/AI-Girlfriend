#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def median_filter(values: np.ndarray, kernel_size: int) -> np.ndarray:
    radius = kernel_size // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, kernel_size)
    return np.median(windows, axis=1)


def detect_track(
    frames: list[np.ndarray],
    *,
    cascade_path: Path | None = None,
) -> tuple[np.ndarray, int]:
    resolved_cascade = cascade_path or (
        Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    )
    if not resolved_cascade.is_file():
        raise FileNotFoundError(
            "OpenCV face cascade is unavailable; pass --cascade with "
            "haarcascade_frontalface_default.xml"
        )
    detector = cv2.CascadeClassifier(str(resolved_cascade))
    boxes = np.full((len(frames), 4), np.nan, dtype=np.float64)
    previous: np.ndarray | None = None
    detected = 0

    for index, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        candidates = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
        )
        if len(candidates) == 0:
            continue
        candidate_array = np.asarray(candidates, dtype=np.float64)
        if previous is None:
            areas = candidate_array[:, 2] * candidate_array[:, 3]
            chosen = candidate_array[int(np.argmax(areas))]
        else:
            centers = candidate_array[:, :2] + candidate_array[:, 2:] / 2
            previous_center = previous[:2] + previous[2:] / 2
            scale = max(previous[2], previous[3], 1)
            center_distance = np.linalg.norm(centers - previous_center, axis=1) / scale
            size_distance = np.abs(
                np.log(
                    np.maximum(candidate_array[:, 2], candidate_array[:, 3])
                    / max(previous[2], previous[3], 1)
                )
            )
            chosen = candidate_array[int(np.argmin(center_distance + size_distance))]
        boxes[index] = chosen
        previous = chosen
        detected += 1

    if detected < 2:
        raise RuntimeError(f"Only {detected} face detections; cannot build a stable track")

    timeline = np.arange(len(frames))
    for column in range(4):
        valid = np.flatnonzero(~np.isnan(boxes[:, column]))
        boxes[:, column] = np.interp(timeline, valid, boxes[valid, column])

    kernel = min(13, len(frames) if len(frames) % 2 else len(frames) - 1)
    if kernel >= 3:
        boxes = np.stack(
            [median_filter(boxes[:, column], kernel) for column in range(4)],
            axis=1,
        )
    return boxes, detected


def crop_face(frame: np.ndarray, box: np.ndarray) -> np.ndarray:
    x, y, width, height = box
    half_size = max(width, height) / 2
    center_x = x + width / 2
    center_y = y + height / 2
    # Match the official SyncNet face-track crop: 40% side padding and
    # additional space below the detected face so the full jaw is retained.
    left = int(round(center_x - half_size * 1.4))
    right = int(round(center_x + half_size * 1.4))
    top = int(round(center_y - half_size))
    bottom = int(round(center_y + half_size * 1.8))

    pad = max(0, -left, -top, right - frame.shape[1], bottom - frame.shape[0])
    if pad:
        frame = np.pad(
            frame,
            ((pad, pad), (pad, pad), (0, 0)),
            mode="constant",
            constant_values=110,
        )
        left += pad
        right += pad
        top += pad
        bottom += pad
    face = frame[top:bottom, left:right]
    if face.size == 0:
        raise RuntimeError("Tracked face crop is empty")
    return cv2.resize(face, (224, 224), interpolation=cv2.INTER_AREA)


def prepare(
    source: Path,
    output: Path,
    *,
    start: float,
    duration: float | None,
    cascade_path: Path | None,
) -> dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="syncnet-face-") as temp_dir:
        temp = Path(temp_dir)
        normalized = temp / "normalized.avi"
        silent = temp / "face-silent.avi"
        normalize_command = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            str(start),
            "-i",
            str(source),
        ]
        if duration is not None:
            normalize_command.extend(["-t", str(duration)])
        normalize_command.extend(
            [
                "-an",
                "-r",
                "25",
                "-c:v",
                "mjpeg",
                "-q:v",
                "2",
                str(normalized),
            ]
        )
        run(normalize_command)

        capture = cv2.VideoCapture(str(normalized))
        frames: list[np.ndarray] = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        capture.release()
        if len(frames) < 10:
            raise RuntimeError(f"Only {len(frames)} normalized frames")

        boxes, detected = detect_track(frames, cascade_path=cascade_path)
        writer = cv2.VideoWriter(
            str(silent),
            cv2.VideoWriter_fourcc(*"MJPG"),
            25,
            (224, 224),
        )
        if not writer.isOpened():
            raise RuntimeError("Could not open SyncNet face video writer")
        for frame, box in zip(frames, boxes):
            writer.write(crop_face(frame, box))
        writer.release()

        mux_command = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(silent),
            "-ss",
            str(start),
            "-i",
            str(source),
        ]
        if duration is not None:
            mux_command.extend(["-t", str(duration)])
        mux_command.extend(
            [
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-shortest",
                str(output),
            ]
        )
        run(mux_command)

    mean_box = np.mean(boxes, axis=0)
    return {
        "source": str(source),
        "output": str(output),
        "start_seconds": start,
        "duration_seconds": duration,
        "frames": len(frames),
        "face_detection_coverage": round(detected / len(frames), 4),
        "mean_face_box": {
            "x": round(float(mean_box[0]), 2),
            "y": round(float(mean_box[1]), 2),
            "width": round(float(mean_box[2]), 2),
            "height": round(float(mean_box[3]), 2),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a tracked 224x224 face video for official SyncNet evaluation."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--cascade", type=Path)
    args = parser.parse_args()
    if not args.source.is_file():
        parser.error(f"Source video does not exist: {args.source}")
    if args.start < 0:
        parser.error("--start must be non-negative")
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be positive")
    if args.cascade is not None and not args.cascade.is_file():
        parser.error(f"Face cascade does not exist: {args.cascade}")
    print(
        json.dumps(
            prepare(
                args.source,
                args.output,
                start=args.start,
                duration=args.duration,
                cascade_path=args.cascade,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
