"""Train lightweight sklearn gesture models from a benchmark manifest."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AppConfig, VideoConfig  # noqa: E402
from src.domain import GestureID  # noqa: E402
from src.evaluation import EvaluationSample, read_manifest  # noqa: E402
from src.recognition.hand_detector import HandDetector  # noqa: E402
from src.recognition.landmark_features import (  # noqa: E402
    DYNAMIC_FEATURE_VERSION,
    STATIC_FEATURE_VERSION,
    extract_static_features,
    extract_trajectory_features,
)
from src.recognition.trajectory_buffer import TrajectoryBuffer, TrajectoryPoint  # noqa: E402
from src.utils.geometry import Landmark  # noqa: E402
from src.utils.metrics import normalize_label  # noqa: E402


@dataclass(frozen=True)
class TrainingData:
    """Feature matrix with labels and sample groups."""

    features: list[tuple[float, ...]]
    labels: list[str]
    groups: list[str]


class GestureModelTrainer:
    """Extract landmarks from manifest media files and train gesture models."""

    def __init__(
        self,
        config: AppConfig,
        frame_stride: int,
        max_frames: int | None,
        min_dynamic_window_points: int,
        dynamic_augmentation_copies: int,
        dynamic_positive_min_window_ratio: float,
        random_state: int,
    ) -> None:
        """Initialize reusable OpenCV and MediaPipe resources."""

        self._config = config
        self._frame_stride = frame_stride
        self._max_frames = max_frames
        self._min_dynamic_window_points = min_dynamic_window_points
        self._dynamic_augmentation_copies = dynamic_augmentation_copies
        self._dynamic_positive_min_window_ratio = dynamic_positive_min_window_ratio
        self._random_state = random_state
        self._cv2 = importlib.import_module("cv2")
        self._detector = HandDetector(config.hand_detection)

    def close(self) -> None:
        """Release MediaPipe resources."""

        self._detector.close()

    def build_static_data(self, samples: Sequence[EvaluationSample]) -> TrainingData:
        """Build frame-level data for static gesture classification."""

        features: list[tuple[float, ...]] = []
        labels: list[str] = []
        groups: list[str] = []

        for sample in samples:
            label = normalize_label(sample.expected_gesture)
            if not _is_static_or_unknown(label):
                continue
            for landmarks in self._extract_landmark_frames(sample):
                features.append(extract_static_features(landmarks))
                labels.append(label)
                groups.append(sample.sample_id)

        return TrainingData(features, labels, groups)

    def build_dynamic_data(self, samples: Sequence[EvaluationSample]) -> TrainingData:
        """Build trajectory-window data for dynamic gesture classification."""

        features: list[tuple[float, ...]] = []
        labels: list[str] = []
        groups: list[str] = []

        for sample in samples:
            label = normalize_label(sample.expected_gesture)
            is_dynamic_sample = _is_dynamic_label(label)
            points = self._extract_trajectory_points(sample)
            if len(points) < self._min_dynamic_window_points:
                continue
            for window_index, window in enumerate(
                _trajectory_windows(
                    points,
                    self._config.dynamic_classifier.buffer_size,
                    min_size=self._min_dynamic_window_points,
                )
            ):
                target_label = _dynamic_window_label(
                    label,
                    is_dynamic_sample=is_dynamic_sample,
                    window_size=len(window),
                    full_size=len(points),
                    positive_min_window_ratio=self._dynamic_positive_min_window_ratio,
                    min_size=self._min_dynamic_window_points,
                )
                for augmented_window in _augmented_trajectory_windows(
                    window,
                    copies=self._dynamic_augmentation_copies,
                    random_state=self._random_state,
                    sample_id=sample.sample_id,
                    window_index=window_index,
                ):
                    features.append(extract_trajectory_features(augmented_window))
                    labels.append(target_label)
                    groups.append(sample.sample_id)

        return TrainingData(features, labels, groups)

    def _extract_landmark_frames(
        self, sample: EvaluationSample
    ) -> list[list[tuple[float, float, float]]]:
        if sample.media_type == "video":
            return self._extract_landmarks_from_video(sample)
        if sample.media_type == "image":
            return self._extract_landmarks_from_image(sample)
        if sample.media_type == "landmarks":
            return _extract_landmark_frames(json.loads(sample.path.read_text(encoding="utf-8")))
        return []

    def _extract_landmarks_from_image(
        self, sample: EvaluationSample
    ) -> list[list[tuple[float, float, float]]]:
        frame = self._cv2.imread(str(sample.path))
        if frame is None:
            return []
        frame = self._prepare_frame(frame)
        rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        detections = self._detector.detect(rgb)
        if not detections:
            return []
        return [max(detections, key=lambda item: item.score).landmarks]

    def _extract_landmarks_from_video(
        self, sample: EvaluationSample
    ) -> list[list[tuple[float, float, float]]]:
        capture = self._cv2.VideoCapture(str(sample.path))
        if not capture.isOpened():
            return []

        frames: list[list[tuple[float, float, float]]] = []
        decoded_frames = 0
        processed_frames = 0
        try:
            while self._max_frames is None or processed_frames < self._max_frames:
                success, frame = capture.read()
                if not success:
                    break
                decoded_frames += 1
                if sample.start_frame is not None and decoded_frames < sample.start_frame:
                    continue
                if sample.end_frame is not None and decoded_frames > sample.end_frame:
                    break
                segment_start = sample.start_frame or 1
                if (decoded_frames - segment_start) % self._frame_stride != 0:
                    continue

                frame = self._prepare_frame(frame)
                rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
                detections = self._detector.detect(rgb)
                processed_frames += 1
                if detections:
                    frames.append(max(detections, key=lambda item: item.score).landmarks)
        finally:
            capture.release()
        return frames

    def _extract_trajectory_points(self, sample: EvaluationSample) -> list[Any]:
        buffer = TrajectoryBuffer(max_size=10_000)
        for landmarks in self._extract_landmark_frames(sample):
            buffer.add_landmarks(landmarks)
        return buffer.points()

    def _prepare_frame(self, frame: Any) -> Any:
        resized = self._cv2.resize(
            frame,
            (self._config.video.frame_width, self._config.video.frame_height),
        )
        return self._cv2.flip(resized, 1) if self._config.video.mirror_frame else resized


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""

    parser = argparse.ArgumentParser(description="Train sklearn gesture models.")
    parser.add_argument("--manifest", required=True, help="Training manifest CSV.")
    parser.add_argument(
        "--static-output",
        default="models/static_gesture_classifier.joblib",
        help="Output path for the static landmark model.",
    )
    parser.add_argument(
        "--dynamic-output",
        default="models/dynamic_gesture_classifier.joblib",
        help="Output path for the dynamic trajectory model.",
    )
    parser.add_argument("--frame-stride", type=int, default=5, help="Process every Nth frame.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=90,
        help="Maximum processed frames per sample.",
    )
    parser.add_argument("--frame-width", type=int, default=480, help="Video resize width.")
    parser.add_argument("--frame-height", type=int, default=640, help="Video resize height.")
    parser.add_argument("--mirror-frame", action="store_true", help="Mirror frames.")
    parser.add_argument("--random-state", type=int, default=42, help="sklearn random seed.")
    parser.add_argument("--static-min-confidence", type=float, default=0.35)
    parser.add_argument("--dynamic-min-confidence", type=float, default=0.35)
    parser.add_argument(
        "--min-dynamic-window-points",
        type=int,
        default=8,
        help="Minimum trajectory length used when generating dynamic training windows.",
    )
    parser.add_argument(
        "--dynamic-unknown-ratio",
        type=float,
        default=1.5,
        help="Maximum UNKNOWN trajectory windows relative to the largest dynamic class.",
    )
    parser.add_argument(
        "--dynamic-augmentation-copies",
        type=int,
        default=0,
        help="Deterministic jittered copies to add for each dynamic trajectory window.",
    )
    parser.add_argument(
        "--dynamic-positive-min-window-ratio",
        type=float,
        default=0.0,
        help=(
            "When > 0, derived windows from dynamic samples shorter than this fraction "
            "of the full trajectory are labeled UNKNOWN for train-only pretrigger control."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Train static and dynamic gesture models."""

    args = build_parser().parse_args(argv)
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive when provided")
    if args.min_dynamic_window_points <= 0:
        raise ValueError("--min-dynamic-window-points must be positive")
    if args.dynamic_augmentation_copies < 0:
        raise ValueError("--dynamic-augmentation-copies must be non-negative")
    if args.dynamic_positive_min_window_ratio < 0.0 or args.dynamic_positive_min_window_ratio > 1.0:
        raise ValueError("--dynamic-positive-min-window-ratio must be in [0, 1]")

    samples = read_manifest(Path(args.manifest))
    config = AppConfig(
        video=VideoConfig(
            frame_width=args.frame_width,
            frame_height=args.frame_height,
            mirror_frame=args.mirror_frame,
        )
    )
    trainer = GestureModelTrainer(
        config=config,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        min_dynamic_window_points=args.min_dynamic_window_points,
        dynamic_augmentation_copies=args.dynamic_augmentation_copies,
        dynamic_positive_min_window_ratio=args.dynamic_positive_min_window_ratio,
        random_state=args.random_state,
    )
    try:
        static_data = trainer.build_static_data(samples)
        dynamic_data = _cap_unknown_class(
            trainer.build_dynamic_data(samples),
            unknown_ratio=args.dynamic_unknown_ratio,
            random_state=args.random_state,
        )
    finally:
        trainer.close()

    static_model = _train_classifier(static_data, args.random_state)
    dynamic_model = _train_classifier(dynamic_data, args.random_state)

    _save_model(
        output_path=Path(args.static_output),
        model=static_model,
        data=static_data,
        model_type="static_landmark_extra_trees",
        feature_version=STATIC_FEATURE_VERSION,
        min_confidence=args.static_min_confidence,
    )
    _save_model(
        output_path=Path(args.dynamic_output),
        model=dynamic_model,
        data=dynamic_data,
        model_type="dynamic_trajectory_extra_trees",
        feature_version=DYNAMIC_FEATURE_VERSION,
        min_confidence=args.dynamic_min_confidence,
    )

    _print_summary("static", static_data)
    _print_summary("dynamic", dynamic_data)
    print(f"Wrote static model: {args.static_output}")
    print(f"Wrote dynamic model: {args.dynamic_output}")
    return 0


def _train_classifier(data: TrainingData, random_state: int) -> Any:
    if not data.features:
        raise ValueError("Cannot train a model without features.")
    if len(set(data.labels)) < 2:
        raise ValueError("Cannot train a model with fewer than two labels.")

    ensemble = importlib.import_module("sklearn.ensemble")
    model = ensemble.ExtraTreesClassifier(
        n_estimators=240,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(data.features, data.labels)
    return model


def _cap_unknown_class(
    data: TrainingData,
    unknown_ratio: float,
    random_state: int,
) -> TrainingData:
    if unknown_ratio <= 0:
        return data

    label_counts = Counter(data.labels)
    dynamic_counts = [
        count for label, count in label_counts.items() if label != GestureID.UNKNOWN.name
    ]
    if not dynamic_counts or GestureID.UNKNOWN.name not in label_counts:
        return data

    max_unknown = max(1, int(max(dynamic_counts) * unknown_ratio))
    if label_counts[GestureID.UNKNOWN.name] <= max_unknown:
        return data

    random = importlib.import_module("random")
    rng = random.Random(random_state)
    unknown_indices = [
        index for index, label in enumerate(data.labels) if label == GestureID.UNKNOWN.name
    ]
    kept_unknown_indices = set(rng.sample(unknown_indices, max_unknown))
    kept_indices = [
        index
        for index, label in enumerate(data.labels)
        if label != GestureID.UNKNOWN.name or index in kept_unknown_indices
    ]
    return TrainingData(
        features=[data.features[index] for index in kept_indices],
        labels=[data.labels[index] for index in kept_indices],
        groups=[data.groups[index] for index in kept_indices],
    )


def _save_model(
    output_path: Path,
    model: Any,
    data: TrainingData,
    model_type: str,
    feature_version: str,
    min_confidence: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib = importlib.import_module("joblib")
    joblib.dump(
        {
            "model": model,
            "model_type": model_type,
            "feature_version": feature_version,
            "label_names": tuple(sorted(set(data.labels))),
            "min_confidence": min_confidence,
            "class_thresholds": {},
            "sample_count": len(data.labels),
            "group_count": len(set(data.groups)),
            "class_counts": dict(Counter(data.labels)),
        },
        output_path,
    )


def _print_summary(name: str, data: TrainingData) -> None:
    counts = ", ".join(f"{label}={count}" for label, count in sorted(Counter(data.labels).items()))
    group_counts: dict[str, set[str]] = defaultdict(set)
    for label, group in zip(data.labels, data.groups, strict=True):
        group_counts[label].add(group)
    group_summary = ", ".join(
        f"{label}={len(groups)}" for label, groups in sorted(group_counts.items())
    )
    print(f"{name} frames/windows: {len(data.labels)}")
    print(f"{name} class counts: {counts}")
    print(f"{name} sample groups: {group_summary}")


def _augmented_trajectory_windows(
    window: list[Any],
    *,
    copies: int,
    random_state: int,
    sample_id: str,
    window_index: int,
) -> list[list[Any]]:
    windows: list[list[Any]] = [window]
    if copies <= 0 or not all(isinstance(point, TrajectoryPoint) for point in window):
        return windows

    trajectory_window = [point for point in window if isinstance(point, TrajectoryPoint)]
    for copy_index in range(copies):
        rng = random.Random(f"{random_state}:{sample_id}:{window_index}:{copy_index}")
        windows.append(_jitter_trajectory_window(trajectory_window, rng))
    return windows


def _jitter_trajectory_window(
    window: Sequence[TrajectoryPoint],
    rng: random.Random,
) -> list[TrajectoryPoint]:
    if not window:
        return []

    origin = window[0].palm_center
    reference_size = max(sum(point.hand_size for point in window) / len(window), 1e-9)
    scale = rng.uniform(0.94, 1.06)
    translation = tuple(rng.uniform(-0.025, 0.025) * reference_size for _ in range(3))
    jitter_radius = reference_size * 0.015

    jittered: list[TrajectoryPoint] = []
    for point in window:
        palm = _jitter_point(point.palm_center, origin, scale, translation, jitter_radius, rng)
        index_tip = _jitter_point(point.index_tip, origin, scale, translation, jitter_radius, rng)
        jittered.append(
            TrajectoryPoint(
                palm_center=palm,
                index_tip=index_tip,
                hand_size=max(point.hand_size * scale * rng.uniform(0.97, 1.03), 1e-9),
                timestamp=point.timestamp,
            )
        )
    return jittered


def _jitter_point(
    point: tuple[float, float, float],
    origin: tuple[float, float, float],
    scale: float,
    translation: tuple[float, float, float],
    jitter_radius: float,
    rng: random.Random,
) -> tuple[float, float, float]:
    return tuple(
        origin[axis]
        + (point[axis] - origin[axis]) * scale
        + translation[axis]
        + rng.uniform(-jitter_radius, jitter_radius)
        for axis in range(3)
    )


def _trajectory_windows(
    points: list[Any],
    max_size: int,
    min_size: int = 8,
) -> list[list[Any]]:
    if len(points) < min_size:
        return []

    full_size = min(len(points), max_size)
    candidate_sizes = sorted(
        {
            min_size,
            max(min_size, min(full_size, len(points) // 2)),
            max(min_size, min(full_size, (len(points) * 3) // 4)),
            full_size,
        }
    )
    windows: list[list[Any]] = []
    seen: set[tuple[int, int]] = set()
    for size in candidate_sizes:
        for start in _window_starts(len(points), size):
            key = (start, size)
            if key in seen:
                continue
            seen.add(key)
            windows.append(points[start : start + size])

    return windows[:12]


def _window_starts(point_count: int, size: int) -> list[int]:
    if point_count <= size:
        return [0]
    last_start = point_count - size
    return sorted({0, last_start // 2, last_start})


def _dynamic_window_label(
    label: str,
    *,
    is_dynamic_sample: bool,
    window_size: int,
    full_size: int,
    positive_min_window_ratio: float,
    min_size: int,
) -> str:
    if not is_dynamic_sample:
        return GestureID.UNKNOWN.name
    if positive_min_window_ratio <= 0.0:
        return label

    positive_min_size = max(min_size, math.ceil(full_size * positive_min_window_ratio))
    return label if window_size >= positive_min_size else GestureID.UNKNOWN.name


def _is_static_or_unknown(label: str) -> bool:
    try:
        gesture = GestureID[label]
    except KeyError:
        return False
    return gesture.is_static or gesture == GestureID.UNKNOWN


def _is_dynamic_label(label: str) -> bool:
    try:
        return GestureID[label].is_dynamic
    except KeyError:
        return False


def _extract_landmark_frames(payload: Any) -> list[list[Landmark]]:
    if isinstance(payload, dict):
        if "frames" in payload:
            return _extract_landmark_frames(payload["frames"])
        if "landmarks_sequence" in payload:
            return _extract_landmark_frames(payload["landmarks_sequence"])
        if "landmarks" in payload:
            return _extract_landmark_frames(payload["landmarks"])

    if _is_landmark_frame(payload):
        return [_to_landmark_frame(payload)]

    if isinstance(payload, list):
        frames: list[list[Landmark]] = []
        for item in payload:
            if isinstance(item, dict) and "landmarks" in item:
                frames.append(_to_landmark_frame(item["landmarks"]))
            elif _is_landmark_frame(item):
                frames.append(_to_landmark_frame(item))
        return frames

    raise ValueError("Unsupported landmarks JSON format")


def _is_landmark_frame(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 21
        and all(isinstance(point, list | tuple) and len(point) >= 2 for point in value)
    )


def _to_landmark_frame(value: Any) -> list[Landmark]:
    return [
        (
            float(point[0]),
            float(point[1]),
            float(point[2]) if len(point) > 2 else 0.0,
        )
        for point in value
    ]


if __name__ == "__main__":
    raise SystemExit(main())
