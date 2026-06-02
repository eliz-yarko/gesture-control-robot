"""Train lightweight sklearn gesture models from a benchmark manifest."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import importlib
import json
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
from src.recognition.trajectory_buffer import TrajectoryBuffer  # noqa: E402
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
    ) -> None:
        """Initialize reusable OpenCV and MediaPipe resources."""

        self._config = config
        self._frame_stride = frame_stride
        self._max_frames = max_frames
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
            target_label = label if _is_dynamic_label(label) else GestureID.UNKNOWN.name
            points = self._extract_trajectory_points(sample)
            if len(points) < 8:
                continue
            for window in _trajectory_windows(points, self._config.dynamic_classifier.buffer_size):
                features.append(extract_trajectory_features(window))
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
        "--dynamic-unknown-ratio",
        type=float,
        default=1.5,
        help="Maximum UNKNOWN trajectory windows relative to the largest dynamic class.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Train static and dynamic gesture models."""

    args = build_parser().parse_args(argv)
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive when provided")

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


def _trajectory_windows(points: list[Any], max_size: int) -> list[list[Any]]:
    if len(points) <= max_size:
        return [points]

    windows: list[list[Any]] = []
    step = max(1, (len(points) - max_size) // 6)
    for start in range(0, len(points) - max_size + 1, step):
        windows.append(points[start : start + max_size])
    return windows[:8]


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
