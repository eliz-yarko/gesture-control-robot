"""Run classifiers on a dataset manifest and write benchmark-ready predictions."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AppConfig, VideoConfig  # noqa: E402
from src.domain import GestureID, GesturePrediction  # noqa: E402
from src.evaluation import EvaluationSample, PredictionRecord, read_manifest  # noqa: E402
from src.evaluation.manifest import write_prediction_records  # noqa: E402
from src.recognition import (
    DynamicGestureClassifier,
    HandDetector,
    SklearnDynamicGestureClassifier,
    SklearnStaticGestureClassifier,
    StaticGestureClassifier,
)  # noqa: E402
from src.recognition.trajectory_buffer import TrajectoryBuffer  # noqa: E402
from src.utils.geometry import Landmark  # noqa: E402
from src.utils.metrics import normalize_label  # noqa: E402


class SampleEvaluator:
    """Evaluate manifest samples with the project classifiers."""

    def __init__(
        self,
        config: AppConfig,
        classifier_mode: str,
        frame_stride: int,
        max_frames: int | None,
        mirror_frame: bool,
        static_model: Path | None = None,
        dynamic_model: Path | None = None,
    ) -> None:
        """Initialize reusable detector and classifier objects."""

        self._config = config
        self._classifier_mode = classifier_mode
        self._frame_stride = frame_stride
        self._max_frames = max_frames
        self._mirror_frame = mirror_frame
        self._cv2 = importlib.import_module("cv2")
        self._detector = HandDetector(config.hand_detection)
        self._static_classifier = (
            SklearnStaticGestureClassifier.load_path(static_model)
            if static_model is not None
            else StaticGestureClassifier(config.static_classifier)
        )
        self._dynamic_classifier = (
            SklearnDynamicGestureClassifier.load_path(dynamic_model)
            if dynamic_model is not None
            else DynamicGestureClassifier(config.dynamic_classifier)
        )

    def evaluate(self, sample: EvaluationSample) -> PredictionRecord:
        """Evaluate one manifest sample."""

        if sample.media_type == "image":
            result = self._evaluate_image(sample)
        elif sample.media_type == "video":
            result = self._evaluate_video(sample)
        elif sample.media_type == "landmarks":
            result = self._evaluate_landmarks(sample)
        else:
            raise ValueError(f"Unsupported media_type={sample.media_type!r}")

        prediction, latency_ms, fps, frame_count, note = result
        return PredictionRecord(
            sample_id=sample.sample_id,
            expected_gesture=sample.expected_gesture,
            predicted_gesture=prediction.gesture_id.name,
            confidence=prediction.confidence,
            latency_ms=latency_ms,
            fps=fps,
            dataset=sample.dataset,
            condition=sample.condition,
            distance=sample.distance,
            media_type=sample.media_type,
            source_path=str(sample.path),
            frame_count=frame_count,
            note=note,
        )

    def close(self) -> None:
        """Release detector resources."""

        self._detector.close()

    def _evaluate_image(
        self,
        sample: EvaluationSample,
    ) -> tuple[GesturePrediction, float, float | None, int, str]:
        frame = self._cv2.imread(str(sample.path))
        if frame is None:
            raise RuntimeError(f"Cannot read image: {sample.path}")
        if self._mirror_frame:
            frame = self._cv2.flip(frame, 1)
        rgb_frame = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)

        started_at = perf_counter()
        detections = self._detector.detect(rgb_frame)
        if not detections:
            elapsed_ms = (perf_counter() - started_at) * 1000
            return GesturePrediction.unknown("no_hand_detected"), elapsed_ms, None, 1, "no_hand"

        detection = max(detections, key=lambda item: item.score)
        prediction = self._static_classifier.classify(detection.landmarks)
        elapsed_ms = (perf_counter() - started_at) * 1000
        return prediction, elapsed_ms, None, 1, ""

    def _evaluate_video(
        self,
        sample: EvaluationSample,
    ) -> tuple[GesturePrediction, float | None, float | None, int, str]:
        capture = self._cv2.VideoCapture(str(sample.path))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {sample.path}")

        static_predictions: list[GesturePrediction] = []
        dynamic_predictions: list[GesturePrediction] = []
        buffer = TrajectoryBuffer(self._config.dynamic_classifier.buffer_size)
        processed_frames = 0
        decoded_frames = 0
        started_at = perf_counter()

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

                frame = self._prepare_video_frame(frame)
                rgb_frame = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
                detections = self._detector.detect(rgb_frame)
                processed_frames += 1

                if not detections:
                    static_predictions.append(GesturePrediction.unknown("no_hand_detected"))
                    continue

                detection = max(detections, key=lambda item: item.score)
                static_predictions.append(self._static_classifier.classify(detection.landmarks))
                buffer.add_landmarks(detection.landmarks)
                dynamic_predictions.append(self._dynamic_classifier.classify(buffer))
        finally:
            capture.release()

        elapsed_seconds = perf_counter() - started_at
        if processed_frames == 0:
            return GesturePrediction.unknown("empty_video"), None, None, 0, "empty_video"

        prediction = self._select_sequence_prediction(
            expected_label=sample.expected_gesture,
            static_predictions=static_predictions,
            dynamic_predictions=dynamic_predictions,
        )
        note = "" if prediction.gesture_id != GestureID.UNKNOWN else "no_sequence_prediction"
        return (
            prediction,
            (elapsed_seconds * 1000) / processed_frames,
            processed_frames / elapsed_seconds if elapsed_seconds > 0 else None,
            processed_frames,
            note,
        )

    def _evaluate_landmarks(
        self,
        sample: EvaluationSample,
    ) -> tuple[GesturePrediction, float | None, float | None, int, str]:
        payload = json.loads(sample.path.read_text(encoding="utf-8-sig"))
        frames = _extract_landmark_frames(payload)
        if not frames:
            return GesturePrediction.unknown("empty_landmarks"), None, None, 0, "empty_landmarks"

        if len(frames) == 1:
            started_at = perf_counter()
            prediction = self._static_classifier.classify(frames[0])
            elapsed_ms = (perf_counter() - started_at) * 1000
            return prediction, elapsed_ms, None, 1, ""

        static_predictions: list[GesturePrediction] = []
        dynamic_predictions: list[GesturePrediction] = []
        buffer = TrajectoryBuffer(self._config.dynamic_classifier.buffer_size)
        started_at = perf_counter()

        for frame in frames[: self._max_frames]:
            static_predictions.append(self._static_classifier.classify(frame))
            buffer.add_landmarks(frame)
            dynamic_predictions.append(self._dynamic_classifier.classify(buffer))

        elapsed_seconds = perf_counter() - started_at
        processed_frames = len(static_predictions)
        prediction = self._select_sequence_prediction(
            expected_label=sample.expected_gesture,
            static_predictions=static_predictions,
            dynamic_predictions=dynamic_predictions,
        )
        note = "" if prediction.gesture_id != GestureID.UNKNOWN else "no_sequence_prediction"
        return (
            prediction,
            (elapsed_seconds * 1000) / processed_frames if processed_frames else None,
            processed_frames / elapsed_seconds if elapsed_seconds > 0 else None,
            processed_frames,
            note,
        )

    def _prepare_video_frame(self, frame: Any) -> Any:
        resized = self._cv2.resize(
            frame,
            (self._config.video.frame_width, self._config.video.frame_height),
        )
        return self._cv2.flip(resized, 1) if self._mirror_frame else resized

    def _select_sequence_prediction(
        self,
        expected_label: str,
        static_predictions: Sequence[GesturePrediction],
        dynamic_predictions: Sequence[GesturePrediction],
    ) -> GesturePrediction:
        mode = self._classifier_mode
        if mode == "auto":
            mode = "dynamic" if _is_dynamic_label(expected_label) else "static"

        if mode == "dynamic":
            return _best_prediction(dynamic_predictions)
        if mode == "static":
            return _majority_prediction(static_predictions)
        return _best_prediction([*static_predictions, *dynamic_predictions])


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate image/video/landmark dataset samples and write predictions.csv "
            "for scripts/benchmark.py."
        )
    )
    parser.add_argument("--manifest", required=True, help="Evaluation manifest CSV.")
    parser.add_argument("--output", required=True, help="Output predictions CSV.")
    parser.add_argument(
        "--classifier-mode",
        choices=("auto", "static", "dynamic", "best"),
        default="auto",
        help="How to aggregate video/sequence predictions.",
    )
    parser.add_argument(
        "--frame-stride", type=int, default=1, help="Process every Nth video frame."
    )
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Max processed frames per sample."
    )
    parser.add_argument("--frame-width", type=int, default=640, help="Video resize width.")
    parser.add_argument("--frame-height", type=int, default=480, help="Video resize height.")
    parser.add_argument(
        "--static-model",
        type=str,
        default=None,
        help="Optional joblib model for static gesture classification.",
    )
    parser.add_argument(
        "--dynamic-model",
        type=str,
        default=None,
        help="Optional joblib model for dynamic gesture classification.",
    )
    parser.add_argument(
        "--mirror-frame", action="store_true", help="Mirror frames before detection."
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Write UNKNOWN rows for failed samples instead of stopping.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run manifest evaluation."""

    args = build_parser().parse_args(argv)
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")

    manifest_path = Path(args.manifest)
    samples = read_manifest(manifest_path)
    config = AppConfig(
        video=VideoConfig(
            frame_width=args.frame_width,
            frame_height=args.frame_height,
            mirror_frame=args.mirror_frame,
        )
    )
    evaluator = SampleEvaluator(
        config=config,
        classifier_mode=args.classifier_mode,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        mirror_frame=args.mirror_frame,
        static_model=Path(args.static_model) if args.static_model is not None else None,
        dynamic_model=Path(args.dynamic_model) if args.dynamic_model is not None else None,
    )
    records: list[PredictionRecord] = []

    try:
        for sample in samples:
            try:
                records.append(evaluator.evaluate(sample))
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                records.append(_failed_record(sample, exc))
    finally:
        evaluator.close()

    output_path = Path(args.output)
    write_prediction_records(output_path, records)
    print(f"Wrote predictions: {output_path}")
    return 0


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


def _is_dynamic_label(label: str) -> bool:
    normalized = normalize_label(label)
    try:
        return GestureID[normalized].is_dynamic
    except KeyError:
        return False


def _majority_prediction(predictions: Sequence[GesturePrediction]) -> GesturePrediction:
    known_predictions = [
        prediction for prediction in predictions if prediction.gesture_id != GestureID.UNKNOWN
    ]
    if not known_predictions:
        return GesturePrediction.unknown("no_known_static_prediction")

    counts = Counter(prediction.gesture_id for prediction in known_predictions)
    confidence_by_label: dict[GestureID, list[float]] = defaultdict(list)
    for prediction in known_predictions:
        confidence_by_label[prediction.gesture_id].append(prediction.confidence)

    selected_label, _ = counts.most_common(1)[0]
    mean_confidence = sum(confidence_by_label[selected_label]) / len(
        confidence_by_label[selected_label]
    )
    return GesturePrediction(selected_label, mean_confidence)


def _best_prediction(predictions: Sequence[GesturePrediction]) -> GesturePrediction:
    known_predictions = [
        prediction for prediction in predictions if prediction.gesture_id != GestureID.UNKNOWN
    ]
    if not known_predictions:
        return GesturePrediction.unknown("no_known_prediction")
    return max(known_predictions, key=lambda prediction: prediction.confidence)


def _failed_record(sample: EvaluationSample, exc: Exception) -> PredictionRecord:
    return PredictionRecord(
        sample_id=sample.sample_id,
        expected_gesture=sample.expected_gesture,
        predicted_gesture=GestureID.UNKNOWN.name,
        confidence=0.0,
        latency_ms=None,
        fps=None,
        dataset=sample.dataset,
        condition=sample.condition,
        distance=sample.distance,
        media_type=sample.media_type,
        source_path=str(sample.path),
        frame_count=0,
        note=f"error: {exc}",
    )


if __name__ == "__main__":
    raise SystemExit(main())
