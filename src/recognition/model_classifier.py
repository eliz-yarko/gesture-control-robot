"""scikit-learn based gesture classifiers."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from src.domain import GestureID, GesturePrediction
from src.recognition.landmark_features import (
    DYNAMIC_FEATURE_VERSION,
    STATIC_FEATURE_VERSION,
    extract_static_features,
    extract_trajectory_features,
)
from src.recognition.trajectory_buffer import TrajectoryBuffer
from src.utils.geometry import LandmarkSequence
from src.utils.metrics import normalize_label


@dataclass(frozen=True)
class ModelBundle:
    """Loaded sklearn model metadata."""

    model: Any
    label_names: tuple[str, ...]
    feature_version: str
    min_confidence: float
    model_type: str


class SklearnStaticGestureClassifier:
    """Classify static gestures with a trained sklearn landmark model."""

    def __init__(self, bundle: ModelBundle) -> None:
        """Initialize the classifier from a loaded model bundle."""

        if bundle.feature_version != STATIC_FEATURE_VERSION:
            raise ValueError(f"Unsupported static feature version: {bundle.feature_version!r}")
        self._bundle = bundle

    @classmethod
    def load_path(
        cls,
        path: str | Path,
        min_confidence: float | None = None,
    ) -> SklearnStaticGestureClassifier:
        """Load a static gesture classifier from a joblib file."""

        return cls(
            _load_bundle(
                Path(path),
                expected_feature_version=STATIC_FEATURE_VERSION,
                min_confidence=min_confidence,
            )
        )

    def classify(self, raw_landmarks: LandmarkSequence) -> GesturePrediction:
        """Classify a static gesture from MediaPipe hand landmarks."""

        features = extract_static_features(raw_landmarks)
        return _predict_from_features(self._bundle, features)


class SklearnDynamicGestureClassifier:
    """Classify dynamic gestures with a trained sklearn trajectory model."""

    def __init__(self, bundle: ModelBundle, min_points: int = 30) -> None:
        """Initialize the classifier from a loaded model bundle."""

        if bundle.feature_version != DYNAMIC_FEATURE_VERSION:
            raise ValueError(f"Unsupported dynamic feature version: {bundle.feature_version!r}")
        self._bundle = bundle
        self._min_points = min_points

    @classmethod
    def load_path(
        cls,
        path: str | Path,
        min_confidence: float | None = None,
        min_points: int = 30,
    ) -> SklearnDynamicGestureClassifier:
        """Load a dynamic gesture classifier from a joblib file."""

        return cls(
            _load_bundle(
                Path(path),
                expected_feature_version=DYNAMIC_FEATURE_VERSION,
                min_confidence=min_confidence,
            ),
            min_points=min_points,
        )

    def classify(self, buffer: TrajectoryBuffer) -> GesturePrediction:
        """Classify a dynamic gesture from the current trajectory buffer."""

        points = buffer.points()
        if len(points) < self._min_points:
            return GesturePrediction.unknown("trajectory_too_short")
        features = extract_trajectory_features(points)
        return _predict_from_features(self._bundle, features)


class FallbackStaticGestureClassifier:
    """Use a rule-based classifier when the model does not produce a gesture."""

    def __init__(self, primary: Any, fallback: Any) -> None:
        """Initialize the primary model classifier and fallback classifier."""

        self._primary = primary
        self._fallback = fallback

    def classify(self, raw_landmarks: LandmarkSequence) -> GesturePrediction:
        """Classify landmarks, falling back when the model returns UNKNOWN."""

        prediction = cast(GesturePrediction, self._primary.classify(raw_landmarks))
        if prediction.gesture_id != GestureID.UNKNOWN:
            return prediction
        fallback_prediction = cast(GesturePrediction, self._fallback.classify(raw_landmarks))
        return GesturePrediction(
            gesture_id=fallback_prediction.gesture_id,
            confidence=fallback_prediction.confidence,
            metadata={
                **fallback_prediction.metadata,
                "fallback_after": prediction.metadata.get("reason", "model_unknown"),
            },
        )


class FallbackDynamicGestureClassifier:
    """Use trajectory heuristics when the model does not produce a dynamic gesture."""

    def __init__(self, primary: Any, fallback: Any) -> None:
        """Initialize the primary model classifier and fallback classifier."""

        self._primary = primary
        self._fallback = fallback

    def classify(self, buffer: TrajectoryBuffer) -> GesturePrediction:
        """Classify a trajectory, falling back when the model returns UNKNOWN."""

        prediction = cast(GesturePrediction, self._primary.classify(buffer))
        if prediction.gesture_id != GestureID.UNKNOWN:
            return prediction
        fallback_prediction = cast(GesturePrediction, self._fallback.classify(buffer))
        return GesturePrediction(
            gesture_id=fallback_prediction.gesture_id,
            confidence=fallback_prediction.confidence,
            metadata={
                **fallback_prediction.metadata,
                "fallback_after": prediction.metadata.get("reason", "model_unknown"),
            },
        )


def _load_bundle(
    path: Path,
    expected_feature_version: str,
    min_confidence: float | None = None,
) -> ModelBundle:
    if not path.exists():
        raise FileNotFoundError(f"Model file does not exist: {path}")

    joblib = importlib.import_module("joblib")
    payload = joblib.load(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Model file must contain a dictionary bundle: {path}")

    feature_version = str(payload.get("feature_version", ""))
    if feature_version != expected_feature_version:
        raise ValueError(
            f"Expected feature version {expected_feature_version!r}, got {feature_version!r}"
        )

    model = payload.get("model")
    if model is None:
        raise ValueError(f"Model bundle does not contain a model: {path}")

    label_names = tuple(str(label) for label in payload.get("label_names", ()))
    if not label_names:
        classes = getattr(model, "classes_", ())
        label_names = tuple(str(label) for label in classes)
    if not label_names:
        raise ValueError(f"Model bundle does not contain label names: {path}")

    bundle_min_confidence = float(payload.get("min_confidence", 0.45))
    effective_min_confidence = (
        bundle_min_confidence
        if min_confidence is None
        else max(bundle_min_confidence, min_confidence)
    )

    return ModelBundle(
        model=model,
        label_names=label_names,
        feature_version=feature_version,
        min_confidence=effective_min_confidence,
        model_type=str(payload.get("model_type", "sklearn")),
    )


def _predict_from_features(bundle: ModelBundle, features: tuple[float, ...]) -> GesturePrediction:
    probabilities = bundle.model.predict_proba([features])[0]
    classes = tuple(str(label) for label in getattr(bundle.model, "classes_", bundle.label_names))
    if not classes:
        classes = bundle.label_names

    best_index = max(range(len(probabilities)), key=lambda index: float(probabilities[index]))
    confidence = float(probabilities[best_index])
    label = normalize_label(classes[best_index])
    if confidence < bundle.min_confidence:
        return GesturePrediction.unknown("model_confidence_below_threshold")

    try:
        gesture_id = GestureID[label]
    except KeyError:
        return GesturePrediction.unknown("model_unknown_label")

    return GesturePrediction(
        gesture_id=gesture_id,
        confidence=confidence,
        metadata={
            "classifier": bundle.model_type,
            "feature_version": bundle.feature_version,
        },
    )
