"""scikit-learn based gesture classifiers."""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from src.domain import GestureID, GesturePrediction
from src.recognition.landmark_features import (
    DYNAMIC_FEATURE_VERSION,
    DYNAMIC_FEATURE_VERSION_V1,
    DYNAMIC_FEATURE_VERSION_V2,
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
    class_thresholds: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ThresholdProfile:
    """Optional confidence thresholds loaded from a tuning JSON file."""

    min_confidence: float | None
    class_thresholds: dict[str, float]


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
        threshold_profile: ThresholdProfile | None = None,
    ) -> SklearnStaticGestureClassifier:
        """Load a static gesture classifier from a joblib file."""

        return cls(
            _load_bundle(
                Path(path),
                expected_feature_version=STATIC_FEATURE_VERSION,
                min_confidence=min_confidence,
                threshold_profile=threshold_profile,
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

        if bundle.feature_version not in {
            DYNAMIC_FEATURE_VERSION,
            DYNAMIC_FEATURE_VERSION_V1,
            DYNAMIC_FEATURE_VERSION_V2,
        }:
            raise ValueError(f"Unsupported dynamic feature version: {bundle.feature_version!r}")
        self._bundle = bundle
        self._min_points = min_points

    @classmethod
    def load_path(
        cls,
        path: str | Path,
        min_confidence: float | None = None,
        min_points: int = 30,
        threshold_profile: ThresholdProfile | None = None,
    ) -> SklearnDynamicGestureClassifier:
        """Load a dynamic gesture classifier from a joblib file."""

        return cls(
            _load_bundle(
                Path(path),
                expected_feature_version=(
                    DYNAMIC_FEATURE_VERSION,
                    DYNAMIC_FEATURE_VERSION_V2,
                    DYNAMIC_FEATURE_VERSION_V1,
                ),
                min_confidence=min_confidence,
                threshold_profile=threshold_profile,
            ),
            min_points=min_points,
        )

    def classify(self, buffer: TrajectoryBuffer) -> GesturePrediction:
        """Classify a dynamic gesture from the current trajectory buffer."""

        points = buffer.points()
        if len(points) < self._min_points:
            return GesturePrediction.unknown("trajectory_too_short")
        features = extract_trajectory_features(points, feature_version=self._bundle.feature_version)
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


class ConfidenceFallbackStaticGestureClassifier:
    """Use a secondary classifier when the primary static classifier is weak."""

    def __init__(
        self,
        primary: Any,
        fallback: Any,
        primary_min_confidence: float,
    ) -> None:
        """Initialize confidence-gated primary/fallback classifiers."""

        if primary_min_confidence < 0.0 or primary_min_confidence > 1.0:
            raise ValueError("primary_min_confidence must be in [0, 1]")
        self._primary = primary
        self._fallback = fallback
        self._primary_min_confidence = primary_min_confidence

    def classify(self, raw_landmarks: LandmarkSequence) -> GesturePrediction:
        """Classify landmarks with primary priority and secondary fallback."""

        primary_prediction = cast(GesturePrediction, self._primary.classify(raw_landmarks))
        if (
            primary_prediction.gesture_id != GestureID.UNKNOWN
            and primary_prediction.confidence >= self._primary_min_confidence
        ):
            return primary_prediction

        fallback_prediction = cast(GesturePrediction, self._fallback.classify(raw_landmarks))
        return GesturePrediction(
            gesture_id=fallback_prediction.gesture_id,
            confidence=fallback_prediction.confidence,
            metadata={
                **fallback_prediction.metadata,
                "fallback_after": primary_prediction.metadata.get(
                    "reason",
                    primary_prediction.gesture_id.name,
                ),
                "primary_confidence": primary_prediction.confidence,
                "primary_gesture": primary_prediction.gesture_id.name,
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
        if prediction.metadata.get("reason") == "trajectory_too_short":
            return prediction
        if prediction.gesture_id != GestureID.UNKNOWN:
            fallback_prediction = cast(GesturePrediction, self._fallback.classify(buffer))
            if _should_prefer_dynamic_fallback(prediction, fallback_prediction):
                return GesturePrediction(
                    gesture_id=fallback_prediction.gesture_id,
                    confidence=fallback_prediction.confidence,
                    metadata={
                        **fallback_prediction.metadata,
                        "overrode_model": prediction.gesture_id.name,
                        "model_confidence": prediction.confidence,
                    },
                )
            return prediction
        fallback_prediction = cast(GesturePrediction, self._fallback.classify(buffer))
        if not _should_use_dynamic_fallback_after_unknown(prediction, fallback_prediction):
            return prediction
        return GesturePrediction(
            gesture_id=fallback_prediction.gesture_id,
            confidence=fallback_prediction.confidence,
            metadata={
                **fallback_prediction.metadata,
                "fallback_after": prediction.metadata.get("reason", "model_unknown"),
                "model_confidence": prediction.confidence,
            },
        )


def _should_prefer_dynamic_fallback(
    model_prediction: GesturePrediction,
    fallback_prediction: GesturePrediction,
) -> bool:
    if fallback_prediction.gesture_id == GestureID.UNKNOWN:
        return False
    if fallback_prediction.confidence < 0.75:
        return False
    if model_prediction.confidence >= 0.55:
        return False
    if model_prediction.gesture_id != GestureID.CIRCLE:
        return False
    return fallback_prediction.gesture_id in {GestureID.WAVE_LR, GestureID.PULL_TOWARD}


def _should_use_dynamic_fallback_after_unknown(
    model_prediction: GesturePrediction,
    fallback_prediction: GesturePrediction,
) -> bool:
    if fallback_prediction.gesture_id == GestureID.UNKNOWN:
        return False
    if fallback_prediction.confidence < 0.90:
        return False
    return model_prediction.confidence < 0.35


def _load_bundle(
    path: Path,
    expected_feature_version: str | tuple[str, ...],
    min_confidence: float | None = None,
    threshold_profile: ThresholdProfile | None = None,
) -> ModelBundle:
    if not path.exists():
        raise FileNotFoundError(f"Model file does not exist: {path}")

    joblib = importlib.import_module("joblib")
    payload = joblib.load(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Model file must contain a dictionary bundle: {path}")

    feature_version = str(payload.get("feature_version", ""))
    expected_versions = (
        (expected_feature_version,)
        if isinstance(expected_feature_version, str)
        else expected_feature_version
    )
    if feature_version not in expected_versions:
        raise ValueError(f"Expected feature version {expected_versions!r}, got {feature_version!r}")

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
    bundle_class_thresholds = _normalize_threshold_mapping(
        cast(Mapping[str, Any], payload.get("class_thresholds") or {})
    )
    if threshold_profile is not None:
        if threshold_profile.min_confidence is not None:
            bundle_min_confidence = threshold_profile.min_confidence
        bundle_class_thresholds.update(threshold_profile.class_thresholds)

    effective_min_confidence = (
        bundle_min_confidence
        if min_confidence is None
        else max(bundle_min_confidence, min_confidence)
    )
    if min_confidence is not None:
        bundle_class_thresholds = {
            label: max(threshold, min_confidence)
            for label, threshold in bundle_class_thresholds.items()
        }

    return ModelBundle(
        model=model,
        label_names=label_names,
        feature_version=feature_version,
        min_confidence=effective_min_confidence,
        class_thresholds=bundle_class_thresholds,
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
    threshold = bundle.class_thresholds.get(label, bundle.min_confidence)
    if confidence < threshold:
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
            "threshold": threshold,
        },
    )


def load_threshold_profile(path: str | Path) -> ThresholdProfile:
    """Load a confidence-threshold profile from JSON."""

    profile_path = Path(path)
    payload = json.loads(profile_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Threshold profile must contain a JSON object: {profile_path}")

    min_confidence_value = payload.get("min_confidence")
    min_confidence = (
        None if min_confidence_value is None else _validated_threshold(min_confidence_value)
    )
    class_thresholds = _normalize_threshold_mapping(
        cast(Mapping[str, Any], payload.get("class_thresholds", {}))
    )
    return ThresholdProfile(min_confidence=min_confidence, class_thresholds=class_thresholds)


def _normalize_threshold_mapping(values: Mapping[str, Any]) -> dict[str, float]:
    return {
        normalize_label(label): _validated_threshold(threshold)
        for label, threshold in values.items()
    }


def _validated_threshold(value: Any) -> float:
    threshold = float(value)
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError(f"Confidence threshold must be in [0, 1], got {threshold}")
    return threshold
