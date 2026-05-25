"""Adaptive calibration logic for personalized gesture recognition."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from src.calibration.feature_extractor import (
    FeatureVector,
    LandmarkFeatureExtractor,
    distance_radius,
    euclidean_distance,
    mean_vector,
    std_vector,
)
from src.calibration.profile import GestureCalibrationStats, UserCalibrationProfile, utc_now_iso
from src.config import CalibrationConfig
from src.domain import GestureID, GesturePrediction
from src.utils.geometry import LandmarkSequence


@dataclass(frozen=True)
class CalibrationMatch:
    """Result of comparing a hand pose with a personalized gesture profile."""

    gesture_id: GestureID
    distance: float
    threshold: float
    confidence: float

    @property
    def is_within_threshold(self) -> bool:
        """Return whether the pose falls within the personalized threshold."""

        return self.distance <= self.threshold


class CalibrationSession:
    """Collect calibration samples and build a user profile."""

    def __init__(
        self,
        config: CalibrationConfig | None = None,
        gestures: Iterable[GestureID] | None = None,
        extractor: LandmarkFeatureExtractor | None = None,
    ) -> None:
        """Initialize a calibration session."""

        self._config = config or CalibrationConfig()
        self._extractor = extractor or LandmarkFeatureExtractor()
        self._gestures = tuple(gestures) if gestures is not None else _default_static_gestures()
        self._samples: dict[GestureID, list[FeatureVector]] = {
            gesture_id: [] for gesture_id in self._gestures
        }

    @property
    def gestures(self) -> tuple[GestureID, ...]:
        """Return gestures included in this session."""

        return self._gestures

    def add_sample(self, gesture_id: GestureID, raw_landmarks: LandmarkSequence) -> int:
        """Add a calibration sample and return the new sample count."""

        if gesture_id not in self._samples:
            raise ValueError(f"Gesture {gesture_id.name} is not part of this calibration session.")

        self._samples[gesture_id].append(self._extractor.extract(raw_landmarks))
        return len(self._samples[gesture_id])

    def sample_count(self, gesture_id: GestureID) -> int:
        """Return collected sample count for a gesture."""

        return len(self._samples.get(gesture_id, []))

    def is_gesture_ready(self, gesture_id: GestureID) -> bool:
        """Return whether enough samples were collected for a gesture."""

        return self.sample_count(gesture_id) >= self._config.samples_per_gesture

    def is_complete(self) -> bool:
        """Return whether all session gestures have enough samples."""

        return all(self.is_gesture_ready(gesture_id) for gesture_id in self._gestures)

    def build_profile(self, user_id: str, allow_partial: bool = False) -> UserCalibrationProfile:
        """Build a profile from collected samples.

        Args:
            user_id: Stable profile identifier.
            allow_partial: Whether to build a profile for ready gestures only.

        Raises:
            ValueError: If required gestures do not have enough samples.
        """

        gesture_stats: dict[GestureID, GestureCalibrationStats] = {}
        for gesture_id in self._gestures:
            samples = self._samples[gesture_id]
            if len(samples) < self._config.samples_per_gesture:
                if allow_partial:
                    continue
                raise ValueError(f"Not enough calibration samples for {gesture_id.name}.")

            center = mean_vector(samples)
            gesture_stats[gesture_id] = GestureCalibrationStats(
                gesture_id=gesture_id,
                sample_count=len(samples),
                centroid=center,
                std=std_vector(samples, center),
                radius=distance_radius(samples, center, self._config.sigma_multiplier),
            )

        if not gesture_stats:
            raise ValueError("Cannot build a calibration profile without samples.")

        feature_size = len(next(iter(gesture_stats.values())).centroid)
        timestamp = utc_now_iso()
        return UserCalibrationProfile(
            user_id=user_id,
            created_at=timestamp,
            updated_at=timestamp,
            feature_size=feature_size,
            samples_per_gesture=self._config.samples_per_gesture,
            sigma_multiplier=self._config.sigma_multiplier,
            gestures=gesture_stats,
        )


class AdaptiveCalibrator:
    """Apply a user calibration profile to classifier predictions."""

    def __init__(
        self,
        profile: UserCalibrationProfile,
        config: CalibrationConfig | None = None,
        extractor: LandmarkFeatureExtractor | None = None,
    ) -> None:
        """Initialize adaptive calibrator."""

        self._profile = profile
        self._config = config or CalibrationConfig()
        self._extractor = extractor or LandmarkFeatureExtractor()

    @property
    def profile(self) -> UserCalibrationProfile:
        """Return the active user profile."""

        return self._profile

    def match(
        self,
        raw_landmarks: LandmarkSequence,
        expected_gesture: GestureID | None = None,
    ) -> CalibrationMatch | None:
        """Match landmarks against a specific or nearest calibrated gesture."""

        feature = self._extractor.extract(raw_landmarks)
        if expected_gesture is not None:
            stats = self._profile.stats_for(expected_gesture)
            if stats is None:
                return None
            return _match_stats(feature, stats)

        matches = [_match_stats(feature, stats) for stats in self._profile.gestures.values()]
        if not matches:
            return None
        return min(matches, key=lambda match: _normalized_distance(match))

    def adjust_prediction(
        self,
        prediction: GesturePrediction,
        raw_landmarks: LandmarkSequence,
    ) -> GesturePrediction:
        """Adjust a static prediction using the active user profile."""

        if prediction.gesture_id.is_dynamic:
            return prediction

        if prediction.gesture_id == GestureID.UNKNOWN:
            return self._recover_unknown_prediction(prediction, raw_landmarks)

        match = self.match(raw_landmarks, prediction.gesture_id)
        if match is None:
            return prediction

        confidence = (
            max(prediction.confidence, match.confidence)
            if match.is_within_threshold
            else min(prediction.confidence, match.confidence)
        )
        return GesturePrediction(
            gesture_id=prediction.gesture_id,
            confidence=confidence,
            metadata=_with_calibration_metadata(prediction, match),
        )

    def _recover_unknown_prediction(
        self,
        prediction: GesturePrediction,
        raw_landmarks: LandmarkSequence,
    ) -> GesturePrediction:
        match = self.match(raw_landmarks)
        if match is None or not match.is_within_threshold:
            return prediction
        if match.confidence < self._config.min_calibrated_confidence:
            return prediction
        return GesturePrediction(
            gesture_id=match.gesture_id,
            confidence=match.confidence,
            metadata=_with_calibration_metadata(prediction, match),
        )


def _match_stats(feature: FeatureVector, stats: GestureCalibrationStats) -> CalibrationMatch:
    distance = euclidean_distance(feature, stats.centroid)
    return CalibrationMatch(
        gesture_id=stats.gesture_id,
        distance=distance,
        threshold=stats.radius,
        confidence=_confidence_from_distance(distance, stats.radius),
    )


def _confidence_from_distance(distance: float, threshold: float) -> float:
    if threshold <= 0.0:
        return 1.0 if distance == 0.0 else 0.0
    if distance <= threshold:
        return max(0.6, 1.0 - 0.4 * (distance / threshold))
    return max(0.0, 0.6 - 0.6 * ((distance - threshold) / threshold))


def _normalized_distance(match: CalibrationMatch) -> float:
    return match.distance / match.threshold if match.threshold > 0.0 else float("inf")


def _with_calibration_metadata(
    prediction: GesturePrediction,
    match: CalibrationMatch,
) -> dict[str, object]:
    return {
        **prediction.metadata,
        "calibration": {
            "gesture_id": int(match.gesture_id),
            "distance": match.distance,
            "threshold": match.threshold,
            "confidence": match.confidence,
            "within_threshold": match.is_within_threshold,
        },
    }


def _default_static_gestures() -> tuple[GestureID, ...]:
    return tuple(gesture_id for gesture_id in GestureID if gesture_id.is_static)
