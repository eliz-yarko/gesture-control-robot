"""Rule-based classifier for static hand gestures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from src.config import StaticClassifierConfig
from src.domain import GestureID, GesturePrediction
from src.utils.geometry import (
    LandmarkSequence,
    axis_direction,
    distance,
    hand_scale,
    is_finger_extended,
    to_landmarks,
)


@dataclass(frozen=True)
class FingerStates:
    """Boolean state for the five fingers."""

    thumb: bool
    index: bool
    middle: bool
    ring: bool
    pinky: bool

    def as_tuple(self) -> tuple[bool, bool, bool, bool, bool]:
        """Return states in thumb-to-pinky order."""

        return (self.thumb, self.index, self.middle, self.ring, self.pinky)


class StaticGestureClassifier:
    """Classify static gestures from a single MediaPipe Hands landmark frame."""

    _FINGER_NAMES: Final[tuple[str, ...]] = ("thumb", "index", "middle", "ring", "pinky")

    def __init__(self, config: StaticClassifierConfig | None = None) -> None:
        """Initialize the classifier.

        Args:
            config: Optional thresholds for rule-based recognition.
        """

        self._config = config or StaticClassifierConfig()

    def classify(self, raw_landmarks: LandmarkSequence) -> GesturePrediction:
        """Classify a static gesture from hand landmarks.

        Args:
            raw_landmarks: Sequence of 21 MediaPipe hand landmarks with x, y, z
                coordinates normalized to image space.

        Returns:
            Detected gesture with confidence, or ``GestureID.UNKNOWN``.
        """

        landmarks = to_landmarks(raw_landmarks)
        states = self._finger_states(landmarks)
        scale = hand_scale(landmarks)
        ok_tip_distance = distance(landmarks[4], landmarks[8]) / scale
        metadata = {
            "finger_states": dict(zip(self._FINGER_NAMES, states.as_tuple(), strict=True)),
            "ok_tip_distance": ok_tip_distance,
        }

        if ok_tip_distance <= self._config.ok_tip_distance_ratio:
            if states.middle or states.ring or states.pinky:
                return GesturePrediction(GestureID.OK_SIGN, 0.95, metadata)

        if states.index and states.middle and states.ring and states.pinky:
            confidence = 0.95 if states.thumb else 0.86
            return GesturePrediction(GestureID.OPEN_PALM, confidence, metadata)

        if states.as_tuple() == (False, False, False, False, False):
            return GesturePrediction(GestureID.FIST, 0.92, metadata)

        if states.thumb and not any((states.index, states.middle, states.ring, states.pinky)):
            thumb_direction = axis_direction(
                landmarks,
                base_index=2,
                tip_index=4,
                margin=self._config.direction_margin,
            )
            metadata["thumb_direction"] = thumb_direction
            if thumb_direction == "up":
                return GesturePrediction(GestureID.THUMB_UP, 0.9, metadata)
            if thumb_direction == "down":
                return GesturePrediction(GestureID.THUMB_DOWN, 0.9, metadata)

        if states.index and not any((states.thumb, states.middle, states.ring, states.pinky)):
            index_direction = axis_direction(
                landmarks,
                base_index=5,
                tip_index=8,
                margin=self._config.direction_margin,
            )
            metadata["index_direction"] = index_direction
            if index_direction == "left":
                return GesturePrediction(GestureID.INDEX_LEFT, 0.9, metadata)
            if index_direction == "right":
                return GesturePrediction(GestureID.INDEX_RIGHT, 0.9, metadata)

        if states.as_tuple() == (False, True, True, False, False):
            return GesturePrediction(GestureID.PEACE, 0.92, metadata)

        if states.as_tuple() == (False, True, True, True, False):
            return GesturePrediction(GestureID.THREE_FINGERS, 0.92, metadata)

        if states.as_tuple() == (False, False, False, False, True):
            return GesturePrediction(GestureID.PINKY, 0.9, metadata)

        return GesturePrediction.unknown("static_rules_no_match")

    def _finger_states(self, landmarks: list[tuple[float, float, float]]) -> FingerStates:
        thumb_extended = distance(landmarks[0], landmarks[4]) > (
            distance(landmarks[0], landmarks[3]) * self._config.thumb_extended_ratio
        )
        return FingerStates(
            thumb=thumb_extended,
            index=is_finger_extended(
                landmarks,
                tip_index=8,
                pip_index=6,
                ratio=self._config.extended_finger_ratio,
            ),
            middle=is_finger_extended(
                landmarks,
                tip_index=12,
                pip_index=10,
                ratio=self._config.extended_finger_ratio,
            ),
            ring=is_finger_extended(
                landmarks,
                tip_index=16,
                pip_index=14,
                ratio=self._config.extended_finger_ratio,
            ),
            pinky=is_finger_extended(
                landmarks,
                tip_index=20,
                pip_index=18,
                ratio=self._config.extended_finger_ratio,
            ),
        )
