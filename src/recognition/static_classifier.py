"""Rule-based classifier for static hand gestures."""

from __future__ import annotations

from typing import Final

from src.config import StaticClassifierConfig
from src.domain import GestureID, GesturePrediction
from src.recognition.gesture_pose import (
    FINGER_NAMES,
    analyze_static_pose,
)
from src.recognition.gesture_pose import (
    FingerStates as FingerStates,
)
from src.utils.geometry import (
    LandmarkSequence,
    to_landmarks,
)


class StaticGestureClassifier:
    """Classify static gestures from a single MediaPipe Hands landmark frame."""

    _FINGER_NAMES: Final[tuple[str, ...]] = FINGER_NAMES

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
        pose = analyze_static_pose(landmarks, self._config)
        states = pose.finger_states
        ok_tip_distance = pose.ok_tip_distance_ratio
        metadata = {
            "finger_states": states.as_dict(),
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
            thumb_direction = pose.thumb_direction
            metadata["thumb_direction"] = thumb_direction
            if thumb_direction == "up":
                return GesturePrediction(GestureID.THUMB_UP, 0.9, metadata)
            if thumb_direction == "down":
                return GesturePrediction(GestureID.THUMB_DOWN, 0.9, metadata)

        if states.index and not any((states.thumb, states.middle, states.ring, states.pinky)):
            index_direction = pose.index_direction
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
