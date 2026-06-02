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
    distance,
    hand_scale,
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
        thumb_tip_extension = _thumb_tip_extension_ratio(landmarks)
        thumb_vertical_clearance = _thumb_vertical_clearance_ratio(landmarks, pose.thumb_direction)
        open_non_thumb_count = sum((states.index, states.middle, states.ring, states.pinky))
        metadata = {
            "finger_states": states.as_dict(),
            "ok_tip_distance": ok_tip_distance,
            "thumb_tip_extension": thumb_tip_extension,
            "thumb_vertical_clearance": thumb_vertical_clearance,
        }

        if ok_tip_distance <= self._config.ok_tip_distance_ratio:
            if sum((states.middle, states.ring, states.pinky)) >= 2:
                return GesturePrediction(GestureID.OK_SIGN, 0.95, metadata)

        if states.index and states.middle and states.ring and states.pinky:
            confidence = 0.95 if states.thumb else 0.86
            return GesturePrediction(GestureID.OPEN_PALM, confidence, metadata)

        non_thumb_closed = not any((states.index, states.middle, states.ring, states.pinky))
        if non_thumb_closed:
            thumb_direction = pose.thumb_direction
            metadata["thumb_direction"] = thumb_direction
            thumb_is_isolated = (
                states.thumb
                and thumb_tip_extension >= self._config.thumb_tip_extension_ratio
                and thumb_vertical_clearance >= self._config.thumb_vertical_clearance_ratio
            )
            if thumb_is_isolated:
                if thumb_direction == "up":
                    return GesturePrediction(GestureID.THUMB_UP, 0.9, metadata)
                if thumb_direction == "down":
                    return GesturePrediction(GestureID.THUMB_DOWN, 0.9, metadata)
            confidence = 0.92 if not states.thumb else 0.86
            return GesturePrediction(GestureID.FIST, confidence, metadata)

        if states.index and not any((states.middle, states.ring, states.pinky)):
            index_direction = pose.index_direction
            metadata["index_direction"] = index_direction
            if index_direction == "left":
                confidence = 0.9 if not states.thumb else 0.86
                return GesturePrediction(GestureID.INDEX_LEFT, confidence, metadata)
            if index_direction == "right":
                confidence = 0.9 if not states.thumb else 0.86
                return GesturePrediction(GestureID.INDEX_RIGHT, confidence, metadata)

        if states.index and states.middle and not any((states.ring, states.pinky)):
            confidence = 0.92 if not states.thumb else 0.88
            return GesturePrediction(GestureID.PEACE, confidence, metadata)

        if open_non_thumb_count == 3:
            confidence = 0.92 if not states.thumb else 0.88
            return GesturePrediction(GestureID.THREE_FINGERS, confidence, metadata)

        if states.pinky and not any((states.index, states.middle, states.ring)):
            confidence = 0.9 if not states.thumb else 0.84
            return GesturePrediction(GestureID.PINKY, confidence, metadata)

        return GesturePrediction.unknown("static_rules_no_match")


def _thumb_tip_extension_ratio(landmarks: LandmarkSequence) -> float:
    normalized_landmarks = to_landmarks(landmarks)
    return distance(normalized_landmarks[2], normalized_landmarks[4]) / hand_scale(
        normalized_landmarks
    )


def _thumb_vertical_clearance_ratio(landmarks: LandmarkSequence, direction: str) -> float:
    normalized_landmarks = to_landmarks(landmarks)
    if direction not in {"up", "down"}:
        return 0.0

    scale = hand_scale(normalized_landmarks)
    thumb_tip_y = normalized_landmarks[4][1]
    folded_finger_y_values = [normalized_landmarks[index][1] for index in (6, 10, 14, 18)]
    if direction == "up":
        clearance = min(folded_finger_y_values) - thumb_tip_y
    else:
        clearance = thumb_tip_y - max(folded_finger_y_values)
    return clearance / scale
