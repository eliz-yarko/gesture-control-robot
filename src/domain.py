"""Domain enums and value objects for gesture-based robot control."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from time import monotonic
from typing import Any


class GestureID(IntEnum):
    """Supported gesture identifiers.

    Static gestures use IDs 0-9. Dynamic gestures use IDs 10-12. The numeric
    values are stable because they are used in datasets, benchmark CSV files,
    and thesis tables.
    """

    UNKNOWN = -1
    OPEN_PALM = 0
    FIST = 1
    THUMB_UP = 2
    THUMB_DOWN = 3
    INDEX_LEFT = 4
    INDEX_RIGHT = 5
    PEACE = 6
    THREE_FINGERS = 7
    PINKY = 8
    OK_SIGN = 9
    WAVE_LR = 10
    CIRCLE = 11
    PULL_TOWARD = 12

    @property
    def is_static(self) -> bool:
        """Return whether the gesture is recognized from a single frame."""

        return 0 <= int(self) <= 9

    @property
    def is_dynamic(self) -> bool:
        """Return whether the gesture is recognized from a frame trajectory."""

        return 10 <= int(self) <= 12


class RobotCommand(StrEnum):
    """Robot commands emitted by the interpretation layer."""

    UNKNOWN = "UNKNOWN"
    STOP = "STOP"
    FORWARD = "FORWARD"
    START = "START"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    INCREASE_SPEED = "INCREASE_SPEED"
    DECREASE_SPEED = "DECREASE_SPEED"
    RETURN_HOME = "RETURN_HOME"
    CONFIRM_ACTION = "CONFIRM_ACTION"
    MODE_TOGGLE = "MODE_TOGGLE"
    ROTATE_360 = "ROTATE_360"
    APPROACH_OPERATOR = "APPROACH_OPERATOR"


@dataclass(frozen=True)
class GesturePrediction:
    """Gesture classifier output.

    Args:
        gesture_id: Recognized gesture identifier.
        confidence: Classifier confidence in the range [0, 1].
        metadata: Optional diagnostic values used by debug overlays and reports.
    """

    gesture_id: GestureID
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def unknown(cls, reason: str = "no_match") -> "GesturePrediction":
        """Build a normalized unknown prediction."""

        return cls(
            gesture_id=GestureID.UNKNOWN,
            confidence=0.0,
            metadata={"reason": reason},
        )


@dataclass(frozen=True)
class CommandEvent:
    """Confirmed robot command after debouncing and safety filtering."""

    command: RobotCommand
    gesture_id: GestureID
    confidence: float
    timestamp: float = field(default_factory=monotonic)
