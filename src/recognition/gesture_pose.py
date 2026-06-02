"""Static hand pose diagnostics derived from MediaPipe landmarks."""

from __future__ import annotations

from dataclasses import dataclass

from src.config import StaticClassifierConfig
from src.domain import GestureID
from src.utils.geometry import (
    Landmark,
    LandmarkSequence,
    axis_direction,
    distance,
    hand_scale,
    is_finger_extended,
    to_landmarks,
)

FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")


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

    def as_dict(self) -> dict[str, bool]:
        """Return states keyed by finger name."""

        return dict(zip(FINGER_NAMES, self.as_tuple(), strict=True))


@dataclass(frozen=True)
class StaticPoseAnalysis:
    """Landmark-level pose values used for classifier diagnostics."""

    landmarks: tuple[Landmark, ...]
    finger_states: FingerStates
    ok_tip_distance_ratio: float
    thumb_direction: str
    index_direction: str


@dataclass(frozen=True)
class GesturePoseSpec:
    """Expected static pose for a gesture."""

    gesture_id: GestureID
    finger_states: dict[str, bool | None]
    thumb_direction: str | None = None
    index_direction: str | None = None
    max_ok_tip_distance_ratio: float | None = None


EXPECTED_STATIC_POSES: dict[GestureID, GesturePoseSpec] = {
    GestureID.OPEN_PALM: GesturePoseSpec(
        gesture_id=GestureID.OPEN_PALM,
        finger_states={
            "thumb": True,
            "index": True,
            "middle": True,
            "ring": True,
            "pinky": True,
        },
    ),
    GestureID.FIST: GesturePoseSpec(
        gesture_id=GestureID.FIST,
        finger_states={
            "thumb": False,
            "index": False,
            "middle": False,
            "ring": False,
            "pinky": False,
        },
    ),
    GestureID.THUMB_UP: GesturePoseSpec(
        gesture_id=GestureID.THUMB_UP,
        finger_states={
            "thumb": True,
            "index": False,
            "middle": False,
            "ring": False,
            "pinky": False,
        },
        thumb_direction="up",
    ),
    GestureID.THUMB_DOWN: GesturePoseSpec(
        gesture_id=GestureID.THUMB_DOWN,
        finger_states={
            "thumb": True,
            "index": False,
            "middle": False,
            "ring": False,
            "pinky": False,
        },
        thumb_direction="down",
    ),
    GestureID.INDEX_LEFT: GesturePoseSpec(
        gesture_id=GestureID.INDEX_LEFT,
        finger_states={
            "thumb": False,
            "index": True,
            "middle": False,
            "ring": False,
            "pinky": False,
        },
        index_direction="left",
    ),
    GestureID.INDEX_RIGHT: GesturePoseSpec(
        gesture_id=GestureID.INDEX_RIGHT,
        finger_states={
            "thumb": False,
            "index": True,
            "middle": False,
            "ring": False,
            "pinky": False,
        },
        index_direction="right",
    ),
    GestureID.PEACE: GesturePoseSpec(
        gesture_id=GestureID.PEACE,
        finger_states={
            "thumb": False,
            "index": True,
            "middle": True,
            "ring": False,
            "pinky": False,
        },
    ),
    GestureID.THREE_FINGERS: GesturePoseSpec(
        gesture_id=GestureID.THREE_FINGERS,
        finger_states={
            "thumb": False,
            "index": True,
            "middle": True,
            "ring": True,
            "pinky": False,
        },
    ),
    GestureID.PINKY: GesturePoseSpec(
        gesture_id=GestureID.PINKY,
        finger_states={
            "thumb": False,
            "index": False,
            "middle": False,
            "ring": False,
            "pinky": True,
        },
    ),
    GestureID.OK_SIGN: GesturePoseSpec(
        gesture_id=GestureID.OK_SIGN,
        finger_states={
            "thumb": None,
            "index": None,
            "middle": True,
            "ring": True,
            "pinky": True,
        },
        max_ok_tip_distance_ratio=0.16,
    ),
}


def analyze_static_pose(
    raw_landmarks: LandmarkSequence,
    config: StaticClassifierConfig | None = None,
) -> StaticPoseAnalysis:
    """Analyze finger states and directions for one static hand pose."""

    classifier_config = config or StaticClassifierConfig()
    landmarks = to_landmarks(raw_landmarks)
    scale = hand_scale(landmarks)
    finger_states = _finger_states(landmarks, classifier_config)
    return StaticPoseAnalysis(
        landmarks=tuple(landmarks),
        finger_states=finger_states,
        ok_tip_distance_ratio=distance(landmarks[4], landmarks[8]) / scale,
        thumb_direction=axis_direction(
            landmarks,
            base_index=2,
            tip_index=4,
            margin=classifier_config.direction_margin,
        ),
        index_direction=axis_direction(
            landmarks,
            base_index=5,
            tip_index=8,
            margin=classifier_config.direction_margin,
        ),
    )


def expected_pose_for(gesture_id: GestureID) -> GesturePoseSpec | None:
    """Return the expected static pose specification for a gesture."""

    return EXPECTED_STATIC_POSES.get(gesture_id)


def _finger_states(
    landmarks: list[Landmark],
    config: StaticClassifierConfig,
) -> FingerStates:
    thumb_extended = distance(landmarks[0], landmarks[4]) > (
        distance(landmarks[0], landmarks[3]) * config.thumb_extended_ratio
    )
    return FingerStates(
        thumb=thumb_extended,
        index=is_finger_extended(
            landmarks,
            tip_index=8,
            pip_index=6,
            ratio=config.extended_finger_ratio,
        ),
        middle=is_finger_extended(
            landmarks,
            tip_index=12,
            pip_index=10,
            ratio=config.extended_finger_ratio,
        ),
        ring=is_finger_extended(
            landmarks,
            tip_index=16,
            pip_index=14,
            ratio=config.extended_finger_ratio,
        ),
        pinky=is_finger_extended(
            landmarks,
            tip_index=20,
            pip_index=18,
            ratio=config.extended_finger_ratio,
        ),
    )
