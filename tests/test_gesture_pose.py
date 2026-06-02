from __future__ import annotations

from src.domain import GestureID
from src.recognition.gesture_pose import analyze_static_pose, expected_pose_for


def test_analyze_static_pose_reports_fingers_and_directions() -> None:
    analysis = analyze_static_pose(_open_palm_landmarks())

    assert analysis.finger_states.index is True
    assert analysis.finger_states.middle is True
    assert analysis.finger_states.ring is True
    assert analysis.finger_states.pinky is True
    assert analysis.index_direction == "up"
    assert len(analysis.landmarks) == 21


def test_expected_pose_for_thumb_up_contains_direction() -> None:
    spec = expected_pose_for(GestureID.THUMB_UP)

    assert spec is not None
    assert spec.finger_states["thumb"] is True
    assert spec.finger_states["index"] is False
    assert spec.thumb_direction == "up"


def _open_palm_landmarks() -> list[list[float]]:
    landmarks = [[0.0, 0.0, 0.0] for _ in range(21)]
    landmarks[0] = [0.0, 0.0, 0.0]
    landmarks[1] = [0.12, -0.08, 0.0]
    landmarks[2] = [0.22, -0.16, 0.0]
    landmarks[3] = [0.32, -0.24, 0.0]
    landmarks[4] = [0.42, -0.32, 0.0]
    for offset, indices in enumerate(
        ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
    ):
        base_x = -0.18 + offset * 0.12
        for joint_offset, index in enumerate(indices):
            landmarks[index] = [base_x, -0.2 - joint_offset * 0.18, 0.0]
    return landmarks
