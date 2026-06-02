from __future__ import annotations

import pytest
from src.domain import GestureID
from src.recognition.static_classifier import StaticGestureClassifier


@pytest.mark.parametrize(
    ("gesture_id", "expected"),
    [
        (GestureID.OPEN_PALM, GestureID.OPEN_PALM),
        (GestureID.FIST, GestureID.FIST),
        (GestureID.THUMB_UP, GestureID.THUMB_UP),
        (GestureID.THUMB_DOWN, GestureID.THUMB_DOWN),
        (GestureID.INDEX_LEFT, GestureID.INDEX_LEFT),
        (GestureID.INDEX_RIGHT, GestureID.INDEX_RIGHT),
        (GestureID.PEACE, GestureID.PEACE),
        (GestureID.THREE_FINGERS, GestureID.THREE_FINGERS),
        (GestureID.PINKY, GestureID.PINKY),
        (GestureID.OK_SIGN, GestureID.OK_SIGN),
    ],
)
def test_static_classifier_recognizes_supported_gestures(
    gesture_id: GestureID,
    expected: GestureID,
) -> None:
    classifier = StaticGestureClassifier()

    prediction = classifier.classify(_landmarks_for(gesture_id))

    assert prediction.gesture_id == expected
    assert prediction.confidence >= 0.9


def test_static_classifier_rejects_invalid_landmark_shape() -> None:
    classifier = StaticGestureClassifier()

    with pytest.raises(ValueError, match="Expected 21 hand landmarks"):
        classifier.classify([[0.0, 0.0, 0.0]])


def test_static_classifier_accepts_open_palm_with_relaxed_thumb() -> None:
    classifier = StaticGestureClassifier()
    landmarks = _landmarks_for(GestureID.OPEN_PALM)
    _set_thumb(landmarks, extended=False, direction="right")

    prediction = classifier.classify(landmarks)

    assert prediction.gesture_id == GestureID.OPEN_PALM
    assert prediction.confidence >= 0.85


def _landmarks_for(gesture_id: GestureID) -> list[list[float]]:
    states = {
        "thumb": False,
        "index": False,
        "middle": False,
        "ring": False,
        "pinky": False,
    }
    thumb_direction = "right"
    index_direction = "up"
    ok_sign = False

    if gesture_id == GestureID.OPEN_PALM:
        states = {key: True for key in states}
    elif gesture_id == GestureID.THUMB_UP:
        states["thumb"] = True
        thumb_direction = "up"
    elif gesture_id == GestureID.THUMB_DOWN:
        states["thumb"] = True
        thumb_direction = "down"
    elif gesture_id == GestureID.INDEX_LEFT:
        states["index"] = True
        index_direction = "left"
    elif gesture_id == GestureID.INDEX_RIGHT:
        states["index"] = True
        index_direction = "right"
    elif gesture_id == GestureID.PEACE:
        states["index"] = True
        states["middle"] = True
    elif gesture_id == GestureID.THREE_FINGERS:
        states["index"] = True
        states["middle"] = True
        states["ring"] = True
    elif gesture_id == GestureID.PINKY:
        states["pinky"] = True
    elif gesture_id == GestureID.OK_SIGN:
        states["middle"] = True
        states["ring"] = True
        states["pinky"] = True
        ok_sign = True

    landmarks = [[0.0, 0.0, 0.0] for _ in range(21)]
    landmarks[0] = [0.0, 0.0, 0.0]
    _set_thumb(landmarks, states["thumb"], thumb_direction)
    _set_finger(landmarks, 5, 6, 7, 8, -0.12, states["index"], index_direction)
    _set_finger(landmarks, 9, 10, 11, 12, 0.0, states["middle"], "up")
    _set_finger(landmarks, 13, 14, 15, 16, 0.12, states["ring"], "up")
    _set_finger(landmarks, 17, 18, 19, 20, 0.22, states["pinky"], "up")

    if ok_sign:
        landmarks[4] = [0.03, -0.45, 0.0]
        landmarks[8] = [0.04, -0.45, 0.0]

    return landmarks


def _set_thumb(landmarks: list[list[float]], extended: bool, direction: str) -> None:
    landmarks[1] = [0.14, -0.08, 0.0]
    landmarks[2] = [0.18, -0.10, 0.0]
    if not extended:
        landmarks[3] = [0.20, -0.10, 0.0]
        landmarks[4] = [0.12, -0.08, 0.0]
        return

    if direction == "up":
        landmarks[3] = [0.18, -0.32, 0.0]
        landmarks[4] = [0.18, -0.55, 0.0]
    elif direction == "down":
        landmarks[3] = [0.18, 0.20, 0.0]
        landmarks[4] = [0.18, 0.45, 0.0]
    else:
        landmarks[3] = [0.32, -0.10, 0.0]
        landmarks[4] = [0.50, -0.10, 0.0]


def _set_finger(
    landmarks: list[list[float]],
    mcp_index: int,
    pip_index: int,
    dip_index: int,
    tip_index: int,
    base_x: float,
    extended: bool,
    direction: str,
) -> None:
    landmarks[mcp_index] = [base_x, -0.20, 0.0]
    if not extended:
        landmarks[pip_index] = [base_x, -0.25, 0.0]
        landmarks[dip_index] = [base_x, -0.16, 0.0]
        landmarks[tip_index] = [base_x, -0.10, 0.0]
        return

    if direction == "left":
        landmarks[pip_index] = [base_x - 0.18, -0.20, 0.0]
        landmarks[dip_index] = [base_x - 0.34, -0.20, 0.0]
        landmarks[tip_index] = [base_x - 0.52, -0.20, 0.0]
    elif direction == "right":
        landmarks[pip_index] = [base_x + 0.18, -0.20, 0.0]
        landmarks[dip_index] = [base_x + 0.34, -0.20, 0.0]
        landmarks[tip_index] = [base_x + 0.52, -0.20, 0.0]
    else:
        landmarks[pip_index] = [base_x, -0.42, 0.0]
        landmarks[dip_index] = [base_x, -0.58, 0.0]
        landmarks[tip_index] = [base_x, -0.75, 0.0]
