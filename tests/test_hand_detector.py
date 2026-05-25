from __future__ import annotations

from dataclasses import dataclass

from src.recognition.hand_detector import HandDetector


def test_hand_detector_converts_mediapipe_result() -> None:
    result = _Result(
        multi_hand_landmarks=[
            _LandmarkList(
                landmark=[
                    _Point(x=float(index), y=float(index + 1), z=float(index + 2))
                    for index in range(21)
                ]
            )
        ],
        multi_handedness=[_Handedness(classification=[_Classification("Right", 0.91)])],
    )
    detector = HandDetector(hands=_FakeHands(result))

    detections = detector.detect(rgb_frame=object())

    assert len(detections) == 1
    assert detections[0].handedness == "Right"
    assert detections[0].score == 0.91
    assert detections[0].landmarks[8] == (8.0, 9.0, 10.0)


def test_hand_detector_returns_empty_list_when_no_hands_detected() -> None:
    detector = HandDetector(hands=_FakeHands(_Result(None, None)))

    detections = detector.detect(rgb_frame=object())

    assert detections == []


def test_hand_detector_close_delegates_to_hands_object() -> None:
    hands = _FakeHands(_Result(None, None))
    detector = HandDetector(hands=hands)

    detector.close()

    assert hands.closed is True


@dataclass
class _Point:
    x: float
    y: float
    z: float


@dataclass
class _LandmarkList:
    landmark: list[_Point]


@dataclass
class _Classification:
    label: str
    score: float


@dataclass
class _Handedness:
    classification: list[_Classification]


@dataclass
class _Result:
    multi_hand_landmarks: list[_LandmarkList] | None
    multi_handedness: list[_Handedness] | None


class _FakeHands:
    def __init__(self, result: _Result) -> None:
        self._result = result
        self.closed = False

    def process(self, rgb_frame: object) -> _Result:
        return self._result

    def close(self) -> None:
        self.closed = True
