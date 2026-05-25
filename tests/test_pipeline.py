from __future__ import annotations

from src.capture.video_capture import CapturedFrame
from src.config import AppConfig, CommandMappingConfig
from src.domain import GestureID, RobotCommand
from src.pipeline import GestureControlPipeline
from src.recognition.hand_detector import HandDetection
from src.transmission.mock_sender import MockCommandSender


def test_pipeline_processes_frame_and_sends_confirmed_command() -> None:
    sender = MockCommandSender()
    pipeline = GestureControlPipeline(
        config=AppConfig(command_mapping=CommandMappingConfig(static_confirmation_frames=1)),
        detector=_FakeDetector([HandDetection(_open_palm_landmarks(), "Right", 0.95)]),
        command_sender=sender,
    )

    result = pipeline.process(CapturedFrame(bgr_frame=object(), rgb_frame=object(), index=0))

    assert result.static_prediction.gesture_id == GestureID.OPEN_PALM
    assert result.selected_prediction.gesture_id == GestureID.OPEN_PALM
    assert result.command_event is not None
    assert result.command_event.command == RobotCommand.STOP
    assert sender.last_event == result.command_event


def test_pipeline_returns_unknown_when_no_hand_detected() -> None:
    pipeline = GestureControlPipeline(detector=_FakeDetector([]))

    result = pipeline.process(CapturedFrame(bgr_frame=object(), rgb_frame=object(), index=7))

    assert result.frame_index == 7
    assert result.detections == []
    assert result.selected_prediction.gesture_id == GestureID.UNKNOWN
    assert result.command_event is None


class _FakeDetector:
    def __init__(self, detections: list[HandDetection]) -> None:
        self._detections = detections
        self.closed = False

    def detect(self, rgb_frame: object) -> list[HandDetection]:
        return self._detections

    def close(self) -> None:
        self.closed = True


def _open_palm_landmarks() -> list[tuple[float, float, float]]:
    landmarks = [(0.0, 0.0, 0.0) for _ in range(21)]
    landmarks[0] = (0.0, 0.0, 0.0)
    _set_thumb(landmarks)
    _set_finger(landmarks, 5, 6, 7, 8, -0.12)
    _set_finger(landmarks, 9, 10, 11, 12, 0.0)
    _set_finger(landmarks, 13, 14, 15, 16, 0.12)
    _set_finger(landmarks, 17, 18, 19, 20, 0.22)
    return landmarks


def _set_thumb(landmarks: list[tuple[float, float, float]]) -> None:
    landmarks[1] = (0.14, -0.08, 0.0)
    landmarks[2] = (0.18, -0.10, 0.0)
    landmarks[3] = (0.18, -0.32, 0.0)
    landmarks[4] = (0.18, -0.55, 0.0)


def _set_finger(
    landmarks: list[tuple[float, float, float]],
    mcp_index: int,
    pip_index: int,
    dip_index: int,
    tip_index: int,
    base_x: float,
) -> None:
    landmarks[mcp_index] = (base_x, -0.20, 0.0)
    landmarks[pip_index] = (base_x, -0.42, 0.0)
    landmarks[dip_index] = (base_x, -0.58, 0.0)
    landmarks[tip_index] = (base_x, -0.75, 0.0)
