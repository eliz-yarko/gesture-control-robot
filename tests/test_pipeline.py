from __future__ import annotations

from src.capture.video_capture import CapturedFrame
from src.config import AppConfig, CommandMappingConfig
from src.domain import GestureID, GesturePrediction, RobotCommand
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


def test_pipeline_keeps_static_thumb_up_over_false_wave_candidate() -> None:
    sender = MockCommandSender()
    pipeline = GestureControlPipeline(
        config=AppConfig(
            command_mapping=CommandMappingConfig(
                static_confirmation_frames=1,
                dynamic_confirmation_frames=1,
            )
        ),
        detector=_FakeDetector([HandDetection(_open_palm_landmarks(), "Right", 0.95)]),
        static_classifier=_FakeStaticClassifier(GestureID.THUMB_UP, 0.9),
        dynamic_classifier=_FakeDynamicClassifier(GestureID.WAVE_LR, 0.98),
        command_sender=sender,
    )

    result = pipeline.process(CapturedFrame(bgr_frame=object(), rgb_frame=object(), index=12))

    assert result.selected_prediction.gesture_id == GestureID.THUMB_UP
    assert result.command_event is not None
    assert result.command_event.command == RobotCommand.START


def test_pipeline_allows_compatible_open_palm_wave_candidate() -> None:
    sender = MockCommandSender()
    pipeline = GestureControlPipeline(
        config=AppConfig(
            command_mapping=CommandMappingConfig(
                static_confirmation_frames=1,
                dynamic_confirmation_frames=1,
            )
        ),
        detector=_FakeDetector([HandDetection(_open_palm_landmarks(), "Right", 0.95)]),
        static_classifier=_FakeStaticClassifier(GestureID.OPEN_PALM, 0.95),
        dynamic_classifier=_FakeDynamicClassifier(GestureID.WAVE_LR, 0.98),
        command_sender=sender,
    )

    result = pipeline.process(CapturedFrame(bgr_frame=object(), rgb_frame=object(), index=13))

    assert result.selected_prediction.gesture_id == GestureID.WAVE_LR
    assert result.command_event is not None
    assert result.command_event.command == RobotCommand.MODE_TOGGLE


def test_pipeline_rejects_wave_candidate_without_open_palm_pose() -> None:
    pipeline = GestureControlPipeline(
        config=AppConfig(
            command_mapping=CommandMappingConfig(
                static_confirmation_frames=1,
                dynamic_confirmation_frames=1,
            )
        ),
        detector=_FakeDetector([HandDetection(_index_point_landmarks(), "Right", 0.95)]),
        static_classifier=_FakeStaticClassifier(GestureID.UNKNOWN, 0.0),
        dynamic_classifier=_FakeDynamicClassifier(GestureID.WAVE_LR, 0.98),
    )

    result = pipeline.process(CapturedFrame(bgr_frame=object(), rgb_frame=object(), index=14))

    assert result.selected_prediction.gesture_id == GestureID.UNKNOWN


def test_pipeline_allows_circle_candidate_with_index_pose() -> None:
    sender = MockCommandSender()
    pipeline = GestureControlPipeline(
        config=AppConfig(
            command_mapping=CommandMappingConfig(
                static_confirmation_frames=1,
                dynamic_confirmation_frames=1,
            )
        ),
        detector=_FakeDetector([HandDetection(_index_point_landmarks(), "Right", 0.95)]),
        static_classifier=_FakeStaticClassifier(GestureID.UNKNOWN, 0.0),
        dynamic_classifier=_FakeDynamicClassifier(GestureID.CIRCLE, 0.98),
        command_sender=sender,
    )

    result = pipeline.process(CapturedFrame(bgr_frame=object(), rgb_frame=object(), index=15))

    assert result.selected_prediction.gesture_id == GestureID.CIRCLE
    assert result.command_event is not None
    assert result.command_event.command == RobotCommand.ROTATE_360


class _FakeDetector:
    def __init__(self, detections: list[HandDetection]) -> None:
        self._detections = detections
        self.closed = False

    def detect(self, rgb_frame: object) -> list[HandDetection]:
        return self._detections

    def close(self) -> None:
        self.closed = True


class _FakeStaticClassifier:
    def __init__(self, gesture_id: GestureID, confidence: float) -> None:
        self._prediction = GesturePrediction(gesture_id, confidence)

    def classify(self, raw_landmarks: object) -> GesturePrediction:
        return self._prediction


class _FakeDynamicClassifier:
    def __init__(self, gesture_id: GestureID, confidence: float) -> None:
        self._prediction = GesturePrediction(gesture_id, confidence)

    def classify(self, buffer: object) -> GesturePrediction:
        return self._prediction


def _open_palm_landmarks() -> list[tuple[float, float, float]]:
    landmarks = [(0.0, 0.0, 0.0) for _ in range(21)]
    landmarks[0] = (0.0, 0.0, 0.0)
    _set_thumb(landmarks)
    _set_finger(landmarks, 5, 6, 7, 8, -0.12)
    _set_finger(landmarks, 9, 10, 11, 12, 0.0)
    _set_finger(landmarks, 13, 14, 15, 16, 0.12)
    _set_finger(landmarks, 17, 18, 19, 20, 0.22)
    return landmarks


def _index_point_landmarks() -> list[tuple[float, float, float]]:
    landmarks = [(0.0, 0.0, 0.0) for _ in range(21)]
    landmarks[0] = (0.0, 0.0, 0.0)
    _set_thumb(landmarks)
    _set_finger(landmarks, 5, 6, 7, 8, -0.12)
    _set_closed_finger(landmarks, 9, 10, 11, 12, 0.0)
    _set_closed_finger(landmarks, 13, 14, 15, 16, 0.12)
    _set_closed_finger(landmarks, 17, 18, 19, 20, 0.22)
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


def _set_closed_finger(
    landmarks: list[tuple[float, float, float]],
    mcp_index: int,
    pip_index: int,
    dip_index: int,
    tip_index: int,
    base_x: float,
) -> None:
    landmarks[mcp_index] = (base_x, -0.20, 0.0)
    landmarks[pip_index] = (base_x, -0.25, 0.0)
    landmarks[dip_index] = (base_x, -0.16, 0.0)
    landmarks[tip_index] = (base_x, -0.10, 0.0)
