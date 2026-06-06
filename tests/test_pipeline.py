from __future__ import annotations

from src.capture.video_capture import CapturedFrame
from src.config import AppConfig, CommandMappingConfig, DynamicClassifierConfig
from src.domain import GestureID, GesturePrediction, RobotCommand
from src.pipeline import GestureControlPipeline, PipelineResult
from src.recognition.hand_detector import HandDetection
from src.recognition.trajectory_buffer import TrajectoryBuffer, TrajectoryPoint
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


def test_pipeline_smooths_short_static_unknown_gap() -> None:
    pipeline = GestureControlPipeline(
        detector=_FakeDetector([HandDetection(_open_palm_landmarks(), "Right", 0.95)]),
        static_classifier=_SequenceStaticClassifier(
            [
                GesturePrediction(GestureID.OPEN_PALM, 0.92),
                GesturePrediction(GestureID.OPEN_PALM, 0.91),
                GesturePrediction.unknown("model_confidence_below_threshold"),
            ]
        ),
    )

    results = _process_frames(pipeline, 3)

    assert results[-1].static_prediction.gesture_id == GestureID.OPEN_PALM
    assert results[-1].static_prediction.metadata["smoothed_from"] == "UNKNOWN"


def test_pipeline_keeps_emergency_stop_over_dynamic_candidate() -> None:
    sender = MockCommandSender()
    pipeline = GestureControlPipeline(
        config=AppConfig(
            command_mapping=CommandMappingConfig(
                static_confirmation_frames=1,
                dynamic_confirmation_frames=1,
                emergency_confirmation_frames=1,
            )
        ),
        detector=_FakeDetector([HandDetection(_open_palm_landmarks(), "Right", 0.95)]),
        static_classifier=_FakeStaticClassifier(GestureID.THUMB_DOWN, 0.9),
        dynamic_classifier=_FakeDynamicClassifier(GestureID.WAVE_LR, 0.98),
        command_sender=sender,
    )

    result = pipeline.process(CapturedFrame(bgr_frame=object(), rgb_frame=object(), index=12))

    assert result.selected_prediction.gesture_id == GestureID.THUMB_DOWN
    assert result.command_event is not None
    assert result.command_event.command == RobotCommand.EMERGENCY_STOP


def test_pipeline_suppresses_static_commands_while_dynamic_motion_is_recording() -> None:
    sender = MockCommandSender()
    pipeline = GestureControlPipeline(
        config=AppConfig(
            dynamic_classifier=DynamicClassifierConfig(early_dynamic_min_points=99),
            command_mapping=CommandMappingConfig(
                static_confirmation_frames=1,
                dynamic_confirmation_frames=1,
            ),
        ),
        detector=_SequenceDetector(_moving_open_palm_detections((0.0, 0.03, 0.07))),
        static_classifier=_FakeStaticClassifier(GestureID.OPEN_PALM, 0.95),
        dynamic_classifier=_FakeDynamicClassifier(GestureID.WAVE_LR, 0.98),
        command_sender=sender,
    )

    results = _process_frames(pipeline, 3)

    assert results[-1].dynamic_state == "motion_started"
    assert results[-1].selected_prediction.gesture_id == GestureID.UNKNOWN
    assert results[-1].command_event is None


def test_pipeline_emits_dynamic_command_after_segment_finishes() -> None:
    sender = MockCommandSender()
    pipeline = GestureControlPipeline(
        config=AppConfig(
            dynamic_classifier=DynamicClassifierConfig(early_dynamic_min_points=99),
            command_mapping=CommandMappingConfig(
                static_confirmation_frames=1,
                dynamic_confirmation_frames=1,
            ),
        ),
        detector=_SequenceDetector(_moving_open_palm_detections(_finished_motion_offsets())),
        static_classifier=_FakeStaticClassifier(GestureID.OPEN_PALM, 0.95),
        dynamic_classifier=_FakeDynamicClassifier(GestureID.WAVE_LR, 0.98),
        command_sender=sender,
    )

    results = _process_frames(pipeline, len(_finished_motion_offsets()))
    result = results[-1]

    assert result.dynamic_state == "confirmed"
    assert result.dynamic_prediction.gesture_id == GestureID.WAVE_LR
    assert result.selected_prediction.gesture_id == GestureID.WAVE_LR
    assert result.command_event is not None
    assert result.command_event.command == RobotCommand.MODE_TOGGLE
    assert sender.last_event == result.command_event


def test_pipeline_emits_high_confidence_dynamic_command_before_motion_finishes() -> None:
    sender = MockCommandSender()
    offsets = _active_motion_offsets()
    pipeline = GestureControlPipeline(
        config=AppConfig(
            command_mapping=CommandMappingConfig(
                static_confirmation_frames=1,
                dynamic_confirmation_frames=1,
            )
        ),
        detector=_SequenceDetector(_moving_open_palm_detections(offsets)),
        static_classifier=_FakeStaticClassifier(GestureID.OPEN_PALM, 0.95),
        dynamic_classifier=_FakeDynamicClassifier(GestureID.WAVE_LR, 0.98),
        command_sender=sender,
    )

    results = _process_frames(pipeline, len(offsets))
    result = results[-1]

    assert result.dynamic_state == "confirmed"
    assert result.dynamic_prediction.gesture_id == GestureID.WAVE_LR
    assert result.dynamic_prediction.metadata["early_dynamic_candidate"] is True
    assert result.selected_prediction.gesture_id == GestureID.WAVE_LR
    assert result.command_event is not None
    assert result.command_event.command == RobotCommand.MODE_TOGGLE
    assert sender.last_event == result.command_event


def test_pipeline_confirms_dynamic_segment_for_configured_frame_count() -> None:
    pipeline = GestureControlPipeline(
        config=AppConfig(
            dynamic_classifier=DynamicClassifierConfig(early_dynamic_min_points=99),
            command_mapping=CommandMappingConfig(
                static_confirmation_frames=1,
                dynamic_confirmation_frames=3,
            ),
        ),
        detector=_SequenceDetector(
            _moving_open_palm_detections((*_finished_motion_offsets(), 0.30, 0.30))
        ),
        static_classifier=_FakeStaticClassifier(GestureID.OPEN_PALM, 0.95),
        dynamic_classifier=_FakeDynamicClassifier(GestureID.CIRCLE, 0.98),
    )

    results = _process_frames(pipeline, len(_finished_motion_offsets()) + 2)

    assert results[-3].dynamic_state == "segment_classified"
    assert results[-3].command_event is None
    assert results[-3].command_state.stable_frames == 1
    assert results[-2].dynamic_state == "confirming_dynamic"
    assert results[-2].command_event is None
    assert results[-2].command_state.stable_frames == 2
    assert results[-1].dynamic_state == "confirmed"
    assert results[-1].command_event is not None
    assert results[-1].command_event.command == RobotCommand.ROTATE_360


def test_pipeline_keeps_trajectory_during_short_detection_gap() -> None:
    trajectory_buffer = TrajectoryBuffer(max_size=30)
    trajectory_buffer.add_point(
        TrajectoryPoint(
            palm_center=(0.5, 0.5, 0.0),
            index_tip=(0.5, 0.3, 0.0),
            hand_size=0.25,
            timestamp=0.0,
        )
    )
    pipeline = GestureControlPipeline(
        config=AppConfig(
            dynamic_classifier=DynamicClassifierConfig(missing_detection_tolerance_frames=2)
        ),
        detector=_FakeDetector([]),
        trajectory_buffer=trajectory_buffer,
    )

    result = pipeline.process(CapturedFrame(bgr_frame=object(), rgb_frame=object(), index=17))

    assert result.selected_prediction.gesture_id == GestureID.UNKNOWN
    assert len(trajectory_buffer) == 1


def test_pipeline_clears_trajectory_after_missing_detection_tolerance() -> None:
    trajectory_buffer = TrajectoryBuffer(max_size=30)
    trajectory_buffer.add_point(
        TrajectoryPoint(
            palm_center=(0.5, 0.5, 0.0),
            index_tip=(0.5, 0.3, 0.0),
            hand_size=0.25,
            timestamp=0.0,
        )
    )
    pipeline = GestureControlPipeline(
        config=AppConfig(
            dynamic_classifier=DynamicClassifierConfig(missing_detection_tolerance_frames=1)
        ),
        detector=_FakeDetector([]),
        trajectory_buffer=trajectory_buffer,
    )

    pipeline.process(CapturedFrame(bgr_frame=object(), rgb_frame=object(), index=17))
    result = pipeline.process(CapturedFrame(bgr_frame=object(), rgb_frame=object(), index=18))

    assert result.selected_prediction.gesture_id == GestureID.UNKNOWN
    assert len(trajectory_buffer) == 0


def test_pipeline_emits_active_dynamic_segment_during_detection_gap() -> None:
    sender = MockCommandSender()
    pipeline = GestureControlPipeline(
        config=AppConfig(
            dynamic_classifier=DynamicClassifierConfig(early_dynamic_min_points=99),
            command_mapping=CommandMappingConfig(
                static_confirmation_frames=1,
                dynamic_confirmation_frames=1,
            ),
        ),
        detector=_SequenceDetector(
            [
                *_moving_open_palm_detections((0.0, 0.03, 0.07, 0.12, 0.18)),
                [],
            ]
        ),
        static_classifier=_FakeStaticClassifier(GestureID.OPEN_PALM, 0.95),
        dynamic_classifier=_FakeDynamicClassifier(GestureID.WAVE_LR, 0.98),
        command_sender=sender,
    )

    results = _process_frames(pipeline, 6)
    result = results[-1]

    assert result.dynamic_prediction.gesture_id == GestureID.WAVE_LR
    assert result.selected_prediction.gesture_id == GestureID.WAVE_LR
    assert result.command_event is not None
    assert result.command_event.command == RobotCommand.MODE_TOGGLE
    assert sender.last_event == result.command_event


def test_pipeline_clears_trajectory_after_dynamic_command() -> None:
    trajectory_buffer = TrajectoryBuffer(max_size=30)
    pipeline = GestureControlPipeline(
        config=AppConfig(
            command_mapping=CommandMappingConfig(
                static_confirmation_frames=1,
                dynamic_confirmation_frames=1,
            )
        ),
        detector=_SequenceDetector(_moving_open_palm_detections(_finished_motion_offsets())),
        static_classifier=_FakeStaticClassifier(GestureID.OPEN_PALM, 0.95),
        dynamic_classifier=_FakeDynamicClassifier(GestureID.WAVE_LR, 0.82),
        trajectory_buffer=trajectory_buffer,
    )

    result = _process_frames(pipeline, len(_finished_motion_offsets()))[-1]

    assert result.command_event is not None
    assert result.command_event.gesture_id == GestureID.WAVE_LR
    assert len(trajectory_buffer) == 0


def test_pipeline_applies_dynamic_cooldown_after_dynamic_command() -> None:
    pipeline = GestureControlPipeline(
        config=AppConfig(
            dynamic_classifier=DynamicClassifierConfig(early_dynamic_min_points=99),
            command_mapping=CommandMappingConfig(
                static_confirmation_frames=1,
                dynamic_confirmation_frames=1,
            ),
        ),
        detector=_SequenceDetector(
            _moving_open_palm_detections((*_finished_motion_offsets(), 0.30))
        ),
        static_classifier=_FakeStaticClassifier(GestureID.OPEN_PALM, 0.95),
        dynamic_classifier=_FakeDynamicClassifier(GestureID.WAVE_LR, 0.98),
    )

    results = _process_frames(pipeline, len(_finished_motion_offsets()) + 1)
    first = results[-2]
    second = results[-1]

    assert first.command_event is not None
    assert first.command_event.gesture_id == GestureID.WAVE_LR
    assert first.dynamic_state == "confirmed"
    assert second.dynamic_prediction.gesture_id == GestureID.UNKNOWN
    assert second.dynamic_prediction.metadata["reason"] == "dynamic_cooldown"
    assert second.dynamic_state == "cooldown"


class _FakeDetector:
    def __init__(self, detections: list[HandDetection]) -> None:
        self._detections = detections
        self.closed = False

    def detect(self, rgb_frame: object) -> list[HandDetection]:
        return self._detections

    def close(self) -> None:
        self.closed = True


class _SequenceDetector:
    def __init__(self, frames: list[list[HandDetection]]) -> None:
        self._frames = frames
        self._index = 0
        self.closed = False

    def detect(self, rgb_frame: object) -> list[HandDetection]:
        if not self._frames:
            return []
        frame_index = min(self._index, len(self._frames) - 1)
        self._index += 1
        return self._frames[frame_index]

    def close(self) -> None:
        self.closed = True


class _FakeStaticClassifier:
    def __init__(self, gesture_id: GestureID, confidence: float) -> None:
        self._prediction = GesturePrediction(gesture_id, confidence)

    def classify(self, raw_landmarks: object) -> GesturePrediction:
        return self._prediction


class _SequenceStaticClassifier:
    def __init__(self, predictions: list[GesturePrediction]) -> None:
        self._predictions = predictions
        self._index = 0

    def classify(self, raw_landmarks: object) -> GesturePrediction:
        prediction = self._predictions[min(self._index, len(self._predictions) - 1)]
        self._index += 1
        return prediction


class _FakeDynamicClassifier:
    def __init__(self, gesture_id: GestureID, confidence: float) -> None:
        self._prediction = GesturePrediction(gesture_id, confidence)

    def classify(self, buffer: object) -> GesturePrediction:
        return self._prediction


def _process_frames(
    pipeline: GestureControlPipeline,
    frame_count: int,
) -> list[PipelineResult]:
    return [
        pipeline.process(CapturedFrame(bgr_frame=object(), rgb_frame=object(), index=index))
        for index in range(frame_count)
    ]


def _finished_motion_offsets() -> tuple[float, ...]:
    return (0.0, 0.03, 0.07, 0.12, 0.18, 0.24, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30)


def _active_motion_offsets() -> tuple[float, ...]:
    return (0.0, 0.03, 0.07, 0.12, 0.18, 0.24, 0.30, 0.36, 0.42, 0.48)


def _moving_open_palm_detections(offsets: tuple[float, ...]) -> list[list[HandDetection]]:
    return [
        [HandDetection(_shift_landmarks(_open_palm_landmarks(), dx=offset), "Right", 0.95)]
        for offset in offsets
    ]


def _shift_landmarks(
    landmarks: list[tuple[float, float, float]],
    *,
    dx: float = 0.0,
    dy: float = 0.0,
) -> list[tuple[float, float, float]]:
    return [(x + dx, y + dy, z) for x, y, z in landmarks]


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


def _fist_landmarks() -> list[tuple[float, float, float]]:
    landmarks = [(0.0, 0.0, 0.0) for _ in range(21)]
    landmarks[0] = (0.0, 0.0, 0.0)
    landmarks[1] = (0.14, -0.08, 0.0)
    landmarks[2] = (0.18, -0.10, 0.0)
    landmarks[3] = (0.18, -0.12, 0.0)
    landmarks[4] = (0.16, -0.08, 0.0)
    _set_closed_finger(landmarks, 5, 6, 7, 8, -0.12)
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
