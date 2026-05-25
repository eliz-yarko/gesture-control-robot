from __future__ import annotations

from pathlib import Path

import pytest
from src.calibration import (
    AdaptiveCalibrator,
    CalibrationProfileStore,
    CalibrationSession,
    LandmarkFeatureExtractor,
)
from src.config import AppConfig, CalibrationConfig, CommandMappingConfig
from src.domain import GestureID, GesturePrediction, RobotCommand
from src.pipeline import GestureControlPipeline
from src.recognition.hand_detector import HandDetection
from src.transmission.mock_sender import MockCommandSender


def test_feature_extractor_is_translation_and_scale_invariant() -> None:
    extractor = LandmarkFeatureExtractor()

    base_features = extractor.extract(_open_palm_landmarks())
    transformed_features = extractor.extract(
        _transform_landmarks(_open_palm_landmarks(), scale=2.0, dx=10.0, dy=-4.0)
    )

    assert transformed_features == pytest.approx(base_features)


def test_calibration_session_builds_profile() -> None:
    session = CalibrationSession(
        CalibrationConfig(samples_per_gesture=2),
        gestures=[GestureID.OPEN_PALM],
    )

    session.add_sample(GestureID.OPEN_PALM, _open_palm_landmarks())
    session.add_sample(GestureID.OPEN_PALM, _transform_landmarks(_open_palm_landmarks(), dx=1.0))
    profile = session.build_profile("operator-1")

    assert profile.user_id == "operator-1"
    assert profile.feature_size == 63
    stats = profile.stats_for(GestureID.OPEN_PALM)
    assert stats is not None
    assert stats.sample_count == 2


def test_calibration_profile_store_roundtrip(tmp_path: Path) -> None:
    session = CalibrationSession(
        CalibrationConfig(samples_per_gesture=1, profile_directory=str(tmp_path)),
        gestures=[GestureID.OPEN_PALM],
    )
    session.add_sample(GestureID.OPEN_PALM, _open_palm_landmarks())
    profile = session.build_profile("operator 1")
    store = CalibrationProfileStore(CalibrationConfig(profile_directory=str(tmp_path)))

    path = store.save(profile)
    loaded = store.load_path(path)

    assert path.name == "operator_1.json"
    assert loaded.user_id == profile.user_id
    assert loaded.stats_for(GestureID.OPEN_PALM) == profile.stats_for(GestureID.OPEN_PALM)


def test_adaptive_calibrator_boosts_matching_prediction() -> None:
    profile = _profile_for_open_palm()
    calibrator = AdaptiveCalibrator(profile)

    adjusted = calibrator.adjust_prediction(
        GesturePrediction(GestureID.OPEN_PALM, 0.7),
        _open_palm_landmarks(),
    )

    assert adjusted.gesture_id == GestureID.OPEN_PALM
    assert adjusted.confidence > 0.7
    assert adjusted.metadata["calibration"]["within_threshold"] is True


def test_adaptive_calibrator_reduces_mismatched_static_prediction() -> None:
    profile = _profile_for_open_palm()
    calibrator = AdaptiveCalibrator(profile)

    adjusted = calibrator.adjust_prediction(
        GesturePrediction(GestureID.OPEN_PALM, 0.95),
        _fist_landmarks(),
    )

    assert adjusted.gesture_id == GestureID.OPEN_PALM
    assert adjusted.confidence < 0.6
    assert adjusted.metadata["calibration"]["within_threshold"] is False


def test_adaptive_calibrator_recovers_unknown_static_prediction() -> None:
    profile = _profile_for_open_palm()
    calibrator = AdaptiveCalibrator(profile)

    adjusted = calibrator.adjust_prediction(
        GesturePrediction.unknown(),
        _open_palm_landmarks(),
    )

    assert adjusted.gesture_id == GestureID.OPEN_PALM
    assert adjusted.confidence >= 0.65


def test_pipeline_uses_calibrator_before_command_mapping() -> None:
    sender = MockCommandSender()
    pipeline = GestureControlPipeline(
        config=AppConfig(command_mapping=CommandMappingConfig(static_confirmation_frames=1)),
        detector=_FakeDetector([HandDetection(_open_palm_landmarks(), "Right", 0.95)]),
        static_classifier=_UnknownStaticClassifier(),
        calibrator=AdaptiveCalibrator(_profile_for_open_palm()),
        command_sender=sender,
    )

    result = pipeline.process(bgr_rgb_frame(index=0))

    assert result.selected_prediction.gesture_id == GestureID.OPEN_PALM
    assert result.command_event is not None
    assert result.command_event.command == RobotCommand.STOP
    assert sender.last_event == result.command_event


def _profile_for_open_palm():
    session = CalibrationSession(
        CalibrationConfig(samples_per_gesture=1),
        gestures=[GestureID.OPEN_PALM],
    )
    session.add_sample(GestureID.OPEN_PALM, _open_palm_landmarks())
    return session.build_profile("operator-1")


def bgr_rgb_frame(index: int):
    from src.capture.video_capture import CapturedFrame

    return CapturedFrame(bgr_frame=object(), rgb_frame=object(), index=index)


class _FakeDetector:
    def __init__(self, detections: list[HandDetection]) -> None:
        self._detections = detections

    def detect(self, rgb_frame: object) -> list[HandDetection]:
        return self._detections

    def close(self) -> None:
        return None


class _UnknownStaticClassifier:
    def classify(self, raw_landmarks: object) -> GesturePrediction:
        return GesturePrediction.unknown()


def _open_palm_landmarks() -> list[tuple[float, float, float]]:
    landmarks = [(0.0, 0.0, 0.0) for _ in range(21)]
    landmarks[0] = (0.0, 0.0, 0.0)
    _set_thumb(landmarks, extended=True)
    _set_finger(landmarks, 5, 6, 7, 8, -0.12, extended=True)
    _set_finger(landmarks, 9, 10, 11, 12, 0.0, extended=True)
    _set_finger(landmarks, 13, 14, 15, 16, 0.12, extended=True)
    _set_finger(landmarks, 17, 18, 19, 20, 0.22, extended=True)
    return landmarks


def _fist_landmarks() -> list[tuple[float, float, float]]:
    landmarks = [(0.0, 0.0, 0.0) for _ in range(21)]
    landmarks[0] = (0.0, 0.0, 0.0)
    _set_thumb(landmarks, extended=False)
    _set_finger(landmarks, 5, 6, 7, 8, -0.12, extended=False)
    _set_finger(landmarks, 9, 10, 11, 12, 0.0, extended=False)
    _set_finger(landmarks, 13, 14, 15, 16, 0.12, extended=False)
    _set_finger(landmarks, 17, 18, 19, 20, 0.22, extended=False)
    return landmarks


def _set_thumb(landmarks: list[tuple[float, float, float]], extended: bool) -> None:
    landmarks[1] = (0.14, -0.08, 0.0)
    landmarks[2] = (0.18, -0.10, 0.0)
    if extended:
        landmarks[3] = (0.18, -0.32, 0.0)
        landmarks[4] = (0.18, -0.55, 0.0)
    else:
        landmarks[3] = (0.20, -0.10, 0.0)
        landmarks[4] = (0.12, -0.08, 0.0)


def _set_finger(
    landmarks: list[tuple[float, float, float]],
    mcp_index: int,
    pip_index: int,
    dip_index: int,
    tip_index: int,
    base_x: float,
    extended: bool,
) -> None:
    landmarks[mcp_index] = (base_x, -0.20, 0.0)
    if extended:
        landmarks[pip_index] = (base_x, -0.42, 0.0)
        landmarks[dip_index] = (base_x, -0.58, 0.0)
        landmarks[tip_index] = (base_x, -0.75, 0.0)
    else:
        landmarks[pip_index] = (base_x, -0.25, 0.0)
        landmarks[dip_index] = (base_x, -0.16, 0.0)
        landmarks[tip_index] = (base_x, -0.10, 0.0)


def _transform_landmarks(
    landmarks: list[tuple[float, float, float]],
    scale: float = 1.0,
    dx: float = 0.0,
    dy: float = 0.0,
) -> list[tuple[float, float, float]]:
    return [(x * scale + dx, y * scale + dy, z * scale) for x, y, z in landmarks]
