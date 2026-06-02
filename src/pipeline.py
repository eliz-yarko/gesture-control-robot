"""Frame-to-command pipeline for the gesture control subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from src.calibration.adaptive_calibrator import AdaptiveCalibrator
from src.capture.video_capture import CapturedFrame
from src.config import AppConfig
from src.domain import CommandEvent, GestureID, GesturePrediction
from src.interpretation.command_mapper import CommandConfirmationState, CommandMapper
from src.recognition.dynamic_classifier import DynamicGestureClassifier
from src.recognition.gesture_pose import StaticPoseAnalysis, analyze_static_pose
from src.recognition.hand_detector import HandDetection, HandDetector
from src.recognition.static_classifier import StaticGestureClassifier
from src.recognition.trajectory_buffer import TrajectoryBuffer
from src.transmission.base_sender import CommandSender
from src.utils.geometry import LandmarkSequence


class HandDetectorProtocol(Protocol):
    """Minimal detector contract used by the pipeline."""

    def detect(self, rgb_frame: object) -> list[HandDetection]:
        """Return hand detections for an RGB frame."""

    def close(self) -> None:
        """Release detector resources."""


class StaticClassifierProtocol(Protocol):
    """Minimal static classifier contract used by the pipeline."""

    def classify(self, raw_landmarks: LandmarkSequence) -> GesturePrediction:
        """Return a static gesture prediction for one landmark frame."""


class DynamicClassifierProtocol(Protocol):
    """Minimal dynamic classifier contract used by the pipeline."""

    def classify(self, buffer: TrajectoryBuffer) -> GesturePrediction:
        """Return a dynamic gesture prediction for the current trajectory."""


@dataclass(frozen=True)
class PipelineResult:
    """Result produced after processing one frame."""

    frame_index: int
    detections: list[HandDetection]
    static_prediction: GesturePrediction
    dynamic_prediction: GesturePrediction
    selected_prediction: GesturePrediction
    command_event: CommandEvent | None
    command_state: CommandConfirmationState = field(
        default_factory=CommandConfirmationState.unknown
    )
    pose_analysis: StaticPoseAnalysis | None = None


class GestureControlPipeline:
    """Connect hand detection, classification, debouncing, and command sending."""

    def __init__(
        self,
        config: AppConfig | None = None,
        detector: HandDetectorProtocol | None = None,
        static_classifier: StaticClassifierProtocol | None = None,
        dynamic_classifier: DynamicClassifierProtocol | None = None,
        trajectory_buffer: TrajectoryBuffer | None = None,
        command_mapper: CommandMapper | None = None,
        command_sender: CommandSender | None = None,
        calibrator: AdaptiveCalibrator | None = None,
    ) -> None:
        """Initialize the processing pipeline."""

        self._config = config or AppConfig()
        self._detector = detector or HandDetector(self._config.hand_detection)
        self._static_classifier = static_classifier or StaticGestureClassifier(
            self._config.static_classifier
        )
        self._dynamic_classifier = dynamic_classifier or DynamicGestureClassifier(
            self._config.dynamic_classifier
        )
        self._trajectory_buffer = trajectory_buffer or TrajectoryBuffer(
            self._config.dynamic_classifier.buffer_size
        )
        self._command_mapper = command_mapper or CommandMapper(self._config.command_mapping)
        self._command_sender = command_sender
        self._calibrator = calibrator

    def process(self, frame: CapturedFrame) -> PipelineResult:
        """Process one captured frame and optionally send a confirmed command."""

        detections = self._detector.detect(frame.rgb_frame)
        if not detections:
            unknown = GesturePrediction.unknown("no_hand_detected")
            command_event = self._command_mapper.update(unknown)
            return PipelineResult(
                frame_index=frame.index,
                detections=[],
                static_prediction=unknown,
                dynamic_prediction=unknown,
                selected_prediction=unknown,
                command_event=command_event,
                command_state=self._command_mapper.confirmation_state,
                pose_analysis=None,
            )

        detection = max(detections, key=lambda item: item.score)
        pose_analysis = analyze_static_pose(detection.landmarks, self._config.static_classifier)
        static_prediction = self._static_classifier.classify(detection.landmarks)
        self._trajectory_buffer.add_landmarks(detection.landmarks, timestamp=frame.timestamp)
        dynamic_prediction = self._dynamic_classifier.classify(self._trajectory_buffer)
        selected_prediction = _select_prediction(
            static_prediction,
            dynamic_prediction,
            pose_analysis,
            dynamic_min_confidence=self._config.dynamic_classifier.selection_min_confidence,
        )
        if self._calibrator is not None:
            selected_prediction = self._calibrator.adjust_prediction(
                selected_prediction,
                detection.landmarks,
            )
        command_event = self._command_mapper.update(selected_prediction)
        command_state = self._command_mapper.confirmation_state

        if command_event is not None and self._command_sender is not None:
            self._command_sender.send(command_event)

        return PipelineResult(
            frame_index=frame.index,
            detections=detections,
            static_prediction=static_prediction,
            dynamic_prediction=dynamic_prediction,
            selected_prediction=selected_prediction,
            command_event=command_event,
            command_state=command_state,
            pose_analysis=pose_analysis,
        )

    def close(self) -> None:
        """Close resources owned by the detector."""

        self._detector.close()


def _select_prediction(
    static_prediction: GesturePrediction,
    dynamic_prediction: GesturePrediction,
    pose_analysis: StaticPoseAnalysis,
    dynamic_min_confidence: float,
) -> GesturePrediction:
    """Select the safest gesture candidate for command confirmation."""

    if (
        dynamic_prediction.gesture_id == GestureID.UNKNOWN
        or dynamic_prediction.confidence < dynamic_min_confidence
    ):
        return static_prediction

    if _is_dynamic_pose_compatible(
        dynamic_prediction.gesture_id,
        static_prediction,
        pose_analysis,
    ):
        return dynamic_prediction

    return static_prediction


def _is_dynamic_pose_compatible(
    gesture_id: GestureID,
    static_prediction: GesturePrediction,
    pose_analysis: StaticPoseAnalysis,
) -> bool:
    states = pose_analysis.finger_states
    if gesture_id == GestureID.WAVE_LR:
        return static_prediction.gesture_id == GestureID.OPEN_PALM
    if gesture_id == GestureID.PULL_TOWARD:
        return static_prediction.gesture_id in {GestureID.OPEN_PALM, GestureID.FIST}
    if gesture_id == GestureID.CIRCLE:
        return states.index and not any((states.middle, states.ring, states.pinky))
    return False
