"""Frame-to-command pipeline for the gesture control subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from src.calibration.adaptive_calibrator import AdaptiveCalibrator
from src.capture.video_capture import CapturedFrame
from src.config import AppConfig, CommandMappingConfig
from src.domain import CommandEvent, GestureID, GesturePrediction
from src.interpretation.command_mapper import CommandConfirmationState, CommandMapper
from src.recognition.dynamic_classifier import DynamicGestureClassifier
from src.recognition.dynamic_segmenter import DynamicGestureSegmenter
from src.recognition.gesture_pose import StaticPoseAnalysis, analyze_static_pose
from src.recognition.hand_detector import HandDetection, HandDetector
from src.recognition.prediction_smoother import StaticPredictionSmoother
from src.recognition.static_classifier import StaticGestureClassifier
from src.recognition.trajectory_buffer import TrajectoryBuffer, trajectory_point_from_landmarks
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
    dynamic_state: str = "idle"


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
        self._missing_detection_frames = 0
        self._dynamic_cooldown_frames = 0
        self._dynamic_segmenter = DynamicGestureSegmenter(self._config.dynamic_classifier)
        self._static_smoother = StaticPredictionSmoother(self._config.prediction_smoothing)
        self._pending_dynamic_prediction: GesturePrediction | None = None
        self._pending_dynamic_frames_remaining = 0

    def process(self, frame: CapturedFrame) -> PipelineResult:
        """Process one captured frame and optionally send a confirmed command."""

        detections = self._detector.detect(frame.rgb_frame)
        if not detections:
            return self._process_missing_detection(frame)

        self._missing_detection_frames = 0
        detection = max(detections, key=lambda item: item.score)
        pose_analysis = analyze_static_pose(detection.landmarks, self._config.static_classifier)
        raw_static_prediction = self._static_classifier.classify(detection.landmarks)
        static_prediction = self._static_smoother.update(raw_static_prediction)
        point = trajectory_point_from_landmarks(detection.landmarks, timestamp=frame.timestamp)
        self._trajectory_buffer.add_point(point)
        segment_update = self._dynamic_segmenter.update(point)
        dynamic_prediction, dynamic_state = self._dynamic_prediction_from_segment(
            segment_update.segment,
            candidate_segment=segment_update.candidate_segment,
            default_state=segment_update.state,
        )
        selected_prediction = _select_prediction_for_motion_state(
            static_prediction,
            dynamic_prediction,
            dynamic_min_confidence=self._config.dynamic_classifier.selection_min_confidence,
            dynamic_state=dynamic_state,
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
        if command_event is not None and command_event.gesture_id.is_dynamic:
            self._reset_dynamic_tracking()
            self._begin_dynamic_cooldown()
            dynamic_state = "confirmed"

        return PipelineResult(
            frame_index=frame.index,
            detections=detections,
            static_prediction=static_prediction,
            dynamic_prediction=dynamic_prediction,
            selected_prediction=selected_prediction,
            command_event=command_event,
            command_state=command_state,
            pose_analysis=pose_analysis,
            dynamic_state=dynamic_state,
        )

    def _process_missing_detection(self, frame: CapturedFrame) -> PipelineResult:
        self._missing_detection_frames += 1
        self._static_smoother.reset()
        unknown = GesturePrediction.unknown("no_hand_detected")
        segment_update = self._dynamic_segmenter.missing()
        dynamic_prediction, dynamic_state = self._dynamic_prediction_from_segment(
            segment_update.segment,
            candidate_segment=segment_update.candidate_segment,
            default_state=segment_update.state,
        )
        selected_prediction = _select_prediction_for_motion_state(
            unknown,
            dynamic_prediction,
            dynamic_min_confidence=self._config.dynamic_classifier.selection_min_confidence,
            dynamic_state=dynamic_state,
        )
        command_event = self._command_mapper.update(selected_prediction)
        command_state = self._command_mapper.confirmation_state

        if command_event is not None and self._command_sender is not None:
            self._command_sender.send(command_event)
        if command_event is not None and command_event.gesture_id.is_dynamic:
            self._reset_dynamic_tracking()
            self._begin_dynamic_cooldown()
            self._missing_detection_frames = 0
            dynamic_state = "confirmed"
        elif (
            self._missing_detection_frames
            > self._config.dynamic_classifier.missing_detection_tolerance_frames
        ):
            self._reset_dynamic_tracking()

        return PipelineResult(
            frame_index=frame.index,
            detections=[],
            static_prediction=unknown,
            dynamic_prediction=dynamic_prediction,
            selected_prediction=selected_prediction,
            command_event=command_event,
            command_state=command_state,
            pose_analysis=None,
            dynamic_state=dynamic_state,
        )

    def close(self) -> None:
        """Close resources owned by the detector."""

        self._detector.close()

    def configure_command_mapping(self, config: CommandMappingConfig) -> None:
        """Apply runtime command confirmation settings."""

        self._command_mapper.configure(config)

    def _begin_dynamic_cooldown(self) -> None:
        self._dynamic_cooldown_frames = max(
            2,
            self._config.dynamic_classifier.min_window_points,
        )

    def _dynamic_prediction_from_segment(
        self,
        segment: TrajectoryBuffer | None,
        *,
        candidate_segment: TrajectoryBuffer | None,
        default_state: str,
    ) -> tuple[GesturePrediction, str]:
        if self._dynamic_cooldown_frames > 0:
            self._dynamic_cooldown_frames -= 1
            self._reset_dynamic_tracking(clear_cooldown=False)
            return GesturePrediction.unknown("dynamic_cooldown"), "cooldown"

        pending = self._consume_pending_dynamic_prediction()
        if pending is not None:
            return pending, "confirming_dynamic"

        if segment is None:
            early_prediction = self._classify_early_dynamic_candidate(candidate_segment)
            if early_prediction is not None:
                self._hold_dynamic_prediction(early_prediction)
                return early_prediction, "early_dynamic_candidate"
            if default_state in {"motion_started", "recording_dynamic"}:
                return GesturePrediction.unknown("dynamic_motion_in_progress"), default_state
            return GesturePrediction.unknown(default_state), default_state

        prediction = self._dynamic_classifier.classify(segment)
        if (
            prediction.gesture_id != GestureID.UNKNOWN
            and prediction.confidence >= self._config.dynamic_classifier.selection_min_confidence
        ):
            self._hold_dynamic_prediction(prediction)
            return prediction, "segment_classified"
        return GesturePrediction.unknown("dynamic_segment_rejected"), "segment_rejected"

    def _hold_dynamic_prediction(self, prediction: GesturePrediction) -> None:
        required_frames = self._config.command_mapping.dynamic_confirmation_frames
        self._pending_dynamic_prediction = prediction
        self._pending_dynamic_frames_remaining = max(0, required_frames - 1)

    def _consume_pending_dynamic_prediction(self) -> GesturePrediction | None:
        if self._pending_dynamic_prediction is None:
            return None
        prediction = self._pending_dynamic_prediction
        if self._pending_dynamic_frames_remaining <= 0:
            self._pending_dynamic_prediction = None
            return None
        self._pending_dynamic_frames_remaining -= 1
        return prediction

    def _reset_dynamic_tracking(self, *, clear_cooldown: bool = True) -> None:
        self._trajectory_buffer.clear()
        self._dynamic_segmenter.reset()
        self._pending_dynamic_prediction = None
        self._pending_dynamic_frames_remaining = 0
        if clear_cooldown:
            self._dynamic_cooldown_frames = 0

    def _classify_early_dynamic_candidate(
        self,
        candidate_segment: TrajectoryBuffer | None,
    ) -> GesturePrediction | None:
        if candidate_segment is None:
            return None

        min_points = max(
            self._config.dynamic_classifier.min_dynamic_segment_points,
            self._config.dynamic_classifier.early_dynamic_min_points,
        )
        if len(candidate_segment) < min_points:
            return None

        prediction = self._dynamic_classifier.classify(candidate_segment)
        if prediction.gesture_id == GestureID.UNKNOWN:
            return None

        min_confidence = max(
            self._config.dynamic_classifier.selection_min_confidence,
            self._config.dynamic_classifier.early_dynamic_min_confidence,
        )
        if prediction.confidence < min_confidence:
            return None

        return GesturePrediction(
            gesture_id=prediction.gesture_id,
            confidence=prediction.confidence,
            metadata={
                **prediction.metadata,
                "early_dynamic_candidate": True,
            },
        )


def _select_prediction(
    static_prediction: GesturePrediction,
    dynamic_prediction: GesturePrediction,
    dynamic_min_confidence: float,
) -> GesturePrediction:
    """Select the gesture candidate for command confirmation."""

    if (
        dynamic_prediction.gesture_id == GestureID.UNKNOWN
        or dynamic_prediction.confidence < dynamic_min_confidence
    ):
        return static_prediction

    if _is_emergency_stop_candidate(static_prediction):
        return static_prediction

    return dynamic_prediction


def _select_prediction_for_motion_state(
    static_prediction: GesturePrediction,
    dynamic_prediction: GesturePrediction,
    dynamic_min_confidence: float,
    dynamic_state: str,
) -> GesturePrediction:
    """Select a command candidate while suppressing static commands during motion."""

    if _is_emergency_stop_candidate(static_prediction):
        return static_prediction

    if dynamic_state in {
        "motion_started",
        "recording_dynamic",
        "cooldown",
        "confirming_dynamic",
        "early_dynamic_candidate",
        "segment_classified",
        "segment_rejected",
    }:
        if (
            dynamic_prediction.gesture_id != GestureID.UNKNOWN
            and dynamic_prediction.confidence >= dynamic_min_confidence
        ):
            return dynamic_prediction
        return GesturePrediction.unknown(dynamic_prediction.metadata.get("reason", dynamic_state))

    return _select_prediction(
        static_prediction,
        dynamic_prediction,
        dynamic_min_confidence=dynamic_min_confidence,
    )


def _is_emergency_stop_candidate(static_prediction: GesturePrediction) -> bool:
    return (
        static_prediction.gesture_id == GestureID.THUMB_DOWN
        and static_prediction.confidence >= 0.65
    )
