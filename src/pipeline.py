"""Frame-to-command pipeline for the gesture control subsystem."""

from __future__ import annotations

import math
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
from src.recognition.static_classifier import StaticGestureClassifier
from src.recognition.trajectory_buffer import TrajectoryBuffer, trajectory_point_from_landmarks
from src.transmission.base_sender import CommandSender
from src.utils.geometry import LandmarkSequence

_STATIC_COMMAND_SUPPRESSION_STATES = frozenset(
    {
        "motion_started",
        "recording_dynamic",
        "early_dynamic_blocked",
        "segment_rejected",
    }
)


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
        self._pending_dynamic_prediction: GesturePrediction | None = None
        self._pending_dynamic_frames_remaining = 0
        self._deferred_dynamic_prediction: GesturePrediction | None = None
        self._last_static_gesture = GestureID.UNKNOWN
        self._stable_static_frames = 0
        self._static_transition_grace_frames = 0

    def process(self, frame: CapturedFrame) -> PipelineResult:
        """Process one captured frame and optionally send a confirmed command."""

        detections = self._detector.detect(frame.rgb_frame)
        if not detections:
            return self._process_missing_detection(frame)

        self._missing_detection_frames = 0
        detection = max(detections, key=lambda item: item.score)
        pose_analysis = analyze_static_pose(detection.landmarks, self._config.static_classifier)
        static_prediction = self._static_classifier.classify(detection.landmarks)
        point = trajectory_point_from_landmarks(detection.landmarks, timestamp=frame.timestamp)
        self._trajectory_buffer.add_point(point)
        segment_update = self._dynamic_segmenter.update(point)
        static_priority_active, transition_grace_active = self._update_static_tracking(
            static_prediction
        )
        dynamic_prediction, dynamic_state = self._dynamic_prediction_from_segment(
            segment_update.segment,
            candidate_segment=segment_update.candidate_segment,
            default_state=segment_update.state,
            static_prediction=static_prediction,
            allow_early_dynamic=not (static_priority_active or transition_grace_active),
            allow_segment_dynamic=not transition_grace_active,
        )
        selected_prediction = _select_prediction_for_motion_state(
            static_prediction,
            dynamic_prediction,
            dynamic_min_confidence=self._config.dynamic_classifier.selection_min_confidence,
            dynamic_state=dynamic_state,
            static_priority_active=static_priority_active,
            transition_grace_active=transition_grace_active,
            dynamic_static_override_min_confidence=(
                self._config.dynamic_classifier.early_dynamic_min_confidence
            ),
        )
        if self._calibrator is not None:
            selected_prediction = self._calibrator.adjust_prediction(
                selected_prediction,
                detection.landmarks,
            )
        selected_prediction = _suppress_static_command_during_motion(
            selected_prediction,
            self._config.command_mapping,
            dynamic_state,
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

    def flush(self, frame_index: int = 0) -> PipelineResult:
        """Flush an active dynamic segment without running hand detection."""

        return self._process_missing_detection(
            CapturedFrame(bgr_frame=None, rgb_frame=None, index=frame_index)
        )

    def _process_missing_detection(self, frame: CapturedFrame) -> PipelineResult:
        self._missing_detection_frames += 1
        unknown = GesturePrediction.unknown("no_hand_detected")
        segment_update = self._dynamic_segmenter.missing()
        dynamic_prediction, dynamic_state = self._dynamic_prediction_from_segment(
            segment_update.segment,
            candidate_segment=segment_update.candidate_segment,
            default_state=segment_update.state,
            static_prediction=unknown,
            allow_early_dynamic=True,
            allow_segment_dynamic=True,
        )
        self._reset_static_tracking()
        selected_prediction = _select_prediction_for_motion_state(
            unknown,
            dynamic_prediction,
            dynamic_min_confidence=self._config.dynamic_classifier.selection_min_confidence,
            dynamic_state=dynamic_state,
            static_priority_active=False,
            transition_grace_active=False,
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
        static_prediction: GesturePrediction,
        allow_early_dynamic: bool,
        allow_segment_dynamic: bool,
    ) -> tuple[GesturePrediction, str]:
        if self._dynamic_cooldown_frames > 0:
            self._dynamic_cooldown_frames -= 1
            self._reset_dynamic_tracking(clear_cooldown=False)
            return GesturePrediction.unknown("dynamic_cooldown"), "cooldown"

        if not allow_segment_dynamic and self._pending_dynamic_prediction is not None:
            self._pending_dynamic_prediction = None
            self._pending_dynamic_frames_remaining = 0
            return GesturePrediction.unknown("dynamic_blocked_by_transition_grace"), "blocked"

        pending = self._consume_pending_dynamic_prediction()
        if pending is not None:
            return pending, "confirming_dynamic"

        if segment is None:
            candidate_prediction = self._classify_dynamic_candidate(
                candidate_segment,
                min_points=max(
                    self._config.dynamic_classifier.min_dynamic_segment_points,
                    self._config.dynamic_classifier.dynamic_min_confirm_points,
                ),
                min_confidence=self._config.dynamic_classifier.selection_min_confidence,
            )
            if candidate_prediction is not None:
                self._remember_deferred_dynamic_candidate(
                    candidate_segment,
                    candidate_prediction,
                    static_prediction,
                )
            early_prediction = self._early_dynamic_prediction_from_candidate(
                candidate_segment,
                candidate_prediction,
            )
            if early_prediction is not None:
                if not allow_early_dynamic:
                    return GesturePrediction.unknown("dynamic_blocked_by_static_priority"), (
                        "early_dynamic_blocked"
                    )
                self._hold_dynamic_prediction(early_prediction)
                return early_prediction, "early_dynamic_candidate"
            if default_state in {"motion_started", "recording_dynamic"}:
                return GesturePrediction.unknown("dynamic_motion_in_progress"), default_state
            return GesturePrediction.unknown(default_state), default_state

        if not allow_segment_dynamic:
            return GesturePrediction.unknown("dynamic_blocked_by_transition_grace"), "blocked"

        prediction = self._dynamic_classifier.classify(segment)
        if (
            prediction.gesture_id != GestureID.UNKNOWN
            and prediction.confidence >= self._config.dynamic_classifier.selection_min_confidence
            and self._passes_dynamic_confirmation_gate(segment, prediction, static_prediction)
        ):
            self._deferred_dynamic_prediction = None
            self._hold_dynamic_prediction(prediction)
            return prediction, "segment_classified"
        if self._deferred_dynamic_prediction is not None:
            deferred_prediction = self._deferred_dynamic_prediction
            self._deferred_dynamic_prediction = None
            self._hold_dynamic_prediction(deferred_prediction)
            return deferred_prediction, "deferred_dynamic_candidate"
        return GesturePrediction.unknown("dynamic_segment_rejected"), "segment_rejected"

    def _update_static_tracking(
        self,
        static_prediction: GesturePrediction,
    ) -> tuple[bool, bool]:
        config = self._config.dynamic_classifier
        high_confidence_static = (
            static_prediction.gesture_id.is_static
            and static_prediction.confidence >= config.static_priority_min_confidence
        )

        if high_confidence_static:
            if static_prediction.gesture_id == self._last_static_gesture:
                self._stable_static_frames += 1
            else:
                if self._last_static_gesture.is_static:
                    self._static_transition_grace_frames = max(
                        self._static_transition_grace_frames,
                        config.static_transition_grace_frames,
                    )
                self._last_static_gesture = static_prediction.gesture_id
                self._stable_static_frames = 1
        else:
            self._stable_static_frames = 0
            if static_prediction.gesture_id == GestureID.UNKNOWN:
                self._last_static_gesture = GestureID.UNKNOWN

        transition_grace_active = self._static_transition_grace_frames > 0
        if self._static_transition_grace_frames > 0:
            self._static_transition_grace_frames -= 1

        static_priority_active = (
            high_confidence_static
            and self._stable_static_frames >= config.static_priority_frames
        )
        return static_priority_active, transition_grace_active

    def _reset_static_tracking(self) -> None:
        self._last_static_gesture = GestureID.UNKNOWN
        self._stable_static_frames = 0
        self._static_transition_grace_frames = 0

    def _passes_dynamic_confirmation_gate(
        self,
        segment: TrajectoryBuffer,
        prediction: GesturePrediction,
        static_prediction: GesturePrediction,
    ) -> bool:
        points = segment.points()
        config = self._config.dynamic_classifier
        min_points = max(config.dynamic_min_confirm_points, config.min_dynamic_segment_points)
        if len(points) < min_points:
            return False

        palm_points = [point.palm_center for point in points]
        index_points = [point.index_tip for point in points]
        palm_path = _path_length_2d(palm_points)
        index_path = _path_length_2d(index_points)
        max_path = max(palm_path, index_path)
        mean_step = max_path / max(len(points) - 1, 1)
        resampled_mean_step = max(
            _resampled_mean_step_2d(palm_points),
            _resampled_mean_step_2d(index_points),
        )
        scale_growth = _scale_growth([point.hand_size for point in points])

        has_planar_trajectory = (
            max_path >= config.dynamic_min_confirm_path
            and (
                mean_step >= config.dynamic_min_confirm_mean_step
                or resampled_mean_step >= config.dynamic_min_confirm_mean_step
            )
        )
        has_depth_trajectory = scale_growth >= config.dynamic_min_confirm_scale_growth

        if prediction.gesture_id == GestureID.PULL_TOWARD:
            return (
                _is_dynamic_static_pose_compatible(
                    prediction.gesture_id,
                    static_prediction,
                    self._last_static_gesture,
                    min_confidence=config.dynamic_static_compatibility_min_confidence,
                )
                and _has_confirmed_pull_motion(
                    [point.hand_size for point in points],
                    min_growth=config.min_pull_confirm_sustained_growth,
                    min_positive_ratio=config.min_pull_confirm_positive_step_ratio,
                )
            )
        if prediction.gesture_id in {GestureID.WAVE_LR, GestureID.CIRCLE}:
            return has_planar_trajectory and _is_dynamic_static_pose_compatible(
                prediction.gesture_id,
                static_prediction,
                self._last_static_gesture,
                min_confidence=config.dynamic_static_compatibility_min_confidence,
            )
        return has_planar_trajectory or has_depth_trajectory

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
        self._deferred_dynamic_prediction = None
        if clear_cooldown:
            self._dynamic_cooldown_frames = 0

    def _remember_deferred_dynamic_candidate(
        self,
        candidate_segment: TrajectoryBuffer | None,
        prediction: GesturePrediction,
        static_prediction: GesturePrediction,
    ) -> None:
        if candidate_segment is None:
            return
        if prediction.gesture_id != GestureID.PULL_TOWARD:
            return
        if not self._passes_dynamic_confirmation_gate(
            candidate_segment,
            prediction,
            static_prediction,
        ):
            return
        if (
            self._deferred_dynamic_prediction is None
            or prediction.confidence > self._deferred_dynamic_prediction.confidence
        ):
            self._deferred_dynamic_prediction = GesturePrediction(
                gesture_id=prediction.gesture_id,
                confidence=prediction.confidence,
                metadata={**prediction.metadata, "deferred_dynamic_candidate": True},
            )

    def _early_dynamic_prediction_from_candidate(
        self,
        candidate_segment: TrajectoryBuffer | None,
        prediction: GesturePrediction | None,
    ) -> GesturePrediction | None:
        if candidate_segment is None or prediction is None:
            return None

        min_points = max(
            self._config.dynamic_classifier.min_dynamic_segment_points,
            self._config.dynamic_classifier.early_dynamic_min_points,
        )
        if len(candidate_segment) < min_points:
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

    def _classify_dynamic_candidate(
        self,
        candidate_segment: TrajectoryBuffer | None,
        *,
        min_points: int,
        min_confidence: float,
    ) -> GesturePrediction | None:
        if candidate_segment is None or len(candidate_segment) < min_points:
            return None
        prediction = self._dynamic_classifier.classify(candidate_segment)
        if prediction.gesture_id == GestureID.UNKNOWN:
            return None
        if prediction.confidence < min_confidence:
            return None
        return prediction


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
    static_priority_active: bool = False,
    transition_grace_active: bool = False,
    dynamic_static_override_min_confidence: float = 0.85,
) -> GesturePrediction:
    """Select a command candidate while suppressing static commands during motion."""

    if _is_emergency_stop_candidate(static_prediction):
        return static_prediction

    if dynamic_state == "cooldown":
        return GesturePrediction.unknown(dynamic_prediction.metadata.get("reason", "cooldown"))

    if (
        dynamic_state in {"segment_classified", "confirming_dynamic", "deferred_dynamic_candidate"}
        and dynamic_prediction.gesture_id != GestureID.UNKNOWN
        and dynamic_prediction.confidence >= dynamic_min_confidence
        and not transition_grace_active
        and (
            not static_priority_active
            or dynamic_prediction.confidence >= dynamic_static_override_min_confidence
        )
    ):
        return dynamic_prediction

    if (
        (transition_grace_active or static_priority_active)
        and static_prediction.gesture_id != GestureID.UNKNOWN
    ):
        return static_prediction

    if dynamic_state in {
        "motion_started",
        "recording_dynamic",
        "confirming_dynamic",
        "early_dynamic_candidate",
        "segment_classified",
        "deferred_dynamic_candidate",
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


def _suppress_static_command_during_motion(
    prediction: GesturePrediction,
    config: CommandMappingConfig,
    dynamic_state: str,
) -> GesturePrediction:
    if not config.suppress_static_commands_during_motion:
        return prediction
    if dynamic_state not in _STATIC_COMMAND_SUPPRESSION_STATES:
        return prediction
    if not prediction.gesture_id.is_static or prediction.gesture_id == GestureID.THUMB_DOWN:
        return prediction
    return GesturePrediction(
        gesture_id=prediction.gesture_id,
        confidence=prediction.confidence,
        metadata={
            **prediction.metadata,
            "suppress_command": True,
            "suppress_reason": "static_command_suppressed_during_motion",
        },
    )


def _path_length_2d(points: list[tuple[float, float, float]]) -> float:
    return sum(
        math.hypot(second[0] - first[0], second[1] - first[1])
        for first, second in zip(points, points[1:], strict=False)
    )


def _resampled_mean_step_2d(
    points: list[tuple[float, float, float]],
    *,
    target_points: int = 8,
) -> float:
    if len(points) < 2:
        return 0.0
    if len(points) <= target_points:
        return _path_length_2d(points) / max(len(points) - 1, 1)

    last_index = len(points) - 1
    sampled_indices = [
        round(index * last_index / (target_points - 1)) for index in range(target_points)
    ]
    sampled_points = [points[index] for index in sampled_indices]
    return _path_length_2d(sampled_points) / max(len(sampled_points) - 1, 1)


def _scale_growth(values: list[float]) -> float:
    if not values:
        return 0.0
    low = max(min(values), 1e-9)
    return (max(values) - low) / low


def _is_dynamic_static_pose_compatible(
    dynamic_gesture: GestureID,
    static_prediction: GesturePrediction,
    last_static_gesture: GestureID,
    *,
    min_confidence: float,
) -> bool:
    if dynamic_gesture == GestureID.WAVE_LR:
        compatible = {GestureID.OPEN_PALM, GestureID.UNKNOWN}
    elif dynamic_gesture == GestureID.PULL_TOWARD:
        compatible = {GestureID.OPEN_PALM, GestureID.UNKNOWN}
    elif dynamic_gesture == GestureID.CIRCLE:
        compatible = {
            GestureID.FIST,
            GestureID.INDEX_LEFT,
            GestureID.INDEX_RIGHT,
            GestureID.PINKY,
            GestureID.UNKNOWN,
        }
    else:
        return True

    if static_prediction.confidence >= min_confidence:
        return static_prediction.gesture_id in compatible
    if last_static_gesture != GestureID.UNKNOWN:
        return last_static_gesture in compatible
    return True


def _has_confirmed_pull_motion(
    hand_sizes: list[float],
    *,
    min_growth: float,
    min_positive_ratio: float,
) -> bool:
    if len(hand_sizes) < 2:
        return False
    early_size, late_size = _edge_medians(hand_sizes)
    sustained_growth = (late_size - early_size) / max(early_size, 1e-9)
    positive_ratio = _positive_step_ratio(
        hand_sizes,
        tolerance=max(_mean(hand_sizes) * 0.01, 1e-4),
    )
    return sustained_growth >= min_growth and positive_ratio >= min_positive_ratio


def _edge_medians(values: list[float]) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    window_size = max(1, len(values) // 4)
    return (_median(values[:window_size]), _median(values[-window_size:]))


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    middle = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[middle]
    return (sorted_values[middle - 1] + sorted_values[middle]) / 2


def _positive_step_ratio(values: list[float], *, tolerance: float) -> float:
    if len(values) < 2:
        return 0.0
    positive_steps = sum(
        1 for first, second in zip(values, values[1:], strict=False) if second >= first - tolerance
    )
    return positive_steps / (len(values) - 1)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
