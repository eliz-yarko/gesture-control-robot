"""Frame-to-command pipeline for the gesture control subsystem."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.capture.video_capture import CapturedFrame
from src.config import AppConfig
from src.domain import CommandEvent, GestureID, GesturePrediction
from src.interpretation.command_mapper import CommandMapper
from src.recognition.dynamic_classifier import DynamicGestureClassifier
from src.recognition.hand_detector import HandDetection, HandDetector
from src.recognition.static_classifier import StaticGestureClassifier
from src.recognition.trajectory_buffer import TrajectoryBuffer
from src.transmission.base_sender import CommandSender


class HandDetectorProtocol(Protocol):
    """Minimal detector contract used by the pipeline."""

    def detect(self, rgb_frame: object) -> list[HandDetection]:
        """Return hand detections for an RGB frame."""

    def close(self) -> None:
        """Release detector resources."""


@dataclass(frozen=True)
class PipelineResult:
    """Result produced after processing one frame."""

    frame_index: int
    detections: list[HandDetection]
    static_prediction: GesturePrediction
    dynamic_prediction: GesturePrediction
    selected_prediction: GesturePrediction
    command_event: CommandEvent | None


class GestureControlPipeline:
    """Connect hand detection, classification, debouncing, and command sending."""

    def __init__(
        self,
        config: AppConfig | None = None,
        detector: HandDetectorProtocol | None = None,
        static_classifier: StaticGestureClassifier | None = None,
        dynamic_classifier: DynamicGestureClassifier | None = None,
        trajectory_buffer: TrajectoryBuffer | None = None,
        command_mapper: CommandMapper | None = None,
        command_sender: CommandSender | None = None,
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
            )

        detection = max(detections, key=lambda item: item.score)
        static_prediction = self._static_classifier.classify(detection.landmarks)
        self._trajectory_buffer.add_landmarks(detection.landmarks, timestamp=frame.timestamp)
        dynamic_prediction = self._dynamic_classifier.classify(self._trajectory_buffer)
        selected_prediction = _select_prediction(static_prediction, dynamic_prediction)
        command_event = self._command_mapper.update(selected_prediction)

        if command_event is not None and self._command_sender is not None:
            self._command_sender.send(command_event)

        return PipelineResult(
            frame_index=frame.index,
            detections=detections,
            static_prediction=static_prediction,
            dynamic_prediction=dynamic_prediction,
            selected_prediction=selected_prediction,
            command_event=command_event,
        )

    def close(self) -> None:
        """Close resources owned by the detector."""

        self._detector.close()


def _select_prediction(
    static_prediction: GesturePrediction,
    dynamic_prediction: GesturePrediction,
) -> GesturePrediction:
    """Prefer confirmed dynamic gestures over per-frame static gestures."""

    if (
        dynamic_prediction.gesture_id != GestureID.UNKNOWN
        and dynamic_prediction.confidence >= static_prediction.confidence
    ):
        return dynamic_prediction
    if (
        dynamic_prediction.gesture_id != GestureID.UNKNOWN
        and static_prediction.gesture_id == GestureID.UNKNOWN
    ):
        return dynamic_prediction
    return static_prediction
