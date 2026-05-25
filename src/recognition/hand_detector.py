"""MediaPipe Hands wrapper."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from src.config import HandDetectionConfig
from src.utils.geometry import Landmark


@dataclass(frozen=True)
class HandDetection:
    """Detected hand landmarks and metadata."""

    landmarks: list[Landmark]
    handedness: str
    score: float


class HandDetector:
    """Detect hand landmarks in RGB frames using MediaPipe Hands."""

    def __init__(
        self,
        config: HandDetectionConfig | None = None,
        hands: Any | None = None,
        mediapipe_module: Any | None = None,
    ) -> None:
        """Initialize the detector.

        Args:
            config: MediaPipe detector thresholds.
            hands: Optional prebuilt MediaPipe Hands-like object for tests.
            mediapipe_module: Optional module injection for tests.
        """

        self._config = config or HandDetectionConfig()
        self._hands = hands
        self._mediapipe = mediapipe_module

    def detect(self, rgb_frame: Any) -> list[HandDetection]:
        """Detect hands and return normalized landmarks."""

        hands = self._require_hands()
        result = hands.process(rgb_frame)
        return self._convert_result(result)

    def close(self) -> None:
        """Close the MediaPipe graph if it was initialized."""

        if self._hands is None:
            return
        close = getattr(self._hands, "close", None)
        if callable(close):
            close()
        self._hands = None

    def __enter__(self) -> HandDetector:
        """Initialize MediaPipe Hands when entering a context manager."""

        self._require_hands()
        return self

    def __exit__(self, *args: object) -> None:
        """Close MediaPipe Hands when leaving a context manager."""

        self.close()

    def _require_hands(self) -> Any:
        if self._hands is not None:
            return self._hands

        mediapipe = self._require_mediapipe()
        self._hands = mediapipe.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=self._config.max_num_hands,
            min_detection_confidence=self._config.min_detection_confidence,
            min_tracking_confidence=self._config.min_tracking_confidence,
        )
        return self._hands

    def _require_mediapipe(self) -> Any:
        if self._mediapipe is not None:
            return self._mediapipe

        try:
            mediapipe = importlib.import_module("mediapipe")
        except ImportError as exc:
            raise RuntimeError("MediaPipe is required for HandDetector.") from exc

        self._mediapipe = mediapipe
        return mediapipe

    @staticmethod
    def _convert_result(result: Any) -> list[HandDetection]:
        hand_landmarks = getattr(result, "multi_hand_landmarks", None) or []
        handedness_items = getattr(result, "multi_handedness", None) or []

        detections: list[HandDetection] = []
        for index, landmarks_item in enumerate(hand_landmarks):
            label, score = _classification_at(handedness_items, index)
            detections.append(
                HandDetection(
                    landmarks=[
                        (float(point.x), float(point.y), float(point.z))
                        for point in landmarks_item.landmark
                    ],
                    handedness=label,
                    score=score,
                )
            )
        return detections


def _classification_at(handedness_items: list[Any], index: int) -> tuple[str, float]:
    if index >= len(handedness_items):
        return ("Unknown", 0.0)

    classification = getattr(handedness_items[index], "classification", [])
    if not classification:
        return ("Unknown", 0.0)

    first = classification[0]
    return (str(getattr(first, "label", "Unknown")), float(getattr(first, "score", 0.0)))
