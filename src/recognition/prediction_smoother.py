"""Temporal smoothing for frame-level gesture predictions."""

from __future__ import annotations

from collections import Counter, defaultdict, deque

from src.config import PredictionSmoothingConfig
from src.domain import GestureID, GesturePrediction


class StaticPredictionSmoother:
    """Stabilize static predictions over a short frame history."""

    def __init__(self, config: PredictionSmoothingConfig | None = None) -> None:
        """Initialize an empty smoothing window."""

        self._config = config or PredictionSmoothingConfig()
        self._history: deque[GesturePrediction] = deque(
            maxlen=max(1, self._config.static_window_size)
        )

    def update(self, prediction: GesturePrediction) -> GesturePrediction:
        """Return a stabilized prediction for the current frame."""

        if not self._config.enabled:
            return prediction

        if prediction.gesture_id == GestureID.THUMB_DOWN:
            self.reset()
            self._history.append(prediction)
            return prediction

        self._history.append(prediction)
        if prediction.gesture_id != GestureID.UNKNOWN:
            return self._smooth_known_prediction(prediction)
        return self._smooth_unknown_prediction(prediction)

    def reset(self) -> None:
        """Clear the smoothing history."""

        self._history.clear()

    def _smooth_known_prediction(self, prediction: GesturePrediction) -> GesturePrediction:
        known_predictions = _known_predictions(self._history, self._config.static_min_confidence)
        if not known_predictions:
            return prediction

        selected_label, selected_confidence, selected_votes = _best_history_label(known_predictions)
        current_votes = sum(
            1 for item in known_predictions if item.gesture_id == prediction.gesture_id
        )
        if (
            selected_label != prediction.gesture_id
            and selected_votes >= self._config.static_min_votes
            and selected_votes > current_votes
            and selected_confidence >= prediction.confidence + self._config.conflict_margin
        ):
            return GesturePrediction(
                gesture_id=selected_label,
                confidence=selected_confidence,
                metadata={
                    "smoothed_from": prediction.gesture_id.name,
                    "smoothing_votes": selected_votes,
                },
            )
        return prediction

    def _smooth_unknown_prediction(self, prediction: GesturePrediction) -> GesturePrediction:
        known_predictions = _known_predictions(self._history, self._config.static_min_confidence)
        if not known_predictions:
            return prediction

        selected_label, selected_confidence, selected_votes = _best_history_label(known_predictions)
        if selected_votes < self._config.static_min_votes:
            return prediction

        return GesturePrediction(
            gesture_id=selected_label,
            confidence=selected_confidence,
            metadata={
                **prediction.metadata,
                "smoothed_from": GestureID.UNKNOWN.name,
                "smoothing_votes": selected_votes,
            },
        )


def _known_predictions(
    history: deque[GesturePrediction],
    min_confidence: float,
) -> list[GesturePrediction]:
    return [
        prediction
        for prediction in history
        if prediction.gesture_id != GestureID.UNKNOWN and prediction.confidence >= min_confidence
    ]


def _best_history_label(
    predictions: list[GesturePrediction],
) -> tuple[GestureID, float, int]:
    counts = Counter(prediction.gesture_id for prediction in predictions)
    confidence_by_label: dict[GestureID, list[float]] = defaultdict(list)
    for prediction in predictions:
        confidence_by_label[prediction.gesture_id].append(prediction.confidence)

    return max(
        (
            (
                gesture_id,
                sum(confidences) / len(confidences),
                counts[gesture_id],
            )
            for gesture_id, confidences in confidence_by_label.items()
        ),
        key=lambda item: (item[2], item[1]),
    )
