"""Baseline dynamic gesture classifier using trajectory heuristics."""

from __future__ import annotations

import math
from statistics import fmean, pstdev

from src.config import DynamicClassifierConfig
from src.domain import GestureID, GesturePrediction
from src.recognition.trajectory_buffer import TrajectoryBuffer, TrajectoryPoint


class DynamicGestureClassifier:
    """Recognize dynamic gestures from a fixed-size trajectory buffer.

    This class is the baseline state-machine layer. The planned LSTM classifier
    will use the same buffer data contract, which keeps experiments comparable.
    """

    def __init__(self, config: DynamicClassifierConfig | None = None) -> None:
        """Initialize the classifier."""

        self._config = config or DynamicClassifierConfig()

    def classify(self, buffer: TrajectoryBuffer) -> GesturePrediction:
        """Classify a dynamic gesture from the current trajectory window."""

        points = buffer.points()
        if len(points) < self._config.buffer_size:
            return GesturePrediction.unknown("trajectory_buffer_not_ready")

        pull = self._detect_pull_toward(points)
        if pull is not None:
            return pull

        circle = self._detect_circle(points)
        if circle is not None:
            return circle

        wave = self._detect_wave_lr(points)
        if wave is not None:
            return wave

        return GesturePrediction.unknown("dynamic_rules_no_match")

    def _detect_wave_lr(self, points: list[TrajectoryPoint]) -> GesturePrediction | None:
        xs = [point.palm_center[0] for point in points]
        ys = [point.palm_center[1] for point in points]
        horizontal_range = max(xs) - min(xs)
        vertical_range = max(ys) - min(ys)
        direction_changes = _direction_changes(xs)

        if (
            horizontal_range >= self._config.min_horizontal_displacement
            and vertical_range <= self._config.max_vertical_drift
            and direction_changes >= self._config.min_wave_direction_changes
        ):
            confidence = min(0.98, 0.65 + horizontal_range + direction_changes * 0.05)
            return GesturePrediction(
                GestureID.WAVE_LR,
                confidence,
                {
                    "horizontal_range": horizontal_range,
                    "vertical_range": vertical_range,
                    "direction_changes": direction_changes,
                },
            )
        return None

    def _detect_circle(self, points: list[TrajectoryPoint]) -> GesturePrediction | None:
        xs = [point.index_tip[0] for point in points]
        ys = [point.index_tip[1] for point in points]
        center_x = fmean(xs)
        center_y = fmean(ys)
        radii = [math.hypot(x - center_x, y - center_y) for x, y in zip(xs, ys, strict=True)]
        mean_radius = fmean(radii)
        if mean_radius < self._config.min_circle_radius:
            return None

        radius_cv = pstdev(radii) / mean_radius if len(radii) > 1 else 0.0
        angles = [math.atan2(y - center_y, x - center_x) for x, y in zip(xs, ys, strict=True)]
        angle_span = _unwrapped_angle_span(angles)

        if (
            radius_cv <= self._config.max_circle_radius_cv
            and angle_span >= self._config.min_circle_angle_span
        ):
            confidence = min(0.98, 0.65 + (angle_span / (2 * math.pi)) * 0.25)
            return GesturePrediction(
                GestureID.CIRCLE,
                confidence,
                {
                    "mean_radius": mean_radius,
                    "radius_cv": radius_cv,
                    "angle_span": angle_span,
                },
            )
        return None

    def _detect_pull_toward(self, points: list[TrajectoryPoint]) -> GesturePrediction | None:
        first_size = points[0].hand_size
        last_size = points[-1].hand_size
        if first_size <= 0.0:
            return None

        scale_growth = (last_size - first_size) / first_size
        if scale_growth >= self._config.min_pull_scale_growth and _mostly_increasing(
            [point.hand_size for point in points]
        ):
            confidence = min(0.98, 0.68 + scale_growth)
            return GesturePrediction(
                GestureID.PULL_TOWARD,
                confidence,
                {"scale_growth": scale_growth},
            )
        return None


def _direction_changes(values: list[float]) -> int:
    changes = 0
    previous_sign = 0
    for first, second in zip(values, values[1:], strict=False):
        delta = second - first
        if abs(delta) < 1e-6:
            continue
        sign = 1 if delta > 0 else -1
        if previous_sign != 0 and sign != previous_sign:
            changes += 1
        previous_sign = sign
    return changes


def _unwrapped_angle_span(angles: list[float]) -> float:
    if not angles:
        return 0.0

    total = 0.0
    previous = angles[0]
    for angle in angles[1:]:
        delta = angle - previous
        while delta > math.pi:
            delta -= 2 * math.pi
        while delta < -math.pi:
            delta += 2 * math.pi
        total += delta
        previous = angle
    return abs(total)


def _mostly_increasing(values: list[float]) -> bool:
    if len(values) < 2:
        return False
    increases = sum(1 for first, second in zip(values, values[1:], strict=False) if second >= first)
    return increases / (len(values) - 1) >= 0.75
