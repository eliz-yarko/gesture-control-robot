"""Dynamic gesture classifier based on trajectory heuristics."""

from __future__ import annotations

import math
from collections.abc import Callable
from statistics import fmean, median, pstdev

from src.config import DynamicClassifierConfig
from src.domain import GestureID, GesturePrediction
from src.recognition.trajectory_buffer import TrajectoryBuffer, TrajectoryPoint


class DynamicGestureClassifier:
    """Recognize dynamic gestures from a fixed-size trajectory buffer.

    The trained model path uses the same trajectory data, so this classifier remains
    useful as a transparent fallback and as a comparison point in benchmark runs.
    """

    def __init__(self, config: DynamicClassifierConfig | None = None) -> None:
        """Initialize the classifier."""

        self._config = config or DynamicClassifierConfig()

    def classify(self, buffer: TrajectoryBuffer) -> GesturePrediction:
        """Classify a dynamic gesture from the current trajectory window."""

        points = buffer.points()
        window_points = min(self._config.min_window_points, self._config.buffer_size)
        if len(points) < window_points:
            return GesturePrediction.unknown("trajectory_buffer_not_ready")

        recent_points = points[-self._config.buffer_size :]
        for detector in (
            self._detect_wave_lr,
            self._detect_circle,
            self._detect_pull_toward,
        ):
            prediction = _best_prediction_for_recent_windows(
                recent_points,
                min_window_points=window_points,
                detector=detector,
            )
            if prediction is not None:
                return prediction

        return GesturePrediction.unknown("dynamic_rules_no_match")

    def _detect_wave_lr(self, points: list[TrajectoryPoint]) -> GesturePrediction | None:
        candidates = [
            self._detect_wave_path(
                points,
                label="palm",
                x_accessor=lambda point: point.palm_center[0],
                y_accessor=lambda point: point.palm_center[1],
            ),
            self._detect_wave_path(
                points,
                label="index",
                x_accessor=lambda point: point.index_tip[0],
                y_accessor=lambda point: point.index_tip[1],
            ),
            self._detect_wave_template_path(
                points,
                label="palm_template",
                x_accessor=lambda point: point.palm_center[0],
                y_accessor=lambda point: point.palm_center[1],
            ),
            self._detect_wave_template_path(
                points,
                label="index_template",
                x_accessor=lambda point: point.index_tip[0],
                y_accessor=lambda point: point.index_tip[1],
            ),
        ]
        predictions = [prediction for prediction in candidates if prediction is not None]
        if not predictions:
            return None
        return max(predictions, key=lambda prediction: prediction.confidence)

    def _detect_wave_path(
        self,
        points: list[TrajectoryPoint],
        *,
        label: str,
        x_accessor: Callable[[TrajectoryPoint], float],
        y_accessor: Callable[[TrajectoryPoint], float],
    ) -> GesturePrediction | None:
        xs = _smoothed([x_accessor(point) for point in points])
        ys = _smoothed([y_accessor(point) for point in points])
        horizontal_range = _range(xs)
        vertical_range = _range(ys)
        vertical_residual_range = _range(_detrended(ys))
        horizontal_path = _path_length_1d(xs)
        vertical_path = _path_length_1d(ys)
        direction_changes = _direction_changes(
            xs,
            min_delta=max(0.01, horizontal_range * 0.08),
        )

        if (
            horizontal_range >= self._config.min_horizontal_displacement
            and vertical_residual_range <= self._config.max_vertical_drift
            and horizontal_path >= max(vertical_path * 1.25, horizontal_range * 1.35)
            and direction_changes >= self._config.min_wave_direction_changes
        ):
            drift_penalty = max(0.0, vertical_residual_range - self._config.max_vertical_drift / 2)
            confidence = min(
                0.98,
                0.62 + horizontal_range + direction_changes * 0.05 - drift_penalty * 0.5,
            )
            return GesturePrediction(
                GestureID.WAVE_LR,
                confidence,
                {
                    "horizontal_range": horizontal_range,
                    "vertical_range": vertical_range,
                    "vertical_residual_range": vertical_residual_range,
                    "horizontal_path": horizontal_path,
                    "vertical_path": vertical_path,
                    "direction_changes": direction_changes,
                    "tracked_point": label,
                },
            )
        return None

    def _detect_wave_template_path(
        self,
        points: list[TrajectoryPoint],
        *,
        label: str,
        x_accessor: Callable[[TrajectoryPoint], float],
        y_accessor: Callable[[TrajectoryPoint], float],
    ) -> GesturePrediction | None:
        xs = _smoothed([x_accessor(point) for point in points])
        ys = _smoothed([y_accessor(point) for point in points])
        horizontal_range = _range(xs)
        vertical_residual_range = _range(_detrended(ys))
        horizontal_path = _path_length_1d(xs)
        direction_changes = _direction_changes(
            xs,
            min_delta=max(0.008, horizontal_range * 0.06),
        )
        if (
            horizontal_range < self._config.min_horizontal_displacement * 0.8
            or vertical_residual_range > self._config.max_vertical_drift * 1.6
            or horizontal_path < horizontal_range * 1.2
            or direction_changes < self._config.min_wave_direction_changes
        ):
            return None

        series = _normalize_series(_resample_series(xs, 16))
        template = _normalize_series(
            _resample_series([-1.0, 0.55, 1.0, 0.2, -0.9, -1.0, -0.1, 0.85, 1.0], 16)
        )
        distance = min(
            _dtw_distance(series, template),
            _dtw_distance([-value for value in series], template),
        )
        if distance > 0.28:
            return None

        confidence = min(
            0.94,
            0.72
            + min(horizontal_range, 0.3) * 0.35
            + direction_changes * 0.035
            + max(0.0, 0.28 - distance) * 0.45,
        )
        return GesturePrediction(
            GestureID.WAVE_LR,
            confidence,
            {
                "horizontal_range": horizontal_range,
                "vertical_residual_range": vertical_residual_range,
                "horizontal_path": horizontal_path,
                "direction_changes": direction_changes,
                "tracked_point": label,
                "template_distance": distance,
            },
        )

    def _detect_circle(self, points: list[TrajectoryPoint]) -> GesturePrediction | None:
        xs = _smoothed([point.index_tip[0] for point in points])
        ys = _smoothed([point.index_tip[1] for point in points])
        center_x = fmean(xs)
        center_y = fmean(ys)
        radii = [math.hypot(x - center_x, y - center_y) for x, y in zip(xs, ys, strict=True)]
        mean_radius = fmean(radii)
        if mean_radius < self._config.min_circle_radius:
            return None

        radius_cv = pstdev(radii) / mean_radius if len(radii) > 1 else 0.0
        angles = [math.atan2(y - center_y, x - center_x) for x, y in zip(xs, ys, strict=True)]
        angle_span = _unwrapped_angle_span(angles)
        x_range = _range(xs)
        y_range = _range(ys)
        axis_balance = min(x_range, y_range) / max(x_range, y_range, 1e-9)
        path_length = _path_length_2d(xs, ys)
        closure_distance = math.hypot(xs[-1] - xs[0], ys[-1] - ys[0])
        closure_ratio = closure_distance / max(x_range, y_range, 1e-9)

        if (
            radius_cv <= self._config.max_circle_radius_cv
            and angle_span >= self._config.min_circle_angle_span
            and angle_span <= self._config.max_circle_angle_span
            and axis_balance >= 0.35
            and path_length >= mean_radius * angle_span * 0.6
            and closure_ratio <= 1.25
        ):
            confidence = min(
                0.98,
                0.62
                + (angle_span / (2 * math.pi)) * 0.24
                + axis_balance * 0.08
                + max(0.0, self._config.max_circle_radius_cv - radius_cv) * 0.04,
            )
            return GesturePrediction(
                GestureID.CIRCLE,
                confidence,
                {
                    "mean_radius": mean_radius,
                    "radius_cv": radius_cv,
                    "angle_span": angle_span,
                    "axis_balance": axis_balance,
                    "path_length": path_length,
                    "closure_ratio": closure_ratio,
                },
            )
        return self._detect_circle_template(
            mean_radius=mean_radius,
            radius_cv=radius_cv,
            angle_span=angle_span,
            axis_balance=axis_balance,
            path_length=path_length,
            closure_ratio=closure_ratio,
        )

    def _detect_circle_template(
        self,
        *,
        mean_radius: float,
        radius_cv: float,
        angle_span: float,
        axis_balance: float,
        path_length: float,
        closure_ratio: float,
    ) -> GesturePrediction | None:
        if (
            mean_radius < self._config.min_circle_radius * 0.9
            or radius_cv > self._config.max_circle_radius_cv * 1.15
            or angle_span < self._config.min_circle_angle_span * 0.92
            or angle_span > self._config.max_circle_angle_span
            or axis_balance < 0.32
            or path_length < mean_radius * angle_span * 0.5
            or closure_ratio > 1.45
        ):
            return None

        confidence = min(
            0.9,
            0.66
            + (angle_span / (2 * math.pi)) * 0.18
            + axis_balance * 0.08
            + max(0.0, self._config.max_circle_radius_cv - radius_cv) * 0.035,
        )
        return GesturePrediction(
            GestureID.CIRCLE,
            confidence,
            {
                "mean_radius": mean_radius,
                "radius_cv": radius_cv,
                "angle_span": angle_span,
                "axis_balance": axis_balance,
                "path_length": path_length,
                "closure_ratio": closure_ratio,
                "template_match": "circle",
            },
        )

    def _detect_pull_toward(self, points: list[TrajectoryPoint]) -> GesturePrediction | None:
        sizes = _smoothed([point.hand_size for point in points])
        palm_xs = _smoothed([point.palm_center[0] for point in points])
        palm_ys = _smoothed([point.palm_center[1] for point in points])
        early_size, late_size = _edge_medians(sizes)
        half_start_size, half_end_size = _half_medians(sizes)
        reference_size = max(min(early_size, half_start_size), 1e-9)
        if reference_size <= 0.0:
            return None

        scale_growth = (late_size - early_size) / reference_size
        half_growth = (half_end_size - half_start_size) / reference_size
        peak_growth = (max(sizes) - min(sizes)) / max(min(sizes), 1e-9)
        positive_step_ratio = _positive_step_ratio(
            sizes,
            tolerance=max(reference_size * 0.01, 1e-4),
        )
        slope = _linear_slope(sizes)
        sustained_growth = max(scale_growth, half_growth)
        planar_range = max(_range(palm_xs), _range(palm_ys))
        planar_path = _path_length_2d(palm_xs, palm_ys)
        has_enough_growth = sustained_growth >= self._config.min_pull_scale_growth
        has_limited_planar_motion = planar_range <= max(
            self._config.min_horizontal_displacement * 1.35,
            sustained_growth * 0.9,
        )

        if (
            has_enough_growth
            and has_limited_planar_motion
            and slope > 0.0
            and positive_step_ratio >= 0.55
        ):
            confidence = min(0.98, 0.68 + max(scale_growth, half_growth, peak_growth * 0.65))
            return GesturePrediction(
                GestureID.PULL_TOWARD,
                confidence,
                {
                    "scale_growth": scale_growth,
                    "half_growth": half_growth,
                    "peak_growth": peak_growth,
                    "sustained_growth": sustained_growth,
                    "positive_step_ratio": positive_step_ratio,
                    "scale_slope": slope,
                    "planar_range": planar_range,
                    "planar_path": planar_path,
                },
            )
        return self._detect_pull_template(
            sizes=sizes,
            scale_growth=scale_growth,
            half_growth=half_growth,
            peak_growth=peak_growth,
            positive_step_ratio=positive_step_ratio,
            slope=slope,
            planar_range=planar_range,
            planar_path=planar_path,
        )

    def _detect_pull_template(
        self,
        *,
        sizes: list[float],
        scale_growth: float,
        half_growth: float,
        peak_growth: float,
        positive_step_ratio: float,
        slope: float,
        planar_range: float,
        planar_path: float,
    ) -> GesturePrediction | None:
        sustained_growth = max(scale_growth, half_growth, peak_growth)
        has_limited_planar_motion = planar_range <= max(
            self._config.min_horizontal_displacement * 1.45,
            sustained_growth,
        )
        if (
            sustained_growth < self._config.min_pull_scale_growth * 0.75
            or not has_limited_planar_motion
            or slope <= 0.0
            or positive_step_ratio < 0.48
        ):
            return None

        series = _normalize_series(_resample_series(sizes, 16))
        template = _normalize_series([index / 15 for index in range(16)])
        distance = _dtw_distance(series, template)
        if distance > 0.24:
            return None

        confidence = min(
            0.92,
            0.66 + sustained_growth + positive_step_ratio * 0.08 + max(0.0, 0.24 - distance) * 0.45,
        )
        return GesturePrediction(
            GestureID.PULL_TOWARD,
            confidence,
            {
                "scale_growth": scale_growth,
                "half_growth": half_growth,
                "peak_growth": peak_growth,
                "sustained_growth": sustained_growth,
                "positive_step_ratio": positive_step_ratio,
                "scale_slope": slope,
                "planar_range": planar_range,
                "planar_path": planar_path,
                "template_distance": distance,
            },
        )


def _direction_changes(values: list[float], min_delta: float = 1e-6) -> int:
    changes = 0
    previous_sign = 0
    for first, second in zip(values, values[1:], strict=False):
        delta = second - first
        if abs(delta) < min_delta:
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


def _smoothed(values: list[float]) -> list[float]:
    if len(values) < 7:
        return values
    return [
        fmean(values[max(0, index - 1) : min(len(values), index + 2)])
        for index in range(len(values))
    ]


def _range(values: list[float]) -> float:
    return max(values) - min(values) if values else 0.0


def _detrended(values: list[float]) -> list[float]:
    if len(values) < 2:
        return values
    start = values[0]
    end = values[-1]
    last_index = len(values) - 1
    return [
        value - (start + (end - start) * (index / last_index)) for index, value in enumerate(values)
    ]


def _path_length_1d(values: list[float]) -> float:
    return sum(abs(second - first) for first, second in zip(values, values[1:], strict=False))


def _path_length_2d(xs: list[float], ys: list[float]) -> float:
    return sum(
        math.hypot(second_x - first_x, second_y - first_y)
        for first_x, first_y, second_x, second_y in zip(xs, ys, xs[1:], ys[1:], strict=False)
    )


def _edge_medians(values: list[float]) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    window_size = max(1, len(values) // 4)
    return (median(values[:window_size]), median(values[-window_size:]))


def _half_medians(values: list[float]) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    middle = max(1, len(values) // 2)
    return (median(values[:middle]), median(values[middle:]) if values[middle:] else values[-1])


def _positive_step_ratio(values: list[float], tolerance: float) -> float:
    if len(values) < 2:
        return 0.0
    positive_steps = sum(
        1 for first, second in zip(values, values[1:], strict=False) if second >= first - tolerance
    )
    return positive_steps / (len(values) - 1)


def _linear_slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_x = (len(values) - 1) / 2
    mean_y = fmean(values)
    denominator = sum((index - mean_x) ** 2 for index in range(len(values)))
    if denominator == 0.0:
        return 0.0
    numerator = sum((index - mean_x) * (value - mean_y) for index, value in enumerate(values))
    return numerator / denominator


def _resample_series(values: list[float], target_count: int) -> list[float]:
    if not values:
        return [0.0] * target_count
    if len(values) == 1:
        return [values[0]] * target_count
    if target_count <= 1:
        return [values[0]]

    last_index = len(values) - 1
    result: list[float] = []
    for output_index in range(target_count):
        source_position = output_index * last_index / (target_count - 1)
        left_index = math.floor(source_position)
        right_index = min(last_index, left_index + 1)
        fraction = source_position - left_index
        result.append(values[left_index] * (1 - fraction) + values[right_index] * fraction)
    return result


def _normalize_series(values: list[float]) -> list[float]:
    if not values:
        return []
    value_range = _range(values)
    if value_range < 1e-9:
        return [0.0 for _ in values]
    center = fmean(values)
    return [(value - center) / (value_range / 2) for value in values]


def _dtw_distance(first: list[float], second: list[float]) -> float:
    if not first or not second:
        return math.inf

    previous = [math.inf] * (len(second) + 1)
    previous[0] = 0.0
    for first_value in first:
        current = [math.inf] * (len(second) + 1)
        for second_index, second_value in enumerate(second, start=1):
            cost = abs(first_value - second_value)
            current[second_index] = cost + min(
                current[second_index - 1],
                previous[second_index],
                previous[second_index - 1],
            )
        previous = current
    return previous[-1] / (len(first) + len(second))


def _best_prediction_for_recent_windows(
    points: list[TrajectoryPoint],
    min_window_points: int,
    detector: Callable[[list[TrajectoryPoint]], GesturePrediction | None],
) -> GesturePrediction | None:
    best_prediction: GesturePrediction | None = None
    for window in _recent_windows(points, min_window_points):
        prediction = detector(window)
        if prediction is None:
            continue
        prediction = GesturePrediction(
            gesture_id=prediction.gesture_id,
            confidence=prediction.confidence,
            metadata={
                **prediction.metadata,
                "window_points": len(window),
            },
        )
        if best_prediction is None or prediction.confidence > best_prediction.confidence:
            best_prediction = prediction
    return best_prediction


def _recent_windows(
    points: list[TrajectoryPoint],
    min_window_points: int,
) -> list[list[TrajectoryPoint]]:
    return [points[-window_size:] for window_size in range(len(points), min_window_points - 1, -1)]
