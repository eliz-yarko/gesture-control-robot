"""Feature extraction for landmark-based gesture models."""

from __future__ import annotations

import math
from statistics import fmean, median, pstdev

from src.recognition.trajectory_buffer import TrajectoryPoint
from src.utils.geometry import (
    LandmarkSequence,
    distance,
    normalize_landmarks,
    palm_center,
    to_landmarks,
)

STATIC_FEATURE_VERSION = "static_landmarks_v1"
DYNAMIC_FEATURE_VERSION = "trajectory_landmarks_v1"


def extract_static_features(raw_landmarks: LandmarkSequence) -> tuple[float, ...]:
    """Extract normalized single-frame features for a static gesture model."""

    landmarks = to_landmarks(raw_landmarks)
    normalized = normalize_landmarks(landmarks)
    center = palm_center(landmarks)
    features: list[float] = []

    for point in normalized:
        features.extend(point)

    for mcp_index, pip_index, dip_index, tip_index in _FINGER_INDICES:
        tip = landmarks[tip_index]
        mcp = landmarks[mcp_index]
        features.extend(_vector_2d(mcp, tip))
        features.append(distance(center, tip))
        features.append(distance(mcp, tip))
        features.append(
            _joint_angle(landmarks[mcp_index], landmarks[pip_index], landmarks[dip_index])
        )
        features.append(
            _joint_angle(landmarks[pip_index], landmarks[dip_index], landmarks[tip_index])
        )

    for first_tip, second_tip in _TIP_PAIRS:
        features.append(distance(landmarks[first_tip], landmarks[second_tip]))

    return tuple(features)


def extract_trajectory_features(points: list[TrajectoryPoint]) -> tuple[float, ...]:
    """Extract aggregate trajectory features for a dynamic gesture model."""

    if not points:
        return (0.0,) * 32

    palm_xs = [point.palm_center[0] for point in points]
    palm_ys = [point.palm_center[1] for point in points]
    index_xs = [point.index_tip[0] for point in points]
    index_ys = [point.index_tip[1] for point in points]
    sizes = [point.hand_size for point in points]

    palm_radius, palm_radius_cv, palm_angle_span = _circular_features(palm_xs, palm_ys)
    index_radius, index_radius_cv, index_angle_span = _circular_features(index_xs, index_ys)
    early_size, late_size = _early_late_medians(sizes)
    half_start_size, half_end_size = _half_medians(sizes)
    min_size = min(sizes)
    max_size = max(sizes)

    return (
        float(len(points)),
        _range(palm_xs),
        _range(palm_ys),
        _range(index_xs),
        _range(index_ys),
        _displacement(palm_xs, palm_ys),
        _displacement(index_xs, index_ys),
        _path_length(palm_xs, palm_ys),
        _path_length(index_xs, index_ys),
        float(_direction_changes(palm_xs)),
        float(_direction_changes(palm_ys)),
        float(_direction_changes(index_xs)),
        float(_direction_changes(index_ys)),
        fmean(sizes),
        pstdev(sizes) if len(sizes) > 1 else 0.0,
        _safe_ratio(sizes[-1] - sizes[0], sizes[0]),
        _safe_ratio(late_size - early_size, early_size),
        _safe_ratio(half_end_size - half_start_size, half_start_size),
        _safe_ratio(max_size - min_size, min_size),
        palm_radius,
        palm_radius_cv,
        palm_angle_span,
        index_radius,
        index_radius_cv,
        index_angle_span,
        _range([point.palm_center[2] for point in points]),
        _range([point.index_tip[2] for point in points]),
        _safe_ratio(
            points[-1].palm_center[2] - points[0].palm_center[2], abs(points[0].palm_center[2])
        ),
        _safe_ratio(points[-1].index_tip[2] - points[0].index_tip[2], abs(points[0].index_tip[2])),
        _safe_ratio(_range(index_xs), max(_range(index_ys), 1e-9)),
        _safe_ratio(_range(palm_xs), max(_range(palm_ys), 1e-9)),
        _safe_ratio(index_radius, max(palm_radius, 1e-9)),
    )


_FINGER_INDICES = (
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
)

_TIP_PAIRS = (
    (4, 8),
    (8, 12),
    (12, 16),
    (16, 20),
    (4, 20),
)


def _vector_2d(
    start: tuple[float, float, float], end: tuple[float, float, float]
) -> tuple[float, float]:
    return (end[0] - start[0], end[1] - start[1])


def _joint_angle(
    first: tuple[float, float, float],
    middle: tuple[float, float, float],
    last: tuple[float, float, float],
) -> float:
    first_vector = (first[0] - middle[0], first[1] - middle[1], first[2] - middle[2])
    second_vector = (last[0] - middle[0], last[1] - middle[1], last[2] - middle[2])
    first_norm = math.sqrt(sum(value * value for value in first_vector))
    second_norm = math.sqrt(sum(value * value for value in second_vector))
    if first_norm == 0.0 or second_norm == 0.0:
        return 0.0
    dot = sum(
        first_value * second_value
        for first_value, second_value in zip(first_vector, second_vector, strict=True)
    )
    cosine = max(-1.0, min(1.0, dot / (first_norm * second_norm)))
    return math.acos(cosine)


def _circular_features(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    if len(xs) < 2:
        return (0.0, 0.0, 0.0)

    center_x = fmean(xs)
    center_y = fmean(ys)
    radii = [math.hypot(x - center_x, y - center_y) for x, y in zip(xs, ys, strict=True)]
    mean_radius = fmean(radii)
    radius_cv = pstdev(radii) / mean_radius if mean_radius > 0.0 and len(radii) > 1 else 0.0
    angles = [math.atan2(y - center_y, x - center_x) for x, y in zip(xs, ys, strict=True)]
    return (mean_radius, radius_cv, _unwrapped_angle_span(angles))


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


def _range(values: list[float]) -> float:
    return max(values) - min(values) if values else 0.0


def _displacement(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    return math.hypot(xs[-1] - xs[0], ys[-1] - ys[0])


def _path_length(xs: list[float], ys: list[float]) -> float:
    return sum(
        math.hypot(second_x - first_x, second_y - first_y)
        for first_x, first_y, second_x, second_y in zip(xs, ys, xs[1:], ys[1:], strict=False)
    )


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


def _early_late_medians(values: list[float]) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    window_size = max(1, len(values) // 4)
    return (median(values[:window_size]), median(values[-window_size:]))


def _half_medians(values: list[float]) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    middle = max(1, len(values) // 2)
    return (median(values[:middle]), median(values[middle:]) if values[middle:] else values[-1])


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-9:
        return 0.0
    return numerator / denominator
