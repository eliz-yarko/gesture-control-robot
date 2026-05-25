"""Geometry helpers for MediaPipe hand landmarks."""

from __future__ import annotations

import math
from collections.abc import Sequence

Landmark = tuple[float, float, float]
LandmarkSequence = Sequence[Sequence[float]]


def to_landmarks(raw_landmarks: LandmarkSequence) -> list[Landmark]:
    """Validate and normalize raw landmarks to a list of 3D points.

    Args:
        raw_landmarks: Sequence with 21 points. Each point must contain x, y, z.

    Returns:
        A list of 21 ``(x, y, z)`` tuples.

    Raises:
        ValueError: If the landmark sequence does not match MediaPipe Hands shape.
    """

    if len(raw_landmarks) != 21:
        raise ValueError("Expected 21 hand landmarks.")

    landmarks: list[Landmark] = []
    for point in raw_landmarks:
        if len(point) != 3:
            raise ValueError("Each hand landmark must contain x, y, and z coordinates.")
        landmarks.append((float(point[0]), float(point[1]), float(point[2])))
    return landmarks


def vector(start: Landmark, end: Landmark) -> Landmark:
    """Return a vector from ``start`` to ``end``."""

    return (end[0] - start[0], end[1] - start[1], end[2] - start[2])


def distance(first: Landmark, second: Landmark) -> float:
    """Return Euclidean distance between two 3D landmarks."""

    dx = first[0] - second[0]
    dy = first[1] - second[1]
    dz = first[2] - second[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def vector_norm(value: Landmark) -> float:
    """Return Euclidean vector norm."""

    return math.sqrt(value[0] * value[0] + value[1] * value[1] + value[2] * value[2])


def angle_between(first: Landmark, second: Landmark) -> float:
    """Return the angle between two vectors in radians."""

    first_norm = vector_norm(first)
    second_norm = vector_norm(second)
    if first_norm == 0.0 or second_norm == 0.0:
        raise ValueError("Cannot calculate an angle for a zero-length vector.")

    cosine = (first[0] * second[0] + first[1] * second[1] + first[2] * second[2]) / (
        first_norm * second_norm
    )
    clamped = max(-1.0, min(1.0, cosine))
    return math.acos(clamped)


def palm_center(landmarks: Sequence[Landmark]) -> Landmark:
    """Return an approximate palm center from wrist and metacarpophalangeal joints."""

    indices = (0, 5, 9, 13, 17)
    return _centroid([landmarks[index] for index in indices])


def hand_scale(landmarks: Sequence[Landmark]) -> float:
    """Return a scale estimate for a hand pose.

    The distance between the wrist and middle finger MCP is stable enough for
    normalized rule thresholds and works for open and closed hands.
    """

    scale = distance(landmarks[0], landmarks[9])
    if scale == 0.0:
        raise ValueError("Hand scale cannot be zero.")
    return scale


def normalize_landmarks(landmarks: Sequence[Landmark]) -> list[Landmark]:
    """Normalize landmarks by palm center and hand scale."""

    center = palm_center(landmarks)
    scale = hand_scale(landmarks)
    return [
        (
            (point[0] - center[0]) / scale,
            (point[1] - center[1]) / scale,
            (point[2] - center[2]) / scale,
        )
        for point in landmarks
    ]


def is_finger_extended(
    landmarks: Sequence[Landmark],
    tip_index: int,
    pip_index: int,
    ratio: float,
) -> bool:
    """Return whether a non-thumb finger is extended."""

    wrist = landmarks[0]
    tip_distance = distance(wrist, landmarks[tip_index])
    pip_distance = distance(wrist, landmarks[pip_index])
    return tip_distance > pip_distance * ratio


def axis_direction(
    landmarks: Sequence[Landmark],
    base_index: int,
    tip_index: int,
    margin: float,
) -> str:
    """Return the dominant 2D direction of a finger vector."""

    dx, dy, _ = vector(landmarks[base_index], landmarks[tip_index])
    if abs(dx) > abs(dy) + margin:
        return "right" if dx > 0 else "left"
    if abs(dy) > abs(dx) + margin:
        return "down" if dy > 0 else "up"
    return "diagonal"


def _centroid(points: Sequence[Landmark]) -> Landmark:
    count = len(points)
    if count == 0:
        raise ValueError("Cannot calculate a centroid for an empty sequence.")
    return (
        sum(point[0] for point in points) / count,
        sum(point[1] for point in points) / count,
        sum(point[2] for point in points) / count,
    )
