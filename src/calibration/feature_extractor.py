"""Feature extraction and vector math for adaptive calibration."""

from __future__ import annotations

import math
from collections.abc import Sequence

from src.utils.geometry import LandmarkSequence, normalize_landmarks, to_landmarks

FeatureVector = tuple[float, ...]


class LandmarkFeatureExtractor:
    """Extract a normalized feature vector from MediaPipe hand landmarks."""

    def extract(self, raw_landmarks: LandmarkSequence) -> FeatureVector:
        """Return a translation- and scale-normalized landmark vector.

        Args:
            raw_landmarks: Sequence of 21 MediaPipe hand landmarks.

        Returns:
            Flattened vector with normalized x, y, z coordinates.
        """

        landmarks = to_landmarks(raw_landmarks)
        normalized = normalize_landmarks(landmarks)
        return tuple(coordinate for point in normalized for coordinate in point)


def euclidean_distance(first: FeatureVector, second: FeatureVector) -> float:
    """Return L2 distance between feature vectors."""

    _validate_same_size([first, second])
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(first, second, strict=True)))


def mean_vector(samples: Sequence[FeatureVector]) -> FeatureVector:
    """Return coordinate-wise mean vector."""

    _validate_samples(samples)
    size = len(samples[0])
    return tuple(sum(sample[index] for sample in samples) / len(samples) for index in range(size))


def std_vector(samples: Sequence[FeatureVector], center: FeatureVector) -> FeatureVector:
    """Return coordinate-wise population standard deviation vector."""

    _validate_samples(samples)
    _validate_same_size([*samples, center])
    return tuple(
        math.sqrt(sum((sample[index] - center[index]) ** 2 for sample in samples) / len(samples))
        for index in range(len(center))
    )


def distance_radius(
    samples: Sequence[FeatureVector],
    center: FeatureVector,
    sigma_multiplier: float,
) -> float:
    """Return adaptive radius from sample distances to centroid."""

    _validate_samples(samples)
    distances = [euclidean_distance(sample, center) for sample in samples]
    mean_distance = sum(distances) / len(distances)
    variance = sum((distance - mean_distance) ** 2 for distance in distances) / len(distances)
    radius = mean_distance + sigma_multiplier * math.sqrt(variance)
    return max(radius, 1e-9)


def _validate_samples(samples: Sequence[FeatureVector]) -> None:
    if not samples:
        raise ValueError("At least one feature vector is required.")
    _validate_same_size(samples)


def _validate_same_size(samples: Sequence[FeatureVector]) -> None:
    if not samples:
        return
    size = len(samples[0])
    if size == 0:
        raise ValueError("Feature vectors must not be empty.")
    if any(len(sample) != size for sample in samples):
        raise ValueError("All feature vectors must have the same size.")
