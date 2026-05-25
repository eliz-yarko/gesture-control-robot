from __future__ import annotations

import math

import pytest
from src.utils.geometry import angle_between, distance, normalize_landmarks, to_landmarks


def test_distance_returns_euclidean_distance() -> None:
    assert distance((0.0, 0.0, 0.0), (3.0, 4.0, 0.0)) == 5.0


def test_angle_between_perpendicular_vectors() -> None:
    angle = angle_between((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))

    assert angle == pytest.approx(math.pi / 2)


def test_normalize_landmarks_keeps_mediapipe_shape() -> None:
    landmarks = [[0.0, 0.0, 0.0] for _ in range(21)]
    landmarks[0] = [0.0, 0.0, 0.0]
    landmarks[5] = [-0.1, -0.2, 0.0]
    landmarks[9] = [0.0, -0.3, 0.0]
    landmarks[13] = [0.1, -0.2, 0.0]
    landmarks[17] = [0.2, -0.1, 0.0]

    normalized = normalize_landmarks(to_landmarks(landmarks))

    assert len(normalized) == 21
    assert normalized[9][1] < 0


def test_to_landmarks_rejects_points_without_z_coordinate() -> None:
    invalid_landmarks = [[0.0, 0.0, 0.0] for _ in range(20)]
    invalid_landmarks.append([0.0, 0.0])

    with pytest.raises(ValueError, match="x, y, and z"):
        to_landmarks(invalid_landmarks)
