from __future__ import annotations

import math

from src.config import DynamicClassifierConfig
from src.domain import GestureID
from src.recognition.dynamic_classifier import DynamicGestureClassifier
from src.recognition.trajectory_buffer import TrajectoryBuffer, TrajectoryPoint


def test_dynamic_classifier_detects_wave_lr() -> None:
    buffer = TrajectoryBuffer(max_size=30)
    xs = [0.3, 0.38, 0.48, 0.58, 0.5, 0.38, 0.28, 0.4, 0.54, 0.62] * 3
    for index, x in enumerate(xs):
        buffer.add_point(
            TrajectoryPoint(
                palm_center=(x, 0.4, 0.0),
                index_tip=(x, 0.2, 0.0),
                hand_size=0.25,
                timestamp=float(index),
            )
        )
    classifier = DynamicGestureClassifier()

    prediction = classifier.classify(buffer)

    assert prediction.gesture_id == GestureID.WAVE_LR
    assert prediction.metadata["direction_changes"] >= 2


def test_dynamic_classifier_detects_wave_with_camera_drift() -> None:
    buffer = TrajectoryBuffer(max_size=30)
    xs = [0.32, 0.46, 0.61, 0.52, 0.37, 0.27, 0.35, 0.52, 0.65, 0.55, 0.41, 0.29]
    for index, x in enumerate(xs):
        y = 0.34 + index * 0.012 + (0.006 if index % 2 == 0 else -0.006)
        buffer.add_point(
            TrajectoryPoint(
                palm_center=(x, y, 0.0),
                index_tip=(x, y - 0.2, 0.0),
                hand_size=0.25,
                timestamp=float(index),
            )
        )
    classifier = DynamicGestureClassifier()

    prediction = classifier.classify(buffer)

    assert prediction.gesture_id == GestureID.WAVE_LR
    assert (
        prediction.metadata["vertical_residual_range"]
        <= DynamicClassifierConfig().max_vertical_drift
    )


def test_dynamic_classifier_detects_recent_wave_before_full_buffer() -> None:
    buffer = TrajectoryBuffer(max_size=30)
    xs = [0.3, 0.42, 0.58, 0.62, 0.48, 0.34, 0.28, 0.4, 0.55, 0.62]
    for index, x in enumerate(xs):
        buffer.add_point(
            TrajectoryPoint(
                palm_center=(x, 0.4, 0.0),
                index_tip=(x, 0.2, 0.0),
                hand_size=0.25,
                timestamp=float(index),
            )
        )
    classifier = DynamicGestureClassifier()

    prediction = classifier.classify(buffer)

    assert prediction.gesture_id == GestureID.WAVE_LR
    assert prediction.metadata["window_points"] == len(xs)


def test_dynamic_classifier_treats_wrist_arc_as_wave_not_circle() -> None:
    buffer = TrajectoryBuffer(max_size=30)
    xs = [0.42, 0.55, 0.65, 0.56, 0.43, 0.35, 0.44, 0.56, 0.66, 0.54, 0.42, 0.34]
    ys = [0.36, 0.31, 0.36, 0.42, 0.46, 0.42, 0.36, 0.31, 0.35, 0.42, 0.46, 0.42]
    for index, (x, y) in enumerate(zip(xs, ys, strict=True)):
        buffer.add_point(
            TrajectoryPoint(
                palm_center=(0.5, 0.5, 0.0),
                index_tip=(x, y, 0.0),
                hand_size=0.25,
                timestamp=float(index),
            )
        )
    classifier = DynamicGestureClassifier()

    prediction = classifier.classify(buffer)

    assert prediction.gesture_id == GestureID.WAVE_LR
    assert prediction.metadata["tracked_point"] == "index"


def test_dynamic_classifier_detects_circle() -> None:
    buffer = TrajectoryBuffer(max_size=30)
    for index in range(30):
        angle = 2 * math.pi * index / 29
        buffer.add_point(
            TrajectoryPoint(
                palm_center=(0.5, 0.5, 0.0),
                index_tip=(0.5 + 0.12 * math.cos(angle), 0.5 + 0.12 * math.sin(angle), 0.0),
                hand_size=0.25,
                timestamp=float(index),
            )
        )
    classifier = DynamicGestureClassifier()

    prediction = classifier.classify(buffer)

    assert prediction.gesture_id == GestureID.CIRCLE
    assert prediction.metadata["angle_span"] >= 5.0


def test_dynamic_classifier_detects_uneven_partial_circle() -> None:
    buffer = TrajectoryBuffer(max_size=30)
    for index in range(18):
        angle = 1.65 * math.pi * index / 17
        radius = 0.11 + 0.04 * math.sin(angle * 2)
        buffer.add_point(
            TrajectoryPoint(
                palm_center=(0.5, 0.5, 0.0),
                index_tip=(0.5 + radius * math.cos(angle), 0.5 + radius * math.sin(angle), 0.0),
                hand_size=0.25,
                timestamp=float(index),
            )
        )
    classifier = DynamicGestureClassifier()

    prediction = classifier.classify(buffer)

    assert prediction.gesture_id == GestureID.CIRCLE
    assert prediction.metadata["axis_balance"] >= 0.35


def test_dynamic_classifier_detects_pull_toward() -> None:
    buffer = TrajectoryBuffer(max_size=30)
    for index in range(30):
        size = 0.2 + index * 0.003
        buffer.add_point(
            TrajectoryPoint(
                palm_center=(0.5, 0.5, 0.0),
                index_tip=(0.5, 0.3, 0.0),
                hand_size=size,
                timestamp=float(index),
            )
        )
    classifier = DynamicGestureClassifier()

    prediction = classifier.classify(buffer)

    assert prediction.gesture_id == GestureID.PULL_TOWARD
    assert prediction.metadata["scale_growth"] >= 0.2


def test_dynamic_classifier_detects_noisy_pull_toward() -> None:
    buffer = TrajectoryBuffer(max_size=30)
    sizes = [0.22, 0.226, 0.223, 0.236, 0.233, 0.248, 0.253, 0.249, 0.266, 0.272, 0.269, 0.286]
    for index, size in enumerate(sizes):
        x = 0.5 + (0.006 if index % 2 == 0 else -0.004)
        y = 0.5 + (0.004 if index % 3 == 0 else -0.003)
        buffer.add_point(
            TrajectoryPoint(
                palm_center=(x, y, 0.0),
                index_tip=(x, y - 0.2, 0.0),
                hand_size=size,
                timestamp=float(index),
            )
        )
    classifier = DynamicGestureClassifier()

    prediction = classifier.classify(buffer)

    assert prediction.gesture_id == GestureID.PULL_TOWARD
    assert prediction.metadata["positive_step_ratio"] >= 0.5


def test_dynamic_classifier_detects_subtle_pull_toward() -> None:
    buffer = TrajectoryBuffer(max_size=30)
    sizes = [0.24, 0.242, 0.245, 0.249, 0.252, 0.258, 0.262, 0.267, 0.272]
    for index, size in enumerate(sizes):
        buffer.add_point(
            TrajectoryPoint(
                palm_center=(0.5, 0.5, 0.0),
                index_tip=(0.5, 0.3, 0.0),
                hand_size=size,
                timestamp=float(index),
            )
        )
    classifier = DynamicGestureClassifier()

    prediction = classifier.classify(buffer)

    assert prediction.gesture_id == GestureID.PULL_TOWARD
    assert (
        prediction.metadata["sustained_growth"] >= DynamicClassifierConfig().min_pull_scale_growth
    )


def test_dynamic_classifier_returns_unknown_for_short_trajectory() -> None:
    buffer = TrajectoryBuffer(max_size=30)
    config = DynamicClassifierConfig(buffer_size=30)
    classifier = DynamicGestureClassifier(config)

    prediction = classifier.classify(buffer)

    assert prediction.gesture_id == GestureID.UNKNOWN


def test_dynamic_classifier_ignores_stationary_jitter() -> None:
    buffer = TrajectoryBuffer(max_size=30)
    for index in range(20):
        jitter_x = 0.006 if index % 2 == 0 else -0.005
        jitter_y = 0.004 if index % 3 == 0 else -0.003
        size_jitter = 0.004 if index % 4 in (0, 1) else -0.003
        buffer.add_point(
            TrajectoryPoint(
                palm_center=(0.5 + jitter_x, 0.5 + jitter_y, 0.0),
                index_tip=(0.5 + jitter_x, 0.3 + jitter_y, 0.0),
                hand_size=0.25 + size_jitter,
                timestamp=float(index),
            )
        )
    classifier = DynamicGestureClassifier()

    prediction = classifier.classify(buffer)

    assert prediction.gesture_id == GestureID.UNKNOWN
