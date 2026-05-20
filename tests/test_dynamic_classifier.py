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


def test_dynamic_classifier_returns_unknown_for_short_trajectory() -> None:
    buffer = TrajectoryBuffer(max_size=30)
    config = DynamicClassifierConfig(buffer_size=30)
    classifier = DynamicGestureClassifier(config)

    prediction = classifier.classify(buffer)

    assert prediction.gesture_id == GestureID.UNKNOWN
