from __future__ import annotations

from src.config import DynamicClassifierConfig
from src.recognition.dynamic_segmenter import DynamicGestureSegmenter
from src.recognition.trajectory_buffer import TrajectoryPoint


def test_dynamic_segmenter_detects_gradual_scale_motion() -> None:
    config = DynamicClassifierConfig()
    segmenter = DynamicGestureSegmenter(config)
    sizes = [
        0.200,
        0.203,
        0.207,
        0.212,
        0.218,
        0.225,
        0.225,
        0.225,
        0.225,
        0.225,
        0.225,
    ]

    updates = [
        segmenter.update(_point(size=size, timestamp=float(index)))
        for index, size in enumerate(sizes)
    ]

    assert "motion_started" in [update.state for update in updates]
    assert updates[-1].state == "segment_ready"
    assert updates[-1].segment is not None
    assert len(updates[-1].segment) >= config.min_dynamic_segment_points


def test_dynamic_segmenter_ignores_stationary_scale_jitter() -> None:
    segmenter = DynamicGestureSegmenter(DynamicClassifierConfig())

    updates = [
        segmenter.update(_point(size=size, timestamp=float(index)))
        for index, size in enumerate((0.250, 0.253, 0.248, 0.252, 0.249, 0.251))
    ]

    assert {update.state for update in updates} == {"stable_static"}


def _point(
    *,
    size: float,
    timestamp: float,
) -> TrajectoryPoint:
    return TrajectoryPoint(
        palm_center=(0.5, 0.5, 0.0),
        index_tip=(0.5, 0.3, 0.0),
        hand_size=size,
        timestamp=timestamp,
    )
