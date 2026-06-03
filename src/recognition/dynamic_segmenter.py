"""Motion spotting for continuous dynamic hand gesture recognition."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from src.config import DynamicClassifierConfig
from src.recognition.trajectory_buffer import TrajectoryBuffer, TrajectoryPoint


@dataclass(frozen=True)
class DynamicSegmentUpdate:
    """Result of ingesting one point into the motion segmenter."""

    state: str
    motion_energy: float
    active_points: int
    segment: TrajectoryBuffer | None = None


class DynamicGestureSegmenter:
    """Split a continuous landmark stream into dynamic gesture candidates."""

    def __init__(self, config: DynamicClassifierConfig) -> None:
        """Initialize segmenter thresholds and empty state."""

        self._config = config
        self._previous: TrajectoryPoint | None = None
        self._pre_motion_points: deque[TrajectoryPoint] = deque(
            maxlen=max(2, config.motion_start_frames + 1)
        )
        self._active_points: list[TrajectoryPoint] = []
        self._moving_frames = 0
        self._still_frames = 0
        self._active = False

    @property
    def is_active(self) -> bool:
        """Return whether a dynamic segment is currently being collected."""

        return self._active

    def update(self, point: TrajectoryPoint) -> DynamicSegmentUpdate:
        """Add a tracked hand point and return the current motion state."""

        instant_energy = _motion_energy(self._previous, point)
        self._previous = point
        self._pre_motion_points.append(point)

        if self._active:
            self._active_points.append(point)
            energy = max(
                instant_energy,
                _window_motion_energy(
                    self._active_points[-(self._config.motion_end_frames + 1) :]
                ),
            )
            if energy <= self._config.motion_end_threshold:
                self._still_frames += 1
            else:
                self._still_frames = 0

            if self._should_finish_segment():
                return self._finish_segment("segment_ready", energy)
            return DynamicSegmentUpdate(
                state="recording_dynamic",
                motion_energy=energy,
                active_points=len(self._active_points),
            )

        energy = max(instant_energy, _window_motion_energy(list(self._pre_motion_points)))
        if energy >= self._config.motion_start_threshold:
            self._moving_frames += 1
        else:
            self._moving_frames = 0

        if self._moving_frames >= self._config.motion_start_frames:
            self._active = True
            self._still_frames = 0
            self._active_points = list(self._pre_motion_points)
            return DynamicSegmentUpdate(
                state="motion_started",
                motion_energy=energy,
                active_points=len(self._active_points),
            )

        return DynamicSegmentUpdate(
            state="stable_static",
            motion_energy=energy,
            active_points=0,
        )

    def missing(self) -> DynamicSegmentUpdate:
        """Handle a missing hand detection and flush an active segment if useful."""

        self._previous = None
        self._moving_frames = 0
        if not self._active:
            return DynamicSegmentUpdate(
                state="missing_hand",
                motion_energy=0.0,
                active_points=0,
            )
        if len(self._active_points) >= self._config.min_dynamic_segment_points:
            return self._finish_segment("segment_ready", 0.0)
        self.reset()
        return DynamicSegmentUpdate(
            state="missing_hand",
            motion_energy=0.0,
            active_points=0,
        )

    def reset(self) -> None:
        """Clear all motion spotting state."""

        self._previous = None
        self._pre_motion_points.clear()
        self._active_points = []
        self._moving_frames = 0
        self._still_frames = 0
        self._active = False

    def _should_finish_segment(self) -> bool:
        if len(self._active_points) >= self._config.max_dynamic_segment_points:
            return True
        return (
            len(self._active_points) >= self._config.min_dynamic_segment_points
            and self._still_frames >= self._config.motion_end_frames
        )

    def _finish_segment(self, state: str, energy: float) -> DynamicSegmentUpdate:
        segment_points = _without_trailing_still_points(
            self._active_points,
            self._still_frames,
            min_points=self._config.min_dynamic_segment_points,
        )
        segment = TrajectoryBuffer(
            max(self._config.max_dynamic_segment_points, len(segment_points))
        )
        for point in segment_points:
            segment.add_point(point)
        active_points = len(segment_points)
        self.reset()
        return DynamicSegmentUpdate(
            state=state,
            motion_energy=energy,
            active_points=active_points,
            segment=segment,
        )


def _motion_energy(previous: TrajectoryPoint | None, current: TrajectoryPoint) -> float:
    if previous is None:
        return 0.0
    palm_motion = _distance_2d(previous.palm_center, current.palm_center)
    index_motion = _distance_2d(previous.index_tip, current.index_tip)
    scale_motion = abs(current.hand_size - previous.hand_size) / max(previous.hand_size, 1e-9)
    return max(palm_motion, index_motion, scale_motion * 0.35)


def _window_motion_energy(points: list[TrajectoryPoint]) -> float:
    if len(points) < 2:
        return 0.0
    first = points[0]
    last = points[-1]
    palm_motion = _distance_2d(first.palm_center, last.palm_center)
    index_motion = _distance_2d(first.index_tip, last.index_tip)
    scale_motion = abs(last.hand_size - first.hand_size) / max(first.hand_size, 1e-9)
    return max(palm_motion, index_motion, scale_motion * 0.7)


def _distance_2d(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> float:
    return math.hypot(second[0] - first[0], second[1] - first[1])


def _without_trailing_still_points(
    points: list[TrajectoryPoint],
    still_frames: int,
    *,
    min_points: int,
) -> list[TrajectoryPoint]:
    if still_frames <= 0 or len(points) - still_frames < min_points:
        return list(points)
    return list(points[:-still_frames])
