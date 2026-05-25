"""Fixed-size trajectory buffer for dynamic gesture recognition."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from time import monotonic

from src.utils.geometry import Landmark, LandmarkSequence, hand_scale, palm_center, to_landmarks


@dataclass(frozen=True)
class TrajectoryPoint:
    """Single frame summary used by dynamic gesture classifiers."""

    palm_center: Landmark
    index_tip: Landmark
    hand_size: float
    timestamp: float


class TrajectoryBuffer:
    """Store the latest N frame summaries for dynamic gestures."""

    def __init__(self, max_size: int) -> None:
        """Initialize the buffer.

        Args:
            max_size: Maximum number of frames retained in the buffer.

        Raises:
            ValueError: If ``max_size`` is not positive.
        """

        if max_size <= 0:
            raise ValueError("Trajectory buffer size must be positive.")
        self._points: deque[TrajectoryPoint] = deque(maxlen=max_size)

    @property
    def max_size(self) -> int:
        """Return the configured buffer size."""

        return self._points.maxlen or 0

    @property
    def is_full(self) -> bool:
        """Return whether the buffer contains ``max_size`` points."""

        return len(self._points) == self.max_size

    def __len__(self) -> int:
        """Return the number of stored trajectory points."""

        return len(self._points)

    def __iter__(self) -> Iterable[TrajectoryPoint]:
        """Iterate over buffered points from oldest to newest."""

        return iter(self._points)

    def clear(self) -> None:
        """Remove all buffered points."""

        self._points.clear()

    def add_landmarks(
        self,
        raw_landmarks: LandmarkSequence,
        timestamp: float | None = None,
    ) -> None:
        """Extract trajectory features from landmarks and append them."""

        landmarks = to_landmarks(raw_landmarks)
        self.add_point(
            TrajectoryPoint(
                palm_center=palm_center(landmarks),
                index_tip=landmarks[8],
                hand_size=hand_scale(landmarks),
                timestamp=timestamp if timestamp is not None else monotonic(),
            )
        )

    def add_point(self, point: TrajectoryPoint) -> None:
        """Append a precomputed trajectory point."""

        self._points.append(point)

    def points(self) -> list[TrajectoryPoint]:
        """Return buffered points as a list."""

        return list(self._points)
