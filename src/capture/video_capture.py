"""Video capture and frame preprocessing utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic
from typing import Any

from src.config import VideoConfig


@dataclass(frozen=True)
class CapturedFrame:
    """Frame returned by the video capture layer.

    Args:
        bgr_frame: Frame in OpenCV BGR color space, used for visualization.
        rgb_frame: Frame in RGB color space, used by MediaPipe Hands.
        index: Zero-based frame index within the current run.
        timestamp: Monotonic timestamp when the frame was read.
    """

    bgr_frame: Any
    rgb_frame: Any
    index: int
    timestamp: float = field(default_factory=monotonic)


class VideoCapture:
    """OpenCV-backed source for camera or video-file frames."""

    def __init__(
        self,
        config: VideoConfig | None = None,
        video_path: str | None = None,
        cv2_module: Any | None = None,
    ) -> None:
        """Initialize capture settings without opening the source."""

        self._config = config or VideoConfig()
        self._video_path = video_path
        self._cv2 = cv2_module
        self._capture: Any | None = None
        self._frame_index = 0

    @property
    def is_opened(self) -> bool:
        """Return whether the underlying source is open."""

        return self._capture is not None and bool(self._capture.isOpened())

    def open(self) -> None:
        """Open configured camera or video file.

        Raises:
            RuntimeError: If OpenCV is not installed or the source cannot be opened.
        """

        cv2 = self._require_cv2()
        source: int | str = (
            self._video_path if self._video_path is not None else self._config.camera_index
        )
        capture = cv2.VideoCapture(source)

        if self._video_path is None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.frame_width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.frame_height)
            capture.set(cv2.CAP_PROP_FPS, self._config.target_fps)

        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video source: {source!r}")

        self._capture = capture
        self._frame_index = 0

    def read(self) -> CapturedFrame | None:
        """Read and preprocess the next frame.

        Returns:
            A captured frame object, or ``None`` when a video file has ended.
        """

        if self._capture is None:
            self.open()

        if self._capture is None:
            raise RuntimeError("Video source is not open.")

        success, frame = self._capture.read()
        if not success:
            return None

        bgr_frame, rgb_frame = self._preprocess(frame)
        captured = CapturedFrame(
            bgr_frame=bgr_frame,
            rgb_frame=rgb_frame,
            index=self._frame_index,
        )
        self._frame_index += 1
        return captured

    def release(self) -> None:
        """Release the underlying OpenCV capture object."""

        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "VideoCapture":
        """Open the capture source when entering a context manager."""

        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        """Release the capture source when leaving a context manager."""

        self.release()

    def _preprocess(self, frame: Any) -> tuple[Any, Any]:
        cv2 = self._require_cv2()
        resized = cv2.resize(frame, (self._config.frame_width, self._config.frame_height))
        bgr_frame = cv2.flip(resized, 1) if self._config.mirror_frame else resized
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        return bgr_frame, rgb_frame

    def _require_cv2(self) -> Any:
        if self._cv2 is not None:
            return self._cv2

        try:
            import cv2  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("OpenCV is required for VideoCapture.") from exc

        self._cv2 = cv2
        return cv2
