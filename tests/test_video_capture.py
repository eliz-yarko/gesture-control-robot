from __future__ import annotations

from src.capture.video_capture import VideoCapture
from src.config import VideoConfig


def test_video_capture_reads_and_preprocesses_frame() -> None:
    fake_cv2 = _FakeCv2(frames=[[[("b", "g", "r"), ("x", "y", "z")]]])
    capture = VideoCapture(
        VideoConfig(frame_width=2, frame_height=1, mirror_frame=True),
        cv2_module=fake_cv2,
    )

    with capture:
        frame = capture.read()

    assert frame is not None
    assert frame.index == 0
    assert frame.bgr_frame == [[("x", "y", "z"), ("b", "g", "r")]]
    assert frame.rgb_frame == [[("z", "y", "x"), ("r", "g", "b")]]
    assert fake_cv2.capture.released is True


def test_video_capture_returns_none_when_source_is_exhausted() -> None:
    fake_cv2 = _FakeCv2(frames=[])
    capture = VideoCapture(cv2_module=fake_cv2)

    frame = capture.read()

    assert frame is None


class _FakeCv2:
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5
    COLOR_BGR2RGB = 6

    def __init__(self, frames: list[object]) -> None:
        self.capture = _FakeCapture(frames)

    def VideoCapture(self, source: int | str) -> _FakeCapture:  # noqa: N802
        self.capture.source = source
        return self.capture

    def resize(self, frame: object, size: tuple[int, int]) -> object:
        return frame

    def flip(
        self,
        frame: list[list[tuple[str, str, str]]],
        mode: int,
    ) -> list[list[tuple[str, str, str]]]:
        return [list(reversed(row)) for row in frame]

    def cvtColor(
        self,
        frame: list[list[tuple[str, str, str]]],
        code: int,
    ) -> list[list[tuple[str, str, str]]]:
        return [[tuple(reversed(pixel)) for pixel in row] for row in frame]


class _FakeCapture:
    def __init__(self, frames: list[object]) -> None:
        self.frames = frames
        self.source: int | str | None = None
        self.released = False
        self.properties: dict[int, int] = {}

    def isOpened(self) -> bool:  # noqa: N802
        return True

    def set(self, key: int, value: int) -> None:
        self.properties[key] = value

    def read(self) -> tuple[bool, object | None]:
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self) -> None:
        self.released = True
