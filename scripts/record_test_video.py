"""Record short gesture clips for the local control dataset."""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""

    parser = argparse.ArgumentParser(description="Record a short local gesture sample.")
    parser.add_argument(
        "--gesture",
        required=True,
        help="Gesture label, for example OPEN_PALM, INDEX_LEFT, CIRCLE, or PULL_TOWARD.",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--seconds", type=float, default=3.0, help="Recording duration.")
    parser.add_argument("--fps", type=float, default=20.0, help="Output video FPS.")
    parser.add_argument("--width", type=int, default=640, help="Output frame width.")
    parser.add_argument("--height", type=int, default=480, help="Output frame height.")
    parser.add_argument(
        "--output-dir",
        default="data/external/own_control",
        help="Root directory for recorded samples.",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Record without an OpenCV preview window.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Record a gesture video sample."""

    args = build_parser().parse_args(argv)
    if args.seconds <= 0:
        raise ValueError("--seconds must be positive")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")

    cv2 = importlib.import_module("cv2")
    output_path = _build_output_path(Path(args.output_dir), args.gesture)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open camera: {args.camera}")

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    writer = _build_writer(cv2, output_path, args.fps, args.width, args.height)

    started_at = perf_counter()
    frame_count = 0
    try:
        while perf_counter() - started_at < args.seconds:
            success, frame = capture.read()
            if not success:
                break
            frame = cv2.resize(frame, (args.width, args.height))
            writer.write(frame)
            frame_count += 1
            if not args.no_preview:
                cv2.imshow("Recording gesture sample", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        writer.release()
        capture.release()
        if not args.no_preview:
            cv2.destroyAllWindows()

    print(f"Wrote video: {output_path}")
    print(f"Frames: {frame_count}")
    return 0


def _build_output_path(output_dir: Path, gesture: str) -> Path:
    gesture_dir = gesture.strip().lower().replace(" ", "_").replace("-", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / gesture_dir / f"{gesture_dir}_{timestamp}.mp4"


def _build_writer(
    cv2: Any,
    output_path: Path,
    fps: float,
    width: int,
    height: int,
) -> Any:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {output_path}")
    return writer


if __name__ == "__main__":
    raise SystemExit(main())
