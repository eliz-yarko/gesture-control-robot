"""Collect a user calibration profile from a camera stream."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration import CalibrationProfileStore, CalibrationSession
from src.capture import VideoCapture
from src.config import AppConfig, CalibrationConfig, VideoConfig
from src.domain import GestureID
from src.recognition import HandDetector


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""

    parser = argparse.ArgumentParser(description="Collect a personalized gesture profile.")
    parser.add_argument("--user-id", required=True, help="Stable user profile identifier.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--samples", type=int, default=5, help="Samples per gesture.")
    parser.add_argument(
        "--gestures",
        nargs="+",
        default=["OPEN_PALM", "FIST", "THUMB_UP", "THUMB_DOWN", "INDEX_LEFT", "INDEX_RIGHT"],
        help="Gesture names to calibrate.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run interactive calibration collection."""

    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    gestures = tuple(GestureID[name] for name in args.gestures)
    config = AppConfig(
        video=VideoConfig(camera_index=args.camera),
        calibration=CalibrationConfig(samples_per_gesture=args.samples),
    )
    session = CalibrationSession(config.calibration, gestures=gestures)

    with VideoCapture(config.video) as capture, HandDetector(config.hand_detection) as detector:
        for gesture_id in gestures:
            input(
                f"Show {gesture_id.name}, keep hand visible, then press Enter "
                f"to collect {args.samples} samples."
            )
            while not session.is_gesture_ready(gesture_id):
                frame = capture.read()
                if frame is None:
                    raise RuntimeError("Camera stream ended during calibration.")
                detections = detector.detect(frame.rgb_frame)
                if not detections:
                    continue
                detection = max(detections, key=lambda item: item.score)
                count = session.add_sample(gesture_id, detection.landmarks)
                logging.info("Collected %s/%s for %s", count, args.samples, gesture_id.name)

    profile = session.build_profile(args.user_id)
    path = CalibrationProfileStore(config.calibration).save(profile)
    logging.info("Saved calibration profile: %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
