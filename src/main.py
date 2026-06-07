"""Command-line entry point for the gesture control subsystem."""

from __future__ import annotations

import argparse
import importlib
import logging
from collections.abc import Sequence
from pathlib import Path

from src.calibration import AdaptiveCalibrator, CalibrationProfileStore
from src.capture.video_capture import VideoCapture
from src.config import AppConfig, VideoConfig
from src.pipeline import GestureControlPipeline, PipelineResult
from src.recognition import (
    DynamicGestureClassifier,
    FallbackDynamicGestureClassifier,
    FallbackStaticGestureClassifier,
    SklearnDynamicGestureClassifier,
    SklearnStaticGestureClassifier,
    StaticGestureClassifier,
)
from src.transmission.base_sender import CommandSender
from src.transmission.mock_sender import MockCommandSender
from src.transmission.serial_sender import SerialCommandSender

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DYNAMIC_MODEL_PATH = PROJECT_ROOT / "models" / "dynamic_gesture_classifier.joblib"


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""

    parser = argparse.ArgumentParser(description="Run the gesture control subsystem.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index for live capture.")
    parser.add_argument("--video", type=str, default=None, help="Path to a video file.")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print recognized gestures and commands.",
    )
    parser.add_argument("--visualize", action="store_true", help="Show the OpenCV preview window.")
    parser.add_argument(
        "--calibration-profile",
        type=str,
        default=None,
        help="Path to a user calibration profile JSON file.",
    )
    parser.add_argument(
        "--static-model",
        type=str,
        default=None,
        help="Optional joblib model for static gesture classification.",
    )
    parser.add_argument(
        "--dynamic-model",
        type=str,
        default=None,
        help=(
            "Optional joblib model for dynamic gesture classification. "
            "Defaults to models/dynamic_gesture_classifier.joblib when present."
        ),
    )
    parser.add_argument(
        "--no-dynamic-model",
        action="store_true",
        help="Use only the heuristic dynamic classifier, even if a bundled model exists.",
    )
    parser.add_argument(
        "--sender",
        choices=("mock", "serial"),
        default="mock",
        help="Command transport backend.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run camera/video processing loop."""

    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = AppConfig(video=VideoConfig(camera_index=args.camera))
    serial_sender: SerialCommandSender | None = None
    sender: CommandSender
    if args.sender == "serial":
        serial_sender = SerialCommandSender(config.sender)
        serial_sender.open()
        sender = serial_sender
    else:
        sender = MockCommandSender()
    capture = VideoCapture(config.video, video_path=args.video)
    calibrator = None
    if args.calibration_profile is not None:
        profile = CalibrationProfileStore(config.calibration).load_path(args.calibration_profile)
        calibrator = AdaptiveCalibrator(profile, config.calibration)
    static_classifier = None
    dynamic_classifier = None
    if args.static_model is not None:
        static_classifier = FallbackStaticGestureClassifier(
            primary=SklearnStaticGestureClassifier.load_path(
                args.static_model,
                min_confidence=config.command_mapping.min_confidence,
            ),
            fallback=StaticGestureClassifier(config.static_classifier),
        )
    dynamic_model_path = (
        None if args.no_dynamic_model else _resolve_dynamic_model_path(args.dynamic_model)
    )
    if dynamic_model_path is not None:
        try:
            dynamic_classifier = FallbackDynamicGestureClassifier(
                primary=SklearnDynamicGestureClassifier.load_path(
                    dynamic_model_path,
                    min_confidence=config.command_mapping.min_confidence,
                    min_points=max(
                        config.dynamic_classifier.min_window_points,
                        config.dynamic_classifier.buffer_size // 3,
                    ),
                ),
                fallback=DynamicGestureClassifier(config.dynamic_classifier),
            )
        except (ImportError, OSError, ValueError) as exc:
            LOGGER.warning(
                "Cannot load dynamic model %s; using heuristics: %s", dynamic_model_path, exc
            )
    pipeline = GestureControlPipeline(
        config=config,
        static_classifier=static_classifier,
        dynamic_classifier=dynamic_classifier,
        command_sender=sender,
        calibrator=calibrator,
    )

    cv2_module = None
    if args.visualize:
        try:
            cv2_module = importlib.import_module("cv2")
        except ImportError as exc:
            raise RuntimeError("OpenCV is required for --visualize.") from exc

    try:
        with capture:
            frame_count = 0
            while args.max_frames is None or frame_count < args.max_frames:
                frame = capture.read()
                if frame is None:
                    break

                result = pipeline.process(frame)
                frame_count += 1

                if args.debug:
                    _log_debug_result(result)

                if cv2_module is not None:
                    cv2_module.imshow("Gesture Control", frame.bgr_frame)
                    if cv2_module.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        pipeline.close()
        if serial_sender is not None:
            serial_sender.close()
        if cv2_module is not None:
            cv2_module.destroyAllWindows()

    return 0


def _log_debug_result(result: PipelineResult) -> None:
    logging.info(
        "frame=%s static=%s(%.2f) dynamic=%s(%.2f) selected=%s command=%s",
        result.frame_index,
        result.static_prediction.gesture_id.name,
        result.static_prediction.confidence,
        result.dynamic_prediction.gesture_id.name,
        result.dynamic_prediction.confidence,
        result.selected_prediction.gesture_id.name,
        result.command_event.command.value if result.command_event is not None else "-",
    )


def _resolve_dynamic_model_path(model_path: str | None) -> Path | None:
    if model_path is not None:
        return Path(model_path)
    return DEFAULT_DYNAMIC_MODEL_PATH if DEFAULT_DYNAMIC_MODEL_PATH.exists() else None


if __name__ == "__main__":
    raise SystemExit(main())
