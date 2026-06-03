"""Local web dashboard for the gesture control subsystem."""

from __future__ import annotations

import argparse
import base64
import binascii
import importlib
import json
import logging
import threading
import time
import webbrowser
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import perf_counter
from typing import Any, cast
from urllib.parse import urlparse

from src.capture.video_capture import CapturedFrame, VideoCapture
from src.config import AppConfig, CommandMappingConfig, VideoConfig
from src.domain import CommandEvent, GestureID, RobotCommand
from src.interpretation.command_mapper import CommandConfirmationState
from src.pipeline import GestureControlPipeline, PipelineResult
from src.recognition import (
    DynamicGestureClassifier,
    FallbackDynamicGestureClassifier,
    FallbackStaticGestureClassifier,
    SklearnDynamicGestureClassifier,
    SklearnStaticGestureClassifier,
    StaticGestureClassifier,
    StaticPoseAnalysis,
    expected_pose_for,
)
from src.transmission.base_sender import CommandSender
from src.transmission.mock_sender import MockCommandSender
from src.transmission.serial_sender import SerialCommandSender

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DYNAMIC_MODEL_PATH = PROJECT_ROOT / "models" / "dynamic_gesture_classifier.joblib"

HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)


@dataclass(frozen=True)
class CommandLogEntry:
    """Command item displayed in the dashboard log."""

    command: str
    gesture: str
    confidence: float
    frame_index: int
    created_at: str


@dataclass(frozen=True)
class DashboardSnapshot:
    """Thread-safe status payload for the browser UI."""

    state: str
    source: str
    transport: str
    frame_index: int
    fps: float | None
    latency_ms: float | None
    hand_count: int
    static_gesture: str
    static_confidence: float
    dynamic_gesture: str
    dynamic_confidence: float
    dynamic_state: str
    selected_gesture: str
    selected_confidence: float
    last_command: str
    last_command_gesture: str
    last_command_confidence: float
    command_count: int
    command_state: CommandConfirmationState
    pose_analysis: StaticPoseAnalysis | None
    error: str
    updated_at: str
    uptime_seconds: float


class DashboardState:
    """Shared state between the capture worker and HTTP handlers."""

    def __init__(
        self,
        source: str,
        transport: str,
        command_limit: int = 40,
        command_config: CommandMappingConfig | None = None,
    ) -> None:
        """Initialize empty dashboard state."""

        self._source = source
        self._transport = transport
        self._command_config = command_config or CommandMappingConfig()
        self._started_at = perf_counter()
        self._lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        self._commands: deque[CommandLogEntry] = deque(maxlen=command_limit)
        self._snapshot = DashboardSnapshot(
            state="starting",
            source=source,
            transport=transport,
            frame_index=0,
            fps=None,
            latency_ms=None,
            hand_count=0,
            static_gesture=GestureID.UNKNOWN.name,
            static_confidence=0.0,
            dynamic_gesture=GestureID.UNKNOWN.name,
            dynamic_confidence=0.0,
            dynamic_state="idle",
            selected_gesture=GestureID.UNKNOWN.name,
            selected_confidence=0.0,
            last_command=RobotCommand.UNKNOWN.value,
            last_command_gesture=GestureID.UNKNOWN.name,
            last_command_confidence=0.0,
            command_count=0,
            command_state=CommandConfirmationState.unknown("waiting"),
            pose_analysis=None,
            error="",
            updated_at=_timestamp(),
            uptime_seconds=0.0,
        )

    def set_state(self, state: str, error: str = "") -> None:
        """Update worker state and optional error text."""

        with self._lock:
            current = self._snapshot
            self._snapshot = DashboardSnapshot(
                state=state,
                source=current.source,
                transport=current.transport,
                frame_index=current.frame_index,
                fps=current.fps,
                latency_ms=current.latency_ms,
                hand_count=current.hand_count,
                static_gesture=current.static_gesture,
                static_confidence=current.static_confidence,
                dynamic_gesture=current.dynamic_gesture,
                dynamic_confidence=current.dynamic_confidence,
                dynamic_state=current.dynamic_state,
                selected_gesture=current.selected_gesture,
                selected_confidence=current.selected_confidence,
                last_command=current.last_command,
                last_command_gesture=current.last_command_gesture,
                last_command_confidence=current.last_command_confidence,
                command_count=current.command_count,
                command_state=current.command_state,
                pose_analysis=current.pose_analysis,
                error=error,
                updated_at=_timestamp(),
                uptime_seconds=self._uptime_seconds(),
            )

    def update_frame(
        self,
        result: PipelineResult,
        latency_ms: float,
        fps: float,
        jpeg_bytes: bytes,
    ) -> None:
        """Publish a processed frame and its recognition state."""

        entry = (
            _command_log_entry(result.command_event, result.frame_index)
            if result.command_event is not None
            else None
        )
        with self._lock:
            last_command = self._snapshot.last_command
            last_command_gesture = self._snapshot.last_command_gesture
            last_command_confidence = self._snapshot.last_command_confidence
            if entry is not None:
                self._commands.appendleft(entry)
                last_command = entry.command
                last_command_gesture = entry.gesture
                last_command_confidence = entry.confidence

            self._latest_jpeg = jpeg_bytes
            self._snapshot = DashboardSnapshot(
                state="running",
                source=self._source,
                transport=self._transport,
                frame_index=result.frame_index,
                fps=fps,
                latency_ms=latency_ms,
                hand_count=len(result.detections),
                static_gesture=result.static_prediction.gesture_id.name,
                static_confidence=result.static_prediction.confidence,
                dynamic_gesture=result.dynamic_prediction.gesture_id.name,
                dynamic_confidence=result.dynamic_prediction.confidence,
                dynamic_state=result.dynamic_state,
                selected_gesture=result.selected_prediction.gesture_id.name,
                selected_confidence=result.selected_prediction.confidence,
                last_command=last_command,
                last_command_gesture=last_command_gesture,
                last_command_confidence=last_command_confidence,
                command_count=len(self._commands),
                command_state=result.command_state,
                pose_analysis=result.pose_analysis,
                error="",
                updated_at=_timestamp(),
                uptime_seconds=self._uptime_seconds(),
            )

    def latest_jpeg(self) -> bytes | None:
        """Return the most recent encoded frame."""

        with self._lock:
            return self._latest_jpeg

    def status_payload(self) -> dict[str, object]:
        """Return a JSON-serializable status payload."""

        with self._lock:
            snapshot = self._snapshot
            uptime_seconds = self._uptime_seconds()
        return {
            "state": snapshot.state,
            "source": snapshot.source,
            "transport": snapshot.transport,
            "frame_index": snapshot.frame_index,
            "fps": _rounded(snapshot.fps),
            "latency_ms": _rounded(snapshot.latency_ms),
            "hand_count": snapshot.hand_count,
            "static_gesture": snapshot.static_gesture,
            "static_confidence": _rounded(snapshot.static_confidence),
            "dynamic_gesture": snapshot.dynamic_gesture,
            "dynamic_confidence": _rounded(snapshot.dynamic_confidence),
            "dynamic_state": snapshot.dynamic_state,
            "selected_gesture": snapshot.selected_gesture,
            "selected_confidence": _rounded(snapshot.selected_confidence),
            "last_command": snapshot.last_command,
            "last_command_gesture": snapshot.last_command_gesture,
            "last_command_confidence": _rounded(snapshot.last_command_confidence),
            "command_count": snapshot.command_count,
            "command_candidate_gesture": snapshot.command_state.gesture_id.name,
            "command_candidate_command": snapshot.command_state.command.value,
            "command_candidate_confidence": _rounded(snapshot.command_state.confidence),
            "command_stable_frames": snapshot.command_state.stable_frames,
            "command_required_frames": snapshot.command_state.required_frames,
            "command_progress": _rounded(_progress_ratio(snapshot.command_state)),
            "command_ready": snapshot.command_state.ready,
            "command_blocked_reason": snapshot.command_state.blocked_reason,
            "command_min_confidence": _rounded(self._command_config.min_confidence),
            "static_confirmation_frames": self._command_config.static_confirmation_frames,
            "dynamic_confirmation_frames": self._command_config.dynamic_confirmation_frames,
            "emergency_confirmation_frames": self._command_config.emergency_confirmation_frames,
            "landmarks": _landmarks_payload(snapshot.pose_analysis),
            "finger_states": _finger_states_payload(snapshot.pose_analysis),
            "pose_directions": _pose_directions_payload(snapshot.pose_analysis),
            "expected_pose": _expected_pose_payload(snapshot.selected_gesture),
            "error": snapshot.error,
            "updated_at": snapshot.updated_at,
            "uptime_seconds": _rounded(uptime_seconds),
        }

    def command_payload(self) -> list[dict[str, object]]:
        """Return recent command events for the browser."""

        with self._lock:
            entries = list(self._commands)
        return [
            {
                "command": entry.command,
                "gesture": entry.gesture,
                "confidence": _rounded(entry.confidence),
                "frame_index": entry.frame_index,
                "created_at": entry.created_at,
            }
            for entry in entries
        ]

    def command_config(self) -> CommandMappingConfig:
        """Return the active command confirmation settings."""

        with self._lock:
            return self._command_config

    def settings_payload(self) -> dict[str, object]:
        """Return runtime settings for the browser controls."""

        with self._lock:
            config = self._command_config
        return _settings_payload(config)

    def update_command_config(self, config: CommandMappingConfig) -> dict[str, object]:
        """Store updated command confirmation settings."""

        with self._lock:
            self._command_config = config
        return _settings_payload(config)

    def _uptime_seconds(self) -> float:
        return perf_counter() - self._started_at


class BrowserFrameProcessor:
    """Process frames uploaded by the browser camera mode."""

    def __init__(
        self,
        state: DashboardState,
        config: AppConfig,
        sender: CommandSender,
        jpeg_quality: int,
        static_model: str | None = None,
        dynamic_model: str | None = None,
    ) -> None:
        """Initialize decoder, pipeline, and per-session timing state."""

        self._state = state
        self._config = config
        self._sender = sender
        self._jpeg_quality = jpeg_quality
        self._cv2 = importlib.import_module("cv2")
        self._numpy = importlib.import_module("numpy")
        self._pipeline = _build_pipeline(
            config=config,
            sender=sender,
            static_model=static_model,
            dynamic_model=dynamic_model,
        )
        self._lock = threading.Lock()
        self._frame_index = 0
        self._last_frame_at: float | None = None
        self._smoothed_fps: float | None = None

    def process_payload(self, payload: dict[str, object]) -> dict[str, object]:
        """Process one JSON frame payload and return updated dashboard data."""

        image_value = payload.get("image")
        if not isinstance(image_value, str) or not image_value:
            raise ValueError("Frame payload must contain a non-empty 'image' string.")

        with self._lock:
            bgr_frame = self._decode_frame(image_value)
            frame = self._captured_frame_from_bgr(bgr_frame)
            started_at = perf_counter()
            self._pipeline.configure_command_mapping(self._state.command_config())
            result = self._pipeline.process(frame)
            latency_ms = (perf_counter() - started_at) * 1000
            now = perf_counter()
            if self._last_frame_at is not None:
                instant_fps = 1.0 / max(now - self._last_frame_at, 1e-6)
                self._smoothed_fps = (
                    instant_fps
                    if self._smoothed_fps is None
                    else (self._smoothed_fps * 0.85) + (instant_fps * 0.15)
                )
            self._last_frame_at = now
            display_fps = self._smoothed_fps if self._smoothed_fps is not None else 0.0
            annotated = _annotate_frame(
                cv2=self._cv2,
                frame=frame.bgr_frame,
                result=result,
                latency_ms=latency_ms,
                fps=display_fps,
            )
            success, encoded = self._cv2.imencode(
                ".jpg",
                annotated,
                [int(self._cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality],
            )
            if not success:
                raise RuntimeError("Cannot encode processed browser frame.")

            self._state.update_frame(
                result=result,
                latency_ms=latency_ms,
                fps=display_fps,
                jpeg_bytes=bytes(encoded),
            )
            return {
                "status": self._state.status_payload(),
                "commands": self._state.command_payload(),
                "frame": _data_url(bytes(encoded)),
            }

    def close(self) -> None:
        """Release pipeline resources."""

        self._pipeline.close()
        if isinstance(self._sender, SerialCommandSender):
            self._sender.close()

    def _decode_frame(self, image_value: str) -> Any:
        if "," in image_value and image_value.lstrip().startswith("data:"):
            _header, image_value = image_value.split(",", 1)
        try:
            image_bytes = base64.b64decode(image_value, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Frame image must be valid base64 data.") from exc

        array = self._numpy.frombuffer(image_bytes, dtype=self._numpy.uint8)
        frame = self._cv2.imdecode(array, self._cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Frame image could not be decoded by OpenCV.")
        return frame

    def _captured_frame_from_bgr(self, frame: Any) -> CapturedFrame:
        resized = self._cv2.resize(
            frame,
            (self._config.video.frame_width, self._config.video.frame_height),
        )
        bgr_frame = self._cv2.flip(resized, 1) if self._config.video.mirror_frame else resized
        rgb_frame = self._cv2.cvtColor(bgr_frame, self._cv2.COLOR_BGR2RGB)
        captured = CapturedFrame(
            bgr_frame=bgr_frame,
            rgb_frame=rgb_frame,
            index=self._frame_index,
        )
        self._frame_index += 1
        return captured


class CaptureWorker:
    """Background capture and recognition loop."""

    def __init__(
        self,
        state: DashboardState,
        config: AppConfig,
        video_path: str | None,
        sender: CommandSender,
        stop_event: threading.Event,
        loop_video: bool,
        jpeg_quality: int,
        static_model: str | None = None,
        dynamic_model: str | None = None,
    ) -> None:
        """Initialize the worker without opening hardware resources."""

        self._state = state
        self._config = config
        self._video_path = video_path
        self._sender = sender
        self._stop_event = stop_event
        self._loop_video = loop_video
        self._jpeg_quality = jpeg_quality
        self._static_model = static_model
        self._dynamic_model = dynamic_model
        self._thread = threading.Thread(target=self._run, name="gesture-ui-capture", daemon=True)

    def start(self) -> None:
        """Start the worker thread."""

        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        """Wait for the worker to finish."""

        self._thread.join(timeout)

    def _run(self) -> None:
        pipeline: GestureControlPipeline | None = None
        serial_sender = self._sender if isinstance(self._sender, SerialCommandSender) else None
        try:
            cv2 = importlib.import_module("cv2")
            pipeline = _build_pipeline(
                config=self._config,
                sender=self._sender,
                static_model=self._static_model,
                dynamic_model=self._dynamic_model,
            )
            capture = VideoCapture(self._config.video, video_path=self._video_path, cv2_module=cv2)
            self._state.set_state("opening")
            with capture:
                self._state.set_state("running")
                smoothed_fps: float | None = None
                last_frame_at = perf_counter()
                while not self._stop_event.is_set():
                    frame = capture.read()
                    if frame is None:
                        if self._video_path is not None and self._loop_video:
                            capture.release()
                            capture.open()
                            continue
                        self._state.set_state("ended")
                        break

                    started_at = perf_counter()
                    pipeline.configure_command_mapping(self._state.command_config())
                    result = pipeline.process(frame)
                    latency_ms = (perf_counter() - started_at) * 1000
                    now = perf_counter()
                    instant_fps = 1.0 / max(now - last_frame_at, 1e-6)
                    last_frame_at = now
                    smoothed_fps = (
                        instant_fps
                        if smoothed_fps is None
                        else (smoothed_fps * 0.85) + (instant_fps * 0.15)
                    )
                    annotated = _annotate_frame(
                        cv2=cv2,
                        frame=frame.bgr_frame,
                        result=result,
                        latency_ms=latency_ms,
                        fps=smoothed_fps,
                    )
                    success, encoded = cv2.imencode(
                        ".jpg",
                        annotated,
                        [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality],
                    )
                    if success:
                        self._state.update_frame(
                            result=result,
                            latency_ms=latency_ms,
                            fps=smoothed_fps,
                            jpeg_bytes=bytes(encoded),
                        )
        except Exception as exc:
            LOGGER.exception("Web UI capture worker failed")
            self._state.set_state("error", str(exc))
        finally:
            if pipeline is not None:
                pipeline.close()
            if serial_sender is not None:
                serial_sender.close()


class GestureDashboardServer(ThreadingHTTPServer):
    """HTTP server carrying dashboard state."""

    def __init__(
        self,
        server_address: tuple[str, int],
        state: DashboardState,
        stop_event: threading.Event,
        frame_processor: BrowserFrameProcessor | None = None,
        cors_origin: str | None = None,
    ) -> None:
        """Initialize the local dashboard server."""

        super().__init__(server_address, GestureDashboardHandler)
        self.state = state
        self.stop_event = stop_event
        self.frame_processor = frame_processor
        self.cors_origin = cors_origin


class GestureDashboardHandler(BaseHTTPRequestHandler):
    """Serve dashboard HTML, JSON status, and MJPEG frames."""

    server_version = "GestureDashboard/1.0"

    def do_OPTIONS(self) -> None:
        """Handle CORS preflight requests."""

        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_common_headers()
        self.end_headers()

    def do_GET(self) -> None:
        """Handle dashboard GET routes."""

        path = urlparse(self.path).path
        if path == "/":
            self._send_text(INDEX_HTML, "text/html; charset=utf-8")
        elif path == "/styles.css":
            self._send_text(STYLES_CSS, "text/css; charset=utf-8")
        elif path == "/app.js":
            self._send_text(APP_JS, "application/javascript; charset=utf-8")
        elif path == "/api/status":
            self._send_json(self._dashboard_server().state.status_payload())
        elif path == "/api/settings":
            self._send_json(self._dashboard_server().state.settings_payload())
        elif path == "/api/commands":
            self._send_json(self._dashboard_server().state.command_payload())
        elif path == "/stream.mjpg":
            self._send_stream()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        """Handle dashboard POST routes."""

        path = urlparse(self.path).path
        if path == "/api/settings":
            try:
                state = self._dashboard_server().state
                payload = self._read_json_body(max_bytes=10_000)
                config = _command_config_from_payload(state.command_config(), payload)
                response = state.update_command_config(config)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(response)
            return

        if path != "/api/frame":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        processor = self._dashboard_server().frame_processor
        if processor is None:
            self._send_json(
                {"error": "Browser camera processing is disabled on this server."},
                status=HTTPStatus.SERVICE_UNAVAILABLE,
            )
            return

        try:
            payload = self._read_json_body(max_bytes=3_000_000)
            response = processor.process_payload(payload)
        except Exception as exc:
            LOGGER.exception("Browser frame processing failed")
            self._dashboard_server().state.set_state("error", str(exc))
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        self._send_json(response)

    def log_message(self, format_value: str, *args: object) -> None:
        """Route HTTP logs through the project logger."""

        LOGGER.debug("HTTP %s", format_value % args)

    def _dashboard_server(self) -> GestureDashboardServer:
        return cast(GestureDashboardServer, self.server)

    def _send_text(self, body: str, content_type: str) -> None:
        payload = body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self._send_common_headers(content_type=content_type, content_length=len(payload))
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self._send_common_headers(
            content_type="application/json; charset=utf-8",
            content_length=len(body),
        )
        self.end_headers()
        self.wfile.write(body)

    def _send_stream(self) -> None:
        self.send_response(HTTPStatus.OK)
        self._send_common_headers(content_type="multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        server = self._dashboard_server()
        while not server.stop_event.is_set():
            jpeg = server.state.latest_jpeg()
            if jpeg is None:
                time.sleep(0.1)
                continue
            try:
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                break
            time.sleep(1 / 15)

    def _send_common_headers(
        self,
        content_type: str | None = None,
        content_length: int | None = None,
    ) -> None:
        if content_type is not None:
            self.send_header("Content-Type", content_type)
        if content_length is not None:
            self.send_header("Content-Length", str(content_length))
        self.send_header("Cache-Control", "no-store")
        cors_origin = self._dashboard_server().cors_origin
        if cors_origin:
            self.send_header("Access-Control-Allow-Origin", cors_origin)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _read_json_body(self, max_bytes: int) -> dict[str, object]:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise ValueError("Missing Content-Length header.")
        length = int(raw_length)
        if length > max_bytes:
            raise ValueError("Request body is too large.")
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object.")
        return payload


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""

    parser = argparse.ArgumentParser(description="Run the local gesture control web UI.")
    parser.add_argument("--host", default="127.0.0.1", help="Dashboard host.")
    parser.add_argument("--port", type=int, default=8000, help="Dashboard port.")
    parser.add_argument(
        "--input-mode",
        choices=("server-camera", "browser-camera"),
        default="server-camera",
        help="Use a server-side camera/video source or frames uploaded from the browser.",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index for live capture.")
    parser.add_argument("--video", type=str, default=None, help="Path to a video file.")
    parser.add_argument("--frame-width", type=int, default=480, help="Capture frame width.")
    parser.add_argument("--frame-height", type=int, default=360, help="Capture frame height.")
    parser.add_argument("--target-fps", type=int, default=30, help="Requested camera FPS.")
    parser.add_argument("--no-mirror", action="store_true", help="Disable mirrored preview.")
    parser.add_argument("--loop-video", action="store_true", help="Loop video files.")
    parser.add_argument("--jpeg-quality", type=int, default=82, help="MJPEG JPEG quality.")
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
    parser.add_argument("--open-browser", action="store_true", help="Open the dashboard URL.")
    parser.add_argument(
        "--cors-origin",
        default=None,
        help="Optional CORS origin for static frontend deployments, for example '*'.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the web dashboard."""

    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.jpeg_quality < 1 or args.jpeg_quality > 100:
        raise ValueError("--jpeg-quality must be in the range 1..100")

    video_config = VideoConfig(
        camera_index=args.camera,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        target_fps=args.target_fps,
        mirror_frame=not args.no_mirror,
    )
    config = AppConfig(video=video_config)
    sender = _build_sender(args.sender, config)
    source = (
        "browser-camera"
        if args.input_mode == "browser-camera"
        else args.video if args.video is not None else f"camera:{args.camera}"
    )
    state = DashboardState(
        source=source,
        transport=args.sender,
        command_config=config.command_mapping,
    )
    dynamic_model = "" if args.no_dynamic_model else args.dynamic_model
    stop_event = threading.Event()
    worker: CaptureWorker | None = None
    frame_processor: BrowserFrameProcessor | None = None
    if args.input_mode == "browser-camera":
        state.set_state("waiting")
        frame_processor = BrowserFrameProcessor(
            state=state,
            config=config,
            sender=sender,
            jpeg_quality=args.jpeg_quality,
            static_model=args.static_model,
            dynamic_model=dynamic_model,
        )
    else:
        worker = CaptureWorker(
            state=state,
            config=config,
            video_path=args.video,
            sender=sender,
            stop_event=stop_event,
            loop_video=args.loop_video,
            jpeg_quality=args.jpeg_quality,
            static_model=args.static_model,
            dynamic_model=dynamic_model,
        )
    server = GestureDashboardServer(
        (args.host, args.port),
        state,
        stop_event,
        frame_processor=frame_processor,
        cors_origin=args.cors_origin,
    )
    url = f"http://{args.host}:{server.server_port}"
    if worker is not None:
        worker.start()

    if args.open_browser:
        webbrowser.open(url)

    LOGGER.info("Gesture dashboard running at %s", url)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        LOGGER.info("Stopping gesture dashboard")
    finally:
        stop_event.set()
        server.server_close()
        if worker is not None:
            worker.join(timeout=3.0)
        if frame_processor is not None:
            frame_processor.close()
    return 0


def _build_sender(sender_name: str, config: AppConfig) -> CommandSender:
    if sender_name == "serial":
        sender = SerialCommandSender(config.sender)
        sender.open()
        return sender
    return MockCommandSender()


def _build_pipeline(
    config: AppConfig,
    sender: CommandSender,
    static_model: str | None,
    dynamic_model: str | None,
) -> GestureControlPipeline:
    static_classifier = None
    dynamic_classifier = None
    if static_model is not None:
        static_classifier = FallbackStaticGestureClassifier(
            primary=SklearnStaticGestureClassifier.load_path(
                static_model,
                min_confidence=config.command_mapping.min_confidence,
            ),
            fallback=StaticGestureClassifier(config.static_classifier),
        )
    dynamic_model_path = _resolve_dynamic_model_path(dynamic_model)
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
                "Cannot load dynamic model %s; using heuristics: %s",
                dynamic_model_path,
                exc,
            )
    return GestureControlPipeline(
        config=config,
        static_classifier=static_classifier,
        dynamic_classifier=dynamic_classifier,
        command_sender=sender,
    )


def _resolve_dynamic_model_path(model_path: str | None) -> Path | None:
    if model_path == "":
        return None
    if model_path is not None:
        return Path(model_path)
    return DEFAULT_DYNAMIC_MODEL_PATH if DEFAULT_DYNAMIC_MODEL_PATH.exists() else None


def _annotate_frame(
    cv2: Any,
    frame: Any,
    result: PipelineResult,
    latency_ms: float,
    fps: float,
) -> Any:
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    _draw_hand_landmarks(cv2, annotated, result, width, height)
    _draw_overlay(cv2, annotated, result, latency_ms, fps)
    return annotated


def _draw_hand_landmarks(
    cv2: Any,
    frame: Any,
    result: PipelineResult,
    width: int,
    height: int,
) -> None:
    if not result.detections:
        return
    detection = max(result.detections, key=lambda item: item.score)
    points = [(int(x * width), int(y * height)) for x, y, _z in detection.landmarks]
    for start, end in HAND_CONNECTIONS:
        if start < len(points) and end < len(points):
            cv2.line(frame, points[start], points[end], (26, 122, 86), 2, cv2.LINE_AA)
    for index, point in enumerate(points):
        cv2.circle(frame, point, 4, (21, 130, 199), -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            str(index),
            (point[0] + 5, point[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (245, 247, 250),
            1,
            cv2.LINE_AA,
        )


def _draw_overlay(
    cv2: Any,
    frame: Any,
    result: PipelineResult,
    latency_ms: float,
    fps: float,
) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 112), (23, 28, 34), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    selected = result.selected_prediction
    command = result.command_event.command.value if result.command_event else "PENDING"
    confidence = f"{selected.confidence:.2f}"
    lines = (
        f"Gesture: {selected.gesture_id.name}  confidence={confidence}",
        f"Command: {command}",
        f"FPS: {fps:.1f}  latency={latency_ms:.1f} ms  frame={result.frame_index}",
    )
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (18, 30 + index * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (245, 247, 250),
            2,
            cv2.LINE_AA,
        )


def _command_log_entry(event: CommandEvent, frame_index: int) -> CommandLogEntry:
    return CommandLogEntry(
        command=event.command.value,
        gesture=event.gesture_id.name,
        confidence=event.confidence,
        frame_index=frame_index,
        created_at=_timestamp(),
    )


def _rounded(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 3)


def _data_url(jpeg_bytes: bytes) -> str:
    encoded = base64.b64encode(jpeg_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _progress_ratio(state: CommandConfirmationState) -> float:
    if state.required_frames <= 0:
        return 0.0
    return min(1.0, state.stable_frames / state.required_frames)


def _landmarks_payload(analysis: StaticPoseAnalysis | None) -> list[dict[str, object]]:
    if analysis is None:
        return []
    return [
        {
            "id": index,
            "x": _rounded(point[0]),
            "y": _rounded(point[1]),
            "z": _rounded(point[2]),
        }
        for index, point in enumerate(analysis.landmarks)
    ]


def _finger_states_payload(analysis: StaticPoseAnalysis | None) -> dict[str, bool]:
    if analysis is None:
        return {}
    return analysis.finger_states.as_dict()


def _pose_directions_payload(analysis: StaticPoseAnalysis | None) -> dict[str, object]:
    if analysis is None:
        return {}
    return {
        "thumb": analysis.thumb_direction,
        "index": analysis.index_direction,
        "ok_tip_distance": _rounded(analysis.ok_tip_distance_ratio),
    }


def _expected_pose_payload(gesture_name: str) -> dict[str, object]:
    try:
        gesture_id = GestureID[gesture_name]
    except KeyError:
        return {}
    spec = expected_pose_for(gesture_id)
    if spec is None:
        return {}
    return {
        "gesture": spec.gesture_id.name,
        "finger_states": spec.finger_states,
        "thumb_direction": spec.thumb_direction,
        "index_direction": spec.index_direction,
        "max_ok_tip_distance": _rounded(spec.max_ok_tip_distance_ratio),
    }


def _settings_payload(config: CommandMappingConfig) -> dict[str, object]:
    return {
        "static_confirmation_frames": config.static_confirmation_frames,
        "dynamic_confirmation_frames": config.dynamic_confirmation_frames,
        "emergency_confirmation_frames": config.emergency_confirmation_frames,
        "min_confidence": _rounded(config.min_confidence),
        "repeat_same_command": config.repeat_same_command,
    }


def _command_config_from_payload(
    current: CommandMappingConfig,
    payload: dict[str, object],
) -> CommandMappingConfig:
    static_frames = _confirmation_frames(
        payload.get("static_confirmation_frames"),
        current.static_confirmation_frames,
    )
    dynamic_frames = _confirmation_frames(
        payload.get("dynamic_confirmation_frames"),
        current.dynamic_confirmation_frames,
    )
    emergency_frames = _confirmation_frames(
        payload.get("emergency_confirmation_frames"),
        current.emergency_confirmation_frames,
    )
    return CommandMappingConfig(
        static_confirmation_frames=static_frames,
        dynamic_confirmation_frames=dynamic_frames,
        emergency_confirmation_frames=emergency_frames,
        min_confidence=current.min_confidence,
        repeat_same_command=current.repeat_same_command,
    )


def _confirmation_frames(value: object, fallback: int) -> int:
    if value is None:
        return fallback
    try:
        frames = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Confirmation frames must be whole numbers.") from exc
    if frames < 1 or frames > 12:
        raise ValueError("Confirmation frames must be in the range 1..12.")
    return frames


def _timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Gesture Control Console</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <main class="app-shell">
    <header class="topbar">
      <div class="title-block">
        <h1>Gesture Control Console</h1>
        <p id="sourceLine">camera:0</p>
      </div>
      <div class="state-strip">
        <span id="stateBadge" class="badge badge-muted">starting</span>
        <span id="transportBadge" class="badge">mock</span>
      </div>
    </header>

    <section class="workspace">
      <section class="video-panel" aria-label="Recognition stream">
        <div class="video-stage">
          <img id="stream" src="stream.mjpg" alt="Live gesture recognition stream">
          <video id="browserVideo" class="browser-video" autoplay playsinline muted hidden></video>
          <canvas id="captureCanvas" hidden></canvas>
          <button id="startVideoButton" class="video-start-button" type="button">
            Start video
          </button>
        </div>
        <p class="video-access-note">
          Camera/video access is required to recognize gestures and control the robot.
        </p>
        <div class="frame-strip">
          <div><span>Selected</span><strong id="selectedGesture">UNKNOWN</strong></div>
          <div><span>Last command</span><strong id="lastCommand">UNKNOWN</strong></div>
          <div><span>Confidence</span><strong id="selectedConfidence">0.00</strong></div>
          <div><span>Dynamic state</span><strong id="dynamicState">idle</strong></div>
        </div>
      </section>

      <aside class="status-rail">
        <section class="panel control-panel">
          <span class="eyebrow">control</span>
          <div class="control-buttons">
            <button id="sideStartButton" class="primary-button" type="button">Start</button>
            <button id="stopVideoButton" class="secondary-button" type="button">Stop</button>
            <button id="reloadButton" class="secondary-button" type="button">Reload</button>
          </div>
          <label class="field-label">
            <span>Backend URL</span>
            <input
              id="apiBaseInput"
              class="api-input"
              type="url"
              inputmode="url"
              placeholder="same-origin"
              aria-label="Backend URL"
            >
          </label>
        </section>

        <section class="panel robot-panel">
          <span class="eyebrow">robot transmission</span>
          <strong id="robotTransferState">WAITING</strong>
          <div class="robot-grid">
            <div><span>Transport</span><b id="transportText">mock</b></div>
            <div><span>Sent</span><b id="robotCommandCount">0</b></div>
            <div><span>Last gesture</span><b id="robotLastGesture">UNKNOWN</b></div>
          </div>
        </section>

        <section class="panel command-panel">
          <span class="eyebrow">last confirmed command</span>
          <strong id="commandDisplay">UNKNOWN</strong>
          <span id="lastCommandMeta">UNKNOWN / 0.00</span>
        </section>

        <section class="panel confirmation-panel">
          <span class="eyebrow">confirmation</span>
          <strong id="candidateCommand">UNKNOWN</strong>
          <span id="candidateGesture">UNKNOWN / 0.00</span>
          <div class="progress-track" aria-hidden="true">
            <span id="confirmationProgressBar"></span>
          </div>
          <span id="confirmationProgressText">0 / 0 frames</span>
        </section>

        <section class="panel settings-panel">
          <div class="section-header compact-header">
            <h2>Confirmation frames</h2>
            <span id="settingsStatus">saved</span>
          </div>
          <div class="settings-grid">
            <label>
              <span>Static</span>
              <input id="staticFramesInput" type="number" min="1" max="12" step="1" value="5">
            </label>
            <label>
              <span>Dynamic</span>
              <input id="dynamicFramesInput" type="number" min="1" max="12" step="1" value="1">
            </label>
            <label>
              <span>Emergency</span>
              <input id="emergencyFramesInput" type="number" min="1" max="12" step="1" value="3">
            </label>
          </div>
          <button id="saveSettingsButton" class="primary-button full-button" type="button">
            Apply
          </button>
        </section>

        <section class="metrics-grid">
          <div class="metric"><span>FPS</span><strong id="fps">0.0</strong></div>
          <div class="metric"><span>Latency</span><strong id="latency">0 ms</strong></div>
          <div class="metric"><span>Hands</span><strong id="handCount">0</strong></div>
          <div class="metric"><span>Frame</span><strong id="frameIndex">0</strong></div>
        </section>

        <section class="panel split-panel">
          <div><span>Static</span><strong id="staticGesture">UNKNOWN</strong></div>
          <div><span>Dynamic</span><strong id="dynamicGesture">UNKNOWN</strong></div>
        </section>

        <section class="panel pose-panel">
          <div class="section-header compact-header">
            <h2>Hand pose</h2>
            <span id="poseDirection">--</span>
          </div>
          <div id="fingerStates" class="finger-states"></div>
        </section>

        <section class="panel landmarks-panel">
          <div class="section-header compact-header">
            <h2>Landmark points</h2>
            <span id="landmarkCount">0 / 21</span>
          </div>
          <div id="landmarkList" class="landmark-list"></div>
        </section>

        <section class="panel error-panel" id="errorPanel" hidden>
          <span class="eyebrow">runtime</span>
          <strong id="errorText"></strong>
        </section>
      </aside>
    </section>

    <section class="lower-grid">
      <section class="log-band">
        <div class="section-header">
          <h2>Command Log</h2>
          <span id="updatedAt">--:--:--</span>
        </div>
        <table>
          <thead>
            <tr>
              <th>Time</th>
              <th>Command</th>
              <th>Gesture</th>
              <th>Confidence</th>
              <th>Frame</th>
            </tr>
          </thead>
          <tbody id="commandRows">
            <tr><td colspan="5" class="empty">No commands emitted</td></tr>
          </tbody>
        </table>
      </section>

      <section class="map-band">
        <div class="section-header">
          <h2>Gesture Map</h2>
          <span>13 gestures</span>
        </div>
        <div id="gestureMap" class="gesture-map"></div>
      </section>
    </section>
  </main>
  <script src="app.js"></script>
</body>
</html>
"""


STYLES_CSS = """
:root {
  color-scheme: light;
  --page: #eef2f5;
  --ink: #17212b;
  --muted: #66727d;
  --line: #d4dde5;
  --panel: #ffffff;
  --panel-soft: #f8fafb;
  --graphite: #252c33;
  --teal: #0f7f68;
  --blue: #1e6ea8;
  --amber: #a96603;
  --red: #b33a3a;
  --violet: #7352a1;
  --shadow: 0 14px 34px rgba(31, 41, 55, 0.08);
}

* {
  box-sizing: border-box;
}

[hidden] {
  display: none !important;
}

body {
  margin: 0;
  min-height: 100vh;
  background: var(--page);
  color: var(--ink);
  font-family: Inter, Segoe UI, Arial, sans-serif;
}

.app-shell {
  width: min(1320px, 100%);
  margin: 0 auto;
  padding: 18px 20px 24px;
}

.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 18px;
  padding: 8px 2px 18px;
}

h1, h2, p {
  margin: 0;
}

h1 {
  font-size: 27px;
  font-weight: 760;
  letter-spacing: 0;
}

.topbar p {
  margin-top: 4px;
  color: var(--muted);
  font-size: 14px;
}

.state-strip {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.button-row {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.api-input {
  width: min(260px, 100%);
  min-height: 32px;
  padding: 6px 10px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: var(--panel);
  color: var(--ink);
  font: inherit;
  font-size: 13px;
}

.api-input:focus {
  border-color: rgba(30, 110, 168, 0.55);
  outline: 2px solid rgba(30, 110, 168, 0.12);
}

.icon-button {
  min-height: 32px;
  padding: 6px 12px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: var(--panel);
  color: var(--graphite);
  font: inherit;
  font-size: 13px;
  font-weight: 750;
  cursor: pointer;
}

.icon-button:hover,
.icon-button.active {
  border-color: rgba(30, 110, 168, 0.45);
  color: var(--blue);
}

.primary-button,
.secondary-button {
  min-height: 36px;
  padding: 8px 12px;
  border: 1px solid var(--line);
  border-radius: 6px;
  font: inherit;
  font-size: 13px;
  font-weight: 780;
  cursor: pointer;
}

.primary-button {
  border-color: rgba(15, 127, 104, 0.38);
  background: var(--teal);
  color: #ffffff;
}

.primary-button:hover {
  background: #0c6f5c;
}

.secondary-button {
  background: var(--panel);
  color: var(--graphite);
}

.secondary-button:hover {
  border-color: rgba(30, 110, 168, 0.45);
  color: var(--blue);
}

.badge {
  min-width: 88px;
  min-height: 32px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 6px 12px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: var(--panel);
  color: var(--graphite);
  font-size: 13px;
  font-weight: 700;
  text-transform: uppercase;
}

.badge-running {
  border-color: rgba(19, 122, 99, 0.35);
  color: var(--teal);
}

.badge-error {
  border-color: rgba(179, 58, 58, 0.35);
  color: var(--red);
}

.badge-muted {
  color: var(--muted);
}

.workspace {
  display: grid;
  grid-template-columns: minmax(520px, 860px) minmax(340px, 390px);
  gap: 18px;
  justify-content: center;
  align-items: start;
}

.video-panel,
.log-band,
.map-band {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: var(--shadow);
}

.video-stage {
  position: relative;
  aspect-ratio: 4 / 3;
  min-height: 0;
  background: #11161b;
  border-radius: 8px;
  overflow: hidden;
  display: grid;
  place-items: center;
}

.video-stage img {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
  background: #11161b;
}

.video-stage.browser-active .browser-video {
  position: absolute;
  width: 1px;
  height: 1px;
  opacity: 0;
  pointer-events: none;
}

.browser-video {
  width: 100%;
  height: 100%;
  object-fit: contain;
  background: #11161b;
  transform: scaleX(-1);
}

.video-start-button {
  position: absolute;
  left: 50%;
  top: 50%;
  transform: translate(-50%, -50%);
  min-width: 154px;
  min-height: 48px;
  padding: 12px 22px;
  border: 1px solid rgba(255, 255, 255, 0.3);
  border-radius: 8px;
  background: rgba(15, 127, 104, 0.94);
  color: #ffffff;
  font: inherit;
  font-size: 16px;
  font-weight: 820;
  cursor: pointer;
  box-shadow: 0 16px 32px rgba(0, 0, 0, 0.28);
}

.video-start-button:hover {
  background: rgba(12, 111, 92, 0.98);
}

.video-stage.video-started .video-start-button {
  opacity: 0;
  pointer-events: none;
}

.video-access-note {
  padding: 12px 14px;
  border-top: 1px solid var(--line);
  color: var(--muted);
  font-size: 13px;
  line-height: 1.45;
}

.frame-strip {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  border-top: 1px solid var(--line);
}

.frame-strip div {
  min-height: 74px;
  padding: 13px 14px;
  border-right: 1px solid var(--line);
}

.frame-strip div:last-child {
  border-right: 0;
}

.frame-strip span {
  display: block;
  color: var(--muted);
  font-size: 12px;
  font-weight: 780;
  text-transform: uppercase;
}

.frame-strip strong {
  display: block;
  margin-top: 8px;
  font-size: 20px;
  overflow-wrap: anywhere;
}

.status-rail {
  display: grid;
  gap: 12px;
  align-content: start;
}

.panel,
.metric {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: var(--shadow);
}

.control-panel,
.robot-panel,
.settings-panel {
  padding: 14px;
}

.control-buttons {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 8px;
  margin-top: 10px;
}

.field-label {
  display: grid;
  gap: 6px;
  margin-top: 12px;
  color: var(--muted);
  font-size: 12px;
  font-weight: 760;
  text-transform: uppercase;
}

.field-label .api-input {
  width: 100%;
  text-transform: none;
  font-weight: 500;
}

.robot-panel {
  border-top: 4px solid var(--blue);
}

.robot-panel > strong {
  display: block;
  margin-top: 8px;
  font-size: 22px;
  line-height: 1.15;
}

.robot-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 7px;
  margin-top: 12px;
}

.robot-grid div {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  min-height: 30px;
  padding: 6px 8px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: var(--panel-soft);
}

.robot-grid span {
  color: var(--muted);
  font-size: 12px;
  font-weight: 760;
  text-transform: uppercase;
}

.robot-grid b {
  font-size: 12px;
  overflow-wrap: anywhere;
  text-align: right;
}

.command-panel {
  padding: 18px;
  border-top: 4px solid var(--teal);
}

.command-panel strong {
  display: block;
  margin-top: 8px;
  font-size: 30px;
  line-height: 1.1;
  overflow-wrap: anywhere;
}

.command-panel span:last-child {
  display: block;
  margin-top: 8px;
  color: var(--muted);
  font-size: 13px;
}

.confirmation-panel {
  padding: 16px;
}

.confirmation-panel strong {
  display: block;
  margin-top: 8px;
  font-size: 23px;
  line-height: 1.15;
  overflow-wrap: anywhere;
}

.confirmation-panel > span:not(.eyebrow) {
  display: block;
  margin-top: 8px;
  color: var(--muted);
  font-size: 13px;
  overflow-wrap: anywhere;
}

.progress-track {
  width: 100%;
  height: 8px;
  margin-top: 12px;
  overflow: hidden;
  border-radius: 999px;
  background: #dfe7ee;
}

.progress-track span {
  display: block;
  width: 0%;
  height: 100%;
  border-radius: inherit;
  background: var(--blue);
  transition: width 160ms ease;
}

.confirmation-panel.ready .progress-track span {
  background: var(--teal);
}

.confirmation-panel.blocked .progress-track span {
  background: var(--amber);
}

.settings-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px;
}

.settings-grid label {
  display: grid;
  gap: 6px;
  min-width: 0;
}

.settings-grid span {
  color: var(--muted);
  font-size: 12px;
  font-weight: 760;
  text-transform: uppercase;
}

.settings-grid input {
  width: 100%;
  min-height: 34px;
  padding: 6px 8px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: var(--panel-soft);
  color: var(--ink);
  font: inherit;
  font-weight: 760;
}

.full-button {
  width: 100%;
  margin-top: 12px;
}

.eyebrow {
  color: var(--muted);
  font-size: 12px;
  font-weight: 800;
  text-transform: uppercase;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}

.metric {
  min-height: 82px;
  padding: 12px;
}

.metric span,
.split-panel span {
  display: block;
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
}

.metric strong {
  display: block;
  margin-top: 8px;
  font-size: 21px;
  overflow-wrap: anywhere;
}

.split-panel {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1px;
  overflow: hidden;
}

.split-panel div {
  min-height: 84px;
  padding: 14px;
  background: var(--panel-soft);
}

.split-panel strong {
  display: block;
  margin-top: 8px;
  font-size: 16px;
  overflow-wrap: anywhere;
}

.pose-panel,
.landmarks-panel {
  padding: 14px;
}

.compact-header {
  margin-bottom: 10px;
}

.compact-header h2 {
  font-size: 15px;
}

.finger-states {
  display: grid;
  grid-template-columns: 1fr;
  gap: 7px;
}

.finger-row {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto auto;
  gap: 8px;
  align-items: center;
  min-height: 28px;
  padding: 5px 8px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: var(--panel-soft);
  font-size: 12px;
}

.finger-row strong,
.finger-row span {
  overflow-wrap: anywhere;
}

.finger-row .match {
  color: var(--teal);
  font-weight: 800;
}

.finger-row .mismatch {
  color: var(--red);
  font-weight: 800;
}

.landmark-list {
  max-height: 230px;
  overflow: auto;
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 6px;
}

.landmark-point {
  min-height: 42px;
  padding: 5px 6px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: var(--panel-soft);
  font-size: 11px;
  line-height: 1.35;
}

.landmark-point strong {
  display: block;
  font-size: 12px;
}

.error-panel {
  padding: 14px;
  border-color: rgba(179, 58, 58, 0.35);
  color: var(--red);
}

.error-panel strong {
  display: block;
  margin-top: 6px;
  font-size: 14px;
  line-height: 1.35;
}

.lower-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 18px;
  margin-top: 18px;
  align-items: start;
}

.log-band,
.map-band {
  padding: 16px;
  overflow-x: auto;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
}

h2 {
  font-size: 18px;
}

.section-header span {
  color: var(--muted);
  font-size: 13px;
}

table {
  width: 100%;
  border-collapse: collapse;
  min-width: 640px;
}

th,
td {
  padding: 10px 8px;
  border-top: 1px solid var(--line);
  text-align: left;
  font-size: 14px;
}

th {
  color: var(--muted);
  font-size: 12px;
  text-transform: uppercase;
}

.empty {
  color: var(--muted);
  text-align: center;
}

.gesture-map {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 8px;
}

.gesture-item {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: 10px;
  align-items: center;
  min-height: 46px;
  padding: 9px 10px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel-soft);
}

.gesture-item.active {
  border-color: rgba(19, 122, 99, 0.4);
  background: #eef8f5;
}

.gesture-item strong,
.gesture-item span {
  overflow-wrap: anywhere;
}

.gesture-item strong {
  font-size: 13px;
}

.gesture-item span {
  color: var(--muted);
  font-size: 12px;
  text-align: right;
}

@media (max-width: 980px) {
  .workspace,
  .lower-grid {
    grid-template-columns: 1fr;
  }

  .status-rail {
    grid-template-columns: 1fr;
  }

  .landmark-list {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 640px) {
  .app-shell {
    padding: 12px;
  }

  .topbar {
    align-items: flex-start;
    flex-direction: column;
  }

  .api-input {
    width: 100%;
  }

  h1 {
    font-size: 24px;
  }

  .metrics-grid {
    grid-template-columns: 1fr;
  }

  .video-stage,
  .browser-video {
    aspect-ratio: 4 / 3;
  }

  .frame-strip {
    grid-template-columns: 1fr;
  }

  .frame-strip div {
    border-right: 0;
    border-bottom: 1px solid var(--line);
  }

  .frame-strip div:last-child {
    border-bottom: 0;
  }
}
"""


APP_JS = """
const $ = (id) => document.getElementById(id);

const queryParams = new URLSearchParams(window.location.search);
let apiBase = window.GESTURE_API_BASE || queryParams.get("api") || "";
const BROWSER_CAMERA_FRAME_INTERVAL_MS = 50;
const GESTURE_COMMANDS = [
  ["OPEN_PALM", "STOP"],
  ["FIST", "FORWARD"],
  ["THUMB_UP", "START"],
  ["THUMB_DOWN", "EMERGENCY_STOP"],
  ["INDEX_LEFT", "TURN_LEFT"],
  ["INDEX_RIGHT", "TURN_RIGHT"],
  ["PEACE", "INCREASE_SPEED"],
  ["THREE_FINGERS", "DECREASE_SPEED"],
  ["PINKY", "RETURN_HOME"],
  ["OK_SIGN", "CONFIRM_ACTION"],
  ["WAVE_LR", "MODE_TOGGLE"],
  ["CIRCLE", "ROTATE_360"],
  ["PULL_TOWARD", "APPROACH_OPERATOR"],
];
const FINGER_LABELS = [
  ["thumb", "Thumb"],
  ["index", "Index"],
  ["middle", "Middle"],
  ["ring", "Ring"],
  ["pinky", "Pinky"],
];

let browserCameraMode = false;
let commandCache = [];
let browserStream = null;
let frameTimer = null;
let frameInFlight = false;
let currentSource = "";
let settingsDirty = false;

function text(id, value) {
  const node = $(id);
  if (node) node.textContent = value;
}

function fmt(value, suffix = "", digits = 2) {
  if (value === null || value === undefined) return "--";
  if (typeof value === "number") return `${value.toFixed(digits)}${suffix}`;
  return `${value}${suffix}`;
}

function setStateBadge(state) {
  const badge = $("stateBadge");
  if (!badge) return;
  badge.textContent = state;
  badge.className = "badge";
  if (state === "running") badge.classList.add("badge-running");
  else if (state === "error") badge.classList.add("badge-error");
  else badge.classList.add("badge-muted");
}

function setVideoStarted(started) {
  const stage = document.querySelector(".video-stage");
  if (stage) stage.classList.toggle("video-started", Boolean(started));
  const startButton = $("startVideoButton");
  if (startButton) startButton.textContent = started ? "Video running" : "Start video";
  const sideStartButton = $("sideStartButton");
  if (sideStartButton) sideStartButton.textContent = started ? "Running" : "Start";
  const stopButton = $("stopVideoButton");
  if (stopButton) stopButton.disabled = !browserCameraMode;
}

function buildUrl(path) {
  const prefix = apiBase.replace(/\\/$/, "");
  return `${prefix}/${path.replace(/^\\//, "")}`;
}

function syncApiBaseFromInput() {
  const input = $("apiBaseInput");
  if (!input) return;
  apiBase = input.value.trim();
  if (apiBase) window.localStorage.setItem("gestureApiBase", apiBase);
  else window.localStorage.removeItem("gestureApiBase");
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function nowTime() {
  return new Date().toLocaleTimeString("uk-UA", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function recognitionLabel(gesture, confidence, minConfidence) {
  const value = `${gesture} / ${fmt(confidence)}`;
  if (gesture !== "UNKNOWN" && confidence < minConfidence) return `${value} low`;
  return value;
}

function reasonLabel(reason) {
  if (!reason) return "";
  return String(reason).replaceAll("_", " ");
}

function applyConfirmation(status) {
  const minConfidence = status.command_min_confidence ?? 0.65;
  const gesture = status.command_candidate_gesture || "UNKNOWN";
  const command = status.command_candidate_command || "UNKNOWN";
  const confidence = status.command_candidate_confidence ?? 0;
  const stableFrames = status.command_stable_frames || 0;
  const requiredFrames = status.command_required_frames || 0;
  const progress = Math.max(0, Math.min(1, status.command_progress || 0));
  const reason = status.command_blocked_reason || "";
  const panel = document.querySelector(".confirmation-panel");
  if (panel) {
    panel.classList.toggle("ready", Boolean(status.command_ready));
    panel.classList.toggle("blocked", reason === "confidence_below_threshold");
  }
  text("candidateCommand", command);
  text("candidateGesture", `${gesture} / ${fmt(confidence)}`);
  const bar = $("confirmationProgressBar");
  if (bar) bar.style.width = `${Math.round(progress * 100)}%`;
  if (reason === "confidence_below_threshold") {
    text("confirmationProgressText", `${fmt(confidence)} < ${fmt(minConfidence)} threshold`);
  } else if (reason === "repeat_suppressed") {
    text("confirmationProgressText", `${stableFrames} / ${requiredFrames} frames, already sent`);
  } else if (requiredFrames > 0) {
    text("confirmationProgressText", `${stableFrames} / ${requiredFrames} frames`);
  } else {
    text("confirmationProgressText", reasonLabel(reason) || "0 / 0 frames");
  }
}

function renderPoseDiagnostics(status) {
  const states = status.finger_states || {};
  const expected = (status.expected_pose && status.expected_pose.finger_states) || {};
  const directions = status.pose_directions || {};
  const node = $("fingerStates");
  if (!node) return;
  node.innerHTML = FINGER_LABELS.map(([key, label]) => {
    const actual = states[key];
    const target = expected[key];
    const actualText = actual === true ? "open" : actual === false ? "closed" : "--";
    const targetText = target === true ? "open" : target === false ? "closed" : "any";
    const matched = target === null || target === undefined || actual === target;
    return `
      <div class="finger-row">
        <strong>${label}</strong>
        <span>${actualText} / ${targetText}</span>
        <span class="${matched ? "match" : "mismatch"}">${matched ? "OK" : "NO"}</span>
      </div>
    `;
  }).join("");

  const parts = [];
  if (directions.thumb) parts.push(`thumb ${directions.thumb}`);
  if (directions.index) parts.push(`index ${directions.index}`);
  if (directions.ok_tip_distance !== undefined) {
    parts.push(`ok ${fmt(directions.ok_tip_distance)}`);
  }
  text("poseDirection", parts.length ? parts.join(" | ") : "--");
}

function renderLandmarks(landmarks) {
  const points = Array.isArray(landmarks) ? landmarks : [];
  text("landmarkCount", `${points.length} / 21`);
  const node = $("landmarkList");
  if (!node) return;
  if (!points.length) {
    node.innerHTML = '<div class="empty">No hand points</div>';
    return;
  }
  node.innerHTML = points.map((point) => `
    <div class="landmark-point">
      <strong>#${escapeHtml(point.id)}</strong>
      x ${escapeHtml(point.x)}<br>
      y ${escapeHtml(point.y)}<br>
      z ${escapeHtml(point.z)}
    </div>
  `).join("");
}

function applyStatus(status) {
  currentSource = status.source || "";
  setStateBadge(status.state);
  text("transportBadge", status.transport);
  text("transportText", status.transport);
  text("sourceLine", status.source);
  text("lastCommand", status.last_command);
  text("commandDisplay", status.last_command);
  text(
    "lastCommandMeta",
    `${status.last_command_gesture} / ${fmt(status.last_command_confidence)}`
  );
  text("selectedGesture", status.selected_gesture);
  text("selectedConfidence", fmt(status.selected_confidence));
  text("dynamicState", status.dynamic_state || "idle");
  text("robotCommandCount", status.command_count);
  text("robotLastGesture", status.last_command_gesture);
  text(
    "robotTransferState",
    status.last_command && status.last_command !== "UNKNOWN" ? "SENT" : "WAITING"
  );
  text("fps", fmt(status.fps, "", 1));
  text("latency", fmt(status.latency_ms, " ms", 1));
  text("handCount", status.hand_count);
  text("frameIndex", status.frame_index);
  const minConfidence = status.command_min_confidence ?? 0.65;
  text(
    "staticGesture",
    recognitionLabel(status.static_gesture, status.static_confidence, minConfidence)
  );
  text(
    "dynamicGesture",
    recognitionLabel(status.dynamic_gesture, status.dynamic_confidence, minConfidence)
  );
  text("updatedAt", status.updated_at);
  applyConfirmation(status);
  renderPoseDiagnostics(status);
  renderLandmarks(status.landmarks);
  renderGestureMap(status.selected_gesture);
  if (!settingsDirty) applySettings(settingsPayloadFromStatus(status));
  if (!browserCameraMode) setVideoStarted(status.state === "running");

  const errorPanel = $("errorPanel");
  if (status.error) {
    errorPanel.hidden = false;
    text("errorText", status.error);
  } else {
    errorPanel.hidden = true;
    text("errorText", "");
  }
}

function reloadStream() {
  const stream = $("stream");
  if (stream) stream.src = `${buildUrl("stream.mjpg")}?t=${Date.now()}`;
}

async function refreshStatus() {
  if (browserCameraMode) return;
  try {
    const response = await fetch(buildUrl("api/status"), { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const status = await response.json();
    applyStatus(status);
  } catch (error) {
    showRuntimeError(
      cameraAccessMessage("Backend is unavailable. Start the video backend.")
    );
  }
}

function settingsPayloadFromStatus(status) {
  return {
    static_confirmation_frames: status.static_confirmation_frames,
    dynamic_confirmation_frames: status.dynamic_confirmation_frames,
    emergency_confirmation_frames: status.emergency_confirmation_frames,
  };
}

function applySettings(settings) {
  const pairs = [
    ["staticFramesInput", settings.static_confirmation_frames],
    ["dynamicFramesInput", settings.dynamic_confirmation_frames],
    ["emergencyFramesInput", settings.emergency_confirmation_frames],
  ];
  for (const [id, value] of pairs) {
    const input = $(id);
    if (input && document.activeElement !== input && value !== undefined) {
      input.value = value;
    }
  }
  if (!settingsDirty) text("settingsStatus", "saved");
}

function readSettings() {
  return {
    static_confirmation_frames: Number($("staticFramesInput")?.value || 1),
    dynamic_confirmation_frames: Number($("dynamicFramesInput")?.value || 1),
    emergency_confirmation_frames: Number($("emergencyFramesInput")?.value || 1),
  };
}

async function loadSettings() {
  try {
    const response = await fetch(buildUrl("api/settings"), { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    applySettings(await response.json());
  } catch (error) {
    text("settingsStatus", "offline");
  }
}

async function saveSettings() {
  syncApiBaseFromInput();
  text("settingsStatus", "saving");
  try {
    const response = await fetch(buildUrl("api/settings"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(readSettings()),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    settingsDirty = false;
    applySettings(payload);
    text("settingsStatus", "saved");
  } catch (error) {
    text("settingsStatus", "error");
    showRuntimeError(error.message);
  }
}

async function refreshCommands() {
  const response = await fetch(buildUrl("api/commands"), { cache: "no-store" });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  commandCache = await response.json();
  renderCommands(commandCache);
}

async function startBrowserCamera() {
  syncApiBaseFromInput();
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showRuntimeError(cameraAccessMessage("Browser camera API is unavailable on this page."));
    return;
  }
  try {
    browserStream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: "user",
      },
      audio: false,
    });
    const video = $("browserVideo");
    if (!video) return;
    video.srcObject = browserStream;
    await video.play();
    browserCameraMode = true;
    const stage = document.querySelector(".video-stage");
    if (stage) stage.classList.add("browser-active");
    video.hidden = false;
    setVideoStarted(true);
    setStateBadge("camera");
    text("sourceLine", apiBase ? `browser-camera -> ${apiBase}` : "browser-camera -> same-origin");
    frameTimer = window.setInterval(captureAndSendFrame, BROWSER_CAMERA_FRAME_INTERVAL_MS);
  } catch (error) {
    showRuntimeError(cameraAccessMessage(error));
  }
}

function stopBrowserCamera() {
  browserCameraMode = false;
  if (frameTimer !== null) {
    window.clearInterval(frameTimer);
    frameTimer = null;
  }
  if (browserStream) {
    for (const track of browserStream.getTracks()) track.stop();
    browserStream = null;
  }
  const video = $("browserVideo");
  if (video) {
    video.pause();
    video.srcObject = null;
    video.hidden = true;
  }
  const stage = document.querySelector(".video-stage");
  if (stage) stage.classList.remove("browser-active");
  setVideoStarted(false);
  reloadStream();
}

async function startVideo() {
  syncApiBaseFromInput();
  if (!currentSource || currentSource === "browser-camera" || browserCameraMode) {
    await startBrowserCamera();
    return;
  }
  setVideoStarted(true);
  reloadStream();
  refreshStatus();
  refreshCommands().catch(() => undefined);
}

function stopVideo() {
  if (browserCameraMode) {
    stopBrowserCamera();
    return;
  }
  setVideoStarted(false);
}

async function captureAndSendFrame() {
  if (!browserCameraMode || frameInFlight) return;
  const video = $("browserVideo");
  const canvas = $("captureCanvas");
  if (!video || !canvas || video.readyState < 2 || !video.videoWidth) return;
  frameInFlight = true;
  try {
    const width = 480;
    const height = Math.max(1, Math.round(width * (video.videoHeight / video.videoWidth)));
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d");
    if (!context) throw new Error("Canvas capture context is unavailable.");
    context.drawImage(video, 0, 0, width, height);
    const image = canvas.toDataURL("image/jpeg", 0.72);
    const response = await fetch(buildUrl("api/frame"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image }),
    });
    if (!response.ok) throw new Error(`Frame API HTTP ${response.status}`);
    const payload = await response.json();
    if (payload.status) applyStatus(payload.status);
    if (payload.frame) {
      const stream = $("stream");
      if (stream) stream.src = payload.frame;
    }
    if (payload.commands) {
      commandCache = payload.commands;
      renderCommands(commandCache);
    }
  } catch (error) {
    showRuntimeError(error.message);
  } finally {
    frameInFlight = false;
  }
}

function renderCommands(commands) {
  const body = $("commandRows");
  if (!body) return;
  if (!commands.length) {
    body.innerHTML = '<tr><td colspan="5" class="empty">No commands emitted</td></tr>';
    return;
  }
  body.innerHTML = commands.map((entry) => `
    <tr>
      <td>${escapeHtml(entry.created_at)}</td>
      <td>${escapeHtml(entry.command)}</td>
      <td>${escapeHtml(entry.gesture)}</td>
      <td>${escapeHtml(entry.confidence)}</td>
      <td>${escapeHtml(entry.frame_index)}</td>
    </tr>
  `).join("");
}

function renderGestureMap(activeGesture = "UNKNOWN") {
  const node = $("gestureMap");
  if (!node) return;
  node.innerHTML = GESTURE_COMMANDS.map(([gesture, command]) => `
    <div class="gesture-item ${gesture === activeGesture ? "active" : ""}">
      <strong>${gesture}</strong>
      <span>${command}</span>
    </div>
  `).join("");
}

function showRuntimeError(message) {
  setStateBadge(browserCameraMode ? "camera" : "error");
  const errorPanel = $("errorPanel");
  if (errorPanel) errorPanel.hidden = false;
  text("errorText", message);
}

function cameraAccessMessage(error) {
  const detail = typeof error === "string" ? error : error?.message || "";
  if (error?.name === "NotAllowedError" || error?.name === "PermissionDeniedError") {
    return [
      "Camera/video access is required to control the robot.",
      "Grant video permission and try again.",
    ].join(" ");
  }
  return detail
    ? `Camera/video access is required to control the robot. ${detail}`
    : "Camera/video access is required to control the robot.";
}

function bindControls() {
  const apiInput = $("apiBaseInput");
  if (apiInput) {
    apiInput.value = apiBase || window.localStorage.getItem("gestureApiBase") || "";
    apiBase = apiInput.value.trim();
    apiInput.addEventListener("change", syncApiBaseFromInput);
  }
  const startVideoButton = $("startVideoButton");
  if (startVideoButton) startVideoButton.addEventListener("click", startVideo);
  const sideStartButton = $("sideStartButton");
  if (sideStartButton) sideStartButton.addEventListener("click", startVideo);
  const stopVideoButton = $("stopVideoButton");
  if (stopVideoButton) stopVideoButton.addEventListener("click", stopVideo);
  const reloadButton = $("reloadButton");
  if (reloadButton) reloadButton.addEventListener("click", reloadStream);
  for (const id of ["staticFramesInput", "dynamicFramesInput", "emergencyFramesInput"]) {
    const input = $(id);
    if (input) {
      input.addEventListener("input", () => {
        settingsDirty = true;
        text("settingsStatus", "unsaved");
      });
    }
  }
  const saveSettingsButton = $("saveSettingsButton");
  if (saveSettingsButton) saveSettingsButton.addEventListener("click", saveSettings);
}

bindControls();
renderGestureMap();
loadSettings();
refreshStatus();
refreshCommands().catch(() => undefined);
setInterval(refreshStatus, 500);
setInterval(() => {
  if (!browserCameraMode) refreshCommands().catch(() => undefined);
}, 1000);
"""


if __name__ == "__main__":
    raise SystemExit(main())
