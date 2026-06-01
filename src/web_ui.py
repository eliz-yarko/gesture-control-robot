"""Local web dashboard for the gesture control subsystem."""

from __future__ import annotations

import argparse
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
from time import perf_counter
from typing import Any, cast
from urllib.parse import urlparse

from src.capture.video_capture import VideoCapture
from src.config import AppConfig, VideoConfig
from src.domain import CommandEvent, GestureID, RobotCommand
from src.pipeline import GestureControlPipeline, PipelineResult
from src.transmission.base_sender import CommandSender
from src.transmission.mock_sender import MockCommandSender
from src.transmission.serial_sender import SerialCommandSender

LOGGER = logging.getLogger(__name__)

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
    selected_gesture: str
    selected_confidence: float
    last_command: str
    last_command_gesture: str
    last_command_confidence: float
    command_count: int
    error: str
    updated_at: str
    uptime_seconds: float


class DashboardState:
    """Shared state between the capture worker and HTTP handlers."""

    def __init__(self, source: str, transport: str, command_limit: int = 40) -> None:
        """Initialize empty dashboard state."""

        self._source = source
        self._transport = transport
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
            selected_gesture=GestureID.UNKNOWN.name,
            selected_confidence=0.0,
            last_command=RobotCommand.UNKNOWN.value,
            last_command_gesture=GestureID.UNKNOWN.name,
            last_command_confidence=0.0,
            command_count=0,
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
                selected_gesture=current.selected_gesture,
                selected_confidence=current.selected_confidence,
                last_command=current.last_command,
                last_command_gesture=current.last_command_gesture,
                last_command_confidence=current.last_command_confidence,
                command_count=current.command_count,
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
                selected_gesture=result.selected_prediction.gesture_id.name,
                selected_confidence=result.selected_prediction.confidence,
                last_command=last_command,
                last_command_gesture=last_command_gesture,
                last_command_confidence=last_command_confidence,
                command_count=len(self._commands),
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
            "selected_gesture": snapshot.selected_gesture,
            "selected_confidence": _rounded(snapshot.selected_confidence),
            "last_command": snapshot.last_command,
            "last_command_gesture": snapshot.last_command_gesture,
            "last_command_confidence": _rounded(snapshot.last_command_confidence),
            "command_count": snapshot.command_count,
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

    def _uptime_seconds(self) -> float:
        return perf_counter() - self._started_at


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
    ) -> None:
        """Initialize the worker without opening hardware resources."""

        self._state = state
        self._config = config
        self._video_path = video_path
        self._sender = sender
        self._stop_event = stop_event
        self._loop_video = loop_video
        self._jpeg_quality = jpeg_quality
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
            pipeline = GestureControlPipeline(config=self._config, command_sender=self._sender)
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
    ) -> None:
        """Initialize the local dashboard server."""

        super().__init__(server_address, GestureDashboardHandler)
        self.state = state
        self.stop_event = stop_event


class GestureDashboardHandler(BaseHTTPRequestHandler):
    """Serve dashboard HTML, JSON status, and MJPEG frames."""

    server_version = "GestureDashboard/1.0"

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
        elif path == "/api/commands":
            self._send_json(self._dashboard_server().state.command_payload())
        elif path == "/stream.mjpg":
            self._send_stream()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, format_value: str, *args: object) -> None:
        """Route HTTP logs through the project logger."""

        LOGGER.debug("HTTP %s", format_value % args)

    def _dashboard_server(self) -> GestureDashboardServer:
        return cast(GestureDashboardServer, self.server)

    def _send_text(self, body: str, content_type: str) -> None:
        payload = body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, payload: object) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_stream(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
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


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""

    parser = argparse.ArgumentParser(description="Run the local gesture control web UI.")
    parser.add_argument("--host", default="127.0.0.1", help="Dashboard host.")
    parser.add_argument("--port", type=int, default=8000, help="Dashboard port.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index for live capture.")
    parser.add_argument("--video", type=str, default=None, help="Path to a video file.")
    parser.add_argument("--frame-width", type=int, default=640, help="Capture frame width.")
    parser.add_argument("--frame-height", type=int, default=480, help="Capture frame height.")
    parser.add_argument("--target-fps", type=int, default=30, help="Requested camera FPS.")
    parser.add_argument("--no-mirror", action="store_true", help="Disable mirrored preview.")
    parser.add_argument("--loop-video", action="store_true", help="Loop video files.")
    parser.add_argument("--jpeg-quality", type=int, default=82, help="MJPEG JPEG quality.")
    parser.add_argument(
        "--sender",
        choices=("mock", "serial"),
        default="mock",
        help="Command transport backend.",
    )
    parser.add_argument("--open-browser", action="store_true", help="Open the dashboard URL.")
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
    source = args.video if args.video is not None else f"camera:{args.camera}"
    state = DashboardState(source=source, transport=args.sender)
    stop_event = threading.Event()
    worker = CaptureWorker(
        state=state,
        config=config,
        video_path=args.video,
        sender=sender,
        stop_event=stop_event,
        loop_video=args.loop_video,
        jpeg_quality=args.jpeg_quality,
    )
    server = GestureDashboardServer((args.host, args.port), state, stop_event)
    url = f"http://{args.host}:{server.server_port}"
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
        worker.join(timeout=3.0)
    return 0


def _build_sender(sender_name: str, config: AppConfig) -> CommandSender:
    if sender_name == "serial":
        sender = SerialCommandSender(config.sender)
        sender.open()
        return sender
    return MockCommandSender()


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
    for point in points:
        cv2.circle(frame, point, 4, (21, 130, 199), -1, cv2.LINE_AA)


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


def _timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Gesture Control Dashboard</title>
  <link rel="stylesheet" href="/styles.css">
</head>
<body>
  <main class="app-shell">
    <header class="topbar">
      <div>
        <h1>Gesture Control</h1>
        <p id="sourceLine">camera:0</p>
      </div>
      <div class="state-strip">
        <span id="stateBadge" class="badge badge-muted">starting</span>
        <span id="transportBadge" class="badge">mock</span>
      </div>
    </header>

    <section class="workspace">
      <div class="video-stage">
        <img id="stream" src="/stream.mjpg" alt="Live gesture recognition stream">
      </div>

      <aside class="status-rail">
        <section class="panel command-panel">
          <span class="eyebrow">confirmed command</span>
          <strong id="lastCommand">UNKNOWN</strong>
          <span id="lastCommandMeta">UNKNOWN / 0.00</span>
        </section>

        <section class="metrics-grid">
          <div class="metric">
            <span>Gesture</span><strong id="selectedGesture">UNKNOWN</strong>
          </div>
          <div class="metric">
            <span>Confidence</span><strong id="selectedConfidence">0.00</strong>
          </div>
          <div class="metric"><span>FPS</span><strong id="fps">0.0</strong></div>
          <div class="metric"><span>Latency</span><strong id="latency">0 ms</strong></div>
          <div class="metric"><span>Hands</span><strong id="handCount">0</strong></div>
          <div class="metric"><span>Frame</span><strong id="frameIndex">0</strong></div>
        </section>

        <section class="panel split-panel">
          <div><span>Static</span><strong id="staticGesture">UNKNOWN</strong></div>
          <div><span>Dynamic</span><strong id="dynamicGesture">UNKNOWN</strong></div>
        </section>

        <section class="panel error-panel" id="errorPanel" hidden>
          <span class="eyebrow">runtime</span>
          <strong id="errorText"></strong>
        </section>
      </aside>
    </section>

    <section class="log-band">
      <div class="log-header">
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
  </main>
  <script src="/app.js"></script>
</body>
</html>
"""


STYLES_CSS = """
:root {
  color-scheme: light;
  --page: #f4f6f8;
  --ink: #182026;
  --muted: #61707d;
  --line: #d9e0e6;
  --panel: #ffffff;
  --graphite: #20262d;
  --teal: #137a63;
  --blue: #1f6fb2;
  --amber: #ad6b00;
  --red: #b33a3a;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  min-height: 100vh;
  background: var(--page);
  color: var(--ink);
  font-family: Inter, Segoe UI, Arial, sans-serif;
}

.app-shell {
  width: min(1440px, 100%);
  margin: 0 auto;
  padding: 18px;
}

.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 18px;
  padding: 10px 2px 18px;
}

h1, h2, p {
  margin: 0;
}

h1 {
  font-size: 28px;
  font-weight: 750;
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
  grid-template-columns: minmax(0, 1fr) 360px;
  gap: 18px;
  align-items: stretch;
}

.video-stage {
  min-height: 360px;
  background: #11161b;
  border: 1px solid #12171c;
  border-radius: 8px;
  overflow: hidden;
  display: grid;
  place-items: center;
}

.video-stage img {
  width: 100%;
  height: 100%;
  min-height: 360px;
  aspect-ratio: 4 / 3;
  object-fit: contain;
  display: block;
  background: #11161b;
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
  min-height: 78px;
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
  background: #fbfcfd;
}

.split-panel strong {
  display: block;
  margin-top: 8px;
  font-size: 16px;
  overflow-wrap: anywhere;
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

.log-band {
  margin-top: 18px;
  padding: 16px;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  overflow-x: auto;
}

.log-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
}

h2 {
  font-size: 18px;
}

.log-header span {
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

@media (max-width: 980px) {
  .workspace {
    grid-template-columns: 1fr;
  }

  .status-rail {
    grid-template-columns: 1fr;
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

  h1 {
    font-size: 24px;
  }

  .metrics-grid {
    grid-template-columns: 1fr;
  }

  .video-stage,
  .video-stage img {
    min-height: 260px;
  }
}
"""


APP_JS = """
const $ = (id) => document.getElementById(id);

function text(id, value) {
  const node = $(id);
  if (node) node.textContent = value;
}

function fmt(value, suffix = "") {
  if (value === null || value === undefined) return "--";
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

async function refreshStatus() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    const status = await response.json();
    setStateBadge(status.state);
    text("transportBadge", status.transport);
    text("sourceLine", status.source);
    text("lastCommand", status.last_command);
    text(
      "lastCommandMeta",
      `${status.last_command_gesture} / ${fmt(status.last_command_confidence)}`
    );
    text("selectedGesture", status.selected_gesture);
    text("selectedConfidence", fmt(status.selected_confidence));
    text("fps", fmt(status.fps));
    text("latency", fmt(status.latency_ms, " ms"));
    text("handCount", status.hand_count);
    text("frameIndex", status.frame_index);
    text("staticGesture", `${status.static_gesture} / ${fmt(status.static_confidence)}`);
    text("dynamicGesture", `${status.dynamic_gesture} / ${fmt(status.dynamic_confidence)}`);
    text("updatedAt", status.updated_at);

    const errorPanel = $("errorPanel");
    if (status.error) {
      errorPanel.hidden = false;
      text("errorText", status.error);
    } else {
      errorPanel.hidden = true;
      text("errorText", "");
    }
  } catch (error) {
    setStateBadge("offline");
    const errorPanel = $("errorPanel");
    errorPanel.hidden = false;
    text("errorText", error.message);
  }
}

async function refreshCommands() {
  const response = await fetch("/api/commands", { cache: "no-store" });
  const commands = await response.json();
  const body = $("commandRows");
  if (!body) return;
  if (!commands.length) {
    body.innerHTML = '<tr><td colspan="5" class="empty">No commands emitted</td></tr>';
    return;
  }
  body.innerHTML = commands.map((entry) => `
    <tr>
      <td>${entry.created_at}</td>
      <td>${entry.command}</td>
      <td>${entry.gesture}</td>
      <td>${entry.confidence}</td>
      <td>${entry.frame_index}</td>
    </tr>
  `).join("");
}

refreshStatus();
refreshCommands();
setInterval(refreshStatus, 500);
setInterval(refreshCommands, 1000);
"""


if __name__ == "__main__":
    raise SystemExit(main())
