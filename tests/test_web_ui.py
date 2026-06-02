from __future__ import annotations

from src.domain import CommandEvent, GestureID, GesturePrediction, RobotCommand
from src.interpretation.command_mapper import CommandConfirmationState
from src.pipeline import PipelineResult
from src.recognition.hand_detector import HandDetection
from src.web_ui import APP_JS, INDEX_HTML, STYLES_CSS, DashboardState


def test_dashboard_state_reports_runtime_error() -> None:
    state = DashboardState(source="camera:0", transport="mock")

    state.set_state("error", "Cannot open camera")

    payload = state.status_payload()
    assert payload["state"] == "error"
    assert payload["source"] == "camera:0"
    assert payload["error"] == "Cannot open camera"


def test_dashboard_state_publishes_frame_and_command_log() -> None:
    state = DashboardState(source="sample.mp4", transport="mock")
    result = PipelineResult(
        frame_index=12,
        detections=[
            HandDetection(
                landmarks=[(0.0, 0.0, 0.0)] * 21,
                handedness="Right",
                score=0.95,
            )
        ],
        static_prediction=GesturePrediction(GestureID.OPEN_PALM, 0.86),
        dynamic_prediction=GesturePrediction.unknown(),
        selected_prediction=GesturePrediction(GestureID.OPEN_PALM, 0.86),
        command_event=CommandEvent(
            command=RobotCommand.STOP,
            gesture_id=GestureID.OPEN_PALM,
            confidence=0.86,
        ),
        command_state=CommandConfirmationState(
            gesture_id=GestureID.OPEN_PALM,
            command=RobotCommand.STOP,
            confidence=0.86,
            stable_frames=5,
            required_frames=5,
            ready=True,
            blocked_reason="",
        ),
    )

    state.update_frame(result, latency_ms=32.4, fps=24.8, jpeg_bytes=b"jpg")

    status = state.status_payload()
    commands = state.command_payload()
    assert status["state"] == "running"
    assert status["selected_gesture"] == "OPEN_PALM"
    assert status["last_command"] == "STOP"
    assert status["command_candidate_gesture"] == "OPEN_PALM"
    assert status["command_candidate_command"] == "STOP"
    assert status["command_ready"] is True
    assert status["hand_count"] == 1
    assert status["fps"] == 24.8
    assert status["latency_ms"] == 32.4
    assert state.latest_jpeg() == b"jpg"
    assert commands[0]["command"] == "STOP"
    assert commands[0]["gesture"] == "OPEN_PALM"


def test_dashboard_frontend_contains_demo_mode_assets() -> None:
    assert "demoToggle" in INDEX_HTML
    assert "gestureMap" in INDEX_HTML
    assert "candidateCommand" in INDEX_HTML
    assert "confirmationProgressBar" in INDEX_HTML
    assert "DEMO_SEQUENCE" in APP_JS
    assert "GESTURE_COMMANDS" in APP_JS
    assert "applyConfirmation" in APP_JS
    assert "demo-frame" in STYLES_CSS
    assert "confirmation-panel" in STYLES_CSS


def test_dashboard_frontend_contains_browser_camera_assets() -> None:
    assert "apiBaseInput" in INDEX_HTML
    assert "browserCameraToggle" in INDEX_HTML
    assert "browserVideo" in INDEX_HTML
    assert "captureCanvas" in INDEX_HTML
    assert "getUserMedia" in APP_JS
    assert "api/frame" in APP_JS
    assert "browserCameraMode" in APP_JS
    assert "api-input" in STYLES_CSS
    assert "browser-active" in STYLES_CSS
    assert "browser-video" in STYLES_CSS
