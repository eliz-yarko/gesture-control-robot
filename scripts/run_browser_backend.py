"""Run the browser-camera backend using deployment-friendly environment variables."""

from __future__ import annotations

import os

from src.web_ui import main


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


if __name__ == "__main__":
    raise SystemExit(
        main(
            [
                "--input-mode",
                "browser-camera",
                "--host",
                _env("HOST", "0.0.0.0"),
                "--port",
                _env("PORT", "8000"),
                "--frame-width",
                _env("FRAME_WIDTH", "640"),
                "--frame-height",
                _env("FRAME_HEIGHT", "480"),
                "--static-model",
                _env("STATIC_MODEL", "models/static_gesture_classifier.joblib"),
                "--dynamic-model",
                _env("DYNAMIC_MODEL", "models/dynamic_gesture_classifier.joblib"),
                "--cors-origin",
                _env("CORS_ORIGIN", "*"),
            ]
        )
    )
