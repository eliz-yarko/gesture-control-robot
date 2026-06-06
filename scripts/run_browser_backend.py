"""Run the browser-camera backend using deployment-friendly environment variables."""

from __future__ import annotations

import os

from src.web_ui import main


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _append_env_option(args: list[str], option: str, name: str) -> None:
    value = os.getenv(name)
    if value not in (None, ""):
        args.extend([option, value])


if __name__ == "__main__":
    cli_args = [
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
        "--cors-origin",
        _env("CORS_ORIGIN", "*"),
    ]
    _append_env_option(cli_args, "--static-model", "STATIC_MODEL")
    _append_env_option(cli_args, "--dynamic-model", "DYNAMIC_MODEL")
    _append_env_option(cli_args, "--static-threshold-profile", "STATIC_THRESHOLD_PROFILE")
    _append_env_option(cli_args, "--dynamic-threshold-profile", "DYNAMIC_THRESHOLD_PROFILE")
    raise SystemExit(main(cli_args))
