"""Export the dashboard frontend as static files for free hosting."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.web_ui import APP_JS, INDEX_HTML, STYLES_CSS  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""

    parser = argparse.ArgumentParser(description="Export the web UI frontend.")
    parser.add_argument(
        "--output",
        default="docs/demo",
        help="Directory where static frontend files will be written.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Write static dashboard files."""

    args = build_parser().parse_args(argv)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "index.html").write_text(INDEX_HTML, encoding="utf-8")
    (output_dir / "styles.css").write_text(STYLES_CSS, encoding="utf-8")
    (output_dir / "app.js").write_text(APP_JS, encoding="utf-8")
    (output_dir / ".nojekyll").write_text("", encoding="utf-8")
    print(f"Wrote static frontend to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
