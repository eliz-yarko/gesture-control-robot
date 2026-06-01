"""Build a benchmark manifest from local dataset files."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.manifest_builder import (  # noqa: E402
    available_dataset_presets,
    build_manifest_from_directory,
    summarize_counts,
)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""

    parser = argparse.ArgumentParser(
        description="Create a benchmark manifest CSV from class-named dataset folders."
    )
    parser.add_argument("--input", required=True, help="Directory with local dataset files.")
    parser.add_argument(
        "--output",
        default="data/processed/benchmark_inputs/manifest.csv",
        help="Destination manifest CSV.",
    )
    parser.add_argument(
        "--dataset",
        choices=available_dataset_presets(),
        default="own_control",
        help="Dataset class-name mapping preset.",
    )
    parser.add_argument(
        "--condition",
        default="normal",
        help="Capture condition label written to each row.",
    )
    parser.add_argument(
        "--distance",
        default="unknown",
        help="Camera distance label written to each row.",
    )
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=None,
        help="Optional max number of samples per target gesture.",
    )
    parser.add_argument(
        "--include-unknown",
        action="store_true",
        help="Include files mapped to UNKNOWN for false-positive analysis.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build a manifest file."""

    args = build_parser().parse_args(argv)
    result = build_manifest_from_directory(
        input_dir=Path(args.input),
        output_path=Path(args.output),
        dataset=args.dataset,
        condition=args.condition,
        distance=args.distance,
        limit_per_class=args.limit_per_class,
        include_unknown=args.include_unknown,
    )
    print(f"Wrote manifest: {result.output_path}")
    print(f"Samples: {len(result.rows)}")
    print(f"Counts: {summarize_counts(result.counts_by_gesture)}")
    print(f"Skipped files: {result.skipped_files}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
