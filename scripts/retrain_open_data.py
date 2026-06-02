"""Build an open-data training manifest and retrain gesture models."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_ipn_manifest import build_ipn_rows  # noqa: E402
from scripts.train_gesture_models import main as train_models_main  # noqa: E402
from src.evaluation.manifest_builder import (  # noqa: E402
    ManifestRow,
    build_manifest_from_directory,
    summarize_counts,
    write_manifest,
)


@dataclass(frozen=True)
class OpenDataBuildResult:
    """Summary of a combined open-data manifest build."""

    output_path: Path
    rows: tuple[ManifestRow, ...]
    skipped_items: int
    counts_by_gesture: dict[str, int]


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""

    parser = argparse.ArgumentParser(
        description="Build a combined open-data manifest and retrain gesture models."
    )
    parser.add_argument(
        "--hagrid-dir",
        type=Path,
        default=None,
        help="Local HaGRID/HaGRIDv2 class-named directory.",
    )
    parser.add_argument(
        "--jester-dir",
        type=Path,
        default=None,
        help="Local Jester class-named directory.",
    )
    parser.add_argument(
        "--own-dir",
        type=Path,
        default=None,
        help="Local own-control class-named directory.",
    )
    parser.add_argument(
        "--ipn-root",
        type=Path,
        default=None,
        help="Local IPN Hand root with annotations/ and videos/ subdirectories.",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("data/processed/training/open_data_manifest.csv"),
        help="Combined manifest path.",
    )
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=250,
        help="Maximum samples per target gesture for each dataset source.",
    )
    parser.add_argument(
        "--include-unknown",
        action="store_true",
        help="Include UNKNOWN samples for false-positive control.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Only build the manifest, without training joblib models.",
    )
    parser.add_argument(
        "--static-output",
        default="models/static_gesture_classifier.joblib",
        help="Output path for the retrained static model.",
    )
    parser.add_argument(
        "--dynamic-output",
        default="models/dynamic_gesture_classifier.joblib",
        help="Output path for the retrained dynamic model.",
    )
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--static-min-confidence", type=float, default=0.65)
    parser.add_argument("--dynamic-min-confidence", type=float, default=0.65)
    parser.add_argument("--mirror-frame", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build the manifest and optionally retrain model files."""

    args = build_parser().parse_args(argv)
    result = build_open_data_manifest(
        output_path=args.output_manifest,
        hagrid_dir=args.hagrid_dir,
        jester_dir=args.jester_dir,
        own_dir=args.own_dir,
        ipn_root=args.ipn_root,
        limit_per_class=args.limit_per_class,
        include_unknown=args.include_unknown,
    )
    print(f"Wrote manifest: {result.output_path}")
    print(f"Samples: {len(result.rows)}")
    print(f"Counts: {summarize_counts(result.counts_by_gesture)}")
    print(f"Skipped items: {result.skipped_items}")

    if args.skip_train:
        return 0

    train_args = [
        "--manifest",
        str(result.output_path),
        "--static-output",
        args.static_output,
        "--dynamic-output",
        args.dynamic_output,
        "--frame-stride",
        str(args.frame_stride),
        "--max-frames",
        str(args.max_frames),
        "--static-min-confidence",
        str(args.static_min_confidence),
        "--dynamic-min-confidence",
        str(args.dynamic_min_confidence),
    ]
    if args.mirror_frame:
        train_args.append("--mirror-frame")
    return train_models_main(train_args)


def build_open_data_manifest(
    output_path: Path,
    hagrid_dir: Path | None = None,
    jester_dir: Path | None = None,
    own_dir: Path | None = None,
    ipn_root: Path | None = None,
    limit_per_class: int | None = 250,
    include_unknown: bool = False,
) -> OpenDataBuildResult:
    """Build a single manifest from selected open-data sources."""

    rows: list[ManifestRow] = []
    skipped_items = 0

    for dataset, input_dir in (
        ("hagrid_v2", hagrid_dir),
        ("jester", jester_dir),
        ("own_control", own_dir),
    ):
        if input_dir is None:
            continue
        result = build_manifest_from_directory(
            input_dir=input_dir,
            output_path=_source_manifest_path(output_path, dataset),
            dataset=dataset,
            condition="open_data",
            distance="unknown",
            limit_per_class=limit_per_class,
            include_unknown=include_unknown,
        )
        rows.extend(result.rows)
        skipped_items += result.skipped_files

    if ipn_root is not None:
        ipn_rows, skipped = build_ipn_rows(
            annotations_dir=ipn_root / "annotations",
            videos_dir=ipn_root / "videos",
            split="train",
            labels=("D0X", "G05", "G06", "G10"),
            include_unknown=include_unknown,
            limit_per_class=limit_per_class,
            distance="unknown",
        )
        rows.extend(ipn_rows)
        skipped_items += skipped

    if not rows:
        raise ValueError("No training rows were collected from the selected open-data sources.")

    rows = sorted(rows, key=lambda row: (row.dataset, row.expected_gesture, row.sample_id))
    write_manifest(output_path, rows)
    return OpenDataBuildResult(
        output_path=output_path,
        rows=tuple(rows),
        skipped_items=skipped_items,
        counts_by_gesture=dict(Counter(row.expected_gesture for row in rows)),
    )


def _source_manifest_path(output_path: Path, dataset: str) -> Path:
    return output_path.with_name(f"{output_path.stem}_{dataset}.csv")


if __name__ == "__main__":
    raise SystemExit(main())
