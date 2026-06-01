"""Build a benchmark manifest from the IPN Hand dataset annotations."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.manifest_builder import (  # noqa: E402
    ManifestRow,
    summarize_counts,
    write_manifest,
)
from src.utils.metrics import normalize_label  # noqa: E402

IPN_LABEL_MAP = {
    "D0X": "UNKNOWN",
    "G05": "WAVE_LR",
    "G06": "WAVE_LR",
    "G10": "PULL_TOWARD",
}


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""

    parser = argparse.ArgumentParser(
        description="Create a segment-level benchmark manifest from IPN Hand annotations."
    )
    parser.add_argument(
        "--annotations",
        default="data/external/ipn_hand/annotations",
        help="Directory with IPN Hand annotation files.",
    )
    parser.add_argument(
        "--videos",
        default="data/external/ipn_hand/videos",
        help="Directory with extracted IPN Hand .avi videos.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/benchmark_inputs/ipn_hand_manifest.csv",
        help="Destination manifest CSV.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "test", "all"),
        default="test",
        help="Annotation split to use.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=("G05", "G06", "G10"),
        help="IPN labels to include. Use D0X with --include-unknown for non-gesture rows.",
    )
    parser.add_argument(
        "--include-unknown",
        action="store_true",
        help="Include labels mapped to UNKNOWN, usually D0X.",
    )
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=30,
        help="Maximum rows per target gesture after mapping to project labels.",
    )
    parser.add_argument(
        "--distance",
        default="unknown",
        help="Camera distance label written to each row.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build an IPN Hand manifest."""

    args = build_parser().parse_args(argv)
    rows, skipped = build_ipn_rows(
        annotations_dir=Path(args.annotations),
        videos_dir=Path(args.videos),
        split=args.split,
        labels=tuple(label.upper() for label in args.labels),
        include_unknown=args.include_unknown,
        limit_per_class=args.limit_per_class,
        distance=args.distance,
    )
    output_path = Path(args.output)
    write_manifest(output_path, rows)
    counts = Counter(row.expected_gesture for row in rows)
    print(f"Wrote manifest: {output_path}")
    print(f"Samples: {len(rows)}")
    print(f"Counts: {summarize_counts(counts)}")
    print(f"Skipped rows: {skipped}")
    return 0


def build_ipn_rows(
    annotations_dir: Path,
    videos_dir: Path,
    split: str,
    labels: Sequence[str],
    include_unknown: bool,
    limit_per_class: int | None,
    distance: str,
) -> tuple[list[ManifestRow], int]:
    """Build manifest rows from IPN annotation CSV-like files."""

    if limit_per_class is not None and limit_per_class <= 0:
        raise ValueError("limit_per_class must be positive when provided")
    annotation_path = _annotation_path(annotations_dir, split)
    metadata = _read_metadata(annotations_dir / "metadata.csv")
    label_filter = set(labels)
    selected_by_gesture: dict[str, list[ManifestRow]] = defaultdict(list)
    skipped = 0

    for row in _read_annotation_rows(annotation_path):
        ipn_label = (row.get("label") or "").strip().upper()
        if label_filter and ipn_label not in label_filter:
            skipped += 1
            continue
        gesture = normalize_label(IPN_LABEL_MAP.get(ipn_label, "UNKNOWN"))
        if gesture == "UNKNOWN" and not include_unknown:
            skipped += 1
            continue
        if limit_per_class is not None and len(selected_by_gesture[gesture]) >= limit_per_class:
            continue

        video_name = _required(row, "video")
        video_path = videos_dir / f"{video_name}.avi"
        if not video_path.exists():
            skipped += 1
            continue

        selected_by_gesture[gesture].append(
            ManifestRow(
                sample_id=_sample_id(video_name, ipn_label, row),
                path=video_path.resolve().as_posix(),
                expected_gesture=gesture,
                media_type="video",
                dataset="ipn_hand",
                condition=_condition(metadata.get(video_name, {})),
                distance=distance,
                start_frame=int(_required(row, "t_start")),
                end_frame=int(_required(row, "t_end")),
            )
        )

    rows: list[ManifestRow] = []
    for gesture in sorted(selected_by_gesture):
        rows.extend(selected_by_gesture[gesture])
    return rows, skipped


def _annotation_path(annotations_dir: Path, split: str) -> Path:
    if split == "train":
        return annotations_dir / "Annot_TrainList.txt"
    if split == "test":
        return annotations_dir / "Annot_TestList.txt"
    return annotations_dir / "Annot_List.txt"


def _read_metadata(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return {
            (row.get("Video Name") or "").strip(): dict(row)
            for row in reader
            if (row.get("Video Name") or "").strip()
        }


def _read_annotation_rows(path: Path) -> list[dict[str, str]]:
    fieldnames = ("video", "label", "id", "t_start", "t_end", "frames")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        first_line = stream.readline()
        stream.seek(0)
        if first_line.lower().startswith("video,label,"):
            return [dict(row) for row in csv.DictReader(stream)]
        reader = csv.DictReader(stream, fieldnames=fieldnames)
        return [dict(row) for row in reader]


def _sample_id(video_name: str, ipn_label: str, row: Mapping[str, str]) -> str:
    start = _required(row, "t_start")
    end = _required(row, "t_end")
    safe_video = video_name.replace("#", "n").replace("/", "_").replace("\\", "_")
    return f"ipn_{safe_video}_{ipn_label}_{start}_{end}"


def _condition(metadata: Mapping[str, str]) -> str:
    if not metadata:
        return "unknown"
    parts = [
        metadata.get("Illumination", ""),
        metadata.get("Background", ""),
        metadata.get("Background Motion", ""),
    ]
    normalized = [part.strip().lower().replace(" ", "_") for part in parts if part.strip()]
    return "+".join(normalized) if normalized else "unknown"


def _required(row: Mapping[str, str], key: str) -> str:
    value = row.get(key)
    if value is None or not value.strip():
        raise ValueError(f"IPN annotation row is missing required field: {key}")
    return value.strip()


if __name__ == "__main__":
    raise SystemExit(main())
