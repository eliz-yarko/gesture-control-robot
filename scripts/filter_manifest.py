"""Filter evaluation manifests by cached metadata such as landmark frame count."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""

    parser = argparse.ArgumentParser(description="Filter a CSV manifest into a clean subset.")
    parser.add_argument("--input", required=True, help="Input manifest CSV.")
    parser.add_argument("--output", required=True, help="Filtered output manifest CSV.")
    parser.add_argument(
        "--min-frame-count",
        type=int,
        default=None,
        help="Keep only rows with frame_count >= N when the column exists.",
    )
    parser.add_argument(
        "--exclude-sample-id",
        action="append",
        default=[],
        help="Sample ID to exclude. Can be provided multiple times.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Filter a manifest and print a short summary."""

    args = build_parser().parse_args(argv)
    if args.min_frame_count is not None and args.min_frame_count < 0:
        raise ValueError("--min-frame-count must be non-negative")

    rows, fieldnames = _read_rows(Path(args.input))
    filtered = filter_rows(
        rows,
        min_frame_count=args.min_frame_count,
        excluded_sample_ids=set(args.exclude_sample_id),
    )
    _write_rows(Path(args.output), fieldnames, filtered)
    print(f"Wrote filtered manifest: {args.output}")
    print(f"Rows: {len(rows)} -> {len(filtered)}")
    return 0


def filter_rows(
    rows: Sequence[dict[str, str]],
    *,
    min_frame_count: int | None = None,
    excluded_sample_ids: set[str] | None = None,
) -> list[dict[str, str]]:
    """Return rows that pass the configured filters."""

    excluded_sample_ids = excluded_sample_ids or set()
    filtered: list[dict[str, str]] = []
    for row in rows:
        if row.get("sample_id", "") in excluded_sample_ids:
            continue
        if min_frame_count is not None and "frame_count" in row:
            frame_count = int(row.get("frame_count") or 0)
            if frame_count < min_frame_count:
                continue
        filtered.append(dict(row))
    return filtered


def _read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {path}")
        return [dict(row) for row in reader], list(reader.fieldnames)


def _write_rows(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
