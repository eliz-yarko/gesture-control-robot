"""Create a train-only manifest with duplicated rows for selected domains/classes."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.metrics import normalize_label  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build CLI arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Duplicate selected manifest rows for train-only sample reweighting. "
            "Holdout/control manifests should not be passed to this tool."
        )
    )
    parser.add_argument("--input", required=True, help="Input training manifest CSV.")
    parser.add_argument("--output", required=True, help="Output reweighted manifest CSV.")
    parser.add_argument(
        "--copies",
        type=int,
        required=True,
        help="Additional copies to append for each matched row.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset value to duplicate. Can be provided multiple times.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="Expected gesture label to duplicate. Can be provided multiple times.",
    )
    parser.add_argument(
        "--condition",
        action="append",
        default=[],
        help="Optional condition value to duplicate. Can be provided multiple times.",
    )
    parser.add_argument(
        "--suffix-template",
        default="rw{copy:02d}",
        help="Suffix appended to sample_id for copied rows.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run manifest reweighting from CLI args."""

    args = build_parser().parse_args(argv)
    if args.copies < 0:
        raise ValueError("--copies must be non-negative.")

    input_path = Path(args.input)
    output_path = Path(args.output)
    rows, fieldnames = _read_rows(input_path)
    reweighted = reweight_rows(
        rows,
        input_dir=input_path.parent,
        output_dir=output_path.parent,
        copies=args.copies,
        datasets=tuple(args.dataset),
        labels=tuple(args.label),
        conditions=tuple(args.condition),
        suffix_template=args.suffix_template,
    )
    _write_rows(output_path, fieldnames, reweighted)

    source_counts = Counter(_normalized_label(row) for row in rows)
    output_counts = Counter(_normalized_label(row) for row in reweighted)
    matched = len(reweighted) - len(rows)
    print(f"Wrote reweighted manifest: {output_path}")
    print(f"Rows: {len(rows)} -> {len(reweighted)} (+{matched})")
    print(f"Input counts: {_format_counts(source_counts)}")
    print(f"Output counts: {_format_counts(output_counts)}")
    return 0


def reweight_rows(
    rows: Sequence[dict[str, str]],
    *,
    input_dir: Path,
    output_dir: Path,
    copies: int,
    datasets: Sequence[str] = (),
    labels: Sequence[str] = (),
    conditions: Sequence[str] = (),
    suffix_template: str = "rw{copy:02d}",
) -> list[dict[str, str]]:
    """Return rows with additional matched copies appended."""

    dataset_filter = {value.strip() for value in datasets if value.strip()}
    label_filter = {normalize_label(value) for value in labels if value.strip()}
    condition_filter = {value.strip() for value in conditions if value.strip()}

    normalized_rows = [
        _row_for_output(row, input_dir=input_dir, output_dir=output_dir) for row in rows
    ]
    copied_rows: list[dict[str, str]] = []
    for row in normalized_rows:
        if not _matches(row, dataset_filter, label_filter, condition_filter):
            continue
        for copy_index in range(1, copies + 1):
            copied = dict(row)
            suffix = suffix_template.format(copy=copy_index)
            copied["sample_id"] = f"{row.get('sample_id', '').strip()}_{suffix}"
            copied_rows.append(copied)

    return [*normalized_rows, *copied_rows]


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
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _row_for_output(
    row: dict[str, str],
    *,
    input_dir: Path,
    output_dir: Path,
) -> dict[str, str]:
    normalized = dict(row)
    normalized["expected_gesture"] = _normalized_label(row)
    if "path" in normalized and normalized["path"].strip():
        normalized["path"] = _relative_or_absolute_path(
            _resolve_path(normalized["path"], input_dir),
            output_dir,
        )
    return normalized


def _resolve_path(raw_path: str, input_dir: Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else input_dir / path


def _relative_or_absolute_path(path: Path, output_dir: Path) -> str:
    resolved_path = path.resolve()
    resolved_output = output_dir.resolve()
    try:
        return resolved_path.relative_to(resolved_output).as_posix()
    except ValueError:
        return str(resolved_path)


def _matches(
    row: dict[str, str],
    datasets: set[str],
    labels: set[str],
    conditions: set[str],
) -> bool:
    if datasets and row.get("dataset", "").strip() not in datasets:
        return False
    if labels and _normalized_label(row) not in labels:
        return False
    if conditions and row.get("condition", "").strip() not in conditions:
        return False
    return True


def _normalized_label(row: dict[str, str]) -> str:
    return normalize_label(row.get("expected_gesture", ""))


def _format_counts(counts: Counter[str]) -> str:
    return ", ".join(f"{label}={count}" for label, count in sorted(counts.items()))


if __name__ == "__main__":
    raise SystemExit(main())
