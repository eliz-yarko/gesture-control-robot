"""Split a manifest into stratified train/control/holdout subsets."""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SplitSummary:
    """Counts produced by one manifest split."""

    train_count: int
    control_count: int
    holdout_count: int
    train_counts: dict[str, int]
    control_counts: dict[str, int]
    holdout_counts: dict[str, int]


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Create group-safe stratified train/control/holdout manifests. This is intended "
            "for independent source samples, not derived windows."
        )
    )
    parser.add_argument("--input", required=True, help="Input manifest CSV.")
    parser.add_argument("--train-output", required=True, help="Output train manifest CSV.")
    parser.add_argument("--control-output", required=True, help="Output control manifest CSV.")
    parser.add_argument("--holdout-output", required=True, help="Output holdout manifest CSV.")
    parser.add_argument(
        "--control-count",
        type=int,
        default=150,
        help="Target number of control samples, stratified by expected_gesture.",
    )
    parser.add_argument(
        "--holdout-count",
        type=int,
        default=70,
        help="Target number of holdout samples, stratified by expected_gesture.",
    )
    parser.add_argument(
        "--group-column",
        default="sample_id",
        help="Column used to keep related rows in the same split.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic shuffle seed.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Split a manifest and write the three output CSV files."""

    args = build_parser().parse_args(argv)
    _validate_args(args)
    rows, fieldnames = _read_rows(Path(args.input))
    train_rows, control_rows, holdout_rows, summary = split_manifest_rows(
        rows,
        control_count=args.control_count,
        holdout_count=args.holdout_count,
        group_column=args.group_column,
        seed=args.seed,
    )
    _write_rows(Path(args.train_output), fieldnames, train_rows)
    _write_rows(Path(args.control_output), fieldnames, control_rows)
    _write_rows(Path(args.holdout_output), fieldnames, holdout_rows)

    print(f"Wrote train manifest: {args.train_output}")
    print(f"Wrote control manifest: {args.control_output}")
    print(f"Wrote holdout manifest: {args.holdout_output}")
    print(
        "Counts: "
        f"train={summary.train_count}, "
        f"control={summary.control_count}, "
        f"holdout={summary.holdout_count}"
    )
    print(f"Train by class: {_format_counts(summary.train_counts)}")
    print(f"Control by class: {_format_counts(summary.control_counts)}")
    print(f"Holdout by class: {_format_counts(summary.holdout_counts)}")
    return 0


def split_manifest_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    control_count: int,
    holdout_count: int,
    group_column: str = "sample_id",
    seed: int = 42,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], SplitSummary]:
    """Return train/control/holdout rows with group-safe stratification."""

    if control_count < 0 or holdout_count < 0:
        raise ValueError("Split target counts must be non-negative")
    if not rows:
        return [], [], [], SplitSummary(0, 0, 0, {}, {}, {})

    groups_by_label = _groups_by_label(rows, group_column=group_column)
    label_counts = {label: len(groups) for label, groups in groups_by_label.items()}
    total_eval_target = control_count + holdout_count
    eval_targets = _allocate_counts(label_counts, total_eval_target)
    effective_eval_count = sum(eval_targets.values())
    effective_control_count = _effective_control_count(
        effective_eval_count,
        control_count=control_count,
        holdout_count=holdout_count,
    )
    control_targets = _allocate_counts(eval_targets, effective_control_count)
    holdout_targets = {
        label: eval_targets[label] - control_targets.get(label, 0) for label in eval_targets
    }

    rng = random.Random(seed)
    train_groups: set[str] = set()
    control_groups: set[str] = set()
    holdout_groups: set[str] = set()

    for label, groups in sorted(groups_by_label.items()):
        shuffled_groups = list(groups)
        rng.shuffle(shuffled_groups)
        control_take = control_targets.get(label, 0)
        holdout_take = holdout_targets.get(label, 0)

        control_selection = shuffled_groups[:control_take]
        holdout_selection = shuffled_groups[control_take : control_take + holdout_take]
        train_selection = shuffled_groups[control_take + holdout_take :]

        control_groups.update(control_selection)
        holdout_groups.update(holdout_selection)
        train_groups.update(train_selection)

    train_rows: list[dict[str, str]] = []
    control_rows: list[dict[str, str]] = []
    holdout_rows: list[dict[str, str]] = []
    for row in rows:
        group = _required(row, group_column)
        row_copy = dict(row)
        if group in control_groups:
            control_rows.append(row_copy)
        elif group in holdout_groups:
            holdout_rows.append(row_copy)
        elif group in train_groups:
            train_rows.append(row_copy)
        else:
            raise ValueError(f"Row group was not assigned to a split: {group}")

    summary = SplitSummary(
        train_count=len(train_rows),
        control_count=len(control_rows),
        holdout_count=len(holdout_rows),
        train_counts=_label_counts(train_rows),
        control_counts=_label_counts(control_rows),
        holdout_counts=_label_counts(holdout_rows),
    )
    return train_rows, control_rows, holdout_rows, summary


def _groups_by_label(
    rows: Sequence[Mapping[str, str]],
    *,
    group_column: str,
) -> dict[str, list[str]]:
    rows_by_group: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_group[_required(row, group_column)].append(row)

    groups_by_label: dict[str, list[str]] = defaultdict(list)
    for group, group_rows in rows_by_group.items():
        labels = {_required(row, "expected_gesture") for row in group_rows}
        if len(labels) != 1:
            labels_text = ", ".join(sorted(labels))
            raise ValueError(f"Group {group!r} contains multiple labels: {labels_text}")
        label = next(iter(labels))
        groups_by_label[label].append(group)
    return {label: sorted(groups) for label, groups in groups_by_label.items()}


def _allocate_counts(label_counts: Mapping[str, int], target_count: int) -> dict[str, int]:
    available_total = sum(max(count, 0) for count in label_counts.values())
    if target_count <= 0 or available_total <= 0:
        return {label: 0 for label in label_counts}

    target_count = min(target_count, available_total)
    allocations: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for label, available in sorted(label_counts.items()):
        if available <= 0:
            allocations[label] = 0
            continue
        exact = target_count * available / available_total
        base = min(available, int(exact))
        allocations[label] = base
        remainders.append((exact - base, label))

    remaining = target_count - sum(allocations.values())
    for _remainder, label in sorted(remainders, reverse=True):
        if remaining <= 0:
            break
        if allocations[label] >= label_counts[label]:
            continue
        allocations[label] += 1
        remaining -= 1

    return allocations


def _effective_control_count(
    effective_eval_count: int,
    *,
    control_count: int,
    holdout_count: int,
) -> int:
    requested_eval_count = control_count + holdout_count
    if effective_eval_count <= 0 or requested_eval_count <= 0:
        return 0
    exact = effective_eval_count * control_count / requested_eval_count
    return min(effective_eval_count, round(exact))


def _label_counts(rows: Sequence[Mapping[str, str]]) -> dict[str, int]:
    return dict(Counter(_required(row, "expected_gesture") for row in rows))


def _read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {path}")
        return [dict(row) for row in reader], list(reader.fieldnames)


def _write_rows(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_counts(counts: Mapping[str, int]) -> str:
    if not counts:
        return "no samples"
    return ", ".join(f"{label}={count}" for label, count in sorted(counts.items()))


def _required(row: Mapping[str, str], column: str) -> str:
    value = row.get(column)
    if value is None or not value.strip():
        raise ValueError(f"Manifest row is missing required column value: {column}")
    return value.strip()


def _validate_args(args: argparse.Namespace) -> None:
    if args.control_count < 0:
        raise ValueError("--control-count must be non-negative")
    if args.holdout_count < 0:
        raise ValueError("--holdout-count must be non-negative")


if __name__ == "__main__":
    raise SystemExit(main())
