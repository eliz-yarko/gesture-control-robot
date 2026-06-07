"""Compute command-level safety metrics from pipeline prediction CSV files."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.domain import GestureID  # noqa: E402
from src.utils.metrics import normalize_label  # noqa: E402

SAFETY_FIELDS = (
    "group",
    "sample_count",
    "confirmed_command_count",
    "confirmed_command_rate",
    "false_confirmed_command_count",
    "false_confirmed_command_rate",
    "critical_false_positive_count",
    "critical_false_positive_rate",
    "false_confirmed_dynamic_count",
    "false_confirmed_dynamic_rate",
    "dynamic_pretrigger_proxy_count",
    "dynamic_pretrigger_proxy_rate",
    "static_dynamic_conflict_proxy_count",
    "static_dynamic_conflict_proxy_rate",
    "command_overwrite_proxy_count",
    "command_overwrite_proxy_rate",
)


@dataclass(frozen=True)
class CommandSafetySample:
    """One evaluated sample with command-confirmation status."""

    sample_id: str
    expected_label: str
    predicted_label: str
    confirmed: bool


@dataclass(frozen=True)
class CommandSafetySummary:
    """Aggregate command-level safety metrics."""

    sample_count: int
    confirmed_command_count: int
    false_confirmed_command_count: int
    critical_false_positive_count: int
    false_confirmed_dynamic_count: int
    dynamic_pretrigger_proxy_count: int
    static_dynamic_conflict_proxy_count: int
    command_overwrite_proxy_count: int

    @property
    def confirmed_command_rate(self) -> float:
        return _safe_divide(self.confirmed_command_count, self.sample_count)

    @property
    def false_confirmed_command_rate(self) -> float:
        return _safe_divide(self.false_confirmed_command_count, self.sample_count)

    @property
    def critical_false_positive_rate(self) -> float:
        return _safe_divide(self.critical_false_positive_count, self.sample_count)

    @property
    def false_confirmed_dynamic_rate(self) -> float:
        return _safe_divide(self.false_confirmed_dynamic_count, self.sample_count)

    @property
    def dynamic_pretrigger_proxy_rate(self) -> float:
        return _safe_divide(self.dynamic_pretrigger_proxy_count, self.sample_count)

    @property
    def static_dynamic_conflict_proxy_rate(self) -> float:
        return _safe_divide(self.static_dynamic_conflict_proxy_count, self.sample_count)

    @property
    def command_overwrite_proxy_rate(self) -> float:
        return _safe_divide(self.command_overwrite_proxy_count, self.sample_count)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Compute command safety metrics from scripts/evaluate_manifest.py "
            "--classifier-mode pipeline predictions."
        )
    )
    parser.add_argument("--input", required=True, help="Pipeline predictions CSV.")
    parser.add_argument("--output", required=True, help="Output command-safety summary CSV.")
    parser.add_argument(
        "--group-by",
        nargs="*",
        default=(),
        help="Optional CSV columns for grouped safety metrics.",
    )
    parser.add_argument(
        "--critical-label",
        default="THUMB_DOWN",
        help="Gesture label treated as critical for false-positive reporting.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Compute and write command-level safety metrics."""

    args = build_parser().parse_args(argv)
    rows = _read_rows(Path(args.input))
    grouped_samples = _group_samples(rows, tuple(args.group_by))
    grouped_samples[("ALL",)] = [_sample_from_row(row, index) for index, row in enumerate(rows, 1)]
    _write_summary(Path(args.output), grouped_samples, critical_label=args.critical_label)
    print(f"Wrote command safety summary: {args.output}")
    return 0


def compute_command_safety_summary(
    samples: Sequence[CommandSafetySample],
    *,
    critical_label: str = "THUMB_DOWN",
) -> CommandSafetySummary:
    """Compute command-confirmation safety metrics."""

    critical = normalize_label(critical_label)
    false_confirmed = 0
    critical_false_positive = 0
    false_dynamic = 0
    dynamic_pretrigger_proxy = 0
    static_dynamic_conflict_proxy = 0
    command_overwrite_proxy = 0
    confirmed_count = 0

    for sample in samples:
        expected = normalize_label(sample.expected_label)
        predicted = normalize_label(sample.predicted_label)
        if not sample.confirmed:
            continue

        confirmed_count += 1
        expected_gesture = _gesture_or_unknown(expected)
        predicted_gesture = _gesture_or_unknown(predicted)
        is_false_confirmed = expected != predicted
        predicted_dynamic = predicted_gesture.is_dynamic
        expected_static = expected_gesture.is_static
        expected_unknown = expected_gesture == GestureID.UNKNOWN

        if is_false_confirmed:
            false_confirmed += 1
        if predicted == critical and expected != critical:
            critical_false_positive += 1
        if predicted_dynamic and is_false_confirmed:
            false_dynamic += 1
        if predicted_dynamic and (expected_static or expected_unknown):
            dynamic_pretrigger_proxy += 1
            static_dynamic_conflict_proxy += 1
        if predicted_dynamic and expected_static:
            command_overwrite_proxy += 1

    return CommandSafetySummary(
        sample_count=len(samples),
        confirmed_command_count=confirmed_count,
        false_confirmed_command_count=false_confirmed,
        critical_false_positive_count=critical_false_positive,
        false_confirmed_dynamic_count=false_dynamic,
        dynamic_pretrigger_proxy_count=dynamic_pretrigger_proxy,
        static_dynamic_conflict_proxy_count=static_dynamic_conflict_proxy,
        command_overwrite_proxy_count=command_overwrite_proxy,
    )


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return [dict(row) for row in reader]


def _group_samples(
    rows: Sequence[dict[str, str]],
    group_columns: Sequence[str],
) -> dict[tuple[str, ...], list[CommandSafetySample]]:
    grouped: dict[tuple[str, ...], list[CommandSafetySample]] = defaultdict(list)
    if not group_columns:
        return grouped
    for index, row in enumerate(rows, 1):
        group_key = tuple((row.get(column) or "UNKNOWN").strip() for column in group_columns)
        grouped[group_key].append(_sample_from_row(row, index))
    return dict(grouped)


def _sample_from_row(row: dict[str, str], index: int) -> CommandSafetySample:
    return CommandSafetySample(
        sample_id=_first_value(row, ("sample_id",), default=str(index)),
        expected_label=normalize_label(
            _first_value(row, ("expected_gesture", "expected_label", "y_true", "label"))
        ),
        predicted_label=normalize_label(
            _first_value(row, ("predicted_gesture", "predicted_label", "y_pred", "prediction"))
        ),
        confirmed=_first_value(row, ("note",), default="").strip() == "confirmed_command",
    )


def _write_summary(
    path: Path,
    grouped_samples: dict[tuple[str, ...], list[CommandSafetySample]],
    *,
    critical_label: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SAFETY_FIELDS)
        writer.writeheader()
        for group_key, samples in grouped_samples.items():
            writer.writerow(
                _summary_row(
                    group_key,
                    compute_command_safety_summary(samples, critical_label=critical_label),
                )
            )


def _summary_row(
    group_key: tuple[str, ...],
    summary: CommandSafetySummary,
) -> dict[str, object]:
    return {
        "group": _group_name(group_key),
        "sample_count": summary.sample_count,
        "confirmed_command_count": summary.confirmed_command_count,
        "confirmed_command_rate": summary.confirmed_command_rate,
        "false_confirmed_command_count": summary.false_confirmed_command_count,
        "false_confirmed_command_rate": summary.false_confirmed_command_rate,
        "critical_false_positive_count": summary.critical_false_positive_count,
        "critical_false_positive_rate": summary.critical_false_positive_rate,
        "false_confirmed_dynamic_count": summary.false_confirmed_dynamic_count,
        "false_confirmed_dynamic_rate": summary.false_confirmed_dynamic_rate,
        "dynamic_pretrigger_proxy_count": summary.dynamic_pretrigger_proxy_count,
        "dynamic_pretrigger_proxy_rate": summary.dynamic_pretrigger_proxy_rate,
        "static_dynamic_conflict_proxy_count": summary.static_dynamic_conflict_proxy_count,
        "static_dynamic_conflict_proxy_rate": summary.static_dynamic_conflict_proxy_rate,
        "command_overwrite_proxy_count": summary.command_overwrite_proxy_count,
        "command_overwrite_proxy_rate": summary.command_overwrite_proxy_rate,
    }


def _first_value(
    row: dict[str, str],
    candidates: Iterable[str],
    default: str | None = None,
) -> str:
    for candidate in candidates:
        value = row.get(candidate)
        if value is not None and value.strip():
            return value.strip()
    if default is not None:
        return default
    raise ValueError(f"CSV row is missing one of required columns: {tuple(candidates)}")


def _gesture_or_unknown(label: str) -> GestureID:
    try:
        return GestureID[normalize_label(label)]
    except KeyError:
        return GestureID.UNKNOWN


def _group_name(group_key: tuple[str, ...]) -> str:
    return " / ".join(group_key)


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
