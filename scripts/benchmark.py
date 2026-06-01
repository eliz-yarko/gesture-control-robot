"""Compute benchmark metrics from labeled gesture prediction CSV files."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.metrics import (  # noqa: E402
    BenchmarkSummary,
    ClassificationSample,
    PerClassMetrics,
    collect_labels,
    compute_benchmark_summary,
    compute_confusion_matrix,
    compute_per_class_metrics,
    normalize_label,
)

SUMMARY_FIELDS = (
    "group",
    "sample_count",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "unknown_rate",
    "critical_false_positive_rate",
    "mean_latency_ms",
    "p95_latency_ms",
    "mean_fps",
    "min_fps",
)
PER_CLASS_FIELDS = (
    "group",
    "label",
    "support",
    "predicted",
    "true_positive",
    "false_positive",
    "false_negative",
    "precision",
    "recall",
    "f1",
)
CONFUSION_FIELDS = ("group", "expected_label", "predicted_label", "count")


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Compute accuracy, F1, latency, FPS, false-positive rate, and confusion matrix "
            "from a CSV with expected and predicted gesture labels."
        )
    )
    parser.add_argument("--input", required=True, help="CSV with labeled predictions.")
    parser.add_argument("--output", required=True, help="Path for aggregate summary CSV.")
    parser.add_argument(
        "--per-class-output",
        help="Path for per-class metrics CSV. Defaults to <output_stem>_per_class.csv.",
    )
    parser.add_argument(
        "--confusion-output",
        help="Path for confusion matrix CSV. Defaults to <output_stem>_confusion_matrix.csv.",
    )
    parser.add_argument(
        "--group-by",
        nargs="*",
        default=(),
        help="Optional CSV columns used for grouped metrics, e.g. dataset condition distance.",
    )
    parser.add_argument(
        "--critical-label",
        default="THUMB_DOWN",
        help="Gesture label used to calculate false-positive rate.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run benchmark metric generation."""

    args = build_parser().parse_args(argv)
    input_path = Path(args.input)
    output_path = Path(args.output)
    per_class_path = (
        Path(args.per_class_output)
        if args.per_class_output
        else output_path.with_name(f"{output_path.stem}_per_class.csv")
    )
    confusion_path = (
        Path(args.confusion_output)
        if args.confusion_output
        else output_path.with_name(f"{output_path.stem}_confusion_matrix.csv")
    )

    rows = _read_rows(input_path)
    grouped_samples = _group_samples(rows, tuple(args.group_by))
    grouped_samples[("ALL",)] = [_sample_from_row(row, index) for index, row in enumerate(rows, 1)]

    _write_summary(output_path, grouped_samples, args.critical_label)
    _write_per_class(per_class_path, grouped_samples)
    _write_confusion(confusion_path, grouped_samples)

    print(f"Wrote benchmark summary: {output_path}")
    print(f"Wrote per-class metrics: {per_class_path}")
    print(f"Wrote confusion matrix: {confusion_path}")
    return 0


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return [dict(row) for row in reader]


def _group_samples(
    rows: Sequence[dict[str, str]],
    group_columns: Sequence[str],
) -> dict[tuple[str, ...], list[ClassificationSample]]:
    grouped: dict[tuple[str, ...], list[ClassificationSample]] = defaultdict(list)
    if not group_columns:
        return grouped

    for index, row in enumerate(rows, 1):
        group_key = tuple((row.get(column) or "UNKNOWN").strip() for column in group_columns)
        grouped[group_key].append(_sample_from_row(row, index))
    return dict(grouped)


def _sample_from_row(row: dict[str, str], index: int) -> ClassificationSample:
    expected = _first_value(row, ("expected_gesture", "expected_label", "y_true", "label"))
    predicted = _first_value(row, ("predicted_gesture", "predicted_label", "y_pred", "prediction"))
    sample_id = _first_value(row, ("sample_id", "frame_id", "video_id"), default=str(index))
    return ClassificationSample(
        sample_id=sample_id,
        expected_label=normalize_label(expected),
        predicted_label=normalize_label(predicted),
        latency_ms=_optional_float(_first_value(row, ("latency_ms",), default="")),
        fps=_optional_float(_first_value(row, ("fps",), default="")),
    )


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


def _optional_float(value: str) -> float | None:
    if not value:
        return None
    return float(value)


def _write_summary(
    path: Path,
    grouped_samples: dict[tuple[str, ...], list[ClassificationSample]],
    critical_label: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for group_key, samples in grouped_samples.items():
            summary = compute_benchmark_summary(samples, critical_label=critical_label)
            writer.writerow(_summary_row(group_key, summary))


def _write_per_class(
    path: Path,
    grouped_samples: dict[tuple[str, ...], list[ClassificationSample]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PER_CLASS_FIELDS)
        writer.writeheader()
        for group_key, samples in grouped_samples.items():
            for metrics in compute_per_class_metrics(samples):
                writer.writerow(_per_class_row(group_key, metrics))


def _write_confusion(
    path: Path,
    grouped_samples: dict[tuple[str, ...], list[ClassificationSample]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CONFUSION_FIELDS)
        writer.writeheader()
        for group_key, samples in grouped_samples.items():
            matrix = compute_confusion_matrix(samples)
            labels = collect_labels(samples)
            for expected_label in labels:
                for predicted_label in labels:
                    writer.writerow(
                        {
                            "group": _group_name(group_key),
                            "expected_label": expected_label,
                            "predicted_label": predicted_label,
                            "count": matrix.get(expected_label, {}).get(predicted_label, 0),
                        }
                    )


def _summary_row(group_key: tuple[str, ...], summary: BenchmarkSummary) -> dict[str, object]:
    return {
        "group": _group_name(group_key),
        "sample_count": summary.sample_count,
        "accuracy": summary.accuracy,
        "macro_precision": summary.macro_precision,
        "macro_recall": summary.macro_recall,
        "macro_f1": summary.macro_f1,
        "weighted_f1": summary.weighted_f1,
        "unknown_rate": summary.unknown_rate,
        "critical_false_positive_rate": summary.critical_false_positive_rate,
        "mean_latency_ms": _empty_if_none(summary.mean_latency_ms),
        "p95_latency_ms": _empty_if_none(summary.p95_latency_ms),
        "mean_fps": _empty_if_none(summary.mean_fps),
        "min_fps": _empty_if_none(summary.min_fps),
    }


def _per_class_row(group_key: tuple[str, ...], metrics: PerClassMetrics) -> dict[str, object]:
    return {
        "group": _group_name(group_key),
        "label": metrics.label,
        "support": metrics.support,
        "predicted": metrics.predicted,
        "true_positive": metrics.true_positive,
        "false_positive": metrics.false_positive,
        "false_negative": metrics.false_negative,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
    }


def _group_name(group_key: tuple[str, ...]) -> str:
    return " / ".join(group_key)


def _empty_if_none(value: float | None) -> float | str:
    return "" if value is None else value


if __name__ == "__main__":
    raise SystemExit(main())
