"""Tune per-class confidence thresholds from benchmark predictions."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.domain import GestureID  # noqa: E402
from src.utils.metrics import (  # noqa: E402
    BenchmarkSummary,
    ClassificationSample,
    compute_benchmark_summary,
    normalize_label,
)


@dataclass(frozen=True)
class ScoredPrediction:
    """Prediction row with confidence retained for threshold simulation."""

    sample_id: str
    expected_label: str
    predicted_label: str
    confidence: float
    latency_ms: float | None = None
    fps: float | None = None


@dataclass(frozen=True)
class TuningResult:
    """Selected thresholds and before/after metrics."""

    min_confidence: float
    class_thresholds: dict[str, float]
    baseline: BenchmarkSummary
    tuned: BenchmarkSummary


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Tune per-class confidence thresholds from predictions.csv. The script "
            "demotes low-confidence predictions to UNKNOWN and chooses thresholds "
            "that improve macro F1 while respecting a critical false-positive limit."
        )
    )
    parser.add_argument("--input", required=True, help="Benchmark predictions CSV.")
    parser.add_argument("--output", required=True, help="Output JSON threshold profile.")
    parser.add_argument("--default-threshold", type=float, default=0.35)
    parser.add_argument("--min-threshold", type=float, default=0.20)
    parser.add_argument("--max-threshold", type=float, default=0.95)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--critical-label", default=GestureID.THUMB_DOWN.name)
    parser.add_argument("--max-critical-fpr", type=float, default=0.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Tune thresholds and write a JSON profile."""

    args = build_parser().parse_args(argv)
    _validate_args(args)
    predictions = read_predictions(Path(args.input))
    result = tune_thresholds(
        predictions,
        default_threshold=args.default_threshold,
        min_threshold=args.min_threshold,
        max_threshold=args.max_threshold,
        step=args.step,
        max_iterations=args.max_iterations,
        critical_label=args.critical_label,
        max_critical_fpr=args.max_critical_fpr,
    )
    write_threshold_profile(Path(args.output), result, source_path=Path(args.input))
    print(f"Wrote threshold profile: {args.output}")
    print(
        "macro_f1 "
        f"{result.baseline.macro_f1:.3f} -> {result.tuned.macro_f1:.3f}, "
        "critical_fpr "
        f"{result.baseline.critical_false_positive_rate:.3f} -> "
        f"{result.tuned.critical_false_positive_rate:.3f}, "
        f"unknown_rate {result.baseline.unknown_rate:.3f} -> {result.tuned.unknown_rate:.3f}"
    )
    return 0


def read_predictions(path: Path) -> list[ScoredPrediction]:
    """Read predictions written by scripts/evaluate_manifest.py."""

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return [_prediction_from_row(row, index) for index, row in enumerate(reader, 1)]


def tune_thresholds(
    predictions: Sequence[ScoredPrediction],
    *,
    default_threshold: float,
    min_threshold: float,
    max_threshold: float,
    step: float,
    max_iterations: int,
    critical_label: str,
    max_critical_fpr: float,
) -> TuningResult:
    """Greedily tune thresholds for labels observed in predictions."""

    labels = sorted(
        {
            prediction.predicted_label
            for prediction in predictions
            if prediction.predicted_label != GestureID.UNKNOWN.name
        }
    )
    thresholds = {label: default_threshold for label in labels}
    grid = _threshold_grid(min_threshold, max_threshold, step)
    baseline = _summary_for_thresholds(
        predictions,
        {label: 0.0 for label in labels},
        default_threshold=0.0,
        critical_label=critical_label,
    )
    best_summary = _summary_for_thresholds(
        predictions,
        thresholds,
        default_threshold=default_threshold,
        critical_label=critical_label,
    )

    for _ in range(max_iterations):
        improved = False
        for label in labels:
            best_label_threshold = thresholds[label]
            best_label_summary = best_summary
            for candidate in grid:
                candidate_thresholds = {**thresholds, label: candidate}
                candidate_summary = _summary_for_thresholds(
                    predictions,
                    candidate_thresholds,
                    default_threshold=default_threshold,
                    critical_label=critical_label,
                )
                if candidate_summary.critical_false_positive_rate > max_critical_fpr:
                    continue
                current_is_allowed = (
                    best_label_summary.critical_false_positive_rate <= max_critical_fpr
                )
                if not current_is_allowed or _score(candidate_summary) > _score(best_label_summary):
                    best_label_threshold = candidate
                    best_label_summary = candidate_summary
            if best_label_threshold != thresholds[label]:
                thresholds[label] = best_label_threshold
                best_summary = best_label_summary
                improved = True
        if not improved:
            break

    return TuningResult(
        min_confidence=default_threshold,
        class_thresholds=thresholds,
        baseline=baseline,
        tuned=best_summary,
    )


def write_threshold_profile(
    path: Path,
    result: TuningResult,
    *,
    source_path: Path,
) -> None:
    """Write the selected thresholds in model_classifier-compatible JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": str(source_path),
        "min_confidence": result.min_confidence,
        "class_thresholds": result.class_thresholds,
        "baseline": asdict(result.baseline),
        "tuned": asdict(result.tuned),
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _prediction_from_row(row: dict[str, str], index: int) -> ScoredPrediction:
    expected = _first_value(row, ("expected_gesture", "expected_label", "y_true", "label"))
    predicted = _first_value(row, ("predicted_gesture", "predicted_label", "y_pred", "prediction"))
    sample_id = _first_value(row, ("sample_id", "frame_id", "video_id"), default=str(index))
    return ScoredPrediction(
        sample_id=sample_id,
        expected_label=normalize_label(expected),
        predicted_label=normalize_label(predicted),
        confidence=_optional_float(_first_value(row, ("confidence",), default="0")) or 0.0,
        latency_ms=_optional_float(_first_value(row, ("latency_ms",), default="")),
        fps=_optional_float(_first_value(row, ("fps",), default="")),
    )


def _summary_for_thresholds(
    predictions: Sequence[ScoredPrediction],
    thresholds: dict[str, float],
    *,
    default_threshold: float,
    critical_label: str,
) -> BenchmarkSummary:
    samples = [
        ClassificationSample(
            sample_id=prediction.sample_id,
            expected_label=prediction.expected_label,
            predicted_label=_thresholded_label(
                prediction.predicted_label,
                prediction.confidence,
                thresholds,
                default_threshold=default_threshold,
            ),
            latency_ms=prediction.latency_ms,
            fps=prediction.fps,
        )
        for prediction in predictions
    ]
    return compute_benchmark_summary(samples, critical_label=critical_label)


def _thresholded_label(
    label: str,
    confidence: float,
    thresholds: dict[str, float],
    *,
    default_threshold: float,
) -> str:
    normalized = normalize_label(label)
    if normalized == GestureID.UNKNOWN.name:
        return normalized
    threshold = thresholds.get(normalized, default_threshold)
    return normalized if confidence >= threshold else GestureID.UNKNOWN.name


def _threshold_grid(min_threshold: float, max_threshold: float, step: float) -> list[float]:
    values: list[float] = []
    current = min_threshold
    while current <= max_threshold + 1e-9:
        values.append(round(current, 4))
        current += step
    return values


def _score(summary: BenchmarkSummary) -> tuple[float, float, float, float]:
    return (
        summary.macro_f1,
        summary.weighted_f1,
        summary.accuracy,
        -summary.unknown_rate,
    )


def _first_value(
    row: dict[str, str],
    candidates: Sequence[str],
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


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("default_threshold", "min_threshold", "max_threshold", "max_critical_fpr"):
        value = float(getattr(args, name))
        if value < 0.0 or value > 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")
    if args.step <= 0.0:
        raise ValueError("--step must be positive")
    if args.min_threshold > args.max_threshold:
        raise ValueError("--min-threshold cannot exceed --max-threshold")
    if args.max_iterations <= 0:
        raise ValueError("--max-iterations must be positive")


if __name__ == "__main__":
    raise SystemExit(main())
