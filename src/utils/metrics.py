"""Benchmark metrics for gesture classification experiments."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil

from src.domain import GestureID


@dataclass(frozen=True)
class ClassificationSample:
    """Single labeled prediction used by benchmark reports."""

    sample_id: str
    expected_label: str
    predicted_label: str
    latency_ms: float | None = None
    fps: float | None = None


@dataclass(frozen=True)
class PerClassMetrics:
    """Precision, recall, and F1 values for one label."""

    label: str
    support: int
    predicted: int
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class BenchmarkSummary:
    """Aggregate benchmark metrics for a set of labeled predictions."""

    sample_count: int
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    unknown_rate: float
    critical_false_positive_rate: float
    mean_latency_ms: float | None
    p95_latency_ms: float | None
    mean_fps: float | None
    min_fps: float | None


ConfusionMatrix = dict[str, dict[str, int]]


def normalize_label(value: str | int | GestureID) -> str:
    """Normalize gesture labels from IDs, enum values, or free-form strings."""

    if isinstance(value, GestureID):
        return value.name

    text = str(value).strip()
    if not text:
        return GestureID.UNKNOWN.name

    if "." in text:
        text = text.rsplit(".", maxsplit=1)[-1]

    try:
        gesture_id = GestureID(int(text))
    except (ValueError, TypeError):
        return text.strip().upper().replace(" ", "_").replace("-", "_")
    return gesture_id.name


def compute_confusion_matrix(samples: Sequence[ClassificationSample]) -> ConfusionMatrix:
    """Build a confusion matrix as ``expected -> predicted -> count``."""

    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for sample in samples:
        expected_label = normalize_label(sample.expected_label)
        predicted_label = normalize_label(sample.predicted_label)
        matrix[expected_label][predicted_label] += 1
    return {expected: dict(predicted) for expected, predicted in matrix.items()}


def compute_per_class_metrics(samples: Sequence[ClassificationSample]) -> list[PerClassMetrics]:
    """Compute per-class precision, recall, and F1 metrics."""

    matrix = compute_confusion_matrix(samples)
    labels = collect_labels(samples)
    metrics: list[PerClassMetrics] = []

    for label in labels:
        true_positive = matrix.get(label, {}).get(label, 0)
        support = sum(matrix.get(label, {}).values())
        predicted = sum(row.get(label, 0) for row in matrix.values())
        false_positive = predicted - true_positive
        false_negative = support - true_positive
        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        f1 = _f1_score(precision, recall)
        metrics.append(
            PerClassMetrics(
                label=label,
                support=support,
                predicted=predicted,
                true_positive=true_positive,
                false_positive=false_positive,
                false_negative=false_negative,
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )

    return metrics


def compute_benchmark_summary(
    samples: Sequence[ClassificationSample],
    critical_label: str = "THUMB_DOWN",
    unknown_label: str = "UNKNOWN",
) -> BenchmarkSummary:
    """Compute aggregate benchmark metrics."""

    normalized_samples = [
        ClassificationSample(
            sample_id=sample.sample_id,
            expected_label=normalize_label(sample.expected_label),
            predicted_label=normalize_label(sample.predicted_label),
            latency_ms=sample.latency_ms,
            fps=sample.fps,
        )
        for sample in samples
    ]
    sample_count = len(normalized_samples)
    if sample_count == 0:
        return BenchmarkSummary(
            sample_count=0,
            accuracy=0.0,
            macro_precision=0.0,
            macro_recall=0.0,
            macro_f1=0.0,
            weighted_f1=0.0,
            unknown_rate=0.0,
            critical_false_positive_rate=0.0,
            mean_latency_ms=None,
            p95_latency_ms=None,
            mean_fps=None,
            min_fps=None,
        )

    correct = sum(
        1 for sample in normalized_samples if sample.expected_label == sample.predicted_label
    )
    per_class = compute_per_class_metrics(normalized_samples)
    unknown = normalize_label(unknown_label)
    target_metrics = [item for item in per_class if item.support > 0 and item.label != unknown]
    total_target_support = sum(item.support for item in target_metrics)
    critical = normalize_label(critical_label)
    non_critical_count = sum(
        1 for sample in normalized_samples if sample.expected_label != critical
    )
    critical_false_positive_count = sum(
        1
        for sample in normalized_samples
        if sample.expected_label != critical and sample.predicted_label == critical
    )
    latencies = [
        sample.latency_ms for sample in normalized_samples if sample.latency_ms is not None
    ]
    fps_values = [sample.fps for sample in normalized_samples if sample.fps is not None]

    return BenchmarkSummary(
        sample_count=sample_count,
        accuracy=correct / sample_count,
        macro_precision=_mean([item.precision for item in target_metrics]),
        macro_recall=_mean([item.recall for item in target_metrics]),
        macro_f1=_mean([item.f1 for item in target_metrics]),
        weighted_f1=_safe_divide(
            sum(item.f1 * item.support for item in target_metrics),
            total_target_support,
        ),
        unknown_rate=_safe_divide(
            sum(1 for sample in normalized_samples if sample.predicted_label == unknown),
            sample_count,
        ),
        critical_false_positive_rate=_safe_divide(
            critical_false_positive_count,
            non_critical_count,
        ),
        mean_latency_ms=_optional_mean(latencies),
        p95_latency_ms=_percentile(latencies, 95),
        mean_fps=_optional_mean(fps_values),
        min_fps=min(fps_values) if fps_values else None,
    )


def collect_labels(samples: Sequence[ClassificationSample]) -> list[str]:
    """Collect labels in stable ``GestureID`` order, followed by unknown custom labels."""

    observed = {normalize_label(sample.expected_label) for sample in samples} | {
        normalize_label(sample.predicted_label) for sample in samples
    }
    ordered = [gesture.name for gesture in sorted(GestureID, key=int) if gesture.name in observed]
    custom = sorted(label for label in observed if label not in set(ordered))
    return ordered + custom


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _f1_score(precision: float, recall: float) -> float:
    return _safe_divide(2 * precision * recall, precision + recall)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _optional_mean(values: Sequence[float]) -> float | None:
    return _mean(values) if values else None


def _percentile(values: Sequence[float], percentile: int) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    index = ceil((percentile / 100) * len(sorted_values)) - 1
    return sorted_values[max(0, min(index, len(sorted_values) - 1))]
