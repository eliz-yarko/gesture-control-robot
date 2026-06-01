from __future__ import annotations

from src.domain import GestureID
from src.utils.metrics import (
    ClassificationSample,
    compute_benchmark_summary,
    compute_confusion_matrix,
    compute_per_class_metrics,
    normalize_label,
)


def test_normalize_label_accepts_ids_enum_and_strings() -> None:
    assert normalize_label(GestureID.OPEN_PALM) == "OPEN_PALM"
    assert normalize_label("GestureID.THUMB_DOWN") == "THUMB_DOWN"
    assert normalize_label("0") == "OPEN_PALM"
    assert normalize_label("thumb up") == "THUMB_UP"


def test_compute_confusion_matrix_counts_expected_vs_predicted() -> None:
    samples = [
        ClassificationSample("1", "OPEN_PALM", "OPEN_PALM"),
        ClassificationSample("2", "OPEN_PALM", "FIST"),
        ClassificationSample("3", "FIST", "FIST"),
    ]

    matrix = compute_confusion_matrix(samples)

    assert matrix["OPEN_PALM"]["OPEN_PALM"] == 1
    assert matrix["OPEN_PALM"]["FIST"] == 1
    assert matrix["FIST"]["FIST"] == 1


def test_compute_per_class_metrics_uses_precision_recall_and_f1() -> None:
    samples = [
        ClassificationSample("1", "OPEN_PALM", "OPEN_PALM"),
        ClassificationSample("2", "OPEN_PALM", "FIST"),
        ClassificationSample("3", "FIST", "FIST"),
        ClassificationSample("4", "THUMB_DOWN", "OPEN_PALM"),
    ]

    by_label = {metric.label: metric for metric in compute_per_class_metrics(samples)}

    assert by_label["OPEN_PALM"].support == 2
    assert by_label["OPEN_PALM"].true_positive == 1
    assert by_label["OPEN_PALM"].false_positive == 1
    assert by_label["OPEN_PALM"].false_negative == 1
    assert by_label["OPEN_PALM"].precision == 0.5
    assert by_label["OPEN_PALM"].recall == 0.5
    assert by_label["OPEN_PALM"].f1 == 0.5


def test_compute_benchmark_summary_reports_core_metrics() -> None:
    samples = [
        ClassificationSample("1", "OPEN_PALM", "OPEN_PALM", latency_ms=100.0, fps=24.0),
        ClassificationSample("2", "FIST", "FIST", latency_ms=130.0, fps=22.0),
        ClassificationSample("3", "THUMB_DOWN", "UNKNOWN", latency_ms=160.0, fps=20.0),
        ClassificationSample("4", "OPEN_PALM", "THUMB_DOWN", latency_ms=190.0, fps=18.0),
    ]

    summary = compute_benchmark_summary(samples, critical_label="THUMB_DOWN")

    assert summary.sample_count == 4
    assert summary.accuracy == 0.5
    assert summary.unknown_rate == 0.25
    assert summary.critical_false_positive_rate == 1 / 3
    assert summary.mean_latency_ms == 145.0
    assert summary.p95_latency_ms == 190.0
    assert summary.mean_fps == 21.0
    assert summary.min_fps == 18.0
