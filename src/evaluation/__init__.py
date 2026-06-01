"""Evaluation helpers for dataset-based benchmark runs."""

from src.evaluation.manifest import (
    EvaluationSample,
    PredictionRecord,
    infer_media_type,
    read_manifest,
    write_prediction_records,
)
from src.evaluation.manifest_builder import (
    ManifestBuildResult,
    ManifestRow,
    available_dataset_presets,
    build_manifest_from_directory,
    summarize_counts,
)

__all__ = [
    "EvaluationSample",
    "ManifestBuildResult",
    "ManifestRow",
    "PredictionRecord",
    "available_dataset_presets",
    "build_manifest_from_directory",
    "infer_media_type",
    "read_manifest",
    "summarize_counts",
    "write_prediction_records",
]
