"""CSV manifest helpers for dataset evaluation scripts."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from src.utils.metrics import normalize_label

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".webm"}
LANDMARK_EXTENSIONS = {".json"}

PREDICTION_FIELDS = (
    "sample_id",
    "expected_gesture",
    "predicted_gesture",
    "confidence",
    "latency_ms",
    "fps",
    "dataset",
    "condition",
    "distance",
    "media_type",
    "source_path",
    "frame_count",
    "note",
)


@dataclass(frozen=True)
class EvaluationSample:
    """One item from an evaluation manifest."""

    sample_id: str
    path: Path
    expected_gesture: str
    media_type: str
    dataset: str = ""
    condition: str = ""
    distance: str = ""


@dataclass(frozen=True)
class PredictionRecord:
    """Prediction row written for downstream benchmark scripts."""

    sample_id: str
    expected_gesture: str
    predicted_gesture: str
    confidence: float
    latency_ms: float | None
    fps: float | None
    dataset: str
    condition: str
    distance: str
    media_type: str
    source_path: str
    frame_count: int
    note: str = ""


def read_manifest(path: Path) -> list[EvaluationSample]:
    """Read an evaluation manifest CSV.

    Required columns:
        ``path`` and ``expected_gesture``.

    Optional columns:
        ``sample_id``, ``media_type``, ``dataset``, ``condition``, and ``distance``.
    """

    base_dir = path.parent
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {path}")
        return [_sample_from_row(row, base_dir, index) for index, row in enumerate(reader, 1)]


def write_prediction_records(path: Path, records: Sequence[PredictionRecord]) -> None:
    """Write prediction records in the format consumed by ``scripts/benchmark.py``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(_record_to_row(record))


def infer_media_type(path: Path) -> str:
    """Infer manifest media type from a file extension."""

    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    if suffix in LANDMARK_EXTENSIONS:
        return "landmarks"
    raise ValueError(f"Cannot infer media_type from extension: {path}")


def _sample_from_row(
    row: dict[str, str],
    base_dir: Path,
    index: int,
) -> EvaluationSample:
    raw_path = _required(row, "path")
    sample_path = Path(raw_path)
    if not sample_path.is_absolute():
        sample_path = base_dir / sample_path
    expected_gesture = normalize_label(_required(row, "expected_gesture"))
    media_type = (row.get("media_type") or "").strip().lower() or infer_media_type(sample_path)

    return EvaluationSample(
        sample_id=(row.get("sample_id") or str(index)).strip(),
        path=sample_path,
        expected_gesture=expected_gesture,
        media_type=media_type,
        dataset=(row.get("dataset") or "").strip(),
        condition=(row.get("condition") or "").strip(),
        distance=(row.get("distance") or "").strip(),
    )


def _required(row: dict[str, str], column: str) -> str:
    value = row.get(column)
    if value is None or not value.strip():
        raise ValueError(f"Manifest row is missing required column: {column}")
    return value.strip()


def _record_to_row(record: PredictionRecord) -> dict[str, object]:
    return {
        "sample_id": record.sample_id,
        "expected_gesture": normalize_label(record.expected_gesture),
        "predicted_gesture": normalize_label(record.predicted_gesture),
        "confidence": record.confidence,
        "latency_ms": _empty_if_none(record.latency_ms),
        "fps": _empty_if_none(record.fps),
        "dataset": record.dataset,
        "condition": record.condition,
        "distance": record.distance,
        "media_type": record.media_type,
        "source_path": record.source_path,
        "frame_count": record.frame_count,
        "note": record.note,
    }


def _empty_if_none(value: float | None) -> float | str:
    return "" if value is None else value
