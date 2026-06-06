from __future__ import annotations

from pathlib import Path

from scripts.export_landmark_manifest import (
    _frame_payload,
    _landmark_payload,
    _relative_manifest_path,
)
from src.evaluation import EvaluationSample


def test_frame_payload_serializes_landmarks_as_lists() -> None:
    payload = _frame_payload(
        7,
        [(0.1, 0.2, 0.3), (0.4, 0.5)],
    )

    assert payload == {
        "frame_index": 7,
        "landmarks": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.0]],
    }


def test_landmark_payload_preserves_sample_metadata() -> None:
    sample = EvaluationSample(
        sample_id="sample_001",
        path=Path("video.mov"),
        expected_gesture="WAVE_LR",
        media_type="video",
        dataset="own_control",
        condition="normal",
        distance="1m",
    )

    payload = _landmark_payload(sample, [{"frame_index": 1, "landmarks": []}])

    assert payload["sample_id"] == "sample_001"
    assert payload["expected_gesture"] == "WAVE_LR"
    assert payload["frame_count"] == 1
    assert payload["dataset"] == "own_control"


def test_relative_manifest_path_uses_manifest_directory(tmp_path: Path) -> None:
    manifest = tmp_path / "manifests" / "landmarks.csv"
    target = tmp_path / "manifests" / "landmarks" / "sample.json"

    assert _relative_manifest_path(manifest, target) == "landmarks/sample.json"


def test_relative_manifest_path_returns_absolute_for_external_target(tmp_path: Path) -> None:
    manifest = tmp_path / "manifests" / "landmarks.csv"
    target = tmp_path / "cache" / "sample.json"

    assert _relative_manifest_path(manifest, target) == str(target.resolve())
