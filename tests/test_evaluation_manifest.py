from __future__ import annotations

import csv
from pathlib import Path

from src.evaluation import PredictionRecord, infer_media_type, read_manifest
from src.evaluation.manifest import write_prediction_records


def test_read_manifest_normalizes_labels_and_relative_paths(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    image_path = tmp_path / "open_palm.jpg"
    image_path.write_text("placeholder", encoding="utf-8")
    manifest_path.write_text(
        "\n".join(
            [
                "sample_id,path,expected_gesture,media_type,dataset,condition,distance",
                "s1,open_palm.jpg,0,,hagrid_v2,normal,1m",
            ]
        ),
        encoding="utf-8",
    )

    samples = read_manifest(manifest_path)

    assert len(samples) == 1
    assert samples[0].sample_id == "s1"
    assert samples[0].path == image_path
    assert samples[0].expected_gesture == "OPEN_PALM"
    assert samples[0].media_type == "image"
    assert samples[0].dataset == "hagrid_v2"


def test_infer_media_type_uses_known_extensions() -> None:
    assert infer_media_type(Path("frame.png")) == "image"
    assert infer_media_type(Path("clip.mp4")) == "video"
    assert infer_media_type(Path("landmarks.json")) == "landmarks"


def test_write_prediction_records_matches_benchmark_input_schema(tmp_path: Path) -> None:
    output_path = tmp_path / "predictions.csv"
    write_prediction_records(
        output_path,
        [
            PredictionRecord(
                sample_id="s1",
                expected_gesture="OPEN_PALM",
                predicted_gesture="OPEN_PALM",
                confidence=0.95,
                latency_ms=35.5,
                fps=28.0,
                dataset="hagrid_v2",
                condition="normal",
                distance="1m",
                media_type="image",
                source_path="frame.png",
                frame_count=1,
            )
        ],
    )

    with output_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert rows == [
        {
            "sample_id": "s1",
            "expected_gesture": "OPEN_PALM",
            "predicted_gesture": "OPEN_PALM",
            "confidence": "0.95",
            "latency_ms": "35.5",
            "fps": "28.0",
            "dataset": "hagrid_v2",
            "condition": "normal",
            "distance": "1m",
            "media_type": "image",
            "source_path": "frame.png",
            "frame_count": "1",
            "note": "",
        }
    ]
