from __future__ import annotations

import csv
from pathlib import Path

from src.evaluation import (
    PredictionRecord,
    build_manifest_from_directory,
    infer_media_type,
    read_manifest,
    summarize_counts,
)
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


def test_read_manifest_supports_video_segments(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    video_path = tmp_path / "clip.avi"
    video_path.write_text("placeholder", encoding="utf-8")
    manifest_path.write_text(
        "\n".join(
            [
                "sample_id,path,expected_gesture,media_type,dataset,condition,distance,"
                "start_frame,end_frame",
                "s1,clip.avi,WAVE_LR,video,ipn_hand,stable,unknown,120,160",
            ]
        ),
        encoding="utf-8",
    )

    samples = read_manifest(manifest_path)

    assert samples[0].path == video_path
    assert samples[0].start_frame == 120
    assert samples[0].end_frame == 160


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


def test_build_manifest_from_directory_maps_hagrid_subset(tmp_path: Path) -> None:
    input_dir = tmp_path / "external" / "hagrid_v2"
    (input_dir / "fist").mkdir(parents=True)
    (input_dir / "peace").mkdir()
    (input_dir / "point").mkdir()
    (input_dir / "fist" / "001.jpg").write_text("placeholder", encoding="utf-8")
    (input_dir / "fist" / "002.jpg").write_text("placeholder", encoding="utf-8")
    (input_dir / "peace" / "001.png").write_text("placeholder", encoding="utf-8")
    (input_dir / "point" / "001.jpg").write_text("placeholder", encoding="utf-8")
    output_path = tmp_path / "processed" / "manifest.csv"

    result = build_manifest_from_directory(
        input_dir=input_dir,
        output_path=output_path,
        dataset="hagrid_v2",
        condition="normal",
        distance="1m",
        limit_per_class=1,
    )

    assert len(result.rows) == 2
    assert result.counts_by_gesture == {"FIST": 1, "PEACE": 1}
    assert result.skipped_files == 1

    samples = read_manifest(output_path)
    assert [sample.expected_gesture for sample in samples] == ["FIST", "PEACE"]
    assert samples[0].dataset == "hagrid_v2"
    assert samples[0].media_type == "image"


def test_build_manifest_can_include_unknown_samples(tmp_path: Path) -> None:
    input_dir = tmp_path / "own_control"
    (input_dir / "unknown").mkdir(parents=True)
    (input_dir / "pull_toward").mkdir()
    (input_dir / "unknown" / "idle.mp4").write_text("placeholder", encoding="utf-8")
    (input_dir / "pull_toward" / "pull.mp4").write_text("placeholder", encoding="utf-8")

    result = build_manifest_from_directory(
        input_dir=input_dir,
        output_path=tmp_path / "manifest.csv",
        dataset="own_control",
        include_unknown=True,
    )

    assert result.counts_by_gesture == {"PULL_TOWARD": 1, "UNKNOWN": 1}
    assert summarize_counts(result.counts_by_gesture) == "PULL_TOWARD=1, UNKNOWN=1"
