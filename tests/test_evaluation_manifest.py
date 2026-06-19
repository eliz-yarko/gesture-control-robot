from __future__ import annotations

import csv
from pathlib import Path

from scripts.evaluate_manifest import SampleEvaluator, _stable_static_prediction, build_parser
from src.domain import GestureID, GesturePrediction
from src.evaluation import (
    PredictionRecord,
    build_manifest_from_directory,
    infer_media_type,
    read_manifest,
    summarize_counts,
)
from src.evaluation.manifest import write_prediction_records
from src.recognition.trajectory_buffer import TrajectoryBuffer


def test_read_manifest_normalizes_labels_and_relative_paths(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    image_path = tmp_path / "open_palm.jpg"
    image_path.write_text("placeholder", encoding="utf-8")
    manifest_path.write_text(
        "\n".join(
            [
                "sample_id,path,expected_gesture,media_type,dataset,condition,distance",
                "s1,open_palm.jpg,0,,hands,normal,1m",
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
    assert samples[0].dataset == "hands"


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
                dataset="hands",
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
            "dataset": "hands",
            "condition": "normal",
            "distance": "1m",
            "media_type": "image",
            "source_path": "frame.png",
            "frame_count": "1",
            "note": "",
        }
    ]


def test_evaluate_parser_accepts_dynamic_confirmation_override() -> None:
    args = build_parser().parse_args(
        [
            "--manifest",
            "manifest.csv",
            "--output",
            "predictions.csv",
            "--dynamic-confirmation-frames",
            "2",
        ]
    )

    assert args.dynamic_confirmation_frames == 2


def test_build_manifest_from_directory_maps_hands_subset(tmp_path: Path) -> None:
    input_dir = tmp_path / "external" / "hands"
    (input_dir / "fist").mkdir(parents=True)
    (input_dir / "two").mkdir()
    (input_dir / "point").mkdir()
    (input_dir / "fist" / "001.jpg").write_text("placeholder", encoding="utf-8")
    (input_dir / "fist" / "002.jpg").write_text("placeholder", encoding="utf-8")
    (input_dir / "two" / "001.png").write_text("placeholder", encoding="utf-8")
    (input_dir / "point" / "001.jpg").write_text("placeholder", encoding="utf-8")
    output_path = tmp_path / "processed" / "manifest.csv"

    result = build_manifest_from_directory(
        input_dir=input_dir,
        output_path=output_path,
        dataset="hands",
        condition="normal",
        distance="1m",
        limit_per_class=1,
    )

    assert len(result.rows) == 2
    assert result.counts_by_gesture == {"FIST": 1, "PEACE": 1}
    assert result.skipped_files == 1

    samples = read_manifest(output_path)
    assert [sample.expected_gesture for sample in samples] == ["FIST", "PEACE"]
    assert samples[0].dataset == "hands"
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


def test_stable_static_prediction_accepts_known_majority_with_unknowns() -> None:
    prediction = _stable_static_prediction(
        [
            GesturePrediction(GestureID.THUMB_UP, 0.7),
            GesturePrediction(GestureID.THUMB_UP, 0.8),
            GesturePrediction.unknown("motion_gap"),
            GesturePrediction(GestureID.THUMB_UP, 0.6),
            GesturePrediction.unknown("motion_gap"),
        ]
    )

    assert prediction.gesture_id == GestureID.THUMB_UP
    assert abs(prediction.confidence - 0.7) < 1e-9


def test_stable_static_prediction_accepts_sparse_known_consensus() -> None:
    prediction = _stable_static_prediction(
        [
            GesturePrediction(GestureID.INDEX_LEFT, 0.7),
            GesturePrediction.unknown("motion_gap"),
            GesturePrediction.unknown("motion_gap"),
            GesturePrediction(GestureID.INDEX_LEFT, 0.8),
            GesturePrediction.unknown("motion_gap"),
            GesturePrediction(GestureID.INDEX_LEFT, 0.6),
            GesturePrediction.unknown("motion_gap"),
        ]
    )

    assert prediction.gesture_id == GestureID.INDEX_LEFT
    assert abs(prediction.confidence - 0.7) < 1e-9


def test_stable_static_prediction_rejects_unstable_known_votes() -> None:
    prediction = _stable_static_prediction(
        [
            GesturePrediction(GestureID.OPEN_PALM, 0.8),
            GesturePrediction(GestureID.OPEN_PALM, 0.7),
            GesturePrediction(GestureID.FIST, 0.9),
            GesturePrediction(GestureID.FIST, 0.8),
            GesturePrediction.unknown("motion_gap"),
        ]
    )

    assert prediction.gesture_id == GestureID.UNKNOWN
    assert prediction.metadata == {"reason": "unstable_pipeline_static_prediction"}


def test_full_dynamic_sequence_aggregation_uses_entire_landmark_sequence() -> None:
    evaluator = SampleEvaluator.__new__(SampleEvaluator)
    classifier = _RecordingDynamicClassifier()
    evaluator._dynamic_classifier = classifier

    prediction, _latency_ms, _fps, frame_count, note = (
        evaluator._evaluate_landmark_dynamic_full([_landmark_frame(index) for index in range(40)])
    )

    assert prediction.gesture_id == GestureID.PULL_TOWARD
    assert frame_count == 40
    assert note == "full_dynamic_sequence"
    assert classifier.seen_lengths == [40]


class _RecordingDynamicClassifier:
    def __init__(self) -> None:
        self.seen_lengths: list[int] = []

    def classify(self, buffer: TrajectoryBuffer) -> GesturePrediction:
        self.seen_lengths.append(len(buffer))
        return GesturePrediction(GestureID.PULL_TOWARD, 0.9)


def _landmark_frame(index: int) -> list[tuple[float, float, float]]:
    offset = index * 0.001
    landmarks = [(offset, 0.0, 0.0) for _ in range(21)]
    landmarks[0] = (offset, 0.0, 0.0)
    landmarks[5] = (offset - 0.1, -0.1, 0.0)
    landmarks[8] = (offset - 0.1, -0.4, 0.0)
    landmarks[9] = (offset, -0.1, 0.0)
    landmarks[12] = (offset, -0.4, 0.0)
    landmarks[17] = (offset + 0.1, -0.1, 0.0)
    landmarks[20] = (offset + 0.1, -0.4, 0.0)
    return landmarks
