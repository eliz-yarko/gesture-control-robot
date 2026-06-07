from __future__ import annotations

import json
import random
from pathlib import Path

from scripts.retrain_open_data import build_open_data_manifest, build_parser
from scripts.train_gesture_models import (
    _augmented_trajectory_windows,
    _dynamic_window_label,
    _extract_landmark_frames,
    _jitter_trajectory_window,
    _trajectory_windows,
)
from src.evaluation import read_manifest
from src.recognition.trajectory_buffer import TrajectoryPoint


def test_extract_landmark_frames_reads_sequence_json() -> None:
    frame = [[float(index), float(index + 1), 0.0] for index in range(21)]
    payload = {"frames": [{"landmarks": frame}, {"landmarks": frame}]}

    frames = _extract_landmark_frames(payload)

    assert len(frames) == 2
    assert frames[0][8] == (8.0, 9.0, 0.0)


def test_trajectory_windows_include_short_and_full_segments() -> None:
    points = list(range(12))

    windows = _trajectory_windows(points, max_size=30, min_size=5)

    assert [len(window) for window in windows] == [5, 5, 5, 6, 6, 6, 9, 9, 9, 12]
    assert windows[0] == [0, 1, 2, 3, 4]
    assert windows[-1] == points


def test_dynamic_augmentation_adds_deterministic_jittered_windows() -> None:
    points = [
        TrajectoryPoint(
            palm_center=(0.2 + index * 0.01, 0.4, 0.0),
            index_tip=(0.2 + index * 0.01, 0.2, 0.0),
            hand_size=0.2,
            timestamp=float(index),
        )
        for index in range(5)
    ]

    windows = _augmented_trajectory_windows(
        points,
        copies=2,
        random_state=42,
        sample_id="sample",
        window_index=0,
    )

    assert len(windows) == 3
    assert windows[0] == points
    assert windows[1] != points
    assert windows[1] == _jitter_trajectory_window(
        points,
        random.Random("42:sample:0:0"),
    )


def test_dynamic_window_label_marks_short_dynamic_prefixes_unknown() -> None:
    assert (
        _dynamic_window_label(
            "WAVE_LR",
            is_dynamic_sample=True,
            window_size=5,
            full_size=20,
            positive_min_window_ratio=0.6,
            min_size=5,
        )
        == "UNKNOWN"
    )
    assert (
        _dynamic_window_label(
            "WAVE_LR",
            is_dynamic_sample=True,
            window_size=12,
            full_size=20,
            positive_min_window_ratio=0.6,
            min_size=5,
        )
        == "WAVE_LR"
    )


def test_build_open_data_manifest_combines_directory_and_ipn_sources(tmp_path: Path) -> None:
    hagrid_dir = tmp_path / "hagrid_v2"
    (hagrid_dir / "fist").mkdir(parents=True)
    (hagrid_dir / "fist" / "001.jpg").write_text("placeholder", encoding="utf-8")
    (hagrid_dir / "ok" / "sample.json").parent.mkdir()
    (hagrid_dir / "ok" / "sample.json").write_text(
        json.dumps({"landmarks": [[0.0, 0.0, 0.0] for _ in range(21)]}),
        encoding="utf-8",
    )

    ipn_root = tmp_path / "ipn_hand"
    annotations_dir = ipn_root / "annotations"
    videos_dir = ipn_root / "videos"
    annotations_dir.mkdir(parents=True)
    videos_dir.mkdir()
    (videos_dir / "sample_001.avi").write_text("placeholder", encoding="utf-8")
    (annotations_dir / "Annot_TrainList.txt").write_text(
        "sample_001,G05,8,10,45,36",
        encoding="utf-8",
    )

    output_path = tmp_path / "open_data_manifest.csv"
    result = build_open_data_manifest(
        output_path=output_path,
        hagrid_dir=hagrid_dir,
        ipn_root=ipn_root,
        limit_per_class=2,
    )

    assert result.counts_by_gesture == {"FIST": 1, "OK_SIGN": 1, "WAVE_LR": 1}
    samples = read_manifest(output_path)
    assert [sample.expected_gesture for sample in samples] == ["FIST", "OK_SIGN", "WAVE_LR"]
    assert samples[1].media_type == "landmarks"


def test_build_open_data_manifest_merges_extra_manifest(tmp_path: Path) -> None:
    landmark_path = tmp_path / "landmarks" / "sample.json"
    landmark_path.parent.mkdir()
    landmark_path.write_text(
        json.dumps({"frames": [{"landmarks": [[0.0, 0.0, 0.0] for _ in range(21)]}]}),
        encoding="utf-8",
    )
    extra_manifest = tmp_path / "extra.csv"
    extra_manifest.write_text(
        "\n".join(
            [
                "sample_id,path,expected_gesture,media_type,dataset,condition,distance",
                f"extra_001,{landmark_path},OPEN_PALM,landmarks,own_control,cached,mixed",
            ]
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "combined.csv"

    result = build_open_data_manifest(
        output_path=output_path,
        extra_manifests=(extra_manifest,),
    )

    samples = read_manifest(output_path)
    assert result.counts_by_gesture == {"OPEN_PALM": 1}
    assert samples[0].sample_id == "extra_001"
    assert samples[0].media_type == "landmarks"


def test_open_data_parser_accepts_dynamic_augmentation_copies() -> None:
    args = build_parser().parse_args(["--dynamic-augmentation-copies", "2"])

    assert args.dynamic_augmentation_copies == 2


def test_open_data_parser_accepts_dynamic_positive_min_window_ratio() -> None:
    args = build_parser().parse_args(["--dynamic-positive-min-window-ratio", "0.65"])

    assert args.dynamic_positive_min_window_ratio == 0.65
