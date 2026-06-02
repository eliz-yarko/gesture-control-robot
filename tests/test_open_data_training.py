from __future__ import annotations

import json
from pathlib import Path

from scripts.retrain_open_data import build_open_data_manifest
from scripts.train_gesture_models import _extract_landmark_frames
from src.evaluation import read_manifest


def test_extract_landmark_frames_reads_sequence_json() -> None:
    frame = [[float(index), float(index + 1), 0.0] for index in range(21)]
    payload = {"frames": [{"landmarks": frame}, {"landmarks": frame}]}

    frames = _extract_landmark_frames(payload)

    assert len(frames) == 2
    assert frames[0][8] == (8.0, 9.0, 0.0)


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
