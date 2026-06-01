from __future__ import annotations

from pathlib import Path

from scripts.build_ipn_manifest import build_ipn_rows


def test_build_ipn_rows_uses_segment_boundaries_and_label_mapping(tmp_path: Path) -> None:
    annotations_dir = tmp_path / "annotations"
    videos_dir = tmp_path / "videos"
    annotations_dir.mkdir()
    videos_dir.mkdir()
    (videos_dir / "sample_001.avi").write_text("placeholder", encoding="utf-8")
    (annotations_dir / "metadata.csv").write_text(
        "\n".join(
            [
                "Video Name,Frames,Sex,Hand,Background,Illumination,People in Scene,"
                "Background Motion,Set",
                "sample_001,200,W,Right,Clutter,Stable,Single,Static,test",
            ]
        ),
        encoding="utf-8",
    )
    (annotations_dir / "Annot_TestList.txt").write_text(
        "\n".join(
            [
                "sample_001,G05,8,10,45,36",
                "sample_001,G10,13,60,100,41",
                "sample_001,D0X,1,101,140,40",
            ]
        ),
        encoding="utf-8",
    )

    rows, skipped = build_ipn_rows(
        annotations_dir=annotations_dir,
        videos_dir=videos_dir,
        split="test",
        labels=("D0X", "G05", "G10"),
        include_unknown=True,
        limit_per_class=None,
        distance="unknown",
    )

    assert skipped == 0
    assert [row.expected_gesture for row in rows] == ["PULL_TOWARD", "UNKNOWN", "WAVE_LR"]
    assert rows[0].start_frame == 60
    assert rows[0].end_frame == 100
    assert rows[0].condition == "stable+clutter+static"
