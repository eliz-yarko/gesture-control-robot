from __future__ import annotations

from pathlib import Path

from scripts.reweight_manifest import build_parser, main, reweight_rows


def test_reweight_rows_duplicates_matching_dataset_and_label(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    landmark_path = input_dir / "landmarks" / "thumb.json"
    landmark_path.parent.mkdir(parents=True)
    landmark_path.write_text("{}", encoding="utf-8")
    output_dir.mkdir()

    rows = [
        {
            "sample_id": "own_thumb",
            "path": "landmarks/thumb.json",
            "expected_gesture": "thumb up",
            "media_type": "landmarks",
            "dataset": "own_control",
            "condition": "phone",
            "distance": "mixed",
        },
        {
            "sample_id": "hands_thumb",
            "path": "external/thumb.json",
            "expected_gesture": "THUMB_UP",
            "media_type": "landmarks",
            "dataset": "hands",
            "condition": "external",
            "distance": "unknown",
        },
    ]

    reweighted = reweight_rows(
        rows,
        input_dir=input_dir,
        output_dir=output_dir,
        copies=2,
        datasets=("own_control",),
        labels=("THUMB_UP",),
    )

    assert [row["sample_id"] for row in reweighted] == [
        "own_thumb",
        "hands_thumb",
        "own_thumb_rw01",
        "own_thumb_rw02",
    ]
    assert [row["expected_gesture"] for row in reweighted] == [
        "THUMB_UP",
        "THUMB_UP",
        "THUMB_UP",
        "THUMB_UP",
    ]
    assert Path(reweighted[0]["path"]).is_absolute()


def test_reweight_parser_accepts_repeated_filters() -> None:
    args = build_parser().parse_args(
        [
            "--input",
            "train.csv",
            "--output",
            "weighted.csv",
            "--copies",
            "3",
            "--dataset",
            "own_control",
            "--label",
            "THUMB_UP",
            "--label",
            "THUMB_DOWN",
        ]
    )

    assert args.copies == 3
    assert args.dataset == ["own_control"]
    assert args.label == ["THUMB_UP", "THUMB_DOWN"]


def test_reweight_main_writes_manifest_with_original_field_order(tmp_path: Path) -> None:
    input_path = tmp_path / "input" / "manifest.csv"
    output_path = tmp_path / "output" / "weighted.csv"
    input_path.parent.mkdir()
    input_path.write_text(
        "\n".join(
            [
                "sample_id,path,expected_gesture,media_type,dataset",
                "sample,sample.json,FIST,landmarks,own_control",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--copies",
            "1",
            "--dataset",
            "own_control",
            "--label",
            "FIST",
        ]
    )

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert exit_code == 0
    assert lines[0] == "sample_id,path,expected_gesture,media_type,dataset"
    assert lines[2].startswith("sample_rw01,")
