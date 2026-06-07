from __future__ import annotations

import csv
from pathlib import Path

import pytest
from scripts.split_manifest import main, split_manifest_rows


def test_split_manifest_rows_stratifies_and_keeps_groups_disjoint() -> None:
    rows = [
        _row("open_1", "OPEN_PALM"),
        _row("open_2", "OPEN_PALM"),
        _row("open_3", "OPEN_PALM"),
        _row("open_4", "OPEN_PALM"),
        _row("fist_1", "FIST"),
        _row("fist_2", "FIST"),
        _row("fist_3", "FIST"),
        _row("fist_4", "FIST"),
    ]

    train_rows, control_rows, holdout_rows, summary = split_manifest_rows(
        rows,
        control_count=4,
        holdout_count=2,
        seed=7,
    )

    assert summary.control_counts == {"OPEN_PALM": 2, "FIST": 2}
    assert summary.holdout_counts == {"OPEN_PALM": 1, "FIST": 1}
    assert summary.train_counts == {"OPEN_PALM": 1, "FIST": 1}

    train_groups = {row["sample_id"] for row in train_rows}
    control_groups = {row["sample_id"] for row in control_rows}
    holdout_groups = {row["sample_id"] for row in holdout_rows}
    assert train_groups.isdisjoint(control_groups)
    assert train_groups.isdisjoint(holdout_groups)
    assert control_groups.isdisjoint(holdout_groups)


def test_split_manifest_rows_rejects_multilabel_group() -> None:
    rows = [
        _row("shared", "OPEN_PALM"),
        _row("shared", "FIST"),
    ]

    with pytest.raises(ValueError, match="multiple labels"):
        split_manifest_rows(rows, control_count=1, holdout_count=0)


def test_split_manifest_rows_shares_shortfall_between_control_and_holdout() -> None:
    rows = [_row(f"open_{index}", "OPEN_PALM") for index in range(1, 11)]

    train_rows, control_rows, holdout_rows, summary = split_manifest_rows(
        rows,
        control_count=15,
        holdout_count=5,
    )

    assert len(train_rows) == 0
    assert len(control_rows) == 8
    assert len(holdout_rows) == 2
    assert summary.control_counts == {"OPEN_PALM": 8}
    assert summary.holdout_counts == {"OPEN_PALM": 2}


def test_split_manifest_cli_writes_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "input.csv"
    train_path = tmp_path / "train.csv"
    control_path = tmp_path / "control.csv"
    holdout_path = tmp_path / "holdout.csv"
    _write_manifest(input_path, [_row(f"open_{index}", "OPEN_PALM") for index in range(1, 5)])

    exit_code = main(
        [
            "--input",
            str(input_path),
            "--train-output",
            str(train_path),
            "--control-output",
            str(control_path),
            "--holdout-output",
            str(holdout_path),
            "--control-count",
            "2",
            "--holdout-count",
            "1",
        ]
    )

    assert exit_code == 0
    assert len(_read_rows(train_path)) == 1
    assert len(_read_rows(control_path)) == 2
    assert len(_read_rows(holdout_path)) == 1


def _row(sample_id: str, label: str) -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "path": f"{sample_id}.jpg",
        "expected_gesture": label,
        "media_type": "image",
        "dataset": "hagrid_v2",
        "condition": "normal",
        "distance": "unknown",
    }


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))
