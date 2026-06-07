from __future__ import annotations

import csv

from scripts.command_safety_benchmark import (
    CommandSafetySample,
    compute_command_safety_summary,
    main,
)


def test_compute_command_safety_summary_counts_false_confirmed_dynamic() -> None:
    samples = [
        CommandSafetySample("1", "OPEN_PALM", "OPEN_PALM", confirmed=True),
        CommandSafetySample("2", "OPEN_PALM", "WAVE_LR", confirmed=True),
        CommandSafetySample("3", "FIST", "THUMB_DOWN", confirmed=True),
        CommandSafetySample("4", "UNKNOWN", "PULL_TOWARD", confirmed=True),
        CommandSafetySample("5", "THUMB_DOWN", "THUMB_DOWN", confirmed=False),
    ]

    summary = compute_command_safety_summary(samples, critical_label="THUMB_DOWN")

    assert summary.sample_count == 5
    assert summary.confirmed_command_count == 4
    assert summary.false_confirmed_command_count == 3
    assert summary.critical_false_positive_count == 1
    assert summary.false_confirmed_dynamic_count == 2
    assert summary.dynamic_pretrigger_proxy_count == 2
    assert summary.static_dynamic_conflict_proxy_count == 2
    assert summary.command_overwrite_proxy_count == 1


def test_command_safety_benchmark_writes_summary_csv(tmp_path) -> None:
    input_path = tmp_path / "predictions.csv"
    output_path = tmp_path / "safety.csv"
    input_path.write_text(
        "\n".join(
            [
                "sample_id,expected_gesture,predicted_gesture,note",
                "1,OPEN_PALM,OPEN_PALM,confirmed_command",
                "2,OPEN_PALM,WAVE_LR,confirmed_command",
                "3,THUMB_DOWN,THUMB_DOWN,selected_prediction_without_command",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["--input", str(input_path), "--output", str(output_path)])

    with output_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert exit_code == 0
    assert rows[0]["group"] == "ALL"
    assert rows[0]["sample_count"] == "3"
    assert rows[0]["false_confirmed_dynamic_count"] == "1"
