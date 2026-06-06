from __future__ import annotations

import json

from scripts.tune_thresholds import main


def test_tune_thresholds_writes_profile_that_blocks_critical_false_positive(tmp_path) -> None:
    predictions_path = tmp_path / "predictions.csv"
    output_path = tmp_path / "thresholds.json"
    predictions_path.write_text(
        "\n".join(
            [
                "sample_id,expected_gesture,predicted_gesture,confidence",
                "1,OPEN_PALM,OPEN_PALM,0.91",
                "2,FIST,FIST,0.88",
                "3,OPEN_PALM,THUMB_DOWN,0.62",
                "4,THUMB_DOWN,THUMB_DOWN,0.93",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--input",
            str(predictions_path),
            "--output",
            str(output_path),
            "--default-threshold",
            "0.35",
            "--step",
            "0.05",
            "--max-critical-fpr",
            "0",
        ]
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["class_thresholds"]["THUMB_DOWN"] > 0.62
    assert payload["tuned"]["critical_false_positive_rate"] == 0.0
