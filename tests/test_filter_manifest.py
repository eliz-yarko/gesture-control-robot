from __future__ import annotations

from scripts.filter_manifest import filter_rows


def test_filter_rows_removes_low_frame_count_and_excluded_ids() -> None:
    rows = [
        {"sample_id": "keep", "frame_count": "3"},
        {"sample_id": "empty", "frame_count": "0"},
        {"sample_id": "manual", "frame_count": "10"},
    ]

    filtered = filter_rows(
        rows,
        min_frame_count=1,
        excluded_sample_ids={"manual"},
    )

    assert filtered == [{"sample_id": "keep", "frame_count": "3"}]
