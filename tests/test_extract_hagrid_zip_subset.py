from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
from scripts.extract_hagrid_zip_subset import extract_hagrid_subset
from src.evaluation import build_manifest_from_directory


def test_extract_hagrid_subset_writes_limited_class_directories(tmp_path: Path) -> None:
    zip_path = tmp_path / "hagrid.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for index in range(3):
            archive.writestr(
                f"hagrid-sample-30k-384p/hagrid_30k/train_val_fist/{index}.jpg",
                b"jpg",
            )

    output_root = tmp_path / "subset"
    counts = extract_hagrid_subset(
        zip_path=zip_path,
        output_root=output_root,
        classes=("fist",),
        limit_per_class=2,
    )

    assert counts == {"fist": 2}
    assert len(list((output_root / "fist").glob("*.jpg"))) == 2


def test_extract_hagrid_subset_errors_when_limit_is_unavailable(tmp_path: Path) -> None:
    zip_path = tmp_path / "hagrid.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("hagrid-sample-30k-384p/hagrid_30k/train_val_fist/1.jpg", b"jpg")

    with pytest.raises(ValueError, match="only found 1"):
        extract_hagrid_subset(
            zip_path=zip_path,
            output_root=tmp_path / "subset",
            classes=("fist",),
            limit_per_class=2,
        )


def test_hagrid_non_command_class_maps_to_unknown(tmp_path: Path) -> None:
    input_root = tmp_path / "hagrid"
    (input_root / "call").mkdir(parents=True)
    (input_root / "call" / "sample.jpg").write_bytes(b"jpg")

    result = build_manifest_from_directory(
        input_dir=input_root,
        output_path=tmp_path / "manifest.csv",
        dataset="hagrid_v2",
        include_unknown=True,
    )

    assert result.counts_by_gesture == {"UNKNOWN": 1}
