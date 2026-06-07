from __future__ import annotations

from pathlib import Path

import pytest
import scripts.download_hf_class_subset as downloader
from scripts.download_hf_class_subset import SplitRequest, select_repo_files


def test_select_repo_files_returns_bounded_class_split_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        downloader,
        "_repo_files",
        lambda _repo: [
            "data/test/fist/a.jpg",
            "data/test/fist/b.jpg",
            "data/test/fist/c.jpg",
            "data/test/ok/a.jpg",
            "data/test/ok/b.jpg",
            "data/val/fist/v.jpg",
            "README.md",
        ],
    )

    selected = select_repo_files(
        repo="owner/repo",
        classes=("fist", "ok"),
        split_requests=(SplitRequest("test", "control", 2),),
    )

    assert selected == [
        ("data/test/fist/a.jpg", Path("control/fist/a.jpg")),
        ("data/test/fist/b.jpg", Path("control/fist/b.jpg")),
        ("data/test/ok/a.jpg", Path("control/ok/a.jpg")),
        ("data/test/ok/b.jpg", Path("control/ok/b.jpg")),
    ]


def test_select_repo_files_errors_when_class_limit_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(downloader, "_repo_files", lambda _repo: ["data/test/fist/a.jpg"])

    with pytest.raises(ValueError, match="only found 1"):
        select_repo_files(
            repo="owner/repo",
            classes=("fist",),
            split_requests=(SplitRequest("test", "control", 2),),
        )


def test_parse_split_request_requires_source_output_limit() -> None:
    request = downloader._parse_split_request("test:control:19")

    assert request == SplitRequest("test", "control", 19)
    with pytest.raises(ValueError, match="source:output:limit"):
        downloader._parse_split_request("test:control")


def test_raw_file_url_uses_dataset_raw_endpoint() -> None:
    url = downloader._raw_file_url(
        repo="owner/repo",
        revision="main",
        remote_path="data/test/no_gesture/a b.jpg",
    )

    assert url == (
        "https://huggingface.co/datasets/owner/repo/raw/main/"
        "data/test/no_gesture/a%20b.jpg"
    )


def test_parse_lfs_pointer_extracts_oid_and_size() -> None:
    payload = (
        b"version https://git-lfs.github.com/spec/v1\n"
        b"oid sha256:abc123\n"
        b"size 456\n"
    )

    assert downloader._parse_lfs_pointer(payload) == ("abc123", 456)
    assert downloader._parse_lfs_pointer(b"\xff\xd8 jpeg bytes") is None
