"""Download a small class-limited image subset from a Hugging Face dataset repo."""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SplitRequest:
    """One source split download request."""

    source_split: str
    output_split: str
    limit_per_class: int


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Download a bounded class-directory subset from Hugging Face dataset files. "
            "The downloader uses the public dataset file API and does not require "
            "huggingface_hub or datasets."
        )
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="Dataset repo id, for example schwein69/hagrid-subset.",
    )
    parser.add_argument("--output-root", required=True, help="Destination directory.")
    parser.add_argument(
        "--class",
        dest="classes",
        action="append",
        required=True,
        help="Class folder to download. Can be provided multiple times.",
    )
    parser.add_argument(
        "--split",
        action="append",
        required=True,
        help=(
            "Split request in source:output:limit format, for example "
            "test:control:19 or val:holdout:9."
        ),
    )
    parser.add_argument("--revision", default="main", help="Dataset revision.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Delay between downloads.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Download selected files from a Hugging Face dataset repo."""

    args = build_parser().parse_args(argv)
    split_requests = [_parse_split_request(value) for value in args.split]
    selected_files = select_repo_files(
        repo=args.repo,
        classes=args.classes,
        split_requests=split_requests,
    )
    output_root = Path(args.output_root)
    downloaded = 0
    skipped_existing = 0
    for remote_path, local_relative_path in selected_files:
        target_path = output_root / local_relative_path
        if target_path.exists() and not args.overwrite:
            skipped_existing += 1
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        _download_file(
            repo=args.repo,
            revision=args.revision,
            remote_path=remote_path,
            target_path=target_path,
        )
        downloaded += 1
        if args.sleep > 0:
            time.sleep(args.sleep)

    print(f"Selected files: {len(selected_files)}")
    print(f"Downloaded files: {downloaded}")
    print(f"Skipped existing files: {skipped_existing}")
    print(f"Output root: {output_root}")
    return 0


def select_repo_files(
    *,
    repo: str,
    classes: Sequence[str],
    split_requests: Sequence[SplitRequest],
) -> list[tuple[str, Path]]:
    """Return remote and local paths for selected dataset files."""

    if not classes:
        raise ValueError("At least one class must be requested")
    if not split_requests:
        raise ValueError("At least one split request must be provided")

    class_set = set(classes)
    selected: list[tuple[str, Path]] = []
    repo_files = _repo_files(repo)
    for request in split_requests:
        if request.limit_per_class <= 0:
            raise ValueError("Split limits must be positive")
        for class_name in classes:
            prefix = f"data/{request.source_split}/{class_name}/"
            class_files = [
                file_path
                for file_path in repo_files
                if file_path.startswith(prefix) and _is_image_file(file_path)
            ]
            if len(class_files) < request.limit_per_class:
                raise ValueError(
                    f"Requested {request.limit_per_class} files for {class_name!r} "
                    f"in split {request.source_split!r}, but only found {len(class_files)}"
                )
            for remote_path in sorted(class_files)[: request.limit_per_class]:
                if _class_from_remote_path(remote_path) not in class_set:
                    continue
                filename = Path(remote_path).name
                local_path = Path(request.output_split) / class_name / filename
                selected.append((remote_path, local_path))
    return selected


def _repo_files(repo: str) -> list[str]:
    api_url = f"https://huggingface.co/api/datasets/{repo}"
    with urllib.request.urlopen(api_url, timeout=60) as response:
        payload = json.load(response)
    siblings = payload.get("siblings", [])
    if not isinstance(siblings, list):
        raise ValueError(f"Unexpected Hugging Face API response for {repo!r}")
    files: list[str] = []
    for item in siblings:
        if isinstance(item, dict) and isinstance(item.get("rfilename"), str):
            files.append(str(item["rfilename"]))
    return files


def _download_file(
    *,
    repo: str,
    revision: str,
    remote_path: str,
    target_path: Path,
) -> None:
    url = _raw_file_url(repo=repo, revision=revision, remote_path=remote_path)
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = response.read()
    pointer = _parse_lfs_pointer(payload)
    if pointer is not None:
        payload = _download_lfs_object(repo=repo, oid=pointer[0], size=pointer[1])
    target_path.write_bytes(payload)


def _raw_file_url(*, repo: str, revision: str, remote_path: str) -> str:
    quoted_path = urllib.parse.quote(remote_path)
    return f"https://huggingface.co/datasets/{repo}/raw/{revision}/{quoted_path}"


def _parse_lfs_pointer(payload: bytes) -> tuple[str, int] | None:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return None
    lines = [line.strip() for line in text.splitlines()]
    if not lines or lines[0] != "version https://git-lfs.github.com/spec/v1":
        return None

    oid = ""
    size = 0
    for line in lines[1:]:
        if line.startswith("oid sha256:"):
            oid = line.removeprefix("oid sha256:").strip()
        elif line.startswith("size "):
            size = int(line.removeprefix("size ").strip())
    if not oid or size <= 0:
        raise ValueError("Invalid Git LFS pointer payload")
    return oid, size


def _download_lfs_object(*, repo: str, oid: str, size: int) -> bytes:
    url = f"https://huggingface.co/datasets/{repo}.git/info/lfs/objects/batch"
    request_payload = json.dumps(
        {
            "operation": "download",
            "transfers": ["basic"],
            "objects": [{"oid": oid, "size": size}],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=request_payload,
        headers={"Content-Type": "application/vnd.git-lfs+json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.load(response)

    objects = payload.get("objects", [])
    if not objects or not isinstance(objects[0], dict):
        raise ValueError("Git LFS batch response did not include an object")
    actions = objects[0].get("actions", {})
    if not isinstance(actions, dict):
        raise ValueError("Git LFS batch response did not include actions")
    download = actions.get("download", {})
    if not isinstance(download, dict) or not isinstance(download.get("href"), str):
        raise ValueError("Git LFS batch response did not include a download URL")

    with urllib.request.urlopen(str(download["href"]), timeout=120) as response:
        return response.read()


def _parse_split_request(value: str) -> SplitRequest:
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError("--split must use source:output:limit format")
    source_split, output_split, limit_text = parts
    return SplitRequest(
        source_split=source_split,
        output_split=output_split,
        limit_per_class=int(limit_text),
    )


def _class_from_remote_path(remote_path: str) -> str:
    parts = remote_path.split("/")
    return parts[2] if len(parts) >= 4 else ""


def _is_image_file(path: str) -> bool:
    return Path(path).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}


if __name__ == "__main__":
    raise SystemExit(main())
