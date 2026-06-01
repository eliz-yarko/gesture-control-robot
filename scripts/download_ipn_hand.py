"""Download and extract the IPN Hand dataset from its public Google Drive files."""

from __future__ import annotations

import argparse
import tarfile
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

DOWNLOAD_URL = (
    "https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
)


@dataclass(frozen=True)
class DriveFile:
    """Public Google Drive file used by IPN Hand."""

    file_id: str
    name: str


ANNOTATION_FILES = (
    DriveFile("1zFRngxukYnVPWqAm2isrOrSUfSe-Sfp3", "Annot_List.txt"),
    DriveFile("13fXL1r62yuVDn6P-QR7GYM4L7d5IdrZa", "Annot_TestList.txt"),
    DriveFile("1Y6yIcckNonsRZmyD833oCLF-ndu3rvT7", "Annot_TrainList.txt"),
    DriveFile("1-3Ue2pMBQuEG26ggzDjdFuoW3akKS6Qq", "classIdx.txt"),
    DriveFile("17d2lsFOJZ-R74PrkS83iO7xSV8ZDiwwT", "metadata.csv"),
    DriveFile("1CK_wnCZz-O2nZ3IcXYo3uifxEQCtwYbU", "Video_TestList.txt"),
    DriveFile("1hj8ERCfjzStMzNxfWnS1ywl0ExKvjT00", "Video_TrainList.txt"),
)

VIDEO_ARCHIVES = (
    DriveFile("1HylyDnApIRNMloREqvKYWWhB88q7Z1Cg", "videos01.tgz"),
    DriveFile("1tqR2FF8OlXGmYACw3TSxyS6QGDit_s3a", "videos02.tgz"),
    DriveFile("1Dk7l-jAAvNLlb0faypAco88XYLWPa9g_", "videos03.tgz"),
    DriveFile("1x0mDr-QHQtDkfcQAm9bKdWBR7Lj3fLcV", "videos04.tgz"),
    DriveFile("1PCjldH6hmVYV7EObPf-jtNWCtDK60o6n", "videos05.tgz"),
)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""

    parser = argparse.ArgumentParser(description="Download the IPN Hand dataset locally.")
    parser.add_argument(
        "--output-dir",
        default="data/external/ipn_hand",
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--annotations-only",
        action="store_true",
        help="Download only small annotation files.",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Download archives without extracting videos.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download existing files.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Download IPN Hand files."""

    args = build_parser().parse_args(argv)
    root = Path(args.output_dir)
    annotations_dir = root / "annotations"
    archives_dir = root / "archives"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    archives_dir.mkdir(parents=True, exist_ok=True)

    _download_files(ANNOTATION_FILES, annotations_dir, overwrite=args.overwrite)
    if args.annotations_only:
        return 0

    _download_files(VIDEO_ARCHIVES, archives_dir, overwrite=args.overwrite)
    if not args.skip_extract:
        _extract_archives(archives_dir, root)
    return 0


def _download_files(files: Sequence[DriveFile], output_dir: Path, overwrite: bool) -> None:
    for file in files:
        target = output_dir / file.name
        if target.exists() and not overwrite:
            print(f"Skipping existing file: {target}")
            continue

        url = DOWNLOAD_URL.format(file_id=file.file_id)
        print(f"Downloading {file.name} -> {target}")
        urllib.request.urlretrieve(url, target)


def _extract_archives(archives_dir: Path, root: Path) -> None:
    for archive_path in sorted(archives_dir.glob("*.tgz")):
        print(f"Extracting {archive_path.name}")
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(root, filter="data")


if __name__ == "__main__":
    raise SystemExit(main())
