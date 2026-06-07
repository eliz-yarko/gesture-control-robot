"""Extract a bounded class subset from the HaGRID sample zip archive."""

from __future__ import annotations

import argparse
import random
import zipfile
from collections.abc import Sequence
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Extract selected class folders from hagrid-sample-30k-384p.zip without "
            "unpacking the entire archive."
        )
    )
    parser.add_argument("--zip", required=True, help="Path to hagrid-sample zip file.")
    parser.add_argument("--output-root", required=True, help="Destination class-directory root.")
    parser.add_argument(
        "--class",
        dest="classes",
        action="append",
        required=True,
        help="HaGRID class name, for example fist, palm, like, or call.",
    )
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=48,
        help="Maximum images to extract per requested class.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic shuffle seed.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Extract selected classes from a HaGRID sample zip."""

    args = build_parser().parse_args(argv)
    if args.limit_per_class <= 0:
        raise ValueError("--limit-per-class must be positive")

    counts = extract_hagrid_subset(
        zip_path=Path(args.zip),
        output_root=Path(args.output_root),
        classes=args.classes,
        limit_per_class=args.limit_per_class,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    for class_name, count in sorted(counts.items()):
        print(f"{class_name}: {count}")
    return 0


def extract_hagrid_subset(
    *,
    zip_path: Path,
    output_root: Path,
    classes: Sequence[str],
    limit_per_class: int,
    seed: int = 42,
    overwrite: bool = False,
) -> dict[str, int]:
    """Extract selected HaGRID class images and return written counts."""

    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file does not exist: {zip_path}")
    if limit_per_class <= 0:
        raise ValueError("limit_per_class must be positive")
    if not classes:
        raise ValueError("At least one class must be requested")

    rng = random.Random(seed)
    counts: dict[str, int] = {}
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        for class_name in classes:
            members = _class_members(names, class_name)
            if len(members) < limit_per_class:
                raise ValueError(
                    f"Requested {limit_per_class} images for {class_name!r}, "
                    f"but only found {len(members)}"
                )
            shuffled = list(members)
            rng.shuffle(shuffled)
            written = 0
            for member in shuffled[:limit_per_class]:
                target_path = output_root / class_name / Path(member).name
                if target_path.exists() and not overwrite:
                    written += 1
                    continue
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_bytes(archive.read(member))
                written += 1
            counts[class_name] = written
    return counts


def _class_members(names: Sequence[str], class_name: str) -> list[str]:
    folder = f"/hagrid_30k/train_val_{class_name}/"
    return sorted(
        name
        for name in names
        if folder in f"/{name}" and Path(name).suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


if __name__ == "__main__":
    raise SystemExit(main())
