"""Build benchmark manifests from locally stored dataset subsets."""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from src.evaluation.manifest import IMAGE_EXTENSIONS, LANDMARK_EXTENSIONS, VIDEO_EXTENSIONS
from src.utils.metrics import normalize_label

MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS | LANDMARK_EXTENSIONS


DATASET_CLASS_MAPS: dict[str, dict[str, str]] = {
    "own_control": {
        "unknown": "UNKNOWN",
        "no_gesture": "UNKNOWN",
        "open_palm": "OPEN_PALM",
        "stop": "OPEN_PALM",
        "fist": "FIST",
        "forward": "FIST",
        "thumb_up": "THUMB_UP",
        "like": "THUMB_UP",
        "start": "THUMB_UP",
        "thumb_down": "THUMB_DOWN",
        "dislike": "THUMB_DOWN",
        "emergency_stop": "THUMB_DOWN",
        "index_left": "INDEX_LEFT",
        "turn_left": "INDEX_LEFT",
        "index_right": "INDEX_RIGHT",
        "turn_right": "INDEX_RIGHT",
        "peace": "PEACE",
        "two_up": "PEACE",
        "increase_speed": "PEACE",
        "three_fingers": "THREE_FINGERS",
        "three": "THREE_FINGERS",
        "decrease_speed": "THREE_FINGERS",
        "pinky": "PINKY",
        "little_finger": "PINKY",
        "return_home": "PINKY",
        "ok": "OK_SIGN",
        "ok_sign": "OK_SIGN",
        "confirm_action": "OK_SIGN",
        "wave_lr": "WAVE_LR",
        "wave_left_right": "WAVE_LR",
        "circle": "CIRCLE",
        "pull_toward": "PULL_TOWARD",
        "pulling_hand_in": "PULL_TOWARD",
    },
    "hagrid": {
        "no_gesture": "UNKNOWN",
        "stop": "OPEN_PALM",
        "palm": "OPEN_PALM",
        "fist": "FIST",
        "like": "THUMB_UP",
        "thumb_up": "THUMB_UP",
        "dislike": "THUMB_DOWN",
        "thumb_down": "THUMB_DOWN",
        "peace": "PEACE",
        "two_up": "PEACE",
        "three": "THREE_FINGERS",
        "three2": "THREE_FINGERS",
        "three3": "THREE_FINGERS",
        "little_finger": "PINKY",
        "pinky": "PINKY",
        "ok": "OK_SIGN",
        "ok_sign": "OK_SIGN",
    },
    "hagrid_v2": {
        "no_gesture": "UNKNOWN",
        "stop": "OPEN_PALM",
        "palm": "OPEN_PALM",
        "fist": "FIST",
        "like": "THUMB_UP",
        "thumb_up": "THUMB_UP",
        "dislike": "THUMB_DOWN",
        "thumb_down": "THUMB_DOWN",
        "peace": "PEACE",
        "two_up": "PEACE",
        "three": "THREE_FINGERS",
        "three2": "THREE_FINGERS",
        "three3": "THREE_FINGERS",
        "little_finger": "PINKY",
        "pinky": "PINKY",
        "ok": "OK_SIGN",
        "ok_sign": "OK_SIGN",
    },
    "jester": {
        "doing_other_things": "UNKNOWN",
        "no_gesture": "UNKNOWN",
        "stop_sign": "OPEN_PALM",
        "thumb_up": "THUMB_UP",
        "thumb_down": "THUMB_DOWN",
        "swiping_left": "WAVE_LR",
        "swiping_right": "WAVE_LR",
        "shaking_hand": "WAVE_LR",
        "pulling_hand_in": "PULL_TOWARD",
        "pulling_two_fingers_in": "PULL_TOWARD",
        "zooming_in_with_full_hand": "PULL_TOWARD",
    },
    "ipn_hand": {
        "d0x": "UNKNOWN",
        "non_gesture": "UNKNOWN",
        "g05": "WAVE_LR",
        "throw_left": "WAVE_LR",
        "g06": "WAVE_LR",
        "throw_right": "WAVE_LR",
        "g10": "PULL_TOWARD",
        "zoom_in": "PULL_TOWARD",
    },
}


@dataclass(frozen=True)
class ManifestRow:
    """One CSV row for a dataset benchmark manifest."""

    sample_id: str
    path: str
    expected_gesture: str
    media_type: str
    dataset: str
    condition: str
    distance: str


@dataclass(frozen=True)
class ManifestBuildResult:
    """Summary of manifest-building work."""

    output_path: Path
    rows: tuple[ManifestRow, ...]
    skipped_files: int
    counts_by_gesture: Mapping[str, int]


def available_dataset_presets() -> tuple[str, ...]:
    """Return supported dataset preset names."""

    return tuple(sorted(DATASET_CLASS_MAPS))


def build_manifest_from_directory(
    input_dir: Path,
    output_path: Path,
    dataset: str,
    condition: str = "normal",
    distance: str = "unknown",
    limit_per_class: int | None = None,
    include_unknown: bool = False,
) -> ManifestBuildResult:
    """Scan a local dataset subset and write a benchmark manifest CSV.

    The scanner expects files to be placed under class-named directories. The class
    directory may be the file's direct parent or any ancestor between ``input_dir``
    and the file.

    Args:
        input_dir: Directory with local dataset files.
        output_path: Destination CSV path.
        dataset: Dataset preset name, for example ``hagrid_v2`` or ``own_control``.
        condition: Capture condition label written to each row.
        distance: Camera distance label written to each row.
        limit_per_class: Optional maximum number of samples per target gesture.
        include_unknown: Whether to include samples mapped to ``UNKNOWN``.

    Returns:
        Build result with written rows and per-class counts.

    Raises:
        ValueError: If arguments are invalid.
        FileNotFoundError: If ``input_dir`` does not exist.
    """

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")
    if limit_per_class is not None and limit_per_class <= 0:
        raise ValueError("limit_per_class must be positive when provided")

    dataset_key = _normalize_alias(dataset)
    class_map = _class_map_for_dataset(dataset_key)
    candidate_files = _collect_media_files(input_dir)
    selected_files: dict[str, list[Path]] = defaultdict(list)
    skipped_files = 0

    for file_path in candidate_files:
        gesture_label = _match_gesture_label(file_path, input_dir, class_map)
        if gesture_label is None:
            skipped_files += 1
            continue
        if gesture_label == "UNKNOWN" and not include_unknown:
            skipped_files += 1
            continue
        if limit_per_class is not None and len(selected_files[gesture_label]) >= limit_per_class:
            continue
        selected_files[gesture_label].append(file_path)

    rows = _build_rows(
        selected_files=selected_files,
        output_dir=output_path.parent,
        dataset=dataset_key,
        condition=condition,
        distance=distance,
    )
    write_manifest(output_path, rows)
    return ManifestBuildResult(
        output_path=output_path,
        rows=tuple(rows),
        skipped_files=skipped_files,
        counts_by_gesture=Counter(row.expected_gesture for row in rows),
    )


def write_manifest(output_path: Path, rows: Sequence[ManifestRow]) -> None:
    """Write manifest rows using the schema consumed by ``evaluate_manifest.py``."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "sample_id",
                "path",
                "expected_gesture",
                "media_type",
                "dataset",
                "condition",
                "distance",
            ),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sample_id": row.sample_id,
                    "path": row.path,
                    "expected_gesture": row.expected_gesture,
                    "media_type": row.media_type,
                    "dataset": row.dataset,
                    "condition": row.condition,
                    "distance": row.distance,
                }
            )


def _class_map_for_dataset(dataset_key: str) -> Mapping[str, str]:
    if dataset_key not in DATASET_CLASS_MAPS:
        presets = ", ".join(available_dataset_presets())
        raise ValueError(f"Unknown dataset preset: {dataset_key}. Available presets: {presets}")
    return DATASET_CLASS_MAPS[dataset_key]


def _collect_media_files(input_dir: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS
        ),
        key=lambda path: path.as_posix(),
    )


def _match_gesture_label(
    file_path: Path,
    input_dir: Path,
    class_map: Mapping[str, str],
) -> str | None:
    relative_parts = file_path.relative_to(input_dir).parts[:-1]
    for part in reversed(relative_parts):
        alias = _normalize_alias(part)
        if alias in class_map:
            return normalize_label(class_map[alias])
    return None


def _build_rows(
    selected_files: Mapping[str, Sequence[Path]],
    output_dir: Path,
    dataset: str,
    condition: str,
    distance: str,
) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    for gesture_label in sorted(selected_files):
        for index, file_path in enumerate(selected_files[gesture_label], start=1):
            rows.append(
                ManifestRow(
                    sample_id=f"{dataset}_{gesture_label.lower()}_{index:04d}",
                    path=_relative_path(file_path, output_dir),
                    expected_gesture=gesture_label,
                    media_type=_media_type(file_path),
                    dataset=dataset,
                    condition=condition,
                    distance=distance,
                )
            )
    return rows


def _relative_path(path: Path, base_dir: Path) -> str:
    try:
        relative_path = path.resolve().relative_to(base_dir.resolve())
    except ValueError:
        relative_path = path.resolve()
    return relative_path.as_posix()


def _media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    if suffix in LANDMARK_EXTENSIONS:
        return "landmarks"
    raise ValueError(f"Unsupported media file extension: {path}")


def _normalize_alias(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return normalized


def summarize_counts(counts_by_gesture: Mapping[str, int]) -> str:
    """Format per-gesture counts for CLI output."""

    if not counts_by_gesture:
        return "no samples"
    parts = [f"{label}={count}" for label, count in sorted(counts_by_gesture.items())]
    return ", ".join(parts)
