"""Export a video/image manifest to cached MediaPipe landmark JSON files."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AppConfig, VideoConfig  # noqa: E402
from src.evaluation import EvaluationSample, read_manifest  # noqa: E402
from src.recognition.hand_detector import HandDetector  # noqa: E402

MANIFEST_FIELDS = (
    "sample_id",
    "path",
    "expected_gesture",
    "media_type",
    "dataset",
    "condition",
    "distance",
    "start_frame",
    "end_frame",
)


class LandmarkManifestExporter:
    """Convert media samples to landmark-sequence JSON files."""

    def __init__(
        self,
        config: AppConfig,
        *,
        frame_stride: int,
        max_frames: int | None,
    ) -> None:
        """Initialize detector and OpenCV resources."""

        self._config = config
        self._frame_stride = frame_stride
        self._max_frames = max_frames
        self._cv2 = importlib.import_module("cv2")
        self._detector = HandDetector(config.hand_detection)

    def close(self) -> None:
        """Release detector resources."""

        self._detector.close()

    def export_sample(
        self,
        sample: EvaluationSample,
        output_path: Path,
    ) -> int:
        """Write one sample's landmark JSON and return extracted frame count."""

        frames = self._extract_frames(sample)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                _landmark_payload(sample, frames),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return len(frames)

    def _extract_frames(self, sample: EvaluationSample) -> list[dict[str, object]]:
        if sample.media_type == "landmarks":
            payload = json.loads(sample.path.read_text(encoding="utf-8-sig"))
            frames = payload.get("frames", []) if isinstance(payload, dict) else []
            return frames if isinstance(frames, list) else []
        if sample.media_type == "image":
            return self._extract_image(sample)
        if sample.media_type == "video":
            return self._extract_video(sample)
        raise ValueError(f"Unsupported media_type={sample.media_type!r}")

    def _extract_image(self, sample: EvaluationSample) -> list[dict[str, object]]:
        frame = self._cv2.imread(str(sample.path))
        if frame is None:
            raise RuntimeError(f"Cannot read image: {sample.path}")
        prepared = self._prepare_frame(frame)
        rgb = self._cv2.cvtColor(prepared, self._cv2.COLOR_BGR2RGB)
        landmarks = self._detect_landmarks(rgb)
        return [_frame_payload(1, landmarks)] if landmarks is not None else []

    def _extract_video(self, sample: EvaluationSample) -> list[dict[str, object]]:
        capture = self._cv2.VideoCapture(str(sample.path))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {sample.path}")

        frames: list[dict[str, object]] = []
        decoded_frames = 0
        processed_frames = 0
        try:
            while self._max_frames is None or processed_frames < self._max_frames:
                success, frame = capture.read()
                if not success:
                    break
                decoded_frames += 1
                if sample.start_frame is not None and decoded_frames < sample.start_frame:
                    continue
                if sample.end_frame is not None and decoded_frames > sample.end_frame:
                    break
                segment_start = sample.start_frame or 1
                if (decoded_frames - segment_start) % self._frame_stride != 0:
                    continue

                prepared = self._prepare_frame(frame)
                rgb = self._cv2.cvtColor(prepared, self._cv2.COLOR_BGR2RGB)
                landmarks = self._detect_landmarks(rgb)
                processed_frames += 1
                if landmarks is not None:
                    frames.append(_frame_payload(decoded_frames, landmarks))
        finally:
            capture.release()
        return frames

    def _detect_landmarks(self, rgb_frame: Any) -> list[tuple[float, float, float]] | None:
        detections = self._detector.detect(rgb_frame)
        if not detections:
            return None
        return max(detections, key=lambda item: item.score).landmarks

    def _prepare_frame(self, frame: Any) -> Any:
        resized = self._cv2.resize(
            frame,
            (self._config.video.frame_width, self._config.video.frame_height),
        )
        return self._cv2.flip(resized, 1) if self._config.video.mirror_frame else resized


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Extract MediaPipe hand landmarks from an image/video manifest and write "
            "a new landmark-only manifest for faster training and evaluation."
        )
    )
    parser.add_argument("--manifest", required=True, help="Input image/video manifest CSV.")
    parser.add_argument("--output-manifest", required=True, help="Output landmark manifest CSV.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where per-sample landmark JSON files will be written.",
    )
    parser.add_argument("--frame-stride", type=int, default=2, help="Process every Nth frame.")
    parser.add_argument("--max-frames", type=int, default=90, help="Maximum processed frames.")
    parser.add_argument("--frame-width", type=int, default=480, help="Frame resize width.")
    parser.add_argument("--frame-height", type=int, default=640, help="Frame resize height.")
    parser.add_argument("--mirror-frame", action="store_true", help="Mirror frames first.")
    parser.add_argument("--limit", type=int, default=None, help="Export only the first N samples.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing per-sample JSON files and only process missing samples.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Write empty JSON records for failed samples instead of stopping.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run landmark export."""

    args = build_parser().parse_args(argv)
    _validate_args(args)
    samples = read_manifest(Path(args.manifest))
    if args.limit is not None:
        samples = samples[: args.limit]
    output_manifest = Path(args.output_manifest)
    output_dir = Path(args.output_dir)
    config = AppConfig(
        video=VideoConfig(
            frame_width=args.frame_width,
            frame_height=args.frame_height,
            mirror_frame=args.mirror_frame,
        )
    )
    exporter = LandmarkManifestExporter(
        config,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
    )
    rows: list[dict[str, object]] = []

    try:
        for sample in samples:
            sample_output = output_dir / f"{sample.sample_id}.json"
            try:
                if args.skip_existing and sample_output.exists():
                    frame_count, note = _existing_landmark_result(sample_output)
                else:
                    frame_count = exporter.export_sample(sample, sample_output)
                    note = ""
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                sample_output.parent.mkdir(parents=True, exist_ok=True)
                sample_output.write_text(
                    json.dumps(
                        _landmark_payload(sample, [], note=f"error: {exc}"),
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                frame_count = 0
                note = f"error: {exc}"
            rows.append(_manifest_row(output_manifest, sample, sample_output, frame_count, note))
    finally:
        exporter.close()

    _write_manifest(output_manifest, rows)
    print(f"Wrote landmark manifest: {output_manifest}")
    print(f"Wrote landmark JSON files: {output_dir}")
    return 0


def _landmark_payload(
    sample: EvaluationSample,
    frames: list[dict[str, object]],
    note: str = "",
) -> dict[str, object]:
    return {
        "sample_id": sample.sample_id,
        "source_path": str(sample.path),
        "expected_gesture": sample.expected_gesture,
        "dataset": sample.dataset,
        "condition": sample.condition,
        "distance": sample.distance,
        "frames": frames,
        "frame_count": len(frames),
        "note": note,
    }


def _existing_landmark_result(path: Path) -> tuple[int, str]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Existing landmark payload must be a JSON object: {path}")
    frame_count = payload.get("frame_count")
    if frame_count is None:
        frames = payload.get("frames", [])
        frame_count = len(frames) if isinstance(frames, list) else 0
    return int(frame_count), str(payload.get("note") or "")


def _frame_payload(
    frame_index: int,
    landmarks: Sequence[Sequence[float]],
) -> dict[str, object]:
    return {
        "frame_index": frame_index,
        "landmarks": [
            [float(point[0]), float(point[1]), float(point[2]) if len(point) > 2 else 0.0]
            for point in landmarks
        ],
    }


def _manifest_row(
    output_manifest: Path,
    sample: EvaluationSample,
    landmark_path: Path,
    frame_count: int,
    note: str,
) -> dict[str, object]:
    return {
        "sample_id": sample.sample_id,
        "path": _relative_manifest_path(output_manifest, landmark_path),
        "expected_gesture": sample.expected_gesture,
        "media_type": "landmarks",
        "dataset": sample.dataset,
        "condition": sample.condition,
        "distance": sample.distance,
        "start_frame": "",
        "end_frame": "",
        "frame_count": frame_count,
        "note": note,
    }


def _relative_manifest_path(output_manifest: Path, target_path: Path) -> str:
    try:
        return target_path.resolve().relative_to(output_manifest.parent.resolve()).as_posix()
    except ValueError:
        return str(target_path.resolve())


def _write_manifest(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (*MANIFEST_FIELDS, "frame_count", "note")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _validate_args(args: argparse.Namespace) -> None:
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive when provided")
    if args.frame_width <= 0 or args.frame_height <= 0:
        raise ValueError("--frame-width and --frame-height must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive when provided")


if __name__ == "__main__":
    raise SystemExit(main())
