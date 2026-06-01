"""Generate thesis figures from benchmark CSV outputs."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""

    parser = argparse.ArgumentParser(description="Generate benchmark figures for thesis section 5.")
    parser.add_argument(
        "--input",
        "--summary",
        dest="summary",
        required=True,
        help="Summary CSV produced by scripts/benchmark.py.",
    )
    parser.add_argument(
        "--per-class",
        help="Per-class CSV. Defaults to <summary_stem>_per_class.csv.",
    )
    parser.add_argument(
        "--confusion",
        help="Confusion matrix CSV. Defaults to <summary_stem>_confusion_matrix.csv.",
    )
    parser.add_argument("--output", required=True, help="Directory for generated PNG figures.")
    parser.add_argument("--group", default="ALL", help="Group to plot for class-level figures.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Generate PNG charts from benchmark outputs."""

    args = build_parser().parse_args(argv)
    summary_path = Path(args.summary)
    per_class_path = (
        Path(args.per_class)
        if args.per_class
        else summary_path.with_name(f"{summary_path.stem}_per_class.csv")
    )
    confusion_path = (
        Path(args.confusion)
        if args.confusion
        else summary_path.with_name(f"{summary_path.stem}_confusion_matrix.csv")
    )
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = _read_rows(summary_path)
    _plot_macro_f1(summary_rows, output_dir / "benchmark_macro_f1.png")
    _plot_runtime(summary_rows, output_dir / "benchmark_runtime.png")

    if per_class_path.exists():
        _plot_per_class_f1(
            _read_rows(per_class_path),
            output_dir / "benchmark_per_class_f1.png",
            args.group,
        )
    if confusion_path.exists():
        _plot_confusion_matrix(
            _read_rows(confusion_path),
            output_dir / "benchmark_confusion_matrix.png",
            args.group,
        )

    print(f"Wrote figures to: {output_dir}")
    return 0


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _plot_macro_f1(rows: Sequence[dict[str, str]], path: Path) -> None:
    if not rows:
        return
    groups = [row["group"] for row in rows]
    values = [_float(row.get("macro_f1")) for row in rows]

    fig, ax = plt.subplots(figsize=(max(7, len(groups) * 1.2), 4.5))
    ax.bar(groups, values, color="#2f6f8f")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Macro-F1 за групами експерименту")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_runtime(rows: Sequence[dict[str, str]], path: Path) -> None:
    runtime_rows = [
        row
        for row in rows
        if row.get("mean_latency_ms") not in (None, "") or row.get("mean_fps") not in (None, "")
    ]
    if not runtime_rows:
        return

    groups = [row["group"] for row in runtime_rows]
    latencies = [_float(row.get("mean_latency_ms")) for row in runtime_rows]
    fps_values = [_float(row.get("mean_fps")) for row in runtime_rows]

    fig, axes = plt.subplots(1, 2, figsize=(max(9, len(groups) * 1.4), 4.5))
    axes[0].bar(groups, latencies, color="#8f4f2f")
    axes[0].set_ylabel("Latency, ms")
    axes[0].set_title("Середня затримка")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(groups, fps_values, color="#3f7f4f")
    axes[1].set_ylabel("FPS")
    axes[1].set_title("Середня частота кадрів")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_per_class_f1(rows: Sequence[dict[str, str]], path: Path, group: str) -> None:
    selected_rows = [
        row for row in rows if row.get("group") == group and row.get("label") != "UNKNOWN"
    ]
    if not selected_rows:
        return
    labels = [row["label"] for row in selected_rows]
    values = [_float(row.get("f1")) for row in selected_rows]

    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 0.75), 4.8))
    ax.bar(labels, values, color="#536c9d")
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1-score")
    ax.set_title(f"F1-score за жестами ({group})")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_confusion_matrix(rows: Sequence[dict[str, str]], path: Path, group: str) -> None:
    selected_rows = [row for row in rows if row.get("group") == group]
    if not selected_rows:
        return
    labels = sorted(
        {row["expected_label"] for row in selected_rows}
        | {row["predicted_label"] for row in selected_rows}
    )
    matrix = [[0 for _ in labels] for _ in labels]
    label_to_index = {label: index for index, label in enumerate(labels)}
    for row in selected_rows:
        expected_index = label_to_index[row["expected_label"]]
        predicted_index = label_to_index[row["predicted_label"]]
        matrix[expected_index][predicted_index] = int(row["count"])

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.55), max(6, len(labels) * 0.5)))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Передбачений жест")
    ax.set_ylabel("Очікуваний жест")
    ax.set_title(f"Матриця помилок ({group})")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    max_value = max((max(row) for row in matrix), default=0)
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            if value == 0:
                continue
            color = "white" if value > max_value * 0.55 else "black"
            ax.text(column_index, row_index, str(value), ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _float(value: str | None) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
