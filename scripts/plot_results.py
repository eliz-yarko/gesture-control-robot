"""Generate thesis figures from benchmark CSV outputs."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path
from textwrap import wrap

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

COLORS = {
    "blue_base": "#A3BEFA",
    "blue_mid": "#5477C4",
    "blue_dark": "#2E4780",
    "gold_base": "#FFE15B",
    "orange_base": "#F0986E",
    "olive_base": "#A3D576",
    "pink_base": "#F390CA",
    "neutral_base": "#C5CAD3",
    "neutral_dark": "#464C55",
}

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "macro_f1": "Macro-F1",
    "weighted_f1": "Weighted-F1",
    "critical_false_positive_rate": "Critical FPR",
    "unknown_rate": "Unknown rate",
}

DEFAULT_COMPARISON_METRICS = ("macro_f1", "accuracy", "weighted_f1")
METRIC_COLORS = (
    COLORS["blue_base"],
    COLORS["gold_base"],
    COLORS["pink_base"],
    COLORS["olive_base"],
    COLORS["orange_base"],
)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""

    parser = argparse.ArgumentParser(description="Generate benchmark figures for thesis section 5.")
    parser.add_argument(
        "--input",
        "--summary",
        dest="summary",
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
    parser.add_argument("--output", required=True, help="Directory for generated PNG/SVG figures.")
    parser.add_argument("--group", default="ALL", help="Group to plot for class-level figures.")
    parser.add_argument(
        "--comparison",
        action="append",
        default=[],
        metavar="LABEL=CSV",
        help="Add a benchmark summary CSV to the comparison figure. Can be repeated.",
    )
    parser.add_argument(
        "--comparison-metrics",
        default=",".join(DEFAULT_COMPARISON_METRICS),
        help="Comma-separated summary metrics for comparison figures.",
    )
    parser.add_argument(
        "--comparison-title",
        default="Порівняння якості режимів оцінювання",
        help="Title for the comparison figure.",
    )
    parser.add_argument(
        "--comparison-subtitle",
        default="Значення метрик у діапазоні 0-1; вище значення означає кращу якість.",
        help="Subtitle for the comparison figure.",
    )
    parser.add_argument(
        "--skip-individual",
        action="store_true",
        help="Only generate comparison figures from --comparison inputs.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Generate PNG and SVG charts from benchmark outputs."""

    args = build_parser().parse_args(argv)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.summary is None and not args.comparison:
        raise ValueError("Provide --input/--summary, at least one --comparison, or both.")

    if args.summary is not None and not args.skip_individual:
        _generate_individual_figures(
            summary_path=Path(args.summary),
            per_class_path=Path(args.per_class) if args.per_class else None,
            confusion_path=Path(args.confusion) if args.confusion else None,
            output_dir=output_dir,
            group=args.group,
        )

    if args.comparison:
        metrics = _parse_metric_list(args.comparison_metrics)
        _plot_comparison(
            _comparison_rows(args.comparison, args.group),
            output_dir / "benchmark_comparison.png",
            metrics=metrics,
            title=args.comparison_title,
            subtitle=args.comparison_subtitle,
        )

    print(f"Wrote figures to: {output_dir}")
    return 0


def _generate_individual_figures(
    *,
    summary_path: Path,
    per_class_path: Path | None,
    confusion_path: Path | None,
    output_dir: Path,
    group: str,
) -> None:
    per_class_path = per_class_path or summary_path.with_name(f"{summary_path.stem}_per_class.csv")
    confusion_path = confusion_path or summary_path.with_name(
        f"{summary_path.stem}_confusion_matrix.csv"
    )

    summary_rows = _read_rows(summary_path)
    _plot_macro_f1(summary_rows, output_dir / "benchmark_macro_f1.png")
    _plot_runtime(summary_rows, output_dir / "benchmark_runtime.png")

    if per_class_path.exists():
        _plot_per_class_f1(
            _read_rows(per_class_path),
            output_dir / "benchmark_per_class_f1.png",
            group,
        )
    if confusion_path.exists():
        _plot_confusion_matrix(
            _read_rows(confusion_path),
            output_dir / "benchmark_confusion_matrix.png",
            group,
        )


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _plot_macro_f1(rows: Sequence[dict[str, str]], path: Path) -> None:
    if not rows:
        return
    groups = [row["group"] for row in rows]
    values = [_float(row.get("macro_f1")) for row in rows]

    fig, ax = _figure(figsize=(max(7.5, len(groups) * 1.2), 4.8))
    ax.bar(groups, values, color=COLORS["blue_base"], edgecolor=COLORS["blue_dark"], linewidth=1)
    _style_value_axis(ax, upper=1.0)
    ax.set_ylabel("Macro-F1", color=TOKENS["ink"])
    _add_value_labels(ax, values)
    _add_header(
        fig,
        "Macro-F1 за групами експерименту",
        "Значення у діапазоні 0-1; вище значення означає кращий баланс precision та recall.",
    )
    ax.tick_params(axis="x", rotation=25)
    _save_figure(fig, path)


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

    fig, axes = _figure(
        figsize=(max(10.0, len(groups) * 1.4), 5.0),
        axes=(1, 2),
    )
    latency_ax, fps_ax = axes
    latency_ax.bar(
        groups,
        latencies,
        color=COLORS["orange_base"],
        edgecolor=COLORS["neutral_dark"],
        linewidth=1,
    )
    latency_ax.set_ylabel("Затримка, мс")
    latency_ax.set_title("Середня затримка", color=TOKENS["ink"], pad=8)
    _style_value_axis(latency_ax)
    _add_value_labels(latency_ax, latencies, suffix=" мс", decimals=1)

    fps_ax.bar(
        groups,
        fps_values,
        color=COLORS["olive_base"],
        edgecolor=COLORS["neutral_dark"],
        linewidth=1,
    )
    fps_ax.set_ylabel("FPS")
    fps_ax.set_title("Середня частота кадрів", color=TOKENS["ink"], pad=8)
    _style_value_axis(fps_ax)
    _add_value_labels(fps_ax, fps_values, decimals=1)

    for ax in axes:
        ax.tick_params(axis="x", rotation=25)

    _add_header(
        fig,
        "Швидкодія benchmark-прогону",
        "Mean latency та mean FPS з CSV scripts/benchmark.py.",
    )
    _save_figure(fig, path)


def _plot_per_class_f1(rows: Sequence[dict[str, str]], path: Path, group: str) -> None:
    selected_rows = [
        row
        for row in rows
        if row.get("group") == group
        and row.get("label") != "UNKNOWN"
        and int(row.get("support") or 0) > 0
    ]
    if not selected_rows:
        return
    selected_rows = sorted(selected_rows, key=lambda row: _float(row.get("f1")))
    labels = [_wrapped_label(row["label"]) for row in selected_rows]
    values = [_float(row.get("f1")) for row in selected_rows]

    fig, ax = _figure(figsize=(9.8, max(5.0, len(labels) * 0.36)))
    y_positions = list(range(len(labels)))
    ax.barh(
        y_positions,
        values,
        color=COLORS["blue_base"],
        edgecolor=COLORS["blue_dark"],
        linewidth=1,
    )
    ax.set_yticks(y_positions, labels=labels)
    ax.set_xlabel("F1-score")
    _style_value_axis(ax, upper=1.0, horizontal=True)
    _add_horizontal_value_labels(ax, values)
    _add_header(
        fig,
        f"F1-score за жестами ({group})",
        "Показано тільки класи з ненульовою підтримкою у тестовому наборі.",
    )
    _save_figure(fig, path)


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

    fig, ax = _figure(figsize=(max(8.8, len(labels) * 0.58), max(7.2, len(labels) * 0.52)))
    image = ax.imshow(matrix, cmap="Blues", vmin=0)
    ax.set_xticks(range(len(labels)), labels=[_wrapped_label(label) for label in labels])
    ax.set_yticks(range(len(labels)), labels=[_wrapped_label(label) for label in labels])
    ax.tick_params(axis="x", rotation=45)
    ax.set_xlabel("Передбачений жест")
    ax.set_ylabel("Очікуваний жест")
    for spine in ax.spines.values():
        spine.set_color(TOKENS["axis"])
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.outline.set_edgecolor(TOKENS["axis"])

    max_value = max((max(row) for row in matrix), default=0)
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            if value == 0:
                continue
            color = TOKENS["panel"] if value > max_value * 0.55 else TOKENS["ink"]
            ax.text(column_index, row_index, str(value), ha="center", va="center", color=color)

    _add_header(
        fig,
        f"Матриця помилок ({group})",
        "Кількість прикладів у клітинці: очікуваний клас проти передбаченого класу.",
    )
    _save_figure(fig, path)


def _plot_comparison(
    rows: Sequence[dict[str, str]],
    path: Path,
    *,
    metrics: Sequence[str],
    title: str,
    subtitle: str,
) -> None:
    if not rows:
        return
    labels = [_wrapped_label(row["label"], width=24) for row in rows]
    y_positions = list(range(len(rows)))
    bar_height = min(0.22, 0.72 / max(len(metrics), 1))

    fig, ax = _figure(figsize=(11.4, max(5.2, len(rows) * 0.62)))
    for metric_index, metric in enumerate(metrics):
        offsets = [
            position + (metric_index - (len(metrics) - 1) / 2) * bar_height
            for position in y_positions
        ]
        values = [_float(row.get(metric)) for row in rows]
        ax.barh(
            offsets,
            values,
            height=bar_height,
            label=METRIC_LABELS.get(metric, metric),
            color=METRIC_COLORS[metric_index % len(METRIC_COLORS)],
            edgecolor=COLORS["neutral_dark"],
            linewidth=0.8,
        )
        for offset, value in zip(offsets, values, strict=True):
            ax.text(
                value + 0.015,
                offset,
                f"{value:.3f}",
                va="center",
                ha="left",
                fontsize=9,
                color=TOKENS["ink"],
            )

    ax.set_yticks(y_positions, labels=labels)
    ax.set_xlabel("Значення метрики")
    ax.set_xlim(0, 1.08)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 1.10),
        ncol=min(len(metrics), 4),
        frameon=False,
    )
    _style_value_axis(ax, upper=1.0, horizontal=True)
    _add_header(fig, title, subtitle)
    _save_figure(fig, path, top=0.82)


def _comparison_rows(values: Sequence[str], group: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for value in values:
        label, separator, path_value = value.partition("=")
        if not separator or not label.strip() or not path_value.strip():
            raise ValueError("--comparison must use LABEL=CSV format")
        summary_rows = _read_rows(Path(path_value.strip()))
        match = next((row for row in summary_rows if row.get("group") == group), None)
        if match is None:
            raise ValueError(f"Group {group!r} is not present in {path_value!r}")
        rows.append({"label": label.strip(), **match})
    return rows


def _parse_metric_list(value: str) -> tuple[str, ...]:
    metrics = tuple(metric.strip() for metric in value.split(",") if metric.strip())
    return metrics or DEFAULT_COMPARISON_METRICS


def _figure(
    *,
    figsize: tuple[float, float],
    axes: tuple[int, int] | None = None,
):
    plt.rcParams.update(
        {
            "font.family": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
            "axes.labelcolor": TOKENS["ink"],
            "xtick.color": TOKENS["muted"],
            "ytick.color": TOKENS["muted"],
        }
    )
    if axes is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor(TOKENS["panel"])
        fig.patch.set_facecolor(TOKENS["surface"])
        return fig, ax

    fig, axis_grid = plt.subplots(*axes, figsize=figsize)
    flat_axes = list(axis_grid.flat) if hasattr(axis_grid, "flat") else [axis_grid]
    fig.patch.set_facecolor(TOKENS["surface"])
    for ax in flat_axes:
        ax.set_facecolor(TOKENS["panel"])
    return fig, flat_axes


def _add_header(fig, title: str, subtitle: str) -> None:
    fig.text(0.02, 0.965, title, ha="left", va="top", fontsize=15, color=TOKENS["ink"])
    fig.text(0.02, 0.925, subtitle, ha="left", va="top", fontsize=10, color=TOKENS["muted"])


def _style_value_axis(ax, upper: float | None = None, *, horizontal: bool = False) -> None:
    if upper is not None:
        if horizontal:
            ax.set_xlim(0, upper)
        else:
            ax.set_ylim(0, upper)
    ax.grid(axis="x" if horizontal else "y", color=TOKENS["grid"], linewidth=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])


def _add_value_labels(
    ax,
    values: Sequence[float],
    *,
    suffix: str = "",
    decimals: int = 3,
) -> None:
    if not values:
        return
    label_offset = max(values, default=1.0) * 0.03
    for bar, value in zip(ax.patches, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + label_offset,
            f"{value:.{decimals}f}{suffix}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=TOKENS["ink"],
        )


def _add_horizontal_value_labels(ax, values: Sequence[float]) -> None:
    for bar, value in zip(ax.patches, values, strict=True):
        ax.text(
            value + 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            ha="left",
            va="center",
            fontsize=9,
            color=TOKENS["ink"],
        )


def _save_figure(fig, path: Path, *, top: float = 0.88) -> None:
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def _wrapped_label(label: str, *, width: int = 18) -> str:
    return "\n".join(wrap(label.replace("_", " "), width=width)) or label


def _float(value: str | None) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
