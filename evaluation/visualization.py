"""Generates ablation comparison plots (Phase 17).

Renders three static bar charts - Retrieval Precision@5, Faithfulness, and
Average Latency - one bar per ablation configuration, saved as PNG files
under `evaluation/plots/`. `matplotlib` is imported lazily (inside
`generate_ablation_plots`) so importing this module, or any other
evaluation module, never requires it to be installed.

Colors follow this project's validated categorical palette (a fixed hue
per configuration, in the same order across all three charts, run through
the palette validator for CVD-safe adjacent contrast) rather than
matplotlib's default color cycle, so a given configuration is visually
identifiable the same way across every chart. Each bar also carries a
direct value label - required here since two of the five palette slots
(aqua, yellow) fall under the 3:1 contrast floor against the light chart
surface, so color alone is not a reliable read.
"""

from __future__ import annotations

from pathlib import Path

from core.exceptions import EvaluationError
from core.logging import get_logger
from evaluation.ablation import AblationReport

logger = get_logger(__name__)

DEFAULT_PLOTS_DIR: Path = Path(__file__).resolve().parent / "plots"

# Fixed categorical slots (validated CVD-safe order - see this project's
# dataviz skill palette), assigned to configurations positionally: the
# Nth configuration always gets the Nth color, consistent across charts.
_CATEGORICAL_COLORS: tuple[str, ...] = (
    "#2a78d6",  # blue
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#008300",  # green
    "#4a3aa7",  # violet
)

_SURFACE = "#fcfcfb"
_INK_PRIMARY = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#898781"
_GRIDLINE = "#e1e0d9"
_BASELINE = "#c3c2b7"


def _bar_chart(
    names: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    value_format: str,
    output_path: Path,
) -> None:
    """Render one categorical bar chart and save it as a PNG.

    Args:
        names: Bar labels (ablation configuration names), in display order.
        values: One value per name.
        title: Chart title.
        ylabel: Y-axis label.
        value_format: `str.format`-style format string for each bar's
            direct value label (e.g. ``"{:.2f}"``).
        output_path: Where to save the rendered PNG.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = [_CATEGORICAL_COLORS[i % len(_CATEGORICAL_COLORS)] for i in range(len(names))]

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=_SURFACE)
    ax.set_facecolor(_SURFACE)

    bars = ax.bar(range(len(names)), values, color=colors, width=0.6)

    ax.set_title(title, color=_INK_PRIMARY, fontsize=13, pad=12)
    ax.set_ylabel(ylabel, color=_INK_SECONDARY, fontsize=10)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", color=_INK_SECONDARY, fontsize=9)
    ax.tick_params(axis="y", colors=_INK_MUTED, labelsize=9)

    ax.yaxis.grid(True, color=_GRIDLINE, linewidth=1)
    ax.set_axisbelow(True)
    for spine_name in ("top", "right", "left"):
        ax.spines[spine_name].set_visible(False)
    ax.spines["bottom"].set_color(_BASELINE)

    for bar, value in zip(bars, values, strict=True):
        ax.annotate(
            value_format.format(value),
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=_INK_PRIMARY,
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor=_SURFACE)
    plt.close(fig)


def generate_ablation_plots(report: AblationReport, output_dir: Path | None = None) -> list[Path]:
    """Render Retrieval Precision@5, Faithfulness, and Average Latency comparison plots.

    Args:
        report: The ablation report to visualize.
        output_dir: Directory plots are written to. Defaults to
            `DEFAULT_PLOTS_DIR` (``evaluation/plots``).

    Returns:
        The paths of every PNG file written, one per metric.

    Raises:
        EvaluationError: If `report` has no results, `matplotlib` is not
            installed, or a plot cannot be rendered/saved.
    """
    if not report.results:
        raise EvaluationError("generate_ablation_plots requires at least one result")

    destination_dir = output_dir or DEFAULT_PLOTS_DIR
    try:
        destination_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise EvaluationError(f"Failed to create plots directory {destination_dir}: {exc}") from exc

    names = [result.config_name for result in report.results]
    metric_specs: tuple[tuple[str, str, str, str, list[float]], ...] = (
        (
            "retrieval_precision.png",
            "Retrieval Precision@5 by Configuration",
            "Precision@5",
            "{:.2f}",
            [result.retrieval_precision_at_5 for result in report.results],
        ),
        (
            "faithfulness.png",
            "Faithfulness by Configuration",
            "Faithfulness",
            "{:.2f}",
            [result.faithfulness for result in report.results],
        ),
        (
            "latency.png",
            "Average Latency by Configuration",
            "Latency (ms)",
            "{:.0f}",
            [result.average_latency_ms for result in report.results],
        ),
    )

    paths: list[Path] = []
    try:
        for filename, title, ylabel, value_format, values in metric_specs:
            output_path = destination_dir / filename
            _bar_chart(names, values, title, ylabel, value_format, output_path)
            paths.append(output_path)
    except ImportError as exc:
        raise EvaluationError(f"matplotlib is required to generate plots: {exc}") from exc
    except EvaluationError:
        raise
    except Exception as exc:  # noqa: BLE001 - third-party plotting boundary
        raise EvaluationError(f"Failed to generate ablation plots: {exc}") from exc

    logger.info("Plots generated: %d file(s) in %s", len(paths), destination_dir)
    return paths
