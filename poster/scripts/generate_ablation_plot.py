"""Bar chart of at-fault collisions and zero-score scenarios across ablations."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np

import poster_style


@dataclass(frozen=True)
class Series:
    label: str
    values: tuple[int, ...]
    colour: str


CONFIGURATIONS: Final = (
    "Mosaic\nw/o verif.",
    "PDM-Closed\nonly",
    "FlowDrive*\nonly",
    "Mosaic\n(full)",
)

SERIES: Final = (
    Series("At-fault collisions", (40, 17, 15, 16), poster_style.PEACH),
    Series("Zero-score scenarios", (38, 28, 23, 17), poster_style.BLUE),
)

TOTAL_SCENARIOS: Final = 1118

# Aspect of the target box, so the figure is placed at 1:1 rather than scaled.
BOX_CONTENT_WIDTH: Final = 1133.8
BOX_CONTENT_HEIGHT: Final = 704.4
FIGURE_WIDTH_INCHES: Final = 7.6

BAR_WIDTH: Final = 0.34
Y_AXIS_HEADROOM: Final = 8
LABEL_FONT_SIZE: Final = 13
TICK_FONT_SIZE: Final = 12.5


def render(output_path: Path) -> None:
    plt.style.use(poster_style.matplotlib_rc_params())

    height = poster_style.figure_height_for(
        FIGURE_WIDTH_INCHES, BOX_CONTENT_WIDTH, BOX_CONTENT_HEIGHT
    )
    figure, axes = plt.subplots(figsize=(FIGURE_WIDTH_INCHES, height))

    positions = np.arange(len(CONFIGURATIONS))
    offsets = (-BAR_WIDTH / 2, BAR_WIDTH / 2)

    for series, offset in zip(SERIES, offsets, strict=True):
        bar_positions = positions + offset
        axes.bar(
            bar_positions,
            series.values,
            BAR_WIDTH,
            color=series.colour,
            label=series.label,
            zorder=3,
        )
        for x, value in zip(bar_positions, series.values, strict=True):
            axes.text(
                x,
                value + 1.4,
                str(value),
                ha="center",
                va="bottom",
                fontsize=LABEL_FONT_SIZE,
                fontweight="bold",
                color=series.colour,
                zorder=5,
            )

    tallest = max(value for series in SERIES for value in series.values)
    axes.set_ylim(0, tallest + Y_AXIS_HEADROOM)
    axes.set_xticks(list(positions))
    axes.set_xticklabels(CONFIGURATIONS, fontsize=TICK_FONT_SIZE)
    axes.set_ylabel(f"Scenarios (of {TOTAL_SCENARIOS})", fontsize=TICK_FONT_SIZE)
    axes.tick_params(axis="y", labelsize=TICK_FONT_SIZE - 1)
    axes.grid(axis="y", color=poster_style.SURFACE, linewidth=0.9, zorder=0)
    axes.set_axisbelow(True)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.legend(
        fontsize=TICK_FONT_SIZE,
        frameon=False,
        loc="upper right",
        handlelength=1.4,
        borderaxespad=0.2,
    )

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, format="svg", bbox_inches="tight", transparent=True)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=poster_style.FIGURES_DIR / "ablation.svg",
        help="where to write the SVG (default: %(default)s)",
    )
    arguments = parser.parse_args()
    render(arguments.output)
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
