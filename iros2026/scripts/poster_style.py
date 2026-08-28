"""Shared colours, paths and figure defaults."""

from pathlib import Path
from typing import Final

SCRIPTS_DIR: Final = Path(__file__).resolve().parent
# Assets are shared with the presentation; single source of truth at the repo root.
FIGURES_DIR: Final = SCRIPTS_DIR.parent / "assets" / "figures"

INK: Final = "#1e1e2e"
PEACH: Final = "#fe640b"
BLUE: Final = "#1e66f5"
SURFACE: Final = "#ccd0da"
TEAL: Final = "#179299"

FONT_NAME: Final = "DejaVu Sans"


def matplotlib_rc_params() -> dict[str, object]:
    return {
        "font.family": "sans-serif",
        "font.sans-serif": [FONT_NAME],
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.edgecolor": INK,
        # Glyphs as paths so the SVG renders without the font installed.
        "svg.fonttype": "path",
    }


def figure_height_for(
    width_inches: float, box_width: float, box_height: float
) -> float:
    return width_inches * box_height / box_width
