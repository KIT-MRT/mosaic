"""Plot subcommand — generate a behavior selection pie chart from cost estimator logs."""

from pathlib import Path
from typing import Union

import click
import matplotlib.pyplot as plt

from mosaic.cli.analyze.data import find_latest_experiment
from mosaic.cli.analyze.logs import load_cost_estimator_counts

# Catppuccin Latte palette
COLOR_SAPPHIRE = "#209fb5"
COLOR_PEACH = "#fe640b"
COLOR_LAVENDER = "#7287fd"


@click.command()
@click.option(
    "--path",
    "-p",
    type=click.Path(exists=True),
    default=None,
    help="Path to experiment output dir (auto-detects latest if omitted).",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="behavior_selection.svg",
    help="Output SVG path.",
)
def plot(path: Union[str, None], output: str) -> None:
    """Generate a behavior selection pie chart from cost estimator logs."""
    if path is None:
        experiment_dir = find_latest_experiment()
        click.echo(f"Using latest experiment: {experiment_dir}")
    else:
        experiment_dir = Path(path)

    mosaic_dir = experiment_dir / "mosaic_logs"
    if not mosaic_dir.exists():
        raise click.ClickException(f"No mosaic_logs directory in {experiment_dir}")

    command_wins, _appearances, _all_commands, tie_count = load_cost_estimator_counts(
        mosaic_dir
    )

    total = sum(command_wins.values()) + tie_count
    if total == 0:
        raise click.ClickException("No cost estimator entries found.")

    # Build slices: one per command winner + one for ties
    labels = []
    sizes = []
    colors = []

    # Map log command names to display labels and colors
    display_names = {
        "flow_drive": "FlowDrive*",
        "pdm_closed": "PDM-Closed",
    }
    command_colors = {
        "flow_drive": COLOR_SAPPHIRE,
        "pdm_closed": COLOR_PEACH,
    }

    for cmd in sorted(command_wins):
        labels.append(display_names.get(cmd, cmd))
        sizes.append(command_wins[cmd])
        colors.append(command_colors.get(cmd, COLOR_LAVENDER))

    labels.append("Tied")
    sizes.append(tie_count)
    colors.append(COLOR_LAVENDER)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        counterclock=False,
    )
    ax.set_aspect("equal")

    plt.rcParams["svg.fonttype"] = "none"
    fig.savefig(output, format="svg", bbox_inches="tight")
    plt.close(fig)
    click.echo(f"Saved pie chart to {output}")
