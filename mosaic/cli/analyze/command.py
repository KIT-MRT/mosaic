"""Click command definition and orchestration for the analyze subcommand."""

import json as json_mod
from pathlib import Path
from typing import Union

import click

from mosaic.cli.analyze.data import find_latest_experiment, load_collision_details, load_results
from mosaic.cli.analyze.export import build_export
from mosaic.cli.analyze.runtime import get_runtime
from mosaic.cli.analyze.summary import print_report


@click.command()
@click.option(
    "--path",
    "-p",
    type=click.Path(exists=True),
    default=None,
    help="Path to experiment output dir (auto-detects latest if omitted).",
)
@click.option(
    "--per-type / --no-per-type",
    default=True,
    help="Show per-scenario-type breakdown.",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    default=False,
    help="Output as machine-readable JSON.",
)
def analyze(
    path: Union[str, None],
    per_type: bool,
    output_json: bool,
) -> None:
    """Analyze simulation results from nuplan metric parquets."""
    if path is None:
        experiment_dir = find_latest_experiment()
        if not output_json:
            click.echo(f"Using latest experiment: {experiment_dir}")
    else:
        experiment_dir = Path(path)

    df = load_results(experiment_dir)
    df = load_collision_details(experiment_dir, df)
    runtime = get_runtime(experiment_dir, len(df))
    mosaic_dir = experiment_dir / "mosaic_logs"

    if output_json:
        click.echo(json_mod.dumps(build_export(df, experiment_dir.name, runtime, mosaic_dir), indent=2))
    else:
        print_report(df, experiment_dir.name, runtime, mosaic_dir, per_type)
