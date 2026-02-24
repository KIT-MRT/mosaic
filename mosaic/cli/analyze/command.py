"""Click command definition and orchestration for the analyze subcommand."""

from pathlib import Path
from typing import Union

import click

from mosaic.cli.analyze.comparison import print_comparison
from mosaic.cli.analyze.data import find_latest_experiment, load_results
from mosaic.cli.analyze.logs import analyze_cost_estimator_logs, analyze_verifier_logs
from mosaic.cli.analyze.runtime import get_runtime
from mosaic.cli.analyze.summary import (
    print_collision_scenarios,
    print_failures,
    print_header,
    print_per_type,
)


@click.command()
@click.option(
    "--path",
    "-p",
    type=click.Path(exists=True),
    default=None,
    help="Path to experiment output dir (auto-detects latest if omitted).",
)
@click.option(
    "--baseline",
    "-b",
    type=click.Path(exists=True),
    default=None,
    help="Path to baseline experiment dir for comparison.",
)
@click.option(
    "--per-type / --no-per-type",
    default=True,
    help="Show per-scenario-type breakdown.",
)
def analyze(path: Union[str, None], baseline: Union[str, None], per_type: bool) -> None:
    """Analyze simulation results from nuplan metric parquets."""
    if path is None:
        experiment_dir = find_latest_experiment()
        click.echo(f"Using latest experiment: {experiment_dir}")
    else:
        experiment_dir = Path(path)

    df = load_results(experiment_dir)
    runtime = get_runtime(experiment_dir, len(df))

    print_header(df, experiment_dir.name, runtime)
    print_failures(df)
    print_collision_scenarios(df)

    if per_type:
        print_per_type(df)

    if baseline is not None:
        baseline_df = load_results(Path(baseline))
        print_comparison(df, baseline_df)

    mosaic_dir = experiment_dir / "mosaic_logs"
    if not mosaic_dir.exists():
        return

    analyze_cost_estimator_logs(mosaic_dir)
    analyze_verifier_logs(mosaic_dir)
