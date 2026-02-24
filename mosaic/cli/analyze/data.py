"""Data loading and metric constants for the analyze command."""

import os
from pathlib import Path

import click
from pandas import DataFrame

HARD_GATES = [
    "no_ego_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "ego_is_making_progress",
]

WEIGHTED_METRICS = {
    "ego_progress_along_expert_route": 5,
    "time_to_collision_within_bound": 5,
    "speed_limit_compliance": 4,
    "ego_is_comfortable": 2,
}


def find_latest_experiment() -> Path:
    """Locate the most recent experiment directory with aggregator_metric/ results."""
    if "NUPLAN_EXP_ROOT" in os.environ:
        output_root = Path(os.environ["NUPLAN_EXP_ROOT"])
    else:
        output_root = Path.home() / "nuplan" / "exp"
    if not output_root.exists():
        raise click.ClickException(
            f"No output directory found at {output_root}. Run a simulation first or provide --path."
        )

    candidates = []
    for d in output_root.rglob("aggregator_metric"):
        if d.is_dir():
            candidates.append(d.parent)
    if not candidates:
        raise click.ClickException(
            f"No aggregator_metric directories found under {output_root}."
        )

    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_results(experiment_dir: Path) -> DataFrame:
    """Load per-scenario results from aggregator_metric parquet files."""
    import pandas as pd

    agg_dir = experiment_dir / "aggregator_metric"
    if not agg_dir.exists():
        raise click.ClickException(
            f"No aggregator_metric directory in {experiment_dir}"
        )

    parquet_files = list(agg_dir.glob("*.parquet"))
    if not parquet_files:
        raise click.ClickException(f"No parquet files in {agg_dir}")

    df = pd.read_parquet(parquet_files[0])

    # Drop nuplan aggregate rows (per-type summaries + final_score)
    aggregate_names = set(df["scenario_type"].unique())
    aggregate_names.add("final_score")
    df = df[~df["scenario"].isin(aggregate_names)].reset_index(drop=True)

    return df
