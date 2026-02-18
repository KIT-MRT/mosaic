import os
from pathlib import Path
from typing import Union

import click
from pandas import DataFrame


def _find_latest_experiment() -> Path:
    if "NUPLAN_EXP_ROOT" in os.environ:
        output_root = Path(os.environ["NUPLAN_EXP_ROOT"])
    else:
        output_root = Path.home() / "nuplan" / "exp"
    if not output_root.exists():
        raise click.ClickException(
            f"No output directory found at {output_root}. Run a simulation first or provide --path."
        )

    # Find most recent directory containing aggregator_metric/
    candidates = []
    for d in output_root.rglob("aggregator_metric"):
        if d.is_dir():
            candidates.append(d.parent)
    if not candidates:
        raise click.ClickException(
            f"No aggregator_metric directories found under {output_root}."
        )

    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_results(experiment_dir: Path) -> DataFrame:
    import pandas as pd

    agg_dir = experiment_dir / "aggregator_metric"
    if not agg_dir.exists():
        raise click.ClickException(
            f"No aggregator_metric directory in {experiment_dir}"
        )

    parquet_files = list(agg_dir.glob("*.parquet"))
    if not parquet_files:
        raise click.ClickException(f"No parquet files in {agg_dir}")

    return pd.read_parquet(parquet_files[0])


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


def _truncate(string: str, width: int) -> str:
    if len(string) <= width:
        return string

    return string[: width - 1] + "…"


def _print_summary(df: DataFrame, label: str) -> None:
    click.echo(f"\n{'=' * 60}")
    click.echo(f"  {label}")
    click.echo(f"{'=' * 60}")
    click.echo(f"  Scenarios: {len(df)}")
    click.echo(f"  Overall score: {df['score'].mean():.4f}")


def _print_failures(df: DataFrame) -> None:
    click.echo(f"\n  Failure Breakdown:")
    click.echo(f"  {'Metric':<40} {'Fail':>5}  {'Rate':>6}")
    click.echo(f"  {'-' * 55}")

    for metric in HARD_GATES:
        if metric in df.columns:
            fails = (df[metric] == 0.0).sum()
            rate = fails / len(df) * 100
            marker = " ***" if fails > 0 else ""
            click.echo(f"  {metric:<40} {fails:>5}  {rate:>5.1f}%{marker}")

    for metric, weight in WEIGHTED_METRICS.items():
        if metric in df.columns:
            fails = (df[metric] == 0.0).sum()
            rate = fails / len(df) * 100
            marker = " ***" if fails > 0 else ""
            click.echo(
                f"  {metric:<40} {fails:>5}  {rate:>5.1f}%  (w={weight}){marker}"
            )

    zero_score = (df["score"] == 0.0).sum()
    click.echo(f"\n  Zero-score scenarios: {zero_score}")


def _print_per_type(df: DataFrame) -> None:
    click.echo(f"\n  Per-Scenario-Type Breakdown:")
    click.echo(
        f"  {'Type':<45} {'n':>4}  {'Score':>6}  {'Zero':>4}  {'TTC=0':>5}  {'Coll':>4}"
    )
    click.echo(f"  {'-' * 75}")

    type_stats = []
    for stype, group in df.groupby("scenario_type"):
        n = len(group)
        score = group["score"].mean()
        zeros = (group["score"] == 0.0).sum()
        ttc_fail = (
            (group["time_to_collision_within_bound"] == 0.0).sum()
            if "time_to_collision_within_bound" in group.columns
            else 0
        )
        coll_fail = (
            (group["no_ego_at_fault_collisions"] == 0.0).sum()
            if "no_ego_at_fault_collisions" in group.columns
            else 0
        )
        type_stats.append((stype, n, score, zeros, ttc_fail, coll_fail))

    for stype, n, score, zeros, ttc_fail, coll_fail in sorted(
        type_stats, key=lambda x: x[2]
    ):
        click.echo(
            f"  {_truncate(stype, 45):<45} {n:>4}  {score:>6.4f}  {zeros:>4}  {ttc_fail:>5}  {coll_fail:>4}"
        )


def _print_collision_scenarios(df: DataFrame) -> None:
    if "no_ego_at_fault_collisions" not in df.columns:
        click.echo("\n  (no_ego_at_fault_collisions column not found)")
        return

    collision_df = df[df["no_ego_at_fault_collisions"] == 0.0].copy()
    if collision_df.empty:
        click.echo("\n  No collision scenarios found.")
        return

    collision_df = collision_df.sort_values("score")

    click.echo(f"\n  Collision Scenarios ({len(collision_df)}):")
    click.echo(f"  {'Scenario Token':<20} {'Type':<40} {'Score':>6}  {'Coll':>5}")
    click.echo(f"  {'-' * 75}")
    for _, row in collision_df.iterrows():
        stype = _truncate(str(row.get("scenario_type", "")), 40)
        click.echo(
            f"  {row['scenario']:<20} {stype:<40} {row['score']:>6.4f}  {row['no_ego_at_fault_collisions']:>5.2f}"
        )


def _print_comparison(df: DataFrame, baseline_df: DataFrame) -> None:
    click.echo(f"\n  Comparison with Baseline:")
    overall_diff = df["score"].mean() - baseline_df["score"].mean()
    sign = "+" if overall_diff >= 0 else ""
    click.echo(f"  Overall score diff: {sign}{overall_diff:.4f}")

    # Per-metric failure comparison
    click.echo(f"\n  {'Metric':<40} {'Now':>5}  {'Base':>5}  {'Diff':>5}")
    click.echo(f"  {'-' * 60}")

    for metric in HARD_GATES + list(WEIGHTED_METRICS.keys()):
        if metric in df.columns and metric in baseline_df.columns:
            now_fails = (df[metric] == 0.0).sum()
            base_fails = (baseline_df[metric] == 0.0).sum()
            diff = now_fails - base_fails
            sign = "+" if diff > 0 else ""
            marker = ""
            if diff < 0:
                marker = " (improved)"
            elif diff > 0:
                marker = " (regressed)"
            click.echo(
                f"  {metric:<40} {now_fails:>5}  {base_fails:>5}  {sign}{diff:>4}{marker}"
            )

    # Per-scenario comparison (matched by token)
    merged = df[["scenario", "score"]].merge(
        baseline_df[["scenario", "score"]],
        on="scenario",
        suffixes=("_new", "_base"),
        how="inner",
    )
    if len(merged) > 0:
        improved = (merged["score_new"] > merged["score_base"]).sum()
        regressed = (merged["score_new"] < merged["score_base"]).sum()
        unchanged = (merged["score_new"] == merged["score_base"]).sum()
        click.echo(f"\n  Matched scenarios: {len(merged)}")
        click.echo(
            f"  Improved: {improved}  Regressed: {regressed}  Unchanged: {unchanged}"
        )

        if regressed > 0:
            regressions = merged[merged["score_new"] < merged["score_base"]].copy()
            regressions["diff"] = regressions["score_new"] - regressions["score_base"]
            regressions = regressions.sort_values("diff")
            click.echo(f"\n  Worst regressions:")
            for _, row in regressions.head(10).iterrows():
                click.echo(
                    f"    {row['scenario']}  {row['score_base']:.4f} -> {row['score_new']:.4f}  ({row['diff']:+.4f})"
                )

        if improved > 0:
            improvements = merged[merged["score_new"] > merged["score_base"]].copy()
            improvements["diff"] = (
                improvements["score_new"] - improvements["score_base"]
            )
            improvements = improvements.sort_values("diff", ascending=False)
            click.echo(f"\n  Best improvements:")
            for _, row in improvements.head(10).iterrows():
                click.echo(
                    f"    {row['scenario']}  {row['score_base']:.4f} -> {row['score_new']:.4f}  ({row['diff']:+.4f})"
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
        experiment_dir = _find_latest_experiment()
        click.echo(f"Using latest experiment: {experiment_dir}")
    else:
        experiment_dir = Path(path)

    df = _load_results(experiment_dir)
    _print_summary(df, experiment_dir.name)
    _print_failures(df)
    _print_collision_scenarios(df)

    if per_type:
        _print_per_type(df)

    if baseline is not None:
        baseline_df = _load_results(Path(baseline))
        _print_comparison(df, baseline_df)
