import json
import os
import re
from collections import Counter, defaultdict
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

    df = pd.read_parquet(parquet_files[0])

    # Drop nuplan aggregate rows (per-type summaries + final_score)
    aggregate_names = set(df["scenario_type"].unique())
    aggregate_names.add("final_score")
    df = df[~df["scenario"].isin(aggregate_names)].reset_index(drop=True)

    return df


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
    click.echo("\n  Failure Breakdown:")
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
    click.echo("\n  Per-Scenario-Type Breakdown:")
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
    click.echo("\n  Comparison with Baseline:")
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
            click.echo("\n  Worst regressions:")
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
            click.echo("\n  Best improvements:")
            for _, row in improvements.head(10).iterrows():
                click.echo(
                    f"    {row['scenario']}  {row['score_base']:.4f} -> {row['score_new']:.4f}  ({row['diff']:+.4f})"
                )


def _analyze_cost_estimator_logs(mosaic_dir: Path) -> None:
    command_score_wins = Counter()
    command_appearances = Counter()
    all_commands = set()
    command_score_ties = 0

    for estimator_log in mosaic_dir.glob("*_trajectory_costs.jsonl"):
        with open(estimator_log) as f:
            for line in f:
                entry = json.loads(line)
                proposals = entry.get("proposals", [])
                if not proposals:
                    continue

                # Track appearances
                for p in proposals:
                    cmd = p["command"]
                    all_commands.add(cmd)
                    command_appearances[cmd] += 1

                # Determine winner(s)
                max_score = max(p["final_score"] for p in proposals)
                winners = [
                    p["command"] for p in proposals if p["final_score"] == max_score
                ]

                if len(winners) > 1:
                    command_score_ties += 1
                else:
                    command_score_wins[winners[0]] += 1

    click.echo("\n=== Cost Estimator Analysis ===\n")

    total_decisions = sum(command_score_wins.values()) + command_score_ties

    if total_decisions == 0:
        click.echo("  No estimator entries found.")
        return

    for cmd in sorted(all_commands):
        wins = command_score_wins[cmd]
        appearances = command_appearances[cmd]

        win_rate_global = wins / total_decisions * 100
        win_rate_local = wins / appearances * 100 if appearances > 0 else 0.0

        click.echo(
            f"  {cmd:<20} "
            f"{wins:>6} wins  "
            f"({win_rate_global:>6.2f}% of decisions, "
            f"{win_rate_local:>6.2f}% of its appearances)"
        )

    tie_rate = command_score_ties / total_decisions * 100

    click.echo(
        f"\n  Tied scores: {command_score_ties} / {total_decisions} ({tie_rate:.2f}%)"
    )


def _analyze_verifier_logs(mosaic_dir: Path) -> None:
    command_fails = Counter()
    command_checks = Counter()
    both_failures = 0
    total_timesteps = 0

    for verifier_log in mosaic_dir.glob("*_verification.jsonl"):
        timestep_results = defaultdict(list)

        with open(verifier_log) as f:
            for line in f:
                entry = json.loads(line)

                cmd = entry["command"]
                timestep_results[entry["time"]].append(entry)

                command_checks[cmd] += 1
                if entry["result"] == "fail":
                    command_fails[cmd] += 1

        for entries in timestep_results.values():
            total_timesteps += 1
            fail_count = sum(e["result"] == "fail" for e in entries)
            if fail_count > 1:
                both_failures += 1

    click.echo("\n=== Verifier Analysis ===\n")

    if not command_checks:
        click.echo("  No verifier entries found.")
        return

    for cmd in sorted(command_checks):
        checks = command_checks[cmd]
        fails = command_fails[cmd]
        rate = fails / checks * 100 if checks > 0 else 0.0

        click.echo(f"  {cmd:<20} {fails:>6} / {checks:<6} ({rate:>6.2f}% fail rate)")

    both_rate = both_failures / total_timesteps * 100 if total_timesteps > 0 else 0.0

    click.echo(
        f"\n  Both commands failed at same timestep: "
        f"{both_failures} / {total_timesteps} "
        f"({both_rate:.2f}%)"
    )


def _parse_duration(duration_str: str) -> float:
    """Parse 'HH:MM:SS' into total seconds."""
    parts = duration_str.strip().split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


_TIMING_PATTERN = re.compile(
    r"(Simulation duration)"
    r":\s+(\d{2}:\d{2}:\d{2})"
)


def _print_runtime(experiment_dir: Path, num_scenarios: int) -> None:
    log_file = experiment_dir / "log.txt"
    if not log_file.exists():
        return

    timings = {}
    with open(log_file) as f:
        for line in f:
            match = _TIMING_PATTERN.search(line)
            if match:
                timings[match.group(1)] = match.group(2)

    if "Simulation duration" not in timings:
        return

    click.echo("\n=== Runtime ===\n")

    sim_seconds = _parse_duration(timings["Simulation duration"])
    click.echo(f"  Simulation duration:      {timings['Simulation duration']}")

    if num_scenarios > 0:
        per_scenario = sim_seconds / num_scenarios
        click.echo(f"  Per scenario:             {per_scenario:.2f}s")


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
    _print_runtime(experiment_dir, len(df))
    _print_failures(df)
    _print_collision_scenarios(df)

    if per_type:
        _print_per_type(df)

    if baseline is not None:
        baseline_df = _load_results(Path(baseline))
        _print_comparison(df, baseline_df)

    mosaic_dir = experiment_dir / "mosaic_logs"
    if not mosaic_dir.exists():
        click.echo("\n  No mosaic_logs directory found.")
        return

    _analyze_cost_estimator_logs(mosaic_dir)
    _analyze_verifier_logs(mosaic_dir)
