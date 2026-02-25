"""Summary, failure breakdown, collision, and per-type reports."""

from typing import Optional, Tuple

import click
from pandas import DataFrame

from mosaic.cli.analyze.data import HARD_GATES, WEIGHTED_METRICS
from mosaic.cli.analyze.formatting import (
    Table,
    fail_marker,
    kv,
    section,
    subsection,
    title_box,
    truncate,
)


def print_header(
    df: DataFrame,
    label: str,
    runtime: Optional[Tuple[str, Optional[str]]],
) -> None:
    """Print the title box and key stats underneath."""
    title_box(label)
    click.echo()
    kv("Scenarios", str(len(df)))
    kv("Overall score", f"{df['score'].mean():.4f}")
    if runtime:
        duration, per_scenario = runtime
        kv("Duration", duration + (f" ({per_scenario})" if per_scenario else ""))


def print_failures(df: DataFrame) -> None:
    """Print failure breakdown for hard gates and weighted metrics."""
    section("Failures")

    widths = [38, 5, 7]
    aligns = ["<", ">", ">"]

    subsection("Hard Gates")
    t = Table(["Metric", "Fail", "Rate"], widths, aligns)
    for metric in HARD_GATES:
        if metric not in df.columns:
            continue
        fails = int((df[metric] == 0.0).sum())
        rate = f"{fails / len(df) * 100:.1f}%"
        t.row([metric, str(fails), rate], fail_marker(fails))
    t.render()

    subsection("Weighted Metrics")
    t = Table(["Metric", "Fail", "Rate"], widths, aligns)
    for metric, weight in WEIGHTED_METRICS.items():
        if metric not in df.columns:
            continue
        fails = int((df[metric] == 0.0).sum())
        rate = f"{fails / len(df) * 100:.1f}%"
        label = f"{metric} (w={weight})"
        t.row([label, str(fails), rate], fail_marker(fails))
    t.render()

    zero_score = int((df["score"] == 0.0).sum())
    click.echo(f"\n  {zero_score} scenarios scored zero")


def print_collision_scenarios(df: DataFrame) -> None:
    """List all scenarios where the ego caused a collision."""
    if "no_ego_at_fault_collisions" not in df.columns:
        return

    collision_df = df[df["no_ego_at_fault_collisions"] == 0.0].copy()
    if collision_df.empty:
        section("Collisions")
        click.echo("\n  None.")
        return

    collision_df = collision_df.sort_values("score")

    section(f"Collisions ({len(collision_df)})")

    t = Table(["Scenario", "Type", "Score"], [18, 30, 6], ["<", "<", ">"])
    for _, row in collision_df.iterrows():
        stype = truncate(str(row.get("scenario_type", "")), 30)
        t.row([str(row["scenario"]), stype, f"{row['score']:.4f}"])
    t.render()


def print_per_type(df: DataFrame) -> None:
    """Print a per-scenario-type breakdown table."""
    section("Per Scenario Type")

    t = Table(
        ["Type", "n", "Score", "Zero", "TTC", "Coll"],
        [36, 4, 6, 4, 4, 4],
        ["<", ">", ">", ">", ">", ">"],
    )

    type_stats = []
    for stype, group in df.groupby("scenario_type"):
        n = len(group)
        score = group["score"].mean()
        zeros = int((group["score"] == 0.0).sum())
        ttc_fail = (
            int((group["time_to_collision_within_bound"] == 0.0).sum())
            if "time_to_collision_within_bound" in group.columns
            else 0
        )
        coll_fail = (
            int((group["no_ego_at_fault_collisions"] == 0.0).sum())
            if "no_ego_at_fault_collisions" in group.columns
            else 0
        )
        type_stats.append((stype, n, score, zeros, ttc_fail, coll_fail))

    for stype, n, score, zeros, ttc_fail, coll_fail in sorted(
        type_stats, key=lambda x: x[2]
    ):
        t.row(
            [
                truncate(stype, 36),
                str(n),
                f"{score:.4f}",
                str(zeros),
                str(ttc_fail),
                str(coll_fail),
            ]
        )

    t.render()
