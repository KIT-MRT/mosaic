"""Baseline comparison report."""

import click
from pandas import DataFrame

from mosaic.cli.analyze.data import HARD_GATES, WEIGHTED_METRICS
from mosaic.cli.analyze.formatting import Table, colored_diff, kv, section


def print_comparison(df: DataFrame, baseline_df: DataFrame) -> None:
    """Compare current results against a baseline experiment."""
    section("Baseline Comparison")

    current_score = df["score"].mean()
    baseline_score = baseline_df["score"].mean()
    overall_diff = current_score - baseline_score
    click.echo()
    kv("Baseline score", f"{baseline_score:.4f}")
    kv("Current score", f"{current_score:.4f}")
    kv("Diff", colored_diff(overall_diff))

    # Per-metric failure comparison
    t = Table(
        ["Metric", "Now", "Base", "Diff"],
        [38, 5, 5, 6],
        ["<", ">", ">", ">"],
    )
    for metric in HARD_GATES + list(WEIGHTED_METRICS.keys()):
        if metric not in df.columns or metric not in baseline_df.columns:
            continue
        now_fails = int((df[metric] == 0.0).sum())
        base_fails = int((baseline_df[metric] == 0.0).sum())
        diff = now_fails - base_fails
        diff_str = colored_diff(diff, fmt="+d", invert=True)
        t.row([metric, str(now_fails), str(base_fails), diff_str])
    t.render()

    # Per-scenario matched comparison
    merged = df[["scenario", "score"]].merge(
        baseline_df[["scenario", "score"]],
        on="scenario",
        suffixes=("_new", "_base"),
        how="inner",
    )
    if len(merged) == 0:
        return

    improved = int((merged["score_new"] > merged["score_base"]).sum())
    regressed = int((merged["score_new"] < merged["score_base"]).sum())
    unchanged = int((merged["score_new"] == merged["score_base"]).sum())

    parts = [f"{len(merged)} matched"]
    if improved:
        parts.append(click.style(f"▲ {improved} improved", fg="green"))
    if regressed:
        parts.append(click.style(f"▼ {regressed} regressed", fg="red"))
    if unchanged:
        parts.append(f"= {unchanged} unchanged")
    click.echo("\n  " + "   ".join(parts))

    if regressed > 0:
        regressions = merged[merged["score_new"] < merged["score_base"]].copy()
        regressions["diff"] = regressions["score_new"] - regressions["score_base"]
        regressions = regressions.sort_values("diff")

        click.echo(f"\n  {click.style('Worst Regressions', bold=True)}")
        for _, row in regressions.head(10).iterrows():
            diff_str = click.style(f"{row['diff']:+.4f}", fg="red")
            click.echo(
                f"    {row['scenario']}  "
                f"{row['score_base']:.4f} → {row['score_new']:.4f}  "
                f"({diff_str})"
            )

    if improved > 0:
        improvements = merged[merged["score_new"] > merged["score_base"]].copy()
        improvements["diff"] = improvements["score_new"] - improvements["score_base"]
        improvements = improvements.sort_values("diff", ascending=False)

        click.echo(f"\n  {click.style('Best Improvements', bold=True)}")
        for _, row in improvements.head(10).iterrows():
            diff_str = click.style(f"{row['diff']:+.4f}", fg="green")
            click.echo(
                f"    {row['scenario']}  "
                f"{row['score_base']:.4f} → {row['score_new']:.4f}  "
                f"({diff_str})"
            )
