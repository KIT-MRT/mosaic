"""Build a machine-readable JSON export from analyze results."""

from pathlib import Path
from typing import Optional

from pandas import DataFrame

from mosaic.cli.analyze.data import (
    HARD_GATES,
    WEIGHTED_METRICS,
    detect_benchmark_extras,
)
from mosaic.cli.analyze.logs import load_cost_estimator_counts, load_verifier_counts
from mosaic.cli.analyze.runtime import RuntimeInfo


def build_export(
    df: DataFrame,
    experiment_name: str,
    runtime: Optional[RuntimeInfo],
    mosaic_dir: Path,
) -> dict:
    """Build a dict suitable for JSON export from analyze results."""
    num_scenarios = len(df)
    cls_r = round(df["score"].mean() * 100, 2)
    zero_score_count = int((df["score"] == 0.0).sum())

    result: dict = {
        "experiment": experiment_name,
        "num_scenarios": num_scenarios,
    }

    if runtime is not None:
        result["duration_s"] = runtime.duration_s
        if num_scenarios > 0:
            result["compute_time_per_scenario_s"] = round(
                runtime.duration_s / num_scenarios, 1
            )

    result["cls_r"] = cls_r
    result["zero_score_count"] = zero_score_count

    if "no_ego_at_fault_collisions" in df.columns:
        result["collision_count"] = int((df["no_ego_at_fault_collisions"] < 1.0).sum())

    if mosaic_dir.exists():
        verifier = load_verifier_counts(mosaic_dir)
        command_checks = verifier["command_checks"]
        command_fails = verifier["command_fails"]
        both_failures = verifier["both_failures"]
        total_timesteps = verifier["total_timesteps"]

        if command_checks:
            rejection_rates = {
                cmd: round(command_fails[cmd] / command_checks[cmd] * 100, 2)
                for cmd in sorted(command_checks)
            }
            simultaneous_pct = (
                round(both_failures / total_timesteps * 100, 2)
                if total_timesteps > 0
                else 0.0
            )
            result["verification"] = {
                "rejection_rates_pct": rejection_rates,
                "simultaneous_failure_pct": simultaneous_pct,
            }

        estimator = load_cost_estimator_counts(mosaic_dir)
        command_wins = estimator["command_wins"]
        all_commands = estimator["all_commands"]
        tie_count = estimator["tie_count"]
        total_decisions = sum(command_wins.values()) + tie_count

        if total_decisions > 0:
            win_pcts = {
                cmd: round(command_wins[cmd] / total_decisions * 100, 2)
                for cmd in sorted(all_commands)
            }
            result["selection"] = {
                "tie_pct": round(tie_count / total_decisions * 100, 2),
                "command_win_pcts": win_pcts,
            }

    extra_gates, extra_weighted = detect_benchmark_extras(df)
    gate_failures = {}
    for metric in HARD_GATES + extra_gates:
        if metric in df.columns:
            gate_failures[metric] = int((df[metric] < 1.0).sum())
    weighted_failures = {}
    for metric in list(WEIGHTED_METRICS) + list(extra_weighted):
        if metric in df.columns:
            weighted_failures[metric] = int((df[metric] == 0.0).sum())
    result["failures"] = {
        "hard_gates": gate_failures,
        "weighted_metrics": weighted_failures,
    }

    per_scenario = []
    for stype, group in df.groupby("scenario_type"):
        per_scenario.append(
            {
                "name": str(stype),
                "n": len(group),
                "cls_r": round(group["score"].mean() * 100, 2),
            }
        )
    per_scenario.sort(key=lambda x: x["cls_r"])
    result["per_scenario"] = per_scenario

    return result
