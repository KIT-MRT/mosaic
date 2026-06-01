"""Analysis of cost estimator and verifier JSONL logs."""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import TypedDict

import click

from mosaic.cli.analyze.formatting import Table, kv, section


class CostEstimatorCounts(TypedDict):
    command_wins: Counter[int]
    command_appearances: Counter[int]
    all_commands: set[str]
    tie_count: int


class VerifierCounts(TypedDict):
    command_fails: Counter[int]
    command_checks: Counter[int]
    both_failures: int
    total_timesteps: int


def load_cost_estimator_counts(mosaic_dir: Path) -> CostEstimatorCounts:
    """Parse trajectory cost JSONL logs and return raw counts."""
    command_wins: Counter[int] = Counter()
    command_appearances: Counter[int] = Counter()
    all_commands: set[str] = set()
    tie_count = 0

    for estimator_log in mosaic_dir.glob("*_trajectory_costs.jsonl"):
        with open(estimator_log) as f:
            for line in f:
                entry = json.loads(line)
                proposals = entry.get("proposals", [])
                if not proposals:
                    continue

                for p in proposals:
                    cmd = p["command"]
                    all_commands.add(cmd)
                    command_appearances[cmd] += 1

                max_score = max(p["final_score"] for p in proposals)
                winners = [
                    p["command"] for p in proposals if p["final_score"] == max_score
                ]

                if len(winners) > 1:
                    tie_count += 1
                else:
                    command_wins[winners[0]] += 1

    return CostEstimatorCounts(
        command_wins=command_wins,
        command_appearances=command_appearances,
        all_commands=all_commands,
        tie_count=tie_count,
    )


def load_verifier_counts(mosaic_dir: Path) -> VerifierCounts:
    """Parse verifier JSONL logs and return raw counts."""
    command_fails: Counter = Counter()
    command_checks: Counter = Counter()
    both_failures = 0
    total_timesteps = 0

    for verifier_log in mosaic_dir.glob("*_verification.jsonl"):
        timestep_results: dict = defaultdict(list)

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
            if sum(e["result"] == "fail" for e in entries) > 1:
                both_failures += 1

    return VerifierCounts(
        command_fails=command_fails,
        command_checks=command_checks,
        both_failures=both_failures,
        total_timesteps=total_timesteps,
    )


def analyze_cost_estimator_logs(mosaic_dir: Path) -> None:
    """Analyze trajectory cost logs to report per-command win rates."""
    counts = load_cost_estimator_counts(mosaic_dir)
    command_wins = counts["command_wins"]
    command_appearances = counts["command_appearances"]
    all_commands = counts["all_commands"]
    tie_count = counts["tie_count"]

    section("Cost Estimator")

    total_decisions = sum(command_wins.values()) + tie_count
    if total_decisions == 0:
        click.echo("\n  No estimator entries found.")
        return

    t = Table(
        ["Command", "Wins", "Of Decisions", "Of Appearances"],
        [18, 6, 13, 15],
        ["<", ">", ">", ">"],
    )
    for cmd in sorted(all_commands):
        wins = command_wins[cmd]
        appearances = command_appearances[cmd]
        rate_global = f"{wins / total_decisions * 100:.1f}%"
        rate_local = f"{wins / appearances * 100:.1f}%" if appearances > 0 else "—"
        t.row([cmd, str(wins), rate_global, rate_local])
    t.render()

    tie_rate = tie_count / total_decisions * 100
    click.echo()
    kv("Tied scores", f"{tie_count} / {total_decisions} ({tie_rate:.1f}%)")


def analyze_verifier_logs(mosaic_dir: Path) -> None:
    """Analyze verifier logs to report per-command fail rates."""
    counts = load_verifier_counts(mosaic_dir)
    command_fails = counts["command_fails"]
    command_checks = counts["command_checks"]
    both_failures = counts["both_failures"]
    total_timesteps = counts["total_timesteps"]

    section("Verifier")

    if not command_checks:
        click.echo("\n  No verifier entries found.")
        return

    t = Table(
        ["Command", "Fails", "Checks", "Fail Rate"],
        [18, 6, 8, 10],
        ["<", ">", ">", ">"],
    )
    for cmd in sorted(command_checks):
        checks = command_checks[cmd]
        fails = command_fails[cmd]
        rate = f"{fails / checks * 100:.1f}%" if checks > 0 else "—"
        t.row([cmd, str(fails), str(checks), rate])
    t.render()

    both_rate = both_failures / total_timesteps * 100 if total_timesteps > 0 else 0.0
    click.echo()
    kv("Both failed", f"{both_failures} / {total_timesteps} ({both_rate:.1f}%)")
