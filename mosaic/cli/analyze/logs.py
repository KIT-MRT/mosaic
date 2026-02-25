"""Analysis of cost estimator and verifier JSONL logs."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import click

from mosaic.cli.analyze.formatting import Table, kv, section


def load_cost_estimator_counts(
    mosaic_dir: Path,
) -> tuple[Counter, Counter, set, int]:
    """Parse trajectory cost JSONL logs and return raw counts.

    Returns (command_score_wins, command_appearances, all_commands, tie_count).
    """
    command_score_wins: Counter = Counter()
    command_appearances: Counter = Counter()
    all_commands: set = set()
    command_score_ties = 0

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
                    command_score_ties += 1
                else:
                    command_score_wins[winners[0]] += 1

    return command_score_wins, command_appearances, all_commands, command_score_ties


def analyze_cost_estimator_logs(mosaic_dir: Path) -> None:
    """Analyze trajectory cost logs to report per-command win rates."""
    command_score_wins, command_appearances, all_commands, command_score_ties = (
        load_cost_estimator_counts(mosaic_dir)
    )

    section("Cost Estimator")

    total_decisions = sum(command_score_wins.values()) + command_score_ties
    if total_decisions == 0:
        click.echo("\n  No estimator entries found.")
        return

    t = Table(
        ["Command", "Wins", "Of Decisions", "Of Appearances"],
        [18, 6, 13, 15],
        ["<", ">", ">", ">"],
    )
    for cmd in sorted(all_commands):
        wins = command_score_wins[cmd]
        appearances = command_appearances[cmd]
        rate_global = f"{wins / total_decisions * 100:.1f}%"
        rate_local = f"{wins / appearances * 100:.1f}%" if appearances > 0 else "—"
        t.row([cmd, str(wins), rate_global, rate_local])
    t.render()

    tie_rate = command_score_ties / total_decisions * 100
    click.echo()
    kv("Tied scores", f"{command_score_ties} / {total_decisions} ({tie_rate:.1f}%)")


def analyze_verifier_logs(mosaic_dir: Path) -> None:
    """Analyze verifier logs to report per-command fail rates."""
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
