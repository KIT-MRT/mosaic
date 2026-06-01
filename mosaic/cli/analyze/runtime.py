"""Runtime extraction from simulation logs."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


_TIMING_PATTERN = re.compile(r"(Simulation duration):\s+(\d{2}:\d{2}:\d{2})")


@dataclass
class RuntimeInfo:
    duration_str: str
    duration_s: float
    per_scenario_str: Optional[str]


def parse_duration(duration_str: str) -> float:
    """Parse 'HH:MM:SS' into total seconds."""
    parts = duration_str.strip().split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


def get_runtime(experiment_dir: Path, num_scenarios: int) -> Optional[RuntimeInfo]:
    """Extract simulation runtime from log.txt. Returns None if unavailable."""
    log_file = experiment_dir / "log.txt"
    if not log_file.exists():
        return None

    timings = {}
    with open(log_file) as f:
        for line in f:
            match = _TIMING_PATTERN.search(line)
            if match:
                timings[match.group(1)] = match.group(2)

    if "Simulation duration" not in timings:
        return None

    duration_str = timings["Simulation duration"]
    duration_s = parse_duration(duration_str)
    per_scenario_str = None
    if num_scenarios > 0:
        per_scenario_str = f"{duration_s / num_scenarios:.1f}s/ea"

    return RuntimeInfo(
        duration_str=duration_str,
        duration_s=duration_s,
        per_scenario_str=per_scenario_str,
    )
