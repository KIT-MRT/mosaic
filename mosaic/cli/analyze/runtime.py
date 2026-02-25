"""Runtime extraction from simulation logs."""

import re
from pathlib import Path
from typing import Optional, Tuple


_TIMING_PATTERN = re.compile(r"(Simulation duration):\s+(\d{2}:\d{2}:\d{2})")


def _parse_duration(duration_str: str) -> float:
    """Parse 'HH:MM:SS' into total seconds."""
    parts = duration_str.strip().split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


def get_runtime(
    experiment_dir: Path, num_scenarios: int
) -> Optional[Tuple[str, Optional[str]]]:
    """Extract simulation runtime from log.txt.

    Returns (duration_str, per_scenario_str) or None if unavailable.
    """
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

    duration = timings["Simulation duration"]
    per_scenario = None
    if num_scenarios > 0:
        secs = _parse_duration(duration) / num_scenarios
        per_scenario = f"{secs:.1f}s/ea"

    return duration, per_scenario
