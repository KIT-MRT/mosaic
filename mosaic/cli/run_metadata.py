import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

RUN_METADATA_FILE_NAME = "run_metadata.json"

_PROJECT_DIR = Path(__file__).resolve().parents[2]


@dataclass
class GitState:
    commit: Optional[str]
    branch: Optional[str]
    changed_files: list[str]
    is_dirty: bool


@dataclass
class HardwareSettings:
    hostname: str
    cuda_visible_devices: Optional[str]
    gpu_names: list[str]
    cpu_count: Optional[int]
    threads: int
    gpus_per_sim: float


@dataclass
class RunMetadata:
    started_at: str
    command_line: list[str]
    git: GitState
    hardware: HardwareSettings
    simulation_overrides: list[str]
    planner_config: dict[str, Any]
    finished_at: Optional[str] = None


def collect_run_metadata(
    threads: int,
    gpus_per_sim: float,
    simulation_overrides: list[str],
    planner_config: dict[str, Any],
) -> RunMetadata:
    return RunMetadata(
        started_at=_now(),
        command_line=sys.argv,
        git=_git_state(),
        hardware=HardwareSettings(
            hostname=platform.node(),
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            gpu_names=_gpu_names(),
            cpu_count=os.cpu_count(),
            threads=threads,
            gpus_per_sim=gpus_per_sim,
        ),
        simulation_overrides=simulation_overrides,
        planner_config=planner_config,
    )


def write_run_metadata(metadata: RunMetadata, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / RUN_METADATA_FILE_NAME
    path.write_text(json.dumps(asdict(metadata), indent=2) + "\n")


def mark_finished(metadata: RunMetadata, output_dir: Path) -> None:
    metadata.finished_at = _now()
    write_run_metadata(metadata, output_dir)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _run_git(*args: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=_PROJECT_DIR,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout


def _git_state() -> GitState:
    status = _run_git("status", "--porcelain", "--untracked-files=no")
    changed_files = [line[3:] for line in (status or "").splitlines()]
    commit = _run_git("rev-parse", "HEAD")
    branch = _run_git("rev-parse", "--abbrev-ref", "HEAD")
    return GitState(
        commit=commit.strip() if commit is not None else None,
        branch=branch.strip() if branch is not None else None,
        changed_files=changed_files,
        is_dirty=len(changed_files) > 0,
    )


def _gpu_names() -> list[str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]
