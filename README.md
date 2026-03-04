<p align="center">
  <img src="assets/mosaic.png" alt="Mosaic logo" height="128">
</p>

# Mosaic

**An Extensible Framework for Composing Rule-Based and Learned Motion Planners**

Mosaic is an autonomous driving motion planning framework that uses [arbitration graphs](https://github.com/KIT-MRT/arbitration_graphs) to combine heterogeneous trajectory planners into a unified, explainable decision-making structure. It integrates a rule-based planner ([PDM-Closed](https://github.com/autonomousvision/tuplan_garage)) and a learning-based planner ([FlowDrive](https://github.com/einsteinguang/flow_drive_planner)) with centralized trajectory verification and scoring, making every selection decision transparent and traceable.

Evaluated on [nuPlan](https://github.com/motional/nuplan-devkit), Mosaic achieves state-of-the-art closed-loop performance — outperforming all existing methods without retraining either planner or requiring additional data.

## Results

### nuPlan Val14 and interPlan benchmarks

| Type | Planner | Val14 CLS-NR | Val14 CLS-R | interPlan CLS-R |
|---|---|:---:|:---:|:---:|
| Expert | Log-replay | 93.53 | 80.32 | 14.76 |
| Learning-based | FlowDrive | 91.21 | 85.37 | 36.96 |
| Rule-based & Hybrid | PDM-Closed | 92.84 | 92.12 | 41.23 |
| | FlowDrive* | 94.81 | 92.96 | 44.05 |
| | GIGAFLOW | - | 93.80 | - |
| | **Mosaic (ours)** | **95.49** | **93.97** | **54.03** |

### Ablation study (Val14 CLS-R)

| Configuration | CLS-R | Collisions | Zero-score |
|---|:---:|:---:|:---:|
| **Mosaic (full)** | **93.97** | **15** | **20** |
| w/o verifier | 93.33 | 27 | 31 |
| FlowDrive* only | 92.91 | 19 | 31 |
| PDM-Closed only | 92.15 | 17 | 33 |

## Architecture

```
Mosaic (PriorityArbitrator)
├── Composer (CostArbitrator)     ← trajectory selector with verification
│   ├── FlowDrive*                ← learning-based planner
│   └── PDM-Closed                ← rule-based planner
└── Emergency Stop   (FALLBACK)   ← hard brake, last resort
```

The **Composer** verifies and scores candidate trajectories from both planners. Unsafe proposals (imminent at-fault collisions) are rejected before scoring. Among verified trajectories, the one with the best score is selected. The scoring function combines multiplicative safety gates (collision, drivable area, driving direction, progress) with a weighted performance score (progress, time-to-collision, comfort).

If neither planner produces a verified trajectory, **Mosaic** falls back to the **Emergency Stop** component.

## Setup

### Prerequisites

- [uv](https://docs.astral.sh/uv/) package manager
- [nuPlan dataset](https://www.nuscenes.org/nuplan) (set `NUPLAN_DATA_ROOT` and `NUPLAN_MAPS_ROOT` environment variables)

### Installation

```bash
uv sync
```

This installs all dependencies and makes the `mosaic` CLI available.

## Usage

```bash
mosaic simulate          # Run simulation (default: Val14 CLS-R)
mosaic analyze           # Print summary of latest experiment
mosaic results           # Launch nuBoard to view results
mosaic plot              # Generate behavior selection pie chart
```

### Quick test

```bash
mosaic simulate -n 1     # Run a single scenario
```

### CLI reference

```
mosaic simulate [OPTIONS]
  -c, --challenge          closed_loop_reactive_agents (default) | closed_loop_nonreactive_agents | interplan
  --scenario-filter        Scenario filter preset (default: val14_split, or interplan10 for interplan)
  --ablation               none (default) | no_verifier | pdm_closed_only | flow_drive_only
  -n, --limit-scenarios    Limit total scenarios
  --experiment-name        Experiment name (default: mosaic)
  --threads                Worker threads per node (default: 160)
  --gpus-per-sim           GPUs per simulation (default: 0.05)
  -o, --override           Arbitrary Hydra overrides (repeatable)

mosaic analyze [OPTIONS]
  -p, --path               Path to experiment output dir (auto-detects latest)
  -b, --baseline           Path to baseline experiment dir for comparison
  --per-type / --no-per-type  Per-scenario-type breakdown (default: on)

mosaic results [OPTIONS]
  -p, --path               Path to output dir or .nuboard file (auto-detects latest)
  --port                   Port number (default: 5006)

mosaic plot [OPTIONS]
  -p, --path               Path to experiment output dir (auto-detects latest)
  -o, --output             Output SVG path (default: behavior_selection.svg)
```

## Reproducing paper experiments

Run all benchmarks and ablations with a single script:

```bash
bash scripts/run_experiments.sh
```

This runs 6 experiments sequentially and saves analysis outputs to `results/`. Use `--quick` for a smoke test with one scenario per experiment:

```bash
bash scripts/run_experiments.sh --quick
```

Individual experiments:

| Experiment | Command |
|---|---|
| Val14 CLS-R | `mosaic simulate` |
| Val14 CLS-NR | `mosaic simulate -c closed_loop_nonreactive_agents` |
| interPlan CLS-R | `mosaic simulate -c interplan` |
| Ablation: no verifier | `mosaic simulate --ablation no_verifier` |
| Ablation: PDM-Closed only | `mosaic simulate --ablation pdm_closed_only` |
| Ablation: FlowDrive* only | `mosaic simulate --ablation flow_drive_only` |

## Development

```bash
uv run pytest                    # Run tests
uv run ruff check .              # Lint
uv run ruff format --check .     # Check formatting
uv run ruff format .             # Auto-format
```

## Citation

TODO
