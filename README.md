<p align="center">
  <img src="assets/mosaic.png" alt="Mosaic logo" height="128">
</p>

# Mosaic

**An Extensible Framework for Composing Rule-Based and Learned Motion Planners**

Mosaic is an autonomous driving motion planning framework that uses [arbitration graphs](https://github.com/KIT-MRT/arbitration_graphs) to combine heterogeneous trajectory planners into a unified, explainable decision-making structure. It integrates a rule-based planner ([PDM-Closed](https://github.com/autonomousvision/tuplan_garage)) and a learning-based planner ([FlowDrive](https://github.com/einsteinguang/flow_drive_planner)) with centralized trajectory verification and scoring, making every selection decision transparent and traceable.

Evaluated on [nuPlan](https://github.com/motional/nuplan-devkit), Mosaic achieves state-of-the-art closed-loop performance — outperforming all existing methods without retraining either planner or requiring additional data.

## Results

### nuPlan Val14 and interPlan benchmarks

| Type                | Planner           | Val14 CLS-NR | Val14 CLS-R | interPlan CLS-R |
| ------------------- | ----------------- | :----------: | :---------: | :-------------: |
| Expert              | Log-replay        |    93.53     |    80.32    |      14.76      |
| Learning-based      | FlowDrive         |    91.21     |    85.37    |      36.96      |
|                     | GIGAFLOW          |      -       |    93.80    |        -        |
| Rule-based & Hybrid | PDM-Closed        |    92.84     |    92.12    |      41.23      |
|                     | FlowDrive*        |    94.81     |    92.96    |      44.05      |
|                     | **Mosaic (ours)** |  **95.48**   |  **93.98**  |    **54.30**    |

### Ablation study (Val14 CLS-R)

| Configuration     |   CLS-R   | Collisions | Zero-score |
| ----------------- | :-------: | :--------: | :--------: |
| **Mosaic (full)** | **93.98** |   **14**   |   **20**   |
| w/o verifier      |   92.82   |     35     |     38     |
| FlowDrive* only   |   92.96   |     18     |     30     |
| PDM-Closed only   |   92.15   |     17     |     33     |

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

## Prerequisites

- [uv](https://docs.astral.sh/uv/) package manager (dependencies are synced automatically on first `uv run`)
- [nuPlan dataset](https://www.nuscenes.org/nuplan) (set `NUPLAN_DATA_ROOT` and `NUPLAN_MAPS_ROOT` environment variables)

## Usage

```bash
uv run mosaic simulate          # Run simulation (default: Val14 CLS-R)
uv run mosaic analyze           # Print summary of latest experiment
uv run mosaic results           # Launch nuBoard to view results
uv run mosaic plot              # Generate behavior selection pie chart
uv run mosaic cite              # Print BibTeX citation
```

### Quick test

```bash
uv run mosaic simulate -n 1     # Run a single scenario
```

### CLI reference

```
uv run mosaic simulate [OPTIONS]
  -c, --challenge          closed_loop_reactive_agents (default) | closed_loop_nonreactive_agents | interplan
  --scenario-filter        Scenario filter preset (default: val14_split, or interplan10 for interplan)
  --ablation               none (default) | no_verifier | pdm_closed_only | flow_drive_only
  -n, --limit-scenarios    Limit total scenarios
  --experiment-name        Experiment name (default: mosaic)
  --threads                Worker threads per node (default: 160)
  --gpus-per-sim           GPUs per simulation (default: 0.05)
  -o, --override           Simulation framework overrides (repeatable, e.g. -o worker.threads_per_node=80)
  -p, --planner-override   Planner parameter overrides (repeatable, e.g. -p cost_estimator.parameters.ttc.weight=10)

uv run mosaic analyze [OPTIONS]
  -p, --path               Path to experiment output dir (auto-detects latest)
  -b, --baseline           Path to baseline experiment dir for comparison
  --per-type / --no-per-type  Per-scenario-type breakdown (default: on)

uv run mosaic results [OPTIONS]
  -p, --path               Path to output dir or .nuboard file (auto-detects latest)
  --port                   Port number (default: 5006)

uv run mosaic plot [OPTIONS]
  -p, --path               Path to experiment output dir (auto-detects latest)
  -o, --output             Output SVG path (default: behavior_selection.svg)

uv run mosaic cite
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

| Experiment                | Command                                                    |
| ------------------------- | ---------------------------------------------------------- |
| Val14 CLS-R               | `uv run mosaic simulate`                                   |
| Val14 CLS-NR              | `uv run mosaic simulate -c closed_loop_nonreactive_agents` |
| interPlan CLS-R           | `uv run mosaic simulate -c interplan`                      |
| Ablation: no verifier     | `uv run mosaic simulate --ablation no_verifier`            |
| Ablation: PDM-Closed only | `uv run mosaic simulate --ablation pdm_closed_only`        |
| Ablation: FlowDrive* only | `uv run mosaic simulate --ablation flow_drive_only`        |

## Why the forks?

Three dependencies are pinned to personal forks ([nuplan-devkit](https://github.com/ll-nick/nuplan-devkit), [tuplan-garage](https://github.com/ll-nick/tuplan_garage), [interPlan](https://github.com/ll-nick/interPlan)). The changes fall into three categories:

**Packaging** (all three forks): The upstream repositories rely on conda and manual install steps. The forks replace `setup.py`/`requirements.txt` with `pyproject.toml`, pin dependency versions to resolve conflicts, and ensure non-editable installs work correctly by explicitly including config files (YAML, JSON, etc.) that the upstream packages omit.

**Data paths** (nuplan-devkit, interPlan): Minor path configuration fixes required to locate dataset files correctly in a standard environment.

**PDM-Closed prediction** (tuplan-garage): Switches from heading-based to velocity-based agent trajectory prediction, following a recommendation from the FlowDrive authors. This is the setup under which FlowDrive was trained and evaluated. Also adds an option to disable PDM-Closed's internal emergency brake fallback, allowing the arbitration graph to handle that decision instead.

## Development

```bash
uv run pytest                    # Run tests
uv run ruff check .              # Lint
uv run ruff format --check .     # Check formatting
uv run ruff format .             # Auto-format
```

## Citation

Please consider citing our paper if our work is helpful to your research:

```bibtex
@misc{lelarge2026mosaic,
  title={Mosaic: An Extensible Framework for Composing Rule-Based and Learned Motion Planners},
  author={Le Large, Nick and Steiner, Marlon and Wang, Lingguang and Poh, Willi and Pauls, Jan-Hendrik and Ta\c{s}, {\"{O}}mer {\c{S}}ahin and Stiller, Christoph},
  year={2026},
  eprint={2604.13853},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2604.13853},
}
```
