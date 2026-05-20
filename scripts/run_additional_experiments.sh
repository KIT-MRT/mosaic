#!/usr/bin/env bash
# Additional experiments for the dissertation (not part of the Mosaic paper).
# Runs single-planner ablations on the interPlan benchmark to isolate the
# composition gain from the verification gain on out-of-distribution scenarios.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$PROJECT_DIR/results"

# Parse --quick flag
QUICK_ARGS=()
if [[ "${1:-}" == "--quick" ]]; then
    QUICK_ARGS=(-n 1)
    echo "=== Quick mode: limiting each experiment to 1 scenario ==="
fi

# Load hardware configuration
source "$SCRIPT_DIR/config.sh"

# Install dependencies and activate venv
echo "=== Installing dependencies ==="
cd "$PROJECT_DIR"
uv sync
source .venv/bin/activate

run_experiment() {
    local name="$1"
    local challenge="$2"
    local ablation="$3"
    shift 3
    local extra_args=("$@")

    echo ""
    echo "========================================"
    echo "  Experiment: $name"
    echo "  Challenge:  $challenge"
    echo "  Ablation:   $ablation"
    echo "========================================"
    echo ""

    local sim_args=(
        -c "$challenge"
        --ablation "$ablation"
        --experiment-name "$name"
        --gpus-per-sim "$GPUS_PER_SIM"
        --threads "$THREADS"
    )
    sim_args+=("${QUICK_ARGS[@]}")
    sim_args+=("${extra_args[@]}")

    mosaic simulate "${sim_args[@]}"

    mosaic analyze | tee "$RESULTS_DIR/$name.txt"
}

run_experiment interplan-pdm-closed-only interplan pdm_closed_only
run_experiment interplan-flowdrive-only interplan flow_drive_only

echo ""
echo "=== Additional experiments complete. Results saved in $RESULTS_DIR ==="
