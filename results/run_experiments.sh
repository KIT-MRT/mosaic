#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR"

# Parse --quick flag
QUICK_ARGS=()
if [[ "${1:-}" == "--quick" ]]; then
    QUICK_ARGS=(-n 1)
    echo "=== Quick mode: limiting each experiment to 1 scenario ==="
fi

# Install dependencies and activate venv
echo "=== Installing dependencies ==="
cd "$PROJECT_DIR"
uv sync
source .venv/bin/activate

run_experiment() {
    local name="$1"
    local challenge="$2"
    local ablation="$3"

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
    )
    sim_args+=("${QUICK_ARGS[@]}")

    mosaic simulate "${sim_args[@]}"

    mosaic analyze | tee "$RESULTS_DIR/$name.txt"
}

run_experiment val14-reactive closed_loop_reactive_agents none
run_experiment val14-nonreactive closed_loop_nonreactive_agents none
run_experiment interplan interplan none
run_experiment ablation-no-verifier closed_loop_reactive_agents no_verifier
run_experiment ablation-pdm-closed-only closed_loop_reactive_agents pdm_closed_only
run_experiment ablation-flowdrive-only closed_loop_reactive_agents flow_drive_only

echo ""
echo "=== All experiments complete. Results saved in $RESULTS_DIR ==="
