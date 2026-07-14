#!/usr/bin/env bash
# Sensitivity study for the cost function weights (TTC and progress).
#
# With three weights (TTC, progress, comfort) there are only two independent
# degrees of freedom — the ratios between them. We fix comfort=2 and sweep
# TTC in {3.5, 7, 14} × progress in {2.5, 5, 10}, covering ±1 octave around
# the defaults (TTC=7, progress=5) in both directions. The center cell
# (TTC=7, progress=5) is already captured by the main val14-reactive result
# and is skipped here.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$PROJECT_DIR/results/sensitivity"

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

mkdir -p "$RESULTS_DIR"

run_cell() {
    local ttc_weight="$1"
    local progress_weight="$2"
    local name="sensitivity-ttc${ttc_weight}-prog${progress_weight}"

    echo ""
    echo "========================================"
    echo "  Experiment: $name"
    echo "  TTC weight:      $ttc_weight"
    echo "  Progress weight: $progress_weight"
    echo "  Comfort weight:  2 (fixed)"
    echo "========================================"
    echo ""

    local sim_args=(
        -c closed_loop_reactive_agents
        --ablation none
        --experiment-name "$name"
        --gpus-per-sim "$GPUS_PER_SIM"
        --threads "$THREADS"
        -p "cost_estimator.weighted_metrics.1.parameters.weight=${ttc_weight}"
        -p "cost_estimator.weighted_metrics.0.parameters.weight=${progress_weight}"
    )
    sim_args+=("${QUICK_ARGS[@]}")

    mosaic simulate "${sim_args[@]}"

    mosaic analyze | tee "$RESULTS_DIR/$name.txt"
}

# 3x3 grid: TTC in {3.5, 7, 14} x progress in {2.5, 5, 10}
# Skip center cell (7, 5) — already in results/val14-reactive.txt
for ttc in 3.5 7 14; do
    for prog in 2.5 5 10; do
        if [[ "$ttc" == "7" && "$prog" == "5" ]]; then
            echo ""
            echo "=== Skipping center cell (TTC=7, progress=5): use results/val14-reactive.txt ==="
            continue
        fi
        run_cell "$ttc" "$prog"
    done
done

echo ""
echo "=== Sensitivity study complete. Results saved in $RESULTS_DIR ==="
echo "=== Center cell baseline: results/val14-reactive.txt ==="
