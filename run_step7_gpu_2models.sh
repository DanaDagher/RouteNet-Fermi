#!/usr/bin/env bash
# run_step7_gpu_2models.sh — Step 7 (TRIMMED SCOPE for GPU rerun).
#
# Trains ONLY the 2 cells needed for the final D3.2 story:
#   1. baseline               — all 10 path scalars (reference)
#   2. principled_relevant    — keep top-3 {sigma, traffic, packets}
#
# The retrained principled_irrelevant model is DROPPED from the plan because
# perturbing the top-3 at inference on the baseline is the necessity test:
# it neutralises {sigma, traffic, packets} the only way the model allows
# (their values become noise), which is what "irrelevant" was trying to test.
# See results/two_doors_and_ml_locus_REPORT.md §2.
#
# Random controls (random_relevant / random_irrelevant) are DROPPED because
# the perturbation experiment (run_step7_perturbation.py) provides fidelity
# controls via XAI-bottom-ranked features (eq_lambda, avg_pkts_lambda).
#
# Hyperparameters: paper section IV.D — 150 epochs of 2000 steps, Adam lr=1e-3,
# MAPE loss, hidden state 32, T=8, seed 42. Full train / test splits.
#
# A cell is SKIPPED if its metrics.json already exists (safe to re-run after
# interruption; run_step7_train.py also resumes mid-cell from the last
# per-epoch checkpoint).
#
# Compute: RTX 4090 (Sogeti sogeti147). Run on the GPU box, NOT locally.
# Usage (from repo root, branch xai-protocol-b, venv active):
#   nohup ./run_step7_gpu_2models.sh > step7_gpu.log 2>&1 &
#   tail -f step7_gpu.log
#
# Expected wall-clock on RTX 4090: ~20-40 min per cell → ~40-80 min total.

set -u
export PYTHONHASHSEED=42

cd "$(dirname "$0")"

CELLS=(
  "configs/baseline/full.json          checkpoints/gpu_full/baseline_seed42"
  "configs/ig/k30_relevant.json        checkpoints/gpu_full/principled/k30_relevant"
)

FAILED=()
for cell in "${CELLS[@]}"; do
    read -r config output <<< "$cell"
    echo
    echo "######################################################################"
    echo "# CELL: $config -> $output"
    echo "# $(date)"
    echo "######################################################################"
    if [ -f "$output/metrics.json" ]; then
        echo "metrics.json exists — cell already complete, skipping."
        continue
    fi
    # --cache: first pass parses the tfrecords, later epochs skip re-parsing
    if python run_step7_train.py --config "$config" --output "$output" --cache; then
        echo "CELL OK: $output"
    else
        echo "CELL FAILED: $output (continuing with next cell)"
        FAILED+=("$output")
    fi
done

echo
echo "######################################################################"
echo "# STEP 7 GPU 2-MODEL RUN FINISHED  $(date)"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "# All 2 cells completed."
    echo "# NEXT: run perturbation pass on the retrained baseline"
    echo "#   python run_step7_perturbation.py --pool 300 --n-seeds 5"
    echo "#   (CELLS in that script already points at checkpoints/gpu_full/baseline_seed42)"
else
    echo "# FAILED cells:"
    printf '#   %s\n' "${FAILED[@]}"
fi
echo "######################################################################"
