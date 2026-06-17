#!/usr/bin/env bash
# run_step7_all.sh — Step 7: run the 13-cell retraining matrix sequentially.
#
# Matrix (13 UNIQUE models): baseline first (sanity gate vs paper MAPE 4.71%),
# then the 6 PRINCIPLED variants, then the 6 RANDOM-control variants.
#
# Why no separate KernelSHAP cells: at k in {30,50,70} the IG and KernelSHAP
# rankings produce IDENTICAL kept-feature SETS at every cell (they differ only
# by adjacent swaps that never cross a cut boundary; verified by
# audit_steps_1_to_6.py). Since the model is order-invariant, training the
# KernelSHAP configs would reproduce the IG models bit-for-bit. The 6 IG cells
# ARE the principled variants (IG == KernelSHAP here). See THESIS_DECISIONS §7.
#
# The random-control cells are now CORE (not conditional): because IG == SHAP,
# the meaningful faithfulness test is principled-vs-random, so random carries
# the fidelity result.
#
# A cell is SKIPPED if its metrics.json already exists, so the script is safe to
# re-run after an interruption (run_step7_train.py also resumes mid-cell).
#
# Compute: GCP (company-provided). Run on the GCP VM, NOT locally.
# Usage (from repo root, branch xai-protocol-b, conda env active):
#   nohup ./run_step7_all.sh > step7_all.log 2>&1 &
#   tail -f step7_all.log

set -u
export PYTHONHASHSEED=42

cd "$(dirname "$0")"

# config-path : output-dir pairs, locked order
# 1 baseline + 6 principled (IG == KernelSHAP) + 6 random = 13 unique models.
CELLS=(
  "configs/baseline/full.json              checkpoints/baseline_seed42"
  "configs/ig/k30_relevant.json            checkpoints/principled/k30_relevant"
  "configs/ig/k30_irrelevant.json          checkpoints/principled/k30_irrelevant"
  "configs/ig/k50_relevant.json            checkpoints/principled/k50_relevant"
  "configs/ig/k50_irrelevant.json          checkpoints/principled/k50_irrelevant"
  "configs/ig/k70_relevant.json            checkpoints/principled/k70_relevant"
  "configs/ig/k70_irrelevant.json          checkpoints/principled/k70_irrelevant"
  "configs/random/k30_relevant.json        checkpoints/random/k30_relevant"
  "configs/random/k30_irrelevant.json      checkpoints/random/k30_irrelevant"
  "configs/random/k50_relevant.json        checkpoints/random/k50_relevant"
  "configs/random/k50_irrelevant.json      checkpoints/random/k50_irrelevant"
  "configs/random/k70_relevant.json        checkpoints/random/k70_relevant"
  "configs/random/k70_irrelevant.json      checkpoints/random/k70_irrelevant"
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
    if python run_step7_train.py --config "$config" --output "$output"; then
        echo "CELL OK: $output"
    else
        echo "CELL FAILED: $output (continuing with next cell)"
        FAILED+=("$output")
    fi
done

echo
echo "######################################################################"
echo "# STEP 7 MATRIX FINISHED  $(date)"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "# All cells completed."
else
    echo "# FAILED cells:"
    printf '#   %s\n' "${FAILED[@]}"
fi
echo "######################################################################"
