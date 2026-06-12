#!/usr/bin/env bash
# run_step7_all.sh — Step 7: run the 13-cell retraining matrix sequentially.
#
# Order: baseline first (sanity gate vs paper MAPE 4.71%), then IG, then
# KernelSHAP. A cell is SKIPPED if its metrics.json already exists, so the
# script is safe to re-run after an interruption (run_step7_train.py also
# resumes mid-cell from the latest checkpoint).
#
# Usage (from repo root, branch xai-protocol-b, conda env active):
#   nohup ./run_step7_all.sh > step7_all.log 2>&1 &
#   tail -f step7_all.log
#
# Random-control cells (PIPELINE section 7.B) are intentionally NOT here —
# they are conditional on the 13 main cells finishing cleanly with time left.

set -u
export PYTHONHASHSEED=42

cd "$(dirname "$0")"

# config-path : output-dir pairs, locked order
CELLS=(
  "configs/baseline/full.json              checkpoints/baseline_seed42"
  "configs/ig/k25_relevant.json            checkpoints/ig/k25_relevant"
  "configs/ig/k25_irrelevant.json          checkpoints/ig/k25_irrelevant"
  "configs/ig/k50_relevant.json            checkpoints/ig/k50_relevant"
  "configs/ig/k50_irrelevant.json          checkpoints/ig/k50_irrelevant"
  "configs/ig/k75_relevant.json            checkpoints/ig/k75_relevant"
  "configs/ig/k75_irrelevant.json          checkpoints/ig/k75_irrelevant"
  "configs/kernel_shap/k25_relevant.json   checkpoints/kernel_shap/k25_relevant"
  "configs/kernel_shap/k25_irrelevant.json checkpoints/kernel_shap/k25_irrelevant"
  "configs/kernel_shap/k50_relevant.json   checkpoints/kernel_shap/k50_relevant"
  "configs/kernel_shap/k50_irrelevant.json checkpoints/kernel_shap/k50_irrelevant"
  "configs/kernel_shap/k75_relevant.json   checkpoints/kernel_shap/k75_relevant"
  "configs/kernel_shap/k75_irrelevant.json checkpoints/kernel_shap/k75_irrelevant"
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
