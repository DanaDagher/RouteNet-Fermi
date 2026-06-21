#!/usr/bin/env bash
# run_step7_k30.sh — Step 7 (REDUCED SCOPE): train the k=30 slice only.
#
# 5 unique models instead of the full 13. Rationale and the full comparison
# logic (sufficiency / necessity / control) are documented in
# results/step7_scope_decision.md (2026-06-17). The full {30,50,70} sweep
# remains available via run_step7_all.sh — nothing was deleted.
#
# Cells:
#   1 baseline           — all 10 path scalars (reference)
#   2 principled relevant   — keep top-3 {sigma,traffic,packets}  (IG == KernelSHAP)
#   3 principled irrelevant — drop top-3, keep bottom-7
#   4 random relevant       — keep random-3 (negative control)
#   5 random irrelevant     — drop random-3 (control mirror)
#
# A cell is SKIPPED if its metrics.json already exists, so the script is safe to
# re-run after an interruption (run_step7_train.py also resumes mid-cell).
#
# Compute: GCP (company-provided). Run on the GCP VM, NOT locally.
# Usage (from repo root, branch xai-protocol-b, conda env active):
#   nohup ./run_step7_k30.sh > step7_k30.log 2>&1 &
#   tail -f step7_k30.log

set -u
export PYTHONHASHSEED=42

cd "$(dirname "$0")"

# config-path : output-dir pairs, locked order (baseline first as sanity gate)
CELLS=(
  "configs/baseline/full.json         checkpoints/baseline_seed42"
  "configs/ig/k30_relevant.json       checkpoints/principled/k30_relevant"
  "configs/ig/k30_irrelevant.json     checkpoints/principled/k30_irrelevant"
  "configs/random/k30_relevant.json   checkpoints/random/k30_relevant"
  "configs/random/k30_irrelevant.json checkpoints/random/k30_irrelevant"
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
echo "# STEP 7 (k=30 slice) FINISHED  $(date)"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "# All 5 cells completed."
else
    echo "# FAILED cells:"
    printf '#   %s\n' "${FAILED[@]}"
fi
echo "######################################################################"
