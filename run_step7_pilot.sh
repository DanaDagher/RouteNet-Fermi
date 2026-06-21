#!/usr/bin/env bash
# run_step7_pilot.sh — Step 7-PILOT: the k=30 comparison on a SUB-DATASET, on CPU.
#
# WHAT THIS IS
#   A compute-constrained PILOT execution of the Step 7 retraining comparison,
#   meant to run on a local PC (no GPU) while GCP/RTX access is pending. It
#   trains the same 5 models as the k=30 slice, but on a FIXED seeded
#   sub-dataset of N_TRAIN simulations, cached so epochs are cheap on CPU.
#   Full rationale + validity discussion: results/step7_pilot_subdataset.md
#
# WHAT IT IS NOT
#   Not a replacement for the full study. The full-data pipelines are untouched
#   and ready for GPU:
#     - run_step7_all.sh  (13 models, full data)
#     - run_step7_k30.sh  (5 models,  full data)
#   When GPU access arrives, run those; they supersede this pilot.
#
# WHY THE BASELINE IS RE-TRAINED HERE
#   All 5 models — baseline included — must train on the IDENTICAL sub-dataset
#   with IDENTICAL hyperparameters, so the only thing that differs between them
#   is the kept input features. The upstream checkpoint was trained on the FULL
#   data, so it cannot serve as the reference for a sub-dataset comparison.
#
# Usage (from repo root, branch xai-protocol-b, venv active):
#   bash run_step7_pilot.sh > step7_pilot.log 2>&1 &
#   tail -f step7_pilot.log
#
# Safe to re-run: a cell whose metrics.json exists is skipped; an interrupted
# cell resumes from its latest epoch checkpoint.

set -u
export PYTHONHASHSEED=42

cd "$(dirname "$0")"

# ── knobs (edit these) ───────────────────────────────────────────────────────
N_TRAIN=500     # training sub-dataset size (fixed, seeded). 500 ≈ overnight; 1000 ≈ ~2x.
EPOCHS=40       # passes over the sub-dataset (cheap once cached)
VAL=100         # per-epoch validation size (kept small so val doesn't dominate)
TEST=300        # FINAL test eval size, SAME for all 5 models (comparable)
PY=venv/Scripts/python.exe   # use 'python' on Linux/GCP

OUT_ROOT="checkpoints/pilot_n${N_TRAIN}"

# config-path : output-dir pairs (5 cells, baseline first)
CELLS=(
  "configs/baseline/full.json         ${OUT_ROOT}/baseline_seed42"
  "configs/ig/k30_relevant.json       ${OUT_ROOT}/principled/k30_relevant"
  "configs/ig/k30_irrelevant.json     ${OUT_ROOT}/principled/k30_irrelevant"
  "configs/random/k30_relevant.json   ${OUT_ROOT}/random/k30_relevant"
  "configs/random/k30_irrelevant.json ${OUT_ROOT}/random/k30_irrelevant"
)

echo "Step 7-PILOT  | N_TRAIN=$N_TRAIN  EPOCHS=$EPOCHS  VAL=$VAL  TEST=$TEST"
echo "Output root: $OUT_ROOT"

FAILED=()
for cell in "${CELLS[@]}"; do
    read -r config output <<< "$cell"
    echo
    echo "######################################################################"
    echo "# CELL: $config -> $output    $(date)"
    echo "######################################################################"
    if [ -f "$output/metrics.json" ]; then
        echo "metrics.json exists — cell already complete, skipping."
        continue
    fi
    if "$PY" run_step7_train.py --config "$config" --output "$output" \
            --max-train-samples "$N_TRAIN" --cache \
            --epochs "$EPOCHS" --steps-per-epoch "$N_TRAIN" \
            --val-samples "$VAL" --max-test-samples "$TEST"; then
        echo "CELL OK: $output"
    else
        echo "CELL FAILED: $output (continuing)"
        FAILED+=("$output")
    fi
done

echo
echo "######################################################################"
echo "# STEP 7-PILOT FINISHED  $(date)"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "# All 5 cells completed.  Results under: $OUT_ROOT"
else
    echo "# FAILED cells:"; printf '#   %s\n' "${FAILED[@]}"
fi
echo "######################################################################"
