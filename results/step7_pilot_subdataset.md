# Step 7-PILOT — The k=30 Comparison on a Sub-Dataset (local CPU)

**Date:** 2026-06-21
**Branch:** xai-protocol-b
**Status:** STAGED — scripts ready and CPU-validated; full run pending (Dana runs on her PC).
**Relation to the contract:** a new INTERIM step inserted between Step 6 and the
full Step 7. It does **not** modify or replace the v7 pipeline; pending the
IMT technical advisor's sign-off to fold into PIPELINE/THESIS_DECISIONS as a documented step.

---

## Why this step exists

The v7 Step 7 retraining matrix needs a GPU. As of 2026-06-21 **neither GCP nor
the Sogeti RTX 4090 is available**, and the deliverable deadline is near. Rather
than block, we run a **pilot** of the faithfulness comparison **on a small,
fixed sub-dataset** that fits on Dana's CPU-only PC.

This is a deliberate, documented compute-constrained design — **not** a change
to the methodology. When GPU access arrives, the full-data study runs unchanged
(`run_step7_all.sh` / `run_step7_k30.sh`) and supersedes this pilot.

## What it does

Trains the **5 models of the k=30 slice**, all on the **same** fixed seeded
sub-dataset of `N_TRAIN` simulations, with the **same** hyperparameters:

| # | Model | Config | Kept path scalars | Tests |
|---|---|---|---|---|
| 1 | Baseline | `configs/baseline/full.json` | all 10 | reference |
| 2 | Principled — relevant | `configs/ig/k30_relevant.json` | sigma, traffic, packets | sufficiency |
| 3 | Principled — irrelevant | `configs/ig/k30_irrelevant.json` | the other 7 | necessity |
| 4 | Random — relevant | `configs/random/k30_relevant.json` | 3 random | control |
| 5 | Random — irrelevant | `configs/random/k30_irrelevant.json` | the other 7 | control mirror |

Principled = IG (IG ≡ KernelSHAP at k=30). Driver: `run_step7_pilot.sh`.

## Why the baseline is RE-TRAINED here (not reused)

All 5 models — baseline included — must train on the **identical sub-dataset**
with **identical hyperparameters**, so the *only* difference between any two
models is the kept input features. The upstream pretrained checkpoint was
trained on the **full** dataset (and an unknown run/seed), so it **cannot** be
the reference for a sub-dataset comparison. Same data + same settings +
only-features-differ ⇒ the baseline must be retrained on this sub-dataset.

## How it stays valid (and where it doesn't)

- ✅ **Internally valid.** The faithfulness claim is *relative*: principled-relevant
  ≈ baseline (sufficiency), principled-irrelevant ≪ baseline (necessity),
  principled-relevant ≫ random-relevant (the ranking matters). Because all 5
  models share the same sub-dataset, hyperparameters, seed, and test set, those
  deltas are meaningful.
- ⚠️ **Absolute accuracy is not comparable to the paper.** Training on ~500–1000
  samples instead of ~75,000 (and fewer effective steps) means test MAPE will be
  far above the paper's 4.71%. That is expected and affects all 5 models equally.
- ⚠️ **External validity is reduced.** "On a small training set, the top-3 features
  dominate" is a weaker statement than the full-data result. Hence: pilot, to be
  confirmed at full scale on GPU.
- ✅ **Verify-baseline-first.** The driver trains the baseline first. If the
  expected pattern (relevant ≈ baseline, irrelevant & random worse) does not
  appear even qualitatively, increase `EPOCHS`/`N_TRAIN` before reading too much
  into it.

## Method details (deterministic + fast)

- **Fixed subsample:** the training set is loaded in deterministic order
  (`shuffle=False`) and the first `N_TRAIN` simulations are taken — so every
  model trains on the exact same simulations. Per-epoch variety comes from a
  TF-level shuffle (seed 42) applied after caching.
- **Caching:** the subsample is parsed once (networkx/DatanetAPI) and cached;
  later epochs reuse it. Measured on Dana's CPU: **~1.4 s/step** first
  (parsing) epoch → **~0.26 s/step** cached — about an **8× speedup**.
- **Benign warning:** TF prints `cache_dataset_ops ... did not fully read the
  dataset being cached` on epoch 1. It is harmless here — the measured epoch-2
  speedup (≈222s → ≈39s) confirms the cache is effective; TF only warns because
  `steps_per_epoch` equals the subsample size. Results and reproducibility are
  unaffected.
- **Hyperparameters:** unchanged from the paper §IV-D except total step budget
  (Adam 1e-3, MAPE loss, hidden 32, T=8, seed 42). `steps_per_epoch = N_TRAIN`
  (one pass/epoch); `EPOCHS` passes total.
- **Evaluation:** every model is finally evaluated on the **same** fixed test
  subsample (`TEST`, default 300) so test MAPE is comparable. A small per-epoch
  validation (`VAL`, default 100) drives the convergence curve / checkpointing.

## Time estimate (CPU, from calibration)

Per model ≈ first-epoch parse + cached epochs + validation. Approximate totals:

| N_TRAIN | EPOCHS | ≈ all 5 models |
|---|---|---|
| **500** | 40 | ~7–8 h (overnight) |
| 1000 | 40 | ~14–16 h (≈ a day) |

Defaults in `run_step7_pilot.sh`: `N_TRAIN=500, EPOCHS=40, VAL=100, TEST=300`.
Change `N_TRAIN=1000` at the top of the script for the larger pilot.

## What is NOT changed (kept intact for GPU)

- `run_step7_all.sh` (13 models, full data) — untouched.
- `run_step7_k30.sh` (5 models, full data) — untouched.
- All 19 `configs/`, `delay_model.py`, `data_generator.py`, `run_step6_smoke.py`
  — untouched.
- `run_step7_train.py` — only **optional** flags added (`--max-train-samples`,
  `--val-samples`, `--max-test-samples`, `--cache`). With no flags it behaves
  exactly as before, so the full-data pipelines are byte-for-byte unaffected.

## Outputs

Under `checkpoints/pilot_n<N_TRAIN>/`: per cell a `metrics.json`,
`training_log.csv`, `training_config.json`, and per-epoch weight checkpoints —
the same schema as the full Step 7, so the Step 8 analysis works on either.

## Next

1. Dana runs `run_step7_pilot.sh` on her PC (baseline first; sanity-check the
   pattern), reports `metrics.json` for the 5 cells.
2. Build the pilot fidelity summary (relevant vs irrelevant vs random).
3. When GPU appears: run the full-data Step 7; the pilot becomes a documented
   compute-constrained precursor.
4. IMT technical advisor sign-off → fold this step into PIPELINE/THESIS_DECISIONS (v8).
