# Step 6 — Build Reduced-Input Dataset Variants (Column Dropping)

**Date:** 2026-06-10
**Branch:** xai-protocol-b (created off xai-features at commit 6bf1113)
**Status:** COMPLETE — verified by smoke test (13/13 PASS), confirmed by Dana before this report was written.

---

## Objective

Implement the column-dropping infrastructure required by the locked retraining
protocol (CLAUDE.md / THESIS_DECISIONS §7): for each XAI method (IG, KernelSHAP)
and each threshold k ∈ {25, 50, 75}, partition the 10 path scalars into a
`relevant_k` variant (keep top-k%) and an `irrelevant_k` variant (keep
bottom-(100−k)%), with the model's `path_embedding` input dimension genuinely
reduced per variant — a real input-dimension change, not a runtime mask.

This is a **branch + code change** (PIPELINE Step 6). All modifications live
exclusively on `xai-protocol-b`; `xai-features` and `main` are untouched.

---

## Files Produced / Modified

| File | Change |
|---|---|
| `delay_model.py` | MODIFIED — variant-aware model (see below) |
| `traffic_models/delay/data_generator.py` | MODIFIED — feature-dropping loader (see below) |
| `configs/baseline/full.json` | NEW — baseline config (10 scalars, dim=17) |
| `configs/ig/k{25,50,75}_{relevant,irrelevant}.json` | NEW — 6 IG variant configs |
| `configs/kernel_shap/k{25,50,75}_{relevant,irrelevant}.json` | NEW — 6 KernelSHAP variant configs |
| `run_step6_smoke.py` | NEW — smoke test over all 13 configs |

Committed on `xai-protocol-b` (commit `5d346a2`), pushed to
`DanaDagher/RouteNet-Fermi`. Review PR: #1 (`xai-protocol-b` → `xai-features`,
marked DO NOT MERGE — the branch is permanent, per contract).

---

## Code changes

### `delay_model.py`

- Added `PATH_SCALAR_FEATURES`, the ordered list of the 10 per-flow path
  scalars (order matches the original model's `tf.concat`).
- `RouteNet_Fermi.__init__` now accepts `kept_path_scalars` (defaults to all
  10 → identical to upstream). Unknown names are rejected by assertion;
  features are always concatenated in `PATH_SCALAR_FEATURES` order regardless
  of the order given.
- `path_embedding` input dimension = `len(kept_path_scalars) + 7`
  (the 7-bit model one-hot is always included — it is one of the 12 structural
  inputs, never droppable).
- `call()` refactored: structural inputs (traffic, packets, length, model,
  capacity, policy, queue_size, priority, weight, 5 graph tensors) are always
  extracted; the path-embedding concat is built dynamically from
  `kept_path_scalars` only.
- All other layers (GRU cells, hidden state 32, T=8 iterations, readout,
  z-score constants) are byte-identical to upstream.

### `traffic_models/delay/data_generator.py`

- Added `_DROPPABLE_PATH_SCALARS` = the 8 path scalars other than
  `traffic`/`packets`.
- `generator()` and `input_fn()` accept `dropped_features`; dropped scalars
  are removed from the yielded input dict, and `input_fn` builds the TF
  `output_signature` dynamically to match.
- Bug found and fixed during smoke testing: TF's
  `tf.data.Dataset.from_generator` serialises Python list `args` to numpy
  arrays of byte-strings before calling the generator, so the idiom
  `set(dropped_features or [])` raised
  `ValueError: truth value of an array ... is ambiguous` for every non-empty
  drop list. Replaced with an explicit decode loop (`f.decode('UTF-8')` per
  element, `str(f)` fallback).

### The traffic/packets dual role (the subtle design point)

`traffic` and `packets` serve two purposes in RouteNet-Fermi:

1. **Path-embedding inputs** — droppable under the protocol (they are 2 of the
   10 XAI-scoped scalars).
2. **Structural computations** — link load (`sum(traffic)/capacity`) and
   transmission delay (`pkt_size = traffic/packets`), which define the
   physics of the model and are never removable.

Resolution: when a variant drops `traffic` or `packets`, they are excluded
from the **path-embedding concat** (so the learned per-flow representation
cannot see them) but remain in the **data dict** for the structural
computations. This is why `_DROPPABLE_PATH_SCALARS` has 8 entries, not 10,
and why the smoke test checks them separately.

---

## The 13 configurations

Rankings source: Step 5 main rankings (`rankings/ig.csv`,
`rankings/kernel_shap.csv`, median-reference runs).

| Config | Kept (path-embedding) | n kept | embed dim |
|---|---|---|---|
| baseline/full | all 10 | 10 | 17 |
| ig/k25_relevant | sigma, traffic | 2 | 9 |
| ig/k25_irrelevant | the other 8 | 8 | 15 |
| ig/k50_relevant | sigma, traffic, packets, pkts_lambda_on, eq_lambda | 5 | 12 |
| ig/k50_irrelevant | avg_t_off, ar_a, exp_max_factor, avg_t_on, avg_pkts_lambda | 5 | 12 |
| ig/k75_relevant | top-7 by IG | 7 | 14 |
| ig/k75_irrelevant | exp_max_factor, avg_t_on, avg_pkts_lambda | 3 | 10 |
| kernel_shap/k25_relevant | traffic, sigma | 2 | 9 |
| kernel_shap/k25_irrelevant | the other 8 | 8 | 15 |
| kernel_shap/k50_relevant | traffic, sigma, packets, eq_lambda, pkts_lambda_on | 5 | 12 |
| kernel_shap/k50_irrelevant | ar_a, avg_t_off, exp_max_factor, avg_t_on, avg_pkts_lambda | 5 | 12 |
| kernel_shap/k75_relevant | top-7 by SHAP | 7 | 14 |
| kernel_shap/k75_irrelevant | exp_max_factor, avg_t_on, avg_pkts_lambda | 3 | 10 |

Each JSON records `xai_method`, `k`, `partition`, `kept_features`,
`dropped_features`, `n_path_scalars_kept`, `path_embedding_input_dim`.
Random-control configs (conditional §7.B) are not generated yet; they will be
produced from `rankings/random.csv` only if §7.B is triggered.

### Convention note (flagged inconsistency)

PIPELINE.md line 175 gives example `irrelevant_k` feature counts
(k=25 → "3 features", k=75 → "2 features") that contradict the locked
convention stated immediately below it (lines 178–179) and in CLAUDE.md:
`relevant_k` and `irrelevant_k` **partition** the 10 scalars, so the counts
are complements — 2/8 at k=25, 5/5 at k=50, 7/3 at k=75. The implementation
follows the locked convention (which both contract documents state
identically); the line-175 parentheticals appear to be a leftover editing
error.

---

## Smoke test (run_step6_smoke.py)

For every one of the 13 configs, on the real `all_multiplexed` test split:

1. **Model instantiation** — `path_embedding` input dim equals the config's
   `path_embedding_input_dim`. ✔ 13/13
2. **Dataset loads** — `input_fn(..., dropped_features=...)` yields a batch
   without error. ✔ 13/13
3. **Dropped features absent** — every droppable feature in the config's
   drop list is missing from the batch dict; `traffic`/`packets` remain
   present even when listed as dropped (structural). ✔ 13/13
4. **Kept features present** — every kept feature is in the batch dict. ✔ 13/13
5. **Forward pass** — `model(batch)` runs to completion, output shape
   (272, 1) on the first test sample batch. ✔ 13/13

**Result: 13/13 PASS** (exceeds the PIPELINE stop criterion, which required
only one variant config to be smoke-tested). Run on CPU, TF 2.6.5,
2026-06-09.

---

## Verdict

Step 6 is complete per the PIPELINE stop criterion:

- Branch `xai-protocol-b` created, committed, and pushed.
- 12 variant configs + 1 baseline config generated from the Step 5 rankings.
- Smoke test confirms all 13 configurations load, initialise with the correct
  reduced input dimension, and complete a forward pass.

**Next: Step 7** — the 13 retrainings (150 epochs × 2,000 samples each,
seed 42), pending GPU access (Sogeti RTX 4090, awaiting Mouna's SSH
confirmation). Random-control retrainings (§7.B) remain conditional.
