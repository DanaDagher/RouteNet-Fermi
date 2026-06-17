# Step 6 — Build Reduced-Input Dataset Variants (Column Dropping)

**Date:** 2026-06-10 (originally for k ∈ {25, 50, 75}); **revised 2026-06-16** for k ∈ {30, 50, 70}.
**Branch:** xai-protocol-b (created off xai-features at commit 6bf1113)
**Status:** COMPLETE — verified by smoke test (13/13 PASS), confirmed by Dana before this report was written.

> **Revision note (2026-06-16, v7):** The threshold sweep was changed from
> {25, 50, 75} to **{30, 50, 70}** so that k=30 isolates exactly the top-3
> features sitting above the ~19–20× attribution cliff observed in both IG and
> KernelSHAP (Step 5). The 8 k25/k75 configs were deleted and 8 k30/k70 configs
> generated; the k50 pair and baseline are unchanged. Model count stays 13.
> Re-verified: audit + smoke test 13/13 PASS. The tables below reflect the
> current (v7) configs. See THESIS_DECISIONS §6/§7 and changelog v7.

---

## Objective

Implement the column-dropping infrastructure required by the locked retraining
protocol (CLAUDE.md / THESIS_DECISIONS §7): for each XAI method (IG, KernelSHAP)
and each threshold k ∈ {30, 50, 70}, partition the 10 path scalars into a
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
| `configs/ig/k{30,50,70}_{relevant,irrelevant}.json` | NEW — 6 IG variant configs |
| `configs/kernel_shap/k{30,50,70}_{relevant,irrelevant}.json` | NEW — 6 KernelSHAP variant configs |
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
| ig/k30_relevant | sigma, traffic, packets | 3 | 10 |
| ig/k30_irrelevant | the other 7 | 7 | 14 |
| ig/k50_relevant | sigma, traffic, packets, pkts_lambda_on, eq_lambda | 5 | 12 |
| ig/k50_irrelevant | avg_t_off, ar_a, exp_max_factor, avg_t_on, avg_pkts_lambda | 5 | 12 |
| ig/k70_relevant | top-7 by IG | 7 | 14 |
| ig/k70_irrelevant | exp_max_factor, avg_t_on, avg_pkts_lambda | 3 | 10 |
| kernel_shap/k30_relevant | traffic, sigma, packets | 3 | 10 |
| kernel_shap/k30_irrelevant | the other 7 | 7 | 14 |
| kernel_shap/k50_relevant | traffic, sigma, packets, eq_lambda, pkts_lambda_on | 5 | 12 |
| kernel_shap/k50_irrelevant | ar_a, avg_t_off, exp_max_factor, avg_t_on, avg_pkts_lambda | 5 | 12 |
| kernel_shap/k70_relevant | top-7 by SHAP | 7 | 14 |
| kernel_shap/k70_irrelevant | exp_max_factor, avg_t_on, avg_pkts_lambda | 3 | 10 |

Each JSON records `xai_method`, `k`, `partition`, `kept_features`,
`dropped_features`, `n_path_scalars_kept`, `path_embedding_input_dim`.

**Update (2026-06-17):** the 6 **random-control** configs (`configs/random/k{30,50,70}_{relevant,irrelevant}.json`) have now been generated from `rankings/random.csv` and are **core, not conditional** — there are 19 config files total (baseline + 6 IG + 6 KernelSHAP + 6 random). Step 7 trains 13 unique models: baseline + 6 principled (IG ≡ KernelSHAP, so the KernelSHAP configs are not trained separately) + 6 random. See THESIS_DECISIONS §7.

### Convention note

`relevant_k` and `irrelevant_k` **partition** the 10 scalars, so the kept
counts are complements: **3/7 at k=30, 5/5 at k=50, 7/3 at k=70**. (The v6
PIPELINE Step 6 block had garbled example counts on its old line 175; that
block was rewritten in v7 to state the exact k×10 mapping, resolving the
earlier inconsistency.)

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

**Next: Step 7** — the 13 unique retrainings (150 epochs × 2,000 samples each,
seed 42): baseline + 6 principled + 6 random. Compute: **GCP** (company-provided;
Dana runs it by hand). The random-control retrainings are now **core** (they
carry the principled-vs-random fidelity result).
