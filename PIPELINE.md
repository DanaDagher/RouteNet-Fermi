# PIPELINE.md

**Project:** XAI for GNN-based QoS Prediction in SDN — M2 Thesis + D3.2 Section 5
**Companion to:** `THESIS_DECISIONS.md` (v7)
**Reference paper:** Ferriol-Galmés et al., *RouteNet-Fermi*, IEEE/ACM ToN 2023, DOI 10.1109/TNET.2023.3269983.
**Version:** v7

Read `THESIS_DECISIONS.md` first. Every choice is justified there.

## Protocol summary (LOCKED)

**Retraining-based feature filtering**, confirmed by the IMT technical advisor. For each XAI method × each threshold k ∈ {30, 50, 70}, two reduced-input dataset variants are built by **dropping columns** (not masking values), and the model is retrained from scratch on each. **13 trained models in the core**; optionally +6 for a random-ranking control.

---

## Workflow overview

```
[Step 1]    Environment setup
[Step 2]    Baseline reproduction probe (upstream checkpoint, N=50)
[Step 2.5]  Sample validity check on all 5 sub-datasets, N=300
[Step 3]    Implement XAI methods (IG + KernelSHAP + random control)
[Step 3.5]  Pre-registration writeup
[Step 4]    Generate explanations on N=300 from all_multiplexed (using upstream pretrained)
[Step 5]    Aggregate and rank features globally (+ half-split check)
[Step 6]    Build reduced-input dataset variants (COLUMN DROPPING)
[Step 7]    Retrain RouteNet-Fermi from scratch — 13 unique models (baseline + 6 principled + 6 random)
[Step 8]    Build fidelity curves and compare
[Step 9]    Stability and cost analysis
[Step 10a]  Plausibility analysis (vs Step 3.5 pre-registration)
[Step 10b]  V2X-transferability writeup
[Step 11]   Write the thesis / D3.2 Section 5
```

After each step: **stop and report** to the user before moving on.

---

## Step 1 — Environment setup

**Goal:** Reproducible Python environment.

**Tasks:**
- Working directory: `C:\Users\ddagher\RouteNet-Fermi\` on branch `xai-features`.
- Activate venv. Python 3.7–3.9 per the repo's `requirements.txt`.
- Install: `tensorflow==2.x`, `numpy`, `pandas`, `matplotlib`, `networkx`, `shap`, `scikit-learn`, `seaborn`, `tqdm`.
- Confirm `data/traffic_models/{constant_bitrate, onoff, autocorrelated, modulated, all_multiplexed}/test/` exist.
- Confirm `traffic_models/delay/<tm>/ckpt_dir_<tm>/` exist for all 5 sub-datasets.
- Smoke test: `predict.py` on one sub-dataset with `ds_test.take(5)` to confirm the model loads.

**Stop criterion:** smoke test completes without error and prints a non-trivial MAPE. Report and wait before Step 2.

---

## Step 2 — Baseline reproduction probe

**Goal:** Quick sanity check on the upstream pretrained checkpoint before committing to N=300.

**Tasks:**
- For `all_multiplexed`: load `ckpt_dir_all_multiplexed/` best checkpoint (by val MAPE in filename).
- Run inference on `ds_test.take(50)`, `shuffle=False`.
- Compute mean and median MAPE on per-flow predictions.
- Save: `results/baseline_validation/probe_50_all_multiplexed.json`.
- Compare to RouteNet-Fermi paper Table V "Mixed" row (delay MAPE 4.71%) — order-of-magnitude check only.

**Stop criterion:** probe MAPE within an order of magnitude of paper's number. Report and wait before Step 2.5.

---

## Step 2.5 — Sample validity check on all 5 sub-datasets

**Goal:** Reproduce the RouteNet-Fermi paper's MAPE table on N=300 simulations across all 5 sub-datasets. Validates environment health and confirms N=300 is representative.

**Tasks:**
- For each sub-dataset `tm` ∈ {`constant_bitrate`, `onoff`, `autocorrelated`, `modulated`, `all_multiplexed`}:
  1. Load `ckpt_dir_<tm>/` best checkpoint.
  2. Inference on `ds_test.take(300)` from `data/traffic_models/<tm>/test/`, `shuffle=False`, deterministic.
  3. Compute mean MAPE and median MAPE on per-flow delay.
  4. Count total flows across the 300 simulations.
  5. Save: `results/baseline_validation/mape_<tm>.json` with `{tm, n_simulations, n_flows_total, mape_mean, mape_median, checkpoint_name}`.

- Build `results/baseline_validation/validation_table.md`:

| Sub-dataset | Paper Table V MAPE (delay) | Our MAPE on N=300 (mean / median) | Delta |
|---|---|---|---|
| `constant_bitrate` | Deterministic 4.43% | … | … |
| `onoff` | On-Off 2.90% | … | … |
| `autocorrelated` | A.Exponentials 2.62% | … | … |
| `modulated` | M.Exponentials 5.21% | … | … |
| `all_multiplexed` | Mixed 4.71% | … | … |

**Stop criterion:** for `all_multiplexed` specifically, mean MAPE on N=300 within ±1 absolute percentage point of paper's 4.71%. If outside, investigate before proceeding. Other 4 sub-datasets are informational. Report and wait before Step 3.

---

## Step 3 — Implement XAI methods

**Goal:** Working IG + KernelSHAP implementations over the 10 path scalars, plus a deterministic random-ranking helper.

### 3.1 — Integrated Gradients
- Baseline input: 10 path scalars replaced with `all_multiplexed` training-set median.
- N=50 interpolation steps.
- Native TF gradient computation.
- Reference: Sundararajan et al., ICML 2017.

### 3.2 — KernelSHAP (custom GNN wrapper, same design as CBR pilot)
- Fix the full graph (links, queues, other flows, topology). Vary only the **flow_idx=0** path scalars.
- 256 perturbations per flow.
- `shap.KernelExplainer` around the wrapper.
- Background: training-set median per feature (single reference, same as pilot).
- Reference: Lundberg & Lee, NeurIPS 2017.

### 3.3 — Random-ranking control
- Deterministic helper: `random_ranking(seed=42)` produces one random permutation of the 10 feature names.
- Used downstream by the **core** random-control retrainings (Step 7 cells 8–13).

**Sanity check for IG and KernelSHAP:** synthetic input where one feature is artificially extreme → method ranks it most important.

**Outputs:** `xai/integrated_gradients.py`, `xai/kernel_shap.py`, `xai/random_control.py`, `xai/tests/`, `xai/README.md`.

**Stop criterion:** IG and KernelSHAP pass their checks. Report and wait before Step 3.5.

---

## Step 3.5 — Pre-registration

**PARTIALLY COMPLETE — file created, content to be written by Dana before Step 5 aggregation is reviewed.**

File: `results/theoretical_expectations.md`

---

## Step 4 — Generate explanations

**Goal:** Per-feature importance scores for the 300 fixed `all_multiplexed` test simulations across both XAI methods, **using the upstream pretrained `ckpt_dir_all_multiplexed/` checkpoint** (this is the model the XAI methods explain; it is *not* the fidelity baseline — that comes from Step 7).

**Tasks:**
- Sample 300 simulations from `all_multiplexed` test with seed=42. Save indices to `explanation_set/indices.npy`.
- For each method M ∈ {IG, KernelSHAP}:
  - For each of the 300 simulations: compute per-feature importance over the 10 path scalars for `flow_idx=0`. Save to `explanations/{M}/sim_{i}.npz`.
  - Log wall-clock time per explanation to `explanations/{M}/timings.csv`.

**Stability runs:**
- IG with an alternate baseline (training-set mean, or uniform-random) → `explanations/ig_stability/`.
- KernelSHAP with a different background subsample → `explanations/kernel_shap_stability/`.

**Stop criterion:** 300 × 2 explanations + 2 stability runs done. Report timings and a quick visual check.

---

## Step 5 — Aggregate and rank features globally

**Goal:** One global ranking per XAI method + the random ranking + a half-split sanity check.

**Tasks:**
- For each method, average **|importance|** per feature across the 300 simulations.
- Sort descending → ranking of the 10 path scalars.
- Save: `rankings/ig.csv`, `rankings/kernel_shap.csv`, `rankings/random.csv`.
- Stability rankings: `rankings/ig_stability.csv`, `rankings/kernel_shap_stability.csv`.
- **Half-split Spearman check:** ranking on sims 0–149 vs 150–299 independently. Spearman per method to `rankings/halfsplit_check.json`. Document the value.
- **Traffic-class distribution at flow_idx=0:** how many of the 300 flow-0 selections were CBR / on-off / autocorrelated / modulated. Save to `rankings/flow0_class_distribution.json`.

**Stop criterion:** rankings committed. Report and wait before Step 6.

---

## Step 6 — Build reduced-input dataset variants (COLUMN DROPPING)

**Goal:** For each (XAI method, k), build two reduced-input configurations that drop specified feature columns from the input pipeline.

**Per XAI method M ∈ {IG, KernelSHAP, random} (all three configs generated; IG ≡ KernelSHAP, so only IG + random are trained — see Step 7):**

For each k ∈ {30, 50, 70}:
- **`relevant_k`**: keep the top-k% most important features; drop the bottom-(100−k)% columns.
- **`irrelevant_k`**: keep the bottom-(100−k)% least important features; drop the top-k% columns.

The mapping to feature counts is exact (k×10): **k=30 → top-3 / bottom-7**, **k=50 → top-5 / bottom-5**, **k=70 → top-7 / bottom-3**. The two partitions at each k are complementary (kept ∪ kept = all 10, disjoint).

**Why {30, 50, 70}** (revised from {25, 50, 75}, v7): Step 5 rankings place the same three features ({traffic, sigma, packets}, ~93% of attribution mass) above a ~19–20× attribution cliff in *both* methods. **k=30 isolates exactly this top-3 cliff set** — the cleanest fidelity test (relevant_30 = signal sufficiency, irrelevant_30 = signal necessity). k=50 is an intermediate control on the monotonicity of the trend; k=70 keeps top-7 to test whether padding the relevant set with noise-tail features degrades performance. Thresholds are derived from attribution *magnitudes*, not from downstream retraining performance, so the design does not condition on its own outcome.

**Implementation (column dropping, real input-dimension change):**

This is a **branch + code change**, not a runtime wrapper. Create branch `xai-protocol-b` off `xai-features` before this step.

On `xai-protocol-b`:
- Modify `data_generator.py` to accept a `dropped_features` list parameter, removing those columns from the path-feature dict during loading.
- Modify `delay_model.py` so the `path_embedding` layer's input dimension matches the kept-feature count for the variant being trained.
- Add a per-experiment config file: `configs/{M}/k{k}_{relevant|irrelevant}.json` listing `dropped_features` and the resulting `path_embedding_input_dim`.

**The 12 structural inputs are never in the `dropped_features` list.** Only path scalars are eligible for dropping.

**Outputs:** branch `xai-protocol-b` with the input-pipeline and model-init changes; 12 (or 18 if random) variant config files in `configs/`.

**Stop criterion:** branch pushed, configs generated, smoke test confirms one variant config loads and the model initializes with the correct reduced input dim. Report and wait before Step 7.

---

## Step 7 — Retrain RouteNet-Fermi — the 13 unique trainings

**Goal:** Train one model per variant configuration, all under identical hyperparameters and seed. Run on **GCP** (Dana executes by hand; Claude never runs remote compute). Driver: `run_step7_all.sh`.

### §7.A — Core matrix (13 unique models, all mandatory)

| # | Cell | Config | Output dir |
|---|---|---|---|
| 1 | Baseline — full 10 path scalars, seed 42 | `configs/baseline/full.json` | `checkpoints/baseline_seed42/` |
| 2 | Principled · k=30 · relevant | `configs/ig/k30_relevant.json` | `checkpoints/principled/k30_relevant/` |
| 3 | Principled · k=30 · irrelevant | `configs/ig/k30_irrelevant.json` | `checkpoints/principled/k30_irrelevant/` |
| 4 | Principled · k=50 · relevant | `configs/ig/k50_relevant.json` | `checkpoints/principled/k50_relevant/` |
| 5 | Principled · k=50 · irrelevant | `configs/ig/k50_irrelevant.json` | `checkpoints/principled/k50_irrelevant/` |
| 6 | Principled · k=70 · relevant | `configs/ig/k70_relevant.json` | `checkpoints/principled/k70_relevant/` |
| 7 | Principled · k=70 · irrelevant | `configs/ig/k70_irrelevant.json` | `checkpoints/principled/k70_irrelevant/` |
| 8 | Random · k=30 · relevant | `configs/random/k30_relevant.json` | `checkpoints/random/k30_relevant/` |
| 9 | Random · k=30 · irrelevant | `configs/random/k30_irrelevant.json` | `checkpoints/random/k30_irrelevant/` |
| 10 | Random · k=50 · relevant | `configs/random/k50_relevant.json` | `checkpoints/random/k50_relevant/` |
| 11 | Random · k=50 · irrelevant | `configs/random/k50_irrelevant.json` | `checkpoints/random/k50_irrelevant/` |
| 12 | Random · k=70 · relevant | `configs/random/k70_relevant.json` | `checkpoints/random/k70_relevant/` |
| 13 | Random · k=70 · irrelevant | `configs/random/k70_irrelevant.json` | `checkpoints/random/k70_irrelevant/` |

**The "principled" cells are trained from `configs/ig/` and equal KernelSHAP** — at k ∈ {30,50,70} the two rankings produce identical kept-feature sets at every cell (verified by `audit_steps_1_to_6.py`), so the 6 principled models cover both methods. Training the 6 KernelSHAP configs would reproduce these bit-for-bit and is therefore skipped. The random control is **core, not conditional**: because IG ≡ KernelSHAP, the fidelity headline is principled-vs-random. See THESIS_DECISIONS §7.

**Hyperparameters for every run (paper §IV.D, locked):**
- 150 epochs × 2,000 samples per epoch.
- Adam optimizer, lr = 0.001.
- Loss: MeanAbsolutePercentageError (delay).
- Hidden state size: 32.
- Message-passing iterations: T = 8.
- Seed: 42 (TF, NumPy, Python `random`).
- Same train/validation/test split for all runs.

**Per-run logging (mandatory):**
- `checkpoints/{cell}/metrics.json`: `{xai_method, k, partition, train_mae_final, val_mae_final, test_mae, test_mape, training_time_minutes, epochs_run, seed, n_features_kept, dropped_features}`.
- `checkpoints/{cell}/training_log.csv`: per-epoch train/val loss.
- `checkpoints/{cell}/training_config.json`: full hyperparameters snapshot (so the run is reproducible from the file alone).

**Stop criterion per run:** training completes 150 epochs without divergence, final val MAPE is in a sensible range (rough sanity: not 100×worse than baseline; if it is, stop and investigate before continuing). Report after the baseline run, then again every 3–4 runs so the user can monitor progress.

---

## Step 8 — Build fidelity curves and compare

**Goal:** Per-(method, k) MAE/MAPE table, the `irrelevant_k − relevant_k` gap statistic, and the headline curves.

**Tasks:**
- Load all 13 `metrics.json` into a DataFrame (1 baseline + 6 principled + 6 random).
- Save: `results/fidelity_summary.csv` with columns `[group, k, partition, test_mae, test_mape, n_features_kept]` where `group ∈ {baseline, principled, random}`.
- Compute the **per-(group, k) MAE gap**: `gap(group, k) = MAE(irrelevant_k) − MAE(relevant_k)`. Larger = stronger fidelity.
- **Headline figure**: at each k ∈ {30, 50, 70}, plot for **principled** and **random**:
  - The retrained baseline MAE as horizontal anchor.
  - `relevant_k` MAE (one bar/point per group).
  - `irrelevant_k` MAE (one bar/point per group).
  - The gap is the visual difference between the two bars.
- **Secondary figure**: gap vs k for principled and random (two lines, three points each).
- Same plots in MAPE alongside MAE.

**Interpretation rules (locked):**
- For (principled, k): `relevant_k` MAE close to baseline = good (the top-k features carry the signal). `irrelevant_k` MAE close to baseline = the ranking picked the wrong features.
- **The headline comparison is principled vs. random** (not IG vs. KernelSHAP — they share identical models, §7.A). The ranking is faithful iff the **principled** gap clearly exceeds the **random** gap. If it does not, the protocol isn't informative on this dataset — itself a finding.
- Across k for the principled group: larger gap at smaller k (k=30, keep top-3) is a stronger statement than larger gap at k=70 (keep top-7), because at k=70 most features are kept either way.
- **Caveat (lower-bound gaps):** `traffic` and `packets` are structurally retained (load, `pkt_size`), so gaps that rely on removing them understate true importance; the test most cleanly validates the 8 truly-removable features + `sigma`. State this with the figure.

**Outputs:**
- `results/fidelity_summary.csv`
- `results/figures/fidelity_bars.pdf` (headline figure)
- `results/figures/fidelity_gap_curves.pdf` (secondary)
- `results/figures/fidelity_mape_versions.pdf` (MAPE versions of both)

**Stop criterion:** figures produced, summary committed. Report and wait before Step 9.

---

## Step 9 — Stability and cost analysis

**Goal:** Secondary metrics.

**Tasks:**
- **Stability:** Spearman between main ranking and stability ranking from Step 4, per method. Save to `results/stability.csv`.
- **Cost:**
  - XAI explanation cost from `explanations/*/timings.csv`. Mean and std per method.
  - Retraining cost from `checkpoints/*/metrics.json` `training_time_minutes`. Mean and std per method × k.
- Plot to `results/figures/stability_cost.pdf`.

---

## Step 10a — Plausibility analysis

**Goal:** Compare top-ranked features against networking domain knowledge.

**Tasks:**
- Open `results/preregistration.md` from Step 3.5.
- For each method, top-5 features from `rankings/{M}.csv`.
- For each: 1–2 paragraphs evaluating against queueing theory in the multiplexed regime (`all_multiplexed`, mixing CBR + on/off + autocorrelated + modulated). Consider:
  - Do ON/OFF and autocorrelation features appear (they should, unlike in CBR-only)?
  - Are bandwidth/arrival-rate parameters dominant as expected?
  - Surprises?
- **Compare to pre-registration:** match in N of M expected features. Honest reporting of agreements and disagreements.

**Output:** `results/plausibility.md`.

---

## Step 10b — V2X-transferability writeup

Per THESIS_DECISIONS §16. **Not a code task.** Write `results/v2x_transferability.md` covering the 5 points listed there.

---

## Step 11 — Write thesis / D3.2 Section 5

Skeleton:
- §5.1 Introduction
- §5.2 Model and dataset (RouteNet-Fermi, `all_multiplexed`, the 10 path scalars + structural 12 inputs)
- §5.3 XAI methods (IG, KernelSHAP, [random control])
- §5.4 Evaluation protocol (retraining-based, column dropping, 13 models, hyperparameters per paper §IV.D)
- §5.5 Results (fidelity gap, stability, cost, plausibility)
- §5.6 V2X transferability
- §5.7 Conclusion

---

## File layout (actual)

```
RouteNet-Fermi/   (xai-features branch)
├── CLAUDE.md, THESIS_DECISIONS.md, PIPELINE.md
├── upstream files (delay_model.py, data_generator.py, etc. — DO NOT MODIFY on this branch)
├── traffic_models/delay/<tm>/ckpt_dir_<tm>/   (used Step 2.5 + Step 4 only)
├── data/
├── results/
│   ├── baseline_validation/     ← Step 2 + 2.5
│   ├── preregistration.md        ← Step 3.5
│   ├── fidelity_summary.csv      ← Step 8
│   ├── stability.csv, figures/   ← Step 9
│   ├── plausibility.md, v2x_transferability.md   ← Steps 10a, 10b
├── xai/                          ← Step 3
├── explanation_set/, explanations/, rankings/   ← Steps 4–5

RouteNet-Fermi/   (xai-protocol-b branch — branched off xai-features before Step 6)
├── (same as above)
├── data_generator.py             ← MODIFIED: accepts dropped_features parameter
├── delay_model.py                ← MODIFIED: path_embedding input dim varies per variant
├── configs/                      ← Step 6: 12–18 variant configs
└── checkpoints/                  ← Step 7: 13 unique trained models
    ├── baseline_seed42/
    ├── principled/k{30,50,70}_{relevant,irrelevant}/   ← IG ≡ KernelSHAP (from configs/ig/)
    └── random/k{30,50,70}_{relevant,irrelevant}/       ← negative control (core)
```

---

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| KernelSHAP too slow on 300 sims | Low (validated by pilot scaling) | Pilot showed ~30 min for 40 flows × 256 perturbations; 300 × 1 target flow scales to ~3.5 h. If wall-clock balloons, drop to N=200 with documented justification. |
| One of the 13 retrainings diverges or fails to converge | Medium | Same hyperparameters as paper §IV.D should converge cleanly. If a variant fails, investigate (likely too few features kept at extreme k) before retrying. |
| `data_generator.py` modifications introduce bugs that silently corrupt the input | High if not checked | Unit test: load one variant config, dump the first batch, verify exactly the expected columns are present and have plausible values. |
| `path_embedding` input dim modification fails to retrain because of TF state | Medium | Recreate the model from scratch per variant (don't try to "patch" a pre-existing model). Always instantiate fresh. |
| GCP access/quota falls through | Medium | Dana provisions the company GCP resource; the trainer/driver are GPU-agnostic so any CUDA-capable VM works. Claude never runs remote compute — Dana executes by hand. |
| One of the 13 retrainings diverges | Medium | Same hyperparameters as paper §IV.D should converge; if a variant fails (likely too few features at k=30), investigate before retrying. |
| Implausible XAI rankings (top-5 don't match queueing-theory intuition) | Possible | Treat as a finding. Pre-registration converts this from embarrassment to publishable result. |
| Half-split Spearman in Step 5 is low | Low | If < 0.7, increase N before Step 6. Document the observed value. |
| Confusion between "10-of-22 XAI scope" and "what's dropped in variants" | High | §5 of THESIS_DECISIONS is explicit: XAI ranks 10 features. Dropping in variants is restricted to those 10. The 12 structural inputs are NEVER dropped. |

---

## Changelog

| Version | Change |
|---|---|
| v1 | 3 XAI methods, 19 retrainings, ~115 GPU-h. |
| v2 | Dropped Saliency. 13 retrainings. |
| v3 | k ∈ {50, 70, 90}. 9 retrainings. |
| v4 | `all_multiplexed` locked. V2X writeup added. Time removed. |
| v5 | Switched to inference-only. 12 inference cells. |
| v5.5 | Both protocols (A inference / B retraining) documented as conditional branches with hard-stop gate. Random control as third method. k ∈ {25, 50, 75}. Pre-registration formalized. Step 2.5 added. |
| v6 | **Protocol locked to retraining-based per the IMT technical advisor's email (June 2026).** Variant generation locked to **column dropping** (per the IMT technical advisor's confirmation). Inference-only branch and hard-stop gate removed. Step 6 rewritten as the dropping-based variant generator on a sister branch `xai-protocol-b`. Step 7 rewritten with the 13-cell core matrix + conditional 6-cell §7.B random control. Step 8 rewritten around the `irrelevant_k − relevant_k` MAE gap as the headline statistic. Risks table updated to reflect dropping-specific risks. |
| v7 | **Post-Step-5 redesign (2026-06-16/17).** (1) Threshold sweep {25,50,75} → {30,50,70} — k=30 isolates the top-3 above the ~19–20× attribution cliff; exact mapping k×10 = 3/5/7. (2) **Step 7 matrix restructured to 13 unique = 1 baseline + 6 principled + 6 random.** IG ≡ KernelSHAP at every cell (identical kept-sets; audit equivalence check), so only the 6 IG-derived "principled" models are trained (`checkpoints/principled/`); KernelSHAP duplicates skipped. (3) **Random control promoted from conditional to core** — it carries the fidelity headline (principled-vs-random). (4) Step 8 reframed to principled-vs-random; (5) traffic/packets lower-bound caveat added; (6) compute → **GCP** (risk table updated). Configs regenerated (incl. `configs/random/`); re-verified audit + smoke **19/19**. See THESIS_DECISIONS §6/§7/§10 and changelog v7. |
