# THESIS_DECISIONS.md

**Project:** Critical and Comparative Study of XAI Approaches for GNN-based QoS Prediction in SDN Network Automation
**Author:** Dana Dagher (M2 thesis)
**Context:** TRAVEL ANR project — WP3, Task 3.2, IMT Contribution 3. Feeds deliverable D3.2 Section 5.
**Reference paper:** Ferriol-Galmés et al., *RouteNet-Fermi: Network Modeling with Graph Neural Networks*, IEEE/ACM ToN 2023, DOI 10.1109/TNET.2023.3269983.
**Status:** Locked.
**Version:** v7

---

## 0. Thesis question

> *Which XAI techniques produce the most faithful and plausible feature-importance explanations for a GNN that predicts QoS metrics in an SDN-managed network, and what is the impact on model accuracy of retraining on inputs filtered by feature importance?*

The work is positioned as a **methodological proof-of-concept for XAI applied to GNN-based network automation**. RouteNet-Fermi serves as a controlled SDN testbed. V2X-specific extensions are out of scope; V2X transferability is discussed qualitatively in §16.

---

## 1. Model and codebase

**Decision:** RouteNet-Fermi (BNN-UPC, github.com/BNN-UPC/RouteNet-Fermi). The internal GNN architecture (message-passing topology, hidden state size 32, T=8 iterations, RNN/MLP components) is unchanged across all 13 trained models. Only the `path_embedding` layer's input dimension changes per variant, reflecting the reduced feature count.

**Rationale:** The reference GNN for per-flow QoS prediction in transport networks, validated in the IEEE/ACM ToN 2023 paper. Keeping architecture identical isolates the effect of input-feature filtering from any model-engineering effect.

---

## 2. Sub-experiment

**Decision:** `traffic_models/` (BNN-UPC dataset-v6-traffic-models).

**Rationale:** Richest per-flow feature set in the BNN-UPC suite — essential for XAI.

---

## 3. Target metric

**Decision:** Delay (per-flow average packet delay).

**Rationale:** Lowest noise floor in the suite, reliable baseline convergence, direct interpretability in queueing-theory terms. The RouteNet-Fermi paper's primary reported metric (paper Table V).

---

## 4. Sub-configuration

**Decision:** **`all_multiplexed`** — one of five sub-datasets under `traffic_models/delay/`.

**Rationale:** Mixes CBR, on/off, autocorrelated, and modulated traffic generators per scenario. Heterogeneous flow types → per-flow input features carry real variance → XAI attributions show real contrast.

**Single-generator variants rejected** because several path features are structurally zero in those settings, making any XAI ranking trivial.

**Step 2.5 exception:** the validity check runs on **all 5 sub-datasets** to reproduce the paper's MAPE table (Table V). Beyond Step 2.5, only `all_multiplexed` continues.

**Quotable rationale:** *"We evaluate on the `all_multiplexed` sub-dataset of `traffic_models/delay`, the most heterogeneous and most realistic scenario in the BNN-UPC benchmark. The mix of CBR, on/off, autocorrelated, and modulated traffic generators produces the per-sample feature variance required for meaningful XAI attribution analysis."*

---

## 5. Features in scope for XAI ranking

**Decision:** The **10 per-flow path scalars**. The 12 other named RouteNet-Fermi inputs are present unchanged in every dataset variant.

**Exclusion table (locked):**

| Input group | Count | Why excluded from XAI ranking |
|---|---|---|
| `length` | 1 | Graph structure — changing it changes the route/topology |
| `model` (categorical) | 1 | One-hot encoded into 7 bits — not a continuous scalar to perturb |
| Link features | 2 | Defined per network link, shared across flows — not per-flow traffic |
| Queue features | 3 | Network infrastructure config — not traffic source params |
| Graph tensors | 5 | Connectivity, not features |
| **Path scalars (XAI target)** | **10** | **Continuous, per-flow, directly perturbable traffic source parameters** |

**Critical clarification (this protocol):**

The 12 "excluded" inputs are excluded only from the **XAI ranking** step. In the **dataset partitioning** step (Step 6), where columns are dropped to create the `relevant_k` and `irrelevant_k` variants, **only the 10 path scalars are candidates for dropping**. The 12 structural inputs are **never dropped from any variant** — they are essential for the GNN to function (without them there is no graph for the model to operate over). Every retrained model under Protocol B sees the same 12 structural inputs unchanged; only its subset of path scalars differs.

**Quotable rationale:** *"We restrict feature attribution and feature filtering to the 10 per-flow path scalars — the continuous, per-flow, directly perturbable traffic source parameters of RouteNet-Fermi. The remaining 12 named inputs define the graph structure the GNN operates over (topology, connectivity, link and queue properties, traffic-model categorical) and are present unchanged in every experimental variant. Graph-structural and queue-level explanations are out of scope and deferred to future work."*

---

## 6. Variant generation: column dropping (NOT value masking)

**Decision:** Confirmed by Karim (IMT) — variants are generated by **dropping feature columns** entirely from the input pipeline, not by replacing values with mean/median.

**For each XAI method M and each threshold k:**
- **`relevant_k`** variant: drop the bottom-(100−k)% feature columns from the input dict. The remaining top-k% features pass through to the model. The `path_embedding` input dimension is reduced to match.
- **`irrelevant_k`** variant: drop the top-k% feature columns. The remaining bottom-(100−k)% features pass through. The `path_embedding` input dimension is reduced to match.

**The 12 structural inputs are not touched in either variant.**

**Why dropping (not masking):**
- Eliminates the out-of-distribution confound. A masked value (e.g., replacing on/off duration with the dataset median for a CBR flow) creates input combinations the model never saw during training, and any performance change confounds "the feature mattered" with "the input was OOD."
- A model retrained on dropped inputs sees only in-distribution feature combinations.
- Directly answers the relevant scientific question: *"are these features necessary to learn the task?"* — not the weaker question *"does the trained model rely on these features at inference?"*
- This is the ROAR-style methodology (Hooker et al. 2019) for evaluating feature importance.

**Procedure:**
1. Run XAI methods on N=300 simulations from `all_multiplexed` test (Step 4).
2. Aggregate per-feature importance, rank the 10 path scalars globally (Step 5).
3. For each k ∈ {30, 50, 70}, build two reduced-input dataset configurations (Step 6).
4. Retrain RouteNet-Fermi from scratch on each (Step 7).

---

## 7. Experimental protocol — RETRAINING-BASED (LOCKED)

### §7.A — Core matrix (13 unique models, all mandatory)

| # | Cell | Trainings |
|---|---|---|
| 1 | Baseline — full 10 path scalars, retrained from scratch with seed 42 | 1 |
| 2–7 | **Principled** (IG ≡ KernelSHAP) · k=30/50/70 · {relevant, irrelevant} | 6 |
| 8–13 | **Random** negative control · k=30/50/70 · {relevant, irrelevant} | 6 |
| | **Total** | **13** |

**Why 6 principled cells, not 12 (IG ≡ KernelSHAP):** at k ∈ {30, 50, 70} the IG and KernelSHAP rankings produce identical kept-feature **sets** at every cell — the two rankings differ only by adjacent swaps that never cross a cut boundary (verified by `audit_steps_1_to_6.py`: "IG set == KernelSHAP set" at all 6 cells). Since the model re-sorts features to canonical order, identical sets → identical trained models (same seed, data, set). Training the KernelSHAP configs would reproduce the IG models bit-for-bit, so the 6 IG-derived cells **are** the principled variants for both methods. This equivalence is a *finding* (convergent validity), not an assumption. The 6 trained models live in `checkpoints/principled/` (trained from `configs/ig/`).

**Why the random control is now core (not conditional):** because IG ≡ KernelSHAP, the IG-vs-KernelSHAP *fidelity* comparison is degenerate (identical models → identical curves). The meaningful faithfulness test is **principled-vs-random**: the ranking is faithful iff `relevant_k` stays near baseline, `irrelevant_k` degrades, AND both beat the random control. Without a random baseline a fidelity claim is unfalsifiable (standard ROAR negative control, Hooker et al. 2019). The random control therefore **carries the fidelity result and is mandatory**.

### §7.B — How IG and KernelSHAP are compared

Since the two methods produce identical retrained models (§7.A), they are **not** compared on fidelity. The comparison lives on two axes that *do* differ, both available from the explanation phase (Step 4–5, no extra retraining):
- **Stability** — half-split Spearman: IG **0.88** vs KernelSHAP **0.95** (KernelSHAP is more self-consistent across data subsets).
- **Cost** — IG ~50 gradient steps vs KernelSHAP 256 perturbations per flow (IG ~5× cheaper).

**Conclusion:** the methods are equivalent in what they select and equally faithful; we recommend **Integrated Gradients** (same ranking, sufficiently stable, far cheaper). The strong cross-method agreement (Spearman 0.96, identical subsets at every threshold) is *convergent-validity* evidence that the ranking reflects the model, not the method — a recognized and valuable result, since disagreement between XAI methods is a known field-wide concern.

**Critical constraints (all 13 models):**
- Identical hyperparameters per paper §IV.D: 150 epochs × 2,000 samples, Adam optimizer at lr=0.001, MAPE loss for delay, hidden state size 32, T=8 message-passing iterations.
- Identical seed: 42 for all training runs.
- Identical training/validation/test split.
- Only the `path_embedding` input dimension differs per variant.

**Why retraining baseline (separate from the upstream pretrained checkpoint):**

The upstream pretrained checkpoint was trained on BNN-UPC's machine with BNN-UPC's seed and recipe. For the 13-model comparison to be **fair**, the baseline must be retrained from scratch on Dana's machine, with the same seed and hyperparameters as the 12 variants. The upstream checkpoint is used **only** for (a) Step 2.5 validity check and (b) Step 4 XAI explanation generation. It is not used as the fidelity comparison baseline.

### §7.C — Why this protocol (defense-ready justification)

The retraining-based protocol was selected after explicit comparison with an inference-only alternative (where the upstream pretrained model is run on masked inputs and predictions are compared to the unmasked-input reference). The inference-only protocol is cheaper (no retraining) but answers a weaker question: it conflates "the model relies on this feature at inference" with "the masked input is out-of-distribution." The retraining protocol eliminates this confound by ensuring every model sees only in-distribution input combinations during training and at evaluation. This was the protocol explicitly requested by the IMT supervisor (Karim Gizzini, Sogeti / IMT contribution) for the deliverable.

**Quotable rationale:** *"We adopt a retraining-based feature-filtering protocol. For each XAI method and each threshold k ∈ {30, 50, 70}, the 10 path-scalar inputs are partitioned into two reduced-input variants (top-k% kept, bottom-(100−k)% kept), the `path_embedding` layer is resized accordingly, and the model is retrained from scratch with identical hyperparameters and seed. This eliminates the out-of-distribution confound that affects inference-only feature-importance evaluation (Hooker et al. 2019)."*

---

## 8. XAI methods compared

**Decision:** Two principled XAI methods + one optional negative control:

| # | Method | Family | Role |
|---|---|---|---|
| 1 | **Integrated Gradients** | Gradient-based | Principled gradient method (Sundararajan 2017) |
| 2 | **KernelSHAP** | Perturbation-based | Principled perturbation method (Lundberg & Lee 2017) |
| 3 | **Random-ranking control** | Negative control (optional) | Uniformly random ordering of the 10 features. Same retraining matrix. If IG and KernelSHAP don't outperform random across the comparison, the protocol isn't informative on this dataset — itself a finding. |

**KernelSHAP wrapper specification (consistent with CBR pilot):**
- Fix the full graph (links, queues, other flows, topology). Vary only the target flow's 10 path scalars.
- One target flow per simulation: **flow_idx=0**.
- 256 perturbations per flow.
- Background reference: training-set median per feature (single-reference, same as pilot).

**§8.A Operational justifications (defense-ready):**

- **Flow selection (flow_idx=0):** preserves methodological consistency with the CBR pilot and avoids introducing a flow-selection heuristic that itself needs justification. On `all_multiplexed` the traffic-model class at index 0 varies across simulations; the empirical distribution is reported in the results for transparency.
- **Single-reference background (median):** consistent with the CBR pilot and with practical compute. Rank-based aggregation reduces sensitivity to reference-set size compared to absolute-magnitude analysis.

**Rejected for the main experiments:** Vanilla Gradients (deferred), GNNExplainer / PGExplainer (TF port too costly, deferred), Graph-LIME (not significantly different from KernelSHAP for this case).

---

## 9. Sample budget

- **Training set (used in Step 2 baseline and Step 7 retraining):** full `all_multiplexed` training split.
- **Validation set:** full `all_multiplexed` validation split.
- **XAI explanation set (Step 4):** **N = 300 simulations** from `all_multiplexed` test, sampled with seed=42. Same 300 across both XAI methods.
- **Evaluation set for retrained models (Step 7 / 8):** full `all_multiplexed` test split (for direct comparability with the paper).

**On what "300 samples" means:** 300 network simulations, each ~100–500 flows. Tens of thousands of flow-level delay predictions per inference pass. 300 is a budget on KernelSHAP wall-clock (~3.5 h CPU, scaled from the pilot), not a limitation of statistical mass.

**Validity check (Step 2.5):** before any XAI work, MAPE of the upstream pretrained checkpoint on N=300 across all 5 sub-datasets is compared against the paper Table V. Confirms environment health and that N=300 is representative.

**Sample-stability sanity check (Step 5):** ranking on simulations 0–149 vs. 150–299, Spearman reported. Document the observed value (no magic-number gate).

---

## 10. Evaluation metrics

**Primary — Fidelity (the three-way comparison at every k, on retrained models):**

At every k ∈ {30, 50, 70}, the figure shows three quantities on the same axes:

- **Baseline reference** — test MAE/MAPE of the **retrained full-feature baseline** (seed 42) on the full `all_multiplexed` test split. Anchor, horizontal line.
- **`relevant_k` line** — test MAE/MAPE of the model retrained on the top-k% features only. For the **principled** ranking (IG ≡ KernelSHAP) and for the **random** control.
- **`irrelevant_k` line** — test MAE/MAPE of the model retrained on the bottom-(100−k)% features only.

**Comparison rules (locked interpretation):**

- **`irrelevant_k` MAE should be close to baseline.** The "irrelevant" features alone should suffice — meaning the XAI method correctly identified them as low-importance (the model can do without the top features). High MAE here = the XAI method missed important features.
- **`relevant_k` MAE should also be close to baseline, and ideally lower than `irrelevant_k` MAE.** Keeping only the top-k% features should preserve performance if those features really are the important ones.
- **`relevant_k` − `irrelevant_k` MAE gap** is the per-(method, k) cell statistic. Larger negative gap (relevant < irrelevant) = stronger XAI fidelity.
- **Principled vs. random (the headline):** the principled ranking's `irrelevant − relevant` gap should clearly exceed the random control's gap across the three k values. IG and KernelSHAP are **not** separated here — they share identical models (§7.A) — so the fidelity comparison is principled-vs-random, not IG-vs-KernelSHAP.
- **Random control (core):** should show no consistent gap. The faithfulness claim rests on the principled gap clearly exceeding random's gap; if it does not, the protocol isn't informative on this dataset (itself a reportable finding).

**Caveat — `traffic` and `packets` are structurally retained (lower-bound gaps):** these two are never removed from the data dict (they feed the load and `pkt_size` computations; `_DROPPABLE_PATH_SCALARS` has 8 entries, not 10 — see §5). When a variant "drops" them they leave only the *learned path embedding*, not the physics. So any gap that depends on removing `traffic`/`packets` is a **lower bound**, and the fidelity test most cleanly validates the ranking of the **8 truly-removable features** plus `sigma`. This is acceptable because `traffic`/`packets` are independently established as important by queueing theory (they are in the delay formula); XAI's discriminating value is on the *learned* features, which are fully testable.

**MAPE reported alongside MAE** for direct comparability with the RouteNet-Fermi paper (which reports MAPE).

**Secondary — Stability:**

Both methods run twice with different seeds at the XAI step:
- IG: training-set median baseline vs. a meaningfully different alternative (training-set mean, or uniform-random baseline).
- KernelSHAP: two different background subsamples of training data.

Spearman rank correlation between the two rankings, per method.

**Secondary — Computational cost:** wall-clock time per explanation (Step 4) and per retraining (Step 7), mean and std reported per method.

**Tertiary — Plausibility:** top-5 features per method (from Step 5 rankings), 1–2 paragraphs each on queueing-theory consistency in the multiplexed regime. **Pre-registered** before Step 7 — see §14.

**Weights for discussion:** Fidelity 60%, Stability 25%, Cost 15%. Plausibility = qualitative sanity check.

---

## 11. Compute requirements

- **Step 1–2:** environment + smoke test. CPU sufficient.
- **Step 2.5:** 5 inference passes (one per sub-dataset) on N=300 with the upstream checkpoints. CPU sufficient.
- **Step 4:** 300 KernelSHAP explanations + 300 IG explanations + 2 stability runs. CPU; KernelSHAP is the bottleneck (~3.5 h).
- **Step 7:** 13 unique retrainings of RouteNet-Fermi (1 baseline + 6 principled + 6 random). Per paper §IV.D: 150 epochs × 2,000 samples per run. **GPU required.** Compute: **GCP** (company-provided).

**Action items:**
1. Provision the GCP VM (Dana, via company resource). Claude never runs anything on it — Dana executes Step 7 by hand.
2. Once provisioned, estimate per-training wall-clock on the GCP GPU to plan the 13 runs.

---

## 12. Out of scope (deferred to future work)

- Graph-structural XAI (GNNExplainer, edge masking)
- V2X-specific datasets (Farreras et al.) — methodology transferability discussed qualitatively (§16)
- Real-world traffic traces (`real_traffic`)
- Scalability experiments
- Local (per-sample) feature ranking
- Causal interpretability
- Vanilla Gradients (Saliency)
- The other four sub-datasets under `traffic_models/delay/` (beyond Step 2.5)
- Jitter and loss prediction tasks
- **Value-based masking variants.** Per Karim's confirmation, all variant generation is column dropping. Masking was considered as an alternative protocol and explicitly rejected in favor of dropping for the reasons in §6 and §7.C.
- **Inference-only XAI evaluation on the frozen upstream checkpoint as the fidelity benchmark.** Inference on the upstream checkpoint is used only for Step 2.5 validity and Step 4 explanation generation. The fidelity comparison itself is on retrained models.

---

## 13. Glossary

| Term | Definition |
|---|---|
| **Upstream pretrained checkpoint** | The model shipped by BNN-UPC in `ckpt_dir_<tm>/`. Used only for Step 2.5 (validity) and Step 4 (XAI explanation generation). Not the fidelity baseline. |
| **Retrained baseline** | RouteNet-Fermi trained from scratch on the full 10-path-scalar `all_multiplexed` training set with seed 42 in Step 7 cell #1. This is the fidelity comparison reference for the 12 variants. |
| **`relevant_k`** | Variant where the top-k% features (by XAI ranking) are kept and the rest are dropped. Model retrained on this reduced input. |
| **`irrelevant_k`** | Variant where the bottom-(100−k)% features (by XAI ranking) are kept and the top-k% dropped. Model retrained on this reduced input. |
| **`irrelevant_k − relevant_k` MAE gap** | The primary per-cell fidelity statistic. Larger negative gap = stronger XAI. |
| **Random-ranking control** | Optional third "method" using a uniformly random ranking of the 10 features. Same 6-cell retraining matrix. Fidelity floor. |
| **Half-split Spearman (Step 5)** | Sanity check on N=300 sufficiency. Spearman between rankings computed on sims 0–149 vs. 150–299. |
| **10 path scalars** | The XAI target feature set. See §5. |
| **12 structural inputs** | `length`, `model`, link features, queue features, graph tensors. Present unchanged in every variant. |

---

## 14. Pre-registration (Step 3.5)

**Decision:** Before running XAI on `all_multiplexed` (Step 4 onward), the user writes `results/preregistration.md` containing:

- Expected top-3 features by aggregate importance, based on queueing-theory intuition for the multiplexed regime.
- Expected qualitative shape of the curves (`relevant_k` MAE vs k; `irrelevant_k` MAE vs k).
- Predictions for how IG and KernelSHAP may differ (or not).

Sealed and dated via a git commit **before any Step 4 results are observed**. Compared against observed rankings in Step 10a.

**Rationale:** converts plausibility analysis from post-hoc rationalization to falsifiable comparison.

---

## 15. Changelog

| Version | Change |
|---|---|
| v1 | 3 XAI methods (Saliency, IG, KernelSHAP), 19 retrainings. |
| v2 | Dropped Saliency. 13 retrainings. |
| v3 | Percentile sweep reduced to {50, 70, 90}. 9 retrainings. |
| v4 | `all_multiplexed` locked. V2X transferability section added. Time estimates removed. |
| v5 | Switched to inference-only protocol. Stretch retrain in §17. |
| v5.5 | Both protocols (A inference-only / B retraining) documented as conditional branches. Hard-stop gate added. Random control promoted to third method. k changed to {25, 50, 75}. Pre-registration formalized. Step 2.5 validity check on all 5 sub-datasets added. RouteNet-Fermi paper named as the validation reference. |
| v6 | **Protocol locked to retraining-based per Karim's email (June 2026).** Variant generation confirmed as **column dropping**, not value masking. Inference-only branch and hard-stop gate removed. §7 rewritten with the locked matrix (13 core + 6 conditional random control). §10 evaluation rewritten around the retrained-baseline reference and the `irrelevant_k − relevant_k` MAE gap. §13 glossary updated. §12 out-of-scope updated to explicitly list value-masking and inference-only as rejected. Random control made conditional rather than mandatory, to keep core scope at 13 retrainings. CLAUDE.md and PIPELINE.md aligned. |
| v7 | **Post-Step-5 redesign (2026-06-16/17, Dana — decisions taken and justified independently).** Six linked decisions: **(1) Threshold sweep {25,50,75} → {30,50,70}** — Step 5 shows both methods placing the same 3 features ({traffic, sigma, packets}, ~93% of attribution mass) above a ~19–20× cliff; k=30 isolates that top-3, k=50 intermediate control, k=70 adds the noise tail; exact mapping k×10 = 3/5/7; thresholds derived from attribution magnitudes (not from the fidelity they evaluate → not circular). **(2) Train 7 unique principled models, not 12** — at every cell IG and KernelSHAP yield identical kept-feature SETS (rankings differ only by adjacent swaps that never cross a cut; enforced by an `audit_steps_1_to_6.py` equivalence check), so the 6 IG-derived models cover both methods; KernelSHAP duplicates would be bit-identical. **(3) Random control promoted to core** — since IG ≡ KernelSHAP the fidelity comparison becomes principled-vs-random; the 6 random retrainings now carry the faithfulness result. New matrix = 1 baseline + 6 principled + 6 random = **13 unique**. **(4) Framing → convergent-validity + trade-off** — IG and KernelSHAP compared on stability (0.88 vs 0.95) and cost (~5×), not fidelity; recommend IG. **(5) traffic/packets structural caveat** stated in §10 (their gaps are lower bounds). **(6) Compute → GCP** (replaces Sogeti RTX 4090). Propagated across configs, `run_step7_all.sh` (checkpoints/principled + checkpoints/random), audit, smoke test, CLAUDE.md, PIPELINE.md, step reports, and the D3.2 deliverable. Re-verified: audit + smoke 19/19 PASS. |

---

## 16. V2X transferability — dedicated deliverable section

**Decision:** D3.2 Section 5 and the thesis include a dedicated written section on V2X-dataset limitations and methodology transferability.

**Rationale:** required by the IMT scope-change agreement.

**Content the section must cover:**

1. **What `all_multiplexed` is not.** Static topologies; no mobility; no V2X-specific KPIs; traffic generators as statistical abstractions, not vehicular flows.

2. **What would need to change for V2X.** Vehicular-mobility datasets (Farreras et al. 2024, BNNetSimulator with mobility traces); mobility-state per-flow features; topology features as first-class XAI targets; safety-related KPIs replacing average delay.

3. **What transfers cleanly.** The retraining-based feature-filtering protocol. The IG-vs-KernelSHAP-vs-random comparison framework. The `irrelevant_k − relevant_k` gap as the fidelity statistic. The fidelity/stability/cost/plausibility stack.

4. **What does not transfer cleanly.** Per-flow-only attribution scope (V2X requires graph-structural XAI). Global rankings assume stationary feature distributions, which mobility breaks. Retraining cost scales unfavorably on large vehicular datasets — for V2X, an inference-only variant of the protocol may need to be considered as a pragmatic compromise.

5. **Concrete handoff to WP4 / Task 3.3.** The reduced-input training pipeline, the IG and KernelSHAP implementations, the curve-comparison code, and the ranking pipeline — all reusable artifacts for the V2X case once a compatible dataset exists.

**Quotable framing:** *"The methodology presented here is dataset-agnostic by design. The `all_multiplexed` benchmark serves as a controlled SDN testbed where XAI behaviour can be isolated from the additional complexity of vehicular mobility. The V2X extension is deferred to Task 3.3 / WP4, where the artifacts produced in this work — the reduced-input training pipeline, the IG and KernelSHAP implementations, and the fidelity-gap evaluation harness — provide a direct starting point."*
