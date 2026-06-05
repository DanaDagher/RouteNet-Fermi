# CLAUDE.md

**Version:** v6

This file orients any Claude Code session that opens this project. Read me first.

## Project

**M2 thesis:** Critical and Comparative Study of XAI Approaches for GNN-based QoS Prediction in SDN Network Automation.

**Author:** Dana Dagher.

**Context:** TRAVEL ANR project, WP3 Task 3.2, IMT Contribution 3. Feeds deliverable **D3.2 Section 5**.

**Reference paper (the baseline for validation):** Ferriol-Galmés et al., *"RouteNet-Fermi: Network Modeling with Graph Neural Networks"*, IEEE/ACM Transactions on Networking 2023, DOI 10.1109/TNET.2023.3269983. All baseline numbers come from this paper.

## Two-document contract

1. **`THESIS_DECISIONS.md` (v6)** — what we're doing and why.
2. **`PIPELINE.md` (v6)** — the step-by-step execution checklist.

Read both before doing anything. Do not invent decisions that contradict them. If asked to do something inconsistent, flag the inconsistency.

## Protocol (LOCKED)

**Retraining-based comparison (formerly "Protocol B" in v5.5).** Confirmed by Karim (IMT) on the email exchange of June 2026, and confirmed as **column-dropping** (not value-masking).

For each XAI method (IG, KernelSHAP) and each threshold k ∈ {25, 50, 75}, the input feature set is partitioned into two reduced-input variants:
- **`relevant_k`** — keep the top-k% features (drop the bottom-(100−k)% columns).
- **`irrelevant_k`** — keep the bottom-(100−k)% features (drop the top-k% columns).

A new model is trained from scratch on each variant. **13 trained models total:**
- 1 baseline (full 10 path scalars, retrained from scratch with seed 42).
- 6 IG variants (3 k × 2 partitions).
- 6 KernelSHAP variants (3 k × 2 partitions).

A random-ranking negative control runs the same 6-cell matrix as a third "method." **The random control's 6 retrainings are conditional on compute availability** — execute them if and only if the 13 main retrainings completed cleanly and time / GPU access remains. See PIPELINE Step 7 and THESIS_DECISIONS §7.B.

## Quick facts

- **Model:** RouteNet-Fermi (BNN-UPC), architecture unchanged. Only the `path_embedding` layer's input dimension is reduced under each (k, partition) variant to match the reduced feature count. All other layers (message passing, hidden state size 32, T=8 iterations) are identical to the upstream code and to the paper §IV.D.
- **Dataset:** BNN-UPC `dataset-v6-traffic-models`. Sub-experiment: `traffic_models/`. Target: **delay**. Sub-configuration: **`all_multiplexed`**.
- **Validity-check exception:** Step 2.5 runs the upstream pretrained checkpoint on all 5 sub-datasets (constant_bitrate, onoff, autocorrelated, modulated, all_multiplexed) to reproduce the RouteNet-Fermi paper Table V. Beyond Step 2.5, only `all_multiplexed` continues.
- **XAI methods compared:** Integrated Gradients + KernelSHAP. Random-ranking control as conditional third method.
- **Features in scope for XAI ranking:** the 10 per-flow path scalars. The other 12 of 22 inputs are **always present** in every dataset variant — they define the graph structure the GNN operates over and are never removed. See THESIS_DECISIONS §5.
- **Variant generation:** **column dropping**, not value masking. The `path_embedding` input dimension changes per variant. The 12 structural inputs pass through unchanged in every variant.
- **Threshold sweep:** k ∈ {25, 50, 75}. Reported as feature counts: top-2 / top-5 / top-7 of 10 (since k×10 with floor/round gives those).
- **Sample budget:**
  - **For XAI explanations:** N = 300 simulations from `all_multiplexed` test (same 300 for IG and KernelSHAP).
  - **For training:** full `all_multiplexed` training split, all 13 (or 19 if random control runs) models.
  - **For evaluation of retrained models:** full `all_multiplexed` test split.
- **Hyperparameters per retraining (paper §IV.D):** 150 epochs of 2,000 samples, Adam optimizer at lr=0.001, MAPE loss for delay, hidden state size 32, T=8 message-passing iterations. **Same for all 13 models.** Seed 42 for all.
- **Reference compute:** Sogeti machine with NVIDIA RTX 4090. Pending Mouna's confirmation of SSH access.

## Repository layout (actual)

The working directory IS the cloned repo: `C:\Users\ddagher\RouteNet-Fermi\` on branch `xai-features`. Protocol B variants live on a sister branch `xai-protocol-b` created off `xai-features` before Step 6.

```
RouteNet-Fermi/
├── CLAUDE.md  THESIS_DECISIONS.md  PIPELINE.md     ← contract
├── README.md, LICENSE, requirements.txt, etc.       ← upstream, do not touch
├── delay_model.py                                   ← upstream model. Modified ONLY on branch xai-protocol-b
│                                                       (path_embedding input dim varies per variant).
│                                                       NEVER modify on main/xai-features.
├── data_generator.py                                ← upstream loader. Modified ONLY on xai-protocol-b
│                                                       (drop specified feature columns at load).
├── jitter_model.py, loss_model.py                   ← upstream (unused)
├── all_mixed/, fat_tree/, real_traffic/,
│   scalability/, scheduling/, testbed/              ← upstream, unused
├── traffic_models/
│   └── delay/{constant_bitrate, onoff, autocorrelated,
│              modulated, all_multiplexed}/
│       └── ckpt_dir_<tm>/                           ← upstream pretrained checkpoints
│                                                       (used in Step 2.5 only)
├── data/
│   └── traffic_models/<tm>/{train,validation,test}/
├── venv/                                            ← gitignored
├── .claude/                                         ← Claude Code workspace
├── results/                                         ← TO CREATE
│   ├── baseline_validation/                         ← Step 2.5 outputs
│   ├── preregistration.md                           ← Step 3.5
│   ├── inference/                                   ← XAI runs (Step 4) on pretrained checkpoint
│   ├── retrained/                                   ← Per-variant retraining metrics (Step 7)
│   ├── fidelity_summary.csv, figures/               ← Step 8
│   └── plausibility.md, v2x_transferability.md      ← Steps 10a, 10b
├── xai/                                             ← TO CREATE in Step 3
├── explanation_set/, explanations/, rankings/       ← Steps 4–5
├── datasets/                                        ← Step 6: reduced-input variant configs
├── checkpoints/                                     ← Step 7: 13 (or 19) trained models
│   ├── baseline_seed42/
│   ├── ig/k{25,50,75}_{relevant,irrelevant}/
│   ├── kernel_shap/k{25,50,75}_{relevant,irrelevant}/
│   └── random/k{25,50,75}_{relevant,irrelevant}/    ← conditional
└── (other dirs created on demand per pipeline)
```

## Working principles

- Match style to current code. Don't impose conventions the existing files don't follow.
- **Never modify `delay_model.py` or `data_generator.py` on `xai-features` or `main`.** Those modifications live only on `xai-protocol-b`, where they are mandatory for the dropping-based variants.
- Random seed: **42** everywhere (TF, NumPy, Python). Same seed for all 13 retrainings.
- If a step is ambiguous, ask before guessing.
- After every step, **stop and report**. Don't chain ahead.
- Reference for any "is this correct?" question about the model: the RouteNet-Fermi paper, not slide decks or pilot scripts.
- The 10-vs-22 feature scope explanation is in THESIS_DECISIONS §5 — internalize it so you don't confuse yourself or the user about what gets dropped.

## Out of scope

- V2X-specific datasets (Farreras et al.) — deferred. Methodology transferability is a written section, see PIPELINE Step 10b.
- GNNExplainer / PGExplainer / Graph-LIME — deferred.
- Vanilla Gradients / Saliency — deferred.
- The `scheduling/`, `scalability/`, `fat_tree/`, `all_mixed/`, `real_traffic/`, `testbed/` directories.
- The other four sub-datasets under `traffic_models/delay/` (only `all_multiplexed/` continues past Step 2.5).
- Jitter and loss models.
- Value-based masking. **Variant generation is column dropping.** Per Karim's confirmation.
- Inference-only XAI evaluation on the frozen upstream checkpoint as the *core* protocol. Inference on the frozen checkpoint is used **only** for (a) Step 2.5 validity check and (b) Step 4 XAI explanations on N=300. The actual fidelity evaluation (Step 7+) is on retrained models.

## On chat continuity

Strategic discussions live on the claude.ai Project (web/desktop), not in Claude Code sessions. If asked about a decision that isn't in `THESIS_DECISIONS.md`, say so — don't fabricate.
