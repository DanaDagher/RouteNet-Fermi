# XAI Protocol for RouteNet-Fermi — Overview

This branch (`xai-protocol-b`) contains the code, configs, and results for an M2 thesis
project comparing two XAI methods — **Integrated Gradients (IG)** and **KernelSHAP** — applied
to RouteNet-Fermi, a GNN for network delay prediction. It feeds Deliverable D3.2 Section 5 of
the TRAVEL ANR project (WP3, Task 3.2).

The base model and architecture are unchanged from the upstream
[BNN-UPC/RouteNet-Fermi](https://github.com/BNN-UPC/RouteNet-Fermi) repo
(Ferriol-Galmés et al., IEEE/ACM ToN 2023). This branch adds the XAI evaluation pipeline on
top of it. Dirs/files unrelated to that pipeline (other upstream experiments, scratch notes)
have been removed for clarity — see `THESIS_DECISIONS.md` for what's in/out of scope and why.

## Read these first

1. **`THESIS_DECISIONS.md`** — what was decided and why (protocol, dataset, methods).
2. **`PIPELINE.md`** — the step-by-step execution checklist (Steps 1–7).
3. **`STEP7_RUNBOOK.md`** — practical instructions for running the retraining step.

## Method in one paragraph

For each XAI method and each keep-threshold k ∈ {30, 50, 70}, the 10 per-flow path-scalar
input features are split into a `relevant_k` (top-k%) and `irrelevant_k` (bottom-(100-k)%)
variant by **column dropping** (not value masking). A model is retrained from scratch on each
variant, plus a random-ranking negative control. Fidelity is measured by how much worse the
`irrelevant_k` model does vs. `relevant_k`. IG and KernelSHAP are compared on ranking stability
and computational cost, not on fidelity (their rankings converge to the same kept-feature sets,
so they share the same retrained models — see `THESIS_DECISIONS.md` §7 for why).

## Directory map

| Path | What it is |
|---|---|
| `delay_model.py` | RouteNet-Fermi model, modified to accept a variable input width per variant. |
| `traffic_models/delay/` | Upstream pretrained checkpoints for all 5 sub-datasets (used once, for the Step 2.5 validity check against the paper's Table V). Only `all_multiplexed` continues past that step. |
| `data/` | Training/validation/test splits actually used (gitignored — large binary data, regenerate from BNN-UPC's dataset release). |
| `xai/` | XAI method implementations: IG, KernelSHAP, random control. See `xai/README.md`. |
| `explanation_set/` | Indices of the N=300 simulations used for generating explanations. |
| `rankings/` | Aggregated global feature-importance rankings (CSV) per method, plus stability checks. |
| `configs/` | Per-variant JSON configs (which features are dropped) for `baseline`, `ig`, `kernel_shap`, `random`. KernelSHAP and IG configs are identical per the convergence finding. |
| `checkpoints/` | Retrained model checkpoints per variant (pilot run, CPU, N=500 subset — see `results/step7_pilot_report_DRAFT.md`). |
| `results/` | All write-ups, metrics, and figures — see `results/README.md`. |
| `deliverable/` | The actual D3.2 deliverable draft document and its build script. |
| `audit_steps_1_to_6.py`, `audit_checks_2_3.py` | Independent consistency checks run across the pipeline. |
| `run_step*.py`, `run_step*.sh` | Entry-point scripts for each pipeline step (4 through 7). |
| `validate_300.py`, `paranoia_checks_ig.py`, `postprocess_step4.py` | Supporting/validation scripts for the explanation-generation step. |

## Reproducing results

See `PIPELINE.md` for the full step-by-step. In short: Steps 1–6 run on CPU in minutes; Step 7
(retraining 13 models) is the expensive step and is documented as a CPU pilot only — see
`results/step7_pilot_report_DRAFT.md` for its scope and limitations.
