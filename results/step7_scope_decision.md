# Step 7 Scope Decision — Focus the Retraining on k=30

**Date:** 2026-06-17
**Branch:** xai-protocol-b
**Status:** DECIDED (in-session with Dana). **Pending IMT technical advisor sign-off** before the
v7 contract is bumped — see "Contract impact" below.
**Decision owner:** Dana Dagher, on the advice of an external XAI expert,
to be confirmed with the IMT technical advisor who confirmed the original protocol.

---

## Context

The v7 contract (THESIS_DECISIONS / PIPELINE / CLAUDE.md) locks a full
threshold sweep k ∈ {30, 50, 70} × {relevant, irrelevant} for two ranking
sources (principled IG≡KernelSHAP, and random control) → 13 unique retrained
models in Step 7.

After Step 5 produced the global rankings, two findings stand out:

1. **The ~19–20× attribution cliff.** The top-3 path scalars
   (**sigma, traffic, packets**) carry ~97% of the total attribution mass; the
   bottom-7 form a flat near-noise tail with scores ~20× smaller. (Step 5,
   `results/step5_report.md`.)
2. **Convergent validity.** IG and KernelSHAP — two different XAI families —
   agree on the same top-3 set (cross-method Spearman ρ = 0.964), and at every
   cut boundary in {30, 50, 70} they produce **identical kept-feature sets**
   (the v7 IG ≡ KernelSHAP finding).

An external XAI expert reviewed these results and advised that, given how
sharply the signal concentrates in the top-3, the **full sweep is not
necessary** to make the faithfulness point — k=30 (top-3 vs bottom-7) already
isolates the effect. k=50 and k=70 mostly add noise-tail features and act as
sanity checks rather than load-bearing evidence.

---

## Decision

**Step 7 trains the k=30 slice only — 5 models — instead of the full 13.**

| # | Cell | Config | Kept path scalars | What it tests |
|---|---|---|---|---|
| 1 | baseline | `configs/baseline/full.json` | all 10 | reference performance |
| 2 | principled relevant | `configs/ig/k30_relevant.json` | sigma, traffic, packets | **sufficiency** — do the top-3 retain baseline performance? |
| 3 | principled irrelevant | `configs/ig/k30_irrelevant.json` | the other 7 (top-3 dropped) | **necessity** — does removing the top-3 collapse performance? |
| 4 | random relevant | `configs/random/k30_relevant.json` | avg_t_on, avg_pkts_lambda, eq_lambda | **control** — is the effect from the *ranking* or just from keeping any 3? |
| 5 | random irrelevant | `configs/random/k30_irrelevant.json` | the other 7 (random-3 dropped) | control mirror |

The principled cells use the IG configs because IG ≡ KernelSHAP at k=30
(identical kept set {sigma, traffic, packets}); training the KernelSHAP k=30
configs would reproduce them bit-for-bit (v7 finding, enforced by
`audit_steps_1_to_6.py`).

### Why 5 models and not 1 (the top-3 alone)

The expert's headline — "only the top-3 matter, just train that" — is correct
about *where the signal is*, but a single top-3 model can only show the top-3
are **sufficient**. To claim they are the **important** ones (the thesis claim)
the comparison is required:

- **sufficiency:** top-3 kept (#2) ≈ baseline (#1).
- **necessity:** top-3 dropped (#3) ≪ baseline.
- **control:** principled-relevant (#2) ≫ random-relevant (#4), proving the
  effect comes from the *ranking*, not merely from reducing dimensionality.

Cells #3–#5 are the same tiny architecture and cost little extra GPU, but they
are what turns "top-3 are enough" into a defensible faithfulness result. This
is why the reduced set is **5**, not 1.

---

## What is explicitly NOT changed

- **The explanation-set sampling (N=300) stays as-is.** The 300 sims only feed
  the *attribution ranking*, not the retraining (which uses the full
  `all_multiplexed` split). The ranking is already shown stable on that 300:
  half-split Spearman ρ = 0.879 (IG) / 0.952 (KernelSHAP). A fresh 300 would
  reproduce the same top-3, so no re-draw is performed. (If a reviewer wants
  replication, a second independent 300 can be drawn later as a robustness
  check — expected to confirm, not change, the result.)
- **Nothing is deleted.** All 19 configs and the full 13-cell driver
  (`run_step7_all.sh`) remain in the repo. The full {30,50,70} sweep stays
  runnable if the IMT technical advisor or a reviewer asks for it. The k=30 run uses a separate
  focused driver, `run_step7_k30.sh`.
- **No model / data-pipeline code changes.** The Step 6 infrastructure already
  supports any subset of configs.

---

## Contract impact

This narrows the v7 Step 7 scope (13 → 5 models). Because the retraining
protocol was confirmed by the IMT technical advisor, this scope reduction should be
reviewed before THESIS_DECISIONS is bumped to v8. Until then:

- This file is the decision record of intent.
- `run_step7_k30.sh` is the execution artifact.
- THESIS_DECISIONS / PIPELINE / CLAUDE.md remain at v7 (full sweep) and are
  **not** edited yet — to avoid locking a contract change ahead of sign-off.

**Action for Dana:** confirm the k=30-only scope with the IMT technical advisor,
then fold this into THESIS_DECISIONS as a v8 changelog entry.

---

## Compute

GCP (company-provided). VM image / GPU type not yet chosen — the GCP runbook is
deferred until that is decided. The training/driver scripts are GPU-agnostic
and need no change for GCP; only the runbook does.
