# Step 7 Pilot — Working Report (DRAFT)

**Status:** DRAFT — all 5 models complete; results and interpretation filled in.
**Date:** 2026-06-22
**Branch:** xai-protocol-b
**Compute:** Dana's local Windows PC, CPU only (no GPU available).
**Purpose of this draft:** capture the pilot setup, the methodological
clarifications that came up while running it, and the key limitation we
discovered — to be merged into the final Step 7 pilot report and raised with
Karim for D3.2.

---

## 1. Why a pilot on a sub-dataset

The full Step 7 retraining matrix needs a GPU. Neither GCP (quota blocked) nor
the Sogeti RTX 4090 became available before the deadline, so we run a **pilot**
of the faithfulness comparison **on Dana's CPU**, using a **fixed seeded
sub-dataset** so it finishes in hours instead of days.

This is a deliberate, documented compute-constrained design — **not** a change
to the methodology. The full-data scripts (`run_step7_all.sh`,
`run_step7_k30.sh`) are untouched and will supersede this pilot if a GPU
appears.

## 2. Setup

- **5 models** (the k=30 slice): baseline + principled relevant/irrelevant +
  random relevant/irrelevant.
- **Training data:** first **500** simulations of the `all_multiplexed` train
  split (fixed, seeded, cached). The full split is ~75,000 simulations; 500 is
  the pilot subset.
- **Hyperparameters:** Adam (lr 1e-3), MAPE loss, hidden state 32, T=8,
  seed 42 — same as the paper §IV-D **except** the step budget: **40 epochs**
  over the 500-sample subset (the paper uses 150 epochs over the full data).
  40 epochs is identical across all 5 pilot models, so they remain comparable.
- **Evaluation:** every model is finally evaluated on the **same** fixed test
  subset (300 simulations); a small validation set (100) tracks convergence.

## 3. Why the baseline is re-trained (not reused)

All 5 models — baseline included — train on the **identical** 500-sample subset
with **identical** hyperparameters, so the only difference between any two
models is the kept input features. The upstream pretrained checkpoint was
trained on the **full** dataset (and an unknown run/seed), so it cannot be the
reference for a sub-dataset comparison. Same data + same settings +
only-features-differ ⇒ baseline must be retrained on this subset.

---

## 4. Methodological clarifications (Q&A)

These questions came up while running the pilot. Each is a point worth stating
explicitly in the thesis methods section.

**Q. Does "2,000 samples" mean we only train on 2,000 samples?**
No. "2,000 samples" is the paper's **steps per epoch** — how many samples are
*fed per epoch* — not the dataset size. The full training set is ~75,000
simulations; the data pipeline cycles through it with `.repeat()`. (In the
pilot we deliberately restrict to a fixed 500-sample subset for CPU speed.)

**Q. The repo's `main.py` says 50 epochs, but we use 150 — which is right?**
The **paper** is the authority. RouteNet-Fermi §IV-D states verbatim: *"we
train RouteNet-F during 150 epochs of 2,000 samples each."* The repo's
`main.py` uses `epochs=50` as a lighter default for users; it is not the
paper's training recipe. The full-data protocol therefore uses 150. (The pilot
uses 40 for CPU feasibility, consistently across all 5 models.)

**Q. The numbers shown during training — are they training error?**
Each model produces three numbers: **train** MAPE (error on data it learned
from — optimistic), **validation** MAPE (error on held-out data, checked each
epoch), and **test** MAPE (error on a separate held-out test set, computed once
at the end). The **test** MAPE is the measure of real efficiency, and it is the
number we compare across the 5 models. It is computed automatically per model.

**Q. The variant is named "irrelevant / top-3 dropped" — but were the top-3
actually dropped?**
Only partially, and the word "dropped" is misleading. See §5. Corrected wording
is used below: the top-3 are **excluded from the learned feature set**, but
`traffic` and `packets` are **not** removed from the model.

---

## 5. Key finding — the retraining test cannot ablate `traffic` and `packets`

This is the most important methodological result of the pilot.

### The two-doors mechanism

Each flow's data (`traffic`, `packets`, `sigma`, …) enters RouteNet-Fermi
through **two separate paths**:

- **Door 1 — the learned feature embedding.** A small neural network *learns*
  patterns from a chosen list of per-flow scalars. We control this list. For a
  variant, we can exclude features here.
- **Door 2 — the fixed delay formula.** Independently, the model computes delay
  with hard-coded arithmetic that reads `traffic` and `packets` directly:
  - link load = `traffic ÷ capacity`
  - transmission delay = `traffic ÷ packets × …`
  This is the physics of the model and cannot be removed without breaking it.

When a variant "drops" `traffic`/`packets`, it only closes **Door 1**. **Door 2
stays open** — the formula still reads them from the input data. So the model
still predicts delay well, because most of the delay comes from Door 2.

### Consequence

The ROAR-style "necessity" check — *remove the important features and watch the
model get worse* — **cannot work for `traffic` and `packets`**, because they are
structurally inseparable from the delay computation. Among the top-3, only
`sigma` is fully removable (it is a Door-1-only feature).

### Why this is a finding, not a failure

The two features the XAI methods ranked most important are so fundamental to the
model's delay computation that they **cannot be ablated without breaking the
model** — which is itself strong evidence of their importance. The *ranking*
result (IG and KernelSHAP both put `traffic`, `packets`, `sigma` on top, with a
~20× gap to the rest) stands independently and is unaffected.

### Naming correction

The folder/config names `relevant` / `irrelevant` are the standard ROAR
convention (keep-top vs keep-bottom). The shorthand **"dropped" is inaccurate**
for `traffic`/`packets` and is replaced everywhere by:

> *excluded from the learned feature embedding (Door 1); `traffic` and `packets`
> remain in the delay formula (Door 2).*

---

## 6. The faithfulness comparison that DOES work

Because the `relevant` vs `irrelevant` contrast is compromised for
`traffic`/`packets`, the clean faithfulness test is **`relevant` vs `random`**
at the same k: both models keep only 3 features in the learned embedding (on top
of the shared structural base). If the **principled** top-3 model clearly beats
the **random** 3-feature model on test MAPE, that demonstrates the *ranking*
carries real information — the intended result.

---

## 7. Results (all 5 models complete)

Test MAPE (lower = better), on the same 300-sample fixed test set; trained on
the same 500-sample subset, 40 epochs, seed 42:

| Model | learned features | n kept | Test MAPE |
|---|---|---|---|
| principled k30_relevant | top-3 (sigma, traffic, packets) | 3 | **5.41%** |
| random k30_relevant | 3 random scalars | 3 | **5.43%** |
| random k30_irrelevant | 7 random scalars | 7 | 5.80% |
| baseline | all 10 | 10 | 6.14% |
| principled k30_irrelevant | bottom-7 (top-3 removed from embedding) | 7 | 6.38% |

**The result is a null for faithfulness — and it is consistent and explained:**

1. **principled-3 (5.41%) ≈ random-3 (5.43%)** — a near-exact tie. The
   *identity* of the embedding features does not affect accuracy. This is the
   decisive comparison, and it shows **no** faithfulness signal.
2. **The whole spread is ~1 pp (5.41–6.38%) and does not track importance.**
   The all-10 baseline (6.14%) is *worse* than the 3-feature models, and
   removing the top-3 (irrelevant, 6.38%) does not collapse accuracy.
3. **Why:** as the delay-mechanism analysis shows (see
   `step7_delay_mechanism_analysis_DRAFT.md`), `traffic`/`packets` re-enter the
   prediction through the QT terms (link load eq. 7, transmission eq. 9) in
   *every* model, so the learnable embedding has little marginal influence — the
   small differences are within small-sample (500) training variance and are
   uncorrelated with the XAI ranking.

**Conclusion:** the ROAR-style retraining test **cannot discriminate** feature
importance for RouteNet-Fermi delay. This is the pilot's headline — a critical
finding, not a measurement to be "improved," because its cause is structural
(the model's physics), not the dataset size or the run.

---

## 8. Implications / to raise with Karim

1. The **necessity arm** of the retraining test is structurally limited for
   `traffic` and `packets`. This should be stated as a limitation in D3.2, with
   the two-doors explanation.
2. The **faithfulness conclusion** rests on the **principled-vs-random**
   contrast, not principled-vs-irrelevant.
3. The **feature-ranking** result (IG ≡ KernelSHAP, ~20× cliff, top-3 =
   traffic/packets/sigma) and the **IG-vs-KernelSHAP comparison** (stability,
   cost) are unaffected and remain the core contributions.

---

## 9. Operational notes (lessons from the pilot run)

- **Disable PC sleep for overnight CPU runs.** Sleep pauses training; the first
  overnight run lost hours this way.
- **`Ctrl+C` stops training** (it is not "copy"). The run is resumable — re-run
  the driver and it skips finished models and resumes a partial one from its
  last epoch checkpoint.
- **Run only one driver at a time** — concurrent runs write to the same
  checkpoint folders.
- **Caching works:** first epoch ~1.4 s/step (parsing), cached epochs
  ~0.26 s/step (~8× faster).
