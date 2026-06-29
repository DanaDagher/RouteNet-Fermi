# Design Decisions Log — Problems Faced and How They Were Resolved

**Author:** Dana Dagher
**Date:** 2026-06-17 (post-Step-5 redesign)
**Scope:** the design decisions taken after the Step 5 rankings were available,
before launching the Step 7 retraining campaign.

This log records, for each problem encountered while preparing Step 7, **what
was observed, what was changed, and why.** It is the source material for the
thesis "Design Decisions / Discussion" section. The formal contract is in
`THESIS_DECISIONS.md` (v7); this document is the narrative behind those entries.

---

## 1. The threshold sweep did not match the structure of the data

**Problem.** The original threshold sweep was k ∈ {25, 50, 75} — round
quartiles chosen before any results existed, with no connection to the actual
attribution distribution. Once the Step 5 rankings were computed, both IG and
KernelSHAP showed a sharp **~19–20× cliff after rank 3**: the top three features
({traffic, sigma, packets}) hold ~93% of the total attribution mass, and the
remaining seven sit at near-noise levels. The quartile thresholds did not align
with this structure — in particular k=25 (top-2) cut *through* the important
group, dropping `packets`.

**Change.** The sweep was changed to **k ∈ {30, 50, 70}** (keep top-3 / top-5 /
top-7). The mapping to feature counts is now exact (k×10 = 3 / 5 / 7).

**Why.** k=30 isolates *exactly* the three features above the cliff — the
cleanest possible fidelity test (relevant_30 tests whether the top-3 are
**sufficient**; irrelevant_30 tests whether they are **necessary**). k=50 is an
intermediate control on the trend; k=70 keeps the top-7 to test whether padding
the relevant set with noise-tail features degrades performance. Crucially, the
thresholds are derived from the **attribution magnitudes** — a property of the
explanations — and not from the downstream retraining performance they are later
used to evaluate, so the design does not condition on its own outcome. The cliff
is robust across both methods and both stability runs, so choosing k=30 reads
obvious structure rather than cherry-picking.

---

## 2. The two XAI methods turned out to be near-identical

**Problem.** The study was designed to *compare* Integrated Gradients and
KernelSHAP and pick the more faithful one. But the two rankings agree almost
perfectly (cross-method Spearman 0.96), and the only differences are **adjacent
swaps** at ranks 1↔2, 4↔5, 6↔7. None of the cut points (after feature 3, 5, 7)
splits a swapped pair, so **at every cell the two methods keep the identical set
of features** (verified by an equality check in `audit_steps_1_to_6.py`). Because
the model re-sorts features to a fixed internal order, identical sets produce
identical retrained models. This had two consequences: (a) training the
KernelSHAP variants would duplicate the IG variants exactly, and (b) the
IG-vs-KernelSHAP *fidelity* comparison is degenerate — identical models give
identical curves, so no method can be ranked above the other on fidelity.

**Change.**
- Train **7 unique principled models** (1 baseline + 6 variants) instead of 13;
  the 6 KernelSHAP variant trainings are skipped because they are bit-identical
  to the IG ones. The 6 trained models live in `checkpoints/principled/`.
- Re-spend the freed compute on the **random control** (see §3).
- Reframe the comparison: IG vs KernelSHAP is decided on **stability** (half-split
  Spearman 0.88 vs 0.95) and **cost** (IG ~5× cheaper: 50 gradient steps vs 256
  perturbations), not fidelity.

**Why.** Training duplicate models would waste compute for zero new information.
And the agreement is not a failure — it is a **finding**: two methods from
opposite families (gradient-based vs perturbation-based) converging on the same
ranking is strong *convergent-validity* evidence that the ranking reflects the
model, not an artifact of any single method. Disagreement between XAI methods is
a well-known concern in the field, so demonstrating convergence is a valued
result. The study still yields a concrete recommendation — **Integrated
Gradients** (same ranking, sufficiently stable, far cheaper).

---

## 3. With the methods equivalent, the fidelity comparison needed a new contrast

**Problem.** If IG ≡ KernelSHAP, what does the fidelity experiment actually
compare? Comparing the two methods to each other is degenerate.

**Change.** The **random-ranking negative control** was promoted from *optional*
(it had been conditional on spare compute) to **core**. The new retraining matrix
is **13 unique models = 1 baseline + 6 principled + 6 random**. Six random
configs were generated from `rankings/random.csv` (`configs/random/`).

**Why.** The meaningful faithfulness test is **principled-vs-random**: the ranking
is faithful if and only if relevant_k stays near baseline, irrelevant_k degrades,
**and both clearly beat the random control**. Without a random baseline a fidelity
claim is unfalsifiable. This is the standard ROAR negative control (Hooker et al.
2019). So the random control now *carries* the fidelity result — it is mandatory,
not optional. (Note: the total compute is the same ~13 models as before, but every
model now does real work instead of 6 being pure duplicates.)

---

## 4. Two of the ten ranked features cannot be fully removed

**Problem.** `traffic` and `packets` are among the 10 XAI-ranked path scalars,
but they also have a **structural** role: RouteNet-Fermi uses them to compute
link load (`traffic/capacity`) and packet size (`pkt_size = traffic/packets`),
which define the physics of the delay prediction. They therefore cannot be
removed from the data without undefining the simulation. The loader's
`_DROPPABLE_PATH_SCALARS` set has 8 entries (not 10) for this reason. So when a
variant "drops" the top-3, only `sigma` is truly removed; `traffic` and `packets`
still feed the model through the physics. This means the k=30 fidelity gap is
weaker than the raw ranking implies.

**Change.** This is stated openly as a **limitation** in `THESIS_DECISIONS §10`,
the PIPELINE Step 8 interpretation rules, and the deliverable: gaps that depend on
removing `traffic`/`packets` are **lower bounds**, and the fidelity test most
cleanly validates the **8 fully-removable features plus `sigma`.**

**Why.** It is an honest, pre-existing design property (documented since §5), not
a flaw discovered late. And it is acceptable because `traffic`/`packets` are
*independently* established as important by queueing theory (they are literally in
the delay formula) — so XAI does not need to "prove" they matter. XAI's
discriminating value is on the **learned** features, which are fully testable.
Stating the limitation plainly strengthens the work's credibility rather than
weakening it.

---

## 5. A reported statistic did not reproduce

**Problem.** The Step 5 report stated that the top-3 features carry "~97%" of the
total attribution mass. Re-computing directly from the committed ranking CSVs gave
**~93%** (IG 92.9%, KernelSHAP 93.3%), and no standard normalization reproduced
97% (linear share = 93%; squared/L2 share = 99.7%; per-simulation median = 93%;
per-simulation mean = 86%).

**Change.** The figure was corrected to **~93%** in `step5_report.md`, with the
exact per-method values and the formula recorded so it cannot drift again.

**Why.** ~93% is the directly reproducible pooled share (sum of top-3 mean
|attribution| ÷ sum across all 10); the 97% was an un-reproducible slip. Pinning
the number to an explicit formula makes it auditable.

---

## 6. The compute platform changed

**Problem.** Step 7 was planned for the Sogeti RTX 4090, whose SSH access never
materialised.

**Change.** Compute moved to **GCP** (a company-provided resource). All references
("Sogeti / RTX 4090") were updated; the old `STEP7_RUNBOOK.md` was marked
**superseded** (its conda/CUDA setup is Sogeti-specific). A GCP-specific runbook is
pending the VM image / GPU type.

**Why.** The trainer and driver scripts are GPU-agnostic and need no changes — only
the runbook does. Execution discipline is unchanged: Dana runs Step 7 on the remote
machine by hand; the scripts are prepared locally.

---

## 7. The raw explanation timing was a misleading cost measure

**Problem.** In the main Step 4 run, IG took ~44 min and KernelSHAP only ~10 min —
which would wrongly suggest the *more complex* method (KernelSHAP, 256 coalition
evaluations) is cheaper than the simpler one (IG, 50 gradient steps). Reviewer
guidance (2026-06-17) was to settle the IG-vs-KernelSHAP comparison on
**complexity**, so a defensible cost metric was needed.

**Change.** The raw sequential wall-clock is **not** used as the cost metric. Two
uncontaminated measures are reported instead: (a) **model evaluations per
explanation** — IG 50 vs KernelSHAP 256 (~5×, hardware- and JIT-independent); and
(b) **equally-warmed per-simulation time** from the reference-sensitivity runs —
IG ≈6.9 s vs KernelSHAP ≈7.9 s. A complexity table was added to the deliverable
(§1.5.5, Table 1-6) and the §1.4.4 timing text now flags the artefact.

**Why.** The 44-vs-10-min gap is a **JIT-compilation artefact**: the two methods
ran back-to-back, so KernelSHAP inherited the TensorFlow graph already compiled by
the IG run — IG paid the one-time compilation, KernelSHAP did not. Both the model-
evaluation count and the warm timing show KernelSHAP is the costlier method,
consistent with theory. Reporting this correctly (rather than hiding the raw
number) turns a measurement pitfall into a rigorous, reviewer-aligned complexity
comparison: equal fidelity + higher complexity ⇒ KernelSHAP is not preferable for
this white-box model; IG is recommended, KernelSHAP reserved for black-box
settings. Note the precise wording: "~5× fewer model evaluations," **not** "~5×
faster" (the warm wall-clock gap is only ~14 %, because each IG evaluation
includes a backward/gradient pass).

## Verification after all changes

- `audit_steps_1_to_6.py` → **ALL CHECKS PASSED** (includes the 6 random-config
  consistency checks and the 6 IG≡KernelSHAP equivalence checks).
- `run_step6_smoke.py` → **19/19 configs PASS** (model instantiates, data loads,
  dropped features absent, forward pass succeeds).

## Open items

- GCP runbook to be written once the VM image / GPU type are confirmed.
- Step 7 (the 13 retrainings) to be executed on GCP by Dana.
