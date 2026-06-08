# Theoretical Expectations — Step 3.5
# XAI Feature Importance Baseline from Queueing Theory
# Author: Dana Dagher
# Date: June 2026
# Status: Written after Step 4 IG raw results were observed, 
#         but BEFORE Step 5 formal aggregation and ranking.
# Framing: Theoretical baseline documentation, not blind pre-registration.

---

## 1. Context and Purpose

This document records the theoretical expectations for feature importance
in RouteNet-Fermi delay prediction on the `all_multiplexed` sub-dataset,
derived from standard queueing theory (M/M/1 and M/G/1 models).

It serves as a theoretical baseline to compare against the observed XAI
rankings produced by Integrated Gradients (IG) and KernelSHAP in Steps 4–5.
The goal is to assess whether the GNN-learned delay model aligns with
classical network theory, or whether it captures more complex dynamics
that simplified queueing models miss.

**Honest framing:** Step 4 raw attributions were observed before this
document was written (IG run completed, basic spot-checks seen). However,
Step 5 formal aggregation and the final comparative ranking between IG and
KernelSHAP have NOT yet been reviewed. The predictions below reflect
standard queueing theory, not reverse-engineered observations.

---

## 2. Background — The 10 Path Scalar Features

RouteNet-Fermi operates on 22 input features per flow. XAI ranking covers
the 10 per-flow path scalars:

| Feature | Description | Traffic type it characterizes |
|---|---|---|
| `traffic` | Mean bit rate (bytes/s) | All types |
| `packets` | Mean packet rate (packets/s) | All types |
| `eq_lambda` | Equivalent Poisson arrival rate | All types |
| `avg_pkts_lambda` | Average packet arrival rate during ON period | ON/OFF |
| `exp_max_factor` | Exponential burst size factor | ON/OFF |
| `pkts_lambda_on` | Packet rate during ON period | ON/OFF |
| `avg_t_off` | Mean OFF period duration | ON/OFF |
| `avg_t_on` | Mean ON period duration | ON/OFF |
| `ar_a` | Autocorrelation coefficient | Autocorrelated |
| `sigma` | Modulation amplitude | Modulated |

---

## 3. Naive Expectation — M/M/1 Reasoning

Under the simplest queueing model (M/M/1: Poisson arrivals, exponential
service times, single server), per-flow delay is governed entirely by
the mean arrival rate relative to service capacity:
E[delay] = 1 / (μ - λ)    where ρ = λ/μ → 1 causes divergence

Under this model, only the **mean arrival rate** matters.
The variability and shape of the arrival process are irrelevant.

**M/M/1 predicted ranking:**

| Rank | Feature | Justification |
|---|---|---|
| 1 | `traffic` | Directly sets utilization ρ = λ/μ. Delay diverges as ρ → 1. Primary driver. |
| 2 | `packets` | Packet rate coupled to traffic. Adds granularity for small-payload flows. |
| 3 | `eq_lambda` | Equivalent Poisson rate. Redundant proxy for traffic but still rate-based. |
| 4–10 | All others | Describe arrival shape, not mean rate. Irrelevant under M/M/1. |

**M/M/1 prediction:** `traffic` > `packets` > `eq_lambda` >> everything else.
Bottom 7 features should be near-zero in importance.

---

## 4. Refined Expectation — M/G/1 and Pollaczek-Khinchine

Real network traffic is not Poisson. The Pollaczek-Khinchine (P-K) formula
for M/G/1 queues (Poisson arrivals, general service times) gives:

W_q = (λ · E[S²]) / (2(1 - ρ))

The mean waiting time depends on the **second moment of service time**
E[S²], not just the mean. This means **variability matters as much as
mean rate** for delay prediction. High variance in inter-arrival times
(burstiness) inflates E[S²] and therefore inflates delay.

For `all_multiplexed`, which mixes CBR, ON/OFF, autocorrelated, and
modulated flows in the same simulations:

- `sigma` (modulation amplitude) directly controls the variance of
  the arrival rate for modulated flows → increases E[S²] → inflates delay
- `ar_a` (autocorrelation) creates long-range dependence in arrivals →
  increases effective burstiness beyond what Poisson captures
- `avg_t_on` / `avg_t_off` control ON/OFF burst structure →
  second moment effect for ON/OFF flows

**M/G/1 predicted ranking (refined):**

| Tier | Features | Reasoning |
|---|---|---|
| Tier 1 — rate-driven | `traffic`, `packets`, `eq_lambda` | First moment of arrival process |
| Tier 2 — burstiness-driven | `sigma`, `ar_a`, `exp_max_factor` | Second moment / variance effects |
| Tier 3 — shape parameters | `avg_t_on`, `avg_t_off`, `pkts_lambda_on` | Burst structure, secondary |
| Bottom | `avg_pkts_lambda` | Redundant with `packets` in most regimes |

**M/G/1 prediction:** Tier 1 and Tier 2 are co-dominant in mixed-traffic
regimes. `sigma` and `ar_a` may challenge `traffic` for top position.

---

## 5. Experimental Scope Note

XAI attributions are computed for **`flow_idx=0` only** per simulation — not
averaged over all flows in the simulation. The traffic-model class distribution
of flow_0 across the 300 test simulations is:

| Class | Count | % |
|---|---|---|
| CBR/Deterministic | 50 | 16.7% |
| Poisson | 55 | 18.3% |
| Modulated | 63 | 21.0% |
| AR1 (Autocorrelated) | 64 | 21.3% |
| AR1-1 (Autocorrelated) | 68 | 22.7% |

No single class dominates. The theoretical predictions in Sections 3–4 apply
to this mixed-class population, not to a single traffic type.

---

## 6. Open Questions — To Be Answered by Steps 5–8


1. **Does M/M/1 or M/G/1 reasoning better describe RouteNet-Fermi's
   learned behavior on `all_multiplexed`?**
   - If `traffic` >> everything else → M/M/1 dominates
   - If `sigma` or `ar_a` challenge `traffic` → M/G/1 / P-K dominates

2. **Do IG and KernelSHAP agree on the ranking?**
   - Prediction: high agreement (Spearman ρ > 0.80) on top-3,
     possible divergence on middle-tier features
   - Gradient-based IG may be more sensitive to local burstiness peaks
   - Perturbation-based KernelSHAP may weight average behavior more
   - **Note (Step 4 pre-verification):** Spearman ρ was already computed
     in Step 4 audit Check 3: ρ = 0.9636, p = 0.000007. The observed value
     exceeds the prediction (> 0.80), confirming very high agreement before
     Step 5 formal aggregation.

3. **Does the fidelity gap in Step 8 confirm the XAI ranking?**
   - If dropping bottom-k% features degrades MAE more than dropping
     top-k% → the ranking is faithful
   - If both variants perform similarly → XAI is not capturing the
     true decision boundary

---

## 7. Expected Fidelity Curve Shape (Step 8 Prediction)

For each method at threshold k:
- `gap(k) = MAE(irrelevant_k) - MAE(relevant_k)`

Expected behavior:
- **k=25** (keep top-2 only): gap should be **moderate** — top-2 features
  carry significant signal but not everything
- **k=50** (keep top-5): gap should be **largest** — clean separation
  between informative and uninformative halves
- **k=75** (keep top-7): gap should be **small** — only 2 features dropped,
  and the bottom-2 are unlikely to carry much signal

Expected ordering: `gap(k=50) > gap(k=25) > gap(k=75)`

**Alternative prediction (to be falsified by Step 8):** if the top-2 features
(sigma, traffic) carry the vast majority of signal, gap(k=25) could exceed
gap(k=50) — i.e. dropping only the top-2 hurts more than dropping the top-5.
The observed fidelity curves in Step 8 will determine which ordering holds.

If IG is more faithful than KernelSHAP:
`gap_IG(k) > gap_KernelSHAP(k)` at most thresholds

If both methods are equally faithful:
`gap_IG(k) ≈ gap_KernelSHAP(k)` → practical recommendation favors IG
(cheaper: 50 gradient steps vs 256 perturbations)

---

## 8. Summary of Predictions

| Prediction | Source | Verifiable at Step |
|---|---|---|
| `traffic` top-1 under naive M/M/1 | M/M/1 theory | Step 5 |
| `sigma` challenges `traffic` in mixed regime | M/G/1 / P-K | Step 5 |
| IG and KernelSHAP agree on top-3 | XAI literature + pilot | Step 5 |
| Spearman ρ(IG, SHAP) > 0.80 | Pilot CBR agreement | Step 5 |
| gap(k=50) is largest | Feature importance structure | Step 8 |
| Random control gap ≈ 0 | Negative control logic | Step 8 |
| Dropping bottom-k% hurts more than dropping top-k% | Fidelity definition | Step 8 |