# Theoretical Expectations — Step 3.5
# XAI Feature Importance Baseline from Queueing Theory
# Author: Dana Dagher
# Date: June 2026
# Status: Written after Step 4 IG raw results were observed, but BEFORE Step 5
#         formal aggregation and ranking. Theoretical baseline, not blind
#         pre-registration.

---

## 1. Purpose

This document records what queueing theory (M/M/1 and M/G/1) predicts for the
importance of the 10 path-scalar features in RouteNet-Fermi delay prediction on
`all_multiplexed`. It is the theoretical baseline that the observed IG and
KernelSHAP rankings (Steps 4–5) are compared against in the plausibility
analysis (Step 10a).

Step 4 raw IG attributions were spot-checked before this was written; Step 5
formal aggregation was not yet reviewed. The predictions below come from theory,
not from the final rankings.

---

## 2. The 10 Path-Scalar Features

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

## 3. Naive Expectation — M/M/1

M/M/1 (Poisson arrivals, exponential service, single server) makes delay depend
only on mean arrival rate vs. capacity: `E[delay] = 1 / (μ − λ)`, diverging as
`ρ = λ/μ → 1`. Arrival shape and variability are irrelevant.

**Predicted ranking:** `traffic` > `packets` > `eq_lambda` >> everything else.
The bottom 7 features should be near-zero, since they describe arrival shape, not
mean rate.

---

## 4. Refined Expectation — M/G/1 (Pollaczek-Khinchine)

Real traffic is not Poisson. For M/G/1, the P-K formula
`W_q = (λ · E[S²]) / (2(1 − ρ))` makes mean waiting time depend on the **second
moment** of service time, not just the mean — so variability (burstiness) matters
as much as rate. In `all_multiplexed` (CBR + ON/OFF + autocorrelated + modulated
mixed together):

- `sigma` controls arrival-rate variance for modulated flows → inflates E[S²].
- `ar_a` adds long-range dependence → effective burstiness beyond Poisson.
- `avg_t_on` / `avg_t_off` set ON/OFF burst structure → second-moment effect.

**Predicted tiers:**

| Tier | Features | Reasoning |
|---|---|---|
| 1 — rate-driven | `traffic`, `packets`, `eq_lambda` | First moment of arrivals |
| 2 — burstiness-driven | `sigma`, `ar_a`, `exp_max_factor` | Second-moment / variance |
| 3 — shape parameters | `avg_t_on`, `avg_t_off`, `pkts_lambda_on` | Burst structure, secondary |
| Bottom | `avg_pkts_lambda` | Redundant with `packets` |

**Prediction:** Tier 1 and Tier 2 are co-dominant in mixed traffic; `sigma` and
`ar_a` may rival `traffic` for the top position.

---

## 5. Scope Note

Attributions are computed for **`flow_idx=0` only** per simulation, not averaged
over all flows. The flow-0 traffic-class distribution across the 300 test
simulations is mixed (no class above ~23%): CBR 16.7%, Poisson 18.3%,
Modulated 21.0%, AR1 21.3%, AR1-1 22.7%. The predictions above apply to this
mixed-class population, not to any single traffic type.
