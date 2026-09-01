# IG per-slice report (V2X, ckpt_v2, per-slice baseline)

**IG run:** `ig_shared_300_perslice.npz`  
**Diagnostics:** `audit_diagnostics.json`  
**Total flows:** 300 (100 eMBB, 100 mMTC, 100 URLLC)

## 1. Convergence

- Overall: 300/300 flows converged within max_steps (100.0%).
- Residual as %% of max(effect, max|IG|): mean=0.33%, median=0.12%, max=9.90%.
- Steps used: min=100, median=100, max=200.

This is the corrected result after switching to a per-slice baseline. The earlier single-baseline pilot showed residuals commonly >10% and adaptive step doubling routinely triggered; here almost every flow converges at the starting step count and no flow exceeds 400 steps.

## 2. Per-slice feature ranking (mean |IG|)

### eMBB (n=100)

| rank | feature | mean |IG| | median |IG| | mean signed IG |
|---|---|---|---|---|
| 1 | `delta` | 0.009065 | 0.0003899 | -0.0005034 |
| 2 | `traffic` | 1.779e-05 | 2.811e-07 | -1.121e-05 |
| 3 | `packets` | 1.156e-06 | 8.748e-08 | -5.938e-07 |
| 4 | `avg_pkts_lambda` | 1.99e-10 | 9.752e-12 | -1.736e-10 |
| 5 | `eq_lambda` | 0 | 0 | +0 |
| 6 | `exp_max_factor` | 0 | 0 | +0 |
| 7 | `slice_type` | 0 | 0 | +0 |
| 8 | `model` | 0 | 0 | +0 |

- **Top-3:** delta, traffic, packets
- **Effect size** (|f_actual - f_baseline|): median=0.000388, p10=2.674e-05, p90=0.005652
- **Converged:** 100.0%  **steps used:** median=100, max=100
- **Completeness residual:** median=0.01%, max=9.90%

### mMTC (n=100)

| rank | feature | mean |IG| | median |IG| | mean signed IG |
|---|---|---|---|---|
| 1 | `delta` | 0.0007391 | 0.0001498 | -0.0002063 |
| 2 | `packets` | 2.769e-08 | 1.799e-09 | +8.953e-09 |
| 3 | `traffic` | 5.425e-09 | 7.306e-10 | +7.658e-10 |
| 4 | `eq_lambda` | 1.202e-11 | 9.225e-13 | -2.5e-14 |
| 5 | `avg_pkts_lambda` | 0 | 0 | +0 |
| 6 | `exp_max_factor` | 0 | 0 | +0 |
| 7 | `slice_type` | 0 | 0 | +0 |
| 8 | `model` | 0 | 0 | +0 |

- **Top-3:** delta, packets, traffic
- **Effect size** (|f_actual - f_baseline|): median=0.0001498, p10=7.454e-07, p90=0.002613
- **Converged:** 100.0%  **steps used:** median=100, max=200
- **Completeness residual:** median=0.27%, max=4.52%

### URLLC (n=100)

| rank | feature | mean |IG| | median |IG| | mean signed IG |
|---|---|---|---|---|
| 1 | `delta` | 0.0007384 | 0.0001145 | +0.0004247 |
| 2 | `packets` | 1.701e-07 | 3.053e-08 | +5.772e-08 |
| 3 | `traffic` | 1.457e-07 | 1.856e-08 | +5.592e-08 |
| 4 | `avg_pkts_lambda` | 8.983e-11 | 1.465e-11 | +2.602e-11 |
| 5 | `eq_lambda` | 0 | 0 | +0 |
| 6 | `exp_max_factor` | 0 | 0 | +0 |
| 7 | `slice_type` | 0 | 0 | +0 |
| 8 | `model` | 0 | 0 | +0 |

- **Top-3:** delta, packets, traffic
- **Effect size** (|f_actual - f_baseline|): median=0.0001147, p10=1.009e-05, p90=0.002123
- **Converged:** 100.0%  **steps used:** median=100, max=100
- **Completeness residual:** median=0.13%, max=2.62%

## 3. Cross-slice rank agreement (Spearman on feature order)

- eMBB vs mMTC:  +0.952
- eMBB vs URLLC: +0.976
- mMTC vs URLLC: +0.976

Rank correlation of 1.0 = identical feature ordering across slices; -1.0 = fully inverted; 0.0 = unrelated. Low values mean the model uses different features on different slices (genuine per-slice explanation).

## 4. Caveats you must state in the thesis

1. **Attribution is on the learned door only.** `traffic` and `packets` also affect delay through the queueing-physics door (`load`, `pkt_size`, `trans_delay` in `forward_from_path_input`) which does not pass through the 16-dim path input. IG here under-reports their total influence. `slice_type` has no physics door -> IG on `slice_type` is a fair measure of learned use.
2. **`slice_type` attribution is not identifiable in isolation.** Training-set correlations of the slice one-hot with `packets` and `avg_pkts_lambda` are ~0.94-0.99. A large IG value on `slice_type` could equivalently be assigned to those numeric features; a small IG value does not prove the model does not use slice identity. Report `slice_type` + traffic-profile features as a jointly-important group, not individually.
3. **Categorical columns are 0 by construction** in the reported IG (`scope=numeric`). This is a deliberate choice, documented in the code: interpolating one-hots through invalid mixes ([0.3, 0, 0]) is a known IG failure mode. It is 'not attributed', not 'attributed and zero'.
4. **`exp_max_factor` and `model` are dead features.** The diagnostic showed `exp_max_factor` is constant (=10) on all 300 test flows and `model` is 100% class 0 in the training sample. Any IG value on them will be exactly 0 in `scope=numeric` (category door closed) and negligible in `scope=full`. Cite the diagnostic; do not interpret.
5. **The mMTC failure regime.** The diagnostic reported 25/100 mMTC flows are 'Type 2' (model output close to baseline but ground truth far from baseline). IG attributions on those flows explain the model, not the physical delay. Report the typology count from the diagnostic alongside the IG results.
