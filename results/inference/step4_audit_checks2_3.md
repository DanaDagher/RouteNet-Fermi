# Step 4 Audit Checks 2 and 3
**Date:** 2026-06-04  
**N:** 300 simulations, all_multiplexed test, flow_idx=0

## Check 2 � KernelSHAP top-attributed feature distribution

| Feature | SHAP top count | SHAP % | IG top count | IG % |
|---|---|---|---|---|
| traffic | 214 | 71.3% | 212 | 70.7% |
| sigma | 65 | 21.7% | 68 | 22.7% |
| avg_t_off | 7 | 2.3% | 9 | 3.0% |
| packets | 4 | 1.3% | 1 | 0.3% |
| ar_a | 4 | 1.3% | 5 | 1.7% |
| exp_max_factor | 3 | 1.0% | 3 | 1.0% |
| avg_pkts_lambda | 2 | 0.7% | 2 | 0.7% |
| pkts_lambda_on | 1 | 0.3% | 0 | 0.0% |

**Cross-check � sigma vs Modulated flows:**
- Modulated flows in class distribution: **63** (21.0%)
- sigma top in SHAP: **65** (21.7%)
- sigma top in IG:   **68** (22.7%)
- Match (within 10 sims): **YES**

## Check 3 � Spearman rank correlation: IG vs KernelSHAP global ranking

| Feature | mean(|IG|) | IG rank | mean(|SHAP|) | SHAP rank |
|---|---|---|---|---|
| sigma | 0.182053 | 1 | 0.176979 | 2 |
| traffic | 0.159653 | 2 | 0.181412 | 1 |
| packets | 0.130660 | 3 | 0.154102 | 3 |
| pkts_lambda_on | 0.007042 | 4 | 0.007108 | 5 |
| eq_lambda | 0.006532 | 5 | 0.007863 | 4 |
| avg_t_off | 0.006378 | 6 | 0.005437 | 7 |
| ar_a | 0.005825 | 7 | 0.005642 | 6 |
| exp_max_factor | 0.005217 | 8 | 0.004735 | 8 |
| avg_t_on | 0.003611 | 9 | 0.004275 | 9 |
| avg_pkts_lambda | 0.001760 | 10 | 0.001666 | 10 |

**Spearman rho = 0.9636,  p-value = 0.000007**

**Verdict:** Very high agreement (rho >= 0.9). Both methods rank features consistently.

**Top-3 agreement check:**
- IG top-3:    sigma, traffic, packets
- SHAP top-3:  traffic, sigma, packets
- Overlap: 3/3 features
