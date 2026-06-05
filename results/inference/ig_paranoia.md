# IG Paranoia Checks
**Date:** 2026-06-04  
**Source:** results/inference/ig/ (300 sims, all_multiplexed)

## Check 1 - Sign sanity for traffic, packets, sigma

Positive = feature pushes predicted delay UP relative to baseline.  
Negative = feature pushes delay DOWN.  
A 50/50 split would mean the feature has no consistent directional effect.

| Feature | Positive | Negative | Zero | Mean | Median | Std | Flag? |
|---|---|---|---|---|---|---|---|
| traffic | 154 | 146 | 0 | 0.0060 | 0.0018 | 0.2043 | FLAG: near 50/50 |
| packets | 148 | 152 | 0 | 0.0164 | -0.0012 | 0.1606 | FLAG: near 50/50 |
| sigma | 68 | 0 | 232 | 0.1821 | 0.0000 | 0.3399 | OK |

**Flagged features (near 50/50 split): traffic, packets**  
Investigate sign consistency in Step 5 plausibility analysis.

## Check 2 - Outlier sensitivity for sigma

Top-5 sigma outlier sims removed (indices: 41, 108, 132, 166, 245).
Their sigma attributions: 1.005, 0.988, 1.148, 0.988, 1.011

| Rank | Full 300 sims | mean(|IG|) | Trimmed 295 sims | mean(|IG|) |
|---|---|---|---|---|
| 1 | sigma | 0.182053 | sigma | 0.167716 |
| 2 | traffic | 0.159653 | traffic | 0.159727 |
| 3 | packets | 0.130660 | packets | 0.130657 |
| 4 | pkts_lambda_on | 0.007042 | pkts_lambda_on | 0.007162 |
| 5 | eq_lambda | 0.006532 | avg_t_off | 0.006486 |

**Robust.** sigma remains #1 after removing top-5 outliers (0.182053 -> 0.167716). sigma dominance is not outlier-driven.
