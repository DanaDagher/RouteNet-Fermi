# Three-method cross-comparison (IG vs KernelSHAP vs Permutation Importance)

Reports per-slice feature rankings from each XAI method and pairwise Spearman rank correlations.

## Method scopes recap

| Method | # features attributed | Handles categoricals? | Baseline / background | Level |
|---|---|---|---|---|
| IG | 6 (numeric only) | No -- interpolation invalid on one-hots | per-slice training median | per-flow, all 300 |
| KernelSHAP grouped | 8 (adds slice_type + model) | Yes -- whole one-hot block swapped, never fractional | numerics = per-slice median; slice_type = 3-way empirical-frequency average; model = single-point (100% class 0) | per-flow, all 300 |
| Permutation Importance | 8 | Yes -- whole one-hot block shuffled | none (measured against baseline model error) | global, 300 flows, 5 repeats |

The 8 semantic groups are always: `traffic, packets, eq_lambda, avg_pkts_lambda, exp_max_factor, delta, slice_type, model`.

Two data-degenerate features (`model` = 100% class 0; `exp_max_factor` = constant 10) are included as positive controls -- any XAI method returning non-zero on either would indicate a methodological artifact.

## eMBB

### Feature scores (higher = more important)

| feature | IG mean\|attribution\| | SHAP mean\|phi\| | Permutation ΔMAE |
|---|---|---|---|
| `traffic` | 1.779e-05 | 2.102e-05 | +0.0293 |
| `packets` | 1.156e-06 | 1.211e-06 | -0.002069 |
| `eq_lambda` | 0 | 1.015e-08 | +0.004376 |
| `avg_pkts_lambda` | 1.99e-10 | 6.569e-09 | +0.0009117 |
| `exp_max_factor` | 0 | 7.258e-09 | +0 |
| `delta` | 0.009065 | 0.006143 | -0.002843 |
| `slice_type` | 0 | 0.01674 | -0.01282 |
| `model` | 0 | 3.501e-09 | +0 |

### Rankings

| Method | on 6 numeric features | on all 8 features |
|---|---|---|
| IG          | delta > traffic > packets > avg_pkts_lambda > eq_lambda > exp_max_factor | *(not applicable -- IG cannot rank categoricals)* |
| SHAP        | delta > traffic > packets > eq_lambda > exp_max_factor > avg_pkts_lambda | slice_type > delta > traffic > packets > eq_lambda > exp_max_factor > avg_pkts_lambda > model |
| Permutation | traffic > eq_lambda > delta > packets > avg_pkts_lambda > exp_max_factor | traffic > slice_type > eq_lambda > delta > packets > avg_pkts_lambda > exp_max_factor > model |

### Rank agreement (Spearman)

- **On 6 numeric features:**
  - IG vs SHAP: **+0.829**
  - IG vs Permutation: **+0.543**
  - SHAP vs Permutation: **+0.657**
- **On 8 features (SHAP + Permutation only):**
  - SHAP vs Permutation: **+0.810**

Top-3 by method: IG=['delta', 'traffic', 'packets'], SHAP-8=['slice_type', 'delta', 'traffic'], Perm-8=['traffic', 'slice_type', 'eq_lambda']

## mMTC

### Feature scores (higher = more important)

| feature | IG mean\|attribution\| | SHAP mean\|phi\| | Permutation ΔMAE |
|---|---|---|---|
| `traffic` | 5.425e-09 | 8.023e-09 | +0.02296 |
| `packets` | 2.769e-08 | 3.667e-08 | +0.001005 |
| `eq_lambda` | 1.202e-11 | 1.073e-09 | +0.004275 |
| `avg_pkts_lambda` | 0 | 7.575e-10 | +0.0007256 |
| `exp_max_factor` | 0 | 4.608e-10 | +2.384e-08 |
| `delta` | 0.0007391 | 0.0007905 | +0.0005744 |
| `slice_type` | 0 | 0.0006039 | +0.001868 |
| `model` | 0 | 9.537e-10 | +1.788e-08 |

### Rankings

| Method | on 6 numeric features | on all 8 features |
|---|---|---|
| IG          | delta > packets > traffic > eq_lambda > avg_pkts_lambda > exp_max_factor | *(not applicable -- IG cannot rank categoricals)* |
| SHAP        | delta > packets > traffic > eq_lambda > avg_pkts_lambda > exp_max_factor | delta > slice_type > packets > traffic > eq_lambda > model > avg_pkts_lambda > exp_max_factor |
| Permutation | traffic > eq_lambda > packets > avg_pkts_lambda > delta > exp_max_factor | traffic > eq_lambda > slice_type > packets > avg_pkts_lambda > delta > exp_max_factor > model |

### Rank agreement (Spearman)

- **On 6 numeric features:**
  - IG vs SHAP: **+1.000**
  - IG vs Permutation: **+0.257**
  - SHAP vs Permutation: **+0.257**
- **On 8 features (SHAP + Permutation only):**
  - SHAP vs Permutation: **+0.357**

Top-3 by method: IG=['delta', 'packets', 'traffic'], SHAP-8=['delta', 'slice_type', 'packets'], Perm-8=['traffic', 'eq_lambda', 'slice_type']

## URLLC

### Feature scores (higher = more important)

| feature | IG mean\|attribution\| | SHAP mean\|phi\| | Permutation ΔMAE |
|---|---|---|---|
| `traffic` | 1.457e-07 | 1.701e-07 | +0.002725 |
| `packets` | 1.701e-07 | 1.756e-07 | +0.0006662 |
| `eq_lambda` | 0 | 1.259e-10 | +0.003956 |
| `avg_pkts_lambda` | 8.983e-11 | 1.976e-10 | +0.001288 |
| `exp_max_factor` | 0 | 1.071e-10 | +0 |
| `delta` | 0.0007384 | 0.000782 | +0.0001798 |
| `slice_type` | 0 | 0.0005372 | -9.971e-05 |
| `model` | 0 | 1.132e-10 | +0 |

### Rankings

| Method | on 6 numeric features | on all 8 features |
|---|---|---|
| IG          | delta > packets > traffic > avg_pkts_lambda > eq_lambda > exp_max_factor | *(not applicable -- IG cannot rank categoricals)* |
| SHAP        | delta > packets > traffic > avg_pkts_lambda > eq_lambda > exp_max_factor | delta > slice_type > packets > traffic > avg_pkts_lambda > eq_lambda > model > exp_max_factor |
| Permutation | eq_lambda > traffic > avg_pkts_lambda > packets > delta > exp_max_factor | eq_lambda > traffic > avg_pkts_lambda > packets > delta > slice_type > exp_max_factor > model |

### Rank agreement (Spearman)

- **On 6 numeric features:**
  - IG vs SHAP: **+1.000**
  - IG vs Permutation: **-0.086**
  - SHAP vs Permutation: **-0.086**
- **On 8 features (SHAP + Permutation only):**
  - SHAP vs Permutation: **+0.190**

Top-3 by method: IG=['delta', 'packets', 'traffic'], SHAP-8=['delta', 'slice_type', 'packets'], Perm-8=['eq_lambda', 'traffic', 'avg_pkts_lambda']

## Interpretation notes for the thesis

1. **Rank correlation > 0.8 between two methods = strong agreement**; that is stronger evidence than any single method could give on its own. If all three agree, the ranking is robust to method choice.
2. **Permutation importance under-estimates correlated features.** In this dataset `slice_type` correlates with `packets` / `avg_pkts_lambda` at r ~ 0.94-0.99 on training. Shuffling `slice_type` alone leaves the model able to reconstruct much of the signal from `packets`, so permutation `slice_type` importance is a LOWER BOUND on the model's true dependence. SHAP with grouped categoricals does not have this pathology.
3. **`model` and `exp_max_factor` should be ~0 in every method.** If either is non-negligible in any method, that method has a bug/artifact.
4. **IG can only rank 6 features.** The `slice_type` and `model` rows in the IG column of the score tables are 0 by construction (documented scope choice), not by the method finding them unimportant.
