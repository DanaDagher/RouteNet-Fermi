# rankings/

Global feature-importance rankings aggregated from the per-simulation explanations in
`results/inference/` (Step 5).

| File | What it is |
|---|---|
| `ig.csv`, `kernel_shap.csv` | Global ranking of the 10 path-scalar features by each method, aggregated over the N=300 explanation set. |
| `random.csv` | The random-ranking negative control used for the principled-vs-random fidelity comparison. |
| `ig_stability.csv`, `kernel_shap_stability.csv` | Half-split stability check: ranking computed on two random halves of the explanation set, compared via Spearman correlation. |
| `halfsplit_check.json` | Summary stats (Spearman ρ) from the half-split stability check. |
| `flow0_class_distribution.json` | Sanity-check distribution used in one of the audit checks. |
