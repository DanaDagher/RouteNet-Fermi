# configs/

Per-variant JSON configs describing which of the 10 per-flow path-scalar features are kept
vs. dropped for each retrained model (Step 6/7). The 12 structural inputs are never dropped
and so don't appear in these configs.

| Dir | What it is |
|---|---|
| `baseline/full.json` | The baseline variant: all 10 path scalars kept. 1 model. |
| `ig/` | 6 variants (k ∈ {30,50,70} × {relevant, irrelevant}) using the IG global ranking. |
| `kernel_shap/` | Same 6 variants using the KernelSHAP global ranking. **Identical kept-feature sets to `ig/` at every (k, partition) cell** — this is the IG ≡ KernelSHAP convergence finding (see `THESIS_DECISIONS.md` §7), checked automatically by `audit_steps_1_to_6.py`. |
| `random/` | Same 6 variants using a random feature ranking, as the negative control that the fidelity headline (principled vs. random) is actually measured against. |

13 unique configs in total (`ig/` and `kernel_shap/` collapse to the same 6 underlying models,
so 1 + 6 + 6 = 13, not 19).
