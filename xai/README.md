# xai/ — XAI Methods for RouteNet-Fermi

XAI scope: the **10 per-flow path scalars** only.
The 12 structural inputs (length, model, link/queue features, graph tensors)
are never perturbed or dropped in these modules. See THESIS_DECISIONS §5.

## Files

| File | Purpose |
|---|---|
| `training_stats.py` | Compute/cache training-set median per feature. Run once; saved to `results/training_stats.json`. |
| `integrated_gradients.py` | Integrated Gradients (Sundararajan 2017). 50 interpolation steps from training-set median to actual. Gradient of output[flow_idx=0] w.r.t. flow[0]'s path scalars. |
| `kernel_shap.py` | KernelSHAP (Lundberg & Lee 2017). Black-box wrapper: varies only flow[0]'s 10 path scalars. 256 perturbations, single background = training-set median. |
| `random_control.py` | Deterministic random ranking (seed=42). Negative control for Step 6/7 (core — Step 7 cells 8–13; carries the principled-vs-random fidelity result). |
| `tests/test_ig.py` | IG sanity checks (real-data + synthetic). |
| `tests/test_kernel_shap.py` | KernelSHAP sanity checks (real-data + synthetic). |

## Quick-start

```bash
# 1. Compute training-set medians (run once)
python xai/training_stats.py

# 2. Run sanity checks
python -m xai.tests.test_ig
python -m xai.tests.test_kernel_shap

# 3. Random ranking
python xai/random_control.py
```

## Design decisions

- **flow_idx=0**: consistent with CBR pilot. The traffic-model class at index 0
  varies across simulations on all_multiplexed; distribution is reported in
  Step 5 (PIPELINE.md).
- **Single background = training-set median**: consistent with CBR pilot.
  Rank-based aggregation reduces sensitivity to background choice.
- **Column dropping** (not value masking): variant generation in Step 6 drops
  feature columns entirely from the input dict and resizes `path_embedding`.
  These modules (IG, KernelSHAP) produce the rankings; the dropping happens
  on branch `xai-protocol-b` in Steps 6–7.
