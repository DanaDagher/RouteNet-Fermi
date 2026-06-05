# Step 1 — Environment Setup Report
**Date:** 2026-06-03  
**Branch:** xai-features  
**Status:** ✅ COMPLETE

---

## Final Environment

| Package | Version | Source |
|---|---|---|
| Python | 3.7.9 | `py -3.7` (AppData/Local/Python) |
| TensorFlow | 2.6.5 | `requirements.txt` pin |
| numpy | 1.19.5 | TF 2.6 dependency |
| pandas | 1.1.5 | `requirements.txt` pin |
| networkx | 2.6.3 | `requirements.txt` pin |
| shap | 0.42.1 | Step 1 extra |
| scikit-learn | 1.0.2 | Step 1 extra |
| matplotlib | 3.5.x | Step 1 extra |
| seaborn | latest py37-compat | Step 1 extra |
| tqdm | latest py37-compat | Step 1 extra |

---

## Data & Checkpoints

| Check | Status |
|---|---|
| `data/traffic_models/constant_bitrate/{train,test}` | ✅ present |
| `data/traffic_models/onoff/{train,test}` | ✅ present |
| `data/traffic_models/autocorrelated/{train,test}` | ✅ present |
| `data/traffic_models/modulated/{train,test}` | ✅ present |
| `data/traffic_models/all_multiplexed/{train,test}` | ✅ present |
| `traffic_models/delay/ckpt_dir_constant_bitrate/` | ✅ present (best: 45-4.29) |
| `traffic_models/delay/ckpt_dir_onoff/` | ✅ present (best: 44-2.74) |
| `traffic_models/delay/ckpt_dir_autocorrelated/` | ✅ present (best: 50-2.46) |
| `traffic_models/delay/ckpt_dir_modulated/` | ✅ present (best: 48-5.26) |
| `traffic_models/delay/ckpt_dir_all_multiplexed/` | ✅ present (best: 48-4.53) |

---

## Smoke Test Results (N=300 per sub-dataset)

| Sub-dataset | Checkpoint | N flows | MAPE mean | Paper MAPE | Delta | Pass? |
|---|---|---|---|---|---|---|
| `constant_bitrate` | 45-4.29 | 81,600 | 4.47% | 4.43% | +0.04 pp | ✅ |
| `onoff` | 44-2.74 | 81,600 | 2.83% | 2.90% | −0.07 pp | ✅ |
| `autocorrelated` | 50-2.46 | 81,600 | 2.51% | 2.62% | −0.11 pp | ✅ |
| `modulated` | 48-5.26 | 81,600 | 5.23% | 5.21% | +0.02 pp | ✅ |
| `all_multiplexed` | 48-4.53 | 81,600 | 4.62% | 4.71% | −0.09 pp | ✅ |

All deltas within ±0.12 pp. Stop criterion met.

---

## Errors Encountered and How They Were Fixed

### Error 1 — Wrong requirements.txt read
**What happened:** The `requirements.txt` in working tree showed `tensorflow>=2.11,<3.0` (a loose bound someone had tried). This caused all downstream version choices to be wrong.  
**Fix:** Re-read the file and found the actual pinned spec: `tensorflow==2.6.*, networkx==2.6.*, pandas==1.1.*`. Rebuilt venv from scratch.

### Error 2 — Python version mismatch
**What happened:** The existing venv used Python 3.11. TF 2.6 has no official wheel for Python 3.11.  
**Fix:** Identified Python 3.7.9 available via `py -3.7`. Recreated venv with `py -3.7 -m venv venv --clear`. TF 2.6.5 wheel for Python 3.7 on Windows exists on PyPI.

### Error 3 — numpy/pandas binary incompatibility (during wrong-version attempt)
**What happened:** With the wrong TF 2.21 stack, `shap 0.51.0` required `numpy>=2`, but `pandas 1.5.3` needed `numpy<2`. Every pip install cycle undid the previous fix.  
**Root cause:** This entire problem was downstream of Error 1 (wrong TF version). Resolved entirely by the correct venv rebuild.

### Error 4 — TF 2.21 DLL crash on Windows
**What happened:** TF 2.21 dropped Windows native pip support (starting TF 2.19, Windows requires WSL2). The wheel installed without error but failed at runtime with `DLL load failed: INITIALIZATION FAILED`.  
**Root cause:** Again downstream of Error 1. Resolved by correct venv rebuild.

### Error 5 — TF 2.13 GRU behavioural incompatibility (during wrong-version attempt)
**What happened:** After downgrading to TF 2.13 (still wrong), the model produced ~30–41% MAPE instead of ~4.7%. Weights loaded correctly (confirmed by manual tensor comparison) but GRU cell variable scoping changed in TF 2.13, causing the forward pass to compute incorrectly with the pre-2.13 checkpoints.  
**Root cause:** Again downstream of Error 1. Resolved by correct venv rebuild.

### Error 6 — Optimizer warnings on checkpoint load (harmless)
**What happened:** `WARNING: Unresolved object in checkpoint: (root).optimizer.iter` etc. printed on every model load.  
**Explanation:** The upstream checkpoints were saved with optimizer state (Adam momentum/velocity). We only need model weights for inference. TF correctly loads the weights and warns that optimizer slots were not consumed. This is expected and harmless for inference-only use. No fix needed.

---

## Notes for All Future Steps

- Always activate venv: `venv\Scripts\activate` (Python 3.7.9, TF 2.6.5)
- Run scripts from `traffic_models/delay/` directory (data_generator.py uses relative `sys.path.append('../')` for datanetAPI)
- Optimizer warnings on `model.load_weights()` are harmless — model weights load correctly
- `cudart64_110.dll not found` warning is harmless — no GPU, CPU-only inference
- Random seed: `tf.random.set_seed(42)` + `np.random.seed(42)` at top of every script
