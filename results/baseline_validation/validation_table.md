# Step 2.5 — Baseline Validation Table
**Date:** 2026-06-03  
**N simulations:** 300 per sub-dataset (`shuffle=False`, deterministic)  
**N flows per sub-dataset:** 81,600  
**Reference:** Ferriol-Galmés et al., RouteNet-Fermi, IEEE/ACM ToN 2023, DOI 10.1109/TNET.2023.3269983, Table V

| Sub-dataset | Paper Table V MAPE (delay) | Our MAPE N=300 (mean) | Our MAPE N=300 (median) | Delta (mean − paper) | Within ±1 pp? |
|---|---|---|---|---|---|
| `constant_bitrate` | 4.43% | 4.4699% | 1.1749% | +0.04 pp | ✅ |
| `onoff` | 2.90% | 2.8259% | 0.9967% | −0.07 pp | ✅ |
| `autocorrelated` | 2.62% | 2.5087% | 0.9490% | −0.11 pp | ✅ |
| `modulated` | 5.21% | 5.2262% | 3.7987% | +0.02 pp | ✅ |
| `all_multiplexed` | 4.71% | 4.6184% | 2.5418% | −0.09 pp | ✅ |

**Stop criterion (PIPELINE.md Step 2.5):** `all_multiplexed` mean MAPE within ±1 pp of 4.71% → 4.62% observed → **PASSED**.  
All 5 sub-datasets within ±0.12 pp of paper numbers → environment health confirmed.

**Verdict: N=300 is a valid and representative sample. Safe to proceed.**
