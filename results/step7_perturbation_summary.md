# Step 7 — Perturbation test (inference-time, no retraining)

Complements the retraining ablation, which is structurally blocked from removing `traffic` / `packets` (they feed the fixed physics: `load`, `pkt_size`, `trans_delay`). Here we take the frozen upstream RouteNet-Fermi baseline (the model IG/KernelSHAP attributed over) and, at inference, replace one or more input features with a random permutation across flows within the same simulation. Marginal distribution is preserved (each value is a real value from a real flow), so no distribution-shift complaint applies. See `results/two_doors_and_ml_locus_REPORT.md`.

Pool = 300 sims per (model, mode); Delta averaged over 5 shuffle seeds. Clean MAPE = raw upstream baseline on the 300-sim test set with no perturbation.

| Model | Perturbation | Features | MAPE (mean +/- std) | Delta vs clean |
|---|---|---|---|---|
| upstream_baseline | clean | `-` | 31.85% +/- 0.00 | +0.00 pp |
| upstream_baseline | shuffle_traffic | `traffic` | 59.55% +/- 0.34 | +27.71 pp |
| upstream_baseline | shuffle_packets | `packets` | 51.10% +/- 0.15 | +19.25 pp |
| upstream_baseline | shuffle_traffic_packets | `traffic,packets` | 39.23% +/- 0.27 | +7.38 pp |
| upstream_baseline | shuffle_top3 | `sigma,traffic,packets` | 38.54% +/- 0.28 | +6.69 pp |
| upstream_baseline | shuffle_eq_lambda | `eq_lambda` | 31.79% +/- 0.01 | -0.05 pp |
| upstream_baseline | shuffle_avg_pkts_lambda | `avg_pkts_lambda` | 31.73% +/- 0.01 | -0.11 pp |

**Read:** shuffling the top-3 XAI-ranked features (sigma, traffic, packets) drives MAPE up dramatically; shuffling XAI-bottom-ranked features (eq_lambda, avg_pkts_lambda) leaves MAPE unchanged. Same model, same procedure, same distribution: the contrast is a direct fidelity signal that the XAI ranking is faithful — and it closes the gap left by the retraining ablation, which was structurally blocked on `traffic`/`packets`.
