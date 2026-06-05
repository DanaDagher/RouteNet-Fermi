# Step 2.5 — Sample Validity Check Report
**Date:** 2026-06-03
**Branch:** xai-features
**Status:** COMPLETE — PASSED
**Script:** `validate_300.py` (repo root)

---

## Objective

Reproduce the RouteNet-Fermi paper's MAPE numbers on N=300 simulations across all 5
traffic_models/delay sub-datasets, in order to:

1. Confirm the local environment matches the paper's reported performance.
2. Validate that N=300 simulations is a representative and stable sample before
   committing it as the fixed evaluation budget for all XAI steps.

Reference: Ferriol-Galmes et al., "RouteNet-Fermi: Network Modeling with Graph Neural
Networks", IEEE/ACM Transactions on Networking 2023, DOI 10.1109/TNET.2023.3269983,
Table V (delay MAPE, traffic_models experiment).

---

## Method

- Sub-datasets: constant_bitrate, onoff, autocorrelated, modulated, all_multiplexed
- For each sub-dataset:
  - Loaded the upstream pretrained checkpoint with lowest val-MAPE in filename
    (same checkpoints shipped with the BNN-UPC repo, used in the paper).
  - Called `input_fn(test_path, shuffle=False).take(300)` — deterministic, same 300
    simulations on every run (no shuffling, fixed order from disk).
  - Collected all per-flow predicted delay and true delay values across the 300 sims.
  - Computed mean MAPE and median MAPE over all flows:
      per_flow_APE = |pred - label| / (|label| + 1e-9) * 100
      mape_mean   = mean(per_flow_APE)
      mape_median = median(per_flow_APE)
- Stop criterion (PIPELINE.md Step 2.5): all_multiplexed mean MAPE within +/-1 pp
  of paper's 4.71%.

---

## Results

N=300 simulations per sub-dataset => 81,600 flows per sub-dataset (272 flows/sim avg).

| Sub-dataset      | Paper label       | Checkpoint | Paper MAPE | Ours mean | Ours median | Delta    |
|------------------|-------------------|------------|------------|-----------|-------------|----------|
| constant_bitrate | Deterministic/CBR | 45-4.29    |     4.43 % |   4.4699% |     1.1749% | +0.04 pp |
| onoff            | On-Off            | 44-2.74    |     2.90 % |   2.8259% |     0.9967% | -0.07 pp |
| autocorrelated   | A.Exponentials    | 50-2.46    |     2.62 % |   2.5087% |     0.9490% | -0.11 pp |
| modulated        | M.Exponentials    | 48-5.26    |     5.21 % |   5.2262% |     3.7987% | +0.02 pp |
| all_multiplexed  | Mixed             | 48-4.53    |     4.71 % |   4.6184% |     2.5418% | -0.09 pp |

Note: the paper also reports Poisson 2.10%. There is no Poisson sub-dataset in the
local BNN-UPC dataset-v6-traffic-models copy — this row is not evaluated here.

Stop criterion: all_multiplexed delta = -0.09 pp (within +/-1 pp). PASSED.
All 5 sub-datasets: max absolute delta = 0.11 pp. ALL PASS.

---

## Interpretation

### Are N=300 simulations representative?

Yes. The tightest delta across all 5 sub-datasets is 0.11 pp against the paper's
full-test-set numbers. This tells us two things:

1. The environment is correctly set up (TF 2.6.5, Python 3.7.9, same checkpoints).
2. The first 300 simulations drawn with shuffle=False track the full test distribution
   to within noise — no need to increase the budget.

### Mean vs median MAPE

The mean MAPE is what the paper reports and is what the Keras MAPE loss computes
during training. The median is reported as a secondary indicator of robustness.

For constant_bitrate, the median (1.17%) is much lower than the mean (4.47%),
indicating a heavy right tail — a small fraction of flows with very small delays
produce large relative errors, inflating the mean. This is a known property of MAPE
on near-zero targets and is consistent with the paper's methodology.

For all_multiplexed the median is 2.54% vs mean 4.62%, showing the same pattern but
less extreme, which is expected for the mixed-traffic regime.

### Implications for XAI steps

- The N=300 budget is confirmed for both XAI explanation generation (Step 4) and
  for fidelity evaluation (Steps 7-8).
- The same shuffle=False, take(300) call must be used in every downstream step to
  guarantee the identical 300 simulations are used throughout.
- all_multiplexed is the only sub-dataset that continues past this step (per
  THESIS_DECISIONS.md §4 and PIPELINE.md). The other 4 sub-datasets are health-check
  only and are now closed.

---

## Files Produced

| File | Description |
|------|-------------|
| `results/baseline_validation/mape_constant_bitrate.json` | Per-sub-dataset result JSON |
| `results/baseline_validation/mape_onoff.json`            | Per-sub-dataset result JSON |
| `results/baseline_validation/mape_autocorrelated.json`   | Per-sub-dataset result JSON |
| `results/baseline_validation/mape_modulated.json`        | Per-sub-dataset result JSON |
| `results/baseline_validation/mape_all_multiplexed.json`  | Per-sub-dataset result JSON |
| `results/baseline_validation/validation_table.md`        | Markdown comparison table   |
| `validate_300.py`                                        | Rerunnable validation script |

JSON schema per file:
{
  "tm":            "<sub-dataset name>",
  "checkpoint":    "<epoch>-<val_mape>",
  "n_simulations": 300,
  "n_flows_total": 81600,
  "mape_mean":     <float, %>
  "mape_median":   <float, %>
  "paper_mape":    <float, %>
  "delta":         <float, pp = mape_mean - paper_mape>
}

---

## Verdict

N=300 is valid and representative. The environment reproduces the paper's numbers to
within +/-0.12 pp on all 5 sub-datasets. Step 2.5 is complete.

Next step: Step 3.5 (pre-registration writeup) is unblocked.
Steps 3, 4, 5, 6, 7, 8 remain blocked pending Mouna's protocol decision.
