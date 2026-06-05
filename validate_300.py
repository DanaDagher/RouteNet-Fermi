"""
Step 2.5 — Sample validity check on all 5 traffic_models/delay sub-datasets.

For each sub-dataset:
  - Loads the upstream pretrained checkpoint (best by val MAPE in filename).
  - Runs inference on ds_test.take(300), shuffle=False (deterministic, same 300 every run).
  - Computes mean MAPE and median MAPE across all flows in those 300 simulations.
  - Saves results/baseline_validation/mape_{tm}.json.

Builds and prints the comparison table vs. RouteNet-Fermi paper Table V (delay).

Usage:
    cd C:/Users/ddagher/RouteNet-Fermi
    venv/Scripts/python validate_300.py
"""

import os
import sys
import re
import json

import numpy as np
import tensorflow as tf

# ── paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT   = os.path.dirname(os.path.abspath(__file__))
DELAY_DIR   = os.path.join(REPO_ROOT, 'traffic_models', 'delay')
DATA_ROOT   = os.path.join(REPO_ROOT, 'data', 'traffic_models')
OUT_DIR     = os.path.join(REPO_ROOT, 'results', 'baseline_validation')

sys.path.insert(0, DELAY_DIR)
sys.path.insert(0, REPO_ROOT)
os.chdir(DELAY_DIR)   # data_generator.py uses sys.path.append('../') for datanetAPI

from data_generator import input_fn
from delay_model import RouteNet_Fermi

# ── reproducibility ────────────────────────────────────────────────────────────
tf.random.set_seed(42)
np.random.seed(42)

# ── paper Table V reference numbers (delay MAPE) ───────────────────────────────
# Ferriol-Galmés et al., IEEE/ACM ToN 2023, Table V, traffic_models experiment
PAPER_MAPE = {
    'constant_bitrate': 4.43,   # "Deterministic / CBR"
    'onoff':            2.90,   # "On-Off"
    'autocorrelated':   2.62,   # "A.Exponentials"
    'modulated':        5.21,   # "M.Exponentials"
    'all_multiplexed':  4.71,   # "Mixed"
}

N_SIMS = 300
os.makedirs(OUT_DIR, exist_ok=True)

results = {}

for tm in ['constant_bitrate', 'onoff', 'autocorrelated', 'modulated', 'all_multiplexed']:
    print(f'\n[{tm}] loading checkpoint...')
    test_path = os.path.join(DATA_ROOT, tm, 'test')
    ckpt_dir  = os.path.join(DELAY_DIR, f'ckpt_dir_{tm}')

    # pick best checkpoint by lowest val-MAPE in filename
    best = None
    best_mre = float('inf')
    for f in os.listdir(ckpt_dir):
        if os.path.isfile(os.path.join(ckpt_dir, f)):
            reg = re.findall(r'\d+\.\d+', f)
            if reg:
                mre = float(reg[0])
                if mre <= best_mre:
                    best = (f.replace('.index', '')
                             .replace('.data', '')
                             .replace('-00000-of-00001', ''))
                    best_mre = mre
    print(f'  checkpoint: {best}  (val MAPE {best_mre}%)')

    model = RouteNet_Fermi()
    model.compile(
        loss=tf.keras.losses.MeanAbsolutePercentageError(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        run_eagerly=False,
    )
    model.load_weights(os.path.join(ckpt_dir, best))

    ds_test = (input_fn(test_path, shuffle=False)
               .take(N_SIMS)
               .prefetch(tf.data.experimental.AUTOTUNE))

    preds_all  = []
    labels_all = []
    for x_batch, y_batch in ds_test:
        p = model(x_batch, training=False).numpy().flatten()
        l = y_batch.numpy().flatten()
        preds_all.append(p)
        labels_all.append(l)

    preds_all  = np.concatenate(preds_all)
    labels_all = np.concatenate(labels_all)
    n_flows    = len(labels_all)

    per_flow_ape = (np.abs(preds_all - labels_all)
                    / (np.abs(labels_all) + 1e-9) * 100)
    mape_mean   = float(np.mean(per_flow_ape))
    mape_median = float(np.median(per_flow_ape))
    delta       = round(mape_mean - PAPER_MAPE[tm], 4)

    results[tm] = {
        'tm':             tm,
        'checkpoint':     best,
        'n_simulations':  N_SIMS,
        'n_flows_total':  n_flows,
        'mape_mean':      round(mape_mean,   4),
        'mape_median':    round(mape_median, 4),
        'paper_mape':     PAPER_MAPE[tm],
        'delta':          delta,
    }

    out_path = os.path.join(OUT_DIR, f'mape_{tm}.json')
    with open(out_path, 'w') as fh:
        json.dump(results[tm], fh, indent=2)
    print(f'  n_flows={n_flows}  MAPE mean={mape_mean:.4f}%  median={mape_median:.4f}%'
          f'  paper={PAPER_MAPE[tm]}%  delta={delta:+.4f} pp  -> saved {out_path}')

    tf.keras.backend.clear_session()

# ── comparison table ───────────────────────────────────────────────────────────
print('\n' + '='*85)
print('STEP 2.5 — Validation Table: N=300 simulations vs. RouteNet-Fermi paper Table V')
print('='*85)
print(f'{"Sub-dataset":<22} {"Paper label":<22} {"Paper MAPE":>10} {"Ours mean":>10}'
      f' {"Ours median":>12} {"Delta":>8} {"±1pp?":>6}')
print('-'*85)

PAPER_LABELS = {
    'constant_bitrate': 'Deterministic/CBR',
    'onoff':            'On-Off',
    'autocorrelated':   'A.Exponentials',
    'modulated':        'M.Exponentials',
    'all_multiplexed':  'Mixed',
}

all_pass = True
for tm, r in results.items():
    ok = abs(r['delta']) <= 1.0
    if not ok:
        all_pass = False
    flag = '✓' if ok else '✗ FAIL'
    print(f'{tm:<22} {PAPER_LABELS[tm]:<22} {r["paper_mape"]:>9.2f}%'
          f' {r["mape_mean"]:>9.4f}%  {r["mape_median"]:>11.4f}%'
          f' {r["delta"]:>+7.4f}  {flag}')

print('-'*85)
print(f'\nNote: paper also reports Poisson 2.10% — no matching sub-dataset in local data.')
print(f'Stop criterion (all_multiplexed within ±1 pp of 4.71%): '
      f'{"PASSED" if abs(results["all_multiplexed"]["delta"]) <= 1.0 else "FAILED"}')
print(f'Overall: {"ALL PASS" if all_pass else "SOME FAILURES"}')
