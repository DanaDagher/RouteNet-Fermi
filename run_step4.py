"""
run_step4.py — Step 4: Generate XAI explanations on N=300 all_multiplexed test sims.

Runs IG and KernelSHAP on the locked 300 simulations (shuffle=False, take(300))
using the upstream pretrained checkpoint ckpt_dir_all_multiplexed/48-4.53.

Outputs (all under results/inference/):
  explanation_set/indices.npy              -- sim indices 0..299
  results/inference/ig/sim_{i:04d}.npz    -- IG attribution dict per sim
  results/inference/ig/timings.csv        -- wall-clock per sim
  results/inference/kernel_shap/sim_{i:04d}.npz
  results/inference/kernel_shap/timings.csv

Each .npz contains:
  - ig_scores / shap_scores: array of 10 values (ordered as PATH_SCALAR_FEATURES)
  - feature_names: the 10 feature name strings

Usage:
    python run_step4.py [--method ig|kernel_shap|both] [--n_sims 300]

Run IG first (faster ~1-2 h CPU), then KernelSHAP (~3.5 h CPU).
"""

import os
import sys
import re
import csv
import time
import argparse

import numpy as np
import tensorflow as tf

REPO_ROOT  = os.path.dirname(os.path.abspath(__file__))
DELAY_DIR  = os.path.join(REPO_ROOT, 'traffic_models', 'delay')
sys.path.insert(0, DELAY_DIR)
sys.path.insert(0, REPO_ROOT)
os.chdir(DELAY_DIR)

tf.random.set_seed(42)
np.random.seed(42)

from data_generator import input_fn
from delay_model import RouteNet_Fermi
from xai.training_stats import PATH_SCALAR_FEATURES, load_training_medians
from xai.integrated_gradients import compute_ig, N_STEPS
from xai.kernel_shap import compute_kernel_shap, N_PERTURBATIONS

# ── config ─────────────────────────────────────────────────────────────────────
TM          = 'all_multiplexed'
CKPT_NAME   = '48-4.53'
N_SIMS      = 300
FLOW_IDX    = 0

TEST_PATH   = os.path.join(REPO_ROOT, 'data', 'traffic_models', TM, 'test')
CKPT_DIR    = os.path.join(DELAY_DIR, f'ckpt_dir_{TM}')
CKPT_PATH   = os.path.join(CKPT_DIR, CKPT_NAME)

EXPL_ROOT   = os.path.join(REPO_ROOT, 'results', 'inference')
EXPL_SET    = os.path.join(REPO_ROOT, 'explanation_set')

os.makedirs(os.path.join(EXPL_ROOT, 'ig'),           exist_ok=True)
os.makedirs(os.path.join(EXPL_ROOT, 'kernel_shap'),  exist_ok=True)
os.makedirs(EXPL_SET,                                 exist_ok=True)


def load_model():
    model = RouteNet_Fermi()
    model.compile(
        loss=tf.keras.losses.MeanAbsolutePercentageError(),
        optimizer=tf.keras.optimizers.Adam(0.001),
        run_eagerly=False,
    )
    model.load_weights(CKPT_PATH)
    print(f'Loaded checkpoint: {CKPT_PATH}')
    return model


def run_ig(model, ds, medians):
    print(f'\n[Step 4 — IG] Running on {N_SIMS} simulations (n_steps={N_STEPS})...')
    timing_rows = [['sim_idx', 'n_flows', 'wall_clock_s']]

    for i, (x_batch, _) in enumerate(ds):
        t0       = time.time()
        scores   = compute_ig(model, x_batch, medians,
                              n_steps=N_STEPS, flow_idx=FLOW_IDX)
        elapsed  = time.time() - t0
        n_flows  = int(x_batch['traffic'].shape[0])

        scores_arr = np.array([scores[f] for f in PATH_SCALAR_FEATURES],
                               dtype=np.float32)
        out_path = os.path.join(EXPL_ROOT, 'ig', f'sim_{i:04d}.npz')
        np.savez(out_path,
                 ig_scores=scores_arr,
                 feature_names=np.array(PATH_SCALAR_FEATURES))
        timing_rows.append([i, n_flows, round(elapsed, 3)])

        if (i + 1) % 25 == 0 or i == 0:
            avg = np.mean([r[2] for r in timing_rows[1:]])
            eta = avg * (N_SIMS - i - 1) / 60
            print(f'  IG sim {i+1:3d}/{N_SIMS}  '
                  f'n_flows={n_flows}  t={elapsed:.1f}s  '
                  f'avg={avg:.1f}s  ETA~{eta:.0f}min')

    timing_path = os.path.join(EXPL_ROOT, 'ig', 'timings.csv')
    with open(timing_path, 'w', newline='') as fh:
        csv.writer(fh).writerows(timing_rows)
    print(f'IG done. Timings saved to {timing_path}')


def run_kernel_shap(model, ds, medians):
    print(f'\n[Step 4 — KernelSHAP] Running on {N_SIMS} sims '
          f'(nsamples={N_PERTURBATIONS})...')
    timing_rows = [['sim_idx', 'n_flows', 'wall_clock_s']]

    for i, (x_batch, _) in enumerate(ds):
        t0      = time.time()
        scores  = compute_kernel_shap(model, x_batch, medians,
                                      n_perturbations=N_PERTURBATIONS,
                                      flow_idx=FLOW_IDX)
        elapsed = time.time() - t0
        n_flows = int(x_batch['traffic'].shape[0])

        scores_arr = np.array([scores[f] for f in PATH_SCALAR_FEATURES],
                               dtype=np.float32)
        out_path = os.path.join(EXPL_ROOT, 'kernel_shap', f'sim_{i:04d}.npz')
        np.savez(out_path,
                 shap_scores=scores_arr,
                 feature_names=np.array(PATH_SCALAR_FEATURES))
        timing_rows.append([i, n_flows, round(elapsed, 3)])

        if (i + 1) % 10 == 0 or i == 0:
            avg = np.mean([r[2] for r in timing_rows[1:]])
            eta = avg * (N_SIMS - i - 1) / 60
            print(f'  SHAP sim {i+1:3d}/{N_SIMS}  '
                  f'n_flows={n_flows}  t={elapsed:.1f}s  '
                  f'avg={avg:.1f}s  ETA~{eta:.0f}min')

    timing_path = os.path.join(EXPL_ROOT, 'kernel_shap', 'timings.csv')
    with open(timing_path, 'w', newline='') as fh:
        csv.writer(fh).writerows(timing_rows)
    print(f'KernelSHAP done. Timings saved to {timing_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['ig', 'kernel_shap', 'both'],
                        default='both')
    parser.add_argument('--n_sims', type=int, default=N_SIMS)
    args = parser.parse_args()

    medians = load_training_medians()
    model   = load_model()

    # The locked 300 simulations: shuffle=False, same order every run
    ds = (input_fn(TEST_PATH, shuffle=False)
          .take(args.n_sims)
          .prefetch(tf.data.experimental.AUTOTUNE))

    # Save simulation indices (0 to n_sims-1, deterministic order)
    indices = np.arange(args.n_sims, dtype=np.int32)
    np.save(os.path.join(EXPL_SET, 'indices.npy'), indices)
    print(f'Saved explanation set indices: {indices[0]}..{indices[-1]}')

    if args.method in ('ig', 'both'):
        run_ig(model, ds, medians)

    if args.method in ('kernel_shap', 'both'):
        # Re-create dataset iterator for KernelSHAP (same 300 sims)
        ds_shap = (input_fn(TEST_PATH, shuffle=False)
                   .take(args.n_sims)
                   .prefetch(tf.data.experimental.AUTOTUNE))
        run_kernel_shap(model, ds_shap, medians)

    print('\nStep 4 complete.')


if __name__ == '__main__':
    main()
