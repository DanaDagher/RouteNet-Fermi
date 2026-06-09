"""
run_step4_stability.py — Step 4 stability runs (THESIS_DECISIONS §10 / PIPELINE Step 4).

Generates the SECOND attribution pass for IG and KernelSHAP on the SAME locked
300 all_multiplexed simulations (shuffle=False, take(300), same indices as the
main Step 4 run), but with a deliberately different reference point, so that
Step 5 can compute a stability Spearman between the two rankings per method.

Per THESIS_DECISIONS §10 ("Secondary — Stability"):
  - IG: training-set median baseline (main run) vs. "a meaningfully different
    alternative (training-set mean, or uniform-random baseline)".
        -> This run uses the TRAINING-SET MEAN as the alternate IG baseline.
           Reuses load_training_means() (already cached in
           results/training_stats.json), mirrors the main-run code path
           exactly (compute_ig() takes a `medians`-shaped reference dict —
           passing the means dict in its place is sufficient and requires
           no change to xai/integrated_gradients.py).
  - KernelSHAP: "two different background subsamples of training data".
        -> This run keeps the single-reference background design used in the
           main run (THESIS_DECISIONS §8 / consistent with the CBR pilot —
           shap.KernelExplainer with a (1, 10) background), but swaps the
           reference POINT from the training-set MEDIAN to the training-set
           MEAN. This is a genuinely different background while preserving
           the single-reference architecture, and it pairs naturally with
           the IG median->mean contrast above for a clean cross-method read.
           (Reuses compute_kernel_shap(), passing `means` in place of
           `medians` — no change to xai/kernel_shap.py needed, since the
           function already takes a generic per-feature reference dict.)

Both choices were the lower-risk / lower-new-surface-area options: they reuse
already-cached training statistics (median AND mean both precomputed in
results/training_stats.json by xai/training_stats.py) and the exact same
compute_ig / compute_kernel_shap call signatures as the main Step 4 run —
only the reference dict passed in changes from `medians` to `means`.

NOTE: if Dana prefers a uniform-random IG baseline or a true multi-point
random-subsample SHAP background instead, those are documented as the
alternative options and this script can be adjusted before the run is
considered final — see the two recommendation blocks above. Flag for review
in the Step 4 stability report.

Outputs (mirrors the main run's layout under results/inference/):
  results/inference/ig_stability/sim_{i:04d}.npz       -- IG attribution dict per sim (mean baseline)
  results/inference/ig_stability/timings.csv
  results/inference/kernel_shap_stability/sim_{i:04d}.npz  -- SHAP attribution dict per sim (mean background)
  results/inference/kernel_shap_stability/timings.csv

Each .npz contains the same schema as the main run:
  - ig_scores / shap_scores: array of 10 values (ordered as PATH_SCALAR_FEATURES)
  - feature_names: the 10 feature name strings
  - reference: the string 'mean' (so downstream code can tell stability vs main apart)

Usage:
    python run_step4_stability.py [--method ig|kernel_shap|both] [--n_sims 300]
"""

import os
import sys
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
from xai.training_stats import (PATH_SCALAR_FEATURES,
                                 load_training_medians, load_training_means)
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
IG_STAB_DIR   = os.path.join(EXPL_ROOT, 'ig_stability')
SHAP_STAB_DIR = os.path.join(EXPL_ROOT, 'kernel_shap_stability')

os.makedirs(IG_STAB_DIR,   exist_ok=True)
os.makedirs(SHAP_STAB_DIR, exist_ok=True)


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


def run_ig_stability(model, ds, means):
    """IG stability run: same 300 sims, baseline = training-set MEAN
    (vs. the main run's training-set MEDIAN)."""
    print(f'\n[Step 4 stability — IG] alt-baseline=mean, '
          f'{N_SIMS} sims, n_steps={N_STEPS}...')
    timing_rows = [['sim_idx', 'n_flows', 'wall_clock_s']]

    for i, (x_batch, _) in enumerate(ds):
        t0       = time.time()
        scores   = compute_ig(model, x_batch, means,
                              n_steps=N_STEPS, flow_idx=FLOW_IDX)
        elapsed  = time.time() - t0
        n_flows  = int(x_batch['traffic'].shape[0])

        scores_arr = np.array([scores[f] for f in PATH_SCALAR_FEATURES],
                               dtype=np.float32)
        out_path = os.path.join(IG_STAB_DIR, f'sim_{i:04d}.npz')
        np.savez(out_path,
                 ig_scores=scores_arr,
                 feature_names=np.array(PATH_SCALAR_FEATURES),
                 reference=np.array('mean'))
        timing_rows.append([i, n_flows, round(elapsed, 3)])

        if (i + 1) % 25 == 0 or i == 0:
            avg = np.mean([r[2] for r in timing_rows[1:]])
            eta = avg * (N_SIMS - i - 1) / 60
            print(f'  IG-stab sim {i+1:3d}/{N_SIMS}  '
                  f'n_flows={n_flows}  t={elapsed:.1f}s  '
                  f'avg={avg:.1f}s  ETA~{eta:.0f}min')

    timing_path = os.path.join(IG_STAB_DIR, 'timings.csv')
    with open(timing_path, 'w', newline='') as fh:
        csv.writer(fh).writerows(timing_rows)
    print(f'IG stability run done. Timings saved to {timing_path}')


def run_kernel_shap_stability(model, ds, means):
    """KernelSHAP stability run: same 300 sims, single-reference background
    = training-set MEAN point (vs. the main run's training-set MEDIAN point).
    Architecture (single-reference, 256 perturbations) unchanged — only the
    reference point differs, per THESIS_DECISIONS §8/§10 trade-off discussed
    in the module docstring above."""
    print(f'\n[Step 4 stability — KernelSHAP] alt-background=mean point, '
          f'{N_SIMS} sims (nsamples={N_PERTURBATIONS})...')
    timing_rows = [['sim_idx', 'n_flows', 'wall_clock_s']]

    for i, (x_batch, _) in enumerate(ds):
        t0      = time.time()
        scores  = compute_kernel_shap(model, x_batch, means,
                                      n_perturbations=N_PERTURBATIONS,
                                      flow_idx=FLOW_IDX)
        elapsed = time.time() - t0
        n_flows = int(x_batch['traffic'].shape[0])

        scores_arr = np.array([scores[f] for f in PATH_SCALAR_FEATURES],
                               dtype=np.float32)
        out_path = os.path.join(SHAP_STAB_DIR, f'sim_{i:04d}.npz')
        np.savez(out_path,
                 shap_scores=scores_arr,
                 feature_names=np.array(PATH_SCALAR_FEATURES),
                 reference=np.array('mean'))
        timing_rows.append([i, n_flows, round(elapsed, 3)])

        if (i + 1) % 10 == 0 or i == 0:
            avg = np.mean([r[2] for r in timing_rows[1:]])
            eta = avg * (N_SIMS - i - 1) / 60
            print(f'  SHAP-stab sim {i+1:3d}/{N_SIMS}  '
                  f'n_flows={n_flows}  t={elapsed:.1f}s  '
                  f'avg={avg:.1f}s  ETA~{eta:.0f}min')

    timing_path = os.path.join(SHAP_STAB_DIR, 'timings.csv')
    with open(timing_path, 'w', newline='') as fh:
        csv.writer(fh).writerows(timing_rows)
    print(f'KernelSHAP stability run done. Timings saved to {timing_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['ig', 'kernel_shap', 'both'],
                        default='both')
    parser.add_argument('--n_sims', type=int, default=N_SIMS)
    args = parser.parse_args()

    medians = load_training_medians()   # loaded for reference/sanity only
    means   = load_training_means()     # the alternate reference used here
    model   = load_model()

    print(f'Main-run reference (median) sample: traffic={medians["traffic"]:.4f}')
    print(f'Stability-run reference (mean)  sample: traffic={means["traffic"]:.4f}')

    # The SAME locked 300 simulations as the main Step 4 run:
    # shuffle=False, take(n_sims) -> identical deterministic order/indices
    # (see explanation_set/indices.npy, 0..299).
    ds = (input_fn(TEST_PATH, shuffle=False)
          .take(args.n_sims)
          .prefetch(tf.data.experimental.AUTOTUNE))

    if args.method in ('ig', 'both'):
        run_ig_stability(model, ds, means)

    if args.method in ('kernel_shap', 'both'):
        # Re-create dataset iterator (same 300 sims, same deterministic order)
        ds_shap = (input_fn(TEST_PATH, shuffle=False)
                   .take(args.n_sims)
                   .prefetch(tf.data.experimental.AUTOTUNE))
        run_kernel_shap_stability(model, ds_shap, means)

    print('\nStep 4 stability runs complete.')


if __name__ == '__main__':
    main()
