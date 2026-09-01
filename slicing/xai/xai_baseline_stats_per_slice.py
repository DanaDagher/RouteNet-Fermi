"""
XAI phase - per-slice baseline statistics (Option 1, step 1).

WHY THIS EXISTS
---------------
The single-baseline design in xai_baseline_stats.py uses training-median
numerics + empirical-frequency one-hots. The 300-flow diagnostic
(xai/results/audit_diagnostics.json) revealed that this single baseline is
NOT slice-neutral: on URLLC the model's baseline prediction is ~16x its
prediction on real URLLC flows and ~16x true URLLC delay, because the
training-median numeric row is dominated by mMTC (75% of training) and
gives the model an inconsistent input when paired with a URLLC one-hot.
Applying the same reference row across three very different slice regimes
biases the completeness residual and the effect-size denominator that
downstream IG convergence and stratification depend on.

FIX
---
Compute medians *within each slice* on the training set and produce a
baseline_row **per slice**. IG then picks the row matching the target flow's
slice, so a URLLC flow is compared against a URLLC-typical reference, an
eMBB flow against an eMBB-typical reference, and an mMTC flow against an
mMTC-typical reference. The one-hot for slice_type is still handled as
before (see below), so this does not create three "always this slice"
baselines that would defeat slice_type attribution: the numeric axes are
per-slice, the categorical axes follow the same rules as xai_baseline_stats.py.

OUTPUT SCHEMA (baseline_stats_per_slice.json)
---------------------------------------------
Top-level keys (backwards-compatible with the single-baseline consumer):
  n_files, n_flows                 # totals over the whole training scan
  scalar_median, scalar_mean       # global medians/means (unchanged)
  model_freq, slice_freq           # global class frequencies (unchanged)
  baseline_row                     # global 16-dim baseline row (unchanged)
  feature_names                    # same column order as xai_common.FEATURE_NAMES
NEW keys (add per-slice info without breaking old consumers):
  per_slice: {
    'eMBB' | 'mMTC' | 'URLLC': {
        n_flows                    # flows in this slice
        scalar_median              # dict, medians *within this slice*
        scalar_mean                # dict, means within this slice
        model_freq                 # class freq within this slice (7-vec)
        baseline_row               # 16-dim row: per-slice medians +
                                   #   model=per-slice model freq +
                                   #   slice_type=pinned to this slice's one-hot
    }
  }

BASELINE ROW COMPOSITION PER SLICE
----------------------------------
Column order matches xai_common.FEATURE_NAMES. For slice S in {eMBB, mMTC, URLLC}:
  traffic, packets, eq_lambda, avg_pkts_lambda, exp_max_factor, delta:
      z-scored median of that column *within slice S* on the training set.
  model one-hot (7 cols):
      empirical model-class frequency *within slice S*. Same "typical mixture"
      idea as the global baseline but conditioned on the slice, in case some
      traffic models are used more on some slices than others.
  slice_type one-hot (3 cols):
      pinned to the S row (e.g. eMBB -> [1, 0, 0]). This is a deliberate
      choice, NOT an accident: in scope='numeric' IG (the correct scope),
      the categorical columns of the baseline are IGNORED anyway because
      build_effective_baseline() pins them to the target flow's actual
      values. Pinning them here to slice S in the stored row is just so the
      row is self-consistent and readable, not so IG uses it.

CONSISTENCY WITH THE SINGLE-BASELINE FILE
-----------------------------------------
The top-level `baseline_row` in this output is byte-identical to what
xai_baseline_stats.py produces. This script is a strict superset. Callers
that only know the old schema keep working. New callers (xai_ig patch in
step 2) look for `per_slice[slice_name].baseline_row`.

Usage (DGX, cheap - no GPU, pure numpy over the cache):
    python slicing/xai/xai_baseline_stats_per_slice.py \
        --train-cache slicing/cache/train \
        --out slicing/xai/baseline_stats_per_slice.json
"""
import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xai_common import FEATURE_NAMES, SLICE_NAMES  # noqa: E402


SCALAR_KEYS = ['traffic', 'packets', 'eq_lambda', 'avg_pkts_lambda',
               'exp_max_factor', 'delta']


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    slic = os.path.dirname(here)
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-cache', default=os.path.join(slic, 'cache', 'train'))
    ap.add_argument('--out',
                    default=os.path.join(here, 'baseline_stats_per_slice.json'))
    return ap.parse_args()


def compute_stats(train_cache_dir):
    """Scan every training .pkl once. Bucket every flow's scalars and
    model-class by its slice_type. Return per-slice + global collectors.

    Memory note: same footprint as the single-baseline script -- one list of
    floats per scalar per slice. On the current training set (~44k flows,
    6 scalars, 3 slices) this is well under 100 MB."""
    files = sorted(glob.glob(os.path.join(train_cache_dir, '*.pkl')))
    if not files:
        raise SystemExit(f"ERROR: no .pkl files in {train_cache_dir}")

    # Per-slice collectors: {slice_int: {scalar_name: [values]}}
    per_slice_scalars = {s: {k: [] for k in SCALAR_KEYS} for s in (0, 1, 2)}
    per_slice_model_counts = {s: np.zeros(7, dtype=np.int64) for s in (0, 1, 2)}
    per_slice_n = {s: 0 for s in (0, 1, 2)}

    # Global collectors (identical to xai_baseline_stats.py output for
    # bit-compatibility of the top-level baseline_row).
    global_scalars = {k: [] for k in SCALAR_KEYS}
    global_model_counts = np.zeros(7, dtype=np.int64)
    global_slice_counts = np.zeros(3, dtype=np.int64)
    n_flows = 0

    for i, fp in enumerate(files):
        with open(fp, 'rb') as fh:
            feats, _ = pickle.load(fh)
        slice_arr = np.asarray(feats['slice_type']).astype(np.int64)
        model_arr = np.asarray(feats['model']).astype(np.int64)
        # per-scalar arrays should be same length as slice_arr (one value per flow)
        scalars_arr = {k: np.asarray(feats[k]).astype(np.float64).flatten()
                       for k in SCALAR_KEYS}
        # Sanity: all scalar arrays share length with slice_arr, otherwise
        # per-slice bucketing is undefined.
        for k, v in scalars_arr.items():
            if len(v) != len(slice_arr):
                raise SystemExit(
                    f"ERROR: {os.path.basename(fp)} scalar '{k}' length "
                    f"{len(v)} != slice_type length {len(slice_arr)}")
        if len(model_arr) != len(slice_arr):
            raise SystemExit(
                f"ERROR: {os.path.basename(fp)} model length "
                f"{len(model_arr)} != slice_type length {len(slice_arr)}")

        for s in (0, 1, 2):
            mask = (slice_arr == s)
            if not mask.any():
                continue
            for k in SCALAR_KEYS:
                per_slice_scalars[s][k].extend(scalars_arr[k][mask].tolist())
            for m in model_arr[mask]:
                per_slice_model_counts[s][int(m)] += 1
            per_slice_n[s] += int(mask.sum())

        # Global (unchanged behaviour).
        for k in SCALAR_KEYS:
            global_scalars[k].extend(scalars_arr[k].tolist())
        for m in model_arr:
            global_model_counts[int(m)] += 1
        for s in slice_arr:
            global_slice_counts[int(s)] += 1
        n_flows += len(slice_arr)

        if (i + 1) % 500 == 0:
            print(f"  scanned {i+1}/{len(files)} files, {n_flows} flows so far",
                  flush=True)

    # Reduce to summary stats.
    global_medians = {k: float(np.median(global_scalars[k])) for k in SCALAR_KEYS}
    global_means = {k: float(np.mean(global_scalars[k])) for k in SCALAR_KEYS}
    global_model_freq = (global_model_counts /
                         max(global_model_counts.sum(), 1)).tolist()
    global_slice_freq = (global_slice_counts /
                         max(global_slice_counts.sum(), 1)).tolist()

    per_slice_summary = {}
    for s in (0, 1, 2):
        name = SLICE_NAMES[s]
        n = per_slice_n[s]
        if n == 0:
            print(f"  WARNING: no training flows in slice {name} (s={s})")
            per_slice_summary[name] = {
                'n_flows': 0,
                'scalar_median': dict.fromkeys(SCALAR_KEYS, float('nan')),
                'scalar_mean': dict.fromkeys(SCALAR_KEYS, float('nan')),
                'model_freq': [float('nan')] * 7,
            }
            continue
        med = {k: float(np.median(per_slice_scalars[s][k])) for k in SCALAR_KEYS}
        mean = {k: float(np.mean(per_slice_scalars[s][k])) for k in SCALAR_KEYS}
        mfreq = (per_slice_model_counts[s] /
                 max(per_slice_model_counts[s].sum(), 1)).tolist()
        per_slice_summary[name] = {
            'n_flows': n,
            'scalar_median': med,
            'scalar_mean': mean,
            'model_freq': mfreq,
        }

    # Console summary so the log is self-documenting like the single-baseline
    # script's is.
    print(f"\nScanned {len(files)} files, {n_flows} flows total.")
    print("\nGLOBAL medians (raw units, same as xai_baseline_stats.py):")
    for k in SCALAR_KEYS:
        print(f"  {k:20s} median={global_medians[k]:.6g}  mean={global_means[k]:.6g}")
    print(f"model empirical freq (7 classes, global): {global_model_freq}")
    print(f"slice_type empirical freq [eMBB, mMTC, URLLC]: {global_slice_freq}")

    print("\nPER-SLICE medians (raw units, medians of that slice only):")
    for name in ('eMBB', 'mMTC', 'URLLC'):
        s = per_slice_summary[name]
        print(f"\n  {name}  (n={s['n_flows']} training flows)")
        for k in SCALAR_KEYS:
            m = s['scalar_median'][k]
            g = global_medians[k]
            # Print the ratio to the global median to make bias obvious.
            ratio = (m / g) if (g not in (0.0,) and not np.isnan(m)) else float('nan')
            print(f"    {k:20s} median={m:.6g}  (global median * {ratio:.3f})")

    return {
        'n_files': len(files),
        'n_flows': n_flows,
        'scalar_median': global_medians,
        'scalar_mean': global_means,
        'model_freq': global_model_freq,
        'slice_freq': global_slice_freq,
        'per_slice': per_slice_summary,
    }


def build_global_baseline_row(stats, z_score):
    """Reproduce xai_baseline_stats.build_baseline_row EXACTLY so the top-level
    baseline_row in this file is byte-identical to the single-baseline output."""
    row = np.zeros(16, dtype=np.float32)
    m = stats['scalar_median']
    row[0] = (m['traffic'] - z_score['traffic'][0]) / z_score['traffic'][1]
    row[1] = (m['packets'] - z_score['packets'][0]) / z_score['packets'][1]
    row[2:9] = stats['model_freq']
    row[9] = (m['eq_lambda'] - z_score['eq_lambda'][0]) / z_score['eq_lambda'][1]
    row[10] = (m['avg_pkts_lambda'] - z_score['avg_pkts_lambda'][0]) / z_score['avg_pkts_lambda'][1]
    row[11] = (m['exp_max_factor'] - z_score['exp_max_factor'][0]) / z_score['exp_max_factor'][1]
    row[12:15] = stats['slice_freq']
    row[15] = (m['delta'] - z_score['delta'][0]) / z_score['delta'][1]
    return row


def build_per_slice_baseline_row(slice_name, slice_stats, z_score):
    """16-dim baseline row for a specific slice:
      scalars = z-scored median WITHIN THAT SLICE
      model one-hot = empirical model frequency WITHIN THAT SLICE
      slice_type one-hot = pinned to slice_name (e.g. eMBB -> [1, 0, 0])
    See module docstring for why the slice one-hot is pinned (does not affect
    scope='numeric' IG; kept only so the stored row is self-consistent)."""
    row = np.zeros(16, dtype=np.float32)
    m = slice_stats['scalar_median']
    if any(np.isnan(v) for v in m.values()):
        return row  # empty slice, return zeros; caller must not use this
    row[0] = (m['traffic'] - z_score['traffic'][0]) / z_score['traffic'][1]
    row[1] = (m['packets'] - z_score['packets'][0]) / z_score['packets'][1]
    row[2:9] = slice_stats['model_freq']
    row[9] = (m['eq_lambda'] - z_score['eq_lambda'][0]) / z_score['eq_lambda'][1]
    row[10] = (m['avg_pkts_lambda'] - z_score['avg_pkts_lambda'][0]) / z_score['avg_pkts_lambda'][1]
    row[11] = (m['exp_max_factor'] - z_score['exp_max_factor'][0]) / z_score['exp_max_factor'][1]
    # slice_type one-hot: pinned to this slice.
    slice_onehot = {'eMBB': (1.0, 0.0, 0.0),
                    'mMTC': (0.0, 1.0, 0.0),
                    'URLLC': (0.0, 0.0, 1.0)}[slice_name]
    row[12], row[13], row[14] = slice_onehot
    row[15] = (m['delta'] - z_score['delta'][0]) / z_score['delta'][1]
    return row


def main():
    args = parse_args()
    stats = compute_stats(args.train_cache)

    # z_score is a fixed constant on the model class; importing does not
    # touch the GPU. Same pattern as xai_baseline_stats.py.
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    from delay_model import RouteNet_Fermi
    z = RouteNet_Fermi().z_score

    global_row = build_global_baseline_row(stats, z)
    stats['baseline_row'] = global_row.tolist()
    stats['feature_names'] = list(FEATURE_NAMES)

    print("\nGlobal 16-dim baseline row (unchanged from xai_baseline_stats.py):")
    for name, val in zip(FEATURE_NAMES, global_row):
        print(f"  {name:15s} {val: .4f}")

    print("\nPer-slice 16-dim baseline rows (NEW):")
    for slice_name in ('eMBB', 'mMTC', 'URLLC'):
        s = stats['per_slice'][slice_name]
        if s['n_flows'] == 0:
            print(f"\n  {slice_name}: no training flows, SKIPPING baseline row")
            s['baseline_row'] = None
            continue
        row = build_per_slice_baseline_row(slice_name, s, z)
        s['baseline_row'] = row.tolist()
        print(f"\n  {slice_name} (n={s['n_flows']}):")
        for name, val in zip(FEATURE_NAMES, row):
            print(f"    {name:15s} {val: .4f}")

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(stats, fh, indent=1)
    print(f"\nSaved: {args.out}")


if __name__ == '__main__':
    main()
