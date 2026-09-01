"""
XAI phase - baseline statistics, replacing the all-zero IG baseline.

Why this exists: the old, validated all_multiplexed IG work (branch
xai-protocol-b, xai/integrated_gradients.py) used the TRAINING-SET MEDIAN as
its baseline, and deliberately never interpolated one-hot categorical inputs at
all -- only the continuous path scalars. Our all-zero baseline instead
interpolates the `model` and `slice_type` one-hot blocks too (we need
slice_type specifically, it's the central question of this phase), from an
all-zero point that represents an invalid "no category selected" state never
seen in real data. That is a plausible cause of the poor IG completeness
convergence measured on some flows (residual larger than the prediction
itself on one test flow) -- interpolating through an out-of-distribution
categorical state can add real curvature a straight-line integral struggles
to resolve in few steps.

Fix, grounded in the old work's actual principle (use real training
statistics, not an arbitrary point) rather than the letter of it (which
can't be copied -- we need slice_type attributed):
  - continuous scalars (traffic, packets, eq_lambda, avg_pkts_lambda,
    exp_max_factor, delta): baseline = TRAINING-SET MEDIAN, z-scored with the
    same z_score dict the model itself uses. Matches the old, validated choice.
  - one-hot blocks (model, slice_type): baseline = EMPIRICAL CLASS FREQUENCY
    over the training set (e.g. slice_type ~ [0.04, 0.75, 0.21]), a real,
    in-distribution "typical mixture" point -- not "no category", not one
    specific class.

Usage (DGX, run once, cheap -- pure numpy/pickle over the cache, no GPU):
    python slicing/xai/xai_baseline_stats.py \
        --train-cache slicing/cache/train --out slicing/xai/baseline_stats.json
"""
import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xai_common import FEATURE_NAMES  # noqa: E402


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    slic = os.path.dirname(here)
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-cache', default=os.path.join(slic, 'cache', 'train'))
    ap.add_argument('--out', default=os.path.join(here, 'baseline_stats.json'))
    return ap.parse_args()


def compute_stats(train_cache_dir):
    files = sorted(glob.glob(os.path.join(train_cache_dir, '*.pkl')))
    if not files:
        raise SystemExit(f"ERROR: no .pkl files in {train_cache_dir}")

    scalar_keys = ['traffic', 'packets', 'eq_lambda', 'avg_pkts_lambda',
                   'exp_max_factor', 'delta']
    collectors = {k: [] for k in scalar_keys}
    model_counts = np.zeros(7, dtype=np.int64)
    slice_counts = np.zeros(3, dtype=np.int64)
    n_flows = 0

    for i, fp in enumerate(files):
        with open(fp, 'rb') as fh:
            feats, _ = pickle.load(fh)
        for k in scalar_keys:
            collectors[k].extend(np.asarray(feats[k]).flatten().tolist())
        for m in feats['model']:
            model_counts[int(m)] += 1
        for s in feats['slice_type']:
            slice_counts[int(s)] += 1
        n_flows += len(feats['slice_type'])
        if (i + 1) % 500 == 0:
            print(f"  scanned {i+1}/{len(files)} files, {n_flows} flows so far")

    medians = {k: float(np.median(collectors[k])) for k in scalar_keys}
    means = {k: float(np.mean(collectors[k])) for k in scalar_keys}
    model_freq = (model_counts / model_counts.sum()).tolist()
    slice_freq = (slice_counts / slice_counts.sum()).tolist()

    print(f"\nScanned {len(files)} files, {n_flows} flows total.")
    print("Medians (raw units):")
    for k in scalar_keys:
        print(f"  {k:20s} median={medians[k]:.6g}  mean={means[k]:.6g}")
    print(f"model one-hot empirical frequency (7 classes): {model_freq}")
    print(f"slice_type empirical frequency [eMBB, mMTC, URLLC]: {slice_freq}")

    return {'n_files': len(files), 'n_flows': n_flows,
            'scalar_median': medians, 'scalar_mean': means,
            'model_freq': model_freq, 'slice_freq': slice_freq}


def build_baseline_row(stats, z_score):
    """Assemble the 16-dim baseline row in the exact FEATURE_NAMES column order
    used by xai_common.assemble_path_input. Scalars are z-scored with the
    model's own z_score dict (same normalization the model was trained under);
    one-hot blocks use empirical class frequency, not all-zero."""
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


def main():
    args = parse_args()
    stats = compute_stats(args.train_cache)

    # z_score is a fixed constant baked into delay_model.py -- import the model
    # class just to read it, no TF graph / GPU needed for this.
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    from delay_model import RouteNet_Fermi
    z = RouteNet_Fermi().z_score

    baseline_row = build_baseline_row(stats, z)
    stats['baseline_row'] = baseline_row.tolist()
    stats['feature_names'] = list(FEATURE_NAMES)

    print("\nFinal 16-dim baseline row (z-scored scalars, empirical-frequency one-hots):")
    for name, val in zip(FEATURE_NAMES, baseline_row):
        print(f"  {name:15s} {val: .4f}")

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(stats, fh, indent=1)
    print(f"\nSaved: {args.out}")


if __name__ == '__main__':
    main()
