"""
Feature-level EDA on the 16 path_embedding inputs, read-only.

Closes gap 2 of METHODOLOGY_JUSTIFICATION.md: we audited the delay LABEL in
detail (inspect_outliers.py) but never did the same pass on the INPUT
features. This script does that, scanning the same train cache.

Reports, per numeric feature (traffic, packets, eq_lambda, avg_pkts_lambda,
exp_max_factor, delta, capacity, queue_size, weight):
  - percentiles (p1, p50, p90, p99, p99.9), min, max, mean, std
  - fraction of exactly-zero values
  - fraction of NaN / inf values (should be 0; flags a real bug if not)

And for the categorical features (model one-hot, slice_type, priority, policy):
  - value counts, to confirm which are constant (dead) vs actually varying

Read-only: only opens the existing .pkl cache files, writes nothing.

Usage:
    python slicing/inspect_features.py                 # train split
    python slicing/inspect_features.py --split test
"""
import argparse
import glob
import os
import pickle

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

NUMERIC_FEATS = ['traffic', 'packets', 'eq_lambda', 'avg_pkts_lambda',
                  'exp_max_factor', 'delta', 'capacity', 'queue_size', 'weight']
CATEGORICAL_FEATS = ['model', 'slice_type', 'priority', 'policy']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cache-root', default=os.path.join(HERE, 'cache'))
    p.add_argument('--split', default='train')
    return p.parse_args()


def main():
    args = parse_args()
    cache_dir = os.path.join(args.cache_root, args.split)
    files = sorted(glob.glob(os.path.join(cache_dir, '*.pkl')))
    print(f'Scanning {len(files)} files in {cache_dir} ...')

    numeric_accum = {k: [] for k in NUMERIC_FEATS}
    categorical_accum = {k: [] for k in CATEGORICAL_FEATS}

    for i, fp in enumerate(files):
        with open(fp, 'rb') as fh:
            raw = pickle.load(fh)
        feats, _ = raw
        for k in NUMERIC_FEATS:
            arr = np.asarray(feats[k], dtype=np.float64).reshape(-1)
            numeric_accum[k].append(arr)
        for k in CATEGORICAL_FEATS:
            arr = np.asarray(feats[k]).reshape(-1)
            categorical_accum[k].append(arr)
        if (i + 1) % 1000 == 0:
            print(f'  ... {i + 1} / {len(files)} files scanned')

    print(f'\nDone scanning {len(files)} files.\n')
    print('=' * 78)
    print('  NUMERIC FEATURES')
    print('=' * 78)
    for k in NUMERIC_FEATS:
        arr = np.concatenate(numeric_accum[k])
        n = len(arr)
        n_nan = int(np.sum(np.isnan(arr)))
        n_inf = int(np.sum(np.isinf(arr)))
        n_zero = int(np.sum(arr == 0))
        finite = arr[np.isfinite(arr)]
        pv = np.percentile(finite, [1, 50, 90, 99, 99.9]) if len(finite) else [float('nan')] * 5
        print(f'\n  {k}  (n={n})')
        print(f'    p1={pv[0]:.6g}  p50={pv[1]:.6g}  p90={pv[2]:.6g}  p99={pv[3]:.6g}  p99.9={pv[4]:.6g}')
        print(f'    min={np.min(finite):.6g}  max={np.max(finite):.6g}  mean={np.mean(finite):.6g}  std={np.std(finite):.6g}')
        print(f'    zero: {n_zero} ({100*n_zero/n:.3f}%)   NaN: {n_nan}   Inf: {n_inf}' +
              ('   <<< BUG: NaN/Inf present' if (n_nan or n_inf) else ''))

    print('\n' + '=' * 78)
    print('  CATEGORICAL / ONE-HOT-SOURCE FEATURES')
    print('=' * 78)
    for k in CATEGORICAL_FEATS:
        arr = np.concatenate(categorical_accum[k])
        n = len(arr)
        vals, counts = np.unique(arr, return_counts=True)
        print(f'\n  {k}  (n={n}, {len(vals)} distinct values)')
        order = np.argsort(-counts)
        for v, c in zip(vals[order][:10], counts[order][:10]):
            flag = '  <<< CONSTANT (dead feature)' if len(vals) == 1 else ''
            print(f'    value={v}   count={c}   ({100*c/n:.2f}%){flag}')
        if len(vals) > 10:
            print(f'    ... and {len(vals) - 10} more distinct values')


if __name__ == '__main__':
    main()
