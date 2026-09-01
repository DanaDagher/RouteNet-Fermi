"""
Inspect the delay label distribution in the cached train and test splits.

Read-only. Does not modify any file, cache, checkpoint or model. Only reads the
pickled samples produced by preprocess_cache.py and reports:

  1. Percentile distribution of per-flow delays (p50, p90, p95, p99, p99.9, p99.99, max)
  2. How many individual flows exceed thresholds (100 ms, 1 s, 10 s, 100 s, 1000 s)
  3. How many whole samples contain at least one flow above each threshold
  4. Same breakdown per slice type (eMBB / mMTC / URLLC)

Delays are stored in the cache in seconds (the raw AvgDelay value from the
dataset), so threshold "1 s" is literally 1.0.

Usage:
    python slicing/inspect_outliers.py                    # both train and test
    python slicing/inspect_outliers.py --splits train     # one split only
"""
import argparse
import glob
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

SLICE_NAMES = {0: 'eMBB', 1: 'mMTC', 2: 'URLLC'}
THRESHOLDS_S = [0.1, 1.0, 10.0, 100.0, 1000.0]  # seconds


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cache-root', default=os.path.join(HERE, 'cache'))
    p.add_argument('--splits', nargs='+', default=['train', 'test'])
    return p.parse_args()


def scan_split(cache_dir):
    """Return (labels, slice_types, per_sample_max) as numpy arrays."""
    files = sorted(glob.glob(os.path.join(cache_dir, '*.pkl')))
    if not files:
        return None
    labels_all, slice_all, per_sample_max = [], [], []
    for i, fp in enumerate(files):
        with open(fp, 'rb') as fh:
            raw = pickle.load(fh)
        feats, labels = raw
        labels = np.asarray(labels, dtype=np.float64)
        slice_type = np.asarray(feats['slice_type'], dtype=np.int32)
        labels_all.append(labels)
        slice_all.append(slice_type)
        per_sample_max.append(float(np.max(labels)))
        if (i + 1) % 1000 == 0:
            print(f'  ... {i + 1} / {len(files)} files scanned')
    return (np.concatenate(labels_all),
            np.concatenate(slice_all),
            np.array(per_sample_max),
            len(files))


def report_split(split, labels, slice_types, per_sample_max, n_samples):
    n_flows = len(labels)
    print()
    print('=' * 74)
    print(f'  SPLIT: {split}   samples={n_samples}   flows={n_flows}')
    print('=' * 74)

    # 1. percentiles overall
    percentiles = [50, 90, 95, 99, 99.9, 99.99]
    pv = np.percentile(labels, percentiles)
    print('\n[1] Overall delay percentiles (seconds):')
    for p, v in zip(percentiles, pv):
        print(f'    p{p:<6}  {v:.6g}')
    print(f'    min     {np.min(labels):.6g}')
    print(f'    max     {np.max(labels):.6g}')
    print(f'    mean    {np.mean(labels):.6g}')

    # 2. flows above thresholds
    print('\n[2] Flows above threshold:')
    print(f'    {"threshold":>12}  {"n_flows":>10}  {"% of flows":>12}')
    for t in THRESHOLDS_S:
        n = int(np.sum(labels > t))
        print(f'    {t:>12}s  {n:>10}  {100 * n / n_flows:>11.4f}%')

    # 3. samples containing at least one flow above each threshold
    print('\n[3] Samples containing >=1 flow above threshold:')
    print(f'    {"threshold":>12}  {"n_samples":>10}  {"% of samples":>14}')
    for t in THRESHOLDS_S:
        n = int(np.sum(per_sample_max > t))
        print(f'    {t:>12}s  {n:>10}  {100 * n / n_samples:>13.2f}%')

    # 4. per slice breakdown
    print('\n[4] Per-slice percentiles and outliers:')
    for k in sorted(SLICE_NAMES):
        mask = slice_types == k
        c = int(mask.sum())
        if c == 0:
            print(f'    {SLICE_NAMES[k]:>5}: 0 flows')
            continue
        arr = labels[mask]
        pv_k = np.percentile(arr, [50, 99, 99.9, 99.99])
        n_over_1s = int(np.sum(arr > 1.0))
        n_over_10s = int(np.sum(arr > 10.0))
        print(f'    {SLICE_NAMES[k]:>5}  n={c:>7}  '
              f'p50={pv_k[0]:.4g}  p99={pv_k[1]:.4g}  '
              f'p99.9={pv_k[2]:.4g}  p99.99={pv_k[3]:.4g}  '
              f'max={np.max(arr):.4g}  '
              f'>1s={n_over_1s}  >10s={n_over_10s}')


def main():
    args = parse_args()

    for split in args.splits:
        cache_dir = os.path.join(args.cache_root, split)
        print(f'\nScanning {cache_dir} ...')
        result = scan_split(cache_dir)
        if result is None:
            print(f'  (no .pkl files in {cache_dir}, skipping)')
            continue
        labels, slice_types, per_sample_max, n_samples = result
        report_split(split, labels, slice_types, per_sample_max, n_samples)


if __name__ == '__main__':
    main()
