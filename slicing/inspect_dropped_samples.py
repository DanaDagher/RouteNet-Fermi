"""
Inspect the 13 training samples dropped by V2's max_delay_cap=1000 filter,
read-only. Closes gap 3 of METHODOLOGY_JUSTIFICATION.md: we assumed these are
simulation artifacts (a queue that never drained) but never actually looked
at what is happening in each one.

For every cached train sample with at least one flow whose delay exceeds the
given threshold (default 1000s, matching V2's filter), prints:
  - filename
  - total flows in the sample
  - how many flows exceed the threshold
  - the single worst flow's delay, traffic, packets, slice_type, delta
  - whether the "bad" flow(s) look isolated (one freak flow in an otherwise
    normal sample) or the whole sample is uniformly extreme (suggesting a
    genuinely saturated/collapsed topology rather than a single artifact)

Read-only: only opens existing .pkl cache files, writes nothing, does not
touch the raw dataset, the cache, or any checkpoint.

Usage:
    python slicing/inspect_dropped_samples.py                  # threshold 1000s
    python slicing/inspect_dropped_samples.py --threshold 100  # matches V1-mod's filter
"""
import argparse
import glob
import os
import pickle

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SLICE_NAMES = {0: 'eMBB', 1: 'mMTC', 2: 'URLLC'}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cache-dir', default=os.path.join(HERE, 'cache', 'train'))
    p.add_argument('--threshold', type=float, default=1000.0)
    return p.parse_args()


def main():
    args = parse_args()
    files = sorted(glob.glob(os.path.join(args.cache_dir, '*.pkl')))
    print(f'Scanning {len(files)} files in {args.cache_dir} for delay > {args.threshold}s ...\n')

    found = 0
    for fp in files:
        with open(fp, 'rb') as fh:
            raw = pickle.load(fh)
        feats, labels = raw
        labels = np.asarray(labels, dtype=np.float64)
        mask = labels > args.threshold
        n_bad = int(mask.sum())
        if n_bad == 0:
            continue

        found += 1
        n_total = len(labels)
        worst_idx = int(np.argmax(labels))

        traffic = np.asarray(feats['traffic']).reshape(-1)
        packets = np.asarray(feats['packets']).reshape(-1)
        slice_type = np.asarray(feats['slice_type']).reshape(-1)
        delta = np.asarray(feats['delta']).reshape(-1)

        print(f'--- {os.path.basename(fp)} ---')
        print(f'  total flows in sample     : {n_total}')
        print(f'  flows above {args.threshold}s          : {n_bad} ({100*n_bad/n_total:.2f}%)')
        print(f'  worst flow delay          : {labels[worst_idx]:.2f}s')
        print(f'  worst flow traffic        : {traffic[worst_idx]:.4g}')
        print(f'  worst flow packets        : {packets[worst_idx]:.4g}')
        print(f'  worst flow slice_type     : {SLICE_NAMES.get(int(slice_type[worst_idx]), "?")}')
        print(f'  worst flow delta          : {delta[worst_idx]:.4g}')

        # Distinguish "one freak flow" from "whole sample is saturated"
        other_delays = labels[~mask]
        if len(other_delays) > 0:
            print(f'  other flows in sample     : p50={np.percentile(other_delays,50):.4g}s  '
                  f'p99={np.percentile(other_delays,99):.4g}s  max={np.max(other_delays):.4g}s')
            verdict = ('ISOLATED SPIKE (other flows look normal)'
                       if np.percentile(other_delays, 99) < 1.0
                       else 'SAMPLE-WIDE CONGESTION (other flows also elevated)')
        else:
            verdict = 'ALL flows in this sample are above threshold'
        print(f'  verdict                   : {verdict}\n')

    print(f'Total samples with delay > {args.threshold}s: {found}')


if __name__ == '__main__':
    main()
