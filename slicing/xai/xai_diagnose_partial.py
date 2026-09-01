"""
Diagnose why some flows fail to converge in the IG run.

Reads the partial .npz saved by xai_ig.py and looks for patterns: is
non-convergence correlated with slice type, delay magnitude, or effect size
(|f_actual - f_baseline|)?

Also cross-references the .pkl files each bad flow came from to check
topology size (number of flows in the sample), since large topologies were
one of the hypotheses for why the DGX-scale dataset is slower/harder to
attribute than the paper's reference topologies.

Usage:
    python slicing/xai/xai_diagnose_partial.py \
        --partial slicing/xai/results/ig_shared_300_partial.npz \
        --test-cache slicing/cache/test
"""
import argparse
import os
import pickle
import sys

import numpy as np

SLICE_NAMES = {0: 'eMBB', 1: 'mMTC', 2: 'URLLC'}


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    slic = os.path.dirname(here)
    ap = argparse.ArgumentParser()
    ap.add_argument('--partial', required=True)
    ap.add_argument('--test-cache', default=os.path.join(slic, 'cache', 'test'))
    return ap.parse_args()


def main():
    args = parse_args()
    d = np.load(args.partial, allow_pickle=True)
    n = int(d['n_done'])
    conv = d['converged'][:n]
    slice_type = d['slice_type'][:n]
    f_actual = d['f_actual'][:n]
    f_baseline = d['f_baseline'][:n]
    resid = np.abs(d['completeness_residual'][:n])
    steps_used = d['steps_used'][:n]
    files = d['sample_file'][:n]
    flow_idx = d['flow_idx'][:n]

    effect = np.abs(f_actual - f_baseline)
    rel_resid = resid / np.maximum(effect, 1e-9)

    print(f"=== Partial results: {n} flows done ===")
    print(f"Converged (residual under 5% of effect): {conv.sum()}/{n} ({100*conv.mean():.1f}%)")
    print(f"\n--- Per-flow detail ---")
    print(f"{'#':>3} {'file':40s} {'flow':>4} {'slice':>5} {'f_actual':>10} "
          f"{'f_base':>10} {'effect':>10} {'resid':>10} {'rel%':>7} "
          f"{'steps':>5}  status")
    for i in range(n):
        status = "OK" if conv[i] else "BAD"
        fn = files[i][-40:] if len(files[i]) > 40 else files[i]
        print(f"{i+1:>3} {fn:40s} {int(flow_idx[i]):>4} "
              f"{SLICE_NAMES[int(slice_type[i])]:>5} "
              f"{f_actual[i]:>10.4g} {f_baseline[i]:>10.4g} "
              f"{effect[i]:>10.4g} {resid[i]:>10.4g} "
              f"{rel_resid[i]*100:>6.1f}% {int(steps_used[i]):>5}  {status}")

    print(f"\n--- Convergence vs slice type ---")
    for s in sorted(SLICE_NAMES):
        mask = slice_type == s
        if not mask.any():
            continue
        n_s = int(mask.sum())
        n_conv = int(conv[mask].sum())
        print(f"  {SLICE_NAMES[s]:>5}: {n_conv}/{n_s} converged "
              f"({100*n_conv/n_s:.0f}%)")

    print(f"\n--- Convergence vs effect size |f_actual - f_baseline| ---")
    # split into small vs large effect at the median
    thr = np.median(effect)
    small = effect < thr
    large = effect >= thr
    print(f"  effect < median ({thr:.4g}): {int(conv[small].sum())}/{int(small.sum())} converged")
    print(f"  effect >= median: {int(conv[large].sum())}/{int(large.sum())} converged")

    print(f"\n--- Bad flows: cross-check topology size ---")
    bad_i = np.where(~conv)[0]
    if len(bad_i) == 0:
        print("  (none bad)")
    else:
        print(f"{'#':>3} {'file':40s} {'n_flows_in_topology':>20} {'rel_resid%':>10}")
        for i in bad_i:
            fp = os.path.join(args.test_cache, str(files[i]))
            try:
                with open(fp, 'rb') as fh:
                    feats, _ = pickle.load(fh)
                n_flows = len(feats['slice_type'])
            except Exception as e:
                n_flows = f"ERR: {e}"
            fn = str(files[i])[-40:] if len(str(files[i])) > 40 else str(files[i])
            print(f"{i+1:>3} {fn:40s} {n_flows:>20} {rel_resid[i]*100:>9.1f}%")

    print(f"\n--- Attribution magnitudes vs residual (per BAD flow) ---")
    print("If |residual| is much smaller than max|IG value|, the attribution "
          "ranking is still trustworthy even if the 'converged' flag is False.")
    ig = d['ig'][:n]
    for i in bad_i:
        max_abs_ig = float(np.max(np.abs(ig[i])))
        sum_abs_ig = float(np.sum(np.abs(ig[i])))
        ratio = resid[i] / max(max_abs_ig, 1e-12)
        print(f"  flow {i+1}: |resid|={resid[i]:.4g}, max|IG|={max_abs_ig:.4g}, "
              f"sum|IG|={sum_abs_ig:.4g}, resid/max_ig={ratio*100:.1f}%")

    print(f"\n--- Good flows: same topology-size check for contrast ---")
    good_i = np.where(conv)[0]
    print(f"{'#':>3} {'file':40s} {'n_flows_in_topology':>20} {'rel_resid%':>10}")
    for i in good_i:
        fp = os.path.join(args.test_cache, str(files[i]))
        try:
            with open(fp, 'rb') as fh:
                feats, _ = pickle.load(fh)
            n_flows = len(feats['slice_type'])
        except Exception as e:
            n_flows = f"ERR: {e}"
        fn = str(files[i])[-40:] if len(str(files[i])) > 40 else str(files[i])
        print(f"{i+1:>3} {fn:40s} {n_flows:>20} {rel_resid[i]*100:>9.1f}%")


if __name__ == '__main__':
    main()
