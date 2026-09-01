"""
Look at WHAT is structurally different between the flows IG handles cleanly
and the flows IG fails on. Reads the numeric-scope pilot .npz and cross-
references each flow's actual raw cache to extract features that MIGHT
explain the split -- without changing anything.

Read-only. No IG re-run. No GPU.

Usage:
    python slicing/xai/xai_investigate_hard_flows.py \
        --npz slicing/xai/results/ig_numeric_pilot10.npz \
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
    ap.add_argument('--npz', required=True)
    ap.add_argument('--test-cache', default=os.path.join(slic, 'cache', 'test'))
    return ap.parse_args()


def load_flow_context(cache_dir, fname, flow_idx):
    """Return a dict of per-flow and per-topology structural info that MIGHT
    explain IG difficulty: flow's actual features, path length, number of
    queues it visits, topology size, delay ranges, etc."""
    with open(os.path.join(cache_dir, fname), 'rb') as fh:
        feats, labels = pickle.load(fh)
    n_flows = len(feats['slice_type'])
    n_links = len(feats['capacity'])
    n_queues = len(feats['queue_size'])

    # this flow's own values
    traffic = float(np.asarray(feats['traffic'])[flow_idx].item())
    packets = float(np.asarray(feats['packets'])[flow_idx].item())
    eq_lambda = float(np.asarray(feats['eq_lambda'])[flow_idx].item())
    delta = float(np.asarray(feats['delta'])[flow_idx].item())
    slice_t = int(feats['slice_type'][flow_idx])
    label = float(labels[flow_idx])

    # path length for this flow (from link_to_path)
    lp = feats['link_to_path']
    if isinstance(lp, list):
        path_len = len(lp[flow_idx])
    else:
        try:
            path_len = int(lp[flow_idx].shape[0])
        except Exception:
            path_len = -1
    qp = feats['queue_to_path']
    if isinstance(qp, list):
        n_q_on_path = len(qp[flow_idx])
    else:
        try:
            n_q_on_path = int(qp[flow_idx].shape[0])
        except Exception:
            n_q_on_path = -1

    # topology-wide delay stats (log-scale, since delays span orders of magnitude)
    lbl = np.asarray(labels, dtype=np.float64)
    lbl_pos = lbl[lbl > 0]
    log_lbl = np.log10(lbl_pos) if len(lbl_pos) else np.array([0.0])
    log_range = float(log_lbl.max() - log_lbl.min()) if len(log_lbl) else 0.0
    log_std = float(log_lbl.std()) if len(log_lbl) else 0.0
    max_delay = float(lbl.max())
    min_delay = float(lbl[lbl > 0].min()) if (lbl > 0).any() else 0.0

    # this flow's z-score-ish position within its topology (log scale)
    if len(log_lbl):
        this_log = float(np.log10(max(label, 1e-12)))
        pos_in_log = (this_log - log_lbl.mean()) / max(log_lbl.std(), 1e-9)
    else:
        pos_in_log = 0.0

    return {
        'n_flows_topology': n_flows,
        'n_links_topology': n_links,
        'n_queues_topology': n_queues,
        'path_len': path_len,
        'n_q_on_path': n_q_on_path,
        'traffic': traffic,
        'packets': packets,
        'eq_lambda': eq_lambda,
        'delta': delta,
        'slice': SLICE_NAMES[slice_t],
        'label_delay': label,
        'topology_log_delay_range_orders': log_range,
        'topology_log_delay_std': log_std,
        'topology_max_delay': max_delay,
        'topology_min_delay': min_delay,
        'this_flow_log_zscore_in_topology': pos_in_log,
    }


def main():
    args = parse_args()
    d = np.load(args.npz, allow_pickle=True)
    n = len(d['converged'])
    conv = d['converged']
    files = d['sample_file']
    flow_idxs = d['flow_idx']
    f_actual = d['f_actual']
    f_baseline = d['f_baseline']
    resid = np.abs(d['completeness_residual'])
    ig = d['ig']

    print(f"=== {n} flows, {int(conv.sum())} converged, {int((~conv).sum())} not ===\n")
    print("Loading per-flow topology context (may take a few seconds)...\n")

    ctxs = []
    for i in range(n):
        ctx = load_flow_context(args.test_cache, str(files[i]), int(flow_idxs[i]))
        ctx['_i'] = i + 1
        ctx['_converged'] = bool(conv[i])
        ctx['_resid'] = float(resid[i])
        ctx['_f_actual'] = float(f_actual[i])
        ctx['_f_baseline'] = float(f_baseline[i])
        ctx['_effect'] = float(abs(f_actual[i] - f_baseline[i]))
        ctx['_max_ig'] = float(np.max(np.abs(ig[i])))
        ctxs.append(ctx)

    # print side-by-side comparison
    keys = ['slice', 'label_delay', '_f_actual', '_f_baseline', '_effect',
            '_max_ig', '_resid', 'path_len', 'n_q_on_path',
            'n_flows_topology', 'topology_log_delay_range_orders',
            'topology_log_delay_std', 'topology_max_delay',
            'this_flow_log_zscore_in_topology', 'traffic', 'packets',
            'eq_lambda']

    print(f"{'#':>3} {'OK?':>4}  " + "  ".join(f"{k[:22]:>22}" for k in keys))
    for c in ctxs:
        tag = "OK" if c['_converged'] else "BAD"
        vals = []
        for k in keys:
            v = c[k]
            if isinstance(v, float):
                vals.append(f"{v:>22.4g}")
            else:
                vals.append(f"{str(v):>22}")
        print(f"{c['_i']:>3} {tag:>4}  " + "  ".join(vals))

    print("\n--- Grouped means (converged vs not) ---")
    numeric_keys = [k for k in keys if not isinstance(ctxs[0][k], str)]
    print(f"{'metric':40s} {'OK mean':>15} {'BAD mean':>15} {'ratio':>10}")
    for k in numeric_keys:
        ok_vals = [c[k] for c in ctxs if c['_converged']]
        bad_vals = [c[k] for c in ctxs if not c['_converged']]
        if not ok_vals or not bad_vals:
            continue
        ok_m = np.mean(ok_vals)
        bad_m = np.mean(bad_vals)
        ratio = bad_m / ok_m if abs(ok_m) > 1e-12 else float('nan')
        print(f"{k:40s} {ok_m:>15.4g} {bad_m:>15.4g} {ratio:>10.3g}")

    print("\nRead the ratio column. A ratio far from 1.0 = this metric "
          "differs strongly between converged and non-converged flows, "
          "and is a candidate structural explanation.")


if __name__ == '__main__':
    main()
