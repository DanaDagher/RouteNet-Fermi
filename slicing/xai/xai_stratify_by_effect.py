"""
XAI phase - Section 14 stratification.

For each of the 300 flows in the shared set, compute the "effect size" =
|f_actual - f_baseline| where the baseline swap is IDENTICAL to what IG /
KernelSHAP will use (numeric-scope baseline: 6 numeric columns replaced with
training-set medians, categorical one-hots pinned per-flow at actuals).

WHY: the numeric-only IG pilot (§11 of XAI_PHASE_STATUS_REPORT.md) showed
that ~40% of flows sit near-baseline -- for those, the model output is
essentially unchanged whether the flow's numeric features are its real
values or the training median. XAI methods (IG, KernelSHAP, perturbation)
all have nothing meaningful to attribute for near-baseline flows, because
the model itself contributes ~0 above baseline. Running XAI on those flows
is wasted compute.

This script does two forward passes per flow (no gradients, no adaptive
stepping), computes the effect size, saves per-flow to JSON, and reports the
distribution so the split threshold can be picked from evidence.

Cost estimate: ~2 forward passes per flow. IG-with-gradients cost ~5s per
forward pass on the pilot; without gradients closer to 1-2s. Plus tf.function
retracing per unique topology shape. Realistic total: ~30-60 min for 300
flows, not hours.

Usage (DGX):
    CUDA_VISIBLE_DEVICES=2 python -u slicing/xai/xai_stratify_by_effect.py \
        --ckpt-dir slicing/ckpt_v2 --test-cache slicing/cache/test \
        --shared slicing/xai/sets/shared_300.json \
        --baseline-stats slicing/xai/baseline_stats.json \
        --out slicing/xai/results/stratification_300.json \
        2>&1 | tee slicing/xai/results/stratification_300.log
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xai_common import (SLICE_NAMES, load_model, load_raw_sample,
                        prepare_inputs)  # noqa: E402

# Column indices in the 16-dim path input (must match FEATURE_NAMES order in
# xai_common.py). Kept in sync with xai_ig.py -- if you change one, change both.
NUMERIC_COL_IDX = [0, 1, 9, 10, 11, 15]
CATEGORICAL_COL_IDX = [2, 3, 4, 5, 6, 7, 8, 12, 13, 14]


def build_effective_baseline(baseline_row_full, X_actual, flow_idx):
    """Same numeric-scope effective baseline as xai_ig.py --scope numeric:
    categorical one-hot columns pinned at per-flow actuals, numeric columns
    replaced with the training-median-based baseline_row_full values."""
    actual_row = tf.gather(X_actual, flow_idx)
    cat_idx = tf.constant(CATEGORICAL_COL_IDX, dtype=tf.int32)
    updates = tf.gather(actual_row, cat_idx)
    return tf.tensor_scatter_nd_update(baseline_row_full,
                                       tf.expand_dims(cat_idx, 1), updates)


@tf.function(reduce_retracing=True)
def _endpoint_outputs(model, inputs, X_actual, flow_idx, baseline_row):
    f_actual = tf.gather(model.forward_from_path_input(X_actual, inputs), flow_idx)
    X_base_full = tf.tensor_scatter_nd_update(X_actual, [[flow_idx]], [baseline_row])
    f_baseline = tf.gather(model.forward_from_path_input(X_base_full, inputs), flow_idx)
    return f_actual, f_baseline


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    slic = os.path.dirname(here)
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-dir', default=os.path.join(slic, 'ckpt_v2'))
    ap.add_argument('--test-cache', default=os.path.join(slic, 'cache', 'test'))
    ap.add_argument('--shared', default=os.path.join(here, 'sets', 'shared_300.json'))
    ap.add_argument('--baseline-stats', default=os.path.join(here, 'baseline_stats.json'))
    ap.add_argument('--out', default=os.path.join(here, 'results', 'stratification_300.json'))
    ap.add_argument('--limit', type=int, default=0,
                    help='process only the first N records (0 = all)')
    return ap.parse_args()


def main():
    args = parse_args()
    for _g in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(_g, True)
        except RuntimeError:
            pass

    with open(args.shared) as fh:
        records = json.load(fh)['records']
    if args.limit > 0:
        records = records[:args.limit]
    print(f"Loaded {len(records)} flows to stratify from {args.shared}", flush=True)

    with open(args.baseline_stats) as fh:
        baseline_row_full = tf.constant(
            json.load(fh)['baseline_row'], dtype=tf.float32)

    model, ckpt = load_model(args.ckpt_dir)
    print(f"Baseline: numeric-scope (training median for 6 scalars, "
          f"per-flow actuals for 10 categorical columns)", flush=True)

    per_flow = []
    t0 = time.time()
    cache = {}
    for i, rec in enumerate(records):
        fname, flow_idx = rec['file'], rec['flow_idx']
        if fname not in cache:
            feats_raw, _ = load_raw_sample(args.test_cache, fname)
            cache[fname] = feats_raw
            if len(cache) > 4:
                cache.pop(next(iter(cache)))
        feats_raw = cache[fname]
        inputs = prepare_inputs(feats_raw)
        X_actual = model.assemble_path_input(inputs)
        flow_idx_t = tf.constant(flow_idx, dtype=tf.int32)
        effective_baseline = build_effective_baseline(baseline_row_full, X_actual, flow_idx_t)

        f_a, f_b = _endpoint_outputs(model, inputs, X_actual, flow_idx_t, effective_baseline)
        f_a = float(f_a.numpy().item())
        f_b = float(f_b.numpy().item())
        effect = abs(f_a - f_b)

        per_flow.append({
            'file': fname, 'flow_idx': flow_idx,
            'slice_type': rec['slice_type'], 'slice_name': SLICE_NAMES[rec['slice_type']],
            'label_delay': rec.get('delay', None),
            'f_actual': f_a, 'f_baseline': f_b, 'effect': effect,
        })

        if (i + 1) % 10 == 0 or i == 0 or i == len(records) - 1:
            elapsed = time.time() - t0
            rate = elapsed / (i + 1)
            eta = rate * (len(records) - i - 1)
            print(f"  [{i+1}/{len(records)}] {rate:.2f}s/flow avg, "
                  f"ETA {eta/60:.1f} min, effect={effect:.4g}", flush=True)

    total_time = time.time() - t0
    print(f"\nDone: {len(per_flow)} flows in {total_time:.1f}s ({total_time/len(per_flow):.2f}s/flow avg)",
          flush=True)

    effects = np.array([r['effect'] for r in per_flow])
    slices = np.array([r['slice_type'] for r in per_flow])
    labels = np.array([r['label_delay'] for r in per_flow if r['label_delay'] is not None])

    print("\n=== Effect-size distribution (raw, model output units) ===")
    print(f"  min    : {np.min(effects):.4g}")
    print(f"  p10    : {np.percentile(effects, 10):.4g}")
    print(f"  p25    : {np.percentile(effects, 25):.4g}")
    print(f"  median : {np.median(effects):.4g}")
    print(f"  p75    : {np.percentile(effects, 75):.4g}")
    print(f"  p90    : {np.percentile(effects, 90):.4g}")
    print(f"  max    : {np.max(effects):.4g}")

    print("\n=== Log-scale distribution (log10 effect) ===")
    log_e = np.log10(np.maximum(effects, 1e-12))
    hist, edges = np.histogram(log_e, bins=20)
    for count, lo, hi in zip(hist, edges[:-1], edges[1:]):
        bar = '#' * min(count, 60)
        print(f"  10^{lo:+5.2f} .. 10^{hi:+5.2f}  ({int(count):>3})  {bar}")

    # Suggest candidate thresholds and report the split counts each gives.
    print("\n=== Split counts for candidate thresholds (per slice) ===")
    print(f"{'threshold':>12}  {'total_meaningful':>17} "
          + "".join(f"{n:>10}" for n in ['eMBB', 'mMTC', 'URLLC']))
    for thr_pct in (10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90):
        thr = float(np.percentile(effects, thr_pct))
        keep = effects >= thr
        per_slice = {k: int(keep[slices == k].sum()) for k in sorted(SLICE_NAMES)}
        n_meaningful = int(keep.sum())
        print(f"  p{thr_pct:2d}>={thr:.3g}  {n_meaningful:>17}  "
              + "".join(f"{per_slice[k]:>10}" for k in sorted(SLICE_NAMES)))

    payload = {
        'meta': {
            'ckpt': ckpt, 'n_flows': len(per_flow),
            'shared_set': args.shared, 'baseline_stats': args.baseline_stats,
            'scope': 'numeric (categorical one-hots pinned per-flow)',
        },
        'summary': {
            'effect_percentiles': {
                'p10': float(np.percentile(effects, 10)),
                'p25': float(np.percentile(effects, 25)),
                'median': float(np.median(effects)),
                'p75': float(np.percentile(effects, 75)),
                'p90': float(np.percentile(effects, 90)),
            },
        },
        'records': per_flow,
    }
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(payload, fh, indent=1)
    print(f"\nSaved: {args.out}")


if __name__ == '__main__':
    main()
