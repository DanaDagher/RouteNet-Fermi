"""
XAI phase - audit diagnostics (no gradients, no attribution).

Produces every measurement the skeptical audit
(slicing/XAI_PHASE_SKEPTICAL_AUDIT.md, PART 8 and PART 10 Step 0) says
must exist BEFORE any more IG or KernelSHAP runs. Two forward passes per
topology in the shared set:
  1. The real inputs -> f_actual per flow.
  2. Each target flow's path row replaced by the baseline row
     -> f_baseline per flow.
Plus ground-truth labels from the cache. No gradients, no adaptive
loops. Intended to run in minutes on the DGX.

Outputs (into --out-dir, default slicing/xai/results):
  audit_diagnostics.npz  raw per-flow arrays
  audit_diagnostics.json summary distributions and cross-tabs

What it computes (per flow, in the .npz):
  y_true, f_actual, f_baseline, |f_actual - y_true|,
  |f_actual - f_baseline|, slice_type, sample_file, flow_idx,
  path_len, n_flows_topology, delta, traffic, packets, eq_lambda,
  avg_pkts_lambda.

What it computes (in the .json):
  - per-slice percentiles (10/25/50/75/90) of y_true, f_actual,
    f_baseline, |f_actual - y_true|, |f_actual - f_baseline|
  - the "Type 1 / Type 2 / Type 3" typology from the audit PART 3,
    counted per slice
  - if --train-cache is passed and --n-train-samples > 0, also compute
    an EMPIRICAL correlation table between slice_type (as int and
    per-column one-hot) and each of the 6 numeric features across a
    random subset of training flows -- one row per flow, sampled from
    up to --n-train-samples training cache files. Pearson AND Spearman.
  - sanity checks: verify exp_max_factor and the model one-hot are
    constant across the sampled data (they should be, per
    METHODOLOGY_JUSTIFICATION.md sec 5).

Nothing here calls or requires xai_ig.py / KernelSHAP / permutation. This
script does not depend on any IG results existing.

Usage (DGX, one command):
  CUDA_VISIBLE_DEVICES=2 python slicing/xai/xai_audit_diagnostics.py \
      --ckpt-dir slicing/ckpt_v2 \
      --test-cache slicing/cache/test \
      --shared slicing/xai/sets/shared_300.json \
      --train-cache slicing/cache/train \
      --n-train-samples 200 \
      --out-dir slicing/xai/results

If --shared points at shared_900.json instead, the whole 900 is used
(still fast: 2 forward passes per topology, and shared sets pick one
flow per topology so ~cache-size forward passes total).
"""
import argparse
import json
import os
import pickle
import random
import sys
import time
from collections import defaultdict

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xai_common import (FEATURE_NAMES, SLICE_NAMES, load_model, load_raw_sample,
                        prepare_inputs)  # noqa: E402

NUMERIC_COL_IDX = [0, 1, 9, 10, 11, 15]
CATEGORICAL_COL_IDX = [2, 3, 4, 5, 6, 7, 8, 12, 13, 14]

NUMERIC_KEYS = ['traffic', 'packets', 'eq_lambda', 'avg_pkts_lambda',
                'exp_max_factor', 'delta']


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    slic = os.path.dirname(here)
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-dir', default=os.path.join(slic, 'ckpt_v2'))
    ap.add_argument('--test-cache', default=os.path.join(slic, 'cache', 'test'))
    ap.add_argument('--shared', default=os.path.join(here, 'sets', 'shared_300.json'))
    ap.add_argument('--baseline-stats', default=os.path.join(here, 'baseline_stats.json'))
    ap.add_argument('--out-dir', default=os.path.join(here, 'results'))
    ap.add_argument('--tag', default='audit_diagnostics')
    ap.add_argument('--train-cache', default='',
                    help='if provided, compute slice_type <-> numeric feature '
                         'correlations from a random subset of training flows')
    ap.add_argument('--n-train-samples', type=int, default=200,
                    help='number of training cache files to sample for the '
                         'correlation table (each contributes all its flows)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--limit', type=int, default=0)
    return ap.parse_args()


def _percentiles(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {p: None for p in (10, 25, 50, 75, 90)}
    return {p: float(np.percentile(x, p)) for p in (10, 25, 50, 75, 90)}


def _typology(f_actual, f_baseline, y_true,
              effect_low_pct=25.0, err_high_frac=0.5):
    """Return per-flow tag: 'type1_typical', 'type2_model_collapsed',
    'type3_high_effect', or 'ambiguous'. Thresholds are declared here so
    the script does NOT touch attribution numbers to pick them.

    Rule:
      effect = |f_actual - f_baseline|
      err    = |f_actual - y_true|
      low_effect = effect < percentile(effect, effect_low_pct)
      high_err   = err / max(|y_true|, eps) > err_high_frac
      Type 1: low_effect AND NOT high_err  -> typical, model matches truth near baseline
      Type 2: low_effect AND high_err      -> collapse
      Type 3: NOT low_effect               -> the model is "using" its features
      Ambiguous otherwise (should be empty by construction)."""
    effect = np.abs(f_actual - f_baseline)
    err = np.abs(f_actual - y_true)
    denom = np.maximum(np.abs(y_true), 1e-12)
    err_frac = err / denom
    thr_effect = np.percentile(effect, effect_low_pct)
    low_effect = effect < thr_effect
    high_err = err_frac > err_high_frac
    tag = np.array(['type3_high_effect'] * len(effect), dtype=object)
    tag[low_effect & ~high_err] = 'type1_typical'
    tag[low_effect & high_err] = 'type2_model_collapsed'
    return tag, {'threshold_effect_pctile': float(effect_low_pct),
                 'threshold_effect_value': float(thr_effect),
                 'threshold_err_frac': float(err_high_frac)}


def _corr(a, b, kind='pearson'):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    if kind == 'pearson':
        return float(np.corrcoef(a, b)[0, 1])
    # Spearman via rank
    ar = np.argsort(np.argsort(a))
    br = np.argsort(np.argsort(b))
    return float(np.corrcoef(ar, br)[0, 1])


def _collect_train_flow_features(train_cache_dir, n_files, seed):
    """Sample n_files random training cache files, gather per-flow
    (slice_type, traffic, packets, eq_lambda, avg_pkts_lambda,
    exp_max_factor, delta). Returns dict of 1D arrays."""
    files = [f for f in os.listdir(train_cache_dir) if not f.startswith('.')]
    random.Random(seed).shuffle(files)
    files = files[:max(1, n_files)]
    out = defaultdict(list)
    for fname in files:
        try:
            with open(os.path.join(train_cache_dir, fname), 'rb') as fh:
                feats, _ = pickle.load(fh)
        except Exception:
            continue
        st = np.asarray(feats['slice_type'], dtype=np.int32)
        out['slice_type'].append(st)
        for k in NUMERIC_KEYS:
            out[k].append(np.asarray(feats[k], dtype=np.float64).reshape(-1))
        try:
            out['model'].append(np.asarray(feats['model'], dtype=np.int32).reshape(-1))
        except Exception:
            pass
    out2 = {k: np.concatenate(v) for k, v in out.items() if len(v)}
    return out2


def main():
    args = parse_args()
    for _g in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(_g, True)
        except RuntimeError:
            pass

    # 1. Load the shared set (any of the JSONs in xai/sets/ will do).
    with open(args.shared) as fh:
        payload = json.load(fh)
    records = payload['records']
    if args.limit > 0:
        records = records[:args.limit]
    print(f"Loaded {len(records)} flows to inspect from {args.shared}", flush=True)

    # 2. Load model + baseline row.
    model, ckpt = load_model(args.ckpt_dir)
    if os.path.exists(args.baseline_stats):
        with open(args.baseline_stats) as fh:
            bstats = json.load(fh)
        baseline_row = tf.constant(bstats['baseline_row'], dtype=tf.float32)
        print(f"Baseline: loaded from {args.baseline_stats}", flush=True)
    else:
        baseline_row = tf.zeros(16, dtype=tf.float32)
        print(f"WARNING: {args.baseline_stats} missing, using all-zero baseline.",
              flush=True)

    # 3. Per-flow forward passes.
    n = len(records)
    y_true = np.zeros(n, dtype=np.float32)
    f_actual = np.zeros(n, dtype=np.float32)
    f_baseline = np.zeros(n, dtype=np.float32)
    slice_arr = np.zeros(n, dtype=np.int32)
    path_len = np.zeros(n, dtype=np.int32)
    n_flows_topo = np.zeros(n, dtype=np.int32)
    numeric_vals = {k: np.zeros(n, dtype=np.float64) for k in NUMERIC_KEYS}

    cache = {}
    t0 = time.time()
    for i, rec in enumerate(records):
        fname, flow_idx = rec['file'], rec['flow_idx']
        if fname not in cache:
            feats_raw, labels = load_raw_sample(args.test_cache, fname)
            cache[fname] = (feats_raw, labels)
            if len(cache) > 4:
                cache.pop(next(iter(cache)))
        feats_raw, labels = cache[fname]
        inputs = prepare_inputs(feats_raw)
        X = model.assemble_path_input(inputs)

        # actual + baseline forward passes
        y_actual_all = model.forward_from_path_input(X, inputs).numpy()
        # baseline-replace ONLY the target flow's row
        cat_idx = tf.constant(CATEGORICAL_COL_IDX, dtype=tf.int32)
        actual_row = tf.gather(X, flow_idx)
        cat_actual = tf.gather(actual_row, cat_idx)
        # Effective baseline: pin categoricals at actuals (matches xai_ig.py 'numeric' scope)
        eff_baseline = tf.tensor_scatter_nd_update(
            baseline_row, tf.expand_dims(cat_idx, 1), cat_actual)
        X_base = tf.tensor_scatter_nd_update(X, [[flow_idx]], [eff_baseline])
        y_base_all = model.forward_from_path_input(X_base, inputs).numpy()

        f_actual[i] = float(y_actual_all[flow_idx])
        f_baseline[i] = float(y_base_all[flow_idx])
        y_true[i] = float(labels[flow_idx])
        slice_arr[i] = int(feats_raw['slice_type'][flow_idx])
        for k in NUMERIC_KEYS:
            numeric_vals[k][i] = float(np.asarray(feats_raw[k]).reshape(-1)[flow_idx])
        n_flows_topo[i] = int(len(feats_raw['slice_type']))
        lp = feats_raw['link_to_path']
        try:
            path_len[i] = int(len(lp[flow_idx])) if isinstance(lp, list) else int(lp[flow_idx].shape[0])
        except Exception:
            path_len[i] = -1

        elapsed = time.time() - t0
        if (i + 1) % 20 == 0 or (i + 1) == n:
            print(f"  [{i+1}/{n}] {elapsed/(i+1):.2f}s/flow avg", flush=True)

    effect = np.abs(f_actual - f_baseline)
    err = np.abs(f_actual - y_true)

    # 4. Typology per flow.
    typology, thr = _typology(f_actual, f_baseline, y_true)

    # 5. Per-slice distributions.
    per_slice = {}
    for s, name in SLICE_NAMES.items():
        m = slice_arr == s
        if not m.any():
            continue
        per_slice[name] = {
            'n': int(m.sum()),
            'y_true_pct': _percentiles(y_true[m]),
            'f_actual_pct': _percentiles(f_actual[m]),
            'f_baseline_pct': _percentiles(f_baseline[m]),
            'abs_pred_err_pct': _percentiles(err[m]),
            'effect_pct': _percentiles(effect[m]),
            'typology_counts': {
                t: int((typology[m] == t).sum())
                for t in ('type1_typical', 'type2_model_collapsed', 'type3_high_effect')
            },
        }

    # 6. Cross-tab: effect vs error.
    thr_eff = np.percentile(effect, 25.0)
    thr_err = np.percentile(err / np.maximum(np.abs(y_true), 1e-12), 75.0)
    low_e = effect < thr_eff
    high_err = (err / np.maximum(np.abs(y_true), 1e-12)) > thr_err
    cross = {
        'threshold_effect_p25': float(thr_eff),
        'threshold_err_frac_p75': float(thr_err),
        'low_effect_and_high_err': int((low_e & high_err).sum()),
        'low_effect_and_low_err': int((low_e & ~high_err).sum()),
        'high_effect_and_high_err': int((~low_e & high_err).sum()),
        'high_effect_and_low_err': int((~low_e & ~high_err).sum()),
    }

    # 7. Sanity checks: exp_max_factor and 'model' one-hot constant?
    sanity = {
        'exp_max_factor_test_unique': int(len(np.unique(np.round(numeric_vals['exp_max_factor'], 8)))),
        'exp_max_factor_test_min': float(np.min(numeric_vals['exp_max_factor'])),
        'exp_max_factor_test_max': float(np.max(numeric_vals['exp_max_factor'])),
    }

    # 8. Training-set slice_type <-> numeric feature correlations (optional).
    correlations = None
    if args.train_cache and os.path.isdir(args.train_cache) and args.n_train_samples > 0:
        print(f"\nSampling up to {args.n_train_samples} training files for correlation table...",
              flush=True)
        train = _collect_train_flow_features(args.train_cache, args.n_train_samples, args.seed)
        if 'slice_type' in train and len(train['slice_type']):
            st = train['slice_type'].astype(np.float64)
            corr_int = {}
            corr_onehot = {}
            for k in NUMERIC_KEYS:
                v = train[k]
                m = np.isfinite(v) & np.isfinite(st)
                corr_int[k] = {
                    'pearson': _corr(st[m], v[m], 'pearson'),
                    'spearman': _corr(st[m], v[m], 'spearman'),
                }
            for name, tgt in (('eMBB', 0), ('mMTC', 1), ('URLLC', 2)):
                onehot = (st == tgt).astype(np.float64)
                corr_onehot[name] = {}
                for k in NUMERIC_KEYS:
                    v = train[k]
                    m = np.isfinite(v)
                    corr_onehot[name][k] = {
                        'pearson': _corr(onehot[m], v[m], 'pearson'),
                        'spearman': _corr(onehot[m], v[m], 'spearman'),
                    }
            model_arr = train.get('model')
            model_summary = None
            if model_arr is not None and len(model_arr):
                vals, cnts = np.unique(model_arr, return_counts=True)
                model_summary = {int(v): int(c) for v, c in zip(vals, cnts)}
            correlations = {
                'n_train_flows_sampled': int(len(st)),
                'slice_type_as_int': corr_int,
                'slice_type_onehot': corr_onehot,
                'model_onehot_value_counts_train_sample': model_summary,
            }

    # 9. Write outputs.
    os.makedirs(args.out_dir, exist_ok=True)
    npz_path = os.path.join(args.out_dir, f'{args.tag}.npz')
    np.savez(npz_path,
             y_true=y_true, f_actual=f_actual, f_baseline=f_baseline,
             effect=effect, abs_pred_err=err, slice_type=slice_arr,
             typology=np.array(typology, dtype=object),
             sample_file=np.array([r['file'] for r in records]),
             flow_idx=np.array([r['flow_idx'] for r in records], dtype=np.int32),
             path_len=path_len, n_flows_topology=n_flows_topo,
             traffic=numeric_vals['traffic'], packets=numeric_vals['packets'],
             eq_lambda=numeric_vals['eq_lambda'],
             avg_pkts_lambda=numeric_vals['avg_pkts_lambda'],
             exp_max_factor=numeric_vals['exp_max_factor'],
             delta=numeric_vals['delta'],
             feature_names=np.array(FEATURE_NAMES))

    summary = {
        'checkpoint': ckpt, 'shared': args.shared, 'n_flows': int(n),
        'thresholds': thr,
        'per_slice': per_slice,
        'effect_vs_error_crosstab': cross,
        'sanity': sanity,
        'correlations_train': correlations,
        'notes': (
            'Type 1 = low effect AND low pred error; the model matches truth near '
            'its baseline. Type 2 = low effect AND high pred error; model collapsed '
            'to baseline on a flow whose true delay is far from baseline. Type 3 = '
            'high effect; the model is using its learnable features on this flow. '
            'These three cohorts should NOT be aggregated into a single "near-baseline" '
            'bucket for the XAI report.'
        ),
    }
    json_path = os.path.join(args.out_dir, f'{args.tag}.json')
    with open(json_path, 'w') as fh:
        json.dump(summary, fh, indent=2, default=str)

    print(f"\nSaved: {npz_path}")
    print(f"Saved: {json_path}")
    print("\nPer-slice typology counts:")
    for name, blk in per_slice.items():
        c = blk['typology_counts']
        print(f"  {name:>5}: n={blk['n']:>4}  "
              f"type1_typical={c['type1_typical']:>4}  "
              f"type2_collapsed={c['type2_model_collapsed']:>4}  "
              f"type3_high_effect={c['type3_high_effect']:>4}")
    print("\nCross-tab (effect low = below p25; err high = above p75 of err/|y_true|):")
    print(f"  low_effect  & low_err : {cross['low_effect_and_low_err']}")
    print(f"  low_effect  & high_err: {cross['low_effect_and_high_err']}  <- Type 2 candidates")
    print(f"  high_effect & low_err : {cross['high_effect_and_low_err']}")
    print(f"  high_effect & high_err: {cross['high_effect_and_high_err']}")


if __name__ == '__main__':
    main()
