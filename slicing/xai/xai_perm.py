"""
XAI phase - Permutation Importance across all 8 feature groups (Option 1, step 6).

WHY THIS EXISTS
---------------
Third XAI method for the comparative study, complementing IG (gradient,
per-flow, 6 features) and KernelSHAP (coalition, per-flow, 8 grouped
features). Permutation importance is:
  - MODEL-AGNOSTIC (no gradients, no baseline, no coalition enumeration);
  - NATURAL FOR CATEGORICALS (shuffle the whole one-hot block together);
  - GLOBAL / AGGREGATE (a single importance number per feature-group, not
    a per-flow attribution).

Reference: Breiman 2001 "Random Forests", section on out-of-bag permutation
importance; and scikit-learn's `sklearn.inspection.permutation_importance`.

WHAT IS MEASURED (be careful with the wording)
----------------------------------------------
For each feature-group g:
  1. Compute the baseline error of the frozen model on the shared set:
       err_base = MAE over all 300 flows of (y_true - y_pred).
  2. For each of `--n-repeats` shuffling repetitions (default 5):
       Shuffle the RAW column(s) of group g across all 300 flows in the
       shared set (topology structure preserved; only the values of that
       feature block are permuted among flows). For grouped one-hots we
       shuffle the WHOLE block together so every row keeps a valid one-hot.
       Recompute predictions on all 300 flows with the permuted feature.
       Record err_perm.
  3. Importance(g) = mean over repeats of (err_perm - err_base).

Higher Importance(g) => model relies more on this feature to make correct
predictions.

WHAT THIS IS NOT
----------------
This is *feature dependence* of the model, not causal importance in the
network. If two features are highly correlated (as slice_type is with
packets/avg_pkts_lambda in this dataset, r ~ 0.94-0.99 per the diagnostic),
shuffling one leaves the model able to reconstruct much of its signal
from the other, and permutation importance for BOTH will be UNDERSTATED.
This is a well-documented limitation of permutation importance under
feature correlation. It must be stated in the report.

POSITIVE CONTROLS
-----------------
`model` (100% class 0 in this dataset) and `exp_max_factor` (constant = 10):
shuffling them changes nothing, so importance ~ 0 is expected. Any non-zero
value on either would flag a bug -- so we include them.

SHUFFLE STRATEGY: across-topology or within-topology?
-----------------------------------------------------
We shuffle within-topology (across flows in the same topology sample) rather
than across topologies. Two reasons:
  1. RouteNet-Fermi's forward pass depends on per-topology graph structure;
     mixing feature values between topologies is off-manifold in a way that
     confuses the "measure feature dependence, not topology dependence"
     question.
  2. It matches the shared-set structure: each shared record is (file,
     flow_idx). A shuffle within a file swaps the target flow's value with
     another random flow's value in the same topology.

For groups with one column (numerics), we swap a scalar. For grouped
one-hots, we swap the 3-vec (slice) or 7-vec (model) block as a unit,
preserving validity.

COST
----
For each group and each repeat, one forward pass over 300 flows. With
--n-repeats 5 and 8 groups, that is 5 * 8 * 300 = 12000 flow predictions.
Since a forward pass computes all flows in a topology at once, actual
GPU time depends on topology sizes; empirically ~2-3 min for the whole
run on one V100.

OUTPUT
------
A JSON summary and a .npz with:
  - importance_mean[8] and importance_std[8] over the repeats
  - baseline_error scalar (MAE)
  - per-slice breakdown (importance measured on eMBB/mMTC/URLLC flows only)

Usage:
    CUDA_VISIBLE_DEVICES=2 python slicing/xai/xai_perm.py \
        --shared slicing/xai/sets/shared_300.json \
        --n-repeats 5 --tag perm_shared_300
"""
import argparse
import copy
import json
import os
import sys
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xai_common import (FEATURE_NAMES, SLICE_NAMES, load_model, load_raw_sample,
                        prepare_inputs)  # noqa: E402


# Feature groups (must match xai_shap.py so cross-method comparison lines up).
# Kind meanings:
#   'scalar_int'   -> integer feature, dtype int32 in the raw dict (model, slice_type)
#   'scalar_float' -> float feature, dtype float32 (traffic, packets, etc.)
# We list the RAW feature key in feats dict; the one-hot happens inside the
# model wrapper, so shuffling the raw integer key is the correct operation
# for grouped one-hots (a shuffled int -> a shuffled valid one-hot).
GROUPS = [
    ('traffic',         'traffic',         'scalar_float'),
    ('packets',         'packets',         'scalar_float'),
    ('eq_lambda',       'eq_lambda',       'scalar_float'),
    ('avg_pkts_lambda', 'avg_pkts_lambda', 'scalar_float'),
    ('exp_max_factor',  'exp_max_factor',  'scalar_float'),
    ('delta',           'delta',           'scalar_float'),
    ('slice_type',      'slice_type',      'scalar_int'),
    ('model',           'model',           'scalar_int'),
]
GROUP_NAMES = [g[0] for g in GROUPS]


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    slic = os.path.dirname(here)
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-dir', default=os.path.join(slic, 'ckpt_v2'))
    ap.add_argument('--test-cache', default=os.path.join(slic, 'cache', 'test'))
    ap.add_argument('--shared', default=os.path.join(here, 'sets', 'shared_300.json'))
    ap.add_argument('--out-dir', default=os.path.join(here, 'results'))
    ap.add_argument('--tag', default='perm_shared_300')
    ap.add_argument('--n-repeats', type=int, default=5,
                    help='number of shuffle repetitions per feature group '
                         '(more = tighter std estimate)')
    ap.add_argument('--seed', type=int, default=42)
    return ap.parse_args()


def predict_all(model, records, cache):
    """Return (y_pred_flat, y_true_flat, slice_type_flat) over the whole
    shared set. Each entry corresponds to ONE record (one flow).

    Predicts the whole topology for each unique file, then indexes the
    target flow. This matches xai_ig.py's loop structure."""
    N = len(records)
    y_pred = np.zeros(N, dtype=np.float32)
    y_true = np.zeros(N, dtype=np.float32)
    slice_arr = np.zeros(N, dtype=np.int32)

    # Group records by file so each topology is processed once.
    by_file = {}
    for i, r in enumerate(records):
        by_file.setdefault(r['file'], []).append((i, r['flow_idx']))

    for fname, idx_list in by_file.items():
        feats_raw, labels = cache[fname]
        inputs = prepare_inputs(feats_raw)
        preds = model(inputs).numpy()  # full topology prediction (N_flows,)
        for i_rec, flow_idx in idx_list:
            y_pred[i_rec] = preds[flow_idx]
            y_true[i_rec] = labels[flow_idx]
            slice_arr[i_rec] = int(np.asarray(feats_raw['slice_type'])[flow_idx])
    return y_pred, y_true, slice_arr


def predict_all_with_shuffle(model, records, cache_files_shuffled, key):
    """Predict the whole shared set with feature `key` shuffled across flows
    WITHIN EACH FILE. `cache_files_shuffled` is a dict fname -> (feats_shuffled,
    labels) where feats_shuffled[key] has been permuted within the file's
    flow axis. Returns y_pred (N,)."""
    N = len(records)
    y_pred = np.zeros(N, dtype=np.float32)
    by_file = {}
    for i, r in enumerate(records):
        by_file.setdefault(r['file'], []).append((i, r['flow_idx']))
    for fname, idx_list in by_file.items():
        feats_raw, _ = cache_files_shuffled[fname]
        inputs = prepare_inputs(feats_raw)
        preds = model(inputs).numpy()
        for i_rec, flow_idx in idx_list:
            y_pred[i_rec] = preds[flow_idx]
    return y_pred


def mae(y_pred, y_true):
    return float(np.mean(np.abs(y_pred - y_true)))


def per_slice_mae(y_pred, y_true, slice_arr):
    """Return {slice_name: mae over that slice}."""
    out = {}
    for s_int, s_name in SLICE_NAMES.items():
        mask = (slice_arr == s_int)
        if not mask.any():
            out[s_name] = float('nan')
        else:
            out[s_name] = mae(y_pred[mask], y_true[mask])
    return out


def shuffle_key_in_place(feats, key, rng):
    """Permute values of `feats[key]` along the flow axis in-place.
    For scalar features (float or int), that is the flow axis == axis 0.
    """
    arr = np.asarray(feats[key]).copy()
    perm = rng.permutation(arr.shape[0])
    arr = arr[perm]
    feats[key] = arr
    return feats


def main():
    args = parse_args()
    for _g in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(_g, True)
        except RuntimeError:
            pass
    rng = np.random.default_rng(args.seed)

    with open(args.shared) as fh:
        payload = json.load(fh)
    records = payload['records']
    print(f"Loaded {len(records)} flows from {args.shared}", flush=True)

    model, ckpt = load_model(args.ckpt_dir)

    # Preload every file's cached (feats, labels) once. Small memory footprint
    # for the shared-300 set (one file per flow at most).
    print(f"Preloading test cache...", flush=True)
    file_cache = {}
    for r in records:
        fname = r['file']
        if fname not in file_cache:
            file_cache[fname] = load_raw_sample(args.test_cache, fname)
    print(f"  {len(file_cache)} unique files loaded", flush=True)

    # Baseline: predict on original (unshuffled) features.
    t0 = time.time()
    y_pred_base, y_true, slice_arr = predict_all(model, records, file_cache)
    err_base = mae(y_pred_base, y_true)
    err_base_slice = per_slice_mae(y_pred_base, y_true, slice_arr)
    print(f"Baseline MAE (all 300): {err_base:.4e}", flush=True)
    print(f"Baseline MAE per slice: "
          f"eMBB={err_base_slice['eMBB']:.4e}, "
          f"mMTC={err_base_slice['mMTC']:.4e}, "
          f"URLLC={err_base_slice['URLLC']:.4e}", flush=True)
    print(f"Baseline pass took {time.time()-t0:.1f}s", flush=True)

    # Permutation importance: per group, per repeat.
    importance = np.zeros((len(GROUPS), args.n_repeats), dtype=np.float64)
    importance_per_slice = {n: np.zeros((len(GROUPS), args.n_repeats),
                                        dtype=np.float64)
                            for n in ('eMBB', 'mMTC', 'URLLC')}

    for g_idx, (name, key, kind) in enumerate(GROUPS):
        print(f"\n--- Group {g_idx+1}/{len(GROUPS)}: {name} ({kind}) ---",
              flush=True)
        for rep in range(args.n_repeats):
            t1 = time.time()
            # Build shuffled cache: copy each file's feats dict, permute key
            # inside it. Labels unchanged.
            shuffled = {}
            rep_rng = np.random.default_rng(args.seed + 1000 * g_idx + rep)
            for fname, (feats, labels) in file_cache.items():
                feats_copy = {k: (v.copy() if isinstance(v, np.ndarray) else
                                  copy.copy(v)) for k, v in feats.items()}
                shuffle_key_in_place(feats_copy, key, rep_rng)
                shuffled[fname] = (feats_copy, labels)
            y_pred_perm = predict_all_with_shuffle(model, records, shuffled, key)
            err_perm = mae(y_pred_perm, y_true)
            importance[g_idx, rep] = err_perm - err_base
            for s_name in ('eMBB', 'mMTC', 'URLLC'):
                s_int = {v: k for k, v in SLICE_NAMES.items()}[s_name]
                mask = (slice_arr == s_int)
                if not mask.any():
                    importance_per_slice[s_name][g_idx, rep] = float('nan')
                    continue
                err_perm_s = mae(y_pred_perm[mask], y_true[mask])
                importance_per_slice[s_name][g_idx, rep] = (
                    err_perm_s - err_base_slice[s_name])
            print(f"  repeat {rep+1}/{args.n_repeats}: "
                  f"err_perm={err_perm:.4e}  "
                  f"delta={importance[g_idx, rep]:+.4e}  "
                  f"({time.time()-t1:.1f}s)", flush=True)

    mean = importance.mean(axis=1)
    std = importance.std(axis=1, ddof=1) if args.n_repeats > 1 else np.zeros_like(mean)

    print("\n" + "=" * 70)
    print(f"PERMUTATION IMPORTANCE (all 300, MAE delta, {args.n_repeats} repeats)")
    print("=" * 70)
    order = np.argsort(-mean)
    for j in order:
        name = GROUP_NAMES[j]
        print(f"  {name:18s}  {mean[j]:+10.4e}  +/- {std[j]:.4e}")
    print("=" * 70)

    per_slice_summary = {}
    for s_name in ('eMBB', 'mMTC', 'URLLC'):
        m = importance_per_slice[s_name].mean(axis=1)
        s = (importance_per_slice[s_name].std(axis=1, ddof=1)
             if args.n_repeats > 1 else np.zeros_like(m))
        order_s = np.argsort(-m)
        top3 = [GROUP_NAMES[k] for k in order_s[:3]]
        per_slice_summary[s_name] = {
            'importance_mean': dict(zip(GROUP_NAMES, m.tolist())),
            'importance_std': dict(zip(GROUP_NAMES, s.tolist())),
            'ranking': [GROUP_NAMES[k] for k in order_s],
            'top3': top3,
        }
        print(f"\n{s_name} top-3: {' > '.join(top3)}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_npz = os.path.join(args.out_dir, f'{args.tag}.npz')
    np.savez(out_npz,
             importance_all=importance,
             importance_mean=mean,
             importance_std=std,
             group_names=np.array(GROUP_NAMES),
             baseline_mae=np.float32(err_base),
             baseline_mae_per_slice=np.array([err_base_slice[n]
                                              for n in ('eMBB', 'mMTC', 'URLLC')]),
             importance_per_slice_eMBB=importance_per_slice['eMBB'],
             importance_per_slice_mMTC=importance_per_slice['mMTC'],
             importance_per_slice_URLLC=importance_per_slice['URLLC'],
             n_repeats=args.n_repeats, seed=args.seed,
             shared=args.shared, ckpt=ckpt)
    print(f"\nSaved: {out_npz}")

    out_json = os.path.join(args.out_dir, f'{args.tag}.json')
    with open(out_json, 'w') as fh:
        json.dump({
            'shared': os.path.basename(args.shared),
            'ckpt': ckpt,
            'n_repeats': args.n_repeats,
            'seed': args.seed,
            'baseline_mae_all': err_base,
            'baseline_mae_per_slice': err_base_slice,
            'importance_all': {
                'mean': dict(zip(GROUP_NAMES, mean.tolist())),
                'std': dict(zip(GROUP_NAMES, std.tolist())),
                'ranking_by_mean_desc': [GROUP_NAMES[k] for k in order],
            },
            'importance_per_slice': per_slice_summary,
        }, fh, indent=1)
    print(f"Saved: {out_json}")


if __name__ == '__main__':
    main()
