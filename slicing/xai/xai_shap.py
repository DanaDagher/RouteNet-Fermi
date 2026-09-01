"""
XAI phase - KernelSHAP with grouped categoricals (Option 1, step 5, revised).

WHY THIS EXISTS
---------------
The IG run (ig_shared_300_perslice.npz) says `delta` dominates every slice
by 3-6 orders of magnitude, uniformly across all three slices. Before
writing this up as a thesis finding, we validate it with a second,
independent attribution method. If two methods that differ in almost every
assumption agree on the ranking, the finding is much stronger.

IMPORTANT CHANGE FROM THE FIRST VERSION
---------------------------------------
The first version of this script matched IG's scope exactly: 6 numeric
features only, categorical columns pinned. That was correct as an
apples-to-apples IG replication but under-used SHAP's actual capability.

This revised version attributes ALL 8 semantic feature groups:
  1. traffic, 2. packets, 3. eq_lambda, 4. avg_pkts_lambda,
  5. exp_max_factor, 6. delta,     -- the 6 numeric features
  7. slice_type                    -- 3-column one-hot, treated AS ONE feature
  8. model                         -- 7-column one-hot, treated AS ONE feature

Why the change
--------------
KernelSHAP CAN handle categorical features cleanly if we group the one-hot
columns as a single feature and swap the WHOLE block between "actual" and
"background", never interpolating fractional one-hots. That is what IG
cannot do (its interpolation walks through [0.3, 0, 0.7] etc.) but SHAP
can, because SHAP toggles feature-presence rather than smoothly walking.

This is why we can attribute `slice_type` here even though IG cannot. It is
also the point of running a second XAI method: to answer a question the
first method can't.

BACKGROUND STRATEGY (deliberate, defensible)
--------------------------------------------
For each feature NOT in the current coalition S, we use a background value:

  - Numeric features (6): the PER-SLICE baseline row from
    baseline_stats_per_slice.json, same as IG. Matches IG apples-to-apples
    on the numerics.
  - `slice_type` (grouped one-hot, 3 cols): we EVALUATE THE MODEL 3 TIMES,
    once per pure one-hot ([1,0,0], [0,1,0], [0,0,1]), and average with
    the empirical training-set slice frequencies [0.04, 0.75, 0.21]. This
    is standard KernelSHAP with a small categorical background dataset,
    keeps every evaluation on-manifold (no fractional one-hot ever fed to
    the model), and matches Molnar's IML textbook's recommendation for
    grouped one-hots.
  - `model` (grouped one-hot, 7 cols): training set is 100% class 0 -> the
    empirical background is [1, 0, 0, 0, 0, 0, 0], a single point.
    Actual == background for every flow, so phi(model) will be EXACTLY 0
    by construction (a positive-control property, not a method failure).

INCLUSION OF DATA-DEGENERATE FEATURES AS POSITIVE CONTROLS
----------------------------------------------------------
`model` and `exp_max_factor` are constants on this dataset (100% model
class 0; exp_max_factor = 10 everywhere), documented by the diagnostic.
They will get phi == 0 by construction. This is DELIBERATE: any XAI
method that returns non-zero on either would indicate a methodological
artifact, so including them and observing phi == 0 is a specificity check
on the method.

EXACT SHAPLEY, NOT SAMPLED
--------------------------
With 8 features there are only 2^8 = 256 subsets to enumerate. For subsets
where `slice_type` is "out of coalition" we do 3 model evaluations (one
per background slice one-hot), so total model evals per flow are:
  128 (slice_type in) * 1 eval + 128 (slice_type out) * 3 evals = 512
At ~0.4 s/eval on GPU, that is ~200 s per flow, ~35 min for 10 flows.
Deterministic, exact Shapley -- no sampling error to bound.

Additivity check: sum(phi_8_groups) should equal f(actual) - f(all-background)
up to numerical noise (Shapley "efficiency"), analogous to IG completeness.

Usage:
    CUDA_VISIBLE_DEVICES=2 python slicing/xai/xai_shap.py \
        --shared slicing/xai/sets/shared_300.json --limit 10 --stratify \
        --tag shap_pilot10_perslice_grouped
"""
import argparse
import itertools
import json
import math
import os
import sys
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xai_common import (FEATURE_NAMES, SLICE_NAMES, load_model, load_raw_sample,
                        prepare_inputs)  # noqa: E402


# The 8 semantic feature groups in the order Shapley enumerates them.
# Each entry: (group_name, list_of_16vec_column_indices, kind)
# kind == 'numeric' -> single column, single background value from baseline row
# kind == 'onehot'  -> multi-column, background is a list of (weight, onehot_vec)
GROUPS = [
    ('traffic',         [0],                 'numeric'),
    ('packets',         [1],                 'numeric'),
    ('eq_lambda',       [9],                 'numeric'),
    ('avg_pkts_lambda', [10],                'numeric'),
    ('exp_max_factor',  [11],                'numeric'),
    ('delta',           [15],                'numeric'),
    ('slice_type',      [12, 13, 14],        'onehot'),
    ('model',           [2, 3, 4, 5, 6, 7, 8], 'onehot'),
]
N_GROUPS = len(GROUPS)  # 8
GROUP_NAMES = [g[0] for g in GROUPS]


def slice_onehot(k, dim=3):
    v = [0.0] * dim
    v[k] = 1.0
    return v


# Categorical background distributions.
# slice_type: empirical training frequency (from xai_baseline_stats_per_slice log)
SLICE_BG = [
    (0.04184784636459094, tf.constant(slice_onehot(0), dtype=tf.float32)),  # eMBB
    (0.75097248058503410, tf.constant(slice_onehot(1), dtype=tf.float32)),  # mMTC
    (0.20717967305037496, tf.constant(slice_onehot(2), dtype=tf.float32)),  # URLLC
]
# model: 100% class 0 in training -> single background point (weight = 1.0).
MODEL_BG = [
    (1.0, tf.constant([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=tf.float32)),
]


def all_subsets(n):
    """Yield every subset of {0..n-1} as a frozenset. 2^n items."""
    for mask in range(1 << n):
        yield frozenset(i for i in range(n) if mask & (1 << i))


def build_replacement_row(actual_row, baseline_row, subset,
                          slice_bg_vec, model_bg_vec):
    """Return a 16-vec representing the flow with features in `subset` at
    actual values and features NOT in subset at their background values.

    slice_bg_vec: the specific 3-vec being used as slice_type background for
                  this evaluation (chosen by the caller when averaging).
    model_bg_vec: same, for the 7-vec model background.

    Numeric groups: baseline value from the per-slice baseline_row.
    Categorical groups: whole one-hot block replaced by the given background."""
    row = tf.identity(actual_row)  # start from actual (categoricals default to
                                    # actual; numeric columns overwritten below)
    idxs = []
    updates = []
    for g_idx, (name, cols, kind) in enumerate(GROUPS):
        if g_idx in subset:
            continue  # keep actual value(s) for this group
        if kind == 'numeric':
            col = cols[0]
            idxs.append([col])
            updates.append(baseline_row[col])
        else:  # onehot
            bg = slice_bg_vec if name == 'slice_type' else model_bg_vec
            for k, col in enumerate(cols):
                idxs.append([col])
                updates.append(bg[k])
    if not idxs:
        return row
    updates_t = tf.stack(updates)
    return tf.tensor_scatter_nd_update(row,
                                       tf.constant(idxs, dtype=tf.int32),
                                       updates_t)


@tf.function(reduce_retracing=True)
def _eval_flow(model, inputs, X_actual, flow_idx, replacement_row):
    """Replace ONE row (the target flow's row) of the topology's X with
    `replacement_row`, run the full model, return that flow's predicted delay.
    All other flows stay at real values."""
    X_mod = tf.tensor_scatter_nd_update(X_actual, [[flow_idx]], [replacement_row])
    out = model.forward_from_path_input(X_mod, inputs)
    return tf.gather(out, flow_idx)


def value_of_subset(model, inputs, X_actual, flow_idx, actual_row,
                    baseline_row, subset):
    """Compute v(S) = expected model output over background categoricals when
    features in S are at actual values, features not in S are at background.

    If slice_type IS in subset (categorical is "actual"), no averaging over
    slice_bg is needed for it. If slice_type IS NOT in subset (categorical
    is "background"), average over the 3 slice_bg entries weighted by
    empirical frequency. Same for model (but model has 1 background entry
    so no actual averaging happens)."""
    slice_g_idx = GROUP_NAMES.index('slice_type')
    model_g_idx = GROUP_NAMES.index('model')
    slice_in = slice_g_idx in subset
    model_in = model_g_idx in subset

    slice_variants = [(1.0, None)] if slice_in else SLICE_BG
    model_variants = [(1.0, None)] if model_in else MODEL_BG

    total = 0.0
    total_w = 0.0
    for w_s, s_vec in slice_variants:
        for w_m, m_vec in model_variants:
            row = build_replacement_row(actual_row, baseline_row, subset,
                                        s_vec, m_vec)
            y = float(_eval_flow(model, inputs, X_actual,
                                 tf.constant(flow_idx, dtype=tf.int32),
                                 row).numpy().item())
            w = w_s * w_m
            total += w * y
            total_w += w
    return total / total_w  # weighted average


def exact_shapley_one_flow(model, inputs, X_actual, flow_idx, baseline_row):
    """Compute exact Shapley values for the 8 grouped features.

    Returns (phi_by_group, f_actual, f_baseline, residual):
      phi_by_group -- np.float64 array of shape (8,), one Shapley value per group
      f_actual     -- v(full coalition) with all groups at actual values
      f_baseline   -- v(empty coalition), i.e. all groups at background
                      (averaged over categorical background distributions)
      residual     -- (f_actual - f_baseline) - sum(phi_by_group), should be
                      tiny numerical noise (Shapley efficiency axiom)."""
    actual_row = tf.gather(X_actual, flow_idx)

    v_cache = {}
    for S in all_subsets(N_GROUPS):
        v_cache[S] = value_of_subset(model, inputs, X_actual, flow_idx,
                                     actual_row, baseline_row, S)
    f_baseline = v_cache[frozenset()]
    f_actual = v_cache[frozenset(range(N_GROUPS))]

    n = N_GROUPS
    n_fact = math.factorial(n)
    w = [math.factorial(k) * math.factorial(n - k - 1) / n_fact for k in range(n)]

    phi = np.zeros(N_GROUPS, dtype=np.float64)
    for i in range(n):
        for S in all_subsets(n):
            if i in S:
                continue
            k = len(S)
            phi[i] += w[k] * (v_cache[S | {i}] - v_cache[S])
    residual = (f_actual - f_baseline) - float(phi.sum())
    return phi, f_actual, f_baseline, residual


def phi_to_16vec(phi_by_group):
    """Expand per-group phi into a 16-vec (one entry per raw column).
    For a grouped one-hot feature, the group's phi is credited to the first
    column of that block; other block columns are 0. This is a display
    convention -- the semantic attribution is the per-group vector.
    Downstream comparison scripts should aggregate by GROUPS the same way
    xai_report_per_slice.py does."""
    out = np.zeros(16, dtype=np.float32)
    for g_idx, (name, cols, kind) in enumerate(GROUPS):
        out[cols[0]] = phi_by_group[g_idx]
    return out


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    slic = os.path.dirname(here)
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-dir', default=os.path.join(slic, 'ckpt_v2'))
    ap.add_argument('--test-cache', default=os.path.join(slic, 'cache', 'test'))
    ap.add_argument('--shared', default=os.path.join(here, 'sets', 'shared_300.json'))
    ap.add_argument('--out-dir', default=os.path.join(here, 'results'))
    ap.add_argument('--tag', default='shap_pilot10_perslice_grouped')
    ap.add_argument('--baseline-stats-per-slice',
                    default=os.path.join(here, 'baseline_stats_per_slice.json'))
    ap.add_argument('--limit', type=int, default=10,
                    help='number of flows to process; 0 = all remaining after '
                         '--start-idx')
    ap.add_argument('--start-idx', type=int, default=0,
                    help='index into the shared set to start from. Combined '
                         'with --limit lets you split a run across multiple '
                         'GPUs: GPU0 --start-idx 0 --limit 150, GPU1 '
                         '--start-idx 150 --limit 150, etc. The two .npz '
                         'outputs can be concatenated locally.')
    ap.add_argument('--stratify', action='store_true',
                    help='pick roughly equal counts per slice (only used when '
                         '--start-idx is 0)')
    return ap.parse_args()


def pick_flows(records, limit, stratify):
    if not stratify:
        return records[:limit]
    per_slice = limit // 3
    extras = limit - per_slice * 3
    quotas = {0: per_slice + (1 if extras > 0 else 0),
              1: per_slice + (1 if extras > 1 else 0),
              2: per_slice}
    picked = []
    counts = {0: 0, 1: 0, 2: 0}
    for r in records:
        s = int(r['slice_type'])
        if counts[s] < quotas[s]:
            picked.append(r)
            counts[s] += 1
        if len(picked) == limit:
            break
    return picked


def main():
    args = parse_args()
    for _g in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(_g, True)
        except RuntimeError:
            pass

    with open(args.shared) as fh:
        payload = json.load(fh)
    all_records = payload['records']
    if args.start_idx > 0:
        # multi-GPU split mode: take a contiguous slice, ignore --stratify
        end = args.start_idx + args.limit if args.limit > 0 else len(all_records)
        records = all_records[args.start_idx:end]
        print(f"Loaded {len(records)} flows to explain from {args.shared} "
              f"[start_idx={args.start_idx}, end={args.start_idx+len(records)}]",
              flush=True)
    else:
        limit = args.limit if args.limit > 0 else len(all_records)
        records = pick_flows(all_records, limit, args.stratify)
        print(f"Loaded {len(records)} flows to explain from {args.shared} "
              f"(stratify={args.stratify})", flush=True)

    model, ckpt = load_model(args.ckpt_dir)

    if not os.path.exists(args.baseline_stats_per_slice):
        raise SystemExit(f"ERROR: {args.baseline_stats_per_slice} not found -- "
                         f"run xai_baseline_stats_per_slice.py first.")
    with open(args.baseline_stats_per_slice) as fh:
        bstats_ps = json.load(fh)
    per_slice_baseline = {}
    for s_int, s_name in SLICE_NAMES.items():
        row = bstats_ps.get('per_slice', {}).get(s_name, {}).get('baseline_row')
        if row is None:
            raise SystemExit(f"ERROR: per_slice.{s_name}.baseline_row missing")
        per_slice_baseline[s_int] = tf.constant(row, dtype=tf.float32)
    print(f"Baseline: per-slice numeric from {args.baseline_stats_per_slice}",
          flush=True)
    print(f"Attribution surface: 8 grouped features "
          f"({', '.join(GROUP_NAMES)}). Categoricals grouped: slice_type as "
          f"one 3-col block with 3-way empirical-frequency background; model "
          f"as one 7-col block with single-point background (100% class 0).",
          flush=True)
    print(f"Expected model evals/flow: 128 (slice_type in) * 1 + 128 "
          f"(slice_type out) * 3 = 512", flush=True)

    phi_matrix = np.zeros((len(records), N_GROUPS), dtype=np.float64)
    phi_16 = np.zeros((len(records), 16), dtype=np.float32)
    f_actual_arr = np.zeros(len(records), dtype=np.float32)
    f_baseline_arr = np.zeros(len(records), dtype=np.float32)
    residual_arr = np.zeros(len(records), dtype=np.float32)

    t0 = time.time()
    cache = {}
    for i, rec in enumerate(records):
        fname, flow_idx = rec['file'], rec['flow_idx']
        s_int = int(rec['slice_type'])
        s_name = SLICE_NAMES[s_int]
        if fname not in cache:
            feats_raw, _ = load_raw_sample(args.test_cache, fname)
            cache[fname] = feats_raw
            if len(cache) > 4:
                cache.pop(next(iter(cache)))
        feats_raw = cache[fname]
        inputs = prepare_inputs(feats_raw)
        X_actual = model.assemble_path_input(inputs)
        baseline_row = per_slice_baseline[s_int]

        phi, f_a, f_b, resid = exact_shapley_one_flow(
            model, inputs, X_actual, flow_idx, baseline_row)
        phi_matrix[i] = phi
        phi_16[i] = phi_to_16vec(phi)
        f_actual_arr[i] = f_a
        f_baseline_arr[i] = f_b
        residual_arr[i] = resid

        elapsed = time.time() - t0
        rate = elapsed / (i + 1)
        eta = rate * (len(records) - i - 1)
        denom = max(abs(f_a - f_b), float(np.max(np.abs(phi))), 1e-12)
        print(f"  [{i+1}/{len(records)}] {s_name} f_actual={f_a:.4e} "
              f"f_baseline={f_b:.4e}  residual/scale={abs(resid)/denom*100:.3f}% "
              f"({rate:.1f}s/flow, ETA {eta/60:.1f}min)", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone: {len(records)} flows in {elapsed:.1f}s "
          f"({elapsed/len(records):.1f}s/flow avg)")

    print("\nPer-flow SHAP by group (phi):")
    header = "  " + " " * 10 + "  ".join(f"{n[:10]:>10s}" for n in GROUP_NAMES)
    print(header)
    for i, rec in enumerate(records):
        s_name = SLICE_NAMES[int(rec['slice_type'])]
        vals = "  ".join(f"{phi_matrix[i, j]:+10.3e}" for j in range(N_GROUPS))
        print(f"  [{i+1}] {s_name:6s} " + vals)

    print("\nPer-slice mean |SHAP| by group:")
    slice_types = np.array([r['slice_type'] for r in records], dtype=np.int32)
    for s_int, s_name in SLICE_NAMES.items():
        mask = (slice_types == s_int)
        if not mask.any():
            continue
        mean_abs = np.abs(phi_matrix[mask]).mean(axis=0)
        order = np.argsort(-mean_abs)
        top3 = [GROUP_NAMES[k] for k in order[:3]]
        print(f"  {s_name:6s} (n={int(mask.sum())})  top3: "
              f"{' > '.join(top3)}   |phi|_max={mean_abs.max():.3e}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f'{args.tag}.npz')
    sample_files = np.array([r['file'] for r in records])
    flow_idxs = np.array([r['flow_idx'] for r in records], dtype=np.int32)
    np.savez(out_path,
             phi_grouped=phi_matrix,
             phi_16=phi_16,
             additivity_residual=residual_arr,
             f_actual=f_actual_arr, f_baseline=f_baseline_arr,
             slice_type=slice_types, sample_file=sample_files, flow_idx=flow_idxs,
             group_names=np.array(GROUP_NAMES),
             feature_names=np.array(FEATURE_NAMES),
             method='exact_shapley_grouped_2pow8',
             baseline_source=args.baseline_stats_per_slice,
             baseline_is_per_slice=True,
             slice_bg_weights=np.array([w for w, _ in SLICE_BG], dtype=np.float32),
             model_bg_weights=np.array([w for w, _ in MODEL_BG], dtype=np.float32))
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
