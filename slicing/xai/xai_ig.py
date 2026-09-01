"""
XAI phase - step 3: Integrated Gradients on the 16 path_embedding inputs.

SCOPE (2026-08-21 correction, see XAI_PHASE_STATUS_REPORT.md §7.3):
  IG interpolates ONLY the 6 numeric columns of the 16-dim path input:
    traffic, packets, eq_lambda, avg_pkts_lambda, exp_max_factor, delta
  The 10 categorical columns (model one-hot 7 cols + slice_type one-hot 3 cols)
  are HELD FIXED at their per-flow actual values throughout the walk. So the
  IG walk never passes through invalid one-hot states like [0.3, 0, 0], which
  is the well-documented reason IG converges poorly on categorical inputs.
  This matches the old-phase IG scope choice (their 10 features were all
  numeric scalars). Attribution values for categorical columns are 0 by
  construction (diff=0), consistent with "not attributed" rather than
  "attributed and zero". Pass --scope full to reproduce the old broken
  16-column behaviour for comparison.

For a target flow, IG walks its 6 numeric-column input from a baseline value
(training-set median, z-scored, from `baseline_stats.json`) to the flow's
actual value in increments (trapezoidal rule), evaluates the model's
output-gradient w.r.t. that flow's row at each increment (holding every OTHER
flow/link/queue in the same topology at its real, unperturbed value),
and integrates:

    IG_i = (x_i - baseline_i) * trapezoidal_integral( d(output)/d(x_i), alpha=0..1 )

Sum(IG) should equal f(actual) - f(baseline) (the completeness axiom). A FIXED
step count does not converge uniformly across flows (measured: some flows hit
near-zero residual at 100 steps, others still have residual larger than their
own predicted value). So step count is ADAPTIVE per flow: start at
--start-steps, double until the residual is under --rel-tol of the actual
effect size |f_actual - f_baseline|, up to --max-steps. Flows that still have
not converged at --max-steps are saved anyway but flagged `converged=False` --
never silently treated as trustworthy.

Runs on the model wrapper in slicing/xai/xai_common.py, which was verified
(max_rel ~1e-5) to reproduce the original RouteNet_Fermi.call() exactly.

Usage (DGX):
    CUDA_VISIBLE_DEVICES=2 python slicing/xai/xai_ig.py \
        --ckpt-dir slicing/ckpt_v2 --test-cache slicing/cache/test \
        --shared slicing/xai/sets/shared_900.json \
        --out-dir slicing/xai/results --limit 20      # small run first, gauge speed

    CUDA_VISIBLE_DEVICES=2 python slicing/xai/xai_ig.py \
        --ckpt-dir slicing/ckpt_v2 --test-cache slicing/cache/test \
        --shared slicing/xai/sets/shared_900.json \
        --out-dir slicing/xai/results                 # full 900 once timing is known

    # Optionally also run on the IG-only extended set (cheap, up to 3000 flows):
    CUDA_VISIBLE_DEVICES=2 python slicing/xai/xai_ig.py \
        --shared slicing/xai/sets/ig_extended.json --out-dir slicing/xai/results \
        --tag ig_extended
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
from xai_common import (FEATURE_NAMES, SLICE_NAMES, load_model, load_raw_sample,
                        prepare_inputs)  # noqa: E402

# Column indices in the 16-dim path input (must match FEATURE_NAMES order in xai_common.py).
# 6 numeric scalars -- IG interpolates these:
NUMERIC_COL_IDX = [0, 1, 9, 10, 11, 15]
# 10 categorical one-hot columns -- IG pins these at their per-flow actual values:
CATEGORICAL_COL_IDX = [2, 3, 4, 5, 6, 7, 8, 12, 13, 14]
assert sorted(NUMERIC_COL_IDX + CATEGORICAL_COL_IDX) == list(range(16))


def build_effective_baseline(baseline_row_full, X_actual, flow_idx, scope):
    """Return the effective baseline for this flow, per --scope:
      - scope='numeric' (default): categorical columns are pinned to the flow's
        actual values (no walk through invalid one-hot mixes); only the 6
        numeric columns move during the IG integration.
      - scope='full': the plain baseline_row_full is used for all 16 columns
        (reproduces the old, categorical-broken behaviour for A/B comparison)."""
    if scope == 'full':
        return baseline_row_full
    actual_row = tf.gather(X_actual, flow_idx)
    cat_idx = tf.constant(CATEGORICAL_COL_IDX, dtype=tf.int32)
    updates = tf.gather(actual_row, cat_idx)
    return tf.tensor_scatter_nd_update(baseline_row_full,
                                       tf.expand_dims(cat_idx, 1), updates)


@tf.function(reduce_retracing=True)
def _one_alpha_grad(model, inputs, X_actual, flow_idx, baseline_row, alpha):
    """One IG step, compiled. flow_idx and alpha are tensors (not python values)
    so retracing is driven by shape/dtype only -- the same compiled graph is
    reused across all `steps` alpha values for a given topology (and even across
    topologies that happen to share a shape)."""
    actual_row = tf.gather(X_actual, flow_idx)
    interp_row = baseline_row + alpha * (actual_row - baseline_row)
    X_alpha = tf.tensor_scatter_nd_update(X_actual, [[flow_idx]], [interp_row])
    with tf.GradientTape() as tape:
        tape.watch(X_alpha)
        out = model.forward_from_path_input(X_alpha, inputs)
        target = tf.gather(out, flow_idx)
    grad = tape.gradient(target, X_alpha)
    return tf.gather(grad, flow_idx)


@tf.function(reduce_retracing=True)
def _endpoint_outputs(model, inputs, X_actual, flow_idx, baseline_row):
    f_actual = tf.gather(model.forward_from_path_input(X_actual, inputs), flow_idx)
    X_base_full = tf.tensor_scatter_nd_update(X_actual, [[flow_idx]], [baseline_row])
    f_baseline = tf.gather(model.forward_from_path_input(X_base_full, inputs), flow_idx)
    return f_actual, f_baseline


def integrated_gradients_adaptive(model, inputs, X_actual, flow_idx, baseline_row,
                                   start_steps=100, max_steps=400, rel_tol=0.10):
    """Run IG at increasing step counts until the completeness residual is small
    RELATIVE TO THE MEANINGFUL SCALE OF THE ATTRIBUTION, or max_steps is hit.

    Convergence metric (validated by diagnostic on 10 real flows):
        rel = |residual| / max(|f_actual - f_baseline|, max|IG|)

    Earlier metric used |f_actual - f_baseline| alone (effect size). Diagnostic
    found that for flows whose prediction happens to be close to the baseline
    prediction (i.e. the model considers the flow "typical"), effect size can
    be tiny while the individual per-feature IG values are much larger --
    dividing by effect size then produces a huge % for a residual that is
    actually small compared to the top attributions we care about. This is
    the same problem MAPE has on this dataset (documented in
    METHODOLOGY_JUSTIFICATION.md).

    Using max(effect, max|IG|) as the denominator:
      * still catches genuinely bad cases (flow 10 in the diagnostic:
        residual is 39% of max|IG| -> flagged, correctly)
      * no longer punishes "typical" flows whose IG values are actually fine
        (flows 5, 6, 8: residual is 6-11% of max|IG|, well under the 10% tol)

    Returns (ig_16, completeness_residual, f_actual, f_baseline, steps_used, converged)."""
    steps = start_steps
    while True:
        ig, resid, f_a, f_b = integrated_gradients(model, inputs, X_actual, flow_idx,
                                                    baseline_row, steps)
        effect_size = abs(f_a - f_b)
        max_abs_ig = float(np.max(np.abs(ig)))
        denom = max(effect_size, max_abs_ig, 1e-9)
        rel = abs(resid) / denom
        converged = rel <= rel_tol
        if converged or steps >= max_steps:
            return ig, resid, f_a, f_b, steps, converged
        steps = min(steps * 2, max_steps)


def integrated_gradients(model, inputs, X_actual, flow_idx, baseline_row, steps):
    """Returns (ig_16, completeness_residual, f_actual, f_baseline).

    Uses the trapezoidal rule (half-weight on the two endpoints), which
    integrates more accurately per step than a plain average -- important here
    because alpha=0 sits exactly at the baseline, a known weak point for
    naive/equal-weight IG (some ReLU units sit right at their kink there)."""
    flow_idx_t = tf.constant(flow_idx, dtype=tf.int32)
    alphas = tf.linspace(0.0, 1.0, steps)  # steps points, endpoints included

    grads = tf.TensorArray(tf.float32, size=steps)
    for i in range(steps):
        g = _one_alpha_grad(model, inputs, X_actual, flow_idx_t, baseline_row, alphas[i])
        grads = grads.write(i, g)
    grads = grads.stack()  # (steps, 16)

    h = 1.0 / (steps - 1)
    weights = tf.fill([steps], h)
    weights = tf.tensor_scatter_nd_update(weights, [[0], [steps - 1]], [h / 2, h / 2])
    avg_grad = tf.reduce_sum(grads * weights[:, None], axis=0)  # trapezoidal integral, already normalized to [0,1]
    diff = tf.gather(X_actual, flow_idx_t) - baseline_row
    ig = diff * avg_grad

    f_actual, f_baseline = _endpoint_outputs(model, inputs, X_actual, flow_idx_t, baseline_row)
    completeness_residual = (f_actual - f_baseline) - tf.reduce_sum(ig)
    return (ig.numpy(), float(completeness_residual.numpy().item()),
            float(f_actual.numpy().item()), float(f_baseline.numpy().item()))


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    slic = os.path.dirname(here)
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-dir', default=os.path.join(slic, 'ckpt_v2'))
    ap.add_argument('--test-cache', default=os.path.join(slic, 'cache', 'test'))
    ap.add_argument('--shared', default=os.path.join(here, 'sets', 'shared_900.json'))
    ap.add_argument('--out-dir', default=os.path.join(here, 'results'))
    ap.add_argument('--tag', default='ig_shared_900',
                    help='output filename prefix')
    ap.add_argument('--start-steps', type=int, default=100)
    ap.add_argument('--max-steps', type=int, default=400,
                    help='cap on doubling; a flow that has not converged by '
                         'this many steps is saved anyway but flagged '
                         'converged=False. Lowered from 800 after diagnostic '
                         'showed genuinely-bad flows do not converge at 800 '
                         'either, and false-alarm flows were wasting the '
                         'extra steps chasing a metric they could not hit.')
    ap.add_argument('--rel-tol', type=float, default=0.10,
                    help='target |residual| / max(|f_actual - f_baseline|, max|IG|). '
                         'Denominator uses whichever is larger so we do not '
                         'punish flows whose prediction is close to the baseline '
                         '(where effect size alone is misleadingly small -- '
                         'same MAPE trap documented in METHODOLOGY_JUSTIFICATION.md).')
    ap.add_argument('--scope', choices=['numeric', 'full'], default='numeric',
                    help="'numeric' (default, CORRECT): IG interpolates only "
                         "the 6 numeric columns; categorical one-hots pinned "
                         "at their per-flow actuals so IG never walks through "
                         "invalid one-hot mixes. 'full' reproduces the earlier "
                         "categorical-broken behaviour for A/B comparison only.")
    ap.add_argument('--limit', type=int, default=0,
                    help='process only the first N records (0 = all); use a '
                         'small number first to gauge per-flow runtime')
    ap.add_argument('--baseline-stats', default=os.path.join(here, 'baseline_stats.json'),
                    help='output of xai_baseline_stats.py; median-based baseline '
                         'for scalars, empirical-frequency baseline for one-hot '
                         'blocks (model, slice_type). Falls back to all-zero '
                         'with a warning if the file does not exist.')
    ap.add_argument('--baseline-stats-per-slice',
                    default=os.path.join(here, 'baseline_stats_per_slice.json'),
                    help='output of xai_baseline_stats_per_slice.py; provides '
                         'a 16-dim baseline_row PER SLICE (scalars = median '
                         'within that slice). If this file exists it takes '
                         'precedence over --baseline-stats and each flow is '
                         'attributed against a reference built from its own '
                         'slice, correcting the mMTC-dominated single baseline '
                         'that the 300-flow audit diagnostic exposed. If it '
                         'does not exist, falls back to --baseline-stats '
                         'unchanged.')
    return ap.parse_args()


def main():
    args = parse_args()
    for _g in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(_g, True)
        except RuntimeError:
            pass

    with open(args.shared) as fh:
        payload = json.load(fh)
    records = payload['records']
    if args.limit > 0:
        records = records[:args.limit]
    print(f"Loaded {len(records)} flows to explain from {args.shared}", flush=True)

    model, ckpt = load_model(args.ckpt_dir)

    # Per-slice baseline takes precedence when available. baseline_row is used
    # as the fallback / single-baseline case; per_slice_baseline_rows is a dict
    # {slice_int: tf.constant(16,)} that IG picks from per flow if not None.
    per_slice_baseline_rows = None
    baseline_source_tag = None
    if os.path.exists(args.baseline_stats_per_slice):
        with open(args.baseline_stats_per_slice) as fh:
            bstats_ps = json.load(fh)
        # Global row (for logging / .npz storage; individual flows use the per-slice row).
        baseline_row = tf.constant(bstats_ps['baseline_row'], dtype=tf.float32)
        per_slice_baseline_rows = {}
        for slice_int, slice_name in SLICE_NAMES.items():
            row = bstats_ps.get('per_slice', {}).get(slice_name, {}).get('baseline_row')
            if row is None:
                raise SystemExit(
                    f"ERROR: {args.baseline_stats_per_slice} has no "
                    f"per_slice.{slice_name}.baseline_row -- regenerate with "
                    f"xai_baseline_stats_per_slice.py.")
            per_slice_baseline_rows[slice_int] = tf.constant(row, dtype=tf.float32)
        baseline_source_tag = args.baseline_stats_per_slice
        print(f"Baseline: loaded PER-SLICE from {args.baseline_stats_per_slice}. "
              f"Each flow's IG uses a reference row built from its own slice's "
              f"training-set medians (fixes the mMTC-dominated single baseline "
              f"the diagnostic exposed).", flush=True)
    elif os.path.exists(args.baseline_stats):
        with open(args.baseline_stats) as fh:
            bstats = json.load(fh)
        baseline_row = tf.constant(bstats['baseline_row'], dtype=tf.float32)
        baseline_source_tag = args.baseline_stats
        print(f"Baseline: loaded SINGLE from {args.baseline_stats} "
              f"(training-set median for scalars, empirical class frequency for "
              f"model/slice_type one-hots -- NOT all-zero). Per-slice file "
              f"{args.baseline_stats_per_slice} not found; using single "
              f"baseline for ALL slices (mMTC-biased -- see diagnostic).",
              flush=True)
    else:
        baseline_row = tf.zeros(16, dtype=tf.float32)
        baseline_source_tag = 'all_zero_fallback'
        print(f"WARNING: neither {args.baseline_stats_per_slice} nor "
              f"{args.baseline_stats} found -- falling back to all-zero "
              f"baseline. Run xai_baseline_stats_per_slice.py first.", flush=True)
    print(f"Scope: {args.scope} -- "
          + ("IG interpolates the 6 numeric columns only; categorical one-hots "
             "pinned per-flow (CORRECT)" if args.scope == 'numeric'
             else "IG interpolates all 16 columns incl. categoricals "
                  "(REPRODUCES CATEGORICAL-BROKEN BEHAVIOUR)"),
          flush=True)

    ig_matrix = np.zeros((len(records), 16), dtype=np.float32)
    completeness = np.zeros(len(records), dtype=np.float32)
    f_actual_arr = np.zeros(len(records), dtype=np.float32)
    f_baseline_arr = np.zeros(len(records), dtype=np.float32)
    steps_used_arr = np.zeros(len(records), dtype=np.int32)
    converged_arr = np.zeros(len(records), dtype=bool)

    t0 = time.time()
    cache = {}  # avoid reloading the same .pkl if a file recurs across records
    for i, rec in enumerate(records):
        fname, flow_idx = rec['file'], rec['flow_idx']
        if fname not in cache:
            feats_raw, _ = load_raw_sample(args.test_cache, fname)
            cache[fname] = feats_raw
            if len(cache) > 4:  # keep memory bounded; shared set is 1 flow/file anyway
                cache.pop(next(iter(cache)))
        feats_raw = cache[fname]
        inputs = prepare_inputs(feats_raw)
        X_actual = model.assemble_path_input(inputs)

        # Pick the baseline row for this flow. Per-slice takes precedence when
        # available (fixes the mMTC-dominated single baseline exposed by the
        # 300-flow diagnostic). rec['slice_type'] is 0/1/2 (eMBB/mMTC/URLLC).
        if per_slice_baseline_rows is not None:
            flow_baseline_row = per_slice_baseline_rows[int(rec['slice_type'])]
        else:
            flow_baseline_row = baseline_row

        # Per-flow effective baseline: in 'numeric' scope, categorical one-hot
        # columns are pinned to this flow's actuals so IG never walks through
        # invalid one-hot mixes (see build_effective_baseline docstring).
        effective_baseline = build_effective_baseline(
            flow_baseline_row, X_actual, tf.constant(flow_idx, dtype=tf.int32), args.scope)

        ig, resid, f_a, f_b, steps_used, converged = integrated_gradients_adaptive(
            model, inputs, X_actual, flow_idx, effective_baseline,
            start_steps=args.start_steps, max_steps=args.max_steps, rel_tol=args.rel_tol)
        ig_matrix[i] = ig
        completeness[i] = resid
        f_actual_arr[i] = f_a
        f_baseline_arr[i] = f_b
        steps_used_arr[i] = steps_used
        converged_arr[i] = converged

        elapsed = time.time() - t0
        rate = elapsed / (i + 1)
        eta = rate * (len(records) - i - 1)
        denom = max(abs(f_a - f_b), float(np.max(np.abs(ig))), 1e-9)
        conv_tag = "OK" if converged else "NOT CONVERGED (hit max_steps)"
        print(f"  [{i+1}/{len(records)}] {rate:.2f}s/flow avg, ETA {eta/60:.1f} min, "
              f"steps_used={steps_used}, resid/scale={abs(resid)/denom*100:.1f}% [{conv_tag}]",
              flush=True)  # flush so this shows immediately even when piped through tee/log

        # Save partial progress every 10 flows so a long run is never all-or-nothing:
        # if interrupted, everything computed so far is on disk, and progress can be
        # inspected without waiting for the full run to finish.
        if (i + 1) % 10 == 0 or (i + 1) == len(records):
            os.makedirs(args.out_dir, exist_ok=True)
            partial_path = os.path.join(args.out_dir, f'{args.tag}_partial.npz')
            n = i + 1
            np.savez(partial_path,
                     ig=ig_matrix[:n], completeness_residual=completeness[:n],
                     f_actual=f_actual_arr[:n], f_baseline=f_baseline_arr[:n],
                     steps_used=steps_used_arr[:n], converged=converged_arr[:n],
                     slice_type=np.array([r['slice_type'] for r in records[:n]], dtype=np.int32),
                     sample_file=np.array([r['file'] for r in records[:n]]),
                     flow_idx=np.array([r['flow_idx'] for r in records[:n]], dtype=np.int32),
                     feature_names=np.array(FEATURE_NAMES), n_done=n, n_total=len(records))

    elapsed = time.time() - t0
    n_conv = int(converged_arr.sum())
    print(f"\nDone: {len(records)} flows in {elapsed:.1f}s "
          f"({elapsed/len(records):.2f}s/flow avg)")
    print(f"Converged within max_steps={args.max_steps}: {n_conv}/{len(records)} "
          f"({100*n_conv/len(records):.1f}%)")
    if n_conv < len(records):
        print(f"WARNING: {len(records)-n_conv} flow(s) did NOT converge (residual still "
              f">{args.rel_tol*100:.0f}% of effect size at max_steps={args.max_steps}). "
              f"These flows' attributions are LESS trustworthy -- they are saved and "
              f"flagged (converged=False in the .npz) rather than silently included.")

    max_abs_ig = np.max(np.abs(ig_matrix), axis=1)
    denom_arr = np.maximum(np.maximum(np.abs(f_actual_arr - f_baseline_arr), max_abs_ig), 1e-9)
    resid_frac = np.abs(completeness) / denom_arr
    print(f"Residual as %% of max(effect, max|IG|) [the adaptive convergence "
          f"metric]: mean={np.mean(resid_frac)*100:.1f}%, "
          f"median={np.median(resid_frac)*100:.1f}%, max={np.max(resid_frac)*100:.1f}%")
    print(f"Steps used: min={steps_used_arr.min()}, median={int(np.median(steps_used_arr))}, "
          f"max={steps_used_arr.max()}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f'{args.tag}.npz')
    slice_types = np.array([r['slice_type'] for r in records], dtype=np.int32)
    sample_files = np.array([r['file'] for r in records])
    flow_idxs = np.array([r['flow_idx'] for r in records], dtype=np.int32)
    np.savez(out_path,
             ig=ig_matrix, completeness_residual=completeness,
             f_actual=f_actual_arr, f_baseline=f_baseline_arr,
             steps_used=steps_used_arr, converged=converged_arr,
             slice_type=slice_types, sample_file=sample_files, flow_idx=flow_idxs,
             feature_names=np.array(FEATURE_NAMES), baseline_row=baseline_row.numpy(),
             scope=args.scope, numeric_cols=np.array(NUMERIC_COL_IDX, dtype=np.int32),
             categorical_cols=np.array(CATEGORICAL_COL_IDX, dtype=np.int32),
             start_steps=args.start_steps, max_steps=args.max_steps, rel_tol=args.rel_tol,
             baseline_source=baseline_source_tag,
             baseline_is_per_slice=(per_slice_baseline_rows is not None))
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
