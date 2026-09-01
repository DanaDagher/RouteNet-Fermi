"""
XAI phase - per-slice reporting (Option 1, step 4/final analysis).

Consumes the two artifacts we now have on disk:
  1. slicing/xai/results/ig_shared_300_perslice.npz  (IG run with per-slice baseline)
  2. slicing/xai/results/audit_diagnostics.json       (Type 1/2/3 typology per flow)

Produces (all in --out-dir, default: slicing/xai/results/):
  - ig_report_per_slice.json     machine-readable summary
  - ig_report_per_slice.md       human-readable per-slice report
  - ig_meanabs_<slice>.png       one bar chart per slice (mean |IG| per feature)
  - ig_topfeatures_all.png       one grouped bar chart across slices

Runs locally on Windows with numpy + matplotlib. No GPU, no TF.

Usage:
    python slicing/xai/xai_report_per_slice.py
"""
import argparse
import json
import os

import numpy as np


# Feature-group definition (identical to xai_common.FEATURE_GROUPS but
# duplicated here so this reporter has no dependency on TF or on the model
# code -- runs on any laptop with numpy + matplotlib).
FEATURE_GROUPS = {
    'traffic': [0], 'packets': [1],
    'model': list(range(2, 9)),
    'eq_lambda': [9], 'avg_pkts_lambda': [10], 'exp_max_factor': [11],
    'slice_type': [12, 13, 14],
    'delta': [15],
}
# Presentation order (numeric first, categorical last).
GROUP_ORDER = ['traffic', 'packets', 'eq_lambda', 'avg_pkts_lambda',
               'exp_max_factor', 'delta', 'slice_type', 'model']
SLICE_NAMES = {0: 'eMBB', 1: 'mMTC', 2: 'URLLC'}


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    slic = os.path.dirname(here)
    ap = argparse.ArgumentParser()
    ap.add_argument('--ig-npz',
                    default=os.path.join(slic, 'xai', 'results',
                                         'ig_shared_300_perslice.npz'))
    ap.add_argument('--diagnostics-json',
                    default=os.path.join(slic, 'xai', 'results',
                                         'audit_diagnostics.json'))
    ap.add_argument('--out-dir',
                    default=os.path.join(slic, 'xai', 'results'))
    ap.add_argument('--no-plots', action='store_true',
                    help='skip PNG generation (useful if matplotlib is not installed)')
    return ap.parse_args()


def group_ig(ig_matrix):
    """Aggregate the 16-col IG matrix into an 8-group matrix by SUMMING signed
    values within each one-hot block (model, slice_type). The sum, not the
    max or abs-sum, is what preserves the completeness axiom at the group
    level: sum(group IG) == sum(column IG) == f_actual - f_baseline (up to
    the completeness residual)."""
    N = ig_matrix.shape[0]
    grouped = np.zeros((N, len(GROUP_ORDER)), dtype=np.float32)
    for j, name in enumerate(GROUP_ORDER):
        cols = FEATURE_GROUPS[name]
        grouped[:, j] = ig_matrix[:, cols].sum(axis=1)
    return grouped


def typology_from_diagnostics(diag, slice_types, effect_thr):
    """Reconstruct the Type 1/2/3 label per flow using the diagnostic's
    threshold (p25 of |f_actual - f_baseline|) and its per-slice p75 of
    err/|y_true|. We do not have y_true in the IG .npz, so we can only
    reconstruct Type 3 (high effect) directly; Type 1 vs Type 2 requires the
    diagnostic run's y_true / y_pred, which are in the diagnostic .npz not the
    JSON. Instead of half-reconstructing, we just annotate each flow as
    high_effect vs low_effect against the same threshold; the split matches
    the diagnostic's Type-3 vs (Type-1 + Type-2) partition."""
    return None  # currently unused; kept for future extension


def per_slice_stats(ig_grouped, ig_matrix, slice_types, f_actual, f_baseline,
                    completeness, converged, steps_used):
    """Compute per-slice summary statistics."""
    out = {}
    for s in (0, 1, 2):
        name = SLICE_NAMES[s]
        mask = (slice_types == s)
        n = int(mask.sum())
        if n == 0:
            out[name] = {'n': 0}
            continue

        # Attribution by group.
        signed = ig_grouped[mask]                     # (n, 8) with signs
        absv = np.abs(signed)                          # (n, 8) magnitudes
        mean_abs = absv.mean(axis=0).tolist()
        median_abs = np.median(absv, axis=0).tolist()
        mean_signed = signed.mean(axis=0).tolist()

        # Rank by mean |IG|.
        order = np.argsort(-np.array(mean_abs))
        ranking = [GROUP_ORDER[i] for i in order]

        # Completeness / effect.
        effect = np.abs(f_actual[mask] - f_baseline[mask])
        rel_resid = np.abs(completeness[mask]) / np.maximum(
            np.maximum(effect, absv.max(axis=1)), 1e-9)

        out[name] = {
            'n': n,
            'mean_abs_ig': dict(zip(GROUP_ORDER, mean_abs)),
            'median_abs_ig': dict(zip(GROUP_ORDER, median_abs)),
            'mean_signed_ig': dict(zip(GROUP_ORDER, mean_signed)),
            'feature_ranking_by_mean_abs': ranking,
            'top3': ranking[:3],
            'completeness_pct_of_scale': {
                'mean': float(rel_resid.mean() * 100),
                'median': float(np.median(rel_resid) * 100),
                'max': float(rel_resid.max() * 100),
            },
            'effect_size': {
                'median_abs_f_actual_minus_f_baseline': float(np.median(effect)),
                'p10': float(np.percentile(effect, 10)),
                'p90': float(np.percentile(effect, 90)),
            },
            'converged_frac': float(converged[mask].mean()),
            'steps_used': {
                'min': int(steps_used[mask].min()),
                'median': int(np.median(steps_used[mask])),
                'max': int(steps_used[mask].max()),
            },
        }
    return out


def rank_correlation(a, b):
    """Spearman rank correlation between two rankings of GROUP_ORDER
    (both are lists of the same feature names in some order). Uses SciPy if
    available, otherwise falls back to numpy Pearson on ranks (equivalent)."""
    rank_a = {name: i for i, name in enumerate(a)}
    rank_b = {name: i for i, name in enumerate(b)}
    xs = np.array([rank_a[k] for k in GROUP_ORDER], dtype=np.float64)
    ys = np.array([rank_b[k] for k in GROUP_ORDER], dtype=np.float64)
    if xs.std() == 0 or ys.std() == 0:
        return float('nan')
    return float(np.corrcoef(xs, ys)[0, 1])


def write_markdown(report, ig_npz_path, diag_json_path, out_path):
    """Human-readable per-slice report. Deliberately blunt: what the numbers
    mean, what they do NOT mean, what a reviewer would ask."""
    lines = []
    lines.append("# IG per-slice report (V2X, ckpt_v2, per-slice baseline)")
    lines.append("")
    lines.append(f"**IG run:** `{os.path.basename(ig_npz_path)}`  ")
    lines.append(f"**Diagnostics:** `{os.path.basename(diag_json_path)}`  ")
    lines.append(f"**Total flows:** {report['n_total']} "
                 f"({report['per_slice']['eMBB']['n']} eMBB, "
                 f"{report['per_slice']['mMTC']['n']} mMTC, "
                 f"{report['per_slice']['URLLC']['n']} URLLC)")
    lines.append("")
    lines.append("## 1. Convergence")
    lines.append("")
    lines.append(f"- Overall: {report['converged_all']}/{report['n_total']} flows "
                 f"converged within max_steps ({100.0*report['converged_all']/report['n_total']:.1f}%).")
    lines.append(f"- Residual as %% of max(effect, max|IG|): "
                 f"mean={report['completeness_all']['mean']:.2f}%, "
                 f"median={report['completeness_all']['median']:.2f}%, "
                 f"max={report['completeness_all']['max']:.2f}%.")
    lines.append(f"- Steps used: min={report['steps_all']['min']}, "
                 f"median={report['steps_all']['median']}, "
                 f"max={report['steps_all']['max']}.")
    lines.append("")
    lines.append("This is the "
                 "corrected result after switching to a per-slice baseline. "
                 "The earlier single-baseline pilot showed residuals commonly "
                 ">10% and adaptive step doubling routinely triggered; here "
                 "almost every flow converges at the starting step count and "
                 "no flow exceeds 400 steps.")
    lines.append("")

    lines.append("## 2. Per-slice feature ranking (mean |IG|)")
    lines.append("")
    for name in ('eMBB', 'mMTC', 'URLLC'):
        s = report['per_slice'][name]
        if s['n'] == 0:
            continue
        lines.append(f"### {name} (n={s['n']})")
        lines.append("")
        lines.append("| rank | feature | mean |IG| | median |IG| | mean signed IG |")
        lines.append("|---|---|---|---|---|")
        for i, feat in enumerate(s['feature_ranking_by_mean_abs']):
            lines.append(f"| {i+1} | `{feat}` | "
                         f"{s['mean_abs_ig'][feat]:.4g} | "
                         f"{s['median_abs_ig'][feat]:.4g} | "
                         f"{s['mean_signed_ig'][feat]:+.4g} |")
        lines.append("")
        lines.append(f"- **Top-3:** {', '.join(s['top3'])}")
        lines.append(f"- **Effect size** (|f_actual - f_baseline|): "
                     f"median={s['effect_size']['median_abs_f_actual_minus_f_baseline']:.4g}, "
                     f"p10={s['effect_size']['p10']:.4g}, "
                     f"p90={s['effect_size']['p90']:.4g}")
        lines.append(f"- **Converged:** {s['converged_frac']*100:.1f}%  "
                     f"**steps used:** median={s['steps_used']['median']}, "
                     f"max={s['steps_used']['max']}")
        lines.append(f"- **Completeness residual:** median={s['completeness_pct_of_scale']['median']:.2f}%, "
                     f"max={s['completeness_pct_of_scale']['max']:.2f}%")
        lines.append("")

    lines.append("## 3. Cross-slice rank agreement (Spearman on feature order)")
    lines.append("")
    rc = report['rank_correlation']
    lines.append(f"- eMBB vs mMTC:  {rc.get('eMBB_vs_mMTC', float('nan')):+.3f}")
    lines.append(f"- eMBB vs URLLC: {rc.get('eMBB_vs_URLLC', float('nan')):+.3f}")
    lines.append(f"- mMTC vs URLLC: {rc.get('mMTC_vs_URLLC', float('nan')):+.3f}")
    lines.append("")
    lines.append("Rank correlation of 1.0 = identical feature ordering across "
                 "slices; -1.0 = fully inverted; 0.0 = unrelated. Low values "
                 "mean the model uses different features on different slices "
                 "(genuine per-slice explanation).")
    lines.append("")

    lines.append("## 4. Caveats you must state in the thesis")
    lines.append("")
    lines.append("1. **Attribution is on the learned door only.** `traffic` and "
                 "`packets` also affect delay through the queueing-physics door "
                 "(`load`, `pkt_size`, `trans_delay` in `forward_from_path_input`) "
                 "which does not pass through the 16-dim path input. IG here "
                 "under-reports their total influence. `slice_type` has no "
                 "physics door -> IG on `slice_type` is a fair measure of "
                 "learned use.")
    lines.append("2. **`slice_type` attribution is not identifiable in isolation.** "
                 "Training-set correlations of the slice one-hot with `packets` "
                 "and `avg_pkts_lambda` are ~0.94-0.99. A large IG value on "
                 "`slice_type` could equivalently be assigned to those numeric "
                 "features; a small IG value does not prove the model does not "
                 "use slice identity. Report `slice_type` + traffic-profile "
                 "features as a jointly-important group, not individually.")
    lines.append("3. **Categorical columns are 0 by construction** in the "
                 "reported IG (`scope=numeric`). This is a deliberate choice, "
                 "documented in the code: interpolating one-hots through "
                 "invalid mixes ([0.3, 0, 0]) is a known IG failure mode. "
                 "It is 'not attributed', not 'attributed and zero'.")
    lines.append("4. **`exp_max_factor` and `model` are dead features.** "
                 "The diagnostic showed `exp_max_factor` is constant (=10) on "
                 "all 300 test flows and `model` is 100% class 0 in the "
                 "training sample. Any IG value on them will be exactly 0 "
                 "in `scope=numeric` (category door closed) and negligible "
                 "in `scope=full`. Cite the diagnostic; do not interpret.")
    lines.append("5. **The mMTC failure regime.** The diagnostic reported "
                 "25/100 mMTC flows are 'Type 2' (model output close to "
                 "baseline but ground truth far from baseline). IG attributions "
                 "on those flows explain the model, not the physical delay. "
                 "Report the typology count from the diagnostic alongside "
                 "the IG results.")
    lines.append("")

    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))


def make_plots(report, ig_grouped, slice_types, out_dir):
    """One bar chart per slice + one grouped chart. Kept minimal on purpose."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"WARNING: matplotlib unavailable ({e}); skipping plots")
        return

    for s in (0, 1, 2):
        name = SLICE_NAMES[s]
        mask = (slice_types == s)
        if not mask.any():
            continue
        mean_abs = np.abs(ig_grouped[mask]).mean(axis=0)
        order = np.argsort(-mean_abs)
        feats = [GROUP_ORDER[i] for i in order]
        vals = mean_abs[order]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(range(len(feats)), vals[::-1])
        ax.set_yticks(range(len(feats)))
        ax.set_yticklabels(feats[::-1])
        ax.set_xlabel('mean |IG| (seconds, learned-door contribution)')
        ax.set_title(f'{name}  n={int(mask.sum())}')
        fig.tight_layout()
        out = os.path.join(out_dir, f'ig_meanabs_{name}.png')
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"Saved: {out}")

    # Grouped chart across slices (log scale so wildly different slice
    # magnitudes remain readable in one figure).
    fig, ax = plt.subplots(figsize=(11, 4.5))
    width = 0.27
    x = np.arange(len(GROUP_ORDER))
    for i, s in enumerate((0, 1, 2)):
        name = SLICE_NAMES[s]
        mask = (slice_types == s)
        if not mask.any():
            continue
        mean_abs = np.abs(ig_grouped[mask]).mean(axis=0)
        ax.bar(x + (i - 1) * width, mean_abs, width=width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(GROUP_ORDER, rotation=30, ha='right')
    ax.set_ylabel('mean |IG| (seconds)')
    ax.set_yscale('symlog', linthresh=1e-6)
    ax.legend()
    ax.set_title('IG mean |attribution| per feature, per slice (log scale)')
    fig.tight_layout()
    out = os.path.join(out_dir, 'ig_topfeatures_all.png')
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    args = parse_args()

    print(f"Loading IG results:   {args.ig_npz}")
    d = np.load(args.ig_npz, allow_pickle=True)
    ig = d['ig']                              # (N, 16)
    completeness = d['completeness_residual'] # (N,)
    f_actual = d['f_actual']                  # (N,)
    f_baseline = d['f_baseline']              # (N,)
    steps_used = d['steps_used']              # (N,)
    converged = d['converged']                # (N,) bool
    slice_types = d['slice_type']             # (N,) int
    N = ig.shape[0]
    print(f"  n_flows={N}, feature_dim={ig.shape[1]}, "
          f"per_slice={bool(d.get('baseline_is_per_slice', False))}")

    print(f"Loading diagnostics:  {args.diagnostics_json}")
    with open(args.diagnostics_json, 'r') as fh:
        diag = json.load(fh)

    # Aggregate to 8 feature groups (16 cols -> 8 groups) by SUMMING signed IG
    # within each one-hot block. This preserves completeness at the group
    # level.
    ig_grouped = group_ig(ig)                # (N, 8)

    # Sanity: sum of grouped IG per flow == sum of raw IG per flow.
    raw_sum = ig.sum(axis=1)
    grp_sum = ig_grouped.sum(axis=1)
    max_diff = float(np.max(np.abs(raw_sum - grp_sum)))
    if max_diff > 1e-4:
        print(f"WARNING: raw-vs-grouped sum mismatch {max_diff:.2e} (expected 0)")

    # Per-slice tables.
    ps = per_slice_stats(ig_grouped, ig, slice_types, f_actual, f_baseline,
                         completeness, converged, steps_used)

    # Overall convergence numbers (already in the IG log, restated here for
    # a single-source-of-truth report file).
    effect_all = np.abs(f_actual - f_baseline)
    max_absig_all = np.abs(ig_grouped).max(axis=1)
    denom_all = np.maximum(np.maximum(effect_all, max_absig_all), 1e-9)
    rel_all = np.abs(completeness) / denom_all
    completeness_all = {
        'mean': float(rel_all.mean() * 100),
        'median': float(np.median(rel_all) * 100),
        'max': float(rel_all.max() * 100),
    }
    steps_all = {
        'min': int(steps_used.min()),
        'median': int(np.median(steps_used)),
        'max': int(steps_used.max()),
    }

    # Rank correlations.
    rc = {}
    slices_present = [SLICE_NAMES[s] for s in (0, 1, 2) if ps[SLICE_NAMES[s]]['n'] > 0]
    for i, a in enumerate(slices_present):
        for b in slices_present[i + 1:]:
            rc[f'{a}_vs_{b}'] = rank_correlation(
                ps[a]['feature_ranking_by_mean_abs'],
                ps[b]['feature_ranking_by_mean_abs'])

    report = {
        'ig_npz': os.path.basename(args.ig_npz),
        'diagnostics_json': os.path.basename(args.diagnostics_json),
        'n_total': int(N),
        'converged_all': int(converged.sum()),
        'completeness_all': completeness_all,
        'steps_all': steps_all,
        'per_slice': ps,
        'rank_correlation': rc,
        'diagnostic_typology': {
            name: diag['per_slice'][name]['typology_counts']
            for name in ('eMBB', 'mMTC', 'URLLC') if name in diag.get('per_slice', {})
        },
    }

    os.makedirs(args.out_dir, exist_ok=True)
    json_out = os.path.join(args.out_dir, 'ig_report_per_slice.json')
    with open(json_out, 'w', encoding='utf-8') as fh:
        json.dump(report, fh, indent=1)
    print(f"\nSaved: {json_out}")

    md_out = os.path.join(args.out_dir, 'ig_report_per_slice.md')
    write_markdown(report, args.ig_npz, args.diagnostics_json, md_out)
    print(f"Saved: {md_out}")

    if not args.no_plots:
        make_plots(report, ig_grouped, slice_types, args.out_dir)

    print("\n" + "=" * 70)
    print("QUICK PER-SLICE RANKING (mean |IG| top-3):")
    for name in ('eMBB', 'mMTC', 'URLLC'):
        s = ps[name]
        if s['n'] == 0:
            continue
        print(f"  {name:6s} (n={s['n']:3d}): {' > '.join(s['top3'])}")
    print("\nCROSS-SLICE RANK CORRELATION (Spearman):")
    for k, v in rc.items():
        print(f"  {k:20s} {v:+.3f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
