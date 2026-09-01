"""
XAI phase - three-method cross-comparison (final step of Option 1).

Reads:
  - slicing/xai/results/ig_shared_300_perslice.npz
  - slicing/xai/results/shap_shared_300_grouped_part{0,1,2}.npz  (auto-merged)
  - slicing/xai/results/perm_shared_300.npz  (+ perm_shared_300.json)

Produces (in --out-dir, default slicing/xai/results/):
  - shap_shared_300_grouped.npz            merged SHAP file, same schema as parts
  - three_methods_comparison.json          machine-readable summary
  - three_methods_comparison.md            human-readable comparison report
  - three_methods_topfeatures_<slice>.png  bar chart per slice (3 methods per bar)
  - three_methods_rank_agreement.png       pairwise Spearman heatmap per slice

WHAT COUNTS AS "AGREEMENT"
--------------------------
Each of the three methods produces, per slice, a ranking of features by
importance. IG ranks 6 numeric features (categorical columns pinned, so 0);
SHAP ranks 8 grouped features; permutation ranks the same 8. To compare on
common ground we intersect on the 6 features IG can attribute
(traffic, packets, eq_lambda, avg_pkts_lambda, exp_max_factor, delta) and
also report SHAP-vs-permutation agreement on all 8. Spearman rank
correlation is the standard metric.

DELIBERATE COMPARISON DESIGN
----------------------------
- IG vs SHAP on 6 numeric features: same 300 flows, same per-slice baseline,
  the ONLY difference is the attribution mechanism. Direct like-for-like.
- SHAP vs permutation on 8 features: SHAP is per-flow aggregated per slice
  (mean |phi|); permutation is per-slice global (delta MAE from shuffling).
  Both handle categoricals natively. They MEASURE different things but should
  agree on ranking if the model's dependence is well-defined.
- IG vs permutation: constrained to the 6 numeric features. Two very
  different lenses; agreement here is a strong triangulation.

Runs locally, numpy + matplotlib only. No TF.
"""
import argparse
import glob
import json
import os

import numpy as np


# The 8 groups in the order xai_shap.py and xai_perm.py both use.
# For IG we aggregate its 16-col output into these same groups (as
# xai_report_per_slice.py does).
GROUP_NAMES_8 = ['traffic', 'packets', 'eq_lambda', 'avg_pkts_lambda',
                 'exp_max_factor', 'delta', 'slice_type', 'model']
# For IG-comparable subset (numeric only):
GROUP_NAMES_6 = ['traffic', 'packets', 'eq_lambda', 'avg_pkts_lambda',
                 'exp_max_factor', 'delta']
# For aggregating IG 16-col matrix into 8 groups.
FEATURE_GROUPS = {
    'traffic': [0], 'packets': [1],
    'model': list(range(2, 9)),
    'eq_lambda': [9], 'avg_pkts_lambda': [10], 'exp_max_factor': [11],
    'slice_type': [12, 13, 14],
    'delta': [15],
}
SLICE_NAMES = ['eMBB', 'mMTC', 'URLLC']


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    slic = os.path.dirname(here)
    default_results = os.path.join(slic, 'xai', 'results')
    ap = argparse.ArgumentParser()
    ap.add_argument('--ig-npz', default=os.path.join(default_results,
                                                     'ig_shared_300_perslice.npz'))
    ap.add_argument('--shap-parts-glob',
                    default=os.path.join(default_results,
                                         'shap_shared_300_grouped_part*.npz'))
    ap.add_argument('--perm-npz', default=os.path.join(default_results,
                                                      'perm_shared_300.npz'))
    ap.add_argument('--out-dir', default=default_results)
    ap.add_argument('--no-plots', action='store_true')
    return ap.parse_args()


def merge_shap_parts(paths, out_path):
    """Concatenate the three SHAP part npz files (matching flow order in the
    shared set) into a single npz with the same schema."""
    paths = sorted(paths)
    print(f"Merging {len(paths)} SHAP parts:")
    parts = []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        parts.append(d)
        print(f"  {os.path.basename(p)}  n_flows={d['phi_grouped'].shape[0]}")

    merged = {}
    for key in parts[0].files:
        arr0 = parts[0][key]
        if arr0.ndim == 0:
            # scalar/metadata - use the value from part0, sanity-check same
            for p in parts[1:]:
                if not np.array_equal(p[key], arr0):
                    # non-matching scalars are ok for progress-related fields,
                    # keep part0's copy
                    pass
            merged[key] = arr0
        else:
            merged[key] = np.concatenate([p[key] for p in parts], axis=0)
    np.savez(out_path, **merged)
    print(f"  merged n_flows={merged['phi_grouped'].shape[0]} -> {out_path}")
    return merged


def ig_group_stats(ig_npz):
    """Aggregate IG's (N, 16) into (N, 8) grouped absolute values, split by
    slice. Return {slice_name: mean_abs_per_group_dict}."""
    d = np.load(ig_npz, allow_pickle=True)
    ig = d['ig']
    st = d['slice_type']
    grouped = np.zeros((ig.shape[0], len(GROUP_NAMES_8)), dtype=np.float64)
    for j, name in enumerate(GROUP_NAMES_8):
        cols = FEATURE_GROUPS[name]
        grouped[:, j] = ig[:, cols].sum(axis=1)  # signed sum within group
    out = {}
    for s_int, s_name in enumerate(SLICE_NAMES):
        m = (st == s_int)
        if not m.any():
            continue
        mean_abs = np.abs(grouped[m]).mean(axis=0)
        out[s_name] = dict(zip(GROUP_NAMES_8, mean_abs.tolist()))
    return out


def shap_group_stats(shap_merged):
    """Return {slice_name: mean_abs_per_group_dict} from merged SHAP."""
    phi = shap_merged['phi_grouped']
    st = shap_merged['slice_type']
    group_names = list(shap_merged['group_names'])
    out = {}
    for s_int, s_name in enumerate(SLICE_NAMES):
        m = (st == s_int)
        if not m.any():
            continue
        mean_abs = np.abs(phi[m]).mean(axis=0)
        # ensure GROUP_NAMES_8 ordering
        idx_map = [group_names.index(n) for n in GROUP_NAMES_8]
        out[s_name] = dict(zip(GROUP_NAMES_8, mean_abs[idx_map].tolist()))
    return out


def perm_group_stats(perm_npz):
    """Return {slice_name: importance_dict} from permutation .npz."""
    d = np.load(perm_npz, allow_pickle=True)
    group_names = list(d['group_names'])
    out = {}
    for s_name in SLICE_NAMES:
        key = f'importance_per_slice_{s_name}'
        if key not in d.files:
            continue
        arr = d[key]  # (n_groups, n_repeats)
        mean_ = arr.mean(axis=1)
        idx_map = [group_names.index(n) for n in GROUP_NAMES_8]
        # Permutation importance can be negative if shuffling improves
        # accuracy by luck; take mean directly, not abs, so a truly-null
        # feature stays at ~0 rather than getting inflated by noise.
        out[s_name] = dict(zip(GROUP_NAMES_8, mean_[idx_map].tolist()))
    return out


def rank_by_score(score_dict, subset=None):
    """Rank feature names by absolute score, high to low. If subset given,
    restrict to those features."""
    items = [(k, abs(v)) for k, v in score_dict.items()
             if subset is None or k in subset]
    items.sort(key=lambda x: -x[1])
    return [k for k, _ in items]


def spearman_of_rankings(a, b):
    """Spearman rank correlation between two orderings of the same feature set."""
    if set(a) != set(b):
        raise ValueError("rankings must be over the same feature set")
    n = len(a)
    if n < 2:
        return float('nan')
    ra = {name: i for i, name in enumerate(a)}
    rb = {name: i for i, name in enumerate(b)}
    xs = np.array([ra[k] for k in a], dtype=np.float64)
    ys = np.array([rb[k] for k in a], dtype=np.float64)
    if xs.std() == 0 or ys.std() == 0:
        return float('nan')
    return float(np.corrcoef(xs, ys)[0, 1])


def build_report(ig_stats, shap_stats, perm_stats):
    """Assemble the per-slice comparison record."""
    per_slice = {}
    for s_name in SLICE_NAMES:
        if s_name not in ig_stats or s_name not in shap_stats or \
           s_name not in perm_stats:
            continue
        ig_scores = ig_stats[s_name]
        shap_scores = shap_stats[s_name]
        perm_scores = perm_stats[s_name]

        rank_ig_6 = rank_by_score(ig_scores, GROUP_NAMES_6)
        rank_shap_6 = rank_by_score(shap_scores, GROUP_NAMES_6)
        rank_shap_8 = rank_by_score(shap_scores, GROUP_NAMES_8)
        rank_perm_6 = rank_by_score(perm_scores, GROUP_NAMES_6)
        rank_perm_8 = rank_by_score(perm_scores, GROUP_NAMES_8)

        per_slice[s_name] = {
            'scores': {
                'IG': {k: ig_scores[k] for k in GROUP_NAMES_8},
                'SHAP': {k: shap_scores[k] for k in GROUP_NAMES_8},
                'Permutation': {k: perm_scores[k] for k in GROUP_NAMES_8},
            },
            'ranking_6feat': {
                'IG': rank_ig_6,
                'SHAP': rank_shap_6,
                'Permutation': rank_perm_6,
            },
            'ranking_8feat': {
                'SHAP': rank_shap_8,
                'Permutation': rank_perm_8,
            },
            'rank_correlation_6feat': {
                'IG_vs_SHAP': spearman_of_rankings(rank_ig_6, rank_shap_6),
                'IG_vs_Permutation': spearman_of_rankings(rank_ig_6, rank_perm_6),
                'SHAP_vs_Permutation': spearman_of_rankings(rank_shap_6, rank_perm_6),
            },
            'rank_correlation_8feat': {
                'SHAP_vs_Permutation': spearman_of_rankings(rank_shap_8, rank_perm_8),
            },
            'top3_by_method': {
                'IG_6': rank_ig_6[:3],
                'SHAP_6': rank_shap_6[:3],
                'Permutation_6': rank_perm_6[:3],
                'SHAP_8': rank_shap_8[:3],
                'Permutation_8': rank_perm_8[:3],
            },
        }
    return per_slice


def write_markdown(per_slice, out_path):
    lines = []
    lines.append("# Three-method cross-comparison (IG vs KernelSHAP vs Permutation Importance)")
    lines.append("")
    lines.append("Reports per-slice feature rankings from each XAI method and pairwise Spearman rank correlations.")
    lines.append("")
    lines.append("## Method scopes recap")
    lines.append("")
    lines.append("| Method | # features attributed | Handles categoricals? | Baseline / background | Level |")
    lines.append("|---|---|---|---|---|")
    lines.append("| IG | 6 (numeric only) | No -- interpolation invalid on one-hots | per-slice training median | per-flow, all 300 |")
    lines.append("| KernelSHAP grouped | 8 (adds slice_type + model) | Yes -- whole one-hot block swapped, never fractional | numerics = per-slice median; slice_type = 3-way empirical-frequency average; model = single-point (100% class 0) | per-flow, all 300 |")
    lines.append("| Permutation Importance | 8 | Yes -- whole one-hot block shuffled | none (measured against baseline model error) | global, 300 flows, 5 repeats |")
    lines.append("")
    lines.append("The 8 semantic groups are always: `traffic, packets, eq_lambda, avg_pkts_lambda, exp_max_factor, delta, slice_type, model`.")
    lines.append("")
    lines.append("Two data-degenerate features (`model` = 100% class 0; `exp_max_factor` = constant 10) are included as positive controls -- any XAI method returning non-zero on either would indicate a methodological artifact.")
    lines.append("")

    for s_name in SLICE_NAMES:
        if s_name not in per_slice:
            continue
        s = per_slice[s_name]
        lines.append(f"## {s_name}")
        lines.append("")
        lines.append("### Feature scores (higher = more important)")
        lines.append("")
        lines.append("| feature | IG mean\\|attribution\\| | SHAP mean\\|phi\\| | Permutation ΔMAE |")
        lines.append("|---|---|---|---|")
        for f in GROUP_NAMES_8:
            ig_v = s['scores']['IG'][f]
            sh_v = s['scores']['SHAP'][f]
            pm_v = s['scores']['Permutation'][f]
            lines.append(f"| `{f}` | {ig_v:.4g} | {sh_v:.4g} | {pm_v:+.4g} |")
        lines.append("")

        lines.append("### Rankings")
        lines.append("")
        lines.append("| Method | on 6 numeric features | on all 8 features |")
        lines.append("|---|---|---|")
        lines.append(f"| IG          | {' > '.join(s['ranking_6feat']['IG'])} | *(not applicable -- IG cannot rank categoricals)* |")
        lines.append(f"| SHAP        | {' > '.join(s['ranking_6feat']['SHAP'])} | {' > '.join(s['ranking_8feat']['SHAP'])} |")
        lines.append(f"| Permutation | {' > '.join(s['ranking_6feat']['Permutation'])} | {' > '.join(s['ranking_8feat']['Permutation'])} |")
        lines.append("")

        lines.append("### Rank agreement (Spearman)")
        lines.append("")
        rc6 = s['rank_correlation_6feat']
        rc8 = s['rank_correlation_8feat']
        lines.append(f"- **On 6 numeric features:**")
        lines.append(f"  - IG vs SHAP: **{rc6['IG_vs_SHAP']:+.3f}**")
        lines.append(f"  - IG vs Permutation: **{rc6['IG_vs_Permutation']:+.3f}**")
        lines.append(f"  - SHAP vs Permutation: **{rc6['SHAP_vs_Permutation']:+.3f}**")
        lines.append(f"- **On 8 features (SHAP + Permutation only):**")
        lines.append(f"  - SHAP vs Permutation: **{rc8['SHAP_vs_Permutation']:+.3f}**")
        lines.append("")
        lines.append(f"Top-3 by method: IG={s['top3_by_method']['IG_6']}, SHAP-8={s['top3_by_method']['SHAP_8']}, Perm-8={s['top3_by_method']['Permutation_8']}")
        lines.append("")

    lines.append("## Interpretation notes for the thesis")
    lines.append("")
    lines.append("1. **Rank correlation > 0.8 between two methods = strong agreement**; that is stronger evidence than any single method could give on its own. If all three agree, the ranking is robust to method choice.")
    lines.append("2. **Permutation importance under-estimates correlated features.** In this dataset `slice_type` correlates with `packets` / `avg_pkts_lambda` at r ~ 0.94-0.99 on training. Shuffling `slice_type` alone leaves the model able to reconstruct much of the signal from `packets`, so permutation `slice_type` importance is a LOWER BOUND on the model's true dependence. SHAP with grouped categoricals does not have this pathology.")
    lines.append("3. **`model` and `exp_max_factor` should be ~0 in every method.** If either is non-negligible in any method, that method has a bug/artifact.")
    lines.append("4. **IG can only rank 6 features.** The `slice_type` and `model` rows in the IG column of the score tables are 0 by construction (documented scope choice), not by the method finding them unimportant.")
    lines.append("")

    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))


def make_plots(per_slice, out_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"WARNING: matplotlib unavailable ({e}); skipping plots")
        return

    # Per-slice bar chart: 3 grouped bars per feature (one per method)
    # Normalize each method's scores to [0,1] within the slice so they're
    # visually comparable despite wildly different absolute magnitudes.
    for s_name, s in per_slice.items():
        methods = ['IG', 'SHAP', 'Permutation']
        raw_matrix = np.zeros((len(GROUP_NAMES_8), 3))
        for j, feat in enumerate(GROUP_NAMES_8):
            for k, m in enumerate(methods):
                raw_matrix[j, k] = abs(s['scores'][m][feat])
        # Normalize per method (column) so each method's bars max out at 1.
        norm = raw_matrix.copy()
        for k in range(3):
            m = norm[:, k].max()
            if m > 0:
                norm[:, k] = norm[:, k] / m

        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(len(GROUP_NAMES_8))
        width = 0.27
        for k, m in enumerate(methods):
            ax.bar(x + (k - 1) * width, norm[:, k], width=width, label=m)
        ax.set_xticks(x)
        ax.set_xticklabels(GROUP_NAMES_8, rotation=30, ha='right')
        ax.set_ylabel('Importance (normalized per method to max=1)')
        ax.set_title(f'{s_name}: three-method comparison (300 flows)')
        ax.legend()
        fig.tight_layout()
        out = os.path.join(out_dir, f'three_methods_topfeatures_{s_name}.png')
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"Saved: {out}")

    # Rank-agreement heatmap: rows = slices, cols = pairs, values = Spearman.
    pairs = ['IG_vs_SHAP', 'IG_vs_Permutation', 'SHAP_vs_Permutation']
    fig, ax = plt.subplots(figsize=(7, 3.2))
    data = np.zeros((len(SLICE_NAMES), len(pairs)))
    for i, s_name in enumerate(SLICE_NAMES):
        if s_name not in per_slice:
            continue
        rc = per_slice[s_name]['rank_correlation_6feat']
        for j, p in enumerate(pairs):
            data[i, j] = rc[p]
    im = ax.imshow(data, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(pairs, rotation=15, ha='right')
    ax.set_yticks(range(len(SLICE_NAMES)))
    ax.set_yticklabels(SLICE_NAMES)
    for i in range(len(SLICE_NAMES)):
        for j in range(len(pairs)):
            ax.text(j, i, f'{data[i, j]:+.2f}',
                    ha='center', va='center',
                    color='black' if abs(data[i, j]) < 0.5 else 'white')
    ax.set_title('Rank agreement (Spearman) on 6 numeric features')
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    out = os.path.join(out_dir, 'three_methods_rank_agreement.png')
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    args = parse_args()

    # Merge SHAP parts into one .npz.
    shap_parts = sorted(glob.glob(args.shap_parts_glob))
    if not shap_parts:
        raise SystemExit(f"ERROR: no SHAP parts matching {args.shap_parts_glob}")
    merged_path = os.path.join(args.out_dir, 'shap_shared_300_grouped.npz')
    shap_merged = merge_shap_parts(shap_parts, merged_path)

    # Load per-slice stats from each method.
    print(f"\nLoading IG:  {args.ig_npz}")
    ig_stats = ig_group_stats(args.ig_npz)
    print(f"Loading SHAP: {merged_path}")
    shap_stats = shap_group_stats(shap_merged)
    print(f"Loading Perm: {args.perm_npz}")
    perm_stats = perm_group_stats(args.perm_npz)

    per_slice = build_report(ig_stats, shap_stats, perm_stats)

    os.makedirs(args.out_dir, exist_ok=True)
    json_out = os.path.join(args.out_dir, 'three_methods_comparison.json')
    with open(json_out, 'w', encoding='utf-8') as fh:
        json.dump({'per_slice': per_slice}, fh, indent=1)
    print(f"\nSaved: {json_out}")

    md_out = os.path.join(args.out_dir, 'three_methods_comparison.md')
    write_markdown(per_slice, md_out)
    print(f"Saved: {md_out}")

    if not args.no_plots:
        make_plots(per_slice, args.out_dir)

    # Terminal summary.
    print("\n" + "=" * 78)
    print("SUMMARY -- rank agreement (Spearman) on 6 numeric features")
    print("=" * 78)
    for s_name in SLICE_NAMES:
        if s_name not in per_slice:
            continue
        rc = per_slice[s_name]['rank_correlation_6feat']
        print(f"  {s_name}:")
        print(f"    IG   vs SHAP        : {rc['IG_vs_SHAP']:+.3f}")
        print(f"    IG   vs Permutation : {rc['IG_vs_Permutation']:+.3f}")
        print(f"    SHAP vs Permutation : {rc['SHAP_vs_Permutation']:+.3f}")
    print("=" * 78)
    print("TOP-3 PER METHOD PER SLICE")
    print("=" * 78)
    for s_name in SLICE_NAMES:
        if s_name not in per_slice:
            continue
        t = per_slice[s_name]['top3_by_method']
        print(f"  {s_name}:")
        print(f"    IG  (6 feat): {' > '.join(t['IG_6'])}")
        print(f"    SHAP (8feat): {' > '.join(t['SHAP_8'])}")
        print(f"    Perm (8feat): {' > '.join(t['Permutation_8'])}")
    print("=" * 78)


if __name__ == '__main__':
    main()
