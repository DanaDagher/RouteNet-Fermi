"""
Extra figures for the final XAI report and the August-consolidated sprint report.

Reads:
  slicing/xai/results/ig_shared_300_perslice.npz
  slicing/xai/results/shap_shared_300_grouped.npz
  slicing/xai/results/perm_shared_300.npz
  slicing/xai/results/slice_swap_shared_300.npz
  slicing/xai/results/audit_diagnostics.json

Writes (all PNG, dpi 120, publication-ready sizing):
  slicing/xai/results/fig_sliceswap_violin.png
  slicing/xai/results/fig_sliceswap_directional_heatmap.png
  slicing/xai/results/fig_urllc_dashboard.png
  slicing/xai/results/fig_method_agreement.png
  slicing/xai/results/fig_ig_completeness_hist.png
  slicing/xai/results/fig_typology_stacked.png

Locally, no GPU, numpy + matplotlib only.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'results')

SLICE_NAMES = ['eMBB', 'mMTC', 'URLLC']
SLICE_COLORS = {'eMBB': '#4C72B0', 'mMTC': '#DD8452', 'URLLC': '#55A868'}
METHOD_COLORS = {'IG': '#4C72B0', 'SHAP': '#DD8452', 'Permutation': '#55A868',
                 'Slice-swap': '#C44E52'}

GROUP_NAMES_8 = ['traffic', 'packets', 'eq_lambda', 'avg_pkts_lambda',
                 'exp_max_factor', 'delta', 'slice_type', 'model']
FEATURE_GROUPS = {
    'traffic': [0], 'packets': [1],
    'model': list(range(2, 9)),
    'eq_lambda': [9], 'avg_pkts_lambda': [10], 'exp_max_factor': [11],
    'slice_type': [12, 13, 14],
    'delta': [15],
}


def load_all():
    d = {}
    d['ig'] = np.load(os.path.join(RESULTS_DIR, 'ig_shared_300_perslice.npz'),
                       allow_pickle=True)
    d['shap'] = np.load(os.path.join(RESULTS_DIR, 'shap_shared_300_grouped.npz'),
                        allow_pickle=True)
    d['perm'] = np.load(os.path.join(RESULTS_DIR, 'perm_shared_300.npz'),
                        allow_pickle=True)
    d['swap'] = np.load(os.path.join(RESULTS_DIR, 'slice_swap_shared_300.npz'),
                        allow_pickle=True)
    with open(os.path.join(RESULTS_DIR, 'audit_diagnostics.json')) as fh:
        d['diag'] = json.load(fh)
    return d


def fig_sliceswap_violin(d, out):
    """Per-source-slice violin plot of relative |Δf| across the 2 target swaps.
    Horizontal reference line at 10%, marker for medians. Log-y so the tails
    are visible without hiding the median."""
    swap = d['swap']
    delta, sa, fa = swap['delta'], swap['slice_actual'], swap['f_actual']

    per_slice_vals = []
    for s in range(3):
        mask = (sa == s)
        others = [c for c in range(3) if c != s]
        abs_d = np.abs(delta[mask][:, others])
        f_col = np.abs(fa[mask])[:, None]  # shape (n, 1) to broadcast with (n, 2)
        f_safe = np.where(f_col > 1e-12, f_col, 1.0)
        rel = abs_d / f_safe
        per_slice_vals.append(rel.flatten() * 100)  # percent

    fig, ax = plt.subplots(figsize=(8.5, 5))
    parts = ax.violinplot(per_slice_vals, showmedians=True,
                          showextrema=False, widths=0.75)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(SLICE_COLORS[SLICE_NAMES[i]])
        pc.set_alpha(0.55)
        pc.set_edgecolor('black')
    if 'cmedians' in parts:
        parts['cmedians'].set_color('black')
    ax.set_yscale('log')
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([f"{n}\n(n=100)" for n in SLICE_NAMES])
    ax.set_ylabel('|Δf| / |f_actual| (%, log scale)')
    ax.set_title('Slice-swap response: relative shift per flow, per source slice')
    ax.axhline(10, ls='--', color='red', alpha=0.6,
               label='10% reactivity threshold')
    ax.legend(loc='upper right')
    ax.grid(True, which='major', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved: {out}")


def fig_sliceswap_directional_heatmap(d, out):
    """3x3 heatmap of signed median relative delta, source slice on rows,
    target slice on columns. Diagonal (self) shown as 'self' text."""
    swap = d['swap']
    delta, sa, fa = swap['delta'], swap['slice_actual'], swap['f_actual']

    grid = np.zeros((3, 3))
    labels = np.empty((3, 3), dtype=object)
    for s_src in range(3):
        mask = (sa == s_src)
        for s_tgt in range(3):
            if s_tgt == s_src:
                grid[s_src, s_tgt] = np.nan
                labels[s_src, s_tgt] = 'self'
            else:
                d_col = delta[mask][:, s_tgt]
                f_col = fa[mask]
                rel = np.where(np.abs(f_col) > 1e-12,
                               d_col / f_col, 0.0) * 100
                med = float(np.median(rel))
                grid[s_src, s_tgt] = med
                labels[s_src, s_tgt] = f'{med:+.1f}%'

    fig, ax = plt.subplots(figsize=(6.5, 5))
    vmax = np.nanmax(np.abs(grid))
    im = ax.imshow(grid, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
    ax.set_xticks(range(3))
    ax.set_xticklabels(SLICE_NAMES)
    ax.set_yticks(range(3))
    ax.set_yticklabels(SLICE_NAMES)
    ax.set_xlabel('Target slice (swap TO this label)')
    ax.set_ylabel('Source slice (flow is really this)')
    ax.set_title('Directional slice-swap response\nmedian signed Δf / f_actual (300 flows)')
    for i in range(3):
        for j in range(3):
            color = 'white' if not np.isnan(grid[i, j]) and abs(grid[i, j]) > vmax * 0.5 else 'black'
            ax.text(j, i, labels[i, j], ha='center', va='center',
                    color=color, fontweight='bold' if labels[i, j] != 'self' else 'normal')
    fig.colorbar(im, ax=ax, shrink=0.75, label='median relative Δf (%)')
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved: {out}")


def fig_urllc_dashboard(d, out):
    """Four-panel URLLC-specific figure:
      A. Typology counts across the three slices (URLLC highlighted).
      B. URLLC top-3 features per method (bar chart).
      C. URLLC slice-swap deltas (violin) per target.
      D. URLLC f_actual vs f_baseline scatter (log-log)."""
    diag = d['diag']
    ig, shap, perm, swap = d['ig'], d['shap'], d['perm'], d['swap']

    fig, axs = plt.subplots(2, 2, figsize=(13, 9))

    # Panel A: typology per slice, URLLC highlighted
    types = ['type1_typical', 'type2_model_collapsed', 'type3_high_effect']
    type_labels = ['Type 1\n(typical)', 'Type 2\n(collapsed)', 'Type 3\n(high effect)']
    values = np.array([[diag['per_slice'][sn]['typology_counts'][t]
                        for t in types] for sn in SLICE_NAMES])
    x = np.arange(3)
    width = 0.27
    ax = axs[0, 0]
    for i, sn in enumerate(SLICE_NAMES):
        alpha = 1.0 if sn == 'URLLC' else 0.35
        edge = 'red' if sn == 'URLLC' else 'none'
        lw = 2.5 if sn == 'URLLC' else 0
        ax.bar(x + (i - 1) * width, values[i], width=width,
               label=sn, color=SLICE_COLORS[sn], alpha=alpha,
               edgecolor=edge, linewidth=lw)
    ax.set_xticks(x)
    ax.set_xticklabels(type_labels)
    ax.set_ylabel('Flow count (out of 100)')
    ax.set_title('A. Type 1/2/3 breakdown (URLLC highlighted)')
    ax.legend(loc='upper left')
    ax.grid(True, axis='y', alpha=0.3)

    # Panel B: URLLC top-features across methods (log scale, normalized per method)
    ax = axs[0, 1]
    # IG
    ig_grouped = np.zeros((ig['ig'].shape[0], 8))
    for j, name in enumerate(GROUP_NAMES_8):
        ig_grouped[:, j] = ig['ig'][:, FEATURE_GROUPS[name]].sum(axis=1)
    urllc_mask_ig = (ig['slice_type'] == 2)
    ig_mean_abs = np.abs(ig_grouped[urllc_mask_ig]).mean(axis=0)
    # SHAP
    urllc_mask_shap = (shap['slice_type'] == 2)
    shap_group_names = list(shap['group_names'])
    idx_map = [shap_group_names.index(n) for n in GROUP_NAMES_8]
    shap_mean_abs = np.abs(shap['phi_grouped'][urllc_mask_shap]).mean(axis=0)[idx_map]
    # Perm
    perm_group_names = list(perm['group_names'])
    idx_map_p = [perm_group_names.index(n) for n in GROUP_NAMES_8]
    perm_urllc = perm['importance_per_slice_URLLC'].mean(axis=1)[idx_map_p]

    # Normalize per method to max = 1 for comparability
    def norm(v):
        m = np.max(np.abs(v))
        return v / m if m > 0 else v
    ig_n = norm(np.abs(ig_mean_abs))
    shap_n = norm(np.abs(shap_mean_abs))
    perm_n = norm(np.abs(perm_urllc))

    x8 = np.arange(len(GROUP_NAMES_8))
    w = 0.27
    ax.bar(x8 - w, ig_n, w, label='IG', color=METHOD_COLORS['IG'])
    ax.bar(x8, shap_n, w, label='SHAP', color=METHOD_COLORS['SHAP'])
    ax.bar(x8 + w, perm_n, w, label='Permutation', color=METHOD_COLORS['Permutation'])
    ax.set_xticks(x8)
    ax.set_xticklabels(GROUP_NAMES_8, rotation=30, ha='right')
    ax.set_ylabel('Importance (normalized per method, max=1)')
    ax.set_title('B. URLLC: top features per XAI method')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    # Panel C: URLLC slice-swap absolute delta, per target slice
    urllc_mask = swap['slice_actual'] == 2
    d_urllc = swap['delta'][urllc_mask]
    fa_urllc = swap['f_actual'][urllc_mask]
    ax = axs[1, 0]
    parts = ax.violinplot(
        [np.abs(d_urllc[:, 0]) / np.abs(fa_urllc) * 100,
         np.abs(d_urllc[:, 1]) / np.abs(fa_urllc) * 100],
        showmedians=True, showextrema=False, widths=0.7)
    for i, pc in enumerate(parts['bodies']):
        color = SLICE_COLORS[['eMBB', 'mMTC'][i]]
        pc.set_facecolor(color); pc.set_alpha(0.55); pc.set_edgecolor('black')
    if 'cmedians' in parts:
        parts['cmedians'].set_color('black')
    ax.set_yscale('log')
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['URLLC → eMBB', 'URLLC → mMTC'])
    ax.set_ylabel('|Δf| / |f_actual| (%, log scale)')
    ax.set_title('C. URLLC slice-swap response per target')
    ax.axhline(10, ls='--', color='red', alpha=0.6, label='10% threshold')
    ax.legend()
    ax.grid(True, which='major', alpha=0.3)

    # Panel D: URLLC f_actual vs f_baseline scatter (log-log)
    urllc_ig = ig['slice_type'] == 2
    fa_ig = ig['f_actual'][urllc_ig]
    fb_ig = ig['f_baseline'][urllc_ig]
    ax = axs[1, 1]
    ax.loglog(fb_ig, fa_ig, 'o', color=SLICE_COLORS['URLLC'], alpha=0.5,
              markersize=6, markeredgecolor='black', markeredgewidth=0.4)
    lo = min(fb_ig.min(), fa_ig.min()) * 0.5
    hi = max(fb_ig.max(), fa_ig.max()) * 2
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.4, label='y = x')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel('f_baseline (per-slice baseline row prediction, s)')
    ax.set_ylabel('f_actual (real-input prediction, s)')
    ax.set_title('D. URLLC: model prediction vs per-slice baseline')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

    fig.suptitle('URLLC deep-dive (n = 100 URLLC flows)', fontsize=14, y=1.00)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved: {out}")


def fig_method_agreement(d, out):
    """Bar chart of pairwise Spearman rank correlation on 6 numeric features,
    per slice. Three groups of three bars each."""
    # Compute rank correlations locally (mirrors xai_compare_three_methods.py).
    ig, shap, perm = d['ig'], d['shap'], d['perm']
    GROUP_NAMES_6 = ['traffic', 'packets', 'eq_lambda', 'avg_pkts_lambda',
                     'exp_max_factor', 'delta']

    ig_grouped = np.zeros((ig['ig'].shape[0], 8))
    for j, name in enumerate(GROUP_NAMES_8):
        ig_grouped[:, j] = ig['ig'][:, FEATURE_GROUPS[name]].sum(axis=1)
    shap_group_names = list(shap['group_names'])
    idx_shap = [shap_group_names.index(n) for n in GROUP_NAMES_8]
    perm_group_names = list(perm['group_names'])
    idx_perm = [perm_group_names.index(n) for n in GROUP_NAMES_8]

    def rank(scores_dict):
        items = [(k, abs(scores_dict[k])) for k in GROUP_NAMES_6]
        items.sort(key=lambda x: -x[1])
        return [k for k, _ in items]

    def spearman(a, b):
        n = len(a)
        ra = {name: i for i, name in enumerate(a)}
        rb = {name: i for i, name in enumerate(b)}
        xs = np.array([ra[k] for k in a], dtype=np.float64)
        ys = np.array([rb[k] for k in a], dtype=np.float64)
        if xs.std() == 0 or ys.std() == 0:
            return float('nan')
        return float(np.corrcoef(xs, ys)[0, 1])

    pairs = ['IG vs SHAP', 'IG vs Perm', 'SHAP vs Perm']
    per_slice_rc = {}
    for s_int, s_name in enumerate(SLICE_NAMES):
        ig_m = (ig['slice_type'] == s_int)
        sh_m = (shap['slice_type'] == s_int)
        ig_mean = dict(zip(GROUP_NAMES_8,
                           np.abs(ig_grouped[ig_m]).mean(axis=0)))
        shap_mean = dict(zip(GROUP_NAMES_8,
                             np.abs(shap['phi_grouped'][sh_m]).mean(axis=0)[idx_shap]))
        perm_mean = dict(zip(GROUP_NAMES_8,
                             perm[f'importance_per_slice_{s_name}'].mean(axis=1)[idx_perm]))
        r_ig = rank(ig_mean); r_shap = rank(shap_mean); r_perm = rank(perm_mean)
        per_slice_rc[s_name] = [spearman(r_ig, r_shap),
                                 spearman(r_ig, r_perm),
                                 spearman(r_shap, r_perm)]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(SLICE_NAMES))
    w = 0.27
    colors = ['#3B82F6', '#EF4444', '#10B981']
    for i, pair in enumerate(pairs):
        vals = [per_slice_rc[sn][i] for sn in SLICE_NAMES]
        bars = ax.bar(x + (i - 1) * w, vals, width=w, label=pair, color=colors[i])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + (0.03 if v >= 0 else -0.08),
                    f'{v:+.2f}', ha='center', va='bottom' if v >= 0 else 'top',
                    fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(SLICE_NAMES)
    ax.set_ylabel('Spearman rank correlation')
    ax.set_title('Cross-method rank agreement on 6 numeric features (300 flows)')
    ax.axhline(0, color='black', lw=0.6)
    ax.axhline(0.8, ls=':', color='gray', alpha=0.5, label='|r|=0.8 (strong agreement)')
    ax.axhline(-0.8, ls=':', color='gray', alpha=0.5)
    ax.set_ylim(-1.1, 1.1)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved: {out}")


def fig_ig_completeness_hist(d, out):
    """Histogram of IG completeness residual as % of max(effect, max|IG|),
    per slice, log-x axis."""
    ig = d['ig']
    completeness = ig['completeness_residual']
    f_a = ig['f_actual']; f_b = ig['f_baseline']
    effect = np.abs(f_a - f_b)

    # Group IG into 8 groups for max abs
    ig_grouped = np.zeros((ig['ig'].shape[0], 8))
    for j, name in enumerate(GROUP_NAMES_8):
        ig_grouped[:, j] = ig['ig'][:, FEATURE_GROUPS[name]].sum(axis=1)
    max_absig = np.abs(ig_grouped).max(axis=1)
    denom = np.maximum(np.maximum(effect, max_absig), 1e-12)
    rel = np.abs(completeness) / denom * 100  # percent

    st = ig['slice_type']
    fig, axs = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    bins = np.logspace(-3, 2, 40)  # 0.001% to 100%
    for i, s in enumerate(SLICE_NAMES):
        mask = (st == i)
        axs[i].hist(rel[mask], bins=bins, color=SLICE_COLORS[s],
                    edgecolor='black', alpha=0.85)
        axs[i].axvline(10, ls='--', color='red', alpha=0.7,
                       label='10% target')
        axs[i].set_xscale('log')
        axs[i].set_xlabel('|residual| / max(effect, max|IG|) (%, log)')
        axs[i].set_title(f'{s} (n = {int(mask.sum())})\n'
                         f'median={np.median(rel[mask]):.2f}%  '
                         f'max={rel[mask].max():.2f}%')
        axs[i].legend(loc='upper right')
        axs[i].grid(True, which='major', alpha=0.3)
    axs[0].set_ylabel('Number of flows')
    fig.suptitle('IG completeness residual per slice, per-slice baseline (300 flows)',
                 fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved: {out}")


def fig_typology_stacked(d, out):
    """Stacked bars of Type 1/2/3 counts per slice, with the mMTC failure
    regime called out."""
    diag = d['diag']
    types = ['type1_typical', 'type2_model_collapsed', 'type3_high_effect']
    labels = ['Type 1 (typical, low effect + low error)',
              'Type 2 (model collapsed, low effect + high error)',
              'Type 3 (high effect, model uses features)']
    colors = ['#B0BEC5', '#EF5350', '#66BB6A']
    values = np.array([[diag['per_slice'][sn]['typology_counts'][t]
                        for t in types] for sn in SLICE_NAMES])

    fig, ax = plt.subplots(figsize=(8.5, 5))
    bottom = np.zeros(3)
    for t_i, (t_name, color, label) in enumerate(zip(types, colors, labels)):
        vals = values[:, t_i]
        bars = ax.bar(SLICE_NAMES, vals, bottom=bottom, color=color,
                      edgecolor='black', linewidth=0.8, label=label)
        for j, (b, v) in enumerate(zip(bars, vals)):
            if v > 3:
                ax.text(b.get_x() + b.get_width() / 2, bottom[j] + v / 2,
                        str(v), ha='center', va='center',
                        fontweight='bold', fontsize=10,
                        color='white' if t_i == 1 else 'black')
        bottom += vals
    ax.annotate('mMTC failure regime\n(25 collapsed flows)',
                xy=(1, 25 + 29 / 2), xytext=(1.7, 55),
                fontsize=10, color='#C62828',
                arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.5))
    ax.set_ylabel('Flow count (out of 100 per slice)')
    ax.set_title('Diagnostic typology across slices (300 flows total)')
    ax.set_ylim(0, 115)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    d = load_all()
    fig_sliceswap_violin(d, os.path.join(RESULTS_DIR, 'fig_sliceswap_violin.png'))
    fig_sliceswap_directional_heatmap(d, os.path.join(RESULTS_DIR, 'fig_sliceswap_directional_heatmap.png'))
    fig_urllc_dashboard(d, os.path.join(RESULTS_DIR, 'fig_urllc_dashboard.png'))
    fig_method_agreement(d, os.path.join(RESULTS_DIR, 'fig_method_agreement.png'))
    fig_ig_completeness_hist(d, os.path.join(RESULTS_DIR, 'fig_ig_completeness_hist.png'))
    fig_typology_stacked(d, os.path.join(RESULTS_DIR, 'fig_typology_stacked.png'))
    print("\nAll 6 figures generated.")


if __name__ == '__main__':
    main()
