"""
paranoia_checks_ig.py - Two paranoia checks on the 300 IG attribution vectors.
Saves results to results/inference/ig_paranoia.md
"""
import numpy as np
import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
ig_dir    = os.path.join(REPO_ROOT, 'results', 'inference', 'ig')
out_path  = os.path.join(REPO_ROOT, 'results', 'inference', 'ig_paranoia.md')

FEATS = ['traffic', 'packets', 'eq_lambda', 'avg_pkts_lambda', 'exp_max_factor',
         'pkts_lambda_on', 'avg_t_off', 'avg_t_on', 'ar_a', 'sigma']
FIDX  = {f: i for i, f in enumerate(FEATS)}

# Load all 300 vectors (300, 10)
M = np.array([
    np.load(os.path.join(ig_dir, 'sim_%04d.npz' % i), allow_pickle=True)['ig_scores']
    for i in range(300)
])

lines = []
lines.append('# IG Paranoia Checks')
lines.append('**Date:** 2026-06-04  ')
lines.append('**Source:** results/inference/ig/ (300 sims, all_multiplexed)')
lines.append('')

# ── CHECK 1: Sign sanity ──────────────────────────────────────────────────────
lines.append('## Check 1 - Sign sanity for traffic, packets, sigma')
lines.append('')
lines.append('Positive = feature pushes predicted delay UP relative to baseline.  ')
lines.append('Negative = feature pushes delay DOWN.  ')
lines.append('A 50/50 split would mean the feature has no consistent directional effect.')
lines.append('')
lines.append('| Feature | Positive | Negative | Zero | Mean | Median | Std | Flag? |')
lines.append('|---|---|---|---|---|---|---|---|')

FINDINGS = []
for feat in ['traffic', 'packets', 'sigma']:
    col = M[:, FIDX[feat]]
    n_pos  = int(np.sum(col > 0))
    n_neg  = int(np.sum(col < 0))
    n_zero = int(np.sum(col == 0))
    mean   = float(np.mean(col))
    median = float(np.median(col))
    std    = float(np.std(col))
    ratio  = min(n_pos, n_neg) / max(n_pos, n_neg) if max(n_pos, n_neg) > 0 else 0
    flag   = 'FLAG: near 50/50' if ratio > 0.6 else 'OK'
    if flag != 'OK':
        FINDINGS.append(feat)
    lines.append('| %s | %d | %d | %d | %.4f | %.4f | %.4f | %s |' %
                 (feat, n_pos, n_neg, n_zero, mean, median, std, flag))
    print('%s: pos=%d neg=%d zero=%d mean=%.4f median=%.4f std=%.4f -> %s' %
          (feat, n_pos, n_neg, n_zero, mean, median, std, flag))

lines.append('')
if FINDINGS:
    lines.append('**Flagged features (near 50/50 split): %s**  ' % ', '.join(FINDINGS))
    lines.append('Investigate sign consistency in Step 5 plausibility analysis.')
else:
    lines.append('**No flags.** All three features show consistent directional dominance.')
lines.append('')

# ── CHECK 2: Outlier sensitivity for sigma ────────────────────────────────────
lines.append('## Check 2 - Outlier sensitivity for sigma')
lines.append('')
sigma_col = M[:, FIDX['sigma']]
top5_idx  = np.argsort(np.abs(sigma_col))[-5:]
mask      = np.ones(300, dtype=bool)
mask[top5_idx] = False
M_trimmed = M[mask]  # (295, 10)

mean_abs_full    = [(f, float(np.mean(np.abs(M[:, j]))))    for j, f in enumerate(FEATS)]
mean_abs_trimmed = [(f, float(np.mean(np.abs(M_trimmed[:, j])))) for j, f in enumerate(FEATS)]
mean_abs_full.sort(key=lambda x: -x[1])
mean_abs_trimmed.sort(key=lambda x: -x[1])

lines.append('Top-5 sigma outlier sims removed (indices: %s).' %
             ', '.join(str(i) for i in sorted(top5_idx)))
lines.append('Their sigma attributions: %s' %
             ', '.join('%.3f' % sigma_col[i] for i in sorted(top5_idx)))
lines.append('')
lines.append('| Rank | Full 300 sims | mean(|IG|) | Trimmed 295 sims | mean(|IG|) |')
lines.append('|---|---|---|---|---|')
for rank in range(min(5, len(FEATS))):
    f_full, v_full     = mean_abs_full[rank]
    f_trim, v_trim     = mean_abs_trimmed[rank]
    lines.append('| %d | %s | %.6f | %s | %.6f |' %
                 (rank+1, f_full, v_full, f_trim, v_trim))

lines.append('')
top1_full    = mean_abs_full[0][0]
top1_trimmed = mean_abs_trimmed[0][0]
robust = (top1_trimmed == top1_full)

if robust:
    verdict = ('**Robust.** sigma remains #1 after removing top-5 outliers '
               '(%.6f -> %.6f). sigma dominance is not outlier-driven.' %
               (mean_abs_full[0][1], mean_abs_trimmed[0][1]))
else:
    verdict = ('**Finding.** Top-1 shifts from %s -> %s after removing top-5 sigma '
               'outliers. sigma dominance is partly outlier-driven. '
               'Report in Step 5.' % (top1_full, top1_trimmed))

lines.append(verdict)
print('\nCheck 2 verdict: %s' % verdict)

# ── Save markdown ─────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w') as fh:
    fh.write('\n'.join(lines) + '\n')
print('\nSaved to %s' % out_path)
