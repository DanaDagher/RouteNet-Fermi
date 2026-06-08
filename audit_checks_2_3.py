"""audit_checks_2_3.py — Step 4 audit checks 2 and 3."""
import numpy as np
import os
import json
from scipy.stats import spearmanr

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
ig_dir    = os.path.join(REPO_ROOT, 'results', 'inference', 'ig')
ks_dir    = os.path.join(REPO_ROOT, 'results', 'inference', 'kernel_shap')
class_f   = os.path.join(REPO_ROOT, 'rankings', 'flow0_class_distribution.json')
out_path  = os.path.join(REPO_ROOT, 'results', 'inference', 'step4_audit_checks2_3.md')

FEATS = ['traffic','packets','eq_lambda','avg_pkts_lambda','exp_max_factor',
         'pkts_lambda_on','avg_t_off','avg_t_on','ar_a','sigma']

# Load all 300 vectors
IG = np.array([np.load(os.path.join(ig_dir, 'sim_%04d.npz' % i),
               allow_pickle=True)['ig_scores'] for i in range(300)])
KS = np.array([np.load(os.path.join(ks_dir, 'sim_%04d.npz' % i),
               allow_pickle=True)['shap_scores'] for i in range(300)])

with open(class_f) as f:
    class_dist = json.load(f)

lines = []
lines.append('# Step 4 Audit Checks 2 and 3')
lines.append('**Date:** 2026-06-04  ')
lines.append('**N:** 300 simulations, all_multiplexed test, flow_idx=0')
lines.append('')

# ── CHECK 2: top-attributed feature distribution for KernelSHAP ───────────────
lines.append('## Check 2 — KernelSHAP top-attributed feature distribution')
lines.append('')

ks_top = {f: 0 for f in FEATS}
for row in KS:
    ks_top[FEATS[int(np.argmax(np.abs(row)))]] += 1

ig_top = {f: 0 for f in FEATS}
for row in IG:
    ig_top[FEATS[int(np.argmax(np.abs(row)))]] += 1

lines.append('| Feature | SHAP top count | SHAP % | IG top count | IG % |')
lines.append('|---|---|---|---|---|')
for f in sorted(ks_top, key=lambda x: -ks_top[x]):
    if ks_top[f] > 0 or ig_top[f] > 0:
        lines.append('| %s | %d | %.1f%% | %d | %.1f%% |' % (
            f, ks_top[f], 100.0*ks_top[f]/300,
            ig_top[f], 100.0*ig_top[f]/300))

# Cross-check: sigma vs Modulated count
modulated_count = class_dist['distribution'].get('Modulated', {}).get('count', 0)
sigma_ks = ks_top['sigma']
sigma_ig = ig_top['sigma']
lines.append('')
lines.append('**Cross-check — sigma vs Modulated flows:**')
lines.append('- Modulated flows in class distribution: **%d** (%.1f%%)' %
             (modulated_count, 100.0*modulated_count/300))
lines.append('- sigma top in SHAP: **%d** (%.1f%%)' % (sigma_ks, 100.0*sigma_ks/300))
lines.append('- sigma top in IG:   **%d** (%.1f%%)' % (sigma_ig, 100.0*sigma_ig/300))
match = abs(sigma_ks - modulated_count) <= 10
lines.append('- Match (within 10 sims): **%s**' % ('YES' % () if match else 'NO — investigate'))
lines.append('')

# ── CHECK 3: Spearman rank correlation between IG and SHAP global rankings ─────
lines.append('## Check 3 — Spearman rank correlation: IG vs KernelSHAP global ranking')
lines.append('')

ig_mean_abs = np.mean(np.abs(IG), axis=0)   # (10,)
ks_mean_abs = np.mean(np.abs(KS), axis=0)   # (10,)

ig_rank  = (-ig_mean_abs).argsort().argsort() + 1   # rank 1=best
ks_rank  = (-ks_mean_abs).argsort().argsort() + 1

rho, pval = spearmanr(ig_rank, ks_rank)

lines.append('| Feature | mean(|IG|) | IG rank | mean(|SHAP|) | SHAP rank |')
lines.append('|---|---|---|---|---|')
order = np.argsort(ig_rank)
for j in order:
    lines.append('| %s | %.6f | %d | %.6f | %d |' % (
        FEATS[j], ig_mean_abs[j], ig_rank[j], ks_mean_abs[j], ks_rank[j]))

lines.append('')
lines.append('**Spearman rho = %.4f,  p-value = %.6f**' % (rho, pval))
lines.append('')
if rho >= 0.9:
    verdict = 'Very high agreement (rho >= 0.9). Both methods rank features consistently.'
elif rho >= 0.7:
    verdict = 'High agreement (rho >= 0.7). Rankings largely consistent with some divergence in mid-tier features.'
elif rho >= 0.5:
    verdict = 'Moderate agreement. Investigate divergences in Step 5.'
else:
    verdict = 'Low agreement — flag for investigation before Step 5.'
lines.append('**Verdict:** %s' % verdict)
lines.append('')
lines.append('**Top-3 agreement check:**')
ig_top3  = [FEATS[j] for j in np.argsort(ig_mean_abs)[::-1][:3]]
ks_top3  = [FEATS[j] for j in np.argsort(ks_mean_abs)[::-1][:3]]
overlap  = len(set(ig_top3) & set(ks_top3))
lines.append('- IG top-3:    %s' % ', '.join(ig_top3))
lines.append('- SHAP top-3:  %s' % ', '.join(ks_top3))
lines.append('- Overlap: %d/3 features' % overlap)

# Print to console
for l in lines:
    print(l)

with open(out_path, 'w') as fh:
    fh.write('\n'.join(lines) + '\n')
print('\nSaved to %s' % out_path)
