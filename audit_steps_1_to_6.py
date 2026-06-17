"""
audit_steps_1_to_6.py — Independent end-to-end audit of pipeline Steps 1-6.

Re-derives every key artifact from raw data and compares against the committed
outputs. Does NOT trust the step reports — recomputes.

Run: venv/Scripts/python.exe audit_steps_1_to_6.py
"""

import os
import sys
import csv
import json
import glob

import numpy as np
from scipy.stats import spearmanr

REPO = r'C:\Users\ddagher\RouteNet-Fermi'
sys.path.insert(0, REPO)

PATH_SCALARS = ['traffic', 'packets', 'eq_lambda', 'avg_pkts_lambda',
                'exp_max_factor', 'pkts_lambda_on', 'avg_t_off', 'avg_t_on',
                'ar_a', 'sigma']

failures = []
def check(label, ok, detail=''):
    mark = 'PASS' if ok else 'FAIL'
    print(f'  [{mark}] {label}' + (f' — {detail}' if detail else ''))
    if not ok:
        failures.append(label)

print('=' * 70)
print('AUDIT — Steps 1-6')
print('=' * 70)

# ── Step 1/2.5: environment + baseline validation artifacts ───────────────────
print('\n[Step 1 / 2.5] Environment & baseline validation')
import tensorflow as tf
check('TF version is 2.6.5 (paper environment)', tf.__version__ == '2.6.5', tf.__version__)
check('Python is 3.7.x', sys.version_info[:2] == (3, 7), sys.version.split()[0])

for tm in ['constant_bitrate', 'onoff', 'autocorrelated', 'modulated', 'all_multiplexed']:
    fp = os.path.join(REPO, 'results', 'baseline_validation', f'mape_{tm}.json')
    ok = os.path.isfile(fp)
    if ok:
        with open(fp) as fh:
            d = json.load(fh)
        ok = 'mape_mean' in d or any('mape' in k for k in d)
    check(f'baseline_validation mape_{tm}.json exists & well-formed', ok)

# paper deltas
paper = {'constant_bitrate': 4.43, 'onoff': 2.90, 'autocorrelated': 2.62,
         'modulated': 5.21, 'all_multiplexed': 4.71}
for tm, ref in paper.items():
    fp = os.path.join(REPO, 'results', 'baseline_validation', f'mape_{tm}.json')
    if os.path.isfile(fp):
        with open(fp) as fh:
            d = json.load(fh)
        mean_key = [k for k in d if 'mean' in k]
        if mean_key:
            ours = float(d[mean_key[0]])
            check(f'{tm}: |ours - paper| <= 1pp', abs(ours - ref) <= 1.0,
                  f'ours={ours:.4f} paper={ref}')

# ── Step 3: XAI module + training stats ───────────────────────────────────────
print('\n[Step 3] XAI module & training statistics')
for f in ['integrated_gradients.py', 'kernel_shap.py', 'random_control.py',
          'training_stats.py', 'tests/test_ig.py', 'tests/test_kernel_shap.py']:
    check(f'xai/{f} exists', os.path.isfile(os.path.join(REPO, 'xai', f)))

ts_fp = os.path.join(REPO, 'results', 'training_stats.json')
ok = os.path.isfile(ts_fp)
check('results/training_stats.json exists', ok)
if ok:
    with open(ts_fp) as fh:
        ts = json.load(fh)
    medians = ts.get('median', ts)
    have_all = all(f in str(ts) for f in PATH_SCALARS)
    check('training stats cover all 10 path scalars', have_all)

# ── Step 4: explanation artifacts (4 x 300 npz) ───────────────────────────────
print('\n[Step 4] Explanation artifacts — integrity of all 1200 attribution files')
matrices = {}
for d, key in [('ig', 'ig_scores'), ('kernel_shap', 'shap_scores'),
               ('ig_stability', 'ig_scores'), ('kernel_shap_stability', 'shap_scores')]:
    files = sorted(glob.glob(os.path.join(REPO, 'results', 'inference', d, 'sim_*.npz')))
    check(f'{d}: exactly 300 files', len(files) == 300, f'{len(files)} found')
    rows, all_finite, names_ok = [], True, True
    for fp in files:
        z = np.load(fp)
        arr = z[key].flatten()
        if arr.shape[0] != 10 or not np.all(np.isfinite(arr)):
            all_finite = False
        if 'feature_names' in z:
            fn = [x.decode() if isinstance(x, bytes) else str(x) for x in z['feature_names']]
            if fn != PATH_SCALARS:
                names_ok = False
        rows.append(arr)
    matrices[d] = np.array(rows)
    check(f'{d}: all vectors are 10-dim and finite', all_finite)
    check(f'{d}: feature_names order matches canonical', names_ok)
    # no all-zero feature column
    col_nonzero = np.any(matrices[d] != 0, axis=0)
    check(f'{d}: no feature identically zero across 300 sims', bool(np.all(col_nonzero)))

# explanation set indices
idx_fp = os.path.join(REPO, 'explanation_set', 'indices.npy')
ok = os.path.isfile(idx_fp)
check('explanation_set/indices.npy exists', ok)
if ok:
    idx = np.load(idx_fp)
    check('indices are 0..299 deterministic', idx.shape[0] == 300 and idx.min() == 0
          and idx.max() == 299 and len(set(idx.tolist())) == 300)

# ── Step 5: recompute rankings independently, compare with committed CSVs ─────
print('\n[Step 5] Rankings — independent recomputation from raw npz')
def load_csv_ranking(name):
    fp = os.path.join(REPO, 'rankings', name)
    out = []
    with open(fp) as fh:
        for row in csv.DictReader(fh):
            out.append((int(row['rank']), row['feature'], float(row['mean_abs_score'])))
    return out

for d, csv_name in [('ig', 'ig.csv'), ('kernel_shap', 'kernel_shap.csv'),
                    ('ig_stability', 'ig_stability.csv'),
                    ('kernel_shap_stability', 'kernel_shap_stability.csv')]:
    mean_abs = np.mean(np.abs(matrices[d]), axis=0)
    order = np.argsort(-mean_abs)
    recomputed = [(i + 1, PATH_SCALARS[j], float(mean_abs[j])) for i, j in enumerate(order)]
    committed = load_csv_ranking(csv_name)
    same_order = [r[1] for r in recomputed] == [r[1] for r in committed]
    max_delta = max(abs(a[2] - b[2]) for a, b in zip(recomputed, committed))
    check(f'rankings/{csv_name}: feature order matches recomputation', same_order)
    check(f'rankings/{csv_name}: scores match recomputation', max_delta < 1e-6,
          f'max delta = {max_delta:.2e}')

# spearman re-derivations
ig_abs = np.mean(np.abs(matrices['ig']), axis=0)
ks_abs = np.mean(np.abs(matrices['kernel_shap']), axis=0)
rho_cross, _ = spearmanr(ig_abs, ks_abs)
check('cross-method Spearman == 0.9636 (reported)', abs(rho_cross - 0.9636) < 0.001,
      f'{rho_cross:.4f}')

for d, reported in [('ig', 0.8788), ('kernel_shap', 0.9515)]:
    m = matrices[d]
    h1 = np.mean(np.abs(m[:150]), axis=0)
    h2 = np.mean(np.abs(m[150:]), axis=0)
    rho, _ = spearmanr(h1, h2)
    check(f'{d} half-split Spearman == {reported} (reported)', abs(rho - reported) < 0.001,
          f'{rho:.4f}')

# random control determinism
from xai.random_control import random_ranking
rr1 = random_ranking(seed=42)
rr2 = random_ranking(seed=42)
check('random control deterministic (seed 42)', rr1 == rr2)
committed_rand = load_csv_ranking('random.csv')
check('rankings/random.csv matches regenerated seed-42 ranking',
      [f for f, _ in rr1] == [r[1] for r in committed_rand])

# ── Step 6: configs vs rankings consistency ───────────────────────────────────
print('\n[Step 6] Variant configs — consistency against Step 5 rankings')
def ranked_features(csv_name):
    return [r[1] for r in load_csv_ranking(csv_name)]

K_KEEP = {30: 3, 50: 5, 70: 7}
for method, csv_name in [('ig', 'ig.csv'), ('kernel_shap', 'kernel_shap.csv'),
                         ('random', 'random.csv')]:
    ranking = ranked_features(csv_name)
    for k in [30, 50, 70]:
        n_top = K_KEEP[k]
        expect_rel = set(ranking[:n_top])
        expect_irr = set(ranking[n_top:])
        for part, expect in [('relevant', expect_rel), ('irrelevant', expect_irr)]:
            fp = os.path.join(REPO, 'configs', method, f'k{k}_{part}.json')
            with open(fp) as fh:
                cfg = json.load(fh)
            kept = set(cfg['kept_features'])
            dropped = set(cfg['dropped_features'])
            check(f'{method}/k{k}_{part}: kept == ranking-derived set', kept == expect)
            check(f'{method}/k{k}_{part}: kept+dropped == all 10, disjoint',
                  kept | dropped == set(PATH_SCALARS) and not (kept & dropped))
            check(f'{method}/k{k}_{part}: dim == n_kept + 7',
                  cfg['path_embedding_input_dim'] == len(kept) + 7,
                  f"dim={cfg['path_embedding_input_dim']}")

# IG == KernelSHAP equivalence at every cell (justifies training 7 unique, not 13):
# at k in {30,50,70} the two rankings yield identical kept-feature SETS, so the
# 6 principled trainings cover both methods. This must hold for the matrix design.
for k in [30, 50, 70]:
    for part in ['relevant', 'irrelevant']:
        ig_kept = set(json.load(open(
            os.path.join(REPO, 'configs', 'ig', f'k{k}_{part}.json')))['kept_features'])
        sh_kept = set(json.load(open(
            os.path.join(REPO, 'configs', 'kernel_shap', f'k{k}_{part}.json')))['kept_features'])
        check(f'IG set == KernelSHAP set at k{k}_{part} (principled equivalence)',
              ig_kept == sh_kept)

# baseline config
with open(os.path.join(REPO, 'configs', 'baseline', 'full.json')) as fh:
    base = json.load(fh)
check('baseline: keeps all 10, dim 17',
      set(base['kept_features']) == set(PATH_SCALARS)
      and base['path_embedding_input_dim'] == 17 and base['dropped_features'] == [])

# model-side: delay_model on this branch accepts kept_path_scalars and dims correctly
from delay_model import RouteNet_Fermi, PATH_SCALAR_FEATURES
check('delay_model.PATH_SCALAR_FEATURES == canonical list', PATH_SCALAR_FEATURES == PATH_SCALARS)
m2 = RouteNet_Fermi(kept_path_scalars=['sigma', 'traffic', 'packets'])
check('model dim for top-3 variant == 10', m2.path_embedding.input_shape[-1] == 10)
m10 = RouteNet_Fermi()
check('default model dim == 17 (upstream-identical)', m10.path_embedding.input_shape[-1] == 17)
# unknown feature rejected
try:
    RouteNet_Fermi(kept_path_scalars=['sigma', 'not_a_feature'])
    check('unknown feature name rejected', False)
except AssertionError:
    check('unknown feature name rejected', True)

# structural features never droppable in loader
sys.path.insert(0, os.path.join(REPO, 'traffic_models'))
sys.path.insert(0, os.path.join(REPO, 'traffic_models', 'delay'))
from data_generator import _DROPPABLE_PATH_SCALARS
check('loader droppable set excludes traffic & packets (structural)',
      'traffic' not in _DROPPABLE_PATH_SCALARS and 'packets' not in _DROPPABLE_PATH_SCALARS
      and len(_DROPPABLE_PATH_SCALARS) == 8)

# ── verdict ───────────────────────────────────────────────────────────────────
print('\n' + '=' * 70)
if failures:
    print(f'AUDIT RESULT: {len(failures)} FAILURE(S)')
    for f in failures:
        print(f'  - {f}')
    sys.exit(1)
else:
    print('AUDIT RESULT: ALL CHECKS PASSED')
print('=' * 70)
