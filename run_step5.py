"""
run_step5.py — Step 5: Aggregate XAI scores and rank features globally.

Inputs (all from Step 4):
  results/inference/ig/sim_NNNN.npz           (×300) — main IG, median baseline
  results/inference/kernel_shap/sim_NNNN.npz  (×300) — main SHAP, median background
  results/inference/ig_stability/sim_NNNN.npz (×300) — IG stability, mean baseline
  results/inference/kernel_shap_stability/sim_NNNN.npz (×300) — SHAP stability, mean

Outputs (all under rankings/):
  rankings/ig.csv                  — main IG global ranking
  rankings/kernel_shap.csv         — main KernelSHAP global ranking
  rankings/random.csv              — deterministic random-ranking control (seed=42)
  rankings/ig_stability.csv        — IG stability ranking
  rankings/kernel_shap_stability.csv — KernelSHAP stability ranking
  rankings/halfsplit_check.json    — half-split Spearman per method (sims 0-149 vs 150-299)

CSV format (all ranking files):
  rank, feature, mean_abs_score

halfsplit_check.json format:
  {
    "ig":          {"rho": ..., "p": ..., "rank_first_half": [...], "rank_second_half": [...]},
    "kernel_shap": {"rho": ..., "p": ..., "rank_first_half": [...], "rank_second_half": [...]}
  }

Usage:
    python run_step5.py
"""

import os
import sys
import csv
import json
import glob

import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from xai.training_stats import PATH_SCALAR_FEATURES
from xai.random_control import random_ranking

INFER_ROOT = os.path.join(REPO_ROOT, 'results', 'inference')
RANK_DIR   = os.path.join(REPO_ROOT, 'rankings')
os.makedirs(RANK_DIR, exist_ok=True)

N_SIMS     = 300
HALF       = 150   # sims 0-149 vs 150-299


# ── helpers ────────────────────────────────────────────────────────────────────

def load_scores(directory, score_key, n_sims=N_SIMS):
    """
    Load all sim_NNNN.npz from a directory and return a (n_sims, 10) array.
    Row order matches file sort order (0..299).
    """
    files = sorted(glob.glob(os.path.join(directory, 'sim_*.npz')))
    if len(files) != n_sims:
        raise ValueError(f'{directory}: expected {n_sims} files, found {len(files)}')
    matrix = []
    for fp in files:
        z   = np.load(fp)
        arr = z[score_key].flatten()   # (10,)
        if len(arr) != len(PATH_SCALAR_FEATURES):
            raise ValueError(f'{fp}: expected 10 scores, got {len(arr)}')
        matrix.append(arr)
    return np.array(matrix, dtype=np.float64)   # (300, 10)


def aggregate_ranking(matrix):
    """
    mean(|score|) per feature → sorted ranking.
    Returns list of (rank, feature, mean_abs_score), rank=1 is most important.
    """
    mean_abs = np.mean(np.abs(matrix), axis=0)   # (10,)
    order    = np.argsort(-mean_abs)
    return [(i + 1, PATH_SCALAR_FEATURES[idx], float(mean_abs[idx]))
            for i, idx in enumerate(order)]


def save_ranking_csv(ranking, path):
    """Save ranking to CSV: rank, feature, mean_abs_score."""
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['rank', 'feature', 'mean_abs_score'])
        w.writerows(ranking)
    print(f'  saved: {os.path.relpath(path, REPO_ROOT)}')


def halfsplit_spearman(matrix):
    """
    Compute Spearman between global rankings derived from sims 0-149 and 150-299.
    Returns rho, p_value and both half-rankings (as ordered feature lists).
    """
    first_half  = matrix[:HALF]
    second_half = matrix[HALF:]

    mean_abs_1 = np.mean(np.abs(first_half),  axis=0)
    mean_abs_2 = np.mean(np.abs(second_half), axis=0)

    rho, p = spearmanr(mean_abs_1, mean_abs_2)

    rank_1 = [PATH_SCALAR_FEATURES[i] for i in np.argsort(-mean_abs_1)]
    rank_2 = [PATH_SCALAR_FEATURES[i] for i in np.argsort(-mean_abs_2)]

    return float(rho), float(p), rank_1, rank_2


def print_ranking(label, ranking):
    print(f'\n  {label}:')
    for rank, feat, score in ranking:
        bar = '#' * int(score * 30 / ranking[0][2]) if ranking[0][2] > 0 else ''
        print(f'    {rank:2d}. {feat:20s}  {score:.6f}  {bar}')


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print('=' * 65)
    print('Step 5 — Aggregate XAI scores and rank features globally')
    print('=' * 65)

    # ── 1. Load all four explanation matrices ─────────────────────────────────
    print('\n[1/4] Loading explanation matrices...')
    ig_main  = load_scores(os.path.join(INFER_ROOT, 'ig'),           'ig_scores')
    ks_main  = load_scores(os.path.join(INFER_ROOT, 'kernel_shap'),  'shap_scores')
    ig_stab  = load_scores(os.path.join(INFER_ROOT, 'ig_stability'),          'ig_scores')
    ks_stab  = load_scores(os.path.join(INFER_ROOT, 'kernel_shap_stability'), 'shap_scores')
    print(f'  ig_main shape:  {ig_main.shape}   (should be (300,10))')
    print(f'  ks_main shape:  {ks_main.shape}')
    print(f'  ig_stab shape:  {ig_stab.shape}')
    print(f'  ks_stab shape:  {ks_stab.shape}')

    # ── 2. Global rankings ────────────────────────────────────────────────────
    print('\n[2/4] Computing global rankings...')

    ig_rank  = aggregate_ranking(ig_main)
    ks_rank  = aggregate_ranking(ks_main)
    ig_s_rank = aggregate_ranking(ig_stab)
    ks_s_rank = aggregate_ranking(ks_stab)
    rand_rank = [(i + 1, feat, score)
                 for i, (feat, score) in enumerate(random_ranking(seed=42))]

    print_ranking('IG (main, median baseline)',      ig_rank)
    print_ranking('KernelSHAP (main, median background)', ks_rank)
    print_ranking('Random control (seed=42)',        rand_rank)
    print_ranking('IG stability (mean baseline)',    ig_s_rank)
    print_ranking('KernelSHAP stability (mean background)', ks_s_rank)

    save_ranking_csv(ig_rank,   os.path.join(RANK_DIR, 'ig.csv'))
    save_ranking_csv(ks_rank,   os.path.join(RANK_DIR, 'kernel_shap.csv'))
    save_ranking_csv(rand_rank, os.path.join(RANK_DIR, 'random.csv'))
    save_ranking_csv(ig_s_rank, os.path.join(RANK_DIR, 'ig_stability.csv'))
    save_ranking_csv(ks_s_rank, os.path.join(RANK_DIR, 'kernel_shap_stability.csv'))

    # ── 3. Half-split Spearman ────────────────────────────────────────────────
    print('\n[3/4] Half-split Spearman (sims 0-149 vs 150-299)...')

    ig_rho,  ig_p,  ig_r1,  ig_r2  = halfsplit_spearman(ig_main)
    ks_rho,  ks_p,  ks_r1,  ks_r2  = halfsplit_spearman(ks_main)

    print(f'  IG:          rho = {ig_rho:.4f}   p = {ig_p:.6f}')
    print(f'  KernelSHAP:  rho = {ks_rho:.4f}   p = {ks_p:.6f}')
    print(f'  IG   first-half  ranking: {ig_r1}')
    print(f'  IG   second-half ranking: {ig_r2}')
    print(f'  SHAP first-half  ranking: {ks_r1}')
    print(f'  SHAP second-half ranking: {ks_r2}')

    halfsplit = {
        'ig': {
            'rho':              ig_rho,
            'p_value':          ig_p,
            'n_sims_each_half': HALF,
            'rank_first_half':  ig_r1,
            'rank_second_half': ig_r2,
        },
        'kernel_shap': {
            'rho':              ks_rho,
            'p_value':          ks_p,
            'n_sims_each_half': HALF,
            'rank_first_half':  ks_r1,
            'rank_second_half': ks_r2,
        },
    }
    halfsplit_path = os.path.join(RANK_DIR, 'halfsplit_check.json')
    with open(halfsplit_path, 'w') as fh:
        json.dump(halfsplit, fh, indent=2)
    print(f'  saved: {os.path.relpath(halfsplit_path, REPO_ROOT)}')

    # ── 4. Cross-method & stability Spearman summary ─────────────────────────
    print('\n[4/4] Cross-method & stability Spearman summary...')

    # IG vs KernelSHAP (both main)
    ig_mean_abs = np.mean(np.abs(ig_main), axis=0)
    ks_mean_abs = np.mean(np.abs(ks_main), axis=0)
    rho_cross, p_cross = spearmanr(ig_mean_abs, ks_mean_abs)
    print(f'  IG vs KernelSHAP (main×main):          rho = {rho_cross:.4f}  p = {p_cross:.6f}')

    # IG: main vs stability
    ig_s_abs = np.mean(np.abs(ig_stab), axis=0)
    rho_ig_s, p_ig_s = spearmanr(ig_mean_abs, ig_s_abs)
    print(f'  IG: main(median) vs stability(mean):   rho = {rho_ig_s:.4f}  p = {p_ig_s:.6f}')

    # KernelSHAP: main vs stability
    ks_s_abs = np.mean(np.abs(ks_stab), axis=0)
    rho_ks_s, p_ks_s = spearmanr(ks_mean_abs, ks_s_abs)
    print(f'  SHAP: main(median) vs stability(mean): rho = {rho_ks_s:.4f}  p = {p_ks_s:.6f}')

    # ── final summary ─────────────────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('STEP 5 COMPLETE')
    print('=' * 65)
    print(f'\nGlobal rankings (main runs):')
    print(f'  IG top-3:   {[f for _,f,_ in ig_rank[:3]]}')
    print(f'  SHAP top-3: {[f for _,f,_ in ks_rank[:3]]}')
    print(f'  Random top-3 (seed=42): {[f for _,f,_ in rand_rank[:3]]}')
    print(f'\nHalf-split Spearman (N=300 sufficiency check):')
    print(f'  IG:          rho = {ig_rho:.4f}')
    print(f'  KernelSHAP:  rho = {ks_rho:.4f}')
    print(f'\nFiles saved to rankings/:')
    for fname in ['ig.csv', 'kernel_shap.csv', 'random.csv',
                  'ig_stability.csv', 'kernel_shap_stability.csv',
                  'halfsplit_check.json']:
        print(f'  rankings/{fname}')
    print(f'  rankings/flow0_class_distribution.json  (from Step 4, already present)')
    print('\nReady for Step 6 (build reduced-input dataset variants).')


if __name__ == '__main__':
    main()
