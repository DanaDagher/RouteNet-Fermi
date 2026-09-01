"""
Compute a full metric panel from saved diagnostic arrays.

Reads results/pred_test.npy, label_test.npy, slice_type_test.npy (produced by
predict_diagnostic.py) and prints multiple metrics per slice and overall.

Why several metrics: MAPE is broken for targets that span many orders of
magnitude (it punishes any error on very small labels enormously, so a model
that learns the true median gets a worse MAPE than one that collapses to zero).
Reporting a panel of metrics lets us see whether the model is actually good
even when MAPE says otherwise.

Metrics:
  MAPE       Mean Absolute Percentage Error. Traditional, but asymmetric.
  SMAPE      Symmetric MAPE, bounded 0 to 200 percent, symmetric under y/y_pred swap.
  MedAPE     Median APE. Robust to the heavy tail; typical per-flow error.
  MAE_log    Mean Absolute Error in log-space. Same units log_mse was trained on.
  RMSE_log   Root mean squared error in log-space.
  R2_log     Coefficient of determination on log-delay. 1 is perfect, 0 is baseline.
  r_lin      Pearson correlation of predictions and labels on the linear scale.
  r_log      Pearson correlation on the log scale.

Usage:
    python slicing/analyze_metrics.py
    python slicing/analyze_metrics.py --results-dir slicing/results
"""
import argparse
import os
import numpy as np

SLICE_NAMES = {0: 'eMBB', 1: 'mMTC', 2: 'URLLC'}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--results-dir',
                   default=os.path.join(os.path.dirname(__file__), 'results'))
    return p.parse_args()


def mape(pred, label):
    return float(np.mean(np.abs((label - pred) / label)) * 100)


def smape(pred, label):
    return float(np.mean(2 * np.abs(pred - label) / (np.abs(pred) + np.abs(label))) * 100)


def med_ape(pred, label):
    return float(np.median(np.abs((label - pred) / label)) * 100)


def log_stats(pred, label):
    eps = 1e-12
    lp = np.log(np.maximum(pred, eps))
    ll = np.log(np.maximum(label, eps))
    diff = ll - lp
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    ss_res = float(np.sum(diff * diff))
    ss_tot = float(np.sum((ll - np.mean(ll)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    r_log = float(np.corrcoef(lp, ll)[0, 1])
    r_lin = float(np.corrcoef(pred, label)[0, 1])
    return mae, rmse, r2, r_log, r_lin


def report(name, pred, label):
    n = len(pred)
    m = mape(pred, label)
    s = smape(pred, label)
    med = med_ape(pred, label)
    mae, rmse, r2, r_log, r_lin = log_stats(pred, label)
    print(f'\n  {name}  (n={n})')
    print(f'    MAPE       {m:8.2f} %       (asymmetric, dominated by small-label errors)')
    print(f'    SMAPE      {s:8.2f} %       (symmetric, 0-200% bounded)')
    print(f'    MedAPE     {med:8.2f} %       (median; robust to heavy tail)')
    print(f'    MAE_log    {mae:8.3f}         (avg log-space error; exp(x) = fold error)')
    print(f'    RMSE_log   {rmse:8.3f}')
    print(f'    R2_log     {r2:8.3f}         (1=perfect, 0=predicting the mean)')
    print(f'    r_lin      {r_lin:8.3f}         (linear-scale Pearson)')
    print(f'    r_log      {r_log:8.3f}         (log-scale Pearson)')


def main():
    args = parse_args()
    d = args.results_dir
    pred = np.load(os.path.join(d, 'pred_test.npy'))
    label = np.load(os.path.join(d, 'label_test.npy'))
    slice_type = np.load(os.path.join(d, 'slice_type_test.npy'))

    print(f'Loaded arrays from {d}')
    print(f'  flows total : {len(pred)}')

    print('\n' + '=' * 70)
    print('  OVERALL')
    print('=' * 70)
    report('overall', pred, label)

    print('\n' + '=' * 70)
    print('  PER SLICE')
    print('=' * 70)
    for k in sorted(SLICE_NAMES):
        mask = slice_type == k
        if mask.sum() == 0:
            continue
        report(SLICE_NAMES[k], pred[mask], label[mask])


if __name__ == '__main__':
    main()
