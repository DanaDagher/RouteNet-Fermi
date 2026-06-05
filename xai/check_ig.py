"""Quick sanity checks on the 300 IG attribution vectors."""
import numpy as np
import os

ig_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'results', 'inference', 'ig')

FEATS = ['traffic', 'packets', 'eq_lambda', 'avg_pkts_lambda', 'exp_max_factor',
         'pkts_lambda_on', 'avg_t_off', 'avg_t_on', 'ar_a', 'sigma']

# Load all 300 vectors into matrix (300, 10)
all_scores = []
for i in range(300):
    d = np.load(os.path.join(ig_dir, 'sim_%04d.npz' % i), allow_pickle=True)
    all_scores.append(d['ig_scores'])
M = np.array(all_scores)

# ── CHECK 1 ───────────────────────────────────────────────────────────────────
print('=' * 72)
print('CHECK 1 -- Distribution of IG attributions across 300 sims')
print('=' * 72)
print('%-22s %9s %9s %9s %9s %11s' % ('Feature', 'Min', 'Max', 'Mean', 'Std', 'Any non-0?'))
print('-' * 72)
for j, f in enumerate(FEATS):
    col  = M[:, j]
    nonz = bool(np.any(col != 0))
    print('%-22s %9.4f %9.4f %9.4f %9.4f %11s' %
          (f, col.min(), col.max(), col.mean(), col.std(), str(nonz)))

# ── CHECK 2 ───────────────────────────────────────────────────────────────────
print()
print('=' * 72)
print('CHECK 2 -- Top-attributed feature per simulation (by |IG|)')
print('=' * 72)
top_counts = {f: 0 for f in FEATS}
for row in M:
    top_feat = FEATS[int(np.argmax(np.abs(row)))]
    top_counts[top_feat] += 1

print('%-22s %7s %7s' % ('Feature', 'Count', '%'))
print('-' * 40)
for f, c in sorted(top_counts.items(), key=lambda x: -x[1]):
    if c > 0:
        print('%-22s %7d  %6.1f%%' % (f, c, 100.0 * c / 300))

traffic_pkt_pct = (top_counts['traffic'] + top_counts['packets']) / 300.0 * 100
onoff_ac = sum(top_counts[f] for f in
               ['avg_pkts_lambda', 'exp_max_factor', 'pkts_lambda_on',
                'avg_t_off', 'avg_t_on', 'ar_a', 'sigma'])
print('\ntraffic + packets combined: %.1f%%' % traffic_pkt_pct)
print('on/off + autocorr + modulated combined: %d sims (%.1f%%)' %
      (onoff_ac, 100.0 * onoff_ac / 300))

# ── CHECK 3 ───────────────────────────────────────────────────────────────────
print()
print('=' * 72)
print('CHECK 3 -- Global mean(|IG|) ranking  [Step 5 preview]')
print('=' * 72)
mean_abs = [(f, float(np.mean(np.abs(M[:, j])))) for j, f in enumerate(FEATS)]
mean_abs.sort(key=lambda x: -x[1])
print('%-6s %-22s %12s' % ('Rank', 'Feature', 'mean(|IG|)'))
print('-' * 44)
for rank, (f, v) in enumerate(mean_abs, 1):
    print('%-6d %-22s %12.6f' % (rank, f, v))
