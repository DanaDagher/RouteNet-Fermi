"""
postprocess_step4.py - Post-processing for Step 4 outputs.
Tasks:
  A. Generate flow0_class_distribution.json (PIPELINE.md Step 5 requirement).
  B. Enrich KernelSHAP and IG timings.csv with n_perturbations column.

Does NOT re-run IG or KernelSHAP. Reads existing .npz and data files only.
"""

import os, sys, json, csv, re

REPO_ROOT  = os.path.dirname(os.path.abspath(__file__))
DELAY_DIR  = os.path.join(REPO_ROOT, 'traffic_models', 'delay')
TEST_PATH  = os.path.join(REPO_ROOT, 'data', 'traffic_models', 'all_multiplexed', 'test')
RANKINGS   = os.path.join(REPO_ROOT, 'rankings')
EXPL_ROOT  = os.path.join(REPO_ROOT, 'results', 'inference')

sys.path.insert(0, DELAY_DIR)
sys.path.insert(0, REPO_ROOT)
os.chdir(DELAY_DIR)

import numpy as np
import tensorflow as tf
tf.random.set_seed(42)
np.random.seed(42)

from data_generator import input_fn

os.makedirs(RANKINGS, exist_ok=True)

# ── Traffic-model codes (from delay_model.py / data_generator.py) ─────────────
# model field is an integer encoding TimeDist enum value:
# 0=CBR/Deterministic, 1=Poisson, 2=OnOff, 3=Autocorrelated, 4=Modulated, etc.
# See datanetAPI for full enum. We use the integer directly + a label map.
MODEL_LABELS = {
    0: 'CBR/Deterministic',
    1: 'Poisson',
    2: 'OnOff',
    3: 'Autocorrelated',
    4: 'Modulated',
    5: 'Other',
    6: 'AR1',
    7: 'AR1-1',
}

# ── Task A: flow_idx=0 traffic-model class distribution ───────────────────────
print('Task A: computing flow_idx=0 traffic-model class distribution...')

ds = (input_fn(TEST_PATH, shuffle=False)
      .take(300)
      .prefetch(tf.data.experimental.AUTOTUNE))

class_counts = {}
model_codes  = []
for i, (x_batch, _) in enumerate(ds):
    model_code = int(x_batch['model'].numpy()[0])   # flow_idx=0
    model_codes.append(model_code)
    label = MODEL_LABELS.get(model_code, 'Unknown_%d' % model_code)
    class_counts[label] = class_counts.get(label, 0) + 1

# Sort by count descending
sorted_counts = sorted(class_counts.items(), key=lambda x: -x[1])
total = sum(class_counts.values())

print('flow_idx=0 traffic-model distribution across 300 sims:')
for label, count in sorted_counts:
    print('  %-22s %3d  (%.1f%%)' % (label, count, 100.0*count/total))

result = {
    'n_simulations': 300,
    'flow_idx':      0,
    'note': 'Traffic-model class of flow[0] in each of the 300 test simulations.',
    'distribution': {label: {'count': c, 'pct': round(100.0*c/total, 2)}
                     for label, c in sorted_counts},
    'model_codes_per_sim': model_codes,
}

out_path = os.path.join(RANKINGS, 'flow0_class_distribution.json')
with open(out_path, 'w') as fh:
    json.dump(result, fh, indent=2)
print('Saved to %s' % out_path)

# ── Task B: enrich timings.csv with n_perturbations column ────────────────────
print('\nTask B: enriching timings.csv files with n_perturbations column...')

for method, n_perturb in [('kernel_shap', 256), ('ig', 50)]:
    timing_path = os.path.join(EXPL_ROOT, method, 'timings.csv')
    with open(timing_path, 'r') as fh:
        rows = list(csv.reader(fh))

    header = rows[0]
    if 'n_perturbations' not in header:
        header.append('n_perturbations')
        for row in rows[1:]:
            row.append(str(n_perturb))

    with open(timing_path, 'w', newline='') as fh:
        csv.writer(fh).writerows(rows)

    # Verify
    with open(timing_path, 'r') as fh:
        sample = list(csv.reader(fh))
    print('  %s timings.csv header: %s' % (method, sample[0]))
    print('  %s timings.csv row[1]: %s' % (method, sample[1]))

print('\nDone.')
