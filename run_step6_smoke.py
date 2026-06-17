"""
run_step6_smoke.py — Step 6 smoke test.

For each config (baseline + 6 IG + 6 KernelSHAP + 6 random = 19) verify:
  1. Model instantiates with the expected path_embedding input_dim.
  2. Dataset loads without error (dropped_features fix works).
  3. Forward pass succeeds on one batch.
  4. Features in config['dropped_features'] that are droppable (not traffic/packets)
     are absent from the batch dict.
  5. All features in config['kept_features'] are present in the batch dict.

Usage:
    python run_step6_smoke.py

Must be run on branch xai-protocol-b.
"""

import os
import sys
import json
import glob

# ── repo path so imports work regardless of CWD ──────────────────────────────
REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)                          # needed by DatanetAPI relative imports
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'traffic_models', 'delay'))
sys.path.insert(0, os.path.join(REPO, 'traffic_models'))   # datanetAPI lives here

import numpy as np
import tensorflow as tf

from delay_model import RouteNet_Fermi, PATH_SCALAR_FEATURES
from data_generator import input_fn, _DROPPABLE_PATH_SCALARS

# ── paths ─────────────────────────────────────────────────────────────────────
CONFIGS_ROOT = os.path.join(REPO, 'configs')
TEST_DATA    = os.path.join(REPO, 'data', 'traffic_models', 'all_multiplexed', 'test')

# Feature is truly removable from the data dict
def is_droppable(feat):
    return feat in _DROPPABLE_PATH_SCALARS


# ── load all configs in a stable order ───────────────────────────────────────
CONFIG_PATHS = sorted(glob.glob(os.path.join(CONFIGS_ROOT, '**', '*.json')))

print('=' * 72)
print('Step 6 Smoke Test')
print(f'TF version: {tf.__version__}')
print(f'Test data : {TEST_DATA}')
print(f'Configs   : {len(CONFIG_PATHS)} found')
print('=' * 72)

all_pass = True
results = []

for cfg_path in CONFIG_PATHS:
    rel = os.path.relpath(cfg_path, CONFIGS_ROOT)
    with open(cfg_path) as fh:
        cfg = json.load(fh)

    kept      = cfg['kept_features']          # path scalars in path_embedding
    dropped   = cfg['dropped_features']       # what to remove from path embedding
    expected_dim = cfg['path_embedding_input_dim']
    method    = cfg['xai_method']
    partition = cfg['partition']

    print(f'\n[{rel}]')
    print(f'  kept={kept}  |  n_kept={len(kept)}  |  expected_dim={expected_dim}')
    print(f'  dropped_features (full list from config): {dropped}')

    errors = []

    # ── 1. Model instantiation ────────────────────────────────────────────────
    try:
        model = RouteNet_Fermi(kept_path_scalars=kept)
        actual_dim = model.path_embedding.input_shape[-1]
        if actual_dim != expected_dim:
            errors.append(f'  ERROR: path_embedding dim={actual_dim}, expected={expected_dim}')
        else:
            print(f'  [OK] path_embedding input_dim = {actual_dim}')
    except Exception as e:
        errors.append(f'  ERROR building model: {e}')

    # ── 2. Dataset loads ──────────────────────────────────────────────────────
    try:
        ds = input_fn(TEST_DATA, shuffle=False, dropped_features=dropped)
        batch = next(iter(ds))
        inputs_dict, labels = batch
        print(f'  [OK] dataset loaded, batch labels shape={labels.shape}')
    except Exception as e:
        errors.append(f'  ERROR loading dataset: {e}')
        for err in errors:
            print(err)
        all_pass = False
        results.append((rel, 'FAIL', errors))
        continue

    # ── 3. Absent / present feature checks ───────────────────────────────────
    truly_dropped = [f for f in dropped if is_droppable(f)]
    not_dropped   = [f for f in dropped if not is_droppable(f)]  # traffic/packets

    for feat in truly_dropped:
        if feat in inputs_dict:
            errors.append(f'  ERROR: {feat} should be absent from batch but is present')
        else:
            print(f'  [OK] droppable feature "{feat}" absent from batch')

    for feat in not_dropped:
        # traffic / packets stay in dict even when listed as "dropped" in config
        if feat not in inputs_dict:
            errors.append(f'  ERROR: {feat} must always be present (structural) but is missing')
        else:
            print(f'  [OK] structural feature "{feat}" kept in dict (correct — not droppable)')

    for feat in kept:
        if feat not in inputs_dict:
            errors.append(f'  ERROR: kept feature "{feat}" is missing from batch')
        else:
            print(f'  [OK] kept feature "{feat}" present in batch')

    # ── 4. Forward pass ───────────────────────────────────────────────────────
    try:
        preds = model(inputs_dict, training=False)
        print(f'  [OK] forward pass, preds shape={preds.shape}')
    except Exception as e:
        errors.append(f'  ERROR in forward pass: {e}')

    if errors:
        for err in errors:
            print(err)
        all_pass = False
        results.append((rel, 'FAIL', errors))
    else:
        results.append((rel, 'PASS', []))

# ── final summary ─────────────────────────────────────────────────────────────
print('\n' + '=' * 72)
print('SUMMARY')
print('=' * 72)
for rel, status, errs in results:
    mark = 'PASS' if status == 'PASS' else 'FAIL'
    print(f'  [{mark}] {rel}')
    for e in errs:
        print(f'         {e}')

n_pass = sum(1 for _, s, _ in results if s == 'PASS')
n_fail = sum(1 for _, s, _ in results if s == 'FAIL')
print(f'\nTotal: {n_pass}/{len(results)} PASS, {n_fail} FAIL')
if all_pass:
    print(f'\nAll {len(results)} configs passed smoke test. Step 6 ready.')
else:
    print('\nSome configs FAILED. Fix errors before proceeding.')
    sys.exit(1)
