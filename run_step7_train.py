"""
run_step7_train.py — Step 7: train ONE variant cell of the retraining matrix.

Trains RouteNet-Fermi from scratch on the reduced-input variant described by a
Step 6 config file, with the locked hyperparameters (PIPELINE Step 7 / paper
section IV.D): 150 epochs x 2,000 samples, Adam lr=0.001, MAPE loss, hidden
state 32, T=8 iterations, seed 42.

Outputs (in --output dir):
  training_config.json   full hyperparameter snapshot, written before training
  training_log.csv       per-epoch train/val loss + MAE (Keras CSVLogger)
  {epoch:03d}-{val_loss} per-epoch weight checkpoints (TF format)
  metrics.json           final metrics, written after test evaluation

Resume: if the output dir already contains checkpoints, the latest one is
loaded and training continues from that epoch (same convention as upstream
traffic_models/delay/main.py). Delete the output dir for a clean restart.

Usage (from repo root, branch xai-protocol-b):
    python run_step7_train.py --config configs/baseline/full.json \
                              --output checkpoints/baseline_seed42

NOTE: export PYTHONHASHSEED=42 in the shell BEFORE invoking python (the
driver script run_step7_all.sh does this); setting it inside the process has
no effect on Python's hash randomization.
"""

import argparse
import json
import os
import re
import sys
import time
import random
import socket

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'traffic_models'))          # datanetAPI
sys.path.insert(0, os.path.join(REPO, 'traffic_models', 'delay'))  # data_generator

import numpy as np
import tensorflow as tf

from delay_model import RouteNet_Fermi
from data_generator import input_fn

SEED = 42


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def parse_args():
    p = argparse.ArgumentParser(description='Step 7: train one variant cell')
    p.add_argument('--config', required=True,
                   help='path to a Step 6 config JSON (configs/...)')
    p.add_argument('--output', required=True,
                   help='checkpoint/metrics output dir for this cell')
    p.add_argument('--data-root',
                   default=os.path.join(REPO, 'data', 'traffic_models', 'all_multiplexed'),
                   help='dataset root containing train/ and test/')
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--steps-per-epoch', type=int, default=2000)
    p.add_argument('--validation-steps', type=int, default=200,
                   help='val batches per epoch (upstream main.py uses 200)')
    p.add_argument('--test-steps', type=int, default=None,
                   help='cap test-eval batches (dry runs only; default: full split)')
    # ── compute-constrained (local CPU) options ───────────────────────────────
    p.add_argument('--max-train-samples', type=int, default=None,
                   help='train on a FIXED seeded subsample of this many simulations '
                        '(deterministic: first N in file order). Default: full split. '
                        'Use with --cache so the subsample is parsed once and reused.')
    p.add_argument('--max-test-samples', type=int, default=None,
                   help='FINAL test eval on the first N test simulations (fixed). Default: full.')
    p.add_argument('--val-samples', type=int, default=None,
                   help='per-epoch validation on the first N test simulations (fixed). '
                        'Default: same as --max-test-samples, else full. Keep small '
                        'on CPU so per-epoch val does not dominate runtime.')
    p.add_argument('--cache', action='store_true',
                   help='cache the (subsampled) training set to disk after the first '
                        'pass so later epochs skip graph re-parsing. Big CPU speedup.')
    return p.parse_args()


def latest_checkpoint_epoch(ckpt_dir):
    """Return (ckpt_path, epoch) of the latest checkpoint, or (None, 0)."""
    latest = tf.train.latest_checkpoint(ckpt_dir)
    if latest is None:
        return None, 0
    m = re.match(r'^(\d+)-', os.path.basename(latest))
    return latest, (int(m.group(1)) if m else 0)


def main():
    args = parse_args()

    with open(args.config) as fh:
        cfg = json.load(fh)

    train_dir = os.path.join(args.data_root, 'train')
    test_dir = os.path.join(args.data_root, 'test')
    for d in (train_dir, test_dir):
        if not os.path.isdir(d):
            sys.exit(f'ERROR: data split not found: {d}')

    os.makedirs(args.output, exist_ok=True)

    # GPU: allocate memory on demand instead of grabbing all 24 GB
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    set_seeds(SEED)

    print('=' * 70)
    print(f'Step 7 cell: {cfg["xai_method"]} / k={cfg["k"]} / {cfg["partition"]}')
    print(f'  kept ({cfg["n_path_scalars_kept"]}): {cfg["kept_features"]}')
    print(f'  config:  {args.config}')
    print(f'  output:  {args.output}')
    print(f'  GPUs:    {tf.config.list_physical_devices("GPU")}')
    print('=' * 70)

    # ── data ──────────────────────────────────────────────────────────────────
    dropped = cfg['dropped_features']

    # Training set. For local CPU runs we (a) take a FIXED seeded subsample so
    # all models train on the exact same simulations, and (b) cache it so the
    # expensive networkx/DatanetAPI parsing happens once, not every epoch.
    # shuffle=False makes the subsample deterministic (first N in file order);
    # per-epoch variety is restored by a tf-level shuffle after caching.
    if args.max_train_samples is not None:
        ds_train = input_fn(train_dir, shuffle=False, dropped_features=dropped)
        ds_train = ds_train.take(args.max_train_samples)
        if args.cache:
            cache_file = os.path.join(args.output, '_train_cache')
            ds_train = ds_train.cache(cache_file)
        buf = min(args.max_train_samples, 1000)
        ds_train = ds_train.shuffle(buf, seed=SEED, reshuffle_each_iteration=True)
        ds_train = ds_train.repeat()
    else:
        ds_train = input_fn(train_dir, shuffle=True, dropped_features=dropped)
        if args.cache:
            ds_train = ds_train.cache(os.path.join(args.output, '_train_cache'))
        ds_train = ds_train.repeat()

    # Per-epoch validation set (for the convergence curve + best-checkpoint).
    # Decoupled from the final test eval so we can keep val small on CPU.
    val_cap = args.val_samples
    if val_cap is None:
        val_cap = args.max_test_samples
    ds_val = input_fn(test_dir, shuffle=False, dropped_features=dropped)
    if val_cap is not None:
        ds_val = ds_val.take(val_cap)
        if args.cache:
            ds_val = ds_val.cache(os.path.join(args.output, '_val_cache'))
    # With a fixed subsample, run the WHOLE subsample each epoch
    # (validation_steps=None) instead of a fixed count that could exceed it.
    val_steps = None if val_cap is not None else args.validation_steps

    # ── model (locked hyperparameters) ────────────────────────────────────────
    model = RouteNet_Fermi(kept_path_scalars=cfg['kept_features'])
    actual_dim = model.path_embedding.input_shape[-1]
    assert actual_dim == cfg['path_embedding_input_dim'], \
        f'path_embedding dim {actual_dim} != config {cfg["path_embedding_input_dim"]}'

    model.compile(loss=tf.keras.losses.MeanAbsolutePercentageError(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae')],
                  run_eagerly=False)

    # ── hyperparameter snapshot (before training, so a crashed run is still
    #    reproducible from the file alone) ──────────────────────────────────────
    train_config = {
        'xai_method': cfg['xai_method'],
        'k': cfg['k'],
        'partition': cfg['partition'],
        'kept_features': cfg['kept_features'],
        'dropped_features': dropped,
        'n_features_kept': cfg['n_path_scalars_kept'],
        'path_embedding_input_dim': cfg['path_embedding_input_dim'],
        'epochs': args.epochs,
        'steps_per_epoch': args.steps_per_epoch,
        'validation_steps': args.validation_steps,
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'loss': 'MeanAbsolutePercentageError',
        'hidden_state_dim': 32,
        'message_passing_iterations': 8,
        'seed': SEED,
        'train_dir': train_dir,
        'validation_dir': test_dir,   # upstream convention: test split as val
        'test_dir': test_dir,
        'max_train_samples': args.max_train_samples,   # None = full split
        'max_test_samples': args.max_test_samples,
        'val_samples': args.val_samples,
        'cache': args.cache,
        'tf_version': tf.__version__,
        'hostname': socket.gethostname(),
        'config_file': args.config,
    }
    with open(os.path.join(args.output, 'training_config.json'), 'w') as fh:
        json.dump(train_config, fh, indent=2)

    # ── resume support ────────────────────────────────────────────────────────
    latest, initial_epoch = latest_checkpoint_epoch(args.output)
    if latest is not None:
        print(f'Resuming from {latest} (epoch {initial_epoch})')
        model.load_weights(latest)
    else:
        print('Training from scratch.')

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.output, '{epoch:03d}-{val_loss:.4f}'),
            verbose=1, mode='min', monitor='val_loss',
            save_best_only=False, save_weights_only=True, save_freq='epoch'),
        tf.keras.callbacks.CSVLogger(
            os.path.join(args.output, 'training_log.csv'), append=True),
    ]

    # ── train ─────────────────────────────────────────────────────────────────
    t0 = time.time()
    if initial_epoch < args.epochs:
        model.fit(ds_train,
                  epochs=args.epochs,
                  initial_epoch=initial_epoch,
                  steps_per_epoch=args.steps_per_epoch,
                  validation_data=ds_val,
                  validation_steps=val_steps,
                  callbacks=callbacks,
                  use_multiprocessing=True)
    else:
        print(f'Already trained to epoch {initial_epoch}; skipping fit.')
    train_minutes = (time.time() - t0) / 60.0

    # ── final train/val numbers from the full per-epoch log (robust to resume)
    log_path = os.path.join(args.output, 'training_log.csv')
    train_mape_final = train_mae_final = val_mape_final = val_mae_final = None
    best_val_mape = best_epoch = None
    if os.path.isfile(log_path):
        import csv as _csv
        with open(log_path) as fh:
            rows = [r for r in _csv.DictReader(fh)]
        if rows:
            last = rows[-1]
            train_mape_final = float(last['loss'])
            train_mae_final = float(last['mae'])
            val_mape_final = float(last['val_loss'])
            val_mae_final = float(last['val_mae'])
            best_row = min(rows, key=lambda r: float(r['val_loss']))
            best_val_mape = float(best_row['val_loss'])
            best_epoch = int(best_row['epoch']) + 1   # CSVLogger is 0-based

    # ── evaluate on the test split ────────────────────────────────────────────
    # All models are evaluated on the SAME test set (full, or the same fixed
    # first-N subsample) so test MAPE is comparable across the 5 cells.
    test_cap = args.max_test_samples if args.max_test_samples is not None else args.test_steps
    if test_cap is None:
        print('\nEvaluating on full test split...')
    else:
        print(f'\nEvaluating on first {test_cap} test simulations (fixed)...')
    ds_test = input_fn(test_dir, shuffle=False, dropped_features=dropped)
    if test_cap is not None:
        ds_test = ds_test.take(test_cap)
    test_mape, test_mae = model.evaluate(ds_test, verbose=2)

    metrics = {
        'xai_method': cfg['xai_method'],
        'k': cfg['k'],
        'partition': cfg['partition'],
        'train_mape_final': train_mape_final,
        'train_mae_final': train_mae_final,
        'val_mape_final': val_mape_final,
        'val_mae_final': val_mae_final,
        'test_mae': float(test_mae),
        'test_mape': float(test_mape),
        'best_val_mape': best_val_mape,
        'best_epoch': best_epoch,
        'training_time_minutes': round(train_minutes, 2),
        'epochs_run': args.epochs,
        'seed': SEED,
        'n_features_kept': cfg['n_path_scalars_kept'],
        'dropped_features': dropped,
        'max_train_samples': args.max_train_samples,
        'max_test_samples': args.max_test_samples,
        'test_eval_full_split': test_cap is None,
    }
    with open(os.path.join(args.output, 'metrics.json'), 'w') as fh:
        json.dump(metrics, fh, indent=2)

    print('\n' + '=' * 70)
    print(f'CELL COMPLETE: {cfg["xai_method"]} / k={cfg["k"]} / {cfg["partition"]}')
    print(f'  test MAPE = {test_mape:.4f} %   test MAE = {test_mae:.6f}')
    print(f'  training time = {train_minutes:.1f} min')
    print(f'  metrics: {os.path.join(args.output, "metrics.json")}')
    print('=' * 70)


if __name__ == '__main__':
    main()
