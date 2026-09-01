"""
Evaluate a trained RouteNet-Fermi checkpoint on the slicing test set.

Saves predictions, labels, and per-sample flow counts to .npy files, then
prints MAPE.

Usage:
    python slicing/predict.py [--test-dir slicing/data/test] [--ckpt-dir slicing/ckpt_dir]
"""
import argparse
import os
import re
import sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_generator import input_fn
from delay_model import RouteNet_Fermi


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--test-dir', default=os.path.join(os.path.dirname(__file__), 'data', 'test'))
    p.add_argument('--ckpt-dir', default=os.path.join(os.path.dirname(__file__), 'ckpt_dir'))
    p.add_argument('--ckpt', default=None, help='Specific checkpoint path (auto-selects best if omitted)')
    p.add_argument('--out-dir', default=os.path.join(os.path.dirname(__file__), 'results'))
    p.add_argument('--max-samples', type=int, default=None)
    return p.parse_args()


def find_best_checkpoint(ckpt_dir):
    best = None
    best_mre = float('inf')
    for f in os.listdir(ckpt_dir):
        if os.path.isfile(os.path.join(ckpt_dir, f)):
            reg = re.findall(r"\d+\.\d+", f)
            if reg:
                mre = float(reg[0])
                if mre <= best_mre:
                    name = f.replace('.index', '').replace('.data', '')
                    name = name.replace('-00000-of-00001', '')
                    best = name
                    best_mre = mre
    return best, best_mre


def main():
    args = parse_args()

    model = RouteNet_Fermi()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_object = tf.keras.losses.MeanAbsolutePercentageError()
    model.compile(loss=loss_object, optimizer=optimizer, run_eagerly=False)

    if args.ckpt:
        ckpt_path = args.ckpt
    else:
        best, best_val = find_best_checkpoint(args.ckpt_dir)
        if best is None:
            print(f"No checkpoint found in {args.ckpt_dir}")
            sys.exit(1)
        ckpt_path = os.path.join(args.ckpt_dir, best)
        print(f"Best checkpoint: {best} (val_loss={best_val:.2f})")

    model.load_weights(ckpt_path)

    ds_test = input_fn(args.test_dir, shuffle=False)
    if args.max_samples:
        ds_test = ds_test.take(args.max_samples)
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    print("Running predictions...")
    predictions = model.predict(ds_test, verbose=1)

    labels_all = []
    flow_counts = []
    for _, labels_batch in ds_test:
        labels_all.append(labels_batch.numpy())
        flow_counts.append(len(labels_batch))

    labels_all = np.concatenate(labels_all)
    flow_counts = np.array(flow_counts)
    predictions = np.squeeze(predictions)[:len(labels_all)]

    mape = np.mean(np.abs((labels_all - predictions) / labels_all)) * 100
    print(f"\nTest MAPE: {mape:.2f}%")
    print(f"Samples: {len(flow_counts)}, Total flows: {len(labels_all)}")

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, 'predictions_delay_slicing.npy'), predictions)
    np.save(os.path.join(args.out_dir, 'labels_delay_slicing.npy'), labels_all)
    np.save(os.path.join(args.out_dir, 'flow_counts_delay_slicing.npy'), flow_counts)
    print(f"Saved to {args.out_dir}/")


if __name__ == '__main__':
    main()
