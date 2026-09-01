"""
Per-slice MAPE diagnostic for RouteNet-Fermi on the slicing test set.

Loads the best checkpoint from --ckpt-dir (lowest val_loss encoded in its
filename), runs inference on the cached test set, prints overall MAPE and MAPE
broken down per slice_type (0=eMBB, 1=mMTC, 2=URLLC), and saves the raw
predictions, labels, and slice types as .npy files for later plotting.

Requires the test cache to exist: run preprocess_cache.py --splits test first.

Usage:
    python slicing/predict_diagnostic.py                    # defaults: cache/test, ckpt_full
    python slicing/predict_diagnostic.py --ckpt-dir slicing/ckpt_full
"""
import argparse
import os
import re
import sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
import tensorflow as tf

for _gpu in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_generator import input_fn_cached
from delay_model import RouteNet_Fermi

SLICE_NAMES = {0: 'eMBB', 1: 'mMTC', 2: 'URLLC'}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cache-dir', default=os.path.join(os.path.dirname(__file__), 'cache', 'test'))
    p.add_argument('--ckpt-dir', default=os.path.join(os.path.dirname(__file__), 'ckpt_full'))
    p.add_argument('--out-dir', default=os.path.join(os.path.dirname(__file__), 'results'))
    return p.parse_args()


def find_best_checkpoint(ckpt_dir):
    best, best_val = None, float('inf')
    for f in os.listdir(ckpt_dir):
        m = re.match(r'^(\d+)-(\d+\.\d+)\.index$', f)
        if not m:
            continue
        val = float(m.group(2))
        if val < best_val:
            best = f.replace('.index', '')
            best_val = val
    return best, best_val


def mape(pred, label):
    return float(np.mean(np.abs((label - pred) / label)) * 100)


def main():
    args = parse_args()

    if not os.path.isdir(args.cache_dir) or not os.listdir(args.cache_dir):
        print(f"ERROR: no cached test set at {args.cache_dir}")
        print("Run: python slicing/preprocess_cache.py --splits test")
        sys.exit(1)

    best, best_val = find_best_checkpoint(args.ckpt_dir)
    if best is None:
        print(f"ERROR: no checkpoint in {args.ckpt_dir}")
        sys.exit(1)
    ckpt_path = os.path.join(args.ckpt_dir, best)
    print(f"Best checkpoint: {best} (val_loss={best_val:.2f})")

    model = RouteNet_Fermi()
    model.compile(
        loss=tf.keras.losses.MeanAbsolutePercentageError(),
        optimizer=tf.keras.optimizers.Adam(1e-3),
    )
    model.load_weights(ckpt_path)

    ds = input_fn_cached(args.cache_dir, shuffle=False)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    print("Running predictions via model.predict...")
    predictions = model.predict(ds, verbose=1)
    predictions = np.squeeze(predictions)

    print("Collecting labels and slice_type per flow...")
    all_label, all_slice = [], []
    n_samples = 0
    for feats, labels in ds:
        all_label.append(labels.numpy())
        all_slice.append(feats['slice_type'].numpy().astype(int))
        n_samples += 1
    label = np.concatenate(all_label)
    slice_type = np.concatenate(all_slice)
    pred = predictions[:len(label)]  # safety trim if predict padded

    n_flows = len(pred)
    print(f"\nTotal: {n_samples} samples, {n_flows} flows")
    if len(pred) != len(label) or len(pred) != len(slice_type):
        print(f"WARNING shape mismatch: pred={len(pred)} label={len(label)} slice={len(slice_type)}")

    print("\n== Overall ==")
    print(f"  MAPE               : {mape(pred, label):.2f} %")
    print(f"  median true delay  : {np.median(label):.6g}")
    print(f"  median pred delay  : {np.median(pred):.6g}")
    print(f"  min / max true     : {np.min(label):.6g}  /  {np.max(label):.6g}")

    print("\n== Per slice ==")
    for k in sorted(SLICE_NAMES):
        mask = slice_type == k
        c = int(mask.sum())
        if c == 0:
            print(f"  {SLICE_NAMES[k]:>5}: 0 flows (not present in test set)")
            continue
        p_k, l_k = pred[mask], label[mask]
        print(f"  {SLICE_NAMES[k]:>5} (n={c:>5}, {100*c/n_flows:5.1f} %): "
              f"MAPE={mape(p_k, l_k):7.2f} %, "
              f"median_true={np.median(l_k):.6g}, "
              f"median_pred={np.median(p_k):.6g}, "
              f"true_range=[{np.min(l_k):.4g}, {np.max(l_k):.4g}]")

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, 'pred_test.npy'), pred)
    np.save(os.path.join(args.out_dir, 'label_test.npy'), label)
    np.save(os.path.join(args.out_dir, 'slice_type_test.npy'), slice_type)
    print(f"\nSaved: pred_test.npy, label_test.npy, slice_type_test.npy -> {args.out_dir}/")


if __name__ == '__main__':
    main()
