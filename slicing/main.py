"""
Train RouteNet-Fermi on the Zenodo network-slicing dataset.

Adam lr=0.001, MAPE loss, seed 42. Default budget is 50 epochs x 500 steps
(~5 passes over the ~4,700 training samples) with early stopping on val_loss,
so it stops automatically once it converges and keeps the best checkpoint.

GPU memory growth is forced ON automatically (both via env var and tf.config),
so this never grabs the whole card -- safe to run alongside another user.

Prerequisites:
    1. Run split_dataset.py to create slicing/data/{train,val,test}/
    2. Run compute_zscore.py and paste the z_score dict into delay_model.py

Usage:
    python slicing/main.py                 # 50 x 500, early stopping
    python slicing/main.py --resume        # continue from latest checkpoint
    python slicing/main.py --epochs 80     # override the cap
"""
import argparse
import os
import sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ['PYTHONHASHSEED'] = '42'
# Never seize the whole GPU -- allocate only what is needed (good neighbour on a
# shared card). Set before TensorFlow initialises CUDA.
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

# Belt-and-suspenders: also request memory growth through the API.
for _gpu in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_generator import input_fn, input_fn_cached
from delay_model import RouteNet_Fermi


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train-dir', default=os.path.join(os.path.dirname(__file__), 'data', 'train'))
    p.add_argument('--val-dir', default=os.path.join(os.path.dirname(__file__), 'data', 'val'))
    p.add_argument('--ckpt-dir', default=os.path.join(os.path.dirname(__file__), 'ckpt_dir'))
    p.add_argument('--epochs', type=int, default=50, help='Max epochs (early stopping may stop sooner)')
    p.add_argument('--steps', type=int, default=500, help='Steps (samples) per epoch')
    p.add_argument('--val-steps', type=int, default=50)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--patience', type=int, default=5, help='Early-stopping patience (epochs)')
    p.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    p.add_argument('--cache', action='store_true',
                   help='Read pre-processed samples from --cache-root (fast). '
                        'Run preprocess_cache.py first.')
    p.add_argument('--cache-root', default=os.path.join(os.path.dirname(__file__), 'cache'))
    p.add_argument('--loss', choices=['mape', 'log_mse'], default='mape',
                   help='mape = V1 loss (MeanAbsolutePercentageError). '
                        'log_mse = V2 loss, mean squared error in log-space, '
                        'symmetric across scales.')
    p.add_argument('--max-delay-cap', type=float, default=0.0,
                   help='Skip cached samples where any flow delay exceeds this '
                        'value (seconds). Use 0 to disable (V1 behaviour). '
                        'V2 recommended: 1000.')
    return p.parse_args()


def log_mse(y_true, y_pred):
    """MSE in log-space. Symmetric across scales: being wrong by a factor of 10
    at 1 microsecond gets the same penalty as being wrong by a factor of 10 at
    1 second. Clips both sides to eps to keep the log finite when a raw
    prediction lands at or below zero (readout has no positivity constraint)."""
    eps = 1e-9
    log_true = tf.math.log(tf.maximum(y_true, eps))
    log_pred = tf.math.log(tf.maximum(y_pred, eps))
    return tf.reduce_mean(tf.square(log_true - log_pred))


def main():
    args = parse_args()

    if args.cache:
        train_src = os.path.join(args.cache_root, 'train')
        val_src = os.path.join(args.cache_root, 'val')
        loader = input_fn_cached
    else:
        train_src = args.train_dir
        val_src = args.val_dir
        loader = input_fn

    print(f"Mode      : {'CACHED (fast)' if args.cache else 'live parsing'}")
    print(f"Train src : {os.path.abspath(train_src)}")
    print(f"Val src   : {os.path.abspath(val_src)}")
    print(f"Ckpt dir  : {os.path.abspath(args.ckpt_dir)}")
    print(f"Epochs    : {args.epochs}")
    print(f"Steps/ep  : {args.steps}")
    print(f"LR        : {args.lr}")
    print(f"Loss      : {args.loss}")
    print(f"Max delay : {args.max_delay_cap if args.max_delay_cap > 0 else 'no filter'}")

    def _load(src, shuffle):
        if args.cache:
            return loader(src, shuffle=shuffle, max_delay_cap=args.max_delay_cap)
        return loader(src, shuffle=shuffle)

    # Bounded prefetch (not AUTOTUNE) to keep host RAM flat during long runs.
    ds_train = _load(train_src, shuffle=True)
    ds_train = ds_train.repeat()
    ds_train = ds_train.prefetch(4)

    ds_val = _load(val_src, shuffle=False)
    ds_val = ds_val.prefetch(4)

    model = RouteNet_Fermi()
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    if args.loss == 'log_mse':
        loss_object = log_mse
    else:
        loss_object = tf.keras.losses.MeanAbsolutePercentageError()

    model.compile(loss=loss_object, optimizer=optimizer, run_eagerly=False)

    os.makedirs(args.ckpt_dir, exist_ok=True)

    if args.resume:
        latest = tf.train.latest_checkpoint(args.ckpt_dir)
        if latest is not None:
            print(f"Restoring from {latest}")
            model.load_weights(latest)
        else:
            print("No checkpoint found, training from scratch.")

    filepath = os.path.join(args.ckpt_dir, "{epoch:03d}-{val_loss:.2f}")

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=filepath,
        verbose=1,
        mode="min",
        monitor='val_loss',
        save_best_only=False,
        save_weights_only=True,
        save_freq='epoch')

    # Stop when val_loss stops improving; keep the best weights.
    es_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=args.patience,
        restore_best_weights=True,
        verbose=1)

    print(f"Early stopping: monitor=val_loss, patience={args.patience} epochs")

    model.fit(ds_train,
              epochs=args.epochs,
              steps_per_epoch=args.steps,
              validation_data=ds_val,
              validation_steps=args.val_steps,
              callbacks=[cp_callback, es_callback])

    print("\nTraining complete. Evaluating on validation set...")
    model.evaluate(ds_val, steps=args.val_steps)


if __name__ == '__main__':
    main()
