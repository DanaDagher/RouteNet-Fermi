"""
XAI phase - shared library.

The one piece of real engineering for this phase: expose the 16-dimensional
path_embedding input vector as an explicit differentiation / perturbation surface,
and run the rest of the trained GNN from it, so that Integrated Gradients and
KernelSHAP both explain *exactly* the same function.

WHY THIS IS NEEDED
------------------
In delay_model.RouteNet_Fermi.call(), the 16 path features are assembled inline
(z-scoring + one-hots for `model` and `slice_type`) and immediately fed to
self.path_embedding. There is no seam to attribute against. XaiRouteNet below
splits call() into two exact halves:

    X      = assemble_path_input(inputs)          # (N_p, 16), the attribution surface
    delays = forward_from_path_input(X, inputs)   # everything from path_embedding onward

forward_from_path_input(assemble_path_input(inputs), inputs) is numerically
identical to call(inputs) (verified by the self-test at the bottom). Attribution
methods then perturb / differentiate a single row of X (one flow) and read the
corresponding scalar delay back out.

SCOPE CAVEAT (important for interpreting results)
-------------------------------------------------
The 16-vector is the "learnable door". `traffic` and `packets` ALSO influence the
delay through the queuing-theory reconstruction (load, pkt_size, trans_delay),
which does not pass through these 16 inputs (the "physics door"). Therefore:
  * IG/SHAP here measure how much the GNN *learned* to use each feature, not its
    total causal effect. `traffic`/`packets` attribution UNDERSTATES their total
    influence (physics door not counted).
  * `slice_type` has ONLY a learnable door (no physics path), so its attribution
    here is a fair, direct measure of whether the embedding uses slice identity
    -- the core V2X question.
  * The step-8 value-perturbation test hits BOTH doors and is the complementary
    necessity check (see slicing/XAI_PHASE_PLAN.md).

Column order of the 16-vector is fixed by FEATURE_NAMES / FEATURE_GROUPS below and
must match assemble_path_input exactly.
"""
import glob
import os
import pickle
import re
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from delay_model import RouteNet_Fermi  # noqa: E402

SLICE_NAMES = {0: 'eMBB', 1: 'mMTC', 2: 'URLLC'}

# Exact column order produced by assemble_path_input (16 columns total).
FEATURE_NAMES = (
    ['traffic', 'packets']
    + [f'model_{i}' for i in range(7)]        # one_hot(model, max_num_models=7)
    + ['eq_lambda', 'avg_pkts_lambda', 'exp_max_factor']
    + ['slice_eMBB', 'slice_mMTC', 'slice_URLLC']  # one_hot(slice_type, 3): 0/1/2
    + ['delta']
)
assert len(FEATURE_NAMES) == 16, len(FEATURE_NAMES)

# Grouping of one-hot blocks for aggregated per-feature reporting.
# Maps a display name -> list of column indices in the 16-vector.
FEATURE_GROUPS = {
    'traffic': [0], 'packets': [1],
    'model': list(range(2, 9)),
    'eq_lambda': [9], 'avg_pkts_lambda': [10], 'exp_max_factor': [11],
    'slice_type': [12, 13, 14],
    'delta': [15],
}

# dense feature -> dtype used when calling the model directly (matches
# data_generator._output_signature).
_FLOAT_KEYS = ['traffic', 'packets', 'eq_lambda', 'avg_pkts_lambda',
               'exp_max_factor', 'delta', 'capacity', 'queue_size', 'weight']
_INT_KEYS = ['length', 'model', 'slice_type', 'policy', 'priority']
_RAGGED_KEYS_RANK0 = ['link_to_path', 'queue_to_path', 'queue_to_link']


class XaiRouteNet(RouteNet_Fermi):
    """RouteNet_Fermi with call() split so the 16-vector path input is explicit.

    Reuses the parent's trained layers (path_embedding, queue/link embeddings,
    the three GRUCells, readout_path) and z_score -- nothing is re-initialized.
    """

    def assemble_path_input(self, inputs):
        """Reproduce, exactly, the 16-column concat that call() feeds to
        path_embedding -- but return it instead of embedding it."""
        z = self.z_score
        traffic = inputs['traffic']
        packets = inputs['packets']
        eq_lambda = inputs['eq_lambda']
        avg_pkts_lambda = inputs['avg_pkts_lambda']
        exp_max_factor = inputs['exp_max_factor']
        delta = inputs['delta']
        slice_type = tf.one_hot(inputs['slice_type'], 3)
        return tf.concat(
            [(traffic - z['traffic'][0]) / z['traffic'][1],
             (packets - z['packets'][0]) / z['packets'][1],
             tf.one_hot(inputs['model'], self.max_num_models),
             (eq_lambda - z['eq_lambda'][0]) / z['eq_lambda'][1],
             (avg_pkts_lambda - z['avg_pkts_lambda'][0]) / z['avg_pkts_lambda'][1],
             (exp_max_factor - z['exp_max_factor'][0]) / z['exp_max_factor'][1],
             slice_type,
             (delta - z['delta'][0]) / z['delta'][1]], axis=1)

    def forward_from_path_input(self, X, inputs):
        """Everything call() does from `path_state = path_embedding(...)` onward,
        with the path input supplied externally as X (N_p, 16). Byte-for-byte the
        same ops as the parent call()."""
        z = self.z_score
        traffic = inputs['traffic']
        packets = inputs['packets']
        length = inputs['length']
        capacity = inputs['capacity']
        policy = tf.one_hot(inputs['policy'], self.num_policies)
        queue_size = inputs['queue_size']
        priority = tf.one_hot(inputs['priority'], self.max_num_queues)
        weight = inputs['weight']

        queue_to_path = inputs['queue_to_path']
        link_to_path = inputs['link_to_path']
        path_to_link = inputs['path_to_link']
        path_to_queue = inputs['path_to_queue']
        queue_to_link = inputs['queue_to_link']

        path_gather_traffic = tf.gather(traffic, path_to_link[:, :, 0])
        load = tf.math.reduce_sum(path_gather_traffic, axis=1) / capacity
        pkt_size = traffic / packets

        path_state = self.path_embedding(X)
        link_state = self.link_embedding(tf.concat([load, policy], axis=1))
        queue_state = self.queue_embedding(
            tf.concat([(queue_size - z['queue_size'][0]) / z['queue_size'][1],
                       priority, weight], axis=1))

        path_state_sequence = None
        for it in range(self.iterations):
            queue_gather = tf.gather(queue_state, queue_to_path)
            link_gather = tf.gather(link_state, link_to_path, name="LinkToPath")
            path_update_rnn = tf.keras.layers.RNN(self.path_update,
                                                  return_sequences=True,
                                                  return_state=True)
            previous_path_state = path_state
            path_state_sequence, path_state = path_update_rnn(
                tf.concat([queue_gather, link_gather], axis=2),
                initial_state=path_state)
            path_state_sequence = tf.concat(
                [tf.expand_dims(previous_path_state, 1), path_state_sequence], axis=1)
            path_gather = tf.gather_nd(path_state_sequence, path_to_queue)
            path_sum = tf.math.reduce_sum(path_gather, axis=1)
            queue_state, _ = self.queue_update(path_sum, [queue_state])
            queue_gather = tf.gather(queue_state, queue_to_link)
            link_gru_rnn = tf.keras.layers.RNN(self.link_update, return_sequences=False)
            link_state = link_gru_rnn(queue_gather, initial_state=link_state)

        capacity_gather = tf.gather(capacity, link_to_path)
        input_tensor = path_state_sequence[:, 1:].to_tensor()
        occupancy_gather = self.readout_path(input_tensor)
        length = tf.ensure_shape(length, [None])
        occupancy_gather = tf.RaggedTensor.from_tensor(occupancy_gather, lengths=length)
        queue_delay = tf.math.reduce_sum(occupancy_gather / capacity_gather, axis=1)
        trans_delay = pkt_size * tf.math.reduce_sum(1 / capacity_gather, axis=1)
        return queue_delay + trans_delay


def find_best_checkpoint(ckpt_dir):
    """Lowest val_loss encoded in the checkpoint filename (NNN-V.VV.index)."""
    best, best_val = None, float('inf')
    for f in os.listdir(ckpt_dir):
        m = re.match(r'^(\d+)-(\d+\.\d+)\.index$', f)
        if m and float(m.group(2)) < best_val:
            best, best_val = f[:-len('.index')], float(m.group(2))
    return best, best_val


def prepare_inputs(feats_raw):
    """Convert one cached sample's raw feature dict (numpy/lists) into the tensor
    dict the model expects, with dtypes matching data_generator._output_signature.
    Returns a plain dict of tensors."""
    f = {}
    for k in _FLOAT_KEYS:
        f[k] = tf.constant(np.asarray(feats_raw[k], dtype=np.float32))
    for k in _INT_KEYS:
        f[k] = tf.constant(np.asarray(feats_raw[k], dtype=np.int32))
    for k in _RAGGED_KEYS_RANK0:
        f[k] = tf.ragged.constant(feats_raw[k], dtype=tf.int32)
    f['path_to_queue'] = tf.ragged.constant(feats_raw['path_to_queue'], ragged_rank=1, dtype=tf.int32)
    f['path_to_link'] = tf.ragged.constant(feats_raw['path_to_link'], ragged_rank=1, dtype=tf.int32)
    return f


def load_raw_sample(cache_dir, fname):
    """Load one cached (feats, labels) tuple by basename."""
    with open(os.path.join(cache_dir, fname), 'rb') as fh:
        feats, labels = pickle.load(fh)
    return feats, np.asarray(labels, dtype=np.float32)


def load_model(ckpt_dir):
    """Build XaiRouteNet, restore the best checkpoint, and return (model, ckpt_name).
    The caller MUST run one forward pass (e.g. via warmup_and_verify) before trusting
    the weights: restore into a subclassed model is deferred until variables exist."""
    best, best_val = find_best_checkpoint(ckpt_dir)
    if best is None:
        raise SystemExit(f"ERROR: no checkpoint in {ckpt_dir}")
    model = XaiRouteNet()
    model.compile(loss=tf.keras.losses.MeanAbsolutePercentageError(),
                  optimizer=tf.keras.optimizers.Adam(1e-3))
    model.load_weights(os.path.join(ckpt_dir, best))
    print(f"Loaded checkpoint {best} (val_loss={best_val:.2f}) from {ckpt_dir}")
    return model, best


def warmup_and_verify(model, feats_raw, tol=1e-3):
    """Force variable creation / complete the deferred restore by running the model
    once, and verify the split reproduces call() exactly on this sample.
    Returns (delays_call, max_abs_diff, max_rel_diff)."""
    inputs = prepare_inputs(feats_raw)
    y_call = model(inputs).numpy()                        # original call()
    X = model.assemble_path_input(inputs)
    y_split = model.forward_from_path_input(X, inputs).numpy()
    abs_diff = np.abs(y_call - y_split)
    rel_diff = abs_diff / np.maximum(np.abs(y_call), 1e-12)
    max_abs, max_rel = float(abs_diff.max()), float(rel_diff.max())
    if max_rel > tol:
        raise RuntimeError(
            f"SPLIT MISMATCH: max_rel_diff={max_rel:.2e} > tol={tol:.0e}. "
            f"forward_from_path_input does not reproduce call() -- do NOT trust "
            f"attributions until this is fixed.")
    return y_call, max_abs, max_rel


if __name__ == '__main__':
    # Self-test: load the model and verify the split on a handful of shared-set
    # samples. Run on the DGX:
    #   CUDA_VISIBLE_DEVICES=2 python slicing/xai/xai_common.py
    import argparse
    import json

    here = os.path.dirname(os.path.abspath(__file__))
    slic = os.path.dirname(here)
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-dir', default=os.path.join(slic, 'ckpt_v2'))
    ap.add_argument('--test-cache', default=os.path.join(slic, 'cache', 'test'))
    ap.add_argument('--shared', default=os.path.join(here, 'sets', 'shared_900.json'))
    ap.add_argument('--n', type=int, default=8, help='distinct samples to verify')
    args = ap.parse_args()

    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    for _g in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(_g, True)
        except RuntimeError:
            pass

    with open(args.shared) as fh:
        records = json.load(fh)['records']
    # distinct files, first n
    seen, files = set(), []
    for r in records:
        if r['file'] not in seen:
            seen.add(r['file'])
            files.append(r['file'])
        if len(files) >= args.n:
            break

    model, ckpt = load_model(args.ckpt_dir)
    print(f"\nVerifying call() == forward_from_path_input(assemble(...)) on "
          f"{len(files)} samples:")
    worst_abs = worst_rel = 0.0
    for i, fname in enumerate(files):
        feats, labels = load_raw_sample(args.test_cache, fname)
        y, ma, mr = warmup_and_verify(model, feats)
        worst_abs, worst_rel = max(worst_abs, ma), max(worst_rel, mr)
        print(f"  [{i+1}/{len(files)}] {fname}: n_flows={len(y)}, "
              f"max_abs={ma:.2e}, max_rel={mr:.2e}")
    print(f"\nWORST over all: max_abs={worst_abs:.2e}, max_rel={worst_rel:.2e}")
    print("OK -- split reproduces call(); safe to build IG/SHAP on this wrapper."
          if worst_rel < 1e-3 else "FAIL -- investigate before proceeding.")
