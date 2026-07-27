"""
compute_delay_decomposition.py — measured values for eqs. 7-10 (reviewer request).

Answers the D3.2 review comment: "il manque des mesures, valeurs associees a ces
metriques" on the delay-mechanism subsection. Produces, on the SAME fixed test
subset as the Step 7 pilot (first 300 simulations of all_multiplexed/test):

  1. Descriptive statistics of the eq. inputs: traffic, packets, pkt_size
     (eq. 9 input), capacity, link load (eq. 7), path length.
  2. The measured decomposition of the baseline model's predicted delay into
     queuing (eq. 8) and transmission (eq. 9) components. trans_delay is
     closed-form from the inputs (pkt_size * sum(1/capacity), no learned
     parameters), so queue_delay = predicted_delay - trans_delay exactly,
     by the model's own output equation (eq. 10).
  3. Prediction accuracy per flow vs measured delay (consistency check with
     the pilot metrics.json test MAPE).

Usage (repo root, branch xai-protocol-b):
    venv/Scripts/python.exe compute_delay_decomposition.py

Output: results/delay_decomposition.json + console summary.
"""

import json
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'traffic_models'))
sys.path.insert(0, os.path.join(REPO, 'traffic_models', 'delay'))

import numpy as np
import tensorflow as tf

from delay_model import RouteNet_Fermi
from data_generator import input_fn

N_TEST = 300   # same fixed subset as the pilot final test eval
# Retrained baseline with saved weights (test MAPE 5.36 %). The old pilot
# checkpoint (6.14 %) was lost; this is the model used everywhere in v12.
CKPT_DIR = os.path.join(REPO, 'checkpoints', 'local_sanity', 'baseline_seed42')
TEST_DIR = os.path.join(REPO, 'data', 'traffic_models', 'all_multiplexed', 'test')


def main():
    tf.random.set_seed(42)
    np.random.seed(42)

    model = RouteNet_Fermi()   # full 10 scalars (baseline)
    ckpt = tf.train.latest_checkpoint(CKPT_DIR)
    if ckpt is None:
        sys.exit(f'ERROR: no checkpoint in {CKPT_DIR}')
    print(f'Loading {ckpt}')

    ds = input_fn(TEST_DIR, shuffle=False).take(N_TEST)

    # Build weights with one forward pass, then load
    for x, y in ds.take(1):
        model(x)
    model.load_weights(ckpt).expect_partial()

    # accumulators (per flow)
    traffic_all, packets_all, pkt_size_all, length_all = [], [], [], []
    capacity_all, load_all = [], []
    pred_all, trans_all, label_all = [], [], []

    for i, (x, y) in enumerate(ds):
        traffic = x['traffic'].numpy().ravel()          # bits/time unit, per flow
        packets = x['packets'].numpy().ravel()          # packets/time unit
        capacity = x['capacity'].numpy().ravel()        # per link
        l2p = x['link_to_path']                         # ragged [flow, hop]
        length = x['length'].numpy().ravel()

        pkt_size = traffic / packets                    # eq. 9 input (bits)

        # eq. 7 link load: sum of traffic of flows crossing the link / capacity
        p2l = x['path_to_link']
        gathered = tf.gather(traffic, p2l[:, :, 0])
        load = (tf.math.reduce_sum(gathered, axis=1).numpy() / capacity)

        # eq. 9 transmission delay: pkt_size * sum over path of 1/capacity
        cap_gather = tf.gather(capacity, l2p)
        inv_sum = tf.math.reduce_sum(1.0 / cap_gather, axis=1).numpy().ravel()
        trans = pkt_size * inv_sum

        pred = model(x).numpy().ravel()                 # eq. 10 total
        label = y.numpy().ravel()                       # measured delay

        traffic_all.append(traffic); packets_all.append(packets)
        pkt_size_all.append(pkt_size); length_all.append(length)
        capacity_all.append(capacity); load_all.append(load)
        pred_all.append(pred); trans_all.append(trans); label_all.append(label)

        if (i + 1) % 50 == 0:
            print(f'  {i + 1}/{N_TEST} samples')

    def cat(lst):
        return np.concatenate(lst)

    traffic, packets = cat(traffic_all), cat(packets_all)
    pkt_size, length = cat(pkt_size_all), cat(length_all)
    capacity, load = cat(capacity_all), cat(load_all)
    pred, trans, label = cat(pred_all), cat(trans_all), cat(label_all)
    queue = pred - trans                                # eq. 8 aggregate term

    trans_share = trans / pred
    mape = np.mean(np.abs(pred - label) / label) * 100

    def stats(a):
        return {'mean': float(np.mean(a)), 'std': float(np.std(a)),
                'min': float(np.min(a)), 'p50': float(np.median(a)),
                'max': float(np.max(a))}

    out = {
        'n_test_samples': N_TEST,
        'n_flows': int(pred.size),
        'n_links': int(capacity.size),
        'checkpoint': os.path.basename(ckpt),
        'inputs': {
            'traffic_bits_per_tu': stats(traffic),
            'packets_per_tu': stats(packets),
            'pkt_size_bits_eq9': stats(pkt_size),
            'path_length_hops': stats(length),
            'link_capacity': stats(capacity),
            'link_load_eq7': stats(load),
        },
        'decomposition': {
            'predicted_delay_eq10': stats(pred),
            'trans_delay_eq9': stats(trans),
            'queue_delay_eq8': stats(queue),
            'trans_share_of_prediction': stats(trans_share),
            'trans_share_mean_pct': float(np.mean(trans_share) * 100),
        },
        'accuracy': {
            'measured_delay': stats(label),
            'test_mape_pct': float(mape),
        },
    }

    os.makedirs(os.path.join(REPO, 'results'), exist_ok=True)
    dst = os.path.join(REPO, 'results', 'delay_decomposition_v12.json')
    with open(dst, 'w') as fh:
        json.dump(out, fh, indent=2)

    print('\n=== eq. 7-10 measured values (first %d test sims) ===' % N_TEST)
    print(f"flows: {pred.size}, links: {capacity.size}")
    print(f"link load (eq.7):        mean {out['inputs']['link_load_eq7']['mean']:.3f}  median {out['inputs']['link_load_eq7']['p50']:.3f}")
    print(f"pkt size (eq.9 input):   mean {out['inputs']['pkt_size_bits_eq9']['mean']:.1f} bits")
    print(f"predicted delay (eq.10): mean {out['decomposition']['predicted_delay_eq10']['mean']:.5f}")
    print(f"  trans (eq.9):          mean {out['decomposition']['trans_delay_eq9']['mean']:.5f}")
    print(f"  queue (eq.8):          mean {out['decomposition']['queue_delay_eq8']['mean']:.5f}")
    print(f"TRANSMISSION SHARE of prediction: {out['decomposition']['trans_share_mean_pct']:.1f} % (mean per flow)")
    print(f"test MAPE (consistency): {mape:.2f} %")
    print(f'\nSaved {dst}')


if __name__ == '__main__':
    main()
