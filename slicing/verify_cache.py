"""
Prove the caching path is LOSSLESS before building the whole cache.

For a few real samples it checks that pickling the raw sample and reloading it
produces byte-identical tensors to the direct (live) conversion. If every
feature and every label matches, the cache stores exactly what the live
pipeline would have produced.

(End-to-end model compatibility is confirmed separately by Step 4: training one
epoch on the cache with the real pipeline.)

Run from the repo root inside the venv:
    python slicing/verify_cache.py
Exit code 0 = PASS.
"""
import os
import sys
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = ''          # no GPU needed / touched
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'new_dataset'))

import numpy as np
import networkx as nx
import tensorflow as tf
from datanetAPI import DatanetAPI
from data_generator import network_to_hypergraph, hypergraph_to_raw, _to_tensors

VAL_DIR = os.path.join(HERE, 'data', 'val')
N = 3


def same(a, b):
    """True if a and b hold identical data. Handles ragged tensors, regular
    tensors, numpy arrays and plain lists -- without printing anything big."""
    if isinstance(a, tf.RaggedTensor) or isinstance(b, tf.RaggedTensor):
        la = a.to_list() if isinstance(a, tf.RaggedTensor) else a
        lb = b.to_list() if isinstance(b, tf.RaggedTensor) else b
        return la == lb
    a = a.numpy() if hasattr(a, 'numpy') else np.asarray(a)
    b = b.numpy() if hasattr(b, 'numpy') else np.asarray(b)
    return a.shape == b.shape and bool(np.array_equal(a, b))


def main():
    print('Reading up to {} samples from {}'.format(N, VAL_DIR))
    api = DatanetAPI(VAL_DIR, shuffle=False)

    checked = 0
    all_ok = True
    for sample in api:
        G = nx.DiGraph(sample.get_topology_object())
        T = sample.get_traffic_matrix()
        R = sample.get_routing_matrix()
        P = sample.get_performance_matrix()
        slices = sample.get_slices()
        HG = network_to_hypergraph(G=G, R=R, T=T, P=P, slices=slices)
        if HG is None:
            continue

        raw = hypergraph_to_raw(HG)
        if not all(x > 0 for x in raw[1]):
            continue

        # direct (live) vs pickle round-trip (cache)
        live_feats, live_labels = _to_tensors(raw)
        reloaded = pickle.loads(pickle.dumps(raw, protocol=pickle.HIGHEST_PROTOCOL))
        cache_feats, cache_labels = _to_tensors(reloaded)

        keys_ok = set(live_feats) == set(cache_feats)
        mismatched = [k for k in live_feats if not same(live_feats[k], cache_feats[k])]
        labels_ok = same(live_labels, cache_labels)

        ok = keys_ok and not mismatched and labels_ok
        all_ok = all_ok and ok
        detail = 'all {} features + labels identical'.format(len(live_feats)) if ok \
            else 'MISMATCH keys={} labels_ok={}'.format(mismatched, labels_ok)
        print('  sample {}: {} -> {}'.format(checked, 'PASS' if ok else 'FAIL', detail))

        checked += 1
        if checked >= N:
            break

    print('\nVERDICT:', 'PASS - caching is lossless' if (all_ok and checked > 0) else 'FAIL')
    sys.exit(0 if (all_ok and checked > 0) else 1)


if __name__ == '__main__':
    main()
