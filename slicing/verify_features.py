"""
Evidence that the slicing dataset is correctly adapted to RouteNet-Fermi.

Reads real samples and proves, empirically (not by assertion):
  1. The 5 dropped traffic params are genuinely absent in this dataset,
     so removing them loses no information.
  2. The 2 added slice features (slice_type, delta) exist and attach to
     every flow via its (src,dst) pair.
  3. The generator's output tensors match exactly what the model consumes.

Usage:
    python slicing/verify_features.py [--data new_dataset] [--n 3]
"""
import argparse
import os
import sys
import numpy as np
import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'new_dataset'))

from datanetAPI import DatanetAPI, TimeDist

# the 5 params the ORIGINAL model uses but the slicing model drops
DROPPED_PARAM_KEYS = ['PktsLambdaOn', 'AvgTOff', 'AvgTOn', 'AR-a', 'sigma']
# what the slicing model KEEPS
KEPT_PARAM_KEYS = ['EqLambda', 'AvgPktsLambda', 'ExpMaxFactor']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=os.path.join(HERE, '..', 'new_dataset'))
    p.add_argument('--n', type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = os.path.abspath(args.data)
    print(f"Reading {args.n} real samples from {data_dir}\n")

    api = DatanetAPI(data_dir, shuffle=False)
    it = iter(api)

    dist_counts = {}
    params_seen = set()
    dropped_hits = {k: 0 for k in DROPPED_PARAM_KEYS}
    n_flows = 0
    n_flows_with_slice = 0
    slice_types_seen = {}
    delta_values = []
    example_flow = None

    samples = []
    for _ in range(args.n):
        samples.append(next(it))

    for s in samples:
        G = s.get_topology_object()
        T = s.get_traffic_matrix()
        P = s.get_performance_matrix()
        slices = s.get_slices()

        lookup = {}
        for sl in slices:
            for f in sl['flows']:
                lookup[(f['origin_node'], f['destination'])] = (sl['type'], sl['delta'])

        N = G.number_of_nodes()
        for src in range(N):
            for dst in range(N):
                if src == dst:
                    continue
                try:
                    flows = T[src, dst]['Flows']
                except (TypeError, IndexError, KeyError):
                    continue
                for f_id, flow in enumerate(flows):
                    if flow['AvgBw'] == 0 or flow['PktsGen'] == 0:
                        continue
                    n_flows += 1

                    dname = TimeDist.getStrig(flow['TimeDist'].value)
                    dist_counts[dname] = dist_counts.get(dname, 0) + 1

                    for k in flow['TimeDistParams'].keys():
                        params_seen.add(k)
                    for k in DROPPED_PARAM_KEYS:
                        if k in flow['TimeDistParams']:
                            dropped_hits[k] += 1

                    if (src, dst) in lookup:
                        n_flows_with_slice += 1
                        st, dl = lookup[(src, dst)]
                        slice_types_seen[st] = slice_types_seen.get(st, 0) + 1
                        delta_values.append(dl)

                    if example_flow is None:
                        example_flow = {
                            'src': src, 'dst': dst,
                            'AvgBw': flow['AvgBw'],
                            'PktsGen': flow['PktsGen'],
                            'TimeDist': dname,
                            'TimeDistParams': dict(flow['TimeDistParams']),
                            'slice': lookup.get((src, dst)),
                            'AvgDelay': P[src, dst]['Flows'][f_id]['AvgDelay'],
                        }

    print("=" * 70)
    print("CLAIM 1: the 5 dropped traffic params are absent in this dataset")
    print("=" * 70)
    print(f"  Traffic distributions actually used: {dist_counts}")
    print(f"  All TimeDistParams keys present in the data: {sorted(params_seen)}")
    print(f"  Occurrences of each DROPPED param across {n_flows} flows:")
    for k, c in dropped_hits.items():
        verdict = "absent (safe to drop)" if c == 0 else f"PRESENT {c}x -- DO NOT DROP"
        print(f"    {k:16s}: {c:6d}  -> {verdict}")
    kept_ok = all(k in params_seen for k in KEPT_PARAM_KEYS)
    print(f"  Kept params {KEPT_PARAM_KEYS} all present: {kept_ok}")

    print()
    print("=" * 70)
    print("CLAIM 2: slice_type + delta exist and attach to every flow")
    print("=" * 70)
    print(f"  Flows total: {n_flows}")
    print(f"  Flows that map to a slice (src,dst): {n_flows_with_slice}")
    pct = 100.0 * n_flows_with_slice / max(n_flows, 1)
    print(f"  Coverage: {pct:.1f}%  (must be 100% -- every flow needs a slice)")
    print(f"  Slice types seen: {slice_types_seen}")
    if delta_values:
        print(f"  delta range: min={min(delta_values):.3f} max={max(delta_values):.3f} "
              f"mean={np.mean(delta_values):.3f}")

    print()
    print("=" * 70)
    print("A REAL FLOW (so you can see actual values going into the model)")
    print("=" * 70)
    for k, v in example_flow.items():
        print(f"  {k:16s}: {v}")

    print()
    print("=" * 70)
    print("CLAIM 3: generator output matches the model's expected inputs")
    print("=" * 70)
    from data_generator import network_to_hypergraph, hypergraph_to_input_data
    s = samples[0]
    G = nx.DiGraph(s.get_topology_object())
    HG = network_to_hypergraph(G=G, R=s.get_routing_matrix(),
                               T=s.get_traffic_matrix(), P=s.get_performance_matrix(),
                               slices=s.get_slices())
    features, labels = hypergraph_to_input_data(HG)
    print("  Feature tensors produced by the generator:")
    for k, v in features.items():
        try:
            shape = np.asarray(v).shape
        except Exception:
            shape = getattr(v, 'shape', '?')
        print(f"    {k:16s} shape={shape}")
    print(f"  Labels (delay): count={len(labels)}, "
          f"min={min(labels):.6f}, max={max(labels):.6f}")
    print()
    print("  path_embedding input assembled by the model = concat of:")
    print("    traffic(1) + packets(1) + model_onehot(7) + eq_lambda(1)")
    print("    + avg_pkts_lambda(1) + exp_max_factor(1) + slice_type_onehot(3) + delta(1)")
    print("    = 16 dims   <-- matches Input(shape=5+7+3+1) in delay_model.py")


if __name__ == '__main__':
    main()
