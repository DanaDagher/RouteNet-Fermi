"""
Compute z-score normalization statistics (mean, std) from the training split
of the slicing dataset.

Uses the DatanetAPI directly (no TF) for speed. Scans N samples, collects
per-feature running stats, then prints the z_score dict to paste into
delay_model.py.

Usage:
    python slicing/compute_zscore.py [--data slicing/data/train] [--n 200]
"""
import argparse
import os
import sys
import numpy as np
import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'new_dataset'))
from datanetAPI import DatanetAPI


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=os.path.join(HERE, 'data', 'train'))
    p.add_argument('--n', type=int, default=200, help='Number of samples to scan')
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = os.path.abspath(args.data)
    print(f"Scanning {args.n} samples from {data_dir} ...")

    accum = {
        'traffic': [], 'packets': [],
        'eq_lambda': [], 'avg_pkts_lambda': [], 'exp_max_factor': [],
        'delta': [],
        'capacity': [], 'queue_size': [],
    }

    tool = DatanetAPI(data_dir, shuffle=False)
    count = 0
    for sample in tool:
        G = nx.DiGraph(sample.get_topology_object())
        T = sample.get_traffic_matrix()
        P = sample.get_performance_matrix()
        slices = sample.get_slices()

        delta_lookup = {}
        for sl in slices:
            for f in sl['flows']:
                delta_lookup[(f['origin_node'], f['destination'])] = sl['delta']

        for src in range(G.number_of_nodes()):
            for dst in range(G.number_of_nodes()):
                if src == dst:
                    continue
                if G.has_edge(src, dst):
                    accum['capacity'].append(G.edges[src, dst]['bandwidth'])
                try:
                    flows = T[src, dst]['Flows']
                except (TypeError, IndexError, KeyError):
                    continue
                for f_id, flow in enumerate(flows):
                    if flow['AvgBw'] == 0 or flow['PktsGen'] == 0:
                        continue
                    if P[src, dst]['Flows'][f_id]['AvgDelay'] <= 0:
                        continue
                    accum['traffic'].append(flow['AvgBw'])
                    accum['packets'].append(flow['PktsGen'])
                    accum['eq_lambda'].append(flow['TimeDistParams'].get('EqLambda', 0))
                    accum['avg_pkts_lambda'].append(flow['TimeDistParams'].get('AvgPktsLambda', 0))
                    accum['exp_max_factor'].append(flow['TimeDistParams'].get('ExpMaxFactor', 0))
                    accum['delta'].append(delta_lookup.get((src, dst), 0.5))

                    for h_1, h_2 in [sample.get_routing_matrix()[src, dst][i:i + 2]
                                     for i in range(len(sample.get_routing_matrix()[src, dst]) - 1)]:
                        q_s = str(G.nodes[h_1]['queueSizes']).split(',')
                        avg_pkt_size = flow['AvgBw'] / flow['PktsGen']
                        for q in range(G.nodes[h_1]['levelsQoS']):
                            q_size = int(q_s[q]) if q < len(q_s) else int(q_s[0])
                            accum['queue_size'].append(q_size * avg_pkt_size)

        count += 1
        if count % 20 == 0:
            print(f"  {count}/{args.n} samples processed")
        if count >= args.n:
            break

    print(f"\nProcessed {count} samples.\n")
    print("# Paste this into slicing/delay_model.py, replacing the existing z_score dict:\n")
    print("        self.z_score = {")
    for feat in ['traffic', 'packets', 'eq_lambda', 'avg_pkts_lambda', 'exp_max_factor',
                 'delta', 'capacity', 'queue_size']:
        arr = np.array(accum[feat], dtype=np.float64)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        if std == 0:
            std = 1.0
        print(f"            '{feat}': [{mean}, {std}],")
    print("        }")


if __name__ == '__main__':
    main()
