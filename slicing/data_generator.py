"""
Data generator for the Zenodo network-slicing dataset (10610616, Farreras et al.)
adapted from scheduling/delay/data_generator.py (RouteNet-Fermi, UPC, Apache 2.0).

Adaptations vs the scheduling generator (each one validated empirically, see
verify_slicing_dataset.py):
  1. schedulingWeights are per output port, ';'-separated (one group per port,
     indexed by the link's 'port' attribute); each group sums to ~100 (%).
  2. Slice features joined onto each flow via its unique (origin, destination)
     pair: slice type (0=eMBB, 1=mMTC, 2=URLLC) and delta (provisioning factor,
     Vreserved = bw*(0.9+0.3*delta) per the Data in Brief paper).
  3. ToS is the origin-node id used as the flow->queue key (tosToQoSqueue),
     NOT a QoS class. It is used for queue mapping only, never as a feature.
  4. Paths come from the routing matrix (ground truth validated against
     measured link utilization). slice['path'] is the reservation-time path
     and must NOT be used to build the graph.
"""
import networkx as nx
import numpy as np
import tensorflow as tf

import os
import sys
import glob
import pickle
import random
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'new_dataset'))
from datanetAPI import DatanetAPI

POLICIES = np.array(['WFQ', 'SP', 'DRR', 'FIFO'])
SLICE_TYPES = {'eMBB': 0, 'mMTC': 1, 'URLLC': 2}


def generator(data_dir, shuffle):
    try:
        data_dir = data_dir.decode('UTF-8')
    except (UnicodeDecodeError, AttributeError):
        pass
    tool = DatanetAPI(data_dir, shuffle=shuffle)
    it = iter(tool)
    for sample in it:
        G = nx.DiGraph(sample.get_topology_object())
        T = sample.get_traffic_matrix()
        R = sample.get_routing_matrix()
        P = sample.get_performance_matrix()
        slices = sample.get_slices()
        HG = network_to_hypergraph(G=G, R=R, T=T, P=P, slices=slices)
        if HG is None:
            continue
        ret = hypergraph_to_input_data(HG)
        # skip samples with zero or negative delays
        if not all(x > 0 for x in ret[1]):
            continue
        yield ret


def _slice_lookup(slices):
    """Map (origin, destination) -> (slice_type_int, delta). Unique per pair
    (verified: no (src,dst) pair is shared by two flows)."""
    lookup = {}
    for sl in slices:
        for f in sl['flows']:
            lookup[(f['origin_node'], f['destination'])] = (SLICE_TYPES[sl['type']], sl['delta'])
    return lookup


def _port_weights(G, h_1, h_2):
    """Per-queue WFQ weights of the output port h_1->h_2, normalized to sum 1.
    schedulingWeights format: one ';'-separated group per output port, indexed
    by the link's 'port' attribute; '-' on FIFO/terminal nodes."""
    w_str = str(G.nodes[h_1].get('schedulingWeights', '-'))
    if w_str == '-':
        return None
    groups = w_str.split(';')
    port = G.edges[h_1, h_2].get('port', 0)
    group = groups[port] if port < len(groups) else groups[0]
    if group == '-' or group == '':
        return None
    q_w = [float(w) for w in group.split(',')]
    w_sum = sum(q_w)
    if w_sum <= 0:
        return None
    return [w / w_sum for w in q_w]


def network_to_hypergraph(G, R, T, P, slices):
    lookup = _slice_lookup(slices)
    D_G = nx.DiGraph()
    for src in range(G.number_of_nodes()):
        for dst in range(G.number_of_nodes()):
            if src == dst:
                continue
            if G.has_edge(src, dst):
                D_G.add_node('l_{}_{}'.format(src, dst),
                             capacity=G.edges[src, dst]['bandwidth'],
                             policy=np.where(G.nodes[src]['schedulingPolicy'] == POLICIES)[0][0])
            try:
                flows = T[src, dst]['Flows']
            except (TypeError, IndexError, KeyError):
                continue
            for f_id in range(len(flows)):
                flow = flows[f_id]
                if flow['AvgBw'] == 0 or flow['PktsGen'] == 0:
                    continue
                if (src, dst) not in lookup:
                    # every flow must belong to a slice; abort sample otherwise
                    return None
                s_type, s_delta = lookup[(src, dst)]
                D_G.add_node('p_{}_{}_{}'.format(src, dst, f_id),
                             source=src,
                             destination=dst,
                             tos=int(flow['ToS']),
                             traffic=flow['AvgBw'],
                             packets=flow['PktsGen'],
                             length=len(R[src, dst]) - 1,
                             model=flow['TimeDist'].value,
                             eq_lambda=flow['TimeDistParams'].get('EqLambda', 0),
                             avg_pkts_lambda=flow['TimeDistParams'].get('AvgPktsLambda', 0),
                             exp_max_factor=flow['TimeDistParams'].get('ExpMaxFactor', 0),
                             slice_type=s_type,
                             delta=s_delta,
                             delay=P[src, dst]['Flows'][f_id]['AvgDelay'])

                for h_1, h_2 in [R[src, dst][i:i + 2] for i in range(0, len(R[src, dst]) - 1)]:
                    D_G.add_edge('l_{}_{}'.format(h_1, h_2), 'p_{}_{}_{}'.format(src, dst, f_id))
                    D_G.add_edge('p_{}_{}_{}'.format(src, dst, f_id), 'l_{}_{}'.format(h_1, h_2))

                    q_s = str(G.nodes[h_1]['queueSizes']).split(',')
                    q_w = _port_weights(G, h_1, h_2)
                    q_map = [m.split(',') for m in str(G.nodes[h_1]['tosToQoSqueue']).split(';')]
                    avg_pkt_size = flow['AvgBw'] / flow['PktsGen']
                    for q in range(G.nodes[h_1]['levelsQoS']):
                        q_size = int(q_s[q]) if q < len(q_s) else int(q_s[0])
                        D_G.add_node('q_{}_{}_{}'.format(h_1, h_2, q),
                                     queue_size=q_size * avg_pkt_size,
                                     priority=q,
                                     weight=q_w[q] if q_w is not None and q < len(q_w) else 0)
                        D_G.add_edge('q_{}_{}_{}'.format(h_1, h_2, q), 'l_{}_{}'.format(h_1, h_2))
                        if q < len(q_map) and str(int(flow['ToS'])) in q_map[q]:
                            D_G.add_edge('p_{}_{}_{}'.format(src, dst, f_id), 'q_{}_{}_{}'.format(h_1, h_2, q))
                            D_G.add_edge('q_{}_{}_{}'.format(h_1, h_2, q), 'p_{}_{}_{}'.format(src, dst, f_id))

    D_G.remove_nodes_from([node for node, in_degree in D_G.in_degree() if in_degree == 0])
    return D_G


def hypergraph_to_raw(HG):
    """Build the per-sample feature dict + labels using plain numpy arrays and
    Python lists only (no TF tensors) so the result is picklable for caching.
    Identical logic to the original hypergraph_to_input_data; the TF conversion
    of the 5 index structures is deferred to _to_tensors()."""
    n_q = 0
    n_p = 0
    n_l = 0
    mapping = {}
    for entity in list(HG.nodes()):
        if entity.startswith('q'):
            mapping[entity] = ('q_{}'.format(n_q))
            n_q += 1
        elif entity.startswith('p'):
            mapping[entity] = ('p_{}'.format(n_p))
            n_p += 1
        elif entity.startswith('l'):
            mapping[entity] = ('l_{}'.format(n_l))
            n_l += 1

    HG = nx.relabel_nodes(HG, mapping)

    link_to_path = []
    queue_to_path = []
    path_to_queue = []
    queue_to_link = []
    path_to_link = []

    for node in HG.nodes:
        in_nodes = [s for s, d in HG.in_edges(node)]
        if node.startswith('q_'):
            path = []
            for n in in_nodes:
                if n.startswith('p_'):
                    path_pos = []
                    for _, d in HG.out_edges(n):
                        if d.startswith('q_'):
                            path_pos.append(d)
                    path.append([int(n.replace('p_', '')), path_pos.index(node)])
            path_to_queue.append(path)
        elif node.startswith('p_'):
            links = []
            queues = []
            for n in in_nodes:
                if n.startswith('l_'):
                    links.append(int(n.replace('l_', '')))
                elif n.startswith('q_'):
                    queues.append(int(n.replace('q_', '')))
            link_to_path.append(links)
            queue_to_path.append(queues)
        elif node.startswith('l_'):
            queues = []
            paths = []
            for n in in_nodes:
                if n.startswith('q_'):
                    queues.append(int(n.replace('q_', '')))
                elif n.startswith('p_'):
                    path_pos = []
                    for _, d in HG.out_edges(n):
                        if d.startswith('l_'):
                            path_pos.append(d)
                    paths.append([int(n.replace('p_', '')), path_pos.index(node)])
            path_to_link.append(paths)
            queue_to_link.append(queues)

    return {"traffic": np.expand_dims(list(nx.get_node_attributes(HG, 'traffic').values()), axis=1),
            "packets": np.expand_dims(list(nx.get_node_attributes(HG, 'packets').values()), axis=1),
            "length": list(nx.get_node_attributes(HG, 'length').values()),
            "model": list(nx.get_node_attributes(HG, 'model').values()),
            "eq_lambda": np.expand_dims(list(nx.get_node_attributes(HG, 'eq_lambda').values()), axis=1),
            "avg_pkts_lambda": np.expand_dims(list(nx.get_node_attributes(HG, 'avg_pkts_lambda').values()), axis=1),
            "exp_max_factor": np.expand_dims(list(nx.get_node_attributes(HG, 'exp_max_factor').values()), axis=1),
            "slice_type": list(nx.get_node_attributes(HG, 'slice_type').values()),
            "delta": np.expand_dims(list(nx.get_node_attributes(HG, 'delta').values()), axis=1),
            "capacity": np.expand_dims(list(nx.get_node_attributes(HG, 'capacity').values()), axis=1),
            "queue_size": np.expand_dims(list(nx.get_node_attributes(HG, 'queue_size').values()), axis=1),
            "policy": list(nx.get_node_attributes(HG, 'policy').values()),
            "priority": list(nx.get_node_attributes(HG, 'priority').values()),
            "weight": np.expand_dims(list(nx.get_node_attributes(HG, 'weight').values()), axis=1),
            "link_to_path": link_to_path,
            "queue_to_path": queue_to_path,
            "queue_to_link": queue_to_link,
            "path_to_queue": path_to_queue,
            "path_to_link": path_to_link
            }, list(nx.get_node_attributes(HG, 'delay').values())


def _to_tensors(raw):
    """Convert the 5 index structures of a raw sample to TF ragged tensors.
    Everything else (numpy arrays / lists) is passed through unchanged and
    converted by from_generator per the output signature."""
    feats, labels = raw
    feats = dict(feats)
    feats["link_to_path"] = tf.ragged.constant(feats["link_to_path"])
    feats["queue_to_path"] = tf.ragged.constant(feats["queue_to_path"])
    feats["queue_to_link"] = tf.ragged.constant(feats["queue_to_link"])
    feats["path_to_queue"] = tf.ragged.constant(feats["path_to_queue"], ragged_rank=1)
    feats["path_to_link"] = tf.ragged.constant(feats["path_to_link"], ragged_rank=1)
    return feats, labels


def hypergraph_to_input_data(HG):
    """Backward-compatible wrapper: build raw sample then convert to tensors."""
    return _to_tensors(hypergraph_to_raw(HG))


def _output_signature():
    return (
        {"traffic": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
         "packets": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
         "length": tf.TensorSpec(shape=None, dtype=tf.int32),
         "model": tf.TensorSpec(shape=None, dtype=tf.int32),
         "eq_lambda": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
         "avg_pkts_lambda": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
         "exp_max_factor": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
         "slice_type": tf.TensorSpec(shape=None, dtype=tf.int32),
         "delta": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
         "capacity": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
         "queue_size": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
         "policy": tf.TensorSpec(shape=None, dtype=tf.int32),
         "priority": tf.TensorSpec(shape=None, dtype=tf.int32),
         "weight": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
         "link_to_path": tf.RaggedTensorSpec(shape=(None, None), dtype=tf.int32),
         "queue_to_path": tf.RaggedTensorSpec(shape=(None, None), dtype=tf.int32),
         "queue_to_link": tf.RaggedTensorSpec(shape=(None, None), dtype=tf.int32),
         "path_to_queue": tf.RaggedTensorSpec(shape=(None, None, 2), dtype=tf.int32, ragged_rank=1),
         "path_to_link": tf.RaggedTensorSpec(shape=(None, None, 2), dtype=tf.int32, ragged_rank=1),
         },
        tf.TensorSpec(shape=None, dtype=tf.float32),
    )


def input_fn(data_dir, shuffle=False):
    ds = tf.data.Dataset.from_generator(generator,
                                        args=[data_dir, shuffle],
                                        output_signature=_output_signature())
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def cached_generator(cache_dir, shuffle, max_delay_cap):
    """Yield pre-processed samples from a cache directory of .pkl files.
    Each .pkl holds the raw (picklable) output of hypergraph_to_raw; we only
    apply the fast TF conversion here -- no tarball parsing, no networkx.

    Samples whose largest per-flow delay exceeds max_delay_cap are skipped.
    Pass max_delay_cap <= 0 or a very large number to disable the filter.
    """
    try:
        cache_dir = cache_dir.decode('UTF-8')
    except (UnicodeDecodeError, AttributeError):
        pass
    try:
        cap = float(max_delay_cap)
    except (TypeError, ValueError):
        cap = float('inf')
    if cap <= 0:
        cap = float('inf')
    files = sorted(glob.glob(os.path.join(cache_dir, '*.pkl')))
    if shuffle:
        random.Random(1234).shuffle(files)
    for fp in files:
        with open(fp, 'rb') as fh:
            raw = pickle.load(fh)
        _, labels = raw
        if labels and max(labels) > cap:
            continue
        yield _to_tensors(raw)


def input_fn_cached(cache_dir, shuffle=False, max_delay_cap=float('inf')):
    ds = tf.data.Dataset.from_generator(cached_generator,
                                        args=[cache_dir, shuffle, max_delay_cap],
                                        output_signature=_output_signature())
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds
