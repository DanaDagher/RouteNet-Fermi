"""
Pre-process the slicing dataset ONCE, in parallel across CPU cores, and cache
each sample's tensors to disk. Training then reads the cache (milliseconds per
sample) instead of re-parsing tarballs + rebuilding graphs (~8s per sample).

Uses multiprocessing (real parallelism, bypasses Python's GIL). The GPU is NOT
touched here (CUDA disabled), so it's safe to run while someone else uses the GPU.

Reuses the *validated* data-building logic from data_generator.py unchanged
(network_to_hypergraph + hypergraph_to_raw); it only pickles the result.

Usage (run from the repo root, inside your venv):
    python slicing/preprocess_cache.py                 # all splits, auto workers
    python slicing/preprocess_cache.py --workers 12
    python slicing/preprocess_cache.py --splits train  # one split only

Resumable: a tarball already cached (its .done marker exists) is skipped, so you
can safely re-run if it gets interrupted.
"""
import argparse
import os
import sys
import time
import pickle
import traceback
from multiprocessing import Pool

# Do NOT grab the GPU during preprocessing (be a good neighbour, and avoid any
# CUDA/fork interaction). Must be set before TF is imported anywhere.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'new_dataset'))

import networkx as nx  # noqa: E402
from datanetAPI import DatanetAPI  # noqa: E402
from data_generator import network_to_hypergraph, hypergraph_to_raw  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', default=os.path.join(HERE, 'data'),
                   help='Dir containing train/ val/ test/ splits')
    p.add_argument('--cache-root', default=os.path.join(HERE, 'cache'),
                   help='Where to write the cache (train/ val/ test/ subdirs)')
    p.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 4) - 2),
                   help='Parallel processes (default: CPU count - 2)')
    return p.parse_args()


def process_tarball(task):
    """Cache every valid sample of one tarball. Runs in a worker process."""
    data_dir, cache_dir, root, fname = task
    done_marker = os.path.join(cache_dir, '.done_' + fname)
    if os.path.exists(done_marker):
        return (fname, 0, 'skipped')
    try:
        api = DatanetAPI(data_dir, shuffle=False)
        # NOTE: DatanetAPI.set_files_to_process is buggy in this dataset version
        # (it shadows the builtin `tuple` and always raises). __iter__ reads
        # _selected_tuple_files directly when it is non-empty, so set it here.
        api._selected_tuple_files = [(root, fname)]
        n = 0
        for i, sample in enumerate(api):
            G = nx.DiGraph(sample.get_topology_object())
            T = sample.get_traffic_matrix()
            R = sample.get_routing_matrix()
            P = sample.get_performance_matrix()
            slices = sample.get_slices()
            HG = network_to_hypergraph(G=G, R=R, T=T, P=P, slices=slices)
            if HG is None:
                continue
            raw = hypergraph_to_raw(HG)
            # same filter the live generator applies: skip non-positive delays
            if not all(x > 0 for x in raw[1]):
                continue
            out = os.path.join(cache_dir, '{}_{:04d}.pkl'.format(fname, i))
            with open(out, 'wb') as fh:
                pickle.dump(raw, fh, protocol=pickle.HIGHEST_PROTOCOL)
            n += 1
        with open(done_marker, 'w') as fh:
            fh.write(str(n))
        return (fname, n, 'ok')
    except Exception as e:
        return (fname, 0, 'ERROR: {}: {}'.format(type(e).__name__, e))


def main():
    args = parse_args()
    print('Workers: {}'.format(args.workers))

    for split in args.splits:
        data_dir = os.path.abspath(os.path.join(args.data_root, split))
        cache_dir = os.path.abspath(os.path.join(args.cache_root, split))
        if not os.path.isdir(data_dir):
            print('  [skip] no data dir: {}'.format(data_dir))
            continue
        os.makedirs(cache_dir, exist_ok=True)

        api = DatanetAPI(data_dir, shuffle=False)
        tarballs = api.get_available_files()  # list of (root, fname)
        tasks = [(data_dir, cache_dir, root, fname) for root, fname in tarballs]

        print('\n== split "{}": {} tarballs -> {} =='.format(split, len(tasks), cache_dir))
        t0 = time.time()
        total = 0
        with Pool(processes=args.workers) as pool:
            for k, (fname, n, status) in enumerate(pool.imap_unordered(process_tarball, tasks), 1):
                total += n
                flag = '' if status in ('ok', 'skipped') else '  <<< ' + status
                print('  [{}/{}] {}  (+{} samples, {}){}'.format(
                    k, len(tasks), fname, n, status, flag))
        dt = time.time() - t0
        print('  split "{}" done: {} samples cached in {:.0f}s'.format(split, total, dt))

    print('\nAll requested splits processed. Cache at: {}'.format(os.path.abspath(args.cache_root)))


if __name__ == '__main__':
    main()
