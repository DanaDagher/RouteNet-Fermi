"""
Quantify exactly how much the max_num_queues=40 truncation actually loses.

delay_model.py does tf.one_hot(inputs['priority'], self.max_num_queues). Any
priority index >= max_num_queues gets silently zeroed by tf.one_hot instead
of raising an error. inspect_features.py found priority values up to 131 in
the training data, meaning truncation is definitely happening; this script
answers the follow-up question that decides whether it is worth fixing:
how many queue nodes, and how many FLOWS (via their queue_to_path /
path_to_queue links), are actually affected.

Two numbers matter:
  1. Fraction of queue nodes with priority >= max_num_queues (cheap upper
     bound on how much of the queue population is affected).
  2. Fraction of FLOWS that touch at least one such queue (the number that
     actually matters, since flows are what the loss is computed on -- a
     queue that no flow ever routes through affected priority is irrelevant).

Read-only, does not modify cache, model, or checkpoints.

Usage:
    python slicing/quantify_priority_truncation.py                    # default cap 40
    python slicing/quantify_priority_truncation.py --cap 40 --split train
"""
import argparse
import glob
import os
import pickle

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cache-root', default=os.path.join(HERE, 'cache'))
    p.add_argument('--split', default='train')
    p.add_argument('--cap', type=int, default=40, help='max_num_queues value used by the model')
    return p.parse_args()


def main():
    args = parse_args()
    cache_dir = os.path.join(args.cache_root, args.split)
    files = sorted(glob.glob(os.path.join(cache_dir, '*.pkl')))
    print(f'Scanning {len(files)} files in {cache_dir}, cap={args.cap} ...\n')

    total_queues = 0
    truncated_queues = 0
    total_flows = 0
    flows_touching_truncated_queue = 0
    samples_with_any_truncation = 0

    for i, fp in enumerate(files):
        with open(fp, 'rb') as fh:
            raw = pickle.load(fh)
        feats, labels = raw

        priority = np.asarray(feats['priority']).reshape(-1)
        n_q = len(priority)
        bad_mask = priority >= args.cap
        n_bad_q = int(bad_mask.sum())

        total_queues += n_q
        truncated_queues += n_bad_q

        n_f = len(labels)
        total_flows += n_f

        if n_bad_q > 0:
            samples_with_any_truncation += 1
            bad_queue_idx = set(np.nonzero(bad_mask)[0].tolist())

            # queue_to_path: for each queue, the list of flow indices that touch it
            queue_to_path = feats['queue_to_path']
            touched_flows = set()
            for q_idx in bad_queue_idx:
                if q_idx < len(queue_to_path):
                    for f_idx in queue_to_path[q_idx]:
                        touched_flows.add(f_idx)
            flows_touching_truncated_queue += len(touched_flows)

        if (i + 1) % 1000 == 0:
            print(f'  ... {i + 1} / {len(files)} files scanned')

    print(f'\nDone.\n')
    print('=' * 70)
    print(f'  Samples scanned                    : {len(files)}')
    print(f'  Samples with >=1 truncated queue    : {samples_with_any_truncation} '
          f'({100*samples_with_any_truncation/len(files):.2f}%)')
    print()
    print(f'  Total queue nodes                  : {total_queues}')
    print(f'  Queue nodes with priority >= {args.cap:<3}   : {truncated_queues} '
          f'({100*truncated_queues/total_queues:.3f}% of all queue nodes)')
    print()
    print(f'  Total flows                        : {total_flows}')
    print(f'  Flows touching >=1 truncated queue  : {flows_touching_truncated_queue} '
          f'({100*flows_touching_truncated_queue/total_flows:.3f}% of all flows)')
    print('=' * 70)
    print('\nInterpretation:')
    print('  The "flows touching a truncated queue" percentage is the one that matters.')
    print('  That is the fraction of training/test flows whose delay prediction was')
    print('  computed with at least one queue along its path missing its priority signal.')


if __name__ == '__main__':
    main()
