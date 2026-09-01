"""
XAI phase - step 1: freeze the sample sets.

Selects, deterministically, the exact flows that the XAI phase will explain, so
that Integrated Gradients and KernelSHAP run on the identical set (required for a
fair method comparison) and the whole thing is reproducible from a seed.

A "flow" is one path node in a cached sample. Within a cached .pkl the arrays are
aligned by path-node index i: slice_type[i], label(delay)[i], traffic[i], ... all
refer to the same flow i. We therefore identify a flow by (cache_file, flow_idx).

Three sets are produced (see slicing/XAI_PHASE_PLAN.md sections 2 and 3):

  1. shared_900.json   - 900 flows, stratified 300 / 300 / 300 across
                         eMBB / mMTC / URLLC, drawn from distinct topologies where
                         possible. Reused by BOTH IG and KernelSHAP.
  2. ig_extended.json  - up to N_EXT per slice (default 1000), same selection rule,
                         a superset used by IG only (IG is cheap) for finer
                         subgroup / delay-bin analysis. Contains shared_900 as a
                         subset so nothing is wasted.
  3. background.json    - ~100 flows sampled from the TRAIN cache, KernelSHAP's
                         background / reference distribution. Never drawn from test.

Selection rules (all deterministic, seed 42):
  - candidate flows must satisfy 0 < delay <= MAX_DELAY_CAP (1000 s), the same
    exclusion used in training (METHODOLOGY_JUSTIFICATION.md section 4). Applied
    per-flow here (training applied it per-sample); the intent is identical: keep
    the XAI set inside the delay range the model was actually fit on.
  - within a slice, prefer topology diversity: iterate distinct cache files
    round-robin taking one flow per file before taking a second from any file.

Read-only on the cache. Writes only JSON under --out-dir. Run on the DGX where
the caches live:

    python slicing/xai/xai_freeze_samples.py \
        --test-cache slicing/cache/test \
        --train-cache slicing/cache/train \
        --out-dir slicing/xai/sets
"""
import argparse
import glob
import json
import os
import pickle
import random

SLICE_NAMES = {0: 'eMBB', 1: 'mMTC', 2: 'URLLC'}
MAX_DELAY_CAP = 1000.0
SEED = 42
N_SHARED_PER_SLICE = 300
N_EXT_PER_SLICE = 1000
N_BACKGROUND = 100


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    slic = os.path.dirname(here)
    p = argparse.ArgumentParser()
    p.add_argument('--test-cache', default=os.path.join(slic, 'cache', 'test'))
    p.add_argument('--train-cache', default=os.path.join(slic, 'cache', 'train'))
    p.add_argument('--out-dir', default=os.path.join(here, 'sets'))
    p.add_argument('--shared-per-slice', type=int, default=N_SHARED_PER_SLICE)
    p.add_argument('--ext-per-slice', type=int, default=N_EXT_PER_SLICE)
    p.add_argument('--n-background', type=int, default=N_BACKGROUND)
    p.add_argument('--max-delay-cap', type=float, default=MAX_DELAY_CAP)
    return p.parse_args()


def scan_candidates(cache_dir, max_delay_cap):
    """Return {slice_int: [ (fname, flow_idx, delay), ... ]} for all flows in the
    cache satisfying 0 < delay <= max_delay_cap. fname is the basename only."""
    by_slice = {k: [] for k in SLICE_NAMES}
    files = sorted(glob.glob(os.path.join(cache_dir, '*.pkl')))
    if not files:
        raise SystemExit(f"ERROR: no .pkl files in {cache_dir}")
    n_flows_total = 0
    n_kept = 0
    for fp in files:
        with open(fp, 'rb') as fh:
            feats, labels = pickle.load(fh)
        slice_type = list(feats['slice_type'])
        fname = os.path.basename(fp)
        for i, (s, d) in enumerate(zip(slice_type, labels)):
            n_flows_total += 1
            if not (0 < d <= max_delay_cap):
                continue
            s = int(s)
            if s not in by_slice:
                continue
            by_slice[s].append((fname, i, float(d)))
            n_kept += 1
    return by_slice, files, n_flows_total, n_kept


def select_diverse(candidates, n_target, rng):
    """Pick n_target flows from `candidates` (list of (fname, idx, delay))
    preferring distinct fnames: round-robin one flow per file, shuffled, until
    n_target reached or candidates exhausted."""
    by_file = {}
    for rec in candidates:
        by_file.setdefault(rec[0], []).append(rec)
    files = list(by_file.keys())
    rng.shuffle(files)
    for f in files:
        rng.shuffle(by_file[f])
    picked = []
    round_i = 0
    while len(picked) < n_target:
        progressed = False
        for f in files:
            if round_i < len(by_file[f]):
                picked.append(by_file[f][round_i])
                progressed = True
                if len(picked) >= n_target:
                    break
        if not progressed:
            break  # exhausted every file
        round_i += 1
    return picked


def to_records(picked, slice_int):
    return [{'file': f, 'flow_idx': idx, 'slice_type': slice_int,
             'slice_name': SLICE_NAMES[slice_int], 'delay': d}
            for (f, idx, d) in picked]


def main():
    args = parse_args()
    rng = random.Random(SEED)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Scanning TEST cache: {args.test_cache}")
    by_slice, _, n_total, n_kept = scan_candidates(args.test_cache, args.max_delay_cap)
    print(f"  {n_total} flows total, {n_kept} candidates after 0 < delay <= "
          f"{args.max_delay_cap:g}s filter")
    for k in sorted(SLICE_NAMES):
        print(f"    {SLICE_NAMES[k]:>5}: {len(by_slice[k]):>7} candidate flows "
              f"across {len({r[0] for r in by_slice[k]})} distinct topologies")

    # Extended IG set first (superset), then shared_900 as its per-slice prefix so
    # shared_900 is guaranteed a subset of ig_extended.
    ext_records, shared_records = [], []
    for k in sorted(SLICE_NAMES):
        # use a per-slice child RNG so each slice's draw is independent & stable
        child = random.Random(SEED * 1000 + k)
        picked_ext = select_diverse(by_slice[k], args.ext_per_slice, child)
        if len(picked_ext) < args.shared_per_slice:
            print(f"  WARNING: slice {SLICE_NAMES[k]} only yielded "
                  f"{len(picked_ext)} flows (< {args.shared_per_slice} requested)")
        ext_records.extend(to_records(picked_ext, k))
        shared_records.extend(to_records(picked_ext[:args.shared_per_slice], k))

    print(f"\nScanning TRAIN cache for background: {args.train_cache}")
    by_slice_tr, _, _, n_kept_tr = scan_candidates(args.train_cache, args.max_delay_cap)
    # flatten with slice tag preserved
    flat_train = []
    for k in sorted(by_slice_tr):
        for (f, idx, d) in by_slice_tr[k]:
            flat_train.append((f, idx, d, k))
    bg_rng = random.Random(SEED + 7)
    bg_rng.shuffle(flat_train)
    bg_pick = flat_train[:args.n_background]
    bg_records = [{'file': f, 'flow_idx': idx, 'slice_type': k,
                   'slice_name': SLICE_NAMES[k], 'delay': d}
                  for (f, idx, d, k) in bg_pick]

    meta = {'seed': SEED, 'max_delay_cap': args.max_delay_cap,
            'shared_per_slice': args.shared_per_slice,
            'ext_per_slice': args.ext_per_slice,
            'test_cache': os.path.abspath(args.test_cache),
            'train_cache': os.path.abspath(args.train_cache)}

    def dump(name, records):
        path = os.path.join(args.out_dir, name)
        with open(path, 'w') as fh:
            json.dump({'meta': meta, 'records': records}, fh, indent=1)
        counts = {SLICE_NAMES[k]: sum(1 for r in records if r['slice_type'] == k)
                  for k in sorted(SLICE_NAMES)}
        print(f"  wrote {path}  ({len(records)} flows, {counts})")

    print("\nWriting sets:")
    dump('shared_900.json', shared_records)
    dump('ig_extended.json', ext_records)
    dump('background.json', bg_records)
    print("\nDone. shared_900.json is the frozen set for BOTH IG and KernelSHAP.")


if __name__ == '__main__':
    main()
