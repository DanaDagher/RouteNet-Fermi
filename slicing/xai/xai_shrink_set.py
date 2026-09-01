"""
Shrink shared_900.json down to N per slice, WITHOUT re-running the freeze
selection. Safe to do this way because xai_freeze_samples.py's per-slice
round-robin order does not depend on the target count -- the first N records
per slice in shared_900.json are byte-identical to what a fresh run with
--shared-per-slice N would have produced. This just takes that same prefix.

Usage:
    python slicing/xai/xai_shrink_set.py \
        --in slicing/xai/sets/shared_900.json \
        --out slicing/xai/sets/shared_300.json --per-slice 100
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--per-slice', type=int, required=True)
    args = ap.parse_args()

    with open(args.inp) as fh:
        payload = json.load(fh)
    records = payload['records']

    kept = {}
    out_records = []
    for r in records:
        s = r['slice_type']
        kept.setdefault(s, 0)
        if kept[s] < args.per_slice:
            out_records.append(r)
            kept[s] += 1

    meta = dict(payload['meta'])
    meta['shrunk_from'] = args.inp
    meta['per_slice'] = args.per_slice
    with open(args.out, 'w') as fh:
        json.dump({'meta': meta, 'records': out_records}, fh, indent=1)
    print(f"Wrote {args.out}: {len(out_records)} flows, per-slice counts={kept}")


if __name__ == '__main__':
    main()
