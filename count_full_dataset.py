"""
count_full_dataset.py — EXACT simulation count of the all_multiplexed dataset.

Each archive results_*.tar.gz contains a simulationResults.txt whose number of
lines equals the number of simulations in that archive. The per-archive count
is NOT constant (it varies by topology: ~5 for GEANT2, ~19 for NSFNET, ~10-13
for GBN), so the only correct total is the SUM of all line counts.

This reads each archive's simulationResults.txt directly from the gzip stream
(no extraction to disk) and counts its lines.

Output: results/dataset_size.json  and prints a summary table.

Run:  venv/Scripts/python.exe count_full_dataset.py
"""
import os, sys, glob, tarfile, json, time

REPO = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(REPO, "data", "traffic_models", "all_multiplexed")

SPLITS = [
    ("train", "geant2-multiplexed"),
    ("train", "nsfnet-multiplexed"),
    ("test",  "gbn-multiplexed"),
]


def count_sims_in_archive(path):
    """Return #lines of simulationResults.txt inside the tar.gz (=#simulations)."""
    with tarfile.open(path, "r:gz") as tar:
        for m in tar:
            if m.name.endswith("simulationResults.txt"):
                f = tar.extractfile(m)
                if f is None:
                    return 0
                # count newlines in binary stream (fast, memory-light)
                n = 0
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    n += chunk.count(b"\n")
                return n
    return 0


def main():
    t0 = time.time()
    result = {"per_split": {}, "grand_total": 0, "archives_total": 0}
    grand = 0
    arch_total = 0
    for split, topo in SPLITS:
        d = os.path.join(SRC, split, topo)
        archives = sorted(glob.glob(os.path.join(d, "results_*.tar.gz")))
        n_arch = len(archives)
        total = 0
        for i, a in enumerate(archives):
            try:
                total += count_sims_in_archive(a)
            except Exception as e:
                print(f"  WARN {os.path.basename(a)}: {e}", flush=True)
            if (i + 1) % 500 == 0:
                print(f"  [{split}/{topo}] {i+1}/{n_arch} archives, "
                      f"{total:,} sims so far ({time.time()-t0:.0f}s)", flush=True)
        result["per_split"][f"{split}/{topo}"] = {
            "archives": n_arch, "simulations": total,
            "sims_per_archive_mean": round(total / n_arch, 2) if n_arch else 0,
        }
        grand += total
        arch_total += n_arch
        print(f"DONE {split}/{topo}: {n_arch:,} archives -> {total:,} simulations",
              flush=True)

    result["grand_total"] = grand
    result["archives_total"] = arch_total
    # convenience sub-totals
    tr = sum(v["simulations"] for k, v in result["per_split"].items() if k.startswith("train/"))
    te = sum(v["simulations"] for k, v in result["per_split"].items() if k.startswith("test/"))
    result["train_pool"] = tr
    result["test_pool"] = te
    result["used_train"] = 500
    result["used_test"] = 300
    result["used_train_pct"] = round(500 / tr * 100, 4) if tr else None
    result["used_test_pct"] = round(300 / te * 100, 4) if te else None
    result["test_flow_predictions_used"] = 81600  # 300 GBN sims x 272 flows
    result["elapsed_seconds"] = round(time.time() - t0, 1)

    out = os.path.join(REPO, "results", "dataset_size.json")
    json.dump(result, open(out, "w"), indent=2)

    print("\n=================== EXACT DATASET SIZE ===================")
    for k, v in result["per_split"].items():
        print(f"  {k:28s}: {v['archives']:>6,} archives  {v['simulations']:>9,} sims  "
              f"({v['sims_per_archive_mean']}/archive)")
    print(f"  {'TRAIN pool (NSFNET+GEANT2)':28s}: {'':>6}          {tr:>9,} sims")
    print(f"  {'TEST pool (GBN)':28s}: {'':>6}          {te:>9,} sims")
    print(f"  {'GRAND TOTAL':28s}: {arch_total:>6,} archives  {grand:>9,} sims")
    print(f"\n  Used 500 train ({result['used_train_pct']}%) + 300 test ({result['used_test_pct']}%)")
    print(f"  300 test sims = 81,600 per-flow delay predictions")
    print(f"  Wrote {os.path.relpath(out, REPO)}  ({result['elapsed_seconds']}s)")


if __name__ == "__main__":
    main()
