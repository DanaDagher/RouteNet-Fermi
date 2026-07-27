"""
run_step7_perturbation.py — INFERENCE-time perturbation test.

Complements the retraining ablation for `traffic` and `packets`, which cannot
be dropped from the model because the delay formula reads them directly
(load = traffic/capacity, pkt_size = traffic/packets, trans_delay = pkt_size *
Sum(1/capacity)). See results/two_doors_and_ml_locus_REPORT.md.

Method: take a trained model (baseline or principled_irrelevant k30) and, at
inference on the 300-sim test set, replace one or two input features with a
random permutation *across flows within the same simulation*. The marginal
distribution is preserved (each value is a real value from a real flow), so
no distribution-shift complaint applies; the signal is destroyed because
flow i now receives flow j's traffic profile.

When `traffic` and `packets` are shuffled together with the SAME permutation,
the per-flow ratio `traffic/packets = pkt_size` stays realistic (it is
flow j's real packet size), so only `load` and the readout are stressed.

Also runs shuffles on two XAI-bottom-ranked features (eq_lambda,
avg_pkts_lambda) as a fidelity control: same procedure, same distribution,
but on features the ranking said are unimportant. If XAI is faithful, those
should give near-zero DeltaMAPE.

Output: results/step7_perturbation_summary.csv, .json, and .md.

Usage:
    python run_step7_perturbation.py --pool 300 --n-seeds 5
"""
import argparse, json, os, sys, csv
REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "traffic_models"))
sys.path.insert(0, os.path.join(REPO, "traffic_models", "delay"))
import numpy as np
import tensorflow as tf
from delay_model import RouteNet_Fermi
from data_generator import input_fn

TEST_DIR = os.path.join(REPO, "data", "traffic_models", "all_multiplexed", "test")

# Perturbation targets. Default = the GPU-retrained baseline (all 10 features,
# 150 epochs, full data). Uncomment the upstream row to also cross-check on the
# frozen paper checkpoint. A cell is silently skipped if its checkpoint dir has
# no weight files.
CELLS = [
    ("local_sanity_baseline", "configs/baseline/full.json", "checkpoints/local_sanity/baseline_seed42"),
    # ("gpu_baseline",        "configs/baseline/full.json", "checkpoints/gpu_full/baseline_seed42"),
    # ("upstream_baseline",   "configs/baseline/full.json", "traffic_models/delay/ckpt_dir_all_multiplexed"),
]

# (mode_name, list_of_features_to_shuffle_with_same_permutation)
# A feature is only perturbed if it is present in the model's input dict
# (i.e. always-present structural features, or in kept_features).
PERT_MODES = [
    ("clean",                    []),
    ("shuffle_sigma",            ["sigma"]),                          # top-3, rank #1
    ("shuffle_traffic",          ["traffic"]),                        # top-3, rank #2
    ("shuffle_packets",          ["packets"]),                        # top-3, rank #3
    ("shuffle_traffic_packets",  ["traffic", "packets"]),             # keeps pkt_size realistic
    ("shuffle_top3",             ["sigma", "traffic", "packets"]),    # full necessity test
    ("shuffle_eq_lambda",        ["eq_lambda"]),                      # bottom-7 control (rank #5)
    ("shuffle_avg_pkts_lambda",  ["avg_pkts_lambda"]),                # bottom-7 control (rank #10)
]
# NOTE ON SILENT SKIPS: a mode is skipped if any of its features is not in the
# model's input dict (data_generator drops 'sigma' from the input for the
# principled-irrelevant variant; 'traffic' and 'packets' are always kept for
# the physics wrapper). On the GPU baseline (all 10 kept), every mode runs.


def apply_shuffle(inputs, feats_to_shuffle, seed):
    """Return a NEW inputs dict with the given features permuted along axis 0.
    Same permutation is used for every feature in `feats_to_shuffle`, so paired
    features (traffic + packets -> pkt_size) stay physically consistent."""
    if not feats_to_shuffle:
        return inputs
    out = dict(inputs)
    n = int(tf.shape(out[feats_to_shuffle[0]])[0])
    perm = tf.constant(np.random.default_rng(seed).permutation(n), dtype=tf.int32)
    for f in feats_to_shuffle:
        if f in out:  # silently skip features not present in this model's input dict
            out[f] = tf.gather(out[f], perm)
    return out


def per_sim_ape(model, ds, pool, feats_to_shuffle, seed):
    """Iterate `pool` sims, apply shuffle, return per-sim (sum APE, #flows)."""
    ape_sum, n_flows = [], []
    for i, (inputs, label) in enumerate(ds):  # ds already .take(pool).cache()'d in main
        inputs_p = apply_shuffle(inputs, feats_to_shuffle, seed=seed + i)
        pred = model(inputs_p, training=False)
        y = tf.reshape(tf.cast(label, tf.float32), [-1]).numpy()
        p = tf.reshape(tf.cast(pred, tf.float32), [-1]).numpy()
        m = y != 0
        ape = np.abs(p[m] - y[m]) / np.abs(y[m]) * 100.0
        ape_sum.append(float(ape.sum()))
        n_flows.append(int(m.sum()))
    return np.array(ape_sum), np.array(n_flows)


def micro_mape(ape_sum, n_flows):
    return float(ape_sum.sum() / n_flows.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=300,
                    help="test sims to score per (model, mode, seed)")
    ap.add_argument("--n-seeds", type=int, default=5,
                    help="number of shuffle seeds to average over")
    args = ap.parse_args()

    rows = []  # list of dicts for CSV
    per_model_clean = {}

    for name, cfg_path, ckpt_dir in CELLS:
        cfg = json.load(open(os.path.join(REPO, cfg_path)))
        latest = tf.train.latest_checkpoint(os.path.join(REPO, ckpt_dir))
        print(f"\n[{name}] loading {os.path.basename(latest) if latest else '(none)'}")
        if latest is None:
            print(f"  SKIP: no checkpoint found in {ckpt_dir}")
            continue
        model = RouteNet_Fermi(kept_path_scalars=cfg["kept_features"])
        # .cache() populates on first pass, all subsequent passes read from RAM
        # -- avoids re-parsing the same 300 sims 36 times.
        ds = input_fn(TEST_DIR, shuffle=False, dropped_features=cfg["dropped_features"])
        ds = ds.take(args.pool).cache()
        for inp, _ in ds.take(1):
            model(inp, training=False)
            break
        model.load_weights(latest).expect_partial()

        # which of the perturbation features are actually present in this model
        # (always-present: traffic, packets; kept only if in cfg["kept_features"])
        always_present = {"traffic", "packets"}
        present = always_present | set(cfg["kept_features"])

        for mode_name, feats in PERT_MODES:
            missing = [f for f in feats if f not in present]
            if missing:
                print(f"  [{mode_name}] SKIP (features not in model: {missing})")
                continue

            seeds = [0] if mode_name == "clean" else list(range(args.n_seeds))
            per_seed_mape = []
            for seed in seeds:
                ape_sum, n_flows = per_sim_ape(model, ds, args.pool, feats, seed=1000 * (seed + 1))
                mape = micro_mape(ape_sum, n_flows)
                per_seed_mape.append(mape)
            per_seed_mape = np.array(per_seed_mape)

            if mode_name == "clean":
                per_model_clean[name] = float(per_seed_mape[0])
                delta = 0.0
            else:
                delta = float(per_seed_mape.mean()) - per_model_clean.get(name, np.nan)

            row = dict(
                model=name,
                mode=mode_name,
                features=",".join(feats) if feats else "-",
                n_seeds=len(seeds),
                mape_mean=float(per_seed_mape.mean()),
                mape_std=float(per_seed_mape.std()) if len(per_seed_mape) > 1 else 0.0,
                mape_min=float(per_seed_mape.min()),
                mape_max=float(per_seed_mape.max()),
                delta_vs_clean=delta,
            )
            rows.append(row)
            print(f"  [{mode_name:<26}] MAPE = {row['mape_mean']:6.2f}% "
                  f"+/- {row['mape_std']:4.2f}   Delta = {delta:+6.2f} pp")

    # --------------- write CSV ---------------
    out_csv = os.path.join(REPO, "results", "step7_perturbation_summary.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # --------------- write JSON ---------------
    out_json = os.path.join(REPO, "results", "step7_perturbation_summary.json")
    json.dump({"args": vars(args), "clean_mape_per_model": per_model_clean, "results": rows},
              open(out_json, "w"), indent=2)

    # --------------- write MD ---------------
    md = []
    md.append("# Step 7 — Perturbation test (inference-time, no retraining)\n")
    md.append(
        "Complements the retraining ablation, which is structurally blocked from removing "
        "`traffic` / `packets` (they feed the fixed physics: `load`, `pkt_size`, `trans_delay`). "
        "Here we take a trained model and, at inference, replace one or two input features "
        "with a random permutation across flows within the same simulation. Marginal "
        "distribution is preserved (each value is a real value from a real flow), so no "
        "distribution-shift complaint applies. See `results/two_doors_and_ml_locus_REPORT.md`.\n"
    )
    md.append(
        f"Pool = {args.pool} sims per (model, mode); Delta averaged over {args.n_seeds} shuffle seeds.\n"
    )
    md.append("| Model | Perturbation | Features | MAPE (mean +/- std) | Delta vs clean |")
    md.append("|---|---|---|---|---|")
    for r in rows:
        md.append(
            f"| {r['model']} | {r['mode']} | `{r['features']}` | "
            f"{r['mape_mean']:.2f}% +/- {r['mape_std']:.2f} | "
            f"{r['delta_vs_clean']:+.2f} pp |"
        )
    md.append("")
    md.append(
        "**Read:** if XAI ranking is faithful, `shuffle_traffic`, `shuffle_packets`, and "
        "`shuffle_traffic_packets` should produce large positive Deltas, while the "
        "bottom-7 controls (`shuffle_eq_lambda`, `shuffle_avg_pkts_lambda`) should be "
        "near zero. A large Delta on `principled_irrelevant` for the top-3 features "
        "confirms that the reason the retraining ablation could not remove them is not a "
        "technicality: their VALUES are load-bearing via the physics wrapper.\n"
    )
    open(os.path.join(REPO, "results", "step7_perturbation_summary.md"), "w").write("\n".join(md))
    print("\nWrote:")
    print(f"  {out_csv}")
    print(f"  {out_json}")
    print(f"  results/step7_perturbation_summary.md")


if __name__ == "__main__":
    main()
