"""
run_step7_eval_robustness.py — EVALUATION robustness check (no retraining).

Loads the 5 already-trained pilot models and re-evaluates them on FRESH random
test samples (drawn from test simulations AFTER the original 300 used for the
reported numbers), to test whether the ~1pp MAPE ordering is stable or an
artifact of which test sims were used.

Method: for each model, run inference over a pool of fresh test sims, store
per-simulation (sum of absolute-percentage-error, #flows); then compute the
micro-MAPE over the full pool and over n seeded random subsets of size
--draw-size (reported as mean +/- std and min..max). Pure inference.

Output: results/step7_robustness_eval.md (+ .json). NOT part of the deliverable
subsection.

Usage:
    python run_step7_eval_robustness.py --pool 1000 --draw-size 300 --n-draws 10
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
CELLS = [
    ("baseline",              "configs/baseline/full.json",         "checkpoints/pilot_n500/baseline_seed42"),
    ("principled_relevant",   "configs/ig/k30_relevant.json",       "checkpoints/pilot_n500/principled/k30_relevant"),
    ("principled_irrelevant", "configs/ig/k30_irrelevant.json",     "checkpoints/pilot_n500/principled/k30_irrelevant"),
    ("random_relevant",       "configs/random/k30_relevant.json",   "checkpoints/pilot_n500/random/k30_relevant"),
    ("random_irrelevant",     "configs/random/k30_irrelevant.json", "checkpoints/pilot_n500/random/k30_irrelevant"),
]


def per_sim_ape(model, ds, pool):
    """Return arrays: per-sim summed APE and per-sim #flows, over `pool` sims."""
    ape_sum, n_flows = [], []
    for i, (inputs, label) in enumerate(ds.take(pool)):
        pred = model(inputs, training=False)
        y = tf.reshape(tf.cast(label, tf.float32), [-1]).numpy()
        p = tf.reshape(tf.cast(pred, tf.float32), [-1]).numpy()
        m = y != 0
        ape = np.abs(p[m] - y[m]) / np.abs(y[m]) * 100.0
        ape_sum.append(float(ape.sum())); n_flows.append(int(m.sum()))
    return np.array(ape_sum), np.array(n_flows)


def micro_mape(ape_sum, n_flows, idx):
    return float(ape_sum[idx].sum() / n_flows[idx].sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=1000, help="fresh test sims to score")
    ap.add_argument("--offset", type=int, default=300, help="skip the original first-N test sims")
    ap.add_argument("--draw-size", type=int, default=300)
    ap.add_argument("--n-draws", type=int, default=10)
    args = ap.parse_args()

    rows = {}
    for name, cfg_path, out_dir in CELLS:
        cfg = json.load(open(os.path.join(REPO, cfg_path)))
        latest = tf.train.latest_checkpoint(os.path.join(REPO, out_dir))
        print(f"[{name}] loading {os.path.basename(latest)}")
        model = RouteNet_Fermi(kept_path_scalars=cfg["kept_features"])
        ds = input_fn(TEST_DIR, shuffle=False, dropped_features=cfg["dropped_features"])
        ds = ds.skip(args.offset)
        # build variables on one batch, then load weights
        for inp, _ in ds.take(1):
            model(inp, training=False); break
        model.load_weights(latest).expect_partial()

        ape_sum, n_flows = per_sim_ape(model, ds, args.pool)
        full = micro_mape(ape_sum, n_flows, np.arange(len(ape_sum)))
        rng = np.random.default_rng(42)
        draws = []
        for _ in range(args.n_draws):
            idx = rng.choice(len(ape_sum), size=min(args.draw_size, len(ape_sum)), replace=False)
            draws.append(micro_mape(ape_sum, n_flows, idx))
        draws = np.array(draws)
        rows[name] = dict(full_pool=full, n_sims=int(len(ape_sum)),
                          draw_mean=float(draws.mean()), draw_std=float(draws.std()),
                          draw_min=float(draws.min()), draw_max=float(draws.max()))
        print(f"  full({len(ape_sum)})={full:.2f}%  draws {args.draw_size}x{args.n_draws}: "
              f"{draws.mean():.2f}+/-{draws.std():.2f}  [{draws.min():.2f},{draws.max():.2f}]")

    # original reported (first-300) for comparison
    orig = {}
    for name, _, out_dir in CELLS:
        m = json.load(open(os.path.join(REPO, out_dir, "metrics.json")))
        orig[name] = m["test_mape"]

    out_json = os.path.join(REPO, "results", "step7_robustness_eval.json")
    json.dump({"args": vars(args), "fresh": rows, "original_first300": orig},
              open(out_json, "w"), indent=2)

    # markdown
    md = []
    md.append("# Step 7 — Evaluation-robustness check (no retraining)\n")
    md.append(f"The 5 trained pilot models re-evaluated on **fresh** test sims "
              f"(skipping the original first {args.offset}). Pool = {args.pool} sims; "
              f"random draws = {args.n_draws} x {args.draw_size} sims (seed 42). "
              f"Pure inference; **separate from the deliverable subsection.**\n")
    md.append("| Model | Reported (first-300) | Fresh full-pool | Fresh random draws (mean +/- std) | [min, max] |")
    md.append("|---|---|---|---|---|")
    order = ["principled_relevant","random_relevant","random_irrelevant","baseline","principled_irrelevant"]
    for name in order:
        r = rows[name]
        md.append(f"| {name} | {orig[name]:.2f}% | {r['full_pool']:.2f}% | "
                  f"{r['draw_mean']:.2f} +/- {r['draw_std']:.2f}% | "
                  f"[{r['draw_min']:.2f}, {r['draw_max']:.2f}] |")
    md.append("\n**Read:** if the ordering and magnitudes are preserved across fresh "
              "samples (and the per-draw std is small relative to the ~1pp spread), the "
              "pilot numbers are stable to the test-sample choice; if the ordering "
              "reshuffles or the std is comparable to the spread, the ~1pp differences "
              "are sampling noise — consistent with the deliverable's interpretation.\n")
    open(os.path.join(REPO, "results", "step7_robustness_eval.md"), "w").write("\n".join(md))
    print("\nWrote results/step7_robustness_eval.md (+ .json)")


if __name__ == "__main__":
    main()
