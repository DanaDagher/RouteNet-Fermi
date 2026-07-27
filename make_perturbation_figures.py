"""
make_perturbation_figures.py — figures for the perturbation-based fidelity test
(deliverable section 2.5.6.3). Reads results/step7_perturbation_summary.csv and
the two retrained models' metrics.json. Outputs 300-dpi PNGs to results/figures/.

Run:  venv/Scripts/python.exe make_perturbation_figures.py
"""
import os, csv, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
})

REPO = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(REPO, "results", "figures")
os.makedirs(FIG, exist_ok=True)

IG_BLUE, KS_RED, GREEN, GREY = "#2c6fbb", "#c0392b", "#27ae60", "#95a5a6"

# ---- load perturbation summary ----------------------------------------------
rows = {}
with open(os.path.join(REPO, "results", "step7_perturbation_summary.csv")) as f:
    for r in csv.DictReader(f):
        rows[r["mode"]] = {
            "mape": float(r["mape_mean"]),
            "std":  float(r["mape_std"]),
            "delta": float(r["delta_vs_clean"]),
        }
clean = rows["clean"]["mape"]

# ===== Figure A: perturbation necessity test (ΔMAPE per feature) =============
# order: top-3 individually, joint top-3, then bottom-7 controls
mode_order = [
    ("shuffle_sigma",           "sigma\n(#1)",           IG_BLUE),
    ("shuffle_traffic",         "traffic\n(#2)",         IG_BLUE),
    ("shuffle_packets",         "packets\n(#3)",         IG_BLUE),
    ("shuffle_top3",            "top-3\njoint",          KS_RED),
    ("shuffle_eq_lambda",       "eq_lambda\n(#5)",       GREY),
    ("shuffle_avg_pkts_lambda", "avg_pkts_\nlambda(#10)", GREY),
]
labels  = [m[1] for m in mode_order]
deltas  = [rows[m[0]]["delta"] for m in mode_order]
stds    = [rows[m[0]]["std"]   for m in mode_order]
colors  = [m[2] for m in mode_order]

fig, ax = plt.subplots(figsize=(6.6, 3.6))
x = np.arange(len(labels))
bars = ax.bar(x, deltas, yerr=stds, capsize=3, color=colors)
for b, d in zip(bars, deltas):
    ax.text(b.get_x()+b.get_width()/2, d + 0.6, f"+{d:.1f}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel("Δ Test MAPE vs clean (pp)")
ax.set_ylim(0, max(deltas) * 1.18)
ax.set_title("Perturbation necessity test on the retrained baseline\n"
             "shuffling top-3 features breaks the model; bottom-ranked features do not")
# legend proxies
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color=IG_BLUE, label="top-3 feature (individual)"),
    Patch(color=KS_RED,  label="top-3 joint"),
    Patch(color=GREY,    label="bottom-7 control"),
], frameon=False, loc="upper right")
plt.tight_layout()
fA = os.path.join(FIG, "fig_perturbation_delta_mape.png")
plt.savefig(fA, dpi=300); plt.close()

# ===== Figure B: absolute MAPE per mode (clean vs perturbed) =================
labels_b = ["clean"] + [m[1].replace("\n", " ") for m in mode_order]
mapes_b  = [clean]   + [rows[m[0]]["mape"] for m in mode_order]
stds_b   = [0.0]     + [rows[m[0]]["std"]  for m in mode_order]
colors_b = [GREEN]   + [m[2] for m in mode_order]

fig, ax = plt.subplots(figsize=(7.0, 3.6))
x = np.arange(len(labels_b))
bars = ax.bar(x, mapes_b, yerr=stds_b, capsize=3, color=colors_b)
for b, m in zip(bars, mapes_b):
    ax.text(b.get_x()+b.get_width()/2, m + 0.6, f"{m:.1f}", ha="center", fontsize=8.5)
ax.axhline(clean, ls="--", lw=1, color=GREEN, alpha=0.8,
           label=f"clean baseline ({clean:.2f}%)")
ax.set_xticks(x); ax.set_xticklabels(labels_b, rotation=25, ha="right", fontsize=8)
ax.set_ylabel("Test MAPE (%)")
ax.set_ylim(0, max(mapes_b) * 1.15)
ax.set_title("Absolute test MAPE under input perturbation (retrained baseline)")
ax.legend(frameon=False, loc="upper left")
plt.tight_layout()
fB = os.path.join(FIG, "fig_perturbation_abs_mape.png")
plt.savefig(fB, dpi=300); plt.close()

# ===== Figure C: sufficiency (baseline all-10 vs principled top-3) ===========
def load_mape(ckpt):
    return json.load(open(os.path.join(REPO, ckpt, "metrics.json")))["test_mape"]
base_mape = load_mape("checkpoints/local_sanity/baseline_seed42")
top3_mape = load_mape("checkpoints/local_sanity/principled_k30_relevant")

fig, ax = plt.subplots(figsize=(4.2, 3.4))
labels_c = ["Baseline\n(all 10)", "Principled\ntop-3"]
mapes_c  = [base_mape, top3_mape]
bars = ax.bar(labels_c, mapes_c, color=[GREEN, IG_BLUE], width=0.55)
for b, m in zip(bars, mapes_c):
    ax.text(b.get_x()+b.get_width()/2, m + 0.05, f"{m:.2f}%", ha="center", fontsize=10)
ax.set_ylabel("Test MAPE (%)")
ax.set_ylim(0, max(mapes_c) * 1.25)
ax.set_title("Sufficiency: top-3 ≈ all-10\n(retrained, 500 sims, seed 42)")
plt.tight_layout()
fC = os.path.join(FIG, "fig_sufficiency_top3.png")
plt.savefig(fC, dpi=300); plt.close()

print("Saved:")
for f in (fA, fB, fC):
    print("  ", os.path.relpath(f, REPO), os.path.getsize(f), "bytes")
print(f"\nclean={clean:.2f}%  baseline_all10={base_mape:.2f}%  top3={top3_mape:.2f}%")
