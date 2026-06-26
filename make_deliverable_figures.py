"""
make_deliverable_figures.py — generate the 3 figures for the XAI deliverable.
Outputs PNGs (300 dpi) to results/figures/ for inclusion in LaTeX.
"""
import os, csv, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Compact + legible-when-small (figures are shown at ~0.5 textwidth in LaTeX)
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
})

REPO = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(REPO, "results", "figures")
os.makedirs(FIG, exist_ok=True)

IG_BLUE, KS_RED = "#2c6fbb", "#c0392b"

# ---- load rankings ----------------------------------------------------------
def load_rank(path):
    d = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            d[r["feature"]] = float(r["mean_abs_score"])
    return d

ig = load_rank(os.path.join(REPO, "rankings", "ig.csv"))
ks = load_rank(os.path.join(REPO, "rankings", "kernel_shap.csv"))
order = ["sigma","traffic","packets","pkts_lambda_on","eq_lambda",
         "avg_t_off","ar_a","exp_max_factor","avg_t_on","avg_pkts_lambda"]
ig_v = [ig[f] for f in order]
ks_v = [ks[f] for f in order]

# ===== Figure 1: attribution ranking + cliff =================================
fig, ax = plt.subplots(figsize=(6.6, 3.2))
x = np.arange(len(order)); w = 0.4
ax.bar(x - w/2, ig_v, w, label="Integrated Gradients", color=IG_BLUE)
ax.bar(x + w/2, ks_v, w, label="KernelSHAP", color=KS_RED)
ax.set_xticks(x); ax.set_xticklabels(order, rotation=40, ha="right", fontsize=9)
ax.set_ylabel("Mean |attribution|")
ax.set_title("Global feature importance — IG vs KernelSHAP (both agree; top-3 dominate)")
ax.legend(frameon=False)
# annotate the cliff
cliff = (ig_v[2]) / (ig_v[3])
ax.annotate(f"~{cliff:.0f}x cliff",
            xy=(2.5, ig_v[2]*0.5), xytext=(4.3, ig_v[2]*0.7),
            arrowprops=dict(arrowstyle="->", color="gray"), color="gray", fontsize=10)
ax.axvline(2.5, ls="--", lw=1, color="gray", alpha=0.6)
plt.tight_layout()
f1 = os.path.join(FIG, "fig_attribution_cliff.png")
plt.savefig(f1, dpi=300); plt.close()

# ===== Figure 2: faithfulness (test MAPE of the 5 retrained models) ==========
models = ["Principled\nrelevant\n(top-3)", "Random\nrelevant\n(3 rand.)",
          "Random\nirrelevant\n(7 rand.)", "Baseline\n(all 10)",
          "Principled\nirrelevant\n(bottom-7)"]
mape = [5.41, 5.43, 5.80, 6.14, 6.38]
colors = [IG_BLUE, "#7f8c8d", "#bdc3c7", "#27ae60", KS_RED]
fig, ax = plt.subplots(figsize=(6.4, 3.4))
bars = ax.bar(models, mape, color=colors)
for b, m in zip(bars, mape):
    ax.text(b.get_x()+b.get_width()/2, m+0.05, f"{m:.2f}", ha="center", fontsize=9)
ax.axhline(6.14, ls="--", lw=1, color="#27ae60", alpha=0.7, label="baseline (all 10)")
ax.set_ylabel("Test MAPE (%)  — lower is better")
ax.set_ylim(0, 7.5)
ax.set_title("Retraining faithfulness: test error barely moves (≈ 1 pp spread)\n"
             "principled top-3 ≈ random-3 → the ranking is not separable here")
ax.legend(frameon=False, loc="lower right")
plt.tight_layout()
f2 = os.path.join(FIG, "fig_faithfulness_mape.png")
plt.savefig(f2, dpi=300); plt.close()

# ===== Figure 3: radar (IG vs KernelSHAP on 4 criteria) ======================
axes_labels = ["Faithfulness", "Stability", "Cost-efficiency", "Plausibility"]
ig_s  = [3, 4, 5, 5]
ks_s  = [3, 5, 2, 5]
N = len(axes_labels)
ang = np.linspace(0, 2*np.pi, N, endpoint=False).tolist(); ang += ang[:1]
def close(v): return v + v[:1]
fig = plt.figure(figsize=(4.6, 4.6))
ax = plt.subplot(111, polar=True)
ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1)
ax.set_xticks(ang[:-1]); ax.set_xticklabels(axes_labels, fontsize=10)
ax.set_ylim(0, 5); ax.set_yticks([1,2,3,4,5]); ax.set_yticklabels(["1","2","3","4","5"], fontsize=8)
ax.plot(ang, close(ig_s), color=IG_BLUE, lw=2, label="Integrated Gradients")
ax.fill(ang, close(ig_s), color=IG_BLUE, alpha=0.15)
ax.plot(ang, close(ks_s), color=KS_RED, lw=2, label="KernelSHAP")
ax.fill(ang, close(ks_s), color=KS_RED, alpha=0.15)
ax.set_title("XAI method comparison (1=weak … 5=strong)", fontsize=11, pad=18)
ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12), frameon=False, fontsize=9)
plt.tight_layout()
f3 = os.path.join(FIG, "fig_radar_ig_vs_kshap.png")
plt.savefig(f3, dpi=300, bbox_inches="tight"); plt.close()

print("Saved:")
for f in (f1, f2, f3):
    print("  ", os.path.relpath(f, REPO), os.path.getsize(f), "bytes")
