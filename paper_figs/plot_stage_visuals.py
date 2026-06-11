"""Two slide visuals (thesis style):
1. stage0_visual.pdf — measured DSPSS field | Mahalanobis anomaly map | truth
   (node scatter, no grid artifacts) on one held-out 4x4 sample.
2. fmpe_posterior_visual.pdf — FMPE posterior from the e2e run: (cx,cy) sample
   cloud + truth, and layer/size marginals with truth lines.
"""
import sys, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "slides", "figure_sources"))
from thesis_style import use
import matplotlib.pyplot as plt

BASE  = "/home/nishioka/CFRP/CFRP_hole/hole_data_inp"
D4    = f"{BASE}/Defect_hole_4x4_Region1_21_npy"
L4    = f"{BASE}/Def4x4_19class_label"
D1    = f"{BASE}/Defect_hole_1x1_Random_npy"
CD    = f"{BASE}/basicdata_for_holegnn"
N     = 13942
rng   = np.random.default_rng(42)

x_c = np.load(f"{CD}/normalized_x_2layer.npy")[:N]
y_c = np.load(f"{CD}/normalized_y_2layer.npy")[:N]

# The nodes trace a thin curved band (the curved interstage section), so plot
# in principal-axis coordinates: rotate so the band lies horizontally.
P      = np.stack([x_c, y_c], 1)
Pc     = P - P.mean(0)
_, _, Vt = np.linalg.svd(Pc, full_matrices=False)
R      = Vt.T
def rot(xy):
    return (xy - P.mean(0)) @ R
xr, yr = rot(P).T

def zfield(p):
    v = np.load(p).astype(np.float32)[:N]
    return (v - v.mean()) / (v.std() + 1e-8)

# ── figure 1: stage-0 detection ──────────────────────────────────────────────
files1 = sorted(glob.glob(f"{D1}/*.npy"))
refs   = np.stack([zfield(files1[i]) for i in rng.choice(len(files1), 800, replace=False)])
mu, sd = refs.mean(0), refs.std(0) + 1e-6

f4   = sorted(glob.glob(f"{D4}/*.npy"))[3]
v    = zfield(f4)
score = np.abs(v - mu) / sd
base  = os.path.splitext(os.path.basename(f4))[0]
lp    = os.path.join(L4, f"{base}_19label.npy")
if not os.path.exists(lp):
    lp = os.path.join(L4, f"{base}.npy")
lbl   = np.load(lp)
mask  = lbl[:, 1:].sum(1) > 0

fig, axes = plt.subplots(3, 1, figsize=use(width_frac=0.92, aspect=0.52))
s0 = axes[0].scatter(xr, yr, c=v, s=2.2, cmap="RdBu_r", vmin=-3, vmax=3,
                     rasterized=True)
axes[0].set_title(r"(a) measured field (z-scored DSPSS)", fontsize=8, pad=2)
fig.colorbar(s0, ax=axes[0], fraction=0.025, pad=0.01)
s1 = axes[1].scatter(xr, yr, c=score, s=2.2, cmap="inferno", vmin=0, vmax=8,
                     rasterized=True)
axes[1].set_title(r"(b) anomaly score $|x_i-\mu_i|/\sigma_i$", fontsize=8, pad=2)
fig.colorbar(s1, ax=axes[1], fraction=0.025, pad=0.01)
axes[2].scatter(xr, yr, c="0.88", s=2.2, rasterized=True)
axes[2].scatter(xr[mask], yr[mask], c="#d7301f", s=9)
axes[2].set_title(r"(c) ground truth (%d defect nodes / 13\,942)" % mask.sum(),
                  fontsize=8, pad=2)
fig.colorbar(s1, ax=axes[2], fraction=0.025, pad=0.01).ax.set_visible(False)
for a in axes:
    a.set_xticks([]); a.set_yticks([]); a.set_aspect("equal")
plt.tight_layout(h_pad=0.4)
plt.savefig("paper_figs/stage0_visual.pdf", bbox_inches="tight", dpi=200)
plt.savefig("paper_figs/stage0_visual.png", bbox_inches="tight", dpi=200)
print("saved stage0_visual")

# ── figure 2: FMPE posterior ─────────────────────────────────────────────────
d    = np.load("results/fmpe_posterior_e2e.npz")
post, truth = d["posterior"], d["truth"]          # (256,4): cx,cy,layer,log2size

fig = plt.figure(figsize=use(width_frac=1.0, aspect=0.32))
gs  = fig.add_gridspec(2, 3, width_ratios=[1.5, 1, 1])
ax0 = fig.add_subplot(gs[:, 0])
ax0.scatter(xr, yr, c=v, s=0.8, cmap="RdBu_r", vmin=-3, vmax=3, alpha=0.35,
            rasterized=True)
pr = rot(post[:, :2]); ax0.scatter(pr[:, 0], pr[:, 1], s=7, c="#1565C0", alpha=0.45,
            label=r"posterior samples")
tr = rot(truth[None, :2]); ax0.scatter([tr[0,0]], [tr[0,1]], marker="*", s=160, c="#d7301f",
            edgecolor="k", linewidth=0.5, zorder=5, label="true defect")
ax0.set_aspect("equal")
ax0.set_xticks([]); ax0.set_yticks([])
ax0.legend(loc="lower left", fontsize=7, markerscale=0.9)
ax0.set_title(r"(a) position posterior $p(c_x,c_y\mid x)$", fontsize=8)

ax1 = fig.add_subplot(gs[0, 1:])
ax1.hist(post[:, 2], bins=18, color="#1565C0", alpha=0.75)
ax1.axvline(truth[2], color="#d7301f", lw=1.6)
ax1.set_title(r"(b) layer (depth) marginal", fontsize=8)
ax1.set_yticks([])

ax2 = fig.add_subplot(gs[1, 1:])
ax2.hist(2 ** post[:, 3], bins=18, color="#1565C0", alpha=0.75)
ax2.axvline(2 ** truth[3], color="#d7301f", lw=1.6)
ax2.set_title(r"(c) defect-size marginal (elements)", fontsize=8)
ax2.set_yticks([])

plt.tight_layout()
plt.savefig("paper_figs/fmpe_posterior_visual.pdf", bbox_inches="tight", dpi=200)
plt.savefig("paper_figs/fmpe_posterior_visual.png", bbox_inches="tight", dpi=200)
print("saved fmpe_posterior_visual")
