#!/usr/bin/env python3
"""Regenerate the WCCM paper-deck field figures LOCALLY with the thesis style.

Uses thesis_style.use() (usetex + lmodern). Perceptually-uniform colormaps,
SHARED + SHRUNK colorbars (short, not floor-to-ceiling), compact figure heights.

Run:  PATH=/home/nishioka/texlive/2025/bin/x86_64-linux:$PATH python3 gen_clean_figs_local.py
Outputs -> ../figures_paper/{diff,noise,ndf,ndf2,result,label}_clean.png
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import thesis_style as ts
ts.use()
matplotlib.rcParams.update({
    "font.size": 15, "axes.titlesize": 16, "axes.labelsize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
})
DATA = os.path.join(HERE, "data")
OUT  = os.path.abspath(os.path.join(HERE, "..", "figures_paper"))

num_cols, num_rows, vpl, total = 57, 125, 6971, 57*125
hcs, hrs1, hrs2, hsc, hsr1, hsr2 = 26, 51, 77, 7, 7, 15
hole = set()
for r in range(hrs1, hrs1+hsr1):
    for c in range(hcs, hcs+hsc): hole.add(r*num_cols+c-1)
for r in range(hrs2, hrs2+hsr2):
    for c in range(hcs, hcs+hsc): hole.add(r*num_cols+c-1)

def grid(data, layer=1):
    seg = data[:vpl] if layer == 1 else data[vpl:]
    full = np.full(total, np.nan); k = 0
    for i in range(total):
        if i not in hole and k < vpl:
            full[i] = seg[k]; k += 1
    g = np.flipud(np.rot90(full.reshape((num_rows, num_cols)), k=1))
    return np.ma.masked_invalid(g)

def fld(ax, g, title, cmap, vmin=None, vmax=None):
    im = ax.imshow(g, cmap=cmap, aspect="equal", interpolation="none", vmin=vmin, vmax=vmax)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_linewidth(0.6)
    ax.set_title(title); return im

def cbar(fig, im, axes, label, ticks=None, shrink=0.62):
    """Short, shared colorbar (shrink < 1 keeps it from running full height)."""
    cb = fig.colorbar(im, ax=axes, fraction=0.040, pad=0.03, shrink=shrink,
                      aspect=14, ticks=ticks)
    cb.set_label(label, fontsize=11); cb.ax.tick_params(labelsize=9, width=0.5)
    cb.outline.set_linewidth(0.5); return cb

CLS = plt.cm.tab20(np.linspace(0, 1, 20))[:19]; CLS[0] = [0.93, 0.93, 0.93, 1]
CLS = ListedColormap(CLS)

spec = "Defect_L10_B100_el2515_H4_W4"
diff   = np.load(f"{DATA}/{spec}_difference.npy")
ref    = np.load(f"{DATA}/real_hole_no_defect_original.npy")
orig   = ref + diff
noised = np.load(f"{DATA}/{spec}_zscore_noise.npy")
zs     = (diff - diff.mean()) / diff.std()
ndf    = np.random.default_rng(0).normal(0.0, 0.1, size=diff.shape)
lab    = np.load(f"{DATA}/{spec}_19label.npy").argmax(1).astype(float)
L = 1
zv = np.nanpercentile(np.abs(np.r_[zs, noised]), 98)

# Fig 1: difference construction — (a,b) share one short colorbar; (c) its own
rv = np.nanpercentile(np.r_[ref, orig], [2, 98]); dv = np.nanpercentile(np.abs(diff), 98)
fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.3))
fld(ax[0], grid(ref, L),  r"(a) No-defect reference", ts.SEQ_CMAP, rv[0], rv[1])
im1 = fld(ax[1], grid(orig, L), r"(b) With defect",   ts.SEQ_CMAP, rv[0], rv[1])
cbar(fig, im1, [ax[0], ax[1]], "DSPSS")
im2 = fld(ax[2], grid(diff, L), r"(c) Difference (b)$-$(a)", ts.DIV_CMAP, -dv, dv)
cbar(fig, im2, ax[2], r"$\Delta$DSPSS")
fig.savefig(f"{OUT}/diff_clean.png", bbox_inches="tight"); plt.close(fig)

# Fig 2: noise comparison — both same scale -> ONE shared short colorbar
fig, ax = plt.subplots(1, 2, figsize=(10, 3.3))
fld(ax[0], grid(zs, L),     r"Z-score input (clean)", ts.DIV_CMAP, -zv, zv)
im = fld(ax[1], grid(noised, L), r"Z-score input + noise", ts.DIV_CMAP, -zv, zv)
cbar(fig, im, [ax[0], ax[1]], "Z-score")
fig.savefig(f"{OUT}/noise_clean.png", bbox_inches="tight"); plt.close(fig)

# Fig 3 (NEW): NDF vs noisy defect — two images, shared scale
fig, ax = plt.subplots(1, 2, figsize=(10, 3.3))
fld(ax[0], grid(ndf, L),    r"Defect-free $+$ 10\% noise", ts.DIV_CMAP, -zv, zv)
im = fld(ax[1], grid(noised, L), r"Defect $+$ structured noise", ts.DIV_CMAP, -zv, zv)
cbar(fig, im, [ax[0], ax[1]], "Z-score")
fig.savefig(f"{OUT}/ndf2_clean.png", bbox_inches="tight"); plt.close(fig)

# (single NDF kept for compatibility)
fig, ax = plt.subplots(figsize=(5.2, 2.9))
im = fld(ax, grid(ndf, L), r"Defect-free sample $+$ 10\% noise", ts.DIV_CMAP, -zv, zv)
cbar(fig, im, ax, "Z-score", shrink=0.85)
fig.savefig(f"{OUT}/ndf_clean.png", bbox_inches="tight"); plt.close(fig)

# Fig 4: noise result — noisy input -> localized target (separate short colorbars)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.3))
imA = fld(ax[0], grid(noised, L), r"Noisy Z-score input", ts.DIV_CMAP, -zv, zv)
cbar(fig, imA, ax[0], "Z-score")
imB = fld(ax[1], grid(lab, L), r"Localized defect (target)", CLS, 0, 18)
cbar(fig, imB, ax[1], "class id", ticks=[0, 6, 12, 18])
fig.savefig(f"{OUT}/result_clean.png", bbox_inches="tight"); plt.close(fig)

# Fig 5: 19-class target (single, kept)
fig, ax = plt.subplots(figsize=(5.2, 2.9))
im = fld(ax, grid(lab, L), r"Ground-truth defect class (target)", CLS, 0, 18)
cbar(fig, im, ax, "class id", ticks=[0, 6, 12, 18], shrink=0.85)
fig.savefig(f"{OUT}/label_clean.png", bbox_inches="tight"); plt.close(fig)

# Fig 6: prior-work schematic — CNN image grid (Kojima et al.) vs our GNN mesh
import matplotlib.patches as mpatches
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.7))
# (left) CNN image grid with a hole that breaks the regular sampling
ng = 8
holecells = {(3, 3), (3, 4), (4, 3), (4, 4)}
for i in range(ng):
    for j in range(ng):
        if (i, j) in holecells:
            ax[0].add_patch(mpatches.Rectangle((j, i), 1, 1, facecolor="white",
                            edgecolor="#a01c1c", lw=1.3, ls="--", hatch="//"))
        else:
            ax[0].add_patch(mpatches.Rectangle((j, i), 1, 1, facecolor="#d7e3f4",
                            edgecolor="#6b7280", lw=0.6))
ax[0].set_xlim(0, ng); ax[0].set_ylim(0, ng); ax[0].set_aspect("equal")
ax[0].invert_yaxis(); ax[0].set_xticks([]); ax[0].set_yticks([])
ax[0].set_title(r"Previous: CNN on an image grid")
ax[0].text(ng/2, ng+0.55, r"holes break the regular grid", ha="center", va="top",
           fontsize=11, color="#a01c1c")
# (right) GNN on the FEM mesh: nodes + edges, hole left empty
rng2 = np.random.default_rng(3)
xs, ys = [], []
for i in range(7):
    for j in range(7):
        if (2 <= i <= 3) and (2 <= j <= 3):   # hole region: no nodes
            continue
        xs.append(j + 0.12*rng2.standard_normal()); ys.append(i + 0.12*rng2.standard_normal())
xs, ys = np.array(xs), np.array(ys)
P = np.c_[xs, ys]
for a in range(len(P)):                        # connect near neighbours -> mesh edges
    d = np.hypot(*(P - P[a]).T)
    for b in np.argsort(d)[1:4]:
        if d[b] < 1.5:
            ax[1].plot([P[a, 0], P[b, 0]], [P[a, 1], P[b, 1]], color="#9aa3b2", lw=0.6, zorder=1)
ax[1].scatter(xs, ys, s=46, c="#142b54", zorder=2, edgecolors="white", linewidths=0.5)
ax[1].set_aspect("equal"); ax[1].invert_yaxis(); ax[1].set_xticks([]); ax[1].set_yticks([])
for s in ax[1].spines.values(): s.set_visible(False)
ax[1].set_title(r"This work: GNN on the FEM mesh")
ax[1].text(3, 7.2, r"nodes $=$ elements, edges $=$ connectivity", ha="center", va="top", fontsize=11, color="#142b54")
fig.savefig(f"{OUT}/prior_cnn_gnn.png", bbox_inches="tight"); plt.close(fig)

# Fig 7: focal-loss curves (gamma = 0 is cross-entropy)
fig, ax = plt.subplots(figsize=(5.6, 3.6))
p = np.linspace(0.012, 1.0, 300)
for g, ls in zip([0, 1, 2, 3], ["-", "--", "-.", ":"]):
    ax.plot(p, -(1 - p)**g * np.log(p), ls, lw=1.8,
            label=(r"$\gamma=0$ (cross-entropy)" if g == 0 else rf"$\gamma={g}$"))
ax.set_xlabel(r"$p_t$ (prob.\ of true class)"); ax.set_ylabel(r"loss")
ax.set_xlim(0, 1); ax.set_ylim(0, 5); ax.legend(frameon=False, fontsize=10)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.set_title(r"Focal loss down-weights easy (high $p_t$) nodes")
fig.savefig(f"{OUT}/focal_curve.png", bbox_inches="tight"); plt.close(fig)
print("done ->", OUT)
