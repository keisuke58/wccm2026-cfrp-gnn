#!/usr/bin/env python3
"""Regenerate the WCCM paper-deck field figures LOCALLY with the thesis style.

Uses thesis_style.use() (usetex + lmodern) so figure text matches the thesis font,
with perceptually-uniform colormaps and clean colorbars. Font sizes are bumped above
the thesis 9 pt because these PNGs are displayed large on slides.

Run:  PATH=/home/nishioka/texlive/2025/bin/x86_64-linux:$PATH \
      python3 gen_clean_figs_local.py
Outputs -> ../figures_paper/{diff,noise,ndf,label}_clean.png
Source data -> ./data/  (specimen Defect_L10_B100_el2515_H4_W4 + no-defect reference)
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import thesis_style as ts

ts.use()                                  # usetex + lmodern + 300 dpi
matplotlib.rcParams.update({              # slide-legible sizes (figs shown large)
    "font.size": 15, "axes.titlesize": 16, "axes.labelsize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
})
DATA = os.path.join(HERE, "data")
OUT  = os.path.abspath(os.path.join(HERE, "..", "figures_paper"))

# ── structured-grid reshape (57x125, 2 layers x 6971 nodes, hole masked) ──────
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

def cbar(fig, im, ax, label, ticks=None):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.035, ticks=ticks)
    cb.set_label(label, fontsize=12); cb.ax.tick_params(labelsize=10, width=0.5)
    cb.outline.set_linewidth(0.5); return cb

def fld(ax, g, title, cmap, vmin=None, vmax=None):
    im = ax.imshow(g, cmap=cmap, aspect="equal", interpolation="none", vmin=vmin, vmax=vmax)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_linewidth(0.6)
    ax.set_title(title); return im

spec = "Defect_L10_B100_el2515_H4_W4"
diff   = np.load(f"{DATA}/{spec}_difference.npy")
ref    = np.load(f"{DATA}/real_hole_no_defect_original.npy")
orig   = ref + diff
noised = np.load(f"{DATA}/{spec}_zscore_noise.npy")
zs     = (diff - diff.mean()) / diff.std()
ndf    = np.random.default_rng(0).normal(0.0, 0.1, size=diff.shape)
lab    = np.load(f"{DATA}/{spec}_19label.npy").argmax(1).astype(float)
L = 1   # L10 -> bottom layer

# Fig 1: difference construction
rv = np.nanpercentile(np.r_[ref, orig], [2, 98]); dv = np.nanpercentile(np.abs(diff), 98)
fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.6))
im0 = fld(ax[0], grid(ref, L),  r"(a) No-defect reference", ts.SEQ_CMAP, rv[0], rv[1]); cbar(fig, im0, ax[0], "DSPSS")
im1 = fld(ax[1], grid(orig, L), r"(b) With defect",         ts.SEQ_CMAP, rv[0], rv[1]); cbar(fig, im1, ax[1], "DSPSS")
im2 = fld(ax[2], grid(diff, L), r"(c) Difference (b)$-$(a)", ts.DIV_CMAP, -dv, dv);    cbar(fig, im2, ax[2], r"$\Delta$DSPSS")
fig.tight_layout(); fig.savefig(f"{OUT}/diff_clean.png"); plt.close(fig)

# Fig 2: noise comparison
zv = np.nanpercentile(np.abs(np.r_[zs, noised]), 98)
fig, ax = plt.subplots(1, 2, figsize=(10, 4.6))
im0 = fld(ax[0], grid(zs, L),     r"Z-score input (clean)",  ts.DIV_CMAP, -zv, zv); cbar(fig, im0, ax[0], "Z-score")
im1 = fld(ax[1], grid(noised, L), r"Z-score input + noise",  ts.DIV_CMAP, -zv, zv); cbar(fig, im1, ax[1], "Z-score")
fig.tight_layout(); fig.savefig(f"{OUT}/noise_clean.png"); plt.close(fig)

# Fig 3: NDF
fig, ax = plt.subplots(figsize=(5.8, 5.0))
im = fld(ax, grid(ndf, L), r"Defect-free sample $+$ 10\% noise", ts.DIV_CMAP, -zv, zv); cbar(fig, im, ax, "Z-score")
fig.tight_layout(); fig.savefig(f"{OUT}/ndf_clean.png"); plt.close(fig)

# Fig 4: 19-class ground-truth target
base = plt.cm.tab20(np.linspace(0, 1, 20))[:19]; base[0] = [0.93, 0.93, 0.93, 1]
fig, ax = plt.subplots(figsize=(5.8, 5.0))
im = ax.imshow(grid(lab, L), cmap=ListedColormap(base), aspect="equal", interpolation="none", vmin=0, vmax=18)
ax.set_xticks([]); ax.set_yticks([]); ax.set_title(r"Ground-truth defect class (target)")
for s in ax.spines.values(): s.set_linewidth(0.6)
cbar(fig, im, ax, "class id", ticks=[0, 6, 12, 18])
fig.tight_layout(); fig.savefig(f"{OUT}/label_clean.png"); plt.close(fig)
print("done ->", OUT)

# Fig 5: noise result (noised input -> ground-truth target)
base2 = plt.cm.tab20(np.linspace(0, 1, 20))[:19]; base2[0] = [0.93, 0.93, 0.93, 1]
fig, ax = plt.subplots(1, 2, figsize=(10, 4.6))
imA = fld(ax[0], grid(noised, L), r"Noisy Z-score input", ts.DIV_CMAP, -zv, zv); cbar(fig, imA, ax[0], "Z-score")
imB = ax[1].imshow(grid(lab, L), cmap=ListedColormap(base2), aspect="equal", interpolation="none", vmin=0, vmax=18)
ax[1].set_xticks([]); ax[1].set_yticks([]); ax[1].set_title(r"Localized defect (target)")
for s in ax[1].spines.values(): s.set_linewidth(0.6)
cbar(fig, imB, ax[1], "class id", ticks=[0, 6, 12, 18])
fig.tight_layout(); fig.savefig(f"{OUT}/result_clean.png"); plt.close(fig)
print("result_clean done")
