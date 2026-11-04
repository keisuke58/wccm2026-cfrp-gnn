"""denso_onepage_panels.py -- panels for the one-page visual summary (slides/build_denso_onepage.js).
  onepage_plate.png  flat plate, measured released strain (Tsuda et al. 2026 Fig. 12, read by eye) vs the physics
                     chain before (T0 = 180 C) and after the identification (T0_eff), Ch.3 (MD) and Ch.1 (TD);
  onepage_part.png   one 3-D part (00010768): data warp, prior, corrected with 10 points (top row of
                     rescue3d_example.png, cropped);
  onepage_bars.png   3-D parts, median relative error: no correction / other parts' factor / 10 points.
Numbers from results/denso_rescue/results.json, abaqus/.../greif3d_rescue/transfer_factor.json.
Run:  python3 slides/figure_sources/denso_onepage_panels.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "validation"))
import denso_rescue_released as rr  # noqa: E402

OUTD = os.path.join(ROOT, "slides", "figures_paper")
ACC = "#1F3A5F"
matplotlib.rcParams.update({"font.family": ["Times New Roman", "Liberation Serif", "IPAPMincho"],
                            "mathtext.fontset": "stix", "font.size": 12, "xtick.direction": "in", "ytick.direction": "in"})
res = json.load(open(os.path.join(ROOT, "results", "denso_rescue", "results.json")))
th = res["R4"]["T0 only"]["theta"]
y0 = rr.released(rr.THETA_PAPER).reshape(3, 10); y1 = rr.released(tuple(th)).reshape(3, 10); ym = rr.Y_MEAS.reshape(3, 10)
d = np.concatenate([[0.0], rr.DEPTH]); z = lambda v: np.concatenate([[0.0], v])

fig, ax = plt.subplots(figsize=(5.0, 3.6))
for k, lab in ((2, "MD"), (0, "TD")):
    ax.plot(rr.DEPTH, ym[k], "s", color="k", ms=5, mfc="white", label="実測" if k == 2 else None)
    ax.plot(d, z(y0[k]), "--", color="0.55", lw=1.6, label="補正前" if k == 2 else None)
    ax.plot(d, z(y1[k]), "-", color=ACC, lw=2.4, label="補正後" if k == 2 else None)
    ax.text(2.02, y1[k][-1], lab, va="center", fontsize=11)
ax.axhline(0, color="0.85", lw=0.8)
ax.set_xlabel("穴の深さ [mm]"); ax.set_ylabel("解放ひずみ [µε]"); ax.set_xlim(0, 2.2)
ax.legend(frameon=False, fontsize=11, loc="center right")
fig.tight_layout(); fig.savefig(os.path.join(OUTD, "onepage_plate.png"), dpi=250); plt.close(fig)

im = Image.open(os.path.join(OUTD, "rescue3d_example.png")); W, H = im.size
im.crop((int(0.15 * W), int(0.075 * H), W, int(0.42 * H))).save(os.path.join(OUTD, "onepage_part.png"))   # titles cut; labels on the slide

tf = json.load(open(os.path.join(ROOT, "abaqus", "denso_plate", "abaqus_results", "greif3d_rescue", "transfer_factor.json")))
vals = [tf["prior (a = 1)"]["median"], tf["LOO factor, no measurement"]["median"], tf["RESCUE 10 nodes, 2 factors"]["median"]]
labs = ["補正なし", "測定なし\n（他の部品の倍率）", "10 点を測って補正"]
fig, ax = plt.subplots(figsize=(5.0, 3.6))
yy = np.arange(3)[::-1]
ax.barh(yy, np.array(vals) * 100, color=["0.75", ACC, ACC], height=0.55)
for y_, v in zip(yy, vals):
    ax.text(v * 100 + 1.5, y_, f"{v * 100:.0f} %", va="center", fontsize=15, fontweight="bold")
ax.set_yticks(yy); ax.set_yticklabels(labs, fontsize=11.5); ax.set_xlim(0, 70)
ax.set_xlabel(f"反りの誤差（中央値、{tf['n']} 形状）[%]")
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUTD, "onepage_bars.png"), dpi=250); plt.close(fig)
print("wrote onepage_plate.png, onepage_part.png, onepage_bars.png")
