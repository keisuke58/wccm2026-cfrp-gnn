"""denso_overview_flow.py -- the whole study on one picture, with as little text as possible:
the prediction chain (orientation -> shrinkage -> FE -> few measurements -> correction -> prediction -> design)
as thumbnails made from the repository's results, and the three stages with their headline numbers.
Every thumbnail is data from this repository (sources below); numbers from the same result files.
  orientation: Tsuda et al. 2026 Fig. 8a (read by eye); eigenstrain: validation/denso_rescue_released.py;
  parts: rescue3d_gallery.png / rescue3d_example.png; hole drilling: onepage_plate.png;
  gate ranking: abaqus/.../greif3d_rescue/gate_ranking.json.
Run:  python3 slides/figure_sources/denso_overview_flow.py   (writes slides/figures_paper/overview_flow.png)
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "validation"))
import denso_rescue_released as rr  # noqa: E402

FIG = os.path.join(ROOT, "slides", "figures_paper")
G3 = os.path.join(ROOT, "abaqus", "denso_plate", "abaqus_results", "greif3d_rescue")
ACC = "#1F3A5F"
matplotlib.rcParams.update({"font.family": ["Times New Roman", "Liberation Serif", "IPAPMincho"], "font.size": 9,
                            "xtick.direction": "in", "ytick.direction": "in"})

fig = plt.figure(figsize=(13.0, 4.6))
bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, 130); bg.set_ylim(0, 46); bg.axis("off")
N = 6; W = 18.0; X0 = 2.0; GAP = (130 - 2 * X0 - N * W) / (N - 1); Y0, H = 24.0, 19.0
xs = [X0 + i * (W + GAP) for i in range(N)]
labels = ["繊維の向き", "縮み方（物理）", "部品の FE", "数点を測る", "補正して予測", "成形条件の設計"]


def frame(i, accent=False):
    bg.add_patch(FancyBboxPatch((xs[i], Y0), W, H, boxstyle="round,pad=0.2,rounding_size=1.0",
                                ec=ACC if accent else "0.6", fc="white", lw=1.4 if accent else 1.0))
    bg.text(xs[i] + W / 2, Y0 - 2.2, labels[i], ha="center", va="top", fontsize=11, fontweight="bold",
            color=ACC if accent else "k")
    if i < N - 1:
        bg.add_patch(FancyArrowPatch((xs[i] + W + 0.5, Y0 + H / 2), (xs[i + 1] - 0.5, Y0 + H / 2),
                                     arrowstyle="-|>", mutation_scale=13, color="0.3", lw=1.2))


def inset(i, pad=1.2):
    return fig.add_axes([(xs[i] + pad) / 130, (Y0 + pad) / 46, (W - 2 * pad) / 130, (H - 2 * pad) / 46])


def picture(i, path, box=None):
    a = inset(i, 0.8); im = Image.open(path)
    if box:
        Wp, Hp = im.size; im = im.crop((int(box[0] * Wp), int(box[1] * Hp), int(box[2] * Wp), int(box[3] * Hp)))
    a.imshow(im); a.axis("off")


for i in range(N):
    frame(i, accent=(i in (3, 4)))
# 1 orientation (paper Fig. 8a)
a = inset(0, 2.2); p = rr.tc.PROF_3MM
a.plot(p["axx"], p["d"], color=ACC, lw=1.8); a.plot(p["ayy"], p["d"], color="0.55", lw=1.4, ls="--")
a.invert_yaxis(); a.set_xticks([]); a.set_yticks([]); a.text(0.97, 0.04, "MD", transform=a.transAxes, ha="right", fontsize=8, color=ACC)
a.text(0.03, 0.04, "TD", transform=a.transAxes, fontsize=8, color="0.4")
# 2 eigenstrain profile
zq, eth = rr.eigenstrain(rr.THETA_PAPER)
a = inset(1, 2.2)
a.plot(eth[:, 0] * 1e3, zq, color=ACC, lw=1.8); a.plot(eth[:, 1] * 1e3, zq, color="0.55", lw=1.4, ls="--")
a.invert_yaxis(); a.set_xticks([]); a.set_yticks([])
# 3 parts (gallery, three of them)
picture(2, os.path.join(FIG, "rescue3d_gallery.png"), (0.0, 0.18, 0.5, 0.92))
# 4 hole drilling
picture(3, os.path.join(FIG, "onepage_plate.png"))
# 5 corrected part
picture(4, os.path.join(FIG, "onepage_part.png"), (0.68, 0.0, 1.0, 1.0))
# 6 gate ranking
a = inset(5, 2.6)
gr = json.load(open(os.path.join(G3, "gate_ranking.json")))
for prt in gr["parts"]:
    d_, p_ = np.array(prt["max_data_mm"]), np.array(prt["max_prior_mm"]) * 1.93
    a.plot(d_, p_, "-", color="0.8", lw=0.8)
    a.plot(d_, p_, "o", ms=3.5, color=ACC if prt["best_agrees"] else "0.5", mfc="white")
m = max(max(x["max_data_mm"]) for x in gr["parts"])
a.plot([0, m], [0, m], "k:", lw=0.8); a.set_xscale("log"); a.set_yscale("log"); a.set_xticks([]); a.set_yticks([])
a.minorticks_off()

# stages
sy, sh = 1.5, 14.0
stages = [(0, 3, "① 残留ひずみ", "平板・実測", "誤差 58 → 16 µε", "済み"),
          (2, 4, "② 反り", "3 次元部品・解析データ", "誤差 53 → 17 %（10 点）", "済み（途中経過）"),
          (5, 5, "③ 設計", "ゲート位置", f"最良のゲート {gr['best_agrees']}/{gr['n_parts']} 一致", "着手")]
SW = (130 - 2 * X0 - 2 * 3.0) / 3
for k, (i0, i1, t, sub, num, st) in enumerate(stages):
    x0 = X0 + k * (SW + 3.0); x1 = x0 + SW
    bg.add_patch(FancyBboxPatch((x0, sy), x1 - x0, sh, boxstyle="round,pad=0.2,rounding_size=1.0", ec="0.75", fc="#F5F6F8", lw=0.8))
    bg.text(x0 + 1.5, sy + sh - 2.0, t, fontsize=12, fontweight="bold", va="top")
    bg.text(x0 + 1.5, sy + sh - 6.3, sub, fontsize=9.5, va="top", color="0.35")
    bg.text(x0 + 1.5, sy + 2.0, num, fontsize=12, fontweight="bold", color=ACC, va="bottom")
    bg.text(x1 - 1.5, sy + sh - 2.0, st, fontsize=9, va="top", ha="right", color="0.35")
fig.savefig(os.path.join(FIG, "overview_flow.png"), dpi=250); plt.close(fig)
print("wrote overview_flow.png")
