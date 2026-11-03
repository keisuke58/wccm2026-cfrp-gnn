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
ACC = "#1F3A5F"                       # done
PART, TODO = "#2E7D6B", "#C77700"      # in progress / not yet (status colours)
TINT = {ACC: "#EEF2F7", PART: "#ECF5F2", TODO: "#FBF3E6"}
matplotlib.rcParams.update({"font.family": ["Times New Roman", "Liberation Serif", "IPAPMincho"], "font.size": 9,
                            "xtick.direction": "in", "ytick.direction": "in"})

YT = 60
fig = plt.figure(figsize=(13.0, 6.0))
bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, 130); bg.set_ylim(0, YT); bg.axis("off")
N = 6; W = 17.4; X0 = 0.6; LEAD = 9.6; GAP = (130 - 2 * X0 - LEAD - N * W) / N; Y0, H = 40.6, 19.0
xs = [X0 + LEAD + GAP + i * (W + GAP) for i in range(N)]
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
    return fig.add_axes([(xs[i] + pad) / 130, (Y0 + pad) / YT, (W - 2 * pad) / 130, (H - 2 * pad) / YT])


def picture(i, path, box=None):
    a = inset(i, 0.8); im = Image.open(path)
    if box:
        Wp, Hp = im.size; im = im.crop((int(box[0] * Wp), int(box[1] * Hp), int(box[2] * Wp), int(box[3] * Hp)))
    a.imshow(im); a.axis("off")


for i in range(N):
    frame(i, accent=(i in (3, 4)))
# lead: moulding conditions -> orientation (flow analysis not working yet)
bg.add_patch(FancyBboxPatch((X0, Y0 + 5), LEAD, H - 10, boxstyle="round,pad=0.2,rounding_size=1.0", ec=TODO, fc=TINT[TODO], lw=1.2, ls="--"))
bg.text(X0 + LEAD / 2, Y0 + H / 2, "成形条件", ha="center", va="center", fontsize=10.5, fontweight="bold", color=TODO)
bg.add_patch(FancyArrowPatch((X0 + LEAD + 0.4, Y0 + H / 2), (xs[0] - 0.4, Y0 + H / 2), arrowstyle="-|>", mutation_scale=12,
                             color=TODO, lw=1.2, ls="--"))
bg.text(X0 + LEAD / 2, Y0 + 3.6, "流動解析は未", ha="center", va="top", fontsize=8.5, color=TODO, fontweight="bold")
bg.text(xs[2] + W / 2, Y0 - 4.5, "＋代わりのモデル（600 枚で事前学習）", ha="center", va="top", fontsize=8, color="0.35")
bg.text(xs[0] + W / 2, Y0 - 4.5, "既報・Moldflow から", ha="center", va="top", fontsize=8, color="0.35")
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
sy, sh = 0.3, 33.6
stages = [(0, 3, "① 残留ひずみ", "平板・実測", "誤差 58 → 16 µε", "済み", ACC),
          (2, 4, "② 反り", "3 次元部品・解析", "誤差 53 → 17 %（10 点）", "途中経過・強度は未", PART),
          (5, 5, "③ 設計", "ゲート位置", f"最良のゲート {gr['best_agrees']}/{gr['n_parts']} 一致", "着手", TODO)]
SW = (130 - 2 * X0 - 2 * 1.4) / 3
axes_res = []
for k, (i0, i1, t, sub, num, st, col) in enumerate(stages):
    x0 = X0 + k * (SW + 1.4); x1 = x0 + SW
    bg.add_patch(FancyBboxPatch((x0, sy), x1 - x0, sh, boxstyle="round,pad=0.2,rounding_size=1.0", ec=col, fc=TINT[col], lw=1.2))
    bg.text(x0 + 1.5, sy + sh - 2.0, f"{t}　{sub}", fontsize=12, fontweight="bold", va="top", color=col)
    bg.text(x1 - 1.5, sy + sh - 2.0, st, fontsize=9.5, va="top", ha="right", color=col, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=col, lw=0.8))
    bg.text(x0 + 1.5, sy + 1.5, num, fontsize=12.5, fontweight="bold", color=col, va="bottom")
    axes_res.append(fig.add_axes([(x0 + 6.0) / 130, (sy + 7.6) / YT, (SW - 7.6) / 130, (sh - 12.8) / YT]))
# results inside the stage boxes
res = json.load(open(os.path.join(ROOT, "results", "denso_rescue", "results.json")))["R4"]["T0 only"]
y0 = rr.released(rr.THETA_PAPER).reshape(3, 10); y1 = rr.released(tuple(res["theta"])).reshape(3, 10)
ym = rr.Y_MEAS.reshape(3, 10); dd = np.concatenate([[0.0], rr.DEPTH]); zz = lambda v: np.concatenate([[0.0], v])
a = axes_res[0]
for k in (2, 0):
    a.plot(rr.DEPTH, ym[k], "s", color="k", ms=3.5, mfc="white", label="実測" if k == 2 else None)
    a.plot(dd, zz(y0[k]), "--", color="0.55", lw=1.2, label="補正前" if k == 2 else None)
    a.plot(dd, zz(y1[k]), "-", color=ACC, lw=2.0, label="補正後" if k == 2 else None)
a.axhline(0, color="0.85", lw=0.6); a.set_xlabel("穴の深さ [mm]", fontsize=8.5); a.set_ylabel("解放ひずみ [µε]", fontsize=8.5)
a.tick_params(labelsize=7.5); a.legend(frameon=False, fontsize=7.5, loc="center right")
tf = json.load(open(os.path.join(G3, "transfer_factor.json")))
a = axes_res[1]
vals = [tf["prior (a = 1)"]["median"], tf["LOO factor, no measurement"]["median"], tf["RESCUE 10 nodes, 2 factors"]["median"]]
yy = np.arange(3)[::-1]
a.barh(yy, np.array(vals) * 100, color=["0.75", PART, PART], height=0.55)
for y_, v in zip(yy, vals):
    a.text(v * 100 + 1.5, y_, f"{v * 100:.0f} %", va="center", fontsize=10, fontweight="bold")
a.set_yticks(yy); a.set_yticklabels(["補正なし", "測定なし\n（他部品の倍率）", "10 点で補正"], fontsize=8)
a.set_xlim(0, 68); a.set_xlabel(f"反りの誤差（中央値、{tf['n']} 形状）[%]", fontsize=8.5); a.tick_params(labelsize=7.5)
for s_ in ("top", "right"):
    a.spines[s_].set_visible(False)
a = axes_res[2]
for prt in gr["parts"]:
    d_, p_ = np.array(prt["max_data_mm"]), np.array(prt["max_prior_mm"]) * tf["a_star_median"]
    a.plot(d_, p_, "-", color="0.8", lw=0.8)
    a.plot(d_[np.argmin(d_)], p_[np.argmin(d_)], "*", ms=9, color=TODO if prt["best_agrees"] else "0.5")
    a.plot(d_, p_, "o", ms=3.5, color="0.35", mfc="white")
a.plot([0.1, m * 1.2], [0.1, m * 1.2], "k:", lw=0.8); a.set_xscale("log"); a.set_yscale("log")
a.set_xlabel("データの反り [mm]", fontsize=8.5); a.set_ylabel("予測の反り [mm]", fontsize=8.5); a.tick_params(labelsize=7.5)
a.text(0.04, 0.95, "線：同じ部品のゲート違い\n★：反り最小のゲート", transform=a.transAxes, va="top", fontsize=7.5, color="0.3")
fig.savefig(os.path.join(FIG, "overview_flow.png"), dpi=250); plt.close(fig)
print("wrote overview_flow.png")
