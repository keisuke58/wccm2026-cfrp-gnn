"""denso_rescue3d_figs.py -- figures for the 3-D RESCUE slides (Greif & Meyer geometries, fifa §19).
  rescue3d_flow.png     schematic of the method (no data);
  rescue3d_errors.png   (a) relative warp error per geometry for each correction; (b) best factor per geometry vs
                        part size; (c) shape agreement (cos) vs error after the 10-node correction.
Data: abaqus/denso_plate/abaqus_results/greif3d_rescue/per_geometry.json (Moldflow results, not measurements;
the material constants of the prior are general PP/GF30 values, example). The no-measurement case is
validation/greif3d_transfer_factor.py (leave-one-out factor, exact from the stored values).
Run:  python3 slides/figure_sources/denso_rescue3d_figs.py
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
D = os.path.join(ROOT, "abaqus", "denso_plate", "abaqus_results")
OUTD = os.path.join(ROOT, "slides", "figures_paper")
ACC = "#1F3A5F"
matplotlib.rcParams.update({"font.family": ["Times New Roman", "Liberation Serif", "IPAPMincho"],
                            "mathtext.fontset": "stix", "font.size": 10, "xtick.direction": "in", "ytick.direction": "in"})

R = json.load(open(os.path.join(D, "greif3d_rescue", "per_geometry.json")))
size = {r["id"]: float(r["diag_mm"]) for r in csv.DictReader(open(os.path.join(D, "greif3d", "greif_survey.csv")))}
c = np.array([r["cos_prior_data"] for r in R]); e0 = np.array([r["prior"] for r in R])
afe = np.array([r["field_1f_a"][0] for r in R])
sq = np.sqrt(np.maximum(c ** 2 - 1 + e0 ** 2, 0.0))
rr = np.where(np.abs(c / (c - sq) - afe) < np.abs(c / (c + sq) - afe), c - sq, c + sq)   # as greif3d_transfer_factor.py
astar = c / rr
loo = np.array([np.median(np.delete(astar, i)) for i in range(len(R))])
eL = np.sqrt(np.maximum(loo ** 2 * rr ** 2 - 2 * loo * rr * c + 1, 0.0))
N = len(R)

# ------------------------------------------------------------------ flow schematic
fig, ax = plt.subplots(figsize=(12.0, 2.6)); ax.set_xlim(0, 120); ax.set_ylim(0, 25); ax.axis("off")


def box(x, y, w, h, title, lines, ec="k"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0", ec=ec, fc="white", lw=1.2))
    ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=11, fontweight="bold", color=ec)
    ax.text(x + w / 2, y + h - 6.0, "\n".join(lines), ha="center", va="top", fontsize=8.8, linespacing=1.5)


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, color="k", lw=1.2))


W, H, Y = 21, 15, 8
xs = [1, 25, 49, 73, 97]
box(xs[0], Y, W, H, "① 繊維の向き", ["データ（Moldflow）の", "各節点の配向テンソル", "196 形状のうち 46"])
box(xs[1], Y, W, H, "② 縮み方", ["配向平均で", "剛性と熱膨張係数", "（材料定数は一般値）"])
box(xs[2], Y, W, H, "③ 反りの予測", ["3 次元 FE（CalculiX）", "等方・異方の 2 成分", "＝補正前の予測"])
box(xs[3], Y, W, H, "④ 数点で補正", ["表面の 5〜20 点の反り", "倍率 2 個＋剛体運動", "（L2-SP）"], ec=ACC)
box(xs[4], Y, W, H, "⑤ 全点で評価", ["残りの全節点で", "データの反りと比べる", "相対誤差"])
for i in range(4):
    arrow(xs[i] + W + 0.4, Y + H / 2, xs[i + 1] - 0.4, Y + H / 2)
ax.text(60, 3.2, "平板（既報の試験片 1 枚）で行った「物理モデル＋少数の計測で補正」を、3 次元の部品で同じ手順で行う。"
        "データの反りは Moldflow の計算結果であり、実測ではない。", ha="center", fontsize=8.5, color="0.35")
fig.savefig(os.path.join(OUTD, "rescue3d_flow.png"), dpi=250, bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------ results
fig = plt.figure(figsize=(12.0, 3.9))
a = fig.add_axes([0.06, 0.17, 0.36, 0.68])
cols = [("補正なし", e0), ("測定なし\n（他の部品の倍率）", eL), ("5 点\n倍率 1 個", np.array([r["rescue_5_1f"] for r in R])),
        ("10 点\n倍率 2 個", np.array([r["rescue_10_2f"] for r in R])), ("全点\n倍率 2 個", np.array([r["field_2f"] for r in R]))]
rng = np.random.default_rng(0)
for i, (lab, v) in enumerate(cols):
    a.plot(i + rng.uniform(-0.18, 0.18, len(v)), v * 100, "o", ms=3.2, color="0.6" if i == 0 else ACC, alpha=0.55, mec="none")
    a.plot([i - 0.3, i + 0.3], [np.median(v) * 100] * 2, "k-", lw=2.0)
    a.text(i, np.median(v) * 100 + 3, f"{np.median(v) * 100:.0f} %", ha="center", va="bottom", fontsize=9, fontweight="bold",
           bbox=dict(fc="white", ec="none", pad=0.5))
a.set_xticks(range(len(cols))); a.set_xticklabels([c_[0] for c_ in cols], fontsize=8)
a.set_ylabel("反りの相対誤差 [%]"); a.set_ylim(0, 90); a.set_xlim(-0.6, len(cols) - 0.4)
a.set_title(f"(a) 補正のしかたと誤差（{N} 形状、横線は中央値）", fontsize=10.5, fontweight="bold")
b = fig.add_axes([0.50, 0.17, 0.21, 0.68])
dg = np.array([size[r["id"]] for r in R])
b.plot(dg, astar, "o", ms=4, color=ACC, mfc="white")
b.axhline(np.median(astar), color="k", lw=0.9, ls="--")
b.text(dg.min(), 2.95, f"破線：中央値 {np.median(astar):.2f}", ha="left", va="top", fontsize=8.5)
b.set_xscale("log"); b.set_xticks([20, 50, 100, 200]); b.set_xticklabels(["20", "50", "100", "200"]); b.minorticks_off(); b.set_xlabel("部品の大きさ（外接箱の対角）[mm]"); b.set_ylabel("最適な倍率")
b.set_ylim(0, 3.2)
b.set_title("(b) 倍率は部品によらずほぼ同じ", fontsize=10.5, fontweight="bold")
d = fig.add_axes([0.78, 0.17, 0.20, 0.68])
d.plot(c, np.array([r["rescue_10_2f"] for r in R]) * 100, "o", ms=4, color=ACC, mfc="white")
d.set_xlabel("補正前の予測とデータの形の一致 cos"); d.set_ylabel("10 点で補正後の誤差 [%]")
d.set_xlim(min(0.8, c.min() - 0.01), 1.0); d.set_ylim(0, 70)
d.set_title("(c) 形が合うほど直る", fontsize=10.5, fontweight="bold")
fig.text(0.5, 0.005, "相対誤差：全節点で、予測とデータの差から剛体運動を除いた rms ÷ データの反りの rms。"
         "データは Greif & Meyer（Zenodo 17831586）の Moldflow 計算、途中経過。補正前の予測の材料定数は一般値（例示）。",
         ha="center", fontsize=7.5, color="0.4")
fig.savefig(os.path.join(OUTD, "rescue3d_errors.png"), dpi=250); plt.close(fig)
print("wrote rescue3d_flow.png, rescue3d_errors.png;", N, "geometries; medians",
      {k: round(float(np.median(v)), 3) for k, v in cols})

# ------------------------------------------------------------------ transfer (no measurement)
fig = plt.figure(figsize=(12.0, 3.2))
ax = fig.add_axes([0.0, 0.0, 0.42, 1.0]); ax.set_xlim(0, 42); ax.set_ylim(0, 32); ax.axis("off")
for i in range(4):
    ax.add_patch(FancyBboxPatch((1.5 + i * 4.3, 17), 3.4, 8, boxstyle="round,pad=0.2,rounding_size=0.6", ec="0.4", fc="0.95"))
    ax.text(3.2 + i * 4.3, 21, f"部品\n{'ABC…'[i]}", ha="center", va="center", fontsize=8.5)
ax.text(9.5, 28.5, "ほかの部品（データあり）", ha="center", fontsize=9.5, fontweight="bold")
ax.text(9.5, 13.5, f"それぞれの最適な倍率\n→ 中央値 {np.median(astar):.2f}", ha="center", va="top", fontsize=9)
ax.add_patch(FancyArrowPatch((19.5, 21), (27.5, 21), arrowstyle="-|>", mutation_scale=16, color=ACC, lw=1.6))
ax.text(23.5, 23.0, "倍率だけ渡す", ha="center", fontsize=8.5, color=ACC)
ax.add_patch(FancyBboxPatch((29, 16), 9, 10, boxstyle="round,pad=0.2,rounding_size=0.6", ec=ACC, fc="white", lw=1.4))
ax.text(33.5, 21, "新しい部品\n（測定なし）", ha="center", va="center", fontsize=9, color=ACC, fontweight="bold")
ax.text(33.5, 13.5, "補正前の予測 × 倍率", ha="center", va="top", fontsize=9)
ax.text(21, 3.5, "一つ抜き：各部品を「新しい部品」とみなし、残りの部品の倍率で予測する", ha="center", fontsize=8.5, color="0.35")
b = fig.add_axes([0.52, 0.2, 0.45, 0.68])
labs = ["補正なし", "測定なし\n（他の部品の倍率）", "5 点\n倍率 1 個", "10 点\n倍率 2 個"]
vals = [np.median(e0), np.median(eL), np.median([r["rescue_5_1f"] for r in R]), np.median([r["rescue_10_2f"] for r in R])]
yy = np.arange(len(labs))[::-1]
b.barh(yy, np.array(vals) * 100, color=["0.7", ACC, "0.45", "0.45"], height=0.6)
for y_, v in zip(yy, vals):
    b.text(v * 100 + 1, y_, f"{v * 100:.0f} %", va="center", fontsize=10, fontweight="bold")
b.set_yticks(yy); b.set_yticklabels(labs, fontsize=9); b.set_xlim(0, 65)
b.set_xlabel(f"反りの相対誤差の中央値 [%]（{N} 形状）")
b.set_title("測定なしでも誤差は半分以下", fontsize=10.5, fontweight="bold")
fig.savefig(os.path.join(OUTD, "rescue3d_transfer.png"), dpi=250); plt.close(fig)
print("wrote rescue3d_transfer.png")
