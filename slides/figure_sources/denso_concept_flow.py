"""denso_concept_flow.py -- explanatory figure: the whole flow of the DENSO residual-strain study, with the
state of each step (done in this repository / needs DENSO data). Schematic; the numbers quoted are from
the scripts named in research/MEMO_denso_variable_ranking.md.
Run:  python3 slides/figure_sources/denso_concept_flow.py  (writes slides/figures_paper/concept_flow.png)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "figures_paper", "concept_flow.png")
ACC = "#1F3A5F"
matplotlib.rcParams.update({"font.family": ["Times New Roman", "Liberation Serif", "IPAPMincho"], "font.size": 10})

fig, ax = plt.subplots(figsize=(12.5, 5.6))
ax.set_xlim(0, 125); ax.set_ylim(0, 56); ax.axis("off")


def box(x, y, w, h, title, lines, ec="k", fc="white", tw="bold"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", ec=ec, fc=fc, lw=1.2))
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=11, fontweight=tw, color=ec)
    ax.text(x + 1.2, y + h - 6.2, "\n".join(lines), ha="left", va="top", fontsize=8.6, linespacing=1.55)


def arrow(x0, y0, x1, y1, color="k", ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, color=color, lw=1.3, ls=ls))


Y, Hh, W = 23, 21, 22
xs = [1, 26, 51, 76, 101]
box(xs[0], Y, W, Hh, "① 入力", ["成形条件", "　金型温度・樹脂温度", "材料の状態", "　結晶化度", "　繊維の向き（MD/TD）"])
box(xs[1], Y, W, Hh, "② 数値解析", ["物理モデル", "　結晶化の式（PPS）", "　配向平均・積層理論", "　Abaqus／Ansys で照合", "条件を変えて 600 枚"])
box(xs[2], Y, W, Hh, "③ 学習", ["代わりのモデル（MLP）", "　入力 → 残留応力", "　400 枚で誤差が頭打ち", "　（0.7 MPa）", "物理計算の代わりに使う"])
box(xs[3], Y, W, Hh, "④ 少しの実測で補正", ["穿孔法（穴あけ）", "　倍率 1 つ：0.67", "　TD・45° の誤差 46 → 11 µε", "　（MD だけで決める）", "≒ 無応力温度 127 °C"], ec=ACC)
box(xs[4], Y, W, Hh, "⑤ 予測・設計", ["残留ひずみ・反り", "成形条件の設計", "　金型温度だけでは", "　反りは直らない", "　→ 繊維の向きが鍵"])
for i in range(4):
    arrow(xs[i] + W + 0.4, Y + Hh / 2, xs[i + 1] - 0.4, Y + Hh / 2)

# status row
st = [("済み（既報の値）", "k"), ("済み", "k"), ("済み（合成データ）", "k"), ("既報の図 1 枚で確認", ACC), ("試作（例示定数）", "k")]
for x, (s, c) in zip(xs, st):
    ax.text(x + W / 2, Y - 2.2, s, ha="center", va="top", fontsize=8.5, color=c)

# what DENSO data would replace
box(1, 1.5, 72, 13, "デンソー様のデータで置き換えたいもの", [
    "流動解析の結果 → ① の繊維の向き（今は既報の図からの読み取り。反りの一番の原因）",
    "DMA（粘弾性）・DSC（結晶化）→ ② の材料定数（今は文献の一般値・例示値）",
    "穿孔法などの実測（多数の条件）→ ④ の補正（今は既報の試験片 1 枚）"], ec="0.35")
arrow(20, 14.8, 12, Y - 4.0, color="0.45", ls="--")
arrow(37, 14.8, 37, Y - 4.0, color="0.45", ls="--")
arrow(62, 14.8, 87, Y - 4.0, color="0.45", ls="--")
ax.text(76, 9.5, "論文：段階①（補正の考え方）を\n実データで示す（2027 年夏 投稿目標）", ha="left", va="center", fontsize=9, color=ACC)
ax.text(62.5, 54.5, "短繊維強化樹脂の射出成形品：残留ひずみ予測の全体の流れ", ha="center", va="top", fontsize=13, fontweight="bold")
ax.text(62.5, 50.0, "数値解析で学習し、少しの実測で補正して、予測と成形条件の設計に使う", ha="center", va="top", fontsize=9.5, color="0.3")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=250, bbox_inches="tight"); print("wrote", OUT)
