"""denso_concept_theory.py -- explanatory schematic (not data) of the theory in five steps, minimal text:
(1) fibres resist shrinkage along their direction; (2) layers that want to shrink differently pull on each other
(residual stress), an up/down difference bends the plate; (3) the stress-free temperature: above it the hot resin
lets the force escape, below it the force is kept; (4) hole drilling releases the locked-in force and the gauge
sees the surface move; (5) correction: the computed shape is right, one factor from a few measurements fixes the size.
The only numbers shown (180 C assumed in the paper, 143 C identified) are from results/denso_rescue/results.json.
Run:  python3 slides/figure_sources/denso_concept_theory.py   (writes slides/figures_paper/concept_theory.png)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, "slides", "figures_paper", "concept_theory.png")
ACC, WARM, GREY = "#1F3A5F", "#C77700", "0.55"
matplotlib.rcParams.update({"font.family": ["Times New Roman", "Liberation Serif", "IPAPMincho"], "font.size": 10})
T0ID = json.load(open(os.path.join(ROOT, "results", "denso_rescue", "results.json")))["R4"]["T0 only"]["theta"][0]

fig = plt.figure(figsize=(14.0, 4.2))
W = 0.19; xs = [0.005 + i * 0.2 for i in range(5)]
titles = ["① 繊維の向きで縮み方が違う", "② 層どうしが引っ張り合う", "③ 力が残り始める温度", "④ 穴を掘って力を逃がす", "⑤ 数点で大きさを直す"]
axs = []
for x, t in zip(xs, titles):
    a = fig.add_axes([x, 0.04, W, 0.80]); a.set_xlim(0, 10); a.set_ylim(0, 10); a.axis("off")
    a.set_title(t, fontsize=12, fontweight="bold", loc="left"); axs.append(a)
for x in xs[1:]:
    fig.add_artist(FancyArrowPatch((x - 0.012, 0.47), (x + 0.002, 0.47), transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=14, color="0.3"))


def arr(a, p, q, c="k", lw=1.4, ms=12):
    a.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=c, lw=lw))


# (1) a block with fibres along x: hot (dashed) -> cold (solid): shrinks little along x, much along y
a = axs[0]
a.add_patch(Rectangle((1.0, 3.0), 8.0, 4.4, fill=False, ls="--", ec=GREY, lw=1.2))
a.add_patch(Rectangle((1.3, 4.0), 7.4, 2.4, fc="#EEF2F7", ec=ACC, lw=1.6))
rng = np.random.default_rng(1)
for _ in range(16):
    y = rng.uniform(4.25, 6.15); x0 = rng.uniform(1.5, 6.0)
    a.plot([x0, x0 + 2.2], [y, y], color=ACC, lw=1.6)
arr(a, (5, 7.4), (5, 6.5), WARM); arr(a, (5, 3.0), (5, 3.9), WARM)
arr(a, (9.0, 5.2), (8.75, 5.2), ACC, ms=8); arr(a, (1.0, 5.2), (1.25, 5.2), ACC, ms=8)
a.text(5, 8.0, "樹脂はよく縮む", ha="center", fontsize=9.5, color=WARM)
a.text(5, 2.2, "繊維の向き（MD）は縮みにくい", ha="center", fontsize=9.5, color=ACC)
a.text(5, 0.9, "点線：熱いとき　実線：冷えたあと", ha="center", fontsize=8.5, color=GREY)

# (2) three layers; the middle one wants to shrink more -> pulled; top/bottom unequal -> bends
a = axs[1]
for k, (y, wid, c) in enumerate(((6.6, 7.6, ACC), (5.4, 6.0, WARM), (4.2, 7.0, ACC))):
    a.add_patch(Rectangle((5 - wid / 2, y), wid, 1.0, fc="white", ec=c, lw=1.4, ls="--"))
a.add_patch(Rectangle((1.6, 4.2), 6.8, 3.4, fill=False, ec="k", lw=1.6))
for y, d in ((6.0, 1), (6.0, -1)):
    arr(a, (5 + d * 2.2, y), (5 + d * 3.1, y), WARM, ms=9)
a.text(5, 8.25, "縮みたい量（点線）が層ごとに違う", ha="center", fontsize=9, color="0.25")
a.text(5, 3.45, "→ 引っ張り合う力＝残留応力", ha="center", fontsize=9.5, color=WARM)
xx = np.linspace(1.6, 8.4, 50); a.plot(xx, 1.4 + 0.25 * ((xx - 5) / 3.4) ** 2 * 4, color=ACC, lw=3)
a.text(5, 0.4, "上下で違うと反る（バイメタル）", ha="center", fontsize=9.5, color=ACC)

# (3) temperature -> locked-in force
a = axs[2]
T = np.linspace(260, 20, 200)
for t0, c, ls, lab in ((180, GREY, "--", "既報の仮定 180 °C"), (T0ID, ACC, "-", f"実測から {T0ID:.0f} °C")):
    f = np.clip(t0 - T, 0, None) / 160 * 6.5
    a.plot(1.0 + (260 - T) / 240 * 8.5, 1.6 + f, color=c, ls=ls, lw=2.2)
    a.text(1.0 + (260 - t0) / 240 * 8.5, 1.25, f"{t0:.0f}", ha="center", va="top", fontsize=8.5, color=c)
a.plot([1.0, 9.5], [1.6, 1.6], color="k", lw=1.0)
a.text(1.2, 2.3, "柔らかい：\n力が逃げる", fontsize=9, color="0.3")
a.text(9.4, 2.3, "固い：力が残る", fontsize=9, ha="right", color="0.3")
a.text(5.25, 0.1, "温度（冷えていく →）", ha="center", fontsize=9)
a.text(1.0, 9.3, f"↑ 残る力\n実測から求めた {T0ID:.0f} °C だと\n残る力は小さい", fontsize=8.5, ha="left", va="top", color=ACC)

# (4) hole drilling cross-section
a = axs[3]
a.add_patch(Rectangle((0.6, 2.0), 8.8, 5.0, fc="#F5F6F8", ec="k", lw=1.2))
a.add_patch(Rectangle((4.4, 4.6), 1.2, 2.4, fc="white", ec="k", lw=1.2))
for y in (5.0, 5.8, 6.6):
    arr(a, (3.0, y), (4.2, y), WARM, ms=9); arr(a, (7.0, y), (5.8, y), WARM, ms=9)
a.add_patch(Rectangle((1.4, 7.0), 1.8, 0.35, fc=ACC, ec=ACC)); a.add_patch(Rectangle((6.8, 7.0), 1.8, 0.35, fc=ACC, ec=ACC))
a.text(2.3, 7.7, "ゲージ", ha="center", fontsize=9, color=ACC); a.text(7.7, 7.7, "ゲージ", ha="center", fontsize=9, color=ACC)
a.text(5, 8.6, "0.2 mm ずつ掘る", ha="center", fontsize=9.5)
a.text(5, 1.2, "掘った分の力が抜けて、表面が少し動く", ha="center", fontsize=9, color=WARM)
a.text(5, 0.3, "深さごとの動き → 中の力の分布", ha="center", fontsize=9, color=ACC)

# (5) correction: computed shape x factor
a = axs[4]
d = np.linspace(0, 2, 60); shape = np.sin(np.pi * d / 2.4) * np.exp(-0.5 * d)
x = 1.0 + d * 4.2
a.plot(x, 2.0 + 7.0 * shape, color=GREY, ls="--", lw=1.8)
a.plot(x, 2.0 + 4.6 * shape, color=ACC, lw=2.4)
for dm in (0.3, 0.6, 0.9):
    sm = np.sin(np.pi * dm / 2.4) * np.exp(-0.5 * dm)
    a.plot(1.0 + dm * 4.2, 2.0 + 4.6 * sm + 0.12, "s", color="k", ms=6, mfc="white")
a.plot([1.0, 9.6], [2.0, 2.0], color="k", lw=1.0)
a.text(7.2, 8.0, "計算：形は合う\n大きさがずれる", fontsize=9, color=GREY, ha="center")
a.text(7.2, 5.7, "× 倍率 1 つ", fontsize=11, color=ACC, ha="center", fontweight="bold")
a.text(2.6, 0.9, "□ 数点だけ測る", fontsize=9, ha="left")
a.text(5.3, 0.0, "→ 全体の予測が直る", fontsize=9.5, color=ACC, ha="center")
fig.text(0.5, 0.955, "模式図（データではない）。数値は既報の仮定 180 °C と、本研究で実測から求めた値のみ。", ha="center", fontsize=8.5, color="0.4")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=220); print("wrote", OUT)
