"""denso_concept_warp.py -- explanatory schematic (not data): why the plate warps and what T0 means.
(a) symmetric fibre orientation through the thickness -> layers pull equally -> flat;
(b) up/down-asymmetric orientation (as read from Tsuda et al. 2026 Fig. 8a) -> bends like a bimetal;
(c) cooling: above the stress-free temperature T0 the melt relaxes, below it the stress is locked in.
The numbers quoted in the labels come from abaqus/denso_plate/process_pps.py and
validation/denso_tsuda2026_transfer.py; the drawing itself is schematic.
Run:  python3 slides/figure_sources/denso_concept_warp.py  (writes slides/figures_paper/concept_warp.png)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "figures_paper", "concept_warp.png")
matplotlib.rcParams.update({"font.family": ["Times New Roman", "Liberation Serif", "IPAPMincho"],
                            "mathtext.fontset": "stix", "font.size": 10})
ACC = "#1F3A5F"


def layers(ax, y0, spec, w=6.0, h=0.5):
    """spec: list of 'MD' / 'TD' from top to bottom."""
    for i, s in enumerate(spec):
        y = y0 - (i + 1) * h
        ax.add_patch(Rectangle((0, y), w, h, fc="0.93" if s == "MD" else "white", ec="0.4", lw=0.6))
        xs = np.linspace(0.3, w - 0.3, 14)
        if s == "MD":
            for x in xs:
                ax.plot([x - 0.15, x + 0.15], [y + h / 2] * 2, color="k", lw=1.2)
        else:
            ax.plot(xs, [y + h / 2] * len(xs), "o", color="k", ms=2.2)
        ax.text(w + 0.15, y + h / 2, "流れ方向（MD）" if s == "MD" else "直角方向（TD）", va="center", fontsize=8)


fig = plt.figure(figsize=(10.5, 3.9))
# (a) symmetric
a = fig.add_axes([0.02, 0.12, 0.28, 0.76]); a.set_xlim(-0.3, 8.6); a.set_ylim(-4.2, 0.9); a.axis("off")
layers(a, 0.0, ["MD", "TD", "TD", "MD"])
a.text(3.0, 0.45, "(a) 上下対称", ha="center", fontsize=11, fontweight="bold")
a.plot([0, 6], [-2.8, -2.8], color=ACC, lw=3)
a.text(3.0, -3.35, "引っぱり合いがつり合う → 平ら", ha="center", fontsize=9)
a.text(3.0, -3.9, "（計算：配向を対称にすると反り 0）", ha="center", fontsize=8, color="0.35")
# (b) asymmetric
b = fig.add_axes([0.34, 0.12, 0.28, 0.76]); b.set_xlim(-0.3, 8.6); b.set_ylim(-4.2, 0.9); b.axis("off")
layers(b, 0.0, ["MD", "MD", "TD", "TD"])
b.text(3.0, 0.45, "(b) 上下非対称（既報の配向）", ha="center", fontsize=11, fontweight="bold")
x = np.linspace(0, 6, 50); b.plot(x, -2.55 - 0.5 * ((x - 3) / 3) ** 2, color=ACC, lw=3)
b.text(3.0, -3.35, "縮み方の違う層を貼った板 → 反る", ha="center", fontsize=9)
b.text(3.0, -3.9, "（計算：100 mm あたり約 2.9 mm、バイメタルと同じ）", ha="center", fontsize=8, color="0.35")
# (c) T0
c = fig.add_axes([0.70, 0.22, 0.28, 0.62])
T = np.linspace(320, 20, 300); T0 = 127.0
sig = np.where(T > T0, 0.0, (T0 - T))
c.plot(T, sig, color=ACC, lw=2.2)
c.axvspan(T0, 320, color="0.93", zorder=0)
c.axvline(T0, color="k", lw=0.8, ls="--")
c.text(225, 70, "柔らかい：\n力が逃げる", ha="center", fontsize=8.5)
c.text(100, 100, "固い：\n力が残る", ha="center", fontsize=8.5)
c.text(T0 + 4, 112, r"$T_0$ ≈ 127 °C", fontsize=8.5)
c.set_xlim(320, 20); c.set_ylim(0, 125)
c.set_xlabel("温度 [°C]（冷えていく →）", fontsize=9); c.set_ylabel("残る応力（模式）", fontsize=9)
c.set_yticks([])
c.set_title("(c) 力が逃げなくなる温度", fontsize=11, fontweight="bold")
fig.text(0.5, 0.01, "模式図。数値は本リポジトリの計算（既報の板 1 枚、定数の一部は例示）。", ha="center", fontsize=7.5, color="0.4")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=250); print("wrote", OUT)
