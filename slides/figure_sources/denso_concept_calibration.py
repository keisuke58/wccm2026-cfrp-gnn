"""denso_concept_calibration.py -- explanatory figure: how the hole-drilling measurement corrects the physics.
(a) schematic of the hole-drilling test (gauges around a small hole drilled in steps);
(b) data: released strain of the MD gauge (used for the correction) and of the TD gauge (predicted):
    measured (paper Fig. 12, read by eye), physics as is (Abaqus, production mesh) and the physics
    multiplied by ONE common factor fitted on the MD gauge only (check H3 of
    validation/denso_tsuda2026_transfer.py).
Run:  python3 slides/figure_sources/denso_concept_calibration.py  (writes slides/figures_paper/concept_calibration.png)
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "validation"))
import denso_tsuda2026_transfer as tr  # noqa: E402

OUT = os.path.join(ROOT, "slides", "figures_paper", "concept_calibration.png")
AR = os.path.join(ROOT, "abaqus", "denso_plate", "abaqus_results")
ACC = "#1F3A5F"
matplotlib.rcParams.update({"font.family": ["Times New Roman", "Liberation Serif", "IPAPMincho"],
                            "mathtext.fontset": "stix", "font.size": 10, "xtick.direction": "in", "ytick.direction": "in"})

H, full, y, yfe, gauge, dd, depth = tr.load(AR, os.path.join(AR, "denso_hole_released.json"), "denso_hole")
p0 = H.sum(1)
md = gauge == 2; td = gauge == 0
c = float(p0[md] @ y[md] / (p0[md] @ p0[md]))

fig = plt.figure(figsize=(10.5, 4.0))
# (a) schematic
a = fig.add_axes([0.02, 0.12, 0.30, 0.74]); a.set_xlim(-5.5, 5.5); a.set_ylim(-5.5, 5.8); a.set_aspect("equal"); a.axis("off")
a.add_patch(Rectangle((-5, -5), 10, 10, fc="0.96", ec="0.5"))
a.add_patch(Circle((0, 0), 0.8, fc="0.3", ec="k"))
a.add_patch(Rectangle((2.5, -0.55), 2.0, 1.1, fc="white", ec=ACC, lw=1.6))
a.add_patch(Rectangle((-0.55, 2.5), 1.1, 2.0, fc="white", ec="0.45", lw=1.6))
a.text(3.5, -1.3, "MD のゲージ", ha="center", fontsize=8.5, color=ACC)
a.text(1.0, 3.5, "TD の\nゲージ", ha="left", va="center", fontsize=8.5, color="0.35")
a.text(0, -1.5, "穴を 0.2 mm ずつ\nあけていく", ha="center", va="top", fontsize=8.5)
a.annotate("", xy=(4.7, -4.5), xytext=(2.7, -4.5), arrowprops=dict(arrowstyle="->"))
a.text(3.7, -4.1, "流れ方向", ha="center", fontsize=8)
a.text(0, 5.3, "(a) 穿孔法：穴の周りで解放されるひずみを測る", ha="center", fontsize=10.5, fontweight="bold")
# (b) data
b = fig.add_axes([0.42, 0.15, 0.55, 0.7])
d = np.r_[0, dd[md]]
f = lambda v, m: np.r_[0, v[m]]
b.axhline(0, color="0.85", lw=0.6)
b.plot(d, f(y, md), "o", color=ACC, ms=4.5, label="実測（既報の図から読み取り）")
b.plot(d, f(p0, md), ":", color=ACC, lw=1.4, label="計算そのまま（約 1.5 倍）")
b.plot(d, f(c * p0, md), "-", color=ACC, lw=2.0, label=f"計算 × {c:.2f}（MD で倍率を決める）")
b.plot(d, f(y, td), "s", color="0.45", ms=4.5, mfc="white")
b.plot(d, f(p0, td), ":", color="0.45", lw=1.4)
b.plot(d, f(c * p0, td), "-", color="0.45", lw=2.0)
b.text(2.02, 235, "MD：倍率を決めるのに使う", ha="right", fontsize=8.5, color=ACC)
b.text(2.02, -215, "TD：使っていないのに合う", ha="right", fontsize=8.5, color="0.35")
b.set_xlabel("穴の深さ [mm]"); b.set_ylabel("解放ひずみ [µε]")
b.set_xlim(0, 2.05); b.set_ylim(-330, 500); b.legend(frameon=False, fontsize=8, loc="upper left")
b.set_title(f"(b) 倍率 1 つ（{c:.2f}）で直す：TD・45° の誤差 46 → 11 µε", fontsize=10.5, fontweight="bold")
fig.text(0.5, 0.01, "計算は Abaqus（既報と同じ入力）。実測は既報 図12 の読み取り値、試験片 1 枚。倍率 0.67 は無応力温度 約 127 °C に相当。",
         ha="center", fontsize=7.5, color="0.4")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=250); print("wrote", OUT, "factor", round(c, 3))
