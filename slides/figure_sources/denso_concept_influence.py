"""denso_concept_influence.py -- explanatory figure for the hole-drilling influence functions (fifa guide §16).
(a) schematic: the plate split into 0.2 mm depth bands; one band at a time gets a unit eigenstrain;
(b) data: released strain of the MD gauge (Ch.3) for a unit xx eigenstrain in each band (Abaqus, fifa);
(c) data: the paper's eigenstrain, averaged per band and superposed with the band responses, against the
    full production run (check I2: 0.8 %).
Run:  python3 slides/figure_sources/denso_concept_influence.py  (writes slides/figures_paper/concept_influence.png)
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
D = os.path.join(ROOT, "abaqus", "denso_plate", "abaqus_results", "influence")
sys.path.insert(0, os.path.join(ROOT, "abaqus", "denso_plate"))
import gen_hole_drilling as g  # noqa: E402

OUT = os.path.join(ROOT, "slides", "figures_paper", "concept_influence.png")
ACC = "#1F3A5F"
matplotlib.rcParams.update({"font.family": ["Times New Roman", "Liberation Serif", "IPAPMincho"],
                            "mathtext.fontset": "stix", "font.size": 10, "xtick.direction": "in", "ytick.direction": "in"})
NB = 16
rel = lambda c, b: np.array(json.load(open(os.path.join(D, f"denso_hole_inf_{c}_{b:02d}_released.json")))["released"]["GCH3"])
Gx = np.stack([rel("xx", b) for b in range(NB)], -1); Gy = np.stack([rel("yy", b) for b in range(NB)], -1)
depth = np.arange(11) * 0.2
mat = g.layer_materials(154); dep = np.asarray(mat["depth"]); eth = np.asarray(mat["eth"])
band = np.minimum((dep // 0.2).astype(int), NB - 1)
ex = np.array([eth[band == b, 0].mean() for b in range(NB)]); ey = np.array([eth[band == b, 1].mean() for b in range(NB)])
yop = Gx @ (ex / 1e-3) + Gy @ (ey / 1e-3)
full = np.array(json.load(open(os.path.join(ROOT, "abaqus", "denso_plate", "abaqus_results", "denso_hole_released.json")))["released"]["GCH3"])

fig = plt.figure(figsize=(11.5, 4.0))
a = fig.add_axes([0.02, 0.12, 0.2, 0.72]); a.set_xlim(-0.5, 6.5); a.set_ylim(-3.8, 0.6); a.axis("off")
for b in range(NB):
    y0 = -0.2 * (b + 1)
    a.add_patch(Rectangle((0, y0), 5, 0.2, fc=ACC if b == 3 else ("0.95" if b % 2 else "white"), ec="0.6", lw=0.4))
a.add_patch(Rectangle((2.2, -2.0), 0.6, 2.0, fc="0.25", ec="k"))
a.text(5.2, -0.7, "← この層だけ\n　ひずみを与える", fontsize=8.5, color=ACC, va="center")
a.text(2.9, -1.6, "← 穴（0.2 mm ずつ）", ha="left", fontsize=8, color="0.2", bbox=dict(fc="white", ec="none", pad=1))
a.text(2.5, -3.35, "0.2 mm × 16 層（板厚 3.08 mm）", ha="center", va="top", fontsize=8.5)
a.text(2.5, 0.35, "(a) 1 層ずつ計算", ha="center", fontsize=10.5, fontweight="bold")
b_ = fig.add_axes([0.30, 0.15, 0.31, 0.68])
cm = plt.cm.Greys(np.linspace(0.35, 1.0, NB))
for k in range(NB):
    b_.plot(depth, Gx[:, k], "-", color=ACC if k == 3 else cm[k], lw=2.0 if k == 3 else 0.9)
b_.text(0.65, Gx[:, 3].min() - 0.3, "層 4（深さ 0.6〜0.8 mm）", ha="left", va="center", fontsize=8, color=ACC)
b_.set_xlabel("穴の深さ [mm]"); b_.set_ylabel("解放ひずみ [µε]（ひずみ 0.001 あたり）")
b_.set_title("(b) 層ごとの応答（MD のゲージ）", fontsize=10.5, fontweight="bold")
c = fig.add_axes([0.70, 0.15, 0.28, 0.68])
c.plot(depth, full, "k-", lw=2.2, label="本番の計算（Abaqus）")
c.plot(depth, yop, "o", color=ACC, ms=5, mfc="white", label="16 層の足し合わせ")
c.set_xlabel("穴の深さ [mm]"); c.set_ylabel("解放ひずみ [µε]")
c.legend(frameon=False, fontsize=8.5, loc="lower right")
c.set_title("(c) 足し合わせで再現（誤差 0.8 %）", fontsize=10.5, fontweight="bold")
fig.text(0.5, 0.01, "どんなひずみの分布でも、層ごとの応答の足し合わせで、穿孔法で測る値を計算できる。計算は fifa の Abaqus（既報の板、本番メッシュ）。",
         ha="center", fontsize=7.5, color="0.4")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=250); print("wrote", OUT, float(np.max(np.abs(yop - full)) / np.max(np.abs(full))))
