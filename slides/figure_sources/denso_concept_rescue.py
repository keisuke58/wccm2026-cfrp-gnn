"""denso_concept_rescue.py -- figure for RESCUE in the measured quantity (validation/denso_rescue_released.py).
(a)-(c) released strain of the three gauges vs drilling depth: the paper's measurement (Tsuda et al. 2026
    Fig. 12, read by eye, +-10 micro-strain), the physics chain at the paper's conditions (T0 = 180 C), and the
    chain at T0_eff identified from the measurement (others at the paper's values);
(d) the values of T0_eff: the paper's assumption, the paper's own estimate, the constant-CTE reading of the
    common factor 0.67, and the identification with the temperature-dependent CTE (this work).
Every curve is the physics chain (influence functions x eigenstrain), not the surrogate.
Honest scope: one plate, values read by eye; the operator carries the paper's stiffness profile.
Run:  python3 slides/figure_sources/denso_concept_rescue.py  (writes slides/figures_paper/concept_rescue.png)
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "validation"))
import denso_rescue_released as rr  # noqa: E402

OUT = os.path.join(ROOT, "slides", "figures_paper", "concept_rescue.png")
ACC = "#1F3A5F"
matplotlib.rcParams.update({"font.family": ["Times New Roman", "Liberation Serif", "IPAPMincho"],
                            "mathtext.fontset": "stix", "font.size": 10, "xtick.direction": "in", "ytick.direction": "in"})
res = json.load(open(os.path.join(ROOT, "results", "denso_rescue", "results.json")))
th = res["R4"]["T0 only"]["theta"]
y0 = rr.released(rr.THETA_PAPER).reshape(3, 10)
y1 = rr.released(tuple(th)).reshape(3, 10)
ym = rr.Y_MEAS.reshape(3, 10)
d = np.concatenate([[0.0], rr.DEPTH])
z = lambda v: np.concatenate([[0.0], v])

fig = plt.figure(figsize=(12.0, 3.9))
names = {"GCH1": "Ch.1（TD）", "GCH2": "Ch.2（45°）", "GCH3": "Ch.3（MD）"}
for k, g in enumerate(rr.CH):
    a = fig.add_axes([0.05 + 0.205 * k, 0.15, 0.17, 0.68])
    a.errorbar(rr.DEPTH, ym[k], yerr=10, fmt="s", color="k", ms=4, mfc="white", capsize=2, lw=0.8, label="実測（既報）")
    a.plot(d, z(y0[k]), "--", color="0.5", lw=1.4, label="$T_0$ = 180 °C")
    a.plot(d, z(y1[k]), "-", color=ACC, lw=2.0, label=f"$T_{{0,\\mathrm{{eff}}}}$ = {th[0]:.0f} °C")
    a.axhline(0, color="0.8", lw=0.6)
    a.set_xlim(0, 2.05); a.set_xlabel("穴の深さ [mm]")
    if k == 0:
        a.set_ylabel("解放ひずみ [µε]")
    a.set_title(f"({'abc'[k]}) {names[g]}", fontsize=10.5, fontweight="bold")
    if k == 2:
        a.legend(frameon=False, fontsize=7.5, loc="lower right", title="計算（実線・破線）と実測", title_fontsize=7.5)
b = fig.add_axes([0.685, 0.15, 0.295, 0.68])
rows = [("既報の解析で仮定（アニール温度）", 180.0, "0.5"),
        ("既報の推定", 120.0, "0.5"),
        ("倍率 0.67、熱膨張係数一定の換算", 127.0, "0.5"),
        ("本研究：解放ひずみから同定", th[0], ACC)]
for i, (lab, v, c) in enumerate(rows[::-1]):
    b.plot([v], [i], "o", color=c, ms=8)
    b.text(v, i + 0.2, f"{lab}：{v:.0f} °C", fontsize=8.5, color=c, va="bottom", ha="right" if v > 170 else "left")
b.errorbar([th[0]], [0], xerr=[[7.6], [7.6]], color=ACC, capsize=3, lw=1.2)
b.set_yticks([])
b.set_xlim(100, 200); b.set_ylim(-0.6, len(rows) - 0.4); b.set_xlabel("$T_{0,\\mathrm{eff}}$ [°C]")
b.set_title("(d) 無応力温度の比較", fontsize=10.5, fontweight="bold")
fig.text(0.5, 0.005, f"実測との RMSE：{res['R4']['T0 only']['rmse_paper_conditions']:.0f} → {res['R4']['T0 only']['rmse_physics_at_theta']:.0f} µε（30 点、試験片 1 枚、図の読み取り値）。"
         "(d) の誤差棒は合成データでの同定誤差（rms 7.6 °C）。曲線はすべて物理計算（影響関数 × 固有ひずみ）。",
         ha="center", fontsize=7.5, color="0.4")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=250); print("wrote", OUT)
