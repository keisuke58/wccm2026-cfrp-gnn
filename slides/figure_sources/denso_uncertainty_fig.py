"""denso_uncertainty_fig.py -- figure of validation/denso_uncertainty.py (style as thesis_style: Times / STIX, Mincho).
(a) chi^2 of the measured released strain vs T0_eff at nominal constants, 68 % interval;
(b) max residual tension of the plate over the Monte Carlo of the constants: prediction only (T0 = 180 C) vs with
    the measurement (T0_eff re-identified per sample);
(c) strength MD / TD (criterion A), median and 5-95 % band, both cases;
(d) which constant moves the result (Spearman rank correlation, with the measurement).
Data: results/denso_uncertainty/uncertainty.json. The constant ranges are ASSUMED (see the script docstring); the
bands are those of the model, not a measured scatter.
Run:  python3 slides/figure_sources/denso_uncertainty_fig.py   (writes slides/figures_paper/uncertainty.png)
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging

import numpy as np

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "validation"))
import denso_rescue_released as rr  # noqa: E402
import denso_uncertainty as du  # noqa: E402

ACC, WARM, GREY = "#1F3A5F", "#C77700", "0.6"
matplotlib.rcParams.update({"font.family": ["Times New Roman", "Liberation Serif", "IPAPMincho"],
                            "mathtext.fontset": "stix", "font.size": 10, "xtick.direction": "in", "ytick.direction": "in"})
U = json.load(open(os.path.join(ROOT, "results", "denso_uncertainty", "uncertainty.json")))
S = U["samples"]; N = len(S)

fig = plt.figure(figsize=(13.0, 3.6))
# (a)
a = fig.add_axes([0.05, 0.17, 0.19, 0.68])
ch = du.Chain(du.NOM); Ts = np.arange(110.0, 200.0 + 1e-9, 0.5)
chi = np.array([np.sum((ch.released(T) - rr.Y_MEAS) ** 2) for T in Ts]) / U["U1"]["sigma"] ** 2
a.plot(Ts, chi - chi.min(), color=ACC, lw=1.8)
a.axvspan(U["U1"]["lo"], U["U1"]["hi"], color=ACC, alpha=0.18, lw=0)
a.axvline(180, color=GREY, ls="--", lw=1.0); a.text(182, 30, "既報の\n仮定\n180 °C", ha="left", va="top", fontsize=8, color="0.4")
a.text(160, 57, f"{U['U0']['T0']:.0f} °C\n({U['U1']['lo']:.0f}–{U['U1']['hi']:.0f})", ha="left", va="top", fontsize=8.5, color=ACC)
a.set_xlim(110, 200); a.set_ylim(0, 60)
a.set_xlabel("無応力温度 T₀ [°C]"); a.set_ylabel("Δχ²（実測との差）")
a.set_title("(a) 実測で決まる温度の幅", fontsize=10.5, fontweight="bold", loc="left")
# (b)
b = fig.add_axes([0.31, 0.17, 0.19, 0.68])
sP = np.array([r["P"]["smax"] for r in S]); sI = np.array([r["I"]["smax"] for r in S])
bins = np.linspace(min(sP.min(), sI.min()) - 1, max(sP.max(), sI.max()) + 1, 32)
b.hist(sP, bins, color=GREY, alpha=0.7, label="予測だけ（180 °C）")
b.hist(sI, bins, color=ACC, alpha=0.75, label="実測で温度を決める")
b.set_xlabel("板の最大の残留引張応力 [MPa]"); b.set_ylabel("サンプル数")
b.legend(fontsize=8, frameon=False, loc="upper right")
wr = U["U2"]["width_ratio_smax_I_over_P"]
b.set_title(f"(b) 応力のぶれ幅 → {wr:.2f} 倍", fontsize=10.5, fontweight="bold", loc="left")
# (c)
c = fig.add_axes([0.57, 0.17, 0.19, 0.68])
for j, (dr, x0) in enumerate((("MD", 0), ("TD", 1))):
    for k, (case, col, dx) in enumerate((("P", "0.45", -0.14), ("I", ACC, 0.14))):
        u = U["U2"][case][dr]
        c.errorbar(x0 + dx, u["median"], yerr=[[u["median"] - u["p05"]], [u["p95"] - u["median"]]], fmt="o", color=col,
                   ms=6, capsize=4, lw=1.6, label=(("予測だけ（180 °C）", "実測で温度を決める")[k] if j == 0 else None))
        c.text(x0 + dx + 0.06, u["median"], f"{u['median']:.0f}", va="center", fontsize=8.5, color=col)
c.axhline(155, color="k", lw=0.8, ls=":"); c.text(1.45, 156, "残留応力なし MD 155", ha="right", va="bottom", fontsize=7.5)
c.set_xticks([0, 1]); c.set_xticklabels(["MD", "TD"]); c.set_xlim(-0.5, 1.5); c.set_ylim(55, 170)
c.set_ylabel("引張強度 [MPa]（基準 A、例示）")
c.legend(fontsize=8, frameon=False, loc="lower center", ncol=2, handletextpad=0.3, columnspacing=0.8)
c.set_title("(c) 強度（中央値と 5–95 %）", fontsize=10.5, fontweight="bold", loc="left")
# (d)
d = fig.add_axes([0.85, 0.17, 0.14, 0.68])
lab = {"kE_matrix": "樹脂の剛性", "kA_matrix": "樹脂の熱膨張", "kE_glass": "ガラスの剛性", "Vf": "繊維の量",
       "s": "繊維の長さ比", "w": "中心層の幅", "m": "配向の強さ", "k_datasheet": "カタログ強度"}
r = U["U3"]["smax_I"]; names = sorted(r, key=lambda k: abs(r[k]))
d.barh(range(len(names)), [r[k] for k in names], color=[ACC if r[k] > 0 else WARM for k in names], height=0.6)
d.set_yticks(range(len(names))); d.set_yticklabels([lab[k] for k in names], fontsize=8.5)
d.axvline(0, color="k", lw=0.8); d.set_xlim(-1, 1); d.set_xlabel("順位相関（応力との）")
d.set_title("(d) 効く定数", fontsize=10.5, fontweight="bold", loc="left")
fig.text(0.5, 0.005, f"板（既報の 3 mm 試験片）、材料定数を仮定した幅で一様に {N} 回ばらつかせた計算（モデルの幅であり、実測のばらつきではない）。"
         "(a) σ は最適点の RMSE。(c) 基準 A は数値例。", ha="center", fontsize=7.5, color="0.4")
out = os.path.join(ROOT, "slides", "figures_paper", "uncertainty.png")
fig.savefig(out, dpi=250); print("wrote", out)
