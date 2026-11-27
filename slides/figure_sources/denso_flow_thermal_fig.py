"""denso_flow_thermal_fig.py -- figure of validation/flow_cfd_vs_1d.py and validation/flow_freeze_thermal.py
(style as thesis_style: Times / STIX, Mincho).
(a) a_xx through the thickness 50 mm from the gate: fifa 2-D openInjMoldSimF run (illustrative material, 74 % filled)
    against the 1-D shear model for three power-law indices (not fitted);
(b) 3 mm plate: thickness of the unfrozen channel against the contact time (finite slab, fixed wall and h = 5e3 W/m2K;
    erf solution dashed), the paper's core 1.60 mm;
(c) 1 mm plate: the same; the paper's core 0.19 mm and the contact time for the same flow rate as (b).
Thermal constants are ASSUMED (see flow_freeze_thermal.py); the paper gives no fill time.
Run:  python3 slides/figure_sources/denso_flow_thermal_fig.py   (writes slides/figures_paper/flow_thermal.png)
"""
import csv
import json
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "validation"))
import denso_flow_coupling as fc  # noqa: E402
import denso_flow_orientation as fo  # noqa: E402
import flow_freeze_thermal as ft  # noqa: E402

ACC, WARM, GREY = "#1F3A5F", "#C77700", "0.55"
matplotlib.rcParams.update({"font.family": ["Times New Roman", "Liberation Serif", "IPAPMincho"],
                            "mathtext.fontset": "stix", "font.size": 10, "xtick.direction": "in", "ytick.direction": "in"})
F3 = json.load(open(os.path.join(ROOT, "results", "denso_flow", "freeze_thermal.json")))["F3"]
fig, (a, b, c) = plt.subplots(1, 3, figsize=(13.0, 4.3))

# (a)
rows = list(csv.DictReader(open(os.path.join(ROOT, "abaqus", "denso_plate", "abaqus_results", "flow", "stage4_strip", "orientation_50mm_2D.csv"))))
d = np.array([float(r["depth_from_top_mm"]) for r in rows]); ax = np.array([float(r["a_xx"]) for r in rows])
iso = (1 / 3, 1 / 3, 1 / 3); tr = fc.trajectory(iso, 0.00207, 1.0)
for n, ls in ((0.394, ":"), (0.5, "-"), (0.7, "--")):
    r = fc.profile(dict(fo.BASE, L=50e-3, n=n, T_nf=fo.BASE["Tb"], a0=iso), 0.0, tr)
    a.plot(r["axx"], r["depth"], color=ACC, ls=ls, lw=1.6, label=f"1 次元モデル n = {n}")
a.plot(ax, d, "o", color="k", ms=4, label="流動計算（2 次元）")
a.invert_yaxis(); a.set_xlim(0, 1); a.set_ylim(3.08, 0)
a.set_xlabel("流れ方向の配向成分 a_xx"); a.set_ylabel("上面からの深さ [mm]")
a.legend(fontsize=8, frameon=False, loc="upper left")
a.set_title("(a) せん断の部分は流動計算と合う", fontsize=10.5, fontweight="bold", loc="left")


def panel(axx, Hmm, tmax, core_paper, tc_mark=None):
    ts = np.linspace(0.01, tmax, 60)
    for h, col, lab in ((None, ACC, "有限の板厚、壁の温度を固定"), (5e3, WARM, "有限の板厚、壁の熱伝達 5×10³ W/m²K")):
        axx.plot(ts, [(Hmm * 1e-3 - 2 * ft.skin(Hmm * 1e-3, t, h, n=200)) * 1e3 for t in ts], color=col, lw=1.8, label=lab)
    axx.plot(ts, [max(Hmm - 2 * ft.erf_skin(t) * 1e3, 0) for t in ts], color=GREY, ls="--", lw=1.2, label="半無限の解（以前の見積もり）")
    axx.axhline(core_paper, color="k", lw=0.9, ls=":")
    axx.text(tmax, core_paper, f"既報 {core_paper:.2f} mm ", ha="right", va="bottom", fontsize=8.5)
    axx.set_xlim(0, tmax); axx.set_ylim(0, Hmm)
    axx.set_xlabel("溶けた樹脂が壁に触れている時間 [s]"); axx.set_ylabel("固まっていない中心の厚さ [mm]")


panel(b, 3.08, 2.0, 1.60)
for k, col in (("fixed wall", ACC), ("h = 5e3", WARM)):
    b.plot([F3[k]["contact_3mm_s"]], [1.60], "o", color=col, ms=6)
b.legend(fontsize=7.5, frameon=False, loc="upper right")
b.set_title("(b) 3 mm 板：時間 1 個を合わせる", fontsize=10.5, fontweight="bold", loc="left")
panel(c, 1.08, 0.6, 0.19)
for k, col in (("fixed wall", ACC), ("h = 5e3", WARM)):
    t1 = F3[k]["contact_1mm_same_flow_rate_s"]
    c.axvline(t1, color=col, lw=0.9, ls="-.")
c.text(F3["h = 5e3"]["contact_1mm_same_flow_rate_s"] + 0.01, 0.75, "(b) と同じ流量なら\nこの時間\n→ 全部固まる", fontsize=8.5, va="center")
c.set_title("(c) 1 mm 板：全固化の境目", fontsize=10.5, fontweight="bold", loc="left")
fig.text(0.5, 0.01, "(a) 流動計算は fifa の openInjMoldSimF（例示の材料で PPS ではない、充填 74 % まで）。n は合わせていない。"
         "(b)(c) 熱伝導だけの計算。熱の定数は仮定。既報に充填時間の記載はない。", ha="center", fontsize=7.5, color="0.4")
fig.tight_layout(rect=(0, 0.04, 1, 1))
out = os.path.join(ROOT, "slides", "figures_paper", "flow_thermal.png")
fig.savefig(out, dpi=250); print("wrote", out)
