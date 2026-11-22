"""denso_flow_twozone_fig.py -- figure of validation/denso_flow_coupling.py (style as thesis_style: Times / STIX, Mincho).
(a) 3 mm plate: paper Fig. 8a (read by eye) vs the 1-D shear model (best of the sweep) and the two-zone rule
    (frozen skins aligned, channel at the gate state); (b) 1 mm plate: paper Fig. 7a vs the two-zone rule
    (fill time scaled with the thickness, assumed). No number is fitted to the orientation data.
Run:  python3 slides/figure_sources/denso_flow_twozone_fig.py   (writes slides/figures_paper/flow_twozone.png)
"""
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

ACC, WARM, GREY = "#1F3A5F", "#C77700", "0.55"
matplotlib.rcParams.update({"font.family": ["Times New Roman", "Liberation Serif", "IPAPMincho"],
                            "mathtext.fontset": "stix", "font.size": 10, "xtick.direction": "in", "ytick.direction": "in"})
P = json.load(open(os.path.join(ROOT, "data", "literature", "tsuda2026_jscm52_88.json"), encoding="utf-8"))
f8, f7 = P["fig8_orientation_3mm"], P["fig7_orientation_1mm"]
base = dict(fo.BASE, L=50e-3)
tr = fc.trajectory(base["a0"], base["Ci"], 0.1)
r1d = fc.profile(dict(base, n=0.3), 0.0, tr)                       # widest core of the sweep
al = fc.trajectory(base["a0"], base["Ci"], 1.0)[1][-1]
a0 = base["a0"]


def twozone(Hmm, d):
    z = np.linspace(0, Hmm, 400)
    sk = (z < d) | (z > Hmm - d)
    return z, np.where(sk, al[0, 0], a0[0]), np.where(sk, al[1, 1], a0[1])


fig, (a, b) = plt.subplots(1, 2, figsize=(9.0, 4.8), gridspec_kw=dict(width_ratios=[1.5, 1]))
d3 = fo.frozen(150.0, base) * 1e3
z, x2, y2 = twozone(3.08, d3)
a.axhspan(0, d3, color=ACC, alpha=0.07, lw=0); a.axhspan(3.08 - d3, 3.08, color=ACC, alpha=0.07, lw=0)
a.plot(f8["a_xx"], f8["depth_mm"], "o", color="k", ms=4, label="a_xx 既報（読み取り）")
a.plot(f8["a_yy"], f8["depth_mm"], "s", color=GREY, ms=4, mfc="white", label="a_yy 既報（読み取り）")
a.plot(r1d["axx"], r1d["depth"], color=WARM, lw=1.2, ls=":", label="1 次元せん断モデル（最良）a_xx")
a.plot(x2, z, color=ACC, lw=2.0, label="2 層の規則 a_xx")
a.plot(y2, z, color=ACC, lw=2.0, ls="--", label="2 層の規則 a_yy")
a.text(0.5, d3 / 2, f"固化層 {d3:.2f} mm", ha="center", va="center", fontsize=8.5, color=ACC, bbox=dict(fc="white", ec="none", pad=1))
a.invert_yaxis(); a.set_xlim(0, 1); a.set_ylim(3.08, 0)
a.set_xlabel("配向テンソルの成分"); a.set_ylabel("上面からの深さ [mm]")
fig.legend(*a.get_legend_handles_labels(), fontsize=8, frameon=False, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.065))
a.set_title("(a) 3 mm 板（ゲートから 50 mm）", fontsize=10.5, fontweight="bold", loc="left")
d1 = fo.frozen(150.0, dict(base, t_fill=base["t_fill"] * 1.08 / 3.08)) * 1e3
z, x2, y2 = twozone(1.08, d1)
zz = np.array(f7["z_mm"])
b.axhspan(0, d1, color=ACC, alpha=0.07, lw=0); b.axhspan(1.08 - d1, 1.08, color=ACC, alpha=0.07, lw=0)
b.plot(f7["a_xx"], zz + 0.54, "o", color="k", ms=4); b.plot(f7["a_yy"], zz + 0.54, "s", color=GREY, ms=4, mfc="white")
b.plot(f7["a_xx"], 0.54 - zz, "o", color="k", ms=4, alpha=0.35); b.plot(f7["a_yy"], 0.54 - zz, "s", color=GREY, ms=4, mfc="white", alpha=0.35)
b.plot(x2, z, color=ACC, lw=2.0); b.plot(y2, z, color=ACC, lw=2.0, ls="--")
b.text(0.5, d1 / 2, f"固化層 {d1:.2f} mm", ha="center", va="center", fontsize=8.5, color=ACC, bbox=dict(fc="white", ec="none", pad=1))
b.invert_yaxis(); b.set_xlim(0, 1); b.set_ylim(1.08, 0)
b.set_xlabel("配向テンソルの成分"); b.set_ylabel("上面からの深さ [mm]")
b.set_title("(b) 1 mm 板（薄い点は対称に折り返し）", fontsize=10.5, fontweight="bold", loc="left")
fig.text(0.5, 0.005, "2 層の規則：固化層（壁からの熱伝導、充填時間で決まる）は流れ方向にそろい、流れの通り道はゲートの向きのまま。"
         "\n配向データに合わせた数はない。充填時間（3 mm で 1 s、1 mm は厚さに比例）と固化温度は仮定。",
         ha="center", fontsize=7.5, color="0.4")
fig.tight_layout(rect=(0, 0.12, 1, 1))
out = os.path.join(ROOT, "slides", "figures_paper", "flow_twozone.png")
fig.savefig(out, dpi=250); print("wrote", out)
