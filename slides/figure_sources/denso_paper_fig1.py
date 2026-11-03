"""denso_paper_fig1.py -- Fig. 1 of the paper draft (research/PAPER_draft_denso.md): (a) specimen and
FRAS-2 gauge layout on the drilled face, (b) quarter finite-element model (in-plane mesh near the hole,
production mesh of abaqus/denso_plate/gen_hole_drilling.py) with the gauge grids, (c) through-thickness
discretisation and drilling steps. Geometry: Tsuda et al. (2026) and the TML catalogue as recorded in
gen_hole_drilling.py; nothing is read off a result.
Run:  python3 slides/figure_sources/denso_paper_fig1.py   (writes slides/figures_paper/paper_fig1_model.png)
"""
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "abaqus", "denso_plate"))
try:                                   # repo convention: thesis_style, graceful fallback
    from thesis_style import use
    use(width_frac=1.0, aspect=0.35)
except Exception:
    pass
matplotlib.rcParams.update({"text.usetex": False, "font.family": "serif",
                            "font.serif": ["Times New Roman", "Liberation Serif", "STIXGeneral"],
                            "mathtext.fontset": "stix", "font.size": 8.5})
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle  # noqa: E402
import gen_hole_drilling as g  # noqa: E402

R0, R1, W, RH, L = (v * 1e3 for v in (*g.GAUGE_R, g.GAUGE_W, g.R_HOLE, g.L_HALF))
GRIDS = (("Ch. 3 (MD)", 0.0), ("Ch. 2", 45.0), ("Ch. 1 (TD)", 90.0))


def grid(ax, ang, **kw):
    c, s = math.cos(math.radians(ang)), math.sin(math.radians(ang))
    pts = [(u, v) for u, v in ((R0, -W / 2), (R1, -W / 2), (R1, W / 2), (R0, W / 2))]
    ax.add_patch(Polygon([(u * c - v * s, u * s + v * c) for u, v in pts], closed=True, **kw))


fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.4), gridspec_kw=dict(width_ratios=[1, 1, 0.75]))
# (a) specimen and gauges
a = ax[0]
a.add_patch(Rectangle((-L, -L), 2 * L, 2 * L, fill=False, ec="k", lw=1.0))
a.add_patch(Circle((0, 0), RH, fc="0.6", ec="k", lw=0.6))
for lab, ang in GRIDS:
    grid(a, ang, fc="none", ec="k", lw=0.8)
    c, s = math.cos(math.radians(ang)), math.sin(math.radians(ang))
    a.text((R1 + 0.9) * c, (R1 + 0.9) * s, lab, fontsize=7.5, ha="left" if ang < 80 else "center", va="center")
a.annotate("", xy=(8.5, -8.5), xytext=(4.5, -8.5), arrowprops=dict(arrowstyle="->", lw=0.8))
a.text(6.5, -7.8, "flow (MD, x)", fontsize=7.5, ha="center")
a.set_xlim(-L - 0.5, L + 0.5); a.set_ylim(-L - 0.5, L + 0.5); a.set_aspect("equal")
a.set_xlabel("x [mm]"); a.set_ylabel("y [mm]")
a.text(-0.14, 1.02, "(a)", transform=a.transAxes, fontsize=10, fontweight="bold", va="bottom")
# (b) quarter-model mesh near the hole
b = ax[1]
P, Q, hole, _ = g.mesh2d(12, 4, 26, 1.15)
P = P * 1e3
for k, q in enumerate(Q):
    b.add_patch(Polygon(P[q], closed=True, fc="0.55" if hole[k] else "none", ec="0.6", lw=0.25))
for lab, ang in GRIDS:
    grid(b, ang, fc="none", ec="k", lw=1.2)
b.set_xlim(0, 6.0); b.set_ylim(0, 6.0); b.set_aspect("equal")
b.set_xlabel("x [mm]"); b.set_ylabel("y [mm]")
b.text(-0.14, 1.02, "(b)", transform=b.transAxes, fontsize=10, fontweight="bold", va="bottom")
# (c) through thickness
c = ax[2]
H, nb = 3.08, 10
c.add_patch(Rectangle((0, 0), 5.5, H, fill=False, ec="k", lw=1.0))
for k in range(nb):
    c.add_patch(Rectangle((0, H - 0.2 * (k + 1)), RH, 0.2, fc="0.85" if k % 2 else "0.7", ec="k", lw=0.3))
c.text(RH + 0.15, H - 1.0, "10 steps × 0.2 mm", fontsize=7.5, va="center")
for z in np.linspace(0, H, 23):
    c.plot([5.2, 5.5], [z, z], color="0.55", lw=0.4)
c.text(5.1, 0.5, "154 layers × 20 µm\n(every 7th shown)", fontsize=7.5, ha="right")
c.plot([R0, R1], [H + 0.08] * 2, "k-", lw=2.5); c.text((R0 + R1) / 2, H + 0.25, "gauge", fontsize=7.5, ha="center")
c.set_xlim(0, 5.7); c.set_ylim(-0.2, H + 0.6)
c.set_xlabel("r [mm]"); c.set_ylabel("height from the back face [mm]")
c.text(-0.14, 1.02, "(c)", transform=c.transAxes, fontsize=10, fontweight="bold", va="bottom")
fig.tight_layout()
out = os.path.join(ROOT, "slides", "figures_paper", "paper_fig1_model.png")
fig.savefig(out, dpi=300); print("wrote", out)
