"""denso_paper_framework.py -- journal-style framework figure of the study (graphical overview, English labels):
(a) injection moulding and the skin-core fibre orientation; (b) micromechanics: Mori-Tanaka with orientation
averaging (orientation-tensor glyphs per layer); (c) eigenstrain through the thickness from T0,eff to RT (computed,
paper's plate, T0 = 180 C and the identified T0,eff); (d) structural FE (plate laminate / 3-D part, C3D10);
(e) sparse measurement (hole drilling, a few surface points) and the L2-SP calibration that feeds back T0,eff /
the factors; (f) outputs: residual strain, warpage, strength margin, moulding design.
Curves in (c) and the part in (d) are data from this repository; the rest is schematic.
Run:  python3 slides/figure_sources/denso_paper_framework.py   (writes slides/figures_paper/paper_framework.png/.pdf)
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "validation"))
import denso_rescue_released as rr  # noqa: E402

OUTD = os.path.join(ROOT, "slides", "figures_paper")
NAVY, TEAL, AMBER, GREY, LIGHT = "#1F3A5F", "#2E7D6B", "#C77700", "0.45", "#F3F5F8"
matplotlib.rcParams.update({"font.family": ["Times New Roman", "Liberation Serif", "STIXGeneral"], "mathtext.fontset": "stix",
                            "font.size": 8.5, "xtick.direction": "in", "ytick.direction": "in", "axes.linewidth": 0.7})
T0ID = json.load(open(os.path.join(ROOT, "results", "denso_rescue", "results.json")))["R4"]["T0 only"]["theta"][0]

fig = plt.figure(figsize=(7.2, 3.9))                       # double-column width (183 mm)
bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, 72); bg.set_ylim(0, 39); bg.axis("off")


def panel(x, y, w, h, tag, title, ec="0.6", fc="white"):
    bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=0.8", ec=ec, fc=fc, lw=0.8))
    bg.text(x + 0.6, y + h - 0.5, f"({tag})", fontsize=9, fontweight="bold", va="top")
    bg.text(x + 2.6, y + h - 0.5, title, fontsize=8.5, va="top")


def arrow(p, q, c="0.25", lw=1.1, rad=0.0, ls="-", ms=10):
    bg.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=c, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


def ax_at(x, y, w, h):
    return fig.add_axes([x / 72, y / 39, w / 72, h / 39])


# ------------------------------------------------------------------ layout (data units 72 x 39)
TOP, BOT = 20.4, 1.0
panel(0.6, TOP, 16.6, 17.8, "a", "Injection moulding")
panel(18.2, TOP, 17.2, 17.8, "b", "Micromechanics")
panel(36.4, TOP, 16.4, 17.8, "c", "Eigenstrain")
panel(53.8, TOP, 17.6, 17.8, "d", "Structural FE")
panel(0.6, BOT, 34.8, 18.2, "e", "Sparse measurement and calibration", ec=NAVY, fc="#F4F7FB")
panel(36.4, BOT, 35.0, 18.2, "f", "Prediction and design")

# ---------------------------------------------------------- (a) mould + skin-core plate
a = ax_at(1.2, TOP + 1.0, 15.4, 14.6); a.set_xlim(0, 10); a.set_ylim(0, 10); a.axis("off")
a.add_patch(Polygon([[0.4, 7.4], [1.6, 7.4], [1.6, 8.2], [2.4, 8.2], [2.4, 6.6], [0.4, 6.6]], fc="0.85", ec="0.3", lw=0.6))
a.add_patch(Rectangle((2.4, 6.6), 7.2, 1.6, fc="white", ec="0.3", lw=0.8))
a.add_patch(FancyArrowPatch((2.8, 7.4), (8.8, 7.4), arrowstyle="-|>", mutation_scale=9, color=AMBER, lw=1.2))
a.text(6.0, 8.5, "melt flow (MD)", ha="center", fontsize=7.5, color=AMBER)
a.text(0.4, 9.2, "gate", fontsize=7, color="0.35")
# cross-section with skin / core orientation
y0, H = 0.8, 4.6
layers = [(0.0, 0.22, "skin", 0), (0.22, 0.78, "core", 90), (0.78, 1.0, "skin", 0)]
rng = np.random.default_rng(3)
for f0, f1, nm, ang in layers:
    a.add_patch(Rectangle((0.6, y0 + f0 * H), 8.8, (f1 - f0) * H, fc="#EEF2F7" if nm == "skin" else "#FBF3E6",
                          ec="0.5", lw=0.4))
    n = 26 if nm == "skin" else 40
    for _ in range(n):
        xc = rng.uniform(1.0, 9.0); yc = rng.uniform(y0 + (f0 + 0.04) * H, y0 + (f1 - 0.04) * H)
        th = np.radians(ang + rng.normal(0, 12)); L = 0.55 if ang == 0 else 0.35
        a.plot([xc - L * np.cos(th), xc + L * np.cos(th)], [yc - 0.25 * L * np.sin(th), yc + 0.25 * L * np.sin(th)],
               color=NAVY if nm == "skin" else AMBER, lw=0.9, solid_capstyle="round")
a.text(9.55, y0 + 0.11 * H, "skin", fontsize=7, va="center"); a.text(9.55, y0 + 0.5 * H, "core", fontsize=7, va="center")
a.text(9.55, y0 + 0.89 * H, "skin", fontsize=7, va="center")
a.text(5.0, 0.05, r"fibre orientation tensor $\mathbf{a}(z)$", ha="center", fontsize=7.5)
a.plot([5.0, 5.0], [6.5, 5.6], color="0.4", lw=0.6, ls=":")

# ---------------------------------------------------------- (b) RVE + orientation glyphs + MT
b = ax_at(18.8, TOP + 1.0, 16.0, 14.6); b.set_xlim(0, 10); b.set_ylim(0, 10); b.axis("off")
b.add_patch(Ellipse((2.0, 6.6), 3.3, 3.6, fc="#FBF3E6", ec="0.4", lw=0.7))
for _ in range(14):
    xc, yc = rng.uniform(0.8, 3.2), rng.uniform(5.2, 8.0)
    if ((xc - 2.0) / 1.35) ** 2 + ((yc - 6.6) / 1.5) ** 2 < 1:
        th = rng.normal(0, 0.35)
        b.add_patch(Ellipse((xc, yc), 0.8, 0.13, angle=np.degrees(th), fc=NAVY, ec="none"))
b.text(2.0, 4.2, "RVE: E-glass / PPS", ha="center", fontsize=7)
b.text(7.3, 8.6, "Mori–Tanaka", ha="center", fontsize=8, fontweight="bold")
b.text(7.3, 7.6, r"$\mathbf{C}^{\mathrm{UD}},\ \alpha_{\mathrm{L}},\ \alpha_{\mathrm{T}}$", ha="center", fontsize=8.5)
b.text(7.3, 6.55, "orientation averaging", ha="center", fontsize=7.5)
b.text(7.3, 5.5, r"$\langle\mathbf{C}\rangle(\mathbf{a}),\ \langle\boldsymbol{\alpha}\rangle(\mathbf{a})$", ha="center", fontsize=8.5)
b.add_patch(FancyArrowPatch((3.8, 6.6), (4.8, 6.6), arrowstyle="-|>", mutation_scale=8, color="0.3", lw=0.9))
for k, (axx, ayy) in enumerate(((0.80, 0.15), (0.45, 0.45), (0.25, 0.70))):
    xc = 1.6 + k * 3.2
    b.add_patch(Ellipse((xc, 2.0), 2.4 * axx + 0.3, 2.4 * ayy + 0.3, fc="white", ec=NAVY if axx > ayy else AMBER, lw=1.0))
    b.text(xc, 0.25, [r"$a_{xx}$ high", "planar random", r"$a_{yy}$ high"][k], ha="center", fontsize=6.8)
b.text(5.0, 3.6, "orientation-tensor glyphs (per layer)", ha="center", fontsize=6.8, color="0.35")

# ---------------------------------------------------------- (c) eigenstrain profile (data)
c = ax_at(40.4, TOP + 2.7, 11.4, 8.3)
for T0, col, ls, lab in ((180.0, GREY, "--", r"$T_0=180\ ^\circ$C (paper)"), (T0ID, NAVY, "-", rf"$T_{{0,\mathrm{{eff}}}}={T0ID:.0f}\ ^\circ$C")):
    zq, eth = rr.eigenstrain((T0, rr.THETA_PAPER[1], rr.THETA_PAPER[2], 1.0, 1.0))
    c.plot(eth[:, 0] * 1e3, zq, color=col, ls=ls, lw=1.4, label=lab)
c.invert_yaxis(); c.set_ylabel(r"depth $z$ (mm)", fontsize=7.5); c.set_xlabel(r"$\varepsilon^{*}_{xx}$ (MD, $10^{-3}$)", fontsize=7.5); c.tick_params(labelsize=6.5)
c.legend(frameon=False, fontsize=6.2, loc="lower left", bbox_to_anchor=(-0.04, 0.98), handlelength=1.6, ncol=1, labelspacing=0.2)
bg.text(44.6, TOP + 14.6, r"$\boldsymbol{\varepsilon}^{*}(z)=\int_{T_{0,\mathrm{eff}}}^{T_{\mathrm{RT}}}\langle\boldsymbol{\alpha}\rangle(\mathbf{a}(z),T)\,\mathrm{d}T$",
        ha="center", fontsize=8.5)

# ---------------------------------------------------------- (d) FE: part image + plate
im = Image.open(os.path.join(OUTD, "rescue3d_example.png")); W, Hh = im.size
part = im.crop((int(0.43 * W), int(0.08 * Hh), int(0.70 * W), int(0.42 * Hh)))
d = ax_at(54.6, TOP + 5.2, 16.0, 10.2); d.imshow(part); d.axis("off")
bg.text(62.6, TOP + 4.3, "3-D part, C3D10 (CalculiX)", ha="center", fontsize=7.5)
bg.text(62.6, TOP + 2.8, r"$\nabla\cdot\boldsymbol{\sigma}=\mathbf{0},\ \ \boldsymbol{\sigma}=\langle\mathbf{C}\rangle:(\boldsymbol{\varepsilon}-\boldsymbol{\varepsilon}^{*})$",
        ha="center", fontsize=8.0)
bg.text(62.6, TOP + 1.3, "plate: laminate theory / Abaqus", ha="center", fontsize=7, color="0.35")

# ---------------------------------------------------------- (e) measurement + calibration
e1 = ax_at(1.2, BOT + 1.4, 10.2, 12.6); e1.set_xlim(0, 10); e1.set_ylim(0, 10); e1.axis("off")
e1.add_patch(Rectangle((0.5, 2.0), 9.0, 5.0, fc="0.95", ec="0.3", lw=0.7))
e1.add_patch(Rectangle((4.4, 4.4), 1.2, 2.6, fc="white", ec="0.3", lw=0.7))
for yy in (4.9, 5.7, 6.5):
    e1.add_patch(FancyArrowPatch((2.6, yy), (4.25, yy), arrowstyle="-|>", mutation_scale=6, color=AMBER, lw=0.8))
    e1.add_patch(FancyArrowPatch((7.4, yy), (5.75, yy), arrowstyle="-|>", mutation_scale=6, color=AMBER, lw=0.8))
for xg in (1.2, 7.0):
    e1.add_patch(Rectangle((xg, 7.0), 1.8, 0.35, fc=NAVY, ec=NAVY))
e1.text(5.0, 8.2, "incremental hole drilling", ha="center", fontsize=7.5)
e1.text(5.0, 0.9, "released strain at 3 gauges", ha="center", fontsize=7, color="0.35")
e2 = ax_at(14.8, BOT + 3.4, 7.8, 9.2)
f12 = rr.Y_MEAS.reshape(3, 10)
y0p = rr.released(rr.THETA_PAPER).reshape(3, 10); y1p = rr.released(tuple(json.load(open(os.path.join(
    ROOT, "results", "denso_rescue", "results.json")))["R4"]["T0 only"]["theta"])).reshape(3, 10)
dd = np.concatenate([[0], rr.DEPTH]); zz = lambda v: np.concatenate([[0], v])
for k in (2, 0):
    e2.plot(rr.DEPTH, f12[k], "s", ms=2.6, mfc="white", color="k", mew=0.6)
    e2.plot(dd, zz(y0p[k]), "--", color=GREY, lw=0.9); e2.plot(dd, zz(y1p[k]), "-", color=NAVY, lw=1.3)
e2.axhline(0, color="0.85", lw=0.5)
e2.set_xlabel("depth (mm)", fontsize=7); e2.set_ylabel(r"released strain ($\mu\varepsilon$)", fontsize=7); e2.tick_params(labelsize=6)
e2.text(0.97, 0.95, "MD", transform=e2.transAxes, ha="right", va="top", fontsize=6.5)
e2.text(0.97, 0.05, "TD", transform=e2.transAxes, ha="right", fontsize=6.5)
bg.text(29.0, BOT + 13.6, "L2-SP calibration", ha="center", fontsize=8, fontweight="bold", color=NAVY)
bg.text(29.0, BOT + 11.0, r"$\min_{\boldsymbol{\theta}}\ \frac{\|\mathbf{y}-\mathbf{G}\,\boldsymbol{\varepsilon}^{*}(\boldsymbol{\theta})\|^{2}}{s_y^{2}}"
        r"+\frac{\|\boldsymbol{\theta}-\boldsymbol{\theta}_{0}\|^{2}}{s_\theta^{2}}$", ha="center", fontsize=8.0)
bg.text(29.0, BOT + 8.3, r"$\boldsymbol{\theta}$: $T_{0,\mathrm{eff}}$ or factors $a$", ha="center", fontsize=7.2)
bg.text(29.0, BOT + 6.6, r"$\mathbf{G}$: influence functions", ha="center", fontsize=7.2)
bg.text(29.0, BOT + 3.4, "few data: hole drilling", ha="center", fontsize=6.9, color="0.35")
bg.text(29.0, BOT + 2.1, "or 5–10 surface points", ha="center", fontsize=6.9, color="0.35")

# ---------------------------------------------------------- (f) outputs
tf = json.load(open(os.path.join(ROOT, "abaqus", "denso_plate", "abaqus_results", "greif3d_rescue", "transfer_factor.json")))
st = json.load(open(os.path.join(ROOT, "results", "denso_strength", "strength.json")))
gr = json.load(open(os.path.join(ROOT, "abaqus", "denso_plate", "abaqus_results", "greif3d_rescue", "gate_ranking.json")))
items = [("Residual strain", f"flat plate, measured: RMSE {58} → {16} " + r"$\mu\varepsilon$", NAVY),
         ("Warpage", f"{tf['n']} 3-D parts (Moldflow data): {round(tf['prior (a = 1)']['median'] * 100)} → "
                     f"{round(tf['RESCUE 10 nodes, 2 factors']['median'] * 100)} % with 10 points", TEAL),
         ("Transfer", f"no measurement on a new part: {round(tf['LOO factor, no measurement']['median'] * 100)} %", TEAL),
         ("Strength margin", "residual stress takes 20–38 % (estimate)", AMBER),
         ("Moulding design", f"best gate agrees in {gr['best_agrees']} of {gr['n_parts']} parts", AMBER)]
for k, (t, sub, col) in enumerate(items):
    yy = BOT + 14.6 - k * 2.9
    bg.add_patch(FancyBboxPatch((38.0, yy - 1.0), 1.0, 2.0, boxstyle="round,pad=0.05,rounding_size=0.2", fc=col, ec="none"))
    bg.text(39.8, yy + 0.35, t, fontsize=8.2, fontweight="bold", va="center", color=col)
    bg.text(39.8, yy - 0.75, sub, fontsize=7.2, va="center", color="0.2")
bg.text(70.6, BOT + 0.9, "values: examples / interim, see text", ha="right", fontsize=6.3, color="0.45")

# ---------------------------------------------------------- arrows between panels
mid = TOP + 8.9
arrow((17.3, mid), (18.1, mid)); arrow((35.5, mid), (36.3, mid)); arrow((52.9, mid), (53.7, mid))
arrow((62.6, TOP - 0.1), (62.6, BOT + 18.3), c="0.25")                                       # FE -> outputs (prior)
bg.text(63.2, TOP - 0.6, "prior prediction", fontsize=6.8, color="0.35", ha="left", va="center")
arrow((32.5, BOT + 18.3), (39.0, TOP + 0.1), c=NAVY, lw=1.3, rad=-0.3, ls="--")              # calibration feeds back
bg.text(31.8, TOP - 0.6, r"update $T_{0,\mathrm{eff}}$ / $a$", fontsize=7.2, color=NAVY, ha="right", va="center")
arrow((35.5, BOT + 9.0), (36.3, BOT + 9.0), c=NAVY, lw=1.3)

os.makedirs(OUTD, exist_ok=True)
fig.savefig(os.path.join(OUTD, "paper_framework.png"), dpi=400); fig.savefig(os.path.join(OUTD, "paper_framework.pdf"))
print("wrote paper_framework.png/.pdf")
