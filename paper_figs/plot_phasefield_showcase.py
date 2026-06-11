"""Showcase figure: phase-field damage evolution in the CFRP laminate.
Three panels: seeded NDT defect -> sub-critical (interface capture) ->
super-critical growth. Laminate structure annotated (ply angles, interfaces).
Thesis style (lmodern usetex). Runtime ~1 min (3 coarse simulations).
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "slides", "figure_sources"))
from thesis_style import use
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from cfrp_phasefield_2d import (LaminateConfig, build_laminate_maps,
                                seed_defect, simulate_growth)

cfg = LaminateConfig()                      # 8 plies, defaults
_, _, _, _, iface = build_laminate_maps(cfg)
ny, nx = iface.shape

# posterior-mean-like defect near a mid interface
d0 = seed_defect(cx=0.45, cy=0.5, layer=9.0, log2size=1.0, cfg=cfg)

loads = [0.0, 0.085, 0.115]                # seed only / sub-critical / super-critical
titles = [r"(a) seeded defect",
          r"(b) sub-critical",
          r"(c) interface delamination"]
fields = [d0]
for L in loads[1:]:
    fields.append(simulate_growth(d0, L, cfg).d_final)

# white -> warm damage colormap (damage pops on white background)
cmap = LinearSegmentedColormap.from_list(
    "dmg", ["#ffffff", "#fde6c4", "#f99e4c", "#d7301f", "#5c0010"])

fig, axes = plt.subplots(1, 3, figsize=use(width_frac=1.0, aspect=0.30))
extent = [0, cfg.Lx, 0, cfg.Ly]
ply_h  = cfg.Ly / len(cfg.ply_angles_deg)

for ax, d, t, L in zip(axes, fields, titles, loads):
    dd = np.where(d < 0.04, 0.0, d)        # suppress faint background wash
    im = ax.imshow(dd, origin="lower", extent=extent, cmap=cmap,
                   vmin=0, vmax=1, aspect="auto", interpolation="bilinear")
    # interface bands
    ys = np.linspace(0, cfg.Ly, len(cfg.ply_angles_deg) + 1)[1:-1]
    for y in ys:
        ax.axhline(y, color="0.55", lw=0.5, ls=(0, (4, 3)))
    # fiber-direction glyphs on the left margin
    for k, th in enumerate(cfg.ply_angles_deg):
        yc = (k + 0.5) * ply_h
        ax.annotate("", xy=(0.055 + 0.038 * np.cos(np.deg2rad(th)),
                            yc + 0.038 * np.sin(np.deg2rad(th))),
                    xytext=(0.055 - 0.038 * np.cos(np.deg2rad(th)),
                            yc - 0.038 * np.sin(np.deg2rad(th))),
                    arrowprops=dict(arrowstyle="-", lw=0.9, color="0.35"))
    ax.set_title(t + (f", load {L:.3f}" if L > 0 else ""), fontsize=7.5, pad=3)
    ax.set_xticks([]); ax.set_yticks([])

# load arrows on the last panel (transverse tension)
ax = axes[2]
for x in np.linspace(0.15, 0.85, 5):
    ax.annotate("", xy=(x, cfg.Ly * 1.005), xytext=(x, cfg.Ly * 0.93),
                annotation_clip=False,
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#1565C0"))

cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.012)
cb.set_label(r"damage $d$")

out = "paper_figs/phasefield_showcase.pdf"
plt.savefig(out, bbox_inches="tight")
plt.savefig(out.replace(".pdf", ".png"), dpi=220, bbox_inches="tight")
print("saved →", out)
