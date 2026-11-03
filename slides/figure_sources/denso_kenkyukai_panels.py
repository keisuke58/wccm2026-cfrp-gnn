"""denso_kenkyukai_panels.py -- figures for the kenkyukai deck that no analysis script writes itself.
  kk_gauge_shift.png : peak released strain / measured against the radial gauge shift, from the fifa
                       overlap-area run (abaqus/denso_plate/abaqus_results/denso_hole_zone_overlap.json,
                       Abaqus 2024, production mesh); measured values are the paper's Fig. 12 read by eye.
Run:  python3 slides/figure_sources/denso_kenkyukai_panels.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, "slides", "figures_paper")
sys.path.insert(0, HERE)
try:                                   # repo convention: thesis_style, graceful fallback
    from thesis_style import use
    use(width_frac=0.6, aspect=0.75)
except Exception:
    pass
matplotlib.rcParams.update({"text.usetex": False, "font.family": "serif",
                            "font.serif": ["Times New Roman", "Liberation Serif", "STIXGeneral"],
                            "mathtext.fontset": "stix", "font.size": 9})
import matplotlib.pyplot as plt  # noqa: E402

d = json.load(open(os.path.join(ROOT, "abaqus", "denso_plate", "abaqus_results", "denso_hole_zone_overlap.json")))
sh = sorted(d["ch"], key=float)
x = [float(k) for k in sh]
fig, ax = plt.subplots(figsize=(4.2, 3.0))
ax.plot(x, [d["ch"][k]["ratio_to_exp"]["GCH3"] for k in sh], "k-o", ms=3.5, label="MD (Ch. 3)")
ax.plot(x, [d["ch"][k]["ratio_to_exp"]["GCH1"] for k in sh], "--s", color="0.45", ms=3.5, label="TD (Ch. 1)")
ax.axhline(1.0, color="0.75", lw=0.8)
ax.text(0.42, 1.03, "measured", ha="right", fontsize=7.5, color="0.4")
ax.set_xlabel("radial shift of the gauge grids [mm]")
ax.set_ylabel("peak released strain / measured [$-$]")
ax.set_xlim(-0.45, 0.45); ax.set_ylim(0.8, 2.7)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, "kk_gauge_shift.png"), dpi=220)
print("wrote", os.path.join(OUT, "kk_gauge_shift.png"))
