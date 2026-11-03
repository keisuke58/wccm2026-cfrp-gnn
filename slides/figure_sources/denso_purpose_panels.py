"""slides/figure_sources/denso_purpose_panels.py — the four small panels of the one-slide purpose
flow (slides/build_denso_purpose_slide.js). Every curve comes from a file in the repository:

  p1  orientation a_ij through the 3.08 mm plate — Tsuda et al. (2026) Fig. 8a, READ BY EYE
      (data/literature/tsuda2026_jscm52_88.json);
  p2  residual stress through the same plate from this repository's chain with the paper's inputs
      (results/denso_tsuda2026/compare.json, total form), with the paper's surface values (text);
  p3  fine-tune vs source-only vs target-only RMSE against the number of measured plates —
      synthetic preliminary study (results/denso_ml_copaam_gpu/results.json);
  p4  released strain against drilling depth: measured and the paper's FE (Tsuda et al. (2026)
      Fig. 12, READ BY EYE) and this work with the paper's inputs
      (abaqus/denso_plate/abaqus_results/denso_hole_released.json, Abaqus 2024, production mesh, fifa).
Run:  python3 slides/figure_sources/denso_purpose_panels.py   (writes slides/figures_paper/purpose_p*.png)
"""
import json
import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, "slides", "figures_paper")
sys.path.insert(0, HERE)
try:                                       # repo convention: thesis_style, graceful fallback
    from thesis_style import use
    use(width_frac=0.3, aspect=0.65)
    ok = shutil.which("latex") is not None
    if ok and matplotlib.rcParams["text.usetex"]:
        try:
            from matplotlib.texmanager import TexManager
            TexManager().get_text_width_height_descent("lp", 10)
        except Exception:
            ok = False
    if not ok:
        matplotlib.rcParams["text.usetex"] = False
except Exception:
    pass
matplotlib.rcParams.update({"font.family": "serif",
                            "font.serif": ["Times New Roman", "Liberation Serif", "STIXGeneral"],
                            "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9,
                            "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
                            "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                            "lines.linewidth": 1.3})
import matplotlib.pyplot as plt

W, H = 2.85, 1.85            # inches; matches the panel slot on the slide
GREY = "0.55"


def save(fig, name):
    fig.tight_layout(pad=0.25)
    fig.savefig(os.path.join(OUT, name), dpi=300, facecolor="white")
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    paper = json.load(open(os.path.join(ROOT, "data", "literature", "tsuda2026_jscm52_88.json"), encoding="utf-8"))
    # p1 orientation
    f8 = paper["fig8_orientation_3mm"]
    fig, ax = plt.subplots(figsize=(W, H))
    ax.plot(f8["a_xx"], f8["depth_mm"], "k-", label=r"$a_{xx}$ (MD)")
    ax.plot(f8["a_yy"], f8["depth_mm"], "k--", label=r"$a_{yy}$ (TD)")
    ax.plot(f8["a_zz"], f8["depth_mm"], color=GREY, lw=1.0, label=r"$a_{zz}$")
    ax.set_ylim(3.08, 0); ax.set_xlim(0, 1)
    ax.set_xlabel(r"orientation tensor $a_{ij}$ [$-$]"); ax.set_ylabel("depth [mm]")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=1.6,
              columnspacing=0.8, borderaxespad=0.1)
    save(fig, "purpose_p1_orientation.png")
    # p2 residual stress
    cmp_ = json.load(open(os.path.join(ROOT, "results", "denso_tsuda2026", "compare.json")))
    c3 = cmp_["C3_stress_3mm"]; d = c3["depth_mm"]
    st = paper["results_text"]["surface_centre_residual_stress_MPa"]
    fig, ax = plt.subplots(figsize=(W, H))
    ax.axvline(0, color="0.85", lw=0.6)
    ax.plot(c3["total"]["sxx"], d, "k-", label=r"$\sigma_{xx}$")
    ax.plot(c3["total"]["syy"], d, color=GREY, label=r"$\sigma_{yy}$")
    ax.plot([st["sigma_xx"]], [0.0], "kx", ms=6, mew=1.3, clip_on=False, label="paper, surface")
    ax.plot([st["sigma_yy"]], [0.0], "x", color=GREY, ms=6, mew=1.3, clip_on=False)
    ax.set_ylim(3.08, 0)
    ax.set_xlabel("residual stress [MPa]"); ax.set_ylabel("depth [mm]")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=1.4,
              columnspacing=0.8, borderaxespad=0.1)
    save(fig, "purpose_p2_stress.png")
    # p3 transfer learning (synthetic)
    ml = json.load(open(os.path.join(ROOT, "results", "denso_ml_copaam_gpu", "results.json")))
    e2 = ml["exp2"]; ns = sorted(int(k) for k in e2["fine-tune"])
    g = lambda m: [e2[m][str(n)]["mean"] for n in ns]
    fig, ax = plt.subplots(figsize=(W, H))
    ax.plot(ns, g("target-only"), "--", color=GREY, marker="o", ms=3, label="measurement only")
    ax.plot(ns, g("source-only"), ":", color="k", lw=1.0, label="simulation only")
    ax.plot(ns, g("fine-tune"), "k-", marker="s", ms=3, label="fine-tune (proposed)")
    ax.set_xscale("log"); ax.set_ylim(0, 1.7)
    ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns]); ax.minorticks_off()
    ax.set_xlabel("measured plates (5 points each)"); ax.set_ylabel(r"RMSE [$10^{-3}$]")
    ax.legend(frameon=False, loc="upper right", handlelength=1.8)
    save(fig, "purpose_p3_transfer.png")
    # p4 released strain: measured vs the paper's FE vs this work (same inputs)
    f12 = paper["fig12_released_strain"]; dd = f12["depth_mm"]
    mine = json.load(open(os.path.join(ROOT, "abaqus", "denso_plate", "abaqus_results", "denso_hole_released.json")))
    fig, ax = plt.subplots(figsize=(W, H))
    ax.axhline(0, color="0.85", lw=0.6)
    for ch, g, c in (("3", "GCH3", "k"), ("1", "GCH1", GREY)):
        lab = (lambda t: t) if ch == "3" else (lambda t: None)
        ax.plot(dd, f12[f"Ch{ch}_exp"], "-", color=c, marker="o", ms=3, label=lab("measured"))
        ax.plot(dd, f12[f"Ch{ch}_sim"], ":", color=c, marker="s", ms=3, mfc="white", lw=1.0, label=lab("FE (paper)"))
        ax.plot(mine["depth_mm"], mine["released"][g], "--", color=c, marker="^", ms=3.5, label=lab("FE (this work)"))
    ax.text(0.05, 290, "MD (Ch. 3)", fontsize=7.5); ax.text(0.05, -290, "TD (Ch. 1)", fontsize=7.5, color=GREY)
    ax.set_xlabel("drilling depth [mm]"); ax.set_ylabel(r"released strain [$\mu\varepsilon$]")
    ax.set_xlim(0, 2.0); ax.set_ylim(-330, 360)
    ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.5), handlelength=2.0)
    save(fig, "purpose_p4_hole_drilling.png")
    print("wrote purpose_p1..p4 to", OUT)


if __name__ == "__main__":
    main()
