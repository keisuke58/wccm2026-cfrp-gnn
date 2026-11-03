"""denso_paper_figs.py -- Figs. 2-5 of the paper draft (research/PAPER_draft_denso.md) in one style
(Times, greyscale, no figure titles: the captions go in the manuscript). Every curve is read from a
result file of the repository; paper values are those read by eye (data/literature/tsuda2026_jscm52_88.json).

  paper_fig2_stress.png      (a) orientation a_xx, a_yy through the thickness (paper Fig. 8a, read);
                             (b) residual stress of the free plate, this work (results/denso_tsuda2026/compare.json)
  paper_fig3_released.png    released strain vs drilling depth: measured, paper FE, this work (Abaqus,
                             production mesh, grid averaged by overlap areas: denso_hole_zone_overlap.json)
  paper_fig4_calibration.png (a) H1: corrected on depth <= 1.0 mm, predicted deeper; (b) held-out RMSE of H1-H3
                             (basis runs abaqus_results/denso_hole_{xx,yy,zz}_released.json; same fit as
                             validation/denso_tsuda2026_transfer.py, recomputed here)
  paper_fig5_discrepancy.png (a) gauge shift (denso_hole_zone_overlap.json); (b) anneal relaxation factor
                             (results/denso_tsuda2026/anneal_relaxation.json, illustrative constants)
Run:  python3 slides/figure_sources/denso_paper_figs.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, "slides", "figures_paper")
AR = os.path.join(ROOT, "abaqus", "denso_plate", "abaqus_results")
TS = os.path.join(ROOT, "results", "denso_tsuda2026")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "validation"))
try:                                   # repo convention: thesis_style, graceful fallback
    from thesis_style import use
    use(width_frac=1.0, aspect=0.4)
except Exception:
    pass
matplotlib.rcParams.update({"text.usetex": False, "font.family": "serif",
                            "font.serif": ["Times New Roman", "Liberation Serif", "STIXGeneral"],
                            "mathtext.fontset": "stix", "font.size": 9, "axes.linewidth": 0.7,
                            "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
                            "ytick.right": True, "legend.frameon": False, "legend.fontsize": 7.5})
import matplotlib.pyplot as plt  # noqa: E402
import denso_tsuda2026_transfer as tr  # noqa: E402

PAPER = json.load(open(os.path.join(ROOT, "data", "literature", "tsuda2026_jscm52_88.json"), encoding="utf-8"))
F12 = PAPER["fig12_released_strain"]
GREY = "0.5"


def label(ax, t):
    ax.text(-0.16, 1.02, t, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(OUT, name), dpi=300); plt.close(fig)
    print("wrote", name)


def fig2():
    c = json.load(open(os.path.join(TS, "compare.json")))["C3_stress_3mm"]
    d = np.array(c["depth_mm"])
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 3.2), sharey=True)
    ax[0].plot(c["axx"], d, "k-", lw=1.2, label=r"$a_{xx}$ (MD)")
    ax[0].plot(c["ayy"], d, "--", color=GREY, lw=1.2, label=r"$a_{yy}$ (TD)")
    ax[0].set_xlabel("orientation tensor component [$-$]"); ax[0].set_ylabel("depth from the drilled face [mm]")
    ax[0].set_xlim(0, 1); ax[0].legend(loc="lower center")
    ax[1].axvline(0, color="0.8", lw=0.6)
    ax[1].plot(c["total"]["sxx"], d, "k-", lw=1.2, label=r"$\sigma_{xx}$")
    ax[1].plot(c["total"]["syy"], d, "--", color=GREY, lw=1.2, label=r"$\sigma_{yy}$")
    ps = PAPER.get("results_text", {})
    ax[1].plot([-24.3], [0.0], "ks", ms=4.5, mfc="white", label="paper FE, surface")
    ax[1].plot([15.2], [0.0], "s", color=GREY, ms=4.5, mfc="white")
    ax[1].set_xlabel("residual stress [MPa]"); ax[1].legend(loc="center right")
    ax[0].invert_yaxis()
    label(ax[0], "(a)"); label(ax[1], "(b)")
    save(fig, "paper_fig2_stress.png")


def fig3():
    o = json.load(open(os.path.join(AR, "denso_hole_zone_overlap.json")))["ch"]["+0.00"]["grids"]
    dd = F12["depth_mm"]; u = F12["uncertainty_ustrain"]
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ax.axhline(0, color="0.8", lw=0.6)
    for ch, g, col, mk in (("3", "GCH3", "k", "o"), ("1", "GCH1", GREY, "s")):
        first = ch == "3"
        ax.errorbar(dd, F12[f"Ch{ch}_exp"], yerr=u, fmt=mk, color=col, ms=3.5, lw=0.8, capsize=1.5,
                    label="measured (paper, read)" if first else None)
        ax.plot(dd, F12[f"Ch{ch}_sim"], ":", color=col, lw=1.1, label="FE (paper, read)" if first else None)
        ax.plot(dd, o[g]["released"], "-", color=col, lw=1.4, label="FE (this work)" if first else None)
    ax.text(1.95, 300, "MD (Ch. 3)", ha="right", fontsize=8); ax.text(1.95, -290, "TD (Ch. 1)", ha="right", fontsize=8, color=GREY)
    ax.set_xlabel("drilling depth [mm]"); ax.set_ylabel(r"released strain [$\mu\varepsilon$]")
    ax.set_xlim(0, 2.05); ax.set_ylim(-340, 370); ax.legend(loc="center right")
    save(fig, "paper_fig3_released.png")


def fig4():
    H, full, y, yfe, gauge, dd, depth = tr.load(AR, os.path.join(AR, "denso_hole_released.json"), "denso_hole")
    p0 = H.sum(1)
    tr_ = dd <= 1.0 + 1e-9
    a1, _ = tr.fit(H[tr_], y[tr_], True)
    res = json.load(open(os.path.join(TS, "transfer_hole_drilling.json")))["checks"]
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 3.2), gridspec_kw=dict(width_ratios=[1.25, 1]))
    a = ax[0]
    a.axvspan(1.1, 2.05, color="0.93", zorder=0); a.text(1.57, 335, "held out", ha="center", fontsize=7.5)
    a.axhline(0, color="0.8", lw=0.6)
    for g, col, mk in ((2, "k", "o"), (0, GREY, "s")):
        m = gauge == g; d = np.r_[0, dd[m]]; f = lambda v: np.r_[0, v[m]]
        first = g == 2
        a.plot(d, f(y), mk, color=col, ms=3.5, label="measured (paper, read)" if first else None)
        a.plot(d, f(p0), ":", color=col, lw=1.1, label="this work, as is" if first else None)
        a.plot(d, f(H @ a1), "-", color=col, lw=1.4, label="this work, corrected" if first else None)
    a.set_xlabel("drilling depth [mm]"); a.set_ylabel(r"released strain [$\mu\varepsilon$]")
    a.set_xlim(0, 2.05); a.set_ylim(-340, 370); a.legend(loc="center right")
    b = ax[1]
    keys = list(res)
    before = [res[k]["rmse_test_physics"] for k in keys]
    after = [res[k].get("rmse_test_l2sp", res[k].get("rmse_test_common")) for k in keys]
    fe = [res[k]["rmse_test_paper_fe"] for k in keys]
    x = np.arange(3)
    b.bar(x - 0.27, before, 0.26, color="white", ec="k", lw=0.7, label="as is")
    b.bar(x, after, 0.26, color="0.25", ec="k", lw=0.7, label="corrected")
    b.bar(x + 0.27, fe, 0.26, color="0.75", ec="k", lw=0.7, label="paper FE")
    for xi, v in zip(x, after):
        b.text(xi, v + 3, f"{v:.0f}", ha="center", fontsize=7)
    b.set_xticks(x); b.set_xticklabels(["H1\nshallow → deep", "H2\nMD → TD,\n3 factors", "H3\nMD → TD,\ncommon factor"], fontsize=7.5)
    b.set_ylabel(r"held-out RMSE [$\mu\varepsilon$]"); b.legend(loc="upper left")
    b.set_ylim(0, 195); b.tick_params(axis="x", top=False)
    label(a, "(a)"); label(b, "(b)")
    save(fig, "paper_fig4_calibration.png")


def fig5():
    o = json.load(open(os.path.join(AR, "denso_hole_zone_overlap.json")))["ch"]
    sh = sorted(o, key=float); x = [float(k) for k in sh]
    an = json.load(open(os.path.join(TS, "anneal_relaxation.json")))
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 3.0))
    a = ax[0]
    a.axhline(1.0, color="0.8", lw=0.6)
    a.plot(x, [o[k]["ratio_to_exp"]["GCH3"] for k in sh], "k-o", ms=3.5, label="MD (Ch. 3)")
    a.plot(x, [o[k]["ratio_to_exp"]["GCH1"] for k in sh], "--s", color=GREY, ms=3.5, label="TD (Ch. 1)")
    a.set_xlabel("radial shift of the gauge grids [mm]"); a.set_ylabel("peak released strain / measured [$-$]")
    a.set_xlim(-0.45, 0.45); a.set_ylim(0.8, 2.7); a.legend(loc="upper right")
    b = ax[1]
    taus = np.array(an["tau_g_s"])
    for r, ls in zip(an["rates_K_min"], ("-", "--", ":")):
        b.semilogx(taus, an["f"][str(r)], "k" + ls, lw=1.2, label=f"{r:.0f} K/min (assumed)")
    b.axhline(an["a_calibrated"], color=GREY, lw=0.8)
    b.text(1.5, an["a_calibrated"] + 0.02, f"calibrated factor {an['a_calibrated']}", fontsize=7.5, color="0.35")
    b.set_xlabel(r"relaxation time at $T_\mathrm{g}$, $\tau_\mathrm{g}$ [s]")
    b.set_ylabel(r"stress factor $\sigma/\sigma_\mathrm{el}$ at 20 °C [$-$]")
    b.set_ylim(0.35, 1.05); b.legend(loc="lower right")
    label(a, "(a)"); label(b, "(b)")
    save(fig, "paper_fig5_discrepancy.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig2(); fig3(); fig4(); fig5()
