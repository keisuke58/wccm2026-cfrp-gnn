"""Thesis figure: Stage-5 design feedback (closing the SHM loop).

Four panels:
  (a) BO convergence — running-best objective vs evaluation (skopt EI / GP-EI);
  (b) baseline vs optimised damage field d_final on a representative WORST-CASE
      fleet defect (two imshow panels, exact FD phase-field forward);
  (c) baseline vs optimised expected log-growth and CVaR (worst-20%) bars;
  (d) objective vs the main design variable (off-axis ply angle), interface
      held at the optimum — showing the optimum.

Reads the cached optimisation summary written by design_feedback.run()
(results/design_feedback/design_feedback_result.json); regenerates it with a
short run if missing.  Thesis style (lmodern usetex) with graceful fallback to
plain matplotlib when LaTeX is unavailable:
  PATH=/home/nishioka/texlive/2025/bin/x86_64-linux:$PATH \
    python paper_figs/plot_design_feedback.py
"""
import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "slides", "figure_sources"))

from thesis_style import (use, PALETTE, clean_ax, field,        # noqa: E402
                          slim_cbar, SEQ_CMAP)
import matplotlib.pyplot as plt                                  # noqa: E402

import design_feedback as df                                     # noqa: E402
from cfrp_phasefield_2d import seed_defect, simulate_growth      # noqa: E402

OUT = os.path.join(HERE, "design_feedback.pdf")
RESULT_JSON = os.path.join(df.CACHE_DIR, "design_feedback_result.json")

C_BASE = PALETTE.get("prior_off", "#BDBDBD")     # grey  — baseline
C_OPT  = PALETTE.get("health", "#2196A6")        # teal  — optimised
C_CVAR = PALETTE.get("dysbiosis", "#D95F02")     # orange — CVaR / worst-case


def _safe_use():
    """thesis_style.use(); fall back to plain mpl if LaTeX is unavailable."""
    try:
        figsize = use(width_frac=1.0, aspect=0.78)
        # probe a tiny render to confirm usetex actually works
        fig = plt.figure(figsize=(1, 1)); fig.text(0.5, 0.5, r"$x$")
        fig.canvas.draw(); plt.close(fig)
        return figsize
    except Exception:
        matplotlib.rcParams.update({"text.usetex": False, "font.size": 9,
                                    "figure.facecolor": "white",
                                    "axes.facecolor": "white",
                                    "savefig.dpi": 300, "savefig.bbox": "tight"})
        return (6.5, 5.1)


def _load_or_run():
    if not os.path.exists(RESULT_JSON):
        print("[fig] no cached result — running a short Stage-5 optimisation ...")
        df.run(n_defects=8, n_init=6, n_iter=14, verbose=False)
    with open(RESULT_JSON) as f:
        return json.load(f)


def _worst_case_defect(design, n=12, seed=0):
    """Pick the fleet defect with the largest baseline log-growth (worst case)."""
    rng = np.random.default_rng(seed)
    prior = df.fleet_defect_prior()
    defects = prior.sample(n, rng)
    logs = []
    for th in defects:
        cfg = df.make_config(df.BASELINE_DESIGN)
        d0 = seed_defect(th[0], th[1], th[2], th[3], cfg)
        logs.append(np.log1p(max(simulate_growth(d0, prior.load, cfg).rel_growth, 0.0)))
    return defects[int(np.argmax(logs))], prior.load


def main():
    res = _load_or_run()
    summ = res.get("summary", {})
    X = np.array(res["X"]); y = np.array(res["y"])
    y_best = np.array(res["y_best_curve"])
    base_design = np.array(summ.get("baseline_design", df.BASELINE_DESIGN))
    opt_design = np.array(summ.get("optimised_design", res["best_design"]))
    load = float(res.get("prior", {}).get("load", 0.11))

    figsize = _safe_use()
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    (axA, axB), (axC, axD) = axes

    # ── (a) BO convergence ──────────────────────────────────────────────────
    evals = np.arange(1, len(y) + 1)
    axA.plot(evals, y, "o", ms=2.5, color=C_BASE, alpha=0.6, label="evaluated")
    axA.plot(evals, y_best, "-", color=C_OPT, lw=1.6, label="best-so-far")
    axA.set_xlabel("FD objective evaluation")
    axA.set_ylabel(r"objective  $\bar{m}+$penalty")
    axA.set_title("(a) Bayesian-optimisation convergence")
    axA.legend(frameon=False, loc="upper right")
    clean_ax(axA)

    # ── (b) baseline vs optimised damage field on worst-case defect ─────────
    defect, wload = _worst_case_defect(base_design, n=12, seed=0)
    cfg_b = df.make_config(base_design)
    cfg_o = df.make_config(opt_design)
    d0_b = seed_defect(defect[0], defect[1], defect[2], defect[3], cfg_b)
    d0_o = seed_defect(defect[0], defect[1], defect[2], defect[3], cfg_o)
    dfin_b = simulate_growth(d0_b, wload, cfg_b).d_final
    dfin_o = simulate_growth(d0_o, wload, cfg_o).d_final

    # split panel B into two imshow sub-axes
    axB.axis("off")
    pos = axB.get_position()
    gap = 0.012
    w = (pos.width - gap) / 2.0
    axB1 = fig.add_axes([pos.x0, pos.y0, w, pos.height])
    axB2 = fig.add_axes([pos.x0 + w + gap, pos.y0, w, pos.height])
    im = field(axB1, dfin_b, cmap=SEQ_CMAP, vmin=0, vmax=1)
    field(axB2, dfin_o, cmap=SEQ_CMAP, vmin=0, vmax=1)
    axB1.set_title(r"baseline $(0/\pm30)_s$", fontsize=8)
    axB2.set_title(f"optimised ${opt_design[0]:.0f}^\\circ$", fontsize=8)
    fig.text(pos.x0 + pos.width / 2, pos.y1 + 0.005,
             "(b) worst-case defect $d_\\mathrm{final}$",
             ha="center", va="bottom", fontsize=9)
    slim_cbar(fig, im, axB2, label=r"$d$", ticks=[0, 0.5, 1])

    # ── (c) expected log-growth + CVaR bars ─────────────────────────────────
    labels = ["mean log-\ngrowth", "CVaR$_{20\\%}$"]
    base_vals = [summ.get("baseline_mean_log_growth"),
                 summ.get("baseline_cvar")]
    opt_vals = [summ.get("optimised_mean_log_growth"),
                summ.get("optimised_cvar")]
    xpos = np.arange(len(labels)); bw = 0.36
    axC.bar(xpos - bw / 2, base_vals, bw, color=C_BASE, label="baseline")
    axC.bar(xpos + bw / 2, opt_vals, bw, color=C_OPT, label="optimised")
    axC.set_xticks(xpos); axC.set_xticklabels(labels)
    axC.set_ylabel(r"$\mathbb{E}[\log(1{+}g)]$")
    rdm = summ.get("pct_reduction_mean_log_growth", 0.0)
    rdc = summ.get("pct_reduction_cvar", 0.0)
    axC.set_title(f"(c) robust growth  ($-${rdm:.0f}\\% mean, $-${rdc:.0f}\\% tail)")
    axC.legend(frameon=False, loc="upper left")
    clean_ax(axC)

    # ── (d) objective vs off-axis angle (interface at optimum) ──────────────
    rng = np.random.default_rng(0)
    defects = df.fleet_defect_prior().sample(8, rng)
    angles = np.linspace(df.OFFAXIS_LO, df.OFFAXIS_HI, 7)
    objs = []
    for a in angles:
        d = [a, opt_design[1], opt_design[2]]
        objs.append(df.evaluate_design(d, defects, load,
                                       n_workers=min(8, os.cpu_count() or 1)
                                       ).mean_log_growth)
    objs = np.array(objs)
    axD.plot(angles, objs, "o-", color=C_OPT, ms=3)
    axD.axvline(opt_design[0], color=C_CVAR, ls="--", lw=1.0,
                label=f"optimum ${opt_design[0]:.0f}^\\circ$")
    axD.axvline(base_design[0], color=C_BASE, ls=":", lw=1.0,
                label=r"baseline $30^\circ$")
    axD.set_xlabel(r"off-axis ply angle $|\theta|$ [deg]")
    axD.set_ylabel(r"mean log-growth")
    axD.set_title("(d) objective vs off-axis angle")
    axD.legend(frameon=False, loc="upper right")
    clean_ax(axD)

    fig.suptitle("Stage-5 design feedback: fleet-robust laminate optimisation "
                 "(exact FD phase-field forward)", y=1.0, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT)
    print(f"[fig] wrote {OUT}")


if __name__ == "__main__":
    main()
