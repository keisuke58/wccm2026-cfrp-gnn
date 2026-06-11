"""Thesis figure: FNO surrogate for the Stage-3 CFRP phase-field growth model.

Four panels:
  (a) FD vs surrogate d_final side-by-side on a held-out test case;
  (b) rel_growth scatter FD vs surrogate (log1p axes, super-critical tail);
  (c) P(grow) vs load, FD ground truth vs surrogate;
  (d) per-call wall-time bar, FD vs surrogate (the "Nx faster" headline).

Reads the cached dataset + trained checkpoint produced by crack_surrogate.py
(run --generate and --train first).  Thesis style (lmodern usetex) — run with
TeX Live 2025 on PATH:
  PATH=/home/nishioka/texlive/2025/bin/x86_64-linux:$PATH \
    python paper_figs/plot_crack_surrogate.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "slides", "figure_sources"))

from thesis_style import use, PALETTE          # noqa: E402
import matplotlib.pyplot as plt                 # noqa: E402

import crack_surrogate as cs                    # noqa: E402
from cfrp_phasefield_2d import (LaminateConfig, seed_defect,  # noqa: E402
                                simulate_growth)

OUT = os.path.join(HERE, "crack_surrogate.pdf")

C_FD = PALETTE.get("dysbiosis", "#D95F02")      # orange  — FD ground truth
C_SU = PALETTE.get("health", "#2196A6")         # teal    — surrogate


def main():
    data = cs.load_data()
    nx, ny = int(data["nx"]), int(data["ny"])
    model, ckpt = cs.load_surrogate()

    # held-out split + metrics (same seed as default training)
    split, _, te_idx = cs._prepare(data, test_frac=0.2, seed=0)
    m = cs.evaluate(model, split)

    # ---- (a) pick a held-out case that actually grew, for a vivid field ----
    grew = np.where(m["true_grow"])[0]
    sel = grew[np.argmax(m["true_rel"][grew])] if len(grew) else 0
    d0_sel = split.raw_test["d0"][sel]
    load_sel = float(split.raw_test["load"][sel])
    fd_field = m["true_field"][sel]
    su_field = m["pred_field"][sel]

    use(width_frac=1.0, aspect=0.78)
    fig = plt.figure()
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0],
                          hspace=0.55, wspace=0.45)

    # panel (a): two stacked imshow (FD top, surrogate bottom) in col 0
    axA1 = fig.add_subplot(gs[0, 0])
    axA2 = fig.add_subplot(gs[1, 0])
    for ax, fld, lab in ((axA1, fd_field, r"FD $d_{\mathrm{final}}$"),
                         (axA2, su_field, r"surrogate $d_{\mathrm{final}}$")):
        im = ax.imshow(fld, origin="lower", cmap="inferno", vmin=0, vmax=1,
                       aspect="auto")
        ax.set_title(lab, pad=2)
        ax.set_xticks([]); ax.set_yticks([])
    axA1.text(-0.05, 1.18, r"(a) load $=%.3f$" % load_sel,
              transform=axA1.transAxes, fontweight="bold", va="bottom")
    cax = fig.add_axes([axA2.get_position().x0,
                        axA2.get_position().y0 - 0.045,
                        axA2.get_position().width, 0.018])
    fig.colorbar(im, cax=cax, orientation="horizontal", label=r"$d$")

    # panel (b): rel_growth scatter FD vs surrogate (log1p)
    axB = fig.add_subplot(gs[0, 1])
    tr = np.log1p(np.clip(m["true_rel"], 0, None))
    pr = np.log1p(np.clip(m["pred_rel"], 0, None))
    lim = [0, max(tr.max(), pr.max()) * 1.05 + 1e-3]
    axB.plot(lim, lim, "k--", lw=0.7, zorder=0)
    axB.scatter(tr, pr, s=8, c=C_SU, alpha=0.7, edgecolors="none")
    axB.set_xlim(lim); axB.set_ylim(lim)
    axB.set_xlabel(r"FD $\log(1+\mathrm{rel})$")
    axB.set_ylabel(r"surrogate $\log(1+\mathrm{rel})$")
    axB.set_title(r"(b) growth, $\rho=%.2f$" % m["rel_growth_logcorr"], pad=3)

    # panel (c): P(grow) vs load — FD fraction (binned) vs surrogate mean
    axC = fig.add_subplot(gs[0, 2])
    loads_te = split.raw_test["load"]
    edges = np.linspace(cs.LOAD_LO, cs.LOAD_HI, 7)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    fd_p, su_p = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (loads_te >= lo) & (loads_te < hi)
        if mask.sum() == 0:
            fd_p.append(np.nan); su_p.append(np.nan); continue
        fd_p.append(m["true_grow"][mask].mean())
        su_p.append(m["p_grow"][mask].mean())
    axC.plot(ctr, fd_p, "o-", color=C_FD, label="FD", ms=4)
    axC.plot(ctr, su_p, "s--", color=C_SU, label="surrogate", ms=4)
    axC.set_xlabel(r"load $\varepsilon_{yy}$")
    axC.set_ylabel(r"$P(\mathrm{grow})$")
    axC.set_ylim(-0.05, 1.05)
    axC.set_title(r"(c) calibration", pad=3)
    axC.legend(frameon=False, loc="lower right")

    # panel (d): speed bar
    axD = fig.add_subplot(gs[1, 1:])
    bench = cs.benchmark_speed(n=16, nx=nx, ny=ny, model=model)
    fd_ms = bench["fd_per_call_s"] * 1e3
    su_ms = bench["surr_per_call_s"] * 1e3
    bars = axD.bar([0, 1], [fd_ms, su_ms], color=[C_FD, C_SU], width=0.6)
    axD.set_yscale("log")
    axD.set_xticks([0, 1])
    axD.set_xticklabels(["FD\nsimulate\\_growth", "FNO\nsurrogate"])
    axD.set_ylabel(r"per-call time [ms]")
    spd = bench["speedup_serial"]
    axD.set_title(r"(d) speed: $%.0f\times$ faster" % spd, pad=3)
    for b, v in zip(bars, (fd_ms, su_ms)):
        axD.text(b.get_x() + b.get_width() / 2, v * 1.25,
                 r"%.1f ms" % v if v >= 1 else r"%.2f ms" % v,
                 ha="center", va="bottom", fontsize=8)

    fig.savefig(OUT)
    print(f"[fig] field MSE={m['field_mse']:.2e} relL2={m['field_rel_l2']:.3f} "
          f"acc={m['grow_acc']:.3f} brier={m['brier']:.3f} "
          f"speedup={spd:.0f}x  →  {OUT}")


if __name__ == "__main__":
    main()
