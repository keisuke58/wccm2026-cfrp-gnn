"""Thesis figure: HONEST split-conformal UQ for the Stage-3 FNO crack-growth
surrogate, and how far the 142x-faster surrogate can be trusted OUT-OF-DISTRIBUTION.

Four panels:
  (a) coverage vs OOD-distance — nominal line + in-dist vs OOD (the marginal
      guarantee breaks under covariate shift);
  (b) mean interval width / point error vs OOD-distance (error explodes while
      the vanilla interval stays flat — why coverage drops);
  (c) predicted rel_growth WITH conformal interval vs FD truth across the
      in->OOD sweep, coloured by OOD distance;
  (d) trust gate: flag rate and coverage with-vs-without deferral to FD.

Reads the cached dataset + checkpoint (crack_surrogate.py) and runs the
conformal experiment (conformal_surrogate.run_experiment); the OOD FD set is
generated/cached on first run. Thesis style (lmodern usetex) — run with
TeX Live on PATH:
  PATH=/home/nishioka/texlive/2025/bin/x86_64-linux:$PATH \
    python paper_figs/plot_conformal_surrogate.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "slides", "figure_sources"))

from thesis_style import use, PALETTE, clean_ax    # noqa: E402
import matplotlib.pyplot as plt                     # noqa: E402

import conformal_surrogate as cf                     # noqa: E402

OUT = os.path.join(HERE, "conformal_surrogate.pdf")

C_IN = PALETTE.get("health", "#2196A6")     # teal   — in-distribution
C_OOD = PALETTE.get("dysbiosis", "#D95F02")  # orange — out-of-distribution
C_NOM = "#444444"                            # nominal reference
C_GATE = PALETTE.get("prior_on", "#1a7431")  # green  — after deferral / gate


def _log1p(x):
    return np.log1p(np.clip(np.asarray(x, float), 0, None))


def main(n_ood=400):
    res = cf.run_experiment(n_ood=n_ood, dist_margin=0.10, verbose=False)
    cm = res["_cm"]; oodp = res["_oodp"]; ind = res["_ind"]
    te_i = res["_te_i"]
    a = res["primary_alpha"]            # 0.10 → 90%
    nom = 1.0 - a

    use(width_frac=1.0, aspect=0.82)
    fig = plt.figure()
    gs = fig.add_gridspec(2, 2, hspace=0.52, wspace=0.42)

    # ── (a) coverage vs OOD distance ──────────────────────────────────────────
    axA = fig.add_subplot(gs[0, 0])
    b = res["binned"]
    axA.axhline(nom, color=C_NOM, ls="--", lw=0.9,
                label=r"nominal %d\%%" % int(nom * 100))
    # in-dist coverage as the leftmost (distance 0) anchor
    cov_in = res["indist_coverage"][a]
    axA.scatter([0.0], [cov_in], s=30, c=C_IN, zorder=5,
                label="in-dist")
    axA.plot(b["centres"], b["coverage"], "o-", color=C_OOD, ms=4,
             label="OOD (binned)")
    axA.set_xlabel(r"OOD distance $\delta$")
    axA.set_ylabel(r"empirical coverage")
    axA.set_ylim(-0.03, 1.03)
    axA.set_title(r"(a) coverage breaks OOD", pad=3)
    axA.legend(frameon=False, loc="lower left", fontsize=7)
    clean_ax(axA)

    # ── (b) interval width (flat) + point error (explodes) vs distance ────────
    axB = fig.add_subplot(gs[0, 1])
    axB.plot(b["centres"], b["width"], "s--", color=C_NOM, ms=4,
             label=r"interval width $2\hat q$")
    axB.plot(b["centres"], b["abs_log_err"], "o-", color=C_OOD, ms=4,
             label=r"$|\Delta\log(1{+}\mathrm{rel})|$")
    axB.set_xlabel(r"OOD distance $\delta$")
    axB.set_ylabel(r"log-space magnitude")
    axB.set_title(r"(b) error grows, width flat", pad=3)
    axB.legend(frameon=False, loc="upper left", fontsize=7)
    clean_ax(axB)

    # ── (c) predicted rel_growth WITH interval vs FD truth, across the sweep ──
    axC = fig.add_subplot(gs[1, 0])
    t = _log1p(oodp["true_rel"])
    p = _log1p(oodp["pred_rel"])
    qh = cm.qhat[a]
    order = np.argsort(oodp["ood_dist"])
    sc = axC.scatter(t, p, c=oodp["ood_dist"], cmap="viridis", s=10,
                     edgecolors="none", zorder=3)
    # conformal band around the y=x reference (interval is ±q̂ on the prediction)
    lim = [0, max(t.max(), p.max()) * 1.05 + 1e-3]
    axC.plot(lim, lim, "k--", lw=0.8, zorder=1)
    axC.fill_between(lim, [x - qh for x in lim], [x + qh for x in lim],
                     color=C_IN, alpha=0.18, zorder=0,
                     label=r"$\pm\hat q$ (%d\%%)" % int(nom * 100))
    axC.set_xlim(lim); axC.set_ylim(lim)
    axC.set_xlabel(r"FD $\log(1+\mathrm{rel})$")
    axC.set_ylabel(r"surrogate $\log(1+\mathrm{rel})$")
    axC.set_title(r"(c) prediction $\pm$ interval", pad=3)
    axC.legend(frameon=False, loc="upper left", fontsize=7)
    cb = fig.colorbar(sc, ax=axC, fraction=0.046, pad=0.03)
    cb.set_label(r"$\delta$", fontsize=8)
    clean_ax(axC)

    # ── (d) trust gate: flag rate + coverage with/without deferral ────────────
    axD = fig.add_subplot(gs[1, 1])
    g = res["trust_gate"][a]
    labels = ["trust\nall", "defer\nflagged"]
    covs = [g["cov_no_defer"], g["cov_with_defer"]]
    bars = axD.bar([0, 1], covs, width=0.55,
                   color=[C_OOD, C_GATE], zorder=3)
    axD.axhline(nom, color=C_NOM, ls="--", lw=0.9, zorder=2)
    axD.set_xticks([0, 1]); axD.set_xticklabels(labels)
    axD.set_ylabel(r"OOD coverage")
    axD.set_ylim(0, 1.05)
    axD.set_title(r"(d) trust gate: %.0f\%% $\to$FD" % (g["flag_rate"] * 100),
                  pad=3)
    for bar, v in zip(bars, covs):
        axD.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                 r"%.0f\%%" % (v * 100), ha="center", va="bottom", fontsize=8)
    axD.text(0.5, nom + 0.015, r"nominal %d\%%" % int(nom * 100),
             ha="center", va="bottom", fontsize=7, color=C_NOM)
    clean_ax(axD)

    fig.savefig(OUT)
    print(f"[fig] in-dist cov(90%)={res['indist_coverage'][a]*100:.1f}%  "
          f"OOD cov={res['ood_coverage_all'][a]*100:.1f}%  "
          f"gate flag={g['flag_rate']*100:.0f}% "
          f"→ cov {g['cov_with_defer']*100:.1f}%  →  {OUT}")


if __name__ == "__main__":
    main()
