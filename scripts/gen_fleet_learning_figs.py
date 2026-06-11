"""gen_fleet_learning_figs.py — figures for Stage-4 fleet learning (Concept A).

Produces, all seeded/reproducible:
  results/fleet_learning/phi_sharpening.png   (a) φ posterior sharpening vs N
  results/fleet_learning/rv_shrinkage.png     (b) per-vehicle r_v shrinkage
  results/fleet_learning/fleet_leader.png     (c) inspections-needed vs fleet N
  results/fleet_learning/summary.txt          numeric summary (report numbers)
  paper_figs/fleet_learning.pdf               thesis-style headline (a|b|c)

The headline PDF uses the shared thesis matplotlib style (lmodern usetex), so
run it with TeX Live 2025 on PATH:

    PATH=/home/nishioka/texlive/2025/bin/x86_64-linux:$PATH \
        python3 scripts/gen_fleet_learning_figs.py

Without TeX Live the standalone PNGs are still written (usetex off); only the
headline PDF needs LaTeX.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "slides", "figure_sources"))

from fleet_learning import (
    FleetConfig,
    fit_fleet,
    fleet_leader_experiment,
    recovery_study,
    simulate_fleet,
    vehicle_growth_estimate,
)

OUT = os.path.join(ROOT, "results", "fleet_learning")
PAPER = os.path.join(ROOT, "paper_figs")
os.makedirs(OUT, exist_ok=True)
os.makedirs(PAPER, exist_ok=True)

PRIOR_ON = "#1a7431"     # with fleet prior
PRIOR_OFF = "#BDBDBD"    # isolation / without
ACCENT = "#1565C0"
TRUTH = "#D95F02"


# ── compute everything once (seeded) ─────────────────────────────────────────
def compute(cfg: FleetConfig):
    rec = recovery_study(cfg, sizes=[2, 3, 4, 6, 8, 10, 12, 16, 20],
                         n_samples=2500, burn=700)
    fl = fleet_leader_experiment(cfg, tol=0.05, n_samples=1500, burn=500)

    # per-vehicle shrinkage in the LOW-DATA regime: each vehicle has only a
    # few inspections (k_low), so its isolated estimate is noisy and partial
    # pooling visibly pulls it toward the fleet mean.  (With the full F=30
    # inspections each vehicle's own data dominates and there is nothing to
    # shrink — shrinkage is, by construction, a small-data effect.)
    k_low = 3
    fleet = simulate_fleet(cfg, np.random.default_rng(cfg.seed))
    ests = [vehicle_growth_estimate(v, cfg, n_flights=k_low) for v in fleet]
    s_hat = np.array([e[0] for e in ests])
    post = fit_fleet(fleet, cfg, n_flights_per_vehicle=k_low,
                     n_samples=2500, burn=700,
                     rng=np.random.default_rng(cfg.seed))
    r_raw = np.exp(np.clip(s_hat, -50.0, 10.0))
    r_post = post.r_v_mean()
    r_true = np.array([v.r_true for v in fleet])
    fleet_mean_r = float(np.exp(post.mu_phi_mean))
    return rec, fl, r_raw, r_post, r_true, fleet_mean_r, post, k_low


# ── standalone PNG panels (no usetex needed) ─────────────────────────────────
def panel_phi(rec, path):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.axhline(rec.mu_phi_true, color=TRUTH, ls="--", lw=1.2,
               label=r"true $\mu_\phi$")
    ax.errorbar(rec.fleet_sizes, rec.mu_phi_mean, yerr=rec.mu_phi_sd,
                marker="o", ms=4, color=ACCENT, capsize=2,
                label=r"posterior $E[\mu_\phi]\pm$SD")
    ax.set_xlabel("cumulative vehicles in fleet  $N$")
    ax.set_ylabel(r"global growth-rate mean  $\mu_\phi=E[\log r_v]$")
    ax.set_title("(a) fleet prior sharpens as flights accumulate")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def panel_shrinkage(r_raw, r_post, r_true, fleet_mean_r, path):
    order = np.argsort(r_true)
    x = np.arange(len(r_true))
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.axhline(fleet_mean_r, color="#777777", ls=":", lw=1.0,
               label=r"fleet mean $E[r_v]$")
    for xi, ir in zip(x, order):
        ax.plot([xi, xi], [r_raw[ir], r_post[ir]], color="#cccccc", lw=0.8,
                zorder=1)
    ax.scatter(x, r_raw[order], s=22, color=PRIOR_OFF, zorder=2,
               label="isolated estimate $\\hat r_v$")
    ax.scatter(x, r_post[order], s=22, color=PRIOR_ON, zorder=3,
               label="hierarchical posterior mean")
    ax.scatter(x, r_true[order], s=14, marker="x", color=TRUTH, zorder=4,
               label="truth $r_v$")
    ax.set_xlabel("vehicle (sorted by true $r_v$)")
    ax.set_ylabel("per-vehicle growth rate $r_v$")
    ax.set_title("(b) partial pooling shrinks low-data vehicles to the fleet")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def panel_fleet_leader(fl, path):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(fl.fleet_sizes, fl.insp_without_prior, color=PRIOR_OFF, lw=1.6,
            ls="--", marker="s", ms=3, label="isolated (flat prior)")
    ax.plot(fl.fleet_sizes, fl.insp_with_prior, color=PRIOR_ON, lw=1.8,
            marker="o", ms=4, label="with fleet prior")
    ax.fill_between(fl.fleet_sizes, fl.insp_with_prior, fl.insp_without_prior,
                    color=PRIOR_ON, alpha=0.12)
    ax.set_xlabel("established vehicles in fleet  $N$")
    ax.set_ylabel("inspections to a stable clearance")
    ax.set_title("(c) fleet-leader effect: fewer inspections per new vehicle")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


# ── thesis-style headline (usetex) ───────────────────────────────────────────
def headline(rec, fl, r_raw, r_post, r_true, fleet_mean_r, path):
    try:
        from thesis_style import use, PALETTE, clean_ax
    except Exception as e:  # pragma: no cover
        print(f"[warn] thesis_style unavailable ({e}); skipping headline PDF")
        return False
    on = PALETTE["prior_on"]; off = PALETTE["prior_off"]
    fig, axes = plt.subplots(1, 3, figsize=use(width_frac=1.0, aspect=0.34))

    ax = axes[0]
    ax.axhline(rec.mu_phi_true, color="#D95F02", ls="--", lw=1.0)
    ax.errorbar(rec.fleet_sizes, rec.mu_phi_mean, yerr=rec.mu_phi_sd,
                marker="o", ms=2.5, color="#1565C0", capsize=1.5, lw=1.0)
    ax.set_xlabel(r"fleet size $N$")
    ax.set_ylabel(r"$\mu_\phi = E[\log r_v]$")
    ax.set_title(r"(a) $\phi$ posterior sharpens")
    clean_ax(ax)

    ax = axes[1]
    order = np.argsort(r_true); x = np.arange(len(r_true))
    ax.axhline(fleet_mean_r, color="#777777", ls=":", lw=0.8)
    for xi, ir in zip(x, order):
        ax.plot([xi, xi], [r_raw[ir], r_post[ir]], color="#cccccc", lw=0.6)
    ax.scatter(x, r_raw[order], s=10, color=off)
    ax.scatter(x, r_post[order], s=10, color=on)
    ax.scatter(x, r_true[order], s=7, marker="x", color="#D95F02", lw=0.6)
    ax.set_xlabel(r"vehicle")
    ax.set_ylabel(r"growth rate $r_v$")
    ax.set_title(r"(b) shrinkage to fleet")
    clean_ax(ax)

    ax = axes[2]
    ax.plot(fl.fleet_sizes, fl.insp_without_prior, color=off, ls="--",
            marker="s", ms=2.5, lw=1.2, label="isolated")
    ax.plot(fl.fleet_sizes, fl.insp_with_prior, color=on, marker="o", ms=2.5,
            lw=1.4, label="fleet prior")
    ax.fill_between(fl.fleet_sizes, fl.insp_with_prior, fl.insp_without_prior,
                    color=on, alpha=0.12)
    ax.set_xlabel(r"fleet size $N$")
    ax.set_ylabel(r"inspections needed")
    ax.set_title(r"(c) fleet-leader effect")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    clean_ax(ax)

    fig.tight_layout(w_pad=1.2)
    fig.savefig(path)
    plt.close(fig)
    return True


def main():
    cfg = FleetConfig()
    print(f"[fleet-figs] computing (N={cfg.n_vehicles}, F={cfg.n_flights}) ...")
    rec, fl, r_raw, r_post, r_true, fleet_mean_r, post, k_low = compute(cfg)

    panel_phi(rec, os.path.join(OUT, "phi_sharpening.png"))
    panel_shrinkage(r_raw, r_post, r_true, fleet_mean_r,
                    os.path.join(OUT, "rv_shrinkage.png"))
    panel_fleet_leader(fl, os.path.join(OUT, "fleet_leader.png"))

    # numeric summary for the report
    shrink = np.mean(np.abs(r_post - r_true)) / np.mean(np.abs(r_raw - r_true))
    _ = k_low
    iso = float(fl.insp_without_prior[0])
    full = float(fl.insp_with_prior[-1])
    lines = [
        "Stage-4 fleet learning — summary",
        f"φ_true: mu={cfg.mu_phi_true}, sigma={cfg.sigma_phi_true}",
        f"φ recovered @N={rec.fleet_sizes[-1]}: "
        f"E[mu_phi]={rec.mu_phi_mean[-1]:.3f} (sd {rec.mu_phi_sd[-1]:.3f}), "
        f"E[sigma_phi]={rec.sigma_phi_mean[-1]:.3f} (sd {rec.sigma_phi_sd[-1]:.3f})",
        f"mu_phi posterior SD: {rec.mu_phi_sd[0]:.3f} (N={rec.fleet_sizes[0]}) "
        f"-> {rec.mu_phi_sd[-1]:.3f} (N={rec.fleet_sizes[-1]})",
        f"shrinkage (low-data, {k_low} insp/vehicle): "
        f"mean |r_post-r_true| / |r_raw-r_true| = {shrink:.3f} "
        f"(<1 = hierarchy beats isolated)",
        f"fleet-leader: isolated needs {iso:.2f} inspections, "
        f"full fleet prior needs {full:.2f} -> saved {iso-full:.2f}",
    ]
    summary = "\n".join(lines)
    with open(os.path.join(OUT, "summary.txt"), "w") as fh:
        fh.write(summary + "\n")
    print(summary)

    ok = headline(rec, fl, r_raw, r_post, r_true, fleet_mean_r,
                  os.path.join(PAPER, "fleet_learning.pdf"))
    print(f"[fleet-figs] wrote PNGs to {OUT}"
          + ("; headline -> paper_figs/fleet_learning.pdf" if ok else ""))


if __name__ == "__main__":
    main()
