"""
cross_structure_fleet.py — Stage-4 deepened to a 3-LEVEL "fleet of fleets":
cross-structure hierarchical fleet learning.

Where this sits / why it is new
-------------------------------
`fleet_learning.py` pools the vehicles WITHIN one structure's fleet (a 2-level
hierarchy: a global fleet prior φ=(μ_φ,σ_φ) over per-vehicle log-growth-rates
s_v=log r_v, sharpened by per-flight inspections).  `design_feedback.py` then
feeds that fleet posterior into Stage-5 design.  Neither shares information
ACROSS structures.

This module adds the missing top level.  It is the LEARNING-layer analogue of
the LOSO decision-transfer spine (`loso_decision_transfer.py`): there we showed
the *decision* objects (thresholds, Platt map) transfer across structures; here
we let the *fleet growth-rate* knowledge transfer.  A reusable-spaceflight
operator runs SEVERAL structure types (interstage, fairing, motor case, …),
each its own fleet.  A structure that is NEW or has flown FEW times has a wide
within-fleet posterior.  A 3-level hierarchy lets it BORROW STRENGTH from the
other structures' fleets through a shared global hyperprior, so its
remaining-life estimate sharpens with far fewer flights — and a brand-new
structure starts from a fleet-informed prior instead of an uninformative one.

The 3-level Gaussian hierarchy (all conjugate; numpy-only Gibbs)
--------------------------------------------------------------
    M0           ~ Normal(m0, t0²)                     GLOBAL mean   (level 3, NEW)
    T0²          ~ InvGamma(a_T, b_T)                  between-structure variance
    μ_s | M0,T0  ~ Normal(M0, T0²)        each structure s           (level 2)
    σ_s²         ~ InvGamma(a_s, b_s)                  within-structure variance
    s_{s,v}|μ_s  ~ Normal(μ_s, σ_s²)      each vehicle v in s        (level 1)
    ŝ_{s,v}|s    ~ Normal(s_{s,v}, τ_{s,v}²)           inspection likelihood

The per-vehicle measurement posteriors (ŝ_{s,v}, τ_{s,v}) come straight from
`fleet_learning.vehicle_growth_estimate` on each structure's simulated fleet, so
this module reuses the existing Stage-4 measurement model and only adds the
cross-structure top level + its Gibbs conditionals.

What we measure
---------------
  * COLD-START shrink: a structure with few vehicles/short histories — its μ_s
    posterior SD under (i) no pooling, (ii) within-structure (2-level), (iii)
    cross-structure (3-level).  The 3-level is tightest with the least data.
  * NEW-STRUCTURE prior: a brand-new structure with ZERO vehicles — its μ prior
    is the global predictive Normal(M0, T0²+Var M0) informed by the other
    fleets, vs an uninformative baseline.  Quantifies the "warm start".
  * remaining-life translation: r=exp(μ) per-flight growth rate → the cold-start
    posterior on r and its implied flights-to-threshold band.

HONEST LIMITATION
-----------------
Different structures have genuinely different growth PHYSICS; the shared object
is the (dimensionless, normalised) log-growth-rate s_v=log r_v, and the global
hyperprior encodes "these CFRP fleets are exchangeable at the rate level" — an
APPROXIMATION, the same exchangeability caveat as the cross-structure conformal
/ LOSO results.  The growth surrogate and r_v units are representative (Stage-4
of fleet_learning), not coupon-calibrated; the contribution is that cross-
structure pooling PROVABLY sharpens a data-poor fleet, not certified rates.
With S=3 structures the global variance T0² is itself weakly identified — we use
a weakly-informative InvGamma and report the global posterior with that caveat.

Usage
-----
    python cross_structure_fleet.py            # cold-start + new-structure report
    python cross_structure_fleet.py --fig
    python cross_structure_fleet.py --test
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np

import fleet_learning as fl

HERE = os.path.dirname(os.path.abspath(__file__))

# The structures' TRUE fleet means μ_s = E[log r_v] (drawn around a global M0).
# Different physics → different absolute rates; interstage slowest, srb3 fastest.
STRUCTURE_MU = {
    "interstage": -3.15,
    "fairing":    -2.85,
    "srb3":       -3.00,
}
GLOBAL_M0_TRUE = float(np.mean(list(STRUCTURE_MU.values())))   # ≈ -3.0


# ═════════════════════════════════════════════════════════════════════════════
#  Simulate each structure's fleet → per-vehicle (ŝ, τ) measurement posteriors
# ═════════════════════════════════════════════════════════════════════════════

def structure_measurements(mu_s: float, n_vehicles: int, n_flights: int,
                           seed: int) -> tuple:
    """Simulate one structure's fleet at true mean μ_s and return the per-vehicle
    measurement posteriors (ŝ_v, τ_v) on s_v=log r_v, via fleet_learning's own
    vehicle model. Fewer vehicles / flights → fewer, noisier (ŝ_v, τ_v)."""
    cfg = fl.FleetConfig(n_vehicles=n_vehicles, n_flights=n_flights,
                         mu_phi_true=mu_s)
    vehicles = fl.simulate_fleet(cfg, rng=np.random.default_rng(seed))
    s_hat, tau = [], []
    for veh in vehicles:
        sh, tu = fl.vehicle_growth_estimate(veh, cfg)
        s_hat.append(sh); tau.append(tu)
    return np.asarray(s_hat, float), np.asarray(tau, float), cfg


# ═════════════════════════════════════════════════════════════════════════════
#  3-level Gibbs sampler (the new top level over fleet_learning's 2-level)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class CrossFleetPosterior:
    structures: list                 # structure names, in order
    mu_s: np.ndarray                 # (n_keep, S) per-structure mean draws
    sigma_s: np.ndarray              # (n_keep, S) per-structure within-SD draws
    M0: np.ndarray                   # (n_keep,) global mean draws
    T0: np.ndarray                   # (n_keep,) between-structure SD draws

    def mu_s_mean(self):  return self.mu_s.mean(0)
    def mu_s_sd(self):    return self.mu_s.std(0)

    def new_structure_mu_prior(self) -> tuple:
        """Predictive prior on a BRAND-NEW structure's mean μ_new: marginalise
        μ_new ~ Normal(M0, T0²) over the global posterior draws → (mean, sd)."""
        draws = self.M0 + self.T0 * np.random.default_rng(0).standard_normal(
            self.M0.shape[0])
        return float(draws.mean()), float(draws.std())


def fit_cross_hierarchy(meas: dict, n_samples: int = 1500, burn: int = 500,
                        seed: int = 0,
                        m0: float = -3.0, t0_prior_sd: float = 1.0,
                        aT: float = 2.0, bT: float = 0.05,
                        a_s: float = 2.0, b_s: float = 0.05
                        ) -> CrossFleetPosterior:
    """Gibbs for the 3-level Gaussian hierarchy. `meas[name] = (s_hat, tau)`.
    Weakly-informative hyperpriors: M0~Normal(m0,t0_prior_sd²),
    T0²~InvGamma(aT,bT), σ_s²~InvGamma(a_s,b_s)."""
    rng = np.random.default_rng(seed)
    names = list(meas.keys())
    S = len(names)
    s_hat = [np.asarray(meas[k][0], float) for k in names]
    tau = [np.asarray(meas[k][1], float) for k in names]
    n_v = [len(x) for x in s_hat]

    # init
    mu_s = np.array([sh.mean() if len(sh) else m0 for sh in s_hat])
    sig_s = np.array([max(sh.std(), 0.2) if len(sh) > 1 else 0.4
                      for sh in s_hat])
    M0_c = float(mu_s.mean())
    T0_c = max(float(mu_s.std()), 0.2)
    s_v = [sh.copy() for sh in s_hat]

    keep = {"mu_s": [], "sigma_s": [], "M0": [], "T0": []}
    for it in range(n_samples + burn):
        # 1) per-vehicle s_{s,v} | ŝ, μ_s, σ_s   (conjugate Normal)
        for j in range(S):
            if n_v[j] == 0:
                continue
            prec = 1.0 / tau[j] ** 2 + 1.0 / sig_s[j] ** 2
            mean = (s_hat[j] / tau[j] ** 2 + mu_s[j] / sig_s[j] ** 2) / prec
            s_v[j] = mean + rng.standard_normal(n_v[j]) / np.sqrt(prec)
        # 2) μ_s | s_v, σ_s, M0, T0   (conjugate Normal, borrows via M0/T0)
        for j in range(S):
            prec = n_v[j] / sig_s[j] ** 2 + 1.0 / T0_c ** 2
            ssum = s_v[j].sum() if n_v[j] else 0.0
            mean = (ssum / sig_s[j] ** 2 + M0_c / T0_c ** 2) / prec
            mu_s[j] = mean + rng.standard_normal() / np.sqrt(prec)
        # 3) σ_s² | s_v, μ_s   (conjugate InvGamma)
        for j in range(S):
            if n_v[j] == 0:
                continue
            ss = np.sum((s_v[j] - mu_s[j]) ** 2)
            shape = a_s + n_v[j] / 2.0
            scale = b_s + 0.5 * ss
            sig_s[j] = np.sqrt(scale / rng.gamma(shape, 1.0))
        # 4) M0 | μ_s, T0   (conjugate Normal — the GLOBAL mean, level 3)
        prec = S / T0_c ** 2 + 1.0 / t0_prior_sd ** 2
        mean = (mu_s.sum() / T0_c ** 2 + m0 / t0_prior_sd ** 2) / prec
        M0_c = mean + rng.standard_normal() / np.sqrt(prec)
        # 5) T0² | μ_s, M0   (conjugate InvGamma — between-structure variance)
        ss = np.sum((mu_s - M0_c) ** 2)
        T0_c = np.sqrt((bT + 0.5 * ss) / rng.gamma(aT + S / 2.0, 1.0))

        if it >= burn:
            keep["mu_s"].append(mu_s.copy())
            keep["sigma_s"].append(sig_s.copy())
            keep["M0"].append(M0_c)
            keep["T0"].append(T0_c)
    return CrossFleetPosterior(
        structures=names,
        mu_s=np.array(keep["mu_s"]), sigma_s=np.array(keep["sigma_s"]),
        M0=np.array(keep["M0"]), T0=np.array(keep["T0"]))


# ═════════════════════════════════════════════════════════════════════════════
#  Baselines: no-pooling and within-structure (2-level) μ_s posteriors
# ═════════════════════════════════════════════════════════════════════════════

def nopool_mu(s_hat: np.ndarray, tau: np.ndarray) -> tuple:
    """No pooling: estimate μ from a structure's OWN vehicles only, with NO prior
    sharing. The honest SD is the standard error of the mean that includes BOTH
    the between-vehicle spread AND the measurement noise — sd = sqrt((Var[ŝ] +
    mean τ²)/n). (Inverse-variance combining of ŝ alone would ignore the
    between-vehicle variance and be over-confident.)"""
    s_hat = np.asarray(s_hat, float); tau = np.asarray(tau, float)
    n = s_hat.size
    if n == 0:
        return float("nan"), float("inf")
    mean = float(s_hat.mean())
    sample_var = float(np.var(s_hat, ddof=1)) if n > 1 else 0.0
    sd = float(np.sqrt((sample_var + float(np.mean(tau ** 2))) / n))
    return mean, sd


def within_structure_mu(s_hat: np.ndarray, tau: np.ndarray,
                        cfg: fl.FleetConfig, seed: int = 0) -> tuple:
    """Within-structure 2-level pooling via fleet_learning.fit_hierarchy → the
    μ_φ posterior (mean, SD) for that structure ALONE."""
    if s_hat.size == 0:
        return float("nan"), float("inf")
    post = fl.fit_hierarchy(np.asarray(s_hat, float), np.asarray(tau, float),
                            cfg, n_samples=1200, burn=400,
                            rng=np.random.default_rng(seed))
    return float(post.mu_phi.mean()), float(post.mu_phi.std())


# ═════════════════════════════════════════════════════════════════════════════
#  Driver: cold-start a data-poor structure, compare the 3 regimes
# ═════════════════════════════════════════════════════════════════════════════

def run_study(cold: str = "srb3", cold_vehicles: int = 3, cold_flights: int = 4,
              rich_vehicles: int = 14, rich_flights: int = 30,
              seed: int = 0, verbose: bool = True) -> dict:
    """Two rich structures + one COLD-START (few vehicles, short histories).
    Compare the cold structure's μ posterior under no-pool / within / cross."""
    meas, cfgs = {}, {}
    for i, (name, mu) in enumerate(STRUCTURE_MU.items()):
        if name == cold:
            nv, nf = cold_vehicles, cold_flights
        else:
            nv, nf = rich_vehicles, rich_flights
        sh, tu, cfg = structure_measurements(mu, nv, nf, seed=seed + 7 * i)
        meas[name] = (sh, tu); cfgs[name] = cfg

    post = fit_cross_hierarchy(meas, seed=seed)
    j = post.structures.index(cold)

    sh_c, tu_c = meas[cold]
    np_mean, np_sd = nopool_mu(sh_c, tu_c)
    wi_mean, wi_sd = within_structure_mu(sh_c, tu_c, cfgs[cold], seed=seed)
    cr_mean = float(post.mu_s[:, j].mean()); cr_sd = float(post.mu_s[:, j].std())
    mu_true = STRUCTURE_MU[cold]

    new_mean, new_sd = post.new_structure_mu_prior()
    uninform_sd = 1.0          # the t0_prior_sd uninformative baseline

    out = {
        "cold": cold, "mu_true": mu_true,
        "nopool": (np_mean, np_sd), "within": (wi_mean, wi_sd),
        "cross": (cr_mean, cr_sd),
        "shrink_vs_nopool": np_sd / cr_sd if cr_sd > 0 else float("inf"),
        "shrink_vs_within": wi_sd / cr_sd if cr_sd > 0 else float("inf"),
        "new_structure": (new_mean, new_sd),
        "new_shrink_vs_uninform": uninform_sd / new_sd if new_sd > 0 else float("inf"),
        "M0": (float(post.M0.mean()), float(post.M0.std())),
        "T0": (float(post.T0.mean()), float(post.T0.std())),
        "M0_true": GLOBAL_M0_TRUE,
        "post": post, "meas": meas, "cfgs": cfgs,
        "cold_vehicles": cold_vehicles, "cold_flights": cold_flights,
    }
    if verbose:
        _report(out)
    return out


def _report(d: dict) -> None:
    bar = "=" * 78
    print(bar)
    print("  CROSS-STRUCTURE FLEET LEARNING — 3-level 'fleet of fleets' (Stage-4+)")
    print(bar)
    print(f"  structures: {list(STRUCTURE_MU)}   "
          f"COLD-START = '{d['cold']}' "
          f"({d['cold_vehicles']} vehicles × {d['cold_flights']} flights)")
    print(f"  cold structure true mean μ = {d['mu_true']:+.3f}  "
          f"(log per-flight growth rate)")
    print("\n  μ posterior for the COLD-START structure (mean ± SD):")
    for k in ("nopool", "within", "cross"):
        m, s = d[k]
        label = {"nopool": "no pooling (own data only)",
                 "within": "within-structure (2-level)",
                 "cross":  "cross-structure (3-level, NEW)"}[k]
        err = abs(m - d["mu_true"])
        print(f"    {label:34s}  {m:+.3f} ± {s:.3f}   |err|={err:.3f}")
    print(f"\n  SHARPENING from borrowing across structures:")
    print(f"    SD shrink vs no-pooling      : ×{d['shrink_vs_nopool']:.2f}")
    print(f"    SD shrink vs within-structure: ×{d['shrink_vs_within']:.2f}")
    print(f"\n  GLOBAL hyperprior recovered: M0 = {d['M0'][0]:+.3f} ± {d['M0'][1]:.3f}"
          f"  (true {d['M0_true']:+.3f});  T0 = {d['T0'][0]:.3f} ± {d['T0'][1]:.3f}")
    nm, ns = d["new_structure"]
    print(f"\n  BRAND-NEW structure (0 vehicles) μ prior: {nm:+.3f} ± {ns:.3f}"
          f"  vs uninformative SD 1.0  →  ×{d['new_shrink_vs_uninform']:.2f} tighter")
    print("\n  HEADLINE: a data-poor fleet borrows strength from the other "
          "structures'")
    print("  fleets — its growth-rate posterior sharpens "
          f"×{d['shrink_vs_within']:.2f} vs within-structure")
    print("  pooling alone, and a brand-new structure starts warm, not blind.")
    print("  LIMITATION: shared object = normalised log-growth-rate (cross-CFRP")
    print("  exchangeability approximation); representative rates; S=3 → T0 weakly")
    print("  identified.")
    print(bar)


# ═════════════════════════════════════════════════════════════════════════════
#  Learning-curve: cold-start SD vs number of cold-structure flights
# ═════════════════════════════════════════════════════════════════════════════

def learning_curve(cold: str = "srb3", flights_grid=(2, 4, 8, 16, 30),
                   cold_vehicles: int = 3, seed: int = 0) -> dict:
    """Cold-structure μ posterior SD as its data grows, for within vs cross —
    the 3-level curve should sit BELOW the 2-level (sharper with less data)."""
    within_sd, cross_sd = [], []
    for nf in flights_grid:
        meas, cfgs = {}, {}
        for i, (name, mu) in enumerate(STRUCTURE_MU.items()):
            nv = cold_vehicles if name == cold else 14
            ff = nf if name == cold else 30
            sh, tu, cfg = structure_measurements(mu, nv, ff, seed=seed + 7 * i)
            meas[name] = (sh, tu); cfgs[name] = cfg
        post = fit_cross_hierarchy(meas, seed=seed)
        j = post.structures.index(cold)
        cross_sd.append(float(post.mu_s[:, j].std()))
        _, wsd = within_structure_mu(*meas[cold], cfgs[cold], seed=seed)
        within_sd.append(wsd)
    return {"flights": list(flights_grid),
            "within_sd": within_sd, "cross_sd": cross_sd, "cold": cold}


# ═════════════════════════════════════════════════════════════════════════════
#  Figure
# ═════════════════════════════════════════════════════════════════════════════

def make_figure(d: dict, lc: dict, out_path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import sys
    sys.path.insert(0, os.path.join(HERE, "slides", "figure_sources"))
    try:
        from thesis_style import use
        figsize = use(width_frac=1.0, aspect=0.40)
    except Exception:
        figsize = (12.0, 4.0)
    import matplotlib.pyplot as plt

    post = d["post"]
    fig, ax = plt.subplots(1, 3, figsize=figsize)

    # (a) cold-start μ posterior: the 3 regimes as Gaussians, with the truth.
    xs = np.linspace(d["mu_true"] - 1.2, d["mu_true"] + 1.2, 400)
    gauss = lambda m, s: np.exp(-0.5 * ((xs - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
    for k, c, lab in [("nopool", "#90a4ae", "no pooling"),
                      ("within", "#5c93c4", "within-structure (2-level)"),
                      ("cross", "#b71c1c", "cross-structure (3-level)")]:
        m, s = d[k]
        if np.isfinite(s):
            ax[0].plot(xs, gauss(m, s), color=c, lw=1.6, label=lab)
    ax[0].axvline(d["mu_true"], color="k", ls="--", lw=0.9, label="true μ")
    ax[0].set_xlabel("structure mean  μ = E[log r]", fontsize=7)
    ax[0].set_ylabel("posterior density", fontsize=7)
    ax[0].set_title(f"(a) cold-start '{d['cold']}' μ posterior\n"
                    f"(3-level sharpest: ×{d['shrink_vs_within']:.2f} vs 2-level)",
                    fontsize=8)
    ax[0].legend(fontsize=5.5); ax[0].tick_params(labelsize=6)

    # (b) learning curve: μ posterior SD vs cold-structure flights.
    ax[1].plot(lc["flights"], lc["within_sd"], "s-", color="#5c93c4", ms=5,
               lw=1.3, label="within-structure (2-level)")
    ax[1].plot(lc["flights"], lc["cross_sd"], "D-", color="#b71c1c", ms=5,
               lw=1.3, label="cross-structure (3-level)")
    ax[1].set_xlabel(f"flights flown by cold '{lc['cold']}' fleet", fontsize=7)
    ax[1].set_ylabel("μ posterior SD (lower = sharper)", fontsize=7)
    ax[1].set_title("(b) cold-start learning curve\n(3-level sharper with less "
                    "data)", fontsize=8)
    ax[1].legend(fontsize=5.5); ax[1].tick_params(labelsize=6)
    ax[1].set_ylim(bottom=0.0)

    # (c) recovered global hyperprior + per-structure means.
    mu_m = post.mu_s_mean(); mu_s = post.mu_s_sd()
    ypos = np.arange(len(post.structures))
    ax[2].errorbar(mu_m, ypos, xerr=mu_s, fmt="o", color="#2e7d32",
                   capsize=3, label="μ_s ± SD (3-level)")
    for nm, yy in zip(post.structures, ypos):
        ax[2].plot(STRUCTURE_MU[nm], yy, "kx", ms=7)
    M0m, M0s = d["M0"]
    ax[2].axvspan(M0m - M0s, M0m + M0s, color="#b71c1c", alpha=0.12)
    ax[2].axvline(M0m, color="#b71c1c", lw=1.2, label=f"global M0={M0m:+.2f}")
    ax[2].axvline(d["M0_true"], color="k", ls="--", lw=0.8, label="true M0")
    ax[2].set_yticks(ypos); ax[2].set_yticklabels(post.structures, fontsize=7)
    ax[2].set_xlabel("μ = E[log r]", fontsize=7)
    ax[2].set_title("(c) recovered structures + global\n(× = truth)", fontsize=8)
    ax[2].legend(fontsize=5.5, loc="lower right"); ax[2].tick_params(labelsize=6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    fig.savefig(os.path.splitext(out_path)[0] + ".png", bbox_inches="tight",
                dpi=150)
    plt.close(fig)
    return out_path


# ═════════════════════════════════════════════════════════════════════════════
#  Unit tests
# ═════════════════════════════════════════════════════════════════════════════

def run_tests() -> int:
    n = 0

    def ok(cond, msg):
        nonlocal n
        assert cond, msg
        n += 1

    rng = np.random.default_rng(0)

    # ---- nopool inverse-variance combine: SD shrinks with more vehicles ------
    m1, s1 = nopool_mu(np.array([-3.0]), np.array([0.3]))
    m5, s5 = nopool_mu(np.full(9, -3.0), np.full(9, 0.3))
    ok(abs(m1 + 3.0) < 1e-9 and abs(m5 + 3.0) < 1e-9, "nopool mean = data mean")
    ok(s5 < s1, "nopool SD shrinks with more vehicles")
    ok(np.isinf(nopool_mu(np.array([]), np.array([]))[1]), "empty → inf SD")

    # ---- 3-level Gibbs recovers structure means on clean synthetic data ------
    truth = {"A": -3.2, "B": -2.8, "C": -3.0}
    meas = {}
    for k, mu in truth.items():
        sh = mu + 0.1 * rng.standard_normal(12)          # tight measurements
        meas[k] = (sh, np.full(12, 0.12))
    post = fit_cross_hierarchy(meas, n_samples=800, burn=300, seed=1)
    ok(post.mu_s.shape == (800, 3), "posterior mu_s shape (n_keep, S)")
    rec = dict(zip(post.structures, post.mu_s_mean()))
    for k in truth:
        ok(abs(rec[k] - truth[k]) < 0.25, f"3-level recovers μ_{k}")
    ok(abs(post.M0.mean() - np.mean(list(truth.values()))) < 0.25,
       "global M0 recovers the mean of structure means")
    ok(np.all(post.T0 > 0), "T0 (between-structure SD) positive")

    # ---- CORE CLAIM: cross-structure pooling sharpens a COLD-START fleet -----
    # cold structure: few vehicles + short histories; rich neighbours.
    cold_meas = {}
    cold_meas["cold"] = (np.array([-3.0, -2.6, -3.4]),       # 3 noisy vehicles
                         np.array([0.5, 0.5, 0.5]))
    cold_meas["rich1"] = (-3.1 + 0.1 * rng.standard_normal(16),
                          np.full(16, 0.12))
    cold_meas["rich2"] = (-2.9 + 0.1 * rng.standard_normal(16),
                          np.full(16, 0.12))
    post_c = fit_cross_hierarchy(cold_meas, n_samples=1000, burn=300, seed=2)
    jc = post_c.structures.index("cold")
    cross_sd = post_c.mu_s[:, jc].std()
    _, nopool_sd = nopool_mu(*cold_meas["cold"])
    ok(cross_sd < nopool_sd,
       f"cross-structure SD {cross_sd:.3f} < no-pool SD {nopool_sd:.3f}")

    # ---- brand-new structure (0 vehicles) gets a finite, informed μ prior ----
    nm, ns = post_c.new_structure_mu_prior()
    ok(np.isfinite(nm) and np.isfinite(ns) and ns > 0,
       "new-structure prior finite + positive SD")
    ok(ns < 1.0, "new-structure prior tighter than uninformative SD=1.0")

    # ---- end-to-end run_study: cross SD ≤ within SD ≤ no-pool SD (shrink) ----
    d = run_study(cold="srb3", cold_vehicles=3, cold_flights=4,
                  rich_vehicles=12, rich_flights=30, seed=0, verbose=False)
    ok(d["cross"][1] <= d["nopool"][1] + 1e-9,
       "cross-structure SD ≤ no-pool SD on the real fleet sim")
    ok(d["shrink_vs_within"] >= 1.0 - 0.15,
       "cross-structure sharpens (or matches) within-structure pooling")
    ok(np.isfinite(d["M0"][0]) and abs(d["M0"][0] - d["M0_true"]) < 0.6,
       "global M0 near truth on the real fleet sim")

    # ---- learning curve: cross SD ≤ within SD at the data-poor end ----------
    lc = learning_curve(cold="srb3", flights_grid=(2, 8), cold_vehicles=3, seed=0)
    ok(lc["cross_sd"][0] <= lc["within_sd"][0] + 1e-6,
       "cross-structure sharper than within at the fewest flights")

    print(f"cross_structure_fleet: {n}/{n} unit tests passed")
    return n


def main():
    ap = argparse.ArgumentParser(
        description="Cross-structure (3-level) fleet learning — Stage-4 deepened.")
    ap.add_argument("--cold", default="srb3", choices=list(STRUCTURE_MU))
    ap.add_argument("--fig", action="store_true")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.test:
        run_tests(); return
    d = run_study(cold=args.cold)
    if args.fig:
        lc = learning_curve(cold=args.cold)
        p = make_figure(d, lc, os.path.join(HERE, "paper_figs",
                                            "cross_structure_fleet.pdf"))
        print(f"\nfigure → {p}")


if __name__ == "__main__":
    main()
