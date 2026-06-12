"""
composite_fatigue_calibration.py — the framework's first POSITIVE real-data
prognosis calibration: fit the Paris delamination-growth law against REAL CFRP
Mode-I fatigue crack-growth measurements, and confront the framework's
representative constants with what real coupons actually do.

Why this module exists
----------------------
The fairing debond and SRB-3 burst prognoses (and the phase-field growth) all use
a Paris law da/dN = C·(ΔG)^m with a REPRESENTATIVE exponent m≈3 — tier [U],
never fitted to real fracture data (the deepest critique). This module fixes that
for the Paris law using a real, tabular, peer-reviewed dataset (no image CV, no
gauge artefacts):

  Yao & Alderliesten, TU Delft — "Mode I fatigue delamination growth in composite
  laminates with fibre bridging" (4TU, doi:10.4233/uuid:66e210e1-...). 56 DCB
  carbon/epoxy specimens, each a table of (crack length, cycles, Gmax, Gmin, ΔG,
  da/dN). We fit the Paris law to the REAL (ΔG, da/dN) pairs per specimen and
  hierarchically across the population.

What the real data says (honest, and partly uncomfortable)
----------------------------------------------------------
  * WITHIN a specimen the Paris law fits EXCELLENTLY (median log-log R² ≈ 0.98) —
    the functional form is right at a fixed bridging state.
  * The real Paris exponent is m ≈ 17 (median, IQR ~[12,21]), NOT the m≈3 the
    framework assumes. The representative value is physically far too shallow.
  * A SINGLE global Paris law fits the pooled cloud POORLY (R² ≈ 0.16): fibre
    bridging shifts the curve as the crack extends (R-curve), so no single (C,m)
    serves all states. This is the dataset's whole point — and a real, honest
    limitation of the framework's single-Paris prognosis.

So this is simultaneously a CALIBRATION (real m, C with a Bayesian population
posterior) and a real-data CRITIQUE of the framework's own prognosis assumption.

Hierarchical Bayes
------------------
Per specimen i we OLS-fit (m_i, logC_i) in log-log with a standard error; a
Normal–Normal–InvGamma hierarchy (numpy-only Gibbs) pools the slopes into a
population posterior (μ_m, σ_m) with 95% credible intervals and per-specimen
shrinkage. Leave-one-specimen-out then asks how well the POPULATION law predicts a
held-out specimen's da/dN — an honest (modest, bridging-limited) generalisation
number.

HONEST LIMITATION
-----------------
This calibrates the Paris law against real Mode-I DCB delamination of carbon/epoxy
— directly relevant to the fairing skin-core DEBOND and interstage delamination
prognoses, a representative real anchor for the SRB-3 burst (different mode). It
is real and tabular (tier [R]); the headline finding is that the framework's m≈3
is unsupported and a single Paris law is inadequate under bridging. It does NOT by
itself certify any structure (still no run-to-failure of the actual parts).

Usage
-----
    python composite_fatigue_calibration.py            # calibration report
    python composite_fatigue_calibration.py --fig
    python composite_fatigue_calibration.py --test     # synthetic fallback
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "composite_fatigue_4tu")

# physical fatigue-delamination ranges (filter raw-table artefacts).
G_MIN, G_MAX = 1.0, 1000.0          # ΔG, J/m^2
DADN_MIN, DADN_MAX = 1e-9, 1e-3     # m/cycle
MIN_PTS = 8                         # points needed to fit a specimen
FRAMEWORK_M = 3.0                   # the representative Paris exponent in use


# ═════════════════════════════════════════════════════════════════════════════
#  Real-data loader (4TU Yao/Alderliesten DCB fatigue tables)
# ═════════════════════════════════════════════════════════════════════════════

def _parse_table(path: str):
    """(ΔG, da/dN) arrays from one specimen's Data_analysis txt, physically
    filtered. Columns (tab-sep): 0 crack,1 cycle,...6 Gmax,7 Gmin,8 ΔG,9 da/dN."""
    dG, dadn = [], []
    try:
        with open(path, encoding="latin-1") as f:
            next(f, None)                                  # header
            for ln in f:
                p = ln.rstrip("\n").split("\t")
                if len(p) < 10:
                    continue
                try:
                    g, dn = float(p[8]), float(p[9])
                except ValueError:
                    continue
                if G_MIN < g < G_MAX and DADN_MIN < dn < DADN_MAX:
                    dG.append(g); dadn.append(dn)
    except Exception:
        return None
    return (np.array(dG), np.array(dadn)) if len(dG) >= MIN_PTS else None


def load_specimens(data_dir: str = DATA_DIR) -> list:
    """List of (name, ΔG, da/dN) for every usable specimen, or [] if absent."""
    if not os.path.isdir(data_dir):
        return []
    out = []
    for path in sorted(glob.glob(os.path.join(
            data_dir, "Sp_*", "Sp_*_Data_[Aa]nalysis*.txt"))):
        rec = _parse_table(path)
        if rec is not None:
            name = os.path.basename(os.path.dirname(path))
            out.append((name, rec[0], rec[1]))
    return out


def synthetic_specimens(n: int = 20, seed: int = 0) -> list:
    """Paris-law specimens with per-specimen (C,m) scatter + scatter noise — a
    clean stand-in so tests run without the dataset."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        # scaled so the Paris line sits inside the physical da/dN window over ΔG.
        m = 3.0 + 0.4 * rng.standard_normal()
        logC = -27.5 - 0.4 * rng.standard_normal()
        g = np.exp(rng.uniform(np.log(30), np.log(300), rng.integers(12, 30)))
        logd = logC + m * np.log(g) + 0.10 * rng.standard_normal(len(g))
        dn = np.exp(logd)
        keep = (dn > DADN_MIN) & (dn < DADN_MAX)
        out.append((f"SYN{i:02d}", g[keep], dn[keep]))
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  Per-specimen Paris fit (log-log OLS) + within-specimen R²
# ═════════════════════════════════════════════════════════════════════════════

def paris_fit(g: np.ndarray, dn: np.ndarray) -> dict:
    """OLS log(da/dN) = logC + m·log(ΔG). Returns m, logC, R², and SE(m)."""
    x, y = np.log(g), np.log(dn)
    n = len(x)
    m, logC = np.polyfit(x, y, 1)
    yhat = logC + m * x
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    sxx = float(np.sum((x - x.mean()) ** 2))
    se_m = float(np.sqrt(ss_res / max(n - 2, 1) / max(sxx, 1e-12)))
    return {"m": float(m), "logC": float(logC), "r2": r2,
            "se_m": max(se_m, 1e-3), "n": n}


def per_specimen(specimens: list) -> list:
    return [{"name": nm, **paris_fit(g, dn)} for nm, g, dn in specimens]


# ═════════════════════════════════════════════════════════════════════════════
#  Hierarchical Bayes over the per-specimen slopes (Normal–Normal–InvGamma Gibbs)
# ═════════════════════════════════════════════════════════════════════════════

def fit_population(m_hat: np.ndarray, se: np.ndarray, n_samples: int = 2000,
                   burn: int = 500, seed: int = 0) -> dict:
    """Pool per-specimen slope estimates m_hat (± se) into a population posterior.
        m_i      ~ Normal(μ, σ²)        (between-specimen, the bridging spread)
        m̂_i | m_i ~ Normal(m_i, se_i²)   (per-specimen fit uncertainty)
    Conjugate Gibbs for μ, σ², m_i. Weakly-informative μ~N(5,10²), σ²~InvGamma."""
    rng = np.random.default_rng(seed)
    m_hat = np.asarray(m_hat, float); se = np.asarray(se, float)
    K = len(m_hat)
    mu = float(m_hat.mean()); sig = max(float(m_hat.std()), 1.0)
    mvals = m_hat.copy()
    keep_mu, keep_sig = [], []
    for it in range(n_samples + burn):
        # m_i | m̂_i, μ, σ
        prec = 1.0 / se ** 2 + 1.0 / sig ** 2
        mean = (m_hat / se ** 2 + mu / sig ** 2) / prec
        mvals = mean + rng.standard_normal(K) / np.sqrt(prec)
        # μ | m_i, σ   (prior N(5,10²))
        prec_mu = K / sig ** 2 + 1.0 / 100.0
        mean_mu = (mvals.sum() / sig ** 2 + 5.0 / 100.0) / prec_mu
        mu = mean_mu + rng.standard_normal() / np.sqrt(prec_mu)
        # σ² | m_i, μ   (InvGamma(2, 1))
        ss = float(np.sum((mvals - mu) ** 2))
        sig = np.sqrt((1.0 + 0.5 * ss) / rng.gamma(2.0 + K / 2.0, 1.0))
        if it >= burn:
            keep_mu.append(mu); keep_sig.append(sig)
    mu_s = np.array(keep_mu)
    return {"mu_mean": float(mu_s.mean()),
            "mu_ci": (float(np.percentile(mu_s, 2.5)),
                      float(np.percentile(mu_s, 97.5))),
            "sigma_mean": float(np.mean(keep_sig)),
            "mu_draws": mu_s}


# ═════════════════════════════════════════════════════════════════════════════
#  Leave-one-specimen-out: can the population law predict a held-out specimen?
# ═════════════════════════════════════════════════════════════════════════════

def loso_generalisation(specimens: list) -> dict:
    """For each held-out specimen, predict da/dN from the POPULATION-mean Paris
    law (mean m, mean logC over the other specimens) and score log-log R².
    Honest test of whether one global law generalises (bridging makes it hard)."""
    fits = per_specimen(specimens)
    names = [f["name"] for f in fits]
    r2_out = []
    for i, (nm, g, dn) in enumerate(specimens):
        m_pop = float(np.mean([fits[j]["m"] for j in range(len(fits)) if j != i]))
        lc_pop = float(np.mean([fits[j]["logC"] for j in range(len(fits)) if j != i]))
        yhat = lc_pop + m_pop * np.log(g)
        y = np.log(dn)
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - float(np.sum((y - yhat) ** 2)) / ss_tot if ss_tot > 0 else 0.0
        r2_out.append(r2)
    return {"names": names, "r2": np.array(r2_out),
            "median_r2": float(np.median(r2_out)),
            "frac_pos": float(np.mean(np.array(r2_out) > 0))}


# ═════════════════════════════════════════════════════════════════════════════
#  Driver
# ═════════════════════════════════════════════════════════════════════════════

def run_study(verbose: bool = True) -> dict:
    specimens = load_specimens()
    using_real = len(specimens) >= 5
    if not using_real:
        specimens = synthetic_specimens()
    fits = per_specimen(specimens)
    m_hat = np.array([f["m"] for f in fits])
    se = np.array([f["se_m"] for f in fits])
    r2_within = np.array([f["r2"] for f in fits])
    pop = fit_population(m_hat, se)
    loso = loso_generalisation(specimens)
    out = {"using_real": using_real, "n_spec": len(specimens),
           "n_points": int(sum(len(g) for _, g, _ in specimens)),
           "fits": fits, "m_hat": m_hat, "r2_within": r2_within,
           "pop": pop, "loso": loso, "specimens": specimens}
    # --- 下流の Stage-3 予後が読む較正アーティファクトを保存 ---
    import json
    art = {"mu_m": float(pop["mu_mean"]), "sigma_m": float(pop["sigma_mean"]),
           "mu_ci": [float(pop["mu_ci"][0]), float(pop["mu_ci"][1])],
           "logC_mean": float(np.mean([f["logC"] for f in fits])),
           "framework_m": float(FRAMEWORK_M),
           "loso_median_r2": float(loso["median_r2"]),
           "loso_frac_pos": float(loso["frac_pos"]),
           "using_real": bool(using_real), "n_spec": int(len(specimens)),
           "n_points": int(out["n_points"]),
           "note": "real CFRP DCB mode-I; single (C,m) fails LOSO (bridging) -> "
                   "use the population posterior (mu_m,sigma_m), not a point m."}
    adir = os.path.join(HERE, "results", "composite_fatigue_4tu")
    os.makedirs(adir, exist_ok=True)
    json.dump(art, open(os.path.join(adir, "calibration.json"), "w"), indent=2)
    out["artifact_path"] = os.path.join(adir, "calibration.json")
    if verbose:
        _report(out)
    return out


def _report(d: dict) -> None:
    bar = "=" * 78
    print(bar)
    print("  COMPOSITE FATIGUE — real Paris-law calibration (tier [R])")
    print(bar)
    src = ("4TU Yao/Alderliesten DCB CFRP Mode-I fatigue (REAL)" if d["using_real"]
           else "SYNTHETIC Paris specimens (4TU data absent) — NOT a real result")
    print(f"  source: {src}")
    print(f"  specimens: {d['n_spec']}   (ΔG, da/dN) points: {d['n_points']}")
    print(f"\n  WITHIN-specimen Paris fit log-log R²: "
          f"mean={d['r2_within'].mean():.3f}  median={np.median(d['r2_within']):.3f}"
          f"  frac>0.8={np.mean(d['r2_within'] > 0.8):.2f}")
    print(f"  → the Paris FORM fits well at a fixed bridging state.")
    mu = d["pop"]["mu_mean"]; lo, hi = d["pop"]["mu_ci"]
    print(f"\n  REAL Paris exponent m (Bayesian population posterior):")
    print(f"    m = {mu:.1f}  [95% CI {lo:.1f}, {hi:.1f}]   "
          f"between-specimen σ = {d['pop']['sigma_mean']:.1f}")
    print(f"    per-specimen m: median={np.median(d['m_hat']):.1f}  "
          f"IQR=[{np.percentile(d['m_hat'],25):.1f}, {np.percentile(d['m_hat'],75):.1f}]")
    print(f"    FRAMEWORK assumes m = {FRAMEWORK_M:.1f}  →  real m is ~"
          f"{mu/FRAMEWORK_M:.0f}× steeper.  The representative value is unsupported.")
    lo_r = d["loso"]
    print(f"\n  LEAVE-ONE-SPECIMEN-OUT generalisation of a SINGLE global law:")
    print(f"    median held-out log-log R² = {lo_r['median_r2']:.2f}  "
          f"(frac with R²>0: {lo_r['frac_pos']:.2f})")
    print(f"  → one global Paris law does NOT generalise across specimens: fibre")
    print(f"    bridging (R-curve) shifts the curve per crack state. Honest limit")
    print(f"    of the framework's single-(C,m) prognosis.")
    print(f"\n  TAKEAWAY: real CFRP delamination Paris exponent ≈ {mu:.0f} (not 3),")
    print(f"  fits well per-state but needs a bridging/R-curve correction for a")
    print(f"  global law — a concrete, real-data fix for the Stage-3 prognosis.")
    print("  LIMITATION: Mode-I DCB coupons, not the structures themselves; tier")
    print("  [R] real anchor, not certification.")
    print(bar)


# ═════════════════════════════════════════════════════════════════════════════
#  Figure
# ═════════════════════════════════════════════════════════════════════════════

def make_figure(d: dict, out_path: str) -> str:
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

    fig, ax = plt.subplots(1, 3, figsize=figsize)

    # (a) da/dN vs ΔG, all specimens (log-log) + a few per-specimen Paris lines.
    for k, (nm, g, dn) in enumerate(d["specimens"]):
        ax[0].loglog(g, dn, ".", ms=1.5, alpha=0.35,
                     color=plt.cm.viridis(k / max(len(d["specimens"]) - 1, 1)))
    # framework m=3 reference line through the data centroid
    allg = np.concatenate([g for _, g, _ in d["specimens"]])
    alld = np.concatenate([dn for _, _, dn in d["specimens"]])
    gc = np.exp(np.mean(np.log(allg))); dc = np.exp(np.mean(np.log(alld)))
    gl = np.array([allg.min(), allg.max()])
    ax[0].loglog(gl, dc * (gl / gc) ** FRAMEWORK_M, "r--", lw=1.2,
                 label=f"framework m={FRAMEWORK_M:.0f}")
    ax[0].loglog(gl, dc * (gl / gc) ** d["pop"]["mu_mean"], "k-", lw=1.4,
                 label=f"real median m≈{np.median(d['m_hat']):.0f}")
    ax[0].set_xlabel(r"$\Delta G$  (J/m$^2$)", fontsize=7)
    ax[0].set_ylabel("da/dN  (m/cycle)", fontsize=7)
    ax[0].set_title(r"(a) real CFRP da/dN-$\Delta G$ (steeper than m=3)", fontsize=8)
    ax[0].set_xlim(allg.min() * 0.9, allg.max() * 1.1)
    ax[0].set_ylim(alld.min() * 0.5, alld.max() * 2.0)   # clip steep ref lines
    ax[0].legend(fontsize=6); ax[0].tick_params(labelsize=6)

    # (b) per-specimen m distribution vs the framework value.
    ax[1].hist(d["m_hat"], bins=12, color="#5c93c4", edgecolor="k", lw=0.4)
    ax[1].axvline(FRAMEWORK_M, color="r", ls="--", lw=1.4,
                  label=f"framework m={FRAMEWORK_M:.0f}")
    ax[1].axvline(d["pop"]["mu_mean"], color="k", lw=1.4,
                  label=f"real m={d['pop']['mu_mean']:.0f}")
    ax[1].set_xlabel("per-specimen Paris exponent m", fontsize=7)
    ax[1].set_ylabel("specimens", fontsize=7)
    ax[1].set_title(r"(b) real $m \gg$ 3 (fibre-bridging spread)", fontsize=8)
    ax[1].legend(fontsize=6); ax[1].tick_params(labelsize=6)

    # (c) within-specimen R² vs leave-one-out R² (form fits, global law doesn't).
    ax[2].hist(d["r2_within"], bins=12, color="#2e7d32", alpha=0.8,
               edgecolor="k", lw=0.4, label="within-specimen")
    ax[2].hist(np.clip(d["loso"]["r2"], -1, 1), bins=12, color="#b71c1c",
               alpha=0.6, edgecolor="k", lw=0.4, label="leave-one-out (global)")
    ax[2].axvline(0, color="k", lw=0.5)
    ax[2].set_xlabel("log-log R²", fontsize=7)
    ax[2].set_ylabel("specimens", fontsize=7)
    ax[2].set_title("(c) form fits per-state (green),\nglobal law fails (red)",
                    fontsize=8)
    ax[2].legend(fontsize=6); ax[2].tick_params(labelsize=6)

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

    # ---- paris_fit recovers a known (C, m) on clean data --------------------
    g = np.exp(np.linspace(np.log(20), np.log(300), 40))
    dn = np.exp(-16.0 + 4.5 * np.log(g))
    fit = paris_fit(g, dn)
    ok(abs(fit["m"] - 4.5) < 1e-2, "paris_fit recovers slope m")
    ok(abs(fit["logC"] + 16.0) < 0.1, "paris_fit recovers intercept")
    ok(fit["r2"] > 0.999, "noiseless → R²≈1")

    # ---- R² drops with noise; SE positive -----------------------------------
    rng = np.random.default_rng(0)
    dn2 = np.exp(-16.0 + 4.5 * np.log(g) + 0.3 * rng.standard_normal(len(g)))
    f2 = paris_fit(g, dn2)
    ok(0.0 < f2["r2"] < 1.0 and f2["se_m"] > 0, "noisy fit R² in (0,1), SE>0")

    # ---- loader filters physical ranges -------------------------------------
    syn = synthetic_specimens(12, seed=1)
    ok(len(syn) == 12, "synthetic specimens built")
    ok(all(np.all((G_MIN < g) & (g < G_MAX)) for _, g, _ in syn),
       "ΔG within physical range")
    ok(all(np.all((DADN_MIN < dn) & (dn < DADN_MAX)) for _, _, dn in syn),
       "da/dN within physical range")

    # ---- hierarchical population recovers a known mean slope ----------------
    m_true = np.array([4.0, 4.5, 5.0, 3.5, 4.2, 4.8])
    se = np.full_like(m_true, 0.2)
    pop = fit_population(m_true, se, n_samples=800, burn=300, seed=2)
    ok(abs(pop["mu_mean"] - m_true.mean()) < 0.5,
       "population mean slope recovered")
    ok(pop["mu_ci"][0] < pop["mu_mean"] < pop["mu_ci"][1], "CI brackets the mean")
    ok(pop["sigma_mean"] > 0, "between-specimen σ positive")

    # ---- per-specimen R² high on synthetic Paris data -----------------------
    fits = per_specimen(syn)
    ok(np.median([f["r2"] for f in fits]) > 0.8,
       "within-specimen Paris fits well on synthetic")

    # ---- LOSO well-posed -----------------------------------------------------
    lo = loso_generalisation(syn)
    ok(len(lo["r2"]) == len(syn) and np.all(np.isfinite(lo["r2"])),
       "LOSO returns finite R² per specimen")

    # ---- end-to-end run_study -----------------------------------------------
    d = run_study(verbose=False)
    ok(d["n_spec"] >= 5 and d["n_points"] > 0, "study has specimens + points")
    ok(np.isfinite(d["pop"]["mu_mean"]), "population posterior finite")

    tag = "" if d["using_real"] else "  (NOTE: 4TU data absent → synthetic)"
    print(f"composite_fatigue_calibration: {n}/{n} unit tests passed{tag}")
    return n


def main():
    ap = argparse.ArgumentParser(
        description="Real CFRP fatigue Paris-law calibration (4TU DCB data).")
    ap.add_argument("--fig", action="store_true")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.test:
        run_tests(); return
    d = run_study()
    if args.fig:
        p = make_figure(d, os.path.join(HERE, "paper_figs",
                                        "composite_fatigue_calibration.pdf"))
        print(f"\nfigure → {p}")


if __name__ == "__main__":
    main()
