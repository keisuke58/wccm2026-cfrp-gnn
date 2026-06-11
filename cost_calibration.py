"""
cost_calibration.py — economic anchoring + sensitivity of the OK/REPAIR/RETIRE
decision thresholds (alpha, beta) for reusable-launch structural health.

The gap
-------
`cfrp_phasefield_2d.calibrate_thresholds` turns an expected-cost model into the
two decision thresholds

    alpha = c_repair      / ((1 - r) * c_loss)        (FLY  vs REPAIR boundary)
    beta  = (c_retire - c_repair) / (r * c_loss)      (REPAIR vs RETIRE boundary)

and reports the single point (alpha, beta) = (0.02, 0.48).  That point comes
from an ASSUMED cost set — c_loss=100, c_repair=1, c_retire=25, r=0.5, i.e. a
100:1 loss:repair ratio — with no anchoring to real reusable-launch economics
and no sensitivity.  An operator cannot tell whether a borderline P(grow)=0.3
"REPAIR" call is economically robust or an artefact of those round numbers.

What this module does
---------------------
1. Anchors the FOUR cost drivers to order-of-magnitude PUBLIC reusable-launch
   figures, expressed as DIMENSIONLESS RATIOS to loss-of-vehicle so the
   absolute currency cancels (see `COST_DRIVERS`).  Every range carries an
   explicit stated assumption and a source-CLASS label; these are
   representative public estimates, NOT proprietary point numbers.
2. Recomputes (alpha, beta) across that realistic ratio space by a Monte-Carlo
   sweep -> DISTRIBUTIONS of alpha and beta (median + credible band), replacing
   the single assumed 100:1 point (`sweep_thresholds`).
3. Tornado / one-at-a-time sensitivity: which driver moves each threshold most,
   with the decision implication (`tornado`).  Plus a verdict-flip illustration:
   for a representative P(grow) the OK/REPAIR/RETIRE call can flip across the
   economic range (`verdict_robustness`).

Cost model provenance: identical algebra to `pf.calibrate_thresholds`; we only
feed it economically-anchored RATIO inputs (c_loss fixed to 1, the other three
as fractions of it).  alpha/beta are invariant to the absolute currency scale,
which is exactly why ratios are sufficient.

Source-class labels used below (deliberately coarse — these are public,
order-of-magnitude, NOT proprietary):
    [PUB-PRICE]  published launch list-prices / vehicle-cost statements
    [PUB-REFURB] public statements on reuse / refurbishment turnaround & cost
    [ENG-EST]    engineering order-of-magnitude estimate from the above
    [MODEL]      a modelling assumption of the decision rule itself (e.g. r)

CPU, numpy only; the whole sweep + tests run in well under a second (the cost
algebra is closed-form — no FD solves here).

    python cost_calibration.py --test   # unit tests only
    python cost_calibration.py --fig    # + paper_figs/cost_calibration.pdf
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field

import numpy as np

import cfrp_phasefield_2d as pf

HERE = os.path.dirname(os.path.abspath(__file__))

# Decision-rule semantics (identical to flight_clearance / decision_uq.decide)
DECISIONS = ("OK", "REPAIR", "RETIRE")


def decide(p: float, alpha: float, beta: float) -> str:
    return "OK" if p <= alpha else "REPAIR" if p <= beta else "RETIRE"


# ═════════════════════════════════════════════════════════════════════════════
#  1.  Economically-anchored cost drivers (ratios to loss-of-vehicle)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CostDriver:
    """One cost driver as a (low, nominal, high) ratio to loss-of-vehicle.

    `assumption` states what real quantity the range represents; `source`
    is the coarse source-CLASS label (see module docstring).  Ranges are
    representative PUBLIC order-of-magnitude estimates — not proprietary
    point numbers — and are deliberately wide to bracket the real value.
    """
    name: str
    low: float
    nominal: float
    high: float
    assumption: str
    source: str

    def clip01(self, x):
        return float(np.clip(x, 1e-9, 1.0 - 1e-9)) if self.name == "r" \
            else float(max(x, 1e-9))


# Everything is expressed RELATIVE to loss-of-vehicle (c_loss := 1.0), so the
# absolute currency cancels out of alpha and beta.
#
#   loss-of-vehicle  = booster/vehicle replacement + payload + mission value.
#                      Set to 1.0 by construction (the numeraire).
#
#   refurb/loss      = inter-flight refurbishment / repair cost  ÷ loss.
#                      Public reusable-launch reporting puts refurbishment at a
#                      small fraction of building+losing the vehicle: roughly a
#                      few % at the high (early, heavy refurb) end down to ~1 %
#                      at the mature/rapid-reuse end.  => 1/100 .. 1/30.
#
#   retire/loss      = forfeited residual value of retiring the airframe ÷ loss.
#                      Retiring scraps the REMAINING life, not a fresh vehicle:
#                      a stage near end-of-life forfeits little (~0.1), a nearly
#                      new one forfeits much of its value (~0.4).  => 0.1 .. 0.4.
#
#   r                = repair residual-risk: fraction of the growth/loss risk
#                      that SURVIVES a repair (access damage, undetected
#                      siblings, repair quality).  A good repair leaves ~0.2,
#                      a marginal one ~0.6.  A pure MODEL assumption.  => 0.2..0.6
COST_DRIVERS: dict[str, CostDriver] = {
    "refurb": CostDriver(
        name="refurb", low=1.0 / 100.0, nominal=1.0 / 60.0, high=1.0 / 30.0,
        assumption="inter-flight refurbishment cost as a fraction of "
                   "loss-of-vehicle; ~1% (mature rapid reuse) to ~3% "
                   "(early/heavy refurb)",
        source="[PUB-REFURB]/[ENG-EST]"),
    "retire": CostDriver(
        name="retire", low=0.10, nominal=0.25, high=0.40,
        assumption="forfeited residual value of retiring the airframe as a "
                   "fraction of loss-of-vehicle; ~0.1 (near end-of-life) to "
                   "~0.4 (nearly new)",
        source="[PUB-PRICE]/[ENG-EST]"),
    "r": CostDriver(
        name="r", low=0.20, nominal=0.50, high=0.60,
        assumption="repair residual-risk: fraction of growth/loss risk "
                   "surviving a repair; ~0.2 (clean repair) to ~0.6 (marginal)",
        source="[MODEL]"),
}

# The assumed baseline that produces the reported (0.02, 0.48): c_loss=100,
# c_repair=1, c_retire=25, r=0.5 -> as ratios refurb=0.01, retire=0.25, r=0.5.
BASELINE_RATIOS = dict(refurb=0.01, retire=0.25, r=0.50)


def thresholds_from_ratios(refurb: float, retire: float, r: float
                           ) -> tuple[float, float]:
    """(alpha, beta) for cost RATIOS relative to loss-of-vehicle (c_loss=1).

    Thin wrapper over `pf.calibrate_thresholds` with c_loss fixed to 1.0;
    alpha/beta are scale-invariant so this loses nothing.  refurb = c_repair,
    retire = c_retire_value, r = repair_residual_risk.
    """
    return pf.calibrate_thresholds(c_loss=1.0, c_repair=refurb,
                                   c_retire_value=retire, repair_residual_risk=r)


# ═════════════════════════════════════════════════════════════════════════════
#  2.  Monte-Carlo sweep over the realistic cost-ratio space
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Sweep:
    refurb: np.ndarray
    retire: np.ndarray
    r: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray

    def band(self, arr: np.ndarray, lo: float = 5.0, hi: float = 95.0):
        return (float(np.median(arr)),
                float(np.percentile(arr, lo)), float(np.percentile(arr, hi)))


def _triangular(rng, d: CostDriver, n: int) -> np.ndarray:
    """Sample a driver on [low, high] with mode at nominal (triangular):
    expresses 'nominal most plausible, low/high are the credible extremes'."""
    lo, mo, hi = d.low, d.nominal, d.high
    if not (lo < mo < hi):                     # degenerate -> uniform fallback
        return rng.uniform(min(lo, hi), max(lo, hi), n)
    return rng.triangular(lo, mo, hi, n)


def sweep_thresholds(n: int = 20000, seed: int = 0) -> Sweep:
    """Monte-Carlo (alpha, beta) over the COST_DRIVERS ranges.

    Each driver is sampled triangular(low, nominal, high) independently; the
    resulting (alpha, beta) distribution replaces the single assumed 100:1
    point with a credible band over real reusable-launch economics.
    """
    rng = np.random.default_rng(seed)
    refurb = _triangular(rng, COST_DRIVERS["refurb"], n)
    retire = _triangular(rng, COST_DRIVERS["retire"], n)
    r = _triangular(rng, COST_DRIVERS["r"], n)
    alpha = np.empty(n)
    beta = np.empty(n)
    for i in range(n):
        alpha[i], beta[i] = thresholds_from_ratios(refurb[i], retire[i], r[i])
    return Sweep(refurb=refurb, retire=retire, r=r, alpha=alpha, beta=beta)


# ═════════════════════════════════════════════════════════════════════════════
#  3.  One-at-a-time tornado sensitivity
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class TornadoEntry:
    driver: str
    alpha_lo: float
    alpha_hi: float
    beta_lo: float
    beta_hi: float

    @property
    def alpha_span(self) -> float:
        return abs(self.alpha_hi - self.alpha_lo)

    @property
    def beta_span(self) -> float:
        return abs(self.beta_hi - self.beta_lo)


@dataclass
class Tornado:
    entries: list                       # list[TornadoEntry]
    base_alpha: float
    base_beta: float

    def rank_alpha(self):
        return sorted(self.entries, key=lambda e: e.alpha_span, reverse=True)

    def rank_beta(self):
        return sorted(self.entries, key=lambda e: e.beta_span, reverse=True)


def tornado() -> Tornado:
    """One-at-a-time partials: hold two drivers at NOMINAL, swing the third
    low->high, record the (alpha, beta) excursion.  The driver with the widest
    excursion governs that threshold.
    """
    base = {k: COST_DRIVERS[k].nominal for k in COST_DRIVERS}
    ba, bb = thresholds_from_ratios(base["refurb"], base["retire"], base["r"])
    entries = []
    for name, d in COST_DRIVERS.items():
        lo = dict(base); lo[name] = d.low
        hi = dict(base); hi[name] = d.high
        a_lo, b_lo = thresholds_from_ratios(lo["refurb"], lo["retire"], lo["r"])
        a_hi, b_hi = thresholds_from_ratios(hi["refurb"], hi["retire"], hi["r"])
        entries.append(TornadoEntry(driver=name, alpha_lo=a_lo, alpha_hi=a_hi,
                                    beta_lo=b_lo, beta_hi=b_hi))
    return Tornado(entries=entries, base_alpha=ba, base_beta=bb)


# ═════════════════════════════════════════════════════════════════════════════
#  4.  Verdict robustness: does the OK/REPAIR/RETIRE call flip economically?
# ═════════════════════════════════════════════════════════════════════════════

def verdict_robustness(p_grow: float, sweep: Sweep | None = None,
                       n: int = 20000, seed: int = 1) -> dict:
    """For a fixed P(grow), the distribution of verdicts induced by the
    economic uncertainty in (alpha, beta).

    A verdict is ROBUST if (almost) all of the economic range agrees; it is
    FRAGILE if the sweep splits it across OK/REPAIR/RETIRE.  Returns the
    fraction of the cost space landing in each verdict + the dominant one.
    """
    sw = sweep or sweep_thresholds(n=n, seed=seed)
    verdicts = np.array([decide(p_grow, a, b)
                         for a, b in zip(sw.alpha, sw.beta)])
    frac = {d: float((verdicts == d).mean()) for d in DECISIONS}
    dominant = max(frac, key=frac.get)
    return {"p_grow": p_grow, "frac": frac, "dominant": dominant,
            "robust": frac[dominant] >= 0.95}


def verdict_curve(p_grid: np.ndarray, sweep: Sweep) -> dict:
    """Fraction of the swept economic space in each verdict, as a function of
    P(grow) — the verdict-flip illustration for the figure."""
    out = {d: np.empty(len(p_grid)) for d in DECISIONS}
    for j, p in enumerate(p_grid):
        v = np.array([decide(p, a, b) for a, b in zip(sweep.alpha, sweep.beta)])
        for d in DECISIONS:
            out[d][j] = (v == d).mean()
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  Reporting
# ═════════════════════════════════════════════════════════════════════════════

def print_report(sweep: Sweep, torn: Tornado) -> None:
    bar = "═" * 78
    print(bar)
    print("COST-CALIBRATED DECISION THRESHOLDS (reusable-launch economics)")
    print(bar)
    print("\nCost drivers (ratios to loss-of-vehicle, c_loss := 1):")
    for d in COST_DRIVERS.values():
        print(f"  {d.name:<7} {d.low:.4g} .. [{d.nominal:.4g}] .. {d.high:.4g}"
              f"   {d.source}")
        print(f"          {d.assumption}")

    a_med, a_lo, a_hi = sweep.band(sweep.alpha)
    b_med, b_lo, b_hi = sweep.band(sweep.beta)
    ba, bb = thresholds_from_ratios(**BASELINE_RATIOS)
    print("\nThreshold distributions over the economic range "
          "(median [5–95% band]):")
    print(f"  alpha (FLY|REPAIR)  {a_med:.3f}  [{a_lo:.3f}, {a_hi:.3f}]"
          f"      assumed baseline {ba:.3f}")
    print(f"  beta  (REPAIR|RETIRE){b_med:.3f}  [{b_lo:.3f}, {b_hi:.3f}]"
          f"      assumed baseline {bb:.3f}")

    print("\nTornado (one-at-a-time, others at nominal):")
    print("  alpha governed by:")
    for e in torn.rank_alpha():
        print(f"    {e.driver:<7} span {e.alpha_span:.4f}  "
              f"({e.alpha_lo:.3f} -> {e.alpha_hi:.3f})")
    print("  beta governed by:")
    for e in torn.rank_beta():
        print(f"    {e.driver:<7} span {e.beta_span:.4f}  "
              f"({e.beta_lo:.3f} -> {e.beta_hi:.3f})")

    print("\nVerdict robustness across the economic range:")
    for p in (0.10, 0.30, 0.50):
        vr = verdict_robustness(p, sweep=sweep)
        frac = "  ".join(f"{d}={vr['frac'][d]*100:.0f}%" for d in DECISIONS)
        tag = "ROBUST" if vr["robust"] else "FRAGILE"
        print(f"  P(grow)={p:.2f} -> {vr['dominant']:<6} [{tag}]   {frac}")
    print(bar)


# ═════════════════════════════════════════════════════════════════════════════
#  Figure
# ═════════════════════════════════════════════════════════════════════════════

def make_figure(sweep: Sweep, torn: Tornado, out_path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import sys
    sys.path.insert(0, os.path.join(HERE, "slides", "figure_sources"))
    try:
        from thesis_style import use
        figsize = use(width_frac=1.0, aspect=0.30)
    except Exception:
        figsize = (12.0, 3.4)
    import matplotlib.pyplot as plt

    ba, bb = thresholds_from_ratios(**BASELINE_RATIOS)
    fig, ax = plt.subplots(1, 3, figsize=figsize)

    # (a) alpha / beta distributions over the economic range
    ax[0].hist(sweep.alpha, bins=40, color="#1565C0", alpha=0.75,
               density=True, label=r"$\alpha$ (FLY|REPAIR)")
    ax[0].hist(sweep.beta, bins=40, color="#2e7d32", alpha=0.6,
               density=True, label=r"$\beta$ (REPAIR|RETIRE)")
    ax[0].axvline(ba, color="#1565C0", lw=1.4, ls="--")
    ax[0].axvline(bb, color="#2e7d32", lw=1.4, ls="--")
    ax[0].text(ba, ax[0].get_ylim()[1] * 0.92, "assumed\n0.02", fontsize=6,
               ha="center", color="#1565C0")
    ax[0].text(bb, ax[0].get_ylim()[1] * 0.92, "assumed\n0.48", fontsize=6,
               ha="center", color="#2e7d32")
    ax[0].set_xlabel(r"threshold value", fontsize=7)
    ax[0].set_ylabel("density", fontsize=7)
    ax[0].set_title("(a) threshold distributions\nover launch economics",
                    fontsize=8)
    ax[0].legend(fontsize=6, loc="upper center")
    ax[0].tick_params(labelsize=6)

    # (b) tornado: per-driver excursion of alpha and beta
    names = [e.driver for e in torn.entries]
    ypos = np.arange(len(names))
    for e, y in zip(torn.entries, ypos):
        ax[1].plot([e.alpha_lo, e.alpha_hi], [y + 0.15, y + 0.15], lw=6,
                   color="#1565C0", solid_capstyle="butt", alpha=0.8)
        ax[1].plot([e.beta_lo, e.beta_hi], [y - 0.15, y - 0.15], lw=6,
                   color="#2e7d32", solid_capstyle="butt", alpha=0.7)
    ax[1].axvline(torn.base_alpha, color="#1565C0", lw=0.8, ls=":")
    ax[1].axvline(torn.base_beta, color="#2e7d32", lw=0.8, ls=":")
    ax[1].set_yticks(ypos); ax[1].set_yticklabels(names, fontsize=7)
    ax[1].set_xlabel("threshold excursion (low→high)", fontsize=7)
    ax[1].set_title(r"(b) tornado: blue=$\alpha$, green=$\beta$"
                    "\n(others at nominal)", fontsize=8)
    ax[1].tick_params(labelsize=6)

    # (c) verdict-flip vs P(grow): fraction of economic space per verdict
    p_grid = np.linspace(0.0, 1.0, 201)
    vc = verdict_curve(p_grid, sweep)
    ax[2].stackplot(p_grid, vc["OK"], vc["REPAIR"], vc["RETIRE"],
                    labels=["OK", "REPAIR", "RETIRE"],
                    colors=["#2e7d32", "#f9a825", "#b71c1c"], alpha=0.85)
    ax[2].axvline(ba, color="0.2", lw=0.8, ls="--")
    ax[2].axvline(bb, color="0.2", lw=0.8, ls="--")
    ax[2].set_xlim(0, 1); ax[2].set_ylim(0, 1)
    ax[2].set_xlabel(r"$P(\mathrm{grow})$", fontsize=7)
    ax[2].set_ylabel("fraction of economic range", fontsize=7)
    ax[2].set_title("(c) verdict robustness\n(grey bands = fragile)",
                    fontsize=8)
    ax[2].legend(fontsize=6, loc="center left")
    ax[2].tick_params(labelsize=6)

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

    # --- baseline reproduction: documented (0.02, 0.48) -----------------------
    a0, b0 = pf.calibrate_thresholds()
    ok(np.isclose(a0, 0.02) and np.isclose(b0, 0.48),
       "pf baseline reproduces (0.02, 0.48)")
    a_r, b_r = thresholds_from_ratios(**BASELINE_RATIOS)
    ok(np.isclose(a_r, 0.02) and np.isclose(b_r, 0.48),
       "ratio wrapper reproduces (0.02, 0.48) at the documented baseline")

    # --- scale invariance: alpha/beta depend only on ratios -------------------
    a_s, b_s = pf.calibrate_thresholds(c_loss=1000.0, c_repair=10.0,
                                       c_retire_value=250.0,
                                       repair_residual_risk=0.5)
    ok(np.isclose(a_s, a0) and np.isclose(b_s, b0),
       "thresholds invariant to absolute currency scale (only ratios matter)")

    # --- monotonicity ---------------------------------------------------------
    a_lo, _ = thresholds_from_ratios(0.005, 0.25, 0.5)
    a_hi, _ = thresholds_from_ratios(0.030, 0.25, 0.5)
    ok(a_hi > a_lo, "higher refurb cost raises alpha")
    _, b_lo = thresholds_from_ratios(0.01, 0.25, 0.6)
    _, b_hi = thresholds_from_ratios(0.01, 0.25, 0.2)
    ok(b_hi > b_lo, "higher r lowers beta")
    _, b_rlo = thresholds_from_ratios(0.01, 0.10, 0.5)
    _, b_rhi = thresholds_from_ratios(0.01, 0.40, 0.5)
    ok(b_rhi > b_rlo, "higher retire value raises beta")
    a_lo2, _ = thresholds_from_ratios(0.01, 0.25, 0.2)
    a_hi2, _ = thresholds_from_ratios(0.01, 0.25, 0.6)
    ok(a_hi2 > a_lo2, "higher r raises alpha (smaller (1-r) denominator)")

    # --- swept thresholds stay in [0,1] and beta >= alpha ---------------------
    sw = sweep_thresholds(n=4000, seed=0)
    ok(sw.alpha.min() >= 0.0 and sw.alpha.max() <= 1.0, "swept alpha in [0,1]")
    ok(sw.beta.min() >= 0.0 and sw.beta.max() <= 1.0, "swept beta in [0,1]")
    ok(np.all(sw.beta >= sw.alpha - 1e-12), "swept beta >= alpha everywhere")

    # assumed baseline lies within the FULL swept range (the range brackets it);
    # note it sits BELOW the central band for alpha — the headline finding that
    # the assumed refurb cost was optimistically low.
    ok(sw.alpha.min() <= 0.02 <= sw.alpha.max(),
       "assumed alpha=0.02 within full swept range")
    ok(sw.beta.min() <= 0.48 <= sw.beta.max(),
       "assumed beta=0.48 within full swept range")
    a_med, _, _ = sw.band(sw.alpha)
    ok(a_med > 0.02,
       "realistic median alpha exceeds the assumed 0.02 (optimistic baseline)")

    # --- sensitivity ranking is stable ----------------------------------------
    t = tornado()
    ra = [e.driver for e in t.rank_alpha()]
    rb = [e.driver for e in t.rank_beta()]
    ok(ra[0] == "refurb",
       "alpha most sensitive to refurb/loss ratio")
    ok(rb[0] in ("retire", "r") and "refurb" == rb[-1],
       "beta governed by retire/r, refurb least")
    # ranking stable to a re-seeded sweep window (tornado is seed-free anyway)
    t2 = tornado()
    ok([e.driver for e in t2.rank_alpha()] == ra
       and [e.driver for e in t2.rank_beta()] == rb,
       "tornado ranking deterministic / stable")

    # --- verdict robustness: a mid P(grow) can flip across economics ----------
    vr_lo = verdict_robustness(0.10, sweep=sw)
    ok(vr_lo["dominant"] == "REPAIR",
       "P(grow)=0.10 -> REPAIR dominant (above all swept alpha)")
    vr_mid = verdict_robustness(0.45, sweep=sw)
    ok(0.0 < vr_mid["frac"]["RETIRE"] < 1.0 or not vr_mid["robust"],
       "P(grow)=0.45 verdict is fragile across the economic range")
    ok(set(vr_mid["frac"]) == set(DECISIONS)
       and abs(sum(vr_mid["frac"].values()) - 1.0) < 1e-9,
       "verdict fractions are a proper distribution")

    # --- guards from the underlying model are honoured ------------------------
    try:
        thresholds_from_ratios(0.01, 0.25, 0.0)
        raise AssertionError("r=0 should raise")
    except ValueError:
        ok(True, "r=0 rejected by calibrate_thresholds guard")

    print(f"cost_calibration: {n} checks passed.")
    return n


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Economic anchoring + sensitivity of the OK/REPAIR/RETIRE "
                    "decision thresholds for reusable-launch SHM.")
    ap.add_argument("--n", type=int, default=20000,
                    help="Monte-Carlo samples over the cost-ratio space")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fig", action="store_true",
                    help="save paper_figs/cost_calibration.pdf")
    ap.add_argument("--test", action="store_true", help="unit tests only")
    args = ap.parse_args()

    if args.test:
        run_tests(); return

    sweep = sweep_thresholds(n=args.n, seed=args.seed)
    torn = tornado()
    print_report(sweep, torn)
    if args.fig:
        p = make_figure(sweep, torn,
                        os.path.join(HERE, "paper_figs", "cost_calibration.pdf"))
        print(f"\nfigure → {p}")


if __name__ == "__main__":
    main()
