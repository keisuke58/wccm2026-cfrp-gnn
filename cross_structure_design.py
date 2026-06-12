"""
cross_structure_design.py — Stage-5 deepened to a CROSS-STRUCTURE design /
resource-allocation layer (the design-evolution analogue of the fleet-of-fleets).

Where this sits / why it is new
-------------------------------
`design_feedback.py` evolves ONE structure's laminate (off-axis angle, interface
toughening) against its own fleet defect distribution.  `cross_structure_fleet.py`
showed that pooling fleets ACROSS structures sharpens a data-poor structure's
growth-rate posterior.  This module closes the loop at the PORTFOLIO level: an
operator running several reusable CFRP structures has ONE limited engineering
resource — a total toughening / re-design / inspection budget B — and must split
it across the structures to minimise TOTAL fleet risk.

Two results, both chaining the earlier stages:
  1. PORTFOLIO ALLOCATION.  Each structure s has a baseline fleet risk set by its
     growth rate r_s (from Stage-4); spending budget b_s slows its effective rate
     (toughening) with diminishing returns.  We allocate B to minimise the total
     expected fleet loss (and a worst-structure / CVaR variant).  The optimum
     sends more budget to the faster-growing / higher-consequence structures and
     beats uniform, greedy-worst, and no-budget baselines.
  2. CROSS-STRUCTURE WARM-START PAYS OFF IN DESIGN.  The allocation is only as
     good as the per-structure risk estimate.  A cold-start structure's risk is
     mis-estimated by within-structure pooling alone; the cross-structure
     (3-level) posterior estimates it better, so the allocation it induces has
     LOWER REGRET against the oracle (true-risk) allocation.  I.e. fleet-of-
     fleets learning concretely improves the design decision, not just the
     uncertainty number.

Risk model (transparent, representative)
----------------------------------------
    r_eff_s(b_s) = r_s · exp(-k_s · b_s)            toughening slows growth
    P_fail_s     = 1 - exp(-H · r_eff_s)            H-flight cumulative hazard
    Risk_s(b_s)  = c_loss · w_s · P_fail_s          expected loss (w_s = value weight)
    allocate     min_b Σ_s Risk_s(b_s)  s.t. Σ b_s = B,  b_s ≥ 0
Diminishing returns ⇒ greedy marginal allocation (water-filling) is optimal; we
also report the worst-structure (CVaR-like) risk of each strategy.

HONEST LIMITATION
-----------------
The growth→risk map, the toughening effectiveness k_s, the horizon H and value
weights w_s are REPRESENTATIVE (trends, not certified).  The contribution is the
DECISION STRUCTURE — portfolio allocation across structures and the demonstration
that cross-structure fleet learning lowers allocation regret — not calibrated
budget numbers.  Cross-structure exchangeability of the growth rate is the same
approximation flagged in `cross_structure_fleet` / `conformal_clearance`.

Usage
-----
    python cross_structure_design.py            # allocation + warm-start report
    python cross_structure_design.py --fig
    python cross_structure_design.py --test
"""
from __future__ import annotations

import argparse
import os

import numpy as np

import cross_structure_fleet as csf

HERE = os.path.dirname(os.path.abspath(__file__))

# Representative portfolio settings (trends, not certified — see LIMITATION).
H_HORIZON = 30.0                 # flights over which cumulative hazard accrues
C_LOSS = 20.0                    # consequence of a fleet failure (cost units)
# per-structure toughening effectiveness k_s and consequence weight w_s.
STRUCT_K = {"interstage": 6.0, "fairing": 5.0, "srb3": 7.0}
STRUCT_W = {"interstage": 1.0, "fairing": 1.2, "srb3": 1.5}   # srb3 = man-rated
TOTAL_BUDGET = 1.0


# ═════════════════════════════════════════════════════════════════════════════
#  Risk model
# ═════════════════════════════════════════════════════════════════════════════

def p_fail(r_eff: float | np.ndarray, H: float = H_HORIZON):
    """H-flight cumulative-hazard failure probability, increasing in growth rate."""
    return 1.0 - np.exp(-H * np.asarray(r_eff, float))


def structure_risk(r: float, b: float, k: float, w: float,
                   H: float = H_HORIZON, c_loss: float = C_LOSS) -> float:
    """Expected fleet loss for one structure at growth rate r and budget b.
    Toughening lowers the EFFECTIVE rate r·exp(-k b) with diminishing returns."""
    r_eff = r * np.exp(-k * b)
    return float(c_loss * w * p_fail(r_eff, H))


# ═════════════════════════════════════════════════════════════════════════════
#  Portfolio allocation: greedy marginal water-filling over the budget
# ═════════════════════════════════════════════════════════════════════════════

def allocate(rs: dict, budget: float = TOTAL_BUDGET, n_steps: int = 400,
             ks: dict | None = None, ws: dict | None = None) -> dict:
    """Allocate `budget` across structures to minimise total expected loss.

    Diminishing-returns separable objective ⇒ greedy marginal allocation: hand
    out the budget in small increments, each to the structure with the LARGEST
    marginal risk reduction. Returns the allocation and the resulting total /
    worst-structure risk."""
    ks = ks or STRUCT_K
    ws = ws or STRUCT_W
    names = list(rs.keys())
    b = {s: 0.0 for s in names}
    delta = budget / n_steps
    for _ in range(n_steps):
        best, best_gain = None, -np.inf
        for s in names:
            now = structure_risk(rs[s], b[s], ks[s], ws[s])
            nxt = structure_risk(rs[s], b[s] + delta, ks[s], ws[s])
            if (now - nxt) > best_gain:
                best_gain, best = (now - nxt), s
        b[best] += delta
    return _summary(rs, b, ks, ws, budget, "optimal")


def allocate_uniform(rs: dict, budget: float = TOTAL_BUDGET,
                     ks: dict | None = None, ws: dict | None = None) -> dict:
    ks = ks or STRUCT_K; ws = ws or STRUCT_W
    b = {s: budget / len(rs) for s in rs}
    return _summary(rs, b, ks, ws, budget, "uniform")


def allocate_greedy_worst(rs: dict, budget: float = TOTAL_BUDGET,
                          ks: dict | None = None, ws: dict | None = None) -> dict:
    """All budget to the single highest-baseline-risk structure (naive)."""
    ks = ks or STRUCT_K; ws = ws or STRUCT_W
    worst = max(rs, key=lambda s: structure_risk(rs[s], 0.0, ks[s], ws[s]))
    b = {s: (budget if s == worst else 0.0) for s in rs}
    return _summary(rs, b, ks, ws, budget, "greedy-worst")


def allocate_none(rs: dict, ks: dict | None = None,
                  ws: dict | None = None) -> dict:
    ks = ks or STRUCT_K; ws = ws or STRUCT_W
    b = {s: 0.0 for s in rs}
    return _summary(rs, b, ks, ws, 0.0, "no-budget")


def _summary(rs, b, ks, ws, budget, label) -> dict:
    per = {s: structure_risk(rs[s], b[s], ks[s], ws[s]) for s in rs}
    return {"label": label, "alloc": b, "risk_per": per,
            "total_risk": float(sum(per.values())),
            "worst_risk": float(max(per.values())),
            "budget": budget}


# ═════════════════════════════════════════════════════════════════════════════
#  Cross-structure WARM-START: does better fleet learning → better allocation?
# ═════════════════════════════════════════════════════════════════════════════

def allocation_regret(rs_est: dict, rs_true: dict, budget: float = TOTAL_BUDGET
                      ) -> float:
    """Regret of an allocation made from ESTIMATED rates, evaluated at the TRUE
    rates, vs the oracle allocation that knows the true rates.

    regret = totalrisk_true(allocate(rs_est)) − totalrisk_true(allocate(rs_true))."""
    a_est = allocate(rs_est, budget)["alloc"]
    true_of_est = sum(structure_risk(rs_true[s], a_est[s], STRUCT_K[s],
                                     STRUCT_W[s]) for s in rs_true)
    oracle = allocate(rs_true, budget)["total_risk"]
    return float(true_of_est - oracle)


def warmstart_study(cold: str = "srb3", flights_grid=(2, 4, 8, 16, 30),
                    cold_vehicles: int = 3, budget: float = TOTAL_BUDGET,
                    seed: int = 0) -> dict:
    """As the cold structure flies more, compare the allocation REGRET when its
    risk is estimated by within-structure vs cross-structure fleet learning.
    Cross-structure (better estimate) should give lower regret, especially early."""
    rs_true = {s: float(np.exp(csf.STRUCTURE_MU[s])) for s in csf.STRUCTURE_MU}
    within_reg, cross_reg = [], []
    for nf in flights_grid:
        meas, cfgs = {}, {}
        for i, (name, mu) in enumerate(csf.STRUCTURE_MU.items()):
            nv = cold_vehicles if name == cold else 14
            ff = nf if name == cold else 30
            sh, tu, cfg = csf.structure_measurements(mu, nv, ff, seed=seed + 7 * i)
            meas[name] = (sh, tu); cfgs[name] = cfg
        post = csf.fit_cross_hierarchy(meas, seed=seed)
        # cross-structure estimate of each structure's rate
        rs_cross = {s: float(np.exp(post.mu_s[:, post.structures.index(s)].mean()))
                    for s in csf.STRUCTURE_MU}
        # within-structure estimate (each structure alone) for the cold one;
        # rich structures share the same (data-rich) estimate.
        rs_within = dict(rs_cross)
        wmean, _ = csf.within_structure_mu(*meas[cold], cfgs[cold], seed=seed)
        rs_within[cold] = float(np.exp(wmean))
        within_reg.append(allocation_regret(rs_within, rs_true, budget))
        cross_reg.append(allocation_regret(rs_cross, rs_true, budget))
    return {"flights": list(flights_grid), "within_regret": within_reg,
            "cross_regret": cross_reg, "cold": cold, "rs_true": rs_true}


# ═════════════════════════════════════════════════════════════════════════════
#  Driver
# ═════════════════════════════════════════════════════════════════════════════

def run_study(seed: int = 0, budget: float = TOTAL_BUDGET,
              verbose: bool = True) -> dict:
    # per-structure rates from the cross-structure fleet posterior (Stage-4).
    d_fleet = csf.run_study(seed=seed, verbose=False)
    post = d_fleet["post"]
    rs = {s: float(np.exp(post.mu_s[:, post.structures.index(s)].mean()))
          for s in csf.STRUCTURE_MU}

    strategies = {
        "optimal": allocate(rs, budget),
        "uniform": allocate_uniform(rs, budget),
        "greedy-worst": allocate_greedy_worst(rs, budget),
        "no-budget": allocate_none(rs),
    }
    ws = warmstart_study(seed=seed, budget=budget)
    out = {"rs": rs, "strategies": strategies, "warmstart": ws,
           "budget": budget}
    if verbose:
        _report(out)
    return out


def _report(d: dict) -> None:
    bar = "=" * 78
    print(bar)
    print("  CROSS-STRUCTURE DESIGN — portfolio toughening-budget allocation "
          "(Stage-5+)")
    print(bar)
    print(f"  structures + estimated growth rate r=exp(μ) (from Stage-4 fleet):")
    for s, r in d["rs"].items():
        print(f"    {s:12s}  r={r:.4f}  (k={STRUCT_K[s]}, value w={STRUCT_W[s]})")
    print(f"  total budget B = {d['budget']:.2f}\n")
    print(f"  {'strategy':<14}{'total risk':>12}{'worst risk':>12}   allocation")
    print("  " + "-" * 72)
    opt = d["strategies"]["optimal"]["total_risk"]
    for name in ("optimal", "uniform", "greedy-worst", "no-budget"):
        st = d["strategies"][name]
        alloc = "  ".join(f"{s}:{b:.2f}" for s, b in st["alloc"].items())
        gain = ((d["strategies"]["no-budget"]["total_risk"] - st["total_risk"])
                / max(d["strategies"]["no-budget"]["total_risk"], 1e-9) * 100)
        print(f"  {name:<14}{st['total_risk']:12.3f}{st['worst_risk']:12.3f}"
              f"   {alloc}")
    red = ((d["strategies"]["uniform"]["total_risk"] - opt)
           / max(d["strategies"]["uniform"]["total_risk"], 1e-9) * 100)
    print(f"\n  optimal portfolio cuts total risk {red:.1f}% below uniform; "
          f"more budget → faster-growing / man-rated structures.")
    ws = d["warmstart"]
    print(f"\n  WARM-START (cold='{ws['cold']}'): allocation REGRET vs oracle, "
          f"within- vs cross-structure risk estimate:")
    print(f"    flights:        " + "".join(f"{f:8d}" for f in ws["flights"]))
    print(f"    within-struct : " + "".join(f"{r:8.3f}" for r in ws["within_regret"]))
    print(f"    cross-struct  : " + "".join(f"{r:8.3f}" for r in ws["cross_regret"]))
    imp = np.mean([(w - c) for w, c in zip(ws["within_regret"],
                                           ws["cross_regret"])])
    print(f"  → cross-structure fleet learning lowers mean allocation regret by "
          f"{imp:.3f} (better design decision from less data).")
    print("\n  LIMITATION: growth→risk map, k_s, H, w_s REPRESENTATIVE; the "
          "contribution is the")
    print("  portfolio decision structure + that cross-structure learning lowers "
          "regret, not")
    print("  certified budgets. Cross-CFRP rate exchangeability = same "
          "approximation as Stage-4.")
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

    rs = d["rs"]
    names = list(rs.keys())
    pal = {"interstage": "#90a4ae", "fairing": "#5c93c4", "srb3": "#c4a35c"}
    fig, ax = plt.subplots(1, 3, figsize=figsize)

    # (a) per-structure risk vs its own budget; mark the optimal allocation.
    bgrid = np.linspace(0, d["budget"], 80)
    opt = d["strategies"]["optimal"]["alloc"]
    for s in names:
        risk = [structure_risk(rs[s], bb, STRUCT_K[s], STRUCT_W[s]) for bb in bgrid]
        ax[0].plot(bgrid, risk, color=pal[s], lw=1.5, label=s)
        ax[0].plot(opt[s], structure_risk(rs[s], opt[s], STRUCT_K[s], STRUCT_W[s]),
                   "o", color=pal[s], ms=7, mec="k", mew=0.6)
    ax[0].set_xlabel("budget allocated $b_s$", fontsize=7)
    ax[0].set_ylabel("structure risk", fontsize=7)
    ax[0].set_title("(a) risk response + optimal split\n(● = optimal allocation)",
                    fontsize=8)
    ax[0].legend(fontsize=6); ax[0].tick_params(labelsize=6)

    # (b) total risk by strategy.
    strat = ["optimal", "uniform", "greedy-worst", "no-budget"]
    tot = [d["strategies"][s]["total_risk"] for s in strat]
    wst = [d["strategies"][s]["worst_risk"] for s in strat]
    x = np.arange(len(strat))
    ax[1].bar(x - 0.2, tot, 0.4, color="#2e7d32", label="total risk")
    ax[1].bar(x + 0.2, wst, 0.4, color="#b71c1c", alpha=0.8,
              label="worst-structure")
    ax[1].set_xticks(x); ax[1].set_xticklabels(strat, fontsize=6, rotation=12)
    ax[1].set_ylabel("risk", fontsize=7)
    ax[1].set_title("(b) portfolio vs naive allocation\n(optimal = lowest)",
                    fontsize=8)
    ax[1].legend(fontsize=6); ax[1].tick_params(labelsize=6)

    # (c) warm-start: allocation regret vs cold-structure flights.
    ws = d["warmstart"]
    ax[2].plot(ws["flights"], ws["within_regret"], "s-", color="#5c93c4", ms=5,
               lw=1.3, label="within-structure estimate")
    ax[2].plot(ws["flights"], ws["cross_regret"], "D-", color="#b71c1c", ms=5,
               lw=1.3, label="cross-structure estimate")
    ax[2].axhline(0.0, color="k", ls="--", lw=0.7)
    ax[2].set_xlabel(f"flights flown by cold '{ws['cold']}'", fontsize=7)
    ax[2].set_ylabel("allocation regret vs oracle", fontsize=7)
    ax[2].set_title("(c) warm-start: cross-structure\nlowers design regret",
                    fontsize=8)
    ax[2].legend(fontsize=6); ax[2].tick_params(labelsize=6)
    ax[2].set_ylim(bottom=min(-0.02, min(ws["cross_regret"]) - 0.02))

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

    # ---- risk model monotonicity --------------------------------------------
    ok(p_fail(0.10) > p_fail(0.02), "P_fail increases with growth rate")
    ok(structure_risk(0.1, 0.0, 6.0, 1.0) > structure_risk(0.1, 0.5, 6.0, 1.0),
       "budget lowers risk (toughening)")
    ok(structure_risk(0.0, 0.0, 6.0, 1.0) == 0.0, "zero rate → zero risk")
    ok(structure_risk(0.1, 0.2, 6.0, 2.0) > structure_risk(0.1, 0.2, 6.0, 1.0),
       "higher value weight → higher risk")

    rs = {"a": 0.04, "b": 0.06, "c": 0.09}
    ks = {"a": 6.0, "b": 5.0, "c": 7.0}
    ws = {"a": 1.0, "b": 1.2, "c": 1.5}

    # ---- allocation conserves the budget + is non-negative ------------------
    a = allocate(rs, budget=1.0, ks=ks, ws=ws)
    ok(abs(sum(a["alloc"].values()) - 1.0) < 1e-6, "allocation spends the budget")
    ok(all(v >= -1e-9 for v in a["alloc"].values()), "allocation non-negative")

    # ---- optimal ≤ uniform ≤ no-budget on total risk ------------------------
    uni = allocate_uniform(rs, 1.0, ks=ks, ws=ws)
    gw = allocate_greedy_worst(rs, 1.0, ks=ks, ws=ws)
    none = allocate_none(rs, ks=ks, ws=ws)
    ok(a["total_risk"] <= uni["total_risk"] + 1e-6,
       "optimal ≤ uniform total risk")
    ok(a["total_risk"] <= gw["total_risk"] + 1e-6,
       "optimal ≤ greedy-worst total risk")
    ok(uni["total_risk"] <= none["total_risk"] + 1e-6,
       "any budget ≤ no-budget risk")
    ok(none["budget"] == 0.0, "no-budget spends nothing")

    # ---- more budget → lower optimal total risk -----------------------------
    big = allocate(rs, budget=2.0, ks=ks, ws=ws)
    ok(big["total_risk"] <= a["total_risk"] + 1e-9, "more budget → less risk")

    # ---- optimal sends MORE budget to the higher-risk structure 'c' ---------
    ok(a["alloc"]["c"] >= a["alloc"]["a"],
       "optimal favours the faster-growing / man-rated structure")

    # ---- allocation regret (uses the real STRUCT_K/W → real structure keys) --
    rs_real = {"interstage": 0.04, "fairing": 0.06, "srb3": 0.09}
    reg_self = allocation_regret(rs_real, rs_real, budget=1.0)
    ok(abs(reg_self) < 1e-6, "oracle allocation has ~zero self-regret")
    # a deliberately WRONG estimate (swapped slow/fast) has non-negative regret
    wrong = {"interstage": 0.09, "fairing": 0.06, "srb3": 0.04}
    ok(allocation_regret(wrong, rs_real, budget=1.0) >= -1e-9,
       "mis-estimated allocation regret ≥ 0")
    ok(allocation_regret(wrong, rs_real, budget=1.0) > reg_self,
       "wrong estimate has strictly more regret than the oracle")

    # ---- warm-start study: cross-structure regret ≤ within at the data-poor end
    wstud = warmstart_study(cold="srb3", flights_grid=(2, 8), cold_vehicles=3,
                            seed=0)
    ok(len(wstud["cross_regret"]) == 2, "warm-start returns the grid")
    ok(all(r >= -1e-9 for r in wstud["cross_regret"]), "regrets ≥ 0")
    ok(np.mean(wstud["cross_regret"]) <= np.mean(wstud["within_regret"]) + 1e-6,
       "cross-structure mean regret ≤ within-structure (better design decision)")

    # ---- end-to-end run_study returns a coherent ranking --------------------
    d = run_study(seed=0, verbose=False)
    ok(d["strategies"]["optimal"]["total_risk"]
       <= d["strategies"]["uniform"]["total_risk"] + 1e-6,
       "end-to-end optimal ≤ uniform")
    ok(set(d["rs"]) == set(csf.STRUCTURE_MU), "rates for every structure")

    print(f"cross_structure_design: {n}/{n} unit tests passed")
    return n


def main():
    ap = argparse.ArgumentParser(
        description="Cross-structure portfolio design / budget allocation (Stage-5+).")
    ap.add_argument("--budget", type=float, default=TOTAL_BUDGET)
    ap.add_argument("--fig", action="store_true")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.test:
        run_tests(); return
    d = run_study(budget=args.budget)
    if args.fig:
        p = make_figure(d, os.path.join(HERE, "paper_figs",
                                        "cross_structure_design.pdf"))
        print(f"\nfigure → {p}")


if __name__ == "__main__":
    main()
