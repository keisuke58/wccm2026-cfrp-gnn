"""
voi_inspection.py — STAGE 2.5: the Value-of-Information INSPECTION decision.

The paper's abstract already asserts that the clearance verdict is *fragile* in a
mid band of the growth posterior — "the verdict is fragile at P~0.3-0.5, where
additional inspection minimises expected cost".  Every other module either makes
the OK/REPAIR/RETIRE call (`decision_uq.decide`, expected-cost α/β) or benchmarks
that call against naive strategies (`system_baseline`).  None of them formalises
the decision that sits BETWEEN detection and clearance: given a damage posterior,
should we CLEAR NOW, or pay for one more INSPECTION and clear on a sharper belief?

This module makes that an explicit decision STAGE (2.5).  It is pure
decision-theoretic value of information, reusing the SAME economics everywhere:

  * State of knowledge = a BELIEF over the growth probability p, modelled as a
    Beta(a,b) whose MEAN is the framework's predicted P(grow) and whose spread
    encodes the posterior (characterisation) uncertainty.  We never invent a new
    cost model — the action cost of any decision at a given p is exactly
    `system_baseline.expected_cost(decision, p)`, the very expectation whose
    minimiser defines `cfrp_phasefield_2d.calibrate_thresholds`' α, β.

  * E[cost of DECIDING NOW] = min over actions {OK,REPAIR,RETIRE} of
    E_p[expected_cost(action, p)] under the belief — the Bayes-optimal immediate
    action averaged over what we don't yet know about p.

  * EVPI (perfect information) = E_now − E_p[ min_action expected_cost(action,p) ]
    : the cost you would save if an oracle told you p EXACTLY before deciding.
    Upper bound on what any inspection can be worth.

  * EVSI (sample / imperfect information) for an inspection of quality κ: the
    inspection is a stylised belief-SHARPENING operator — draw the "true" p from
    the belief, observe it through Gaussian noise σ_insp (smaller κ ⇒ noisier),
    form the post-inspection posterior, and take its Bayes action.  EVSI is the
    expected cost reduction of that imperfect observation; 0 ≤ EVSI ≤ EVPI.

  * VoI decision rule (Stage 2.5): INSPECT iff EVSI(belief) > c_inspect, the
    cost/time of an inspection.  Swept over predicted P(grow) ∈ [0,1] this carves
    out an INSPECT-REGION that peaks in the ambiguous mid band (~0.3–0.5, where
    the OK/REPAIR/RETIRE action flips) and collapses to ~0 at p≈0 (clearly OK)
    and p≈1 (clearly RETIRE) — exactly the abstract's claim, now quantified.

We also run it on the THREE structures' actual `system_baseline.Scenarios`
(interstage / fairing / SRB-3): the fraction of scenarios Stage 2.5 flags for
inspection, and the expected-cost SAVING of the VoI policy versus always-decide-
now and versus always-inspect (which wastes inspection budget on clear cases).

LIMITATION (honest)
-------------------
The inspection is a STYLISED belief-sharpening operator (Beta/Gaussian) with a
representative quality κ and a representative inspection cost c_inspect — NOT a
calibrated NDT model.  It quantifies WHERE (which P-band) and HOW MUCH (in the
same cost units as α/β) one more inspection pays for itself in the decision-
theoretic sense.  Absolute savings and the exact inspect-band edges depend on the
cost anchors (c_loss, c_repair, c_retire_value, r, c_inspect) and on κ; this is
not a claim about calibrated NDT economics or real inspection sensitivities.

Usage
-----
    python voi_inspection.py            # VoI sweep + per-structure policy report
    python voi_inspection.py --fig      # + paper_figs/voi_inspection.pdf
    python voi_inspection.py --test     # unit tests only
"""
from __future__ import annotations

import argparse
import os

import numpy as np

import system_baseline as sb

HERE = os.path.dirname(os.path.abspath(__file__))

DECISIONS = sb.DECISIONS                       # ("OK", "REPAIR", "RETIRE")

# Representative inspection economics (same cost units as COSTS / α / β).
# c_inspect: cost/time of one inspection.  KAPPA: inspection quality in (0,1],
# 1 = perfect (σ_insp→0), small = noisy.  These are the stylised anchors the
# LIMITATION refers to.
C_INSPECT = 0.6
KAPPA = 0.7
# belief spread: Beta concentration κ_conc = a+b; larger ⇒ tighter belief.
# Two representative posterior-spread levels for the sweep.
CONC_LEVELS = (12.0, 40.0)                      # (broad, tight) posterior


# ═════════════════════════════════════════════════════════════════════════════
#  Belief over p: a Beta(a,b) with a prescribed mean and concentration
# ═════════════════════════════════════════════════════════════════════════════

def beta_ab(mean: float, conc: float) -> tuple[float, float]:
    """Beta(a,b) with E[p]=mean and a+b=conc (concentration).  Mean is clipped
    just inside (0,1) so a,b stay strictly positive (a degenerate belief at
    p=0/1 is the no-uncertainty limit, handled by the sampler clip)."""
    m = float(np.clip(mean, 1e-4, 1 - 1e-4))
    a = m * conc
    b = (1.0 - m) * conc
    return max(a, 1e-3), max(b, 1e-3)


def sample_belief(mean: float, conc: float, n: int,
                  rng: np.random.Generator) -> np.ndarray:
    """Draw n samples of p from the Beta belief (the state of knowledge)."""
    a, b = beta_ab(mean, conc)
    return np.clip(rng.beta(a, b, n), 0.0, 1.0)


# ═════════════════════════════════════════════════════════════════════════════
#  Bayes action + expected costs under a belief (reusing system_baseline)
# ═════════════════════════════════════════════════════════════════════════════

def _action_costs_under_belief(p_samples: np.ndarray,
                               costs: dict | None = None) -> dict:
    """E_p[expected_cost(action, p)] for each action, averaged over the belief
    samples.  expected_cost is REUSED verbatim from system_baseline."""
    return {d: float(np.mean([sb.expected_cost(d, p, costs) for p in p_samples]))
            for d in DECISIONS}


def expected_cost_decide_now(p_samples: np.ndarray,
                             costs: dict | None = None) -> float:
    """E[cost | decide now] = min_action E_p[expected_cost(action,p)] — the
    Bayes-optimal immediate action's expected cost under the current belief."""
    ac = _action_costs_under_belief(p_samples, costs)
    return min(ac.values())


def expected_cost_perfect_info(p_samples: np.ndarray,
                               costs: dict | None = None) -> float:
    """E_p[ min_action expected_cost(action, p) ]: if you knew p exactly you would
    pick the per-p cost-minimising action, then average that over the belief."""
    per = [min(sb.expected_cost(d, p, costs) for d in DECISIONS)
           for p in p_samples]
    return float(np.mean(per))


def evpi(mean: float, conc: float, n: int = 4000, seed: int = 0,
         costs: dict | None = None) -> float:
    """Expected value of PERFECT information for a Beta(mean,conc) belief."""
    rng = np.random.default_rng(seed)
    p = sample_belief(mean, conc, n, rng)
    return max(expected_cost_decide_now(p, costs)
               - expected_cost_perfect_info(p, costs), 0.0)


# ═════════════════════════════════════════════════════════════════════════════
#  EVSI: a stylised noisy inspection that SHARPENS the belief
# ═════════════════════════════════════════════════════════════════════════════

def _logit(x):  return np.log(x / (1.0 - x))
def _sigmoid(x): return 1.0 / (1.0 + np.exp(-x))


def evsi(mean: float, conc: float, kappa: float = KAPPA,
         n_outer: int = 600, n_inner: int = 200, seed: int = 0,
         costs: dict | None = None) -> float:
    """Expected value of SAMPLE (imperfect) information for an inspection of
    quality `kappa` ∈ (0,1].

    Model: the inspection returns a noisy reading of the true p.  We work in
    logit space so the signal stays in (0,1): observe y = logit(p_true)+ε with
    ε ~ N(0, σ²), σ = σ0·(1/κ − 1) (κ→1 ⇒ σ→0 ⇒ perfect; κ small ⇒ noisy).
    Conjugate-style Gaussian update of the logit-belief gives a SHARPER
    post-inspection belief whose Bayes action we take.  EVSI = E[cost|now] −
    E_signal[ E[cost | post-inspection Bayes action] ].  By construction
    0 ≤ EVSI ≤ EVPI (perfect κ→1 recovers EVPI; finite noise can only lose
    information)."""
    rng = np.random.default_rng(seed)
    prior_p = sample_belief(mean, conc, n_outer, rng)        # the belief
    now_cost = expected_cost_decide_now(prior_p, costs)

    # logit-Gaussian approximation of the belief (mean/var of logit p)
    lp = _logit(np.clip(prior_p, 1e-4, 1 - 1e-4))
    mu0, var0 = float(lp.mean()), float(lp.var() + 1e-9)
    # observation noise in logit space: σ shrinks to 0 as κ→1
    sigma = (np.sqrt(var0) + 1e-6) * (1.0 / max(kappa, 1e-3) - 1.0) + 1e-6
    var_obs = sigma ** 2

    post_costs = np.empty(n_outer)
    for i in range(n_outer):
        p_true = prior_p[i]                                  # the would-be truth
        y = _logit(np.clip(p_true, 1e-4, 1 - 1e-4)) + rng.normal(0, sigma)
        # Gaussian posterior on logit p after the noisy reading
        var_post = 1.0 / (1.0 / var0 + 1.0 / var_obs)
        mu_post = var_post * (mu0 / var0 + y / var_obs)
        # sharpened post-inspection belief → its Bayes action's expected cost
        zp = _sigmoid(mu_post + np.sqrt(var_post) * rng.standard_normal(n_inner))
        zp = np.clip(zp, 0.0, 1.0)
        post_costs[i] = expected_cost_decide_now(zp, costs)
    exp_post = float(post_costs.mean())
    return max(now_cost - exp_post, 0.0)


# ═════════════════════════════════════════════════════════════════════════════
#  Sweep over predicted P(grow): EVPI / EVSI / inspect-region
# ═════════════════════════════════════════════════════════════════════════════

def voi_sweep(conc: float, kappa: float = KAPPA, n_grid: int = 41,
              seed: int = 0, costs: dict | None = None) -> dict:
    """Sweep predicted P(grow) ∈ [0,1] at a fixed posterior concentration.
    Returns EVPI, EVSI, and the (decide-now / always-inspect / VoI-policy)
    expected costs at each grid point."""
    P = np.linspace(0.0, 1.0, n_grid)
    EVPI = np.empty(n_grid)
    EVSI = np.empty(n_grid)
    cost_now = np.empty(n_grid)
    for i, m in enumerate(P):
        rng = np.random.default_rng(seed + i)
        p = sample_belief(m, conc, 4000, rng)
        cn = expected_cost_decide_now(p, costs)
        cost_now[i] = cn
        EVPI[i] = max(cn - expected_cost_perfect_info(p, costs), 0.0)
        EVSI[i] = evsi(m, conc, kappa, seed=seed + i, costs=costs)
    EVSI = np.minimum(EVSI, EVPI)                  # enforce EVSI ≤ EVPI
    # cost of three policies (per unit decision, ignoring realised inspection
    # outcome on the action — the inspection's BENEFIT is EVSI):
    #   decide-now      : cost_now
    #   always-inspect  : cost_now − EVSI + c_inspect (pay c_inspect, gain EVSI)
    #   VoI-policy      : inspect iff EVSI > c_inspect, else decide now
    cost_always_insp = cost_now - EVSI + C_INSPECT
    inspect = EVSI > C_INSPECT
    cost_voi = np.where(inspect, cost_always_insp, cost_now)
    return {"P": P, "EVPI": EVPI, "EVSI": EVSI, "inspect": inspect,
            "cost_now": cost_now, "cost_always_insp": cost_always_insp,
            "cost_voi": cost_voi, "conc": conc, "kappa": kappa}


def inspect_band(P: np.ndarray, inspect: np.ndarray) -> tuple[float, float]:
    """The contiguous P-band where the VoI rule says inspect (lo, hi).  Returns
    (nan, nan) if it never inspects."""
    idx = np.where(inspect)[0]
    if len(idx) == 0:
        return float("nan"), float("nan")
    return float(P[idx.min()]), float(P[idx.max()])


# ═════════════════════════════════════════════════════════════════════════════
#  Stage 2.5 applied to the three structures' actual scenario sets
# ═════════════════════════════════════════════════════════════════════════════

def voi_on_scenarios(sc: sb.Scenarios, conc: float = CONC_LEVELS[0],
                     kappa: float = KAPPA, seed: int = 0,
                     costs: dict | None = None) -> dict:
    """Apply Stage 2.5 to one structure's Scenarios.

    For every scenario the framework's predicted P(grow) (p_pipeline) is the
    belief mean; we compute EVSI, decide INSPECT iff EVSI > c_inspect, and score
    the EXPECTED cost under the belief (the decision-theoretic quantity EVSI is
    defined against, reusing system_baseline.expected_cost) of three policies:
      * decide-now     : E[cost | Bayes action on the belief]  (= cost_now).
      * always-inspect : pay c_inspect and gain the inspection's value EVSI on
                         every scenario  (= cost_now − EVSI + c_inspect).
      * VoI-policy     : always-inspect cost iff EVSI > c_inspect (the inspection
                         pays for itself), else decide-now.
    Scoring every policy by its expected cost under the belief makes the three
    consistent — the inspection's BENEFIT is exactly EVSI, so the VoI rule
    (inspect iff EVSI > c_inspect) is the cost-minimiser per scenario by
    construction.  (p_oracle is used only as the scenario's centre of mass via
    p_pipeline; no separate realised-vs-expected mixing.)
    """
    M = len(sc.p_pipeline)
    EVSI = np.empty(M)
    cost_now = np.empty(M)
    for i in range(M):
        m = float(sc.p_pipeline[i])
        EVSI[i] = evsi(m, conc, kappa, seed=seed + i, costs=costs)
        rng = np.random.default_rng(seed + 10_000 + i)
        cost_now[i] = expected_cost_decide_now(
            sample_belief(m, conc, 2000, rng), costs)
    cost_insp = cost_now - EVSI + C_INSPECT
    inspect = EVSI > C_INSPECT
    cost_voi = np.where(inspect, cost_insp, cost_now)
    return {
        "EVSI": EVSI, "inspect": inspect,
        "frac_inspect": float(inspect.mean()),
        "mean_cost_now": float(cost_now.mean()),
        "mean_cost_always_insp": float(cost_insp.mean()),
        "mean_cost_voi": float(cost_voi.mean()),
        "save_vs_now": float(cost_now.mean() - cost_voi.mean()),
        "save_vs_always_insp": float(cost_insp.mean() - cost_voi.mean()),
    }


def voi_all_structures(seed: int = 0, M: int = 40, conc: float = CONC_LEVELS[0],
                       kappa: float = KAPPA) -> dict:
    out = {}
    for name, loader in sb.STRUCTURES.items():
        sc = loader(seed=seed) if name == "interstage" else loader(seed=seed, M=M)
        out[name] = {"sc": sc,
                     "voi": voi_on_scenarios(sc, conc=conc, kappa=kappa,
                                             seed=seed)}
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  Report
# ═════════════════════════════════════════════════════════════════════════════

def print_report(sweeps: dict, per: dict) -> None:
    bar = "=" * 78
    print(bar)
    print("  STAGE 2.5 — VALUE-OF-INFORMATION INSPECTION DECISION")
    print("  CLEAR NOW vs INSPECT-then-clear, by expected cost")
    print(bar)
    a, b = sb.pf.calibrate_thresholds(**sb.COSTS)
    print(f"  cost model : c_loss={sb.COSTS['c_loss']:.0f}  "
          f"c_repair={sb.COSTS['c_repair']:.0f}  "
          f"c_retire={sb.COSTS['c_retire_value']:.0f}  "
          f"r={sb.COSTS['repair_residual_risk']:.2f}   (α={a:.2f}, β={b:.2f})")
    print(f"  inspection : c_inspect={C_INSPECT:.2f}  quality κ={KAPPA:.2f}  "
          f"(stylised belief-sharpening operator)")
    print()

    # the headline sweep (broad belief)
    for conc, sw in sweeps.items():
        i_evpi = int(np.argmax(sw["EVPI"]))
        i_evsi = int(np.argmax(sw["EVSI"]))
        lo, hi = inspect_band(sw["P"], sw["inspect"])
        band = ("never inspects" if lo != lo else
                f"inspect when P∈[{lo:.2f},{hi:.2f}]")
        print(f"  belief spread: concentration a+b={conc:.0f} "
              f"({'broad' if conc <= 15 else 'tight'} posterior)")
        print(f"     EVPI peak : {sw['EVPI'][i_evpi]:6.3f} at "
              f"P(grow)={sw['P'][i_evpi]:.2f}")
        print(f"     EVSI peak : {sw['EVSI'][i_evsi]:6.3f} at "
              f"P(grow)={sw['P'][i_evsi]:.2f}   (κ={sw['kappa']:.2f})")
        print(f"     inspect-region (EVSI > c_inspect={C_INSPECT:.2f}): {band}")
        print(f"     EVPI≈0 at the extremes: P=0 → {sw['EVPI'][0]:.3f}, "
              f"P=1 → {sw['EVPI'][-1]:.3f}")
        print()

    print("  " + "-" * 74)
    print("  PER-STRUCTURE Stage-2.5 policy (predicted P(grow) = p_pipeline):")
    print(f"    {'structure':<18}{'%inspect':>9}{'cost_now':>10}"
          f"{'cost_voi':>10}{'cost_alw.insp':>14}"
          f"{'save_now':>10}{'save_alw':>10}")
    for name, d in per.items():
        v = d["voi"]
        print(f"    {name:<18}{v['frac_inspect']*100:8.1f}%"
              f"{v['mean_cost_now']:10.2f}{v['mean_cost_voi']:10.2f}"
              f"{v['mean_cost_always_insp']:14.2f}"
              f"{v['save_vs_now']:10.2f}{v['save_vs_always_insp']:10.2f}")
    print()
    print("  HEADLINE: value of information PEAKS in the ambiguous mid-band of")
    print("  P(grow) (where the OK/REPAIR/RETIRE action flips) and collapses to")
    print("  ~0 at P≈0 (clearly OK) and P≈1 (clearly RETIRE) — inspection pays for")
    print("  itself ONLY there.  The VoI policy tracks the lower cost envelope:")
    print("  cheaper than deciding-now (it buys the decisive inspections) AND than")
    print("  always-inspecting (it skips the wasteful ones on clear-cut cases).")
    print()
    print("  LIMITATION: the inspection is a STYLISED belief-sharpening operator")
    print("  (Beta/Gaussian) with representative κ and c_inspect — it quantifies")
    print("  WHERE and HOW MUCH inspection helps decision-theoretically, NOT a")
    print("  calibrated NDT model; absolute savings depend on the cost anchors.")
    print(bar)


# ═════════════════════════════════════════════════════════════════════════════
#  Figure
# ═════════════════════════════════════════════════════════════════════════════

def make_figure(sweeps: dict, per: dict, out_path: str) -> str:
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

    # headline sweep = the broad-belief one (first concentration level)
    conc0 = list(sweeps)[0]
    sw = sweeps[conc0]
    P = sw["P"]
    lo, hi = inspect_band(P, sw["inspect"])

    fig, ax = plt.subplots(1, 3, figsize=figsize)

    # (a) THE MONEY PLOT: EVPI & EVSI vs P, c_inspect line, shaded inspect-region
    ax[0].plot(P, sw["EVPI"], "-", color="#1565C0", lw=1.6,
               label="EVPI (perfect info)")
    ax[0].plot(P, sw["EVSI"], "-", color="#b71c1c", lw=1.6,
               label=f"EVSI ($\\kappa$={sw['kappa']:.1f})")
    # second concentration (tighter belief) for context
    if len(sweeps) > 1:
        conc1 = list(sweeps)[1]
        ax[0].plot(P, sweeps[conc1]["EVSI"], "--", color="#b71c1c", lw=1.0,
                   alpha=0.6, label=f"EVSI tight (a+b={conc1:.0f})")
    ax[0].axhline(C_INSPECT, color="0.4", lw=0.9, ls=":",
                  label=f"c_inspect={C_INSPECT:.1f}")
    if lo == lo:
        ax[0].axvspan(lo, hi, color="#b71c1c", alpha=0.12,
                      label=f"inspect $P\\in$[{lo:.2f},{hi:.2f}]")
    ax[0].set_xlabel("predicted $P(\\mathrm{grow})$", fontsize=7)
    ax[0].set_ylabel("value of information (cost units)", fontsize=7)
    ax[0].set_title("(a) VoI peaks in the ambiguous mid-band", fontsize=8)
    ax[0].legend(fontsize=5.5, loc="upper right")
    ax[0].tick_params(labelsize=6)

    # (b) expected cost vs P: decide-now / always-inspect / VoI (lower envelope)
    ax[1].plot(P, sw["cost_now"], "-", color="#455a64", lw=1.4,
               label="decide-now")
    ax[1].plot(P, sw["cost_always_insp"], "-", color="#2e7d32", lw=1.2,
               label="always-inspect")
    ax[1].plot(P, sw["cost_voi"], "-", color="#b71c1c", lw=2.0, alpha=0.85,
               label="VoI-policy")
    if lo == lo:
        ax[1].axvspan(lo, hi, color="#b71c1c", alpha=0.10)
    ax[1].set_xlabel("predicted $P(\\mathrm{grow})$", fontsize=7)
    ax[1].set_ylabel("expected cost", fontsize=7)
    ax[1].set_title("(b) VoI tracks the lower cost envelope", fontsize=8)
    ax[1].legend(fontsize=6, loc="upper left")
    ax[1].tick_params(labelsize=6)

    # (c) per-structure expected-cost saving (VoI vs decide-now & always-inspect)
    names = list(per)
    short = {"interstage": "inter-\nstage", "fairing": "fairing",
             "srb3-motorcase": "SRB-3"}
    labels = [short.get(n, n) for n in names]
    x = np.arange(len(names))
    w = 0.38
    save_now = [per[n]["voi"]["save_vs_now"] for n in names]
    save_ins = [per[n]["voi"]["save_vs_always_insp"] for n in names]
    ax[2].bar(x - w / 2, save_now, w, color="#455a64", edgecolor="k",
              linewidth=0.3, label="vs decide-now")
    ax[2].bar(x + w / 2, save_ins, w, color="#2e7d32", edgecolor="k",
              linewidth=0.3, label="vs always-inspect")
    ax[2].axhline(0, color="k", lw=0.6)
    ax[2].set_xticks(x); ax[2].set_xticklabels(labels, fontsize=6.5)
    ax[2].set_ylabel("expected-cost saving of VoI policy", fontsize=7)
    ax[2].set_title("(c) Stage-2.5 saving per structure", fontsize=8)
    ax[2].legend(fontsize=6, loc="upper right")
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

    # reuse system_baseline.expected_cost (no duplicate cost model)
    ok(expected_cost_decide_now.__module__ == __name__, "module sanity")
    p0 = np.full(50, 0.3)
    ac = _action_costs_under_belief(p0)
    ok(np.isclose(ac["OK"], sb.expected_cost("OK", 0.3)),
       "action cost reuses system_baseline.expected_cost")

    # beta_ab respects the requested mean
    a, b = beta_ab(0.4, 20.0)
    ok(np.isclose(a / (a + b), 0.4, atol=1e-6), "beta mean = requested")
    ok(a > 0 and b > 0, "beta params positive")

    # decide-now ≥ perfect-info floor (knowing p can only help) → EVPI ≥ 0
    rng = np.random.default_rng(0)
    for m in (0.05, 0.2, 0.4, 0.6, 0.9):
        p = sample_belief(m, 12.0, 3000, rng)
        cn = expected_cost_decide_now(p)
        cf = expected_cost_perfect_info(p)
        ok(cn >= cf - 1e-9, f"decide-now ≥ perfect-info floor at mean={m}")

    # sweep at the broad belief
    sw = voi_sweep(12.0, kappa=KAPPA, n_grid=41, seed=0)
    P, EVPI, EVSI = sw["P"], sw["EVPI"], sw["EVSI"]

    # EVPI ≥ EVSI ≥ 0 everywhere
    ok(np.all(EVSI >= -1e-9), "EVSI ≥ 0 everywhere")
    ok(np.all(EVPI >= -1e-9), "EVPI ≥ 0 everywhere")
    ok(np.all(EVPI >= EVSI - 1e-9), "EVPI ≥ EVSI everywhere")

    # EVPI ≈ 0 at the extremes (clear OK / clear RETIRE), positive in the middle
    ok(EVPI[0] < 0.02, "EVPI ≈ 0 at P≈0 (clearly OK)")
    ok(EVPI[-1] < 0.02, "EVPI ≈ 0 at P≈1 (clearly RETIRE)")
    mid = EVPI[(P > 0.2) & (P < 0.6)]
    ok(mid.max() > 0.05, "EVPI positive in the mid band")
    # single interior peak: argmax is interior, not at an endpoint
    i_peak = int(np.argmax(EVPI))
    ok(0 < i_peak < len(P) - 1, "EVPI peak is interior (single mid-band hump)")
    ok(P[i_peak] > 0.05 and P[i_peak] < 0.95, "EVPI peak in interior P band")

    # inspect-region is a CONTIGUOUS mid band (no holes) when non-empty
    insp = sw["inspect"]
    if insp.any():
        idx = np.where(insp)[0]
        ok(np.all(np.diff(idx) == 1), "inspect-region is contiguous")
        lo, hi = inspect_band(P, insp)
        ok(0.0 < lo and hi < 1.0, "inspect band strictly interior")
        ok(not insp[0] and not insp[-1],
           "no inspect at the clear extremes")

    # EVSI monotone-ish in quality: higher κ ⇒ EVSI closer to EVPI at the peak
    m_peak = float(P[i_peak])
    lo_k = evsi(m_peak, 12.0, kappa=0.4, seed=1)
    hi_k = evsi(m_peak, 12.0, kappa=0.95, seed=1)
    ok(hi_k >= lo_k - 1e-3, "higher inspection quality ⇒ ≥ EVSI at the peak")
    ok(hi_k <= evpi(m_peak, 12.0, seed=1) + 1e-2, "EVSI(κ→1) ≤ EVPI at peak")

    # VoI-policy cost ≤ decide-now AND ≤ always-inspect across the sweep
    ok(np.all(sw["cost_voi"] <= sw["cost_now"] + 1e-9),
       "VoI policy ≤ decide-now over the sweep")
    ok(np.all(sw["cost_voi"] <= sw["cost_always_insp"] + 1e-9),
       "VoI policy ≤ always-inspect over the sweep (c_inspect>0)")

    # synthetic scenario set: VoI policy mean cost ≤ both alternatives
    sc = sb._synthetic_scenarios(M=30, seed=2)
    v = voi_on_scenarios(sc, conc=12.0, kappa=KAPPA, seed=0)
    ok(0.0 <= v["frac_inspect"] <= 1.0, "inspect fraction in [0,1]")
    ok(v["mean_cost_voi"] <= v["mean_cost_now"] + 1e-9,
       "VoI mean cost ≤ decide-now on scenario set")
    ok(v["mean_cost_voi"] <= v["mean_cost_always_insp"] + 1e-9,
       "VoI mean cost ≤ always-inspect on scenario set")
    ok(v["save_vs_now"] >= -1e-9 and v["save_vs_always_insp"] >= -1e-9,
       "VoI savings non-negative vs both baselines")

    print(f"voi_inspection: {n}/{n} unit tests passed")
    return n


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Stage 2.5: value-of-information inspection decision "
                    "(CLEAR NOW vs INSPECT-then-clear) by expected cost.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kappa", type=float, default=KAPPA,
                    help="inspection quality in (0,1] (1=perfect)")
    ap.add_argument("--fig", action="store_true",
                    help="save paper_figs/voi_inspection.pdf")
    ap.add_argument("--test", action="store_true", help="unit tests only")
    args = ap.parse_args()

    if args.test:
        run_tests()
        return

    sweeps = {conc: voi_sweep(conc, kappa=args.kappa, n_grid=41, seed=args.seed)
              for conc in CONC_LEVELS}
    per = voi_all_structures(seed=args.seed, kappa=args.kappa)
    print_report(sweeps, per)
    if args.fig:
        p = make_figure(sweeps, per,
                        os.path.join(HERE, "paper_figs", "voi_inspection.pdf"))
        print(f"\nfigure → {p}")


if __name__ == "__main__":
    main()
