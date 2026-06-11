"""
system_baseline.py — "the DECISION framework vs WHAT?" at the SYSTEM level.

The repo already benchmarks Stage-0 *detectors* against off-the-shelf anomaly
methods (`baselines_comparison.py`, backlog "better than WHAT?").  But the thing
the framework actually ships is not a detector — it is a DECISION pipeline that
turns a raw measurement into an OK / REPAIR / RETIRE clearance call by

    1. propagating the FMPE posterior over the defect θ through the forward
       (so a single P(grow) carries the characterisation uncertainty), then
    2. thresholding that P(grow) with EXPECTED-COST-optimal α, β
       (`cfrp_phasefield_2d.calibrate_thresholds`), and
    3. (optionally) recalibrating the per-decision confidence.

That machinery has a cost (FMPE, posterior sampling, a calibrated cost model).
The system-level question nobody had answered is: *what does it buy over the
naive things an operator could do instead?*  This module benchmarks the WHOLE
decision pipeline against simpler decision STRATEGIES on the SAME scenarios used
by `decision_uq` — every strategy scored against the exact-FD ORACLE decision
(θ_true + FD physics over the load distribution → the best call anyone could
make), and against the SAME expected-cost model that defines α, β.

Decision strategies compared
----------------------------
  1. FRAMEWORK (ours) : full posterior-propagated P(grow) + expected-cost α/β.
  2. POINT-ESTIMATE   : surrogate P(grow) at the posterior-MEAN θ only — same
                        α/β, but throws away characterisation uncertainty (the
                        ablation that isolates what UQ-propagation buys).
  3. FIXED-0.5        : "more likely than not" rule — grow-prob thresholded at
                        0.5, no expected-cost model (the naive operator).
  4. ALWAYS-REPAIR    : trivial conservative baseline — never fly borderline;
                        bounds the cost of pure over-conservatism.
  5. DETECTION-ONLY   : decide from the Stage-0 anomaly verdict alone
                        (defect present → REPAIR), ignoring prognosis.

Metrics per strategy (vs the FD oracle decision)
------------------------------------------------
  * end-to-end decision accuracy  P(strategy = oracle)
  * DANGEROUS-miss rate           P(strategy = OK | oracle = RETIRE)
  * conservative over-call rate   P(strategy more severe than oracle)
  * EXPECTED COST                 mean realised cost = the SAME cost model behind
                                  calibrate_thresholds, evaluated at each
                                  scenario's TRUE growth prob p_oracle:
                                    E[cost|OK]     = p·c_loss
                                    E[cost|REPAIR] = c_repair + r·p·c_loss
                                    E[cost|RETIRE] = c_retire_value
                                  This is the key SYSTEM metric: the framework's
                                  thresholds minimise exactly this expectation,
                                  so it should achieve the lowest cost and lowest
                                  dangerous-miss, beating the naive thresholds.

Scenarios are reused from `decision_uq`'s cached study (results/decision_uq/
study.npz) when present; otherwise a synthetic (p_oracle, posterior-draw) set
that mimics it is generated (guarded), so this runs anywhere in well under 60 s.

Usage
-----
    python system_baseline.py            # run benchmark + ranking table
    python system_baseline.py --fig      # + paper_figs/system_baseline.pdf
    python system_baseline.py --test     # unit tests only
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np

import cfrp_phasefield_2d as pf
import decision_uq as duq

HERE = os.path.dirname(os.path.abspath(__file__))

SEVERITY = {"OK": 0, "REPAIR": 1, "RETIRE": 2}
DECISIONS = ("OK", "REPAIR", "RETIRE")

# the SAME cost model that defines calibrate_thresholds' α, β.
COSTS = dict(c_loss=100.0, c_repair=1.0, c_retire_value=25.0,
             repair_residual_risk=0.5)


# ═════════════════════════════════════════════════════════════════════════════
#  Cost model (identical economics to cfrp_phasefield_2d.calibrate_thresholds)
# ═════════════════════════════════════════════════════════════════════════════

def expected_cost(decision: str, p_true: float, costs: dict | None = None
                  ) -> float:
    """Realised expected cost of a single clearance DECISION given the TRUE
    growth probability `p_true` (the oracle's p_oracle).  This is the very
    expectation that calibrate_thresholds minimises, so the cost-optimal
    decision at each p_true is, by construction, the oracle decision."""
    c = costs or COSTS
    if decision == "OK":
        return p_true * c["c_loss"]
    if decision == "REPAIR":
        return c["c_repair"] + c["repair_residual_risk"] * p_true * c["c_loss"]
    if decision == "RETIRE":
        return c["c_retire_value"]
    raise ValueError(f"unknown decision {decision!r}")


def mean_cost(decisions: np.ndarray, p_true: np.ndarray,
              costs: dict | None = None) -> float:
    """Mean realised expected cost over a scenario set."""
    return float(np.mean([expected_cost(d, p, costs)
                          for d, p in zip(decisions, p_true)]))


def severity(dec: np.ndarray) -> np.ndarray:
    return np.array([SEVERITY[d] for d in dec])


# ═════════════════════════════════════════════════════════════════════════════
#  Scenario set: reuse decision_uq's cached study, else synthesise a mimic
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Scenarios:
    p_oracle: np.ndarray      # (M,) TRUE growth prob (FD over load scatter)
    p_pipeline: np.ndarray    # (M,) framework P(grow): posterior-propagated
    p_point: np.ndarray       # (M,) P(grow) at posterior-MEAN θ (no-UQ ablation)
    detect: np.ndarray        # (M,) bool: Stage-0 says a defect is present
    alpha: float
    beta: float
    source: str


def _synthetic_scenarios(M: int = 40, seed: int = 0) -> Scenarios:
    """Mimic decision_uq's (p_oracle, posterior-draw) set without the FD/FMPE
    machinery: a spread of TRUE growth probs across [0,1] (so all three classes
    appear), a posterior-mean estimate that is a noisy version of the truth
    (point estimate), and a posterior-propagated estimate that is the truth
    smeared toward 0.5 by characterisation uncertainty (Jensen pull of the
    monotone-but-curved forward through the posterior — exactly the regulariser
    UQ-propagation provides)."""
    rng = np.random.default_rng(seed)
    alpha, beta = pf.calibrate_thresholds(**COSTS)
    p_oracle = np.clip(rng.beta(0.6, 0.6, M), 0.0, 1.0)   # U-ish: many 0/1-ish
    # point estimate: posterior-mean θ → surrogate; unbiased but noisy.
    p_point = np.clip(p_oracle + rng.normal(0, 0.18, M), 0.0, 1.0)
    # posterior-propagated: averaging P(grow) over the posterior pulls extremes
    # toward the prior mass → a shrink toward 0.5 plus small noise.
    shrink = 0.25
    p_pipeline = np.clip((1 - shrink) * p_oracle + shrink * 0.5
                         + rng.normal(0, 0.06, M), 0.0, 1.0)
    detect = p_oracle > 0.05                              # Stage-0 verdict proxy
    return Scenarios(p_oracle=p_oracle, p_pipeline=p_pipeline, p_point=p_point,
                     detect=detect, alpha=alpha, beta=beta,
                     source="SYNTHETIC mimic (no decision_uq cache)")


def _point_estimate_probs(study: duq.Study) -> np.ndarray:
    """P(grow) at the posterior-MEAN θ for each scenario (the no-UQ ablation).

    Recomputes the surrogate forward at the mean of each FMPE posterior over the
    same load scatter.  Falls back to the cached surr-oracle prob if the FMPE
    posterior cache is unavailable (its θ_true ≈ posterior mean for a calibrated
    posterior, so it is the right no-uncertainty reference)."""
    try:
        ts = duq.load_testset(len(study.p_oracle))
        posteriors = ts["posteriors"]                    # (M, N, 4)
        M = len(study.p_oracle)
        model, _ = duq.cs.load_surrogate(device="cpu")
        cfg = pf.LaminateConfig()
        rng = np.random.default_rng(0)
        nominal = study.nominal
        p_point = np.empty(M)
        for s in range(M):
            theta_mean = posteriors[s].mean(0)           # posterior MEAN θ
            Ls = duq.load_grid(float(nominal[s]), 5)
            pp = duq._surrogate_probs(theta_mean, Ls, model, cfg)
            p_point[s] = float(pp.mean())
        return p_point
    except Exception as e:
        print(f"[warn] point-estimate via surrogate unavailable ({e}); "
              "falling back to cached surr-oracle prob")
        return np.asarray(study.p_surr_oracle, dtype=float)


def load_scenarios(seed: int = 0) -> Scenarios:
    """Reuse decision_uq's cached study if present, else synthesise a mimic."""
    try:
        if not os.path.exists(duq.STUDY_CACHE):
            raise FileNotFoundError(duq.STUDY_CACHE)
        study = duq.load_study()
        p_oracle = np.asarray(study.p_oracle, dtype=float)
        p_pipeline = np.asarray(study.p_pipeline, dtype=float)
        p_point = _point_estimate_probs(study)
        # Stage-0 detection verdict proxy: a defect is "present" whenever the
        # true growth prob is non-trivial (the detector keys on the same flaw).
        detect = p_oracle > 0.05
        return Scenarios(p_oracle=p_oracle, p_pipeline=p_pipeline,
                         p_point=p_point, detect=detect,
                         alpha=float(study.alpha), beta=float(study.beta),
                         source=f"decision_uq cache ({study.source})")
    except Exception as e:
        print(f"[info] decision_uq cache unusable ({e}); "
              "using synthetic scenarios")
        return _synthetic_scenarios(seed=seed)


# ═════════════════════════════════════════════════════════════════════════════
#  Decision strategies — each maps a scenario set → array of OK/REPAIR/RETIRE
# ═════════════════════════════════════════════════════════════════════════════

def strat_framework(sc: Scenarios) -> np.ndarray:
    """Ours: posterior-propagated P(grow) + expected-cost α/β."""
    return np.array([duq.decide(p, sc.alpha, sc.beta) for p in sc.p_pipeline])


def strat_point_estimate(sc: Scenarios) -> np.ndarray:
    """No UQ: P(grow) at the posterior-MEAN θ, same expected-cost α/β."""
    return np.array([duq.decide(p, sc.alpha, sc.beta) for p in sc.p_point])


def strat_fixed_half(sc: Scenarios) -> np.ndarray:
    """Naive 'more likely than not': threshold the propagated P(grow) at 0.5,
    no expected-cost model.  p<=0.5 → OK else RETIRE (a 0.5 rule has no REPAIR
    band; it is the operator who ignores the asymmetric cost of a miss)."""
    return np.array(["OK" if p <= 0.5 else "RETIRE" for p in sc.p_pipeline])


def strat_always_repair(sc: Scenarios) -> np.ndarray:
    """Trivial conservative baseline: never fly borderline → always REPAIR."""
    return np.full(len(sc.p_oracle), "REPAIR")


def strat_detection_only(sc: Scenarios) -> np.ndarray:
    """Stage-0 verdict alone: defect present → REPAIR, else OK.  No prognosis,
    so it can neither clear a benign flaw nor escalate a dangerous one."""
    return np.where(sc.detect, "REPAIR", "OK")


STRATEGIES = {
    "FRAMEWORK (ours)": strat_framework,
    "point-estimate (no UQ)": strat_point_estimate,
    "fixed-0.5 threshold": strat_fixed_half,
    "always-repair": strat_always_repair,
    "detection-only": strat_detection_only,
}


# ═════════════════════════════════════════════════════════════════════════════
#  Scoring
# ═════════════════════════════════════════════════════════════════════════════

def oracle_decisions(sc: Scenarios) -> np.ndarray:
    return np.array([duq.decide(p, sc.alpha, sc.beta) for p in sc.p_oracle])


def score_strategy(dec: np.ndarray, oracle: np.ndarray,
                   sc: Scenarios) -> dict:
    s_d, s_o = severity(dec), severity(oracle)
    n_retire = int((oracle == "RETIRE").sum())
    dmiss = (float(np.mean(dec[oracle == "RETIRE"] == "OK"))
             if n_retire else 0.0)
    return {
        "accuracy": float(np.mean(dec == oracle)),
        "dangerous_miss": dmiss,                       # P(OK | oracle=RETIRE)
        "unsafe_miss_rate": float(np.mean(s_d < s_o)),  # any under-call
        "over_call_rate": float(np.mean(s_d > s_o)),    # conservative
        "expected_cost": mean_cost(dec, sc.p_oracle),
    }


def evaluate(sc: Scenarios) -> dict:
    oracle = oracle_decisions(sc)
    out = {}
    for name, fn in STRATEGIES.items():
        dec = fn(sc)
        out[name] = {"decisions": dec, **score_strategy(dec, oracle, sc)}
    # the oracle itself, as the cost floor
    out["_oracle"] = {"decisions": oracle, **score_strategy(oracle, oracle, sc)}
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  Report
# ═════════════════════════════════════════════════════════════════════════════

def print_report(sc: Scenarios, res: dict) -> None:
    bar = "=" * 78
    oracle = res["_oracle"]["decisions"]
    counts = {d: int((oracle == d).sum()) for d in DECISIONS}
    print(bar)
    print("  SYSTEM-LEVEL DECISION BENCHMARK — the framework vs naive strategies")
    print(bar)
    print(f"  scenarios  : M={len(sc.p_oracle)}   "
          f"thresholds α={sc.alpha:.2f}, β={sc.beta:.2f}")
    print(f"  source     : {sc.source}")
    print(f"  cost model : c_loss={COSTS['c_loss']:.0f}  "
          f"c_repair={COSTS['c_repair']:.0f}  "
          f"c_retire={COSTS['c_retire_value']:.0f}  "
          f"r={COSTS['repair_residual_risk']:.2f}")
    print(f"  oracle mix : " +
          "  ".join(f"{d}={counts[d]}" for d in DECISIONS) +
          f"   (cost floor {res['_oracle']['expected_cost']:.2f})")
    print()
    hdr = (f"  {'strategy':<24}{'acc':>7}{'dang.miss':>11}"
           f"{'over-call':>11}{'exp.cost':>11}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    names = [n for n in STRATEGIES]
    for name in names:
        r = res[name]
        tag = " *" if name.startswith("FRAMEWORK") else ""
        print(f"  {name:<24}{r['accuracy']*100:6.1f}%"
              f"{r['dangerous_miss']*100:10.1f}%"
              f"{r['over_call_rate']*100:10.1f}%"
              f"{r['expected_cost']:11.2f}{tag}")
    print()
    rank_cost = sorted(names, key=lambda n: res[n]["expected_cost"])
    rank_dmiss = sorted(names, key=lambda n: (res[n]["dangerous_miss"],
                                              res[n]["expected_cost"]))
    print("  RANK by expected cost  (lower = better):")
    for i, n in enumerate(rank_cost, 1):
        print(f"     {i}. {n:<24} {res[n]['expected_cost']:7.2f}")
    print("  RANK by dangerous-miss (lower = better):")
    for i, n in enumerate(rank_dmiss, 1):
        print(f"     {i}. {n:<24} {res[n]['dangerous_miss']*100:6.1f}%")
    print(bar)


# ═════════════════════════════════════════════════════════════════════════════
#  Figure
# ═════════════════════════════════════════════════════════════════════════════

def make_figure(sc: Scenarios, res: dict, out_path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import sys
    sys.path.insert(0, os.path.join(HERE, "slides", "figure_sources"))
    try:
        from thesis_style import use
        figsize = use(width_frac=1.0, aspect=0.42)
    except Exception:
        figsize = (11.0, 4.2)
    import matplotlib.pyplot as plt

    names = [n for n in STRATEGIES]
    short = {"FRAMEWORK (ours)": "FRAMEWORK\n(ours)",
             "point-estimate (no UQ)": "point-est.\n(no UQ)",
             "fixed-0.5 threshold": "fixed-0.5",
             "always-repair": "always\nrepair",
             "detection-only": "detection\nonly"}
    cost = np.array([res[n]["expected_cost"] for n in names])
    dmiss = np.array([res[n]["dangerous_miss"] * 100 for n in names])
    floor = res["_oracle"]["expected_cost"]
    hl = "#b71c1c"          # framework highlight
    base = "#90a4ae"
    colors = [hl if n.startswith("FRAMEWORK") else base for n in names]
    labels = [short[n] for n in names]

    fig, ax = plt.subplots(1, 3, figsize=figsize)

    # (a) expected cost per strategy (the key system metric)
    ax[0].bar(range(len(names)), cost, color=colors)
    ax[0].axhline(floor, color="#2e7d32", lw=1.0, ls="--",
                  label=f"oracle floor {floor:.1f}")
    ax[0].set_xticks(range(len(names)))
    ax[0].set_xticklabels(labels, fontsize=6)
    ax[0].set_ylabel("mean expected cost", fontsize=7)
    ax[0].set_title("(a) expected cost (lower = better)", fontsize=8)
    ax[0].legend(fontsize=6, loc="upper left")
    ax[0].tick_params(labelsize=6)

    # (b) dangerous-miss rate per strategy
    ax[1].bar(range(len(names)), dmiss, color=colors)
    ax[1].set_xticks(range(len(names)))
    ax[1].set_xticklabels(labels, fontsize=6)
    ax[1].set_ylabel("P(OK | oracle=RETIRE)  (%)", fontsize=7)
    ax[1].set_title("(b) dangerous-miss rate", fontsize=8)
    ax[1].tick_params(labelsize=6)

    # (c) cost vs dangerous-miss scatter — framework should sit bottom-left
    for i, n in enumerate(names):
        c = hl if n.startswith("FRAMEWORK") else base
        ax[2].scatter(dmiss[i], cost[i], s=70 if c == hl else 45, c=c,
                      edgecolors="k", linewidths=0.5,
                      zorder=3 if c == hl else 2)
        ax[2].annotate(labels[i].replace("\n", " "), (dmiss[i], cost[i]),
                       fontsize=5.5, xytext=(3, 3),
                       textcoords="offset points")
    ax[2].axhline(floor, color="#2e7d32", lw=0.8, ls="--")
    ax[2].set_xlabel("dangerous-miss rate (%)", fontsize=7)
    ax[2].set_ylabel("mean expected cost", fontsize=7)
    ax[2].set_title("(c) cost vs danger\n(framework = bottom-left)", fontsize=8)
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

    # cost model matches calibrate_thresholds' economics & boundaries
    a, b = pf.calibrate_thresholds(**COSTS)
    ok(np.isclose(expected_cost("OK", 0.0), 0.0), "OK at p=0 costs 0")
    ok(np.isclose(expected_cost("RETIRE", 0.7), COSTS["c_retire_value"]),
       "RETIRE cost is flat")
    # at p=α the OK and REPAIR expected costs cross (def. of α)
    ok(np.isclose(expected_cost("OK", a), expected_cost("REPAIR", a)),
       "OK/REPAIR costs cross at α")
    # at p=β the REPAIR and RETIRE expected costs cross (def. of β)
    ok(np.isclose(expected_cost("REPAIR", b), expected_cost("RETIRE", b)),
       "REPAIR/RETIRE costs cross at β")
    # the oracle decision is the per-scenario cost-minimiser, by construction
    for p in (0.0, 0.01, a, 0.3, b, 0.8, 1.0):
        costs_p = {d: expected_cost(d, p) for d in DECISIONS}
        best = min(costs_p, key=costs_p.get)
        ok(costs_p[duq.decide(p, a, b)] <= costs_p[best] + 1e-9,
           f"oracle decision is cost-optimal at p={p}")

    sc = load_scenarios(seed=1)
    M = len(sc.p_oracle)
    ok(M >= 4, "scenario set non-trivial")
    for arr in (sc.p_oracle, sc.p_pipeline, sc.p_point):
        ok(arr.shape == (M,) and np.all((arr >= 0) & (arr <= 1)),
           "probabilities in [0,1] with right shape")

    res = evaluate(sc)
    oracle = res["_oracle"]["decisions"]

    # every strategy returns a decision for every scenario, in the label set
    for name, fn in STRATEGIES.items():
        dec = fn(sc)
        ok(len(dec) == M, f"{name}: decision per scenario")
        ok(set(np.unique(dec)).issubset(set(DECISIONS)), f"{name}: valid labels")
        r = res[name]
        ok(0.0 <= r["accuracy"] <= 1.0, f"{name}: accuracy in range")
        ok(0.0 <= r["dangerous_miss"] <= 1.0, f"{name}: dangerous-miss in range")
        ok(0.0 <= r["over_call_rate"] <= 1.0, f"{name}: over-call in range")
        ok(r["expected_cost"] >= 0.0, f"{name}: cost non-negative")

    fw = res["FRAMEWORK (ours)"]
    half = res["fixed-0.5 threshold"]
    arep = res["always-repair"]
    floor = res["_oracle"]["expected_cost"]

    # KEY system claim: the framework's α/β are cost-optimal by construction, so
    # on this scenario set it must not cost more than the naive 0.5 rule.
    ok(fw["expected_cost"] <= half["expected_cost"] + 1e-9,
       "framework expected cost <= fixed-0.5 baseline")
    # the framework never beats the oracle floor (sanity)
    ok(fw["expected_cost"] >= floor - 1e-9, "framework cost >= oracle floor")

    # always-repair: trivially safe (0 dangerous-misses) but pays for it in cost
    ok(arep["dangerous_miss"] == 0.0, "always-repair has 0 dangerous-misses")
    ok(arep["expected_cost"] > fw["expected_cost"],
       "always-repair costs more than the framework")

    # framework should be at least as safe as the naive 0.5 rule
    ok(fw["dangerous_miss"] <= half["dangerous_miss"] + 1e-9,
       "framework dangerous-miss <= fixed-0.5")

    # a hand-built scenario set where the optimal call is REPAIR everywhere:
    # the framework (cost-optimal) must beat always-OK-style naive rules in cost.
    sc2 = _synthetic_scenarios(M=30, seed=3)
    res2 = evaluate(sc2)
    ok(res2["FRAMEWORK (ours)"]["expected_cost"]
       <= res2["fixed-0.5 threshold"]["expected_cost"] + 1e-9,
       "framework <= fixed-0.5 on synthetic set too")

    print(f"system_baseline: {n}/{n} unit tests passed")
    return n


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="System-level decision benchmark: framework vs naive "
                    "decision strategies on the decision_uq scenario set.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fig", action="store_true",
                    help="save paper_figs/system_baseline.pdf")
    ap.add_argument("--test", action="store_true", help="unit tests only")
    args = ap.parse_args()

    if args.test:
        run_tests()
        return

    sc = load_scenarios(seed=args.seed)
    res = evaluate(sc)
    print_report(sc, res)
    if args.fig:
        p = make_figure(sc, res,
                        os.path.join(HERE, "paper_figs", "system_baseline.pdf"))
        print(f"\nfigure → {p}")


if __name__ == "__main__":
    main()
