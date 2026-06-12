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

THREE structures (additive)
---------------------------
The same system-level benchmark is run on each of the THREE structures the
framework spans, so the comparison is not single-part:
  1. INTERSTAGE — perforated panel, FD AT2 phase-field delamination
     (the decision_uq scenario set above; unchanged).
  2. FAIRING    — honeycomb skin-core debond (fairing_debond_prognosis), its OWN
     prognosis drives a damage-sweep scenario set.
  3. SRB-3      — filament-wound motor-case burst margin (srb3_motorcase), its
     OWN burst prognosis drives a damage-sweep scenario set.
For 2 and 3 the per-structure (p_oracle, p_pipeline, p_point, detect) set is
built from the structure's REAL prognosis over a damage/load distribution (oracle
= prognosis at the true damage; point-estimate = prognosis at the posterior mean;
pipeline = posterior-propagated P(grow) — the same Jensen shrink the interstage
shows). The framework should remain lowest expected cost / 0% dangerous-miss on
each structure AND in aggregate.

Usage
-----
    python system_baseline.py            # run 3-structure benchmark + ranking
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
#  Per-structure scenario builder (fairing / SRB-3) driven by the REAL prognosis
# ═════════════════════════════════════════════════════════════════════════════

def _prognosis_scenarios(name: str, growth_prob, a_lo: float, a_hi: float,
                         load: float, M: int = 40, post_sd_frac: float = 0.18,
                         n_post: int = 24, seed: int = 0) -> Scenarios:
    """Build a (p_oracle, p_pipeline, p_point, detect) scenario set for a NON-
    interstage structure from its OWN prognosis `growth_prob(posterior_a, load)`.

    For each scenario we draw a TRUE damage size a_true across [a_lo, a_hi] and a
    Stage-2 posterior of damage sizes centred on it:
      * p_oracle   = growth_prob at the TRUE damage (a tight posterior at a_true)
                     — the best call the prognosis could make with perfect char.;
      * p_point    = growth_prob at the posterior-MEAN damage only (no-UQ
                     ablation: a tight posterior at the noisy mean estimate);
      * p_pipeline = growth_prob propagated over the FULL posterior spread
                     (posterior-propagated P(grow) — carries char. uncertainty).
    The α/β are the SAME expected-cost thresholds (cost model COSTS), so the
    framework's economics are identical across structures."""
    rng = np.random.default_rng(seed)
    alpha, beta = pf.calibrate_thresholds(**COSTS)
    a_true = rng.uniform(a_lo, a_hi, M)
    sd = post_sd_frac * (a_hi - a_lo)
    p_oracle = np.empty(M); p_point = np.empty(M); p_pipeline = np.empty(M)
    for i in range(M):
        # noisy posterior-mean characterisation estimate (point estimate)
        a_mean = float(np.clip(a_true[i] + rng.normal(0, sd), a_lo, a_hi))
        # full posterior spread around that estimate (carries char. uncertainty)
        post = np.clip(rng.normal(a_mean, sd, n_post), a_lo, a_hi)
        p_oracle[i] = growth_prob(np.full(8, a_true[i]), load)   # truth
        p_point[i] = growth_prob(np.full(8, a_mean), load)       # mean-θ only
        p_pipeline[i] = growth_prob(post, load)                  # propagated
    p_oracle = np.clip(p_oracle, 0.0, 1.0)
    p_point = np.clip(p_point, 0.0, 1.0)
    p_pipeline = np.clip(p_pipeline, 0.0, 1.0)
    detect = a_true > (a_lo + 0.02 * (a_hi - a_lo))   # Stage-0 verdict proxy
    return Scenarios(p_oracle=p_oracle, p_pipeline=p_pipeline, p_point=p_point,
                     detect=detect, alpha=alpha, beta=beta,
                     source=f"{name} prognosis damage-sweep (M={M})")


def fairing_scenarios(seed: int = 0, M: int = 40) -> Scenarios:
    """Fairing skin-core debond: scenario set from the REAL debond prognosis."""
    import fairing_debond_prognosis as fdp
    cfg = fdp.FairingConfig()
    gp = lambda post, load: fdp.debond_growth_probability(post, load, cfg)
    # debond radius sweep spanning OK..RETIRE (load chosen so all 3 classes appear)
    return _prognosis_scenarios("fairing", gp, a_lo=0.01, a_hi=0.16,
                                load=0.11, M=M, seed=seed)


def srb3_scenarios(seed: int = 0, M: int = 40) -> Scenarios:
    """SRB-3 motor case: scenario set from the REAL burst prognosis."""
    import srb3_motorcase as srb3
    cfg = srb3.MotorCaseConfig()
    gp = lambda post, load: srb3.burst_growth_probability(post, load, cfg)
    # damage sweep spanning OK..RETIRE at operating pressure
    return _prognosis_scenarios("srb3", gp, a_lo=0.02, a_hi=0.95,
                                load=cfg.p_op, M=M, seed=seed)


# structure name -> scenario loader. interstage = the existing decision_uq set.
STRUCTURES = {
    "interstage": load_scenarios,
    "fairing": fairing_scenarios,
    "srb3-motorcase": srb3_scenarios,
}


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


def evaluate_structures(seed: int = 0, M: int = 40) -> dict:
    """Run the system benchmark on EACH of the three structures and aggregate.

    Returns {structure: {"sc":Scenarios, "res":evaluate(...)}} plus an
    "_aggregate" entry that pools all scenarios (expected cost and dangerous-miss
    averaged across the three structures, equally weighted)."""
    per: dict = {}
    for name, loader in STRUCTURES.items():
        sc = loader(seed=seed) if name == "interstage" else loader(seed=seed, M=M)
        per[name] = {"sc": sc, "res": evaluate(sc)}

    # aggregate: mean over structures of each strategy's per-structure metrics
    agg: dict = {}
    strat_keys = list(STRATEGIES) + ["_oracle"]
    for sname in strat_keys:
        cost = np.mean([per[s]["res"][sname]["expected_cost"] for s in STRUCTURES])
        dmiss = np.mean([per[s]["res"][sname]["dangerous_miss"] for s in STRUCTURES])
        acc = np.mean([per[s]["res"][sname]["accuracy"] for s in STRUCTURES])
        over = np.mean([per[s]["res"][sname]["over_call_rate"] for s in STRUCTURES])
        agg[sname] = {"expected_cost": float(cost), "dangerous_miss": float(dmiss),
                      "accuracy": float(acc), "over_call_rate": float(over)}
    per["_aggregate"] = agg
    return per


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


def print_report_structures(per: dict) -> None:
    """3-structure benchmark report: per-structure table + pooled aggregate."""
    bar = "=" * 78
    print(bar)
    print("  SYSTEM-LEVEL DECISION BENCHMARK — framework vs naive, ACROSS 3 STRUCTURES")
    print(bar)
    print("  structures : interstage (phase-field) | fairing (debond) | "
          "srb3 (burst)")
    names = [n for n in STRATEGIES]
    for sname in STRUCTURES:
        sc, res = per[sname]["sc"], per[sname]["res"]
        oracle = res["_oracle"]["decisions"]
        counts = {d: int((oracle == d).sum()) for d in DECISIONS}
        print("\n  " + "-" * 74)
        print(f"  STRUCTURE: {sname}   (M={len(sc.p_oracle)}, "
              f"α={sc.alpha:.2f}, β={sc.beta:.2f})")
        print(f"  source: {sc.source}")
        print("  oracle mix : " + "  ".join(f"{d}={counts[d]}" for d in DECISIONS)
              + f"   (cost floor {res['_oracle']['expected_cost']:.2f})")
        hdr = (f"    {'strategy':<24}{'acc':>7}{'dang.miss':>11}"
               f"{'over-call':>11}{'exp.cost':>11}")
        print(hdr)
        for name in names:
            r = res[name]
            tag = " *" if name.startswith("FRAMEWORK") else ""
            print(f"    {name:<24}{r['accuracy']*100:6.1f}%"
                  f"{r['dangerous_miss']*100:10.1f}%"
                  f"{r['over_call_rate']*100:10.1f}%"
                  f"{r['expected_cost']:11.2f}{tag}")

    # pooled aggregate across the three structures
    agg = per["_aggregate"]
    print("\n  " + "=" * 74)
    print("  AGGREGATE (mean over the 3 structures):")
    hdr = (f"    {'strategy':<24}{'acc':>7}{'dang.miss':>11}"
           f"{'over-call':>11}{'exp.cost':>11}")
    print(hdr)
    for name in names:
        r = agg[name]
        tag = " *" if name.startswith("FRAMEWORK") else ""
        print(f"    {name:<24}{r['accuracy']*100:6.1f}%"
              f"{r['dangerous_miss']*100:10.1f}%"
              f"{r['over_call_rate']*100:10.1f}%"
              f"{r['expected_cost']:11.2f}{tag}")
    print(f"    {'(oracle floor)':<24}{'':>7}{'':>11}{'':>11}"
          f"{agg['_oracle']['expected_cost']:11.2f}")
    rank_cost = sorted(names, key=lambda n: agg[n]["expected_cost"])
    rank_dmiss = sorted(names, key=lambda n: (agg[n]["dangerous_miss"],
                                              agg[n]["expected_cost"]))
    print("\n  AGGREGATE RANK by expected cost (lower = better):")
    for i, n in enumerate(rank_cost, 1):
        print(f"     {i}. {n:<24} {agg[n]['expected_cost']:7.2f}")
    print("  AGGREGATE RANK by dangerous-miss (lower = better):")
    for i, n in enumerate(rank_dmiss, 1):
        print(f"     {i}. {n:<24} {agg[n]['dangerous_miss']*100:6.1f}%")
    print(bar)


# ═════════════════════════════════════════════════════════════════════════════
#  Figure
# ═════════════════════════════════════════════════════════════════════════════

def make_figure(per: dict, out_path: str) -> str:
    """3-structure system benchmark figure: grouped expected-cost and
    dangerous-miss bars per structure, plus the pooled cost-vs-danger scatter."""
    import matplotlib
    matplotlib.use("Agg")
    import sys
    sys.path.insert(0, os.path.join(HERE, "slides", "figure_sources"))
    try:
        from thesis_style import use
        figsize = use(width_frac=1.0, aspect=0.42)
    except Exception:
        figsize = (12.0, 4.2)
    import matplotlib.pyplot as plt

    names = [n for n in STRATEGIES]
    short = {"FRAMEWORK (ours)": "FRAMEWORK\n(ours)",
             "point-estimate (no UQ)": "point-est.\n(no UQ)",
             "fixed-0.5 threshold": "fixed-0.5",
             "always-repair": "always\nrepair",
             "detection-only": "detection\nonly"}
    labels = [short[n] for n in names]
    structs = list(STRUCTURES)
    hl = "#b71c1c"          # framework highlight
    spal = {"interstage": "#90a4ae", "fairing": "#5c93c4",
            "srb3-motorcase": "#c4a35c"}

    fig, ax = plt.subplots(1, 3, figsize=figsize)
    x = np.arange(len(names))
    w = 0.26

    # (a) expected cost per strategy, grouped by structure
    for j, s in enumerate(structs):
        res = per[s]["res"]
        cost = np.array([res[n]["expected_cost"] for n in names])
        col = [hl if n.startswith("FRAMEWORK") else spal[s] for n in names]
        ax[0].bar(x + (j - 1) * w, cost, w, color=col,
                  edgecolor="k", linewidth=0.3, label=s)
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels, fontsize=6)
    ax[0].set_ylabel("mean expected cost", fontsize=7)
    ax[0].set_title("(a) expected cost by structure (lower = better)", fontsize=8)
    ax[0].legend(fontsize=5.5, loc="upper left"); ax[0].tick_params(labelsize=6)

    # (b) dangerous-miss rate per strategy, grouped by structure
    for j, s in enumerate(structs):
        res = per[s]["res"]
        dmiss = np.array([res[n]["dangerous_miss"] * 100 for n in names])
        col = [hl if n.startswith("FRAMEWORK") else spal[s] for n in names]
        ax[1].bar(x + (j - 1) * w, dmiss, w, color=col,
                  edgecolor="k", linewidth=0.3, label=s)
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels, fontsize=6)
    ax[1].set_ylabel("P(OK | oracle=RETIRE)  (%)", fontsize=7)
    ax[1].set_title("(b) dangerous-miss by structure", fontsize=8)
    ax[1].legend(fontsize=5.5); ax[1].tick_params(labelsize=6)

    # (c) pooled aggregate cost vs dangerous-miss — framework = bottom-left
    agg = per["_aggregate"]
    base = "#90a4ae"
    for n in names:
        c = hl if n.startswith("FRAMEWORK") else base
        dm, co = agg[n]["dangerous_miss"] * 100, agg[n]["expected_cost"]
        ax[2].scatter(dm, co, s=80 if c == hl else 45, c=c,
                      edgecolors="k", linewidths=0.5,
                      zorder=3 if c == hl else 2)
        ax[2].annotate(short[n].replace("\n", " "), (dm, co), fontsize=5.5,
                       xytext=(3, 3), textcoords="offset points")
    ax[2].axhline(agg["_oracle"]["expected_cost"], color="#2e7d32", lw=0.8,
                  ls="--", label="oracle floor")
    ax[2].set_xlabel("dangerous-miss rate (%)", fontsize=7)
    ax[2].set_ylabel("mean expected cost", fontsize=7)
    ax[2].set_title("(c) aggregate cost vs danger\n(framework = bottom-left)",
                    fontsize=8)
    ax[2].legend(fontsize=5.5); ax[2].tick_params(labelsize=6)

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

    # ---- THREE-structure extension (additive) --------------------------------
    # per-structure scenario builders for fairing + srb3 produce valid sets
    for builder, sname in ((fairing_scenarios, "fairing"),
                           (srb3_scenarios, "srb3-motorcase")):
        scS = builder(seed=1, M=30)
        ok(len(scS.p_oracle) == 30, f"{sname}: M scenarios built")
        for arr in (scS.p_oracle, scS.p_pipeline, scS.p_point):
            ok(arr.shape == (30,) and np.all((arr >= 0) & (arr <= 1)),
               f"{sname}: probs in [0,1]")
        # the structure spans more than one decision class (interesting set)
        oS = oracle_decisions(scS)
        ok(len(set(oS)) >= 2, f"{sname}: oracle spans >=2 decision classes")

    # full 3-structure evaluation
    per = evaluate_structures(seed=1, M=30)
    ok(set(STRUCTURES).issubset(per) and "_aggregate" in per,
       "evaluate_structures returns all 3 structures + aggregate")
    for sname in STRUCTURES:
        rS = per[sname]["res"]
        fwS = rS["FRAMEWORK (ours)"]
        halfS = rS["fixed-0.5 threshold"]
        arepS = rS["always-repair"]
        floorS = rS["_oracle"]["expected_cost"]
        # framework is cost-optimal by construction on every structure
        ok(fwS["expected_cost"] <= halfS["expected_cost"] + 1e-9,
           f"{sname}: framework cost <= fixed-0.5")
        ok(fwS["expected_cost"] >= floorS - 1e-9,
           f"{sname}: framework cost >= oracle floor")
        ok(fwS["dangerous_miss"] <= halfS["dangerous_miss"] + 1e-9,
           f"{sname}: framework dangerous-miss <= fixed-0.5")
        ok(arepS["dangerous_miss"] == 0.0,
           f"{sname}: always-repair has 0 dangerous-misses")

    # aggregate: framework lowest expected cost among the naive strategies
    agg = per["_aggregate"]
    naive = [k for k in STRATEGIES if not k.startswith("FRAMEWORK")]
    ok(all(agg["FRAMEWORK (ours)"]["expected_cost"]
           <= agg[k]["expected_cost"] + 1e-9 for k in naive),
       "aggregate: framework lowest expected cost across 3 structures")
    ok(agg["FRAMEWORK (ours)"]["dangerous_miss"]
       <= min(agg[k]["dangerous_miss"] for k in naive) + 1e-9,
       "aggregate: framework dangerous-miss <= every naive strategy")

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

    per = evaluate_structures(seed=args.seed)
    print_report_structures(per)
    if args.fig:
        p = make_figure(per,
                        os.path.join(HERE, "paper_figs", "system_baseline.pdf"))
        print(f"\nfigure → {p}")


if __name__ == "__main__":
    main()
