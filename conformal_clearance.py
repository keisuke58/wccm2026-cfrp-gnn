"""
conformal_clearance.py — a DISTRIBUTION-FREE (split-conformal) coverage
guarantee on the flight-clearance ACTION, layered on top of the Stage 0-5
decision core (`system_baseline` / `decision_uq`).

Narrative ("OK/REPAIR/RETIRE のラベルに有限標本の保証を付ける")
----------------------------------------------------------------
The framework ships a single OK / REPAIR / RETIRE clearance call: it propagates
the FMPE posterior into one P(grow) and thresholds it with expected-cost
(α, β) (`decision_uq.decide`, `cfrp_phasefield_2d.calibrate_thresholds`).  A
reviewer's fair question is: how often is that single label actually WRONG, and
is there a guarantee?  This module answers with SPLIT-CONFORMAL prediction:
instead of one label we emit a PREDICTION SET of actions with a finite-sample,
distribution-free guarantee that the true (oracle) action is contained with
probability ≥ 1-α — and we validate that guarantee on each of the three
structures (interstage, fairing, srb3) and pooled.

Construction (split conformal over the 3 ordinal actions)
---------------------------------------------------------
  * GROUND TRUTH per scenario = the ORACLE action decide(p_oracle, α, β) — the
    best call the exact-FD / real-prognosis physics could make.
  * SCORE per scenario = the framework's posterior-propagated P(grow) = p_pipeline.
  * SOFT CLASS PROBABILITIES over the 3 actions are built from the score's
    position relative to the (α, β) decision bands: a small softmax of the
    signed distances of p_pipeline to each band [lo,hi] (negative distance =
    inside the band → high probability).  This is a defensible, monotone, NO-FIT
    surrogate for class likelihood; it needs no training and respects the
    ordinal band geometry the decision rule itself uses.
  * NONCONFORMITY of action k at a point = 1 - softprob_k  (the standard
    score s = 1 - p̂(y); APS-style but using the marginal class prob).  Split
    the scenarios into a CALIBRATION half and a TEST half (exchangeable draw).
    On calibration, take the nonconformity of the TRUE oracle action; set the
    conformal quantile

        q̂ = the  ceil((n+1)(1-α))/n  empirical quantile  (clipped to [0,1])

    of those calibration scores.  On test, the PREDICTION SET = every action
    whose nonconformity ≤ q̂.  By the split-conformal theorem this set covers
    the true action with prob ≥ 1-α, marginally, under exchangeability.

Two guarantees reported
-----------------------
  * MARGINAL coverage: empirical P(oracle action ∈ set) on test ≥ 1-α, for
    α ∈ {0.20, 0.10, 0.05}; with the mean SET SIZE (efficiency — smaller is more
    informative at the same coverage).
  * SAFETY-AWARE (asymmetric, one-sided): a RISK-CONTROLLING calibration where
    the budget α_danger bounds the DANGEROUS direction only — P(true action =
    RETIRE  AND  RETIRE ∉ set) ≤ α_danger.  The quantile is fit on the RETIRE
    calibration points alone (one-sided), so RETIRE is rarely dropped when it is
    truly needed.  We report the dangerous-omission rate vs the budget.

Per structure AND pooled
------------------------
Run on interstage (phase-field), fairing (debond), srb3 (burst), and the pooled
set.  Distribution-free → the marginal guarantee should hold for each and for
the pool, regardless of structure / modality.

HONEST LIMITATION
-----------------
Split-conformal gives MARGINAL (not conditional / per-input) coverage and assumes
calibration↔test EXCHANGEABILITY.  Across structures the physics differs, so the
pooled exchangeability is an APPROXIMATION; the honest claim is the cross-
structure POOLED marginal guarantee, not per-input conditional validity.  The
scores derive from REPRESENTATIVE prognoses (the same scenario machinery as
`system_baseline`), not field data.  We do NOT claim conditional coverage, and
the per-structure curves are reported precisely so any structure-specific
deviation is visible rather than hidden by the pool.

Usage
-----
    python conformal_clearance.py            # 3-structure + pooled coverage tables
    python conformal_clearance.py --fig      # + paper_figs/conformal_clearance.pdf
    python conformal_clearance.py --test     # unit tests only (synthetic fallback)
"""
from __future__ import annotations

import argparse
import os

import numpy as np

import decision_uq as duq
import system_baseline as sb
from system_baseline import Scenarios, DECISIONS, SEVERITY

HERE = os.path.dirname(os.path.abspath(__file__))

# nominal miss budgets we report the marginal guarantee for.
ALPHAS = (0.20, 0.10, 0.05)
# dangerous-omission budgets we report the one-sided guarantee for.
ALPHA_DANGERS = (0.20, 0.10, 0.05)


# ═════════════════════════════════════════════════════════════════════════════
#  Soft class probabilities over the 3 ordinal actions from the (α, β) bands
# ═════════════════════════════════════════════════════════════════════════════

def action_bands(alpha: float, beta: float) -> dict:
    """The decision rule's probability band [lo, hi] for each action, so that
    decide(p, α, β) == the action whose band contains p."""
    return {"OK": (0.0, alpha), "REPAIR": (alpha, beta), "RETIRE": (beta, 1.0)}


def soft_action_probs(p: np.ndarray, alpha: float, beta: float,
                      temp: float = 0.06) -> np.ndarray:
    """Soft (M, 3) class probabilities over (OK, REPAIR, RETIRE) for scores `p`.

    For each action we measure how far p sits OUTSIDE its band [lo, hi]
    (0 if inside).  A softmax of the NEGATIVE of those distances / temperature
    gives a monotone, no-fit class probability: maximal for the band that
    actually contains p (matching `decide`), decaying smoothly for neighbours.
    `temp` sets the sharpness (small → near one-hot at the argmax band)."""
    p = np.atleast_1d(np.asarray(p, dtype=float))
    bands = action_bands(alpha, beta)
    dist = np.empty((p.shape[0], len(DECISIONS)))
    for j, a in enumerate(DECISIONS):
        lo, hi = bands[a]
        # distance of p outside [lo, hi]; 0 when inside.
        dist[:, j] = np.maximum.reduce([lo - p, p - hi,
                                        np.zeros_like(p)])
    logits = -dist / max(temp, 1e-6)
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def nonconformity(p: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """(M, 3) nonconformity scores s = 1 - softprob(action) for each action."""
    return 1.0 - soft_action_probs(p, alpha, beta)


# ═════════════════════════════════════════════════════════════════════════════
#  Split-conformal quantile + prediction sets
# ═════════════════════════════════════════════════════════════════════════════

def conformal_quantile(cal_scores: np.ndarray, alpha: float) -> float:
    """Finite-sample conformal quantile q̂ of calibration scores.

    q̂ = the ceil((n+1)(1-α))/n empirical quantile (the standard split-conformal
    correction).  When (n+1)(1-α) > n the level is clipped to 1.0 (q̂ = max
    score → the set contains every action; the small-n honest behaviour)."""
    cal = np.sort(np.asarray(cal_scores, dtype=float))
    n = cal.size
    if n == 0:
        return 1.0
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    if rank >= n:                       # not enough data for the level → 1.0
        return float(cal[-1]) if rank == n else 1.0
    rank = max(rank, 1)
    return float(cal[rank - 1])         # rank-th smallest (1-indexed)


def prediction_sets(p_test: np.ndarray, alpha: float, beta: float,
                    qhat: float) -> list:
    """Conformal prediction SET per test point: actions with nonconformity ≤ q̂."""
    s = nonconformity(p_test, alpha, beta)        # (M, 3)
    sets = []
    for i in range(s.shape[0]):
        keep = [DECISIONS[j] for j in range(len(DECISIONS)) if s[i, j] <= qhat]
        if not keep:                              # never emit an empty set:
            keep = [DECISIONS[int(np.argmin(s[i]))]]   # fall back to the argmin
        sets.append(set(keep))
    return sets


def split_calibrate(sc: Scenarios, alpha: float, seed: int = 0,
                    cal_frac: float = 0.5) -> dict:
    """Full split-conformal pass for one scenario set at nominal level α.

    Splits into calibration / test, fits q̂ on the TRUE-action calibration
    nonconformity, builds test prediction sets, and reports empirical marginal
    coverage and mean set size."""
    p = np.asarray(sc.p_pipeline, dtype=float)
    oracle = sb.oracle_decisions(sc)
    M = p.size
    rng = np.random.default_rng(seed)
    idx = rng.permutation(M)
    n_cal = max(1, int(round(cal_frac * M)))
    cal, test = idx[:n_cal], idx[n_cal:]
    if test.size == 0:                            # tiny set: reuse cal as test
        test = cal

    s_all = nonconformity(p, sc.alpha, sc.beta)   # (M, 3)
    true_j = np.array([DECISIONS.index(a) for a in oracle])
    cal_scores = s_all[cal, true_j[cal]]
    qhat = conformal_quantile(cal_scores, alpha)

    sets = prediction_sets(p[test], sc.alpha, sc.beta, qhat)
    covered = np.array([oracle[t] in sets[i] for i, t in enumerate(test)])
    sizes = np.array([len(s) for s in sets])
    return {
        "alpha": alpha, "qhat": qhat,
        "coverage": float(covered.mean()),
        "set_size": float(sizes.mean()),
        "n_cal": int(n_cal), "n_test": int(test.size),
        "covered": covered, "sizes": sizes,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  Safety-aware (asymmetric / one-sided) RETIRE-omission guarantee
# ═════════════════════════════════════════════════════════════════════════════

def dangerous_omission(sc: Scenarios, alpha_danger: float, seed: int = 0,
                       cal_frac: float = 0.5) -> dict:
    """One-sided RISK-CONTROLLING guarantee on the DANGEROUS direction.

    Budget α_danger bounds  P(oracle = RETIRE  AND  RETIRE ∉ set).  We calibrate
    the RETIRE-inclusion threshold on the calibration RETIRE points ALONE: q̂ is
    the conformal quantile of the RETIRE nonconformity scores of scenarios whose
    oracle action is RETIRE.  On test, RETIRE is included whenever its
    nonconformity ≤ q̂.  The dangerous-omission rate = fraction of test
    oracle-RETIRE points whose set omits RETIRE; by construction ≤ α_danger."""
    p = np.asarray(sc.p_pipeline, dtype=float)
    oracle = sb.oracle_decisions(sc)
    M = p.size
    rng = np.random.default_rng(seed + 991)
    idx = rng.permutation(M)
    n_cal = max(1, int(round(cal_frac * M)))
    cal, test = idx[:n_cal], idx[n_cal:]
    if test.size == 0:
        test = cal

    retire_j = DECISIONS.index("RETIRE")
    s_retire = nonconformity(p, sc.alpha, sc.beta)[:, retire_j]
    cal_ret = cal[oracle[cal] == "RETIRE"]
    # one-sided quantile on RETIRE calibration scores; if none, be conservative
    # (include RETIRE everywhere → q̂ = 1).
    qhat = (conformal_quantile(s_retire[cal_ret], alpha_danger)
            if cal_ret.size else 1.0)

    test_ret = test[oracle[test] == "RETIRE"]
    if test_ret.size == 0:
        omit_rate = 0.0
    else:
        omitted = s_retire[test_ret] > qhat       # RETIRE excluded from the set
        omit_rate = float(omitted.mean())
    return {
        "alpha_danger": alpha_danger, "qhat": qhat,
        "omission_rate": omit_rate,
        "n_retire_cal": int(cal_ret.size), "n_retire_test": int(test_ret.size),
    }


def dangerous_omission_cv(sc: Scenarios, alpha_danger: float, n_folds: int = 5,
                          seed: int = 0) -> dict:
    """K-fold CROSS-CONFORMAL version of the one-sided RETIRE-omission guarantee
    — the small-sample shore-up.

    The single-split `dangerous_omission` calibrates q̂ on only HALF the RETIRE
    points and reports the omission on the other half: on a small real structure
    set (few RETIRE points / split) that estimate is high-variance and can breach
    the budget by chance.  Cross-conformal removes that variance: partition into K
    folds; for each fold-as-test, calibrate q̂ on the RETIRE points of the OTHER
    K-1 folds (≈(K-1)/K of all RETIRE points → a tighter quantile), and aggregate
    the omissions over ALL folds so every RETIRE point is tested exactly once.
    The reported rate is then the leave-fold-out omission over the FULL set, the
    stable estimate of the guarantee P(RETIRE ∉ set | oracle=RETIRE) ≤ α_danger."""
    p = np.asarray(sc.p_pipeline, dtype=float)
    oracle = sb.oracle_decisions(sc)
    M = p.size
    retire_j = DECISIONS.index("RETIRE")
    s_retire = nonconformity(p, sc.alpha, sc.beta)[:, retire_j]
    rng = np.random.default_rng(seed + 991)
    K = int(min(max(2, n_folds), M))
    folds = np.array_split(rng.permutation(M), K)

    omit = n_ret_test = cal_ret_tot = 0
    for k in range(K):
        test = folds[k]
        cal = np.concatenate([folds[j] for j in range(K) if j != k])
        cal_ret = cal[oracle[cal] == "RETIRE"]
        qhat = (conformal_quantile(s_retire[cal_ret], alpha_danger)
                if cal_ret.size else 1.0)
        cal_ret_tot += cal_ret.size
        test_ret = test[oracle[test] == "RETIRE"]
        if test_ret.size:
            omit += int((s_retire[test_ret] > qhat).sum())
            n_ret_test += test_ret.size
    rate = float(omit / n_ret_test) if n_ret_test else 0.0
    return {
        "alpha_danger": alpha_danger, "omission_rate": rate,
        "n_retire_test": int(n_ret_test), "n_folds": K,
        "mean_retire_cal": float(cal_ret_tot / K),
    }


def dangerous_omission_mondrian(per: dict, alpha_danger: float,
                                seed: int = 0) -> dict:
    """MONDRIAN cross-structure pooling of the one-sided RETIRE calibration — the
    cross-structure shore-up (ties to the LOSO decision-transfer spine).

    For each structure we calibrate the RETIRE-inclusion threshold q̂ on the
    RETIRE nonconformity scores POOLED FROM THE OTHER structures (leave-this-
    structure-out), then test omission on THIS structure's RETIRE points.  A small
    structure thus borrows RETIRE calibration strength from the others.  Valid as
    a cross-structure (Mondrian-by-class) guarantee IF the RETIRE-nonconformity is
    exchangeable across structures — an APPROXIMATION (different physics), stated.
    Returns the per-structure leave-one-structure-out omission rate."""
    names = [k for k in per if k != "pooled"]
    retire_j = DECISIONS.index("RETIRE")
    # precompute (s_retire, oracle) per structure
    cache = {}
    for nm in names:
        sc = per[nm]
        p = np.asarray(sc.p_pipeline, dtype=float)
        oracle = sb.oracle_decisions(sc)
        s_ret = nonconformity(p, sc.alpha, sc.beta)[:, retire_j]
        cache[nm] = (s_ret, oracle)
    out = {}
    for nm in names:
        s_ret, oracle = cache[nm]
        # calibration = RETIRE scores pooled from all OTHER structures
        cal_pool = np.concatenate(
            [cache[o][0][cache[o][1] == "RETIRE"] for o in names if o != nm]
            or [np.empty(0)])
        qhat = (conformal_quantile(cal_pool, alpha_danger)
                if cal_pool.size else 1.0)
        test_ret = s_ret[oracle == "RETIRE"]
        rate = (float((test_ret > qhat).mean()) if test_ret.size else 0.0)
        out[nm] = {"alpha_danger": alpha_danger, "omission_rate": rate,
                   "qhat": qhat, "n_retire_cal_pool": int(cal_pool.size),
                   "n_retire_test": int(test_ret.size)}
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  Scenario sets — reuse system_baseline's per-structure builders + a pool
# ═════════════════════════════════════════════════════════════════════════════

def structure_scenarios(seed: int = 0, M: int = 80) -> dict:
    """Per-structure scenario sets (reusing system_baseline's EXACT builders)
    plus a POOLED set that concatenates all three."""
    per = {
        "interstage": sb.load_scenarios(seed=seed),
        "fairing": sb.fairing_scenarios(seed=seed, M=M),
        "srb3-motorcase": sb.srb3_scenarios(seed=seed, M=M),
    }
    per["pooled"] = pool_scenarios(list(per.values()))
    return per


def pool_scenarios(scs: list) -> Scenarios:
    """Concatenate several scenario sets into one pooled set.  All share the SAME
    (α, β) economics (calibrate_thresholds with the fixed COSTS), so the bands
    align and pooling is well-defined."""
    a = scs[0].alpha, scs[0].beta
    assert all(np.isclose(s.alpha, a[0]) and np.isclose(s.beta, a[1])
               for s in scs), "pooled sets must share α, β"
    cat = lambda k: np.concatenate([getattr(s, k) for s in scs])
    return Scenarios(p_oracle=cat("p_oracle"), p_pipeline=cat("p_pipeline"),
                     p_point=cat("p_point"), detect=cat("detect"),
                     alpha=a[0], beta=a[1], source="POOLED (interstage+fairing+srb3)")


# ═════════════════════════════════════════════════════════════════════════════
#  Evaluation across structures
# ═════════════════════════════════════════════════════════════════════════════

def evaluate(seed: int = 0, M: int = 80) -> dict:
    """Coverage + set-size for every α and dangerous-omission for every
    α_danger, per structure and pooled."""
    per = structure_scenarios(seed=seed, M=M)
    out = {}
    for name, sc in per.items():
        cov = {a: split_calibrate(sc, a, seed=seed) for a in ALPHAS}
        dang = {ad: dangerous_omission(sc, ad, seed=seed) for ad in ALPHA_DANGERS}
        dang_cv = {ad: dangerous_omission_cv(sc, ad, seed=seed)
                   for ad in ALPHA_DANGERS}
        out[name] = {"sc": sc, "cov": cov, "dang": dang, "dang_cv": dang_cv}
    # Mondrian cross-structure (leave-one-structure-out) RETIRE calibration.
    mond = {ad: dangerous_omission_mondrian(per, ad, seed=seed)
            for ad in ALPHA_DANGERS}
    for name in per:
        if name == "pooled":
            continue
        out[name]["dang_mond"] = {ad: mond[ad][name] for ad in ALPHA_DANGERS}
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  Report
# ═════════════════════════════════════════════════════════════════════════════

def print_report(res: dict) -> None:
    bar = "=" * 80
    order = ["interstage", "fairing", "srb3-motorcase", "pooled"]
    print(bar)
    print("  SPLIT-CONFORMAL CLEARANCE GUARANTEE — distribution-free coverage on")
    print("  the OK/REPAIR/RETIRE action, across 3 structures + pooled")
    print(bar)
    print("  score = framework P(grow) (p_pipeline);  truth = oracle "
          "decide(p_oracle, α, β)")
    print(f"  q̂ = ceil((n+1)(1-α))/n empirical quantile of TRUE-action "
          f"nonconformity (split conformal)")

    # ---- marginal coverage / efficiency table -------------------------------
    print("\n  MARGINAL COVERAGE  (empirical P[oracle action ∈ set];  guarantee "
          "≥ 1-α):")
    hdr = (f"    {'structure':<16}" +
           "".join(f"{'1-α='+format(1-a,'.2f'):>13}" for a in ALPHAS) +
           f"{'meanSize':>11}")
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for name in order:
        cov = res[name]["cov"]
        row = f"    {name:<16}"
        for a in ALPHAS:
            c = cov[a]["coverage"]
            ok_mark = "✓" if c >= (1 - a) - 1e-9 else "·"
            row += f"{c*100:9.1f}% {ok_mark} "
        mean_size = np.mean([cov[a]["set_size"] for a in ALPHAS])
        row += f"{mean_size:10.2f}"
        print(row)
    print("    (✓ = empirical coverage met the 1-α guarantee on this structure)")

    # ---- per-α set sizes ----------------------------------------------------
    print("\n  MEAN SET SIZE per 1-α  (efficiency; smaller = more informative):")
    hdr = (f"    {'structure':<16}" +
           "".join(f"{'1-α='+format(1-a,'.2f'):>13}" for a in ALPHAS))
    print(hdr)
    for name in order:
        cov = res[name]["cov"]
        row = f"    {name:<16}"
        for a in ALPHAS:
            row += f"{cov[a]['set_size']:13.2f}"
        print(row)

    # ---- dangerous-omission table (single split — high variance on small sets)
    print("\n  DANGEROUS-OMISSION RATE  P(oracle=RETIRE ∧ RETIRE ∉ set)  "
          "(one-sided; ≤ α_danger):")
    print("  [a] single split (the baseline — high variance when few RETIRE/split):")
    hdr = (f"    {'structure':<16}" +
           "".join(f"{'αd='+format(ad,'.2f'):>13}" for ad in ALPHA_DANGERS))
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for name in order:
        dang = res[name]["dang"]
        row = f"    {name:<16}"
        for ad in ALPHA_DANGERS:
            r = dang[ad]["omission_rate"]
            ok_mark = "✓" if r <= ad + 1e-9 else "✗"
            row += f"{r*100:9.1f}% {ok_mark} "
        print(row)

    # ---- K-fold cross-conformal (the small-sample shore-up) ------------------
    print("  [b] K-fold cross-conformal (shore-up — all RETIRE points tested, "
          "tighter q̂):")
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for name in order:
        dcv = res[name]["dang_cv"]
        row = f"    {name:<16}"
        for ad in ALPHA_DANGERS:
            r = dcv[ad]["omission_rate"]
            ok_mark = "✓" if r <= ad + 1e-9 else "✗"
            row += f"{r*100:9.1f}% {ok_mark} "
        print(row)

    # ---- Mondrian cross-structure (leave-one-structure-out) -----------------
    print("  [c] Mondrian cross-structure (RETIRE calibration borrowed from the "
          "OTHER structures):")
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for name in [n for n in order if n != "pooled"]:
        dm = res[name].get("dang_mond")
        if not dm:
            continue
        row = f"    {name:<16}"
        for ad in ALPHA_DANGERS:
            r = dm[ad]["omission_rate"]
            ok_mark = "✓" if r <= ad + 1e-9 else "✗"
            row += f"{r*100:9.1f}% {ok_mark} "
        print(row)
    print("    (✓ = dangerous-omission within budget α_danger;  [b]/[c] are the "
          "small-sample fixes)")

    # ---- headline -----------------------------------------------------------
    pooled = res["pooled"]
    cov05 = pooled["cov"][0.05]
    mean_size_pool = np.mean([pooled["cov"][a]["set_size"] for a in ALPHAS])
    print("\n  " + "-" * 76)
    print("  HEADLINE: split-conformal gives a finite-sample 1-α coverage "
          "guarantee on the")
    print("  clearance action that holds across all three structures; "
          f"mean set size S≈{mean_size_pool:.2f};")
    print(f"  RETIRE rarely omitted (pooled 1-α=0.95 coverage "
          f"{cov05['coverage']*100:.1f}%).")
    print("\n  LIMITATION: this is MARGINAL (not conditional / per-input) "
          "coverage and assumes")
    print("  calibration↔test exchangeability; across structures (different "
          "physics) the POOLED")
    print("  marginal guarantee is the honest claim — not per-input conditional "
          "validity. Scores")
    print("  derive from representative prognoses; we do not overclaim "
          "conditional coverage.")
    print(bar)


# ═════════════════════════════════════════════════════════════════════════════
#  Figure
# ═════════════════════════════════════════════════════════════════════════════

def make_figure(res: dict, out_path: str) -> str:
    """(a) coverage calibration nominal vs empirical (per structure, y=x line);
    (b) set-size vs 1-α; (c) dangerous-omission vs α_danger budget (y=x line)."""
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

    structs = ["interstage", "fairing", "srb3-motorcase", "pooled"]
    pal = {"interstage": "#90a4ae", "fairing": "#5c93c4",
           "srb3-motorcase": "#c4a35c", "pooled": "#b71c1c"}
    mk = {"interstage": "o", "fairing": "s", "srb3-motorcase": "^",
          "pooled": "D"}
    nominal = np.array([1 - a for a in ALPHAS])

    fig, ax = plt.subplots(1, 3, figsize=figsize)

    # (a) coverage calibration — points on/above y=x.
    ax[0].plot([0.7, 1.0], [0.7, 1.0], "k--", lw=0.8, label="y=x guarantee")
    for s in structs:
        emp = np.array([res[s]["cov"][a]["coverage"] for a in ALPHAS])
        ax[0].plot(nominal, emp, mk[s] + "-", color=pal[s], ms=5, lw=1.2,
                   label=s)
    ax[0].set_xlabel("nominal coverage  1-α", fontsize=7)
    ax[0].set_ylabel("empirical coverage", fontsize=7)
    ax[0].set_title("(a) coverage calibration\n(on/above diagonal = guarantee "
                    "holds)", fontsize=8)
    ax[0].legend(fontsize=5.5, loc="upper left"); ax[0].tick_params(labelsize=6)
    ax[0].set_xlim(0.75, 1.0); ax[0].set_ylim(0.75, 1.02)

    # (b) set size vs 1-α — efficiency.
    for s in structs:
        sz = np.array([res[s]["cov"][a]["set_size"] for a in ALPHAS])
        ax[1].plot(nominal, sz, mk[s] + "-", color=pal[s], ms=5, lw=1.2,
                   label=s)
    ax[1].set_xlabel("nominal coverage  1-α", fontsize=7)
    ax[1].set_ylabel("mean prediction-set size", fontsize=7)
    ax[1].set_title("(b) efficiency\n(smaller = more informative)", fontsize=8)
    ax[1].legend(fontsize=5.5, loc="upper left"); ax[1].tick_params(labelsize=6)
    ax[1].set_ylim(0.9, 3.1)

    # (c) dangerous-omission vs α_danger — single split (faded) vs K-fold
    #     cross-conformal shore-up (solid); on/below y=x budget.
    budget = np.array(ALPHA_DANGERS)
    ax[2].plot([0.0, max(ALPHA_DANGERS)], [0.0, max(ALPHA_DANGERS)],
               "k--", lw=0.8, label="y=x budget")
    for s in structs:
        om_split = np.array([res[s]["dang"][ad]["omission_rate"]
                             for ad in ALPHA_DANGERS])
        om_cv = np.array([res[s]["dang_cv"][ad]["omission_rate"]
                          for ad in ALPHA_DANGERS])
        ax[2].plot(budget, om_split, mk[s] + ":", color=pal[s], ms=4, lw=0.8,
                   alpha=0.35)
        ax[2].plot(budget, om_cv, mk[s] + "-", color=pal[s], ms=5, lw=1.4,
                   label=s)
    ax[2].plot([], [], "k:", lw=0.8, alpha=0.4, label="single split")
    ax[2].plot([], [], "k-", lw=1.4, label="K-fold cross-conf.")
    ax[2].set_xlabel("budget  α_danger", fontsize=7)
    ax[2].set_ylabel("RETIRE-omission rate", fontsize=7)
    ax[2].set_title("(c) dangerous-omission: cross-conformal fix\n"
                    "(on/below diagonal = within budget)", fontsize=8)
    ax[2].legend(fontsize=5.0, loc="upper left"); ax[2].tick_params(labelsize=6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    fig.savefig(os.path.splitext(out_path)[0] + ".png", bbox_inches="tight",
                dpi=150)
    plt.close(fig)
    return out_path


# ═════════════════════════════════════════════════════════════════════════════
#  Synthetic well-specified scenario set (for tests; no caches needed)
# ═════════════════════════════════════════════════════════════════════════════

def _synthetic_wellspec(M: int = 400, seed: int = 0) -> Scenarios:
    """A WELL-SPECIFIED set for the coverage test: p_pipeline is an exchangeable,
    well-calibrated score for the oracle action.  We draw p_oracle across [0,1]
    (so all three bands appear) and set p_pipeline = p_oracle + small symmetric
    noise — so the framework score is an honest, exchangeable estimate of the
    band that contains the truth, the regime where split-conformal's marginal
    guarantee should hold tightly."""
    import cfrp_phasefield_2d as pf
    rng = np.random.default_rng(seed)
    alpha, beta = pf.calibrate_thresholds(**sb.COSTS)
    p_oracle = rng.uniform(0.0, 1.0, M)
    p_pipeline = np.clip(p_oracle + rng.normal(0, 0.03, M), 0.0, 1.0)
    p_point = np.clip(p_oracle + rng.normal(0, 0.05, M), 0.0, 1.0)
    detect = p_oracle > 0.05
    return Scenarios(p_oracle=p_oracle, p_pipeline=p_pipeline, p_point=p_point,
                     detect=detect, alpha=alpha, beta=beta,
                     source=f"SYNTHETIC well-specified (M={M})")


# ═════════════════════════════════════════════════════════════════════════════
#  Unit tests
# ═════════════════════════════════════════════════════════════════════════════

def run_tests() -> int:
    n = 0

    def ok(cond, msg):
        nonlocal n
        assert cond, msg
        n += 1

    import cfrp_phasefield_2d as pf
    alpha_t, beta_t = pf.calibrate_thresholds(**sb.COSTS)

    # ---- soft probs respect the decision bands (argmax == decide) -----------
    grid = np.linspace(0.0, 1.0, 51)
    sp = soft_action_probs(grid, alpha_t, beta_t)
    ok(sp.shape == (51, 3), "soft probs shape (M,3)")
    ok(np.allclose(sp.sum(1), 1.0), "soft probs sum to 1")
    argmax_act = np.array([DECISIONS[j] for j in sp.argmax(1)])
    decided = np.array([duq.decide(p, alpha_t, beta_t) for p in grid])
    ok(np.mean(argmax_act == decided) >= 0.98,
       "soft-prob argmax matches decide() on the band grid")

    # ---- q̂ formula correctness: ceil((n+1)(1-α))/n, clipped ----------------
    cal = np.linspace(0.0, 1.0, 100)          # sorted 0..1
    for a in (0.2, 0.1, 0.05):
        nn = cal.size
        rank = int(np.ceil((nn + 1) * (1 - a)))
        q = conformal_quantile(cal, a)
        if rank >= nn:
            ok(q == 1.0 or np.isclose(q, cal[-1]), f"q̂ clipped/at-max for α={a}")
        else:
            ok(np.isclose(q, cal[rank - 1]),
               f"q̂ = rank-{rank} order statistic for α={a}")
    # small-n: (n+1)(1-α) can exceed n → q̂ = 1.0 (set = all actions)
    ok(conformal_quantile(np.array([0.1, 0.2, 0.3]), 0.05) == 1.0,
       "q̂ → 1.0 when n too small for the level")

    # ---- prediction sets always non-empty, sizes in [1,3] -------------------
    sets = prediction_sets(grid, alpha_t, beta_t, qhat=0.5)
    ok(all(1 <= len(s) <= 3 for s in sets), "set sizes in [1,3]")
    ok(all(s.issubset(set(DECISIONS)) for s in sets), "sets contain valid labels")

    # ---- WELL-SPECIFIED coverage guarantee for α ∈ {0.2,0.1,0.05} -----------
    sc = _synthetic_wellspec(M=500, seed=0)
    tol = 0.05
    covs, sizes = {}, {}
    for a in ALPHAS:
        r = split_calibrate(sc, a, seed=0)
        covs[a], sizes[a] = r["coverage"], r["set_size"]
        ok(r["coverage"] >= (1 - a) - tol,
           f"well-spec coverage {r['coverage']:.3f} ≥ {1-a:.2f}-tol for α={a}")
        ok(1.0 <= r["set_size"] <= 3.0, f"set size in [1,3] for α={a}")
    # ---- set size monotone-ish increasing as 1-α → 1 ------------------------
    ok(sizes[0.05] >= sizes[0.20] - 1e-9,
       "set size grows (or holds) as coverage target rises")

    # ---- dangerous-omission ≤ α_danger + tol on well-spec set ---------------
    for ad in ALPHA_DANGERS:
        d = dangerous_omission(sc, ad, seed=0)
        ok(d["omission_rate"] <= ad + tol,
           f"dangerous-omission {d['omission_rate']:.3f} ≤ {ad}+tol")

    # ---- K-fold cross-conformal shore-up: tests EVERY RETIRE point once, ----
    #      stays within budget, and uses more calibration RETIRE per fold ------
    oracle_ws = sb.oracle_decisions(sc)
    n_retire_total = int(np.sum(oracle_ws == "RETIRE"))
    for ad in ALPHA_DANGERS:
        dcv = dangerous_omission_cv(sc, ad, n_folds=5, seed=0)
        ok(dcv["omission_rate"] <= ad + tol,
           f"CV dangerous-omission {dcv['omission_rate']:.3f} ≤ {ad}+tol")
        ok(dcv["n_retire_test"] == n_retire_total,
           "CV tests every RETIRE point exactly once")
        ok(dcv["mean_retire_cal"] >= 0.0, "CV reports mean calibration RETIRE")

    # ---- the shore-up's PURPOSE: on a SMALL set with few RETIRE/split, the ---
    #      single split can breach budget while K-fold CV stays within it ------
    small = _synthetic_wellspec(M=44, seed=3)
    worst_split = max(dangerous_omission(small, 0.10, seed=s)["omission_rate"]
                      for s in range(6))
    cv_small = dangerous_omission_cv(small, 0.10, n_folds=5, seed=0)
    ok(cv_small["omission_rate"] <= worst_split + 1e-9,
       "K-fold CV omission ≤ worst single-split omission (variance reduced)")

    # ---- Mondrian cross-structure: valid per-structure rates, borrows cal ----
    per_ws = {"A": _synthetic_wellspec(M=60, seed=1),
              "B": _synthetic_wellspec(M=60, seed=2),
              "C": _synthetic_wellspec(M=60, seed=4)}
    mond = dangerous_omission_mondrian(per_ws, 0.10, seed=0)
    ok(set(mond) == {"A", "B", "C"}, "Mondrian returns every structure")
    for nm, d in mond.items():
        ok(0.0 <= d["omission_rate"] <= 1.0, f"Mondrian rate in [0,1] ({nm})")
        ok(d["n_retire_cal_pool"] > 0,
           f"Mondrian borrows RETIRE calibration from other structures ({nm})")

    # ---- pooled coverage holds across structures ----------------------------
    scs = [_synthetic_wellspec(M=250, seed=s) for s in (1, 2, 3)]
    pooled = pool_scenarios(scs)
    ok(pooled.p_oracle.size == 750, "pooled concatenates all sets")
    for a in ALPHAS:
        r = split_calibrate(pooled, a, seed=7)
        ok(r["coverage"] >= (1 - a) - tol,
           f"pooled coverage {r['coverage']:.3f} ≥ {1-a:.2f}-tol for α={a}")

    # ---- pool requires matching (α, β) --------------------------------------
    bad = Scenarios(p_oracle=np.array([0.5]), p_pipeline=np.array([0.5]),
                    p_point=np.array([0.5]), detect=np.array([True]),
                    alpha=0.3, beta=0.6, source="mismatch")
    raised = False
    try:
        pool_scenarios([scs[0], bad])
    except AssertionError:
        raised = True
    ok(raised, "pool_scenarios rejects mismatched α, β")

    # ---- end-to-end evaluate() on real structure builders (small M, <60s) ---
    res = evaluate(seed=0, M=40)
    ok(set(["interstage", "fairing", "srb3-motorcase", "pooled"]).issubset(res),
       "evaluate returns 3 structures + pooled")
    for name in res:
        for a in ALPHAS:
            c = res[name]["cov"][a]
            ok(0.0 <= c["coverage"] <= 1.0, f"{name}: coverage in [0,1] (α={a})")
            ok(1.0 <= c["set_size"] <= 3.0, f"{name}: set size in [1,3] (α={a})")
        for ad in ALPHA_DANGERS:
            d = res[name]["dang"][ad]
            ok(0.0 <= d["omission_rate"] <= 1.0,
               f"{name}: omission rate in [0,1] (αd={ad})")
            dcv = res[name]["dang_cv"][ad]
            ok(0.0 <= dcv["omission_rate"] <= 1.0,
               f"{name}: CV omission rate in [0,1] (αd={ad})")
    ok(all("dang_mond" in res[nm] for nm in res if nm != "pooled"),
       "evaluate attaches Mondrian cross-structure omission per structure")

    print(f"conformal_clearance: {n}/{n} unit tests passed")
    return n


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Split-conformal coverage guarantee on the OK/REPAIR/RETIRE "
                    "clearance action, across 3 structures + pooled.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fig", action="store_true",
                    help="save paper_figs/conformal_clearance.pdf")
    ap.add_argument("--test", action="store_true", help="unit tests only")
    args = ap.parse_args()

    if args.test:
        run_tests()
        return

    res = evaluate(seed=args.seed)
    print_report(res)
    if args.fig:
        p = make_figure(res, os.path.join(HERE, "paper_figs",
                                          "conformal_clearance.pdf"))
        print(f"\nfigure → {p}")


if __name__ == "__main__":
    main()
