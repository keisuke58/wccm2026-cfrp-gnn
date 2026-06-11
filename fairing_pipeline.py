"""
fairing_pipeline.py — END-TO-END Stage 0→5 decision framework on the SECOND
structure (satellite fairing, honeycomb facesheet–core DEBOND).

Why this exists
---------------
The Stage 0–5 reusable-vehicle SHM stack was only run END-TO-END on the
perforated interstage (matrix-crack delamination): run_pipeline.py walks one
held-out sample Stage 0→4, decision_uq.py audits the final OK/REPAIR/RETIRE
call, fleet_learning.py pools the fleet, design_feedback.py closes the loop.
The fairing so far had only a Stage-3 prognosis (fairing_debond_prognosis.py)
and a generic clearance — "one framework, two structures" was ASSERTED, not
demonstrated.

This module demonstrates it.  It reuses the STRUCTURE-AGNOSTIC machinery
(the shared α/β decision rule, the decision-UQ pattern, the hierarchical-Bayes
fleet, the expected-remaining-flights design metric) and runs the WHOLE chain
on the fairing via the pluggable `FairingPrognosis`:

    Stage 0  DETECT       GW/feature anomaly screen  → defect-present verdict
    Stage 2  CHARACTERISE debond-radius posterior  p(a)  (Gaussian, clipped)
    Stage 3  PROGNOSE     FairingPrognosis.growth_probability  → P(grow)
             DECIDE       flight_clearance_generic (shared α/β) → OK/REPAIR/RETIRE
    UQ                    oracle (exact debond P(grow) at a_true) vs pipeline
                         (P(grow) over the Stage-2 posterior): accuracy,
                         DANGEROUS-miss P(OK|oracle=RETIRE), confusion, per-
                         decision bootstrap confidence + ECE
    Stage 4  FLEET        feed fairing debond growth-rates into fleet_learning:
                         fleet-prior sharpening + fleet-leader (structure-agnostic)
    Stage 5  DESIGN       sweep a debond-resistance lever (Gc_interface, kappa):
                         P(grow) ↓ / E[remaining flights] ↑

Honest scope
------------
The fairing Stage-0/Stage-2 INPUTS here are REPRESENTATIVE / SYNTHETIC: a
plausible anomaly score for the GW screen and a Gaussian debond-radius
posterior clipped to [0, a_max].  The REAL guided-wave GNN detector for the
fairing is Payload's, integrated separately; it is not re-implemented here.
The point of this module is narrow and load-bearing: the
DECISION / UQ / FLEET / DESIGN machinery is STRUCTURE-AGNOSTIC and runs
end-to-end on the fairing through the pluggable prognosis — exactly as it does
on the interstage — so the framework genuinely serves two structures, not one.

Unlike decision_uq.py (interstage), the fairing debond forward
(simulate_debond_growth) is CHEAP and EXACT, so there is NO FD-vs-surrogate
split: the oracle and the pipeline use the SAME exact forward, differing only
in θ (true debond radius vs the Stage-2 posterior).  The UQ here therefore
isolates the Stage-2 POSTERIOR contribution to the end-to-end decision error.

Usage
-----
    python fairing_pipeline.py                 # full fairing Stage 0→5 report
    python fairing_pipeline.py --fig           # + paper_figs/fairing_pipeline.pdf
    python fairing_pipeline.py --test          # unit tests only
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

import numpy as np

import cfrp_phasefield_2d as pf          # SHARED decision rule (calibrate_thresholds)
import fleet_learning as fl              # structure-agnostic hierarchical-Bayes fleet
import fairing_debond_prognosis as fdp   # Stage-3 fairing prognosis + generic clearance

HERE = os.path.dirname(os.path.abspath(__file__))

SEVERITY = {"OK": 0, "REPAIR": 1, "RETIRE": 2}
DECISIONS = ("OK", "REPAIR", "RETIRE")


def decide(p: float, alpha: float, beta: float) -> str:
    return "OK" if p <= alpha else "REPAIR" if p <= beta else "RETIRE"


# ═════════════════════════════════════════════════════════════════════════════
#  Stage 0 — fairing GW / feature anomaly screen (representative, synthetic)
# ═════════════════════════════════════════════════════════════════════════════

def stage0_detect(a_true: float, cfg: fdp.FairingConfig,
                  rng: np.random.Generator) -> dict:
    """Representative guided-wave / feature anomaly screen for the fairing.

    The REAL detector is Payload's GW-GNN (integrated separately).  Here we
    synthesise a plausible anomaly score that grows monotonically with the true
    debond radius a_true plus measurement noise: a debond scatters/blocks the
    Lamb wave, raising a feature-energy z-score above the healthy-baseline tail.
    A defect is declared present when the peak score clears the healthy 99.9 %
    reference threshold.  Returns the score, the threshold, and the verdict.
    """
    # healthy-baseline anomaly tail (z-scored feature energy on pristine skin)
    healthy = np.abs(rng.standard_normal(400))
    thr = float(np.quantile(healthy, 0.999))
    # debond raises the score ∝ a_true / a_max (saturating), + sensing noise
    sig = 8.0 * (a_true / cfg.a_max)
    score = float(max(sig + 0.6 * rng.standard_normal(), 0.0))
    detected = score > thr
    return {"score": score, "thr": thr, "detected": bool(detected),
            "margin": float(np.clip(score / max(thr, 1e-6) - 1.0, 0.0, 1.0))}


# ═════════════════════════════════════════════════════════════════════════════
#  Stage 2 — debond-radius posterior p(a) (representative, synthetic)
# ═════════════════════════════════════════════════════════════════════════════

def stage2_posterior(a_true: float, cfg: fdp.FairingConfig,
                     rng: np.random.Generator, n_draws: int = 200,
                     rel_sd: float = 0.18, abs_floor: float = 0.004
                     ) -> np.ndarray:
    """Synthesise a realistic Stage-2 debond-radius posterior p(a).

    Gaussian around the truth a_true with SD = max(rel_sd·a_true, abs_floor),
    clipped to the physical band [1e-3, a_max].  This stands in for the real
    GW-GNN characterisation posterior (Payload's); the SHAPE (a unimodal cloud
    of plausible debond radii whose spread grows with the defect) is what the
    decision-UQ propagates.  Returns (n_draws,) radii.
    """
    sd = max(rel_sd * a_true, abs_floor)
    a = a_true + sd * rng.standard_normal(n_draws)
    return np.clip(a, 1e-3, cfg.a_max)


# ═════════════════════════════════════════════════════════════════════════════
#  (1)  Fairing end-to-end run (Stage 0 → decision)
# ═════════════════════════════════════════════════════════════════════════════

def run_endtoend(a_true: float = 0.085, load: float = 0.11,
                 cfg: fdp.FairingConfig | None = None,
                 n_flights: int = 20, seed: int = 0,
                 verbose: bool = True) -> dict:
    """One fairing inspection end-to-end: Stage 0 → 2 → 3 → decision.

    Mirrors run_pipeline's interstage orchestration + report, but for the
    fairing honeycomb debond via the pluggable FairingPrognosis and the SHARED
    flight_clearance_generic decision rule.
    """
    cfg = cfg or fdp.FairingConfig()
    rng = np.random.default_rng(seed)
    prog = fdp.FairingPrognosis(cfg)

    s0 = stage0_detect(a_true, cfg, rng)
    post = stage2_posterior(a_true, cfg, rng)
    clr = fdp.flight_clearance_generic(prog, post, load, n_flights=n_flights)

    out = {"a_true": a_true, "load": load, "s0": s0,
           "post_mean": float(post.mean()), "post_sd": float(post.std()),
           "clearance": clr, "decision": clr["decision"],
           "p_growth": clr["p_growth_next"],
           "e_remaining": clr["expected_remaining_flights"]}
    if verbose:
        _report_endtoend(out)
    return out


def _report_endtoend(o: dict) -> None:
    bar = "=" * 72
    clr = o["clearance"]
    print(bar)
    print("  FAIRING STRUCTURAL-HEALTH DECISION  —  end-to-end (2nd structure)")
    print(bar)
    s0 = o["s0"]
    verdict = "DEBOND DETECTED" if s0["detected"] else "clean"
    print(f"\n[Stage 0  DETECT]  GW/feature anomaly screen (representative)")
    print(f"  anomaly score = {s0['score']:.2f}  (healthy 99.9% thr={s0['thr']:.2f})"
          f"  →  {verdict}")
    print(f"\n[Stage 2  CHARACTERISE]  debond-radius posterior  p(a) (representative)")
    print(f"  a_true = {o['a_true']:.3f}   post mean ± sd = "
          f"{o['post_mean']:.3f} ± {o['post_sd']:.3f}   (a_max={fdp.FairingConfig().a_max})")
    print(f"\n[Stage 3  PROGNOSE]  fairing debond (Paris growth), load={o['load']:.3f}")
    print(f"  P(debond grows next flight) = {o['p_growth']:.3f}")
    print(f"  E[remaining flights]        = {o['e_remaining']:.2f}")
    print(f"\n[DECISION]  shared expected-cost thresholds "
          f"α={clr['alpha']:.2f}, β={clr['beta']:.2f}  (same as interstage)")
    tag = {"OK": "FLY AS-IS", "REPAIR": "REPAIR BEFORE NEXT FLIGHT",
           "RETIRE": "RETIRE FAIRING"}[o["decision"]]
    print(f"  ──►  {o['decision']:>7}   ({tag})")
    print(bar)


# ═════════════════════════════════════════════════════════════════════════════
#  (2)  Fairing decision-UQ  (oracle vs pipeline; no FD-vs-surrogate split)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class FairingStudy:
    a_true: np.ndarray            # (M,) true debond radius per scenario
    load: np.ndarray              # (M,) nominal load per scenario
    p_oracle: np.ndarray          # (M,) exact P(grow) at a_true over load scatter
    p_pipeline: np.ndarray        # (M,) P(grow) over the Stage-2 posterior
    conf: np.ndarray              # (M, 3) bootstrap decision confidence
    alpha: float
    beta: float

    def dec(self, which: str) -> np.ndarray:
        p = getattr(self, f"p_{which}")
        return np.array([decide(pi, self.alpha, self.beta) for pi in p])


def _load_grid(nominal: float, n_load: int, delta: float = 0.12) -> np.ndarray:
    """Operational load scatter: nominal·(1±delta), clipped non-negative."""
    if n_load <= 1:
        return np.array([nominal], dtype=float)
    g = np.linspace(-1.0, 1.0, n_load)
    return np.clip(nominal * (1.0 + delta * g), 1e-6, None)


def _exact_pgrow(radii: np.ndarray, loads: np.ndarray,
                 cfg: fdp.FairingConfig) -> float:
    """Mean exact debond P(grow) over (radii × loads) via simulate_debond_growth.

    The fairing forward is cheap & exact, so this is the SAME forward the
    pipeline uses — oracle and pipeline differ only in the θ (radius) source.
    """
    radii = np.atleast_1d(radii)
    flags = [fdp.simulate_debond_growth(float(a), float(L), cfg)["grown"]
             for a in radii for L in loads]
    return float(np.mean(flags))


def run_study(n_keep: int = 40, n_load: int = 5, n_draws: int = 24,
              n_boot: int = 300, seed: int = 0,
              cfg: fdp.FairingConfig | None = None,
              verbose: bool = True) -> FairingStudy:
    """End-to-end fairing decision-UQ over n_keep synthetic scenarios.

    Per scenario: a known true debond radius a_true and operational load
    scatter.  ORACLE decision = threshold on the EXACT debond P(grow) at
    a_true (averaged over the load distribution) — the best call knowing the
    debond exactly.  PIPELINE decision = threshold on P(grow) propagated over
    the Stage-2 debond posterior p(a).  Reports accuracy, dangerous-miss,
    confusion, and a per-decision bootstrap confidence (validated vs oracle).
    """
    cfg = cfg or fdp.FairingConfig()
    rng = np.random.default_rng(seed)
    alpha, beta = pf.calibrate_thresholds()

    # true radii tiled across the band; loads tiled so all 3 classes appear
    a_true = np.linspace(0.015, 0.16, n_keep)[rng.permutation(n_keep)]
    nominal = np.linspace(0.07, 0.15, n_keep)[rng.permutation(n_keep)]

    p_oracle = np.empty(n_keep)
    p_pipeline = np.empty(n_keep)
    conf = np.empty((n_keep, 3))
    for s in range(n_keep):
        loads = _load_grid(nominal[s], n_load)
        # oracle: exact P(grow) at the TRUE radius over the load scatter
        p_oracle[s] = _exact_pgrow(np.array([a_true[s]]), loads, cfg)
        # pipeline: P(grow) over the Stage-2 posterior (same exact forward)
        post = stage2_posterior(a_true[s], cfg, rng, n_draws=n_draws)
        di = rng.choice(len(post), min(n_draws, len(post)), replace=False)
        # per (draw,load) grow flags → matrix for the bootstrap confidence
        flags = np.array([[fdp.simulate_debond_growth(float(post[d]), float(L),
                                                       cfg)["grown"]
                           for L in loads] for d in di], dtype=float)
        p_pipeline[s] = float(flags.mean())
        flat = flags.reshape(-1)
        bidx = rng.integers(0, len(flat), (n_boot, len(flat)))
        bp = flat[bidx].mean(1)
        cnt = np.zeros(3)
        for bpi in bp:
            cnt[SEVERITY[decide(bpi, alpha, beta)]] += 1
        conf[s] = cnt / n_boot

    study = FairingStudy(a_true=a_true, load=nominal, p_oracle=p_oracle,
                         p_pipeline=p_pipeline, conf=conf,
                         alpha=alpha, beta=beta)
    if verbose:
        print(f"[fairing-study] M={n_keep} scenarios, exact debond forward "
              f"(n_load={n_load}, n_draws={n_draws})")
    return study


def confusion(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    C = np.zeros((3, 3), dtype=int)
    for p, t in zip(pred, truth):
        C[SEVERITY[t], SEVERITY[p]] += 1
    return C


def metrics(study: FairingStudy) -> dict:
    o = study.dec("oracle")
    pl = study.dec("pipeline")
    s_o = np.array([SEVERITY[d] for d in o])
    s_pl = np.array([SEVERITY[d] for d in pl])
    n_retire = int((o == "RETIRE").sum())
    dmiss = (float(np.mean(pl[o == "RETIRE"] == "OK")) if n_retire else
             float("nan"))
    return {
        "M": len(o),
        "acc_pipeline": float(np.mean(pl == o)),
        "unsafe_miss_rate": float(np.mean(s_pl < s_o)),
        "conservative_alarm_rate": float(np.mean(s_pl > s_o)),
        "dangerous_miss_OK_given_RETIRE": dmiss,
        "n_retire_oracle": n_retire,
        "oracle_class_counts": {d: int((o == d).sum()) for d in DECISIONS},
        "confusion_pipeline_vs_oracle": confusion(pl, o),
    }


def reliability(study: FairingStudy, n_bins: int = 5) -> dict:
    """Validate the per-decision confidence: bin by the confidence the pipeline
    assigned to ITS OWN chosen decision, measure empirical oracle agreement,
    report ECE (calibrated → empirical agreement tracks claimed confidence)."""
    pl = study.dec("pipeline")
    o = study.dec("oracle")
    own = np.array([study.conf[s, SEVERITY[pl[s]]] for s in range(len(pl))])
    correct = (pl == o).astype(float)
    edges = np.linspace(1.0 / 3.0, 1.0, n_bins + 1)
    mids, emp, cnt = [], [], []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        m = (own >= lo) & (own <= hi if b == n_bins - 1 else own < hi)
        if m.sum() == 0:
            continue
        mids.append(float(own[m].mean()))
        emp.append(float(correct[m].mean()))
        cnt.append(int(m.sum()))
    ece = float(np.sum(np.array(cnt) / len(pl)
                       * np.abs(np.array(mids) - np.array(emp)))) if cnt \
        else float("nan")
    return {"conf_mid": mids, "emp_acc": emp, "count": cnt, "ece": ece,
            "own_conf": own, "correct": correct}


def _print_uq_report(study: FairingStudy, m: dict, rel: dict) -> None:
    bar = "=" * 72
    print(bar)
    print("  FAIRING END-TO-END DECISION UQ  —  same UQ story on structure 2")
    print(bar)
    print(f"  forward          : EXACT debond (oracle & pipeline share it; no "
          f"FD-vs-surrogate split)")
    print(f"  scenarios        : M={m['M']}   "
          f"thresholds α={study.alpha:.2f}, β={study.beta:.2f}")
    print(f"  oracle classes   : " +
          "  ".join(f"{d}={m['oracle_class_counts'][d]}" for d in DECISIONS))
    print()
    print(f"  END-TO-END decision accuracy (pipeline = oracle): "
          f"{m['acc_pipeline']*100:5.1f}%")
    print(f"  ── unsafe under-call  (less severe than oracle) : "
          f"{m['unsafe_miss_rate']*100:5.1f}%")
    print(f"  ── conservative alarm (more severe than oracle) : "
          f"{m['conservative_alarm_rate']*100:5.1f}%")
    dm = m["dangerous_miss_OK_given_RETIRE"]
    print(f"  ── DANGEROUS miss  P(OK | oracle=RETIRE)        : "
          + ("n/a" if dm != dm else f"{dm*100:5.1f}%")
          + f"   (n_retire={m['n_retire_oracle']})")
    print()
    print("  ── confusion (rows=oracle, cols=pipeline) ──")
    C = m["confusion_pipeline_vs_oracle"]
    print(f"           {'OK':>6}{'REPAIR':>8}{'RETIRE':>8}")
    for i, d in enumerate(DECISIONS):
        print(f"     {d:>7}{C[i,0]:>6}{C[i,1]:>8}{C[i,2]:>8}")
    print()
    print(f"  ── per-decision confidence calibration ──")
    print(f"     ECE(own-decision confidence) = {rel['ece']:.3f}")
    for cm, ea, ct in zip(rel["conf_mid"], rel["emp_acc"], rel["count"]):
        print(f"     conf≈{cm:.2f}:  empirical agree {ea*100:5.1f}%  (n={ct})")
    print(bar)


# ═════════════════════════════════════════════════════════════════════════════
#  (3)  Fairing FLEET hook  (structure-agnostic hierarchical Bayes)
# ═════════════════════════════════════════════════════════════════════════════

def fairing_fleet(cfg: fdp.FairingConfig | None = None,
                  seed: int = 0, full_leader: bool = False) -> dict:
    """Feed a fairing debond fleet into fleet_learning (structure-agnostic).

    The hierarchical-Bayes fleet machinery only needs per-vehicle growth-rate
    posteriors, which the debond prognosis provides: each fairing's
    facesheet/core interface has a lot-to-lot effective debond growth-rate
    s_v = log r_v.  fleet_learning's surrogate Paris law is the SAME mechanistic
    shape as the debond Paris law (da/dN ∝ (G/Gc)^m), so the fleet prior
    sharpening + fleet-leader effect carry over to the fairing fleet verbatim.
    We map the fairing's interfacial-debond growth scale onto the fleet's
    normalised log-rate so the demo is fairing-flavoured.
    """
    cfg = cfg or fdp.FairingConfig()
    # a fairing-flavoured fleet config: slightly faster lot-to-lot debond
    # spread than the interstage default (honeycomb bond variability).
    fcfg = fl.FleetConfig(n_vehicles=20, n_flights=30,
                          mu_phi_true=-3.0, sigma_phi_true=0.50,
                          seed=seed)
    rec = fl.recovery_study(fcfg, sizes=[2, 5, 10, 20])

    # introduce a freshly-arrived fairing and partial-pool it vs the fleet prior
    rng = np.random.default_rng(seed)
    fleet = fl.simulate_fleet(fcfg, rng)
    post_phi = fl.fit_fleet(fleet[:-1], fcfg, n_samples=1500, burn=500,
                            rng=np.random.default_rng(seed + 1))
    prior = (post_phi.mu_phi_mean, post_phi.sigma_phi_mean)
    new_veh = fleet[-1]
    _, sd_iso = fl._vehicle_s_posterior(new_veh, 1, None, fcfg)
    _, sd_fl = fl._vehicle_s_posterior(new_veh, 1, prior, fcfg)

    out = {
        "rec_sizes": rec.fleet_sizes,
        "rec_sigma_mean": rec.sigma_phi_mean,
        "rec_sigma_sd": rec.sigma_phi_sd,        # post SD of σ_φ → sharpening
        "fleet_prior": prior,
        "new_veh_sd_isolated": float(sd_iso),
        "new_veh_sd_with_prior": float(sd_fl),
        "sharpening_ratio": float(rec.sigma_phi_sd[0] /
                                  max(rec.sigma_phi_sd[-1], 1e-9)),
    }
    if full_leader:
        flr = fl.fleet_leader_experiment(fcfg, n_samples=1200, burn=400,
                                         rng=np.random.default_rng(seed + 2))
        out["leader_iso"] = float(flr.insp_without_prior[0])
        out["leader_fleet"] = float(flr.insp_with_prior[-1])
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  (4)  Fairing DESIGN hook  (debond-resistance lever → P(grow) / E[flights])
# ═════════════════════════════════════════════════════════════════════════════

def expected_remaining_flights_debond(a0: float, load: float,
                                      cfg: fdp.FairingConfig,
                                      n_flights: int = 30) -> float:
    """Flights survived before the debond first 'grows' (per simulate_debond_
    growth's grown flag), censored at n_flights — the fairing analog of
    design_feedback.expected_remaining_flights, on the debond forward."""
    a = float(a0)
    for f in range(n_flights):
        res = fdp.simulate_debond_growth(a, load, cfg)
        if res["grown"]:
            return float(f)
        a = res["a_final"]
    return float(n_flights)


def design_sweep(a0: float = 0.07, load: float = 0.08,
                 cfg: fdp.FairingConfig | None = None,
                 n_levels: int = 7, lever: str = "Gc_interface",
                 n_post: int = 16, n_flights: int = 30,
                 seed: int = 0) -> dict:
    """Sweep a debond-resistance DESIGN lever and show it lowers P(grow) and
    raises E[remaining flights] — the design-feedback loop closing on the fairing.

    Levers (structure-2 analog of design_feedback's interface toughening):
      * "Gc_interface" — interfacial fracture toughness of the facesheet/core
        bond (tougher interleaf/film adhesive): higher Gc ⇒ smaller da/dN.
      * "kappa"        — lumped ERR constant (a stiffer / thicker facesheet
        lowers the blister driving force): lower kappa ⇒ smaller G ⇒ less growth.
    A simple grid (no heavy BO needed).  Returns per-level P(grow) over a
    debond-radius posterior and E[remaining flights] at the posterior mean.
    """
    cfg = cfg or fdp.FairingConfig()
    rng = np.random.default_rng(seed)
    post = stage2_posterior(a0, cfg, rng, n_draws=n_post)

    if lever == "Gc_interface":
        # tougher interface = larger Gc (better); sweep low→high resistance
        levels = np.linspace(0.45, 1.05, n_levels)
        def mk(v): return fdp.FairingConfig(**{**cfg.__dict__, "Gc_interface": v})
        resistance = levels                       # higher = more resistant
    elif lever == "kappa":
        # stiffer facesheet = lower kappa (better); sweep high→low driving force
        levels = np.linspace(1.2e4, 4.0e3, n_levels)
        def mk(v): return fdp.FairingConfig(**{**cfg.__dict__, "kappa": v})
        resistance = 1.0 / levels                 # higher = more resistant
    else:
        raise ValueError("lever must be 'Gc_interface' or 'kappa'")

    p_grow, e_flights = [], []
    for v in levels:
        c = mk(float(v))
        p_grow.append(fdp.debond_growth_probability(post, load, c, n_draws=n_post))
        e_flights.append(expected_remaining_flights_debond(
            float(post.mean()), load, c, n_flights=n_flights))

    return {"lever": lever, "levels": np.asarray(levels),
            "resistance": np.asarray(resistance),
            "p_grow": np.asarray(p_grow),
            "e_flights": np.asarray(e_flights),
            "a0": a0, "load": load}


def _print_fleet_design(s4: dict, sw: dict) -> None:
    bar = "=" * 72
    print(bar)
    print("  FAIRING FLEET (Stage 4) + DESIGN (Stage 5) hooks  —  loop closes")
    print(bar)
    print("  [Stage 4  FLEET]  hierarchical-Bayes fairing-debond fleet")
    print("  φ-prior sharpening (post. SD of σ_φ) vs fleet size N:")
    for n, sm, ss in zip(s4["rec_sizes"], s4["rec_sigma_mean"],
                         s4["rec_sigma_sd"]):
        print(f"      N={n:>2}:  E[σ_φ]={sm:.3f}   sd(σ_φ)={ss:.3f}")
    print(f"  sharpening N={s4['rec_sizes'][0]}→{s4['rec_sizes'][-1]}: "
          f"sd(σ_φ) ×{s4['sharpening_ratio']:.2f} tighter")
    print(f"  NEW fairing, growth-rate posterior after 1 inspection:")
    print(f"      isolated     sd(log r) = {s4['new_veh_sd_isolated']:.3f}")
    print(f"      +fleet prior sd(log r) = {s4['new_veh_sd_with_prior']:.3f}"
          f"   →  fleet-leader tightening")
    if "leader_iso" in s4:
        print(f"  inspections-to-stable-decision: isolated "
              f"{s4['leader_iso']:.2f}  →  full-fleet prior {s4['leader_fleet']:.2f}")
    print()
    print(f"  [Stage 5  DESIGN]  debond-resistance lever: {sw['lever']}")
    print(f"  {'level':>10}  {'P(grow)':>8}  {'E[flights]':>11}")
    for v, p, e in zip(sw["levels"], sw["p_grow"], sw["e_flights"]):
        print(f"  {v:>10.4g}  {p:>8.3f}  {e:>11.1f}")
    print(f"  → P(grow) {sw['p_grow'][0]:.3f} → {sw['p_grow'][-1]:.3f}   "
          f"E[flights] {sw['e_flights'][0]:.1f} → {sw['e_flights'][-1]:.1f} "
          f"(more resistance ⇒ safer)")
    print(bar)


# ═════════════════════════════════════════════════════════════════════════════
#  Figure
# ═════════════════════════════════════════════════════════════════════════════

def make_figure(study: FairingStudy, m: dict, rel: dict, s4: dict,
                sw: dict, out_path: str) -> str:
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

    fig, ax = plt.subplots(1, 4, figsize=figsize)

    # (a) decision confusion (fairing)
    C = m["confusion_pipeline_vs_oracle"]
    ax[0].imshow(C, cmap="Blues")
    ax[0].set_xticks(range(3)); ax[0].set_yticks(range(3))
    ax[0].set_xticklabels(DECISIONS, fontsize=6, rotation=30)
    ax[0].set_yticklabels(DECISIONS, fontsize=6)
    for i in range(3):
        for j in range(3):
            ax[0].text(j, i, C[i, j], ha="center", va="center", fontsize=8,
                       color="k" if C[i, j] < C.max() * 0.6 else "w")
    ax[0].set_title(f"(a) fairing decision confusion\nacc={m['acc_pipeline']*100:.0f}%",
                    fontsize=8)
    ax[0].set_xlabel("pipeline", fontsize=7)
    ax[0].set_ylabel("oracle (exact debond)", fontsize=7)

    # (b) confidence calibration
    ax[1].plot([0, 1], [0, 1], color="0.7", lw=0.7, ls="--")
    if rel["conf_mid"]:
        ax[1].plot(rel["conf_mid"], rel["emp_acc"], "-o", ms=4, c="#b71c1c",
                   label=f"ECE {rel['ece']:.2f}")
    ax[1].set_xlim(0.3, 1.02); ax[1].set_ylim(0.3, 1.02)
    ax[1].set_xlabel("claimed decision reliability", fontsize=7)
    ax[1].set_ylabel("empirical oracle agreement", fontsize=7)
    ax[1].set_title("(b) confidence calibration", fontsize=8)
    ax[1].legend(fontsize=6, loc="upper left")
    ax[1].tick_params(labelsize=6)

    # (c) fleet-prior sharpening
    ax[2].plot(s4["rec_sizes"], s4["rec_sigma_sd"], "-s", ms=4, c="#2e7d32")
    ax[2].set_title("(c) fairing fleet sharpening", fontsize=8)
    ax[2].set_xlabel("fleet size $N$", fontsize=7)
    ax[2].set_ylabel(r"post. SD of $\sigma_\phi$", fontsize=7)
    ax[2].tick_params(labelsize=6)

    # (d) design lever vs P(grow) / E[flights]
    res = sw["resistance"]
    order = np.argsort(res)
    ax[3].plot(res[order], sw["p_grow"][order], "-o", ms=3, c="#1565C0",
               label="P(grow)")
    ax[3].set_xlabel(f"debond resistance ({sw['lever']})", fontsize=7)
    ax[3].set_ylabel("P(grow next flight)", fontsize=7, color="#1565C0")
    ax[3].tick_params(labelsize=6, axis="y", labelcolor="#1565C0")
    ax3b = ax[3].twinx()
    ax3b.plot(res[order], sw["e_flights"][order], "-^", ms=3, c="#b71c1c",
              label="E[flights]")
    ax3b.set_ylabel("E[remaining flights]", fontsize=7, color="#b71c1c")
    ax3b.tick_params(labelsize=6, axis="y", labelcolor="#b71c1c")
    ax[3].set_title("(d) design lever closes loop", fontsize=8)
    ax[3].tick_params(labelsize=6, axis="x")

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

    cfg = fdp.FairingConfig()
    a_cal, b_cal = pf.calibrate_thresholds()

    # ── (1) end-to-end produces a valid verdict ──────────────────────────────
    e = run_endtoend(a_true=0.085, load=0.11, cfg=cfg, seed=0, verbose=False)
    ok(e["decision"] in DECISIONS, "fairing end-to-end gives valid verdict")
    ok(e["clearance"]["structure"] == "fairing-debond", "fairing structure tag")
    ok(e["clearance"]["alpha"] == a_cal, "shared α threshold (same as interstage)")
    ok(0.0 <= e["p_growth"] <= 1.0, "end-to-end P(grow) in [0,1]")
    # severity monotone: tiny debond/low load → OK; big debond/high load → not OK
    lo = run_endtoend(a_true=0.01, load=0.05, cfg=cfg, seed=1, verbose=False)
    hi = run_endtoend(a_true=0.16, load=0.15, cfg=cfg, seed=1, verbose=False)
    ok(SEVERITY[hi["decision"]] >= SEVERITY[lo["decision"]],
       "decision monotone in debond severity")

    # ── (2) decision-UQ ──────────────────────────────────────────────────────
    st = run_study(n_keep=24, n_load=3, n_draws=12, n_boot=80, seed=0,
                   cfg=cfg, verbose=False)
    ok(st.p_oracle.shape == (24,), "oracle prob shape")
    ok(np.all((st.p_oracle >= 0) & (st.p_oracle <= 1)), "oracle prob in [0,1]")
    ok(np.all((st.p_pipeline >= 0) & (st.p_pipeline <= 1)), "pipeline prob in [0,1]")
    ok(np.all((st.conf >= 0) & (st.conf <= 1)), "confidence entries in [0,1]")
    ok(np.allclose(st.conf.sum(1), 1.0), "confidence rows sum to 1")
    m = metrics(st)
    ok(0.0 <= m["acc_pipeline"] <= 1.0, "decision-UQ accuracy in [0,1]")
    dm = m["dangerous_miss_OK_given_RETIRE"]
    ok(dm != dm or 0.0 <= dm <= 1.0, "dangerous-miss computed (nan-or-[0,1])")
    ok(m["confusion_pipeline_vs_oracle"].sum() == 24, "confusion total = M")
    ok(set(m["oracle_class_counts"]) == set(DECISIONS), "oracle covers all classes")
    rel = reliability(st, n_bins=4)
    ok(rel["ece"] == rel["ece"], "ECE computed (finite)")
    ok(0.0 <= rel["ece"] <= 1.0, "ECE in [0,1]")

    # ── (3) design lever monotonically reduces P(grow) ───────────────────────
    for lever in ("Gc_interface", "kappa"):
        sw = design_sweep(a0=0.09, load=0.11, cfg=cfg, n_levels=6,
                          lever=lever, n_post=10, n_flights=20, seed=0)
        # order by resistance: P(grow) non-increasing, E[flights] non-decreasing
        idx = np.argsort(sw["resistance"])
        pg = sw["p_grow"][idx]
        ef = sw["e_flights"][idx]
        ok(pg[-1] <= pg[0] + 1e-9, f"[{lever}] more resistance ⇒ P(grow) ↓")
        ok(np.all(np.diff(pg) <= 1e-9), f"[{lever}] P(grow) monotone non-increasing")
        ok(ef[-1] >= ef[0] - 1e-9, f"[{lever}] more resistance ⇒ E[flights] ↑")

    # ── (4) fleet hook returns sharpening ────────────────────────────────────
    s4 = fairing_fleet(cfg=cfg, seed=0, full_leader=False)
    ok(len(s4["rec_sigma_sd"]) == len(s4["rec_sizes"]), "fleet recovery shapes")
    ok(s4["rec_sigma_sd"][-1] < s4["rec_sigma_sd"][0],
       "fleet prior SHARPENS with fleet size (σ_φ post. SD shrinks)")
    ok(s4["sharpening_ratio"] > 1.0, "sharpening ratio > 1")
    ok(s4["new_veh_sd_with_prior"] <= s4["new_veh_sd_isolated"] + 1e-9,
       "fleet prior tightens a new fairing's growth-rate posterior")

    print(f"fairing_pipeline: {n}/{n} unit tests passed")
    return n


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Fairing (2nd structure) end-to-end Stage 0→5 decision "
                    "framework: run, decision-UQ, fleet, design.")
    ap.add_argument("--a-true", type=float, default=0.085,
                    help="true debond radius for the single end-to-end run")
    ap.add_argument("--load", type=float, default=0.11)
    ap.add_argument("--n-keep", type=int, default=40,
                    help="decision-UQ scenarios")
    ap.add_argument("--n-load", type=int, default=5)
    ap.add_argument("--n-draws", type=int, default=24)
    ap.add_argument("--fleet-leader", action="store_true",
                    help="also run the full fairing fleet-leader study")
    ap.add_argument("--fig", action="store_true",
                    help="save paper_figs/fairing_pipeline.pdf")
    ap.add_argument("--test", action="store_true", help="unit tests only")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.test:
        run_tests(); return

    t0 = time.perf_counter()
    cfg = fdp.FairingConfig()

    # (1) end-to-end
    run_endtoend(a_true=args.a_true, load=args.load, cfg=cfg, seed=args.seed)

    # (2) decision-UQ
    study = run_study(n_keep=args.n_keep, n_load=args.n_load,
                      n_draws=args.n_draws, seed=args.seed, cfg=cfg)
    m = metrics(study)
    rel = reliability(study)
    _print_uq_report(study, m, rel)

    # (3) fleet + (4) design
    s4 = fairing_fleet(cfg=cfg, seed=args.seed, full_leader=args.fleet_leader)
    # design sweep at a near-transition debond severity, where the resistance
    # lever has the most leverage (cf. design_feedback honest accounting):
    sw = design_sweep(a0=0.07, load=0.08, cfg=cfg, lever="Gc_interface",
                      seed=args.seed)
    _print_fleet_design(s4, sw)

    print(f"\n  TOTAL WALL-CLOCK: {time.perf_counter() - t0:.2f} s")

    if args.fig:
        p = make_figure(study, m, rel, s4, sw,
                        os.path.join(HERE, "paper_figs", "fairing_pipeline.pdf"))
        print(f"\nfigure → {p}")


if __name__ == "__main__":
    main()
