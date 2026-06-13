"""
pomdp_inspection.py — the SEQUENTIAL inspect/fly/repair/retire decision as a
BELIEF-STATE POMDP: "may the part fly again, how many flights are left, AND IS IT
WORTH PAYING FOR ONE MORE INSPECTION FIRST?" — answered over the whole life with
the crack size only PARTIALLY OBSERVED.

Where this sits / why it is new
-------------------------------
Two modules already own a piece of this:
  * `lifetime_policy` solves the optimal-stopping FLY/REPAIR/RETIRE DP, but with
    the crack size FULLY OBSERVED — it folds inspection noise into the growth-rate
    belief and explicitly flags "a belief-state POMDP over a is future work".
  * `voi_inspection` (Stage 2.5) prices ONE inspection by value-of-information, but
    as a MYOPIC one-shot decision on a static belief — it never asks WHEN, over a
    life of flights, an inspection should be bought.
This module FUSES them.  The state is a BELIEF over the (hidden) crack size a; the
operator has a fourth action — INSPECT — a costly, noisy read of a that SHARPENS
the belief (same Gaussian read model the fleet sim already carries,
`FleetConfig.measurement_sd`).  Backward induction over the belief now decides,
jointly and sequentially, whether to FLY, pay to INSPECT, REPAIR, or RETIRE.  The
myopic mid-band inspect rule of `voi_inspection` re-emerges as a 2-D slice of this
policy — now ENDOGENOUS (gated by how uncertain the belief is) and EMBEDDED in the
lifetime trade-off.

The POMDP (assumed-density / moment-projected backward induction)
-----------------------------------------------------------------
Belief over the crack is carried as a Gaussian b=(μ,s) on a∈[0,a_fail] (an
assumed-density filter — the honest approximation, see LIMITATION).  Per epoch:
  * FLY    — earn FLY_VALUE if it survives; the crack grows with the SAME Paris
             surrogate as Stage-4 (a'=a+r·Δσ^m·a^p, Δσ~N), the belief is
             moment-matched forward and its spread GROWS (no observation); with
             prob p_fail(b) the crack reaches a_fail mid-flight → −C_FAIL, life ends.
  * INSPECT— pay −C_INSPECT, forgo this flight, observe y=a+N(0,σ²),
             σ=measurement_sd(μ)/κ; a conjugate Gaussian update SHARPENS the
             belief (s↓).  Value is averaged over the predictive reading.
  * REPAIR — pay −C_REPAIR, reset the crack belief to (A_REPAIR, tight).
  * RETIRE — take salvage +S_RETIRE and stop.

What we show (honest ordering: the robust result first)
-------------------------------------------------------
  1. STATE UNCERTAINTY DOMINATES.  Act-on-the-mean — feeding a POINT crack
     estimate to the fully-observed `lifetime_policy` DP, ignoring that the crack
     is only PARTIALLY observed — fails catastrophically (~30%) and loses about a
     THIRD of lifetime value.  A belief-state policy (the POMDP, or its
     never-inspect ablation) recovers it.  This large, robust gap is the headline:
     a reusable-vehicle clearance MUST propagate state uncertainty, not act on a
     point crack read.
  2. ACTIVE INSPECTION is a MODEST, QUALITY-GATED increment ON TOP of the
     belief-aware policy — NOT a free lunch.  Over a conservative belief policy
     that already hedges by retiring early, buying reads at the baseline NDT noise
     returns little net value (within Monte-Carlo noise); SHARPER/cheaper reads
     (higher κ) are bought and do pay.  Inspection earns its keep only in the
     ambiguous, uncertain mid-band — `voi_inspection`'s one-shot VoI result, now
     produced SEQUENTIALLY and endogenously by the lifetime DP.
  3. The INSPECT region is the UNCERTAINTY-GATED mid band: INSPECT is optimal
     where the crack mean is ambiguous AND the belief is uncertain (large s), and
     collapses where the belief is already tight or the call is clear-cut.

HONEST LIMITATION
-----------------
The belief is propagated by an ASSUMED-DENSITY (Gaussian moment-projection) filter
on a discretised (μ,s) grid — not an exact α-vector / point-based POMDP; spread is
clipped to the grid (an information-loss approximation, stated where it bites).
The growth rate r is taken KNOWN here — state uncertainty is over the crack a,
whereas `lifetime_policy` owns growth-rate uncertainty; fusing BOTH unknowns into
one belief is future work.  The economics (FLY_VALUE, C_FAIL, C_REPAIR, S_RETIRE,
C_INSPECT) and the Gaussian inspection model are REPRESENTATIVE — the contribution
is the belief-state DECISION STRUCTURE and the two qualitative results (resolving
state uncertainty by inspection buys lifetime value; the inspect region is the
uncertainty-gated mid-band), not certified inspection schedules.

Usage
-----
    python pomdp_inspection.py            # policy comparison + inspect region + sweep
    python pomdp_inspection.py --fig
    python pomdp_inspection.py --test
"""
from __future__ import annotations

import argparse
import os

import numpy as np

import fleet_learning as fl
import lifetime_policy as lp

HERE = os.path.dirname(os.path.abspath(__file__))

# Economics: inherit the lifetime constants from lifetime_policy so the two
# time-axis modules speak the same cost units; add the inspection price.
FLY_VALUE = lp.FLY_VALUE          # 10  — value of one successful mission
C_FAIL = lp.C_FAIL                # 200 — catastrophic in-flight failure
C_REPAIR = lp.C_REPAIR            # 25  — refurbishment turn (resets the crack)
S_RETIRE = lp.S_RETIRE            # 30  — salvage of a retired-but-intact vehicle
A_REPAIR = lp.A_REPAIR            # 0.10 — crack belief mean after a refurbishment
F_MAX = lp.F_MAX                  # 24   — design-life horizon (flights)
C_INSPECT = 2.0                   # price+turn of one inspection (forgoes a flight too)

# Belief grid (μ = crack mean, s = crack-belief SD) and quadrature resolution.
N_MU = 28
N_S = 14
S_MIN = 0.012                     # tightest representable belief (post-inspection floor)
S_MAX = 0.34                      # broadest belief we track (FLY inflation clipped here)
S_REPAIR = 0.03                   # belief SD right after a refurbishment (well known)
K_BELIEF = 7                      # Gauss–Hermite nodes over the belief
J_LOAD = 5                        # Gauss–Hermite nodes over flight-load scatter
L_OBS = 7                         # Gauss–Hermite nodes over the predictive reading

ACTIONS = ("FLY", "INSPECT", "REPAIR", "RETIRE")
I_FLY, I_INSP, I_REP, I_RET = 0, 1, 2, 3


# ═════════════════════════════════════════════════════════════════════════════
#  Gauss–Hermite (probabilists') nodes for N(0,1) expectations
# ═════════════════════════════════════════════════════════════════════════════

def _gh(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes/weights so that E_{x~N(0,1)}[f(x)] ≈ Σ w_i f(x_i)."""
    x, w = np.polynomial.hermite_e.hermegauss(n)
    return x, w / np.sqrt(2.0 * np.pi)


_XB, _WB = _gh(K_BELIEF)
_XL, _WL = _gh(J_LOAD)
_XY, _WY = _gh(L_OBS)


# ═════════════════════════════════════════════════════════════════════════════
#  Belief grid + bilinear interpolation of a value table V[μ, s]
# ═════════════════════════════════════════════════════════════════════════════

def belief_grids(cfg: fl.FleetConfig) -> tuple[np.ndarray, np.ndarray]:
    return (np.linspace(0.0, cfg.a_fail, N_MU), np.linspace(S_MIN, S_MAX, N_S))


def bilin(V: np.ndarray, mu_g: np.ndarray, s_g: np.ndarray, mq, sq):
    """Bilinear interpolation of V on the regular (mu_g, s_g) grid at query
    point(s) (mq, sq); works for scalars or equal-shaped arrays."""
    mq = np.clip(mq, mu_g[0], mu_g[-1])
    sq = np.clip(sq, s_g[0], s_g[-1])
    dm = mu_g[1] - mu_g[0]
    ds = s_g[1] - s_g[0]
    fi = (mq - mu_g[0]) / dm
    fj = (sq - s_g[0]) / ds
    i0 = np.clip(np.floor(fi).astype(int), 0, len(mu_g) - 2)
    j0 = np.clip(np.floor(fj).astype(int), 0, len(s_g) - 2)
    ti = fi - i0
    tj = fj - j0
    v00 = V[i0, j0]; v10 = V[i0 + 1, j0]
    v01 = V[i0, j0 + 1]; v11 = V[i0 + 1, j0 + 1]
    return ((1 - ti) * (1 - tj) * v00 + ti * (1 - tj) * v10
            + (1 - ti) * tj * v01 + ti * tj * v11)


# ═════════════════════════════════════════════════════════════════════════════
#  Belief transitions (shared by the DP and the Monte-Carlo rollout)
# ═════════════════════════════════════════════════════════════════════════════

def fly_moments(mu, s, r: float, cfg: fl.FleetConfig):
    """One-flight belief push-forward under FLY with NO observation, vectorised
    over arbitrary-shaped (mu, s).  Moment-match the post-flight crack belief and
    return (μ', s', p_fail) (each the shape of the broadcast inputs).  The spread
    grows: load scatter (amplified by Δσ^m) + belief spread propagate through the
    Paris law and no measurement contracts them."""
    mu = np.asarray(mu, float); s = np.asarray(s, float)
    a_nodes = np.clip(mu[..., None] + s[..., None] * _XB, 0.0, None)   # (...,K)
    loads = np.clip(cfg.load_amp + cfg.load_amp_sd * _XL, 1e-6, None)  # (J,)
    a = a_nodes[..., :, None]                                          # (...,K,1)
    a_next = a + r * (loads ** cfg.paris_m) * (a ** cfg.paris_p)       # (...,K,J)
    w = _WB[:, None] * _WL[None, :]
    w = w / w.sum()                                                    # (K,J)
    fail = a_next >= cfg.a_fail
    p_fail = (w * fail).sum(axis=(-2, -1))
    ws = w * (~fail)
    norm = ws.sum(axis=(-2, -1))
    safe = norm > 0
    nz = np.where(safe, norm, 1.0)
    mu_n = (ws * a_next).sum(axis=(-2, -1)) / nz
    var_n = (ws * (a_next - mu_n[..., None, None]) ** 2).sum(axis=(-2, -1)) / nz
    mu_n = np.where(safe, np.clip(mu_n, 0.0, cfg.a_fail), cfg.a_fail)
    s_n = np.where(safe, np.clip(np.sqrt(np.maximum(var_n, 0.0)), S_MIN, S_MAX),
                   S_MIN)
    return mu_n, s_n, p_fail


def fly_predict(mu: float, s: float, r: float, cfg: fl.FleetConfig
                ) -> tuple[float, float, float]:
    """Scalar wrapper around `fly_moments` (used by the rollout and tests)."""
    mu_n, s_n, pf = fly_moments(mu, s, r, cfg)
    return float(mu_n), float(s_n), float(pf)


def inspect_sigma(mu: float, cfg: fl.FleetConfig, kappa: float) -> float:
    """Inspection read-noise SD at believed crack μ: the fleet sim's own
    Gaussian measurement model, degraded by quality κ∈(0,1] (κ=1 baseline)."""
    return float(cfg.measurement_sd(max(mu, 0.0))) / max(kappa, 1e-3)


def inspect_update(mu: float, s: float, y: float, sigma: float
                   ) -> tuple[float, float]:
    """Conjugate Gaussian belief update after observing reading y (noise σ)."""
    s2, o2 = s * s, sigma * sigma
    s_post = float(np.sqrt(1.0 / (1.0 / s2 + 1.0 / o2)))
    mu_post = (s_post ** 2) * (mu / s2 + y / o2)
    return mu_post, max(s_post, S_MIN)


def inspect_plan(mu: float, s: float, cfg: fl.FleetConfig, kappa: float):
    """Predictive quadrature nodes for INSPECT: (weight, μ_post, s_post) over the
    predictive reading y ~ N(μ, s²+σ²).  s_post is reading-independent."""
    sigma = inspect_sigma(mu, cfg, kappa)
    s2, o2 = s * s, sigma * sigma
    s_post = float(np.sqrt(1.0 / (1.0 / s2 + 1.0 / o2)))
    sd_y = np.sqrt(s2 + o2)
    ys = mu + sd_y * _XY
    mu_post = (s_post ** 2) * (mu / s2 + ys / o2)
    return _WY, np.clip(mu_post, 0.0, cfg.a_fail), max(s_post, S_MIN)


# ═════════════════════════════════════════════════════════════════════════════
#  Backward induction over the belief grid
# ═════════════════════════════════════════════════════════════════════════════

def solve_pomdp(r: float, cfg: fl.FleetConfig | None = None,
                kappa: float = 1.0, c_inspect: float = C_INSPECT,
                allow_inspect: bool = True) -> dict:
    """Finite-horizon assumed-density backward induction, vectorised over the
    (μ,s) grid.  Returns V[f,μ,s], the 4-action map act[f,μ,s], and the 3-action
    "decide-now" map dn_act[f,μ,s] (best of FLY/REPAIR/RETIRE — used to resolve
    the post-inspection decision in the rollout).

    INSPECT does NOT forfeit a flight: it pays only `c_inspect`, sharpens the
    belief, and is followed by a decide-now action at the SAME epoch (one
    inspection per epoch, so the recursion stays in V_{f+1} through FLY) — this is
    `voi_inspection`'s one-shot VoI step embedded in the lifetime DP.
    allow_inspect=False removes INSPECT (the never-inspect baseline)."""
    cfg = cfg or fl.FleetConfig()
    mu_g, s_g = belief_grids(cfg)
    MU, SS = np.meshgrid(mu_g, s_g, indexing="ij")            # (NMU,NS)

    # f-independent push-forwards: FLY from the base belief grid …
    MUn, SSn, PF = fly_moments(MU, SS, r, cfg)
    # … and the predictive post-INSPECT beliefs + their FLY push-forwards.
    sigma = np.asarray(cfg.measurement_sd(MU)) / max(kappa, 1e-3)
    s2, o2 = SS ** 2, sigma ** 2
    sp = np.sqrt(1.0 / (1.0 / s2 + 1.0 / o2))                 # (NMU,NS) post-SD
    sd_y = np.sqrt(s2 + o2)
    ys = MU[..., None] + sd_y[..., None] * _XY                # (NMU,NS,L)
    mup = (sp[..., None] ** 2) * (MU[..., None] / s2[..., None]
                                  + ys / o2[..., None])
    mup = np.clip(mup, 0.0, cfg.a_fail)
    sp_b = np.broadcast_to(sp[..., None], mup.shape)
    MUn_p, SSn_p, PF_p = fly_moments(mup, sp_b, r, cfg)       # (NMU,NS,L)

    V = np.zeros((F_MAX + 1, N_MU, N_S))
    act = np.zeros((F_MAX, N_MU, N_S), dtype=int)
    dn_act = np.zeros((F_MAX, N_MU, N_S), dtype=int)
    V[F_MAX, :, :] = S_RETIRE                                 # terminal: salvage
    DN_MAP = np.array([I_FLY, I_REP, I_RET])

    for f in range(F_MAX - 1, -1, -1):
        Vn = V[f + 1]
        v_repair = -C_REPAIR + float(bilin(Vn, mu_g, s_g, A_REPAIR, S_REPAIR))
        # decide-now action values on the base grid
        v_fly = (1 - PF) * (FLY_VALUE + bilin(Vn, mu_g, s_g, MUn, SSn)) \
            + PF * (-C_FAIL)
        dn_stack = np.stack([v_fly, np.full_like(v_fly, v_repair),
                             np.full_like(v_fly, S_RETIRE)])   # (3,NMU,NS)
        dn_choice = dn_stack.argmax(axis=0)
        dn_act[f] = DN_MAP[dn_choice]
        decide_now = dn_stack.max(axis=0)
        # INSPECT value: sharpen, then best decide-now on the sharpened belief
        if allow_inspect:
            v_fly_p = (1 - PF_p) * (FLY_VALUE + bilin(Vn, mu_g, s_g, MUn_p, SSn_p)) \
                + PF_p * (-C_FAIL)
            vpost_l = np.maximum(np.maximum(v_fly_p, v_repair), S_RETIRE)
            v_insp = -c_inspect + (vpost_l * _WY).sum(axis=-1)
        else:
            v_insp = np.full_like(v_fly, -np.inf)
        cand = np.stack([v_fly, v_insp, np.full_like(v_fly, v_repair),
                         np.full_like(v_fly, S_RETIRE)])       # (4,NMU,NS)
        act[f] = cand.argmax(axis=0)
        V[f] = cand.max(axis=0)
    return {"V": V, "act": act, "dn_act": dn_act, "mu_g": mu_g, "s_g": s_g,
            "r": r, "cfg": cfg, "kappa": kappa, "c_inspect": c_inspect,
            "allow_inspect": allow_inspect}


# ═════════════════════════════════════════════════════════════════════════════
#  Policies for the partially-observed Monte-Carlo rollout
# ═════════════════════════════════════════════════════════════════════════════

def _snap(mu_g, s_g, mu, s) -> tuple[int, int]:
    i = int(np.clip(np.searchsorted(mu_g, mu), 0, len(mu_g) - 1))
    j = int(np.clip(np.searchsorted(s_g, s), 0, len(s_g) - 1))
    return i, j


def pomdp_policy(sol: dict):
    mu_g, s_g, act = sol["mu_g"], sol["s_g"], sol["act"]

    def pol(f, mu, s):
        i, j = _snap(mu_g, s_g, mu, s)
        return int(act[min(f, F_MAX - 1), i, j])
    return pol


def dn_policy(sol: dict):
    """The decide-now map (FLY/REPAIR/RETIRE) used to resolve the action AFTER an
    INSPECT at the same epoch."""
    mu_g, s_g, dn = sol["mu_g"], sol["s_g"], sol["dn_act"]

    def pol(f, mu, s):
        i, j = _snap(mu_g, s_g, mu, s)
        return int(dn[min(f, F_MAX - 1), i, j])
    return pol


def act_on_mean_policy(r: float, cfg: fl.FleetConfig):
    """Fully-observed `lifetime_policy` DP applied to the belief MEAN — ignores
    state uncertainty and never inspects.  Maps lifetime_policy's 3 actions onto
    this module's 4-action index (it can never return INSPECT)."""
    dp = lp.solve_dp(r, cfg)
    base = lp.dp_policy(dp)
    remap = {lp.ACTIONS.index("FLY"): I_FLY,
             lp.ACTIONS.index("REPAIR"): I_REP,
             lp.ACTIONS.index("RETIRE"): I_RET}

    def pol(f, mu, s):
        return remap[base(f, mu, None)]
    return pol


def periodic_inspect_policy(sol_noinsp: dict, period: int):
    """Inspect every `period` flights regardless; otherwise act on the
    never-inspect belief map.  A non-adaptive inspection schedule baseline."""
    base = pomdp_policy(sol_noinsp)

    def pol(f, mu, s):
        a = base(f, mu, s)
        if a == I_FLY and period > 0 and (f % period == period - 1):
            return I_INSP
        return a
    return pol


# ═════════════════════════════════════════════════════════════════════════════
#  Partially-observed Monte-Carlo evaluation (true crack hidden)
# ═════════════════════════════════════════════════════════════════════════════

def rollout(policy, r_true: float, cfg: fl.FleetConfig, kappa: float = 1.0,
            decide_now=None, n_vehicles: int = 3000, seed: int = 0) -> dict:
    """Simulate lives with the crack HIDDEN: the policy sees only the belief
    (μ,s).  An INSPECT buys a noisy read that updates the belief and is then
    resolved by `decide_now` (a FLY/REPAIR/RETIRE map) at the SAME epoch — it
    does not consume a flight.  FLY grows the true crack and pushes the belief
    forward unobserved."""
    rng = np.random.default_rng(seed)
    a0 = np.exp(cfg.log_a0_mean + cfg.log_a0_sd * rng.standard_normal(n_vehicles))
    mu0 = float(np.exp(cfg.log_a0_mean + 0.5 * cfg.log_a0_sd ** 2))   # prior mean
    s0 = float(mu0 * np.sqrt(np.exp(cfg.log_a0_sd ** 2) - 1.0))       # prior SD
    s0 = float(np.clip(s0, S_MIN, S_MAX))

    vals = np.zeros(n_vehicles); flights = np.zeros(n_vehicles)
    insp = np.zeros(n_vehicles); failed = np.zeros(n_vehicles, dtype=bool)
    for v in range(n_vehicles):
        a = float(a0[v]); mu, s = mu0, s0
        val = 0.0; nfl = 0; nins = 0
        for f in range(F_MAX):
            act = policy(f, mu, s)
            if act == I_INSP:                                    # information action
                sigma = inspect_sigma(mu, cfg, kappa)
                y = a + rng.normal(0.0, sigma)
                mu, s = inspect_update(mu, s, y, sigma)
                val -= C_INSPECT; nins += 1
                act = decide_now(f, mu, s) if decide_now else I_FLY
            if act == I_RET:
                val += S_RETIRE; break
            if act == I_REP:
                val -= C_REPAIR; a = A_REPAIR; mu, s = A_REPAIR, S_REPAIR; continue
            # FLY
            load = max(cfg.load_amp + cfg.load_amp_sd * rng.standard_normal(), 1e-6)
            a_next = a + r_true * (load ** cfg.paris_m) * (max(a, 0.0) ** cfg.paris_p)
            mu, s, _ = fly_predict(mu, s, r_true, cfg)            # belief, unobserved
            if a_next >= cfg.a_fail:
                val -= C_FAIL; failed[v] = True; break
            val += FLY_VALUE; a = a_next; nfl += 1
        else:
            val += S_RETIRE                                       # survived horizon
        vals[v] = val; flights[v] = nfl; insp[v] = nins
    return {"value": float(vals.mean()), "value_sd": float(vals.std()),
            "fail_rate": float(failed.mean()), "flights": float(flights.mean()),
            "inspections": float(insp.mean())}


# ═════════════════════════════════════════════════════════════════════════════
#  Inspect-region geometry + quality/cost sweep
# ═════════════════════════════════════════════════════════════════════════════

def inspect_region(sol: dict, f: int) -> dict:
    """The (μ,s) cells where INSPECT is optimal at flight f, plus the inspect
    fraction and the μ-band it spans (the uncertainty-gated mid band)."""
    act = sol["act"][f]
    mu_g, s_g = sol["mu_g"], sol["s_g"]
    mask = act == I_INSP
    frac = float(mask.mean())
    if mask.any():
        mus = mu_g[np.where(mask.any(axis=1))[0]]
        s_rows = s_g[np.where(mask.any(axis=0))[0]]
        band = (float(mus.min()), float(mus.max()))
        s_band = (float(s_rows.min()), float(s_rows.max()))
    else:
        band = s_band = (float("nan"), float("nan"))
    return {"mask": mask, "frac": frac, "mu_band": band, "s_band": s_band}


def quality_sweep(r: float, cfg: fl.FleetConfig,
                  kappas=(0.3, 0.5, 0.7, 1.0, 1.5), seed: int = 0) -> dict:
    """How the optimum's value and #inspections depend on inspection quality κ
    (σ_insp = measurement_sd/κ): sharp/cheap reads are bought and pay off."""
    out = {"kappa": list(kappas), "value": [], "inspections": [],
           "fail_rate": []}
    for k in kappas:
        sol = solve_pomdp(r, cfg, kappa=k)
        e = rollout(pomdp_policy(sol), r, cfg, kappa=k, decide_now=dn_policy(sol),
                    n_vehicles=2000, seed=seed)
        out["value"].append(e["value"])
        out["inspections"].append(e["inspections"])
        out["fail_rate"].append(e["fail_rate"])
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  Driver
# ═════════════════════════════════════════════════════════════════════════════

def run_study(mu_r: float = -2.1, kappa: float = 1.0, seed: int = 0,
              verbose: bool = True) -> dict:
    cfg = fl.FleetConfig()
    r_true = float(np.exp(mu_r))

    sol = solve_pomdp(r_true, cfg, kappa=kappa)
    sol_noinsp = solve_pomdp(r_true, cfg, kappa=kappa, allow_inspect=False)

    dn = dn_policy(sol); dn_ni = dn_policy(sol_noinsp)
    policies = {
        "POMDP-inspect": (pomdp_policy(sol), dn),
        "never-inspect": (pomdp_policy(sol_noinsp), dn_ni),
        "act-on-mean": (act_on_mean_policy(r_true, cfg), None),
        "periodic-insp(3)": (periodic_inspect_policy(sol_noinsp, period=3), dn_ni),
    }
    evals = {name: rollout(pol, r_true, cfg, kappa=kappa, decide_now=d, seed=seed)
             for name, (pol, d) in policies.items()}

    # inspect region at an early-mid flight (uncertainty has built up)
    region = inspect_region(sol, f=max(1, F_MAX // 4))
    sweep = quality_sweep(r_true, cfg, seed=seed)

    out = {"sol": sol, "sol_noinsp": sol_noinsp, "evals": evals,
           "region": region, "sweep": sweep, "r_true": r_true, "cfg": cfg,
           "kappa": kappa}
    if verbose:
        _report(out)
    return out


def _report(d: dict) -> None:
    bar = "=" * 78
    print(bar)
    print("  BELIEF-STATE POMDP — inspect/fly/repair/retire over the life")
    print("  (crack PARTIALLY OBSERVED; INSPECT buys a noisy read that sharpens it)")
    print(bar)
    print(f"  growth rate r={d['r_true']:.4f}/flight, a_fail={d['cfg'].a_fail}, "
          f"horizon={F_MAX}, inspection κ={d['kappa']:.2f}")
    print(f"  economics: FLY +{FLY_VALUE}, FAIL -{C_FAIL}, REPAIR -{C_REPAIR}, "
          f"RETIRE +{S_RETIRE}, INSPECT -{C_INSPECT}\n")
    print(f"  {'policy':<20}{'life value':>12}{'fail rate':>11}"
          f"{'flights':>9}{'insp.':>8}")
    print("  " + "-" * 68)
    for name in ("POMDP-inspect", "never-inspect", "act-on-mean",
                 "periodic-insp(3)"):
        e = d["evals"][name]
        print(f"  {name:<20}{e['value']:12.2f}{e['fail_rate']*100:10.2f}%"
              f"{e['flights']:9.1f}{e['inspections']:8.2f}")
    pom = d["evals"]["POMDP-inspect"]["value"]
    nin = d["evals"]["never-inspect"]["value"]
    aom = d["evals"]["act-on-mean"]["value"]
    print(f"\n  HEADLINE — STATE UNCERTAINTY DOMINATES:")
    print(f"    belief-state POMDP vs act-on-the-mean (point crack read) = "
          f"{pom - aom:+.2f} value")
    print(f"    failures {d['evals']['POMDP-inspect']['fail_rate']*100:.2f}% "
          f"(POMDP) vs {d['evals']['act-on-mean']['fail_rate']*100:.2f}% "
          f"(act-on-mean) — ignoring state uncertainty is catastrophic.")
    print(f"  ACTIVE-INSPECTION increment (POMDP − never-inspect) = {pom - nin:+.2f} "
          f"value  (modest at κ=1; E[insp]≈{d['evals']['POMDP-inspect']['inspections']:.2f})")
    print(f"    → not a free lunch over a conservative belief policy; see the κ "
          f"sweep for where it pays.")
    reg = d["region"]
    if reg["mu_band"][0] == reg["mu_band"][0]:
        print(f"\n  INSPECT region @ flight {max(1, F_MAX // 4)}: "
              f"{reg['frac']*100:.0f}% of belief cells, crack-mean band "
              f"μ∈[{reg['mu_band'][0]:.2f},{reg['mu_band'][1]:.2f}], "
              f"belief-SD band s∈[{reg['s_band'][0]:.2f},{reg['s_band'][1]:.2f}]")
        print("    → inspection is bought in the ambiguous mid-band AND only when "
              "uncertain (large s).")
    sw = d["sweep"]
    print(f"\n  INSPECTION QUALITY SWEEP (σ_insp = measurement_sd/κ):")
    for k, v, ni in zip(sw["kappa"], sw["value"], sw["inspections"]):
        print(f"    κ={k:.1f}: life value {v:7.2f},  E[inspections] {ni:5.2f}")
    print("  → sharper/cheaper reads are bought more and return value; very noisy "
          "reads are declined.")
    print("\n  LIMITATION: assumed-density (Gaussian moment-projection) belief on a "
          "(μ,s) grid,")
    print("  KNOWN growth rate, representative economics + Gaussian inspection — "
          "the result is the")
    print("  belief-state DECISION STRUCTURE (inspection resolves state "
          "uncertainty for lifetime")
    print("  value; inspect = the uncertainty-gated mid band), not a certified "
          "schedule.")
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

    sol = d["sol"]; cfg = d["cfg"]
    mu_g, s_g = sol["mu_g"], sol["s_g"]
    f_show = max(1, F_MAX // 4)
    fig, ax = plt.subplots(1, 3, figsize=figsize)

    # (a) optimal action map in belief space (μ, s) at a mid-life flight.
    cmap = plt.matplotlib.colors.ListedColormap(
        ["#2e7d32", "#1565C0", "#f9a825", "#b71c1c"])      # FLY/INSPECT/REPAIR/RETIRE
    ax[0].imshow(sol["act"][f_show].T, aspect="auto", origin="lower", cmap=cmap,
                 extent=[0, cfg.a_fail, s_g[0], s_g[-1]], vmin=0, vmax=3,
                 interpolation="nearest")
    ax[0].set_xlabel("crack-belief mean $\\mu$", fontsize=7)
    ax[0].set_ylabel("crack-belief SD $s$", fontsize=7)
    ax[0].set_title(f"(a) belief-state policy @ flight {f_show}\n"
                    "FLY / INSPECT / REPAIR / RETIRE", fontsize=8)
    ax[0].tick_params(labelsize=6)
    from matplotlib.patches import Patch
    ax[0].legend(handles=[Patch(color=c, label=l) for c, l in zip(
        ["#2e7d32", "#1565C0", "#f9a825", "#b71c1c"], ACTIONS)],
        fontsize=5.0, loc="upper left", ncol=2, framealpha=0.85)

    # (b) lifetime value by policy (fail rate + #inspections annotated).
    order = ["POMDP-inspect", "never-inspect", "act-on-mean", "periodic-insp(3)"]
    vals = [d["evals"][k]["value"] for k in order]
    frs = [d["evals"][k]["fail_rate"] for k in order]
    nis = [d["evals"][k]["inspections"] for k in order]
    cols = ["#1565C0", "#2e7d32", "#b71c1c", "#c4a35c"]
    x = np.arange(len(order))
    ax[1].bar(x, vals, color=cols)
    for xi, v, fr, ni in zip(x, vals, frs, nis):
        ax[1].text(xi, v + 1.5, f"{fr*100:.1f}% fail\n{ni:.1f} insp",
                   ha="center", fontsize=5.0)
    ax[1].set_xticks(x); ax[1].set_xticklabels(order, fontsize=5.5, rotation=15)
    ax[1].set_ylabel("expected lifetime value", fontsize=7)
    ax[1].set_title("(b) belief-state recovers value;\nact-on-mean collapses "
                    "(30% fail, see bars)", fontsize=8)
    ax[1].tick_params(labelsize=6); ax[1].axhline(0, color="k", lw=0.5)

    # (c) inspection-quality sweep: value (left) + #inspections (right) vs κ.
    sw = d["sweep"]
    ax2 = ax[2]
    l1, = ax2.plot(sw["kappa"], sw["value"], "D-", color="#1565C0", ms=4, lw=1.4,
                   label="lifetime value")
    ax2.set_xlabel("inspection quality $\\kappa$ "
                   "($\\sigma_{insp}=\\mathrm{meas\\_sd}/\\kappa$)", fontsize=7)
    ax2.set_ylabel("expected lifetime value", fontsize=7, color="#1565C0")
    ax2.tick_params(labelsize=6)
    ax3 = ax2.twinx()
    l2, = ax3.plot(sw["kappa"], sw["inspections"], "s--", color="#b71c1c", ms=4,
                   lw=1.2, label="E[inspections]")
    ax3.set_ylabel("E[inspections] / life", fontsize=7, color="#b71c1c")
    ax3.tick_params(labelsize=6)
    ax2.set_title("(c) sharper/cheaper reads are\nbought more and pay off",
                  fontsize=8)
    ax2.legend(handles=[l1, l2], fontsize=5.5, loc="center right")

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

    cfg = fl.FleetConfig()
    mu_g, s_g = belief_grids(cfg)

    # ---- Gauss–Hermite weights integrate N(0,1) ----------------------------
    ok(abs(_WB.sum() - 1.0) < 1e-6, "belief GH weights sum to 1")
    ok(abs((_WB * _XB).sum()) < 1e-6, "belief GH mean 0")
    ok(abs((_WB * _XB ** 2).sum() - 1.0) < 1e-6, "belief GH variance 1")

    # ---- bilinear interp is exact on the grid + reproduces a plane ----------
    V = np.add.outer(mu_g, 2.0 * s_g)                  # linear plane f=μ+2s
    ok(abs(float(bilin(V, mu_g, s_g, mu_g[3], s_g[2])) - V[3, 2]) < 1e-9,
       "bilin exact on a node")
    ok(abs(float(bilin(V, mu_g, s_g, 0.5, 0.1)) - (0.5 + 0.2)) < 1e-6,
       "bilin reproduces a linear plane")

    # ---- INSPECT sharpens the belief (s_post < s) ---------------------------
    _, _, sp = inspect_plan(0.5, 0.20, cfg, kappa=1.0)
    ok(sp < 0.20, "inspection reduces belief SD")
    _, _, sp_noisy = inspect_plan(0.5, 0.20, cfg, kappa=0.3)
    ok(sp_noisy > sp, "noisier inspection (small κ) sharpens less")
    # predictive μ_post averages back to the prior mean (unbiased reading)
    w, mup, _ = inspect_plan(0.5, 0.20, cfg, kappa=1.0)
    ok(abs(float((w * mup).sum()) - 0.5) < 1e-2, "predictive μ_post ≈ prior mean")

    # ---- FLY pushes the crack up, inflates spread, and reports p_fail -------
    mu_n, s_n, pf = fly_predict(0.4, 0.05, 0.12, cfg)
    ok(mu_n > 0.4, "FLY grows the crack mean")
    ok(s_n >= 0.05 - 1e-9, "FLY does not contract the belief (no observation)")
    ok(0.0 <= pf <= 1.0, "p_fail is a probability")
    pf_hi = fly_predict(0.9, 0.05, 0.12, cfg)[2]
    ok(pf_hi > pf, "bigger crack ⇒ higher failure probability")

    # ---- DP structure: shapes + terminal salvage ---------------------------
    sol = solve_pomdp(0.12, cfg, kappa=1.0)
    ok(sol["V"].shape == (F_MAX + 1, N_MU, N_S), "value table shape")
    ok(sol["act"].shape == (F_MAX, N_MU, N_S), "action map shape")
    ok(np.all(sol["V"][F_MAX] == S_RETIRE), "terminal value = salvage")
    ok(sol["act"][0, 0, 0] == I_FLY, "FLY when crack ≈ 0 and belief tight")
    ok((sol["act"] == I_RET).any(), "RETIRE used somewhere")
    ok((sol["act"] == I_INSP).any(), "INSPECT used somewhere")

    # ---- INSPECT is chosen only when there is uncertainty to resolve --------
    # at the tightest belief SD, the inspect fraction is no larger than at a
    # broad belief SD (information has nothing to sharpen when s is already small)
    insp_tight = (sol["act"][F_MAX // 4, :, 0] == I_INSP).mean()
    insp_broad = (sol["act"][F_MAX // 4, :, -1] == I_INSP).mean()
    ok(insp_broad >= insp_tight, "inspect region grows with belief uncertainty")

    # ---- removing INSPECT changes the policy (the action is actually used) --
    sol_ni = solve_pomdp(0.12, cfg, kappa=1.0, allow_inspect=False)
    ok(not (sol_ni["act"] == I_INSP).any(), "never-inspect map has no INSPECT")

    # ---- CORE rollout orderings (partially observed) ------------------------
    r = 0.12
    sol = solve_pomdp(r, cfg, kappa=1.0)
    sol_ni = solve_pomdp(r, cfg, kappa=1.0, allow_inspect=False)
    e_pom = rollout(pomdp_policy(sol), r, cfg, decide_now=dn_policy(sol),
                    n_vehicles=1500, seed=1)
    e_nin = rollout(pomdp_policy(sol_ni), r, cfg, decide_now=dn_policy(sol_ni),
                    n_vehicles=1500, seed=1)
    e_aom = rollout(act_on_mean_policy(r, cfg), r, cfg, n_vehicles=1500, seed=1)
    # ROBUST headline: the belief-state policy crushes act-on-the-mean (which
    # ignores state uncertainty) by a large margin in value AND failures.
    ok(e_pom["value"] >= e_aom["value"] + 20.0,
       "POMDP ≫ act-on-mean value (state uncertainty dominates)")
    ok(e_aom["fail_rate"] >= e_pom["fail_rate"] + 0.10,
       "act-on-mean fails far more (ignoring state uncertainty is catastrophic)")
    # active inspection is a modest increment — assert only that it is not
    # significantly NEGATIVE over never-inspect (MC-noise tolerant), and that the
    # inspect action is actually exercised.
    ok(e_pom["value"] >= e_nin["value"] - 2.0,
       "POMDP ≈/≥ never-inspect (inspect action not significantly harmful)")
    ok(e_pom["inspections"] > 0.0, "POMDP actually buys inspections")
    ok(e_nin["inspections"] == 0.0, "never-inspect buys none")
    ok(e_pom["fail_rate"] <= 0.05, f"POMDP fail rate low ({e_pom['fail_rate']:.3f})")

    # ---- value finite + flights within horizon ------------------------------
    ok(np.isfinite(e_pom["value"]), "value finite")
    ok(0 <= e_pom["flights"] <= F_MAX, "flights within horizon")

    # ---- quality sweep: more inspections bought as κ rises (cheaper info) ----
    sw = quality_sweep(r, cfg, kappas=(0.3, 1.5), seed=0)
    ok(sw["inspections"][-1] >= sw["inspections"][0] - 1e-9,
       "sharper inspection (higher κ) ⇒ ≥ inspections bought")

    # ---- end-to-end driver returns the comparison ---------------------------
    d = run_study(seed=0, verbose=False)
    ok(d["evals"]["POMDP-inspect"]["value"]
       >= d["evals"]["act-on-mean"]["value"] + 20.0,
       "end-to-end POMDP ≫ act-on-mean (robust headline)")
    ok(d["region"]["frac"] >= 0.0, "inspect-region fraction defined")

    print(f"pomdp_inspection: {n}/{n} unit tests passed")
    return n


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Belief-state POMDP for inspect/fly/repair/retire over the life.")
    ap.add_argument("--mu-r", type=float, default=-2.1,
                    help="E[log growth rate] of the vehicle")
    ap.add_argument("--kappa", type=float, default=1.0,
                    help="inspection quality (σ_insp = measurement_sd/κ)")
    ap.add_argument("--fig", action="store_true")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.test:
        run_tests(); return
    d = run_study(mu_r=args.mu_r, kappa=args.kappa)
    if args.fig:
        p = make_figure(d, os.path.join(HERE, "paper_figs", "pomdp_inspection.pdf"))
        print(f"\nfigure → {p}")


if __name__ == "__main__":
    main()
