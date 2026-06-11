"""
design_feedback.py — Stage-5 design-evolution layer (closes the SHM loop).

Pipeline position (CFRP NDT, WCCM2026 / thesis), the Stage 0→5 reusable-
vehicle stack:
  Stage-0 Mahalanobis screening → Stage-1 GNN classification
  → Stage-2 FMPE amortised posterior θ=(cx,cy,layer,log2size)
  → Stage-3 phase-field growth prognosis  (cfrp_phasefield_2d.py)
  → Stage-4 hierarchical-Bayes FLEET learning (fleet_learning.py)
  → Stage-5 THIS MODULE: design feedback.

What it does (the loop-closing stage).  Stage-4 accumulates, across the whole
recovered fleet, WHERE and HOW BADLY defects grow — a fleet defect
distribution over θ=(cx,cy,layer,log2size) and the flight load level.  Stage-5
asks the inverse-design question: given that the fleet WILL keep arriving with
defects drawn from that distribution, what LAMINATE DESIGN (ply layup +
interface toughening) minimises the EXPECTED crack growth over that
distribution — and, crucially for a reusable vehicle, the WORST-CASE
(tail / CVaR) growth?  We Bayesian-optimise a small, manufacturable design
vector against a robust expected-growth objective evaluated by the exact FD
phase-field forward model.

────────────────────────────────────────────────────────────────────────────
WHY the FNO/CNN surrogate (crack_surrogate.py) is NOT the forward model here
────────────────────────────────────────────────────────────────────────────
The Stage-3 neural surrogate is trained with inputs (d0 seeded-damage field,
load) ONLY — it never sees the laminate layup.  `LaminateConfig` is held at
its default (0/±30)_s, 8-ply layup for every training sample (crack_surrogate
._sim_one builds `LaminateConfig(nx, ny)` with default angles).  Therefore:

  * the surrogate has NO INPUT through which a design change (ply angle,
    Gc_interface_ratio) could even be expressed, and
  * any design that perturbs the layup is fully OUT-OF-DISTRIBUTION for it.

Using it as the Stage-5 forward model would mean optimising a model that is
blind to the very variables we are designing — it would return the same
prediction for every design.  So Stage-5 design evaluation MUST call the exact
FD phase-field forward `cfrp_phasefield_2d.simulate_growth`, which DOES take
the layup via `LaminateConfig(ply_angles_deg=..., Gc_interface_ratio=...)`.
This is slower (~1.4 s / sim at the coarse grid used here) but honest.  A
LAYUP-CONDITIONED neural operator — a surrogate that takes the ply-angle /
interface-toughness fields as additional input channels so design search can
be amortised — is the clean motivation for the Keio Phase-3 richer-surrogate
plan and is left as future work.

────────────────────────────────────────────────────────────────────────────
Honest accounting (read before quoting numbers)
────────────────────────────────────────────────────────────────────────────
  (a) FD-only forward (above): the surrogate cannot see the layup.
  (b) Coarse grid (nx≈52) + a SYNTHETIC fleet defect prior (interior cx, load
      near the sub-/super-critical transition) standing in for the Stage-4
      posterior.  `fleet_defect_prior` will derive load/centroid statistics
      from a fitted `fleet_learning.FleetPosterior` when one is passed, but the
      θ-geometry prior (cx, layer, size) is a documented synthetic envelope
      matching crack_surrogate's in-distribution box — Stage-4's latent is a
      scalar growth-rate s_v, not a θ-field, so the geometry prior cannot be
      read off it directly.
  (c) AT2 nominal strength scales as ℓ^(-1/2) (cfrp_phasefield_2d memo §6), so
      ABSOLUTE growth magnitudes are indicative only.  The trustworthy output
      is the design RANKING / TREND: which layup+interface reduces expected and
      worst-case growth, and by roughly how much.
  (d) Layup-conditioned operator = future work (see above).

Objective.  At loads just above the static-critical strain the growth flag is
near-binary (P(grow)≈0 below, ≈1 above) and saturates, so a pure P(grow)
objective has little gradient.  We therefore use a SMOOTH primary objective —
mean log-growth  E[log(1+rel_growth)]  over the fleet defect draws — and ALSO
report P(grow).  Worst-case safety is captured by the CVaR (mean of the
worst-q fraction) of the SAME per-defect log-growth.  A small manufacturability
/ weight penalty discourages gratuitous interface toughening.  Lower = better.

Optimiser.  Bayesian optimisation: skopt.gp_minimize (Expected Improvement) if
available, else a compact hand-rolled RBF-GP + EI on the normalised cube (no
new hard dependency).  Each objective eval runs N FD sims; the N defect draws
are parallelised with a multiprocessing pool (cf. crack_surrogate.py).

CPU, numpy/scipy only.  ~30–50 evals × N≈10 FD sims ≈ 15–25 min on a 12-core
box at the coarse grid.
"""
from __future__ import annotations

import os
import time
import json
import hashlib
from dataclasses import dataclass, field, asdict

import numpy as np

from cfrp_phasefield_2d import (LaminateConfig, seed_defect, simulate_growth,
                                calibrate_thresholds)

# ─────────────────────────────────────────────────────────────────────────────
#  Constants — coarse FD grid + design landscape
# ─────────────────────────────────────────────────────────────────────────────

GRID_NX, GRID_NY = 52, 44        # coarse design-eval grid (~1.4 s / FD sim)

# Design-variable bounds (manufacturable bands).
OFFAXIS_LO, OFFAXIS_HI = 15.0, 75.0     # off-axis ply angle |θ| [deg]
GCINT_LO,   GCINT_HI   = 0.30, 0.90     # interface toughening Gc_int/Gc_ply
REINF_LO,   REINF_HI   = 1.00, 1.40     # local interface-toughness multiplier

# Baseline design = the (0/±30)_s default layup, default interface ratio,
# no extra reinforcement.  (Matches cfrp_phasefield_2d.LaminateConfig defaults.)
BASELINE_DESIGN = np.array([30.0, 0.40, 1.00])

# Manufacturability / weight penalty.  Interleaf toughening and local
# reinforcement add areal weight; we charge a small linear cost in the
# *fraction of the band's toughening* used, scaled to be << a unit of
# log-growth so it only breaks ties between near-equal designs.
W_GCINT = 0.06      # per unit of (Gc_int - GCINT_LO)/(GCINT_HI-GCINT_LO)
W_REINF = 0.04      # per unit of (reinf  - REINF_LO )/(REINF_HI -REINF_LO )

CVAR_Q = 0.20       # CVaR tail fraction (worst 20 % of defects)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "results", "design_feedback")


# ─────────────────────────────────────────────────────────────────────────────
#  1.  Design vector  →  LaminateConfig   (balanced + symmetric by construction)
# ─────────────────────────────────────────────────────────────────────────────

def balanced_symmetric_layup(offaxis_deg: float) -> tuple:
    """(0/±θ)_s 8-ply layup as a tuple of ply angles.

    Construction GUARANTEES the laminate is BALANCED (every +θ has a matching
    −θ) and SYMMETRIC (the stack reads the same top-to-bottom), so it is a
    manufacturable layup for any θ — no membrane–bending coupling.  θ is the
    off-axis magnitude; the 0° plies are kept fixed (axial stiffness).
    """
    t = float(offaxis_deg)
    return (0.0, t, -t, 0.0, 0.0, -t, t, 0.0)


def make_config(design, nx: int = GRID_NX, ny: int = GRID_NY,
                **cfg_kwargs) -> LaminateConfig:
    """Map a design vector → a valid (balanced+symmetric) LaminateConfig.

    design = (offaxis_deg, gc_interface_ratio, reinf_mult):
      * offaxis_deg        — |θ| of the ±θ plies, clipped to [15,75]°
                             (steeper plies deflect the dominant matrix crack).
      * gc_interface_ratio — Gc_int/Gc_ply interface toughness, clipped to
                             [0.30,0.90] (tougher interleaf / stitching).
      * reinf_mult         — local interface-toughness multiplier in [1.0,1.4]
                             applied ON TOP of the ratio (a modest extra
                             reinforcement; folded into beta_interface band via
                             the effective ratio so it stays a single FD config).

    The reinforcement multiplier scales the *effective* interface ratio (capped
    at GCINT_HI so the config never leaves the calibrated band).
    """
    design = np.asarray(design, dtype=float).ravel()
    if design.shape[0] == 2:
        offaxis, gcint = design
        reinf = 1.0
    else:
        offaxis, gcint, reinf = design[:3]
    offaxis = float(np.clip(offaxis, OFFAXIS_LO, OFFAXIS_HI))
    gcint   = float(np.clip(gcint,   GCINT_LO,   GCINT_HI))
    reinf   = float(np.clip(reinf,   REINF_LO,   REINF_HI))

    gcint_eff = float(np.clip(gcint * reinf, GCINT_LO, GCINT_HI))
    return LaminateConfig(nx=nx, ny=ny,
                          ply_angles_deg=balanced_symmetric_layup(offaxis),
                          Gc_interface_ratio=gcint_eff,
                          **cfg_kwargs)


def manufacturability_penalty(design) -> float:
    """Small linear weight penalty for toughening / reinforcement (lower=cheaper).

    Charged in the design's normalised position within the band, so the
    baseline (low toughening) pays ~0 and the fully-toughened corner pays
    ~W_GCINT + W_REINF.  Kept << one unit of log-growth (tie-breaker only).
    """
    design = np.asarray(design, dtype=float).ravel()
    offaxis, gcint = design[0], design[1]
    reinf = design[2] if design.shape[0] >= 3 else 1.0
    f_gc = (np.clip(gcint, GCINT_LO, GCINT_HI) - GCINT_LO) / (GCINT_HI - GCINT_LO)
    f_re = (np.clip(reinf, REINF_LO, REINF_HI) - REINF_LO) / (REINF_HI - REINF_LO)
    return float(W_GCINT * f_gc + W_REINF * f_re)


# ─────────────────────────────────────────────────────────────────────────────
#  2.  Fleet defect distribution  (Stage-4 → robust-design prior)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DefectPrior:
    """Fleet defect distribution over θ=(cx,cy,layer,log2size) and flight load.

    cx ~ U(cx_lo,cx_hi) interior; cy carried for API compatibility (the 2D
    section is taken through the defect — cfrp_phasefield_2d.seed_defect
    ignores cy); layer ~ U(layer_lo,layer_hi) interior plies; log2size ~
    U(l2s_lo,l2s_hi) (defect 2**log2size elements).  load = the flight peel
    strain at which growth is judged (one scalar; near the sub-/super-critical
    transition where design has leverage).
    """
    cx_lo: float = 0.25
    cx_hi: float = 0.75
    layer_lo: float = 4.0
    layer_hi: float = 14.0
    l2s_lo: float = 0.6
    l2s_hi: float = 2.0
    load: float = 0.11
    source: str = "synthetic"

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Return (n, 4) array of θ draws (cx, cy, layer, log2size)."""
        cx = rng.uniform(self.cx_lo, self.cx_hi, n)
        cy = np.full(n, 0.5)                       # section plane only
        layer = rng.uniform(self.layer_lo, self.layer_hi, n)
        l2s = rng.uniform(self.l2s_lo, self.l2s_hi, n)
        return np.column_stack([cx, cy, layer, l2s])


def fleet_defect_prior(fleet_posterior=None,
                       fleet_cfg=None) -> DefectPrior:
    """Build the Stage-5 robust-design defect prior.

    If a `fleet_learning.FleetPosterior` (and its `FleetConfig`) is supplied,
    the FLIGHT LOAD severity is scaled from the fleet's posterior-mean growth
    rate r̄ = E[exp s_v]: a faster-growing fleet (higher r̄) is mapped to a
    higher design peel strain, so the design is hardened against the fleet
    actually observed.  The θ-GEOMETRY envelope (cx, layer, size) remains the
    documented synthetic box (honest accounting (b)): Stage-4's latent is a
    scalar growth rate, not a θ-field, so the geometry prior is not derivable
    from it.

    With no posterior, returns the default synthetic prior centred on the
    super-critical transition load (~0.11) where the layup/interface design has
    the most leverage.
    """
    prior = DefectPrior()
    if fleet_posterior is None:
        return prior
    try:
        r_bar = float(np.median(fleet_posterior.r_v_mean()))
    except Exception:
        return prior
    # map fleet growth rate (calibrated mean r≈exp(-3)≈0.05) onto the FD peel
    # load band [0.10,0.13]: r̄ at the calibrated point → 0.11, scale gently.
    r_ref = float(np.exp(fleet_cfg.mu_phi_true)) if fleet_cfg is not None else 0.05
    scale = np.clip(r_bar / max(r_ref, 1e-6), 0.5, 2.0)
    prior.load = float(np.clip(0.11 * (0.85 + 0.15 * scale), 0.10, 0.13))
    prior.source = "fleet_posterior"
    return prior


# ─────────────────────────────────────────────────────────────────────────────
#  3.  Robust objective  (FD forward, parallel over defect draws)
# ─────────────────────────────────────────────────────────────────────────────

def _sim_one_defect(args) -> dict:
    """Worker: one FD simulate_growth for (design, θ, load).  Top-level/picklable.

    Rebuilds the LaminateConfig inside the worker (cfg is cheap; the FD solve
    dominates).  Returns the per-defect grow flag and log-growth.
    """
    from cfrp_phasefield_2d import seed_defect as _seed, simulate_growth as _sim
    import design_feedback as _df

    design, theta, load, nx, ny = args
    cfg = _df.make_config(design, nx=nx, ny=ny)
    cx, cy, layer, l2s = theta
    d0 = _seed(cx, cy, layer, l2s, cfg)
    res = _sim(d0, float(load), cfg)
    return {"grown": bool(res.grown),
            "rel_growth": float(res.rel_growth),
            "log_growth": float(np.log1p(max(res.rel_growth, 0.0)))}


@dataclass
class ObjResult:
    """Outcome of one robust-objective evaluation."""
    objective: float          # penalised mean log-growth (LOWER = better)
    mean_log_growth: float
    cvar_log_growth: float    # CVaR_q of per-defect log-growth (worst tail)
    p_grow: float             # mean grow flag
    penalty: float
    per_defect_log: np.ndarray
    per_defect_grown: np.ndarray


def cvar(values: np.ndarray, q: float = CVAR_Q) -> float:
    """CVaR_q (mean of the worst-q fraction, i.e. largest values).

    At least one sample is always included, so CVaR >= mean by construction.
    """
    v = np.sort(np.asarray(values, dtype=float))[::-1]      # descending
    k = max(1, int(np.ceil(q * len(v))))
    return float(v[:k].mean())


def evaluate_design(design, defects: np.ndarray, load: float,
                    nx: int = GRID_NX, ny: int = GRID_NY,
                    n_workers: int | None = None,
                    cvar_q: float = CVAR_Q) -> ObjResult:
    """Robust expected-growth objective for one design over the fleet defects.

    For each θ in `defects` run the exact FD forward at `load`; aggregate to
      mean_log_growth  = E[log(1+rel_growth)]   (smooth primary signal)
      cvar_log_growth  = CVaR_q (worst tail, reusable-vehicle safety)
      p_grow           = E[grow flag]
    and add the manufacturability penalty:
      objective = mean_log_growth + penalty   (LOWER = better).

    Defect draws are parallelised with a multiprocessing pool (cf.
    crack_surrogate.py).  Set n_workers=1 (or 0) for serial / tests.
    """
    defects = np.atleast_2d(np.asarray(defects, dtype=float))
    jobs = [(np.asarray(design, float), tuple(map(float, th)), float(load),
             nx, ny) for th in defects]

    if n_workers is None:
        n_workers = min(len(jobs), os.cpu_count() or 1)
    if n_workers and n_workers > 1 and len(jobs) > 1:
        import multiprocessing as mp
        with mp.Pool(n_workers) as pool:
            recs = pool.map(_sim_one_defect, jobs, chunksize=1)
    else:
        recs = [_sim_one_defect(j) for j in jobs]

    logs = np.array([r["log_growth"] for r in recs])
    grown = np.array([r["grown"] for r in recs])
    mean_log = float(logs.mean())
    cvar_log = cvar(logs, cvar_q)
    pen = manufacturability_penalty(design)
    return ObjResult(objective=mean_log + pen,
                     mean_log_growth=mean_log,
                     cvar_log_growth=cvar_log,
                     p_grow=float(grown.mean()),
                     penalty=pen,
                     per_defect_log=logs,
                     per_defect_grown=grown)


# ─────────────────────────────────────────────────────────────────────────────
#  4.  Bayesian optimisation  (skopt EI if available, else hand-rolled RBF-GP+EI)
# ─────────────────────────────────────────────────────────────────────────────

DESIGN_BOUNDS = [(OFFAXIS_LO, OFFAXIS_HI),
                 (GCINT_LO,   GCINT_HI),
                 (REINF_LO,   REINF_HI)]


def _normalise(x, bounds):
    x = np.asarray(x, float)
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    return (x - lo) / (hi - lo)


def _denormalise(u, bounds):
    u = np.asarray(u, float)
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    return lo + u * (hi - lo)


class _RBFGP:
    """Tiny zero-mean RBF Gaussian-process regressor on the unit cube.

    GP with constant length-scale RBF kernel + noise jitter.  Enough for EI
    acquisition over a 3-D normalised cube and ~40 points; not a full
    marginal-likelihood-tuned GP (we standardise y and fix a sane length scale).
    """
    def __init__(self, length_scale: float = 0.35, noise: float = 1e-4):
        self.ls = length_scale
        self.noise = noise

    def _kernel(self, A, B):
        d2 = (np.sum(A**2, 1)[:, None] + np.sum(B**2, 1)[None, :]
              - 2.0 * A @ B.T)
        return np.exp(-0.5 * np.maximum(d2, 0.0) / self.ls**2)

    def fit(self, X, y):
        self.X = np.atleast_2d(X)
        y = np.asarray(y, float)
        self.y_mean, self.y_std = y.mean(), y.std() + 1e-9
        yn = (y - self.y_mean) / self.y_std
        K = self._kernel(self.X, self.X) + self.noise * np.eye(len(self.X))
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, yn))
        return self

    def predict(self, Xq):
        Xq = np.atleast_2d(Xq)
        Ks = self._kernel(self.X, Xq)
        mu = (Ks.T @ self.alpha) * self.y_std + self.y_mean
        v = np.linalg.solve(self.L, Ks)
        var = (1.0 - np.sum(v**2, 0)) * self.y_std**2
        return mu, np.sqrt(np.maximum(var, 1e-12))


def _expected_improvement(mu, sigma, y_best, xi=0.01):
    """EI for MINIMISATION (improvement below the incumbent y_best)."""
    from math import sqrt, pi
    sigma = np.maximum(sigma, 1e-9)
    imp = y_best - mu - xi
    z = imp / sigma
    Phi = 0.5 * (1.0 + np.vectorize(_erf)(z / np.sqrt(2.0)))
    phi = np.exp(-0.5 * z**2) / np.sqrt(2.0 * pi)
    return imp * Phi + sigma * phi


def _erf(x):
    # Abramowitz-Stegun 7.1.26 (≈1e-7) — avoids a scipy.special import here.
    t = 1.0 / (1.0 + 0.3275911 * abs(x))
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return float(np.sign(x) * y)


@dataclass
class BOResult:
    best_design: np.ndarray
    best_objective: float
    best_obj_result: ObjResult
    X: np.ndarray                 # all evaluated designs (n_eval, d)
    y: np.ndarray                 # all objectives (n_eval,)
    y_best_curve: np.ndarray      # running best objective (n_eval,)
    obj_results: list             # ObjResult per eval
    backend: str
    wall_clock_s: float


def _hand_rolled_bo(objective_fn, bounds, n_init, n_iter, rng,
                    n_cand=512, verbose=True):
    """Compact GP-EI BO on the normalised cube (minimisation)."""
    d = len(bounds)
    U = rng.random((n_init, d))                       # Latin-ish random init
    X = [_denormalise(u, bounds) for u in U]
    obj_results, y = [], []
    for i, x in enumerate(X):
        r = objective_fn(x)
        obj_results.append(r); y.append(r.objective)
        if verbose:
            print(f"  [init {i+1}/{n_init}] obj={r.objective:.4f} "
                  f"design={np.round(x,3)}")
    X = [np.asarray(x, float) for x in X]; y = list(y)

    for it in range(n_iter):
        Xn = np.array([_normalise(x, bounds) for x in X])
        gp = _RBFGP().fit(Xn, np.array(y))
        cand = rng.random((n_cand, d))
        mu, sig = gp.predict(cand)
        ei = _expected_improvement(mu, sig, np.min(y))
        u_next = cand[int(np.argmax(ei))]
        x_next = _denormalise(u_next, bounds)
        r = objective_fn(x_next)
        X.append(np.asarray(x_next, float)); y.append(r.objective)
        obj_results.append(r)
        if verbose:
            print(f"  [iter {it+1}/{n_iter}] obj={r.objective:.4f} "
                  f"best={np.min(y):.4f} design={np.round(x_next,3)}")
    return np.array(X), np.array(y), obj_results


def optimise_design(defects: np.ndarray, load: float,
                    n_init: int = 8, n_iter: int = 24,
                    nx: int = GRID_NX, ny: int = GRID_NY,
                    n_workers: int | None = None,
                    seed: int = 0, prefer_skopt: bool = True,
                    cvar_q: float = CVAR_Q, verbose: bool = True) -> BOResult:
    """Bayesian-optimise the design vector against the robust objective.

    Budget = n_init + n_iter objective evaluations; each eval runs len(defects)
    FD sims (parallelised).  Tries skopt.gp_minimize (EI) first, falls back to
    the hand-rolled RBF-GP+EI if skopt is unavailable (no new dependency).
    """
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    obj_results_store = []

    def objective_fn(design):
        r = evaluate_design(design, defects, load, nx=nx, ny=ny,
                            n_workers=n_workers, cvar_q=cvar_q)
        return r

    backend = "hand_rolled_gp_ei"
    skopt_ok = False
    if prefer_skopt:
        try:
            from skopt import gp_minimize           # noqa: F401
            from skopt.space import Real             # noqa: F401
            skopt_ok = True
        except Exception:
            skopt_ok = False

    if skopt_ok:
        from skopt import gp_minimize
        from skopt.space import Real
        backend = "skopt_gp_minimize"
        space = [Real(lo, hi) for (lo, hi) in DESIGN_BOUNDS]
        X_log, y_log = [], []

        def _scalar(design):
            r = objective_fn(design)
            X_log.append(np.asarray(design, float)); y_log.append(r.objective)
            obj_results_store.append(r)
            if verbose:
                print(f"  [skopt {len(y_log)}] obj={r.objective:.4f} "
                      f"design={np.round(design,3)}")
            return r.objective

        res = gp_minimize(_scalar, space, n_calls=n_init + n_iter,
                          n_initial_points=n_init, acq_func="EI",
                          random_state=seed)
        X = np.array(X_log); y = np.array(y_log)
        obj_results = obj_results_store
    else:
        X, y, obj_results = _hand_rolled_bo(
            objective_fn, DESIGN_BOUNDS, n_init, n_iter, rng, verbose=verbose)

    y_best_curve = np.minimum.accumulate(y)
    best_i = int(np.argmin(y))
    return BOResult(best_design=X[best_i], best_objective=float(y[best_i]),
                    best_obj_result=obj_results[best_i], X=X, y=y,
                    y_best_curve=y_best_curve, obj_results=obj_results,
                    backend=backend, wall_clock_s=time.perf_counter() - t0)


# ─────────────────────────────────────────────────────────────────────────────
#  5.  E[remaining flights]  (Stage-3 fatigue API, before vs after)
# ─────────────────────────────────────────────────────────────────────────────

def _fatigue_one(args) -> int:
    """Worker: flights survived before first growth for one defect.  Picklable."""
    from cfrp_phasefield_2d import simulate_fatigue_flights
    import design_feedback as _df
    design, theta, load, n_flights, nx, ny = args
    cfg = _df.make_config(design, nx=nx, ny=ny)
    cx, cy, layer, l2s = theta
    d0 = _df.seed_defect(cx, cy, layer, l2s, cfg)
    res = simulate_fatigue_flights(d0, float(load), n_flights, cfg)
    return (res.fail_flight - 1) if res.fail_flight is not None else n_flights


def expected_remaining_flights(design, defects: np.ndarray, load: float,
                               n_flights: int = 12, nx: int = GRID_NX,
                               ny: int = GRID_NY,
                               n_workers: int | None = None) -> float:
    """Mean flights-survived before first growth, over the fleet defects.

    Uses the Stage-3 fatigue forward (simulate_fatigue_flights) at the design's
    LaminateConfig; censored at `n_flights`.  Reported before vs after design.
    Parallelised over defect draws (cf. crack_surrogate.py).
    """
    defects = np.atleast_2d(np.asarray(defects, float))
    jobs = [(np.asarray(design, float), tuple(map(float, th)), float(load),
             n_flights, nx, ny) for th in defects]
    if n_workers is None:
        n_workers = min(len(jobs), os.cpu_count() or 1)
    if n_workers and n_workers > 1 and len(jobs) > 1:
        import multiprocessing as mp
        with mp.Pool(n_workers) as pool:
            surv = pool.map(_fatigue_one, jobs, chunksize=1)
    else:
        surv = [_fatigue_one(j) for j in jobs]
    return float(np.mean(surv))


# ─────────────────────────────────────────────────────────────────────────────
#  6.  Caching helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cache_key(design, load, nx, ny) -> str:
    s = json.dumps([np.round(np.asarray(design, float), 4).tolist(),
                    round(float(load), 4), nx, ny])
    return hashlib.md5(s.encode()).hexdigest()[:16]


def save_result(bo: BOResult, prior: DefectPrior, extra: dict | None = None,
                path: str | None = None) -> str:
    """Persist the optimisation summary (designs, objectives, prior) to JSON."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = path or os.path.join(CACHE_DIR, "design_feedback_result.json")
    out = {
        "backend": bo.backend,
        "wall_clock_s": bo.wall_clock_s,
        "best_design": np.asarray(bo.best_design).tolist(),
        "best_objective": bo.best_objective,
        "X": np.asarray(bo.X).tolist(),
        "y": np.asarray(bo.y).tolist(),
        "y_best_curve": np.asarray(bo.y_best_curve).tolist(),
        "prior": asdict(prior),
    }
    if extra:
        out.update(extra)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  Driver
# ─────────────────────────────────────────────────────────────────────────────

FATIGUE_LOAD = 0.09   # sub-critical peel strain for the E[flights] comparison
                      # (at the quasi-static objective load ~0.11 the seed
                      #  grows in flight 1 for every design, so fatigue life
                      #  only discriminates designs in the sub-critical band).


def run(n_defects: int = 10, n_init: int = 8, n_iter: int = 24,
        nx: int = GRID_NX, ny: int = GRID_NY, seed: int = 0,
        n_workers: int | None = None, verbose: bool = True,
        fatigue_load: float = FATIGUE_LOAD) -> dict:
    """End-to-end Stage-5 run: prior → BO → baseline-vs-optimised report.

    Returns a summary dict (also saved to results/design_feedback/).
    """
    rng = np.random.default_rng(seed)
    prior = fleet_defect_prior()
    defects = prior.sample(n_defects, rng)
    load = prior.load

    if verbose:
        print(f"=== Stage-5 design feedback ===")
        print(f"grid {nx}x{ny}, {n_defects} fleet defects, load={load:.3f}, "
              f"prior={prior.source}")
        print(f"baseline design (0/±30)_s, Gc_int=0.40: {BASELINE_DESIGN}")

    # baseline objective on the SAME defect set (fair comparison)
    base = evaluate_design(BASELINE_DESIGN, defects, load, nx=nx, ny=ny,
                           n_workers=n_workers)
    if verbose:
        print(f"baseline: obj={base.objective:.4f} mean_log={base.mean_log_growth:.4f} "
              f"CVaR={base.cvar_log_growth:.4f} P(grow)={base.p_grow:.2f}")

    bo = optimise_design(defects, load, n_init=n_init, n_iter=n_iter,
                         nx=nx, ny=ny, n_workers=n_workers, seed=seed,
                         verbose=verbose)
    opt = bo.best_obj_result

    # E[remaining flights] before / after, evaluated at a SUB-CRITICAL fatigue
    # load (see FATIGUE_LOAD): at the quasi-static objective load the seed grows
    # in flight 1 for every design, so the fatigue metric only discriminates
    # designs in the sub-critical band.
    erf_base = expected_remaining_flights(BASELINE_DESIGN, defects, fatigue_load,
                                          nx=nx, ny=ny, n_workers=n_workers)
    erf_opt = expected_remaining_flights(bo.best_design, defects, fatigue_load,
                                         nx=nx, ny=ny, n_workers=n_workers)

    def _pct(a, b):
        return 100.0 * (a - b) / a if a > 1e-9 else 0.0

    summary = {
        "load": load,
        "n_defects": n_defects,
        "baseline_design": BASELINE_DESIGN.tolist(),
        "optimised_design": np.asarray(bo.best_design).tolist(),
        "baseline_mean_log_growth": base.mean_log_growth,
        "optimised_mean_log_growth": opt.mean_log_growth,
        "baseline_cvar": base.cvar_log_growth,
        "optimised_cvar": opt.cvar_log_growth,
        "baseline_p_grow": base.p_grow,
        "optimised_p_grow": opt.p_grow,
        "pct_reduction_mean_log_growth": _pct(base.mean_log_growth,
                                              opt.mean_log_growth),
        "pct_reduction_cvar": _pct(base.cvar_log_growth, opt.cvar_log_growth),
        "pct_reduction_p_grow": _pct(base.p_grow, opt.p_grow),
        "fatigue_load": fatigue_load,
        "erf_baseline": erf_base,
        "erf_optimised": erf_opt,
        "backend": bo.backend,
        "wall_clock_s": bo.wall_clock_s,
        "n_evals": int(len(bo.y)),
    }
    save_result(bo, prior, extra={"summary": summary,
                                  "baseline": {"mean_log_growth": base.mean_log_growth,
                                               "cvar": base.cvar_log_growth,
                                               "p_grow": base.p_grow}})
    if verbose:
        print("\n=== RESULT ===")
        for k, v in summary.items():
            print(f"  {k}: {v}")
    return {"summary": summary, "bo": bo, "baseline": base, "prior": prior,
            "defects": defects}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-defects", type=int, default=10)
    ap.add_argument("--n-init", type=int, default=8)
    ap.add_argument("--n-iter", type=int, default=24)
    ap.add_argument("--nx", type=int, default=GRID_NX)
    ap.add_argument("--ny", type=int, default=GRID_NY)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(n_defects=args.n_defects, n_init=args.n_init, n_iter=args.n_iter,
        nx=args.nx, ny=args.ny, seed=args.seed, n_workers=args.workers)
