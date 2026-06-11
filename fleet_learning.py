"""
fleet_learning.py — Stage-4 hierarchical Bayesian fleet learning (Concept A).

Pipeline position (CFRP NDT / reusable-vehicle SHM, WCCM2026 / thesis):
  Stage-0 Mahalanobis screening → Stage-1 GNN classification
  → Stage-2 FMPE amortised posterior θ = (cx, cy, layer, log2size)
  → Stage-3 cfrp_phasefield_2d: anisotropic phase-field growth prognosis
    + Carrara-2020 fatigue + expected-cost flight clearance (ONE vehicle)
  → Stage-4 THIS MODULE: pool the fleet.

Spec: research/STAGE4_5_RESEARCH_CONCEPTS.md, Concept A.

Idea (the SpaceX "fleet-leader" loop, made explicit and quantitative)
--------------------------------------------------------------------
A reusable fleet of N vehicles flies the same mission repeatedly.  Each
booster has its own manufacturing/lot idiosyncrasy (an effective
fatigue/growth-rate r_v), yet they share a common physics.  Treating each
vehicle in isolation wastes the fleet; pooling naively ignores individual
variation.  A HIERARCHICAL Bayesian model sits in between: a global (fleet)
prior φ = (μ_φ, σ_φ) over the per-vehicle log-growth-rate s_v = log r_v, and
per-flight inspections as the likelihood.  As flights accumulate across the
fleet, φ sharpens, so the prior on a *new* vehicle's r_v at its first
inspection is already tight — fewer inspections are needed to reach the same
clearance confidence.  That is the fleet-leader effect.

Physical basis of the growth law (NOT a re-derivation)
------------------------------------------------------
The full Stage-3 phase-field + Carrara-2020 fatigue solver
(`cfrp_phasefield_2d.simulate_fatigue_flights`) costs ~15 s/vehicle/flight —
far too slow to drive a fleet of N≈20 × F≈30 flights × MCMC.  We use a CHEAP
SURROGATE growth law of the SAME mechanistic shape: a Paris-law-like
accumulation of an effective crack size a per flight,

    a_{f+1} = a_f + r_v · (Δσ_f)^m · a_f^{p},        (surrogate)

with r_v the per-vehicle growth-rate constant.  This mirrors, in closed form,
the monotone fatigue accumulation that Carrara, Ambati, Alessi & De Lorenzis
(2020, CMAME 361:112731) produce through the fatigue history variable ᾱ and
the asymptotic degradation function f(ᾱ) (see cfrp_phasefield_2d's
`fatigue_degradation` / `simulate_fatigue_flights`): repeated sub-critical
flights erode the effective toughness and grow the crack, with HIGHER load
amplitude and HIGHER r_v giving SHORTER life (Wöhler-like).  The surrogate
keeps that ordering while running in microseconds.  Calibrating r_v against
the phase-field fatigue life is Stage-4→3 future work; here r_v lives in the
surrogate's normalised units.

Inference
---------
LogNormal–Normal hierarchy → tractable Metropolis-within-Gibbs / conjugate
updates.  Per vehicle we form a Gaussian "measurement posterior" on s_v from
its inspections (the Stage-0→2 posterior stand-in: a Gaussian centred near the
true growth, with realistic spread).  The hierarchy is then the textbook
Normal–Normal–Inverse-Gamma model:

    s_v        ~ Normal(μ_φ, σ_φ²)                         (vehicle level)
    ŝ_v | s_v  ~ Normal(s_v, τ_v²)        τ_v from inspections (likelihood)
    μ_φ        ~ Normal(m0, v0)                            (hyper-prior)
    σ_φ²       ~ Inverse-Gamma(a0, b0)                     (hyper-prior)

Gibbs: conjugate full conditionals for s_v, μ_φ, σ_φ² (all closed-form
Normal / Inverse-Gamma).  TMCMC is overkill for this conjugate structure
(per the spec); we keep it simple and correct.  As the fleet grows the φ
posterior concentrates on φ_true (consistency), and low-data vehicles are
shrunk toward the fleet mean (partial pooling).

CPU, numpy/scipy only, seeded, reproducible — the whole study runs in
seconds.

Honest limitations
-------------------
  * Surrogate growth law (closed-form Paris/Carrara-shaped), NOT the phase
    field — see above; r_v in normalised units.
  * The per-inspection measurement posterior is a Gaussian stand-in for the
    real Stage-0→2 (Mahalanobis→GNN→FMPE) posterior, not that pipeline run
    end-to-end.
  * Single scalar growth parameter per vehicle (r_v); defect-incidence is
    modelled but not jointly inferred with r_v in the hierarchy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FleetConfig:
    """Fleet simulator + hierarchy configuration (normalised units).

    The TRUE global distribution φ_true governs the per-vehicle log-growth
    rate s_v = log r_v ~ Normal(mu_phi_true, sigma_phi_true²).  Recovering
    (mu_phi_true, sigma_phi_true) from fleet data is the consistency target.
    """
    # fleet / mission size
    n_vehicles: int = 20
    n_flights: int = 30                 # max flights per vehicle (F)
    # TRUE global distribution φ_true over s_v = log r_v.  Calibrated so a
    # ~0.1 seed crack grows detectably (well above measurement noise) toward
    # a_fail over the F-flight horizon: r_v ≈ exp(-3.0) ≈ 0.05 per flight.
    mu_phi_true: float = -3.0           # E[log r_v]
    sigma_phi_true: float = 0.45        # SD[log r_v] (lot-to-lot spread)
    # defect-incidence: per-vehicle initial crack size a0_v ~ LogNormal
    log_a0_mean: float = np.log(0.10)
    log_a0_sd: float = 0.30
    # surrogate Paris/Carrara growth law  a_{f+1}=a_f + r_v (Δσ)^m a_f^p
    paris_m: float = 3.0                # stress-amplitude exponent
    paris_p: float = 0.5               # crack-size exponent (sub-linear)
    load_amp: float = 1.0               # nominal flight load amplitude Δσ
    load_amp_sd: float = 0.05           # flight-to-flight load scatter
    a_fail: float = 1.0                 # crack size at which the next flight is unsafe
    cert_horizon: int = 12              # future flights the clearance must cover
    early_life_flight: int = 2          # anchor crack state at this early flight
    # inspection (Stage-0→2 posterior stand-in): noisy Gaussian read of a
    meas_rel_noise: float = 0.08        # measurement SD = rel_noise · a  (+floor)
    meas_abs_floor: float = 0.004       # absolute SD floor
    # hyper-priors for the Normal–Normal–InvGamma hierarchy (weak)
    m0: float = -3.0                    # prior mean of μ_φ
    v0: float = 9.0                     # prior var  of μ_φ (very weak: SD 3)
    a0: float = 2.0                     # InvGamma shape for σ_φ²
    b0: float = 0.5                     # InvGamma scale for σ_φ²
    seed: int = 0

    def measurement_sd(self, a: np.ndarray | float) -> np.ndarray | float:
        return np.maximum(self.meas_rel_noise * np.asarray(a),
                          self.meas_abs_floor)


# ─────────────────────────────────────────────────────────────────────────────
#  Fleet simulator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VehicleHistory:
    """One vehicle's latent truth + observed inspection stream."""
    r_true: float                 # true growth-rate constant
    s_true: float                 # log r_true
    a0: float                     # true initial crack size
    a_true: np.ndarray            # (n_flights+1,) true crack size per inspection
    a_meas: np.ndarray            # (n_flights+1,) noisy measured crack size
    meas_sd: np.ndarray           # (n_flights+1,) measurement SD
    loads: np.ndarray             # (n_flights,) per-flight load amplitude


def _grow_one_flight(a: float, r_v: float, load: float,
                     cfg: FleetConfig) -> float:
    """Surrogate Paris/Carrara growth increment for a single flight.

    a_{f+1} = a_f + r_v · load^m · a_f^p   (clipped non-negative, monotone).
    Mechanistic basis: Carrara-2020 fatigue (cfrp_phasefield_2d) — see module
    docstring.  Closed form so a fleet × MCMC runs in seconds.
    """
    return a + r_v * (load ** cfg.paris_m) * (max(a, 0.0) ** cfg.paris_p)


def simulate_vehicle(r_true: float, a0: float, cfg: FleetConfig,
                     rng: np.random.Generator) -> VehicleHistory:
    """Fly one vehicle for cfg.n_flights flights, inspecting after each."""
    F = cfg.n_flights
    s_true = float(np.log(r_true))
    loads = cfg.load_amp + cfg.load_amp_sd * rng.standard_normal(F)
    loads = np.clip(loads, 1e-6, None)

    a_true = np.empty(F + 1)
    a_true[0] = a0
    for f in range(F):
        a_true[f + 1] = _grow_one_flight(a_true[f], r_true, loads[f], cfg)

    meas_sd = np.asarray(cfg.measurement_sd(a_true), dtype=float)
    a_meas = a_true + meas_sd * rng.standard_normal(F + 1)
    a_meas = np.clip(a_meas, 1e-6, None)
    return VehicleHistory(r_true=r_true, s_true=s_true, a0=a0,
                          a_true=a_true, a_meas=a_meas, meas_sd=meas_sd,
                          loads=loads)


def simulate_fleet(cfg: FleetConfig | None = None,
                   rng: np.random.Generator | None = None
                   ) -> list[VehicleHistory]:
    """Draw N vehicles from φ_true and fly each for F flights.

    r_v = exp(s_v),  s_v ~ Normal(mu_phi_true, sigma_phi_true²);
    a0_v ~ LogNormal(log_a0_mean, log_a0_sd) (defect-incidence level).
    """
    cfg = cfg or FleetConfig()
    rng = rng or np.random.default_rng(cfg.seed)
    s = cfg.mu_phi_true + cfg.sigma_phi_true * rng.standard_normal(cfg.n_vehicles)
    a0 = np.exp(cfg.log_a0_mean + cfg.log_a0_sd * rng.standard_normal(cfg.n_vehicles))
    return [simulate_vehicle(float(np.exp(s[v])), float(a0[v]), cfg, rng)
            for v in range(cfg.n_vehicles)]


# ─────────────────────────────────────────────────────────────────────────────
#  Per-vehicle measurement posterior on s_v = log r_v  (Stage-0→2 stand-in)
# ─────────────────────────────────────────────────────────────────────────────

def vehicle_growth_estimate(veh: VehicleHistory, cfg: FleetConfig,
                            n_flights: int | None = None
                            ) -> tuple[float, float]:
    """Gaussian "measurement posterior" (ŝ_v, τ_v) on s_v from inspections.

    Each consecutive inspection pair (a_f, a_{f+1}) gives a noisy estimate of
    r_v through the surrogate law, hence of s_v = log r_v.  Linearising the
    measurement noise, inspection f contributes

        r̂_f  = (a_meas_{f+1} − a_meas_f) / (load_f^m · a_meas_f^p)
        ŝ_f  = log r̂_f ,    var(ŝ_f) ≈ var(r̂_f) / r̂_f²

    and we pool the per-pair estimates by inverse-variance weighting → a
    single Gaussian (ŝ_v, τ_v²).  This is the Gaussian stand-in for the real
    Stage-0→2 (Mahalanobis→GNN→FMPE) posterior on the vehicle's growth state:
    more inspections (larger n_flights) → smaller τ_v.

    Estimator.  Single-flight differences (a_{f+1}−a_f ~ 1e-2) are below the
    measurement noise, so pairwise rate estimates are pure noise.  Instead we
    fit s_v = log r_v by minimising the weighted misfit between the measured
    crack trajectory and the deterministic forward integration a(f; r_v) of
    the surrogate law over the first F inspections — cumulative growth, which
    sits well above the noise floor.  The misfit χ²(s) is smooth and convex
    in this regime; we minimise it on a coarse-then-fine grid (no SciPy
    needed) and read τ_v from its curvature (Laplace approximation), giving
    the Gaussian (ŝ_v, τ_v) "measurement posterior".

    n_flights : use only the first n_flights inspections (temporal ordering
                for the fleet-leader experiment).  None → all available.
    """
    F = cfg.n_flights if n_flights is None else min(n_flights, cfg.n_flights)
    if F < 1:
        # no observations: improper / flat — return a huge variance
        return cfg.mu_phi_true, 1e6

    a_obs = veh.a_meas[:F + 1]
    sd = veh.meas_sd[:F + 1]
    loads = veh.loads[:F]
    a0_obs = a_obs[0]                       # anchor the trajectory at insp. 0

    def chi2(s: float) -> float:
        r = np.exp(s)
        a = a0_obs
        resid = 0.0
        for f in range(F):
            a = a + r * (loads[f] ** cfg.paris_m) * (max(a, 0.0) ** cfg.paris_p)
            resid += ((a - a_obs[f + 1]) / sd[f + 1]) ** 2
        return resid

    # coarse grid around the prior, then refine — robust, SciPy-free
    grid = np.linspace(cfg.mu_phi_true - 5.0, cfg.mu_phi_true + 3.0, 81)
    vals = np.array([chi2(s) for s in grid])
    s0 = grid[int(np.argmin(vals))]
    fine = np.linspace(s0 - 0.2, s0 + 0.2, 81)
    fvals = np.array([chi2(s) for s in fine])
    s_hat = float(fine[int(np.argmin(fvals))])

    # τ from local curvature of ½χ²  (Laplace):  1/τ² = ½ d²χ²/ds²
    h = 0.05
    c0 = chi2(s_hat)
    cp = chi2(s_hat + h)
    cm = chi2(s_hat - h)
    d2 = (cp - 2.0 * c0 + cm) / h ** 2          # d²χ²/ds²
    prec = 0.5 * max(d2, 1e-6)                   # 1/τ²
    tau = float(np.sqrt(1.0 / prec))
    tau = float(np.clip(tau, 1e-3, 1e3))
    return s_hat, tau


# ─────────────────────────────────────────────────────────────────────────────
#  Hierarchical Bayesian inference (Normal–Normal–InvGamma Gibbs sampler)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FleetPosterior:
    """Posterior samples of the global prior φ and per-vehicle s_v."""
    mu_phi: np.ndarray            # (n_samples,) posterior draws of μ_φ
    sigma_phi: np.ndarray         # (n_samples,) posterior draws of σ_φ
    s_v: np.ndarray               # (n_samples, n_vehicles) draws of s_v
    s_hat: np.ndarray             # (n_vehicles,) per-vehicle measurement mean
    tau: np.ndarray               # (n_vehicles,) per-vehicle measurement SD

    @property
    def mu_phi_mean(self) -> float:
        return float(self.mu_phi.mean())

    @property
    def sigma_phi_mean(self) -> float:
        return float(self.sigma_phi.mean())

    @property
    def s_v_mean(self) -> np.ndarray:
        return self.s_v.mean(axis=0)

    def r_v_mean(self) -> np.ndarray:
        """Posterior-mean per-vehicle growth rate (E[exp s_v] via samples)."""
        return np.exp(self.s_v).mean(axis=0)


def fit_hierarchy(s_hat: np.ndarray, tau: np.ndarray, cfg: FleetConfig,
                  n_samples: int = 2000, burn: int = 500,
                  rng: np.random.Generator | None = None) -> FleetPosterior:
    """Gibbs sampler for the Normal–Normal–InvGamma fleet hierarchy.

        s_v       ~ Normal(μ_φ, σ_φ²)
        ŝ_v | s_v ~ Normal(s_v, τ_v²)          (measurement posterior)
        μ_φ       ~ Normal(m0, v0)
        σ_φ²      ~ InvGamma(a0, b0)

    Conjugate full conditionals (all closed form):
        s_v | · ~ Normal( (ŝ_v/τ_v² + μ_φ/σ_φ²) / (1/τ_v² + 1/σ_φ²),
                          1/(1/τ_v² + 1/σ_φ²) )                  [shrinkage]
        μ_φ | · ~ Normal( (m0/v0 + Σ s_v/σ_φ²)/(1/v0 + V/σ_φ²),
                          1/(1/v0 + V/σ_φ²) )
        σ_φ²| · ~ InvGamma( a0 + V/2, b0 + ½ Σ (s_v − μ_φ)² )

    Returns FleetPosterior with the post-burn-in draws.
    """
    rng = rng or np.random.default_rng(cfg.seed)
    s_hat = np.asarray(s_hat, dtype=float)
    tau = np.asarray(tau, dtype=float)
    V = len(s_hat)
    tau2 = tau ** 2

    # init
    mu = float(np.mean(s_hat))
    sig2 = float(max(np.var(s_hat), 1e-3))
    s_v = s_hat.copy()

    keep = max(n_samples - burn, 1)
    mu_keep = np.empty(keep)
    sig_keep = np.empty(keep)
    sv_keep = np.empty((keep, V))

    for it in range(n_samples):
        # s_v | ·  (partial-pooling / shrinkage)
        prec = 1.0 / tau2 + 1.0 / sig2
        mean = (s_hat / tau2 + mu / sig2) / prec
        s_v = mean + rng.standard_normal(V) / np.sqrt(prec)

        # μ_φ | ·
        prec_mu = 1.0 / cfg.v0 + V / sig2
        mean_mu = (cfg.m0 / cfg.v0 + s_v.sum() / sig2) / prec_mu
        mu = float(mean_mu + rng.standard_normal() / np.sqrt(prec_mu))

        # σ_φ² | ·   (Inverse-Gamma)
        a_post = cfg.a0 + 0.5 * V
        b_post = cfg.b0 + 0.5 * float(np.sum((s_v - mu) ** 2))
        sig2 = float(b_post / rng.gamma(a_post, 1.0))   # InvGamma via 1/Gamma

        if it >= burn:
            k = it - burn
            mu_keep[k] = mu
            sig_keep[k] = np.sqrt(sig2)
            sv_keep[k] = s_v

    return FleetPosterior(mu_phi=mu_keep, sigma_phi=sig_keep, s_v=sv_keep,
                          s_hat=s_hat, tau=tau)


def fit_fleet(vehicles: list[VehicleHistory], cfg: FleetConfig,
              n_flights_per_vehicle: int | None = None,
              n_samples: int = 2000, burn: int = 500,
              rng: np.random.Generator | None = None) -> FleetPosterior:
    """Convenience: per-vehicle measurement posteriors → hierarchy fit."""
    ests = [vehicle_growth_estimate(v, cfg, n_flights_per_vehicle)
            for v in vehicles]
    s_hat = np.array([e[0] for e in ests])
    tau = np.array([e[1] for e in ests])
    return fit_hierarchy(s_hat, tau, cfg, n_samples=n_samples, burn=burn,
                         rng=rng)


# ─────────────────────────────────────────────────────────────────────────────
#  Clearance decision for a (new) vehicle — reuses the Stage-3 cost thresholds
# ─────────────────────────────────────────────────────────────────────────────

def _clearance_thresholds() -> tuple[float, float]:
    """Reuse Stage-3 expected-cost α/β (cfrp_phasefield_2d).

    Imported lazily so this CPU module has no hard dependency at import time;
    falls back to the documented defaults (α=0.02, β=0.48) if unavailable.
    """
    try:
        from cfrp_phasefield_2d import calibrate_thresholds
        return calibrate_thresholds()
    except Exception:
        return 0.02, 0.48


def predict_failure_prob(s_samples: np.ndarray, a_current: float,
                         load: float, cfg: FleetConfig,
                         horizon: int = 1) -> float:
    """P(crack exceeds a_fail within `horizon` future flights) under the
    per-vehicle growth-rate posterior.

    Push the per-vehicle s = log r posterior samples through `horizon`
    surrogate growth steps from the current crack size; fraction reaching
    a_fail by the end.  horizon=1 → next-flight risk; horizon>1 → the
    forward-looking clearance used in the fleet-leader study (survive to the
    end of the certification block).  The SPREAD of the s posterior — which
    the fleet prior sharpens — is what drives the decision's stability.
    """
    r = np.exp(np.clip(np.asarray(s_samples), -50.0, 10.0))
    a = np.full_like(r, float(a_current))
    failed = a >= cfg.a_fail
    for _ in range(max(horizon, 1)):
        a = a + r * (load ** cfg.paris_m) * np.maximum(a, 0.0) ** cfg.paris_p
        failed = failed | (a >= cfg.a_fail)
    return float(np.mean(failed))


def clearance_decision(p_fail: float,
                       alpha: float | None = None,
                       beta: float | None = None) -> str:
    """OK / REPAIR / RETIRE from the Stage-3 expected-cost thresholds."""
    if alpha is None or beta is None:
        a_cal, b_cal = _clearance_thresholds()
        alpha = a_cal if alpha is None else alpha
        beta = b_cal if beta is None else beta
    if p_fail <= alpha:
        return "OK"
    if p_fail <= beta:
        return "REPAIR"
    return "RETIRE"


def remaining_life_estimate(s_samples: np.ndarray, a_current: float,
                            cfg: FleetConfig) -> tuple[float, float]:
    """Posterior remaining-life (flights to a_fail): mean and SD over samples.

    Demonstrates the tightening of the per-vehicle prognosis as φ sharpens:
    a tighter s posterior → tighter remaining-life distribution.
    """
    r = np.exp(np.clip(np.asarray(s_samples), -50.0, 10.0))
    lives = np.empty(len(r))
    for i, ri in enumerate(r):
        a = a_current
        n = 0
        while a < cfg.a_fail and n < 10 * cfg.n_flights:
            a = a + ri * (cfg.load_amp ** cfg.paris_m) * (max(a, 0.0) ** cfg.paris_p)
            n += 1
        lives[i] = n
    return float(lives.mean()), float(lives.std())


# ─────────────────────────────────────────────────────────────────────────────
#  Fleet-leader experiment: inspections-to-confidence, with vs without prior
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FleetLeaderResult:
    fleet_sizes: np.ndarray             # (n_points,) cumulative #established vehicles
    insp_with_prior: np.ndarray         # (n_points,) inspections needed WITH fleet prior
    insp_without_prior: np.ndarray      # (n_points,) inspections needed WITHOUT (flat)
    mu_phi_mean: np.ndarray             # (n_points,) running φ-mean estimate
    sigma_phi_mean: np.ndarray          # (n_points,) running φ-spread estimate
    sigma_phi_post_sd: np.ndarray       # (n_points,) posterior SD of σ_φ (sharpening)
    tol: float
    p_threshold: float


def _decision_stable(decisions: list[str], p_seq: np.ndarray,
                     tol: float, window: int = 3) -> int | None:
    """First inspection index after which the clearance decision AND the
    failure-probability estimate have stabilised: the decision is unchanged
    over the trailing `window` inspections and |Δp| < tol there.

    Returns the 1-based number of inspections needed, or None if never.
    """
    n = len(decisions)
    for k in range(window, n + 1):
        dwin = decisions[k - window:k]
        pwin = p_seq[k - window:k]
        if len(set(dwin)) == 1 and (pwin.max() - pwin.min()) < tol:
            return k - window + 1
    return None


def _vehicle_s_posterior(new_veh: VehicleHistory, k: int,
                         fleet_prior: tuple[float, float] | None,
                         cfg: FleetConfig) -> tuple[float, float]:
    """Per-vehicle s=log r posterior after k inspections.

    With a fleet prior → Normal–Normal partial pooling (mean, post_sd);
    without → the raw measurement posterior (isolation, data only).
    """
    s_hat, tau = vehicle_growth_estimate(new_veh, cfg, n_flights=k)
    if fleet_prior is not None:
        mu0, sig0 = fleet_prior
        prec = 1.0 / tau ** 2 + 1.0 / sig0 ** 2
        mean = (s_hat / tau ** 2 + mu0 / sig0 ** 2) / prec
        return float(mean), float(1.0 / np.sqrt(prec))
    return float(s_hat), float(tau)


def _inspections_to_confidence(new_veh: VehicleHistory,
                               fleet_prior: tuple[float, float] | None,
                               cfg: FleetConfig, tol: float,
                               load: float,
                               rng: np.random.Generator,
                               max_insp: int | None = None,
                               n_post: int = 2000) -> int:
    """How many early-life inspections a (new/held-out) vehicle needs before
    its certification decision stabilises.

    The certification question is forward-looking: "will this vehicle survive
    the next `cfg.cert_horizon` flights?"  We anchor the crack at its
    early-life state (cfg.early_life_flight) and push the per-vehicle growth-
    rate posterior over that horizon (`predict_failure_prob`), giving an
    OK/REPAIR/RETIRE decision (Stage-3 expected-cost thresholds).  The
    decision is driven by the SPREAD of the s=log r posterior:

      * fleet_prior given → Normal–Normal partial pooling: the fleet prior
        (σ_φ ≈ 0.45, sharpening with fleet size) makes the s posterior tight
        even after ONE inspection → the decision is stable immediately;
      * fleet_prior None → ISOLATION: only the vehicle's own inspections
        inform s; one inspection gives a near-flat posterior (τ huge), so the
        failure probability swings wildly and the decision needs SEVERAL
        inspections to settle.

    Returns the first inspection count at which the decision is stable
    (unchanged + |Δp|<tol over a trailing window).
    """
    max_insp = max_insp or cfg.n_flights
    alpha, beta = _clearance_thresholds()
    a_anchor = new_veh.a_meas[cfg.early_life_flight]   # early-life crack state

    decisions: list[str] = []
    p_seq = []
    for k in range(1, max_insp + 1):
        mean, post_sd = _vehicle_s_posterior(new_veh, k, fleet_prior, cfg)
        s_samples = mean + post_sd * rng.standard_normal(n_post)
        p = predict_failure_prob(s_samples, a_anchor, load, cfg,
                                 horizon=cfg.cert_horizon)
        decisions.append(clearance_decision(p, alpha, beta))
        p_seq.append(p)

    stable = _decision_stable(decisions, np.asarray(p_seq), tol)
    return stable if stable is not None else max_insp


def fleet_leader_experiment(cfg: FleetConfig | None = None,
                            tol: float = 0.05,
                            load: float | None = None,
                            n_samples: int = 1500, burn: int = 500,
                            rng: np.random.Generator | None = None
                            ) -> FleetLeaderResult:
    """The headline fleet-leader study.

    Simulate the fleet, hold out the LAST vehicle as the "new" vehicle, and
    process the remaining vehicles in temporal order.  After each established
    vehicle is added, refit the fleet prior φ and measure how many
    inspections the new vehicle needs to reach a stable clearance decision —
    WITH the running fleet prior vs WITHOUT it (isolation, flat prior).

    The without-prior curve is FLAT (the new vehicle's own data is all it
    has, independent of fleet size); the with-prior curve DROPS as the fleet
    grows and φ sharpens.  Their gap is the fleet-leader reduction.
    """
    cfg = cfg or FleetConfig()
    rng = rng or np.random.default_rng(cfg.seed)
    load = cfg.load_amp if load is None else load

    fleet = simulate_fleet(cfg, rng)
    # hold out the last n_holdout vehicles as "new" vehicles; average the
    # inspections-to-confidence over them for a robust (less noisy) curve.
    n_holdout = max(3, cfg.n_vehicles // 5)
    holdout = fleet[-n_holdout:]
    established = fleet[:-n_holdout]

    # isolation baseline (no fleet prior) — independent of fleet size
    insp_iso = float(np.mean([
        _inspections_to_confidence(v, None, cfg, tol, load,
                                   np.random.default_rng(cfg.seed + 7000 + j))
        for j, v in enumerate(holdout)]))

    sizes, with_p, mu_run, sig_run, sig_sd = [], [], [], [], []
    for n in range(1, len(established) + 1):
        post = fit_fleet(established[:n], cfg, n_samples=n_samples, burn=burn,
                         rng=np.random.default_rng(cfg.seed + n))
        prior = (post.mu_phi_mean, post.sigma_phi_mean)
        k = float(np.mean([
            _inspections_to_confidence(
                v, prior, cfg, tol, load,
                np.random.default_rng(cfg.seed + 1000 + 31 * n + j))
            for j, v in enumerate(holdout)]))
        sizes.append(n)
        with_p.append(k)
        mu_run.append(post.mu_phi_mean)
        sig_run.append(post.sigma_phi_mean)
        sig_sd.append(float(post.sigma_phi.std()))

    return FleetLeaderResult(
        fleet_sizes=np.array(sizes),
        insp_with_prior=np.array(with_p),
        insp_without_prior=np.full(len(sizes), insp_iso),
        mu_phi_mean=np.array(mu_run),
        sigma_phi_mean=np.array(sig_run),
        sigma_phi_post_sd=np.array(sig_sd),
        tol=tol, p_threshold=cfg.a_fail,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  φ-recovery / consistency study (φ posterior sharpens with fleet size)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RecoveryResult:
    fleet_sizes: np.ndarray
    mu_phi_mean: np.ndarray
    mu_phi_sd: np.ndarray         # posterior SD of μ_φ (shrinks → consistency)
    sigma_phi_mean: np.ndarray
    sigma_phi_sd: np.ndarray
    mu_phi_true: float
    sigma_phi_true: float


def recovery_study(cfg: FleetConfig | None = None,
                   sizes: list[int] | None = None,
                   n_samples: int = 2000, burn: int = 500,
                   rng: np.random.Generator | None = None) -> RecoveryResult:
    """Fit the hierarchy on growing fleet sizes; track the φ posterior.

    As N grows, E[μ_φ|data] → mu_phi_true, E[σ_φ|data] → sigma_phi_true, and
    the posterior SD of both shrinks (the fleet prior sharpens).
    """
    cfg = cfg or FleetConfig()
    rng = rng or np.random.default_rng(cfg.seed)
    big = FleetConfig(**{**cfg.__dict__})
    big.n_vehicles = max(cfg.n_vehicles, sizes[-1] if sizes else cfg.n_vehicles)
    fleet = simulate_fleet(big, np.random.default_rng(cfg.seed))
    sizes = sizes or [2, 4, 6, 8, 12, 16, 20]
    sizes = [s for s in sizes if s <= len(fleet)]

    mu_m, mu_s, sg_m, sg_s = [], [], [], []
    for n in sizes:
        post = fit_fleet(fleet[:n], cfg, n_samples=n_samples, burn=burn,
                         rng=np.random.default_rng(cfg.seed + n))
        mu_m.append(post.mu_phi_mean); mu_s.append(float(post.mu_phi.std()))
        sg_m.append(post.sigma_phi_mean); sg_s.append(float(post.sigma_phi.std()))
    return RecoveryResult(
        fleet_sizes=np.array(sizes),
        mu_phi_mean=np.array(mu_m), mu_phi_sd=np.array(mu_s),
        sigma_phi_mean=np.array(sg_m), sigma_phi_sd=np.array(sg_s),
        mu_phi_true=cfg.mu_phi_true, sigma_phi_true=cfg.sigma_phi_true,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    cfg = FleetConfig()
    print("=== Stage-4 hierarchical Bayesian fleet learning (Concept A) ===")
    print(f"fleet: N={cfg.n_vehicles} vehicles × F={cfg.n_flights} flights; "
          f"φ_true=(μ={cfg.mu_phi_true}, σ={cfg.sigma_phi_true})")

    t0 = time.perf_counter()
    rec = recovery_study(cfg)
    print("\n-- φ recovery vs fleet size --")
    print(f"{'N':>4} {'E[μ_φ]':>9} {'sd(μ_φ)':>9} {'E[σ_φ]':>9} {'sd(σ_φ)':>9}")
    for i, n in enumerate(rec.fleet_sizes):
        print(f"{n:>4} {rec.mu_phi_mean[i]:>9.3f} {rec.mu_phi_sd[i]:>9.3f} "
              f"{rec.sigma_phi_mean[i]:>9.3f} {rec.sigma_phi_sd[i]:>9.3f}")
    print(f"   (truth μ_φ={rec.mu_phi_true}, σ_φ={rec.sigma_phi_true})")

    fl = fleet_leader_experiment(cfg)
    print("\n-- fleet-leader: inspections-to-confidence --")
    print(f"isolated (no fleet prior): {fl.insp_without_prior[0]} inspections")
    print(f"{'fleet_N':>8} {'insp(prior)':>12} {'E[μ_φ]':>9} {'E[σ_φ]':>9}")
    for i, n in enumerate(fl.fleet_sizes):
        print(f"{n:>8} {fl.insp_with_prior[i]:>12} "
              f"{fl.mu_phi_mean[i]:>9.3f} {fl.sigma_phi_mean[i]:>9.3f}")
    k_iso = float(fl.insp_without_prior[0])
    k_fleet = float(fl.insp_with_prior[-1])
    print(f"\nfleet-leader reduction: isolated needs {k_iso:.2f} inspections, "
          f"full fleet prior needs {k_fleet:.2f} → saved {k_iso - k_fleet:.2f}")
    print(f"[total {time.perf_counter() - t0:.1f}s]")
