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

# Areal-mass proxy coefficients (Section 1b): fraction of areal mass ADDED by
# the fully-toughened / fully-reinforced corner over the bare structural plies
# (interleaf + stitching mass).  ~10 % total — the right order for interleaf
# toughening.  Used only by laminate_mass_proxy / the multi-objective layer.
M_GCINT = 0.06
M_REINF = 0.04

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
#  1b.  Cheap analytical STRUCTURAL proxies  (competing objectives for Stage-5)
# ─────────────────────────────────────────────────────────────────────────────
#
# WHY these exist.  The single-objective Stage-5 (mean log-growth + CVaR +
# manufacturability) has NO competing structural cost: making the laminate
# heavier (more interface toughening / reinforcement) and turning the off-axis
# plies steeper only ever HELPS deflect the matrix crack, so the optimum rails
# to the manufacturable cap (off-axis 74–75°, max reinforcement).  That is an
# honest sign the design model is incomplete: a real laminate trades
# crack-resistance against MASS and STIFFNESS.  Below are cheap, closed-form
# (NO new FEM) analytical proxies for those competing objectives, derived from
# classical laminate theory (CLT) on the (0/±θ)_s layup, so an INTERIOR optimum
# can exist once they are weighted in.
#
# All proxies are NORMALISED to the baseline design (BASELINE_DESIGN) so they
# read as dimensionless ratios ≈1 at baseline; only RATIOS/TRENDS are claimed
# (cf. honest accounting (c)), never absolute MPa / kg.  They are deterministic
# and microsecond-cheap, so the Pareto sweep is essentially free.

# Reduced-stiffness anchors for a unidirectional CFRP ply (normalised, typical
# carbon/epoxy ratios — memo §5 notes the FD growth model itself ignores E1/E2,
# so these constants live ONLY in the CLT proxies, not in the FD forward):
_PLY_E1   = 130.0    # axial (fiber) modulus  [normalised]
_PLY_E2   = 9.0      # transverse modulus
_PLY_G12  = 5.0      # in-plane shear modulus
_PLY_NU12 = 0.30     # major Poisson ratio


def _ply_Qbar(theta_deg: float):
    """Transformed reduced-stiffness matrix Q̄(θ) for one UD ply (CLT).

    Standard plane-stress CLT (e.g. Jones, *Mechanics of Composite Materials*):
    build the on-axis reduced stiffness Q from (E1,E2,G12,ν12), then rotate by
    the ply angle θ to get Q̄.  Returns (Qb11, Qb22, Qb66) — the entries the
    membrane (axial) and we need for the laminate A- and D-matrix proxies.
    """
    nu21 = _PLY_NU12 * _PLY_E2 / _PLY_E1
    denom = 1.0 - _PLY_NU12 * nu21
    Q11 = _PLY_E1 / denom
    Q22 = _PLY_E2 / denom
    Q12 = _PLY_NU12 * _PLY_E2 / denom
    Q66 = _PLY_G12
    c = np.cos(np.radians(theta_deg)); s = np.sin(np.radians(theta_deg))
    c2, s2 = c * c, s * s
    c4, s4 = c2 * c2, s2 * s2
    cs2 = c2 * s2
    Qb11 = Q11 * c4 + 2.0 * (Q12 + 2.0 * Q66) * cs2 + Q22 * s4
    Qb22 = Q11 * s4 + 2.0 * (Q12 + 2.0 * Q66) * cs2 + Q22 * c4
    Qb66 = (Q11 + Q22 - 2.0 * Q12 - 2.0 * Q66) * cs2 + Q66 * (c4 + s4)
    return Qb11, Qb22, Qb66


def axial_stiffness_proxy(design, normalised: bool = True) -> float:
    """In-plane AXIAL modulus proxy E_x of the (0/±θ)_s laminate (CLT).

    Assumptions / construction:
      * Classical laminate theory, all plies equal thickness; the laminate
        membrane stiffness in the loading (x / 0°) direction is the
        thickness-average of Q̄11(θ_k) over the stack.
      * We report the A11-density A11/h = mean_k Q̄11(θ_k) as the axial-modulus
        proxy (a faithful monotone surrogate for E_x = (A11·A22−A12²)/(h·A22);
        the simpler A11/h keeps the trend and is cheaper).
      * The 0° plies are fixed; the ±θ plies LOSE axial stiffness as θ steepens
        (Q̄11(θ) decreases monotonically from 0°→90°).  So steeper off-axis →
        LOWER axial stiffness — the competing cost that pulls θ off the cap.
    Interface toughening / reinforcement do not enter (they are interleaf
    resin, ~0 axial load path).  normalised → ratio to BASELINE_DESIGN.
    """
    design = np.asarray(design, float).ravel()
    offaxis = float(np.clip(design[0], OFFAXIS_LO, OFFAXIS_HI))
    angles = balanced_symmetric_layup(offaxis)
    A11 = float(np.mean([_ply_Qbar(a)[0] for a in angles]))
    if not normalised:
        return A11
    A11_base = axial_stiffness_proxy(BASELINE_DESIGN, normalised=False)
    return A11 / A11_base


def bending_stiffness_proxy(design, normalised: bool = True) -> float:
    """BUCKLING / bending-stiffness proxy D11 of the (0/±θ)_s laminate (CLT).

    Assumptions / construction:
      * Classical laminate theory D-matrix: D11 = Σ_k Q̄11(θ_k)·(z_k³−z_{k-1}³)/3
        with z the ply through-thickness coordinates (mid-plane = 0, unit total
        thickness).  D11 governs panel bending / buckling resistance, so this is
        a (panel-)buckling proxy.
      * Because z³ weights the OUTER plies most, and the (0/±θ)_s stack here
        keeps 0° plies on the faces, D11 stays high; but the ±θ plies that sit
        off the faces still erode D11 as θ steepens — a second, weaker
        competing structural cost (its trend with θ is monotone-decreasing).
    Interface toughening / reinforcement do not enter.  normalised → ratio to
    BASELINE_DESIGN.
    """
    design = np.asarray(design, float).ravel()
    offaxis = float(np.clip(design[0], OFFAXIS_LO, OFFAXIS_HI))
    angles = balanced_symmetric_layup(offaxis)
    n = len(angles)
    # equal-thickness plies, total thickness 1, mid-plane at 0 → z from -1/2..1/2
    z = np.linspace(-0.5, 0.5, n + 1)
    D11 = 0.0
    for k, a in enumerate(angles):
        D11 += _ply_Qbar(a)[0] * (z[k + 1] ** 3 - z[k] ** 3) / 3.0
    if not normalised:
        return float(D11)
    D11_base = bending_stiffness_proxy(BASELINE_DESIGN, normalised=False)
    return float(D11 / D11_base)


def laminate_mass_proxy(design, normalised: bool = True) -> float:
    """Areal-density / MASS proxy for the design (heavier toughening = more mass).

    Assumptions / construction:
      * The base laminate (the 8 structural plies) has a fixed areal mass m0=1.
      * Interface toughening (interleaf veils / tougher resin, Gc_int/Gc_ply)
        and local reinforcement (reinf_mult) ADD areal mass: interleaving and
        stitching deposit extra resin/fibre.  We charge their mass LINEARLY in
        the fraction of each band used, with coefficients M_GCINT / M_REINF
        chosen so a fully-toughened, fully-reinforced laminate is ~10 % heavier
        — the right order for interleaf toughening (a few % per interface).
      * Ply ANGLE does not change mass (same plies, re-oriented).
    So mass competes with growth-resistance ONLY through the toughening /
    reinforcement axes; the off-axis angle is traded by stiffness, not mass.
    normalised=True (default) returns the ratio to BASELINE_DESIGN (≈1).
    """
    design = np.asarray(design, float).ravel()
    gcint = float(np.clip(design[1], GCINT_LO, GCINT_HI)) if design.shape[0] >= 2 \
        else GCINT_LO
    reinf = float(np.clip(design[2], REINF_LO, REINF_HI)) if design.shape[0] >= 3 \
        else 1.0
    f_gc = (gcint - GCINT_LO) / (GCINT_HI - GCINT_LO)
    f_re = (reinf - REINF_LO) / (REINF_HI - REINF_LO)
    m = 1.0 + M_GCINT * f_gc + M_REINF * f_re
    if not normalised:
        return float(m)
    return float(m / laminate_mass_proxy(BASELINE_DESIGN, normalised=False))


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


# ── Stage-4 → Stage-5 handoff: REAL fleet posterior → fleet defect-θ samples ──
#
# WHY a separate sampler (Task #4).  `fleet_defect_prior` above only re-scales
# the SCALAR design load from the fleet — the θ-GEOMETRY envelope (cx, layer,
# size) stays a synthetic uniform box.  That box is the last synthetic stand-in
# in the Stage-4→5 chain.  Stage-4's per-vehicle latent is a scalar growth rate
# s_v = log r_v (NOT a θ-field), so we cannot read a full θ posterior off it.
# But we CAN, in a principled and DOCUMENTED way, turn the fleet's growth-rate
# DISTRIBUTION into a defect-SEVERITY distribution, because a faster-growing
# vehicle is — to first order — one carrying more severe (larger / deeper)
# defects.  We make that link explicit and MONOTONE below; everything else of
# the θ-vector that the scalar latent genuinely cannot constrain (the in-plane
# centroid cx, cy) stays the documented in-distribution envelope.

# Link calibration: how the per-vehicle growth-rate z-score maps onto the
# log2size defect-severity axis.  log2size ∈ [L2S_LO, L2S_HI] (DefectPrior
# band); the fleet MEDIAN growth rate maps to the band MIDPOINT and ±2σ of the
# fleet log-rate spans (most of) the band.  Strictly increasing in r_v.
LINK_L2S_LO, LINK_L2S_HI = 0.6, 2.0      # defect log2size band (matches DefectPrior)
LINK_LAYER_LO, LINK_LAYER_HI = 4.0, 14.0  # interior-ply band (synthetic, kept)


@dataclass
class FleetDefectSamples:
    """Stage-4-derived fleet defect-θ sample bank for Stage-5 optimisation.

    theta : (n, 4) draws (cx, cy, layer, log2size) — the per-defect θ Stage-5
            optimises against, with log2size (defect SEVERITY) derived from the
            REAL Stage-4 fleet growth-rate posterior via a monotone link (below).
    load  : scalar design peel strain (scaled from the fleet posterior).
    source: provenance tag ("fleet_posterior_theta").
    link  : human-readable description of the growth-rate → severity link.
    """
    theta: np.ndarray
    load: float
    source: str = "fleet_posterior_theta"
    link: str = ""

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Return n θ draws (with replacement) from the fleet-derived bank."""
        idx = rng.integers(0, len(self.theta), size=n)
        return self.theta[idx]


def fleet_growthrate_to_log2size(r_v: np.ndarray, r_med: float,
                                 r_spread: float) -> np.ndarray:
    """MONOTONE link: per-vehicle growth rate r_v → defect severity log2size.

    Documented link assumption (Task #4).  Stage-4 only gives a scalar growth
    rate per vehicle; we posit that a vehicle's growth rate is a monotone,
    saturating function of the severity of the defect it carries — bigger /
    deeper defects (larger log2size) drive faster effective growth.  We invert
    that qualitatively: map the LOG growth-rate z-score
        ζ_v = (log r_v − log r_med) / (√2 · r_spread)
    through the smooth, bounded, STRICTLY-INCREASING logistic-like squash
        f(ζ) = ½(1 + erf(ζ))          ∈ (0,1)
    onto the log2size band [L2S_LO, L2S_HI].  Properties: r_v = r_med → band
    midpoint; r_v ↑ → log2size ↑ (monotone); bounded to the calibrated band so
    the FD config never leaves the in-distribution box (honest accounting (b)).
    `r_spread` is the SD of log r_v over the fleet (the φ-spread σ_φ).
    """
    r_v = np.maximum(np.asarray(r_v, float), 1e-12)
    z = (np.log(r_v) - np.log(max(r_med, 1e-12))) / (np.sqrt(2.0) * max(r_spread, 1e-6))
    f = 0.5 * (1.0 + np.array([_erf(zi) for zi in np.atleast_1d(z)]))
    return LINK_L2S_LO + f * (LINK_L2S_HI - LINK_L2S_LO)


def fleet_defect_prior_from_posterior(fleet_posterior, fleet_cfg=None,
                                      n: int = 400,
                                      seed: int = 0) -> FleetDefectSamples:
    """Stage-4 → Stage-5 handoff: REAL fleet posterior → fleet defect-θ samples.

    Replaces the synthetic θ-box with a sample bank whose defect SEVERITY axis
    (log2size) is derived from the Stage-4 hierarchical posterior:

      1. Draw growth rates from the fleet: for each posterior draw of the global
         prior φ=(μ_φ, σ_φ) we draw a per-vehicle s = log r ~ Normal(μ_φ, σ_φ²)
         and set r = exp(s).  This propagates BOTH the fleet-level spread σ_φ
         AND the posterior uncertainty in φ into the defect population (the full
         posterior-predictive over a NEW vehicle's growth rate).
      2. Map each r through `fleet_growthrate_to_log2size` (monotone link, with
         r_med = posterior-median fleet rate and r_spread = E[σ_φ]).
      3. The centroid cx ~ U and interior layer ~ U remain the documented
         synthetic in-distribution envelope (the scalar latent cannot constrain
         them — honest accounting (b)); only the SEVERITY axis is now real.
      4. The scalar design load is scaled from the fleet exactly as
         `fleet_defect_prior` does (reuse).

    Returns a FleetDefectSamples bank.  `n` draws keeps the Pareto/BO sweep
    tractable.  This is the principled Stage-4→5 coupling: the fleet defect
    distribution Stage-5 optimises against now comes from the observed fleet,
    not a hand-set box.
    """
    rng = np.random.default_rng(seed)
    # posterior-predictive growth rates for a NEW vehicle: integrate over φ draws
    mu = np.asarray(fleet_posterior.mu_phi, float)
    sig = np.asarray(fleet_posterior.sigma_phi, float)
    m = min(len(mu), len(sig))
    idx = rng.integers(0, m, size=n)
    s_pred = mu[idx] + sig[idx] * rng.standard_normal(n)
    r_pred = np.exp(np.clip(s_pred, -50.0, 10.0))

    r_med = float(np.median(r_pred))
    r_spread = float(np.mean(sig)) if np.mean(sig) > 0 else float(np.std(s_pred) + 1e-6)
    l2s = fleet_growthrate_to_log2size(r_pred, r_med, r_spread)

    cx = rng.uniform(0.25, 0.75, n)
    cy = np.full(n, 0.5)
    layer = rng.uniform(LINK_LAYER_LO, LINK_LAYER_HI, n)
    theta = np.column_stack([cx, cy, layer, l2s])

    # reuse the documented load-scaling link from fleet_defect_prior
    load = fleet_defect_prior(fleet_posterior, fleet_cfg).load
    link = ("log2size = link(r_v): monotone erf-squash of the fleet log-growth-"
            "rate z-score onto [%.2f, %.2f]; r_med→midpoint, r_v↑→severity↑"
            % (LINK_L2S_LO, LINK_L2S_HI))
    return FleetDefectSamples(theta=theta, load=float(load),
                              source="fleet_posterior_theta", link=link)


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
#  4b.  MULTI-OBJECTIVE design  (growth ⟷ mass / stiffness trade-off; Pareto)
# ─────────────────────────────────────────────────────────────────────────────
#
# Task #6.  The single objective above rails the off-axis angle to the cap
# because nothing competes with crack-resistance.  Here we (a) expose a
# SCALARISED weighted objective that adds the structural costs (mass, lost
# stiffness) with tunable weights, and (b) compute a PARETO FRONT over the
# design space so the growth-vs-mass / growth-vs-stiffness trade-off — and the
# resulting INTERIOR optimum — is explicit.
#
# Growth term.  For the Pareto SWEEP (a dense grid) we use a CHEAP analytical
# growth PROXY (no FD) so the sweep is free; the proxy is monotone in the same
# direction as the FD growth (steeper off-axis & tougher interface → less
# growth) and is documented below.  The scalarised objective can ALSO be run
# with the exact FD growth (use_fd=True) for the final knee design — explicit
# per honest accounting.

# Growth-proxy weights (chosen so the proxy decreases with off-axis angle and
# interface toughening, matching the FD trend the single-objective study finds).
GP_W_OFFAXIS = 0.9    # growth ↓ as off-axis steepens (crack deflection)
GP_W_GCINT   = 0.6    # growth ↓ as interface toughens
GP_W_REINF   = 0.3    # growth ↓ with local reinforcement


def growth_proxy(design) -> float:
    """Cheap analytical surrogate for the design's expected log-growth (LOWER=better).

    NO FD.  A monotone, dimensionless stand-in for evaluate_design().mean_log_
    growth used ONLY for the dense Pareto sweep.  Construction (documented
    assumptions):
      * Steeper off-axis plies deflect the dominant matrix crack → growth ↓ with
        the off-axis fraction f_off ∈[0,1] of the band.
      * Tougher interface (gcint) and local reinforcement (reinf) raise the
        crack-arrest energy → growth ↓ with their band fractions f_gc, f_re.
      * Mapped through a smooth saturating exp so the proxy is positive and
        bounded, mirroring the near-binary-but-smoothed FD growth signal.
    Calibrated to be ~1 at baseline and to fall toward ~0.3 at the toughest /
    steepest corner — the SAME ORDERING the single-objective FD study shows.
    Trends only (honest accounting (c)); not a quantitative growth prediction.
    """
    design = np.asarray(design, float).ravel()
    offaxis = float(np.clip(design[0], OFFAXIS_LO, OFFAXIS_HI))
    gcint = float(np.clip(design[1], GCINT_LO, GCINT_HI)) if design.shape[0] >= 2 else GCINT_LO
    reinf = float(np.clip(design[2], REINF_LO, REINF_HI)) if design.shape[0] >= 3 else 1.0
    f_off = (offaxis - OFFAXIS_LO) / (OFFAXIS_HI - OFFAXIS_LO)
    f_gc = (gcint - GCINT_LO) / (GCINT_HI - GCINT_LO)
    f_re = (reinf - REINF_LO) / (REINF_HI - REINF_LO)
    z = GP_W_OFFAXIS * f_off + GP_W_GCINT * f_gc + GP_W_REINF * f_re
    return float(np.exp(-z))


def scalarised_objective(design, w_mass: float = 0.0, w_stiff: float = 0.0,
                         w_buckle: float = 0.0, growth=None,
                         use_fd: bool = False, defects=None, load=None,
                         nx: int = GRID_NX, ny: int = GRID_NY,
                         n_workers: int | None = None) -> dict:
    """Weighted multi-objective scalarisation (LOWER = better).

        J(design) = growth + w_mass·(mass−1) + w_stiff·(1/stiff − 1)
                            + w_buckle·(1/buckle − 1) + manufacturability_penalty

    where mass, stiff, buckle are the normalised proxies (≈1 at baseline).  The
    stiffness/buckling terms penalise LOST stiffness (1/ratio − 1 grows as
    stiffness drops), so steep off-axis plies — which shed axial/bending
    stiffness — are charged, giving an INTERIOR optimum off-axis angle once
    w_stiff > 0.  Weights are exposed so the trade-off can be swept.

    growth term: by default the cheap `growth_proxy`; if use_fd=True the EXACT
    FD mean-log-growth (needs `defects` and `load`).  Returns a dict with the
    scalar `J` and all components for inspection.
    """
    design = np.asarray(design, float).ravel()
    if use_fd:
        assert defects is not None and load is not None, \
            "use_fd requires defects and load"
        g = evaluate_design(design, defects, load, nx=nx, ny=ny,
                            n_workers=n_workers).mean_log_growth
    elif growth is not None:
        g = float(growth)
    else:
        g = growth_proxy(design)

    mass = laminate_mass_proxy(design)
    stiff = axial_stiffness_proxy(design)
    buckle = bending_stiffness_proxy(design)
    pen = manufacturability_penalty(design)
    J = (g + w_mass * (mass - 1.0) + w_stiff * (1.0 / stiff - 1.0)
         + w_buckle * (1.0 / buckle - 1.0) + pen)
    return {"J": float(J), "growth": float(g), "mass": float(mass),
            "stiffness": float(stiff), "buckling": float(buckle),
            "penalty": float(pen)}


def _pareto_mask(costs: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated rows (minimisation in every column).

    Point i is non-dominated if no other point is ≤ it in every objective and
    strictly < in at least one.
    """
    costs = np.atleast_2d(np.asarray(costs, float))
    n = costs.shape[0]
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        # flag every point STRICTLY DOMINATED by i (≥ in all, > in at least one)
        worse = np.all(costs >= costs[i], axis=1) & np.any(costs > costs[i], axis=1)
        mask[worse] = False
    return mask


@dataclass
class ParetoResult:
    designs: np.ndarray            # (n_grid, d) all swept designs
    growth: np.ndarray             # (n_grid,) growth objective
    mass: np.ndarray               # (n_grid,) mass proxy
    stiffness: np.ndarray          # (n_grid,) axial-stiffness proxy
    buckling: np.ndarray           # (n_grid,) bending-stiffness proxy
    pareto_mask: np.ndarray        # (n_grid,) bool, growth-vs-objective front
    objective_name: str            # "mass" or "stiffness"


def pareto_front(n_off: int = 13, n_gc: int = 5, n_re: int = 3,
                 objective: str = "mass") -> ParetoResult:
    """Pareto front growth-vs-{mass|stiffness} over a tractable design grid.

    Sweeps the 3-D design box on a small grid (default 13×5×3 = 195 designs,
    all CHEAP proxies → microseconds) and returns the non-dominated set for
    minimising BOTH the growth proxy AND the chosen structural cost:
      * objective="mass"      → minimise (growth, mass)
      * objective="stiffness" → minimise (growth, 1/stiffness)  [lost stiffness]
    The front is non-empty and non-dominated by construction.  Use the knee
    (max-curvature point) of the front as the balanced design.
    """
    offs = np.linspace(OFFAXIS_LO, OFFAXIS_HI, n_off)
    gcs = np.linspace(GCINT_LO, GCINT_HI, n_gc)
    res = np.linspace(REINF_LO, REINF_HI, n_re)
    designs = np.array([[o, g, r] for o in offs for g in gcs for r in res])

    growth = np.array([growth_proxy(d) for d in designs])
    mass = np.array([laminate_mass_proxy(d) for d in designs])
    stiff = np.array([axial_stiffness_proxy(d) for d in designs])
    buckle = np.array([bending_stiffness_proxy(d) for d in designs])

    if objective == "mass":
        costs = np.column_stack([growth, mass])
    elif objective == "stiffness":
        costs = np.column_stack([growth, 1.0 / stiff])
    else:
        raise ValueError("objective must be 'mass' or 'stiffness'")
    mask = _pareto_mask(costs)
    return ParetoResult(designs=designs, growth=growth, mass=mass,
                        stiffness=stiff, buckling=buckle, pareto_mask=mask,
                        objective_name=objective)


def pareto_knee(pr: ParetoResult) -> np.ndarray:
    """Knee design of a Pareto front: the front point closest to the utopia
    (per-objective minimum) corner after min-max normalising both objectives.

    A standard, weight-free knee selector — the balanced design where giving up
    a little growth-resistance no longer buys much mass/stiffness (or vice
    versa).  Returns the design vector.
    """
    g = pr.growth[pr.pareto_mask]
    second = (pr.mass if pr.objective_name == "mass"
              else 1.0 / pr.stiffness)[pr.pareto_mask]
    D = pr.designs[pr.pareto_mask]

    def _nrm(x):
        rng = x.max() - x.min()
        return (x - x.min()) / rng if rng > 1e-12 else np.zeros_like(x)
    gn, sn = _nrm(g), _nrm(second)
    dist = np.sqrt(gn ** 2 + sn ** 2)        # distance to utopia (0,0)
    return D[int(np.argmin(dist))]


def scalarised_optimum(w_stiff: float = 0.0, w_mass: float = 0.0,
                       w_buckle: float = 0.0, n_off: int = 25,
                       n_gc: int = 7, n_re: int = 4) -> np.ndarray:
    """Grid-minimise the scalarised (cheap-growth) objective; return best design.

    Dense off-axis grid so the interior optimum's off-axis angle is resolved.
    Used to SHOW that the optimum moves interior as w_stiff increases.
    """
    offs = np.linspace(OFFAXIS_LO, OFFAXIS_HI, n_off)
    gcs = np.linspace(GCINT_LO, GCINT_HI, n_gc)
    res = np.linspace(REINF_LO, REINF_HI, n_re)
    best, best_J = None, np.inf
    for o in offs:
        for g in gcs:
            for r in res:
                J = scalarised_objective([o, g, r], w_mass=w_mass,
                                         w_stiff=w_stiff, w_buckle=w_buckle)["J"]
                if J < best_J:
                    best_J, best = J, np.array([o, g, r])
    return best


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
#  6b.  Pareto / trade-off figure  (optional; robust Agg + thesis_style fallback)
# ─────────────────────────────────────────────────────────────────────────────

def make_pareto_figure(out_path: str | None = None,
                       w_stiff_sweep=(0.0, 0.3, 0.8, 1.5)) -> str:
    """Figure: growth-vs-mass & growth-vs-stiffness Pareto fronts + the interior
    move of the scalarised optimum's off-axis angle as the stiffness weight grows.

    Robust: matplotlib Agg backend, thesis_style import guarded by try/except,
    saves both .pdf and .png.
    """
    import matplotlib
    matplotlib.use("Agg")
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "slides", "figure_sources"))
    try:
        from thesis_style import use
        figsize = use(width_frac=1.0, aspect=0.34)
    except Exception:
        figsize = (12.0, 4.0)
    import matplotlib.pyplot as plt

    out_path = out_path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "paper_figs", "design_feedback_pareto.pdf")

    pr_m = pareto_front(objective="mass")
    pr_s = pareto_front(objective="stiffness")

    fig, ax = plt.subplots(1, 3, figsize=figsize)

    # (a) growth vs mass
    ax[0].scatter(pr_m.mass, pr_m.growth, s=10, c="0.7", label="all designs")
    om = np.argsort(pr_m.mass[pr_m.pareto_mask])
    ax[0].plot(pr_m.mass[pr_m.pareto_mask][om], pr_m.growth[pr_m.pareto_mask][om],
               "-o", ms=4, c="#b71c1c", label="Pareto front")
    knee_m = pareto_knee(pr_m)
    ax[0].scatter([laminate_mass_proxy(knee_m)], [growth_proxy(knee_m)],
                  marker="*", s=120, c="#1565C0", zorder=5, label="knee")
    ax[0].set_xlabel("mass proxy (norm.)", fontsize=7)
    ax[0].set_ylabel("growth proxy", fontsize=7)
    ax[0].set_title("(a) growth vs mass", fontsize=8)
    ax[0].legend(fontsize=6); ax[0].tick_params(labelsize=6)

    # (b) growth vs (lost) stiffness
    inv = 1.0 / pr_s.stiffness
    ax[1].scatter(inv, pr_s.growth, s=10, c="0.7", label="all designs")
    os_ = np.argsort(inv[pr_s.pareto_mask])
    ax[1].plot(inv[pr_s.pareto_mask][os_], pr_s.growth[pr_s.pareto_mask][os_],
               "-o", ms=4, c="#b71c1c", label="Pareto front")
    knee_s = pareto_knee(pr_s)
    ax[1].scatter([1.0 / axial_stiffness_proxy(knee_s)], [growth_proxy(knee_s)],
                  marker="*", s=120, c="#1565C0", zorder=5,
                  label=f"knee (off={knee_s[0]:.0f}°)")
    ax[1].set_xlabel("lost axial stiffness  1/$E_x$ (norm.)", fontsize=7)
    ax[1].set_ylabel("growth proxy", fontsize=7)
    ax[1].set_title("(b) growth vs stiffness", fontsize=8)
    ax[1].legend(fontsize=6); ax[1].tick_params(labelsize=6)

    # (c) optimum off-axis angle vs stiffness weight (interior move)
    ws = np.linspace(0.0, max(w_stiff_sweep) if w_stiff_sweep else 1.5, 16)
    offs = [scalarised_optimum(w_stiff=float(w))[0] for w in ws]
    ax[2].plot(ws, offs, "-o", ms=3, c="#2e7d32")
    ax[2].axhline(OFFAXIS_HI, color="0.6", lw=0.7, ls="--", label="manuf. cap")
    ax[2].set_xlabel("stiffness weight $w_{stiff}$", fontsize=7)
    ax[2].set_ylabel("optimum off-axis angle [°]", fontsize=7)
    ax[2].set_title("(c) optimum moves interior", fontsize=8)
    ax[2].legend(fontsize=6); ax[2].tick_params(labelsize=6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    fig.savefig(os.path.splitext(out_path)[0] + ".png", bbox_inches="tight",
                dpi=150)
    plt.close(fig)
    return out_path


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


# ═════════════════════════════════════════════════════════════════════════════
#  Unit tests
# ═════════════════════════════════════════════════════════════════════════════

def run_tests() -> int:
    n = 0

    def ok(cond, msg):
        nonlocal n
        assert cond, msg
        n += 1

    # ── design → config mapping ───────────────────────────────────────────────
    cfg = make_config([45.0, 0.5, 1.1], nx=20, ny=18)
    ok(abs(cfg.ply_angles_deg[1]) == 45.0, "off-axis angle threaded into layup")
    ok(cfg.ply_angles_deg[0] == 0.0, "0° plies kept on faces")
    ok(GCINT_LO <= cfg.Gc_interface_ratio <= GCINT_HI, "gcint_eff in band")
    lay = balanced_symmetric_layup(40.0)
    ok(abs(sum(lay)) < 1e-9 and lay == lay[::-1], "layup balanced & symmetric")

    # ── manufacturability penalty monotone ────────────────────────────────────
    ok(manufacturability_penalty([30, 0.30, 1.0]) <
       manufacturability_penalty([30, 0.90, 1.4]), "penalty ↑ with toughening")

    # ── NEW structural proxies (Task #6) ──────────────────────────────────────
    ok(abs(laminate_mass_proxy(BASELINE_DESIGN) - 1.0) < 1e-9,
       "mass proxy =1 at baseline")
    ok(abs(axial_stiffness_proxy(BASELINE_DESIGN) - 1.0) < 1e-9,
       "stiffness proxy =1 at baseline")
    ok(abs(bending_stiffness_proxy(BASELINE_DESIGN) - 1.0) < 1e-9,
       "buckling proxy =1 at baseline")
    # more reinforcement / toughening → more mass
    ok(laminate_mass_proxy([30, 0.90, 1.40]) > laminate_mass_proxy([30, 0.30, 1.0]),
       "more reinforcement → more mass")
    ok(laminate_mass_proxy([30, 0.30, 1.40]) > laminate_mass_proxy([30, 0.30, 1.0]),
       "more local reinforcement alone → more mass")
    # mass independent of off-axis angle
    ok(abs(laminate_mass_proxy([15, 0.5, 1.1]) -
           laminate_mass_proxy([75, 0.5, 1.1])) < 1e-9, "mass ⟂ off-axis angle")
    # steeper off-axis → lower axial & bending stiffness (monotone)
    sa = [axial_stiffness_proxy([o, 0.4, 1.0]) for o in (15, 35, 55, 75)]
    ok(all(np.diff(sa) < 0), "axial stiffness ↓ monotone as off-axis steepens")
    sb = [bending_stiffness_proxy([o, 0.4, 1.0]) for o in (15, 35, 55, 75)]
    ok(all(np.diff(sb) < 0), "bending/buckling stiffness ↓ as off-axis steepens")
    # toughening/reinforcement do not change stiffness proxies
    ok(abs(axial_stiffness_proxy([40, 0.3, 1.0]) -
           axial_stiffness_proxy([40, 0.9, 1.4])) < 1e-9, "stiffness ⟂ toughening")

    # ── growth proxy sanity ───────────────────────────────────────────────────
    ok(growth_proxy([75, 0.9, 1.4]) < growth_proxy([15, 0.3, 1.0]),
       "growth proxy ↓ at tough/steep corner")
    ok(growth_proxy([30, 0.9, 1.0]) < growth_proxy([30, 0.3, 1.0]),
       "growth proxy ↓ with interface toughening")

    # ── scalarised objective components ───────────────────────────────────────
    s0 = scalarised_objective(BASELINE_DESIGN)
    ok(set(["J", "growth", "mass", "stiffness", "buckling"]) <= set(s0),
       "scalarised objective returns components")
    # stiffness weight raises J of a steep design (it sheds stiffness)
    steep = [75, 0.9, 1.4]
    ok(scalarised_objective(steep, w_stiff=2.0)["J"] >
       scalarised_objective(steep, w_stiff=0.0)["J"],
       "stiffness weight penalises steep design")

    # ── Pareto front non-empty & non-dominated (Task #6) ──────────────────────
    pr = pareto_front(n_off=9, n_gc=4, n_re=3, objective="mass")
    ok(pr.pareto_mask.sum() >= 1, "mass Pareto front non-empty")
    front = np.column_stack([pr.growth, pr.mass])[pr.pareto_mask]
    nd = True
    for i in range(len(front)):
        for j in range(len(front)):
            if i != j and np.all(front[j] <= front[i]) and np.any(front[j] < front[i]):
                nd = False
    ok(nd, "mass Pareto front is non-dominated")
    prs = pareto_front(n_off=9, n_gc=4, n_re=3, objective="stiffness")
    ok(prs.pareto_mask.sum() >= 1, "stiffness Pareto front non-empty")
    knee = pareto_knee(prs)
    ok(OFFAXIS_LO <= knee[0] <= OFFAXIS_HI, "stiffness knee design in bounds")

    # ── scalarised optimum moves INTERIOR under stiffness weighting (Task #6) ──
    off_w0 = scalarised_optimum(w_stiff=0.0)[0]
    off_w1 = scalarised_optimum(w_stiff=1.5)[0]
    ok(off_w0 >= OFFAXIS_HI - 1e-6, "no stiffness weight → off-axis rails to cap")
    ok(off_w1 < off_w0 - 1.0, "stiffness weight → optimum off-axis moves interior")
    ok(off_w1 < OFFAXIS_HI - 1.0, "weighted optimum is interior (off cap)")

    # ── Stage-4 → Stage-5 fleet-derived θ distribution (Task #4) ──────────────
    import fleet_learning as fl
    fcfg = fl.FleetConfig(n_vehicles=8, n_flights=10)
    fleet = fl.simulate_fleet(fcfg, np.random.default_rng(0))
    post = fl.fit_fleet(fleet, fcfg, n_samples=400, burn=150,
                        rng=np.random.default_rng(1))
    fds = fleet_defect_prior_from_posterior(post, fcfg, n=200, seed=2)
    ok(fds.theta.shape == (200, 4), "fleet-derived θ has shape (n,4)")
    l2s = fds.theta[:, 3]
    ok(np.all(l2s >= LINK_L2S_LO - 1e-6) and np.all(l2s <= LINK_L2S_HI + 1e-6),
       "fleet-derived log2size within calibrated band")
    ok(l2s.std() > 0, "fleet-derived severity has spread")
    ok(0.10 <= fds.load <= 0.13, "fleet-derived design load in band")
    ok(fds.source == "fleet_posterior_theta" and len(fds.link) > 0,
       "fleet θ samples tagged + documented link")
    # monotone link: higher growth rate → larger log2size
    r = np.array([0.02, 0.05, 0.20])
    ll = fleet_growthrate_to_log2size(r, r_med=0.05, r_spread=0.45)
    ok(ll[0] < ll[1] < ll[2], "growth-rate→severity link strictly increasing")
    ok(abs(ll[1] - 0.5 * (LINK_L2S_LO + LINK_L2S_HI)) < 1e-6,
       "median growth rate → band midpoint")
    # samples are usable as a θ source for evaluate_design (shape contract)
    samp = fds.sample(5, np.random.default_rng(3))
    ok(samp.shape == (5, 4), "FleetDefectSamples.sample returns (n,4)")

    # ── CVaR + cheap FD objective smoke (serial, tiny grid) ───────────────────
    ok(cvar(np.array([1.0, 2.0, 3.0, 4.0]), q=0.25) == 4.0, "CVaR picks worst tail")
    ok(cvar(np.array([1.0, 1.0])) >= np.mean([1.0, 1.0]) - 1e-9, "CVaR ≥ mean")
    dprior = DefectPrior()
    dd = dprior.sample(2, np.random.default_rng(0))
    rr = evaluate_design(BASELINE_DESIGN, dd, dprior.load, nx=18, ny=16,
                         n_workers=1)
    ok(rr.mean_log_growth >= 0.0, "FD mean-log-growth non-negative")
    ok(rr.per_defect_log.shape == (2,), "per-defect log array shape")

    print(f"design_feedback: {n}/{n} unit tests passed")
    return n


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
    ap.add_argument("--test", action="store_true", help="unit tests only")
    ap.add_argument("--fig", action="store_true",
                    help="save paper_figs/design_feedback_pareto.pdf")
    args = ap.parse_args()
    if args.test:
        run_tests()
    elif args.fig:
        p = make_pareto_figure()
        print(f"[fig] wrote {p}")
    else:
        run(n_defects=args.n_defects, n_init=args.n_init, n_iter=args.n_iter,
            nx=args.nx, ny=args.ny, seed=args.seed, n_workers=args.workers)
