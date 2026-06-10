"""Tests for cfrp_phasefield_2d (Stage-3 prognosis layer).

Pure numpy/scipy — no torch/CUDA.  Runs the real FD phase-field solver on
the default coarse grid (nx=60, ny=48); the whole file takes ~2-3 min on CPU.

Load calibration (default LaminateConfig, transverse-strain proxy):
  load <= 0.06  → no quasi-static growth; fatigue life > 25 flights @ K=100
  load ~ 0.08   → no quasi-static growth; FINITE fatigue life (~12-19 flights)
  load ~ 0.10   → quasi-static transition (size-dependent)
  load >= 0.14  → severe growth / laminate-spanning delamination
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cfrp_phasefield_2d import (
    LaminateConfig,
    build_laminate_maps,
    build_var_coeff_divM,
    calibrate_thresholds,
    damage_area,
    fatigue_degradation,
    flight_clearance,
    growth_probability,
    seed_defect,
    simulate_fatigue_flights,
    simulate_growth,
)

LOAD_LOW = 0.06
LOAD_MID = 0.10
LOAD_HIGH = 0.14


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def growth_angle_deg(result, d0, cfg):
    """Principal-axis angle [deg] of the newly damaged (d>0.5) cell cloud."""
    new = (result.d_final > 0.5) & ~(d0 > 0.5)
    ys, xs = np.nonzero(new)
    assert len(xs) >= 6, "not enough growth to measure an orientation"
    x = xs * cfg.hx
    y = ys * cfg.hy
    x = x - x.mean()
    y = y - y.mean()
    sxx, syy, sxy = (x * x).mean(), (y * y).mean(), (x * y).mean()
    return 0.5 * np.degrees(np.arctan2(2.0 * sxy, sxx - syy))


def ang_dist(a, b):
    """Distance between crack orientations (mod 180°)."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


# --------------------------------------------------------------------------- #
# seeding
# --------------------------------------------------------------------------- #
def test_seed_defect_basic():
    cfg = LaminateConfig()
    d0 = seed_defect(0.5, 0.5, 9.0, 1.0, cfg)
    assert d0.shape == (cfg.ny, cfg.nx)
    assert d0.max() == 1.0 and d0.min() == 0.0
    assert (d0 > 0.5).sum() > 0
    # centroid near requested position
    ys, xs = np.nonzero(d0 > 0.5)
    assert abs(xs.mean() * cfg.hx - 0.5 * cfg.Lx) < 3 * cfg.hx
    # bigger log2size → bigger seed
    big = seed_defect(0.5, 0.5, 9.0, 2.0, cfg)
    assert (big > 0.5).sum() > (d0 > 0.5).sum()
    # out-of-range posterior draws are clipped, not fatal
    d_clip = seed_defect(-0.3, 1.7, 99.0, 10.0, cfg)
    assert np.isfinite(d_clip).all() and (d_clip > 0.5).sum() > 0


# --------------------------------------------------------------------------- #
# 1. isotropic limit sane
# --------------------------------------------------------------------------- #
def test_isotropic_limit_sane():
    cfg = LaminateConfig(ply_angles_deg=(0.0,), beta_ply=0.0)  # 1 ply, isotropic
    d0 = seed_defect(0.5, 0.5, 9.0, 1.0, cfg)

    # constant-coefficient operator: row sums vanish (conservation / Neumann)
    _, Mxx, Myy, Mxy, imask = build_laminate_maps(cfg)
    assert not imask.any(), "single ply must have no interface band"
    L = build_var_coeff_divM(Mxx, Myy, Mxy, cfg.hx, cfg.hy)
    assert np.abs(np.asarray(L.sum(axis=1))).max() < 1e-10

    # low load: bounded, irreversible, no growth
    r = simulate_growth(d0, LOAD_LOW, cfg)
    assert np.all(r.d_final >= -1e-12) and np.all(r.d_final <= 1.0 + 1e-12)
    assert np.all(r.d_final >= d0 - 1e-12), "irreversibility violated"
    assert r.area_final >= r.area0 - 1e-12
    assert not r.grown

    # high load: the crack does grow (the model is not inert)
    r_hi = simulate_growth(d0, LOAD_MID, cfg)
    assert r_hi.grown
    assert damage_area(r_hi.d_final, cfg) > damage_area(r.d_final, cfg)


# --------------------------------------------------------------------------- #
# 2. crack deflects toward the fiber direction
# --------------------------------------------------------------------------- #
def test_crack_deflects_toward_fiber_direction():
    theta_fiber = 30.0
    load = 0.12  # enough to drive growth in the beta-toughened ply

    cfg_iso = LaminateConfig(ply_angles_deg=(theta_fiber,), beta_ply=0.0)
    cfg_ani = LaminateConfig(ply_angles_deg=(theta_fiber,), beta_ply=25.0)

    d0 = seed_defect(0.5, 0.5, 9.0, 1.0, cfg_iso)
    ang_iso = growth_angle_deg(simulate_growth(d0, load, cfg_iso), d0, cfg_iso)
    ang_ani = growth_angle_deg(simulate_growth(d0, load, cfg_ani), d0, cfg_ani)

    # isotropic: crack normal to the transverse load → runs along x (~0°)
    assert ang_dist(ang_iso, 0.0) < 8.0
    # anisotropic: structural tensor steers the crack toward the fibers
    assert ang_dist(ang_ani, theta_fiber) < ang_dist(ang_iso, theta_fiber) - 10.0
    assert ang_dist(ang_ani, theta_fiber) < 15.0


# --------------------------------------------------------------------------- #
# 3. low-Gc interface band captures the crack (delamination channel)
# --------------------------------------------------------------------------- #
def test_interface_band_captures_crack():
    layer = 9.8   # mid-ply seed close to (but not on) an interface
    load = 0.11

    def run(ratio):
        cfg = LaminateConfig(Gc_interface_ratio=ratio)
        d0 = seed_defect(0.5, 0.5, layer, 1.0, cfg)
        _, _, _, _, imask = build_laminate_maps(cfg)
        r = simulate_growth(d0, load, cfg)
        new = (r.d_final > 0.5) & ~(d0 > 0.5)
        return int((new & imask).sum()), int(new.sum()), imask

    n_iface_low, n_tot_low, imask = run(0.3)    # weak interface (DCB-like)
    n_iface_ctl, n_tot_ctl, _ = run(1.0)        # control: no Gc contrast

    # the weak interface promotes growth (delamination is competitive) ...
    assert n_tot_low > n_tot_ctl
    assert n_iface_low > n_iface_ctl
    # ... and the growth concentrates in the bands beyond their area share
    band_share = imask.mean()
    assert n_iface_low / max(n_tot_low, 1) > band_share


# --------------------------------------------------------------------------- #
# 4. growth_probability monotone in load
# --------------------------------------------------------------------------- #
def test_growth_probability_monotone_in_load():
    # 3 posterior draws spanning the FMPE size classes (1x1 / 2x2 / 4x4)
    posterior = np.array([
        [0.5, 0.5, 9.0, 0.0],
        [0.5, 0.5, 9.0, 1.0],
        [0.5, 0.5, 9.0, 2.0],
    ])
    cfg = LaminateConfig()
    rng = np.random.default_rng(0)
    probs = [growth_probability(posterior, load, cfg, n_draws=3, rng=rng)
             for load in (LOAD_LOW, LOAD_MID, LOAD_HIGH)]
    assert probs[0] <= probs[1] <= probs[2]
    assert probs[0] == 0.0, "no FMPE-range defect should grow at the low load"
    assert probs[2] == 1.0, "every defect should grow at the severe load"
    assert 0.0 < probs[1] < 1.0, "transition load should split the posterior"


# --------------------------------------------------------------------------- #
# 5. flight_clearance decisions
# --------------------------------------------------------------------------- #
def test_flight_clearance_ok_low_load_retire_high_load():
    posterior = np.array([
        [0.5, 0.5, 9.0, 0.0],
        [0.5, 0.5, 9.0, 1.0],
        [0.5, 0.5, 9.0, 2.0],
    ])
    cfg = LaminateConfig()

    out_lo = flight_clearance(posterior, load_profile=[0.5 * LOAD_LOW, LOAD_LOW],
                              n_flights=10, cfg=cfg, n_draws=3,
                              rng=np.random.default_rng(0))
    assert out_lo["decision"] == "OK"
    assert out_lo["p_growth_next"] == 0.0
    assert out_lo["p_survive_n"] == 1.0
    assert out_lo["expected_remaining_flights"] > 1.0
    assert np.isfinite(out_lo["expected_remaining_flights"])

    out_hi = flight_clearance(posterior, load_profile=[LOAD_HIGH],
                              n_flights=10, cfg=cfg, n_draws=3,
                              rng=np.random.default_rng(0))
    assert out_hi["decision"] == "RETIRE"
    assert out_hi["p_growth_next"] > 0.3
    assert out_hi["p_survive_n"] < out_lo["p_survive_n"]
    assert (out_hi["expected_remaining_flights"]
            < out_lo["expected_remaining_flights"])

    for out in (out_lo, out_hi):
        assert set(out) == {"decision", "p_growth_next", "p_survive_n",
                            "p_survive_curve", "expected_remaining_flights",
                            "alpha", "beta", "fatigue", "cycles_per_flight"}


def test_flight_clearance_rejects_bad_posterior_shape():
    with pytest.raises(ValueError):
        growth_probability(np.zeros((5, 3)), 0.1)
    with pytest.raises(ValueError):
        flight_clearance(np.zeros((5, 3)), 0.1)


# --------------------------------------------------------------------------- #
# 6. fatigue (Carrara et al. 2020): sub-critical loads eventually grow
# --------------------------------------------------------------------------- #
LOAD_SUBCRIT = 0.08   # statically safe, finite fatigue life (~16 flights @K=100)


def test_fatigue_degradation_function():
    aT = 2.0
    a = np.array([0.0, 1.0, 2.0, 6.0, 1e6])
    f = fatigue_degradation(a, aT)
    # below/at threshold: no degradation; above: Carrara Eq. 37, monotone → 0
    assert np.allclose(f[:3], 1.0)
    assert np.isclose(f[3], (2 * aT / (6.0 + aT)) ** 2)   # = 0.25
    assert f[4] < 1e-9
    assert np.all(np.diff(f) <= 1e-12)


def test_fatigue_subcritical_accumulation_grows_crack():
    """Wöhler-like behaviour: a load with NO quasi-static growth propagates
    the crack after enough identical cycles, and ᾱ accumulates monotonically
    across flights (true cycle-block accumulation, not i.i.d. per flight)."""
    cfg = LaminateConfig()
    d0 = seed_defect(0.5, 0.5, 9.0, 1.0, cfg)

    # statically sub-critical (rate-independent baseline would never grow)
    assert not simulate_growth(d0, LOAD_SUBCRIT, cfg).grown

    # flight-by-flight continuation: ᾱ monotone, damage irreversible
    res1 = simulate_fatigue_flights(d0, LOAD_SUBCRIT, n_flights=1, cfg=cfg)
    a1 = res1.state.alpha_bar.copy()
    d1 = res1.state.d.copy()
    assert np.all(a1 >= 0.0) and a1.max() > 0.0
    res2 = simulate_fatigue_flights(d0, LOAD_SUBCRIT, n_flights=1, cfg=cfg,
                                    state=res1.state)
    assert np.all(res2.state.alpha_bar >= a1 - 1e-12)
    assert res2.state.alpha_bar.max() > a1.max()
    assert np.all(res2.state.d >= d1 - 1e-12)
    assert res2.state.cycles == pytest.approx(2 * cfg.cycles_per_flight)

    # repeated identical sub-critical flights eventually grow the crack
    res = simulate_fatigue_flights(d0, LOAD_SUBCRIT, n_flights=25, cfg=cfg)
    assert res.grown
    assert res.fail_flight is not None and res.fail_flight > 1
    rel = res.rel_growth_per_flight[:res.fail_flight]
    assert np.all(np.diff(rel) >= -1e-12), "crack area must grow monotonically"


def test_fatigue_woehler_monotone_in_load():
    """Higher cyclic load amplitude → shorter fatigue life (S-N trend),
    and a statically critical load still fails in flight 1."""
    cfg = LaminateConfig()
    d0 = seed_defect(0.5, 0.5, 9.0, 1.0, cfg)
    fails = {}
    for load in (LOAD_SUBCRIT, 0.09, LOAD_HIGH):
        r = simulate_fatigue_flights(d0, load, n_flights=25, cfg=cfg)
        assert r.grown, f"load {load} should have finite fatigue life"
        fails[load] = r.fail_flight
    assert fails[LOAD_HIGH] == 1, "statically critical load must fail flight 1"
    assert fails[0.09] < fails[LOAD_SUBCRIT]
    assert fails[LOAD_SUBCRIT] > 5, "sub-critical life should be many flights"


def test_fatigue_survival_curve_decays_at_constant_load():
    """flight_clearance with fatigue: P(survive n) decays even at CONSTANT
    load, unlike the rate-independent legacy mode (i.i.d. growth events)."""
    posterior = np.array([
        [0.5, 0.5, 9.0, 0.0],
        [0.5, 0.5, 9.0, 1.0],
        [0.5, 0.5, 9.0, 2.0],
    ])
    cfg = LaminateConfig()
    out_f = flight_clearance(posterior, load_profile=[LOAD_SUBCRIT],
                             n_flights=20, fatigue=True, cfg=cfg, n_draws=3,
                             rng=np.random.default_rng(0))
    out_l = flight_clearance(posterior, load_profile=[LOAD_SUBCRIT],
                             n_flights=20, fatigue=False, cfg=cfg, n_draws=3,
                             rng=np.random.default_rng(0))

    curve = out_f["p_survive_curve"]
    assert len(curve) == 20
    assert np.all(np.diff(curve) <= 1e-12), "survival curve must be monotone"
    assert curve[0] == 1.0, "no draw fails on the first flight at 0.08"
    assert curve[-1] < 1.0, "fatigue must produce failures within the horizon"
    assert out_f["p_survive_n"] < out_l["p_survive_n"] == 1.0
    assert out_l["p_growth_next"] == 0.0, \
        "legacy rate-independent mode sees a sub-critical load as zero risk"
    # like-for-like: expected flights survived within the horizon (Σ S(n))
    assert (out_f["expected_remaining_flights"]
            < float(np.sum(out_l["p_survive_curve"])))
    assert out_f["fatigue"] is True and out_l["fatigue"] is False


# --------------------------------------------------------------------------- #
# 7. expected-cost decision thresholds
# --------------------------------------------------------------------------- #
def test_calibrate_thresholds_expected_cost():
    # defaults: vehicle-loss:repair = 100:1, retire = 25, residual risk 0.5
    alpha, beta = calibrate_thresholds()
    assert alpha == pytest.approx(0.02)
    assert beta == pytest.approx(0.48)
    assert 0.0 < alpha < beta <= 1.0

    # more expensive vehicle loss → both thresholds tighten
    a2, b2 = calibrate_thresholds(c_loss=1000.0)
    assert a2 < alpha and b2 < beta
    # cheaper repair → flying as-is less attractive (alpha smaller)
    a3, _ = calibrate_thresholds(c_repair=0.1)
    assert a3 < alpha

    with pytest.raises(ValueError):
        calibrate_thresholds(repair_residual_risk=0.0)
    with pytest.raises(ValueError):
        calibrate_thresholds(c_loss=-1.0)

    # flight_clearance uses calibrated thresholds by default
    posterior = np.array([[0.5, 0.5, 9.0, 1.0]])
    out = flight_clearance(posterior, load_profile=[LOAD_LOW], n_flights=1,
                           fatigue=False, cfg=LaminateConfig(), n_draws=1,
                           rng=np.random.default_rng(0))
    assert out["alpha"] == pytest.approx(0.02)
    assert out["beta"] == pytest.approx(0.48)
