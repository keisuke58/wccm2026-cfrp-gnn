"""Tests for cfrp_phasefield_2d (Stage-3 prognosis layer).

Pure numpy/scipy — no torch/CUDA.  Runs the real FD phase-field solver on
the default coarse grid (nx=60, ny=48); the whole file takes ~1 min on CPU.

Load calibration (default LaminateConfig, transverse-strain proxy):
  load <= 0.06  → no growth for any FMPE-range defect
  load ~ 0.10   → transition (size-dependent)
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
    damage_area,
    flight_clearance,
    growth_probability,
    seed_defect,
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
                            "expected_remaining_flights"}


def test_flight_clearance_rejects_bad_posterior_shape():
    with pytest.raises(ValueError):
        growth_probability(np.zeros((5, 3)), 0.1)
