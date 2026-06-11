"""Tests for fleet_learning (Stage-4 hierarchical Bayesian fleet learning).

Pure numpy/scipy — no torch/CUDA, no phase-field FD solve (the fleet uses the
cheap surrogate growth law).  The whole file runs in a few seconds on CPU.

Covered claims (Concept A spec):
  1. consistency  — the φ posterior recovers φ_true as the fleet grows;
  2. fleet-leader — inspections-to-confidence WITH the fleet prior ≤ WITHOUT;
  3. shrinkage    — low-data vehicles are pulled toward the fleet mean.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fleet_learning import (
    FleetConfig,
    fit_fleet,
    fit_hierarchy,
    fleet_leader_experiment,
    predict_failure_prob,
    recovery_study,
    remaining_life_estimate,
    simulate_fleet,
    vehicle_growth_estimate,
)


# --------------------------------------------------------------------------- #
# simulator + estimator sanity
# --------------------------------------------------------------------------- #
def test_simulate_fleet_shapes_and_monotone_growth():
    cfg = FleetConfig(n_vehicles=8, n_flights=20)
    fleet = simulate_fleet(cfg, np.random.default_rng(0))
    assert len(fleet) == 8
    for v in fleet:
        assert v.a_true.shape == (cfg.n_flights + 1,)
        assert v.a_meas.shape == (cfg.n_flights + 1,)
        # the surrogate growth law is monotone non-decreasing in the truth
        assert np.all(np.diff(v.a_true) >= -1e-12)
        assert v.r_true > 0.0
        assert np.isclose(v.s_true, np.log(v.r_true))


def test_vehicle_growth_estimate_recovers_s_true_and_tightens():
    """The trajectory-fit estimator recovers s_true, and its uncertainty τ
    shrinks as more inspections accumulate (more cumulative growth signal)."""
    cfg = FleetConfig()
    fleet = simulate_fleet(cfg, np.random.default_rng(1))
    errs, taus_few, taus_many = [], [], []
    for v in fleet:
        s_hat, tau = vehicle_growth_estimate(v, cfg)          # all flights
        errs.append(abs(s_hat - v.s_true))
        taus_few.append(vehicle_growth_estimate(v, cfg, n_flights=3)[1])
        taus_many.append(vehicle_growth_estimate(v, cfg, n_flights=cfg.n_flights)[1])
    assert np.mean(errs) < 0.15, "full-trajectory fit should recover s_true"
    # more inspections → tighter measurement posterior (per vehicle)
    assert np.mean(taus_many) < np.mean(taus_few)


# --------------------------------------------------------------------------- #
# 1. consistency: φ posterior recovers φ_true as the fleet grows
# --------------------------------------------------------------------------- #
def test_phi_recovery_consistency():
    cfg = FleetConfig(n_vehicles=20)
    rec = recovery_study(cfg, sizes=[3, 20], n_samples=1500, burn=500)

    mu_err_small = abs(rec.mu_phi_mean[0] - cfg.mu_phi_true)
    mu_err_large = abs(rec.mu_phi_mean[-1] - cfg.mu_phi_true)

    # the large fleet recovers μ_φ accurately ...
    assert mu_err_large < 0.20, f"μ_φ not recovered: {rec.mu_phi_mean[-1]}"
    # ... and the φ posterior SHARPENS: posterior SD of μ_φ shrinks with N
    assert rec.mu_phi_sd[-1] < rec.mu_phi_sd[0]
    assert rec.sigma_phi_sd[-1] < rec.sigma_phi_sd[0]
    # σ_φ also recovered to the right ballpark at the full fleet
    assert abs(rec.sigma_phi_mean[-1] - cfg.sigma_phi_true) < 0.20


def test_phi_posterior_sd_decreases_monotonically_on_average():
    cfg = FleetConfig(n_vehicles=20)
    rec = recovery_study(cfg, sizes=[2, 5, 10, 20], n_samples=1500, burn=500)
    # not strictly monotone every step (MC), but the trend is clearly down
    assert rec.mu_phi_sd[0] > rec.mu_phi_sd[-1]
    # at least the endpoints bracket the truth direction
    assert np.all(rec.mu_phi_sd > 0)


# --------------------------------------------------------------------------- #
# 2. fleet-leader: inspections WITH fleet prior ≤ WITHOUT (monotone benefit)
# --------------------------------------------------------------------------- #
def test_fleet_leader_benefit():
    cfg = FleetConfig(n_vehicles=20)
    fl = fleet_leader_experiment(cfg, tol=0.05, n_samples=1200, burn=400)

    iso = fl.insp_without_prior[0]
    full = fl.insp_with_prior[-1]

    # with the fleet prior, a new vehicle never needs MORE inspections ...
    assert np.all(fl.insp_with_prior <= fl.insp_without_prior + 1e-9)
    # ... and the full-fleet prior strictly cuts the inspection count
    assert full < iso, f"no fleet-leader reduction: {full} vs {iso}"
    # the benefit grows: the large-fleet prior beats the small-fleet prior
    assert fl.insp_with_prior[-1] <= fl.insp_with_prior[0] + 1e-9
    # the fleet prior sharpens as flights accumulate (σ_φ posterior SD down)
    assert fl.sigma_phi_post_sd[-1] < fl.sigma_phi_post_sd[0]


def test_fleet_leader_curve_trends_down():
    """The with-prior inspections-to-confidence curve trends downward as the
    fleet grows (correlation with fleet size is negative)."""
    cfg = FleetConfig(n_vehicles=20)
    fl = fleet_leader_experiment(cfg, tol=0.05, n_samples=1200, burn=400)
    n = fl.fleet_sizes.astype(float)
    y = fl.insp_with_prior.astype(float)
    # Pearson correlation between fleet size and inspections needed < 0
    c = np.corrcoef(n, y)[0, 1]
    assert c < -0.3, f"fleet-leader curve not decreasing (corr={c:.2f})"


# --------------------------------------------------------------------------- #
# 3. hierarchical shrinkage: low-data vehicles pulled toward the fleet mean
# --------------------------------------------------------------------------- #
def test_hierarchical_shrinkage_toward_fleet_mean():
    """A vehicle with a NOISY (high-τ) measurement posterior is pulled toward
    the fleet mean; a vehicle with a precise (low-τ) one is barely moved."""
    cfg = FleetConfig()
    fleet_mean = cfg.mu_phi_true                     # ≈ where μ_φ will sit

    # 9 well-measured vehicles near the fleet mean + 1 noisy outlier
    s_true = np.full(10, cfg.mu_phi_true)
    s_hat = s_true.copy()
    s_hat[-1] = cfg.mu_phi_true + 1.2                # outlier raw estimate
    tau = np.full(10, 0.05)
    tau[-1] = 0.8                                    # outlier is poorly measured

    post = fit_hierarchy(s_hat, tau, cfg, n_samples=3000, burn=1000,
                         rng=np.random.default_rng(0))
    s_post = post.s_v_mean

    raw_gap = abs(s_hat[-1] - fleet_mean)
    shrunk_gap = abs(s_post[-1] - fleet_mean)
    # the noisy outlier's posterior is pulled a long way toward the fleet mean
    assert shrunk_gap < 0.5 * raw_gap, "no shrinkage of the low-data vehicle"

    # a precisely-measured vehicle is hardly shrunk (its data dominates)
    well_idx = 0
    moved_well = abs(s_post[well_idx] - s_hat[well_idx])
    moved_noisy = abs(s_post[-1] - s_hat[-1])
    assert moved_noisy > moved_well, \
        "low-data vehicle should move MORE than a well-measured one"


def test_shrinkage_scales_with_fleet_concentration():
    """Tighter fleet prior (more established vehicles) → stronger shrinkage of
    a fixed noisy vehicle."""
    cfg = FleetConfig()

    def shrunk_gap(n_good):
        s_hat = np.concatenate([np.full(n_good, cfg.mu_phi_true),
                                [cfg.mu_phi_true + 1.2]])
        tau = np.concatenate([np.full(n_good, 0.05), [0.8]])
        post = fit_hierarchy(s_hat, tau, cfg, n_samples=2500, burn=800,
                             rng=np.random.default_rng(1))
        return abs(post.s_v_mean[-1] - cfg.mu_phi_true)

    gap_small = shrunk_gap(3)
    gap_large = shrunk_gap(18)
    assert gap_large < gap_small + 1e-9


# --------------------------------------------------------------------------- #
# clearance / remaining-life integration (Stage-3 thresholds reused)
# --------------------------------------------------------------------------- #
def test_predict_failure_prob_monotone_in_horizon_and_load():
    cfg = FleetConfig()
    s = cfg.mu_phi_true + 0.05 * np.random.default_rng(0).standard_normal(2000)
    p1 = predict_failure_prob(s, 0.3, cfg.load_amp, cfg, horizon=1)
    p12 = predict_failure_prob(s, 0.3, cfg.load_amp, cfg, horizon=12)
    assert 0.0 <= p1 <= p12 <= 1.0          # longer horizon → more failure
    p_lo = predict_failure_prob(s, 0.3, 0.8, cfg, horizon=12)
    p_hi = predict_failure_prob(s, 0.3, 1.3, cfg, horizon=12)
    assert p_hi >= p_lo                      # higher load → more failure


def test_remaining_life_tightens_with_fleet_prior():
    """The per-vehicle remaining-life estimate tightens (smaller SD) when the
    fleet prior sharpens the growth-rate posterior."""
    cfg = FleetConfig()
    fleet = simulate_fleet(cfg, np.random.default_rng(2))
    new_veh = fleet[-1]
    established = fleet[:-1]

    # ONE inspection only: isolation posterior is near-flat
    s_hat, tau = vehicle_growth_estimate(new_veh, cfg, n_flights=1)
    rng = np.random.default_rng(3)
    s_iso = s_hat + tau * rng.standard_normal(2000)

    # same one inspection, but combined with the full fleet prior
    post = fit_fleet(established, cfg, n_samples=1200, burn=400,
                     rng=np.random.default_rng(4))
    mu0, sig0 = post.mu_phi_mean, post.sigma_phi_mean
    prec = 1.0 / tau ** 2 + 1.0 / sig0 ** 2
    mean = (s_hat / tau ** 2 + mu0 / sig0 ** 2) / prec
    s_fleet = mean + (1.0 / np.sqrt(prec)) * rng.standard_normal(2000)

    a_anchor = new_veh.a_meas[cfg.early_life_flight]
    _, sd_iso = remaining_life_estimate(s_iso, a_anchor, cfg)
    _, sd_fleet = remaining_life_estimate(s_fleet, a_anchor, cfg)
    assert sd_fleet < sd_iso, "fleet prior should tighten remaining-life"


def test_reproducible_seeded():
    cfg = FleetConfig(n_vehicles=6, n_flights=15)
    a = simulate_fleet(cfg, np.random.default_rng(123))
    b = simulate_fleet(cfg, np.random.default_rng(123))
    assert np.allclose(a[0].a_meas, b[0].a_meas)
    p1 = fit_hierarchy(np.array([-3.0, -2.9, -3.1]), np.array([0.1, 0.1, 0.1]),
                       cfg, n_samples=800, burn=200, rng=np.random.default_rng(5))
    p2 = fit_hierarchy(np.array([-3.0, -2.9, -3.1]), np.array([0.1, 0.1, 0.1]),
                       cfg, n_samples=800, burn=200, rng=np.random.default_rng(5))
    assert np.allclose(p1.mu_phi, p2.mu_phi)
