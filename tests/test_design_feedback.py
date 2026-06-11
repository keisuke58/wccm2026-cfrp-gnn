"""Tests for design_feedback (Stage-5 design-evolution layer).

FAST by construction: the pure-logic tests use a cheap MONOTONE stand-in for
simulate_growth (more growth when the interface is weaker / plies shallower),
so no FD solves run.  Two small tests touch the real FD forward at a tiny grid
with N=2 defects to keep the suite under a few seconds total.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import design_feedback as df                         # noqa: E402
from cfrp_phasefield_2d import LaminateConfig          # noqa: E402


# --------------------------------------------------------------------------- #
# 1. make_config: balanced + symmetric + valid for any in-bounds design
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("design", [
    [15.0, 0.30, 1.00],
    [45.0, 0.60, 1.20],
    [75.0, 0.90, 1.40],
    [30.0, 0.40, 1.00],                      # baseline
    [200.0, 5.0, 9.0],                       # out of bounds → clipped
])
def test_make_config_balanced_symmetric(design):
    cfg = df.make_config(design)
    assert isinstance(cfg, LaminateConfig)
    angles = np.array(cfg.ply_angles_deg)
    # symmetric: stack reads the same top-to-bottom
    assert np.allclose(angles, angles[::-1])
    # balanced: every +θ has a matching −θ (sum of off-axis angles = 0)
    assert abs(angles.sum()) < 1e-9
    # interface ratio stays inside the calibrated band
    assert df.GCINT_LO - 1e-9 <= cfg.Gc_interface_ratio <= df.GCINT_HI + 1e-9
    # 8-ply (0/±θ)_s structure
    assert cfg.n_plies == 8


def test_make_config_offaxis_steepens():
    """Off-axis design variable actually sets the ±θ ply magnitude."""
    cfg = df.make_config([55.0, 0.5, 1.0])
    assert pytest.approx(55.0) == max(cfg.ply_angles_deg)
    assert pytest.approx(-55.0) == min(cfg.ply_angles_deg)


# --------------------------------------------------------------------------- #
# 2. CVaR >= mean, and penalty monotone in toughening
# --------------------------------------------------------------------------- #
def test_cvar_ge_mean():
    rng = np.random.default_rng(0)
    for _ in range(20):
        v = rng.normal(size=12)
        assert df.cvar(v, 0.2) >= v.mean() - 1e-9


def test_penalty_monotone():
    base = df.manufacturability_penalty([30.0, 0.30, 1.00])
    more = df.manufacturability_penalty([30.0, 0.90, 1.40])
    assert more > base
    assert base == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# 3. evaluate_design with a cheap monotone mock forward (pure logic, no FD)
# --------------------------------------------------------------------------- #
def _patch_mock_forward(monkeypatch):
    """Replace the worker's FD call with a cheap monotone stand-in.

    Growth increases when the interface is WEAKER (low Gc_int) and plies are
    SHALLOWER (low |θ|) — the physically expected direction — and with bigger
    defects.  Deterministic, no FD solve.
    """
    def fake(args):
        design, theta, load, nx, ny = args
        offaxis, gcint = float(design[0]), float(design[1])
        reinf = float(design[2]) if len(design) >= 3 else 1.0
        gc_eff = np.clip(gcint * reinf, df.GCINT_LO, df.GCINT_HI)
        cx, cy, layer, l2s = theta
        # weaker interface & shallower plies & bigger defect → more growth
        rel = max(0.0, (load * 40.0)
                  * (1.2 - gc_eff) * (1.5 - offaxis / 90.0) * (0.5 + l2s) - 0.3)
        return {"grown": rel > 0.5, "rel_growth": rel,
                "log_growth": float(np.log1p(rel))}
    monkeypatch.setattr(df, "_sim_one_defect", fake)


def test_objective_finite_and_better_design_lower(monkeypatch):
    _patch_mock_forward(monkeypatch)
    rng = np.random.default_rng(1)
    defects = df.DefectPrior().sample(6, rng)
    load = 0.11
    pathological = [15.0, 0.30, 1.00]        # shallow plies, weak interface
    good = [70.0, 0.85, 1.30]                # steep plies, tough interface
    r_bad = df.evaluate_design(pathological, defects, load, n_workers=1)
    r_good = df.evaluate_design(good, defects, load, n_workers=1)
    assert np.isfinite(r_bad.objective) and np.isfinite(r_good.objective)
    assert r_good.objective < r_bad.objective
    # CVaR >= mean within a single evaluation
    assert r_bad.cvar_log_growth >= r_bad.mean_log_growth - 1e-9
    assert r_good.cvar_log_growth >= r_good.mean_log_growth - 1e-9


# --------------------------------------------------------------------------- #
# 4. BO loop runs, returns best-so-far, and beats the pathological corner
# --------------------------------------------------------------------------- #
def test_bo_runs_and_returns_best(monkeypatch):
    _patch_mock_forward(monkeypatch)
    rng = np.random.default_rng(2)
    defects = df.DefectPrior().sample(5, rng)
    bo = df.optimise_design(defects, load=0.11, n_init=4, n_iter=4,
                            n_workers=1, seed=3, verbose=False)
    assert len(bo.y) == 8
    # best-so-far curve is non-increasing and matches the reported best
    assert np.all(np.diff(bo.y_best_curve) <= 1e-9)
    assert bo.best_objective == pytest.approx(bo.y.min())
    assert bo.best_objective == pytest.approx(bo.y_best_curve[-1])
    # optimum no worse than a pathological design on the same defects
    bad = df.evaluate_design([15.0, 0.30, 1.00], defects, 0.11, n_workers=1)
    assert bo.best_objective <= bad.objective + 1e-9


def test_bo_optimum_le_baseline(monkeypatch):
    """Optimised objective <= baseline objective on a small deterministic set."""
    _patch_mock_forward(monkeypatch)
    rng = np.random.default_rng(4)
    defects = df.DefectPrior().sample(5, rng)
    base = df.evaluate_design(df.BASELINE_DESIGN, defects, 0.11, n_workers=1)
    bo = df.optimise_design(defects, load=0.11, n_init=5, n_iter=5,
                            n_workers=1, seed=5, verbose=False)
    assert bo.best_objective <= base.objective + 1e-9


# --------------------------------------------------------------------------- #
# 5. fleet_defect_prior wiring (synthetic + posterior-scaled load)
# --------------------------------------------------------------------------- #
def test_fleet_prior_default_synthetic():
    p = df.fleet_defect_prior()
    assert p.source == "synthetic"
    assert 0.10 <= p.load <= 0.13
    s = p.sample(8, np.random.default_rng(0))
    assert s.shape == (8, 4)
    assert (s[:, 0] >= p.cx_lo - 1e-9).all() and (s[:, 0] <= p.cx_hi + 1e-9).all()


def test_fleet_prior_from_posterior():
    class _FakePost:
        def r_v_mean(self):
            return np.array([0.05, 0.06, 0.05])
    class _FakeCfg:
        mu_phi_true = -3.0
    p = df.fleet_defect_prior(_FakePost(), _FakeCfg())
    assert p.source == "fleet_posterior"
    assert 0.10 <= p.load <= 0.13


# --------------------------------------------------------------------------- #
# 6. REAL FD forward smoke test (tiny grid, N=2) — slow but bounded
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_real_fd_evaluate_runs():
    rng = np.random.default_rng(0)
    defects = df.DefectPrior().sample(2, rng)
    r = df.evaluate_design(df.BASELINE_DESIGN, defects, load=0.11,
                           nx=32, ny=28, n_workers=1)
    assert np.isfinite(r.objective)
    assert r.cvar_log_growth >= r.mean_log_growth - 1e-9
    assert 0.0 <= r.p_grow <= 1.0
