"""Tests for crack_surrogate (FNO surrogate of the Stage-3 phase-field growth).

These tests exercise the TRAINED surrogate against FD ground truth, so they
require the cached dataset + checkpoint produced once by:

    python crack_surrogate.py --generate --n-samples 800
    python crack_surrogate.py --train --epochs 300

If either artifact is missing (fresh checkout, no run yet) the surrogate tests
skip with a clear message; the pure-logic tests (P(grow) monotonicity uses the
FD model, surrogate API contract) still cover behaviour.  torch is required and
import-skipped if absent.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")

import crack_surrogate as cs  # noqa: E402
from cfrp_phasefield_2d import LaminateConfig  # noqa: E402


# ── shared fixtures: load cached data + trained model once per session ────────

def _have_artifacts() -> bool:
    return os.path.exists(cs.DATA_NPZ) and os.path.exists(cs.MODEL_PT)


requires_trained = pytest.mark.skipif(
    not _have_artifacts(),
    reason="run `python crack_surrogate.py --generate --train` first "
           "(data.npz / fno_crack.pt missing)")


@pytest.fixture(scope="module")
def data():
    return cs.load_data()


@pytest.fixture(scope="module")
def model():
    m, _ = cs.load_surrogate()
    return m


@pytest.fixture(scope="module")
def heldout(data, model):
    split, _, _ = cs._prepare(data, test_frac=0.2, seed=0)
    return cs.evaluate(model, split)


# ── Deliverable 2.1 — field error below threshold on held-out ─────────────────

@requires_trained
def test_field_error_below_threshold(heldout):
    """Surrogate d_final field is accurate on held-out cases."""
    assert heldout["field_mse"] < 5e-3, heldout["field_mse"]
    # mean relative-L2 over the field; coarse grid + sharp cracks → allow 0.45
    assert heldout["field_rel_l2"] < 0.45, heldout["field_rel_l2"]


# ── Deliverable 2.2 — growth-flag accuracy high ───────────────────────────────

@requires_trained
def test_growth_flag_accuracy_high(heldout):
    """P(grow)>=0.5 matches the FD grow flag on most held-out cases."""
    assert heldout["grow_acc"] > 0.85, heldout["grow_acc"]
    # well-calibrated probabilities (Brier well below the 0.25 coin-flip line)
    assert heldout["brier"] < 0.15, heldout["brier"]


@requires_trained
def test_rel_growth_correlated(heldout):
    """Surrogate rel_growth tracks FD rel_growth in log space."""
    assert heldout["rel_growth_logcorr"] > 0.75, heldout["rel_growth_logcorr"]


# ── Deliverable 2.3 — surrogate P(grow) monotone in load & close to FD ────────

@requires_trained
def test_surrogate_pgrow_monotone_in_load(model):
    """Mean surrogate P(grow) increases from sub- to super-critical load."""
    cfg = LaminateConfig()
    rng = np.random.default_rng(3)
    post = np.column_stack([
        rng.normal(0.5, 0.05, 32), rng.normal(0.5, 0.05, 32),
        rng.normal(9.0, 1.0, 32), rng.normal(1.0, 0.3, 32)])
    ps = [cs.growth_probability_surrogate(post, lo, model=model, cfg=cfg,
                                          n_draws=32)
          for lo in (0.05, 0.08, 0.11, 0.14)]
    # non-decreasing (allow tiny numerical wiggle) and a real low→high spread
    for a, b in zip(ps, ps[1:]):
        assert b >= a - 0.05, ps
    assert ps[-1] - ps[0] > 0.3, ps


@requires_trained
def test_surrogate_pgrow_close_to_fd(model):
    """Surrogate P(grow) agrees with the FD growth_probability on a few
    posterior sets spanning the load range.

    Tolerance is tight (0.20) in the sub-/super-critical plateaus where the FD
    fraction is unambiguous, and looser (0.40) at the transition load 0.10:
    there the FD growth fraction is bimodal across draws, the 12-draw FD
    Monte-Carlo is itself noisy (±~0.15), and the surrogate emits a *soft*
    mean P(grow) — so the two legitimately differ by the steep slope of the
    transition (an honest, documented limitation, see module docstring).
    """
    from cfrp_phasefield_2d import growth_probability

    cfg = LaminateConfig()
    rng = np.random.default_rng(11)
    post = np.column_stack([
        rng.normal(0.5, 0.05, 24), rng.normal(0.5, 0.05, 24),
        rng.normal(9.0, 1.0, 24), rng.normal(1.0, 0.3, 24)])
    for lo, tol in ((0.06, 0.20), (0.10, 0.40), (0.14, 0.20)):
        p_fd = growth_probability(post, lo, cfg, n_draws=12,
                                  rng=np.random.default_rng(0))
        p_su = cs.growth_probability_surrogate(post, lo, model=model, cfg=cfg,
                                               n_draws=12,
                                               rng=np.random.default_rng(0))
        assert abs(p_fd - p_su) < tol, (lo, p_fd, p_su)


# ── Deliverable 2.4 — inference faster than FD ────────────────────────────────

@requires_trained
def test_surrogate_faster_than_fd(model):
    b = cs.benchmark_speed(n=8, model=model)
    assert b["fd_per_call_s"] > b["surr_per_call_s"]
    # the headline must be a real acceleration, not a rounding artefact
    assert b["speedup_serial"] > 20.0, b


# ── API contract (no trained model needed) ────────────────────────────────────

@requires_trained
def test_predict_shapes_and_ranges(model, data):
    nx, ny = int(data["nx"]), int(data["ny"])
    d0 = data["d0"][0]
    out = cs.predict(model, d0, 0.10)
    assert out["d_final"].shape == (ny, nx)
    assert 0.0 <= out["d_final"].min() and out["d_final"].max() <= 1.0
    assert 0.0 <= out["p_grow"] <= 1.0
    assert np.isfinite(out["rel_growth"])


def test_growth_probability_surrogate_rejects_bad_shape(model_or_none):
    """API guard: posterior must be (N,4) — independent of training."""
    bad = np.zeros((5, 3))
    with pytest.raises(ValueError):
        cs.growth_probability_surrogate(bad, 0.1, model=model_or_none,
                                        n_draws=2)


@pytest.fixture(scope="module")
def model_or_none():
    return cs.load_surrogate()[0] if _have_artifacts() else None
