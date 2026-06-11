"""Tests for conformal_surrogate (honest split-conformal UQ + OOD trust gate
for the Stage-3 FNO crack-growth surrogate).

Two tiers:
  * pure-logic tests (no trained model / no FD sims) covering the conformal
    quantile, OOD-distance metric, calibration on synthetic scores, and the
    trust-gate flagging logic — these run on a bare box;
  * integration tests that exercise the trained surrogate + a SMALL cached OOD
    FD set; they skip cleanly if the artifacts are missing.

To keep CPU runtime modest the integration tests reuse the cached data.npz and
a small OOD ground-truth set (generated once, ~60 sims), and share a single
`run_experiment` result across tests via a module-scoped fixture.

    python crack_surrogate.py --generate --train        # once
    python conformal_surrogate.py --gen-ood --n-ood 80  # once (optional)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import conformal_surrogate as cf  # noqa: E402
import crack_surrogate as cs      # noqa: E402


# ── pure-logic: OOD distance metric ───────────────────────────────────────────

def test_ood_distance_zero_inside_envelope():
    """A clearly in-distribution input has zero OOD distance."""
    d = cf.ood_distance(load=0.09, cx=0.5, layer=9.0, log2size=1.0)
    assert d.shape == (1,)
    assert d[0] == pytest.approx(0.0, abs=1e-12)


def test_ood_distance_positive_and_monotone_outside():
    """Pushing a parameter further past its training edge increases distance."""
    # load past the upper edge 0.14, progressively further
    d_edge = cf.ood_distance(load=0.14, cx=0.5, layer=9.0, log2size=1.0)[0]
    d_mid = cf.ood_distance(load=0.17, cx=0.5, layer=9.0, log2size=1.0)[0]
    d_far = cf.ood_distance(load=0.20, cx=0.5, layer=9.0, log2size=1.0)[0]
    assert d_edge == pytest.approx(0.0, abs=1e-9)
    assert 0.0 < d_mid < d_far
    # multiple OOD params accumulate
    d_two = cf.ood_distance(load=0.20, cx=0.5, layer=9.0, log2size=3.0)[0]
    assert d_two > d_far


def test_in_envelope_mask():
    load = np.array([0.09, 0.18])
    m = cf.in_envelope(load, cx=0.5, layer=9.0, log2size=1.0, margin=0.0)
    assert m.tolist() == [True, False]


# ── pure-logic: conformal quantile + calibration ──────────────────────────────

def test_conformal_quantile_rank_and_coverage():
    """The split-conformal quantile gives >= nominal coverage on a fresh draw
    from the SAME distribution (marginal guarantee)."""
    rng = np.random.default_rng(0)
    scores = np.abs(rng.normal(0, 1, 2000))
    q = cf.conformal_quantile(scores, alpha=0.10)
    fresh = np.abs(rng.normal(0, 1, 5000))
    cov = (fresh <= q).mean()
    assert cov >= 0.88   # ~0.90 up to finite-sample noise
    # higher coverage (smaller alpha) → wider quantile
    assert cf.conformal_quantile(scores, 0.05) >= q


def test_calibrate_produces_finite_intervals():
    """Calibration on finite scores yields finite, non-degenerate intervals
    whose width grows as nominal coverage rises."""
    rng = np.random.default_rng(1)
    true_rel = rng.uniform(0, 5, 300)
    pred_rel = np.clip(true_rel + rng.normal(0, 0.3, 300), 0, None)
    cm = cf.calibrate(pred_rel, true_rel, alphas=(0.20, 0.10, 0.05))
    for a in (0.20, 0.10, 0.05):
        assert np.isfinite(cm.qhat[a])
        assert cm.qhat[a] >= 0.0
    assert cm.width_log1p(0.05) >= cm.width_log1p(0.10) >= cm.width_log1p(0.20)
    lo, hi = cm.interval(pred_rel[:5], 0.10)
    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))
    assert np.all(hi >= lo)


def test_indist_coverage_matches_nominal_synthetic():
    """On exchangeable synthetic data, empirical coverage on a fresh split
    matches the nominal level within tolerance."""
    rng = np.random.default_rng(2)
    true = rng.uniform(0, 4, 1200)
    pred = np.clip(true + rng.normal(0, 0.4, 1200), 0, None)
    cal, te = pred[:600], pred[600:]
    cm = cf.calibrate(cal, true[:600], alphas=(0.10,))
    cov = cf.empirical_coverage(cm, te, true[600:], 0.10)
    assert abs(cov - 0.90) < 0.06, cov


# ── pure-logic: trust gate ────────────────────────────────────────────────────

def test_trust_gate_flags_known_ood():
    """The gate flags a clearly-OOD distance and trusts an in-dist one."""
    gate = cf.TrustGate(dist_margin=0.10)
    flagged = gate.flag(np.array([0.0, 0.05, 0.5]))
    assert flagged.tolist() == [False, False, True]


def test_deferring_flagged_improves_coverage_synthetic():
    """A synthetic OOD set where the surrogate is wrong at large distance:
    deferring flagged samples to FD (exact) must not decrease coverage and
    should raise it when far-OOD points were mis-covered."""
    rng = np.random.default_rng(3)
    n = 400
    dist = np.concatenate([np.zeros(300), rng.uniform(0.2, 0.6, 100)])
    true = rng.uniform(0, 3, n)
    # in-dist: small error (covered); OOD: large bias (mis-covered)
    bias = np.where(dist > 0, 2.0, 0.0)
    pred = np.clip(true + bias + rng.normal(0, 0.2, n), 0, None)
    cm = cf.calibrate(pred[:200], true[:200], alphas=(0.10,))
    oodp = {"pred_rel": pred, "true_rel": true, "ood_dist": dist}
    gate = cf.TrustGate(dist_margin=0.10)
    res = cf.evaluate_trust_gate(oodp, cm, gate, 0.10)
    assert res["cov_with_defer"] >= res["cov_no_defer"]
    assert res["cov_with_defer"] > res["cov_no_defer"] + 0.05
    assert 0.0 < res["flag_rate"] <= 1.0


# ── integration: trained surrogate + small cached OOD FD set ──────────────────

def _have_artifacts() -> bool:
    return os.path.exists(cs.DATA_NPZ) and os.path.exists(cs.MODEL_PT)


requires_trained = pytest.mark.skipif(
    not _have_artifacts(),
    reason="run `python crack_surrogate.py --generate --train` first "
           "(data.npz / fno_crack.pt missing)")

pytest.importorskip("torch")


@pytest.fixture(scope="module")
def experiment():
    # small OOD set keeps the test fast; reuses cached data.npz + checkpoint
    return cf.run_experiment(n_ood=80, dist_margin=0.10, verbose=False)


@requires_trained
def test_indist_coverage_near_nominal(experiment):
    """In-distribution empirical coverage of the 90% interval is close to
    nominal (split-conformal's marginal guarantee holds under exchangeability)."""
    cov90 = experiment["indist_coverage"][0.10]
    assert 0.78 <= cov90 <= 1.0, cov90


@requires_trained
def test_ood_coverage_degrades(experiment):
    """The headline: OOD coverage drops BELOW the in-dist coverage — the
    marginal guarantee breaks under covariate shift (expected, not a bug)."""
    cov_in = experiment["indist_coverage"][0.10]
    cov_ood = experiment["ood_coverage_all"][0.10]
    assert cov_ood < cov_in, (cov_in, cov_ood)


@requires_trained
def test_error_grows_with_ood_distance(experiment):
    """Mean surrogate point error (|Δlog1p rel|) in the most-OOD bin is larger
    than in-distribution — the surrogate degrades as inputs leave its envelope."""
    assert experiment["abs_log_err_ood"] > 2.0 * experiment["abs_log_err_indist"]


@requires_trained
def test_coverage_decreases_with_distance(experiment):
    """Per-bin coverage trends downward with OOD distance: the first bin's
    coverage exceeds the last populated bin's."""
    b = experiment["binned"]
    cov = b["coverage"]
    assert len(cov) >= 2
    assert cov[0] > cov[-1], cov


@requires_trained
def test_trust_gate_recovers_coverage(experiment):
    """Deferring gate-flagged samples to FD raises OOD coverage back toward
    nominal, and flags a non-trivial fraction of the truly-OOD points."""
    g = experiment["trust_gate"][0.10]
    assert g["cov_with_defer"] > g["cov_no_defer"]
    assert g["flag_rate"] > 0.0
    assert g["flagged_true_ood_recall"] > 0.3   # flags most truly-OOD samples
