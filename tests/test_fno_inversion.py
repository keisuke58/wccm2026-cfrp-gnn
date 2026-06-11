"""Tests for fno_inversion (Concept B v2: FNO surrogate as a differentiable,
fast inverse-problem forward map).

These tests require the trained FNO checkpoint (results/crack_surrogate/fno_crack.pt)
produced by `python crack_surrogate.py --generate --train`.  If it is missing
the surrogate-dependent tests skip with a clear message.  torch is required and
import-skipped if absent.  Kept CPU-light: small grids, few iterations.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")

import crack_surrogate as cs  # noqa: E402
import fno_inversion as fi  # noqa: E402
from cfrp_phasefield_2d import LaminateConfig, seed_defect, simulate_growth  # noqa: E402


requires_ckpt = pytest.mark.skipif(
    not os.path.exists(cs.MODEL_PT),
    reason="trained FNO checkpoint missing — run "
           "`python crack_surrogate.py --generate --train` first")


@pytest.fixture(scope="module")
def cfg():
    return LaminateConfig()


# ── 1. differentiable_seed: shape / finite / gradient-exists ──────────────────

def test_differentiable_seed_shape_and_finite(cfg):
    th = torch.tensor([0.5, 0.5, 9.0, 1.0])
    d0 = fi.differentiable_seed(th, cfg)
    assert d0.shape == (cfg.ny, cfg.nx)
    assert torch.isfinite(d0).all()
    assert float(d0.max()) <= 1.0 + 1e-5
    # peak is < 1 in general (nearest grid node sits a fraction of a cell off
    # the continuous centre), but must be a strong damage signal
    assert float(d0.max()) > 0.5
    assert float(d0.min()) >= 0.0


def test_differentiable_seed_batch(cfg):
    th = torch.tensor([[0.4, 0.5, 5.0, 0.5], [0.6, 0.5, 12.0, 1.5]])
    d0 = fi.differentiable_seed(th, cfg)
    assert d0.shape == (2, 1, cfg.ny, cfg.nx)


def test_differentiable_seed_grad_exists(cfg):
    th = torch.tensor([0.5, 0.5, 9.0, 1.0], requires_grad=True)
    d0 = fi.differentiable_seed(th, cfg)
    d0.sum().backward()
    assert th.grad is not None
    g = th.grad.numpy()
    assert np.all(np.isfinite(g))
    # cx, layer, log2size move the blob/its size → non-zero gradient
    assert abs(g[0]) > 0 and abs(g[2]) > 0 and abs(g[3]) > 0


def test_differentiable_seed_centroid_matches_numpy(cfg):
    """The smooth seed's centroid must match seed_defect's hard-ellipse centroid
    (same x0,y0 convention) so the FNO sees the defect in the right place."""
    th = np.array([0.35, 0.5, 7.0, 1.0])
    d_hard = seed_defect(*th, cfg)
    d_soft = fi.differentiable_seed(torch.as_tensor(th), cfg).detach().numpy()
    xs = np.linspace(0, cfg.Lx, cfg.nx); ys = np.linspace(0, cfg.Ly, cfg.ny)
    X, Y = np.meshgrid(xs, ys)
    cx_h = (X * (d_hard > 0.5)).sum() / (d_hard > 0.5).sum()
    cx_s = (X * d_soft).sum() / d_soft.sum()
    assert abs(cx_h - cx_s) < 0.03        # centroids within 3% of Lx


# ── 2. autograd-vs-FD gradient agreement ──────────────────────────────────────

@requires_ckpt
def test_autograd_vs_fd_smooth_outputs(cfg):
    """For the smooth scalar heads (p_grow, log1p_rel) autograd must match
    central FD of the SAME forward to high precision."""
    for which, tol in [("p_grow", 5e-3), ("log1p_rel", 5e-3)]:
        r = fi.verify_gradient(which=which, cfg=cfg, verbose=False)
        assert r["max_rel_active"] < tol, (which, r["max_rel_active"])


@requires_ckpt
def test_autograd_vs_fd_field_objective(cfg):
    """The field-sum objective is noisier (many near-cancelling pixel
    sensitivities); still expect autograd-vs-FD agreement within a loose tol."""
    r = fi.verify_gradient(which="field_sum", cfg=cfg, eps=5e-4, verbose=False)
    assert r["max_rel_active"] < 0.15


@requires_ckpt
def test_cy_gradient_is_inactive(cfg):
    """cy is the out-of-plane section coordinate — ∂out/∂cy must be ~0."""
    th = torch.tensor([0.5, 0.5, 9.0, 1.0], requires_grad=True)
    out = fi.forward_fno(th, cfg=cfg)
    out.p_grow.backward()
    assert abs(float(th.grad[1])) < 1e-5


# ── 3. forward_fno output ranges ──────────────────────────────────────────────

@requires_ckpt
def test_forward_fno_ranges(cfg):
    th = torch.tensor([0.5, 0.5, 9.0, 1.0])
    with torch.no_grad():
        out = fi.forward_fno(th, load=0.10, cfg=cfg)
    assert 0.0 <= float(out.p_grow) <= 1.0
    assert float(out.d_final.min()) >= 0.0
    assert float(out.d_final.max()) <= 1.0 + 1e-5
    assert out.d_final.shape == (cfg.ny, cfg.nx)


# ── 4. inversion recovers an in-distribution theta ────────────────────────────

@requires_ckpt
def test_inversion_recovers_indist(cfg):
    """MAP on a TRUE FD observation should recover cx and layer of an in-dist
    target to a modest tolerance (coarse grid, smooth surrogate)."""
    model, _ = fi.get_model()
    th_true = np.array([0.45, 0.5, 9.0, 1.0])
    rng = np.random.default_rng(0)
    d_obs = fi._make_observation(th_true, fi.DEFAULT_LOAD, 0.01, cfg, rng)
    # light settings to keep CPU runtime modest (1 FD obs + small MAP)
    res = fi.invert(d_obs, load=fi.DEFAULT_LOAD, sigma=0.02, cfg=cfg,
                    model=model, n_iter=120, n_post=100, n_restarts=2)
    est = res["theta_map"]
    assert abs(est[0] - th_true[0]) < 0.15        # cx within 0.15
    assert abs(est[2] - th_true[2]) < 3.0         # layer within ~3 plies


# ── 5. Laplace covariance is PSD ──────────────────────────────────────────────

@requires_ckpt
def test_laplace_covariance_psd(cfg):
    model, _ = fi.get_model()
    th_map = np.array([0.5, 0.5, 9.0, 1.0])
    _, Sigma = fi.laplace_posterior(th_map, load=fi.DEFAULT_LOAD, sigma=0.02,
                                    n_samples=50, cfg=cfg, model=model)
    assert Sigma.shape == (4, 4)
    assert np.allclose(Sigma, Sigma.T, atol=1e-10)
    eig = np.linalg.eigvalsh(Sigma)
    assert np.all(eig > 0), eig         # strictly PD (floor guarantees it)
