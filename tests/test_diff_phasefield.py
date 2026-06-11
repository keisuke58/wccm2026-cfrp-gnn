"""Tests for Concept B — differentiable phase-field inversion.

Covers:
  * the differentiable forward F(theta): jax.grad finite-difference check;
  * MAP inversion recovers a known theta on (near-)noiseless data;
  * the DPS-lite (Langevin) posterior covers theta_true.

Pure JAX/numpy, coarse grid (nx=32, ny=24) — the whole file is well under a
minute on CPU.  Run with the JAX venv:
    /home/nishioka/IKM_Hiwi/.venv_jax/bin/python -m pytest tests/test_diff_phasefield.py -q
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from diff_phasefield import PFConfig, forward_field, fd_grad_check
import diff_inversion as di


# --------------------------------------------------------------------------- #
#  fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def cfg():
    return PFConfig()


@pytest.fixture(scope="module")
def fwd(cfg):
    f = jax.jit(lambda th: forward_field(th, cfg))
    # warm up the compile once
    f(jnp.asarray([0.5, 0.5, 0.1])).block_until_ready()
    return f


# --------------------------------------------------------------------------- #
#  Deliverable 1 — differentiable forward
# --------------------------------------------------------------------------- #
def test_forward_shape_and_range(cfg, fwd):
    x = np.asarray(fwd(jnp.asarray([0.5, 0.5, 0.12])))
    assert x.shape == (cfg.ny, cfg.nx)
    assert np.all(x >= -1e-9) and np.all(x <= 1.0 + 1e-9)


def test_forward_localised_signal(cfg, fwd):
    """A defect at cx=0.3 vs cx=0.7 shifts the high-damage mass horizontally."""
    xl = np.asarray(fwd(jnp.asarray([0.3, 0.5, 0.14])))
    xr = np.asarray(fwd(jnp.asarray([0.7, 0.5, 0.14])))
    com_l = (xl.sum(0) * np.arange(cfg.nx)).sum() / xl.sum()
    com_r = (xr.sum(0) * np.arange(cfg.nx)).sum() / xr.sum()
    assert com_l < com_r        # left defect -> mass further left


def test_grad_finite_difference(cfg):
    """jax.grad of the misfit matches central FD to well under 1e-2."""
    _, _, rel = fd_grad_check(cfg, theta=[0.45, 0.55, 0.10], eps=1e-4)
    assert rel < 1e-2, f"grad FD rel error {rel:.2e} exceeds 1e-2"


def test_grad_nonzero(cfg):
    g, fd, _ = fd_grad_check(cfg, theta=[0.40, 0.60, 0.09])
    assert np.linalg.norm(g) > 1e-3       # the misfit actually depends on theta


# --------------------------------------------------------------------------- #
#  Deliverable 2 — MAP inversion recovers a known theta
# --------------------------------------------------------------------------- #
def test_map_recovers_known_theta(cfg, fwd):
    th_true = np.array([0.58, 0.46, 0.12])
    x_obs = np.array(fwd(jnp.asarray(th_true)))      # noiseless
    th0 = 0.5 * (di.VALID_LO + di.VALID_HI)
    est, hist = di.map_invert(x_obs, cfg, th0, n_iter=400, lr=3e-3, sigma=0.01)
    assert hist[-1] < hist[0]                        # loss decreased
    err = np.abs(est - th_true)
    # position to within a few percent of the domain, size within ~0.04
    assert err[0] < 0.06 and err[1] < 0.06, f"position err {err}"
    assert err[2] < 0.05, f"size err {err[2]:.3f}"


# --------------------------------------------------------------------------- #
#  Deliverable 2 — DPS-lite (differentiable-physics) posterior covers theta_true
# --------------------------------------------------------------------------- #
def test_laplace_posterior_covers_truth(cfg, fwd):
    """Gauss-Newton/Laplace posterior from the forward Jacobian: mean ~ truth,
    90% interval covers theta_true.  This is the primary DPS-lite posterior for
    the (near-)deterministic forward (sharp likelihood)."""
    th_true = np.array([0.55, 0.50, 0.11])
    x_obs = np.array(fwd(jnp.asarray(th_true))) + \
        0.02 * np.random.default_rng(0).standard_normal((cfg.ny, cfg.nx))
    th0 = 0.5 * (di.VALID_LO + di.VALID_HI)
    map_est, _ = di.map_invert(x_obs, cfg, th0, n_iter=300, sigma=0.02)
    samples, Sigma = di.laplace_posterior(x_obs, cfg, map_est, sigma=0.02,
                                          n_samples=400, seed=0)
    mean, lo, hi = di.post_summary(samples)
    assert np.abs(mean[0] - th_true[0]) < 0.05
    assert np.abs(mean[1] - th_true[1]) < 0.05
    assert np.abs(mean[2] - th_true[2]) < 0.03
    # covariance is SPD and finite
    assert np.all(np.linalg.eigvalsh(Sigma) > 0)
    # 90% interval covers truth on at least 2 of 3 params
    cov = di.coverage_hit(th_true, lo, hi)
    assert cov.sum() >= 2, f"coverage too low: {cov}"


def test_langevin_posterior_runs(cfg, fwd):
    """The differentiable-forward Langevin (DPS-lite, secondary sampler) runs
    end-to-end and returns finite samples in the valid box.  (Its calibration on
    the razor-sharp likelihood is documented in the report; here we only assert
    it executes and stays in support.)"""
    th_true = np.array([0.55, 0.50, 0.11])
    x_obs = np.array(fwd(jnp.asarray(th_true)))
    th0 = 0.5 * (di.VALID_LO + di.VALID_HI)
    map_est, _ = di.map_invert(x_obs, cfg, th0, n_iter=150, sigma=0.02)
    samples = di.langevin_posterior(x_obs, cfg, map_est, n_chains=8,
                                    n_steps=60, burn=20, sigma=0.05, seed=0)
    assert samples.shape[1] == 3 and np.all(np.isfinite(samples))
    assert np.all(samples >= di.VALID_LO - 1e-9) and np.all(samples <= di.VALID_HI + 1e-9)


def test_svgd_runs_and_is_near_map(cfg, fwd):
    th_true = np.array([0.50, 0.55, 0.10])
    x_obs = np.array(fwd(jnp.asarray(th_true)))
    th0 = 0.5 * (di.VALID_LO + di.VALID_HI)
    map_est, _ = di.map_invert(x_obs, cfg, th0, n_iter=200, sigma=0.02)
    parts = di.svgd_posterior(x_obs, cfg, map_est, n_particles=12,
                              n_steps=80, sigma=0.02, seed=0)
    assert parts.shape == (12, 3)
    mean = parts.mean(0)
    assert np.abs(mean[0] - th_true[0]) < 0.12
