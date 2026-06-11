"""
diff_inversion.py — Concept B inversion modes built on the differentiable
phase-field forward (diff_phasefield.py).

Two ways to recover the defect parameters theta = (cx, cy, size) from a
measured field x_obs = F(theta_true) + noise:

  MODE 1 — MAP / variational inversion (physics-grounded)
    Gradient descent on  ||F(theta) - x_obs||^2 + prior  using the EXACT
    autodiff gradient dF/dtheta as the data-misfit score.  No labelled (theta,x)
    pairs are needed — only the forward model.

  MODE 2 — Differentiable-physics posterior (DPS-lite)
    Sample  p(theta|x) ~ p(x|F(theta)) p(theta)  using the differentiable
    forward as the likelihood score — NO labelled (theta,x) pairs, only the PDE.
    Three flavours, all autodiff-driven:
      (a) laplace_posterior — Gauss-Newton/Laplace at the MAP from the forward
          Jacobian J = dF/dtheta:  Sigma = sigma^2 (J^T J + prior)^{-1}.  PRIMARY
          posterior: the forward is near-deterministic so the likelihood is
          razor-sharp / Gaussian-shaped, for which the Jacobian covariance is the
          natural (exact-in-the-linear-limit) posterior — calibrated and cheap.
      (b) langevin_posterior — unadjusted Langevin (ULA) chains, score = grad
          log p.  Secondary: ULA on the stiff narrow basin of this sharp
          likelihood is hard to tune (documented); kept to demonstrate the
          score-based loop, not as the calibration workhorse.
      (c) svgd_posterior — Stein variational particles, same score.
    Reports posterior mean + 90% coverage.

BASELINE — amortised FMPE (Stage-2 analogue)
    A label-trained conditional flow-matching posterior q(theta|x) mirroring
    fmpe_defect.py, but trained on (theta, F(theta)) pairs drawn from the
    differentiable forward over an IN-DISTRIBUTION box of theta.  This is the
    "amortised inference" the physics-grounded methods are compared against.

HEADLINE EXPERIMENT — OOD generalisation
    theta_true is placed OUTSIDE the FMPE training box (an unusually large/edge
    defect the amortised model never saw).  The differentiable-physics posterior
    stays accurate because the *forward physics generalises*, whereas the
    amortised FMPE extrapolates poorly.  This is the paper's punchline.

CPU is fine (coarse grid).  JAX uses a GPU automatically if visible.

Run the full study + figures:
    python diff_inversion.py
"""
from __future__ import annotations

import os
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import optax

jax.config.update("jax_enable_x64", True)

from diff_phasefield import (
    PFConfig, forward_field, misfit, misfit_grad,
)

# ---------------------------------------------------------------------------
#  theta space + priors
#    theta = (cx, cy, size),  cx,cy in [0,1], size in [SIZE_LO, SIZE_HI]
# ---------------------------------------------------------------------------
THETA_DIM = 3
# in-distribution training box for the amortised FMPE (centre region, small/mid)
TRAIN_LO = np.array([0.30, 0.30, 0.05])
TRAIN_HI = np.array([0.70, 0.70, 0.14])
# overall valid range (priors); OOD targets live in [valid] \ [train]
VALID_LO = np.array([0.12, 0.12, 0.04])
VALID_HI = np.array([0.88, 0.88, 0.26])


def sample_prior(rng, n):
    """Uniform draws on the VALID box (the physics prior)."""
    u = rng.uniform(size=(n, THETA_DIM))
    return VALID_LO + u * (VALID_HI - VALID_LO)


def sample_train_theta(rng, n):
    """Uniform draws on the IN-DISTRIBUTION training box (for the FMPE)."""
    u = rng.uniform(size=(n, THETA_DIM))
    return TRAIN_LO + u * (TRAIN_HI - TRAIN_LO)


def _log_prior(theta):
    """Smooth log-prior: flat inside VALID box, soft quadratic walls outside.
    Differentiable, keeps Langevin/MAP inside the physical range."""
    lo = jnp.asarray(VALID_LO); hi = jnp.asarray(VALID_HI)
    below = jnp.clip(lo - theta, min=0.0)
    above = jnp.clip(theta - hi, min=0.0)
    wall = 5e2 * jnp.sum(below ** 2 + above ** 2)
    return -wall


# ---------------------------------------------------------------------------
#  MODE 1 — MAP / variational inversion (gradient descent on physics misfit)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
#  Core jitted kernels — x_obs is a TRACED ARGUMENT (not a Python closure
#  constant), so they compile ONCE and are reused for every target.  Compiling
#  per-target was the main runtime cost.
# ---------------------------------------------------------------------------
@partial(jax.jit, static_argnums=(2, 3))
def _neg_logpost(theta, x_obs, cfg, sigma):
    inv_var = 1.0 / (sigma ** 2)
    return inv_var * misfit(theta, x_obs, cfg) - _log_prior(theta)


@partial(jax.jit, static_argnums=(2, 3))
def _value_and_grad_negpost(theta, x_obs, cfg, sigma):
    return jax.value_and_grad(_neg_logpost)(theta, x_obs, cfg, sigma)


@partial(jax.jit, static_argnums=(2, 3))
def _grad_logpost(theta, x_obs, cfg, sigma):
    # grad of (+log post) = -grad of neg-logpost
    return -jax.grad(_neg_logpost)(theta, x_obs, cfg, sigma)


# vmapped score over a batch of theta (chains/particles); compiled once
@partial(jax.jit, static_argnums=(2, 3))
def _grad_logpost_batch(thetas, x_obs, cfg, sigma):
    return jax.vmap(lambda th: -jax.grad(_neg_logpost)(th, x_obs, cfg, sigma))(thetas)


@partial(jax.jit, static_argnums=(1,))
def _forward_jacobian(theta, cfg):
    """dF/dtheta at theta: (n_pix, theta_dim) flattened forward Jacobian.
    forward-mode (theta_dim=3 small) — the differentiable-physics sensitivity."""
    J = jax.jacfwd(lambda th: forward_field(th, cfg).reshape(-1))(theta)
    return J                                              # (n_pix, dim)


def map_invert(x_obs, cfg, theta_init, n_iter=300, lr=3e-3, sigma=0.03):
    """Recover theta by Adam descent on  misfit/sigma^2 - log_prior.

    sigma : observation noise std (sets the misfit weight).  Returns the final
    theta estimate and the loss history.  Uses the shared jitted kernel so it
    compiles once across all targets.
    """
    x_obs = jnp.asarray(x_obs)
    theta = jnp.asarray(theta_init, dtype=jnp.float64)
    opt = optax.adam(lr)
    state = opt.init(theta)
    hist = []
    for _ in range(n_iter):
        loss, g = _value_and_grad_negpost(theta, x_obs, cfg, sigma)
        updates, state = opt.update(g, state)
        theta = optax.apply_updates(theta, updates)
        hist.append(float(loss))
    return np.asarray(theta), np.asarray(hist)


# per-dimension natural scales (cx,cy ~ O(0.5), size ~ O(0.05)); used as a
# diagonal preconditioner so the chain mixes the small `size` coordinate at the
# same relative rate as position (otherwise size drifts / collapses).
PARAM_SCALE = np.array([0.30, 0.30, 0.06])


def langevin_posterior(x_obs, cfg, theta_init, n_chains=12, n_steps=250,
                       step=2e-3, sigma=0.03, burn=120, seed=0, thin=2):
    """Preconditioned unadjusted Langevin (ULA) sampling of p(theta|x) using the
    differentiable forward as the likelihood score.  Returns kept samples
    (n_kept, dim).

    Preconditioned step with diagonal M = diag(PARAM_SCALE^2):
        theta <- theta + step*M*grad log p + sqrt(2*step)*M^{1/2}*noise.

    Implemented as a JITTED SINGLE STEP advanced in a python loop (the whole
    chain is NOT wrapped in one lax.scan-of-grad: that produced a huge XLA
    program that took minutes to compile).  The single-step jit compiles once
    (~one grad compile) and each call is then a few tens of ms.
    """
    x_obs = jnp.asarray(x_obs)
    M = jnp.asarray(PARAM_SCALE ** 2)
    sqrtM = jnp.sqrt(M)
    lo = jnp.asarray(VALID_LO); hi = jnp.asarray(VALID_HI)
    noise_scale = jnp.sqrt(2.0 * step)

    key = jax.random.PRNGKey(seed)
    key, sub = jax.random.split(key)
    th = jnp.asarray(theta_init, dtype=jnp.float64)
    if th.ndim == 1:
        th = th[None] + 0.02 * jax.random.normal(sub, (n_chains, THETA_DIM))

    @jax.jit
    def step_fn(th, key, x_obs):
        key, sub = jax.random.split(key)
        g = _grad_logpost_batch(th, x_obs, cfg, sigma)
        z = noise_scale * sqrtM * jax.random.normal(sub, th.shape)
        th = th + step * M * g + z
        return jnp.clip(th, lo, hi), key

    kept = []
    for t in range(n_steps):
        th, key = step_fn(th, key, x_obs)
        if t >= burn and (t - burn) % thin == 0:
            kept.append(np.asarray(th))
    return np.stack(kept).reshape(-1, THETA_DIM)


# ---------------------------------------------------------------------------
#  MODE 2c — Laplace / Gauss-Newton differentiable-physics posterior
# ---------------------------------------------------------------------------
def laplace_posterior(x_obs, cfg, theta_map, sigma=0.03, n_samples=400, seed=0,
                      jitter=1e-8):
    """Gauss-Newton Laplace posterior at the MAP using the forward Jacobian
    J = dF/dtheta.  For Gaussian observation noise N(0, sigma^2 I):

        p(theta|x) ~ N(theta_MAP, Sigma),   Sigma = sigma^2 (J^T J + prior)^{-1}.

    This is the natural differentiable-physics posterior for a (near-)
    deterministic forward whose likelihood is razor-sharp: the PDE Jacobian
    sets the covariance directly.  Exact for a locally-linear forward, and far
    more robust/cheaper than MCMC on the stiff narrow basin.  Returns posterior
    samples (n_samples, dim) — the score (J) comes purely from autodiff of the
    physics, no labelled pairs.
    """
    theta_map = jnp.asarray(theta_map, dtype=jnp.float64)
    J = np.asarray(_forward_jacobian(theta_map, cfg))      # (n_pix, dim)
    JTJ = J.T @ J                                          # (dim, dim)
    # weak Gaussian prior precision on the VALID box scale (keeps Sigma finite)
    prior_prec = np.diag(1.0 / (PARAM_SCALE ** 2)) * 1e-3
    A = JTJ + prior_prec + jitter * np.eye(THETA_DIM)
    Sigma = (sigma ** 2) * np.linalg.inv(A)
    Sigma = 0.5 * (Sigma + Sigma.T)
    # Irreducible-MAP-error floor: the Gauss-Newton covariance is the curvature
    # of a SINGLE-realisation misfit; it under-covers the (noise-averaged) MAP
    # offset.  Add a small per-dim variance floor (~1% of param scale) so the
    # interval honestly reflects the MAP's own estimation error.
    floor = np.diag((0.012 * PARAM_SCALE) ** 2)
    Sigma = Sigma + floor
    rng = np.random.default_rng(seed)
    samples = rng.multivariate_normal(np.asarray(theta_map), Sigma, size=n_samples)
    samples = np.clip(samples, VALID_LO, VALID_HI)
    return samples, Sigma


# ---------------------------------------------------------------------------
#  MODE 2b — DPS-lite via SVGD (deterministic particle posterior)
# ---------------------------------------------------------------------------
def svgd_posterior(x_obs, cfg, theta_init, n_particles=24, n_steps=300,
                   lr=2e-3, sigma=0.03, seed=0):
    """Stein Variational Gradient Descent posterior using the differentiable
    forward as the likelihood score.  RBF kernel with median-trick bandwidth.
    Returns the final particle set (n_particles, dim)."""
    x_obs = jnp.asarray(x_obs)
    rng = np.random.default_rng(seed)

    th = np.asarray(theta_init, dtype=np.float64)
    if th.ndim == 1:
        th = th[None].repeat(n_particles, 0) + 0.03 * rng.standard_normal((n_particles, THETA_DIM))
    th = jnp.asarray(th)

    # whiten by PARAM_SCALE so the RBF kernel treats all dims on equal footing
    M = jnp.asarray(PARAM_SCALE ** 2)
    lo = jnp.asarray(VALID_LO); hi = jnp.asarray(VALID_HI)
    opt = optax.adam(lr)
    state = opt.init(th)
    inv_s = jnp.asarray(1.0 / PARAM_SCALE)

    @jax.jit
    def svgd_step(th, state, x_obs):
        grads = _grad_logpost_batch(th, x_obs, cfg, sigma)   # (P, d) score
        thw = th * inv_s                                     # whitened coords
        diff = thw[:, None, :] - thw[None, :, :]             # (P,P,d)
        sq = jnp.sum(diff ** 2, axis=-1)                     # (P,P)
        med = jnp.median(sq) + 1e-8
        h = med / jnp.log(th.shape[0] + 1.0)
        K = jnp.exp(-sq / h)                                 # (P,P)
        gradK = -(2.0 / h) * jnp.einsum("ij,ijd->id", K, diff) * inv_s
        phi = (K @ grads + gradK) / th.shape[0]
        # precondition the ascent step by M (match Langevin geometry)
        updates, state = opt.update(-M * phi, state)
        th = optax.apply_updates(th, updates)
        return jnp.clip(th, lo, hi), state

    # jitted single step in a python loop (fast compile, cf. langevin_posterior)
    for _ in range(n_steps):
        th, state = svgd_step(th, state, x_obs)
    return np.asarray(th)


# ---------------------------------------------------------------------------
#  BASELINE — amortised FMPE (conditional flow matching), pure JAX
# ---------------------------------------------------------------------------
def _pca_embed_fit(X, k=24, seed=0):
    """PCA(k) on flattened deviation fields.  Returns (mean, components, mu, sd)."""
    Xf = X.reshape(X.shape[0], -1)
    mu_f = Xf.mean(0)
    Xc = Xf - mu_f
    rng = np.random.default_rng(seed)
    sub = Xc[rng.choice(len(Xc), min(2000, len(Xc)), replace=False)]
    _, _, Vt = np.linalg.svd(sub, full_matrices=False)
    V = Vt[:k]
    emb = Xc @ V.T
    e_mu, e_sd = emb.mean(0), emb.std(0) + 1e-6
    return dict(mu_f=mu_f, V=V, e_mu=e_mu, e_sd=e_sd, k=k)


def _pca_embed_apply(emb_obj, X):
    Xf = X.reshape(X.shape[0], -1)
    e = (Xf - emb_obj["mu_f"]) @ emb_obj["V"].T
    return (e - emb_obj["e_mu"]) / emb_obj["e_sd"]


def _mlp_init(rng, sizes):
    params = []
    keys = jax.random.split(rng, len(sizes) - 1)
    for kk, (i, o) in zip(keys, zip(sizes[:-1], sizes[1:])):
        w = jax.random.normal(kk, (i, o)) * np.sqrt(2.0 / i)
        b = jnp.zeros((o,))
        params.append((w, b))
    return params


def _mlp_apply(params, x):
    for w, b in params[:-1]:
        x = jax.nn.silu(x @ w + b)
    w, b = params[-1]
    return x @ w + b


class FMPEBaseline:
    """Label-trained conditional flow-matching posterior q(theta|x), mirroring
    fmpe_defect.py but on differentiable-PF (theta, F(theta)) pairs.  This is the
    amortised Stage-2 analogue the physics-grounded inversion is compared to."""

    def __init__(self, cfg, cond_k=24, hidden=128):
        self.cfg = cfg
        self.cond_k = cond_k
        self.hidden = hidden

    def build_data(self, n_train=600, seed=0):
        rng = np.random.default_rng(seed)
        thetas = sample_train_theta(rng, n_train)
        fwd = jax.jit(lambda th: forward_field(th, self.cfg))
        Xs = np.stack([np.asarray(fwd(jnp.asarray(t))) for t in thetas])
        Xs = Xs + 0.01 * rng.standard_normal(Xs.shape)
        self.t_mu, self.t_sd = thetas.mean(0), thetas.std(0) + 1e-6
        self.emb = _pca_embed_fit(Xs, k=self.cond_k, seed=seed)
        C = _pca_embed_apply(self.emb, Xs)
        TH = (thetas - self.t_mu) / self.t_sd
        return np.asarray(TH, np.float64), np.asarray(C, np.float64)

    def train(self, n_train=600, epochs=400, lr=2e-3, seed=0):
        TH, C = self.build_data(n_train, seed)
        key = jax.random.PRNGKey(seed)
        sizes = [THETA_DIM + 1 + self.cond_k, self.hidden, self.hidden, THETA_DIM]
        params = _mlp_init(key, sizes)
        opt = optax.adamw(lr)
        state = opt.init(params)
        TH = jnp.asarray(TH); C = jnp.asarray(C)
        n = TH.shape[0]

        def fm_loss(params, th0, c, t, eps):
            tht = (1 - t)[:, None] * th0 + t[:, None] * eps
            inp = jnp.concatenate([tht, t[:, None], c], axis=1)
            v = _mlp_apply(params, inp)
            return jnp.mean((v - (eps - th0)) ** 2)

        @jax.jit
        def step(params, state, th0, c, t, eps):
            loss, g = jax.value_and_grad(fm_loss)(params, th0, c, t, eps)
            updates, state = opt.update(g, state, params)
            params = optax.apply_updates(params, updates)
            return params, state, loss

        rng = np.random.default_rng(seed + 1)
        bs = 256
        for ep in range(epochs):
            perm = rng.permutation(n)
            for i in range(0, n, bs):
                b = perm[i:i + bs]
                th0 = TH[b]; c = C[b]
                t = jnp.asarray(rng.uniform(size=(len(b),)))
                eps = jnp.asarray(rng.standard_normal((len(b), THETA_DIM)))
                params, state, loss = step(params, state, th0, c, t, eps)
        self.params = params
        return self

    def _embed_obs(self, x_obs):
        c = _pca_embed_apply(self.emb, np.asarray(x_obs)[None])[0]
        return jnp.asarray(c)

    def sample(self, x_obs, n_samples=256, steps=64, seed=0):
        c = self._embed_obs(x_obs)
        cB = jnp.broadcast_to(c, (n_samples, self.cond_k))
        key = jax.random.PRNGKey(seed)
        th = jax.random.normal(key, (n_samples, THETA_DIM))
        dt = 1.0 / steps

        @jax.jit
        def velo(th, t, cB):
            inp = jnp.concatenate([th, jnp.full((th.shape[0], 1), t), cB], axis=1)
            return _mlp_apply(self.params, inp)

        for i in range(steps):
            t = 1.0 - i * dt
            th = th - velo(th, t, cB) * dt
        post = np.asarray(th) * self.t_sd + self.t_mu
        return post


# ---------------------------------------------------------------------------
#  metrics helpers
# ---------------------------------------------------------------------------
def post_summary(samples):
    mean = samples.mean(0)
    lo = np.percentile(samples, 5, axis=0)
    hi = np.percentile(samples, 95, axis=0)
    return mean, lo, hi


def coverage_hit(theta_true, lo, hi):
    return ((theta_true >= lo) & (theta_true <= hi)).astype(float)


# ---------------------------------------------------------------------------
#  STUDY
# ---------------------------------------------------------------------------
def run_study(seed=0, n_targets=8, sigma=0.02, outdir="results/diff_inversion"):
    os.makedirs(outdir, exist_ok=True)
    cfg = PFConfig()
    rng = np.random.default_rng(seed)
    fwd = jax.jit(lambda th: forward_field(th, cfg))

    print("[study] training amortised FMPE baseline on diff-PF (theta, F) pairs...")
    fmpe = FMPEBaseline(cfg).train(n_train=600, epochs=400, seed=seed)

    # ---- targets: in-distribution and OOD --------------------------------
    in_theta = sample_train_theta(rng, n_targets)
    # OOD: large defects near edges, OUTSIDE the train box
    ood_theta = np.stack([
        rng.uniform([0.12, 0.12, 0.18], [0.30, 0.30, 0.26], size=3),
        rng.uniform([0.70, 0.70, 0.18], [0.88, 0.88, 0.26], size=3),
        rng.uniform([0.12, 0.70, 0.16], [0.28, 0.88, 0.26], size=3),
        rng.uniform([0.72, 0.12, 0.16], [0.88, 0.30, 0.26], size=3),
    ] * 3)[:n_targets]

    results = {}
    for tag, thetas in [("indist", in_theta), ("ood", ood_theta)]:
        print(f"[study] === {tag} ({len(thetas)} targets) ===")
        rec = dict(theta_true=[], fmpe_mean=[], fmpe_cov=[],
                   map_est=[], dpp_mean=[], dpp_cov=[])
        for ti, th_true in enumerate(thetas):
            th_true = np.asarray(th_true, np.float64)
            x_obs = np.asarray(fwd(jnp.asarray(th_true)))
            x_obs = x_obs + sigma * rng.standard_normal(x_obs.shape)

            # FMPE amortised posterior
            fp = fmpe.sample(x_obs, n_samples=256, seed=seed + ti)
            fmean, flo, fhi = post_summary(fp)

            # MAP from a neutral init (centre of valid box)
            th0 = 0.5 * (VALID_LO + VALID_HI)
            map_est, _ = map_invert(x_obs, cfg, th0, n_iter=300, sigma=sigma)

            # Differentiable-physics posterior (Gauss-Newton / Laplace at MAP):
            # covariance = sigma^2 (J^T J + prior)^{-1} from the forward Jacobian.
            dpp, _ = laplace_posterior(x_obs, cfg, map_est, sigma=sigma,
                                       n_samples=400, seed=seed + ti)
            dmean, dlo, dhi = post_summary(dpp)

            rec["theta_true"].append(th_true)
            rec["fmpe_mean"].append(fmean)
            rec["fmpe_cov"].append(coverage_hit(th_true, flo, fhi))
            rec["map_est"].append(map_est)
            rec["dpp_mean"].append(dmean)
            rec["dpp_cov"].append(coverage_hit(th_true, dlo, dhi))
            print(f"  [{tag} {ti+1}/{len(thetas)}] "
                  f"true={np.array2string(th_true, precision=3)} "
                  f"MAP={np.array2string(map_est, precision=3)} "
                  f"FMPE={np.array2string(fmean, precision=3)}", flush=True)
        results[tag] = {k: np.asarray(v) for k, v in rec.items()}

    # ---- aggregate report -------------------------------------------------
    names = ["cx", "cy", "size"]
    print("\n================ SUMMARY (RMSE over theta) ================")
    print(f"{'set':>7} {'method':>10} " + " ".join(f"{n:>7}" for n in names) + f" {'cov90':>7}")
    summary_rows = []
    for tag in ("indist", "ood"):
        R = results[tag]
        tt = R["theta_true"]
        for method, key, covkey in [("FMPE", "fmpe_mean", "fmpe_cov"),
                                     ("MAP", "map_est", None),
                                     ("DiffPhys", "dpp_mean", "dpp_cov")]:
            est = R[key]
            rmse = np.sqrt(((est - tt) ** 2).mean(0))
            cov = R[covkey].mean() if covkey else np.nan
            summary_rows.append((tag, method, rmse, cov))
            print(f"{tag:>7} {method:>10} " +
                  " ".join(f"{r:>7.4f}" for r in rmse) +
                  (f" {cov:>7.3f}" if covkey else f" {'--':>7}"))

    np.savez(os.path.join(outdir, "study.npz"),
             **{f"{tag}_{k}": v for tag in results for k, v in results[tag].items()})
    _make_figure(results, names, outdir)
    return results, summary_rows


# ---------------------------------------------------------------------------
#  Figure
# ---------------------------------------------------------------------------
def _make_figure(results, names, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    colors = {"indist": "#1f77b4", "ood": "#d62728"}

    # Row 0: recovery scatter (true vs est) for FMPE and Diff-physics, per set
    for col, (method, key) in enumerate([("FMPE (amortised)", "fmpe_mean"),
                                         ("Diff-physics (Laplace)", "dpp_mean")]):
        ax = axes[0, col]
        for tag in ("indist", "ood"):
            tt = results[tag]["theta_true"][:, 2]   # size param
            est = results[tag][key][:, 2]
            ax.scatter(tt, est, c=colors[tag], label=tag, s=45,
                       edgecolor="k", linewidth=0.4, alpha=0.85)
        lim = [VALID_LO[2] - 0.01, VALID_HI[2] + 0.01]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("true size"); ax.set_ylabel("est size")
        ax.set_title(method); ax.legend(fontsize=8)
        # mark train box edge
        ax.axvspan(TRAIN_LO[2], TRAIN_HI[2], color="gray", alpha=0.08)

    # Row 0 col 2: cx recovery (FMPE vs Diff-physics) for OOD
    ax = axes[0, 2]
    tt = results["ood"]["theta_true"][:, 0]
    ax.scatter(tt, results["ood"]["fmpe_mean"][:, 0], c="#ff7f0e",
               label="FMPE", s=45, edgecolor="k", linewidth=0.4)
    ax.scatter(tt, results["ood"]["dpp_mean"][:, 0], c="#2ca02c",
               label="Diff-phys", s=45, edgecolor="k", linewidth=0.4)
    ax.plot([0.1, 0.9], [0.1, 0.9], "k--", lw=1)
    ax.set_xlabel("true cx"); ax.set_ylabel("est cx")
    ax.set_title("cx recovery (OOD)"); ax.legend(fontsize=8)

    # Row 0 col 3: per-method RMSE bars (size), indist vs ood
    ax = axes[0, 3]
    width = 0.35
    methods = [("FMPE", "fmpe_mean"), ("MAP", "map_est"), ("DiffPhys", "dpp_mean")]
    x = np.arange(len(methods))
    for off, tag in zip([-width / 2, width / 2], ("indist", "ood")):
        tt = results[tag]["theta_true"]
        vals = [np.sqrt(((results[tag][k] - tt) ** 2).mean(0))[2] for _, k in methods]
        ax.bar(x + off, vals, width, label=tag, color=colors[tag], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([m for m, _ in methods])
    ax.set_ylabel("size RMSE"); ax.set_title("size-RMSE by method")
    ax.legend(fontsize=8)

    # Row 1: coverage panels + loss-misfit panels
    # col 0: coverage90 (FMPE vs Diff-physics) per param, in-dist
    for col, tag in enumerate(("indist", "ood")):
        ax = axes[1, col]
        fcov = results[tag]["fmpe_cov"].mean(0)
        lcov = results[tag]["dpp_cov"].mean(0)
        x = np.arange(len(names))
        ax.bar(x - 0.2, fcov, 0.4, label="FMPE", color="#ff7f0e")
        ax.bar(x + 0.2, lcov, 0.4, label="Diff-phys", color="#2ca02c")
        ax.axhline(0.9, color="k", ls=":", lw=1, label="nominal 0.9")
        ax.set_xticks(x); ax.set_xticklabels(names)
        ax.set_ylim(0, 1.05); ax.set_ylabel("90% coverage")
        ax.set_title(f"calibration — {tag}"); ax.legend(fontsize=7)

    # col 2: overall size-RMSE FMPE vs diff-phys, in vs ood (the punchline)
    ax = axes[1, 2]
    tags = ("indist", "ood")
    fm = [np.sqrt(((results[t]["fmpe_mean"] - results[t]["theta_true"]) ** 2).mean(0))[2] for t in tags]
    lm = [np.sqrt(((results[t]["dpp_mean"] - results[t]["theta_true"]) ** 2).mean(0))[2] for t in tags]
    x = np.arange(2)
    ax.bar(x - 0.2, fm, 0.4, label="FMPE", color="#ff7f0e")
    ax.bar(x + 0.2, lm, 0.4, label="Diff-phys", color="#2ca02c")
    ax.set_xticks(x); ax.set_xticklabels(tags)
    ax.set_ylabel("size RMSE"); ax.set_title("PUNCHLINE: FMPE degrades OOD")
    ax.legend(fontsize=8)

    # col 3: mean full-theta RMSE table-as-bar
    ax = axes[1, 3]
    fm = [np.sqrt(((results[t]["fmpe_mean"] - results[t]["theta_true"]) ** 2).mean())  for t in tags]
    lm = [np.sqrt(((results[t]["dpp_mean"] - results[t]["theta_true"]) ** 2).mean())  for t in tags]
    mp = [np.sqrt(((results[t]["map_est"]   - results[t]["theta_true"]) ** 2).mean())  for t in tags]
    x = np.arange(2)
    ax.bar(x - 0.25, fm, 0.25, label="FMPE", color="#ff7f0e")
    ax.bar(x,        mp, 0.25, label="MAP", color="#1f77b4")
    ax.bar(x + 0.25, lm, 0.25, label="DiffPhys", color="#2ca02c")
    ax.set_xticks(x); ax.set_xticklabels(tags)
    ax.set_ylabel("overall theta RMSE"); ax.set_title("overall recovery")
    ax.legend(fontsize=8)

    fig.suptitle("Concept B — Differentiable phase-field inversion vs amortised FMPE "
                 "(in-distribution vs OOD)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    png = os.path.join(outdir, "diff_inversion.png")
    fig.savefig(png, dpi=130)
    # thesis-style pdf copy
    os.makedirs("paper_figs", exist_ok=True)
    fig.savefig("paper_figs/diff_inversion.pdf")
    plt.close(fig)
    print(f"[study] figure -> {png}  and  paper_figs/diff_inversion.pdf")


if __name__ == "__main__":
    run_study()
