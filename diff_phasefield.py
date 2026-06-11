"""
diff_phasefield.py — differentiable (JAX) AT2 phase-field forward for
Concept B (research/STAGE4_5_RESEARCH_CONCEPTS.md): "Differentiable FMPE x
Phase-Field Inversion".

Goal
----
Close the inverse<->forward loop with ONE gradient.  The Stage-3 numpy solver
(cfrp_phasefield_2d.py) uses scipy.sparse.linalg.spsolve and a hard (d=1)
elliptical seed -> not differentiable in the defect parameters theta.  Here we
port a MINIMAL version of the same anisotropic AT2 damage physics to JAX so
that

    x = F(theta),     theta = (cx, cy, size[, log_scale])

is end-to-end differentiable: jax.grad of ||F(theta) - x_obs||^2 w.r.t. theta
runs and finite-difference-checks to ~1e-2.  This is the differentiable
"likelihood forward" used by diff_inversion.py.

What is kept vs the full Stage-3 model
--------------------------------------
KEPT (so it is the *same physics family*):
  * AT2 surface-density damage operator  -ell div(M grad d) + (Gc/ell + 2H) d = 2H
    with an ANISOTROPIC structural tensor  M = Gc (I + beta a(x)(x)a) coming
    from the ply layup (Teichtmeister 2017; same form as build_laminate_maps).
  * Staggered displacement<->damage fixed-point iterations with a history
    field H = max_t psi_e+ (Miehe 2010).
  * Transverse (Mode-I peel) loading, the same driver as the numpy model.

SIMPLIFIED (so autodiff is cheap and the solve is a clean linear system):
  * Elasticity reduced to a SCALAR anti-plane / peel proxy: one degraded
    Poisson solve  -div(g(d) grad u) = 0  with u=0 (bottom) / u=load (top),
    g(d) = (1-d)^2 + kappa.  psi_e+ = 0.5 g |grad u|^2.  This is the standard
    scalar surrogate of Mode-I energy release; it keeps crack-tip energy
    concentration without the coupled (u_x,u_y) plane-strain system.
  * Defect seeded as a SMOOTH (soft) Gaussian damage bump centred at
    (cx,cy) with radius set by `size`, instead of the hard d=1 ellipse.
    Smoothness is what makes F differentiable in theta.
  * Dense jnp.linalg.solve on a COARSE grid (nx~32) so a forward+backward is
    sub-second on CPU; no sparse adjoint bookkeeping needed (autodiff through
    the linear solve is exact).

Everything is float64, jit-compiled, and differentiable.  CPU is plenty at
nx=32; JAX picks GPU automatically if one is visible.

Run a self-check (grad finite-difference) directly:
    python diff_phasefield.py
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
#  Configuration (coarse by design)
# ---------------------------------------------------------------------------
class PFConfig:
    """Coarse differentiable-PF config.  Plain object (not a dataclass) so it
    is trivially hashable for jax.jit static args."""

    def __init__(self, nx=28, ny=20, Lx=1.0, Ly=0.75,
                 Gc=2e-2, beta=8.0, ply_angles_deg=(0.0, 60.0, -60.0, 0.0),
                 ell=None, kappa=1e-3, n_stagger=3, load=0.08):
        self.nx = int(nx)
        self.ny = int(ny)
        self.Lx = float(Lx)
        self.Ly = float(Ly)
        self.Gc = float(Gc)
        self.beta = float(beta)
        self.ply_angles_deg = tuple(float(a) for a in ply_angles_deg)
        self.kappa = float(kappa)
        self.n_stagger = int(n_stagger)
        self.load = float(load)
        self.hx = Lx / (self.nx - 1)
        self.hy = Ly / (self.ny - 1)
        self.ell = float(ell) if ell is not None else 2.2 * max(self.hx, self.hy)

    # hashable identity so PFConfig can be a jit static argument
    def _key(self):
        return (self.nx, self.ny, self.Lx, self.Ly, self.Gc, self.beta,
                self.ply_angles_deg, self.kappa, self.n_stagger, self.load,
                self.ell)

    def __hash__(self):
        return hash(self._key())

    def __eq__(self, other):
        return isinstance(other, PFConfig) and self._key() == other._key()


# ---------------------------------------------------------------------------
#  Static (theta-independent) maps: anisotropic structural tensor M = Gc*A
# ---------------------------------------------------------------------------
def build_aniso_maps(cfg: PFConfig):
    """Per-node Mxx, Myy, Mxy for M = Gc (I + beta a(x)a),  a = ply fibre dir.

    Horizontal ply bands of equal thickness (same idealisation as the numpy
    build_laminate_maps).  Returns three (ny, nx) NUMPY arrays — these are
    theta-independent and static (depend only on cfg), so they are computed
    with numpy and used to build the static FD operator outside the autodiff
    graph.
    """
    import numpy as np
    ny, nx = cfg.ny, cfg.nx
    ys = np.linspace(0.0, cfg.Ly, ny)
    n_plies = len(cfg.ply_angles_deg)
    ply_t = cfg.Ly / n_plies
    ply_of_row = np.minimum((ys / ply_t).astype(int), n_plies - 1)
    angles = np.radians(np.asarray(cfg.ply_angles_deg))
    th_row = angles[ply_of_row]                       # (ny,)
    ax = np.cos(th_row)
    ay = np.sin(th_row)
    Mxx_row = cfg.Gc * (1.0 + cfg.beta * ax * ax)
    Myy_row = cfg.Gc * (1.0 + cfg.beta * ay * ay)
    Mxy_row = cfg.Gc * (cfg.beta * ax * ay)
    ones_x = np.ones((nx,))
    Mxx = Mxx_row[:, None] * ones_x
    Myy = Myy_row[:, None] * ones_x
    Mxy = Mxy_row[:, None] * ones_x
    return Mxx, Myy, Mxy


# ---------------------------------------------------------------------------
#  Dense FD operators (coarse grid -> dense is fine and exactly differentiable)
# ---------------------------------------------------------------------------
def _grid(cfg: PFConfig):
    xs = jnp.linspace(0.0, cfg.Lx, cfg.nx)
    ys = jnp.linspace(0.0, cfg.Ly, cfg.ny)
    X, Y = jnp.meshgrid(xs, ys)            # (ny, nx)
    return X, Y


def _laplacian_divM(Mxx, Myy, Mxy, cfg: PFConfig):
    """Dense matrix L (N,N) for L d = div(M grad d), conservative flux form
    with face-averaged Mxx/Myy and cell-centred cross term — the dense analogue
    of cfrp_phasefield_2d.build_var_coeff_divM.  theta-INDEPENDENT (Mxx/Myy/Mxy
    are static), so built once with numpy and returned as a dense jnp array."""
    import numpy as np
    ny, nx = cfg.ny, cfg.nx
    N = nx * ny
    hx2, hy2 = cfg.hx ** 2, cfg.hy ** 2
    hxy4 = 4.0 * cfg.hx * cfg.hy

    def idx(i, j):
        return i + j * nx

    L = np.zeros((N, N))
    for j in range(ny):
        for i in range(nx):
            k = idx(i, j)
            if i < nx - 1:
                Me = 0.5 * (Mxx[j][i] + Mxx[j][i + 1])
                L[k, k] += -Me / hx2; L[k, idx(i + 1, j)] += Me / hx2
            if i > 0:
                Mw = 0.5 * (Mxx[j][i] + Mxx[j][i - 1])
                L[k, k] += -Mw / hx2; L[k, idx(i - 1, j)] += Mw / hx2
            if j < ny - 1:
                Mn = 0.5 * (Myy[j][i] + Myy[j + 1][i])
                L[k, k] += -Mn / hy2; L[k, idx(i, j + 1)] += Mn / hy2
            if j > 0:
                Ms = 0.5 * (Myy[j][i] + Myy[j - 1][i])
                L[k, k] += -Ms / hy2; L[k, idx(i, j - 1)] += Ms / hy2
            if 0 < i < nx - 1 and 0 < j < ny - 1 and abs(Mxy[j][i]) > 1e-14:
                cxy = 2.0 * Mxy[j][i] / hxy4
                L[k, idx(i + 1, j + 1)] += cxy
                L[k, idx(i - 1, j - 1)] += cxy
                L[k, idx(i + 1, j - 1)] += -cxy
                L[k, idx(i - 1, j + 1)] += -cxy
    return jnp.asarray(L)


def _scalar_peel_solve(d, cfg: PFConfig):
    """Degraded scalar Poisson (Mode-I peel proxy):
        -div(g(d) grad u) = 0,   u=0 on bottom row, u=load on top row,
        g(d) = (1-d)^2 + kappa.
    Returns u (ny, nx) and the elastic energy density psi = 0.5 g |grad u|^2.

    Fully differentiable in d (and hence in theta through the seed).
    Dense build + jnp.linalg.solve; autodiff backprops through the solve.
    """
    ny, nx = cfg.ny, cfg.nx
    N = nx * ny
    hx2, hy2 = cfg.hx ** 2, cfg.hy ** 2
    g = (1.0 - d) ** 2 + cfg.kappa                    # (ny, nx)

    def idx(i, j):
        return i + j * nx

    rows, cols, vals = [], [], []

    def add(r, c, v):
        rows.append(r); cols.append(c); vals.append(v)

    # build connectivity with python loops (static), coefficient values are
    # gathered from g via fancy indexing so they stay in the autodiff graph.
    gi_self, gi_nbr, e_rows, e_cols, e_h2, e_sign = [], [], [], [], [], []

    def face(k, knb, ii, jj, inb, jnb, h2):
        # -d/dx(g du/dx): face coeff = 0.5(g_self+g_nbr)/h2
        e_rows.append(k); e_cols.append(k); gi_self.append((jj, ii)); gi_nbr.append((jnb, inb)); e_h2.append(h2); e_sign.append(-1.0)
        e_rows.append(k); e_cols.append(knb); gi_self.append((jj, ii)); gi_nbr.append((jnb, inb)); e_h2.append(h2); e_sign.append(+1.0)

    for j in range(ny):
        for i in range(nx):
            k = idx(i, j)
            if i < nx - 1:
                face(k, idx(i + 1, j), i, j, i + 1, j, hx2)
            if i > 0:
                face(k, idx(i - 1, j), i, j, i - 1, j, hx2)
            if j < ny - 1:
                face(k, idx(i, j + 1), i, j, i, j + 1, hy2)
            if j > 0:
                face(k, idx(i, j - 1), i, j, i, j - 1, hy2)

    er = jnp.asarray(e_rows); ec = jnp.asarray(e_cols)
    gs_j = jnp.asarray([a[0] for a in gi_self]); gs_i = jnp.asarray([a[1] for a in gi_self])
    gn_j = jnp.asarray([a[0] for a in gi_nbr]);  gn_i = jnp.asarray([a[1] for a in gi_nbr])
    h2 = jnp.asarray(e_h2); sgn = jnp.asarray(e_sign)
    gface = 0.5 * (g[gs_j, gs_i] + g[gn_j, gn_i])
    coeff = sgn * gface / h2
    K = jnp.zeros((N, N)).at[er, ec].add(coeff)

    rhs = jnp.zeros((N,))
    # Dirichlet rows: bottom u=0, top u=load.  Replace the row by identity.
    bot = jnp.asarray([idx(i, 0) for i in range(nx)])
    top = jnp.asarray([idx(i, ny - 1) for i in range(nx)])
    dir_rows = jnp.concatenate([bot, top])
    dir_vals = jnp.concatenate([jnp.zeros((nx,)), jnp.full((nx,), cfg.load)])
    # zero those rows then set diagonal=1
    keep = jnp.ones((N,)).at[dir_rows].set(0.0)
    K = K * keep[:, None]
    K = K.at[dir_rows, dir_rows].add(1.0)
    rhs = rhs.at[dir_rows].set(dir_vals)

    u = jnp.linalg.solve(K, rhs).reshape(ny, nx)

    ux = jnp.gradient(u, cfg.hx, axis=1)
    uy = jnp.gradient(u, cfg.hy, axis=0)
    psi = 0.5 * g * (ux ** 2 + uy ** 2)
    return u, psi


def _solve_damage(H, L_divM, Gc, cfg: PFConfig):
    """One AT2 damage solve:  -ell div(M grad d) + (Gc/ell + 2H) d = 2H.
    Dense linear solve, differentiable in H (and through it, in theta)."""
    ny, nx = cfg.ny, cfg.nx
    N = nx * ny
    diag = Gc / cfg.ell + 2.0 * H.reshape(-1)
    K = -cfg.ell * L_divM + jnp.diag(diag)
    rhs = 2.0 * H.reshape(-1)
    d = jnp.linalg.solve(K, rhs).reshape(ny, nx)
    return jnp.clip(d, 0.0, 1.0)


# ---------------------------------------------------------------------------
#  Smooth (differentiable) defect seed
# ---------------------------------------------------------------------------
def soft_seed(theta, cfg: PFConfig, X, Y):
    """Smooth damage bump d0 in [0, ~0.95] from theta.

    theta = (cx, cy, size)  or  (cx, cy, size, log_scale):
      cx, cy   : centre in normalised [0,1] (mapped to [0,Lx]x[0,Ly])
      size     : Gaussian radius in normalised units (~0.03..0.25)
      log_scale: optional scalar log-amplitude / sharpness knob (Gc/beta-like);
                 here it gently scales the seed amplitude.  Optional 4th dim.

    A smooth bump (vs the hard ellipse of seed_defect) is what makes F
    differentiable in (cx, cy, size).
    """
    cx, cy, size = theta[0], theta[1], theta[2]
    amp = 0.92
    if theta.shape[0] > 3:
        amp = amp * jax.nn.sigmoid(theta[3]) * 2.0   # 0..~1.84, clipped below
    x0 = cx * cfg.Lx
    y0 = cy * cfg.Ly
    r = jnp.maximum(size, 1.5 * cfg.hx / cfg.Lx) * cfg.Lx
    g2 = jnp.exp(-(((X - x0) ** 2 + (Y - y0) ** 2) / (2.0 * r ** 2)))
    return jnp.clip(amp * g2, 0.0, 0.95)


# ---------------------------------------------------------------------------
#  The differentiable forward  x = F(theta)
# ---------------------------------------------------------------------------
@partial(jax.jit, static_argnums=(1,))
def forward_field(theta, cfg: PFConfig):
    """Differentiable AT2 phase-field forward.

    Returns the final damage field d (ny, nx) after `n_stagger` staggered
    displacement<->damage iterations under transverse peel load.  This d-field
    is the "measured field" x for the inverse problem (it stands in for the
    full-field NDT response that Stage-2 FMPE conditions on).
    """
    X, Y = _grid(cfg)
    Mxx, Myy, Mxy = build_aniso_maps(cfg)
    L_divM = _laplacian_divM(Mxx, Myy, Mxy, cfg)

    d = soft_seed(theta, cfg, X, Y)
    d0 = d
    H = jnp.zeros_like(d)
    for _ in range(cfg.n_stagger):
        _, psi = _scalar_peel_solve(d, cfg)
        # do not let the seed itself drive runaway history; cap at non-seed psi
        H = jnp.maximum(H, psi)
        d_new = _solve_damage(H, L_divM, cfg.Gc, cfg)
        # irreversibility + keep the seed present (soft max with seed)
        d = jnp.maximum(jnp.maximum(d_new, d), d0)
        d = jnp.clip(d, 0.0, 1.0)
    return d


def forward_field_np(theta, cfg: PFConfig):
    """numpy convenience wrapper (returns np array)."""
    import numpy as np
    return np.asarray(forward_field(jnp.asarray(theta, dtype=jnp.float64), cfg))


# ---------------------------------------------------------------------------
#  Misfit + grad (the "likelihood score" of Concept B)
# ---------------------------------------------------------------------------
@partial(jax.jit, static_argnums=(2,))
def misfit(theta, x_obs, cfg: PFConfig):
    """0.5 * ||F(theta) - x_obs||^2  (sum of squares)."""
    x = forward_field(theta, cfg)
    r = x - x_obs
    return 0.5 * jnp.sum(r * r)


misfit_grad = jax.jit(jax.grad(misfit, argnums=0), static_argnums=(2,))


# ---------------------------------------------------------------------------
#  Self-check: finite-difference gradient verification
# ---------------------------------------------------------------------------
def fd_grad_check(cfg: PFConfig | None = None, theta=None, eps=1e-4, seed=0):
    """Central finite-difference check of jax.grad(misfit).  Returns
    (analytic_grad, fd_grad, rel_err)."""
    import numpy as np
    cfg = cfg or PFConfig()
    if theta is None:
        theta = jnp.asarray([0.45, 0.55, 0.10], dtype=jnp.float64)
    else:
        theta = jnp.asarray(theta, dtype=jnp.float64)

    # synthesise an observation at a different theta_true
    rng = np.random.default_rng(seed)
    theta_true = jnp.asarray([0.55, 0.45, 0.14], dtype=jnp.float64)
    x_obs = forward_field(theta_true, cfg)
    x_obs = x_obs + 0.005 * jnp.asarray(rng.standard_normal(x_obs.shape))

    g = np.asarray(misfit_grad(theta, x_obs, cfg))
    fd = np.zeros_like(g)
    th = np.asarray(theta)
    for k in range(len(th)):
        tp = th.copy(); tp[k] += eps
        tm = th.copy(); tm[k] -= eps
        fp = float(misfit(jnp.asarray(tp), x_obs, cfg))
        fm = float(misfit(jnp.asarray(tm), x_obs, cfg))
        fd[k] = (fp - fm) / (2 * eps)
    denom = np.linalg.norm(g) + np.linalg.norm(fd) + 1e-12
    rel = np.linalg.norm(g - fd) / denom
    return g, fd, rel


if __name__ == "__main__":
    import time
    cfg = PFConfig()
    print(f"[diff_phasefield] grid {cfg.nx}x{cfg.ny}, ell={cfg.ell:.4f}, "
          f"n_stagger={cfg.n_stagger}, load={cfg.load}")
    t0 = time.time()
    theta0 = jnp.asarray([0.45, 0.55, 0.10])
    x = forward_field(theta0, cfg)              # triggers jit compile
    x.block_until_ready()
    print(f"  forward (incl. compile): {time.time()-t0:.2f}s, "
          f"d range [{float(x.min()):.3f}, {float(x.max()):.3f}], "
          f"mean {float(x.mean()):.4f}")
    t0 = time.time()
    for _ in range(5):
        forward_field(jnp.asarray([0.5, 0.5, 0.12]), cfg).block_until_ready()
    print(f"  warm forward: {(time.time()-t0)/5*1000:.1f} ms each")

    g, fd, rel = fd_grad_check(cfg)
    print(f"  analytic grad: {g}")
    print(f"  fd       grad: {fd}")
    print(f"  rel L2 error : {rel:.3e}  ({'PASS' if rel < 1e-2 else 'FAIL'})")
