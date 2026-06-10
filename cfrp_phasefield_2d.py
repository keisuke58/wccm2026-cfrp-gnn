"""
cfrp_phasefield_2d.py — Stage-3 CFRP laminate phase-field prognosis layer.

Pipeline position (CFRP NDT, WCCM2026 / thesis):
  Stage-0 Mahalanobis screening → Stage-1 GNN classification
  → Stage-2 FMPE amortised posterior θ = (cx, cy, layer, log2size)
    (see fmpe_defect.py; cx, cy normalised [0,1], layer integer-ish float,
     defect size = 2**log2size elements)
  → Stage-3 THIS MODULE: anisotropic phase-field growth prognosis
    P(growth | NDT measurement) and a flight-clearance decision.

Formulation (adopted in wafer-proc-sim/research/phasefield/
CFRP_PHASEFIELD_LITERATURE.md, §5 — that memo is the spec):

  * Bulk: AT2 phase-field with history field H = max_t ψ_e⁺ (Miehe 2010),
    staggered displacement/damage solves on a 2D plane-strain FD grid.
  * Ply anisotropy: per-ply structural tensor in the crack surface density,
        γ(d, ∇d) = d²/(2ℓ) + (ℓ/2) ∇d · A(x) ∇d,
        A(x) = I + β a(θ_ply)⊗a(θ_ply),   a = fiber direction,
    (Teichtmeister 2017; Quintanas-Corominas 2019).  Effective toughness
    Gc_eff(n) ≈ Gc (1 + β (n·a)²): crack normals parallel to the fibers are
    penalised, so matrix cracks run ALONG fibers.  Elasticity is kept
    isotropic — a documented approximation (crack-path selection is dominated
    by the surface-energy anisotropy when β >> 1; load-displacement curves
    are not quantitative).
  * Interface / delamination channel: thin inter-ply bands (width ~ℓ) with
    reduced Gc (DCB/ENF anchors: GIc_interface / Gc_ply,transverse ≈ 0.3–0.6,
    memo §2) and an interface-aligned structural tensor.  This is a PURE-PF
    SURROGATE of the PF+CZM hybrids of Paggi–Reinoso (CMAME 2017) and
    Carollo–Reinoso–Paggi (Compos. Struct. 2017) — memo §2.4: it reproduces
    the deflection-vs-penetration competition qualitatively but has no
    traction–separation law, no mode-mix dependence, no interface stiffness.
  * Damage operator with spatially varying coefficients, conservative
    (flux) form:  -ℓ ∇·(Gc(x) A(x) ∇d) + (Gc(x)/ℓ + 2H) d = 2H,
    face-averaged Gc·A on the axis terms, cell-centred Axy on cross terms.

Geometry: 2D laminate cross-section [0,Lx]×[0,Ly]; x = in-plane direction,
y = through-thickness; horizontal ply bands.  `ply_angles_deg` are the
effective in-plane projections of the fiber directions onto the modelled
cross-section (a 2D idealisation of the 3D layup).

Loading: `load` is a nominal TRANSVERSE (through-thickness, Mode-I peel)
strain ε_yy = u_y(top)/Ly.  Rationale: the FMPE defects are planar
(hole/delamination-like) flaws lying in the laminate plane; in a 2D
cross-section they appear as x-aligned slits, which in-plane tension ε_xx
barely loads (and fiber-cutting growth is β-penalised by design).  The
physically dominant growth driver for such flaws is the local interlaminar
peel/shear stress produced by flight loads (panel bending, sublaminate
buckling); we use transverse tension as a scalar proxy for that local
driver.  Mapping vehicle-level loads to this local peel strain requires an
outer structural analysis and is out of scope here — P(growth) curves are
therefore monotone trends in a normalised load, not absolute clearances.

Provenance: the FD elasticity solver, the staggered AT2 loop and the
history-field logic are copied/adapted from
  /home/nishioka/git/wafer-proc-sim/research/phasefield/at2_simulator_2d.py
(wafer-proc-sim is read-only reference; this module is self-contained).
The damage operator generalises `_build_2d_laplacian` from constant A to
spatially varying Gc(x)·A(x); the constant-coefficient case recovers it.

Known limitations (memo §6 — honest accounting):
  * AT2 nominal strength scales as ℓ^(-1/2): absolute failure loads are NOT
    quantitative; report P(growth) trends vs load, calibrate ℓ before
    quoting strengths (or migrate to Wu's PF-CZM).
  * Isotropic ply elasticity: the crack DRIVING force ignores E1/E2 ≈ 10–20.
  * Interface band is a Gc surrogate, not a cohesive law (see above).
  * Quasi-static only: no fatigue.  True fatigue PF = Carrara et al. 2020.

CPU, numpy/scipy only, coarse grids (nx ~ 60): one growth simulation runs in
seconds; growth_probability over ~20 posterior draws in minutes.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LaminateConfig:
    """Laminate cross-section + solver configuration (normalised units).

    Physical anchors (memo §5): Gc_ply = matrix-dominated transverse
    intralaminar toughness (0.2–1.0 kJ/m²); Gc_interface_ratio anchored to
    DCB mode-I GIc ≈ 0.2–0.5 kJ/m²; beta_ply = 20–50 (effective fiber/matrix
    toughness ratio 1+β, conservative vs the experimental ~10²; β > ~100
    degrades FD conditioning at nx ≈ 60); ell ≥ 2 grid spacings.
    """
    # domain / grid
    Lx: float = 1.0
    Ly: float = 0.5
    nx: int = 60
    ny: int = 48
    # isotropic ply elasticity (normalised)
    E:  float = 1.0
    nu: float = 0.25
    # fracture
    Gc_ply: float = 8e-4          # matrix-dominated transverse Gc (normalised)
    beta_ply: float = 25.0        # fiber-direction toughening (memo: 20–50)
    ply_angles_deg: tuple = (0.0, 30.0, -30.0, 0.0, 0.0, -30.0, 30.0, 0.0)
    Gc_interface_ratio: float = 0.4   # Gc_int / Gc_ply (memo: 0.3–0.6)
    beta_interface: float = 25.0      # interface-aligned structural tensor
    interface_cells: int = 1          # interface band half... full width [rows]
    ell: float | None = None          # PF length; default 2.2*max(hx,hy)
    kappa_res: float = 1e-6
    # FMPE θ mapping
    n_layers_data: int = 19           # layer index range of the FMPE posterior
    defect_elem_frac: float = 0.03    # one defect "element" as fraction of Lx
    # growth criterion / load stepping
    growth_rel_threshold: float = 0.5  # relative crack-area increase = growth
    n_load_steps: int = 12

    def __post_init__(self):
        if self.ell is None:
            self.ell = 2.2 * max(self.hx, self.hy)

    @property
    def hx(self) -> float:
        return self.Lx / (self.nx - 1)

    @property
    def hy(self) -> float:
        return self.Ly / (self.ny - 1)

    @property
    def n_plies(self) -> int:
        return len(self.ply_angles_deg)

    def lame(self) -> tuple[float, float]:
        lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        mu  = self.E / (2.0 * (1.0 + self.nu))
        return lam, mu


@dataclass
class GrowthResult:
    grown: bool
    rel_growth: float       # (area_final - area0) / area0
    area0: float
    area_final: float
    d0: np.ndarray          # seeded damage field (ny, nx)
    d_final: np.ndarray     # final damage field (ny, nx)
    peak_F: float           # peak reaction force (qualitative only, see memo)


# ─────────────────────────────────────────────────────────────────────────────
#  Laminate coefficient maps:  Gc(x) and M(x) = Gc(x) · A(x)
# ─────────────────────────────────────────────────────────────────────────────

def build_laminate_maps(cfg: LaminateConfig
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                   np.ndarray, np.ndarray]:
    """Build per-node fracture coefficient maps for the laminate.

    Returns (Gc_map, Mxx, Myy, Mxy, interface_mask), each (ny, nx):
      Gc_map         — local fracture toughness
      Mxx, Myy, Mxy  — components of M = Gc · A,  A = I + β a⊗a
      interface_mask — True on inter-ply interface band rows

    Plies are horizontal bands of equal thickness; between consecutive plies
    an interface band of `interface_cells` rows gets Gc_int = ratio·Gc_ply
    and an interface-aligned structural tensor (a = x̂), i.e. delamination
    (cracks running along x) is the cheap direction inside the band.
    """
    ny, nx = cfg.ny, cfg.nx
    ys = np.linspace(0.0, cfg.Ly, ny)
    ply_t = cfg.Ly / cfg.n_plies

    Gc_map = np.full((ny, nx), cfg.Gc_ply)
    Mxx = np.empty((ny, nx)); Myy = np.empty((ny, nx)); Mxy = np.empty((ny, nx))
    interface_mask = np.zeros((ny, nx), dtype=bool)

    # ply assignment per row
    ply_of_row = np.minimum((ys / ply_t).astype(int), cfg.n_plies - 1)
    for j in range(ny):
        th = np.radians(cfg.ply_angles_deg[ply_of_row[j]])
        a = np.array([np.cos(th), np.sin(th)])
        A = np.eye(2) + cfg.beta_ply * np.outer(a, a)
        M = cfg.Gc_ply * A
        Mxx[j, :], Myy[j, :], Mxy[j, :] = M[0, 0], M[1, 1], M[0, 1]

    # interface bands: rows nearest each inter-ply line, `interface_cells` wide
    Gc_int = cfg.Gc_interface_ratio * cfg.Gc_ply
    a_int = np.array([1.0, 0.0])                       # interface tangent = x̂
    A_int = np.eye(2) + cfg.beta_interface * np.outer(a_int, a_int)
    M_int = Gc_int * A_int
    for k in range(1, cfg.n_plies):
        y_k = k * ply_t
        j0 = int(np.argmin(np.abs(ys - y_k)))
        lo = max(j0 - (cfg.interface_cells - 1) // 2, 0)
        hi = min(lo + cfg.interface_cells, ny)
        Gc_map[lo:hi, :] = Gc_int
        Mxx[lo:hi, :], Myy[lo:hi, :], Mxy[lo:hi, :] = (M_int[0, 0],
                                                       M_int[1, 1],
                                                       M_int[0, 1])
        interface_mask[lo:hi, :] = True

    return Gc_map, Mxx, Myy, Mxy, interface_mask


# ─────────────────────────────────────────────────────────────────────────────
#  Damage operator: -∇·(M(x) ∇d), conservative flux form
#  (generalises at2_simulator_2d._build_2d_laplacian to spatially varying M;
#   constant M recovers it exactly — memo §5 "numerical implications")
# ─────────────────────────────────────────────────────────────────────────────

def build_var_coeff_divM(Mxx: np.ndarray, Myy: np.ndarray, Mxy: np.ndarray,
                         hx: float, hy: float) -> sp.csr_matrix:
    """Sparse FD matrix for  L d = ∇·(M(x)·∇d),  M = [[Mxx,Mxy],[Mxy,Myy]].

    Axis terms in conservative form with face-averaged coefficients
    (zero-flux Neumann: boundary faces simply contribute nothing).
    Cross terms 2·Mxy·∂²d/∂x∂y with cell-centred Mxy on interior nodes
    (they vanish at boundary nodes, mirroring the elasticity builder).
    """
    ny, nx = Mxx.shape
    N = nx * ny
    rows, cols, vals = [], [], []

    def add(r, c, v):
        rows.append(r); cols.append(c); vals.append(v)

    for j in range(ny):
        for i in range(nx):
            k = i + j * nx
            # x-axis flux faces
            if i < nx - 1:
                Me = 0.5 * (Mxx[j, i] + Mxx[j, i + 1])
                add(k, k, -Me / hx**2); add(k, k + 1, Me / hx**2)
            if i > 0:
                Mw = 0.5 * (Mxx[j, i] + Mxx[j, i - 1])
                add(k, k, -Mw / hx**2); add(k, k - 1, Mw / hx**2)
            # y-axis flux faces
            if j < ny - 1:
                Mn = 0.5 * (Myy[j, i] + Myy[j + 1, i])
                add(k, k, -Mn / hy**2); add(k, k + nx, Mn / hy**2)
            if j > 0:
                Ms = 0.5 * (Myy[j, i] + Myy[j - 1, i])
                add(k, k, -Ms / hy**2); add(k, k - nx, Ms / hy**2)
            # cross terms: 2 Mxy ∂²d/∂x∂y  (interior only)
            if 0 < i < nx - 1 and 0 < j < ny - 1 and abs(Mxy[j, i]) > 1e-14:
                cxy = 2.0 * Mxy[j, i] / (4.0 * hx * hy)
                add(k, (i + 1) + (j + 1) * nx,  cxy)
                add(k, (i - 1) + (j - 1) * nx,  cxy)
                add(k, (i + 1) + (j - 1) * nx, -cxy)
                add(k, (i - 1) + (j + 1) * nx, -cxy)

    return sp.csr_matrix((vals, (rows, cols)), shape=(N, N))


def _solve_damage(H: np.ndarray, d_prev: np.ndarray, L_divM: sp.csr_matrix,
                  Gc_map: np.ndarray, ell: float) -> np.ndarray:
    """One staggered AT2 damage solve with spatially varying Gc, A:

        -ℓ ∇·(Gc A ∇d) + (Gc/ℓ + 2H) d = 2H,    d ≥ d_prev (irreversible).

    Solve-then-clip (standard AT2 treatment of irreversibility).
    """
    ny, nx = H.shape
    diag = Gc_map.ravel() / ell + 2.0 * H.ravel()
    K = -ell * L_divM + sp.diags(diag, format="csr")
    rhs = 2.0 * H.ravel()
    d_flat = spla.spsolve(K, rhs)
    return np.clip(d_flat.reshape(ny, nx), d_prev, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
#  2D plane-strain elasticity (u_x, u_y), degraded by g(d)
#  Copied from at2_simulator_2d._build_full_elasticity_matrix (provenance:
#  wafer-proc-sim/research/phasefield/at2_simulator_2d.py), signature adapted
#  to scalars; the stencil is unchanged.
# ─────────────────────────────────────────────────────────────────────────────

def _dof(i: int, j: int, c: int, nx: int) -> int:
    """DOF index: c=0 → u_x, c=1 → u_y."""
    return 2 * (i + j * nx) + c


def _build_full_elasticity_matrix(g: np.ndarray, lam: float, mu: float,
                                  nx: int, ny: int, hx: float, hy: float
                                  ) -> sp.csr_matrix:
    """Full 2D plane-strain FD elasticity with (u_x, u_y) DOFs.

    Neumann BCs use the flux-based formulation: zero-flux faces contribute
    nothing.  Cross terms (λ+μ)·g·∂²u/∂x∂y vanish at boundary nodes.
    """
    M = 2 * nx * ny
    rows_l, cols_l, vals_l = [], [], []

    def add(r, c, v):
        rows_l.append(r); cols_l.append(c); vals_l.append(v)

    def _face_term(row, self_c, nbr_c, coeff, g_face, h2):
        add(row, self_c, -coeff * g_face / h2)
        add(row, nbr_c,  coeff * g_face / h2)

    for j in range(ny):
        for i in range(nx):
            ip = i + 1 if i < nx - 1 else i
            im = i - 1 if i > 0      else i
            jp = j + 1 if j < ny - 1 else j
            jm = j - 1 if j > 0      else j

            gxp = 0.5 * (g[j, i] + g[j, min(i + 1, nx - 1)])
            gxm = 0.5 * (g[j, i] + g[j, max(i - 1, 0)])
            gyp = 0.5 * (g[j, i] + g[min(j + 1, ny - 1), i])
            gym = 0.5 * (g[j, i] + g[max(j - 1, 0),    i])
            gc  = g[j, i]

            at_xL = (i == 0); at_xR = (i == nx - 1)
            at_yB = (j == 0); at_yT = (j == ny - 1)
            cross_ok = not (at_xL or at_xR or at_yB or at_yT)

            rx = _dof(i, j, 0, nx)
            ry = _dof(i, j, 1, nx)
            a = lam + 2 * mu

            # x-equilibrium
            if not at_xL:
                _face_term(rx, _dof(i, j, 0, nx), _dof(im, j, 0, nx), a, gxm, hx**2)
            if not at_xR:
                _face_term(rx, _dof(i, j, 0, nx), _dof(ip, j, 0, nx), a, gxp, hx**2)
            if not at_yB:
                _face_term(rx, _dof(i, j, 0, nx), _dof(i, jm, 0, nx), mu, gym, hy**2)
            if not at_yT:
                _face_term(rx, _dof(i, j, 0, nx), _dof(i, jp, 0, nx), mu, gyp, hy**2)
            if cross_ok:
                cc = (lam + mu) * gc / (4.0 * hx * hy)
                add(rx, _dof(ip, jp, 1, nx),  cc)
                add(rx, _dof(im, jm, 1, nx),  cc)
                add(rx, _dof(ip, jm, 1, nx), -cc)
                add(rx, _dof(im, jp, 1, nx), -cc)

            # y-equilibrium
            if not at_xL:
                _face_term(ry, _dof(i, j, 1, nx), _dof(im, j, 1, nx), mu, gxm, hx**2)
            if not at_xR:
                _face_term(ry, _dof(i, j, 1, nx), _dof(ip, j, 1, nx), mu, gxp, hx**2)
            if not at_yB:
                _face_term(ry, _dof(i, j, 1, nx), _dof(i, jm, 1, nx), a, gym, hy**2)
            if not at_yT:
                _face_term(ry, _dof(i, j, 1, nx), _dof(i, jp, 1, nx), a, gyp, hy**2)
            if cross_ok:
                cc = (lam + mu) * gc / (4.0 * hx * hy)
                add(ry, _dof(ip, jp, 0, nx),  cc)
                add(ry, _dof(im, jm, 0, nx),  cc)
                add(ry, _dof(ip, jm, 0, nx), -cc)
                add(ry, _dof(im, jp, 0, nx), -cc)

    return sp.csr_matrix((vals_l, (rows_l, cols_l)), shape=(M, M))


def _solve_displacement_ytension(d: np.ndarray, u_applied: float,
                                 cfg: LaminateConfig
                                 ) -> tuple[np.ndarray, np.ndarray]:
    """Solve (u_x, u_y) under transverse (Mode-I peel) tension — BCs as in
    at2_simulator_2d._solve_displacement_2d (roller-grip Mode I):

      Bottom (j=0):        u_y = 0           (roller)
      Top    (j=ny-1):     u_y = u_applied
      One node (nx//2, 0): u_x = 0           (pins rigid-body x-translation
                                              away from the defect zone;
                                              rotation blocked by edge u_y)
      Left/right: traction-free (Poisson contraction allowed).
    """
    nx, ny = cfg.nx, cfg.ny
    lam, mu = cfg.lame()
    M = 2 * nx * ny

    g = (1.0 - d) ** 2 + cfg.kappa_res
    K = _build_full_elasticity_matrix(g, lam, mu, nx, ny, cfg.hx, cfg.hy).tolil()
    rhs = np.zeros(M)

    for i in range(nx):
        r = _dof(i, 0, 1, nx)
        K[r, :] = 0; K[r, r] = 1.0; rhs[r] = 0.0
        r = _dof(i, ny - 1, 1, nx)
        K[r, :] = 0; K[r, r] = 1.0; rhs[r] = u_applied

    r = _dof(nx // 2, 0, 0, nx)
    K[r, :] = 0; K[r, r] = 1.0; rhs[r] = 0.0

    u_flat = spla.spsolve(K.tocsr(), rhs)
    return u_flat[0::2].reshape(ny, nx), u_flat[1::2].reshape(ny, nx)


def _compute_H_field(u_x: np.ndarray, u_y: np.ndarray, cfg: LaminateConfig,
                     d: np.ndarray | None = None) -> np.ndarray:
    """Plane-strain elastic energy density ψ_e⁺ (adapted from
    at2_simulator_2d._compute_H_field).

    Crack-aware one-sided differences avoid measuring crack-face opening at
    nodes adjacent to fully cracked (d ≥ 0.99) nodes — applied to ε_yy across
    horizontal crack faces AND ε_xx across vertical crack faces (the original
    handled only horizontal Mode-I cracks).
    """
    hx, hy = cfg.hx, cfg.hy
    lam, mu = cfg.lame()

    eps_xx = np.gradient(u_x, hx, axis=1)
    eps_yy = np.gradient(u_y, hy, axis=0)
    eps_xy = 0.5 * (np.gradient(u_x, hy, axis=0) + np.gradient(u_y, hx, axis=1))

    if d is not None:
        cracked = d >= 0.99
        # ε_yy: vertical neighbours cracked → one-sided in y
        mask_lo = cracked[:-2, :] & ~cracked[1:-1, :]
        eps_yy[1:-1][mask_lo] = (u_y[2:][mask_lo] - u_y[1:-1][mask_lo]) / hy
        mask_hi = cracked[2:, :] & ~cracked[1:-1, :]
        eps_yy[1:-1][mask_hi] = (u_y[1:-1][mask_hi] - u_y[:-2][mask_hi]) / hy
        # ε_xx: horizontal neighbours cracked → one-sided in x
        mask_l = cracked[:, :-2] & ~cracked[:, 1:-1]
        eps_xx[:, 1:-1][mask_l] = (u_x[:, 2:][mask_l] - u_x[:, 1:-1][mask_l]) / hx
        mask_r = cracked[:, 2:] & ~cracked[:, 1:-1]
        eps_xx[:, 1:-1][mask_r] = (u_x[:, 1:-1][mask_r] - u_x[:, :-2][mask_r]) / hx

    psi = (0.5 * (lam + 2 * mu) * (eps_xx**2 + eps_yy**2)
           + lam * eps_xx * eps_yy
           + 2.0 * mu * eps_xy**2)
    return np.maximum(psi, 0.0)


def _reaction_force_y(u_x: np.ndarray, u_y: np.ndarray, d: np.ndarray,
                      cfg: LaminateConfig) -> float:
    """Integrate degraded σ_yy over the bottom edge (copied from
    at2_simulator_2d._reaction_force)."""
    lam, mu = cfg.lame()
    eps_xx = np.gradient(u_x, cfg.hx, axis=1)[0, :]
    eps_yy = np.gradient(u_y, cfg.hy, axis=0)[0, :]
    g = (1.0 - d[0, :]) ** 2 + cfg.kappa_res
    sig_yy = g * ((lam + 2 * mu) * eps_yy + lam * eps_xx)
    return float(np.sum(sig_yy) * cfg.hx)


# ─────────────────────────────────────────────────────────────────────────────
#  Defect seeding from FMPE posterior θ = (cx, cy, layer, log2size)
# ─────────────────────────────────────────────────────────────────────────────

def seed_defect(cx: float, cy: float, layer: float, log2size: float,
                cfg: LaminateConfig | None = None) -> np.ndarray:
    """Initial damage field d0 (ny, nx) from one FMPE posterior draw.

    FMPE θ format (fmpe_defect.py): cx, cy normalised [0,1] in-plane defect
    centroid; layer = through-thickness layer index (integer-ish float in
    [0, n_layers_data-1]); defect size = 2**log2size elements.

    Mapping onto the modelled x–y cross-section (x in-plane, y thickness):
      * x0 = cx · Lx  (cross-section is taken THROUGH the defect, so the
        second in-plane coordinate cy only fixes the section plane and does
        not appear in the 2D geometry — accepted for API compatibility).
      * y0 = centre of the ply the layer index falls in (linear map of
        layer/(n_layers_data-1) onto the n_plies bands).
      * Defect = fully damaged (d=1) ellipse, x-semi-axis
        = size_elements · defect_elem_frac · Lx (floor 1.5·hx), y-semi-axis
        clipped to 45% of the ply thickness so the seed stays intra-ply.

    Out-of-range posterior draws are clipped (layer to [0, n_layers-1],
    cx to [0,1], log2size to [-1, 3]).
    """
    cfg = cfg or LaminateConfig()
    _ = cy  # out-of-plane of the modelled section; see docstring
    cx = float(np.clip(cx, 0.0, 1.0))
    layer = float(np.clip(layer, 0.0, cfg.n_layers_data - 1))
    size_elems = 2.0 ** float(np.clip(log2size, -1.0, 3.0))

    ply_t = cfg.Ly / cfg.n_plies
    frac = layer / max(cfg.n_layers_data - 1, 1)
    ply_idx = frac * (cfg.n_plies - 1)
    y0 = (ply_idx + 0.5) * ply_t
    x0 = cx * cfg.Lx

    rx = max(size_elems * cfg.defect_elem_frac * cfg.Lx, 1.5 * cfg.hx)
    ry = min(rx, 0.45 * ply_t)
    ry = max(ry, 1.0 * cfg.hy)

    xs = np.linspace(0.0, cfg.Lx, cfg.nx)
    ys = np.linspace(0.0, cfg.Ly, cfg.ny)
    X, Y = np.meshgrid(xs, ys)
    inside = ((X - x0) / rx) ** 2 + ((Y - y0) / ry) ** 2 <= 1.0

    d0 = np.zeros((cfg.ny, cfg.nx))
    d0[inside] = 1.0
    return d0


def damage_area(d: np.ndarray, cfg: LaminateConfig,
                threshold: float = 0.5) -> float:
    """Crack area proxy: cell area where d exceeds threshold."""
    return float((d > threshold).sum()) * cfg.hx * cfg.hy


# ─────────────────────────────────────────────────────────────────────────────
#  Forward growth simulation (staggered AT2, quasi-static ramp)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_growth(d0: np.ndarray, load: float,
                    cfg: LaminateConfig | None = None,
                    n_steps: int | None = None) -> GrowthResult:
    """Quasi-static ramp to nominal transverse (peel) strain `load`
    (= u_y(top)/Ly, see module docstring on the loading proxy) with a seeded
    defect; staggered AT2 (Miehe 2010) with the laminate Gc/A maps.

    Growth criterion: relative increase of the d>0.5 crack area beyond
    cfg.growth_rel_threshold.  Rate-independent: for a monotonic ramp only
    the peak load matters (history field).
    """
    cfg = cfg or LaminateConfig()
    n_steps = n_steps or cfg.n_load_steps

    Gc_map, Mxx, Myy, Mxy, _ = build_laminate_maps(cfg)
    L_divM = build_var_coeff_divM(Mxx, Myy, Mxy, cfg.hx, cfg.hy)

    d = d0.copy()
    H = np.zeros_like(d)
    area0 = damage_area(d0, cfg)
    seed_mask = d0 >= 0.99
    peak_F = 0.0

    u_max = load * cfg.Ly
    for u_app in np.linspace(0.0, u_max, n_steps + 1)[1:]:
        ux, uy = _solve_displacement_ytension(d, u_app, cfg)
        psi = _compute_H_field(ux, uy, cfg, d=d)
        psi_masked = psi.copy()
        psi_masked[d >= 0.99] = 0.0     # fully damaged nodes cannot re-fracture
        H = np.maximum(H, psi_masked)
        d = _solve_damage(H, d, L_divM, Gc_map, cfg.ell)
        d[seed_mask] = 1.0              # re-enforce seed
        peak_F = max(peak_F, _reaction_force_y(ux, uy, d, cfg))

    area_final = damage_area(d, cfg)
    rel = (area_final - area0) / max(area0, 1e-12)
    return GrowthResult(grown=rel > cfg.growth_rel_threshold,
                        rel_growth=rel, area0=area0, area_final=area_final,
                        d0=d0, d_final=d, peak_F=peak_F)


# ─────────────────────────────────────────────────────────────────────────────
#  Stage-3 prognosis API
# ─────────────────────────────────────────────────────────────────────────────

def _growth_flags(posterior_samples: np.ndarray, load: float,
                  cfg: LaminateConfig, n_draws: int,
                  rng: np.random.Generator) -> np.ndarray:
    """Boolean growth flag per (sub-sampled) posterior draw at peak `load`."""
    samples = np.atleast_2d(np.asarray(posterior_samples, dtype=float))
    if samples.shape[1] != 4:
        raise ValueError("posterior_samples must be (N, 4): "
                         "(cx, cy, layer, log2size)")
    if len(samples) > n_draws:
        idx = rng.choice(len(samples), n_draws, replace=False)
        samples = samples[idx]
    flags = np.zeros(len(samples), dtype=bool)
    for k, (cx, cy, layer, l2s) in enumerate(samples):
        d0 = seed_defect(cx, cy, layer, l2s, cfg)
        flags[k] = simulate_growth(d0, load, cfg).grown
    return flags


def growth_probability(posterior_samples: np.ndarray, load: float,
                       cfg: LaminateConfig | None = None,
                       n_draws: int = 20,
                       rng: np.random.Generator | None = None) -> float:
    """P(crack growth | NDT measurement) at nominal transverse strain `load`.

    Monte-Carlo over the Stage-2 FMPE posterior: fraction of `n_draws`
    posterior draws θ=(cx,cy,layer,log2size) whose seeded crack grows beyond
    cfg.growth_rel_threshold in a quasi-static ramp to `load`.

    Cost: one FD phase-field simulation per draw (~seconds at nx≈60), i.e.
    minutes total for n_draws ≈ 20.
    """
    cfg = cfg or LaminateConfig()
    rng = rng or np.random.default_rng(0)
    flags = _growth_flags(posterior_samples, load, cfg, n_draws, rng)
    return float(flags.mean())


def flight_clearance(posterior_samples: np.ndarray,
                     load_profile: float | np.ndarray,
                     n_flights: int = 1,
                     alpha: float = 0.05,
                     beta: float = 0.3,
                     cfg: LaminateConfig | None = None,
                     n_draws: int = 20,
                     rng: np.random.Generator | None = None) -> dict:
    """Go/no-go decision for a reusable vehicle given the Stage-2 posterior.

    One flight = a sequence of quasi-static load blocks (`load_profile`,
    nominal strains).  The AT2 model is rate-independent with an
    irreversible history field, so within a flight only the PEAK block
    matters and identical repeated flights add no further growth; we
    therefore simulate one ramp to max(load_profile) per posterior draw and
    treat the n_flights extension statistically (per-flight growth events
    i.i.d. across flights).  This is an honest simplification: true fatigue
    phase-field (cyclic Gc degradation, Carrara et al. CMAME 2020) is future
    work, as is load-scatter between flights.

    Decision thresholds: alpha/beta are placeholders; in production they
    should come from expected-cost minimisation over the decision
    (cost(vehicle loss) >> cost(refurbishment) >> cost(inspection)), so the
    OK-threshold alpha must be far smaller than the REPAIR-threshold beta.

      p_growth_next <= alpha → "OK"
      p_growth_next <= beta  → "REPAIR"
      otherwise              → "RETIRE"

    Returns dict with keys:
      decision, p_growth_next, p_survive_n, expected_remaining_flights
      (expected_remaining_flights uses Laplace-smoothed p to stay finite when
       no draw grows: p~ = (k+1)/(n+2), E[N] = (1-p~)/p~ — a geometric-model
       point estimate, i.e. a lower-confidence proxy, not a fatigue life.)
    """
    cfg = cfg or LaminateConfig()
    rng = rng or np.random.default_rng(0)
    peak = float(np.max(np.atleast_1d(load_profile)))

    flags = _growth_flags(posterior_samples, peak, cfg, n_draws, rng)
    n = len(flags)
    k = int(flags.sum())
    p = k / n

    p_survive_n = (1.0 - p) ** n_flights
    p_smooth = (k + 1.0) / (n + 2.0)                 # Laplace smoothing
    expected_remaining = (1.0 - p_smooth) / p_smooth  # geometric model

    if p <= alpha:
        decision = "OK"
    elif p <= beta:
        decision = "REPAIR"
    else:
        decision = "RETIRE"

    return {
        "decision": decision,
        "p_growth_next": p,
        "p_survive_n": p_survive_n,
        "expected_remaining_flights": expected_remaining,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    cfg = LaminateConfig()
    print("=== CFRP laminate phase-field prognosis (Stage-3) ===")
    print(f"grid {cfg.nx}x{cfg.ny}, {cfg.n_plies} plies "
          f"{cfg.ply_angles_deg}, ell={cfg.ell:.4f}")

    rng = np.random.default_rng(7)
    # mock FMPE posterior: defect mid-laminate, 2x2-ish
    post = np.column_stack([
        rng.normal(0.5, 0.05, 64),     # cx
        rng.normal(0.5, 0.05, 64),     # cy
        rng.normal(9.0, 1.0, 64),      # layer
        rng.normal(1.0, 0.3, 64),      # log2size
    ])

    # calibrated load scale (this grid/Gc/ell): no growth <= 0.06,
    # transition ~0.09-0.11, severe >= 0.14
    for load in (0.06, 0.10, 0.14):
        t0 = time.perf_counter()
        p = growth_probability(post, load, cfg, n_draws=8, rng=rng)
        dt = time.perf_counter() - t0
        print(f"load={load:.3f}  P(growth)={p:.2f}   [{dt:.1f}s]")

    out = flight_clearance(post, load_profile=[0.5 * 0.10, 0.10],
                           n_flights=10, cfg=cfg, n_draws=8, rng=rng)
    print("clearance:", out)
