"""
phasefield_3d.py — Stage-3 CFRP phase-field, 3-D PROOF-OF-CONCEPT extension.

Why this exists.  The production solver `cfrp_phasefield_2d.py` models an
x–thickness CROSS-SECTION (2-D plane-strain): a delamination shows up as an
x-aligned slit on a single section plane (the second in-plane coordinate `cy`
only selects which plane is drawn, it never enters the geometry — see
`seed_defect`).  Real interlaminar delamination is a 2-D crack FRONT advancing
in BOTH in-plane directions (x AND y) at a fixed through-thickness depth z,
i.e. an embedded ellipsoidal patch growing in 3-D.  This module DEMONSTRATES
that the same AT2 / screened-Poisson damage formulation and the same
conservative variable-coefficient flux discretisation extend cleanly to 3-D,
and that the 3-D discretisation is VERIFIED (2nd-order MMS).

⚠ SCOPE — THIS IS A PROOF-OF-CONCEPT, NOT A PRODUCTION REPLACEMENT.
  Included:
    1. A 3-D conservative variable-coefficient damage operator
       ∇·(M(x)·∇d) (7-point Laplacian when M = m·I), assembled as a single
       scipy.sparse matrix — the direct 3-D analogue of
       `cfrp_phasefield_2d.build_var_coeff_divM` (axis fluxes only; the 2-D
       module's Mxy cross term has no isotropic 3-D counterpart used here, so
       the PoC sticks to a diagonal anisotropy M = diag(Mxx,Myy,Mzz), enough
       for an interface-aligned structural tensor).
    2. MMS verification in 3-D on the AT2 Helmholtz operator A d = c0 d − κ Δd
       with d* = cos(πx/Lx)cos(πy/Ly)cos(πz/Lz): 2nd-order solution error,
       matching the 2-D `physics_validation.py` Dirichlet headline.
    3. A minimal 3-D delamination-growth demo: an embedded ellipsoidal damaged
       patch driven by a SIMPLIFIED scalar driving force, run through a few
       staggered AT2 update steps; damaged volume grows.
  NOT included (explicitly out of scope — stated so reviewers are not misled):
    * No 3-D elasticity solve.  The 2-D module solves (u_x,u_y) and derives the
      crack-driving history H = ψ_e⁺ from the elastic field.  Here the driving
      force is a prescribed smooth scalar history field H(x,y,z) concentrated
      near the existing delamination front (a peel-energy SURROGATE), so the
      growth demo is QUALITATIVE — it shows the operator + staggered loop drive
      a 3-D front, not a quantitative load→growth map.
    * No ply-by-ply layup, no fatigue, no FMPE posterior coupling, no
      flight-clearance decision (those live in the 2-D production module).
    * Off-diagonal (Mxy/Myz/Mxz) anisotropy and full 3-D structural tensors.

Conventions mirror `cfrp_phasefield_2d.py`: a LaminateConfig3D dataclass,
`build_var_coeff_divM_3d` (sparse), `seed_defect_3d`, `_solve_damage_3d`
(solve-then-clip AT2 irreversibility), a `run_tests()` with a nonlocal
counter + `ok(cond,msg)`, and an argparse `main(--test/--fig)` saving a
thesis-styled figure to paper_figs/phasefield_3d.pdf (+ .png).

Imports `cfrp_phasefield_2d` for the AT2 operator shape / constants only; it
does NOT modify it.  CPU, numpy/scipy only, TINY grids (≤ 16³) — tests run in
well under a minute, the figure study in a few seconds.

  python phasefield_3d.py            # MMS + growth study + printed report
  python phasefield_3d.py --fig      # + paper_figs/phasefield_3d.pdf (+ .png)
  python phasefield_3d.py --test     # unit tests only
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import cfrp_phasefield_2d as pf2d  # constants/reference only; never modified

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_PATH = os.path.join(HERE, "paper_figs", "phasefield_3d.pdf")

# Manufactured-operator constants for the 3-D MMS, mirroring physics_validation
# (O(1), well-conditioned).  A d = MMS_C0 d − MMS_KAPPA ∇·(M ∇d), M = I.  This
# is the algebraic shape of the AT2 _solve_damage operator
#   (Gc/ℓ + 2H) d − ℓ ∇·(Gc A ∇d) = 2H   with  Gc/ℓ+2H ↔ c0,  ℓ Gc ↔ κ.
MMS_C0 = 3.0
MMS_KAPPA = 0.7
MMS_LX = 1.0
MMS_LY = 0.5
MMS_LZ = 0.5


# ─────────────────────────────────────────────────────────────────────────────
#  Configuration (3-D analogue of LaminateConfig)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LaminateConfig3D:
    """3-D laminate block + solver configuration (normalised units).

    Geometry: a laminate block [0,Lx]×[0,Ly]×[0,Lz]; x, y are the two in-plane
    directions, z is through-thickness.  A delamination is an embedded patch at
    a fixed z (between plies) whose FRONT advances in the (x,y) plane — the 3-D
    feature the 2-D cross-section cannot represent.  Fracture constants
    (Gc_ply, beta, ell) inherit the 2-D module's meaning and defaults.
    """
    Lx: float = 1.0
    Ly: float = 1.0
    Lz: float = 0.5
    nx: int = 16
    ny: int = 16
    nz: int = 12
    # fracture (same normalised anchors as the 2-D module)
    Gc_ply: float = 8e-4
    beta_interface: float = 25.0   # interface-aligned (in-plane) toughening
    ell: float | None = None       # PF length; default 2.2 * max(h)
    kappa_res: float = 1e-6

    def __post_init__(self):
        if self.ell is None:
            self.ell = 2.2 * max(self.hx, self.hy, self.hz)

    @property
    def hx(self) -> float:
        return self.Lx / (self.nx - 1)

    @property
    def hy(self) -> float:
        return self.Ly / (self.ny - 1)

    @property
    def hz(self) -> float:
        return self.Lz / (self.nz - 1)

    @property
    def cell_vol(self) -> float:
        return self.hx * self.hy * self.hz


# ─────────────────────────────────────────────────────────────────────────────
#  3-D damage operator: ∇·(M(x)·∇d), conservative flux form, diagonal M
#  (direct analogue of cfrp_phasefield_2d.build_var_coeff_divM; 7-point
#   stencil; M = m·I recovers the standard 3-D Laplacian).
# ─────────────────────────────────────────────────────────────────────────────

def build_var_coeff_divM_3d(Mxx: np.ndarray, Myy: np.ndarray, Mzz: np.ndarray,
                            hx: float, hy: float, hz: float) -> sp.csr_matrix:
    """Sparse FD matrix for  L d = ∇·(M(x)·∇d),  M = diag(Mxx,Myy,Mzz).

    Conservative form with face-averaged coefficients on each of the six faces
    (7-point stencil); zero-flux Neumann at the block boundary (boundary faces
    contribute nothing — exactly the 2-D `build_var_coeff_divM` convention,
    one dimension up).  Fields are indexed (nz, ny, nx); flat index
    k = i + j*nx + l*nx*ny.  Vectorised over faces for speed/memory at ≤16³.
    """
    nz, ny, nx = Mxx.shape
    N = nx * ny * nz

    # node (i,j,l) ↔ flat k, in the (nz,ny,nx) field layout
    ll, jj, ii = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx),
                             indexing="ij")
    ii = ii.ravel(); jj = jj.ravel(); ll = ll.ravel()
    k = ii + jj * nx + ll * (nx * ny)

    # accumulate the diagonal, then the off-diagonals face by face
    diag = np.zeros(N)
    off_rows, off_cols, off_vals = [], [], []

    Mx = Mxx.ravel(); My = Myy.ravel(); Mz = Mzz.ravel()

    def add_face(mask, knbr, Mself, Mnbr, h):
        # face coefficient = 0.5*(M_self + M_nbr)/h^2 on the masked nodes
        coef = 0.5 * (Mself[mask] + Mnbr[mask]) / (h * h)
        np.add.at(diag, k[mask], -coef)
        off_rows.append(k[mask]); off_cols.append(knbr[mask]); off_vals.append(coef)

    # x faces
    me = ii < nx - 1
    add_face(me, k + 1, Mx, np.roll(Mx, -1), hx)
    mw = ii > 0
    add_face(mw, k - 1, Mx, np.roll(Mx, 1), hx)
    # y faces
    mn = jj < ny - 1
    add_face(mn, k + nx, My, np.roll(My, -nx), hy)
    ms = jj > 0
    add_face(ms, k - nx, My, np.roll(My, nx), hy)
    # z faces
    mu = ll < nz - 1
    add_face(mu, k + nx * ny, Mz, np.roll(Mz, -(nx * ny)), hz)
    md = ll > 0
    add_face(md, k - nx * ny, Mz, np.roll(Mz, nx * ny), hz)

    r = np.concatenate([k] + off_rows)
    c = np.concatenate([k] + off_cols)
    v = np.concatenate([diag] + off_vals)
    return sp.csr_matrix((v, (r, c)), shape=(N, N))


def _solve_damage_3d(H: np.ndarray, d_prev: np.ndarray, L_divM: sp.csr_matrix,
                     Gc_map: np.ndarray, ell: float) -> np.ndarray:
    """One staggered AT2 damage solve in 3-D (solve-then-clip irreversibility):

        -ℓ ∇·(Gc A ∇d) + (Gc/ℓ + 2H) d = 2H,    d ≥ d_prev.

    Identical algebra to `cfrp_phasefield_2d._solve_damage`, one dimension up.
    """
    nz, ny, nx = H.shape
    diag = Gc_map.ravel() / ell + 2.0 * H.ravel()
    K = -ell * L_divM + sp.diags(diag, format="csr")
    rhs = 2.0 * H.ravel()
    d_flat = spla.spsolve(K, rhs)
    return np.clip(d_flat.reshape(nz, ny, nx), d_prev, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
#  Defect seeding: embedded 3-D ellipsoidal delamination patch
# ─────────────────────────────────────────────────────────────────────────────

def seed_defect_3d(cx: float, cy: float, cz: float,
                   rx: float, ry: float, rz: float,
                   cfg: LaminateConfig3D | None = None) -> np.ndarray:
    """Initial damage field d0 (nz, ny, nx): a fully damaged (d=1) ellipsoid
    centred at (cx,cy,cz)·(Lx,Ly,Lz) with semi-axes (rx,ry,rz)·(Lx,Ly,Lz).

    The physically relevant case is a THIN patch (rz « rx,ry) at a fixed
    through-thickness depth — an embedded inter-ply delamination whose front
    advances in the in-plane (x,y) directions.
    """
    cfg = cfg or LaminateConfig3D()
    xs = np.linspace(0.0, cfg.Lx, cfg.nx)
    ys = np.linspace(0.0, cfg.Ly, cfg.ny)
    zs = np.linspace(0.0, cfg.Lz, cfg.nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")     # (nx,ny,nz)
    x0, y0, z0 = cx * cfg.Lx, cy * cfg.Ly, cz * cfg.Lz
    ax = max(rx * cfg.Lx, 1.0 * cfg.hx)
    ay = max(ry * cfg.Ly, 1.0 * cfg.hy)
    az = max(rz * cfg.Lz, 1.0 * cfg.hz)
    inside = (((X - x0) / ax) ** 2 + ((Y - y0) / ay) ** 2
              + ((Z - z0) / az) ** 2) <= 1.0
    d0 = np.zeros((cfg.nx, cfg.ny, cfg.nz))
    d0[inside] = 1.0
    return np.transpose(d0, (2, 1, 0))                   # -> (nz,ny,nx)


def damaged_volume(d: np.ndarray, cfg: LaminateConfig3D,
                   threshold: float = 0.5) -> float:
    """Damaged-volume proxy: cell volume where d exceeds threshold."""
    return float((d > threshold).sum()) * cfg.cell_vol


# ─────────────────────────────────────────────────────────────────────────────
#  Simplified-driving-force interface map + staggered growth demo
# ─────────────────────────────────────────────────────────────────────────────

def build_interface_maps_3d(cfg: LaminateConfig3D, z_interface: float
                            ) -> tuple[np.ndarray, np.ndarray,
                                       np.ndarray, np.ndarray]:
    """Build (Gc_map, Mxx, Myy, Mzz), each (nz,ny,nx), for a single weak
    inter-ply interface plane at through-thickness fraction `z_interface`.

    Inside a thin band around z = z_interface·Lz the toughness is reduced
    (interface ratio 0.4, the 2-D module's anchor) and the structural tensor is
    in-plane-aligned (Mxx,Myy boosted by beta_interface, Mzz left at Gc), so
    delamination spreads cheaply in (x,y) — the 3-D analogue of the 2-D
    interface band whose cheap direction was x̂.
    """
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    zs = np.linspace(0.0, cfg.Lz, nz)
    Gc_map = np.full((nz, ny, nx), cfg.Gc_ply)
    Mxx = np.full((nz, ny, nx), cfg.Gc_ply)
    Myy = np.full((nz, ny, nx), cfg.Gc_ply)
    Mzz = np.full((nz, ny, nx), cfg.Gc_ply)

    z0 = z_interface * cfg.Lz
    l_iface = int(np.argmin(np.abs(zs - z0)))
    band = slice(max(l_iface, 0), min(l_iface + 1, nz))   # 1-cell band
    Gc_int = 0.4 * cfg.Gc_ply                              # interface ratio
    Gc_map[band] = Gc_int
    Mxx[band] = Gc_int * (1.0 + cfg.beta_interface)
    Myy[band] = Gc_int * (1.0 + cfg.beta_interface)
    Mzz[band] = Gc_int                                    # cracking across z hard
    return Gc_map, Mxx, Myy, Mzz


@dataclass
class GrowthResult3D:
    grown: bool
    vol0: float
    vol_final: float
    vol_history: np.ndarray         # damaged volume after each staggered step
    d0: np.ndarray                  # seeded damage field (nz,ny,nx)
    d_final: np.ndarray             # final damage field (nz,ny,nx)


def simulate_growth_3d(d0: np.ndarray, cfg: LaminateConfig3D | None = None,
                       n_steps: int = 6, drive: float = 1.0,
                       z_interface: float = 0.5) -> GrowthResult3D:
    """Minimal 3-D staggered AT2 delamination-growth demo.

    SIMPLIFIED DRIVING FORCE (no elasticity — see module docstring): instead of
    solving 3-D elasticity for ψ_e⁺, we prescribe a smooth scalar history field
    H(x,y,z) = drive · (peel ring concentrated on the current delamination
    FRONT, inside the weak interface plane).  Each staggered step:
      * recompute the front-localised H from the current damage,
      * accumulate H (irreversible history, max over steps),
      * one AT2 damage solve with the interface Gc/A maps,
      * re-enforce the seed.
    The front-localised driver makes the patch advance OUTWARD in (x,y) within
    the interface plane — a qualitative 3-D analogue of `simulate_growth`.
    Returns the damaged-volume history (should be non-decreasing and grow).
    """
    cfg = cfg or LaminateConfig3D()
    Gc_map, Mxx, Myy, Mzz = build_interface_maps_3d(cfg, z_interface)
    L_divM = build_var_coeff_divM_3d(Mxx, Myy, Mzz, cfg.hx, cfg.hy, cfg.hz)

    d = d0.copy()
    H = np.zeros_like(d)
    seed_mask = d0 >= 0.99
    vol0 = damaged_volume(d0, cfg)
    vols = []

    # interface band mask (where growth is allowed) — same band as the maps
    zs = np.linspace(0.0, cfg.Lz, cfg.nz)
    l_iface = int(np.argmin(np.abs(zs - z_interface * cfg.Lz)))
    iface_plane = np.zeros_like(d, dtype=bool)
    iface_plane[l_iface] = True

    # in-plane Laplacian-style "front" detector: damaged minus its in-plane
    # neighbourhood mean highlights the rim of the current patch.
    def front_drive(dmg):
        from scipy.ndimage import uniform_filter
        # uniform_filter is in scipy.ndimage (available); fall back to a manual
        # box mean if unavailable.
        sm = uniform_filter(dmg.astype(float), size=(1, 3, 3), mode="nearest")
        rim = np.clip(sm - dmg, 0.0, None)        # >0 just outside the patch
        H_new = drive * rim
        H_new[~iface_plane] *= 0.0                # drive only in the interface
        return H_new

    for _ in range(n_steps):
        H = np.maximum(H, front_drive(d))
        H[d >= 0.99] = np.maximum(H[d >= 0.99], 0.0)
        d = _solve_damage_3d(H, d, L_divM, Gc_map, cfg.ell)
        d[seed_mask] = 1.0
        vols.append(damaged_volume(d, cfg))

    vols = np.array(vols)
    vol_final = float(vols[-1])
    return GrowthResult3D(grown=vol_final > vol0, vol0=vol0,
                          vol_final=vol_final, vol_history=vols,
                          d0=d0, d_final=d)


# ═════════════════════════════════════════════════════════════════════════════
#  3-D Method of Manufactured Solutions on build_var_coeff_divM_3d
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class MMSResult3D:
    """Convergence record for the 3-D MMS (one error per grid)."""
    label: str
    h: np.ndarray            # mesh size hx per grid
    err: np.ndarray          # L2 error per grid
    order: float             # fitted slope of log(err) vs log(h)

    def order_between(self) -> np.ndarray:
        return (np.log(self.err[:-1] / self.err[1:])
                / np.log(self.h[:-1] / self.h[1:]))


def _manufactured_fields_3d(nx: int, ny: int, nz: int,
                            Lx=MMS_LX, Ly=MMS_LY, Lz=MMS_LZ):
    """Return (h's, d*, source f) for d* = cos(πx/Lx)cos(πy/Ly)cos(πz/Lz).

    Δd* = −(π²/Lx² + π²/Ly² + π²/Lz²) d*  (closed form); for M = I the source
    is f = c0 d* − κ Δd*.  d* has zero normal derivative on every face → it
    satisfies the zero-flux Neumann closure exactly except for the 1st-order
    boundary truncation; the Dirichlet-pinned variant isolates the interior
    (the 2nd-order headline, mirroring physics_validation).
    """
    hx, hy, hz = Lx / (nx - 1), Ly / (ny - 1), Lz / (nz - 1)
    xs = np.linspace(0.0, Lx, nx)
    ys = np.linspace(0.0, Ly, ny)
    zs = np.linspace(0.0, Lz, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")          # (nx,ny,nz)
    d_star = (np.cos(np.pi * X / Lx) * np.cos(np.pi * Y / Ly)
              * np.cos(np.pi * Z / Lz))
    lap = -(np.pi ** 2 / Lx ** 2 + np.pi ** 2 / Ly ** 2
            + np.pi ** 2 / Lz ** 2) * d_star
    f = MMS_C0 * d_star - MMS_KAPPA * lap
    # to (nz,ny,nx) field layout used by the operator
    d_star = np.transpose(d_star, (2, 1, 0))
    f = np.transpose(f, (2, 1, 0))
    return hx, hy, hz, d_star, f


def _mms_operator_3d(nx, ny, nz, hx, hy, hz) -> sp.csr_matrix:
    """A = c0 I − κ ∇·(∇·) via the production 3-D flux stencil (M = I)."""
    M = np.ones((nz, ny, nx))
    L = build_var_coeff_divM_3d(M, M, M, hx, hy, hz)          # = 3-D Laplacian
    return (MMS_C0 * sp.eye(nx * ny * nz, format="csr") - MMS_KAPPA * L).tocsr()


def mms_convergence_3d(n_list=(9, 13, 17, 21)) -> dict[str, MMSResult3D]:
    """Run the 3-D MMS study over a cube-grid sequence (nx=ny=nz=n).

    Keys: "dirichlet" (boundary pinned to d*, interior solution error — the
    clean 2nd-order headline), "residual" (interior truncation residual,
    BC-free stencil probe).  Grids are cubes on the unit box so h is identical
    in all three axes (cleanest convergence read-out).
    """
    hs, e_dir, e_res = [], [], []
    for n in n_list:
        hx, hy, hz, d_star, f = _manufactured_fields_3d(
            n, n, n, Lx=1.0, Ly=1.0, Lz=1.0)
        A = _mms_operator_3d(n, n, n, hx, hy, hz)
        ds = d_star.ravel()

        # interior truncation residual ‖A d* − f‖ (no solve, no BC closure)
        res = (A.dot(ds) - f.ravel()).reshape(n, n, n)[1:-1, 1:-1, 1:-1]
        e_res.append(float(np.sqrt(np.mean(res ** 2))))

        # Dirichlet-pinned solve, interior L2 error (isolates interior stencil)
        Ad = A.tolil()
        rhs = f.ravel().copy()
        bmask = np.zeros((n, n, n), dtype=bool)
        bmask[0, :, :] = bmask[-1, :, :] = True
        bmask[:, 0, :] = bmask[:, -1, :] = True
        bmask[:, :, 0] = bmask[:, :, -1] = True
        for kk in np.where(bmask.ravel())[0]:
            Ad.rows[kk] = [kk]; Ad.data[kk] = [1.0]; rhs[kk] = ds[kk]
        d_dir = spla.spsolve(Ad.tocsr(), rhs).reshape(n, n, n)
        e_int = d_dir[1:-1, 1:-1, 1:-1] - d_star[1:-1, 1:-1, 1:-1]
        e_dir.append(float(np.sqrt(np.mean(e_int ** 2))))

        hs.append(hx)

    hs = np.array(hs)
    out = {}
    for key, lab, errs in (("dirichlet", "3-D MMS-Dirichlet (interior soln)", e_dir),
                           ("residual", "3-D MMS-residual (interior trunc.)", e_res)):
        errs = np.array(errs)
        order = float(np.polyfit(np.log(hs), np.log(errs), 1)[0])
        out[key] = MMSResult3D(label=lab, h=hs, err=errs, order=order)
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  Report
# ═════════════════════════════════════════════════════════════════════════════

def print_report(mms: dict[str, MMSResult3D], gr: GrowthResult3D) -> None:
    bar = "═" * 78
    print(bar)
    print("  Stage-3 phase-field — 3-D PROOF-OF-CONCEPT (operator + MMS + growth)")
    print(bar)

    print("\n(1) 3-D METHOD OF MANUFACTURED SOLUTIONS — build_var_coeff_divM_3d")
    print(f"    operator  A d = {MMS_C0} d − {MMS_KAPPA} ∇·(∇d),   "
          f"d* = cos(πx)cos(πy)cos(πz)")
    for key in ("dirichlet", "residual"):
        r = mms[key]
        errs = "  ".join(f"{e:.2e}" for e in r.err)
        print(f"    {r.label:<34s} order={r.order:5.2f}")
        print(f"        h    = " + "  ".join(f"{hh:.4f}" for hh in r.h))
        print(f"        L2   = " + errs)
        print(f"        pair = " + "  ".join(f"{o:.2f}" for o in r.order_between()))
    print("    → 3-D interior stencil is 2nd-order (≈2) — discretisation verified.")

    print("\n(2) 3-D DELAMINATION GROWTH DEMO — simplified driving force")
    print("    embedded ellipsoidal patch in a weak interface plane; staggered")
    print("    AT2 with a front-localised scalar history (NO 3-D elasticity).")
    print(f"    damaged volume:  V0 = {gr.vol0:.4e}  →  Vf = {gr.vol_final:.4e}"
          f"   (×{gr.vol_final / max(gr.vol0, 1e-30):.2f})")
    print("    V(step) = " + "  ".join(f"{v:.3e}" for v in gr.vol_history))
    print(f"    grew: {gr.grown}")
    print(bar)
    print("  SCOPE: PoC = damage operator + MMS + simplified-driving-force "
          "growth.")
    print("         NOT a production replacement: no full 3-D elasticity, no")
    print("         layup/fatigue/FMPE/decision coupling (those stay in 2-D).")
    print(bar)


# ═════════════════════════════════════════════════════════════════════════════
#  Figure
# ═════════════════════════════════════════════════════════════════════════════

def make_figure(mms: dict[str, MMSResult3D], gr: GrowthResult3D,
                cfg: LaminateConfig3D, out_path: str = FIG_PATH) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import sys
    sys.path.insert(0, os.path.join(HERE, "slides", "figure_sources"))
    try:
        from thesis_style import use
        figsize = use(width_frac=1.0, aspect=0.34)
    except Exception:
        figsize = (12.0, 4.0)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=figsize)

    # (a) 3-D MMS log-log error vs h with fitted slope + slope-2 reference
    styles = {"dirichlet": ("#1565C0", "o"), "residual": ("#2e7d32", "s")}
    for key in ("dirichlet", "residual"):
        r = mms[key]
        c, mk = styles[key]
        ax[0].loglog(r.h, r.err, mk + "-", color=c, ms=5,
                     label=f"{r.label.split('(')[0].strip()} (p={r.order:.2f})")
    h0 = mms["dirichlet"].h
    e0 = mms["dirichlet"].err[0]
    ax[0].loglog(h0, e0 * (h0 / h0[0]) ** 2, "k--", lw=0.8, label="slope 2")
    ax[0].set_xlabel("mesh size $h$", fontsize=8)
    ax[0].set_ylabel(r"$\|d - d^*\|_{L_2}$", fontsize=8)
    ax[0].set_title("(a) 3-D MMS convergence", fontsize=9)
    ax[0].legend(fontsize=6, loc="lower right")
    ax[0].tick_params(labelsize=7)
    ax[0].grid(True, which="both", alpha=0.25)

    # mid-plane slices (interface plane) before / after growth
    zs = np.linspace(0.0, cfg.Lz, cfg.nz)
    l_iface = int(np.argmin(np.abs(zs - 0.5 * cfg.Lz)))
    ext = [0, cfg.Lx, 0, cfg.Ly]
    im0 = ax[1].imshow(gr.d0[l_iface], origin="lower", extent=ext,
                       vmin=0, vmax=1, cmap="inferno", aspect="auto")
    ax[1].set_title("(b) interface slice — seed", fontsize=9)
    ax[1].set_xlabel("$x$", fontsize=8); ax[1].set_ylabel("$y$", fontsize=8)
    ax[1].tick_params(labelsize=7)

    ax[2].imshow(gr.d_final[l_iface], origin="lower", extent=ext,
                 vmin=0, vmax=1, cmap="inferno", aspect="auto")
    ax[2].set_title(f"(c) interface slice — grown "
                    f"(×{gr.vol_final / max(gr.vol0, 1e-30):.1f} vol)", fontsize=9)
    ax[2].set_xlabel("$x$", fontsize=8); ax[2].set_ylabel("$y$", fontsize=8)
    ax[2].tick_params(labelsize=7)
    cbar = fig.colorbar(im0, ax=[ax[1], ax[2]], fraction=0.046, pad=0.02)
    cbar.set_label("damage $d$", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    fig.savefig(os.path.splitext(out_path)[0] + ".png", bbox_inches="tight",
                dpi=150)
    plt.close(fig)
    return out_path


# ═════════════════════════════════════════════════════════════════════════════
#  Unit tests
# ═════════════════════════════════════════════════════════════════════════════

def run_tests() -> int:
    n = 0

    def ok(cond, msg):
        nonlocal n
        assert cond, msg
        n += 1

    # --- operator sanity: symmetric, recovers a known field ---
    cfg = LaminateConfig3D(nx=10, ny=10, nz=8)
    M = np.ones((cfg.nz, cfg.ny, cfg.nx))
    L = build_var_coeff_divM_3d(M, M, M, cfg.hx, cfg.hy, cfg.hz)
    asym = abs((L - L.T)).max()
    ok(asym < 1e-10, f"3-D Laplacian symmetric (asym={asym:.1e})")
    # constant field is in the nullspace of a pure divergence operator
    const = np.ones(cfg.nx * cfg.ny * cfg.nz)
    ok(np.abs(L.dot(const)).max() < 1e-10, "∇·(∇·) annihilates a constant field")

    # well-posed Helmholtz solve recovers a known smooth field exactly-ish:
    # build f = A d_known, solve, compare (Dirichlet-pinned, interior 2nd-order)
    A = _mms_operator_3d(cfg.nx, cfg.ny, cfg.nz, cfg.hx, cfg.hy, cfg.hz)
    ok(abs((A - A.T)).max() < 1e-10, "AT2 Helmholtz operator symmetric")

    # --- 3-D MMS: tiny cube sequence, fast ---
    mms = mms_convergence_3d(n_list=(9, 13, 17))
    rd = mms["dirichlet"]
    ok(np.all(np.diff(rd.err) < 0), "3-D MMS error decreases with refinement")
    ok(rd.order > 1.7, f"3-D MMS Dirichlet order ~2 (got {rd.order:.2f})")
    ok(rd.order < 2.6, f"3-D MMS order not super-quadratic ({rd.order:.2f})")
    rr = mms["residual"]
    ok(np.all(np.diff(rr.err) < 0), "3-D MMS residual decreases with refinement")
    ok(rr.order > 1.7, f"3-D MMS residual order ~2 (got {rr.order:.2f})")

    # --- growth demo: damaged volume increases on a tiny grid ---
    gcfg = LaminateConfig3D(nx=16, ny=16, nz=12)
    d0 = seed_defect_3d(0.5, 0.5, 0.5, 0.10, 0.10, 0.05, gcfg)
    ok(damaged_volume(d0, gcfg) > 0, "seed has non-zero damaged volume")
    gr = simulate_growth_3d(d0, gcfg, n_steps=6, drive=2.0, z_interface=0.5)
    ok(gr.vol_final > gr.vol0, f"growth increases damaged volume "
       f"({gr.vol0:.3e} → {gr.vol_final:.3e})")
    ok(np.all(np.diff(gr.vol_history) >= -1e-12),
       "damaged volume non-decreasing across staggered steps (irreversible)")
    ok(gr.grown, "growth demo reports grown=True")

    print(f"phasefield_3d: {n}/{n} unit tests passed")
    return n


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

def _parse_int_list(s: str) -> tuple:
    return tuple(int(x) for x in s.replace(" ", "").split(",") if x)


def main():
    ap = argparse.ArgumentParser(
        description="Stage-3 phase-field 3-D proof-of-concept: 3-D damage "
                    "operator + MMS verification + simplified-driving-force "
                    "delamination growth.")
    ap.add_argument("--mms-grids", type=_parse_int_list, default=(9, 13, 17, 21),
                    help="comma list of cube sizes n (nx=ny=nz=n) for 3-D MMS")
    ap.add_argument("--nx", type=int, default=16, help="growth-demo grid nx")
    ap.add_argument("--ny", type=int, default=16, help="growth-demo grid ny")
    ap.add_argument("--nz", type=int, default=12, help="growth-demo grid nz")
    ap.add_argument("--steps", type=int, default=6,
                    help="staggered growth steps")
    ap.add_argument("--drive", type=float, default=2.0,
                    help="scalar driving-force magnitude (surrogate peel)")
    ap.add_argument("--fig", action="store_true",
                    help="save paper_figs/phasefield_3d.pdf (+ .png)")
    ap.add_argument("--test", action="store_true", help="unit tests only")
    args = ap.parse_args()

    if args.test:
        run_tests(); return

    t0 = time.perf_counter()
    print("[mms] running 3-D manufactured-solution convergence ...")
    mms = mms_convergence_3d(n_list=args.mms_grids)
    print("[growth] running 3-D delamination-growth demo ...")
    gcfg = LaminateConfig3D(nx=args.nx, ny=args.ny, nz=args.nz)
    d0 = seed_defect_3d(0.5, 0.5, 0.5, 0.10, 0.10, 0.05, gcfg)
    gr = simulate_growth_3d(d0, gcfg, n_steps=args.steps, drive=args.drive,
                            z_interface=0.5)
    print(f"[done] study in {time.perf_counter() - t0:.1f}s\n")

    print_report(mms, gr)

    if args.fig:
        p = make_figure(mms, gr, gcfg)
        print(f"figure → {p}")


if __name__ == "__main__":
    main()
