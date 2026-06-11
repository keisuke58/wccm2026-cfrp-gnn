"""
physics_validation.py — Stage-3 FD phase-field solver verification.

Makes the finite-difference (FD) phase-field solver of `cfrp_phasefield_2d.py`
defensible to reviewers via two standard code-verification exercises:

  (1) METHOD OF MANUFACTURED SOLUTIONS (MMS) on the damage / screened-Poisson
      Helmholtz operator that the staggered AT2 loop inverts every step,
          A d = c0 · d − κ · ∇·(M ∇d),
      built from the SAME `build_var_coeff_divM` stencil the solver uses (so we
      verify the production operator, not a re-implementation).  We manufacture
      a smooth field d*(x,y) = cos(πx/Lx) cos(πy/Ly) — which has zero normal
      derivative on every edge and therefore exactly satisfies the zero-flux
      Neumann BCs the flux operator imposes — form the analytic source f = A d*,
      solve A d = f on a sequence of refined grids, and measure the L2 error
      ‖d − d*‖ vs mesh size h.

      HONEST SCOPE.  `build_var_coeff_divM` is a 2nd-order centred stencil in
      the INTERIOR but its zero-flux Neumann closure is only 1st-order at the
      boundary.  A global Neumann solve therefore shows ≈1st-order solution
      error (boundary pollution spreads through the elliptic inverse), even
      though the interior stencil is genuinely 2nd-order.  We report this
      honestly with three complementary measures:
        * MMS-Dirichlet : boundary nodes pinned to d* — isolates the interior
          stencil; solution L2 error converges at order ≈2 (the headline).
        * MMS-residual  : interior truncation residual ‖(A d* − f)|_int‖ —
          a BC-free probe of the stencil; converges at order ≈2.
        * MMS-Neumann   : the production BC, global solution error — order ≈1,
          reported so the boundary limitation is on the record.

  (2) MESH-CONVERGENCE study on a physical QoI of the real solver.  We run the
      actual `simulate_growth` for a FIXED seeded defect at increasing nx/ny
      and record the peak reaction force `peak_F` (a smooth, mesh-robust QoI)
      and the final crack area.  The PF length ℓ is FROZEN at its absolute
      default value across all grids, so the mesh-refinement error is separated
      from the AT2 ℓ-modelling choice (ℓ^(-1/2) strength scaling, memo §6): we
      are measuring "does the discretisation converge for fixed physics", not
      "is the absolute load quantitative" (it is not — see the module docstring
      of cfrp_phasefield_2d).  Richardson extrapolation on peak_F gives an
      asymptotic value and a per-mesh relative error; the default nx=60 grid is
      then placed on that error curve.

      Caveat reported in the output: the crack AREA / rel_growth QoI is NOT
      monotone under refinement at fixed absolute ℓ (a coarse grid barely
      resolves the ℓ-band and the seed-in-elements changes), so only peak_F is
      claimed mesh-convergent; area is shown as a trend with its own residual.

CPU, numpy/scipy only.  Tests use tiny grids and run in well under a minute;
the full study (5 grids up to nx=80) runs in a few seconds.  Heavy arrays may
be cached to results/physics_validation/*.npz (gitignored).

  python physics_validation.py            # full study + printed report
  python physics_validation.py --fig      # + paper_figs/physics_validation.pdf
  python physics_validation.py --test     # unit tests only
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import cfrp_phasefield_2d as pf

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "results", "physics_validation")
FIG_PATH = os.path.join(HERE, "paper_figs", "physics_validation.pdf")

# manufactured-operator constants (normalised; chosen O(1), well-conditioned):
#   A d = MMS_C0 · d − MMS_KAPPA · ∇·(M ∇d),  M = I  (constant-coefficient Lap)
# this is exactly the algebraic shape of `_solve_damage`'s operator
#   (Gc/ℓ + 2H) d − ℓ ∇·(Gc A ∇d) = 2H  with Gc/ℓ+2H ↔ c0 and ℓ Gc ↔ κ.
MMS_C0 = 3.0
MMS_KAPPA = 0.7
MMS_LX = 1.0
MMS_LY = 0.5


# ═════════════════════════════════════════════════════════════════════════════
#  (1) Method of manufactured solutions on build_var_coeff_divM
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class MMSResult:
    """Convergence record for one MMS variant (one error per grid)."""
    label: str
    h: np.ndarray            # mesh size hx per grid
    err: np.ndarray          # L2 error per grid
    order: float             # fitted slope of log(err) vs log(h)

    def order_between(self) -> np.ndarray:
        """Pairwise observed orders log(e_i/e_{i+1}) / log(h_i/h_{i+1})."""
        return (np.log(self.err[:-1] / self.err[1:])
                / np.log(self.h[:-1] / self.h[1:]))


def _manufactured_fields(nx: int, ny: int,
                         Lx: float = MMS_LX, Ly: float = MMS_LY):
    """Return (hx, hy, X, Y, d_star, source f) for the manufactured problem.

    d* = cos(πx/Lx) cos(πy/Ly);  ∇²d* = −(π²/Lx² + π²/Ly²) d*  (closed form),
    so for M = I the source is  f = c0 d* − κ ∇²d*.
    """
    hx, hy = Lx / (nx - 1), Ly / (ny - 1)
    xs = np.linspace(0.0, Lx, nx)
    ys = np.linspace(0.0, Ly, ny)
    X, Y = np.meshgrid(xs, ys)
    d_star = np.cos(np.pi * X / Lx) * np.cos(np.pi * Y / Ly)
    lap = -(np.pi ** 2 / Lx ** 2 + np.pi ** 2 / Ly ** 2) * d_star
    f = MMS_C0 * d_star - MMS_KAPPA * lap
    return hx, hy, X, Y, d_star, f


def _mms_operator(nx: int, ny: int, hx: float, hy: float) -> sp.csr_matrix:
    """A = c0 I − κ ∇·(M ∇·),  M = I, via the production flux stencil."""
    M = np.ones((ny, nx))
    L = pf.build_var_coeff_divM(M, M, np.zeros((ny, nx)), hx, hy)   # = Laplacian
    return (MMS_C0 * sp.eye(nx * ny, format="csr") - MMS_KAPPA * L).tocsr()


def mms_convergence(nx_list=(21, 31, 41, 61, 81),
                    Lx: float = MMS_LX, Ly: float = MMS_LY
                    ) -> dict[str, MMSResult]:
    """Run the MMS study over a grid sequence, return three MMSResults.

    Keys: "dirichlet" (boundary pinned to d*, interior solution error — the
    clean 2nd-order headline), "residual" (interior truncation residual,
    BC-free stencil probe), "neumann" (production zero-flux BC, global solution
    error — the honest ≈1st-order boundary-limited number).
    """
    hs = []
    e_dir, e_res, e_neu = [], [], []
    for nx in nx_list:
        ny = int(round((nx - 1) * Ly / Lx)) + 1
        hx, hy, _, _, d_star, f = _manufactured_fields(nx, ny, Lx, Ly)
        A = _mms_operator(nx, ny, hx, hy)
        ds = d_star.ravel()

        # interior truncation residual ‖A d* − f‖ (no solve, no BC closure)
        res = (A.dot(ds) - f.ravel()).reshape(ny, nx)[1:-1, 1:-1]
        e_res.append(float(np.sqrt(np.mean(res ** 2))))

        # production Neumann solve, global L2 error
        d_neu = spla.spsolve(A, f.ravel()).reshape(ny, nx)
        e_neu.append(float(np.sqrt(np.mean((d_neu - d_star) ** 2))))

        # Dirichlet-pinned solve, interior L2 error (isolates interior stencil)
        Ad = A.tolil()
        rhs = f.ravel().copy()
        bmask = np.zeros((ny, nx), dtype=bool)
        bmask[0, :] = bmask[-1, :] = bmask[:, 0] = bmask[:, -1] = True
        for k in np.where(bmask.ravel())[0]:
            Ad.rows[k] = [k]; Ad.data[k] = [1.0]; rhs[k] = ds[k]
        d_dir = spla.spsolve(Ad.tocsr(), rhs).reshape(ny, nx)
        e_int = d_dir[1:-1, 1:-1] - d_star[1:-1, 1:-1]
        e_dir.append(float(np.sqrt(np.mean(e_int ** 2))))

        hs.append(hx)

    hs = np.array(hs)
    out = {}
    for key, lab, errs in (("dirichlet", "MMS-Dirichlet (interior soln)", e_dir),
                           ("residual", "MMS-residual (interior trunc.)", e_res),
                           ("neumann", "MMS-Neumann (global soln)", e_neu)):
        errs = np.array(errs)
        order = float(np.polyfit(np.log(hs), np.log(errs), 1)[0])
        out[key] = MMSResult(label=lab, h=hs, err=errs, order=order)
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  (2) Mesh-convergence study on the real solver (peak_F QoI)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class MeshConvResult:
    """Mesh-convergence record for the physical QoIs."""
    nx: np.ndarray
    ny: np.ndarray
    h: np.ndarray
    peak_F: np.ndarray
    area_final: np.ndarray
    rel_growth: np.ndarray
    ell: float               # frozen absolute PF length (same on every grid)
    load: float
    seconds: np.ndarray
    # Richardson extrapolation on peak_F (finest triplet)
    qoi_asymptote: float     # extrapolated peak_F as h→0
    rel_err: np.ndarray      # |peak_F − asymptote| / |asymptote| per grid
    default_nx: int          # the production default (60)
    default_rel_err: float   # its relative QoI error


def _richardson_asymptote(h: np.ndarray, q: np.ndarray) -> tuple[float, float]:
    """Richardson extrapolation of QoI q(h)→q0 from the three FINEST grids.

    Fits q(h) ≈ q0 + C h^p (generic non-integer order from the triplet ratio,
    robust to the non-uniform h sequence).  Falls back to the finest value if
    the triplet is degenerate (non-monotone / equal spacing issues).
    """
    h3, q3 = h[-3:], q[-3:]
    r = (q3[1] - q3[0]) / (q3[2] - q3[1] + 1e-30)
    hr01 = h3[0] / h3[1]
    hr12 = h3[1] / h3[2]
    # solve r = (hr01^p − 1)/(1 − hr12^-p) approximately by assuming hr01≈hr12=H
    H = 0.5 * (hr01 + hr12)
    if r > 1.0 and H > 1.0:
        p = float(np.log(r) / np.log(H))
        p = float(np.clip(p, 0.3, 4.0))
        q0 = q3[2] + (q3[2] - q3[1]) / (H ** p - 1.0)
    else:
        p, q0 = 2.0, float(q3[2])
    return float(q0), p


def mesh_convergence(nx_list=(30, 40, 52, 60, 80),
                     load: float = 0.10, default_nx: int = 60,
                     verbose: bool = False) -> MeshConvResult:
    """Run `simulate_growth` for a FIXED physical defect at increasing nx/ny.

    ny is scaled to keep the nominal aspect (ny ≈ nx·48/60) and ℓ is FROZEN at
    the default config's absolute value so refinement error is separated from
    the ℓ-modelling choice.  The defect is seeded by the same (cx,cy,layer,
    log2size) on every grid → a fixed physical flaw.
    """
    base = pf.LaminateConfig()
    ell_fixed = base.ell
    nxs, nys, hs, Fs, areas, rels, secs = [], [], [], [], [], [], []
    for nx in nx_list:
        ny = int(round(nx * 48 / 60))
        cfg = pf.LaminateConfig(nx=nx, ny=ny, ell=ell_fixed)
        d0 = pf.seed_defect(0.5, 0.5, 9.0, 1.0, cfg)
        t0 = time.perf_counter()
        r = pf.simulate_growth(d0, load, cfg)
        dt = time.perf_counter() - t0
        nxs.append(nx); nys.append(ny); hs.append(cfg.hx)
        Fs.append(r.peak_F); areas.append(r.area_final); rels.append(r.rel_growth)
        secs.append(dt)
        if verbose:
            print(f"  nx={nx:3d} ny={ny:3d} h={cfg.hx:.4f}  "
                  f"peak_F={r.peak_F:.5e}  area={r.area_final:.4e}  "
                  f"rel={r.rel_growth:.3f}  [{dt:.1f}s]")

    h = np.array(hs); F = np.array(Fs)
    q0, _ = _richardson_asymptote(h, F)
    rel_err = np.abs(F - q0) / (abs(q0) + 1e-30)
    if default_nx in nxs:
        d_err = float(rel_err[nxs.index(default_nx)])
    else:
        d_err = float("nan")
    return MeshConvResult(
        nx=np.array(nxs), ny=np.array(nys), h=h, peak_F=F,
        area_final=np.array(areas), rel_growth=np.array(rels),
        ell=ell_fixed, load=load, seconds=np.array(secs),
        qoi_asymptote=q0, rel_err=rel_err,
        default_nx=default_nx, default_rel_err=d_err)


# ═════════════════════════════════════════════════════════════════════════════
#  Report
# ═════════════════════════════════════════════════════════════════════════════

def print_report(mms: dict[str, MMSResult], mc: MeshConvResult) -> None:
    bar = "═" * 78
    print(bar)
    print("  Stage-3 phase-field solver verification — MMS + mesh convergence")
    print(bar)

    print("\n(1) METHOD OF MANUFACTURED SOLUTIONS — build_var_coeff_divM")
    print(f"    operator  A d = {MMS_C0} d − {MMS_KAPPA} ∇·(∇d),   "
          f"d* = cos(πx/Lx)cos(πy/Ly)")
    for key in ("dirichlet", "residual", "neumann"):
        r = mms[key]
        errs = "  ".join(f"{e:.2e}" for e in r.err)
        print(f"    {r.label:<32s} order={r.order:5.2f}")
        print(f"        h    = " + "  ".join(f"{hh:.4f}" for hh in r.h))
        print(f"        L2   = " + errs)
    print("    → interior stencil is 2nd-order (Dirichlet & residual ≈2);")
    print("      zero-flux Neumann closure is 1st-order at the boundary")
    print("      (global Neumann order ≈1) — a known, documented limitation.")

    print("\n(2) MESH CONVERGENCE — simulate_growth, fixed defect, frozen ℓ")
    print(f"    load={mc.load:.3f}   ℓ={mc.ell:.4f} (absolute, same on all grids)"
          f"   QoI = peak reaction force")
    print(f"    {'nx':>4s} {'ny':>4s} {'h':>8s} {'peak_F':>12s} "
          f"{'rel.err':>9s} {'area_f':>11s} {'rel_grow':>9s} {'t[s]':>6s}")
    for i in range(len(mc.nx)):
        mark = "  ← default" if mc.nx[i] == mc.default_nx else ""
        print(f"    {mc.nx[i]:4d} {mc.ny[i]:4d} {mc.h[i]:8.4f} "
              f"{mc.peak_F[i]:12.5e} {mc.rel_err[i]*100:8.2f}% "
              f"{mc.area_final[i]:11.4e} {mc.rel_growth[i]:9.3f} "
              f"{mc.seconds[i]:6.1f}{mark}")
    succ = np.abs(np.diff(mc.peak_F)) / np.abs(mc.peak_F[1:])
    print(f"    successive |Δpeak_F|/peak_F : "
          + "  ".join(f"{s*100:.2f}%" for s in succ))
    print(f"    Richardson asymptote peak_F → {mc.qoi_asymptote:.5e}")
    print(f"    default nx={mc.default_nx}: peak_F mesh error "
          f"≈ {mc.default_rel_err*100:.2f}% (vs Richardson asymptote)")
    print("    NOTE: peak_F is the mesh-convergent QoI; crack AREA/rel_growth")
    print("      are NOT monotone at fixed absolute ℓ (coarse grids underresolve")
    print("      the ℓ-band) — area is a trend, not a converged value.")
    print(bar)


# ═════════════════════════════════════════════════════════════════════════════
#  Figure
# ═════════════════════════════════════════════════════════════════════════════

def make_figure(mms: dict[str, MMSResult], mc: MeshConvResult,
                out_path: str = FIG_PATH) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import sys
    sys.path.insert(0, os.path.join(HERE, "slides", "figure_sources"))
    try:
        from thesis_style import use
        figsize = use(width_frac=1.0, aspect=0.42)
    except Exception:
        figsize = (10.0, 4.0)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=figsize)

    # (a) MMS log-log error vs h with fitted slopes
    styles = {"dirichlet": ("#1565C0", "o"),
              "residual": ("#2e7d32", "s"),
              "neumann": ("#b71c1c", "^")}
    for key in ("dirichlet", "residual", "neumann"):
        r = mms[key]
        c, mk = styles[key]
        ax[0].loglog(r.h, r.err, mk + "-", color=c, ms=5,
                     label=f"{r.label} (p={r.order:.2f})")
    # reference order-2 and order-1 triangles anchored to the dirichlet curve
    h0 = mms["dirichlet"].h
    e0 = mms["dirichlet"].err[0]
    ax[0].loglog(h0, e0 * (h0 / h0[0]) ** 2, "k--", lw=0.8, label="slope 2")
    ax[0].loglog(h0, mms["neumann"].err[0] * (h0 / h0[0]) ** 1, "k:", lw=0.8,
                 label="slope 1")
    ax[0].set_xlabel("mesh size $h$", fontsize=8)
    ax[0].set_ylabel(r"$\|d - d^*\|_{L_2}$", fontsize=8)
    ax[0].set_title("(a) MMS convergence — damage operator", fontsize=9)
    ax[0].legend(fontsize=6, loc="lower right")
    ax[0].tick_params(labelsize=7)
    ax[0].grid(True, which="both", alpha=0.25)

    # (b) QoI peak_F vs h with Richardson asymptote
    ax[1].plot(mc.h, mc.peak_F, "o-", color="#1565C0", ms=5, label="peak $F$")
    ax[1].axhline(mc.qoi_asymptote, color="0.4", ls="--", lw=0.9,
                  label=f"Richardson $h\\to0$ = {mc.qoi_asymptote:.4f}")
    # mark default grid
    di = list(mc.nx).index(mc.default_nx) if mc.default_nx in mc.nx else None
    if di is not None:
        ax[1].plot(mc.h[di], mc.peak_F[di], "*", color="#b71c1c", ms=14,
                   label=f"default nx={mc.default_nx} "
                         f"({mc.default_rel_err*100:.1f}% err)")
    ax[1].set_xlabel("mesh size $h$", fontsize=8)
    ax[1].set_ylabel("peak reaction force", fontsize=8)
    ax[1].set_title(f"(b) QoI mesh convergence (load={mc.load:.2f}, "
                    r"$\ell$ frozen)", fontsize=9)
    ax[1].legend(fontsize=6, loc="lower right")
    ax[1].tick_params(labelsize=7)
    ax[1].grid(True, alpha=0.25)
    ax[1].invert_xaxis()   # refinement = left→right

    plt.tight_layout()
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

    # --- MMS: small grid sequence, fast ---
    mms = mms_convergence(nx_list=(11, 17, 25, 33))

    rd = mms["dirichlet"]
    ok(np.all(np.diff(rd.err) < 0), "MMS-Dirichlet error decreases with refine")
    ok(rd.order > 1.5, f"MMS-Dirichlet order ~2 (got {rd.order:.2f})")
    ok(rd.order < 2.6, f"MMS-Dirichlet order not super-quadratic "
                       f"({rd.order:.2f})")

    rr = mms["residual"]
    ok(np.all(np.diff(rr.err) < 0), "MMS-residual decreases with refinement")
    ok(rr.order > 1.5, f"MMS-residual (interior stencil) order ~2 "
                       f"(got {rr.order:.2f})")

    rn = mms["neumann"]
    # the production BC really is only 1st-order — assert it is sub-quadratic
    ok(np.all(np.diff(rn.err) < 0), "MMS-Neumann decreases with refinement")
    ok(rn.order < rd.order + 1e-6,
       "Neumann global order ≤ interior order (boundary-limited)")

    # --- mesh convergence: 3 tiny grids, peak_F Cauchy-ish ---
    mc = mesh_convergence(nx_list=(24, 32, 44), load=0.10, default_nx=44)
    ok(np.all(mc.peak_F > 0), "peak_F positive on all grids")
    succ = np.abs(np.diff(mc.peak_F)) / np.abs(mc.peak_F[1:])
    ok(succ[-1] <= succ[0] + 0.05,
       f"peak_F successive differences shrink-ish ({succ})")
    ok(np.isfinite(mc.qoi_asymptote), "Richardson asymptote finite")
    ok(0.0 <= mc.default_rel_err < 0.5,
       f"default-grid rel QoI error sane ({mc.default_rel_err:.3f})")
    ok(np.all(mc.h[:-1] > mc.h[1:]), "mesh size strictly decreasing")

    print(f"physics_validation: {n}/{n} unit tests passed")
    return n


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

def _parse_int_list(s: str) -> tuple:
    return tuple(int(x) for x in s.replace(" ", "").split(",") if x)


def main():
    ap = argparse.ArgumentParser(
        description="Stage-3 FD phase-field verification: MMS + mesh "
                    "convergence.")
    ap.add_argument("--mms-grids", type=_parse_int_list,
                    default=(21, 31, 41, 61, 81),
                    help="comma list of nx for the MMS study")
    ap.add_argument("--mesh-grids", type=_parse_int_list,
                    default=(30, 40, 52, 60, 80),
                    help="comma list of nx for the QoI mesh study")
    ap.add_argument("--load", type=float, default=0.10,
                    help="transverse peel strain for the mesh study")
    ap.add_argument("--default-nx", type=int, default=60,
                    help="production default grid to place on the error curve")
    ap.add_argument("--cache", action="store_true",
                    help="cache study arrays to results/physics_validation/")
    ap.add_argument("--fig", action="store_true",
                    help="save paper_figs/physics_validation.pdf (+ .png)")
    ap.add_argument("--test", action="store_true", help="unit tests only")
    args = ap.parse_args()

    if args.test:
        run_tests(); return

    t0 = time.perf_counter()
    print("[mms] running manufactured-solution convergence ...")
    mms = mms_convergence(nx_list=args.mms_grids)
    print("[mesh] running QoI mesh-convergence study ...")
    mc = mesh_convergence(nx_list=args.mesh_grids, load=args.load,
                          default_nx=args.default_nx, verbose=True)
    print(f"[done] study in {time.perf_counter() - t0:.1f}s\n")

    print_report(mms, mc)

    if args.cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez(os.path.join(CACHE_DIR, "study.npz"),
                 mms_h=mms["dirichlet"].h,
                 mms_err_dir=mms["dirichlet"].err,
                 mms_err_res=mms["residual"].err,
                 mms_err_neu=mms["neumann"].err,
                 mms_order_dir=mms["dirichlet"].order,
                 mms_order_res=mms["residual"].order,
                 mms_order_neu=mms["neumann"].order,
                 mc_nx=mc.nx, mc_h=mc.h, mc_peak_F=mc.peak_F,
                 mc_area=mc.area_final, mc_rel=mc.rel_growth,
                 mc_asymptote=mc.qoi_asymptote, mc_rel_err=mc.rel_err,
                 mc_default_rel_err=mc.default_rel_err)
        print(f"cached → {os.path.join(CACHE_DIR, 'study.npz')}")

    if args.fig:
        p = make_figure(mms, mc)
        print(f"figure → {p}")


if __name__ == "__main__":
    main()
