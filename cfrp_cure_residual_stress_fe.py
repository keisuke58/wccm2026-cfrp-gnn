"""cfrp_cure_residual_stress_fe.py — UPGRADE of the CLT seed (cfrp_cure_residual_stress.py)
to a weak-form finite-element, INCREMENTALLY CURED laminate that outputs both the
residual STRESS and residual STRAIN fields (Daikin/NEDO theme,
research/MEMO_cfrp_residual_stress_collab.md). This is the "格上げ": from analytic CLT
to a coupled cure-hardening FE driven by a real cure cycle.

Coupling (cure-hardening / CHILE picture, cf. LITREVIEW_cfrp_cure_residual_stress.md):
  * cure kinetics — degree of cure alpha(t) evolves along a cure cycle T(t) via an
    Arrhenius nth-order law  dalpha/dt = A e^{-Ea/RT} (1-alpha)^n,
  * CHILE stiffness — the ply modulus develops with cure, g(alpha) in [g0, 1], so
    stress only locks in after gelation (before gel the resin flows, ~stress free),
  * eigenstrains — thermal  alpha_CTE * dT  and chemical cure shrinkage  beta * dalpha
    (fibres inert; transverse only), applied INCREMENTALLY with the current modulus.
The residual state is built step by step by a plane-stress CST finite-element solve
K(g) du = f(deigen); stress and strain are accumulated. A [0/90] cross-section is used
so the ply CTE/stiffness mismatch locks in interlaminar residual stress on cool-down.

Validated: a SINGLE uniform ply, statically-determinate supports, contracts freely on
cool-down -> ~zero residual stress (machine-precision), the composite analogue of the
free-expansion check. The [0/90] laminate then develops real residual stress/strain.

Outputs (the process -> residual state map the collaboration wants):
  * cure cycle T(t) and degree of cure alpha(t) (the process input),
  * modulus development g(alpha) and the residual-stress build-up history,
  * residual STRESS field sigma_xx on the cross-section,
  * residual STRAIN field eps_xx on the cross-section (the added output).

Honest scope: plane-stress CST, CHILE (instantaneously linear-elastic per step, no
viscoelastic stress relaxation), cure shrinkage as a linear eigenstrain in dalpha,
no tool-part interaction; illustrative T300/epoxy-like properties and cure cycle. A
seed toward a viscoelastic cure FE. Physics is the lead; a process->residual surrogate
(repo CFRP-GNN / DeepONet, cf. ㉑) is the subordinate layer. No ML here.

Run:  python3 cfrp_cure_residual_stress_fe.py     (writes cfrp_cure_residual_stress_fe.png)
      python3 cfrp_cure_residual_stress_fe.py --help
"""
from __future__ import annotations

import argparse

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

SEED = 20260726
LX, LZ = 20e-3, 0.6e-3          # laminate cross-section: length x thickness [m]
# T300/epoxy-like ply (plane-stress, cross-section x=length, z=thickness)
E1, E2 = 135e9, 9e9            # fibre-dir / transverse modulus [Pa]
NU = 0.30
G12 = 5e9
A1, A2 = -0.3e-6, 28e-6        # CTE fibre / transverse [1/K]
BETA_T = -4e-3                 # transverse chemical cure-shrinkage (per unit dalpha)
# cure kinetics (illustrative Arrhenius nth-order)
A_K, EA_K, NEXP = 1e8, 5.5e4, 1.6
RGAS = 8.314
ALPHA_GEL = 0.5               # gelation: stress builds only above this
G0 = 0.02                     # rubbery/ungelled stiffness fraction
T_CURE, T_ROOM = 180.0, 25.0


def build_mesh(nx, nz):
    xs = np.linspace(0, LX, nx + 1); zs = np.linspace(0, LZ, nz + 1)
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    nodes = np.column_stack([X.ravel(), Z.ravel()])
    idx = lambda i, j: i * (nz + 1) + j
    tris, ply = [], []
    for i in range(nx):
        for j in range(nz):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            tris.append((a, b, c)); tris.append((a, c, d))
            # layup [0/90] through thickness: lower half 0deg, upper half 90deg
            deg = 0 if (j + 0.5) / nz < 0.5 else 90
            ply.append(deg); ply.append(deg)
    return nodes, np.asarray(tris, np.int64), np.asarray(ply)


def ply_material(deg):
    """Plane-stress orthotropic C (3x3) and eigenstrain coefficients (thermal a_x,a_z;
    shrink s_x,s_z) for a ply in the x-z cross-section. 0deg: fibre along x. 90deg:
    fibre along y (out of plane) -> both x,z transverse."""
    if deg == 0:
        Ex, Ez, ax, az = E1, E2, A1, A2
        sx, sz = 0.0, BETA_T                      # shrink transverse (z) only
    else:
        Ex, Ez, ax, az = E2, E2, A2, A2
        sx, sz = BETA_T, BETA_T                   # both in-plane transverse
    nuxz = NU * Ez / E1 if deg == 0 else NU
    nuzx = nuxz * Ez / Ex
    d = 1.0 - nuxz * nuzx
    C = np.array([[Ex / d, nuzx * Ex / d, 0.0],
                  [nuxz * Ez / d, Ez / d, 0.0],
                  [0.0, 0.0, G12]])
    return C, np.array([ax, az, 0.0]), np.array([sx, sz, 0.0])


def cst_B(p):
    x1, y1 = p[0]; x2, y2 = p[1]; x3, y3 = p[2]
    A2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    area = 0.5 * abs(A2)
    b = np.array([y2 - y3, y3 - y1, y1 - y2]) / A2
    c = np.array([x3 - x2, x1 - x3, x2 - x1]) / A2
    B = np.zeros((3, 6))
    for k in range(3):
        B[0, 2 * k] = b[k]; B[1, 2 * k + 1] = c[k]
        B[2, 2 * k] = c[k]; B[2, 2 * k + 1] = b[k]
    return B, area


def cure_cycle(nstep):
    """Illustrative cure cycle T(t): heat 25->180, hold, cool 180->25 (deg C)."""
    t = np.linspace(0, 1.0, nstep)
    T = np.piecewise(t, [t < 0.25, (t >= 0.25) & (t < 0.55), t >= 0.55],
                     [lambda tt: T_ROOM + (T_CURE - T_ROOM) * tt / 0.25,
                      lambda tt: T_CURE,
                      lambda tt: T_CURE + (T_ROOM - T_CURE) * (tt - 0.55) / 0.45])
    return t, T


def integrate_cure(t, T):
    a = np.zeros_like(t)
    for i in range(1, len(t)):
        Tk = T[i - 1] + 273.15
        da = A_K * np.exp(-EA_K / (RGAS * Tk)) * (1 - a[i - 1]) ** NEXP * (t[i] - t[i - 1])
        a[i] = min(1.0, a[i - 1] + da)
    return a


def g_chile(alpha):
    """Cure-hardening stiffness fraction: g0 below gel, ramps to 1 after gel."""
    x = np.clip((alpha - ALPHA_GEL) / (1 - ALPHA_GEL), 0, 1)
    return G0 + (1 - G0) * x


def solve(nx, nz, nstep, single=False):
    nodes, tris, ply = build_mesh(nx, nz)
    if single:
        ply[:] = 0
    N = len(nodes); ndof = 2 * N
    Bs, areas, Cs, eth, esh = [], [], [], [], []
    for e, t in enumerate(tris):
        B, area = cst_B(nodes[list(t)])
        C, ath, ash = ply_material(ply[e])
        Bs.append(B); areas.append(area); Cs.append(C); eth.append(ath); esh.append(ash)

    # statically-determinate supports: allow free contraction + warp (validation-friendly)
    fixed = []
    n00 = np.argmin(nodes[:, 0] ** 2 + nodes[:, 1] ** 2)          # (0,0): u=w=0
    nL0 = np.argmin((nodes[:, 0] - LX) ** 2 + nodes[:, 1] ** 2)   # (LX,0): w=0
    fixed = [2 * n00, 2 * n00 + 1, 2 * nL0 + 1]
    free = np.setdiff1d(np.arange(ndof), fixed)

    t, T = cure_cycle(nstep); alpha = integrate_cure(t, T)
    g = g_chile(alpha)
    sig = np.zeros((len(tris), 3)); eps = np.zeros((len(tris), 3))
    u = np.zeros(ndof); hist = []

    for i in range(1, nstep):
        dT = T[i] - T[i - 1]; da = alpha[i] - alpha[i - 1]
        gi = g[i]
        rows, cols, vals = [], [], []
        F = np.zeros(ndof)
        for e, tri in enumerate(tris):
            dof = np.array([2 * tri[0], 2 * tri[0] + 1, 2 * tri[1], 2 * tri[1] + 1,
                            2 * tri[2], 2 * tri[2] + 1])
            Ce = gi * Cs[e]
            de = eth[e] * dT + esh[e] * da                # incremental eigenstrain
            Ke = areas[e] * (Bs[e].T @ Ce @ Bs[e])
            fe = areas[e] * (Bs[e].T @ Ce @ de)
            for a_ in range(6):
                F[dof[a_]] += fe[a_]
                for b_ in range(6):
                    rows.append(dof[a_]); cols.append(dof[b_]); vals.append(Ke[a_, b_])
        K = sp.csr_matrix((vals, (rows, cols)), shape=(ndof, ndof))
        du = np.zeros(ndof)
        du[free] = spla.spsolve(K[free][:, free].tocsc(), F[free])
        u += du
        for e, tri in enumerate(tris):
            dof = np.array([2 * tri[0], 2 * tri[0] + 1, 2 * tri[1], 2 * tri[1] + 1,
                            2 * tri[2], 2 * tri[2] + 1])
            deps = Bs[e] @ du[dof]
            de = eth[e] * dT + esh[e] * da
            sig[e] += g[i] * Cs[e] @ (deps - de)
            eps[e] += deps
        hist.append(np.max(np.abs(sig[:, 0])) / 1e6)
    return nodes, tris, ply, sig, eps, u, t, T, alpha, g, np.array(hist)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nx", type=int, default=40)
    ap.add_argument("--nz", type=int, default=16)
    ap.add_argument("--nstep", type=int, default=40)
    ap.add_argument("--out", type=str, default="cfrp_cure_residual_stress_fe.png")
    args = ap.parse_args()
    np.random.seed(SEED)

    # validation: a single uniform ply contracts freely -> ~zero residual stress
    sig_single = solve(args.nx, args.nz, args.nstep, single=True)[3]
    print(f"[validate] single uniform ply, free contraction: max |sigma_xx| "
          f"{np.max(np.abs(sig_single[:,0]))/1e6:.2e} MPa (should be ~0)")

    nodes, tris, ply, sig, eps, u, t, T, alpha, g, hist = solve(args.nx, args.nz, args.nstep)
    sxx = sig[:, 0] / 1e6
    exx = eps[:, 0]
    print(f"CFRP cure FE ([0/90], CHILE, cure cycle): final cure alpha={alpha[-1]:.3f}")
    print(f"  residual sigma_xx range [{sxx.min():.1f}, {sxx.max():.1f}] MPa")
    print(f"  residual eps_xx  range [{exx.min()*1e3:.3f}, {exx.max()*1e3:.3f}] milli-strain")

    _plot(args.out, nodes, tris, sig, eps, u, t, T, alpha, g, hist)
    print(f"wrote {args.out}")


def _plot(out, nodes, tris, sig, eps, u, t, T, alpha, g, hist):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    triang = mtri.Triangulation(nodes[:, 0] * 1e3, nodes[:, 1] * 1e3, tris)
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    # cure cycle + degree of cure
    ax0 = ax[0, 0]; ax0b = ax0.twinx()
    ax0.plot(t, T, "-", color="#d62728", label="temperature T(t)")
    ax0b.plot(t, alpha, "-", color="#1f77b4", label="degree of cure α(t)")
    ax0b.axhline(ALPHA_GEL, color="#1f77b4", ls=":", alpha=0.6)
    ax0.set_xlabel("normalized cure time"); ax0.set_ylabel("T [°C]", color="#d62728")
    ax0b.set_ylabel("degree of cure α", color="#1f77b4")
    ax0.set_title("cure cycle (process input): T(t) and cure α(t)")
    ax0.grid(True, alpha=0.3)

    # modulus development + residual-stress build-up (both vs cure time)
    ax1 = ax[0, 1]; ax1b = ax1.twinx()
    ax1.plot(t, g, "-", color="#2ca02c", label="stiffness g(α)")
    ax1.set_xlabel("normalized cure time")
    ax1.set_ylabel("stiffness fraction g(α)", color="#2ca02c")
    ax1b.plot(t[1:], hist, "-o", ms=2, color="#b5651d")
    ax1b.set_ylabel("max |σxx| build-up [MPa]", color="#b5651d")
    ax1.set_title("CHILE: stiffness develops during hold →\nresidual stress locks in on cool-down")
    ax1.grid(True, alpha=0.3)

    tp = ax[1, 0].tripcolor(triang, facecolors=sig[:, 0] / 1e6, cmap="coolwarm")
    fig.colorbar(tp, ax=ax[1, 0], label="residual σxx [MPa]")
    ax[1, 0].set_title("residual STRESS field σxx ([0/90], interlaminar mismatch)")
    ax[1, 0].set_xlabel("x (length) [mm]"); ax[1, 0].set_ylabel("z (thickness) [mm]")
    ax[1, 0].set_aspect("auto")

    tp2 = ax[1, 1].tripcolor(triang, facecolors=eps[:, 0] * 1e3, cmap="PuOr")
    fig.colorbar(tp2, ax=ax[1, 1], label="residual εxx [milli-strain]")
    ax[1, 1].set_title("residual STRAIN field εxx (added output)")
    ax[1, 1].set_xlabel("x (length) [mm]"); ax[1, 1].set_ylabel("z (thickness) [mm]")
    ax[1, 1].set_aspect("auto")

    fig.suptitle("CFRP cure residual stress & strain — weak-form FE + degree-of-cure + CHILE "
                 "(upgrade of CLT): cure cycle -> residual stress AND strain fields", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main()
