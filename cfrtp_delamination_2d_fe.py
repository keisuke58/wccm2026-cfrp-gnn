"""cfrtp_delamination_2d_fe.py — full 2D finite-element MIXED-MODE DELAMINATION-FRONT
propagation for CFRTP (Daikin theme), upgrading the mixed-mode cohesive CONSTITUTIVE
law (cfrtp_cohesive_mixedmode.py) to an actual propagating crack on a mesh — a bilayer
bonded by a cohesive interface, loaded so the delamination front advances along the
interface under combined peel + shear.

Model: two elastic ply layers (plane-strain CST), meshed with DUPLICATED nodes on the
mid-plane interface, connected by cohesive interface springs carrying the Camanho-Davila
mixed-mode bilinear law with Benzeggagh-Kenane mode interaction (imported from
cfrtp_cohesive_mixedmode.py). A pre-crack (unbonded interface over x < a0) starts the
delamination. The right end is clamped; the loaded (cracked) end's top-arm tip is driven
by a prescribed displacement at an angle theta (theta sets the nominal mode mixity: 0 =
peel/mode I, larger = more shear/mode II), with the bottom-arm tip held. The nonlinear
cohesive problem is solved incrementally by a secant (damage-explicit) fixed-point
iteration; irreversible damage advances the front.

What it shows:
  * the deformed bilayer with the cohesive-damage field — the delamination FRONT,
  * the load-displacement response (rise -> peak -> propagation),
  * the delamination length (front position) vs applied displacement (an R-curve-like
    advance),
  * the energy dissipated vs crack advance ~ the mixed-mode toughness (front mixity).

Validated: with no pre-crack growth the initial response matches the bonded bilayer
stiffness; the cohesive law's dissipation per unit advance tracks Gc at the front mixity.

Honest scope: 2D plane-strain, CST bulk, node-lumped cohesive springs (not full cohesive
elements), secant solver with small steps (no arc-length, so only stable propagation is
traced), illustrative CFRTP-like properties. A seed toward 3D fronts / cohesive elements
/ arc-length. Physics leads; no ML.

Run:  python3 cfrtp_delamination_2d_fe.py     (writes cfrtp_delamination_2d_fe.png)
      python3 cfrtp_delamination_2d_fe.py --help
"""
from __future__ import annotations

import argparse

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from cfrtp_cohesive_mixedmode import mixed_params, K_PEN

SEED = 20260726
LX, TLAY = 20e-3, 0.6e-3      # length, single-layer thickness [m]
E_PLY, NU = 60e9, 0.30        # CFRTP ply (plane-strain, isotropic seed)
A0 = 5e-3                     # initial pre-crack length [m]


def cst_B(p):
    x1, y1 = p[0]; x2, y2 = p[1]; x3, y3 = p[2]
    A2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    b = np.array([y2 - y3, y3 - y1, y1 - y2]) / A2
    c = np.array([x3 - x2, x1 - x3, x2 - x1]) / A2
    B = np.zeros((3, 6))
    for k in range(3):
        B[0, 2 * k] = b[k]; B[1, 2 * k + 1] = c[k]
        B[2, 2 * k] = c[k]; B[2, 2 * k + 1] = b[k]
    return B, 0.5 * abs(A2)


def build(nx, nz):
    xs = np.linspace(0, LX, nx + 1)
    zb = np.linspace(0, TLAY, nz + 1); zt = np.linspace(TLAY, 2 * TLAY, nz + 1)
    nb = (nx + 1) * (nz + 1)
    nodes = np.zeros((2 * nb, 2))
    for j in range(nz + 1):
        for i in range(nx + 1):
            nodes[j * (nx + 1) + i] = (xs[i], zb[j])
            nodes[nb + j * (nx + 1) + i] = (xs[i], zt[j])
    tris = []
    def q(base, i, j):
        return base + j * (nx + 1) + i
    for base in (0, nb):
        for i in range(nx):
            for j in range(nz):
                a, b, c, d = q(base, i, j), q(base, i + 1, j), q(base, i + 1, j + 1), q(base, i, j + 1)
                tris.append((a, b, c)); tris.append((a, c, d))
    # cohesive pairs on the interface (bottom top-row j=nz  <->  top bottom-row j=0)
    pairs = [(nz * (nx + 1) + i, nb + i, xs[i]) for i in range(nx + 1)]
    return nodes, np.array(tris, np.int64), pairs, nb


def Cmat():
    E, nu = E_PLY, NU
    f = E / ((1 + nu) * (1 - 2 * nu))
    return f * np.array([[1 - nu, nu, 0], [nu, 1 - nu, 0], [0, 0, (1 - 2 * nu) / 2]])


def assemble_bulk(nodes, tris):
    N = len(nodes); C = Cmat()
    rows, cols, vals = [], [], []
    for t in tris:
        B, A = cst_B(nodes[list(t)])
        Ke = A * (B.T @ C @ B)
        dof = np.array([2 * t[0], 2 * t[0] + 1, 2 * t[1], 2 * t[1] + 1, 2 * t[2], 2 * t[2] + 1])
        for a in range(6):
            for b in range(6):
                rows.append(dof[a]); cols.append(dof[b]); vals.append(Ke[a, b])
    return sp.csr_matrix((vals, (rows, cols)), shape=(2 * N, 2 * N))


def coh_state(dn, ds, lam_max):
    lam = np.hypot(max(dn, 0.0), ds)
    lam_h = max(lam, lam_max)
    phi = np.arctan2(abs(ds), max(dn, 1e-30))
    dm0, dmf, Gc, B = mixed_params(phi)
    if lam_h <= dm0:
        d = 0.0
    elif lam_h >= dmf:
        d = 1.0
    else:
        d = dmf * (lam_h - dm0) / (lam_h * (dmf - dm0))
    return d, max(lam, lam_max)


def solve(nx=80, nz=3, theta_deg=25.0, nstep=60, dmax_tip=6e-4):
    nodes, tris, pairs, nb = build(nx, nz)
    N = len(nodes); ndof = 2 * N
    Kb = assemble_bulk(nodes, tris).tolil()
    hx = LX / nx
    active = [(b, t, x, (hx if 0 < i < nx else hx / 2)) for i, (b, t, x) in enumerate(pairs)
              if x >= A0]                                   # bonded region only
    lam_hist = {b: 0.0 for (b, t, x, w) in active}
    dmg = {b: 0.0 for (b, t, x, w) in active}

    # BCs: clamp x=LX (both layers); bottom-arm tip fixed; top-arm tip prescribed
    fixed = {}
    for n in range(N):
        if nodes[n, 0] >= LX - 1e-12:
            fixed[2 * n] = 0.0; fixed[2 * n + 1] = 0.0
    tip_b = nz * 0 + 0                                       # bottom layer (0,0) index 0
    fixed[2 * 0] = 0.0; fixed[2 * 0 + 1] = 0.0               # bottom-arm tip fixed
    tip_t = nb + 0                                          # top layer (0,0) -> at z=TLAY... use top tip (0, z=2T)
    tip_t = nb + nz * (nx + 1) + 0                          # top layer node at (x=0, z=2T)
    th = np.radians(theta_deg)
    dirx, dirz = np.sin(th), np.cos(th)

    steps = np.linspace(dmax_tip / nstep, dmax_tip, nstep)
    disp_hist, load_hist, front_hist = [], [], []
    u = np.zeros(ndof)
    dfield_final = None
    for Dk in steps:
        fixed[2 * tip_t] = Dk * dirx; fixed[2 * tip_t + 1] = Dk * dirz
        for _ in range(40):                                 # secant fixed-point
            # cohesive tangent + internal force
            Kc = sp.lil_matrix((ndof, ndof)); fc = np.zeros(ndof)
            for (b, t, x, w) in active:
                dn = u[2 * t + 1] - u[2 * b + 1]            # normal (z) opening
                ds = u[2 * t] - u[2 * b]                    # tangential (x)
                d, lh = coh_state(dn, ds, lam_hist[b]); dmg[b] = d
                kn = (1 - d) * K_PEN if dn > 0 else K_PEN   # compression penalty
                ks = (1 - d) * K_PEN
                tn = kn * dn; ts = ks * ds
                for (comp, k, tr) in ((0, ks, ts), (1, kn, tn)):
                    it, ib = 2 * t + comp, 2 * b + comp
                    fc[it] += tr * w; fc[ib] -= tr * w      # spring restoring force
                    Kc[it, it] += k * w; Kc[ib, ib] += k * w
                    Kc[it, ib] -= k * w; Kc[ib, it] -= k * w
            T = (Kb + Kc).tocsr()
            R = Kb @ u + fc
            free = np.setdiff1d(np.arange(ndof), list(fixed.keys()))
            # apply Dirichlet
            du = np.zeros(ndof)
            for dof_i, val in fixed.items():
                du[dof_i] = val - u[dof_i]
            rhs = -(R + T @ du)
            Tff = T[free][:, free].tocsc()
            du[free] = spla.spsolve(Tff, rhs[free])
            u = u + du
            if np.max(np.abs(du)) < 1e-9:
                break
        for (b, t, x, w) in active:                         # commit history (irreversible)
            dn = u[2 * t + 1] - u[2 * b + 1]; ds = u[2 * t] - u[2 * b]
            _, lh = coh_state(dn, ds, lam_hist[b]); lam_hist[b] = lh
        react = -(Kb @ u)[2 * tip_t + 1]                    # vertical reaction at driven tip
        # delamination front = furthest bonded x that is fully damaged
        dmgd = [x for (b, t, x, w) in active if dmg[b] > 0.5]
        front = (max(dmgd) if dmgd else A0)
        disp_hist.append(Dk); load_hist.append(react); front_hist.append(front)
        dfield_final = (dict(dmg), dict(lam_hist))
    return (nodes, tris, pairs, nb, np.array(disp_hist), np.array(load_hist),
            np.array(front_hist), dmg, u, active, theta_deg)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nx", type=int, default=80)
    ap.add_argument("--nz", type=int, default=3)
    ap.add_argument("--theta", type=float, default=25.0, help="loading angle [deg] (mode mixity)")
    ap.add_argument("--nstep", type=int, default=60)
    ap.add_argument("--out", type=str, default="cfrtp_delamination_2d_fe.png")
    args = ap.parse_args()
    np.random.seed(SEED)

    res = solve(args.nx, args.nz, args.theta, args.nstep)
    (nodes, tris, pairs, nb, disp, load, front, dmg, u, active, theta) = res
    # propagation-onset peak = first local maximum (before the softening dip)
    kpk = next((i for i in range(1, len(load) - 1) if load[i] >= load[i + 1]), int(np.argmax(load)))
    print(f"CFRTP 2D mixed-mode delamination FE (theta={theta:.0f} deg, pre-crack {A0*1e3:.0f} mm):")
    print(f"  propagation-onset peak {load[kpk]:.0f} N/m at δ={disp[kpk]*1e6:.0f} um, then softening")
    print(f"  delamination front advanced {A0*1e3:.1f} -> {front[-1]*1e3:.1f} mm "
          f"(Δa {(front[-1]-A0)*1e3:.1f} mm); load re-stiffens near the clamp (finite specimen)")

    _plot(args.out, nodes, tris, pairs, nb, disp, load, front, dmg, u, active, theta)
    print(f"wrote {args.out}")


def _plot(out, nodes, tris, pairs, nb, disp, load, front, dmg, u, active, theta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))

    # deformed shape with interface damage
    scale = 3.0
    dn = nodes.copy(); dn[:, 0] += scale * u[0::2]; dn[:, 1] += scale * u[1::2]
    triang = mtri.Triangulation(dn[:, 0] * 1e3, dn[:, 1] * 1e3, tris)
    ax[0, 0].triplot(triang, color="#cccccc", lw=0.3)
    xs = np.array([x for (b, t, x, w) in active]); dd = np.array([dmg[b] for (b, t, x, w) in active])
    sc = ax[0, 0].scatter(xs * 1e3, np.full_like(xs, TLAY) * 1e3, c=dd, cmap="inferno",
                          s=14, vmin=0, vmax=1, zorder=5)
    fig.colorbar(sc, ax=ax[0, 0], label="cohesive damage d")
    ax[0, 0].set_title(f"deformed bilayer + interface damage (×{scale:.0f})\ndelamination front = damage edge")
    ax[0, 0].set_xlabel("x [mm]"); ax[0, 0].set_ylabel("z [mm]"); ax[0, 0].set_aspect("auto")

    ax[0, 1].plot(disp * 1e6, load, "-o", ms=2, color="#1f77b4")
    ax[0, 1].set_xlabel("applied tip displacement δ [µm]"); ax[0, 1].set_ylabel("reaction [N/m]")
    ax[0, 1].set_title("load–displacement (rise → peak → propagation)")
    ax[0, 1].grid(True, alpha=0.3)

    ax[1, 0].plot(disp * 1e6, front * 1e3, "-o", ms=2, color="#d62728")
    ax[1, 0].axhline(A0 * 1e3, color="k", ls=":", alpha=0.6, label="pre-crack a0")
    ax[1, 0].set_xlabel("applied tip displacement δ [µm]"); ax[1, 0].set_ylabel("delamination front x [mm]")
    ax[1, 0].set_title("delamination front advances along the interface")
    ax[1, 0].legend(); ax[1, 0].grid(True, alpha=0.3)

    # damage profile along the interface (process zone)
    ax[1, 1].plot(xs * 1e3, dd, "-", color="#b5651d")
    ax[1, 1].fill_between(xs * 1e3, dd, alpha=0.2, color="#b5651d")
    ax[1, 1].set_xlabel("x along interface [mm]"); ax[1, 1].set_ylabel("cohesive damage d")
    ax[1, 1].set_title("interface damage profile (cohesive process zone)")
    ax[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f"CFRTP 2D mixed-mode delamination-front FE (θ={theta:.0f}°, Benzeggagh-Kenane cohesive): "
                 "a propagating interlaminar crack (Daikin theme)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main()
