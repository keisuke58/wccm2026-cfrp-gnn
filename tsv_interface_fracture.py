"""tsv_interface_fracture.py — seed demo for research theme B (see
research/RESEARCH_THEMES_muramatsu.md): thermo-mechanical phase-field fracture of
a TSV Cu/Si interface (delamination), driven by CTE mismatch, with continuation
warm-start. Fuses this repo's TSV thermal-stress demos (tsv_thermal_stress.py ⑥,
tsv_3d_stress.py ⑦) with the phase-field fracture solver (phasefield_fracture_
warmstart.py ⑬) — i.e. the Muramatsu-lab fracture-coupling core applied to the
Post-5G back-end (3D packaging) reliability problem (POST5G_PROJECT_MAP.md).

Model: a Cu via (disk) embedded in a Si matrix (square), plane strain. Cooling
imposes a thermal eigenstrain eps0 = alpha(T) * tau * [1,1,0] (tau = scaled
thermal-mismatch load, ramped). The Cu/Si CTE mismatch (alpha_Cu >> alpha_Si)
concentrates stress at the interface; a reduced interface fracture toughness
Gc(interface) < Gc(bulk) makes damage localise there -> delamination ring.
AT2 phase-field, P1/CST weak form, staggered (alternate-minimisation) solve.

Idea C (continuation): each thermal-load step warm-starts (u,d,H) from the
previous step; cold re-solves each level from the undamaged scratch state.
Every solve is the exact FE staggered solution; warm-start cuts the total
staggered iterations, most around delamination onset.

Honest scope: AT2, no tension/compression split, illustrative *scaled* thermal
load tau (absolute magnitudes not calibrated; alpha ratio Cu:Si is physical),
structured mesh, weak-interface band as a delamination proxy. Concept seed for
theme B, not a validated reliability study. Reuses build_mesh from ⑬.

Run:  python3 tsv_interface_fracture.py      (writes tsv_interface_fracture.png)
      python3 tsv_interface_fracture.py --help
"""
from __future__ import annotations

import argparse

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from phasefield_fracture_warmstart import build_mesh

NU = 0.3
E_CU, E_SI = 1.1, 1.6           # relative stiffness (Cu softer than Si)
A_CU, A_SI = 1.0, 0.153         # CTE ratio (~17e-6 : 2.6e-6), Cu normalised to 1
R_VIA = 0.28                    # Cu via radius
ELL = 0.03
GC_BULK, GC_INT = 1.0e-3, 8.0e-5   # bulk vs (weak) interface toughness
KRES = 1e-7
STAG_TOL = 1e-4
STAG_MAX = 150


def cmat(Emod):
    lam = Emod * NU / ((1 + NU) * (1 - 2 * NU))
    mu = Emod / (2 * (1 + NU))
    return np.array([[lam + 2 * mu, lam, 0], [lam, lam + 2 * mu, 0], [0, 0, mu]])


def precompute(nodes, tris):
    nel = len(tris)
    cent = nodes[tris].mean(axis=1)
    rc = np.hypot(cent[:, 0] - 0.5, cent[:, 1] - 0.5)
    is_cu = rc < R_VIA
    Emod = np.where(is_cu, E_CU, E_SI)
    alpha = np.where(is_cu, A_CU, A_SI)
    Gc = np.where(np.abs(rc - R_VIA) < 1.5 * ELL, GC_INT, GC_BULK)   # weak interface band

    B = np.zeros((nel, 3, 6)); area = np.zeros(nel); bcg = np.zeros((nel, 2, 3))
    Ke0 = np.zeros((nel, 6, 6)); fth0 = np.zeros((nel, 6)); Cel = np.zeros((nel, 3, 3))
    for e, t in enumerate(tris):
        p = nodes[t]; x, y = p[:, 0], p[:, 1]
        A = 0.5 * ((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))
        b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]]) / (2 * A)
        c = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]]) / (2 * A)
        area[e] = abs(A)
        for k in range(3):
            B[e, 0, 2 * k] = b[k]; B[e, 1, 2 * k + 1] = c[k]
            B[e, 2, 2 * k] = c[k]; B[e, 2, 2 * k + 1] = b[k]
        bcg[e, 0] = b; bcg[e, 1] = c
        C = cmat(Emod[e]); Cel[e] = C
        Ke0[e] = abs(A) * B[e].T @ C @ B[e]
        fth0[e] = abs(A) * B[e].T @ (C @ np.array([1.0, 1.0, 0.0]))   # eigenstrain load shape
    udof = np.zeros((nel, 6), np.int64)
    udof[:, 0::2] = 2 * tris; udof[:, 1::2] = 2 * tris + 1
    rows_u = np.repeat(udof, 6, axis=1).ravel(); cols_u = np.tile(udof, (1, 6)).ravel()
    # damage constant part with per-element Gc
    Lap = (Gc * ELL)[:, None, None] * area[:, None, None] * np.einsum("eai,eaj->eij", bcg, bcg)
    rows_d = np.repeat(tris, 3, axis=1).ravel(); cols_d = np.tile(tris, (1, 3)).ravel()
    massL = area / 3.0
    N = len(nodes)
    Kd_const = sp.csr_matrix((Lap.ravel(), (rows_d, cols_d)), shape=(N, N))
    Kd_const += sp.diags(np.bincount(tris.ravel(),
                         weights=np.repeat((Gc / ELL) * massL, 3), minlength=N))
    return dict(B=B, area=area, Ke0=Ke0, fth0=fth0, Cel=Cel, alpha=alpha, is_cu=is_cu,
                udof=udof, rows_u=rows_u, cols_u=cols_u, Kd_const=Kd_const, massL=massL,
                tris=tris, N=N)


def staggered(P, tau, fix_u, free_u, u0, d0, H0, dlock):
    B, area, Ke0, fth0, Cel = P["B"], P["area"], P["Ke0"], P["fth0"], P["Cel"]
    alpha, udof, rows_u, cols_u = P["alpha"], P["udof"], P["rows_u"], P["cols_u"]
    Kd_const, massL, tris, N = P["Kd_const"], P["massL"], P["tris"], P["N"]
    eps0 = (alpha * tau)[:, None] * np.array([1.0, 1.0, 0.0])[None, :]   # element eigenstrain
    u, d, H = u0.copy(), d0.copy(), H0.copy()
    for it in range(1, STAG_MAX + 1):
        gd = (1 - d[tris].mean(axis=1)) ** 2 + KRES
        K = sp.csr_matrix(((Ke0 * gd[:, None, None]).ravel(), (rows_u, cols_u)), shape=(2 * N, 2 * N))
        f = np.bincount(udof.ravel(),
                        weights=((gd * alpha * tau)[:, None] * fth0).ravel(), minlength=2 * N)
        u = np.zeros(2 * N)
        u[free_u] = spla.spsolve(K[np.ix_(free_u, free_u)].tocsc(), f[free_u])
        eps = np.einsum("eij,ej->ei", B, u[udof]) - eps0
        psi = 0.5 * np.einsum("ei,eij,ej->e", eps, Cel, eps)
        H = np.maximum(H, psi)
        mvar = np.bincount(tris.ravel(), weights=np.repeat(2 * H * massL, 3), minlength=N)
        Kd = (Kd_const + sp.diags(mvar)).tocsc()
        rhs = np.bincount(tris.ravel(), weights=np.repeat(2 * H * massL, 3), minlength=N)
        d_new = spla.spsolve(Kd, rhs)
        d_new = np.clip(np.maximum(d_new, dlock), 0, 1)
        change = np.max(np.abs(d_new - d)); d = d_new
        if change < STAG_TOL:
            break
    energy = float(np.sum(gd * 0.5 * np.einsum("ei,eij,ej->e", eps, Cel, eps) * area))
    return u, d, H, it, energy


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=52, help="cells per side")
    ap.add_argument("--nstep", type=int, default=26, help="thermal-load increments")
    ap.add_argument("--taumax", type=float, default=0.03, help="max scaled thermal mismatch")
    ap.add_argument("--out", type=str, default="tsv_interface_fracture.png")
    args = ap.parse_args()
    if args.nstep < 3:
        ap.error("--nstep must be >= 3")

    nodes, tris = build_mesh(args.n)
    P = precompute(nodes, tris)
    N = P["N"]
    x, y = nodes[:, 0], nodes[:, 1]
    edge = np.where((x < 1e-9) | (x > 1 - 1e-9) | (y < 1e-9) | (y > 1 - 1e-9))[0]
    fix = np.concatenate([2 * edge, 2 * edge + 1])          # outer boundary clamped
    free_u = np.setdiff1d(np.arange(2 * N), fix)
    taus = np.linspace(args.taumax / args.nstep, args.taumax, args.nstep)

    uw = np.zeros(2 * N); dw = np.zeros(N); Hw = np.zeros(len(tris)); dlock = np.zeros(N)
    warm_it, energy = [], []
    for tau in taus:
        u, d, H, it, en = staggered(P, tau, fix, free_u, uw, dw, Hw, dlock)
        uw, dw, Hw = u, d, H; dlock = np.maximum(dlock, d)
        warm_it.append(it); energy.append(en)

    cold_it = []
    for tau in taus:
        _, _, _, it, _ = staggered(P, tau, fix, free_u,
                                   np.zeros(2 * N), np.zeros(N), np.zeros(len(tris)), np.zeros(N))
        cold_it.append(it)

    tc, tw = int(np.sum(cold_it)), int(np.sum(warm_it))
    dmax = dw[np.where(np.abs(np.hypot(nodes[:, 0] - 0.5, nodes[:, 1] - 0.5) - R_VIA) < 2 * ELL)]
    print(f"\nTSV Cu/Si interface delamination: {args.nstep} thermal steps, mesh {args.n}x{args.n}")
    print(f"  staggered iterations  cold {tc}   warm(continuation) {tw}   "
          f"({100*(tc-tw)/tc:.0f}% fewer)")
    print(f"  peak per-step cold {max(cold_it)} warm {max(warm_it)};  max interface damage {dw.max():.2f}")

    _plot(args.out, nodes, tris, P["is_cu"], dw, taus, energy, cold_it, warm_it, tc, tw)
    print(f"wrote {args.out}")


def _plot(out, nodes, tris, is_cu, d, taus, energy, cold_it, warm_it, tc, tw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    tr = Triangulation(nodes[:, 0], nodes[:, 1], tris)
    fig, ax = plt.subplots(2, 2, figsize=(13, 11))

    ax[0, 0].tripcolor(tr, facecolors=is_cu.astype(float), cmap="copper_r", alpha=0.5)
    th = np.linspace(0, 2 * np.pi, 200)
    ax[0, 0].plot(0.5 + R_VIA * np.cos(th), 0.5 + R_VIA * np.sin(th), "k--", lw=1)
    ax[0, 0].set_aspect("equal"); ax[0, 0].set_title("Cu via (dark) in Si — weak interface (dashed)")

    tp = ax[0, 1].tripcolor(tr, d, shading="gouraud", cmap="inferno", vmin=0, vmax=1)
    ax[0, 1].plot(0.5 + R_VIA * np.cos(th), 0.5 + R_VIA * np.sin(th), "c--", lw=0.8)
    ax[0, 1].set_aspect("equal"); ax[0, 1].set_title("damage d at max cooling — interface delamination")
    fig.colorbar(tp, ax=ax[0, 1], fraction=0.046)

    ax[1, 0].plot(taus, energy, "-o", ms=3, color="#1f77b4")
    ax[1, 0].set_title("stored elastic energy vs thermal load — drop at delamination")
    ax[1, 0].set_xlabel("scaled thermal mismatch τ"); ax[1, 0].set_ylabel("stored energy")
    ax[1, 0].grid(True, alpha=0.3)

    xs = np.arange(len(cold_it))
    ax[1, 1].plot(xs, cold_it, "-o", ms=3, color="#d62728", label="cold (from scratch)")
    ax[1, 1].plot(xs, warm_it, "-o", ms=3, color="#1f77b4", label="warm (continuation)")
    ax[1, 1].set_title(f"staggered iterations per thermal step\ntotal cold {tc} → warm {tw} "
                       f"({100*(tc-tw)/tc:.0f}% fewer)")
    ax[1, 1].set_xlabel("thermal-load step"); ax[1, 1].set_ylabel("staggered iterations")
    ax[1, 1].legend(); ax[1, 1].grid(True, alpha=0.3)

    fig.suptitle("TSV Cu/Si interface delamination (thermo-mechanical phase-field): thermal-load "
                 "continuation warm-starts the coupled solve — theme-B seed", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main()
