"""phasefield_fracture_warmstart.py — seed demo for research theme A (see
research/RESEARCH_THEMES_muramatsu.md): warm-start / load continuation for the
coupled, expensive phase-field brittle-fracture solve — the Muramatsu-lab core
(phase-field / fracture coupling) met with this repo's idea C (continuation as
accelerator; cf. gaa_material_sweep_warmstart.py, dd_breakdown_continuation.py).

Model: AT2 phase-field fracture on a single-edge-notched tension (SENT) square,
P1 / CST finite elements (weak form). Two coupled fields:
  * displacement u  : -div( g(d) sigma(u) ) = 0 ,  g(d)=(1-d)^2 + k
  * damage d in[0,1]: (Gc/l + 2H) d - Gc*l*lap(d) = 2H ,  H = max history of psi+
solved by the standard staggered (alternate-minimisation) scheme; a horizontal
pre-crack is imposed as d=1 on a thin notch band. Vertical tension is applied in
load increments.

Idea C (continuation warm-start):
  * warm : each load step initialises (u,d,H) from the previous converged step
  * cold : each load step is solved from the undamaged scratch state (u=0,
           d=notch, H=0) at that load level
Every solve is the exact FE staggered solution; only the staggered iteration
count changes. Warm-start cuts iterations most where the crack propagates —
exactly where a single-step (no-continuation) solve struggles.

Honest scope: AT2, no tension/compression energy split (tension-dominated SENT),
lumped damage mass, structured mesh with l~2h. This is a concept seed, not a
validated fracture study; it demonstrates the continuation/warm-start value that
theme A would develop (operator-learned initial guesses, crack-path operators).

Run:  python3 phasefield_fracture_warmstart.py     (writes phasefield_fracture_warmstart.png)
      python3 phasefield_fracture_warmstart.py --help
"""
from __future__ import annotations

import argparse

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

SEED = 20260725
E, NU = 1.0, 0.3            # non-dim Young / Poisson (plane strain)
GC, ELL = 5.0e-5, 0.03      # fracture toughness, regularisation length
KRES = 1e-7                 # residual stiffness
STAG_TOL = 1e-4
STAG_MAX = 150


def build_mesh(n):
    xs = np.linspace(0, 1, n + 1)
    X, Y = np.meshgrid(xs, xs)
    nodes = np.column_stack([X.ravel(), Y.ravel()])
    idx = lambda i, j: j * (n + 1) + i
    tris = []
    for j in range(n):
        for i in range(n):
            a, b, c, dd = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            tris.append((a, b, c)); tris.append((a, c, dd))
    return nodes, np.asarray(tris, np.int64)


def cmat():
    lam = E * NU / ((1 + NU) * (1 - 2 * NU))
    mu = E / (2 * (1 + NU))
    return np.array([[lam + 2 * mu, lam, 0], [lam, lam + 2 * mu, 0], [0, 0, mu]]), lam, mu


def precompute(nodes, tris):
    C, _, _ = cmat()
    nel = len(tris)
    B = np.zeros((nel, 3, 6)); area = np.zeros(nel)
    bcg = np.zeros((nel, 2, 3))          # damage gradient [b;c]
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
    # element base stiffness Ke0 = |A| B^T C B  (6x6), scaled by g(d) each iter
    Ke0 = np.einsum("e,eki,kl,elj->eij", area, B, C, B)
    udof = np.zeros((nel, 6), np.int64)
    udof[:, 0::2] = 2 * tris; udof[:, 1::2] = 2 * tris + 1
    rows_u = np.repeat(udof, 6, axis=1).ravel()
    cols_u = np.tile(udof, (1, 6)).ravel()
    # damage constant part: Gc*l*|A|*(bb+cc) + (Gc/l)*lumped-mass
    Lap = GC * ELL * area[:, None, None] * np.einsum("eai,eaj->eij", bcg, bcg)
    rows_d = np.repeat(tris, 3, axis=1).ravel()
    cols_d = np.tile(tris, (1, 3)).ravel()
    massL = area / 3.0                    # lumped mass contribution per node
    Kd_const = sp.csr_matrix((Lap.ravel(), (rows_d, cols_d)), shape=(len(nodes),) * 2)
    Kd_const = Kd_const + sp.diags(np.bincount(tris.ravel(),
                                   weights=np.repeat((GC / ELL) * massL, 3), minlength=len(nodes)))
    return C, B, area, Ke0, rows_u, cols_u, udof, Kd_const, massL


def solve_u(Ke0, rows_u, cols_u, gd, free_u, fix_u, uval, N):
    data = (Ke0 * gd[:, None, None]).ravel()
    K = sp.csr_matrix((data, (rows_u, cols_u)), shape=(2 * N, 2 * N))
    u = np.zeros(2 * N); u[fix_u] = uval
    rhs = -(K @ u)
    u[free_u] = spla.spsolve(K[np.ix_(free_u, free_u)].tocsc(), rhs[free_u])
    return u, K


def solve_d(Kd_const, massL, tris, H, d_lock, notch, N):
    mvar = np.bincount(tris.ravel(), weights=np.repeat(2 * H * massL, 3), minlength=N)
    K = (Kd_const + sp.diags(mvar)).tocsc()
    rhs = np.bincount(tris.ravel(), weights=np.repeat(2 * H * massL, 3), minlength=N)
    free = np.ones(N, bool); free[notch] = False
    d = np.zeros(N); d[notch] = 1.0      # Dirichlet lift: notch fixed = 1, rest baseline 0
    rhs2 = rhs - K @ d
    d[free] = spla.spsolve(K[np.ix_(free, free)], rhs2[free])
    return np.clip(np.maximum(d, d_lock), 0, 1)


def staggered(pc, tris, N, uval, fix_u, free_u, u0, d0, H0, dlock):
    C, B, area, Ke0, rows_u, cols_u, udof, Kd_const, massL = pc
    notch = pc_notch
    u, d, H = u0.copy(), d0.copy(), H0.copy()
    for it in range(1, STAG_MAX + 1):
        gd = (1 - d[tris].mean(axis=1)) ** 2 + KRES
        u, K = solve_u(Ke0, rows_u, cols_u, gd, free_u, fix_u, uval, N)
        eps = np.einsum("eij,ej->ei", B, u[udof])                 # element strain (Voigt)
        psi = 0.5 * np.einsum("ei,ij,ej->e", eps, C, eps)
        H = np.maximum(H, psi)
        d_new = solve_d(Kd_const, massL, tris, H, dlock, notch, N)
        change = np.max(np.abs(d_new - d))
        d = d_new
        if change < STAG_TOL:
            break
    reaction = float((K @ u)[fix_top_uy].sum())
    return u, d, H, it, reaction


def main():
    global pc_notch, fix_top_uy
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=48, help="cells per side")
    ap.add_argument("--nload", type=int, default=30, help="load increments")
    ap.add_argument("--umax", type=float, default=0.02, help="max top displacement")
    ap.add_argument("--out", type=str, default="phasefield_fracture_warmstart.png")
    args = ap.parse_args()
    if args.nload < 3:
        ap.error("--nload must be >= 3")

    nodes, tris = build_mesh(args.n)
    N = len(nodes)
    pc = precompute(nodes, tris)

    x, y = nodes[:, 0], nodes[:, 1]
    h = 1.0 / args.n
    bottom = np.where(y < 1e-9)[0]
    top = np.where(y > 1 - 1e-9)[0]
    pc_notch = np.where((np.abs(y - 0.5) < h) & (x <= 0.5))[0]   # pre-crack band
    # Dirichlet dof bookkeeping (u): bottom fully fixed, top ux=0 & uy=load
    fix = np.concatenate([2 * bottom, 2 * bottom + 1, 2 * top, 2 * top + 1])
    fix_top_uy = 2 * top + 1
    free_u = np.setdiff1d(np.arange(2 * N), fix)

    def uval_of(delta):
        v = np.zeros(2 * N); v[2 * top + 1] = delta      # top uy = delta; others 0
        return v[fix], fix

    loads = np.linspace(args.umax / args.nload, args.umax, args.nload)

    # ---- warm: continuation (carry state) ----
    uw = np.zeros(2 * N); dw = np.zeros(N); Hw = np.zeros(len(tris)); dlock = np.zeros(N)
    warm_it, reac, cmod = [], [], []
    for k, dl in enumerate(loads):
        uv, _ = uval_of(dl)
        u, d, H, it, R = staggered(pc, tris, N, uv, fix, free_u, uw, dw, Hw, dlock)
        uw, dw, Hw = u, d, H; dlock = np.maximum(dlock, d)
        warm_it.append(it); reac.append(R); cmod.append(float(d.mean()))

    # ---- cold: each load level from undamaged scratch ----
    cold_it = []
    for k, dl in enumerate(loads):
        uv, _ = uval_of(dl)
        _, _, _, it, _ = staggered(pc, tris, N, uv, fix, free_u,
                                   np.zeros(2 * N), np.zeros(N), np.zeros(len(tris)), np.zeros(N))
        cold_it.append(it)

    tot_c, tot_w = int(np.sum(cold_it)), int(np.sum(warm_it))
    print(f"\nphase-field SENT: {args.nload} load steps, mesh {args.n}x{args.n}")
    print(f"  staggered iterations  cold {tot_c}   warm(continuation) {tot_w}"
          f"   ({100 * (tot_c - tot_w) / tot_c:.0f}% fewer)")
    print(f"  peak per-step  cold {max(cold_it)}   warm {max(warm_it)}   (crack-propagation step)")

    _plot(args.out, nodes, tris, dw, uw, loads, reac, cold_it, warm_it, tot_c, tot_w)
    print(f"wrote {args.out}")


def _plot(out, nodes, tris, d, u, loads, reac, cold_it, warm_it, tot_c, tot_w):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    tr = Triangulation(nodes[:, 0], nodes[:, 1], tris)
    fig, ax = plt.subplots(2, 2, figsize=(13, 11))

    tp = ax[0, 0].tripcolor(tr, d, shading="gouraud", cmap="inferno", vmin=0, vmax=1)
    ax[0, 0].set_aspect("equal"); ax[0, 0].set_title("damage d (crack) at max load — SENT")
    fig.colorbar(tp, ax=ax[0, 0], fraction=0.046)

    sc = 0.0
    dnodes = nodes + sc * u.reshape(-1, 2)
    trd = Triangulation(dnodes[:, 0], dnodes[:, 1], tris)
    tp = ax[0, 1].tripcolor(trd, d, shading="gouraud", cmap="inferno", vmin=0, vmax=1)
    mag = np.sqrt(u[0::2] ** 2 + u[1::2] ** 2)
    tp2 = ax[0, 1].tricontour(tr, mag, levels=6, colors="c", linewidths=0.6)
    ax[0, 1].set_aspect("equal"); ax[0, 1].set_title("damage + displacement-magnitude contours")
    fig.colorbar(tp, ax=ax[0, 1], fraction=0.046)

    ax[1, 0].plot(loads, reac, "-o", ms=3, color="#1f77b4")
    ax[1, 0].set_title("load–displacement (reaction) — softening at crack growth")
    ax[1, 0].set_xlabel("top displacement"); ax[1, 0].set_ylabel("reaction force")
    ax[1, 0].grid(True, alpha=0.3)

    xs = np.arange(len(cold_it))
    ax[1, 1].plot(xs, cold_it, "-o", ms=3, color="#d62728", label="cold (from scratch)")
    ax[1, 1].plot(xs, warm_it, "-o", ms=3, color="#1f77b4", label="warm (continuation)")
    ax[1, 1].set_title(f"staggered iterations per load step\ntotal cold {tot_c} → warm {tot_w} "
                       f"({100*(tot_c-tot_w)/tot_c:.0f}% fewer)")
    ax[1, 1].set_xlabel("load step"); ax[1, 1].set_ylabel("staggered iterations")
    ax[1, 1].legend(); ax[1, 1].grid(True, alpha=0.3)

    fig.suptitle("Phase-field brittle fracture (AT2, SENT): load continuation warm-starts the "
                 "coupled staggered solve — seed for theme A (FEM keeps accuracy)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main()
