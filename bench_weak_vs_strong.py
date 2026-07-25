"""bench_weak_vs_strong.py — weak-form (FE Galerkin) vs strong-form (naive FD) on a
variable-permittivity Poisson problem with a GAA-like interface.

Why this benchmark. The referenced GAA study (Otsuki & Mori, WCCM-ECCOMAS 2026,
STS415) solves the Poisson update by *finite differences* — a strong-form,
pointwise stencil. The differentiator in pi_deeponet_fem_gaa.py is to put the
*weak/Galerkin* form at the core. This script isolates that choice at the
discretisation level, with NO neural-network optimiser in the loop (which would
confound the comparison), so the gap it shows is intrinsic to the formulation.

Problem:  -div(eps(x) grad phi) = rho,  phi = 0 on the boundary of the unit square,
with a smooth high-k core / low-k surround interface of width `delta` (Si / oxide).

Two solvers, both solved exactly (sparse direct), scored by relative L2 against a
fine FE reference:
  * weak (FE)   : P1 Galerkin. Element assembly integrates eps per element, so the
                  interface flux-continuity condition [eps dphi/dn] = 0 is built in.
  * strong (FD) : naive 5-point stencil  eps_ij (4 phi_ij - neighbours)/h^2 = rho_ij,
                  i.e. the pointwise strong form eps*lap(phi)=-rho that drops the
                  grad(eps).grad(phi) interface term — the same pointwise treatment a
                  strong-form PINN residual makes. (A flux-consistent FD with
                  harmonic face coefficients would converge; the point is that the
                  pointwise strong form does not.)

Sweeps:
  A. fix interface sharpness, refine the mesh   -> weak converges O(h^2); strong stalls
  B. fix a coarse mesh, sharpen the interface   -> strong error grows; weak stays low
Plus a spatial error map (strong-form error concentrates on the interface ring).

Run:  python3 bench_weak_vs_strong.py            (writes bench_weak_vs_strong.png)
      python3 bench_weak_vs_strong.py --help
"""
from __future__ import annotations

import argparse

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree

from pi_deeponet_fem_gaa import build_mesh, assemble  # reuse FE mesh + P1 assembly

EPS_HI, EPS_LO = 11.7, 3.9       # Si / SiO2 relative permittivity
R_IF = 0.25                      # interface (core) radius


def eps_np(nodes, delta):
    r = np.hypot(nodes[:, 0] - 0.5, nodes[:, 1] - 0.5)
    s = 0.5 * (1.0 - np.tanh((r - R_IF) / delta))     # 1 in the core, 0 outside
    return (EPS_LO + (EPS_HI - EPS_LO) * s).astype(np.float64)


def rho_np(nodes):
    return np.exp(-((nodes[:, 0] - 0.5) ** 2 + (nodes[:, 1] - 0.5) ** 2) / (2 * 0.12 ** 2))


# ----------------------------------------------------------------------------
# Solvers
# ----------------------------------------------------------------------------
def fe_weak_solve(n, delta):
    """P1 Galerkin (weak form): assemble K(eps), M; solve K_ff phi = (M rho)_f."""
    nodes, tris, on_bnd = build_mesh(n)
    K, M = assemble(nodes, tris, eps_np(nodes, delta))
    free = ~on_bnd
    b = M @ rho_np(nodes)
    phi = np.zeros(len(nodes))
    phi[free] = spla.spsolve(K[free][:, free].tocsc(), b[free])
    return nodes, phi


def grid_nodes(n):
    xs = np.linspace(0, 1, n)
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    return np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)


def fd_strong_solve(n, delta):
    """Naive strong-form 5-point stencil: eps_ij (4 phi_ij - neighbours)/h^2 = rho_ij."""
    h = 1.0 / (n - 1)
    nodes = grid_nodes(n)
    eps, rho = eps_np(nodes, delta), rho_np(nodes)
    idx = np.arange(n * n).reshape(n, n)   # idx[row(y), col(x)]
    on = np.zeros(n * n, bool)
    on[idx[0, :]] = on[idx[-1, :]] = on[idx[:, 0]] = on[idx[:, -1]] = True
    rows, cols, vals = [], [], []
    b = np.zeros(n * n)
    for j in range(n):
        for i in range(n):
            k = idx[j, i]
            if on[k]:
                rows.append(k); cols.append(k); vals.append(1.0)
                continue
            rows.append(k); cols.append(k); vals.append(4 * eps[k] / h ** 2)
            for jj, ii in ((j + 1, i), (j - 1, i), (j, i + 1), (j, i - 1)):
                rows.append(k); cols.append(idx[jj, ii]); vals.append(-eps[k] / h ** 2)
            b[k] = rho[k]
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n * n, n * n))
    return nodes, spla.spsolve(A.tocsc(), b)


def rel_to_ref(nodes, phi, tree, phi_ref):
    _, idx = tree.query(nodes)
    return float(np.linalg.norm(phi - phi_ref[idx]) / (np.linalg.norm(phi_ref[idx]) + 1e-12))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n_ref", type=int, default=49, help="fine FE reference nodes/side")
    ap.add_argument("--out", type=str, default="bench_weak_vs_strong.png")
    args = ap.parse_args()

    meshes = [7, 13, 25]                       # (n-1) divides (n_ref-1)=48
    deltas = [0.15, 0.06, 0.03, 0.015]
    delta_A = 0.03
    n_B = 13

    def ref_tree(delta):
        rn, phi = fe_weak_solve(args.n_ref, delta)
        return cKDTree(rn), phi, rn

    # ---- sweep A: refine the mesh, sharp interface ----
    treeA, refA, _ = ref_tree(delta_A)
    weakA, strongA = [], []
    for n in meshes:
        nw, pw = fe_weak_solve(n, delta_A)
        ns, ps = fd_strong_solve(n, delta_A)
        weakA.append(rel_to_ref(nw, pw, treeA, refA))
        strongA.append(rel_to_ref(ns, ps, treeA, refA))
        print(f"[A] n={n:2d} ({(n-1)**2*2:4d} tris)  weak-FE {weakA[-1]:.4f}   "
              f"strong-FD {strongA[-1]:.4f}")

    # ---- sweep B: sharpen the interface, coarse mesh ----
    weakB, strongB = [], []
    for d in deltas:
        tree, ref, _ = ref_tree(d)
        nw, pw = fe_weak_solve(n_B, d)
        ns, ps = fd_strong_solve(n_B, d)
        weakB.append(rel_to_ref(nw, pw, tree, ref))
        strongB.append(rel_to_ref(ns, ps, tree, ref))
        print(f"[B] delta={d:.3f}  weak-FE {weakB[-1]:.4f}   strong-FD {strongB[-1]:.4f}")

    # ---- spatial error map at one coarse+sharp case (dense on the coarse mesh) ----
    tree, ref, _ = ref_tree(delta_A)
    nodes_B, tris_B, _ = build_mesh(n_B)
    nw, pw = fe_weak_solve(n_B, delta_A)
    ns, ps = fd_strong_solve(n_B, delta_A)
    _, iw = tree.query(nw)
    _, is_ = tree.query(ns)
    err_w = np.abs(pw - ref[iw])    # length n_B**2, defined on nodes_B
    err_s = np.abs(ps - ref[is_])   # length n_B**2, defined on nodes_B

    _plot(args.out, meshes, weakA, strongA, deltas, weakB, strongB,
          nodes_B, tris_B, err_w, err_s, delta_A, n_B)
    print(f"wrote {args.out}")


def _plot(out, meshes, weakA, strongA, deltas, weakB, strongB,
          coarse_nodes, coarse_tris, err_w, err_s, delta_A, n_B):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    fig, ax = plt.subplots(2, 2, figsize=(13, 11))
    tri_counts = [(n - 1) ** 2 * 2 for n in meshes]

    ax[0, 0].plot(tri_counts, strongA, "-o", color="#d62728", label="strong-form (naive FD)")
    ax[0, 0].plot(tri_counts, weakA, "-o", color="#1f77b4", label="weak-form (FE Galerkin)")
    ax[0, 0].set_xscale("log"); ax[0, 0].set_yscale("log")
    ax[0, 0].set_xlabel("elements (mesh resolution)")
    ax[0, 0].set_ylabel("rel-L2 vs fine FE reference")
    ax[0, 0].set_title(f"A. refine mesh (sharp interface, delta={delta_A})\n"
                       "weak converges; strong stalls at the interface floor")
    ax[0, 0].legend(); ax[0, 0].grid(True, which="both", alpha=0.3)

    ax[0, 1].plot(deltas, strongB, "-o", color="#d62728", label="strong-form (naive FD)")
    ax[0, 1].plot(deltas, weakB, "-o", color="#1f77b4", label="weak-form (FE Galerkin)")
    ax[0, 1].set_xscale("log"); ax[0, 1].set_yscale("log"); ax[0, 1].invert_xaxis()
    ax[0, 1].set_xlabel("interface width delta  (sharper ->)")
    ax[0, 1].set_ylabel("rel-L2 vs fine FE reference")
    ax[0, 1].set_title(f"B. sharpen interface (coarse mesh, n={n_B})")
    ax[0, 1].legend(); ax[0, 1].grid(True, which="both", alpha=0.3)

    triang = Triangulation(coarse_nodes[:, 0], coarse_nodes[:, 1], coarse_tris)
    vmax = float(max(err_w.max(), err_s.max()))
    t = np.linspace(0, 2 * np.pi, 200)
    for a, err, lab in ((ax[1, 0], err_s, "strong-form (naive FD)"),
                        (ax[1, 1], err_w, "weak-form (FE Galerkin)")):
        tp = a.tripcolor(triang, err, shading="gouraud", cmap="inferno", vmin=0, vmax=vmax)
        a.plot(0.5 + R_IF * np.cos(t), 0.5 + R_IF * np.sin(t), "c--", lw=1.2,
               label="eps interface")
        a.set_aspect("equal")
        a.set_title(f"|phi - phi_ref|  {lab}\n(n={n_B}, delta={delta_A})")
        a.legend(loc="upper right", fontsize=8)
        fig.colorbar(tp, ax=a, fraction=0.046)

    fig.suptitle("Weak-form (FE) vs strong-form (FD) on a variable-eps GAA Poisson problem "
                 "— discretisation-level, no NN in the loop", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main()
