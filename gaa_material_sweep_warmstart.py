"""gaa_material_sweep_warmstart.py — idea C on the exact workload of a comparative
multi-material TCAD study: a cylindrical GAA nanowire cross-section whose *channel
material* is swept {Si, Ge, GeSn, InGaAs, MoS2} across a gate-bias range, mirroring

    S. Balaji, T.S. Balaji, P. Rathinakumar, S. Karthik,
    "Comparative TCAD investigation of gate-all-around nanowire MOSFETs with
     emerging channel materials", Next Materials 13 (2026) 102743
     (Synopsys Sentaurus; identical geometry/gate-stack/doping, material swapped).

That paper runs, per figure, a grid of *independent* self-consistent solves
(5 channel materials x a V_g transfer sweep). Each one is a fresh Poisson-(drift-
diffusion) Newton/Gummel loop from scratch. This is exactly the regime idea C
amortizes: identical geometry, only eps and the carrier-screening strength change,
so a neighbouring solution is an excellent Newton initial guess.

Physics (scaled equilibrium Poisson-Boltzmann on the circular cross-section):

    -div(eps_r grad u) + kappa * sinh(u) = f_dop        in the semiconductor core
    -div(eps_r grad u)                   = 0             in the gate oxide ring
                                       u = u_gate        on the gate (outer boundary)

P1 finite elements on an unstructured *disk* mesh (weak form — the repo's core; the
curved cylindrical boundary is where FE beats a Cartesian finite-difference grid,
cf. bench_weak_vs_strong.py). eps_r is the channel permittivity; kappa is a
non-dimensional carrier-screening strength, monotone in the material's intrinsic
carrier density n_i (small-gap Ge/GeSn screen strongly, wide-gap MoS2 weakly).
Absolute kappa magnitudes are illustrative; the ordering and trends are physical.

Two warm starts stack, both pure continuation (no training — deterministic):
  * bias continuation:   within a material, solution at V_g(k-1) seeds V_g(k)
  * material transfer:   first bias of a new material is seeded by the previous
                         material's solution at the same bias (identical geometry)
cold = every (material, bias) Newton from u0 = 0.

Every solve is the exact FE solution; only the Newton iteration count changes.
FEM keeps accuracy, continuation cuts the campaign's total iterations.

Run:  python3 gaa_material_sweep_warmstart.py     (writes gaa_material_sweep_warmstart.png)
      python3 gaa_material_sweep_warmstart.py --help
"""
from __future__ import annotations

import argparse

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

SEED = 20260724
NEWTON_TOL = 1e-9
NEWTON_MAX = 60
EPS_OX = 3.9            # SiO2 gate oxide (identical gate stack across materials)
C_DOP = 1.0            # scaled light n-type body doping in the channel

# Channel materials, ordered by intrinsic carrier density n_i (300 K, cm^-3).
# eps_r are standard static permittivities; n_i drives the screening strength.
# (MoS2 eps_r ~ 4-7 depending on layer count / direction; 7.0 used here.)
MATERIALS = [
    #  name       eps_r    n_i (cm^-3)
    ("MoS2",       7.0,     1.0e5),
    ("InGaAs",    13.9,     6.3e11),
    ("Si",        11.7,     1.0e10),
    ("Ge",        16.0,     2.0e13),
    ("GeSn",      16.5,     1.5e14),
]


def kappa_of(n_i):
    """Non-dimensional screening strength, monotone in n_i, compressed to a
    well-conditioned demo range (Debye screening ~ sqrt(n_i); the exponent keeps
    the ~9-decade n_i span inside an instructive band). Si is normalised to ~1."""
    return 1.0 * (n_i / 1.0e10) ** 0.18


# ----------------------------------------------------------------------------
# Unstructured disk mesh for the cylindrical GAA cross-section
# ----------------------------------------------------------------------------
def build_disk_mesh(nr=16, nt=48, r_core=0.70, r_gate=1.0):
    """Triangulated disk: a central node, nr concentric rings, nt sectors.
    Returns nodes (N,2), tris (T,3), gate_mask (outer ring, Dirichlet),
    and elem_semi (T,) True where the element centroid is inside the core radius."""
    nodes = [(0.0, 0.0)]
    for i in range(1, nr + 1):
        r = r_gate * i / nr
        for j in range(nt):
            th = 2.0 * np.pi * j / nt
            nodes.append((r * np.cos(th), r * np.sin(th)))
    nodes = np.asarray(nodes, dtype=np.float64)

    def idx(i, j):                       # node id on ring i (1..nr), sector j
        return 1 + (i - 1) * nt + (j % nt)

    tris = []
    for j in range(nt):                  # innermost fan from the centre
        tris.append((0, idx(1, j), idx(1, j + 1)))
    for i in range(1, nr):               # quad strips between rings -> 2 tris
        for j in range(nt):
            a, b = idx(i, j), idx(i, j + 1)
            c, d = idx(i + 1, j + 1), idx(i + 1, j)
            tris.append((a, b, c))
            tris.append((a, c, d))
    tris = np.asarray(tris, dtype=np.int64)

    gate_mask = np.zeros(len(nodes), dtype=bool)
    gate_mask[[idx(nr, j) for j in range(nt)]] = True    # outer ring = gate

    cent = nodes[tris].mean(axis=1)
    elem_semi = np.hypot(cent[:, 0], cent[:, 1]) < r_core
    return nodes, tris, gate_mask, elem_semi


def assemble(nodes, tris, eps_elem, semi_elem):
    """P1 stiffness K(eps) and lumped semiconductor mass vector ML_semi."""
    N = len(nodes)
    rows, cols, vals = [], [], []
    ML = np.zeros(N)
    for t, (ia, ib, ic) in enumerate(tris):
        p = nodes[[ia, ib, ic]]
        x, y = p[:, 0], p[:, 1]
        # signed area sets the P1 gradient denominators; |area| scales K and mass
        # so element orientation (CW/CCW) does not matter.
        area = 0.5 * ((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))
        if abs(area) < 1e-14:
            continue
        b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]]) / (2 * area)
        c = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]]) / (2 * area)
        aarea = abs(area)
        ke = eps_elem[t] * aarea * (np.outer(b, b) + np.outer(c, c))
        nd = [ia, ib, ic]
        for a in range(3):
            for bb in range(3):
                rows.append(nd[a]); cols.append(nd[bb]); vals.append(ke[a, bb])
        if semi_elem[t]:
            ML[nd] += aarea / 3.0
    K = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    return K, ML


# ----------------------------------------------------------------------------
# Damped FE Newton for the nonlinear Poisson (Dirichlet gate bias on outer ring)
# ----------------------------------------------------------------------------
def newton(K, ML, kappa, fdop, u_gate, free, gate, u0):
    u = np.asarray(u0, dtype=np.float64).copy()
    u[gate] = u_gate

    def resid_full(uu):
        return K @ uu + kappa * ML * np.sinh(uu) - fdop

    hist = []
    ff = np.ix_(free, free)
    for it in range(1, NEWTON_MAX + 1):
        R = resid_full(u)
        rn = np.linalg.norm(R[free])
        hist.append(rn)
        if rn < NEWTON_TOL:
            return u, it - 1, hist
        J = (K[ff] + sp.diags(kappa * ML[free] * np.cosh(u[free]))).tocsc()
        du = spla.spsolve(J, -R[free])
        alpha = 1.0
        for _ in range(30):
            u_try = u.copy(); u_try[free] = u[free] + alpha * du
            if np.linalg.norm(resid_full(u_try)[free]) < rn:
                break
            alpha *= 0.5
        u[free] = u[free] + alpha * du
    return u, NEWTON_MAX, hist


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nr", type=int, default=16, help="radial rings")
    ap.add_argument("--nt", type=int, default=48, help="angular sectors")
    ap.add_argument("--nbias", type=int, default=9, help="gate-bias points per material")
    ap.add_argument("--vmax", type=float, default=8.0, help="max scaled gate bias u_gate")
    ap.add_argument("--out", type=str, default="gaa_material_sweep_warmstart.png")
    args = ap.parse_args()
    if args.nbias < 2:
        ap.error("--nbias must be >= 2 (a transfer sweep needs multiple bias points)")

    nodes, tris, gate, elem_semi = build_disk_mesh(args.nr, args.nt)
    free = ~gate
    N = len(nodes)
    biases = np.linspace(0.0, args.vmax, args.nbias)

    # Pre-assemble K(eps) and ML per material (eps differs only in the core).
    per_mat = []
    for name, eps_r, n_i in MATERIALS:
        eps_elem = np.where(elem_semi, eps_r, EPS_OX)
        K, ML = assemble(nodes, tris, eps_elem, elem_semi)
        fdop = ML * C_DOP
        per_mat.append((name, K.tocsr(), ML, fdop, kappa_of(n_i)))

    # ---- cold campaign: every (material, bias) Newton from u0 = 0 ----
    cold_iters = np.zeros((len(MATERIALS), args.nbias), dtype=int)
    sol = {}                    # (m,k) -> converged u   (reused by the warm run)
    charge = np.zeros((len(MATERIALS), args.nbias))     # transfer-like observable
    for m, (name, K, ML, fdop, kap) in enumerate(per_mat):
        for k, vg in enumerate(biases):
            u, it, _ = newton(K, ML, kap, fdop, vg, free, gate, np.zeros(N))
            cold_iters[m, k] = it
            sol[(m, k)] = u
            charge[m, k] = float(np.sum(kap * ML * np.sinh(u)))   # integrated mobile charge

    # ---- warm campaign: bias continuation + cross-material transfer ----
    warm_iters = np.zeros((len(MATERIALS), args.nbias), dtype=int)
    prev_mat_first = None       # previous material's u at bias 0 (transfer seed)
    for m, (name, K, ML, fdop, kap) in enumerate(per_mat):
        u_prev_bias = None
        for k, vg in enumerate(biases):
            if k == 0:
                u0 = prev_mat_first if prev_mat_first is not None else np.zeros(N)
            else:
                u0 = u_prev_bias
            u, it, _ = newton(K, ML, kap, fdop, vg, free, gate, u0)
            warm_iters[m, k] = it
            u_prev_bias = u
            if k == 0:
                prev_mat_first = u.copy()

    tot_cold, tot_warm = int(cold_iters.sum()), int(warm_iters.sum())
    print(f"\nCampaign = {len(MATERIALS)} materials x {args.nbias} bias points "
          f"= {len(MATERIALS) * args.nbias} self-consistent FE solves")
    print(f"  total Newton iterations   cold {tot_cold:4d}   warm {tot_warm:4d}"
          f"   ({100 * (tot_cold - tot_warm) / tot_cold:.0f}% fewer)")
    for m, (name, *_ ) in enumerate(per_mat):
        print(f"  {name:7s}  cold {cold_iters[m].sum():3d}   warm {warm_iters[m].sum():3d}")

    # residual traces at a hard point (highest bias, strongest screener = last material)
    mh = len(MATERIALS) - 1
    Kh, MLh, fdoph, kaph = per_mat[mh][1], per_mat[mh][2], per_mat[mh][3], per_mat[mh][4]
    _, _, res_cold = newton(Kh, MLh, kaph, fdoph, biases[-1], free, gate, np.zeros(N))
    _, _, res_warm = newton(Kh, MLh, kaph, fdoph, biases[-1], free, gate, sol[(mh, args.nbias - 2)])

    _plot(args.out, nodes, tris, elem_semi, sol[(2, args.nbias - 1)],
          biases, charge, cold_iters, warm_iters, res_cold, res_warm,
          [m[0] for m in MATERIALS], tot_cold, tot_warm)
    print(f"wrote {args.out}")


def _plot(out, nodes, tris, elem_semi, u_repr, biases, charge,
          cold_iters, warm_iters, res_cold, res_warm, names, tot_cold, tot_warm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    triang = Triangulation(nodes[:, 0], nodes[:, 1], tris)
    fig, ax = plt.subplots(2, 3, figsize=(17, 10))

    # (0,0) material map of the cross-section (core vs gate-oxide ring)
    ax[0, 0].tripcolor(triang, facecolors=elem_semi.astype(float),
                       cmap="Pastel1", edgecolors="k", linewidth=0.15)
    ax[0, 0].set_aspect("equal")
    ax[0, 0].set_title("cylindrical GAA cross-section\n(core = channel, ring = gate oxide)")

    # (0,1) representative converged potential (Si at max bias)
    tp = ax[0, 1].tripcolor(triang, u_repr, shading="gouraud", cmap="viridis")
    ax[0, 1].set_aspect("equal")
    ax[0, 1].set_title("converged FE potential u  (Si, max V_g)")
    fig.colorbar(tp, ax=ax[0, 1], fraction=0.046)

    # (0,2) transfer-like curves: integrated mobile charge vs gate bias, per material
    for m, nm in enumerate(names):
        ax[0, 2].plot(biases, charge[m], "-o", ms=3, label=nm)
    ax[0, 2].set_title("material-dependent electrostatics\n(integrated mobile charge vs gate bias)")
    ax[0, 2].set_xlabel("scaled gate bias u_gate"); ax[0, 2].set_ylabel("integrated charge")
    ax[0, 2].legend(fontsize=8); ax[0, 2].grid(True, alpha=0.3)

    # (1,0) Newton residual at the hardest point (highest bias, strongest screener)
    ax[1, 0].semilogy(range(len(res_cold)), res_cold, "-o", color="#d62728", label="cold (u0=0)")
    ax[1, 0].semilogy(range(len(res_warm)), res_warm, "-o", color="#1f77b4",
                      label="warm (continuation)")
    ax[1, 0].set_title(f"Newton residual, hardest solve ({names[-1]}, max V_g)")
    ax[1, 0].set_xlabel("Newton iteration"); ax[1, 0].set_ylabel("||R||")
    ax[1, 0].legend(); ax[1, 0].grid(True, which="both", alpha=0.3)

    # (1,1) total Newton iterations per material, cold vs warm
    gx = np.arange(len(names))
    ax[1, 1].bar(gx - 0.2, cold_iters.sum(axis=1), 0.4, color="#d62728", label="cold")
    ax[1, 1].bar(gx + 0.2, warm_iters.sum(axis=1), 0.4, color="#1f77b4", label="warm")
    ax[1, 1].set_xticks(gx); ax[1, 1].set_xticklabels(names, rotation=20)
    ax[1, 1].set_ylabel("total Newton iterations (bias sweep)")
    ax[1, 1].set_title("per-material campaign cost")
    ax[1, 1].legend()

    # (1,2) campaign total
    ax[1, 2].bar([0, 1], [tot_cold, tot_warm], color=["#d62728", "#1f77b4"])
    ax[1, 2].set_xticks([0, 1]); ax[1, 2].set_xticklabels(["cold", "warm\n(continuation)"])
    ax[1, 2].set_ylabel("total Newton iterations")
    red = 100 * (tot_cold - tot_warm) / tot_cold
    ax[1, 2].set_title(f"whole material x bias campaign\n{red:.0f}% fewer iterations "
                       f"(every solve exact FE)")
    for i, v in enumerate([tot_cold, tot_warm]):
        ax[1, 2].text(i, v + 1, str(v), ha="center", fontsize=11)

    fig.suptitle("Multi-material GAA nanowire TCAD sweep (cf. Balaji et al., Next Materials 2026): "
                 "warm-start/continuation amortizes the material x bias campaign — "
                 "FEM keeps accuracy", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main()
