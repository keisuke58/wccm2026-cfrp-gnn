"""cfet_thermal_crosstalk_3d.py — 3D seed demo for research theme G (device
self-heating), extending the 2D CFET crosstalk (⑰ cfet_thermal_crosstalk.py) to a
full 3D cell layout on linear tetrahedra. The 3D-only payoff is directional:
VERTICAL (intra-cell, nFET-under-pFET) thermal crosstalk vs LATERAL (inter-cell,
neighbouring stack) thermal crosstalk — a distinction a 2D cross-section cannot make.

Why 3D (LITREVIEW_G_electrothermal.md §3): in a real CFET layout each cell stacks an
nFET under a pFET, and cells sit side by side. Heat couples strongly UP the stack
(thin inter-device dielectric, both far from the substrate) but weakly ACROSS to the
neighbouring cell. Quantifying that anisotropy needs the third dimension.

Physics (3D, steady, reduced electro-thermal — same conduction law as ⑯/⑰):
    -kappa (T_xx + T_yy + T_zz) + h (T - T0) = Q(x,y,z; T),
    Q = sum_dev chi_dev(x,y,z) J_dev^2 / sigma(T),  sigma(T) = exp(EA (1/T0 - 1/T)),
on P1 tetrahedra (weak form, constant-gradient elements), damped Newton. The SUBSTRATE
face z=0 is a heat sink (Dirichlet T=T0); the other five faces are insulated (natural
Neumann) with a weak distributed sink h for package leakage. z is the stacking axis.

Layout: two CFET cells side by side in x, each an nFET slab (low z) under a pFET slab
(high z), same x-y footprint within the cell:
    cell A  x in [0.12,0.42],  cell B  x in [0.58,0.88];  both y in [0.30,0.70]
    nFET z in [0.20,0.32],     pFET z in [0.60,0.72].

What it shows:
  * a 3D lattice-temperature field (x-z and x-y slices via the structured node grid),
  * VERTICAL crosstalk theta_vert = (pFET_A rise from nFET_A) / (nFET_A self-rise) vs
    LATERAL crosstalk theta_lat = (nFET_B rise from nFET_A) / (nFET_A self-rise), by
    powering nFET_A alone — vertical >> lateral,
  * Idea C: power (current) continuation warm-start cuts Newton iterations; FEM keeps
    accuracy.

Honest scope: reduced electro-thermal model (temperature-activated Ohmic conduction,
fixed per-device current density), not non-isothermal 2-carrier drift-diffusion (⑱ is
the 1D non-isothermal DD seed; the 2D/3D coupling is the documented next axis).
Non-dimensional units, illustrative geometry, coarse structured mesh. No ML.

Run:  python3 cfet_thermal_crosstalk_3d.py     (writes cfet_thermal_crosstalk_3d.png)
      python3 cfet_thermal_crosstalk_3d.py --help
"""
from __future__ import annotations

import argparse

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

SEED = 20260725
KAPPA = 0.02
HSINK = 0.3
T0 = 1.0
EA = 15.0
NEWTON_TOL = 1e-9
NEWTON_MAX = 80


def sigma(T):
    return np.exp(EA * (1.0 / T0 - 1.0 / T))


def dsigma(T):
    return sigma(T) * EA / T ** 2


def build_tet_mesh(n):
    lin = np.linspace(0, 1, n)
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing="ij")
    nodes = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    def nid(i, j, k):
        return (i * n + j) * n + k

    order = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
             (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    hex_tets = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
                (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)]
    tets = []
    for i in range(n - 1):
        for j in range(n - 1):
            for k in range(n - 1):
                c = [nid(i + di, j + dj, k + dk) for di, dj, dk in order]
                for a, b, d, e in hex_tets:
                    tets.append((c[a], c[b], c[d], c[e]))
    return nodes, np.array(tets, dtype=np.int64)


def assemble(nodes, tets):
    """P1 scalar stiffness K (Laplacian) and lumped mass ml on linear tetrahedra."""
    N = len(nodes)
    rows, cols, vals = [], [], []
    ml = np.zeros(N)
    for t in tets:
        p = nodes[t]
        M = np.column_stack([np.ones(4), p])          # [1 x y z] per node
        detM = np.linalg.det(M)
        vol = abs(detM) / 6.0
        if vol < 1e-14:
            continue
        grads = np.linalg.inv(M)[1:, :].T             # (4,3): grad of each shape fn
        Ke = vol * (grads @ grads.T)
        for a in range(4):
            ml[t[a]] += vol / 4.0
            for b in range(4):
                rows.append(t[a]); cols.append(t[b]); vals.append(Ke[a, b])
    K = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    return K, ml


def slab(nodes, x0, x1, y0, y1, z0, z1):
    return ((nodes[:, 0] >= x0) & (nodes[:, 0] <= x1)
            & (nodes[:, 1] >= y0) & (nodes[:, 1] <= y1)
            & (nodes[:, 2] >= z0) & (nodes[:, 2] <= z1)).astype(float)


def solve_T(Kk, ml, src2, free, T_init):
    """Damped Newton for the 3D reduced electro-thermal problem (Dirichlet sink)."""
    T = T_init.copy()
    Hm = sp.diags(HSINK * ml)

    def resid(TT):
        return Kk @ TT + HSINK * ml * (TT - T0) - ml * (src2 / sigma(TT))

    for it in range(1, NEWTON_MAX + 1):
        R = resid(T)
        rn = np.linalg.norm(R[free])
        if rn < NEWTON_TOL:
            return T, it - 1
        dQ = -src2 * dsigma(T) / sigma(T) ** 2
        Jac = (Kk + Hm - sp.diags(ml * dQ)).tocsc()
        dT = np.zeros_like(T)
        dT[free] = spla.spsolve(Jac[free][:, free], -R[free])
        a = 1.0
        for _ in range(40):
            if np.linalg.norm(resid(T + a * dT)[free]) < rn:
                break
            a *= 0.5
        T = T + a * dT
    return T, NEWTON_MAX


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=17, help="nodes per side (structured cube)")
    ap.add_argument("--jdev", type=float, default=6.0, help="per-device current density")
    ap.add_argument("--nstep", type=int, default=30, help="power-continuation steps")
    ap.add_argument("--out", type=str, default="cfet_thermal_crosstalk_3d.png")
    ap.add_argument("--view", type=str, default="cfet_thermal_crosstalk_3d_view.png",
                    help="extra true-3D perspective render (volumetric temperature)")
    args = ap.parse_args()

    nodes, tets = build_tet_mesh(args.n)
    K, ml = assemble(nodes, tets)
    Kk = (KAPPA * K).tocsc()
    bottom = nodes[:, 2] <= 1e-9                       # substrate sink face z=0
    free = ~bottom
    T0f = np.full(len(nodes), T0)

    # device slabs
    nA = slab(nodes, 0.12, 0.42, 0.30, 0.70, 0.20, 0.32)
    pA = slab(nodes, 0.12, 0.42, 0.30, 0.70, 0.60, 0.72)
    nB = slab(nodes, 0.58, 0.88, 0.30, 0.70, 0.20, 0.32)
    pB = slab(nodes, 0.58, 0.88, 0.30, 0.70, 0.60, 0.72)
    J = args.jdev
    mnA, mpA, mnB = nA > 0, pA > 0, nB > 0

    # power ONLY nFET_A -> measure vertical (pFET_A) and lateral (nFET_B) crosstalk
    T_nA, _ = solve_T(Kk, ml, nA * J ** 2, free, T0f)
    self_nA = T_nA[mnA].mean() - T0
    rise_pA = T_nA[mpA].mean() - T0                   # vertical (intra-cell)
    rise_nB = T_nA[mnB].mean() - T0                   # lateral (inter-cell)
    theta_vert = rise_pA / max(self_nA, 1e-12)
    theta_lat = rise_nB / max(self_nA, 1e-12)

    # all four devices on -> full field + peak
    src_all = (nA + pA + nB + pB) * J ** 2
    T_all, _ = solve_T(Kk, ml, src_all, free, T0f)
    peak = T_all.max()

    # power (current) continuation warm-start on the full layout
    Js = np.linspace(J / args.nstep, J, args.nstep)
    warm_it, cold_it = [], []
    Tw = T0f.copy()
    for Jk in Js:
        Tw, it = solve_T(Kk, ml, (nA + pA + nB + pB) * Jk ** 2, free, Tw)
        warm_it.append(it)
    for Jk in Js:
        _, it = solve_T(Kk, ml, (nA + pA + nB + pB) * Jk ** 2, free, T0f)
        cold_it.append(it)
    tc, tw = int(sum(cold_it)), int(sum(warm_it))

    print(f"\n3D CFET thermal crosstalk: {len(nodes)} nodes, {len(tets)} tetrahedra")
    print(f"  peak T (all devices on): {peak/T0:.2f} x T0;  nFET_A self-rise {self_nA:.3f}")
    print(f"  VERTICAL crosstalk (pFET_A <- nFET_A, intra-cell): theta_vert = {theta_vert:.2f}")
    print(f"  LATERAL  crosstalk (nFET_B <- nFET_A, inter-cell): theta_lat  = {theta_lat:.2f}")
    print(f"  -> vertical stacking couples {theta_vert/max(theta_lat,1e-12):.1f}x more than "
          f"lateral separation (3D-only anisotropy)")
    print(f"  Newton iters (power continuation)  cold {tc}  warm {tw}  "
          f"({100*(tc-tw)/max(tc,1):.0f}% fewer)")

    _plot(args.out, args.n, nodes, T_all, T_nA, theta_vert, theta_lat, self_nA,
          peak, Js, cold_it, warm_it, tc, tw, T0)
    print(f"wrote {args.out}")
    if args.view:
        devices = dict(nA=(0.12, 0.42, 0.30, 0.70, 0.20, 0.32),
                       pA=(0.12, 0.42, 0.30, 0.70, 0.60, 0.72),
                       nB=(0.58, 0.88, 0.30, 0.70, 0.20, 0.32),
                       pB=(0.58, 0.88, 0.30, 0.70, 0.60, 0.72))
        _plot3d(args.view, nodes, T_all, T_nA, devices, theta_vert, theta_lat, T0)
        print(f"wrote {args.view}")


def _plot(out, n, nodes, T_all, T_nA, theta_vert, theta_lat, self_nA, peak,
          Js, cold_it, warm_it, tc, tw, T0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lin = np.linspace(0, 1, n)
    Tg = T_all.reshape(n, n, n)                        # [i(x), j(y), k(z)]
    Tng = T_nA.reshape(n, n, n)
    jmid = n // 2
    kp = int(np.argmin(np.abs(lin - 0.66)))           # pFET height slice
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    # x-z slice at y=mid (all devices on): shows the vertical stack + substrate sink
    im0 = ax[0, 0].imshow(Tg[:, jmid, :].T, origin="lower", extent=[0, 1, 0, 1],
                          cmap="inferno", aspect="equal")
    fig.colorbar(im0, ax=ax[0, 0], label="temperature T")
    ax[0, 0].set_title("x-z slice (y=0.5), all devices on\ntwo stacks; substrate sink at z=0")
    ax[0, 0].set_xlabel("x"); ax[0, 0].set_ylabel("z (stacking axis)")

    # x-z slice, ONLY nFET_A powered: vertical vs lateral heat spread
    im1 = ax[0, 1].imshow(Tng[:, jmid, :].T, origin="lower", extent=[0, 1, 0, 1],
                          cmap="inferno", aspect="equal")
    fig.colorbar(im1, ax=ax[0, 1], label="temperature T")
    ax[0, 1].set_title("only nFET_A powered (x-z, y=0.5)\nheat rises to pFET_A, little reaches cell B")
    ax[0, 1].set_xlabel("x"); ax[0, 1].set_ylabel("z")

    ax[1, 0].bar(["θ_vert\n(pFET_A←nFET_A)\nintra-cell", "θ_lat\n(nFET_B←nFET_A)\ninter-cell"],
                 [theta_vert, theta_lat], color=["#d62728", "#2ca02c"])
    ax[1, 0].set_ylabel("crosstalk coefficient θ")
    ax[1, 0].set_title(f"3D anisotropy: vertical couples "
                       f"{theta_vert/max(theta_lat,1e-12):.1f}x more than lateral\n"
                       f"peak T (all on) {peak/T0:.2f} xT0")
    ax[1, 0].grid(True, axis="y", alpha=0.3)

    xs = np.arange(len(cold_it))
    ax[1, 1].plot(xs, cold_it, "-o", ms=3, color="#d62728", label="cold (ambient)")
    ax[1, 1].plot(xs, warm_it, "-o", ms=3, color="#1f77b4", label="warm (continuation)")
    ax[1, 1].set_title(f"power-continuation warm-start\ntotal Newton cold {tc} → warm {tw} "
                       f"({100*(tc-tw)/max(tc,1):.0f}% fewer)")
    ax[1, 1].set_xlabel("power step"); ax[1, 1].set_ylabel("Newton iterations")
    ax[1, 1].legend(); ax[1, 1].grid(True, alpha=0.3)

    fig.suptitle("Weak-form FE 3D CFET self-heating (theme G): vertical (intra-cell) vs "
                 "lateral (inter-cell) thermal crosstalk — a 3D-only anisotropy",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130)


def _box_edges(ax, b, color="#00bfff", lw=1.2):
    """Wireframe of an axis-aligned box b=(x0,x1,y0,y1,z0,z1)."""
    x0, x1, y0, y1, z0, z1 = b
    corners = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    for a, c in edges:
        ax.plot(*zip(corners[a], corners[c]), color=color, lw=lw, alpha=0.9)


def _plot3d(out, nodes, T_all, T_nA, devices, theta_vert, theta_lat, T0):
    """True 3D perspective render: hot nodes as a volumetric scatter (colour = T),
    device slabs as wireframe boxes, substrate sink plane at z=0."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

    fig = plt.figure(figsize=(15, 7))

    def render(ax, T, title, thr):
        hot = T > T0 + thr
        p = nodes[hot]; tv = T[hot]
        rng = max(T_all.max() - T0, 1e-9)
        sizes = 12 + 70 * (tv - T0) / rng          # hotter nodes render larger
        sc = ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=tv, cmap="inferno",
                        s=sizes, alpha=0.72, vmin=T0, vmax=T_all.max(),
                        edgecolors="none", depthshade=True)
        for name, b in devices.items():
            col = "#1f77b4" if name.startswith("n") else "#e377c2"
            _box_edges(ax, b, color=col)
        # substrate sink plane at z=0
        xx, yy = np.meshgrid([0, 1], [0, 1])
        ax.plot_surface(xx, yy, np.zeros_like(xx), color="#4444aa", alpha=0.12)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z (stack)")
        ax.set_title(title, fontsize=11)
        ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=18, azim=-58)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
        return sc

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    render(ax1, T_all, "all devices on — two CFET stacks glow;\nsubstrate (z=0) stays cool",
           0.10)
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    sc = render(ax2, T_nA, f"only nFET_A on — heat climbs to pFET_A (θ_vert={theta_vert:.2f})\n"
                           f"barely reaches cell B (θ_lat={theta_lat:.2f})", 0.04)
    fig.colorbar(sc, ax=[ax1, ax2], shrink=0.6, label="temperature T", pad=0.02)
    fig.suptitle("3D CFET self-heating — volumetric view: vertical (intra-cell) vs lateral "
                 "(inter-cell) thermal crosstalk\n(blue=nFET, pink=pFET wireframes; "
                 "hot nodes coloured by T)", fontsize=12)
    fig.savefig(out, dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main()
