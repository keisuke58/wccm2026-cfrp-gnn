"""dd_nonisothermal_2d.py — capstone seed for research theme G: a 2-D
NON-ISOTHERMAL drift-diffusion device, fusing the 2-D/3-D self-heating heat
solve (⑰ cfet_thermal_crosstalk.py, ⑲ ...3d.py) with the 1-D non-isothermal DD
coupling (⑱ dd_nonisothermal_1d.py). This is the "2-D non-isothermal DD" next
axis flagged in RESEARCH_THEMES theme G and LITREVIEW_G §4.

Model (scaled, steady, unipolar electrons — an n-i-n resistive channel so a real
current dissipates power):
  * Electrical: nonlinear Poisson for psi (5-point) + a 2-D BOX-METHOD (finite-volume)
    Scharfetter-Gummel electron-continuity solve, coupled by a Gummel map. Bias is
    applied across x (anode x=0, cathode x=L); the y-edges are insulated.
  * Thermal: a 2-D lattice heat equation  -kappa (T_xx+T_yy) + h (T-T0) = beta Q,
    Q = J . E the electron Joule power, on P1 triangles (weak form), distributed
    substrate sink h, insulated boundaries.
  * Feedback: phonon-limited mobility mu(T)=(T/T0)^-alpha carried PER EDGE from the
    local lattice temperature, so heating lowers mobility, raises resistance, and
    degrades the current (self-heating ION roll-off) — now with a 2-D hot spot.

Self-consistency is an outer, under-relaxed electro-thermal loop (solve DD at the
current T -> Joule power -> solve heat -> update T). Idea C: a bias-continuation
warm-start cuts outer iterations; the FE/DD solve keeps accuracy.

Shows: a 2-D self-consistent lattice-temperature field with a Joule hot spot, the
self-heating current penalty vs the isothermal device, and the warm-start saving.

Honest scope: 2-D, scaled units, UNIPOLAR (electron) transport on a structured grid
box method (not the full 2-carrier Full-Newton system, which needs variable scaling),
mobility the only explicit T channel. The Poisson, continuity and heat solves are all
genuinely 2-D, but the doping throttle spans the full width, so the current (and hence
the hot spot) is a y-uniform band — a series resistor. A laterally-varying geometry
(constriction, L-channel, or the CFET footprints of ⑰/⑲) is the natural next step
that makes the field 2-D in both axes. A seed toward full 2-D non-isothermal DD. No ML.

Run:  python3 dd_nonisothermal_2d.py       (writes dd_nonisothermal_2d.png)
      python3 dd_nonisothermal_2d.py --help
"""
from __future__ import annotations

import argparse

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

SEED = 20260725
LAMBDA2 = 3e-4          # scaled Debye length squared
C0 = 40.0               # scaled contact doping (N/n_i)
XD, WD = 0.5, 0.10      # centre / width of the lightly-doped throttle (x)
KAPPA = 5e-3            # lattice heat conduction
HSINK = 0.5             # distributed substrate sink
T0 = 1.0
ALPHA = 1.5             # phonon-limited mobility exponent
BETA_Q = 3.0e-2         # Joule-power -> temperature coupling
OMEGA = 0.5             # outer under-relaxation


def bern(x):
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    big = np.abs(x) > 1e-10
    out[big] = x[big] / np.expm1(x[big])
    return out


def doping_1d(xs):
    return C0 * (1.0 - 0.97 * np.exp(-((xs - XD) / WD) ** 2))


def mobility(T):
    return (T / T0) ** (-ALPHA)


def build_grid(nx, ny):
    xs = np.linspace(0, 1, nx); ys = np.linspace(0, 1, ny)
    hx = xs[1] - xs[0]; hy = ys[1] - ys[0]
    X, Y = np.meshgrid(xs, ys, indexing="ij")         # X[i,j], i->x, j->y
    nodes = np.column_stack([X.ravel(), Y.ravel()])
    return xs, ys, hx, hy, nodes


def nid(i, j, ny):
    return i * ny + j


def heat_fe(nodes, nx, ny, hx, hy):
    """P1 stiffness K and lumped mass ml on the two-triangle split of each cell."""
    N = nx * ny
    rows, cols, vals = [], [], []
    ml = np.zeros(N)
    tris = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a = nid(i, j, ny); b = nid(i + 1, j, ny)
            c = nid(i + 1, j + 1, ny); d = nid(i, j + 1, ny)
            tris.append((a, b, c)); tris.append((a, c, d))
    for t in tris:
        p = nodes[list(t)]
        bco = np.array([p[1, 1] - p[2, 1], p[2, 1] - p[0, 1], p[0, 1] - p[1, 1]])
        cco = np.array([p[2, 0] - p[1, 0], p[0, 0] - p[2, 0], p[1, 0] - p[0, 0]])
        area = 0.5 * abs((p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1])
                         - (p[2, 0] - p[0, 0]) * (p[1, 1] - p[0, 1]))
        Ke = (np.outer(bco, bco) + np.outer(cco, cco)) / (4.0 * area)
        for a in range(3):
            ml[t[a]] += area / 3.0
            for b in range(3):
                rows.append(t[a]); cols.append(t[b]); vals.append(Ke[a, b])
    K = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    return K, ml


def edges(nx, ny):
    """List of (a, b, dir) grid edges; dir 0 = x-edge, 1 = y-edge."""
    ex = []
    for i in range(nx - 1):
        for j in range(ny):
            ex.append((nid(i, j, ny), nid(i + 1, j, ny), 0))
    for i in range(nx):
        for j in range(ny - 1):
            ex.append((nid(i, j, ny), nid(i, j + 1, ny), 1))
    return np.array(ex, dtype=np.int64)


def poisson_gummel(psi, n_k, psi_k, C, Lap, interior, dirich, psi_bc, lam2):
    """One nonlinear-Poisson Newton solve (Gummel exp update, unipolar electrons).
    The Newton step is clamped (|dpsi| <= 2) and the exp argument capped so a large
    applied bias cannot make the linearization overflow / go singular."""
    N = len(psi)
    psi = psi.copy(); psi[dirich] = psi_bc[dirich]
    for _ in range(30):
        nn = n_k * np.exp(np.clip(psi - psi_k, -40.0, 40.0))
        F = -lam2 * (Lap @ psi) - (C - nn)
        Jm = (-lam2 * Lap + sp.diags(nn)).tocsr()
        d = np.zeros(N)
        d[interior] = spla.spsolve(Jm[interior][:, interior].tocsc(), -F[interior])
        d = np.clip(d, -2.0, 2.0)                      # damped Newton (global convergence)
        psi = psi + d
        if np.max(np.abs(d[interior])) < 1e-10:
            break
    n = n_k * np.exp(np.clip(psi - psi_k, -40.0, 40.0))
    return psi, n


def continuity_box(psi, muE, edge, we, N, interior, dirich, n_bc):
    """Box-method Scharfetter-Gummel electron continuity: assemble and solve for n."""
    a = edge[:, 0]; b = edge[:, 1]
    dpsi = psi[b] - psi[a]
    Bp = bern(dpsi) * muE * we           # coefficient multiplying n_b in I_{a->b}
    Bm = bern(-dpsi) * muE * we          # coefficient multiplying n_a
    rows = np.concatenate([a, a, b, b])
    cols = np.concatenate([a, b, b, a])
    # node a balance: -Bm n_a + Bp n_b ; node b balance (opposite sign of I_ab): +Bm n_a - Bp n_b
    data = np.concatenate([-Bm, Bp, -Bp, Bm])
    A = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    rhs = np.zeros(N)
    n = np.zeros(N); n[dirich] = n_bc[dirich]
    rhs[interior] = -(A[interior][:, dirich] @ n_bc[dirich])
    n[interior] = spla.spsolve(A[interior][:, interior].tocsc(), rhs[interior])
    n = np.clip(n, 1e-30, None)
    return n


def _bc(V, nodes, C):
    left = nodes[:, 0] <= 1e-9; right = nodes[:, 0] >= 1 - 1e-9
    dirich = left | right; interior = np.where(~dirich)[0]
    psi_eq = np.arcsinh(C / 2.0); n_eq = np.exp(psi_eq)
    psi_bc = psi_eq.copy(); psi_bc[left] = psi_eq[left] + V
    return dirich, interior, psi_bc, n_eq


def dd_sweep(V, xs, ys, hx, hy, nodes, edge, Lap, muE, we, C, psi, n):
    """ONE coupled Gummel sweep: nonlinear Poisson (given n) then box-SG electron
    continuity (given psi). Returns psi, n, nodal Joule power Q, terminal current I."""
    nx = len(xs); N = nx * len(ys)
    dirich, interior, psi_bc, n_eq = _bc(V, nodes, C)
    psi, n = poisson_gummel(psi, n, psi.copy(), C, Lap, interior, dirich, psi_bc, LAMBDA2)
    n[dirich] = n_eq[dirich]
    n = continuity_box(psi, muE, edge, we, N, interior, dirich, n_eq)

    dpsi = psi[edge[:, 1]] - psi[edge[:, 0]]
    Ie = (bern(dpsi) * n[edge[:, 1]] - bern(-dpsi) * n[edge[:, 0]]) * muE * we
    q_edge = np.abs(Ie * dpsi)
    Q = np.zeros(N)
    np.add.at(Q, edge[:, 0], 0.5 * q_edge)
    np.add.at(Q, edge[:, 1], 0.5 * q_edge)
    imid = nx // 2
    xmask = (edge[:, 2] == 0) & (nodes[edge[:, 0], 0] <= xs[imid] + 1e-9) \
        & (nodes[edge[:, 1], 0] >= xs[imid] - 1e-9)
    Itot = float(np.sum(Ie[xmask]))
    return psi, n, Q, Itot


def edge_weights(edge, hx, hy):
    return np.where(edge[:, 2] == 0, hy / hx, hx / hy)   # FV face/length weights


def isothermal_dd(V, xs, ys, hx, hy, nodes, edge, Lap, we, C, psi0, n0, max_sweep=40):
    """DD solved at ambient mobility (mu at T0) — the isothermal reference current."""
    muE = np.ones(len(edge))
    psi, n = psi0.copy(), n0.copy()
    I = 0.0
    for _ in range(max_sweep):
        psi_k = psi.copy()
        psi, n, _, I = dd_sweep(V, xs, ys, hx, hy, nodes, edge, Lap, muE, we, C, psi, n)
        if np.max(np.abs(psi - psi_k)) < 1e-6:
            break
    return I


def _laplacian5(nx, ny, hx, hy):
    N = nx * ny
    rows, cols, vals = [], [], []
    for i in range(nx):
        for j in range(ny):
            k = nid(i, j, ny); diag = 0.0
            for di, dj, h in ((1, 0, hx), (-1, 0, hx), (0, 1, hy), (0, -1, hy)):
                ii, jj = i + di, j + dj
                if 0 <= ii < nx and 0 <= jj < ny:
                    w = 1.0 / h ** 2
                    rows.append(k); cols.append(nid(ii, jj, ny)); vals.append(w)
                    diag -= w
            rows.append(k); cols.append(k); vals.append(diag)
    return sp.csr_matrix((vals, (rows, cols)), shape=(N, N))


def electro_thermal(V, xs, ys, hx, hy, nodes, edge, Lap, we, C, Kk, ml, psi0, n0, T0f,
                    tol=3e-4, max_it=60):
    """Fully-coupled sweep: each outer iteration does one DD Gummel sweep (mobility
    from the current lattice T) then one heat solve (under-relaxed). DD and thermal
    fields converge together."""
    psi, n, T = psi0.copy(), n0.copy(), T0f.copy()
    Asys = (Kk + sp.diags(HSINK * ml)).tocsc()
    Itot = 0.0
    for it in range(1, max_it + 1):
        psi_k = psi.copy(); T_k = T.copy()
        muE = mobility(0.5 * (T[edge[:, 0]] + T[edge[:, 1]]))
        psi, n, Q, Itot = dd_sweep(V, xs, ys, hx, hy, nodes, edge, Lap, muE, we, C, psi, n)
        rhs = ml * (BETA_Q * Q) + HSINK * ml * T0
        T = T + OMEGA * (spla.spsolve(Asys, rhs) - T)
        if max(np.max(np.abs(psi - psi_k)), np.max(np.abs(T - T_k))) < tol:
            break
    return psi, n, T, it, Itot


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nx", type=int, default=41, help="grid nodes in x (transport)")
    ap.add_argument("--ny", type=int, default=21, help="grid nodes in y (width)")
    ap.add_argument("--vmax", type=float, default=8.0, help="max bias (units of Vt)")
    ap.add_argument("--nbias", type=int, default=14, help="bias steps")
    ap.add_argument("--out", type=str, default="dd_nonisothermal_2d.png")
    args = ap.parse_args()

    xs, ys, hx, hy, nodes = build_grid(args.nx, args.ny)
    edge = edges(args.nx, args.ny)
    Kf, ml = heat_fe(nodes, args.nx, args.ny, hx, hy)
    Kk = (KAPPA * Kf).tocsc()
    Lap = _laplacian5(args.nx, args.ny, hx, hy)       # grid-only, build once
    we = edge_weights(edge, hx, hy)
    N = args.nx * args.ny
    C = doping_1d(nodes[:, 0]); psi_eq = np.arcsinh(C / 2.0); n_eq = np.exp(psi_eq)
    T0f = np.full(N, T0)

    # equilibrium check
    _, _, Teq, _, I0 = electro_thermal(0.0, xs, ys, hx, hy, nodes, edge, Lap, we, C, Kk, ml,
                                       psi_eq, n_eq, T0f)
    print(f"[validate] V=0: current I={I0:.2e} (~0), max T rise={Teq.max()-T0:.1e} (~0)")

    biases = np.linspace(0.0, args.vmax, args.nbias)
    I_iso, I_niso, Tmax, warm_it, cold_it = [], [], [], [], []
    Tfield = None

    psw, nw = psi_eq.copy(), n_eq.copy()
    Tw = T0f.copy()
    for V in biases:
        psw, nw, Tw, it, I = electro_thermal(V, xs, ys, hx, hy, nodes, edge, Lap, we, C,
                                             Kk, ml, psw, nw, Tw)
        warm_it.append(it); I_niso.append(I); Tmax.append(Tw.max())
        # isothermal reference (mobility at T0)
        Iiso = isothermal_dd(V, xs, ys, hx, hy, nodes, edge, Lap, we, C, psi_eq, n_eq)
        I_iso.append(Iiso)
        Tfield = Tw.copy()

    for V in biases:
        _, _, _, it, _ = electro_thermal(V, xs, ys, hx, hy, nodes, edge, Lap, we, C,
                                         Kk, ml, psi_eq, n_eq, T0f)
        cold_it.append(it)

    I_iso = np.abs(np.array(I_iso)); I_niso = np.abs(np.array(I_niso))
    penalty = 100.0 * (1.0 - I_niso[-1] / max(I_iso[-1], 1e-30))
    tc, tw = int(sum(cold_it)), int(sum(warm_it))
    print(f"\n2D non-isothermal drift-diffusion: {args.nx}x{args.ny} grid, "
          f"{args.nbias} bias steps")
    print(f"  self-heating current penalty at Vmax: {penalty:.1f}% "
          f"(I_niso {I_niso[-1]:.3e} < I_iso {I_iso[-1]:.3e})")
    print(f"  max lattice temperature rise: {Tmax[-1]-T0:.3f} (x T0={T0})")
    print(f"  outer electro-thermal iters  cold {tc}  warm {tw}  "
          f"({100*(tc-tw)/max(tc,1):.0f}% fewer via bias-continuation warm-start)")

    _plot(args.out, xs, ys, args.nx, args.ny, nodes, biases, I_iso, I_niso,
          np.array(Tmax), Tfield, cold_it, warm_it, penalty, tc, tw, T0)
    print(f"wrote {args.out}")


def _plot(out, xs, ys, nx, ny, nodes, biases, I_iso, I_niso, Tmax, Tfield,
          cold_it, warm_it, penalty, tc, tw, T0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    ax[0, 0].plot(biases, I_iso, "-o", ms=3, color="#2ca02c", label="isothermal (μ at T0)")
    ax[0, 0].plot(biases, I_niso, "-o", ms=3, color="#d62728", label="non-isothermal (self-heating)")
    ax[0, 0].fill_between(biases, I_niso, I_iso, color="#d62728", alpha=0.12)
    ax[0, 0].set_xlabel("applied bias V / V_t"); ax[0, 0].set_ylabel("terminal current I")
    ax[0, 0].set_title(f"self-heating current penalty (ION roll-off)\n{penalty:.1f}% lower at Vmax")
    ax[0, 0].legend(); ax[0, 0].grid(True, alpha=0.3)

    Tg = Tfield.reshape(nx, ny)
    im = ax[0, 1].imshow(Tg.T, origin="lower", extent=[0, 1, 0, 1], cmap="inferno",
                         aspect="auto")
    fig.colorbar(im, ax=ax[0, 1], label="lattice temperature T")
    ax[0, 1].set_title("2D self-consistent lattice T at Vmax\n(Joule hot spot at the throttle)")
    ax[0, 1].set_xlabel("x (anode→cathode)"); ax[0, 1].set_ylabel("y (width)")

    ax[1, 0].plot(biases, Tmax, "-o", ms=3, color="#b5651d")
    ax[1, 0].set_xlabel("applied bias V / V_t"); ax[1, 0].set_ylabel("peak lattice T")
    ax[1, 0].set_title("Joule self-heating: peak lattice temperature vs bias")
    ax[1, 0].grid(True, alpha=0.3)

    steps = np.arange(len(cold_it))
    ax[1, 1].plot(steps, cold_it, "-o", ms=3, color="#d62728", label="cold (equilibrium)")
    ax[1, 1].plot(steps, warm_it, "-o", ms=3, color="#1f77b4", label="warm (continuation)")
    ax[1, 1].set_title(f"outer electro-thermal iterations\ntotal cold {tc} → warm {tw} "
                       f"({100*(tc-tw)/max(tc,1):.0f}% fewer)")
    ax[1, 1].set_xlabel("bias step"); ax[1, 1].set_ylabel("outer iterations")
    ax[1, 1].legend(); ax[1, 1].grid(True, alpha=0.3)

    fig.suptitle("2D non-isothermal drift-diffusion (theme G capstone): box-method SG + "
                 "2D lattice heat → self-heating hot spot and ION roll-off", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main()
