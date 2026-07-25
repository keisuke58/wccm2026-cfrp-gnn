"""cfet_thermal_crosstalk.py — 2D seed demo for research theme G (device
self-heating = electro-thermal coupling), extending the 1D reduced model
(⑯ electrothermal_selfheating.py, ⑯-dataset electrothermal_dataset.py) to a 2D
CFET cross-section, where the headline physics is DEVICE-TO-DEVICE THERMAL
CROSSTALK.

Why 2D / why CFET (LITREVIEW_G_electrothermal.md §3): a complementary FET stacks
an nFET and a pFET vertically. The stack is thermally isolated, so self-heating is
~2x a planar CMOS and — crucially — heat from one transistor raises the *other's*
lattice temperature (intra-cell thermal crosstalk), which a 1D model cannot show.

Physics (2D, steady, reduced electro-thermal — same conduction law as ⑯):
    -kappa (T_xx + T_yy) + h (T - T0) = Q(x, y; T),
    Q = [chi_n(x,y) J_n^2 + chi_p(x,y) J_p^2] / sigma(T),  sigma(T)=exp(EA(1/T0-1/T)),
on P1 triangles (weak form), damped Newton (sigma(T) makes the source nonlinear).
Boundary conditions: the SUBSTRATE edge (y=0) is a heat sink (Dirichlet T=T0); the
other three edges are insulated (natural Neumann); a weak distributed sink h models
lateral leakage to the package. Each device band (chi_n bottom, chi_p top) carries
its own current density J_n, J_p.

What it demonstrates:
  * a 2D lattice-temperature field for a CFET cross-section (top device hotter — it
    is farther from the substrate sink),
  * CROSSTALK coefficients: powering the nFET alone still heats the pFET (and vice
    versa); theta = (neighbour's rise from my heat) / (my own rise),
  * STACKED (CFET) vs PLANAR (side-by-side) layout: stacking amplifies the crosstalk
    (~1.8x here) and modestly raises peak T; the literature ~2x self-heating figure
    for CFET vs planar CMOS (IEEE 9633122) is the motivation, not a fitted result,
  * Idea C: a power (current) continuation warm-start cuts Newton iterations; FEM
    keeps accuracy.

Honest scope: reduced electro-thermal model (temperature-activated Ohmic conduction,
fixed current density per band), not non-isothermal 2-carrier drift-diffusion (that
is the ⑯-companion 1D non-isothermal DD demo and the eventual 2D coupling).
Non-dimensional units, illustrative geometry. No ML (pure physics/numerics).

Run:  python3 cfet_thermal_crosstalk.py     (writes cfet_thermal_crosstalk.png)
      python3 cfet_thermal_crosstalk.py --help
"""
from __future__ import annotations

import argparse

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

SEED = 20260725
KAPPA = 0.02
HSINK = 0.3            # weak distributed lateral loss (substrate edge is the main sink)
T0 = 1.0
EA = 15.0
NEWTON_TOL = 1e-9
NEWTON_MAX = 80


def sigma(T):
    return np.exp(EA * (1.0 / T0 - 1.0 / T))


def dsigma(T):
    return sigma(T) * EA / T ** 2


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


def assemble(nodes, tris):
    """P1 scalar stiffness K (Laplacian) and lumped mass ml on triangles."""
    N = len(nodes)
    rows, cols, vals = [], [], []
    ml = np.zeros(N)
    for t in tris:
        p = nodes[t]
        b = np.array([p[1, 1] - p[2, 1], p[2, 1] - p[0, 1], p[0, 1] - p[1, 1]])
        c = np.array([p[2, 0] - p[1, 0], p[0, 0] - p[2, 0], p[1, 0] - p[0, 0]])
        area = 0.5 * abs((p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1])
                         - (p[2, 0] - p[0, 0]) * (p[1, 1] - p[0, 1]))
        Ke = (np.outer(b, b) + np.outer(c, c)) / (4.0 * area)
        for a in range(3):
            ml[t[a]] += area / 3.0
            for bb in range(3):
                rows.append(t[a]); cols.append(t[bb]); vals.append(Ke[a, bb])
    K = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    return K, ml


def band(nodes, x0, x1, y0, y1):
    return ((nodes[:, 0] >= x0) & (nodes[:, 0] <= x1)
            & (nodes[:, 1] >= y0) & (nodes[:, 1] <= y1))


def solve_T(Kk, ml, chi_n, chi_p, Jn, Jp, free, T_init):
    """Damped Newton for the 2D reduced electro-thermal problem with a Dirichlet
    substrate sink (fixed nodes = ~free)."""
    T = T_init.copy()
    Hm = sp.diags(HSINK * ml)
    src2 = chi_n * Jn ** 2 + chi_p * Jp ** 2          # per-node current^2 weight

    def resid(TT):
        return Kk @ TT + HSINK * ml * (TT - T0) - ml * (src2 / sigma(TT))

    for it in range(1, NEWTON_MAX + 1):
        R = resid(T)
        rn = np.linalg.norm(R[free])
        if rn < NEWTON_TOL:
            return T, it - 1
        dQ = -src2 * dsigma(T) / sigma(T) ** 2         # d(src2/sigma)/dT
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


def mean_on(T, mask):
    return float(T[mask].mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=48, help="mesh divisions per side")
    ap.add_argument("--jdev", type=float, default=6.0, help="per-device current density")
    ap.add_argument("--nstep", type=int, default=40, help="power-continuation steps")
    ap.add_argument("--out", type=str, default="cfet_thermal_crosstalk.png")
    args = ap.parse_args()

    nodes, tris = build_mesh(args.n)
    K, ml = assemble(nodes, tris)
    Kk = (KAPPA * K).tocsc()
    bottom = nodes[:, 1] <= 1e-9                      # substrate heat sink (Dirichlet)
    free = ~bottom
    T0f = np.full(len(nodes), T0)

    # ---- STACKED (CFET): nFET low, pFET high, same x-column ----
    chi_n = band(nodes, 0.25, 0.75, 0.22, 0.34).astype(float)   # bottom device
    chi_p = band(nodes, 0.25, 0.75, 0.60, 0.72).astype(float)   # top device
    J = args.jdev
    T_both, _ = solve_T(Kk, ml, chi_n, chi_p, J, J, free, T0f)
    T_nonly, _ = solve_T(Kk, ml, chi_n, chi_p, J, 0.0, free, T0f)
    T_ponly, _ = solve_T(Kk, ml, chi_n, chi_p, 0.0, J, free, T0f)

    maskn = chi_n > 0; maskp = chi_p > 0
    rise_n_self = mean_on(T_nonly, maskn) - T0        # nFET's own rise (n powered)
    rise_p_self = mean_on(T_ponly, maskp) - T0        # pFET's own rise (p powered)
    rise_p_from_n = mean_on(T_nonly, maskp) - T0      # pFET heated by nFET only
    rise_n_from_p = mean_on(T_ponly, maskn) - T0      # nFET heated by pFET only
    theta_pn = rise_p_from_n / max(rise_n_self, 1e-12)
    theta_np = rise_n_from_p / max(rise_p_self, 1e-12)
    peak_stack = T_both.max()

    # ---- PLANAR (side-by-side, both near the sink) ----
    chi_a = band(nodes, 0.12, 0.42, 0.22, 0.34).astype(float)
    chi_b = band(nodes, 0.58, 0.88, 0.22, 0.34).astype(float)
    T_planar, _ = solve_T(Kk, ml, chi_a, chi_b, J, J, free, T0f)
    T_a_only, _ = solve_T(Kk, ml, chi_a, chi_b, J, 0.0, free, T0f)
    maska = chi_a > 0; maskb = chi_b > 0
    rise_a_self = mean_on(T_a_only, maska) - T0
    rise_b_from_a = mean_on(T_a_only, maskb) - T0
    theta_planar = rise_b_from_a / max(rise_a_self, 1e-12)
    peak_planar = T_planar.max()

    # ---- power (current) continuation warm-start on the stacked device ----
    Js = np.linspace(J / args.nstep, J, args.nstep)
    warm_it, cold_it, peak_curve = [], [], []
    Tw = T0f.copy()
    for Jk in Js:
        Tw, it = solve_T(Kk, ml, chi_n, chi_p, Jk, Jk, free, Tw)
        warm_it.append(it); peak_curve.append(Tw.max())
    for Jk in Js:
        _, it = solve_T(Kk, ml, chi_n, chi_p, Jk, Jk, free, T0f)
        cold_it.append(it)
    tc, tw = int(sum(cold_it)), int(sum(warm_it))

    print(f"\n2D CFET thermal crosstalk: {len(nodes)} nodes, {len(tris)} triangles")
    print(f"  STACKED (CFET): peak T {peak_stack/T0:.2f}xT0;  self-rise nFET "
          f"{rise_n_self:.3f}, pFET {rise_p_self:.3f}")
    print(f"    crosstalk theta(pFET<-nFET) {theta_pn:.2f},  theta(nFET<-pFET) {theta_np:.2f}"
          f"  (fraction of neighbour's own rise induced by the other device)")
    print(f"  PLANAR (side-by-side): peak T {peak_planar/T0:.2f}xT0;  crosstalk theta "
          f"{theta_planar:.2f}")
    print(f"  -> stacking raises peak T {peak_stack/peak_planar:.2f}x and crosstalk "
          f"{theta_pn/max(theta_planar,1e-12):.1f}x vs planar")
    print(f"  Newton iters (power continuation)  cold {tc}  warm {tw}  "
          f"({100*(tc-tw)/max(tc,1):.0f}% fewer)")

    _plot(args.out, nodes, tris, T_both, T_nonly, Js, peak_curve, cold_it, warm_it,
          theta_pn, theta_np, theta_planar, peak_stack, peak_planar, tc, tw, T0)
    print(f"wrote {args.out}")


def _plot(out, nodes, tris, T_both, T_nonly, Js, peak_curve, cold_it, warm_it,
          theta_pn, theta_np, theta_planar, peak_stack, peak_planar, tc, tw, T0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    triang = mtri.Triangulation(nodes[:, 0], nodes[:, 1], tris)
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    tpc = ax[0, 0].tripcolor(triang, T_both, shading="gouraud", cmap="inferno")
    fig.colorbar(tpc, ax=ax[0, 0], label="temperature T")
    ax[0, 0].set_title("CFET cross-section: lattice T (both devices on)\n"
                       "top pFET hotter — farther from substrate sink")
    ax[0, 0].set_xlabel("x"); ax[0, 0].set_ylabel("y (substrate sink at y=0)")
    ax[0, 0].set_aspect("equal")

    tpc2 = ax[0, 1].tripcolor(triang, T_nonly, shading="gouraud", cmap="inferno")
    fig.colorbar(tpc2, ax=ax[0, 1], label="temperature T")
    ax[0, 1].set_title("only nFET powered → pFET still heats up\n"
                       f"crosstalk θ(pFET←nFET) = {theta_pn:.2f}")
    ax[0, 1].set_xlabel("x"); ax[0, 1].set_ylabel("y")
    ax[0, 1].set_aspect("equal")

    labels = ["θ(pFET←nFET)\nstacked", "θ(nFET←pFET)\nstacked", "θ side-by-side\nplanar"]
    ax[1, 0].bar(labels, [theta_pn, theta_np, theta_planar],
                 color=["#d62728", "#d62728", "#2ca02c"])
    ax[1, 0].set_ylabel("crosstalk coefficient θ")
    ax[1, 0].set_title(f"stacking amplifies crosstalk\npeak T stacked {peak_stack/T0:.2f} "
                       f"vs planar {peak_planar/T0:.2f} (xT0)")
    ax[1, 0].grid(True, axis="y", alpha=0.3)

    xs = np.arange(len(cold_it))
    ax[1, 1].plot(xs, cold_it, "-o", ms=3, color="#d62728", label="cold (ambient)")
    ax[1, 1].plot(xs, warm_it, "-o", ms=3, color="#1f77b4", label="warm (continuation)")
    ax[1, 1].set_title(f"power-continuation warm-start\ntotal Newton cold {tc} → warm {tw} "
                       f"({100*(tc-tw)/max(tc,1):.0f}% fewer)")
    ax[1, 1].set_xlabel("power step"); ax[1, 1].set_ylabel("Newton iterations")
    ax[1, 1].legend(); ax[1, 1].grid(True, alpha=0.3)

    fig.suptitle("Weak-form FE 2D CFET self-heating (theme G): vertical stacking → "
                 "device-to-device thermal crosstalk (~1.8x planar here), continuation warm-start",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main()
