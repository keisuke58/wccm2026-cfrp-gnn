"""dd_nonisothermal_1d.py — physics-deepening seed for research theme G: a
1-D NON-ISOTHERMAL drift-diffusion device, i.e. the real self-heating coupling
(Poisson + electron/hole continuity + a lattice heat equation), rather than the
reduced temperature-activated-Ohmic model of ⑯. This is the "grow the physics"
step flagged in RESEARCH_THEMES theme G and LITREVIEW_G_electrothermal.md §4.

Coupling (scaled, steady):
  * Electrical (van Roosbroeck): nonlinear Poisson for psi + Scharfetter-Gummel
    continuity for n and p, solved by a Gummel map (reuses the validated scheme of
    dd_full_1d.py). Reverse-biased n-i-n-like resistor so a real current flows and
    dissipates power.
  * Thermal: a lattice heat equation  -kappa T'' + h (T - T0) = Q(x),  with the
    Joule/recombination power density  Q = J . E  (E = -dpsi/dx the electric field),
    weak-form P1 FE, distributed substrate sink h, insulated ends.
  * Feedback: phonon-limited mobility  mu(T) = mu0 (T/T0)^(-alpha)  (alpha~1.5), so
    heating LOWERS mobility, RAISES resistance, and DEGRADES the current — the
    hallmark self-heating signature (ION roll-off) that a reduced or isothermal model
    misses. mu enters the SG edge currents per-edge (local T).

Self-consistency is an OUTER electro-thermal loop: solve DD at the current T(x) ->
power density -> solve heat -> update T -> repeat. Idea C: a bias continuation
warm-start (ramp the applied bias, reuse the previous psi/n/p/T) cuts outer
iterations; FEM/DD keeps accuracy.

What it shows:
  * a self-consistent lattice-temperature profile driven by Joule heating,
  * the SELF-HEATING current penalty: I(non-isothermal) < I(isothermal) at the same
    bias, growing with bias (ION degradation),
  * bias-continuation warm-start reducing outer electro-thermal iterations.

Honest scope: 1-D, scaled units, drift-diffusion (no hydrodynamic/energy-transport
or ballistic correction), constant thermal properties, mobility the only explicit
T-channel (band-gap narrowing etc. omitted). A seed for the 2-D non-isothermal DD.
No ML (pure physics/numerics).

Run:  python3 dd_nonisothermal_1d.py       (writes dd_nonisothermal_1d.png)
      python3 dd_nonisothermal_1d.py --help
"""
from __future__ import annotations

import argparse

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

SEED = 20260725
LAMBDA2 = 2e-4          # scaled Debye length squared
C0 = 40.0               # scaled net doping magnitude
TAU = 1.0               # SRH lifetime (scaled)
XD, WD = 0.5, 0.05      # centre of a lightly-doped throttle region, width
# thermal / feedback
KAPPA = 5e-3            # lattice heat conduction (scaled)
HSINK = 0.5             # distributed substrate sink
T0 = 1.0                # ambient (scaled)
ALPHA = 1.5             # phonon-limited mobility exponent  mu ~ T^-alpha
BETA_Q = 5e-5           # Joule-power -> temperature coupling (scaled)
OMEGA = 0.6             # under-relaxation of the outer temperature update (stability)


def bern(x):
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    big = np.abs(x) > 1e-10
    out[big] = x[big] / np.expm1(x[big])
    return out


def doping(x):
    # n-i-n-like: high n doping at the contacts, a lightly-doped throttle in the
    # middle so a finite resistance dissipates power under bias (unipolar electrons).
    return C0 * (1.0 - 0.97 * np.exp(-((x - XD) / WD) ** 2))


def equilibrium(x):
    C = doping(x)
    psi = np.arcsinh(C / 2.0)
    return psi, np.exp(psi), np.exp(-psi)


def laplacian(N, h):
    e = np.ones(N)
    return sp.diags([e[1:], -2 * e, e[1:]], [-1, 0, 1], (N, N)) / h ** 2


def fe_heat_operator(N, h):
    """P1 stiffness K and lumped mass ml for -kappa T'' + h(T-T0) = Q (insulated ends)."""
    main = np.full(N, 2.0 / h); main[0] = main[-1] = 1.0 / h
    off = np.full(N - 1, -1.0 / h)
    K = sp.diags([off, main, off], [-1, 0, 1]).tocsc()
    ml = np.full(N, h); ml[0] = ml[-1] = h / 2
    return K, ml


def mobility(T):
    return (T / T0) ** (-ALPHA)                       # mu0 = 1 at T0


def gummel(V, x, h, psi0, n0, p0, muE, tol=1e-6, max_outer=200):
    """Gummel map with a per-edge mobility muE (length N-1) from the local lattice
    temperature. Returns (psi,n,p, outer_iters, field E, current density J(x))."""
    N = len(x)
    C = doping(x)
    L = laplacian(N, h)
    interior = np.arange(1, N - 1)
    psi_eq, n_eq, p_eq = equilibrium(x)
    psi = psi0.copy(); n = n0.copy(); p = p0.copy()
    psi[0] = psi_eq[0] + V; psi[-1] = psi_eq[-1]
    n[0] = n_eq[0]; n[-1] = n_eq[-1]; p[0] = p_eq[0]; p[-1] = p_eq[-1]

    def sg_matrix(psi, sign):
        """Vectorized tridiagonal Scharfetter-Gummel operator (mobility per edge)."""
        dpl = psi[1:] - psi[:-1]                      # (N-1,) edge potential drops
        Bp = bern(dpl) * muE; Bm = bern(-dpl) * muE   # mobility scales the edge current
        main = np.zeros(N); sup = np.zeros(N - 1); sub = np.zeros(N - 1)
        if sign > 0:                                  # electrons
            sup[:] = Bp / h                           # A[i, i+1]
            sub[:] = Bm / h                           # A[i+1, i]
            main[1:N - 1] = -(Bm[1:] + Bp[:-1]) / h   # A[i, i]
        else:                                         # holes
            sup[:] = -Bm / h
            sub[:] = Bp / h
            main[1:N - 1] = (Bp[1:] + Bm[:-1]) / h
        return sp.diags([sub, main, sup], [-1, 0, 1], (N, N)).tocsr()

    outer = 0
    for outer in range(1, max_outer + 1):
        psi_k = psi.copy(); nk = n.copy(); pk = p.copy()
        for _ in range(30):
            nn = nk * np.exp(psi - psi_k)
            pp = pk * np.exp(-(psi - psi_k))
            F = -LAMBDA2 * (L @ psi) - (pp - nn + C)
            Jm = (-LAMBDA2 * L + sp.diags(nn + pp)).tocsr()
            d = np.zeros(N)
            d[interior] = spla.spsolve(Jm[interior][:, interior].tocsc(), -F[interior])
            psi = psi + d
            if np.max(np.abs(d[interior])) < 1e-10:
                break
        n = nk * np.exp(psi - psi_k); p = pk * np.exp(-(psi - psi_k))
        n[0] = n_eq[0]; n[-1] = n_eq[-1]; p[0] = p_eq[0]; p[-1] = p_eq[-1]

        R = (n * p - 1.0) / (TAU * (n + p + 2.0))
        An = sg_matrix(psi, +1); Ap = sg_matrix(psi, -1)
        bn = h * R.copy(); bp = h * R.copy()
        n_new = n.copy(); p_new = p.copy()
        bn[interior] -= An[interior][:, [0, -1]] @ np.array([n_eq[0], n_eq[-1]])
        bp[interior] -= Ap[interior][:, [0, -1]] @ np.array([p_eq[0], p_eq[-1]])
        n_new[interior] = spla.spsolve(An[interior][:, interior].tocsc(), bn[interior])
        p_new[interior] = spla.spsolve(Ap[interior][:, interior].tocsc(), bp[interior])
        n_new = np.clip(n_new, 1e-30, None); p_new = np.clip(p_new, 1e-30, None)

        dmax = np.max(np.abs(psi - psi_k))
        n, p = n_new, p_new
        if dmax < tol:
            break

    dpl = psi[1:] - psi[:-1]
    Jn = (bern(dpl) * n[1:] - bern(-dpl) * n[:-1]) / h * muE
    Jp = (bern(-dpl) * p[1:] - bern(dpl) * p[:-1]) / h * muE
    Jedge = Jn + Jp                                   # (N-1,) edge current density
    E_edge = -(psi[1:] - psi[:-1]) / h                # electric field per edge
    return psi, n, p, outer, E_edge, Jedge


def electro_thermal(V, x, h, Kk, ml, psi_i, n_i, p_i, T_i, tol=2e-4, max_it=60):
    """Outer self-consistent electro-thermal loop at applied bias V."""
    N = len(x)
    psi, n, p, T = psi_i.copy(), n_i.copy(), p_i.copy(), T_i.copy()
    Hm = sp.diags(HSINK * ml)
    Asys = (Kk + Hm).tocsc()
    it = 0
    for it in range(1, max_it + 1):
        muE = mobility(0.5 * (T[1:] + T[:-1]))        # edge mobility from local T
        psi, n, p, _, E_edge, Jedge = gummel(V, x, h, psi, n, p, muE)
        # nodal Joule power density Q = J.E (edge -> node average), scaled by BETA_Q
        q_edge = np.abs(Jedge * E_edge)
        q_node = np.zeros(N)
        q_node[1:-1] = 0.5 * (q_edge[1:] + q_edge[:-1])
        q_node[0] = q_edge[0]; q_node[-1] = q_edge[-1]
        rhs = ml * (BETA_Q * q_node) + HSINK * ml * T0
        T_solve = spla.spsolve(Asys, rhs)
        T_new = T + OMEGA * (T_solve - T)             # under-relax for a stable loop
        dT = np.max(np.abs(T_new - T))
        T = T_new
        if dT < tol:
            break
    Jmid = float(np.mean(Jedge))
    return psi, n, p, T, it, Jmid


def isothermal_current(V, x, h):
    """Same device solved WITHOUT self-heating (mu at T0) — the reference I(V)."""
    N = len(x)
    muE = np.ones(N - 1)
    psi_eq, n_eq, p_eq = equilibrium(x)
    _, _, _, _, _, Jedge = gummel(V, x, h, psi_eq, n_eq, p_eq, muE)
    return float(np.mean(Jedge))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int, default=161, help="grid nodes")
    ap.add_argument("--vmax", type=float, default=8.0, help="max bias (units of Vt)")
    ap.add_argument("--nbias", type=int, default=20, help="bias steps")
    ap.add_argument("--out", type=str, default="dd_nonisothermal_1d.png")
    args = ap.parse_args()

    x = np.linspace(0, 1, args.N); h = x[1] - x[0]
    Kf, ml = fe_heat_operator(args.N, h)
    Kk = (KAPPA * Kf).tocsc()
    psi_eq, n_eq, p_eq = equilibrium(x)

    # equilibrium check: V=0 -> ~0 current, ~ambient temperature
    p0, n0, pp0, T00, _, J0 = electro_thermal(0.0, x, h, Kk, ml, psi_eq, n_eq, p_eq,
                                              np.full(args.N, T0))
    print(f"[validate] V=0: current J={J0:.2e} (~0), max T rise={T00.max()-T0:.1e} (~0)")

    biases = np.linspace(0.0, args.vmax, args.nbias)
    I_iso, I_niso, Tmax, warm_it, cold_it = [], [], [], [], []
    Tprof = {}

    # warm continuation (ramp bias, reuse previous state)
    psw, nw, pw, Tw = psi_eq.copy(), n_eq.copy(), p_eq.copy(), np.full(args.N, T0)
    for k, V in enumerate(biases):
        psw, nw, pw, Tw, it, J = electro_thermal(V, x, h, Kk, ml, psw, nw, pw, Tw)
        warm_it.append(it); I_niso.append(J); Tmax.append(Tw.max())
        I_iso.append(isothermal_current(V, x, h))
        if k in (len(biases) // 2, len(biases) - 1):
            Tprof[round(float(V), 2)] = Tw.copy()

    # cold (each bias from ambient/equilibrium)
    for V in biases:
        _, _, _, _, it, _ = electro_thermal(V, x, h, Kk, ml, psi_eq, n_eq, p_eq,
                                            np.full(args.N, T0))
        cold_it.append(it)

    I_iso = np.array(I_iso); I_niso = np.array(I_niso)
    penalty = 100.0 * (1.0 - I_niso[-1] / max(I_iso[-1], 1e-30))
    tc, tw = int(sum(cold_it)), int(sum(warm_it))
    print(f"\nnon-isothermal drift-diffusion: {args.N} nodes, {args.nbias} bias steps")
    print(f"  self-heating current penalty at Vmax: {penalty:.1f}% "
          f"(I_niso {I_niso[-1]:.3e} < I_iso {I_iso[-1]:.3e})")
    print(f"  max lattice temperature rise: {Tmax[-1]-T0:.3f} (x T0={T0})")
    print(f"  outer electro-thermal iters  cold {tc}  warm {tw}  "
          f"({100*(tc-tw)/max(tc,1):.0f}% fewer via bias-continuation warm-start)")

    _plot(args.out, x, biases, I_iso, I_niso, np.array(Tmax), Tprof, cold_it, warm_it,
          penalty, tc, tw, T0)
    print(f"wrote {args.out}")


def _plot(out, x, biases, I_iso, I_niso, Tmax, Tprof, cold_it, warm_it,
          penalty, tc, tw, T0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    ax[0, 0].plot(biases, I_iso, "-o", ms=3, color="#2ca02c", label="isothermal (μ at T0)")
    ax[0, 0].plot(biases, I_niso, "-o", ms=3, color="#d62728", label="non-isothermal (self-heating)")
    ax[0, 0].fill_between(biases, I_niso, I_iso, color="#d62728", alpha=0.12)
    ax[0, 0].set_xlabel("applied bias V / V_t"); ax[0, 0].set_ylabel("current density J")
    ax[0, 0].set_title(f"self-heating current penalty (ION roll-off)\n"
                       f"{penalty:.1f}% lower current at Vmax")
    ax[0, 0].legend(); ax[0, 0].grid(True, alpha=0.3)

    ax[0, 1].plot(biases, Tmax, "-o", ms=3, color="#b5651d")
    ax[0, 1].set_xlabel("applied bias V / V_t"); ax[0, 1].set_ylabel("peak lattice T")
    ax[0, 1].set_title("Joule self-heating: peak lattice temperature vs bias")
    ax[0, 1].grid(True, alpha=0.3)

    for V, T in sorted(Tprof.items()):
        ax[1, 0].plot(x, T, label=f"V={V}")
    ax[1, 0].set_xlabel("position x"); ax[1, 0].set_ylabel("lattice temperature T")
    ax[1, 0].set_title("self-consistent T(x) profile (hot spot at the throttle)")
    ax[1, 0].legend(fontsize=8); ax[1, 0].grid(True, alpha=0.3)

    xs = np.arange(len(cold_it))
    ax[1, 1].plot(xs, cold_it, "-o", ms=3, color="#d62728", label="cold (equilibrium)")
    ax[1, 1].plot(xs, warm_it, "-o", ms=3, color="#1f77b4", label="warm (continuation)")
    ax[1, 1].set_title(f"outer electro-thermal iterations\ntotal cold {tc} → warm {tw} "
                       f"({100*(tc-tw)/max(tc,1):.0f}% fewer)")
    ax[1, 1].set_xlabel("bias step"); ax[1, 1].set_ylabel("outer iterations")
    ax[1, 1].legend(); ax[1, 1].grid(True, alpha=0.3)

    fig.suptitle("Non-isothermal drift-diffusion (theme G): Poisson+SG continuity coupled "
                 "to a lattice heat equation → self-heating current penalty (ION roll-off)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main()
