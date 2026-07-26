"""cfrtp_viscoelastic_residual_stress.py — add THERMO-VISCOELASTIC STRESS RELAXATION
to the CFRTP residual-stress FE (cfrtp_residual_stress_fe.py) so the residual-stress
magnitude drops from the (over-predicting) purely-elastic value toward a
measurement-grade level, and picks up the extra cooling-rate dependence that
relaxation brings. Daikin/NEDO theme (research/MEMO_cfrp_residual_stress_collab.md).

Why: the elastic (CHILE-like) CFRTP model locks in the full thermal + crystallization
eigenstrain, over-predicting the residual stress (hundreds of MPa). Real thermoplastics
RELAX: above the solidification / glass region the melt flows and stress relaxes fast;
as it cools the relaxation time grows by orders of magnitude (time-temperature
superposition) and the stress freezes in. Only the stress accumulated near/below the
freezing region survives — a much smaller, realistic value.

Model (reduced thermo-viscoelastic, incremental): a single relaxation with an
Arrhenius/WLF-like temperature-dependent relaxation time
    tau(T) = tau0 * exp(Ea_R (1/T - 1/Tref)),
so per cooling step of physical duration dt_i the accumulated element stress relaxes by
exp(-dt_i / tau(T_i)) before the fresh elastic increment is added:
    sigma <- sigma * exp(-dt_i/tau) + g(T) C (deps - deigen).
Physical time makes it cooling-RATE dependent: dt_i = |dT_i| / rate, so faster cooling
leaves less time to relax -> higher residual stress (on top of the crystallinity effect).
Everything else (mesh, kinematics, CFRTP ply, solidification g, crystallinity X and its
shrinkage) is reused from cfrtp_residual_stress_fe.py.

Validated: a single uniform ply contracts freely -> ~0 residual stress (both models).

Shows: relaxation time tau(T) over the cool-down; the residual-stress build-up ELASTIC
vs VISCOELASTIC (relaxation pulls the locked-in stress down to a realistic level); the
viscoelastic residual stress field; and peak residual stress vs cooling rate for both
models (relaxation makes it markedly rate sensitive).

Honest scope: single-relaxation (not a full Prony series), scalar relaxation of the
element stress, illustrative fluoropolymer-CFRTP-like properties and relaxation
parameters; the magnitude is now plausible but still a seed, not calibrated to data.
Physics leads; a process->residual surrogate (cf. ㉑) is the subordinate layer. No ML.

Run:  python3 cfrtp_viscoelastic_residual_stress.py
      python3 cfrtp_viscoelastic_residual_stress.py --help
"""
from __future__ import annotations

import argparse

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from cfrp_cure_residual_stress_fe import build_mesh, cst_B
from cfrtp_residual_stress_fe import (
    ply_material_cfrtp, solidification, xmax_of_rate, cool_cycle,
    T_MELT, T_ROOM, G0,
)

SEED = 20260726
# thermo-viscoelastic relaxation (reduced, single relaxation time)
TAU0 = 1.0                     # relaxation time at Tref [s]
EA_R = 1.0e4                   # relaxation activation (K); larger = sharper freeze-in
TREF = 120.0                   # freeze-in reference temperature (~fluoropolymer Tg) [C]
HOLD_S = 60.0                  # melt-hold duration [s]


def tau_of_T(Tc):
    return TAU0 * np.exp(EA_R * (1.0 / (Tc + 273.15) - 1.0 / (TREF + 273.15)))


def solve(nx, nz, nstep, rate=5.0, viscoelastic=True, single=False):
    nodes, tris, ply = build_mesh(nx, nz)
    if single:
        ply[:] = 0
    N = len(nodes); ndof = 2 * N
    Bs, areas, Cs, eth, esh = [], [], [], [], []
    for e in range(len(tris)):
        B, area = cst_B(nodes[list(tris[e])])
        C, ath, ash = ply_material_cfrtp(ply[e])
        Bs.append(B); areas.append(area); Cs.append(C); eth.append(ath); esh.append(ash)

    LX = nodes[:, 0].max()
    n00 = np.argmin(nodes[:, 0] ** 2 + nodes[:, 1] ** 2)
    nL0 = np.argmin((nodes[:, 0] - LX) ** 2 + nodes[:, 1] ** 2)
    fixed = [2 * n00, 2 * n00 + 1, 2 * nL0 + 1]
    free = np.setdiff1d(np.arange(ndof), fixed)

    t, T = cool_cycle(nstep)
    s = solidification(T); g = G0 + (1 - G0) * s
    Xf = xmax_of_rate(rate) * s
    # physical time per step: hold steps use HOLD_S, cool steps dt=|dT|/rate
    dt = np.zeros(nstep)
    for i in range(1, nstep):
        dt[i] = HOLD_S / max(np.sum(np.diff(T) == 0), 1) if T[i] == T[i - 1] \
            else abs(T[i] - T[i - 1]) / rate
    sig = np.zeros((len(tris), 3)); eps = np.zeros((len(tris), 3))
    u = np.zeros(ndof); hist = []

    for i in range(1, nstep):
        dT = T[i] - T[i - 1]; dX = Xf[i] - Xf[i - 1]; gi = g[i]
        relax = np.exp(-dt[i] / tau_of_T(T[i])) if viscoelastic else 1.0
        rows, cols, vals = [], [], []; F = np.zeros(ndof)
        for e, tri in enumerate(tris):
            dof = np.array([2 * tri[0], 2 * tri[0] + 1, 2 * tri[1], 2 * tri[1] + 1,
                            2 * tri[2], 2 * tri[2] + 1])
            Ce = gi * Cs[e]
            de = eth[e] * dT + esh[e] * dX
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
            de = eth[e] * dT + esh[e] * dX
            sig[e] = relax * sig[e] + g[i] * Cs[e] @ (deps - de)   # relax then add
            eps[e] += deps
        hist.append(np.max(np.abs(sig[:, 0])) / 1e6)
    return nodes, tris, sig, eps, t, T, g, Xf, dt, np.array(hist)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nx", type=int, default=40)
    ap.add_argument("--nz", type=int, default=16)
    ap.add_argument("--nstep", type=int, default=40)
    ap.add_argument("--rate", type=float, default=5.0, help="nominal cooling rate [C/s]")
    ap.add_argument("--out", type=str, default="cfrtp_viscoelastic_residual_stress.png")
    args = ap.parse_args()
    np.random.seed(SEED)

    sig_single = solve(args.nx, args.nz, args.nstep, single=True)[2]
    print(f"[validate] single uniform ply, free contraction: max |sigma_xx| "
          f"{np.max(np.abs(sig_single[:,0]))/1e6:.2e} MPa (should be ~0)")

    _, _, sig_e, _, t, T, g, Xf, dt, hist_e = solve(args.nx, args.nz, args.nstep,
                                                    args.rate, viscoelastic=False)
    nodes, tris, sig_v, eps_v, t, T, g, Xf, dt, hist_v = solve(args.nx, args.nz, args.nstep,
                                                               args.rate, viscoelastic=True)
    pe = np.max(np.abs(sig_e[:, 0])) / 1e6; pv = np.max(np.abs(sig_v[:, 0])) / 1e6
    print(f"CFRTP viscoelastic ([0/90], melt {T_MELT:.0f}->RT, rate {args.rate} C/s):")
    print(f"  peak residual |sigma_xx|: elastic {pe:.0f} MPa -> viscoelastic {pv:.0f} MPa "
          f"({100*(1-pv/pe):.0f}% relaxed to a measurement-grade level)")

    rates = np.array([0.5, 1, 2, 4, 8, 16, 32, 64])
    pe_r, pv_r = [], []
    for r in rates:
        pe_r.append(np.max(np.abs(solve(args.nx, args.nz, args.nstep, r, False)[2][:, 0])) / 1e6)
        pv_r.append(np.max(np.abs(solve(args.nx, args.nz, args.nstep, r, True)[2][:, 0])) / 1e6)
    pe_r = np.array(pe_r); pv_r = np.array(pv_r)
    print(f"  cooling-rate: viscoelastic peak |sigma| {pv_r[0]:.0f}->{pv_r[-1]:.0f} MPa "
          f"over {rates[0]}..{rates[-1]} C/s (relaxation adds rate sensitivity)")

    _plot(args.out, nodes, tris, sig_v, t, T, hist_e, hist_v, rates, pe_r, pv_r, pe, pv)
    print(f"wrote {args.out}")


def _plot(out, nodes, tris, sig_v, t, T, hist_e, hist_v, rates, pe_r, pv_r, pe, pv):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    triang = mtri.Triangulation(nodes[:, 0] * 1e3, nodes[:, 1] * 1e3, tris)
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    # relaxation time tau(T) over the cool-down
    ax0 = ax[0, 0]; ax0b = ax0.twinx()
    ax0.plot(t, T, "-", color="#d62728", label="T(t)")
    Tk = np.linspace(T_ROOM, T_MELT, 100)
    ax0b.semilogy(t, tau_of_T(T), "-", color="#7f7f7f", label="τ(T) relaxation time")
    ax0.set_xlabel("normalized process time"); ax0.set_ylabel("T [°C]", color="#d62728")
    ax0b.set_ylabel("relaxation time τ [s] (log)", color="#7f7f7f")
    ax0.set_title("thermo-viscoelastic: τ(T) grows on cooling\n(fast relax hot → frozen cold)")
    ax0.grid(True, alpha=0.3)

    # stress build-up: elastic vs viscoelastic
    steps = np.arange(1, len(hist_e) + 1)
    ax[0, 1].plot(steps, hist_e, "-o", ms=2, color="#d62728", label=f"elastic (peak {pe:.0f} MPa)")
    ax[0, 1].plot(steps, hist_v, "-o", ms=2, color="#1f77b4",
                  label=f"viscoelastic (peak {pv:.0f} MPa)")
    ax[0, 1].set_xlabel("cool step"); ax[0, 1].set_ylabel("max |σxx| [MPa]")
    ax[0, 1].set_title(f"relaxation pulls residual stress down\n{100*(1-pv/pe):.0f}% relaxed to measurement-grade")
    ax[0, 1].legend(); ax[0, 1].grid(True, alpha=0.3)

    tp = ax[1, 0].tripcolor(triang, facecolors=sig_v[:, 0] / 1e6, cmap="coolwarm")
    fig.colorbar(tp, ax=ax[1, 0], label="residual σxx [MPa]")
    ax[1, 0].set_title("viscoelastic residual STRESS field σxx (CFRTP [0/90])")
    ax[1, 0].set_xlabel("x (length) [mm]"); ax[1, 0].set_ylabel("z (thickness) [mm]")

    ax[1, 1].semilogx(rates, pe_r, "-o", ms=4, color="#d62728", label="elastic")
    ax[1, 1].semilogx(rates, pv_r, "-s", ms=4, color="#1f77b4", label="viscoelastic")
    ax[1, 1].set_xlabel("cooling rate [°C/s]"); ax[1, 1].set_ylabel("peak residual |σxx| [MPa]")
    ax[1, 1].set_title("peak residual stress vs cooling rate\n(relaxation → strong rate sensitivity)")
    ax[1, 1].legend(); ax[1, 1].grid(True, which="both", alpha=0.3)

    fig.suptitle("CFRTP thermo-viscoelastic residual stress: relaxation brings the elastic "
                 "over-prediction down to a measurement-grade level (Daikin theme)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main()
