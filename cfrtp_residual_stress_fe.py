"""cfrtp_residual_stress_fe.py — CFRTP (carbon-fibre-reinforced THERMOPLASTIC)
version of the cure residual-stress FE, matching Daikin's actual material: a
fluoropolymer-matrix CFRTP (not thermoset epoxy). See
research/MEMO_cfrp_residual_stress_collab.md — Daikin's CFRP is CFRTP, so the
constitutive picture is MELT -> CRYSTALLIZE -> COOL, not cure.

Difference from the cure FE (cfrp_cure_residual_stress_fe.py):
  * process = melt above the melting point, then COOL to room temperature (a large
    dT, because fluoropolymers process hot ~330 C),
  * solidification (not cure) develops the modulus g(T) as the melt freezes on cooling
    through the solidification window -> stress locks in on cool-down,
  * crystallization: a degree of crystallinity X develops as it solidifies, with a
    CRYSTALLIZATION SHRINKAGE eigenstrain (semicrystalline polymers shrink on
    crystallizing), and X is COOLING-RATE dependent (faster cooling -> less
    crystalline), which is the CFRTP process sensitivity (a known manufacturing issue),
  * fluoropolymer-matrix ply properties: high transverse CTE, soft transverse modulus
    -> large residual stress from the big cool-down.
The incremental plane-stress CST FE (mesh / kinematics reused from the cure FE) locks
in residual stress/strain step by step across the cool-down.

Validated: a single uniform ply contracts freely -> ~0 residual stress.

Outputs (process -> residual state for the Daikin theme):
  * cool-down T(t), solidification g(T), crystallinity X(t),
  * residual STRESS field sigma_xx and residual STRAIN field eps_xx ([0/90] section),
  * COOLING-RATE sensitivity: residual stress and crystallinity vs cooling rate — the
    CFRTP manufacturing knob.

Honest scope: reduced thermoplastic model (solidification-hardening analogous to
CHILE, algebraic non-isothermal crystallization, crystallization shrinkage as a linear
eigenstrain in dX, no viscoelastic relaxation, no tool-part interaction); illustrative
fluoropolymer-CFRTP-like properties. Because there is no viscoelastic relaxation, the
elastic model OVER-predicts the stress magnitude (real fluoropolymer CFRTP relaxes at
high temperature); the process TRENDS — cooling-rate sensitivity, the ply-mismatch
pattern, that CFRTP residual stress is large — are the takeaway, not the absolute MPa.
Physics is the lead; a process->residual surrogate (repo CFRP-GNN / DeepONet, cf. ㉑)
is the subordinate layer. No ML here.

Run:  python3 cfrtp_residual_stress_fe.py     (writes cfrtp_residual_stress_fe.png)
      python3 cfrtp_residual_stress_fe.py --help
"""
from __future__ import annotations

import argparse

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from cfrp_cure_residual_stress_fe import build_mesh, cst_B

SEED = 20260726
# fluoropolymer-matrix CFRTP ply (plane-stress; soft, high-CTE transverse)
E1, E2 = 100e9, 4e9           # fibre-dir / transverse modulus [Pa]
NU = 0.30
G12 = 3e9
A1, A2 = 0.0e-6, 110e-6       # CTE fibre / transverse [1/K] (fluoropolymer: high transverse)
BETA_CRYST = -3.5e-2          # transverse crystallization shrinkage per unit crystallinity
# thermoplastic process (deg C)
T_MELT = 340.0                # processing / melt temperature
T_ROOM = 25.0
TS_HIGH, TS_LOW = 300.0, 240.0   # solidification window (melt freezes between these)
G0 = 0.01                     # molten stiffness fraction
X_INF = 0.55                  # max attainable crystallinity (slow cooling)
RATE0 = 12.0                  # cooling-rate sensitivity scale [C/s]


def ply_material_cfrtp(deg):
    """Plane-stress orthotropic C and eigenstrain coeffs (thermal a_x,a_z; crystallization
    shrink s_x,s_z per unit crystallinity) for a CFRTP ply in the x-z cross-section."""
    if deg == 0:
        Ex, Ez, ax, az = E1, E2, A1, A2
        sx, sz = 0.0, BETA_CRYST
        nuxz = NU * Ez / E1
    else:
        Ex, Ez, ax, az = E2, E2, A2, A2
        sx, sz = BETA_CRYST, BETA_CRYST
        nuxz = NU
    nuzx = nuxz * Ez / Ex
    d = 1.0 - nuxz * nuzx
    C = np.array([[Ex / d, nuzx * Ex / d, 0.0],
                  [nuxz * Ez / d, Ez / d, 0.0],
                  [0.0, 0.0, G12]])
    return C, np.array([ax, az, 0.0]), np.array([sx, sz, 0.0])


def solidification(T):
    """Solidification fraction s(T): 0 molten (T>=TS_HIGH) -> 1 solid (T<=TS_LOW)."""
    return np.clip((TS_HIGH - T) / (TS_HIGH - TS_LOW), 0.0, 1.0)


def xmax_of_rate(rate):
    """Attainable crystallinity vs cooling rate (faster cooling -> less crystalline)."""
    return X_INF * np.exp(-rate / RATE0)


def cool_cycle(nstep, hold_frac=0.15):
    """Melt hold then linear cool T_MELT -> T_ROOM (deg C)."""
    t = np.linspace(0, 1, nstep)
    T = np.where(t < hold_frac, T_MELT,
                 T_MELT + (T_ROOM - T_MELT) * (t - hold_frac) / (1 - hold_frac))
    return t, T


def solve(nx, nz, nstep, rate=5.0, single=False):
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
    Xf = xmax_of_rate(rate) * s                        # crystallinity develops with solidification
    sig = np.zeros((len(tris), 3)); eps = np.zeros((len(tris), 3))
    u = np.zeros(ndof); hist = []

    for i in range(1, nstep):
        dT = T[i] - T[i - 1]; dX = Xf[i] - Xf[i - 1]; gi = g[i]
        rows, cols, vals = [], [], []; F = np.zeros(ndof)
        for e, tri in enumerate(tris):
            dof = np.array([2 * tri[0], 2 * tri[0] + 1, 2 * tri[1], 2 * tri[1] + 1,
                            2 * tri[2], 2 * tri[2] + 1])
            Ce = gi * Cs[e]
            de = eth[e] * dT + esh[e] * dX             # thermal + crystallization shrink
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
            sig[e] += g[i] * Cs[e] @ (deps - de)
            eps[e] += deps
        hist.append(np.max(np.abs(sig[:, 0])) / 1e6)
    return nodes, tris, ply, sig, eps, u, t, T, g, Xf, np.array(hist)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nx", type=int, default=40)
    ap.add_argument("--nz", type=int, default=16)
    ap.add_argument("--nstep", type=int, default=40)
    ap.add_argument("--rate", type=float, default=5.0, help="nominal cooling rate [C/s]")
    ap.add_argument("--out", type=str, default="cfrtp_residual_stress_fe.png")
    args = ap.parse_args()
    np.random.seed(SEED)

    sig_single = solve(args.nx, args.nz, args.nstep, single=True)[3]
    print(f"[validate] single uniform ply, free contraction: max |sigma_xx| "
          f"{np.max(np.abs(sig_single[:,0]))/1e6:.2e} MPa (should be ~0)")

    nodes, tris, ply, sig, eps, u, t, T, g, Xf, hist = solve(args.nx, args.nz, args.nstep, args.rate)
    sxx = sig[:, 0] / 1e6; exx = eps[:, 0]
    print(f"CFRTP residual stress FE (fluoropolymer, [0/90], melt {T_MELT:.0f}->RT, "
          f"rate {args.rate} C/s): final crystallinity {Xf.max():.3f}")
    print(f"  residual sigma_xx range [{sxx.min():.1f}, {sxx.max():.1f}] MPa")
    print(f"  residual eps_xx  range [{exx.min()*1e3:.3f}, {exx.max()*1e3:.3f}] milli-strain")

    # cooling-rate sensitivity (the CFRTP manufacturing knob)
    rates = np.array([0.5, 1, 2, 4, 8, 16, 32, 64])
    s_rate, x_rate = [], []
    for r in rates:
        _, _, _, sg, _, _, _, _, _, Xr, _ = solve(args.nx, args.nz, args.nstep, r)
        s_rate.append(np.max(np.abs(sg[:, 0])) / 1e6); x_rate.append(Xr.max())
    s_rate = np.array(s_rate); x_rate = np.array(x_rate)
    print(f"  cooling-rate sweep: crystallinity {x_rate[0]:.2f}->{x_rate[-1]:.2f}, "
          f"max|sigma| {s_rate[0]:.0f}->{s_rate[-1]:.0f} MPa across {rates[0]}..{rates[-1]} C/s")

    _plot(args.out, nodes, tris, sig, eps, t, T, g, Xf, rates, s_rate, x_rate, args.rate)
    print(f"wrote {args.out}")


def _plot(out, nodes, tris, sig, eps, t, T, g, Xf, rates, s_rate, x_rate, rate):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    triang = mtri.Triangulation(nodes[:, 0] * 1e3, nodes[:, 1] * 1e3, tris)
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    ax0 = ax[0, 0]; ax0b = ax0.twinx()
    ax0.plot(t, T, "-", color="#d62728", label="T(t) melt→cool")
    ax0b.plot(t, g, "-", color="#2ca02c", label="solidification g")
    ax0b.plot(t, Xf, "-", color="#1f77b4", label="crystallinity X")
    ax0.axhspan(TS_LOW, TS_HIGH, color="gray", alpha=0.12)
    ax0.set_xlabel("normalized process time"); ax0.set_ylabel("T [°C]", color="#d62728")
    ax0b.set_ylabel("g, X"); ax0b.set_ylim(0, 1)
    ax0.set_title("CFRTP process: melt→cool; solidify + crystallize\n(grey = solidification window)")
    lines = ax0.get_lines() + ax0b.get_lines()
    ax0.legend(lines, [l.get_label() for l in lines], fontsize=7, loc="center right")
    ax0.grid(True, alpha=0.3)

    tp = ax[0, 1].tripcolor(triang, facecolors=sig[:, 0] / 1e6, cmap="coolwarm")
    fig.colorbar(tp, ax=ax[0, 1], label="residual σxx [MPa]")
    ax[0, 1].set_title("residual STRESS field σxx (CFRTP [0/90])")
    ax[0, 1].set_xlabel("x (length) [mm]"); ax[0, 1].set_ylabel("z (thickness) [mm]")

    tp2 = ax[1, 0].tripcolor(triang, facecolors=eps[:, 0] * 1e3, cmap="PuOr")
    fig.colorbar(tp2, ax=ax[1, 0], label="residual εxx [milli-strain]")
    ax[1, 0].set_title("residual STRAIN field εxx")
    ax[1, 0].set_xlabel("x (length) [mm]"); ax[1, 0].set_ylabel("z (thickness) [mm]")

    ax3 = ax[1, 1]; ax3b = ax3.twinx()
    ax3.semilogx(rates, s_rate, "-o", ms=4, color="#b5651d", label="max |σxx| [MPa]")
    ax3b.semilogx(rates, x_rate, "-s", ms=4, color="#1f77b4", label="crystallinity X")
    ax3.set_xlabel("cooling rate [°C/s]"); ax3.set_ylabel("max |σxx| [MPa]", color="#b5651d")
    ax3b.set_ylabel("attained crystallinity X", color="#1f77b4")
    ax3.set_title("CFRTP process sensitivity:\nfaster cooling → less crystalline, less shrink stress")
    lines = ax3.get_lines() + ax3b.get_lines()
    ax3.legend(lines, [l.get_label() for l in lines], fontsize=8)
    ax3.grid(True, which="both", alpha=0.3)

    fig.suptitle("CFRTP (fluoropolymer/carbon) cure-free residual stress & strain — melt→crystallize→cool "
                 "FE: process → residual σ, ε and cooling-rate sensitivity (Daikin theme)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main()
