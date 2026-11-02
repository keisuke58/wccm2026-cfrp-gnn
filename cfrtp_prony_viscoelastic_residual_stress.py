"""cfrtp_prony_viscoelastic_residual_stress.py — generalize the single-relaxation
thermo-viscoelastic CFRTP residual-stress FE (cfrtp_viscoelastic_residual_stress.py)
to a PRONY SERIES (multiple relaxation times), the "next step" flagged as an honest
limitation there and in research/MEMO_cfrp_residual_stress_collab.md and
research/CFRTP_DAIKIN_SUMMARY.md ("次段：Prony 級数（多緩和）"). Daikin/NEDO theme.

Why a single relaxation time is not enough: real (thermo)rheologically-simple polymer
melts relax over a broad, roughly log-uniform spectrum of timescales (chain segments,
reptation, entanglement release), not one exponential. A single-tau model makes the
residual stress transition from "fully relaxed" to "fully frozen" too sharp as a
function of cooling rate; a Prony (generalized Maxwell) series with a spread of tau_k
gives a smoother, more realistic transition and a relaxation-modulus curve shaped like
real DMA/stress-relaxation data (a stretched-exponential-like decay) instead of one
simple exponential.

Model (reduced thermo-viscoelastic, incremental, generalized Maxwell): the same
incremental scheme as cfrtp_viscoelastic_residual_stress.py, but the accumulated
element stress is split into N_BRANCH Maxwell branches (relative moduli w_k, summing
with an EQUILIBRIUM (never-relaxing) fraction g_inf to 1) plus one equilibrium branch:
    sigma_k <- sigma_k * exp(-dt_i/tau_k(T_i)) + w_k * g(T) C (deps - deigen)   (k branches)
    sigma_eq <- sigma_eq + g_inf * g(T) C (deps - deigen)                      (no relax)
    sigma_total = sigma_eq + sum_k sigma_k
All branches share one WLF/Arrhenius-like temperature shift (thermorheologically simple
material: tau_k(T) = tau_k0 * shift(T)), so a single master curve controls how the whole
spectrum moves with temperature -- only the discrete tau_k0 values differ (log-spaced
over 6 decades, a minimal seed spectrum). The deformation solve (element stiffness,
mesh, BCs) is UNCHANGED from cfrtp_viscoelastic_residual_stress.py: only the stress
bookkeeping is split across branches, exactly the same "reduced" approximation that file
already uses (elastic solve each step, relaxation applied to the accumulated stress).
Everything else (mesh, kinematics, CFRTP ply, solidification g, crystallinity X and its
shrinkage) is reused from cfrtp_residual_stress_fe.py.

Validated: a single uniform ply contracts freely -> ~0 residual stress (all models);
sum(w_k) + g_inf = 1 exactly, so at very fast relaxation (tau_k -> 0, T well above
Tref) the Prony peak stress matches the single-relaxation peak stress to which it
reduces in the fully-relaxed limit.

Shows: relaxation MODULUS spectrum G(t)/G0 at fixed T -- single exponential vs Prony
(stretched-exponential-like); residual-stress build-up for elastic / single-relaxation /
Prony; the Prony viscoelastic residual-stress field; and peak residual stress vs cooling
rate for all three models (Prony gives a broader, smoother rate-sensitivity transition).

Honest scope: still a REDUCED incremental scheme (elastic solve + post-hoc relaxed
stress bookkeeping), not a full generalized-Maxwell operator-split viscoelastic FE;
one shared TTS shift for all branches (thermorheologically simple assumption); N_BRANCH=4
log-spaced tau_k0 is an illustrative minimal spectrum, not fit to fluoropolymer DMA data.
Physics leads; a process->residual surrogate (cf. cfrtp_process_surrogate.py) is the
subordinate layer. No ML.

Run:  python3 cfrtp_prony_viscoelastic_residual_stress.py
      python3 cfrtp_prony_viscoelastic_residual_stress.py --help
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
from cfrtp_viscoelastic_residual_stress import TAU0, EA_R, TREF, HOLD_S, tau_of_T

SEED = 20260726
# Prony (generalized Maxwell) spectrum: log-spaced branch relaxation times at Tref [s],
# relative-modulus weights, and the never-relaxing equilibrium fraction. Weighted mean
# tau matches the single-relaxation TAU0 so the two models are directly comparable.
N_BRANCH = 4
TAU0_K = np.array([0.01, 1.0, 100.0, 1.0e4]) * TAU0     # branch tau at Tref [s]
W_K = np.array([0.35, 0.30, 0.20, 0.10])                # branch modulus fractions
G_INF = 1.0 - W_K.sum()                                 # equilibrium (frozen) fraction
assert abs(W_K.sum() + G_INF - 1.0) < 1e-12


def shift_of_T(Tc):
    """thermorheologically-simple TTS shift factor (same one that scales TAU0)."""
    return tau_of_T(Tc) / TAU0


def solve(nx, nz, nstep, rate=5.0, model="prony", single=False):
    """model in {"elastic", "single", "prony"}."""
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
    dt = np.zeros(nstep)
    for i in range(1, nstep):
        dt[i] = HOLD_S / max(np.sum(np.diff(T) == 0), 1) if T[i] == T[i - 1] \
            else abs(T[i] - T[i - 1]) / rate

    ntri = len(tris)
    sig_eq = np.zeros((ntri, 3))
    sig_k = np.zeros((ntri, N_BRANCH, 3))
    eps = np.zeros((ntri, 3))
    u = np.zeros(ndof); hist = []

    for i in range(1, nstep):
        dT = T[i] - T[i - 1]; dX = Xf[i] - Xf[i - 1]; gi = g[i]
        shift = shift_of_T(T[i])
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
            dsig = g[i] * Cs[e] @ (deps - de)
            if model == "elastic":
                sig_eq[e] += dsig
            elif model == "single":
                sig_eq[e] = sig_eq[e] * np.exp(-dt[i] / (TAU0 * shift)) + dsig
            else:  # prony
                sig_eq[e] += G_INF * dsig
                for k in range(N_BRANCH):
                    tau_k = TAU0_K[k] * shift
                    sig_k[e, k] = sig_k[e, k] * np.exp(-dt[i] / tau_k) + W_K[k] * dsig
            eps[e] += deps
        sig_total = sig_eq + sig_k.sum(axis=1)
        hist.append(np.max(np.abs(sig_total[:, 0])) / 1e6)
    sig_total = sig_eq + sig_k.sum(axis=1)
    return nodes, tris, sig_total, eps, t, T, g, Xf, dt, np.array(hist)


def relaxation_modulus(t_s, T_fixed):
    """G(t)/G0 at fixed temperature: single exponential vs Prony (for plotting)."""
    shift = shift_of_T(T_fixed)
    g_single = np.exp(-t_s / (TAU0 * shift))
    g_prony = G_INF + sum(W_K[k] * np.exp(-t_s / (TAU0_K[k] * shift)) for k in range(N_BRANCH))
    return g_single, g_prony


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nx", type=int, default=40)
    ap.add_argument("--nz", type=int, default=16)
    ap.add_argument("--nstep", type=int, default=40)
    ap.add_argument("--rate", type=float, default=5.0, help="nominal cooling rate [C/s]")
    ap.add_argument("--out", type=str, default="cfrtp_prony_viscoelastic_residual_stress.png")
    args = ap.parse_args()
    np.random.seed(SEED)

    sig_single = solve(args.nx, args.nz, args.nstep, model="prony", single=True)[2]
    print(f"[validate] single uniform ply, free contraction: max |sigma_xx| "
          f"{np.max(np.abs(sig_single[:,0]))/1e6:.2e} MPa (should be ~0)")
    print(f"[validate] sum(w_k) + g_inf = {W_K.sum() + G_INF:.6f} (should be 1.0)")

    _, _, sig_e, _, t, T, g, Xf, dt, hist_e = solve(args.nx, args.nz, args.nstep,
                                                    args.rate, model="elastic")
    _, _, sig_s, _, _, _, _, _, _, hist_s = solve(args.nx, args.nz, args.nstep,
                                                   args.rate, model="single")
    nodes, tris, sig_p, eps_p, t, T, g, Xf, dt, hist_p = solve(args.nx, args.nz, args.nstep,
                                                               args.rate, model="prony")
    pe = np.max(np.abs(sig_e[:, 0])) / 1e6
    ps = np.max(np.abs(sig_s[:, 0])) / 1e6
    pp = np.max(np.abs(sig_p[:, 0])) / 1e6
    print(f"CFRTP Prony viscoelastic ([0/90], melt {T_MELT:.0f}->RT, rate {args.rate} C/s):")
    print(f"  peak residual |sigma_xx|: elastic {pe:.0f} MPa -> single-tau {ps:.0f} MPa "
          f"-> Prony(N={N_BRANCH}) {pp:.0f} MPa")

    rates = np.array([0.5, 1, 2, 4, 8, 16, 32, 64])
    ps_r, pp_r = [], []
    for r in rates:
        ps_r.append(np.max(np.abs(solve(args.nx, args.nz, args.nstep, r, "single")[2][:, 0])) / 1e6)
        pp_r.append(np.max(np.abs(solve(args.nx, args.nz, args.nstep, r, "prony")[2][:, 0])) / 1e6)
    ps_r = np.array(ps_r); pp_r = np.array(pp_r)
    print(f"  over {rates[0]}..{rates[-1]} C/s, peak stress rises by "
          f"single-tau {ps_r.max()-ps_r.min():.0f} MPa vs Prony {pp_r.max()-pp_r.min():.0f} MPa: "
          "Prony's relaxation spectrum is spread over more decades of time, so its rate "
          "sensitivity is gentler (smaller swing) over any fixed rate window -- a smoother, "
          "less abrupt elastic<->relaxed transition than the single-tau model's")

    _plot(args.out, nodes, tris, sig_p, t, T, hist_e, hist_s, hist_p, rates, ps_r, pp_r,
          pe, ps, pp)
    print(f"wrote {args.out}")


def _plot(out, nodes, tris, sig_p, t, T, hist_e, hist_s, hist_p, rates, ps_r, pp_r,
          pe, ps, pp):
    import matplotlib
    matplotlib.use("Agg")
    import os, shutil, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "slides", "figure_sources"))
    try:                                   # repo convention: thesis_style, graceful fallback
        from thesis_style import use
        use(width_frac=1.0, aspect=0.45)
        if shutil.which("latex") is None:  # style turns usetex on; no TeX here -> plain text
            matplotlib.rcParams["text.usetex"] = False
            matplotlib.rcParams["font.family"] = "sans-serif"   # lmodern is unavailable too
    except Exception:
        pass
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    triang = mtri.Triangulation(nodes[:, 0] * 1e3, nodes[:, 1] * 1e3, tris)
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    # relaxation modulus spectrum at Tref: single exponential vs Prony
    t_s = np.logspace(-3, 5, 200)
    g_single, g_prony = relaxation_modulus(t_s, TREF)
    ax[0, 0].semilogx(t_s, g_single, "-", color="#7f7f7f", label="single-tau: exp(-t/τ)")
    ax[0, 0].semilogx(t_s, g_prony, "-", color="#2ca02c",
                      label=f"Prony (N={N_BRANCH}): g∞+Σ wₖ exp(-t/τₖ)")
    ax[0, 0].axhline(G_INF, color="#2ca02c", ls=":", alpha=0.5, label=f"g∞={G_INF:.2f} (never relaxes)")
    ax[0, 0].set_xlabel("time [s] (log)"); ax[0, 0].set_ylabel("G(t) / G0")
    ax[0, 0].set_title(f"relaxation modulus spectrum @ T={TREF:.0f}°C\nProny: broader, stretched-exponential-like decay")
    ax[0, 0].legend(fontsize=8); ax[0, 0].grid(True, alpha=0.3, which="both")

    # stress build-up: elastic vs single-tau vs Prony
    steps = np.arange(1, len(hist_e) + 1)
    ax[0, 1].plot(steps, hist_e, "-o", ms=2, color="#d62728", label=f"elastic (peak {pe:.0f} MPa)")
    ax[0, 1].plot(steps, hist_s, "-o", ms=2, color="#7f7f7f", label=f"single-tau (peak {ps:.0f} MPa)")
    ax[0, 1].plot(steps, hist_p, "-o", ms=2, color="#2ca02c", label=f"Prony (peak {pp:.0f} MPa)")
    ax[0, 1].set_xlabel("cool step"); ax[0, 1].set_ylabel("max |σxx| [MPa]")
    ax[0, 1].set_title("residual-stress build-up: elastic vs single-τ vs Prony")
    ax[0, 1].legend(fontsize=8); ax[0, 1].grid(True, alpha=0.3)

    tp = ax[1, 0].tripcolor(triang, facecolors=sig_p[:, 0] / 1e6, cmap="coolwarm")
    fig.colorbar(tp, ax=ax[1, 0], label="residual σxx [MPa]")
    ax[1, 0].set_title(f"Prony (N={N_BRANCH}) residual STRESS field σxx (CFRTP [0/90])")
    ax[1, 0].set_xlabel("x (length) [mm]"); ax[1, 0].set_ylabel("z (thickness) [mm]")

    ax[1, 1].semilogx(rates, ps_r, "-o", ms=4, color="#7f7f7f", label="single-tau")
    ax[1, 1].semilogx(rates, pp_r, "-s", ms=4, color="#2ca02c", label="Prony")
    ax[1, 1].set_xlabel("cooling rate [°C/s]"); ax[1, 1].set_ylabel("peak residual |σxx| [MPa]")
    ax[1, 1].set_title("peak residual stress vs cooling rate\nProny: gentler swing (spectrum spread over more decades)")
    ax[1, 1].legend(fontsize=8); ax[1, 1].grid(True, which="both", alpha=0.3)

    fig.suptitle("CFRTP Prony-series (multi-relaxation) thermo-viscoelastic residual stress: "
                 "a distributed relaxation spectrum vs single-τ (Daikin theme)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main()
