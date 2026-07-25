"""electrothermal_selfheating.py — seed demo for research theme G
(research/RESEARCH_THEMES_muramatsu.md, LITREVIEW_G_electrothermal.md): weak-form
finite-element, self-consistent ELECTRO-THERMAL self-heating, where current
continuation is decisive. Non-fracture multiphysics that couples this repo's
electrical (drift-diffusion, ⑧⑨) and thermal (⑥⑦) sides.

Physics (1D, steady, self-consistent). A conductor of length L carries current
density J. Conductivity is temperature-activated, sigma(T) = exp(EA (1/T0 - 1/T))
(rises with T — leakage/intrinsic-like), so Joule heating raises T, raises sigma,
raises the current path — the electro-thermal feedback that produces self-heating.
Current continuity in 1D gives a uniform J, local field E(x) = J / sigma(T(x)),
Joule source J*E = J^2 / sigma(T), and the steady heat equation with a distributed
substrate heat sink (thermal-network picture; cf. CFET self-heating literature):

    -kappa T''(x) + h (T - T0) = J^2 / sigma(T(x))   (insulated ends),

solved by damped Newton on P1 finite elements (weak form). The terminal voltage is
V(J) = J * integral dx / sigma(T(x)). As J rises, sigma rises fast, so V rises then
*falls* — an S-shaped I-V with negative differential resistance (a hallmark of
self-heating, cf. non-isothermal drift-diffusion literature).

Idea C, made necessary by the physics: sweeping the CURRENT J (continuation) traces
the whole folded I-V including the NDR branch; sweeping the VOLTAGE cannot — past the
peak it snaps through (the NDR branch is voltage-inaccessible). Warm-starting each J
step from the previous solution also cuts Newton iterations. FEM keeps accuracy.

Honest scope: reduced electro-thermal model (temperature-activated Ohmic conduction),
not full non-isothermal 2-carrier drift-diffusion (the natural extension — couple ⑧⑨
with this heat equation). Illustrative non-dimensional units. No ML (pure physics /
numerics), by design for this theme.

Run:  python3 electrothermal_selfheating.py   (writes electrothermal_selfheating.png)
      python3 electrothermal_selfheating.py --help
"""
from __future__ import annotations

import argparse

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

SEED = 20260725
L = 1.0
KAPPA = 0.02            # lateral heat conduction
HSINK = 1.0            # distributed heat loss to the substrate (thermal-network sink)
T0 = 1.0
EA = 15.0              # activation (dimensionless): larger = stronger feedback
NEWTON_TOL = 1e-9
NEWTON_MAX = 80


def sigma(T):
    return np.exp(EA * (1.0 / T0 - 1.0 / T))          # sigma(T0)=1, rises with T


def dsigma(T):
    return sigma(T) * EA / T ** 2


def assemble(n):
    x = np.linspace(0, L, n)
    h = x[1] - x[0]
    # P1 stiffness (Laplacian) and lumped mass
    main = np.full(n, 2.0 / h); main[0] = main[-1] = 1.0 / h
    off = np.full(n - 1, -1.0 / h)
    K = sp.diags([off, main, off], [-1, 0, 1]).tocsc()
    ml = np.full(n, h); ml[0] = ml[-1] = h / 2
    return x, K, ml


def solve_T(J, Kk, ml, T_init):
    """Damped Newton for  -kappa T'' + h(T-T0) = J^2/sigma(T)  (distributed
    substrate heat sink; no cold end-clamps, so the whole device can heat)."""
    T = T_init.copy()
    Hm = sp.diags(HSINK * ml)

    def resid(TT):
        return Kk @ TT + HSINK * ml * (TT - T0) - ml * (J ** 2 / sigma(TT))

    hist = []
    for it in range(1, NEWTON_MAX + 1):
        R = resid(T)
        rn = np.linalg.norm(R); hist.append(rn)
        if rn < NEWTON_TOL:
            return T, it - 1, hist
        dS = -(J ** 2) * dsigma(T) / sigma(T) ** 2       # d(J^2/sigma)/dT
        Jac = (Kk + Hm - sp.diags(ml * dS)).tocsc()
        dT = spla.spsolve(Jac, -R)
        a = 1.0
        for _ in range(40):
            if np.linalg.norm(resid(T + a * dT)) < rn:
                break
            a *= 0.5
        T = T + a * dT
    return T, NEWTON_MAX, hist


def voltage(J, T, x):
    # V = J * integral dx / sigma(T)  (trapezoidal)
    f = 1.0 / sigma(T)
    return J * float(np.sum(0.5 * (f[:-1] + f[1:]) * np.diff(x)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=81, help="FE nodes")
    ap.add_argument("--nstep", type=int, default=90, help="current-continuation steps")
    ap.add_argument("--jmax", type=float, default=7.0, help="max current density")
    ap.add_argument("--out", type=str, default="electrothermal_selfheating.png")
    ap.add_argument("--data", type=str, default="electrothermal_selfheating",
                    help="dataset basename (writes .csv sweep + .npz with T(x) profiles)")
    args = ap.parse_args()
    if args.nstep < 3:
        ap.error("--nstep must be >= 3")

    x, K, ml = assemble(args.n)
    Kk = (KAPPA * K).tocsc()
    Js = np.linspace(args.jmax / args.nstep, args.jmax, args.nstep)

    # ---- current continuation (warm) ----
    Tw = np.full(args.n, T0); warm_it, V, Tmax = [], [], []
    Tsnap = {}
    for k, J in enumerate(Js):
        Tw, it, _ = solve_T(J, Kk, ml, Tw)
        warm_it.append(it); V.append(voltage(J, Tw, x)); Tmax.append(Tw.max())
        Tsnap[k] = Tw.copy()
    V = np.array(V); Tmax = np.array(Tmax)

    # ---- cold (each J from ambient T0) ----
    cold_it = []
    for J in Js:
        _, it, _ = solve_T(J, Kk, ml, np.full(args.n, T0))
        cold_it.append(it)

    # detect the NDR fold (V peak) -> voltage control can't pass it
    kpk = int(np.argmax(V))
    ndr = kpk < len(V) - 1 and V[-1] < V[kpk]
    tc, tw = int(sum(cold_it)), int(sum(warm_it))
    print(f"\nelectro-thermal self-heating: {args.nstep} current steps, {args.n} nodes")
    print(f"  I-V: V peaks at J={Js[kpk]:.2f} (V={V[kpk]:.3f}), then "
          f"{'FOLDS BACK (NDR/S-shape)' if ndr else 'monotone'} -> V_end={V[-1]:.3f}")
    print(f"  -> current continuation traces the fold; voltage control snaps through it")
    print(f"  Newton iters (per step)  cold {tc}  warm {tw}  ({100*(tc-tw)/tc:.0f}% fewer)")
    print(f"  max lattice temperature rise: {Tmax[-1]/T0:.2f} x T0")

    # ---- dataset export (sweep table + temperature-field profiles) ----
    if args.data:
        import csv
        Tprof = np.array([Tsnap[k] for k in range(len(Js))])          # (nstep, n)
        np.savez(args.data + ".npz", J=Js, V=V, Tmax=Tmax, x=x, T_profiles=Tprof,
                 cold_iters=np.array(cold_it), warm_iters=np.array(warm_it),
                 params=dict(EA=EA, KAPPA=KAPPA, HSINK=HSINK, T0=T0))
        with open(args.data + ".csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "J", "V", "Tmax", "cold_iters", "warm_iters"])
            for k in range(len(Js)):
                w.writerow([k, f"{Js[k]:.6g}", f"{V[k]:.6g}", f"{Tmax[k]:.6g}",
                            cold_it[k], warm_it[k]])
        print(f"wrote dataset {args.data}.csv ({len(Js)} rows) and {args.data}.npz "
              f"(T(x) profiles {Tprof.shape})")

    _plot(args.out, x, Js, V, Tmax, Tsnap, cold_it, warm_it, kpk, ndr, tc, tw)
    print(f"wrote {args.out}")


def _plot(out, x, Js, V, Tmax, Tsnap, cold_it, warm_it, kpk, ndr, tc, tw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    # I-V (current vs voltage), fold marked
    ax[0, 0].plot(V, Js, "-o", ms=3, color="#1f77b4")
    ax[0, 0].plot(V[kpk], Js[kpk], "s", ms=9, color="#d62728",
                  label="V peak → NDR / snap" if ndr else "V peak")
    ax[0, 0].annotate("negative differential\nresistance (NDR)\nvoltage control snaps",
                      xy=(V[kpk], Js[kpk]), xytext=(V[kpk] * 0.55, Js[kpk] * 1.02),
                      fontsize=9, color="#d62728",
                      arrowprops=dict(arrowstyle="->", color="#d62728"))
    ax[0, 0].set_xlabel("terminal voltage V"); ax[0, 0].set_ylabel("current density J")
    ax[0, 0].set_title("self-heating I-V (S-shape): current continuation traces the fold")
    ax[0, 0].legend(); ax[0, 0].grid(True, alpha=0.3)

    # temperature profiles at increasing current
    for k in range(0, len(Js), max(1, len(Js) // 6)):
        ax[0, 1].plot(x, Tsnap[k], label=f"J={Js[k]:.2f}")
    ax[0, 1].set_xlabel("position x"); ax[0, 1].set_ylabel("temperature T")
    ax[0, 1].set_title("lattice temperature profile vs current (Joule self-heating)")
    ax[0, 1].legend(fontsize=8); ax[0, 1].grid(True, alpha=0.3)

    # peak temperature vs current
    ax[1, 0].plot(Js, Tmax, "-o", ms=3, color="#b5651d")
    ax[1, 0].set_xlabel("current density J"); ax[1, 0].set_ylabel("peak temperature T_max")
    ax[1, 0].set_title("thermal runaway trend (T rises, σ rises → feedback)")
    ax[1, 0].grid(True, alpha=0.3)

    xs = np.arange(len(cold_it))
    ax[1, 1].plot(xs, cold_it, "-o", ms=3, color="#d62728", label="cold (from ambient)")
    ax[1, 1].plot(xs, warm_it, "-o", ms=3, color="#1f77b4", label="warm (continuation)")
    ax[1, 1].set_title(f"Newton iterations per current step\ntotal cold {tc} → warm {tw} "
                       f"({100*(tc-tw)/tc:.0f}% fewer)")
    ax[1, 1].set_xlabel("current step"); ax[1, 1].set_ylabel("Newton iterations")
    ax[1, 1].legend(); ax[1, 1].grid(True, alpha=0.3)

    fig.suptitle("Weak-form FE electro-thermal self-heating (theme G): temperature-activated "
                 "conduction → S-shaped I-V/NDR where current continuation is decisive", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main()
