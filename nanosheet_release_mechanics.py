"""nanosheet_release_mechanics.py — solid-mechanics reliability of GAA nanosheet
RELEASE, the step directly downstream of Tokyo-Electron's signature selective SiGe
etch. After the SiGe is etched away the ultra-thin Si nanosheets are left as
suspended doubly-clamped bridges between the source/drain anchors, and two mechanical
failures gate the process window:

  1. BUCKLING — process residual (compressive) stress in the released sheet can
     exceed the Euler buckling load of the slender bridge, bowing the sheet
     (clamped-clamped critical stress  sigma_cr = pi^2 E t^2 / (3 L^2)).
  2. STICTION — during wet processing / drying, adhesion (van der Waals / capillary)
     to the neighbouring sheet or substrate can collapse the bridge if the adhesion
     energy beats the bending energy of peeling it back off. A clamped bridge stays
     stuck once its half-length exceeds the equilibrium detachment length
     s_eq = (18 E I g^2 / Gamma)^{1/4}  (I = t^3/12 per width, g the gap, Gamma the
     work of adhesion).

Method: weak-form Euler-Bernoulli beam finite elements (Hermite cubic, 2 dof/node).
The buckling load is the geometric-stiffness eigenvalue  K phi = P Kg phi, VALIDATED
against the analytic clamped-clamped value. Stiction is an energy balance (bending of
the peeled segment vs adhesion of the stuck segment) with the equilibrium peel front.
Combining both over a (length L, thickness t) grid gives the RELEASE PROCESS WINDOW:
which nanosheet geometries survive, which buckle, which stick — the quantity a process
house (etch/clean/dry) actually designs around.

Idea C / ML (kept subordinate): the window is swept by direct FE here; a surrogate on
this map is the natural fast-screening layer (cf. the theme-G operator demo), with FE
the accuracy authority. Physics is the substance.

Honest scope: linear Euler-Bernoulli (small-deflection buckling onset + linear-elastic
peel energy), isotropic Si modulus, per-unit-width 2D beam idealization of a sheet,
adhesion as a single work-of-adhesion Gamma. Illustrative but physical nm/GPa values.

Run:  python3 nanosheet_release_mechanics.py     (writes nanosheet_release_mechanics.png)
      python3 nanosheet_release_mechanics.py --help
"""
from __future__ import annotations

import argparse

import numpy as np
import scipy.linalg as sla

SEED = 20260726
E_SI = 170e9            # Si Young's modulus [Pa]
GAP = 6e-9            # gap to the neighbouring sheet / substrate [m]
GAMMA = 0.10          # work of adhesion [J/m^2] (van der Waals / capillary scale)
SIGMA_RES = 0.8e9     # process residual COMPRESSIVE stress [Pa]


def beam_matrices(L, t, n):
    """Assemble Hermite-cubic Euler-Bernoulli stiffness K and geometric stiffness Kg
    (per unit width) for a clamped-clamped beam of length L, thickness t, n elements.
    Kg is built for unit compressive axial load, so the buckling eigenvalue is P_cr."""
    I = t ** 3 / 12.0                        # second moment per unit width
    EI = E_SI * I
    le = L / n
    Ndof = 2 * (n + 1)
    K = np.zeros((Ndof, Ndof)); Kg = np.zeros((Ndof, Ndof))
    ke = EI / le ** 3 * np.array([
        [12, 6 * le, -12, 6 * le],
        [6 * le, 4 * le ** 2, -6 * le, 2 * le ** 2],
        [-12, -6 * le, 12, -6 * le],
        [6 * le, 2 * le ** 2, -6 * le, 4 * le ** 2]])
    kg = 1.0 / (30 * le) * np.array([        # unit compressive load P=1
        [36, 3 * le, -36, 3 * le],
        [3 * le, 4 * le ** 2, -3 * le, -le ** 2],
        [-36, -3 * le, 36, -3 * le],
        [3 * le, -le ** 2, -3 * le, 4 * le ** 2]])
    for e in range(n):
        d = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
        K[np.ix_(d, d)] += ke
        Kg[np.ix_(d, d)] += kg
    return K, Kg


def buckling_stress(L, t, n=40):
    """Smallest buckling load from the FE geometric-stiffness eigenproblem, returned
    as a critical stress sigma_cr = P_cr / (t * 1). Solved as the symmetric problem
    Kg phi = mu K phi with K SPD (well conditioned; Kg is singular so the (K,Kg) form
    is not), so P_cr = 1 / max(positive mu)."""
    K, Kg = beam_matrices(L, t, n)
    Ndof = K.shape[0]
    fixed = [0, 1, Ndof - 2, Ndof - 1]                 # clamped-clamped: w=w'=0 both ends
    free = [i for i in range(Ndof) if i not in fixed]
    mu = sla.eigh(Kg[np.ix_(free, free)], K[np.ix_(free, free)], eigvals_only=True)
    Pcr = 1.0 / float(np.max(mu))                      # per unit width [N/m]
    return Pcr / t, Pcr                                # sigma_cr [Pa], Pcr [N/m]


def buckling_mode(L, t, n=60):
    K, Kg = beam_matrices(L, t, n)
    Ndof = K.shape[0]
    fixed = [0, 1, Ndof - 2, Ndof - 1]
    free = [i for i in range(Ndof) if i not in fixed]
    mu, V = sla.eigh(Kg[np.ix_(free, free)], K[np.ix_(free, free)])
    k = int(np.argmax(mu))                             # largest mu = smallest P_cr = first mode
    phi = np.zeros(Ndof); phi[free] = V[:, k].real
    x = np.linspace(0, L, n + 1)
    wshape = phi[0::2]                                  # transverse deflections
    wshape = wshape / np.max(np.abs(wshape) + 1e-30)
    return x, wshape


def detachment_length(t, gap=GAP, gamma=GAMMA):
    """Equilibrium peel/detachment length for adhesion collapse (stiction).
    s_eq = (18 E I gap^2 / Gamma)^{1/4}; a bridge sticks if half-length >= s_eq."""
    I = t ** 3 / 12.0
    return (18.0 * E_SI * I * gap ** 2 / gamma) ** 0.25


def stiction_energy(s, a, t, gap=GAP, gamma=GAMMA):
    """Total energy per width of a collapsed bridge with peel length s and half-length
    a: bending of the peeled segment (fixed-guided, deflection=gap) minus adhesion of
    the stuck segment.  U(s) = 6 E I gap^2 / s^3 - Gamma (a - s)."""
    I = t ** 3 / 12.0
    return 6.0 * E_SI * I * gap ** 2 / s ** 3 - gamma * np.clip(a - s, 0, None)


def sticks(a, t, gap=GAP, gamma=GAMMA):
    """True stiction criterion: the collapsed state is favourable (min_s U(s) < 0).
    The energy minimum sits at the detachment length s_eq (or at the boundary s=a if
    the bridge is shorter), so evaluate U there."""
    s_star = min(detachment_length(t, gap, gamma), a)
    return stiction_energy(s_star, a, t, gap, gamma) < 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=40, help="beam elements")
    ap.add_argument("--out", type=str, default="nanosheet_release_mechanics.png")
    args = ap.parse_args()
    np.random.seed(SEED)

    # ---- 1. buckling: FE vs analytic clamped-clamped, across length ----
    t0 = 6e-9
    Ls = np.linspace(30e-9, 220e-9, 18)
    sig_fe = np.array([buckling_stress(L, t0, args.n)[0] for L in Ls])
    sig_an = np.pi ** 2 * E_SI * t0 ** 2 / (3.0 * Ls ** 2)     # clamped-clamped
    rel = float(np.max(np.abs(sig_fe - sig_an) / sig_an))
    print(f"buckling FE vs analytic (clamped-clamped sigma_cr=pi^2 E t^2/3L^2): "
          f"max rel err {rel:.2e}  (t={t0*1e9:.0f} nm)")

    # ---- 2. a representative buckled mode ----
    xm, wm = buckling_mode(120e-9, t0)

    # ---- 3. stiction energy landscape + critical length ----
    a_half = 100e-9                                    # half bridge length
    s = np.linspace(15e-9, a_half, 200)
    U6 = stiction_energy(s, a_half, 6e-9)
    U8 = stiction_energy(s, a_half, 8e-9)
    seq6 = detachment_length(6e-9); seq8 = detachment_length(8e-9)
    # true critical half-length where min_s U = 0 (energy criterion), by bisection
    def a_crit(t):
        lo, hi = 1e-9, 2e-6
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if sticks(mid, t):
                hi = mid
            else:
                lo = mid
        return hi
    ac6 = a_crit(6e-9)
    print(f"stiction: energy-min detachment s_eq(t=6nm)={seq6*1e9:.1f} nm; collapsed state "
          f"favourable (min_s U<0) once half-length >= {ac6*1e9:.0f} nm -> sticks if "
          f"L>={2*ac6*1e9:.0f} nm (gap={GAP*1e9:.0f} nm, Gamma={GAMMA} J/m^2)")

    # ---- 4. process window over (L, t): safe / buckled / stuck ----
    Lg = np.linspace(30e-9, 350e-9, 44)
    tg = np.linspace(4e-9, 14e-9, 30)
    regime = np.zeros((len(tg), len(Lg)), dtype=int)   # 0 safe,1 buckle,2 stick,3 both
    for it_, t in enumerate(tg):
        for il, L in enumerate(Lg):
            scr, _ = buckling_stress(L, t, 24)
            buck = SIGMA_RES > scr
            stick = sticks(L / 2.0, t)                  # energy criterion min_s U < 0
            regime[it_, il] = (1 if buck else 0) + (2 if stick else 0)
    frac = {k: int(np.sum(regime == k)) for k in range(4)}
    print(f"process window (sigma_res={SIGMA_RES/1e9:.1f} GPa): "
          f"safe {frac[0]}, buckle {frac[1]}, stick {frac[2]}, both {frac[3]} "
          f"(of {regime.size} geometries)")

    _plot(args.out, Ls, sig_fe, sig_an, rel, xm, wm, s, U6, U8, seq6, seq8, a_half,
          Lg, tg, regime)
    print(f"wrote {args.out}")


def _plot(out, Ls, sig_fe, sig_an, rel, xm, wm, s, U6, U8, seq6, seq8, a_half,
          Lg, tg, regime):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    ax[0, 0].loglog(Ls * 1e9, sig_an / 1e9, "-", color="#1f77b4", label="analytic (clamped-clamped)")
    ax[0, 0].loglog(Ls * 1e9, sig_fe / 1e9, "o", ms=5, color="#d62728", label="weak-form beam FE")
    ax[0, 0].set_xlabel("nanosheet length L [nm]"); ax[0, 0].set_ylabel("critical buckling stress [GPa]")
    ax[0, 0].set_title(f"buckling: FE validated vs analytic\nmax rel. error {rel:.1e}")
    ax[0, 0].legend(); ax[0, 0].grid(True, which="both", alpha=0.3)

    ax[0, 1].plot(xm * 1e9, wm, "-", color="#2ca02c", lw=2)
    ax[0, 1].axhline(0, color="k", lw=0.5)
    ax[0, 1].fill_between([0, xm[-1] * 1e9], -1, 1, color="none")
    ax[0, 1].set_xlabel("x [nm]"); ax[0, 1].set_ylabel("normalized deflection")
    ax[0, 1].set_title("first buckling mode of the released sheet\n(clamped-clamped bridge, L=120 nm)")
    ax[0, 1].grid(True, alpha=0.3)

    ax[1, 0].plot(s * 1e9, U6 * 1e9, "-", color="#d62728", label="t=6 nm")
    ax[1, 0].plot(s * 1e9, U8 * 1e9, "-", color="#1f77b4", label="t=8 nm")
    ax[1, 0].axvline(seq6 * 1e9, color="#d62728", ls="--", alpha=0.6)
    ax[1, 0].axvline(seq8 * 1e9, color="#1f77b4", ls="--", alpha=0.6)
    ax[1, 0].axhline(0, color="k", lw=0.5)
    ax[1, 0].set_ylim(-8, 40)                           # zoom so the sub-zero dip (stiction) shows
    ax[1, 0].set_xlabel("peel (detachment) length s [nm]")
    ax[1, 0].set_ylabel("total energy per width [nJ/m]")
    ax[1, 0].set_title(f"stiction energy balance (bending − adhesion), half-length {a_half*1e9:.0f} nm\n"
                       f"U<0 ⇒ collapsed state wins (t=6 nm sticks, t=8 nm does not)")
    ax[1, 0].legend(); ax[1, 0].grid(True, alpha=0.3)

    cmap = ListedColormap(["#2ca02c", "#ff7f0e", "#1f77b4", "#d62728"])
    im = ax[1, 1].imshow(regime, origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=3,
                         extent=[Lg[0] * 1e9, Lg[-1] * 1e9, tg[0] * 1e9, tg[-1] * 1e9])
    cb = fig.colorbar(im, ax=ax[1, 1], ticks=[0.4, 1.1, 1.9, 2.6])
    cb.ax.set_yticklabels(["safe", "buckle", "stick", "both"])
    ax[1, 1].set_xlabel("nanosheet length L [nm]"); ax[1, 1].set_ylabel("thickness t [nm]")
    ax[1, 1].set_title("RELEASE PROCESS WINDOW (L, t)\nthin+long → buckling & stiction risk")

    fig.suptitle("GAA nanosheet release mechanics (post SiGe selective etch): weak-form "
                 "beam FE buckling + stiction → process window (for a process/etch house)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main()
