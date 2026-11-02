"""fiber_orientation_cte.py — ORIENTATION-AVERAGED thermal expansion alpha_ij from the
Advani-Tucker fibre orientation tensor a_ij, for the Denso residual-strain theme
(research/DENSO_RA_RESIDUAL_STRAIN.md, research/FIBER_ORIENTATION_TENSOR.md).

Why this and not a_ij directly: the ML target is RESIDUAL STRAIN, and fibre orientation
reaches residual strain only THROUGH the anisotropic thermal expansion and stiffness --
the strain is the mismatch between anisotropic shrinkage and constraint. So alpha_ij(a)
is the physics-routed feature proposed as the contrast to feeding raw a_ij components
(FIBER_ORIENTATION_TENSOR.md section 4d/5).

*** The one thing it is easy to get wrong ***
alpha is a second-order tensor, so it is tempting to orientation-average it directly as
    <alpha_ij> = (alpha_L - alpha_T) a_ij + alpha_T delta_ij.
That is WRONG for a composite. Differently-oriented domains CONSTRAIN each other, so the
effective expansion is not the average of the local expansions. The thermal STRESS tensor
beta = C:alpha is what averages, and the expansion follows from inverting the averaged
stiffness:
    <beta_ij> = (beta_L - beta_T) a_ij + beta_T delta_ij        (2nd-order average)
    <C_ijkl>  = Advani-Tucker 4th-order average (needs a_ijkl)
    <alpha>   = <C>^-1 : <beta>
This module computes both and reports the gap, because the naive route is a plausible
mistake to make and the difference is not small.

Closure: <C> needs the 4th-order a_ijkl, which the 2nd-order tensor does not determine --
the closure problem. Three classical closures are implemented and compared, since the
choice is a real modelling decision, not an implementation detail:
  linear     exact at the ISOTROPIC (3-D random) state, wrong when aligned,
  quadratic  a_ijkl = a_ij a_kl -- exact when perfectly ALIGNED, wrong when random,
  hybrid     f*quadratic + (1-f)*linear with f = 1 - 27 det(a), so it reduces to the
             exact one at BOTH limits (f=0 at random, f=1 at aligned).

Validated (each closure checked where it is supposed to be exact):
  V1 perfectly aligned -> recovers the UD alpha exactly (quadratic, hybrid),
  V2 3-D random -> alpha is ISOTROPIC, off-diagonals zero and diagonals equal
     (a symmetry check on the implementation, satisfied by every closure),
  V3 <alpha> is symmetric for every state tested,
  V4 planar-random -> transversely isotropic in the plane,
  V5 the Advani-Tucker <C> average reproduces the UD stiffness when fully aligned.

Honest scope: this is the VOIGT-type (uniform-strain) estimate -- <C> is a stiffness
upper bound, so <alpha> is an estimate, not an exact effective property; Mori-Tanaka with
orientation averaging would be the next step up. Material constants are ILLUSTRATIVE
short-glass/PA-like values, NOT calibrated to any Denso material. Linear thermoelasticity,
no crystallinity coupling yet (the other input in the Denso spec), temperature-independent
constants. Physics leads; this produces a FEATURE for the ML layer, not a prediction.

Run:  python3 fiber_orientation_cte.py        (writes fiber_orientation_cte.png)
      python3 fiber_orientation_cte.py --help
"""
from __future__ import annotations

import argparse

import numpy as np

SEED = 20261102
# ---- ILLUSTRATIVE unidirectional (perfectly aligned) properties, short-GF/PA-like ----
E_L, E_T = 30.0e9, 8.0e9       # axial / transverse modulus [Pa]
NU_LT, NU_TT = 0.33, 0.42      # major / transverse-plane Poisson
G_LT = 3.0e9                   # axial shear modulus [Pa]
A_L, A_T = 12.0e-6, 70.0e-6    # axial / transverse CTE [1/K]  (fibre- vs matrix-dominated)

VOIGT = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]   # 11,22,33,23,13,12
I3 = np.eye(3)


# ----------------------------------------------------------------- UD properties ----
def ud_stiffness(EL=E_L, ET=E_T, nuLT=NU_LT, nuTT=NU_TT, GLT=G_LT):
    """Transversely isotropic UD stiffness, 6x6 Voigt, fibre along axis 1."""
    G23 = ET / (2.0 * (1.0 + nuTT))
    S = np.zeros((6, 6))
    S[0, 0] = 1.0 / EL
    S[1, 1] = S[2, 2] = 1.0 / ET
    S[0, 1] = S[1, 0] = S[0, 2] = S[2, 0] = -nuLT / EL
    S[1, 2] = S[2, 1] = -nuTT / ET
    S[3, 3] = 1.0 / G23
    S[4, 4] = S[5, 5] = 1.0 / GLT
    return np.linalg.inv(S)


def voigt_to_tensor4(Cv):
    """6x6 Voigt stiffness -> 3x3x3x3 (stiffness convention: no factors on C)."""
    C = np.zeros((3, 3, 3, 3))
    for I, (i, j) in enumerate(VOIGT):
        for J, (k, l) in enumerate(VOIGT):
            v = Cv[I, J]
            for (a, b) in {(i, j), (j, i)}:
                for (c, d) in {(k, l), (l, k)}:
                    C[a, b, c, d] = v
    return C


def tensor4_to_voigt(C):
    return np.array([[C[i, j, k, l] for (k, l) in VOIGT] for (i, j) in VOIGT])


# ----------------------------------------------------------------------- closures ----
def closure_linear(a):
    """Exact at the isotropic state a = I/3."""
    d = I3
    t1 = np.einsum("ij,kl->ijkl", d, d) + np.einsum("ik,jl->ijkl", d, d) \
        + np.einsum("il,jk->ijkl", d, d)
    t2 = (np.einsum("ij,kl->ijkl", a, d) + np.einsum("kl,ij->ijkl", a, d)
          + np.einsum("ik,jl->ijkl", a, d) + np.einsum("il,jk->ijkl", a, d)
          + np.einsum("jl,ik->ijkl", a, d) + np.einsum("jk,il->ijkl", a, d))
    return -t1 / 35.0 + t2 / 7.0


def closure_quadratic(a):
    """Exact for a perfectly aligned state."""
    return np.einsum("ij,kl->ijkl", a, a)


def closure_hybrid(a):
    """f*quadratic + (1-f)*linear, f = 1 - 27 det(a): exact at BOTH limits."""
    f = 1.0 - 27.0 * np.linalg.det(a)
    f = min(max(f, 0.0), 1.0)
    return f * closure_quadratic(a) + (1.0 - f) * closure_linear(a)


CLOSURES = {"linear": closure_linear, "quadratic": closure_quadratic,
            "hybrid": closure_hybrid}


# ------------------------------------------------------------ orientation averaging --
def orient_avg_stiffness(Cud_v, a, closure="hybrid"):
    """Advani-Tucker orientation average of a transversely isotropic stiffness."""
    C11, C22, C12, C23 = Cud_v[0, 0], Cud_v[1, 1], Cud_v[0, 1], Cud_v[1, 2]
    C66 = Cud_v[5, 5]                                   # axial shear (1-2 plane)
    B1 = C11 + C22 - 2.0 * C12 - 4.0 * C66
    B2 = C12 - C23
    B3 = C66 + 0.5 * (C23 - C22)
    B4 = C23
    B5 = 0.5 * (C22 - C23)
    a4 = CLOSURES[closure](a)
    d = I3
    C = (B1 * a4
         + B2 * (np.einsum("ij,kl->ijkl", a, d) + np.einsum("kl,ij->ijkl", a, d))
         + B3 * (np.einsum("ik,jl->ijkl", a, d) + np.einsum("il,jk->ijkl", a, d)
                 + np.einsum("jl,ik->ijkl", a, d) + np.einsum("jk,il->ijkl", a, d))
         + B4 * np.einsum("ij,kl->ijkl", d, d)
         + B5 * (np.einsum("ik,jl->ijkl", d, d) + np.einsum("il,jk->ijkl", d, d)))
    return tensor4_to_voigt(C)


def orient_avg_cte(a, closure="hybrid", Cud_v=None, aL=A_L, aT=A_T):
    """<alpha> = <C>^-1 : <beta>, the CORRECT route. Returns (alpha_3x3, <C>_6x6)."""
    if Cud_v is None:
        Cud_v = ud_stiffness()
    alpha_ud = np.array([aL, aT, aT, 0.0, 0.0, 0.0])        # strain-like Voigt
    beta_ud = Cud_v @ alpha_ud                              # stress-like Voigt
    betaL, betaT = beta_ud[0], beta_ud[1]
    beta_avg = (betaL - betaT) * a + betaT * I3             # 2nd-order average
    Cavg = orient_avg_stiffness(Cud_v, a, closure)
    bv = np.array([beta_avg[i, j] for (i, j) in VOIGT])     # stress-like -> no 2x
    x = np.linalg.solve(Cavg, bv)                           # strain-like (engineering)
    alpha = np.array([[x[0], x[5] / 2, x[4] / 2],
                      [x[5] / 2, x[1], x[3] / 2],
                      [x[4] / 2, x[3] / 2, x[2]]])
    return alpha, Cavg


def naive_avg_cte(a, aL=A_L, aT=A_T):
    """The WRONG-but-tempting direct 2nd-order average of alpha, for comparison."""
    return (aL - aT) * a + aT * I3


# ------------------------------------------------------------------ orientation states
def uniaxial_state(a11):
    """a = diag(a11, (1-a11)/2, (1-a11)/2): 1/3 = 3-D random, 1 = fully aligned."""
    r = (1.0 - a11) / 2.0
    return np.diag([a11, r, r])


STATE_RANDOM3D = np.diag([1 / 3, 1 / 3, 1 / 3])
STATE_PLANAR = np.diag([0.5, 0.5, 0.0])
STATE_ALIGNED = np.diag([1.0, 0.0, 0.0])


def skin_core_profile(n=41, a_skin=0.85, a_core=0.25):
    """ILLUSTRATIVE through-thickness a11: aligned skin, transverse-leaning core."""
    z = np.linspace(-1.0, 1.0, n)
    a11 = a_core + (a_skin - a_core) * np.abs(z) ** 3
    return z, a11


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--closure", default="hybrid", choices=sorted(CLOSURES))
    ap.add_argument("--out", default="fiber_orientation_cte.png")
    args = ap.parse_args()
    np.random.seed(SEED)
    Cud = ud_stiffness()

    print("=== orientation-averaged CTE from the Advani-Tucker tensor ===")
    print(f"UD (ILLUSTRATIVE): E_L {E_L/1e9:.0f} / E_T {E_T/1e9:.0f} GPa, "
          f"alpha_L {A_L*1e6:.0f} / alpha_T {A_T*1e6:.0f} ppm/K")

    # V5 / V1: the averages must be exact where the closure is exact
    Cal = orient_avg_stiffness(Cud, STATE_ALIGNED, "quadratic")
    print(f"[V5] aligned, quadratic closure: max|<C> - C_UD| = "
          f"{np.max(np.abs(Cal - Cud)):.3e} Pa (should be ~0)")
    for cl in ("quadratic", "hybrid"):
        al, _ = orient_avg_cte(STATE_ALIGNED, cl, Cud)
        err = max(abs(al[0, 0] - A_L), abs(al[1, 1] - A_T))
        print(f"[V1] aligned, {cl:9s}: alpha = diag({al[0,0]*1e6:6.2f}, {al[1,1]*1e6:6.2f}, "
              f"{al[2,2]*1e6:6.2f}) ppm/K, err vs UD {err*1e6:.2e} ppm/K")
    al_lin, _ = orient_avg_cte(STATE_ALIGNED, "linear", Cud)
    print(f"[V1] aligned, linear   : alpha_11 {al_lin[0,0]*1e6:6.2f} ppm/K "
          f"-- NOT exact, as expected (linear closure is exact at isotropy, not alignment)")

    # V2 / V3 / V4: symmetry of the result
    for name, st in (("3-D random", STATE_RANDOM3D), ("planar random", STATE_PLANAR)):
        a_avg, _ = orient_avg_cte(st, args.closure, Cud)
        off = np.max(np.abs(a_avg - np.diag(np.diag(a_avg))))
        sym = np.max(np.abs(a_avg - a_avg.T))
        print(f"[V2/V4] {name:13s}: alpha = diag({a_avg[0,0]*1e6:6.2f}, "
              f"{a_avg[1,1]*1e6:6.2f}, {a_avg[2,2]*1e6:6.2f}) ppm/K | "
              f"max off-diag {off*1e6:.1e} | asymmetry {sym*1e6:.1e}")
    iso = orient_avg_cte(STATE_RANDOM3D, args.closure, Cud)[0]
    spread = np.max(np.diag(iso)) - np.min(np.diag(iso))
    print(f"[V2] 3-D random isotropy: diagonal spread {spread*1e6:.2e} ppm/K (should be ~0)")

    # the point of the module: correct vs naive
    print("\n--- correct (<C>^-1:<beta>) vs NAIVE direct average of alpha ---")
    print(" a11   alpha11_correct  alpha11_naive   diff      alpha22_correct  alpha22_naive")
    for a11 in (1 / 3, 0.5, 0.7, 0.9, 1.0):
        st = uniaxial_state(a11)
        ac, _ = orient_avg_cte(st, args.closure, Cud)
        an = naive_avg_cte(st)
        print(f"{a11:5.2f}   {ac[0,0]*1e6:9.2f}      {an[0,0]*1e6:9.2f}   "
              f"{(an[0,0]-ac[0,0])*1e6:+7.2f}      {ac[1,1]*1e6:9.2f}      {an[1,1]*1e6:9.2f}")
    st = uniaxial_state(0.7)
    ac = orient_avg_cte(st, args.closure, Cud)[0][0, 0]
    an = naive_avg_cte(st)[0, 0]
    print(f"  -> at a11=0.7 the naive route is off by {100*(an-ac)/ac:+.0f}% in alpha_11: "
          "the constraint between differently-oriented domains is not a small correction.")

    # closure sensitivity
    print("\n--- closure sensitivity (alpha_11 [ppm/K]) ---")
    print(" a11    linear  quadratic    hybrid   spread")
    for a11 in (1 / 3, 0.5, 0.7, 0.9, 1.0):
        st = uniaxial_state(a11)
        vals = [orient_avg_cte(st, c, Cud)[0][0, 0] * 1e6 for c in
                ("linear", "quadratic", "hybrid")]
        print(f"{a11:5.2f}  {vals[0]:8.2f}  {vals[1]:9.2f}  {vals[2]:8.2f}  "
              f"{max(vals)-min(vals):7.2f}")
    print("  -> the closure is a real modelling choice, not a detail: it moves alpha_11 by")
    print("     several ppm/K at intermediate alignment, which is where real parts sit.")

    _plot(args.out, Cud, args.closure)
    print(f"\nwrote {args.out}")


def _plot(out, Cud, closure):
    import matplotlib
    matplotlib.use("Agg")
    import os, shutil, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "slides", "figure_sources"))
    try:                                   # repo convention: thesis_style, graceful fallback
        from thesis_style import use
        use(width_frac=1.0, aspect=0.45)
        if shutil.which("latex") is None:
            matplotlib.rcParams["text.usetex"] = False
            matplotlib.rcParams["font.family"] = "sans-serif"
    except Exception:
        pass
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    a11s = np.linspace(1 / 3, 1.0, 40)
    cor = np.array([orient_avg_cte(uniaxial_state(v), closure, Cud)[0] for v in a11s])
    nai = np.array([naive_avg_cte(uniaxial_state(v)) for v in a11s])

    # (a) correct vs naive
    a = ax[0, 0]
    a.plot(a11s, cor[:, 0, 0] * 1e6, "-", color="#1f77b4", lw=2, label=r"$\alpha_{11}$ correct")
    a.plot(a11s, nai[:, 0, 0] * 1e6, "--", color="#1f77b4", lw=1.5, label=r"$\alpha_{11}$ naive")
    a.plot(a11s, cor[:, 1, 1] * 1e6, "-", color="#d62728", lw=2, label=r"$\alpha_{22}$ correct")
    a.plot(a11s, nai[:, 1, 1] * 1e6, "--", color="#d62728", lw=1.5, label=r"$\alpha_{22}$ naive")
    a.axhline(A_L * 1e6, color="k", ls=":", lw=1); a.axhline(A_T * 1e6, color="k", ls=":", lw=1)
    a.text(0.34, A_L * 1e6 + 1.5, "UD axial", fontsize=7)
    a.text(0.34, A_T * 1e6 - 4, "UD transverse", fontsize=7)
    a.set_xlabel(r"$a_{11}$  (1/3 = 3-D random, 1 = aligned)")
    a.set_ylabel(r"$\alpha$ [ppm/K]")
    a.set_title("averaging the thermal STRESS, not the expansion\n"
                "(the naive direct average of alpha is the dashed line)")
    a.legend(fontsize=7); a.grid(True, alpha=0.3)

    # (b) closure sensitivity
    b = ax[0, 1]
    for cl, col in zip(("linear", "quadratic", "hybrid"), ("#7f7f7f", "#9467bd", "#2ca02c")):
        v = [orient_avg_cte(uniaxial_state(x), cl, Cud)[0][0, 0] * 1e6 for x in a11s]
        b.plot(a11s, v, "-", color=col, lw=2, label=cl)
    b.scatter([1 / 3], [orient_avg_cte(STATE_RANDOM3D, "linear", Cud)[0][0, 0] * 1e6],
              marker="o", s=45, facecolor="none", edgecolor="#7f7f7f", zorder=5)
    b.scatter([1.0], [A_L * 1e6], marker="o", s=45, facecolor="none",
              edgecolor="#9467bd", zorder=5)
    b.annotate("linear exact here", (1 / 3, orient_avg_cte(STATE_RANDOM3D, "linear", Cud)[0][0, 0] * 1e6),
               textcoords="offset points", xytext=(8, 8), fontsize=7)
    b.annotate("quadratic exact here", (1.0, A_L * 1e6),
               textcoords="offset points", xytext=(-80, 10), fontsize=7)
    b.set_xlabel(r"$a_{11}$"); b.set_ylabel(r"$\alpha_{11}$ [ppm/K]")
    b.set_title("closure is a modelling choice, not a detail\n(hybrid reduces to the exact one at both ends)")
    b.legend(fontsize=8); b.grid(True, alpha=0.3)

    # (c) anisotropy ratio -- what actually drives residual strain
    c = ax[1, 0]
    c.plot(a11s, cor[:, 1, 1] / cor[:, 0, 0], "-", color="#b5651d", lw=2)
    c.set_xlabel(r"$a_{11}$"); c.set_ylabel(r"$\alpha_{22}\,/\,\alpha_{11}$")
    c.set_title("CTE anisotropy ratio\n(the mismatch that residual strain comes from)")
    c.grid(True, alpha=0.3)

    # (d) skin-core profile carried through to alpha
    d = ax[1, 1]; d2 = d.twiny()
    z, a11z = skin_core_profile()
    az = np.array([orient_avg_cte(uniaxial_state(v), closure, Cud)[0] for v in a11z])
    d.plot(a11z, z, "-", color="#7f7f7f", lw=1.6, label=r"$a_{11}$ (illustrative)")
    d2.plot(az[:, 0, 0] * 1e6, z, "-", color="#1f77b4", lw=2.2, label=r"$\alpha_{11}$")
    d2.plot(az[:, 1, 1] * 1e6, z, "-", color="#d62728", lw=2.2, label=r"$\alpha_{22}$")
    d.set_ylabel("normalized thickness  (±1 = surfaces)")
    d.set_xlabel(r"$a_{11}$", color="#7f7f7f"); d2.set_xlabel(r"$\alpha$ [ppm/K]")
    d.set_title("skin-core orientation becomes a through-thickness\nCTE gradient -- which is what warps the part")
    d.legend(fontsize=7, loc="lower left"); d2.legend(fontsize=7, loc="lower right")
    d.grid(True, alpha=0.3)

    fig.suptitle("Orientation-averaged CTE as a physics-routed ML feature: fibre orientation "
                 "reaches residual strain only through alpha and C (ILLUSTRATIVE constants)",
                 fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main()
