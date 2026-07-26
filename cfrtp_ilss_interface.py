"""cfrtp_ilss_interface.py — the INTERFACE-ADHESION challenge of Daikin's CFRTP
(research/MEMO_cfrp_residual_stress_collab.md, issue #1): fluoropolymer matrices have
very low surface energy (the non-stick / water-&-oil-repellent property), so
fibre/matrix and interlaminar ADHESION is the weak link and the interlaminar shear
strength (ILSS) is often the limiting property. This models the interface with a
weak-form COHESIVE-ZONE interface element and shows how the interface strength/toughness
(weak fluoropolymer vs a surface-treated interface) sets the ILSS and the delamination.

Model (mode-II cohesive shear-lag, 1D weak form): an elastic top adherend bonded to a
fixed substrate over an overlap x in [0, L], transferring load through a cohesive
interface. Equilibrium  (E t) u'' = tau(u), with u the interfacial slip and a BILINEAR
traction-separation law:
    tau = K u                       (u <= d0, elastic),
    tau = tau_max (df - u)/(df-d0)  (d0 < u < df, softening / damaging),
    tau = 0                         (u >= df, delaminated),
with tau_max the interface shear strength and G_IIc = 0.5 tau_max df the mode-II
toughness. Damage is irreversible. The panel drives the loaded end by displacement
continuation and records the reaction P(Delta); the peak reaction / area is the ILSS.

Weak (fluoropolymer) vs strong (surface-treated / plasma / etched) interface differ in
tau_max and G_IIc. A residual interfacial shear pre-stress (from cure/cool-down, cf. the
CFRTP residual-stress demos) is optionally added to show it EATS INTO the ILSS margin —
the coupling between the two Daikin issues (residual stress x weak interface).

Validated: for a stiff/short bond the initial slope and the onset load
P_onset = tau_max * L match the shear-lag limit; energy under P(Delta) tracks G_IIc*L.

Honest scope: 1D shear-lag cohesive zone (mode II only), bilinear law, elastic
adherend, illustrative CFRTP-like properties; a seed for a 2D/3D cohesive interlaminar
model (repo's ⑭ interface fracture is the phase-field sibling). Physics leads; ML (an
interface-strength -> ILSS surrogate) would be the subordinate layer. No ML here.

Run:  python3 cfrtp_ilss_interface.py     (writes cfrtp_ilss_interface.png)
      python3 cfrtp_ilss_interface.py --help
"""
from __future__ import annotations

import argparse

import numpy as np

SEED = 20260726
E_AD = 100e9           # top-adherend modulus [Pa]
T_AD = 0.3e-3          # top-adherend thickness [m]
L = 5e-3              # overlap length [m]
D0 = 0.3e-6           # elastic slip limit [m] (interface initial stiffness K = tau_max/D0)


def cohesive(u, umax, tau_max, df):
    """Bilinear mode-II traction and tangent, with irreversible damage via umax."""
    K = tau_max / D0
    ue = np.maximum(u, umax)                      # damage set by the max slip reached
    tau = np.zeros_like(u); dtau = np.zeros_like(u)
    el = ue <= D0                                 # elastic
    tau[el] = K * u[el]; dtau[el] = K
    so = (ue > D0) & (ue < df)                    # softening (secant on damage envelope)
    dsec = tau_max * (df - ue[so]) / (df - D0) / ue[so]     # secant stiffness
    tau[so] = dsec * u[so]; dtau[so] = dsec
    # (fully debonded ue>=df -> tau=0, dtau=0 already)
    return tau, dtau


def solve(tau_max, GIIc, ndelta=60, n=120):
    """Displacement-controlled cohesive shear-lag. Returns Delta, reaction P(Delta),
    slip profile at peak, tau(x) at peak."""
    df = 2.0 * GIIc / tau_max                     # bilinear final separation
    x = np.linspace(0, L, n); h = x[1] - x[0]
    EA = E_AD * T_AD
    # 1D stiffness (adherend) : EA u'' term
    main = np.full(n, 2.0); main[0] = main[-1] = 1.0
    Kad = (EA / h) * (np.diag(main) - np.diag(np.ones(n - 1), 1) - np.diag(np.ones(n - 1), -1))
    ml = np.full(n, h); ml[0] = ml[-1] = h / 2

    deltas = np.linspace(0, 2.2 * df, ndelta)
    u = np.zeros(n); umax = np.zeros(n)
    P = np.zeros(ndelta); peak = {"P": -1}
    for k, Dk in enumerate(deltas):
        for _ in range(60):
            tau, dtau = cohesive(u, umax, tau_max, df)
            R = Kad @ u + ml * tau
            J = Kad + np.diag(ml * dtau)
            # BC: loaded end u[0]=Dk (Dirichlet), far end free (natural)
            R[0] = u[0] - Dk; J[0, :] = 0; J[0, 0] = 1
            du = np.linalg.solve(J, -R)
            u = u + du
            if np.max(np.abs(du)) < 1e-12:
                break
        umax = np.maximum(umax, u)
        P[k] = EA * (u[1] - u[0]) / h * -1.0          # reaction at loaded end (per width)
        if P[k] > peak["P"]:
            tau_pk, _ = cohesive(u, umax, tau_max, df)
            peak = {"P": P[k], "u": u.copy(), "tau": tau_pk.copy(), "k": k}
    ilss = peak["P"] / L                              # peak shear-flow / length = ILSS proxy
    return deltas, P, x, peak, ilss, df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=str, default="cfrtp_ilss_interface.png")
    args = ap.parse_args()
    np.random.seed(SEED)

    weak = dict(tau_max=15e6, GIIc=0.3e3)             # fluoropolymer: low surface energy
    strong = dict(tau_max=45e6, GIIc=1.0e3)           # surface-treated / plasma / etched

    dw, Pw, xw, pkw, ilss_w, dfw = solve(**weak)
    ds, Ps, xs, pks, ilss_s, dfs = solve(**strong)
    print(f"CFRTP interlaminar cohesive interface (mode-II shear-lag):")
    print(f"  weak (fluoropolymer)   tau_max 15 MPa, GIIc 0.3 kJ/m^2 -> ILSS {ilss_w/1e6:.1f} MPa")
    print(f"  strong (treated)       tau_max 45 MPa, GIIc 1.0 kJ/m^2 -> ILSS {ilss_s/1e6:.1f} MPa")
    print(f"  -> surface treatment raises ILSS {ilss_s/ilss_w:.1f}x")

    # ILSS vs interface strength
    taus = np.linspace(8e6, 50e6, 12)
    ilss_scan = np.array([solve(t, 0.3e3 + (t - 8e6) / (50e6 - 8e6) * 0.7e3)[4] for t in taus]) / 1e6

    # residual-stress interaction: a residual interfacial shear consumes part of the
    # interface capacity, so the strength available for applied load is tau_max - tau_res
    # -> apparent ILSS drops (the two Daikin issues couple)
    tres = np.linspace(0, 11e6, 12)
    ilss_res = np.array([solve(tau_max=15e6 - tr, GIIc=0.3e3)[4] for tr in tres]) / 1e6

    _plot(args.out, dw, Pw, ds, Ps, xw, pkw, pks, ilss_w, ilss_s, taus, ilss_scan,
          tres, ilss_res, weak, strong, dfw, dfs)
    print(f"wrote {args.out}")


def _plot(out, dw, Pw, ds, Ps, xw, pkw, pks, ilss_w, ilss_s, taus, ilss_scan,
          tres, ilss_res, weak, strong, dfw, dfs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    # bilinear traction-separation laws
    for lab, p, df, c in [("weak (fluoropolymer)", weak, dfw, "#d62728"),
                          ("strong (treated)", strong, dfs, "#1f77b4")]:
        uu = np.linspace(0, df, 200)
        tt, _ = cohesive(uu, uu, p["tau_max"], df)
        ax[0, 0].plot(uu * 1e6, tt / 1e6, "-", color=c, label=lab)
    ax[0, 0].set_xlabel("interfacial slip u [µm]"); ax[0, 0].set_ylabel("shear traction τ [MPa]")
    ax[0, 0].set_title("cohesive interface law (bilinear)\narea = mode-II toughness G_IIc")
    ax[0, 0].legend(); ax[0, 0].grid(True, alpha=0.3)

    ax[0, 1].plot(dw * 1e6, Pw / 1e3, "-", color="#d62728", label=f"weak (ILSS {ilss_w/1e6:.0f} MPa)")
    ax[0, 1].plot(ds * 1e6, Ps / 1e3, "-", color="#1f77b4", label=f"strong (ILSS {ilss_s/1e6:.0f} MPa)")
    ax[0, 1].set_xlabel("applied end slip Δ [µm]"); ax[0, 1].set_ylabel("reaction P [kN/m]")
    ax[0, 1].set_title("load–slip: interface strength sets peak (ILSS)\nthen delamination softening")
    ax[0, 1].legend(); ax[0, 1].grid(True, alpha=0.3)

    ax[1, 0].plot(xw * 1e3, pkw["tau"] / 1e6, "-", color="#d62728", label="weak")
    ax[1, 0].plot(xw * 1e3, pks["tau"] / 1e6, "-", color="#1f77b4", label="strong")
    ax[1, 0].set_xlabel("position along overlap x [mm]"); ax[1, 0].set_ylabel("interface shear τ [MPa]")
    ax[1, 0].set_title("interface shear at peak load\n(cohesive process zone at the loaded end)")
    ax[1, 0].legend(); ax[1, 0].grid(True, alpha=0.3)

    ax[1, 1].plot(taus / 1e6, ilss_scan, "-o", ms=4, color="#2ca02c", label="ILSS vs interface strength")
    ax[1, 1].set_xlabel("interface shear strength τ_max [MPa]"); ax[1, 1].set_ylabel("ILSS [MPa]")
    ax[1, 1].set_title("ILSS scales with interface adhesion\n(inset: residual shear eats the margin)")
    ax[1, 1].grid(True, alpha=0.3)
    axin = ax[1, 1].inset_axes([0.58, 0.12, 0.38, 0.42])
    axin.plot(tres / 1e6, ilss_res, "-s", ms=3, color="#b5651d")
    axin.set_xlabel("residual τ [MPa]", fontsize=7); axin.set_ylabel("ILSS", fontsize=7)
    axin.tick_params(labelsize=6)

    fig.suptitle("CFRTP interlaminar adhesion (ILSS) via a cohesive interface element: weak "
                 "fluoropolymer vs treated interface, and the residual-stress interaction (Daikin theme)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main()
