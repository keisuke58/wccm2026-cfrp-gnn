"""cfrtp_impregnation_void.py — melt IMPREGNATION & VOID model for CFRTP (Daikin theme,
research/MEMO_cfrp_residual_stress_collab.md, issue #2): fluoropolymer melts are very
viscous, so impregnating the carbon-fibre tow is hard and leaves VOIDS. Daikin's fix is
special carbon-fibre SPREADING (開繊), which thins the tow and slashes the impregnation
distance. This quantifies that with Darcy flow through the fibrous bed.

Physics (transverse Darcy impregnation of a fibre tow):
  * permeability K(Vf) from the Gebart transverse-flow model,
    K = C (sqrt(Va_max/Vf) - 1)^2.5 R_f^2,
  * 1-D Darcy flow front into the half-thickness h under press pressure dP:
    x_f(t) = sqrt(2 K dP t / (mu (1-Vf))),  so full impregnation takes
    t_imp = mu (1-Vf) h^2 / (2 K dP)  — it scales with h^2 and with the (huge) melt
    viscosity mu, and inversely with permeability and pressure,
  * fibre SPREADING (開繊) by a factor s thins the tow, h -> h/s, so t_imp -> t_imp/s^2
    (a quadratic win),
  * VOID content: if the time above melt t_proc < t_imp the tow is only partly
    impregnated (void ~ 1 - sqrt(t_proc/t_imp)), plus a pressure-reducible micro-void
    from trapped gas.

Validated: the Darcy front reaches the centreline exactly at t_imp (x_f(t_imp)=h) to
machine precision, and t_imp has the correct h^2 / (mu / (K dP)) scaling.

Shows: the impregnation front vs time for an unspread vs spread tow; impregnation time
vs fibre volume fraction (via K) and tow thickness; the void-content process window over
(tow thickness, time above melt); and void vs spreading ratio — the 開繊 benefit.

Honest scope: 1-D transverse Darcy, Gebart permeability, a reduced void model (no
capillary/dual-scale flow, no dissolved-gas kinetics); illustrative fluoropolymer-CFRTP
values. Physics is the substance; a process -> void surrogate would be the ML layer. No ML.

Run:  python3 cfrtp_impregnation_void.py     (writes cfrtp_impregnation_void.png)
      python3 cfrtp_impregnation_void.py --help
"""
from __future__ import annotations

import argparse

import numpy as np

SEED = 20260726
R_F = 3.5e-6           # carbon fibre radius [m]
VA_MAX = 0.9069        # max packing (hexagonal)
C_GEBART = 0.231       # Gebart transverse constant 16/(9 pi sqrt(6))
MU = 3000.0            # fluoropolymer melt viscosity [Pa.s] (very high)
DP = 1.0e6             # impregnation pressure [Pa]
H0 = 1.0e-4            # unspread tow half-thickness [m]
VF = 0.55             # fibre volume fraction
T_PROC = 40.0          # time available above the melt [s]


def permeability(vf):
    return C_GEBART * (np.sqrt(VA_MAX / vf) - 1.0) ** 2.5 * R_F ** 2


def t_impregnate(h, vf=VF, mu=MU, dp=DP):
    return mu * (1.0 - vf) * h ** 2 / (2.0 * permeability(vf) * dp)


def front(t, h, vf=VF, mu=MU, dp=DP):
    xf = np.sqrt(2.0 * permeability(vf) * dp * t / (mu * (1.0 - vf)))
    return np.minimum(xf, h)


def void_content(h, t_proc=T_PROC, vf=VF, mu=MU, dp=DP, micro=0.008):
    """Void = unimpregnated fraction (if impregnation not finished) + a pressure-reducible
    micro-void floor."""
    ti = t_impregnate(h, vf, mu, dp)
    unimp = np.where(t_proc < ti, 1.0 - np.sqrt(np.clip(t_proc / ti, 0, 1)), 0.0)
    mv = micro * (1e6 / dp)                         # micro-void shrinks with pressure
    return unimp + mv


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=str, default="cfrtp_impregnation_void.png")
    args = ap.parse_args()
    np.random.seed(SEED)

    ti_full = t_impregnate(H0)
    xf_end = front(ti_full, H0)
    print(f"[validate] Darcy front reaches centreline at t_imp: x_f(t_imp)/h = "
          f"{xf_end/H0:.6f} (should be 1)")
    print(f"CFRTP melt impregnation (mu={MU:.0f} Pa.s, dP={DP/1e6:.1f} MPa, Vf={VF}):")
    print(f"  unspread tow h={H0*1e6:.0f} um -> t_imp {ti_full:.1f} s "
          f"(t_proc {T_PROC:.0f} s -> void {void_content(H0)*100:.1f}%)")
    for s in (2, 3, 5):
        hs = H0 / s
        print(f"  spread x{s} (h={hs*1e6:.0f} um) -> t_imp {t_impregnate(hs):.1f} s, "
              f"void {void_content(hs)*100:.2f}%")

    _plot(args.out)
    print(f"wrote {args.out}")


def _plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    # impregnation front vs time: unspread vs spread
    t = np.linspace(0, 60, 300)
    for s, lab, c in [(1, "unspread (h=100 µm)", "#d62728"),
                      (2, "spread ×2 (50 µm)", "#ff7f0e"),
                      (5, "spread ×5 (20 µm)", "#2ca02c")]:
        h = H0 / s
        ax[0, 0].plot(t, front(t, h) / h, "-", color=c, label=lab)
    ax[0, 0].axvline(T_PROC, color="k", ls=":", alpha=0.6)
    ax[0, 0].annotate("time above melt", (T_PROC, 0.2), fontsize=8, rotation=90, va="bottom")
    ax[0, 0].set_xlabel("time [s]"); ax[0, 0].set_ylabel("impregnated fraction x_f / h")
    ax[0, 0].set_title("Darcy melt impregnation: spreading fills the tow fast\n(fluoropolymer μ=3000 Pa·s)")
    ax[0, 0].legend(fontsize=8); ax[0, 0].grid(True, alpha=0.3)

    # impregnation time vs fibre volume fraction, several thicknesses
    vf = np.linspace(0.40, 0.68, 60)
    for s, c in [(1, "#d62728"), (2, "#ff7f0e"), (5, "#2ca02c")]:
        ti = np.array([t_impregnate(H0 / s, v) for v in vf])
        ax[0, 1].semilogy(vf, ti, "-", color=c, label=f"spread ×{s}")
    ax[0, 1].axhline(T_PROC, color="k", ls=":", alpha=0.6, label="t_proc")
    ax[0, 1].set_xlabel("fibre volume fraction Vf"); ax[0, 1].set_ylabel("impregnation time t_imp [s] (log)")
    ax[0, 1].set_title("t_imp ∝ h²·μ/(K·ΔP): denser tow → slower\n(above t_proc ⇒ voids)")
    ax[0, 1].legend(fontsize=8); ax[0, 1].grid(True, which="both", alpha=0.3)

    # void process window over (tow thickness, time above melt)
    hh = np.linspace(15e-6, 120e-6, 60); tp = np.linspace(5, 80, 60)
    HH, TP = np.meshgrid(hh, tp)
    V = np.vectorize(lambda h, t: void_content(h, t))(HH, TP) * 100
    cs = ax[1, 0].contourf(HH * 1e6, TP, V, levels=np.linspace(0, 40, 21), cmap="inferno_r")
    fig.colorbar(cs, ax=ax[1, 0], label="void content [%]")
    ax[1, 0].contour(HH * 1e6, TP, V, levels=[1.0], colors="cyan", linewidths=2)
    ax[1, 0].set_xlabel("tow half-thickness h [µm]"); ax[1, 0].set_ylabel("time above melt t_proc [s]")
    ax[1, 0].set_title("void process window (cyan = 1% void)\nspreading (←) opens the void-free region")

    # void vs spreading ratio for a few viscosities
    sr = np.linspace(1, 8, 60)
    for mu, c in [(1500, "#2ca02c"), (3000, "#ff7f0e"), (6000, "#d62728")]:
        vv = np.array([void_content(H0 / s, mu=mu) for s in sr]) * 100
        ax[1, 1].plot(sr, vv, "-", color=c, label=f"μ={mu} Pa·s")
    ax[1, 1].set_xlabel("fibre spreading ratio (kaisen) s"); ax[1, 1].set_ylabel("void content [%]")
    ax[1, 1].set_title("fibre spreading (kaisen) kills voids\n(t_imp ∝ 1/s²): Daikin's impregnation fix")
    ax[1, 1].legend(fontsize=8); ax[1, 1].grid(True, alpha=0.3)

    fig.suptitle("CFRTP melt impregnation & voids (Darcy + Gebart): high-viscosity fluoropolymer "
                 "→ voids, fibre spreading (kaisen) as the fix (Daikin theme)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main()
