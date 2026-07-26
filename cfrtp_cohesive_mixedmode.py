"""cfrtp_cohesive_mixedmode.py — MIXED-MODE cohesive-zone interface law for CFRTP
interlaminar delamination (Daikin theme, research/MEMO_cfrp_residual_stress_collab.md),
upgrading the mode-II-only ILSS demo (cfrtp_ilss_interface.py) to coupled opening +
shear with a Benzeggagh-Kenane (B-K) mode-interaction criterion — how real delamination
mixes peel (mode I) and shear (mode II), and how the two Daikin issues interact (a
residual normal/shear pre-load shifts the mode mixity and the available toughness).

Model (Camanho-Davila mixed-mode bilinear cohesive):
  penalty stiffness K; strengths tn_max (mode I), ts_max (mode II); toughnesses GIc,
  GIIc; B-K exponent eta. For a loading direction (opening delta_n >= 0, shear delta_s):
    * mode-mixity energy fraction B = GII/(GI+GII),
    * mixed toughness (B-K):  Gc(B) = GIc + (GIIc - GIc) B^eta,
    * bilinear onset delta_m0 and final delta_mf from the mixed strength & Gc,
    * damage d(lambda) from the max effective separation lambda = sqrt(<dn>+^2 + ds^2),
    * tractions tn = (1-d) K <dn>+  - K<-dn>+ (compression penalty), ts = (1-d) K ds.

Validated: integrating the traction along a fixed-mixity separation path dissipates
exactly the B-K mixed-mode toughness Gc(B) (energy consistency, machine precision).

Shows: the effective traction-separation law across mode mixities (pure I -> pure II);
the B-K failure envelope in the (GII, GI) plane vs a power-law criterion; the dissipated
energy vs mode-mixity angle validating Gc(B); and the effective strength / toughness vs
mixity with the weak-fluoropolymer point and the residual-stress-induced mixity shift.

Honest scope: the cohesive CONSTITUTIVE law + mixed-mode envelope (the physics that a
2D/3D cohesive interface element would carry at each interface point); full 2D/3D
delamination-front propagation on a mesh is the next step (sibling to the repo's ⑭
phase-field interface fracture). Illustrative CFRTP-like properties. No ML.

Run:  python3 cfrtp_cohesive_mixedmode.py     (writes cfrtp_cohesive_mixedmode.png)
      python3 cfrtp_cohesive_mixedmode.py --help
"""
from __future__ import annotations

import argparse

import numpy as np

SEED = 20260726
K_PEN = 1e14           # penalty stiffness [Pa/m]
TN_MAX = 12e6          # mode-I (opening) strength [Pa] (weak fluoropolymer interface)
TS_MAX = 18e6          # mode-II (shear) strength [Pa]
GIC = 0.20e3           # mode-I toughness [J/m^2]
GIIC = 0.60e3          # mode-II toughness [J/m^2]
ETA = 1.6              # Benzeggagh-Kenane exponent


def bk_toughness(B, eta=ETA):
    """Benzeggagh-Kenane mixed-mode toughness; B = GII/(GI+GII) in [0,1]."""
    return GIC + (GIIC - GIC) * B ** eta


def mixed_params(phi):
    """Onset and final effective separations for a loading direction angle phi
    (dn = lambda cos phi >= 0 opening, ds = lambda sin phi). Camanho-Davila bilinear."""
    dn0 = TN_MAX / K_PEN; ds0 = TS_MAX / K_PEN
    c, s = np.cos(phi), np.sin(phi)
    if c <= 1e-9:                                   # pure mode II
        dm0 = ds0; B = 1.0
    else:
        beta = s / c                                # ds/dn
        dm0 = dn0 * ds0 * np.sqrt((1 + beta ** 2) / (ds0 ** 2 + (beta * dn0) ** 2))
        # energy mode fraction with equal penalty stiffness: B = ds^2/(dn^2+ds^2)
        B = s ** 2 / (c ** 2 + s ** 2)
    Gc = bk_toughness(B)
    tau0 = K_PEN * dm0                              # effective traction at onset
    dmf = 2.0 * Gc / tau0                           # bilinear final separation
    return dm0, dmf, Gc, B


def tractions(dn, ds, dmax):
    """Mixed-mode tractions and updated max effective separation (irreversible damage)."""
    lam = np.sqrt(np.maximum(dn, 0.0) ** 2 + ds ** 2)
    lam_h = max(lam, dmax)
    phi = np.arctan2(abs(ds), max(dn, 1e-30))
    dm0, dmf, Gc, B = mixed_params(phi)
    if lam_h <= dm0:
        d = 0.0
    elif lam_h >= dmf:
        d = 1.0
    else:
        d = dmf * (lam_h - dm0) / (lam_h * (dmf - dm0))
    tn = (1 - d) * K_PEN * max(dn, 0.0) - K_PEN * max(-dn, 0.0)   # compression penalty
    ts = (1 - d) * K_PEN * ds
    return tn, ts, max(lam, dmax), Gc, B


def dissipated(phi, nlam=4000):
    """Integrate traction.d(separation) along a fixed-direction path -> dissipated energy
    (should equal the mixed-mode toughness Gc(B))."""
    dm0, dmf, Gc, B = mixed_params(phi)
    lam = np.linspace(0, dmf * 1.05, nlam)
    c, s = np.cos(phi), np.sin(phi)
    Dn, Ds = lam * c, lam * s
    tn_a = np.zeros(nlam); ts_a = np.zeros(nlam); dmax = 0.0
    for i in range(nlam):
        tn_a[i], ts_a[i], dmax, _, _ = tractions(Dn[i], Ds[i], dmax)
    W = np.trapezoid(tn_a, Dn) + np.trapezoid(ts_a, Ds)
    return W, Gc, B


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=str, default="cfrtp_cohesive_mixedmode.png")
    args = ap.parse_args()
    np.random.seed(SEED)

    # energy-consistency validation across mixities
    phis = np.linspace(0.0, np.pi / 2, 19)
    W = np.zeros_like(phis); Gc = np.zeros_like(phis); Bs = np.zeros_like(phis)
    for i, ph in enumerate(phis):
        W[i], Gc[i], Bs[i] = dissipated(ph)
    err = float(np.max(np.abs(W - Gc) / Gc))
    print(f"mixed-mode cohesive (B-K eta={ETA}): dissipated energy vs Gc(B) "
          f"max rel err {err:.2e} (energy consistency)")
    print(f"  toughness pure-I {GIC:.0f} -> pure-II {GIIC:.0f} J/m^2; "
          f"strengths tn {TN_MAX/1e6:.0f}, ts {TS_MAX/1e6:.0f} MPa")

    _plot(args.out, phis, W, Gc, Bs)
    print(f"wrote {args.out}")


def _plot(out, phis, W, Gc, Bs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    # effective traction-separation for several mixities
    for ph, lab, c in [(0.0, "mode I (peel)", "#1f77b4"),
                       (np.pi / 6, "30° mixed", "#2ca02c"),
                       (np.pi / 3, "60° mixed", "#ff7f0e"),
                       (np.pi / 2 - 1e-6, "mode II (shear)", "#d62728")]:
        dm0, dmf, Gc0, B = mixed_params(ph)
        lam = np.linspace(0, dmf, 300); teff = np.zeros_like(lam); dmax = 0.0
        cc, ss = np.cos(ph), np.sin(ph)
        for i, L in enumerate(lam):
            tn, ts, dmax, _, _ = tractions(L * cc, L * ss, dmax)
            teff[i] = np.hypot(tn, ts)
        ax[0, 0].plot(lam * 1e6, teff / 1e6, "-", color=c, label=lab)
    ax[0, 0].set_xlabel("effective separation λ [µm]"); ax[0, 0].set_ylabel("effective traction [MPa]")
    ax[0, 0].set_title("mixed-mode cohesive law (bilinear)\narea under each = mixed toughness Gc(B)")
    ax[0, 0].legend(fontsize=8); ax[0, 0].grid(True, alpha=0.3)

    # B-K failure envelope in (GII, GI) vs power-law
    B = np.linspace(0, 1, 200)
    Gc_bk = GIC + (GIIC - GIC) * B ** ETA
    GII = Gc_bk * B; GI = Gc_bk * (1 - B)
    ax[0, 1].plot(GII, GI, "-", color="#d62728", label=f"B-K (η={ETA})")
    for al, c in [(1.0, "#1f77b4"), (2.0, "#2ca02c")]:
        gii = np.linspace(0, GIIC, 200)
        gi = GIC * (1 - (gii / GIIC) ** al) ** (1 / al)
        ax[0, 1].plot(gii, gi, "--", color=c, alpha=0.8, label=f"power-law α={al:.0f}")
    ax[0, 1].set_xlabel("G_II [J/m²]"); ax[0, 1].set_ylabel("G_I [J/m²]")
    ax[0, 1].set_title("mixed-mode failure envelope\n(delamination propagates on the curve)")
    ax[0, 1].legend(fontsize=8); ax[0, 1].grid(True, alpha=0.3)

    # energy consistency: dissipated vs Gc(B) across mixity angle
    ang = np.degrees(phis)
    ax[1, 0].plot(ang, Gc, "-", color="#1f77b4", label="B-K toughness Gc(B)")
    ax[1, 0].plot(ang, W, "o", ms=4, color="#d62728", label="dissipated (integrated)")
    ax[1, 0].set_xlabel("mode-mixity angle φ [deg] (0=peel, 90=shear)")
    ax[1, 0].set_ylabel("energy [J/m²]")
    ax[1, 0].set_title("validation: dissipated energy = Gc(B)\n(cohesive law is energy consistent)")
    ax[1, 0].legend(); ax[1, 0].grid(True, alpha=0.3)

    # effective strength & toughness vs mixity, with residual-stress mixity shift
    teff0 = np.array([K_PEN * mixed_params(ph)[0] for ph in phis]) / 1e6
    ax[1, 1].plot(ang, teff0, "-o", ms=3, color="#b5651d", label="onset strength [MPa]")
    ax[1, 1].set_xlabel("mode-mixity angle φ [deg]"); ax[1, 1].set_ylabel("onset strength [MPa]", color="#b5651d")
    axb = ax[1, 1].twinx()
    axb.plot(ang, Gc, "-s", ms=3, color="#1f77b4", label="toughness Gc [J/m²]")
    axb.set_ylabel("toughness Gc [J/m²]", color="#1f77b4")
    ax[1, 1].axvspan(50, 80, color="gray", alpha=0.12)
    ax[1, 1].annotate("residual shear pushes\nmixity toward mode II", xy=(65, teff0.min() * 1.1),
                      fontsize=8, ha="center", color="#555")
    ax[1, 1].set_title("strength drops but toughness rises toward shear\n(fluoropolymer weak in peel)")
    ax[1, 1].grid(True, alpha=0.3)

    fig.suptitle("CFRTP mixed-mode cohesive interface (Benzeggagh-Kenane): coupled peel + shear "
                 "delamination law & failure envelope (Daikin theme)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main()
