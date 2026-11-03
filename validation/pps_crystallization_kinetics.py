"""validation/pps_crystallization_kinetics.py -- crystallisation kinetics of PPS from literature constants,
in the Nakamura / Hoffman-Lauritzen form used by the process model (validation/cfrtp_peek_validation.py
K_hl, abaqus/denso_plate/denso_cryst_umat.f), checked against data NOT used to build it.

Growth rate (Lovinger, Davis & Padden, Polymer 26, 1595, 1985, Table 2, medium M.wt. PPS):
    G(T) = G0 exp[-U*/R(T - Tinf)] exp[-Kg/(T dT f)],  dT = Tm0 - T, f = 2T/(Tm0 + T)          (1)
    U* = 1400 cal/mol, Tinf = Tg - 30 K, Tg = 92 C, Tm0 = 315 C;
    regime III (T < 208 C): Kg = 7.970e5 K^2, G0 = 3.86e4 cm/s; regime II: Kg = 3.841e5 K^2, G0 = 5.43 cm/s.
    NOTE on Kg: Table 2 lists "slope" (3.841, 7.970 x 1e5 K^2) and "Kg" (8.84, 18.35 x 1e5 K^2 = 2.303 x slope).
    With the "Kg" row in the natural-exponential form (1), the regime II and III fits differ by a factor
    ~1e5 at the 208 C transition where the paper says they meet; with the "slope" row they meet to within
    a few per cent (check C3), and the G0 ratio 7.1e3 quoted in the text is then reproduced. The "Kg" row
    therefore belongs to a log10 form of (1); the "slope" row is used here. This is our reading, not stated
    in the paper.
Nakamura rate: K_N(T) = (ln 2)^(1/n) / t_half(T). For athermal nucleation with a temperature-independent
nucleus density, K_N is proportional to G (Avrami K = (4/3) pi N G^3, n = 3), so
    K_N(T) = c G(T),                                                                            (2)
with ONE scale c fitted to ONE isothermal half-time: Nohara et al. (Polimeros 16, 104, 2006), Table 1,
t_half = 4.1 min at 265 C. n = 3 (Lopez & Wilkes, Polymer 29, 106, 1988: n = 2.6-3.1).

Checks against data not used in (1)-(2) (printed):
  C1 temperature of the maximum rate: Lopez, Wilkes & Geibel (Polymer 30, 147, 1989) found the maximum of
     the overall rate K near 180 C (third-order polynomial fit of ln K);
  C2 the other two half-times of Nohara et al.: 12.6 min at 268 C and 26.8 min at 270 C.
  C3 continuity: G of regimes II and III at the 208 C transition (the paper's two fits should meet).
Then the relative crystallinity after constant-rate cooling from 320 C (1, 10, 100 K/min) and the
temperature of the maximum crystallisation rate during cooling, as a reading of the kinetics.

Honest scope: constants are for neat Phillips PPS (Mw ~ 51 000), not DURAFIDE 1130A6 (GF 30 %): glass
fibre nucleates and branching/endgroups change K by more than tenfold (Lopez et al. 1989). The nucleus
density is taken temperature-independent, which Lopez & Wilkes (1988) show it is not (it falls with T).
Nohara's material is a different grade with its own Tm (~285 C); its half-times near Tm are very
sensitive to Tm0. Induction time, secondary crystallisation and the memory effect are ignored.
NOT a material model of 1130A6; a literature-based starting point for it, to be replaced by DSC data.

Run:  python3 validation/pps_crystallization_kinetics.py   (writes results/pps_kinetics/{kinetics.json,kinetics.png})
"""
from __future__ import annotations

import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "pps_kinetics")
TABS = 273.15
R = 1.98720                                   # cal/(mol K)
USTAR, TG, TM0 = 1400.0, 92.0, 315.0
TINF = TG - 30.0
T_TR = 208.0                                  # regime II -> III
KG = {"II": 3.841e5, "III": 7.970e5}          # K^2, the "slope" row of Table 2 (see note below)
G0 = {"II": 5.43, "III": 3.86e4}              # cm/s
N_AV = 3.0
FIT = (265.0, 4.1)                            # Nohara et al. Table 1: (T [C], t_half [min])
HOLD = ((268.0, 12.6), (270.0, 26.8))


def growth(T, regime=None):
    """Spherulite growth rate G [cm/s], eq. (1); regime by T unless given."""
    T = np.asarray(T, float); TK = T + TABS; Tm0K = TM0 + TABS
    ok = (T < TM0) & (T > TINF)
    TKs = np.where(ok, TK, 0.5 * (Tm0K + TINF + TABS))
    dT = Tm0K - TKs; f = 2 * TKs / (Tm0K + TKs)
    out = np.zeros_like(TKs)
    for rg in ("II", "III"):
        sel = (T >= T_TR) if rg == "II" else (T < T_TR)
        if regime is not None:
            sel = np.ones_like(sel) if regime == rg else np.zeros_like(sel)
        g = G0[rg] * np.exp(-USTAR / (R * (TKs - TINF - TABS)) - KG[rg] / (TKs * dT * f))
        out = np.where(sel, g, out)
    return np.where(ok, out, 0.0)


def main():
    os.makedirs(OUT, exist_ok=True)
    # scale c from one half-time, eq. (2)
    kN_fit = math.log(2) ** (1 / N_AV) / (FIT[1] * 60.0)
    c = kN_fit / float(growth(FIT[0]))
    KN = lambda T: c * growth(T)
    th = lambda T: math.log(2) ** (1 / N_AV) / float(KN(T)) / 60.0          # min
    print(f"scale c = {c:.3e} 1/cm (from t_half {FIT[1]} min at {FIT[0]:.0f} C, Nohara et al. 2006)")
    # C3 continuity at the regime transition
    gII, gIII = float(growth(T_TR, "II")), float(growth(T_TR, "III"))
    print(f"[C3] G at {T_TR:.0f} C: regime II fit {gII:.3e}, regime III fit {gIII:.3e} cm/s (ratio {gIII/gII:.2f})")
    # C1 maximum rate
    T = np.linspace(TINF + 1, TM0 - 1, 5000)
    Tmax = float(T[np.argmax(growth(T))])
    print(f"[C1] maximum rate at {Tmax:.0f} C (Lopez, Wilkes & Geibel 1989: ~180 C, not used in the fit)")
    # C2 held-out half-times
    c2 = []
    for Tc, tm in HOLD:
        tp = th(Tc); c2.append(dict(T=Tc, t_half_measured_min=tm, t_half_model_min=tp))
        print(f"[C2] t_half at {Tc:.0f} C: model {tp:.1f} min, measured {tm} min (Nohara et al. 2006, not used)")
    # non-isothermal: constant-rate cooling from 320 C, Nakamura dX/dt = n K (1-X)[-ln(1-X)]^((n-1)/n)
    cool = []
    for rate in (1.0, 10.0, 100.0):
        dt = 0.2 / (rate / 60.0); Tt = 320.0; a = 1e-6; Tpk, rmax = None, 0.0
        while Tt > 25.0:
            k = float(KN(Tt))
            r = N_AV * k * (1 - a) * (-math.log(1 - a)) ** ((N_AV - 1) / N_AV)
            if r > rmax:
                rmax, Tpk = r, Tt
            a = min(a + r * dt, 1 - 1e-12); Tt -= 0.2
        cool.append(dict(rate_K_min=rate, alpha_end=a, T_peak=Tpk))
        print(f"cooling {rate:5.0f} K/min from 320 C: final relative crystallinity {a:.3f}, "
              f"fastest crystallisation at {Tpk:.0f} C")
    res = dict(constants=dict(Ustar_cal_mol=USTAR, Tg=TG, Tinf=TINF, Tm0=TM0, Kg=KG, G0_cm_s=G0, T_transition=T_TR,
                              n=N_AV, scale_c_per_cm=c, fit=FIT),
               C1_T_max=Tmax, C3_ratio=gIII / gII, C2=c2, cooling=cool,
               nakamura_K_for_umat=dict(note="K_N(T) = K0 exp(-Ustar_R/(T-Tinf)) exp(-Kg/(T dT f)) per regime",
                                        Ustar_R_K=USTAR / R, K0_II=c * G0["II"], K0_III=c * G0["III"]))
    json.dump(res, open(os.path.join(OUT, "kinetics.json"), "w"), indent=1)
    _plot(T, KN, th, cool)
    print(f"wrote {OUT}/kinetics.json and kinetics.png")


def _plot(T, KN, th, cool):
    sys.path.insert(0, os.path.join(ROOT, "slides", "figure_sources"))
    try:                                   # repo convention: thesis_style, graceful fallback
        from thesis_style import use
        use(width_frac=1.0, aspect=0.4)
    except Exception:
        pass
    matplotlib.rcParams.update({"text.usetex": False, "font.family": "serif",
                                "font.serif": ["Times New Roman", "Liberation Serif", "STIXGeneral"],
                                "mathtext.fontset": "stix", "font.size": 9, "xtick.direction": "in",
                                "ytick.direction": "in", "legend.frameon": False})
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 3.0))
    ax[0].semilogy(T, KN(T), "k-", lw=1.2)
    ax[0].axvline(T_TR, color="0.8", lw=0.6); ax[0].text(T_TR + 2, 2e-6, "regime II/III", fontsize=7, color="0.4")
    for Tc, tm in (FIT,) + HOLD:
        ax[0].plot(Tc, math.log(2) ** (1 / N_AV) / (tm * 60), "o" if (Tc, tm) == FIT else "s", mfc="white", mec="k", ms=4)
    ax[0].axvline(180, color="0.6", lw=0.8, ls=":"); ax[0].text(150, 2e-6, "max. rate\n(Lopez 1989)", fontsize=7, color="0.4")
    ax[0].set_xlabel("temperature [°C]"); ax[0].set_ylabel(r"Nakamura rate $K_\mathrm{N}$ [1/s]")
    ax[0].set_ylim(1e-6, 1e2); ax[0].set_xlim(60, 320)
    ax[0].text(0.03, 0.95, "(a)", transform=ax[0].transAxes, fontweight="bold", va="top")
    for c in cool:
        ax[1].plot([c["rate_K_min"]], [c["T_peak"]], "ko", ms=4)
    ax[1].plot([c["rate_K_min"] for c in cool], [c["T_peak"] for c in cool], "k-", lw=0.8)
    ax[1].set_xscale("log"); ax[1].set_xlabel("cooling rate [K/min]"); ax[1].set_ylabel("temperature of fastest crystallisation [°C]")
    ax[1].text(0.95, 0.95, "(b)", transform=ax[1].transAxes, fontweight="bold", va="top", ha="right")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "kinetics.png"), dpi=300); plt.close(fig)


if __name__ == "__main__":
    main()
