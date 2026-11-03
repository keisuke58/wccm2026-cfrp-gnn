"""validation/denso_anneal_relaxation.py -- how much stress relaxation during the cool-down after the
anneal (180 C x 4 h, Tsuda et al. 2026) would be needed to explain the calibrated factor of the
hole-drilling comparison (a = 0.67, T0_eff ~ 127 C; validation/denso_tsuda2026_transfer.py).

Model (deliberately minimal, ILLUSTRATIVE): one material point, fully constrained (total strain 0),
cooled from T_a = 180 C to 20 C at a constant rate r; stress-free at T_a (the anneal is taken to
relax everything, which is what the elastic analysis with T0 = 180 C already assumes). One Maxwell
element with constant modulus E and CTE alpha:

    d sigma / dt = -E alpha dT/dt - sigma / tau(T),                                        (1)
    tau(T) = tau_g * a_T(T),   log10 a_T = -C1 (T - Tg) / (C2 + T - Tg)   (WLF, T >= Tg),     (2)
    tau = infinity for T < Tg (glassy, no relaxation).

The elastic stress is sigma_el = E alpha (T_a - 20). The ratio f = sigma(20 C) / sigma_el is the
factor that multiplies the eigenstrain; it is compared with the calibrated a = 0.67.
Printed: f against tau_g (relaxation time at Tg) for three cooling rates, and the tau_g that gives
f = 0.67.

Inputs and their status:
  Tg = 85 C (Nohara et al., Polimeros 16(2) 2006, PPS; NOT measured for 1130A6);
  C1 = 17.44, C2 = 51.6 K (the "universal" WLF constants, Williams-Landel-Ferry 1955; NOT PPS data);
  cooling rates 1, 5, 20 K/min (ASSUMED; the paper does not state the rate after the anneal).
Partial relaxation (printed): with GENERAL values (Tg 85/90 C, tau_g 10..1000 s, WLF constants +-20 %,
the three rates), the fraction phi of the stiffness that must relax for the calibrated factor,
from f = (1 - phi) + phi f_M. phi is compared with nothing measured: PPS/GF data would be needed.
Honest scope: no PPS relaxation data exist in the repository, so this does NOT predict the factor;
it only says what relaxation time would be needed for relaxation alone to explain it. A single
Maxwell element, constant E and alpha, crystallinity fixed (the anneal is assumed complete). The
4 h hold itself does not enter, because the state at T_a is taken as stress-free.

Run:  python3 validation/denso_anneal_relaxation.py     (writes results/denso_tsuda2026/anneal_relaxation.{json,png})
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "denso_tsuda2026")
TA, TE, TG = 180.0, 20.0, 85.0
C1, C2 = 17.44, 51.6
RATES = (1.0, 5.0, 20.0)                     # K/min, assumed
A_CAL = 0.67                                 # calibrated factor (Abaqus full mesh)


def factor(tau_g, rate_K_min, n=4000, tg=None, c1=None, c2=None):
    """f = sigma(20 C) / sigma_el for eq. (1); exact exponential update per temperature increment."""
    tg = TG if tg is None else tg; c1 = C1 if c1 is None else c1; c2 = C2 if c2 is None else c2
    rate = rate_K_min / 60.0
    T = np.linspace(TA, TE, n + 1)
    s = 0.0                                  # sigma / (E alpha) in K
    for k in range(n):
        dT = T[k] - T[k + 1]; dt = dT / rate; Tm = 0.5 * (T[k] + T[k + 1])
        if Tm < tg:
            s += dT                          # glassy: elastic
            continue
        tau = tau_g * 10 ** (-c1 * (Tm - tg) / (c2 + Tm - tg))
        x = dt / tau
        s = s * np.exp(-x) + dT * (1 - np.exp(-x)) / x if x > 1e-12 else s + dT
    return s / (TA - TE)


def main():
    os.makedirs(OUT, exist_ok=True)
    taus = np.logspace(0, 14, 141)
    res = dict(Tg=TG, C1=C1, C2=C2, rates_K_min=list(RATES), a_calibrated=A_CAL, tau_g_s=taus.tolist(), f={})
    # check: no relaxation -> f = 1; instant relaxation above Tg -> f = (Tg - 20)/(180 - 20)
    f_hi, f_lo = factor(1e30, 5.0), factor(1e-30, 5.0)
    print(f"[check] tau_g -> inf: f = {f_hi:.6f} (exact 1); tau_g -> 0: f = {f_lo:.6f} "
          f"(exact (Tg-20)/(180-20) = {(TG - TE) / (TA - TE):.6f})")
    res["check"] = dict(f_no_relaxation=f_hi, f_instant=f_lo, f_instant_exact=(TG - TE) / (TA - TE))
    for r in RATES:
        f = np.array([factor(t, r) for t in taus]); res["f"][str(r)] = f.tolist()
        i = int(np.argmin(np.abs(f - A_CAL)))
        print(f"rate {r:4.0f} K/min: f = {A_CAL} at tau_g ~ {taus[i]:.1e} s "
              f"(T0_eff = {20 + f[i] * 160:.0f} C); f range {f.min():.3f} .. {f.max():.3f}")
        res.setdefault("tau_g_for_a_cal", {})[str(r)] = float(taus[i])
    # partial relaxation: only a fraction phi of the stress-carrying stiffness relaxes (amorphous phase);
    # crystals and fibres stay elastic. Parallel branches under the same strain: f = (1 - phi) + phi f_M,
    # so the calibrated factor needs phi = (1 - a) / (1 - f_M). Swept over GENERAL (not PPS) values.
    sweep = []
    for tg in (85.0, 90.0):                          # Nohara et al. 2006 / Tsuda et al. 2026
        for tau_g in (1e1, 1e2, 1e3):                # "tau ~ 100 s at Tg", one decade either side
            for k1 in (0.8, 1.0, 1.2):               # +-20 % on the universal WLF constants
                for r in RATES:
                    fm = factor(tau_g, r, tg=tg, c1=C1 * k1, c2=C2 * k1)
                    sweep.append(dict(Tg=tg, tau_g=tau_g, wlf_scale=k1, rate=r, f_M=fm, phi=(1 - A_CAL) / (1 - fm)))
    ph = np.array([d["phi"] for d in sweep]); fm = np.array([d["f_M"] for d in sweep])
    base = [d for d in sweep if d["Tg"] == 90.0 and d["tau_g"] == 1e2 and d["wlf_scale"] == 1.0 and d["rate"] == 5.0][0]
    print(f"[partial relaxation] f_M over the general-value sweep ({len(sweep)} cases): {fm.min():.3f} .. {fm.max():.3f}")
    print(f"     relaxing fraction needed for a = {A_CAL}: phi = {ph.min():.2f} .. {ph.max():.2f} "
          f"(Tg 90 C, tau_g 100 s, universal WLF, 5 K/min: phi = {base['phi']:.2f})")
    res["partial_relaxation"] = dict(sweep=sweep, phi_range=[float(ph.min()), float(ph.max())], base=base)
    json.dump(res, open(os.path.join(OUT, "anneal_relaxation.json"), "w"), indent=1)
    _plot(taus, res)
    print(f"wrote {OUT}/anneal_relaxation.json and .png")


def _plot(taus, res):
    sys.path.insert(0, os.path.join(ROOT, "slides", "figure_sources"))
    try:                                   # repo convention: thesis_style, graceful fallback
        from thesis_style import use
        use(width_frac=0.6, aspect=0.75)
    except Exception:
        pass
    matplotlib.rcParams.update({"text.usetex": False, "font.family": "serif",
                                "font.serif": ["Times New Roman", "Liberation Serif", "STIXGeneral"],
                                "mathtext.fontset": "stix"})
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    for r, ls in zip(RATES, ("-", "--", ":")):
        ax.semilogx(taus, res["f"][str(r)], "k" + ls, lw=1.2, label=f"{r:.0f} K/min (assumed)")
    ax.axhline(A_CAL, color="0.5", lw=0.8); ax.text(2, A_CAL + 0.02, "calibrated a = 0.67", fontsize=7, color="0.4")
    ax.axhline((TG - TE) / (TA - TE), color="0.8", lw=0.6)
    ax.set_xlabel(r"relaxation time at $T_\mathrm{g}$, $\tau_\mathrm{g}$ [s]")
    ax.set_ylabel(r"$f = \sigma / \sigma_\mathrm{el}$ at 20 $^\circ$C")
    ax.set_ylim(0.3, 1.05); ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.set_title("cool-down after the anneal: one Maxwell element, WLF (illustrative)", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "anneal_relaxation.png"), dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
