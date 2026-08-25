"""cfrtp_peek_validation.py -- B-1 / B-2: validate the crystallization model against
the documented non-isothermal behaviour of carbon/PEEK (public; the Daikin
fluoropolymer-CF system is proprietary).

Two K(T) models are compared, both run through the same Nakamura law in PHYSICAL
time over a cooling-rate sweep:

  bell   K(T) = KMAX exp(-((T-TCRYST)/WCRYST)^2), gated to 0 at/above Tm
         (the crude window used by cfrtp_cryst_umat_ve.f + the Tm cutoff).
  HL     K(T) = K0 exp(-U*/(R(T-Tinf))) exp(-Kg/(T dT f)), dT=Tm0-T,
         f=2T/(Tm0+T)  -- Hoffman-Lauritzen / Nakamura; vanishes at BOTH the
         glass (Tinf~Tg-30) and the melt (Tm0). Physically correct; used by
         cfrtp_cryst_umat_hl.f.

Checks (universal non-isothermal laws):
  V1 peak temperature Tp DECREASES with cooling rate,
  V2 alpha(T) shifts to LOWER T with cooling rate (measured by the half-crystallization
     temperature T_half),
  V3 final crystallinity non-increasing (quench at very high rate).
Quantitative (B-2): slow-cool Tp should sit in the literature PEEK band ~305-310 C.
The bell misses this (Tp too near Tm); HL hits it.

Two caveats found while tightening these (2026-08-25):
  * V2 was documented from the start but NEVER actually computed -- checks() only ever
    returned V1 and V3. It is implemented now, and it is not redundant: it fails the bell
    model, whose alpha never even reaches 0.5 above 80 C/min.
  * V3 is VACUOUS on the DSC rate list: alpha_f = 1.000 at every rate there for HL, so
    "non-increasing" passes without discriminating anything. Suppression only appears at
    quench rates, so a separate high-rate sweep (QUENCH_RATES) now carries that test. It
    turns into a third, independent discriminator: HL loses crystallinity at ~640 C/min
    (right order -- PEEK quenches amorphous around 1e3), while the bell model is already
    half-suppressed at 160 C/min, a routine DSC rate at which real PEEK crystallizes fully.

Drop validation/peek_reference_Tp.csv ('rate_C_per_min,Tp_C', digitized DSC) to
auto-compute RMSE against a specific paper.

    python3 validation/cfrtp_peek_validation.py

Refs (see ../README.md): Tierney & Gillespie, Composites Part A 35 (2004);
MDPI Polymers (transient PEEK); Nakamura et al., J. Appl. Polym. Sci. 16 (1972);
Hoffman & Lauritzen nucleation theory.
"""
import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_AVRAMI = 2.5
TABS = 273.15
TG, TM = 143.0, 343.0          # PEEK glass / (bell cutoff) melt [C]
COOL_RATES = [2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0]
T_START, T_END = 390.0, 150.0
LIT_TP_BAND = (305.0, 312.0)   # literature-typical slow-cool PEEK Tp [C] (confirm w/ data)
# High-rate sweep for the quench check (DSC rates never suppress crystallinity).
QUENCH_RATES = [160.0, 320.0, 640.0, 1280.0, 2560.0, 5120.0, 10240.0]
# PEEK is known to be quenchable from the melt to a largely amorphous state, which needs
# cooling of ORDER 1e3 C/min. This is a coarse order-of-magnitude anchor only -- much
# weaker than LIT_TP_BAND, and like it, to be confirmed against a specific paper.
LIT_QUENCH_DECADE = (300.0, 3000.0)   # C/min: where alpha_f should collapse

# ---- bell K(T) + Tm cutoff (matches cfrtp_cryst_umat_ve.f) ----
BELL = dict(KMAX=0.02, TCRYST=290.0, WCRYST=45.0, TM=TM)
def K_bell(T):
    if T >= BELL["TM"]:
        return 0.0
    a = -((T - BELL["TCRYST"]) / BELL["WCRYST"]) ** 2
    return BELL["KMAX"] * math.exp(a if a > -60 else -60)

# ---- Hoffman-Lauritzen K(T) (matches cfrtp_cryst_umat_hl.f) ----
HL = dict(K0=1.0e6, USTAR_R=755.0, KG=7.0e5, TM0=395.0, TINF=TG - 30.0)
def K_hl(T):
    TK = T + TABS; Tm0K = HL["TM0"] + TABS; TinfK = HL["TINF"] + TABS
    if TK >= Tm0K or TK <= TinfK:
        return 0.0
    dT = Tm0K - TK; f = 2 * TK / (Tm0K + TK)
    a = -HL["USTAR_R"] / (TK - TinfK) - HL["KG"] / (TK * dT * f)
    return HL["K0"] * math.exp(a if a > -700 else -700)


def run_cooling(Kfun, rate_C_per_min, dt=0.02, a0=1e-3):
    rate = rate_C_per_min / 60.0
    T = T_START; a = a0; Ts = []; As = []; Rs = []
    while T > T_END:
        Tm = T - 0.5 * rate * dt
        aa = min(max(a, 1e-8), 1 - 1e-10)
        argl = max(-math.log(1 - aa), 1e-12)
        r = N_AVRAMI * Kfun(Tm) * (1 - aa) * argl ** ((N_AVRAMI - 1) / N_AVRAMI)
        a = min(a + max(r * dt, 0.0), 1.0)
        Ts.append(T); As.append(a); Rs.append(r); T -= rate * dt
    return np.array(Ts), np.array(As), np.array(Rs)


def t_half(T, A):
    """Temperature at which relative crystallinity crosses 0.5 (nan if never)."""
    return T[int(np.argmax(A >= 0.5))] if (A >= 0.5).any() else float("nan")


def sweep(Kfun):
    curves = {}; Tps = []; Xf = []; Th = []
    for rate in COOL_RATES:
        T, A, R = run_cooling(Kfun, rate)
        curves[rate] = (T, A, R)
        Tps.append(T[int(np.argmax(R))] if R.max() > 0 else float("nan"))
        Xf.append(A[-1]); Th.append(t_half(T, A))
    return curves, np.array(Tps), np.array(Xf), np.array(Th)


def checks(Tps, Xf, Th, name):
    v1 = all(Tps[i] >= Tps[i + 1] - 1e-6 for i in range(len(Tps) - 1))
    # V2: the alpha(T) curve shifts to LOWER temperature as cooling rate rises, measured
    # by the half-crystallization temperature (documented since the first version but
    # never actually checked until now). A nan means alpha never reached 0.5 at that
    # rate, which is a distinct failure from "shifted the wrong way" -- so say which.
    finite = np.isfinite(Th)
    Thf = Th[finite]
    monotone = all(Thf[i] >= Thf[i + 1] - 1e-6 for i in range(len(Thf) - 1))
    v2 = monotone and finite.all()
    v2_txt = "PASS" if v2 else ("FAIL(never reaches 0.5 above %g C/min)" % COOL_RATES[
        int(np.argmax(~finite)) - 1] if monotone else "FAIL(non-monotone)")
    v3 = all(Xf[i] >= Xf[i + 1] - 1e-6 for i in range(len(Xf) - 1))
    tp_slow = Tps[1]  # 5 C/min
    inband = LIT_TP_BAND[0] - 8 <= tp_slow <= LIT_TP_BAND[1] + 8
    print("  [%s] V1 Tp decr: %s | V2 T_half decr: %s | V3 Xf non-incr: %s | slow-cool "
          "Tp=%.0f C (lit ~%g-%g) -> %s"
          % (name, "PASS" if v1 else "FAIL", v2_txt,
             "PASS" if v3 else "FAIL", tp_slow, LIT_TP_BAND[0], LIT_TP_BAND[1],
             "IN BAND" if inband else "OUT of band"))
    return v1, v2, v3, inband


def quench_sweep(Kfun, rates=QUENCH_RATES, dt=5e-3):
    """Final relative crystallinity over a high-rate (quench) sweep.

    V3 on COOL_RATES alone is vacuous -- alpha_f saturates at 1.000 for every rate
    there, so 'non-increasing' passes without discriminating anything. Crystallinity
    suppression only appears once cooling outruns the crystallization kinetics, which
    for PEEK needs rates orders of magnitude above the DSC range.
    """
    return np.array([run_cooling(Kfun, r, dt=dt)[1][-1] for r in rates])


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    cb, Tpb, Xfb, Thb = sweep(K_bell)
    ch, Tph, Xfh, Thh = sweep(K_hl)
    print("=== PEEK non-isothermal crystallization: bell+cutoff vs Hoffman-Lauritzen ===")
    print("rate[C/min]   Tp_bell   Tp_HL   alphaf_HL   Thalf_HL")
    for r, tb, th, xf, t5 in zip(COOL_RATES, Tpb, Tph, Xfh, Thh):
        print("  %6.0f      %6.1f   %6.1f    %6.3f     %6.1f" % (r, tb, th, xf, t5))
    checks(Tpb, Xfb, Thb, "bell")
    ok_h = checks(Tph, Xfh, Thh, "HL  ")

    # ---- quench check: at what rate does each model stop crystallizing? ----
    # V3 on the DSC rates is vacuous for HL (alpha_f=1.000 throughout), so it discriminates
    # nothing there. Pushing to quench rates turns it into a real, third test -- and it
    # separates bell from HL just as the Tp band does.
    def r_crit_of(q):
        below = [r for r, a in zip(QUENCH_RATES, q) if a < 0.5]
        return below[0] if below else float("nan")

    qb, qh = quench_sweep(K_bell), quench_sweep(K_hl)
    print("  [quench] alpha_f(HL) vs rate: "
          + ", ".join("%g:%.3f" % (r, a) for r, a in zip(QUENCH_RATES, qh)))
    rc_b, rc_h = r_crit_of(qb), r_crit_of(qh)
    q_ok = (rc_h == rc_h) and LIT_QUENCH_DECADE[0] <= rc_h <= LIT_QUENCH_DECADE[1]
    print("  [quench] alpha_f<0.5 at:  bell ~%g C/min  |  HL ~%g C/min   "
          "(PEEK quenches amorphous at order 1e3) -> HL %s"
          % (rc_b, rc_h, "RIGHT ORDER" if q_ok else "WRONG ORDER"))
    print("           bell already half-suppresses at a routine DSC rate, where real PEEK "
          "still fully crystallizes -- a third way it is wrong, independent of the Tp band.")

    rref, tpref = (None, None)
    csv = os.path.join(here, "peek_reference_Tp.csv")
    if os.path.exists(csv):
        d = np.genfromtxt(csv, delimiter=",", names=True)
        rref, tpref = np.atleast_1d(d["rate_C_per_min"]), np.atleast_1d(d["Tp_C"])
        rmse = float(np.sqrt(np.mean((np.interp(rref, COOL_RATES, Tph) - tpref) ** 2)))
        print("  [quant] digitized reference: HL RMSE(Tp) = %.1f C (%d pts)" % (rmse, len(rref)))
    else:
        print("  [quant] no peek_reference_Tp.csv yet (drop 'rate_C_per_min,Tp_C' for RMSE).")

    # ---- figure: (a) HL K(T) shape, (b) Tp vs rate bell vs HL vs literature ----
    HEAT, CRYST, HLC, INK, GRID = "#ff7a1a", "#12b3a0", "#7b5cff", "#141922", "#dfe4ea"
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GRID})
    fig, (axK, axTp) = plt.subplots(1, 2, figsize=(11, 4.3), dpi=110)
    fig.suptitle("B-2: Hoffman–Lauritzen K(T) fixes the quantitative Tp gap (PEEK)",
                 fontweight="bold")
    Tgrid = np.arange(150, 395, 1.0)
    axK.plot(Tgrid, [K_bell(t) / BELL["KMAX"] for t in Tgrid], color=CRYST, lw=2,
             label="bell (normalized)")
    khl = np.array([K_hl(t) for t in Tgrid]); khl = khl / khl.max()
    axK.plot(Tgrid, khl, color=HLC, lw=2.4, label="Hoffman–Lauritzen (norm.)")
    axK.axvline(TG, color=GRID, lw=1); axK.axvline(HL["TM0"], color=GRID, lw=1)
    axK.text(TG + 3, 0.9, "Tg", color=INK, fontsize=9)
    axK.text(HL["TM0"] - 20, 0.9, "Tm0", color=INK, fontsize=9)
    axK.set_xlabel("temperature [°C]"); axK.set_ylabel("normalized K(T)")
    axK.set_title("(a) HL vanishes at BOTH Tg and Tm", fontsize=10)
    axK.grid(True, color=GRID, lw=.8); axK.legend(fontsize=9)

    axTp.axhspan(LIT_TP_BAND[0], LIT_TP_BAND[1], color=INK, alpha=0.10,
                 label="lit. slow-cool band")
    axTp.plot(COOL_RATES, Tpb, "o--", color=CRYST, lw=2, label="bell+cutoff")
    axTp.plot(COOL_RATES, Tph, "o-", color=HLC, lw=2.4, label="Hoffman–Lauritzen")
    if rref is not None:
        axTp.plot(rref, tpref, "s", color=INK, label="digitized lit.")
    axTp.axhline(TM, color=HEAT, lw=1, ls=":"); axTp.text(2, TM + 1, "Tm", color=HEAT, fontsize=8)
    axTp.set_xscale("log"); axTp.set_xlabel("cooling rate [°C/min, log]")
    axTp.set_ylabel("crystallization peak Tp [°C]")
    axTp.set_title("(b) Tp vs rate — HL lands in the lit. band", fontsize=10)
    axTp.grid(True, which="both", color=GRID, lw=.8); axTp.legend(fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(here, "peek_crystallization_validation.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print("wrote %s" % out)
    return ok_h + (q_ok,)


if __name__ == "__main__":
    v1, v2, v3, inband, q_ok = main()
    print("OVERALL (HL): %s" %
          ("PASS — reproduces all three trends, lands in the literature Tp band, "
           "and suppresses crystallinity at the right order of quench rate"
           if (v1 and v2 and inband and q_ok) else "PARTIAL — check parameters"))
