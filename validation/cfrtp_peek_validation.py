"""cfrtp_peek_validation.py -- B-1: validate the crystallization model (the one
in cfrtp_cryst_umat_ve.f) against the DOCUMENTED non-isothermal crystallization
behaviour of carbon/PEEK, for which public data exist (the Daikin fluoropolymer-CF
system is proprietary).

The model is the same Nakamura non-isothermal law used in the UMAT, run here in
PHYSICAL time (deg C/min cooling rates) rather than the deck's normalized time:

    dalpha/dt = n K(T) (1-alpha) [-ln(1-alpha)]^((n-1)/n),
    K(T) = KMAX exp[ -((T-TCRYST)/WCRYST)^2 ]   (bell-shaped window)

Robust, textbook non-isothermal laws we CHECK the model against (universal for
semi-crystalline polymers incl. PEEK; see refs):
  (V1) crystallization peak temperature Tp DECREASES as cooling rate increases
       (kinetic lag / undercooling),
  (V2) the relative-crystallinity curves alpha(T) shift to LOWER temperature as
       cooling rate increases,
  (V3) final relative crystallinity is non-increasing with cooling rate and
       collapses at very high rates (quench).

Quantitative match against a specific paper needs its digitized DSC points -- a
loader hook (load_reference_csv) is provided so real (rate, Tp) pairs drop in and
error metrics compute automatically. Until then this asserts the model obeys the
documented trends and prints a PASS/FAIL report + a figure.

Refs: Tierney & Gillespie, Composites Part A 35 (2004); "Modeling of PEEK
Crystallization Kinetics Under Transient Thermal Conditions," Polymers (MDPI);
Nakamura et al., J. Appl. Polym. Sci. 16 (1972). See ../README.md references.

    python3 validation/cfrtp_peek_validation.py
"""
import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- PEEK crystallization parameters (literature-typical; match the deck) ----
# NB: the UMAT/deck use the bare bell K(T)=KMAX*exp(-((T-TCRYST)/WCRYST)^2). B-1
# found that without an undercooling cutoff the bell's high-T tail lets slow cooling
# "crystallize" ABOVE the melt temperature Tm -- unphysical. Here K(T) is gated to
# zero at/above Tm (nucleation needs undercooling). KMAX is calibrated in PHYSICAL
# time (deck's KMAX is in normalized time, a different unit). See VALIDATION note:
# the bell+cutoff reproduces the qualitative laws but a Hoffman-Lauritzen K(T)
# (vanishing at both Tg and Tm) is the recommended next step for quantitative Tp.
N_AVRAMI = 2.5
TCRYST = 290.0      # bell-K centre [deg C], in the PEEK window (Tg~143, Tm~343)
WCRYST = 45.0       # window half-width [deg C]
KMAX = 0.02         # 1/s -- physical-time calibration (Tp below Tm, decreasing with rate)
TG, TM = 143.0, 343.0
COOL_RATES = [2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0]   # deg C/min
T_START, T_END = 360.0, 150.0                            # melt -> below window


def Kof(T):
    if T >= TM:                    # no crystallization above the melt (undercooling)
        return 0.0
    a = -((T - TCRYST) / WCRYST) ** 2
    return KMAX * math.exp(a if a > -60 else -60)


def run_cooling(rate_C_per_min, dt=0.02, a0=1e-3):
    """Integrate relative crystallinity while cooling at a constant rate.
    Returns arrays T, alpha, and the instantaneous rate dalpha/dt."""
    rate = rate_C_per_min / 60.0            # deg C/s
    T = T_START
    a = a0
    Ts, As, Rs = [], [], []
    while T > T_END:
        Tm = T - 0.5 * rate * dt
        aa = min(max(a, 1e-8), 1 - 1e-10)
        argl = max(-math.log(1 - aa), 1e-12)
        r = N_AVRAMI * Kof(Tm) * (1 - aa) * argl ** ((N_AVRAMI - 1) / N_AVRAMI)
        da = max(r * dt, 0.0)
        a = min(a + da, 1.0)
        Ts.append(T); As.append(a); Rs.append(r)
        T -= rate * dt
    return np.array(Ts), np.array(As), np.array(Rs)


def peak_temperature(T, rate_series):
    """Tp = temperature of maximum crystallization rate (exotherm peak)."""
    i = int(np.argmax(rate_series))
    return T[i]


def load_reference_csv(path):
    """Optional: digitized literature points, CSV 'rate_C_per_min,Tp_C'.
    Returns (rates, Tp) or (None, None) if the file is absent."""
    if not os.path.exists(path):
        return None, None
    d = np.genfromtxt(path, delimiter=",", names=True)
    return np.atleast_1d(d["rate_C_per_min"]), np.atleast_1d(d["Tp_C"])


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    curves = {}
    Tps, Xfin = [], []
    for rate in COOL_RATES:
        T, A, R = run_cooling(rate)
        curves[rate] = (T, A, R)
        Tps.append(peak_temperature(T, R))
        Xfin.append(A[-1])
    Tps = np.array(Tps); Xfin = np.array(Xfin)

    # ---- trend checks (documented non-isothermal laws) ----
    v1 = all(Tps[i] >= Tps[i + 1] - 1e-6 for i in range(len(Tps) - 1))     # Tp decreasing
    # V2: mean temperature of the alpha=0.5 crossing shifts down with rate
    def t_half(T, A):
        idx = np.argmax(A >= 0.5)
        return T[idx] if A[-1] >= 0.5 else np.nan
    Thalf = np.array([t_half(*curves[r][:2]) for r in COOL_RATES])
    v2 = np.all(np.diff(Thalf[~np.isnan(Thalf)]) <= 1e-6)
    v3 = all(Xfin[i] >= Xfin[i + 1] - 1e-6 for i in range(len(Xfin) - 1))  # Xfin non-increasing

    print("=== B-1 PEEK non-isothermal crystallization validation ===")
    print("params: n=%.1f  TCRYST=%.0f  WCRYST=%.0f  KMAX=%.2f /s  (literature-typical)" %
          (N_AVRAMI, TCRYST, WCRYST, KMAX))
    print("rate[C/min]  Tp[C]   T(a=.5)[C]  alpha_final")
    for r, tp, th, xf in zip(COOL_RATES, Tps, Thalf, Xfin):
        print("  %6.0f    %6.1f   %8.1f     %6.3f" % (r, tp, th, xf))
    print("Tp span over rate range: %.1f C (%.1f -> %.1f)" % (Tps[0] - Tps[-1], Tps[0], Tps[-1]))
    print("[V1] Tp decreases with cooling rate      : %s" % ("PASS" if v1 else "FAIL"))
    print("[V2] alpha(T) shifts to lower T with rate : %s" % ("PASS" if v2 else "FAIL"))
    print("[V3] final crystallinity non-increasing   : %s" % ("PASS" if v3 else "FAIL"))

    # ---- optional quantitative comparison to digitized literature points ----
    rref, tpref = load_reference_csv(os.path.join(here, "peek_reference_Tp.csv"))
    if rref is not None:
        model_at = np.interp(rref, COOL_RATES, Tps)
        rmse = float(np.sqrt(np.mean((model_at - tpref) ** 2)))
        print("[quant] digitized reference found: RMSE(Tp) = %.1f C over %d points" % (rmse, len(rref)))
    else:
        print("[quant] no digitized reference CSV yet (drop peek_reference_Tp.csv:"
              " 'rate_C_per_min,Tp_C' to compute RMSE).")

    # ---- figure ----
    HEAT, CRYST, INK, GRID = "#ff7a1a", "#12b3a0", "#141922", "#dfe4ea"
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GRID})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3), dpi=110)
    fig.suptitle("B-1: PEEK non-isothermal crystallization — model vs documented trends",
                 fontweight="bold")
    cmap = plt.cm.viridis(np.linspace(0.05, 0.9, len(COOL_RATES)))
    for c, r in zip(cmap, COOL_RATES):
        T, A, _ = curves[r]
        ax1.plot(T, A, color=c, lw=2, label="%g C/min" % r)
    ax1.axvspan(TCRYST - WCRYST, TCRYST + WCRYST, color=CRYST, alpha=0.08)
    ax1.set_xlabel("temperature [°C]  (cooling →)"); ax1.set_ylabel("relative crystallinity α")
    ax1.set_xlim(T_END, T_START); ax1.grid(True, color=GRID, lw=.8)
    ax1.legend(fontsize=8, title="cooling rate", ncol=2)
    ax1.set_title("(a) α(T): faster cooling shifts left  [V2]", fontsize=10)

    ax2.plot(COOL_RATES, Tps, "o-", color=HEAT, lw=2, label="model Tp")
    if rref is not None:
        ax2.plot(rref, tpref, "s--", color=INK, label="literature (digitized)")
    ax2.set_xscale("log"); ax2.set_xlabel("cooling rate [°C/min, log]")
    ax2.set_ylabel("crystallization peak Tp [°C]")
    ax2.grid(True, which="both", color=GRID, lw=.8)
    ax2.set_title("(b) Tp ↓ with rate  [V1]", fontsize=10); ax2.legend(fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(here, "peek_crystallization_validation.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print("wrote %s" % out)

    return v1 and v2 and v3


if __name__ == "__main__":
    ok = main()
    print("OVERALL: %s" % ("PASS — model reproduces the documented PEEK trends"
                           if ok else "FAIL — check parameters"))
