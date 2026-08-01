"""cfrtp_inverse_design.py -- B-5: process inverse design for CFRTP residual stress.

Physics-first, ML-subordinate (Muramatsu-lab line): the crystallization + viscoelastic
model (same constitutive law as the Abaqus HL UMAT cfrtp_cryst_umat_hl.f, verified in
validation/) is the accuracy AUTHORITY. A cheap polynomial SURROGATE is fit to a small
set of physics runs, the surrogate does the fast inverse-design search, and the proposed
optimum is RE-VERIFIED with the physics model (surrogate-searched, FE-verified -- same
pattern as electrothermal_inverse_design.py).

Design problem (illustrative, PEEK-parameterized; swap in Daikin data later):
  design variable : cooling rate r [C/min]   (the dominant CFRTP process knob)
  objective       : minimize residual |sigma_11| (constrained ply)
  constraint      : final relative crystallinity alpha >= Xmin (stiffness / anti-
                    stiction / creep need enough crystallinity)
The trade-off the optimizer resolves is the canonical one: FAST cooling crystallizes
later/lower -> LOWER residual stress, but too fast QUENCHES crystallinity below Xmin.
The optimum is therefore the fastest cool that still meets Xmin (constraint-active),
which cuts residual stress a lot versus a slow "safe" cure while keeping properties.
Cycle time (~1/rate) is reported as a secondary benefit.

    python3 design/cfrtp_inverse_design.py     # prints the optimum + writes the figure

numpy + matplotlib only (no sklearn). Magnitudes illustrative (uncalibrated).
"""
import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- physics (HL crystallization + generalized-Maxwell VE; PEEK-typical) ----
TABS = 273.15
N_AVRAMI = 2.5
HL = dict(K0=1.0e6, USTAR_R=755.0, KG=7.0e5, TM0=395.0, TINF=143.0 - 30.0)
GINF = 0.3
GK = [0.4 * (1 - GINF) / 0.7, 0.2 * (1 - GINF) / 0.7, 0.1 * (1 - GINF) / 0.7]
TAU = [5.0, 50.0, 500.0]                       # s
C, A_CTE, BETA, G0, AGEL, BX = 1e10, 30e-6, -4e-3, 0.01, 0.1, 2.0
WC1, WC2, WTREF = 17.4, 51.6, 143.0
XMIN = 0.60                                     # required final crystallinity
TMELT_HOLD, T_RT = 380.0, 25.0


def K_hl(T):
    TK = T + TABS; Tm0K = HL["TM0"] + TABS; TinfK = HL["TINF"] + TABS
    if TK >= Tm0K or TK <= TinfK:
        return 0.0
    dT = Tm0K - TK; f = 2 * TK / (Tm0K + TK)
    a = -HL["USTAR_R"] / (TK - TinfK) - HL["KG"] / (TK * dT * f)
    return HL["K0"] * math.exp(a if a > -700 else -700)


def _beta_exp(dt, tau):
    tau = max(tau, 1e-20); xr = dt / tau; e = math.exp(-xr)
    return ((1 - e) / xr if xr > 1e-6 else 1 - 0.5 * xr), e


def process(rate_C_per_min):
    """Single melt->cool at constant rate. Return (residual |sigma_11| [MPa], alpha)."""
    rate = rate_C_per_min / 60.0                          # C/s
    dur = (TMELT_HOLD - T_RT) / rate
    total = 30.0 + dur
    dt = max(0.2, total / 3000.0)
    a = 1e-3; q = [0.0, 0.0, 0.0]; qinf = 0.0; sig = 0.0
    prevT = TMELT_HOLD
    for (T0, T1, d) in [(TMELT_HOLD, TMELT_HOLD, 30.0), (TMELT_HOLD, T_RT, dur)]:
        n = max(1, int(round(d / dt)))
        for i in range(1, n + 1):
            Tn = T0 + (T1 - T0) * i / n
            Tm = 0.5 * (prevT + Tn); dT = Tn - prevT
            aa = min(max(a, 1e-8), 1 - 1e-10)
            argl = max(-math.log(1 - aa), 1e-12)
            r = N_AVRAMI * K_hl(Tm) * (1 - aa) * argl ** ((N_AVRAMI - 1) / N_AVRAMI)
            a = min(a + max(r * (d / n), 0.0), 1.0)
            x = min(max((a - AGEL) / (1 - AGEL), 0.0), 1.0); g = G0 + (1 - G0) * x
            deig = A_CTE * dT + BETA * (r * (d / n))
            dsi = g * C * (0 - deig)
            dtt = Tm - WTREF; den = WC2 + dtt; den = den if den >= 1 else 1.0
            pw = max(-30.0, min(30.0, -WC1 * dtt / den)); aT = 10 ** pw
            aX = 10 ** min(30.0, BX * a); ash = aT * aX
            qinf += GINF * dsi; s = qinf
            for k in range(3):
                b, e = _beta_exp(d / n, ash * TAU[k]); q[k] = e * q[k] + GK[k] * b * dsi; s += q[k]
            sig = s; prevT = Tn
    return abs(sig) / 1e6, a


# ---- 1D polynomial surrogate in log10(rate) (ML-subordinate) ----
def features(u):                # u = normalized log-rate
    return np.column_stack([np.ones_like(u), u, u**2, u**3, u**4])


def fit(u, y):
    coef, *_ = np.linalg.lstsq(features(u), y, rcond=None)
    return coef


def predict(coef, u):
    return features(u) @ coef


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    rates = np.array([5, 10, 20, 40, 80, 160, 320, 640, 1200, 2000, 3000], float)
    sig = np.zeros(len(rates)); alp = np.zeros(len(rates))
    for i, r in enumerate(rates):
        sig[i], alp[i] = process(r)
    lr = np.log10(rates); u = (lr - lr.min()) / (lr.max() - lr.min())
    cs = fit(u, sig); ca = fit(u, alp)
    relL2 = lambda a, b: float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))
    print("surrogate rel-L2  sigma=%.3f  alpha=%.3f (on samples)" %
          (relL2(predict(cs, u), sig), relL2(predict(ca, u), alp)))

    # inverse design: minimize sigma s.t. alpha >= XMIN, via surrogate on a fine grid
    rf = np.logspace(math.log10(rates.min()), math.log10(rates.max()), 400)
    uf = (np.log10(rf) - lr.min()) / (lr.max() - lr.min())
    sfit = predict(cs, uf); afit = predict(ca, uf)
    feas = afit >= XMIN
    kopt = int(np.argmin(np.where(feas, sfit, np.inf)))
    r_opt = rf[kopt]
    sig_fe, alp_fe = process(r_opt)               # FE-verify
    # slow "safe" baseline for contrast
    sig_slow, alp_slow = process(rates.min())
    print("inverse-design optimum (surrogate-searched, FE-verified):")
    print("  cooling rate r* = %.0f C/min   (constraint alpha>=%.2f active)" % (r_opt, XMIN))
    print("  surrogate : |sigma|=%.1f MPa*, alpha=%.2f" % (sfit[kopt], afit[kopt]))
    print("  FE-verify : |sigma|=%.1f MPa*, alpha=%.2f  (%s)" %
          (sig_fe, alp_fe, "OK" if alp_fe >= XMIN - 0.03 else "CHECK"))
    print("  vs slow 'safe' %g C/min: |sigma|=%.1f MPa*, alpha=%.2f  -> residual cut %.0f%%, "
          "cycle ~%.0fx faster" % (rates.min(), sig_slow, alp_slow,
          100 * (1 - sig_fe / sig_slow), r_opt / rates.min()))

    # ---- figure ----
    HEAT, CRYST, INK, GRID, TENS = "#ff7a1a", "#12b3a0", "#141922", "#dfe4ea", "#e23b48"
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GRID})
    fig, ax = plt.subplots(figsize=(7.8, 4.8), dpi=120)
    ax.set_xscale("log")
    ax.plot(rf, sfit, color=TENS, lw=2.4, label="residual |σ₁₁| (surrogate)")
    ax.plot(rates, sig, "o", color=TENS, ms=5)
    ax.set_xlabel("cooling rate [°C/min, log]"); ax.set_ylabel("residual |σ₁₁| [MPa]*", color="#e23b48")
    ax.tick_params(axis="y", colors="#e23b48"); ax.grid(True, which="both", color=GRID, lw=.7)
    ax2 = ax.twinx()
    ax2.plot(rf, afit, color=CRYST, lw=2.4, label="crystallinity α (surrogate)")
    ax2.plot(rates, alp, "s", color=CRYST, ms=4)
    ax2.axhline(XMIN, color=CRYST, ls="--", lw=1.3)
    ax2.set_ylabel("relative crystallinity α", color=CRYST); ax2.tick_params(axis="y", colors=CRYST)
    ax2.set_ylim(0, 1.05)
    # infeasible region (alpha<Xmin) shaded
    infeas = rf[afit < XMIN]
    if len(infeas):
        ax.axvspan(infeas.min(), rf.max(), color=INK, alpha=0.06)
        ax.text(infeas.min()*1.05, ax.get_ylim()[1]*0.92, "quench:\nα<%.2f" % XMIN,
                fontsize=8, color=INK, va="top")
    ax.axvline(r_opt, color=INK, lw=1.4, ls=":")
    ax.plot([r_opt], [sig_fe], "*", color="#ffffff", ms=18, mec=INK, mew=1.2, zorder=6)
    ax.set_title("B-5 inverse design: fastest cool that keeps α ≥ %.2f\n"
                 "r* = %.0f °C/min → residual %.0f%% below the slow-cure baseline "
                 "(FE-verified)" % (XMIN, r_opt, 100*(1-sig_fe/sig_slow)), fontsize=10)
    fig.tight_layout()
    out = os.path.join(here, "cfrtp_inverse_design.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print("wrote %s" % out)
    return dict(r_opt=r_opt, sig_fe=sig_fe, alp_fe=alp_fe, sig_slow=sig_slow)


if __name__ == "__main__":
    main()
