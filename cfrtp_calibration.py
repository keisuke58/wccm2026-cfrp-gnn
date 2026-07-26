"""cfrtp_calibration.py — CALIBRATION FRAMEWORK for the CFRTP residual-stress model
(Daikin/NEDO, research/MEMO_cfrp_residual_stress_collab.md). The CFRTP demos so far use
illustrative (non-calibrated) parameters; this builds the machinery to fit the model's
uncertain parameters to measurements, so that when real Daikin data arrives it is a
drop-in replacement.

Honest data situation: we do NOT have Daikin's proprietary measurements, so we cannot
truly calibrate here. Instead this is a TWIN EXPERIMENT that proves the calibration
works: pick a ground-truth parameter set, generate synthetic "pseudo-measurements" of
residual stress vs cooling rate (FE + Gaussian noise), then RECOVER the parameters by
minimizing the misfit over the FE forward model. Replacing the synthetic targets with
real residual-stress measurements (hole-drilling / XRD / curvature) turns this into a
real calibration.

Calibrated parameters:
  * TREF   — the thermo-viscoelastic freeze-in temperature (~fluoropolymer Tg),
  * X_INF  — the attainable crystallinity (material / grade).
IDENTIFIABILITY note: residual stress alone under-determines these two (both scale the
stress magnitude -> a misfit ridge), so TWO observables are used — peak residual
sigma_xx AND attained crystallinity, each at a few cooling rates. Crystallinity pins
X_INF independently, the stress-vs-rate shape pins TREF, and the ridge collapses. Misfit
= noise-weighted sum of squared residuals; the forward model is the viscoelastic CFRTP
FE (cfrtp_viscoelastic_residual_stress.py).

Shows: the pseudo-measurements with the initial-guess vs calibrated model; the misfit
surface over (TREF, BETA) with the true and recovered points; the recovered-vs-true
parameters; and the fit residuals.

Honest scope: 2-parameter least-squares over a modest FE grid (coarse mesh), synthetic
targets (a twin experiment), Gaussian noise; a real calibration needs real data and a
proper uncertainty treatment (Bayesian / bootstrap). Physics/FE is the authority.

Run:  python3 cfrtp_calibration.py     (runs a small FE grid, then calibrates)
      python3 cfrtp_calibration.py --help
"""
from __future__ import annotations

import argparse

import numpy as np

import cfrtp_residual_stress_fe as base
import cfrtp_viscoelastic_residual_stress as ve

SEED = 20260726
RATES = np.array([1.0, 4.0, 16.0])       # cooling rates at which stress & crystallinity are "measured"
NOISE_MPA = 6.0                          # residual-stress measurement noise (1 sigma)
NOISE_X = 0.02                           # crystallinity measurement noise (1 sigma)


def forward(tref, x_inf, rate, nx=24, nz=10, nstep=26):
    """FE forward model: (freeze-in TREF, attainable crystallinity X_INF) + cooling rate
    -> (peak residual sigma_xx [MPa], attained crystallinity). Sets module globals."""
    ve.TREF = tref; base.X_INF = x_inf
    _, _, sig, _, _, _, _, Xf, _, _ = ve.solve(nx, nz, nstep, rate=rate, viscoelastic=True)
    return float(np.max(np.abs(sig[:, 0])) / 1e6), float(Xf.max())


def signature(tref, x_inf):
    out = [forward(tref, x_inf, r) for r in RATES]
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])


def misfit(sig_s, x_s, sig_m, x_m):
    return np.sum(((sig_s - sig_m) / NOISE_MPA) ** 2) + np.sum(((x_s - x_m) / NOISE_X) ** 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=str, default="cfrtp_calibration.png")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    # ground truth (unknown to the calibrator) + synthetic pseudo-measurements
    tref_true, xinf_true = 118.0, 0.50
    sig_true, x_true = signature(tref_true, xinf_true)
    sig_m = sig_true + rng.normal(0, NOISE_MPA, size=sig_true.shape)
    x_m = x_true + rng.normal(0, NOISE_X, size=x_true.shape)
    print(f"twin experiment: true TREF={tref_true} C, X_inf={xinf_true}")
    print(f"  pseudo-meas stress (MPa): {np.round(sig_m,1)}; crystallinity: {np.round(x_m,3)}")

    trefs = np.linspace(95, 150, 8)
    xinfs = np.linspace(0.40, 0.60, 8)
    SSE = np.zeros((len(xinfs), len(trefs)))
    print(f"scanning {len(trefs)}x{len(xinfs)} FE grid (two observables) ...")
    for j, tr in enumerate(trefs):
        for i, xi in enumerate(xinfs):
            ss, xs = signature(tr, xi)
            SSE[i, j] = misfit(ss, xs, sig_m, x_m)
    imin, jmin = np.unravel_index(np.argmin(SSE), SSE.shape)
    tref_hat, xinf_hat = trefs[jmin], xinfs[imin]
    print(f"recovered: TREF={tref_hat:.1f} C (true {tref_true}), "
          f"X_inf={xinf_hat:.3f} (true {xinf_true})")

    rate_dense = np.array([0.5, 1, 2, 4, 8, 16, 32])
    sig_cal = np.array([forward(tref_hat, xinf_hat, r)[0] for r in rate_dense])
    sig_ini = np.array([forward(135.0, 0.42, r)[0] for r in rate_dense])   # initial guess
    x_cal = np.array([forward(tref_hat, xinf_hat, r)[1] for r in rate_dense])
    ss_hat, xs_hat = signature(tref_hat, xinf_hat)
    print(f"  stress residuals (MPa): {np.round(ss_hat - sig_m,1)} (noise {NOISE_MPA})")

    _plot(args.out, sig_m, x_m, rate_dense, sig_cal, sig_ini, x_cal, trefs, xinfs, SSE,
          tref_true, xinf_true, tref_hat, xinf_hat)
    print(f"wrote {args.out}")


def _plot(out, sig_m, x_m, rate_dense, sig_cal, sig_ini, x_cal, trefs, xinfs, SSE,
          tref_true, xinf_true, tref_hat, xinf_hat):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    ax[0, 0].errorbar(RATES, sig_m, yerr=NOISE_MPA, fmt="o", ms=6, color="k",
                      capsize=3, label="pseudo-measurements", zorder=5)
    ax[0, 0].plot(rate_dense, sig_ini, "--", color="#999999", label="initial guess")
    ax[0, 0].plot(rate_dense, sig_cal, "-", color="#d62728", label="calibrated model")
    ax[0, 0].set_xscale("log")
    ax[0, 0].set_xlabel("cooling rate [°C/s]"); ax[0, 0].set_ylabel("peak residual σxx [MPa]")
    ax[0, 0].set_title("observable 1: residual stress vs cooling rate")
    ax[0, 0].legend(); ax[0, 0].grid(True, which="both", alpha=0.3)

    ax[0, 1].errorbar(RATES, x_m, yerr=NOISE_X, fmt="o", ms=6, color="k",
                      capsize=3, label="pseudo-measurements", zorder=5)
    ax[0, 1].plot(rate_dense, x_cal, "-", color="#1f77b4", label="calibrated model")
    ax[0, 1].set_xscale("log")
    ax[0, 1].set_xlabel("cooling rate [°C/s]"); ax[0, 1].set_ylabel("attained crystallinity")
    ax[0, 1].set_title("observable 2: crystallinity (pins X_inf, breaks the ridge)")
    ax[0, 1].legend(); ax[0, 1].grid(True, which="both", alpha=0.3)

    cs = ax[1, 0].contourf(trefs, xinfs, np.log10(SSE + 1e-9), levels=18, cmap="magma")
    fig.colorbar(cs, ax=ax[1, 0], label="log10 misfit")
    ax[1, 0].plot(tref_true, xinf_true, "*", ms=16, color="#2ca02c", mec="k", label="true")
    ax[1, 0].plot(tref_hat, xinf_hat, "X", ms=11, color="#00d0ff", mec="k", label="recovered")
    ax[1, 0].set_xlabel("freeze-in TREF [°C]"); ax[1, 0].set_ylabel("attainable crystallinity X_inf")
    ax[1, 0].set_title("joint misfit surface: single minimum (identifiable)")
    ax[1, 0].legend()

    names = ["TREF [°C]", "X_inf×100"]
    true_v = [tref_true, xinf_true * 100]; rec_v = [tref_hat, xinf_hat * 100]
    xpos = np.arange(2)
    ax[1, 1].bar(xpos - 0.2, true_v, width=0.4, color="#2ca02c", label="true")
    ax[1, 1].bar(xpos + 0.2, rec_v, width=0.4, color="#00a0d0", label="recovered")
    ax[1, 1].set_xticks(xpos); ax[1, 1].set_xticklabels(names)
    ax[1, 1].set_title("recovered vs true parameters (twin experiment)")
    ax[1, 1].legend(); ax[1, 1].grid(True, axis="y", alpha=0.3)

    fig.suptitle("CFRTP residual-stress model CALIBRATION harness (twin experiment): two "
                 "observables identify the parameters — swap in real Daikin data (Daikin theme)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main()
