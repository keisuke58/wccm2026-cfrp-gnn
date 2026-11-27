"""validation/pps_material_fit.py -- an openInjMoldSim material card for a glass-fibre-filled PPS from PUBLIC
manufacturer curves (Toray TORELINA A504X90, PPS + 40 wt% GF), to replace the illustrative "HDPE shifted by +160 K"
card of abaqus/denso_plate/abaqus_results/flow/stage3_strip/case/constant/thermophysicalProperties.poly.

Input: data/literature/pps_gf40_toray_a504x90_curves.json (points read from the published figure images).

Models (the coefficient names are those of the openInjMoldSim dictionary; SI units: Pa, K, m3/kg, Pa s):
  Cross-WLF     eta(T, gdot) = eta0 / (1 + (eta0 gdot / tau*)^(1 - n)),                                         (1)
                eta0(T) = D1 exp(-A1 (T - T*) / (A2~ + T - T*)),  T* = D2 + D3 p,                               (2)
  two-domain modified Tait
                v(T, p) = v0(T) [1 - C ln(1 + p / B(T))] + vt(T, p),  C = 0.0894,                               (3)
                melt  (T > Tt): v0 = b1m + b2m (T - b5),  B = b3m exp(-b4m (T - b5)),  vt = 0,
                solid (T < Tt): v0 = b1s + b2s (T - b5),  B = b3s exp(-b4s (T - b5)),
                                vt = b7 exp(b8 (T - b5) - b9 p),
                Tt(p) = b5 + b6 p.                                                                             (4)

What is fitted and what is ASSUMED:
  F1 Cross-WLF: D2 = 363.15 K (Tg = 90 C, public sources) and A2~ = 51.6 K (the "universal" WLF value) are
     ASSUMED, D3 = 0 (no pressure-dependent viscosity data).  n, tau*, D1, A1 are fitted by least squares on
     ln(eta).  The plotted curves are straight lines in the log-log plot (60 .. 6000 1/s): the Newtonian plateau is
     not visible.  In that limit (1) reduces to eta = eta0^n tau*^(1-n) gdot^(n-1), so only n, A1 and the product
     D1^n tau*^(1-n) are determined; tau* and D1 separately are NOT.  The script shows this with a profile of the
     RMS error over fixed tau*.  Selection rule for the card (a rule, not a measurement): the largest tau* (the
     lowest zero-shear plateau) whose RMS error in ln(eta) stays within 1.5 x the minimum of the profile.
     The data give only an upper bound of tau* (a lower bound of eta0); the viscosity of the card below about
     60 1/s is an extrapolation.
  F2 Tait: all 13 coefficients are fitted to the PVT points (20 .. 160 MPa, 100 .. 350 C), b5 and b6 by an outer
     search (the model jumps at Tt), the others by least squares inside.  Reported: RMS error in specific volume,
     separately for the points away from the transition and for all points.
  F3 cp(T): the digitized curve (cooling method, apparent value: it contains the crystallization peak near
     235 C) is written as a table.  Between about 80 and 205 C the A504X90 curve is hidden under another grade's
     curve; the stored values there are that curve's centre (flagged in the JSON).
  F4 conductivity: constant 0.30 W/(m K) (the public table value of A504X90, through-thickness, 80 C; public range
     of PPS + GF40 0.2 .. 0.4).  Check printed: k = alpha rho cp from the digitized diffusivity, cp and the fitted
     Tait volume -- a consistency check between three public curves, not a measurement of k.
  F5 no-flow temperature: NO public value.  ASSUMED = the peak temperature of the apparent cp of the same grade
     on cooling (digitized, about 235 C).  Alternatives printed: 220 C (melt recrystallization peak of an unfilled
     PPS, Fortron FN-10 design guide) and the fitted Tait transition b5.  The margin "melt 320 C minus no-flow" is
     printed for each.
Self-checks (printed): S1 the fitted models re-evaluated at every digitized point (residual tables);
  S2 eta(310 C, 1000 1/s) against the single public value 260 Pa s of DURAFIDE 1140A6 (ISO 11443; secondary
  source, a different grade): the ratio is reported, nothing is fitted to it; S3 the magnitudes of the Tait
  coefficients against the HDPE card of the existing case (same units).
Outputs (written by default; --no-write to skip), in abaqus/denso_plate/abaqus_results/flow/pps_material/:
  thermophysicalProperties.poly.pps_a504x90, cpTable_pps_a504x90.csv, kappaTable_pps_a504x90.csv.
  The cpTable / kappaTable files of the existing case are not in this repository, so their OpenFOAM layout could
  not be checked; the tables are written as CSV (T in K) and must be put into the layout of the example's files.
Honest scope: the points are READ BY EYE / traced from raster figures of a web page (uncertainty in the JSON), not
original data; the shear rate is APPARENT (capillary L/D 40/1, no Bagley or Rabinowitsch correction); D2, A2~ are
assumed and D3 = 0; tau* and D1 are not identified separately (see F1); three temperatures 20 K apart carry the
whole temperature dependence, so eta0(T) far below 310 C is an extrapolation of an assumed WLF form; the PVT curves
start at 100 C and their measuring mode and cooling rate are not known, so the transition (b5, b6) is that of the
PVT test, not of a moulding (fast cooling crystallizes at a lower temperature); the no-flow temperature is assumed;
the conductivity is one constant and isotropic; the grade (A504X90, 40 % GF, Toray) is NOT the DENSO material
(DURAFIDE 1130A6, 30 % GF, Polyplastics) -- more glass means a higher viscosity and a smaller specific volume.
The card is illustrative and uncalibrated: it replaces a shifted HDPE by a PPS/GF of the right family, nothing more.
Run:  python3 validation/pps_material_fit.py
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy.optimize import least_squares, minimize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "data", "literature", "pps_gf40_toray_a504x90_curves.json")
OUT = os.path.join(ROOT, "abaqus", "denso_plate", "abaqus_results", "flow", "pps_material")
HDPE = os.path.join(ROOT, "abaqus", "denso_plate", "abaqus_results", "flow", "stage3_strip", "case", "constant",
                    "thermophysicalProperties.poly")
K0 = 273.15
D2_ASSUMED = 363.15      # K, Tg = 90 C (public sources) -- ASSUMED as the WLF reference
A2_ASSUMED = 51.6        # K, universal WLF value -- ASSUMED
D3_ASSUMED = 0.0         # K/Pa, no pressure-dependent viscosity data
C_TAIT = 0.0894
T_MELT_C = 320.0         # melt temperature of the strip case (comparison point for the no-flow margin)
K_CONST = 0.30           # W/(m K), public table value, A504X90 through-thickness at 80 C
TNOFLOW_ALT_C = 220.0    # melt recrystallization peak, unfilled PPS, Fortron FN-10 design guide
ETA_1140A6 = 260.0       # Pa s at 310 C, 1000 1/s, DURAFIDE 1140A6, ISO 11443 (secondary source; not fitted)
RMS_FACTOR = 1.5         # selection rule of F1


# ------------------------------------------------------------------ Cross-WLF
def eta0_wlf(T, D1, A1, D2=D2_ASSUMED, A2=A2_ASSUMED):
    dT = T - D2
    return D1 * np.exp(-A1 * dT / (A2 + dT))


def cross_wlf(T, gd, n, tau, D1, A1):
    e0 = eta0_wlf(T, D1, A1)
    return e0 / (1.0 + (e0 * gd / tau) ** (1.0 - n))


def load_viscosity(doc):
    T, g, e = [], [], []
    for key, c in doc["curves"].items():
        if key.startswith("viscosity_"):
            for a, b in zip(c["shear_rate_1_per_s"], c["viscosity_Pa_s"]):
                T.append(c["temperature_C"] + K0), g.append(a), e.append(b)
    return np.array(T), np.array(g, float), np.array(e, float)


def fit_cross(T, g, e, tau_fixed=None):
    """least squares on ln(eta); unknowns n, ln tau*, ln D1, A1 (tau* fixed if given)."""
    le = np.log(e)

    def unpack(x):
        if tau_fixed is None:
            return x[0], np.exp(x[1]), np.exp(x[2]), x[3]
        return x[0], tau_fixed, np.exp(x[1]), x[2]

    def res(x):
        return np.log(cross_wlf(T, g, *unpack(x))) - le

    best = None
    for n0 in (0.3, 0.5):
        for lD in (25.0, 35.0, 45.0):
            for A0 in (20.0, 30.0):
                if tau_fixed is None:
                    x0 = [n0, np.log(1e4), lD, A0]
                    lo, hi = [0.05, np.log(1.0), 5.0, 1.0], [0.95, np.log(1e8), 120.0, 200.0]
                else:
                    x0, lo, hi = [n0, lD, A0], [0.05, 5.0, 1.0], [0.95, 120.0, 200.0]
                r = least_squares(res, x0, bounds=(lo, hi), x_scale="jac", max_nfev=4000)
                if best is None or r.cost < best.cost:
                    best = r
    rms = float(np.sqrt(np.mean(best.fun ** 2)))
    return unpack(best.x), rms, best


# ------------------------------------------------------------------ Tait
TAIT_NAMES = ["b1m", "b2m", "b3m", "b4m", "b1s", "b2s", "b3s", "b4s", "b5", "b6", "b7", "b8", "b9"]


def tait_v(T, p, c):
    """two-domain modified Tait, eq. (3)-(4); T in K, p in Pa, v in m3/kg; c = dict of b-coefficients."""
    T = np.asarray(T, float)
    p = np.asarray(p, float)
    Tb = T - c["b5"]
    melt = T > c["b5"] + c["b6"] * p
    v0 = np.where(melt, c["b1m"] + c["b2m"] * Tb, c["b1s"] + c["b2s"] * Tb)
    B = np.where(melt, c["b3m"] * np.exp(-c["b4m"] * Tb), c["b3s"] * np.exp(-c["b4s"] * Tb))
    vt = np.where(melt, 0.0, c["b7"] * np.exp(np.clip(c["b8"] * Tb - c["b9"] * p, -700.0, 50.0)))
    return v0 * (1.0 - C_TAIT * np.log(1.0 + p / B)) + vt


def load_pvt(doc):
    T, p, v, tr = [], [], [], []
    for key, c in doc["curves"].items():
        if not key.startswith("pvt_"):
            continue
        for a, b in zip(c["temperature_C"], c["specific_volume_cm3_per_g"]):
            T.append(a + K0), p.append(c["pressure_MPa"] * 1e6), v.append(b * 1e-3), tr.append(False)
        it = c["in_transition"]
        for a, b in zip(it["temperature_C"], it["specific_volume_cm3_per_g"]):
            T.append(a + K0), p.append(c["pressure_MPa"] * 1e6), v.append(b * 1e-3), tr.append(True)
    return np.array(T), np.array(p), np.array(v), np.array(tr)


def fit_tait(T, p, v):
    """outer search over (b5, b6) (the model is discontinuous at Tt), inner least squares for the other 11."""
    # scaled inner unknowns: b1m, b2m, b3m, b4m, b1s, b2s, b3s, b4s, b7, b8, b9
    sc = np.array([1e-3, 1e-6, 1e8, 1e-3, 1e-3, 1e-7, 1e8, 1e-3, 1e-5, 1e-1, 1e-8])
    lo = np.array([0.5, 0.01, 0.2, 0.0, 0.5, 0.01, 0.2, 0.0, 0.0, 0.01, 0.0])
    hi = np.array([0.9, 5.0, 20.0, 20.0, 0.9, 50.0, 50.0, 20.0, 10.0, 5.0, 10.0])
    x0 = np.array([0.68, 0.45, 1.5, 4.0, 0.655, 3.0, 3.0, 3.0, 1.0, 1.0, 0.5])
    state = {"x": x0.copy()}

    def cdict(x, b5, b6):
        y = x * sc
        return dict(b1m=y[0], b2m=y[1], b3m=y[2], b4m=y[3], b1s=y[4], b2s=y[5], b3s=y[6], b4s=y[7],
                    b5=b5, b6=b6, b7=y[8], b8=y[9], b9=y[10])

    def inner(b5, b6, start):
        r = least_squares(lambda x: (tait_v(T, p, cdict(x, b5, b6)) - v) * 1e6, start, bounds=(lo, hi),
                          x_scale="jac", max_nfev=600)
        return r

    best = None
    for b5 in np.arange(505.0, 531.0, 2.5):          # coarse grid, K
        for b6 in np.arange(3.0e-7, 6.01e-7, 0.25e-7):
            r = inner(b5, b6, x0)
            if best is None or r.cost < best[0]:
                best = (r.cost, b5, b6, r.x)
    state["x"] = best[3]

    def outer(z):
        r = inner(z[0], z[1] * 1e-7, state["x"])
        return r.cost

    o = minimize(outer, [best[1], best[2] * 1e7], method="Nelder-Mead",
                 options=dict(xatol=0.02, fatol=1e-6, maxiter=200))
    b5, b6 = (o.x[0], o.x[1] * 1e-7) if o.fun < best[0] else (best[1], best[2])
    r = inner(b5, b6, state["x"])
    return cdict(r.x, b5, b6), r


# ------------------------------------------------------------------ tables
def cp_table(doc):
    c = doc["specific_heat"]
    T = np.array(c["temperature_C"], float)
    cp = np.array(c["cp_J_per_kgK"], float)
    o = np.argsort(T)
    return T[o], cp[o], [c["flag"][i] for i in o]


def read_card_numbers(path):
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path, encoding="utf-8", errors="replace"):
        s = line.split("//")[0].strip().rstrip(";").split()
        if len(s) == 2:
            try:
                out[s[0]] = float(s[1])
            except ValueError:
                pass
    return out


CARD = r"""/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  3.0.1                                 |
|   \\  /    A nd           | Web:      www.OpenFOAM.org                      |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      thermophysicalProperties;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
// PPS + 40 wt%% glass fibre, Toray TORELINA A504X90.  ILLUSTRATIVE, FROM PUBLIC CURVES, UNCALIBRATED.
// Written by validation/pps_material_fit.py from data/literature/pps_gf40_toray_a504x90_curves.json
// (points read from figure images; do not edit by hand, re-run the script).
// Sources: viscosity  Fig. 6.13, https://www.plastics.toray/technical/torelina/tec_017.html
//                     (APPARENT shear rate, L/D 40/1, no Bagley / Rabinowitsch correction; 310, 320, 330 C)
//          PVT        Fig. 6.22, https://www.plastics.toray/technical/torelina/tec_018.html (20 .. 160 MPa)
//          cp         Fig. 6.3,  https://www.plastics.toray/technical/torelina/tec_013.html (see cpTable CSV)
// Fit errors: Cross-WLF RMS %(rms_eta_pct).1f %% in eta (%(n_eta)d points);
//             Tait RMS %(rms_v).2e m3/kg away from the transition, %(rms_v_all).2e m3/kg all points.
// ASSUMED (not fitted): D2 = Tg = 90 C, A2 = 51.6 K (universal WLF), D3 = 0, TnoFlow (no public value).
// Tau and D1 are NOT identified separately (no Newtonian plateau in the data); see the script docstring, F1.
// Not the DENSO grade (DURAFIDE 1130A6, 30 %% GF).

thermoType
{
    type            mojHeRhoThermo;//leave alone
    mixture         pureMixture;//leave alone
    //transport     crossWLF;// specify thermal conductivity kappa with the crossWLF constants.
    transport       crossWLFTTC;// TTC = T-abulated T-hermal C-onductivity // see file: kappaTable
    //thermo        hPolynomial; // Use this also for constant Cp.
    thermo          hTabular; //see file: constant/cpTable
    equationOfState polymerPVT;//TaitEquation. Might be renamed in the future.
    specie          specie;//leave alone
    energy          sensibleInternalEnergy;//leave alone
}

mixture
{
    specie//leave alone
    {
        nMoles      1;
        molWeight   18.0;
    }
    equationOfState // two-domain Tait fitted to Toray Fig. 6.22 (m3/kg, K, Pa)
    {
        b1m         %(b1m).6e;// m3/kg
        b2m         %(b2m).6e;// m3/(kg K)
        b3m         %(b3m).6e;// Pa
        b4m         %(b4m).6e;// 1/K
        b1s         %(b1s).6e;// m3/kg
        b2s         %(b2s).6e;// m3/(kg K)
        b3s         %(b3s).6e;// Pa
        b4s         %(b4s).6e;// 1/K
        b5          %(b5).2f;// K  (%(b5C).1f C) transition of the PVT test at p = 0, fitted
        b6          %(b6).4e;// K/Pa
        b7          %(b7).6e;// m3/kg
        b8          %(b8).6e;// 1/K
        b9          %(b9).6e;// 1/Pa
    }
    thermodynamics
    {
        Hf              0;//leave alone
        Sf              0;//leave alone
        //CpCoeffs<8>     (1000.0    0.0 0.0 0.0 0.0 0.0 0.0 0.0);//J/(kgK)
    }
    transport // PPS + GF40, Toray A504X90, Cross-WLF fitted to Fig. 6.13 (apparent shear rate)
    {
        n          %(n).5f;//1   fitted
        Tau        %(tau).5e;//Pa  selection rule, upper bound only (plateau not visible)
        D1         %(D1).5e;//Pa s  tied to Tau, not identified separately
        D2         %(D2).2f;//K  ASSUMED = Tg 90 C
        D3         0.0;//K/Pa  ASSUMED (no pressure data)
        A1         %(A1).4f;//1   fitted (310 .. 330 C only)
        A2         %(A2).1f;//K  ASSUMED (universal WLF value)
        //lambda     0.30;//W/(mK) see kappaTable (constant 0.30, public table value)
        etaMin     5;//Pas  numerical limit of the example case, unchanged
        etaMax     1e6;//Pas  numerical limit of the example case, unchanged
        TnoFlow    %(TnoFlow).2f;//K  ASSUMED = %(TnoFlowC).1f C (cp peak on cooling, same grade); no public value
        deltaTempInterp 10.0;//K  unchanged
    }
}
//etaMax - High values cause instability during solidification, packing. High viscosity must reduce the shear rate and
//then elastic shear stress supports the material. See the constant/solidificationProperties file. If that file is not
//present, no elastic shear stress will be calculated even though the solver reads the related fvSchemes.


// ************************************************************************* //
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-write", action="store_true", help="do not write the card and the tables")
    a = ap.parse_args()
    doc = json.load(open(SRC, encoding="utf-8"))
    print("source:", doc["grade"], "|", doc["status"][:60], "...")

    # ---------------- F1 Cross-WLF
    T, g, e = load_viscosity(doc)
    print("\n== F1 Cross-WLF  (ASSUMED D2 = %.2f K, A2~ = %.1f K, D3 = 0; %d points, %g .. %g 1/s, apparent)"
          % (D2_ASSUMED, A2_ASSUMED, len(e), g.min(), g.max()))
    for Tc in sorted(set(T - K0)):
        m = np.isclose(T - K0, Tc)
        s = np.polyfit(np.log(g[m]), np.log(e[m]), 2)
        print("   %3.0f C: log-log slope %.3f (power-law n = %.3f), curvature d2ln(eta)/dln(gdot)2 = %+.4f"
              % (Tc, np.polyfit(np.log(g[m]), np.log(e[m]), 1)[0], 1 + np.polyfit(np.log(g[m]), np.log(e[m]), 1)[0],
                 2 * s[0]))
    (n, tau, D1, A1), rms_free, r = fit_cross(T, g, e)
    print("   free 4-parameter fit: n = %.4f, tau* = %.4g Pa, D1 = %.4g Pa s, A1 = %.3f; RMS ln(eta) = %.4f (%.1f %%)"
          % (n, tau, D1, A1, rms_free, 100 * (np.exp(rms_free) - 1)))
    try:
        J = r.jac
        sv = np.linalg.svd(J / np.linalg.norm(J, axis=0), compute_uv=False)
        print("   conditioning of the scaled Jacobian: singular values %s -> ratio %.1e"
              % (np.array2string(sv, precision=2), sv[0] / sv[-1]))
    except Exception as ex:  # pragma: no cover
        print("   (conditioning not computed: %s)" % ex)
    print("   profile over fixed tau* (n, D1, A1 re-fitted):")
    print("      tau* [Pa]      n       D1 [Pa s]     A1    eta0(320 C) [Pa s]  RMS ln(eta)")
    taus = 10.0 ** np.arange(2.0, 6.01, 0.25)
    prof = []
    for tf in taus:
        (n_, t_, D_, A_), rm, _ = fit_cross(T, g, e, tau_fixed=tf)
        prof.append((tf, n_, D_, A_, rm))
        if abs(np.log10(tf) % 0.5) < 1e-9:
            print("      %9.3g  %.4f  %12.4g  %6.2f  %12.4g        %.4f"
                  % (tf, n_, D_, A_, eta0_wlf(T_MELT_C + K0, D_, A_), rm))
    prof = np.array(prof)
    rmin = prof[:, 4].min()
    print("   minimum of the profile: RMS %.4f; RMS varies by a factor %.2f between tau* = %.0e and %.0e Pa"
          % (rmin, prof[:, 4].max() / rmin, taus[0], taus[-1]))
    # selection rule: the largest tau* with RMS <= RMS_FACTOR x minimum (bisection on a fine grid)
    fine = 10.0 ** np.arange(2.0, 6.001, 0.05)
    sel = None
    for tf in fine:
        par, rm, _ = fit_cross(T, g, e, tau_fixed=tf)
        if rm <= RMS_FACTOR * rmin:
            sel = (par, rm)
    if sel is None:
        raise SystemExit("F1 FAILED: no tau* satisfies the selection rule")
    (n, tau, D1, A1), rms_eta = sel
    flat = prof[prof[:, 0] <= tau]
    print("   -> tau* and D1 are WEAKLY DETERMINED: for tau* <= %.3g Pa the RMS stays within %.1f x the minimum"
          % (tau, RMS_FACTOR))
    print("      while eta0(320 C) ranges over %.3g .. %.3g Pa s; the Newtonian plateau is not visible in the data."
          % (min(eta0_wlf(T_MELT_C + K0, f[2], f[3]) for f in flat),
             max(eta0_wlf(T_MELT_C + K0, f[2], f[3]) for f in flat)))
    print("   CARD (selection rule: largest tau* with RMS <= %.1f x minimum):" % RMS_FACTOR)
    print("      n = %.5f, Tau = %.5e Pa, D1 = %.5e Pa s, A1 = %.4f, D2 = %.2f K, D3 = 0, A2 = %.1f K"
          % (n, tau, D1, A1, D2_ASSUMED, A2_ASSUMED))
    print("      RMS ln(eta) = %.4f  (= %.1f %% in eta);  eta0(310/320/330 C) = %.4g / %.4g / %.4g Pa s (extrapolated)"
          % (rms_eta, 100 * (np.exp(rms_eta) - 1), *[eta0_wlf(t + K0, D1, A1) for t in (310, 320, 330)]))
    print("   S1 residuals at the digitized points (model / read - 1, %):")
    for Tc in sorted(set(T - K0)):
        m = np.isclose(T - K0, Tc)
        rr = 100 * (cross_wlf(T[m], g[m], n, tau, D1, A1) / e[m] - 1)
        print("      %3.0f C: " % Tc + " ".join("%+5.1f" % x for x in rr))
    e310 = float(cross_wlf(310 + K0, 1000.0, n, tau, D1, A1))
    print("   S2 eta(310 C, 1000 1/s) = %.1f Pa s (card);  DURAFIDE 1140A6 public value %.0f Pa s (ISO 11443,"
          % (e310, ETA_1140A6))
    print("      secondary source, different grade, corrected or apparent not stated):  ratio card / 1140A6 = %.3f"
          "  (reported only, not fitted)" % (e310 / ETA_1140A6))

    # ---------------- F2 Tait
    Tp, pp, vv, tr = load_pvt(doc)
    print("\n== F2 two-domain Tait  (%d points, of which %d in the steep transition; %g .. %g MPa, %g .. %g C)"
          % (len(vv), tr.sum(), pp.min() / 1e6, pp.max() / 1e6, Tp.min() - K0, Tp.max() - K0))
    c, rt = fit_tait(Tp, pp, vv)
    dv = tait_v(Tp, pp, c) - vv
    rms_all = float(np.sqrt(np.mean(dv ** 2)))
    rms_reg = float(np.sqrt(np.mean(dv[~tr] ** 2)))
    for k in TAIT_NAMES:
        print("      %-4s %.6e" % (k, c[k]))
    print("   transition Tt(p) = %.1f C + %.3f K/MPa x p" % (c["b5"] - K0, c["b6"] * 1e6))
    print("   RMS error in v: %.3e m3/kg (%.2f %% of v) away from the transition (%d points); %.3e m3/kg all points"
          % (rms_reg, 100 * rms_reg / vv.mean(), (~tr).sum(), rms_all))
    print("   max |error| away from the transition: %.3e m3/kg;  reading uncertainty about 1e-6 m3/kg"
          % np.abs(dv[~tr]).max())
    print("   S1 residuals per isobar (model - read, 1e-6 m3/kg; * = point in the steep transition):")
    for p0 in sorted(set(pp)):
        m = pp == p0
        print("      %3.0f MPa: " % (p0 / 1e6) + " ".join("%+.1f%s" % (x * 1e6, "*" if t else "")
                                                         for x, t in zip(dv[m], tr[m])))
    for p0 in sorted(set(pp)):
        m = (pp == p0) & ~tr
        print("      %3.0f MPa: RMS %.2e, bias %+.2e m3/kg" % (p0 / 1e6, np.sqrt(np.mean(dv[m] ** 2)), dv[m].mean()))
    h = read_card_numbers(HDPE)
    if h:
        print("   S3 magnitudes against the existing (HDPE-based) card, same units (ratio PPS / HDPE):")
        print("      " + ", ".join("%s %.2f" % (k, c[k] / h[k]) for k in TAIT_NAMES if k in h and h[k]))
        print("      density of the card at 20 C, 0.1 MPa: %.0f kg/m3 (EXTRAPOLATED below the 100 C of the data);"
              % (1 / tait_v(293.15, 1e5, c)))
        print("      at 320 C, 0.1 MPa: %.0f kg/m3" % (1 / tait_v(593.15, 1e5, c)))
    else:
        print("   S3 skipped: the existing card was not found")

    # ---------------- F3 cp, F4 conductivity
    Tc_, cp_, fl_ = cp_table(doc)
    print("\n== F3 cp(T): %d points, %g .. %g C; cp(25 C) ~ %.0f, cp(320 C) ~ %.0f J/(kg K); apparent (cooling method)"
          % (len(Tc_), Tc_.min(), Tc_.max(), np.interp(25, Tc_, cp_), np.interp(320, Tc_, cp_)))
    ipk = int(np.argmax(cp_))
    print("   peak of the apparent cp: %.0f J/(kg K) at %.1f C (crystallization on cooling);"
          " %d of the points are hidden under another curve (flag)" % (cp_[ipk], Tc_[ipk],
                                                                       sum(f.startswith("hidden") for f in fl_)))
    print("== F4 conductivity: constant %.2f W/(m K) (public table value, through-thickness, 80 C) -- a choice" % K_CONST)
    d = doc["thermal_diffusivity"]
    Td, al = np.array(d["temperature_C"], float), np.array(d["alpha_1e-6_m2_per_s"], float) * 1e-6
    print("   check k = alpha rho cp from three public curves (rho from the fitted Tait at 0.1 MPa; below 100 C the")
    print("   Tait volume is extrapolated; near 235 .. 290 C the peaks of cp and alpha make the product meaningless):")
    for tc in (50, 80, 100, 150, 200, 300, 320):
        k = np.interp(tc, Td, al) * np.interp(tc, Tc_, cp_) / tait_v(tc + K0, 1e5, c)
        print("      %3d C: k = %.3f W/(m K)   (ratio to the card %.2f)" % (tc, k, k / K_CONST))

    # ---------------- F5 no-flow
    Tnf_c = float(Tc_[ipk])
    print("\n== F5 no-flow temperature (NO public value; ASSUMPTION)")
    print("   card: TnoFlow = %.1f C = %.2f K  (peak of the apparent cp on cooling, same grade, digitized)"
          % (Tnf_c, Tnf_c + K0))
    print("   margin melt %.0f C - no-flow = %.1f K" % (T_MELT_C, T_MELT_C - Tnf_c))
    print("   alternatives: %.0f C (recrystallization peak, unfilled PPS, Fortron FN-10)  -> margin %.0f K;"
          % (TNOFLOW_ALT_C, T_MELT_C - TNOFLOW_ALT_C))
    print("                 Tait b5 = %.1f C (transition of the PVT test at p = 0)   -> margin %.1f K"
          % (c["b5"] - K0, T_MELT_C - (c["b5"] - K0)))
    print("   (the HDPE-based card had 320 - 270 = 50 K)")

    # ---------------- write
    if a.no_write:
        return
    os.makedirs(OUT, exist_ok=True)
    par = dict(c)
    par.update(n=n, tau=tau, D1=D1, A1=A1, D2=D2_ASSUMED, A2=A2_ASSUMED, TnoFlow=Tnf_c + K0, TnoFlowC=Tnf_c,
               b5C=c["b5"] - K0, rms_eta_pct=100 * (np.exp(rms_eta) - 1), n_eta=len(e), rms_v=rms_reg,
               rms_v_all=rms_all)
    f1 = os.path.join(OUT, "thermophysicalProperties.poly.pps_a504x90")
    open(f1, "w", encoding="ascii").write(CARD % par)
    f2 = os.path.join(OUT, "cpTable_pps_a504x90.csv")
    with open(f2, "w", encoding="ascii") as fh:
        fh.write("# apparent cp (cooling method) of Toray TORELINA A504X90, digitized from Fig. 6.3, tec_013; "
                 "illustrative, uncalibrated\n")
        fh.write("# flag: visible / hidden_under_A575W20 / visible_row_scan / by_eye_zoom (see the JSON). "
                 "No data below 20 C or above 330 C.\n")
        fh.write("T_K,cp_J_per_kgK,T_C,flag\n")
        for t, q, f in zip(Tc_, cp_, fl_):
            fh.write("%.2f,%.0f,%.1f,%s\n" % (t + K0, q, t, f))
    f3 = os.path.join(OUT, "kappaTable_pps_a504x90.csv")
    with open(f3, "w", encoding="ascii") as fh:
        fh.write("# constant conductivity, public table value of A504X90 (through-thickness, 80 C); a choice, "
                 "not a fit; public range of PPS + GF40 0.2 .. 0.4 W/(m K)\n")
        fh.write("T_K,kappa_W_per_mK\n")
        for t in (273.15, 373.15, 473.15, 573.15, 673.15):
            fh.write("%.2f,%.2f\n" % (t, K_CONST))
    print("\nwrote:\n  %s\n  %s\n  %s" % (f1, f2, f3))


if __name__ == "__main__":
    main()
