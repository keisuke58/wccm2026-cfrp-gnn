"""validation/pps_gf40_angle_check.py -- check of the strength criteria of validation/denso_strength_residual.py against
MEASURED off-axis tensile data of an injection-moulded GF/PPS plate (a public source; not DENSO's grade).

Data (Ricotta, Sorgato, Zappalorto, Procedia Structural Integrity 33 (2021) 695-703, IGF26, Table 2; open access):
PPS + 40 wt% short glass fibre (Solvay), 200 x 200 x 1.8 mm plates (mould 80 C, melt 310 C, packing 850 bar / 21 s),
specimens milled at 0, 45, 90 deg to the flow, ISO 527 based, at least 3 per angle (mean values only):
    angle [deg]   E_x [MPa]   UTS [MPa]   strain at UTS [%]
       0          14780       174.4         1.71
      45           8760        93.4         1.41
      90           7490        64.9         1.00
Weight-average fibre length 0.226 mm, diameter 12 um; skin-core morphology with a core of about 0.15 mm (8.3 %).
Model (no fit to strengths): Vf from 40 wt% (glass 2.54, PPS 1.35 g/cm3) = 0.262, aspect ratio 0.226 / 0.012 = 18.8;
matrix and glass constants as denso_tsuda2026_compare.py (1130A6 paper, 20 C).
Orientation: shell layers a_xx = p, a_yy = 0.95 - p, core (8.3 %) swapped, a_zz = 0.05. p is the ONE fitted number,
set from the measured modulus ratio E_0 / E_90 (stiffness, not strength).
Checks:
  K1 the modulus at 45 deg and the absolute modulus at 0 deg (not fitted);
  K2 criterion A (one critical strain, calibrated at 0 deg as in the DENSO script) -> strength at 45, 90 deg;
     it also predicts the same failure strain at every angle (measured 1.71 / 1.41 / 1.00 %);
  K3 criterion B (pseudo-ply Tsai-Hill, Kelly-Tyson X_L, example Y_T, S as in the DENSO script), absolute at 0, 45, 90;
  K4 criterion B CALIBRATED on these three strengths: interface shear tau (in X_L, eq. 3 of the DENSO script), Y_T, S
     (three numbers from three values: an exact fit, not a check; plausibility of the values only);
     result: FAILED (the 0 deg strength is out of reach of first-pseudo-ply failure; printed cap).
Loading: uniaxial in-plane stress at angle theta, membrane only, N = sigma H (c^2, s^2, c s).
Honest scope: a different grade (40 wt%, Solvay) and plate; mean values only; residual stress of the specimens ignored;
orientation profile assumed (SEM showed skin-core, no tensor); linear elastic to failure (the curves are non-linear).
A failure is reported as a failure.
Run:  python3 validation/pps_gf40_angle_check.py   (writes results/denso_strength/pps_gf40_angle_check.json)
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import denso_strength_residual as sr  # noqa: E402
import denso_tsuda2026_compare as tc  # noqa: E402

MEAS = {0: (14780.0, 174.4, 1.71), 45: (8760.0, 93.4, 1.41), 90: (7490.0, 64.9, 1.00)}
WF, RHO_G, RHO_M = 0.40, 2.54, 1.35
VF = (WF / RHO_G) / (WF / RHO_G + (1 - WF) / RHO_M)
ASP = 0.226 / 0.012
H = 1.8e-3
CORE = 0.083


def plate(p, n=61):
    z = np.linspace(-H / 2, H / 2, n)
    core = np.abs(z) <= CORE * H / 2
    ax = np.where(core, 0.95 - p, p); ay = 0.95 - ax
    return z, *tc.layers(z, ax, ay)


def load_vec(th):
    c, s = np.cos(np.radians(th)), np.sin(np.radians(th))
    return np.array([c * c, s * s, c * s])


def props(p):
    z, ax, ay, az = plate(p)
    Q, _ = tc.layer_props(tc.T_RT, ax, ay, az)
    A, _, _, _ = tc.abd(z, Q)
    return z, ax, ay, Q, A


def modulus(A, th):
    n = load_vec(th); e = np.linalg.solve(A, n)              # strain per unit resultant
    c, s = np.cos(np.radians(th)), np.sin(np.radians(th))
    e_th = c * c * e[0] + s * s * e[1] + c * s * e[2]        # strain along the load
    return 1.0 / (e_th * H)


def strength_A(A, th, ec):
    e1 = np.linalg.solve(A, load_vec(th))
    return ec / float(sr.max_principal(e1)) / H              # linear: max principal strain = lambda * that of e1


def strength_B(A, ax, ay, th, XL):
    Qud = sr.reduced_Q(tc.ud(tc.T_RT)[0])
    W = np.array([sr.weights(x / (x + y)) for x, y in zip(ax, ay)]); act = W > 0.02
    e1 = np.linalg.solve(A, load_vec(th))
    el = sr.rot_strain(np.broadcast_to(e1, (len(ax), 1, 3)), sr.THETA[None, :])
    sg = el @ Qud.T                                           # per unit resultant; scale with lambda
    s1, s2, t = sg[..., 0], sg[..., 1], sg[..., 2]

    def idx(lam):
        a1, a2, a3 = lam * s1, lam * s2, lam * t
        X = np.where(a1 >= 0, XL, 2 * XL); Y = np.where(a2 >= 0, sr.YT, 2.5 * sr.YT)
        return np.where(act, a1 ** 2 / X ** 2 - a1 * a2 / X ** 2 + a2 ** 2 / Y ** 2 + a3 ** 2 / sr.SS ** 2, 0.0).max()
    lo, hi = 0.0, 1e3 * H * 1e6
    for _ in range(80):
        mid = 0.5 * (lo + hi); lo, hi = (lo, mid) if idx(mid) >= 1 else (mid, hi)
    return hi / H


def main():
    tc.VF, tc.ASPECT = float(VF), float(ASP)
    target = MEAS[0][0] / MEAS[90][0]
    lo, hi = 0.48, 0.92
    for _ in range(40):                                       # fit p to E0/E90 (bisection, ratio grows with p)
        p = 0.5 * (lo + hi); _, _, _, _, A = props(p)
        lo, hi = (p, hi) if modulus(A, 0) / modulus(A, 90) < target else (lo, p)
    z, ax, ay, Q, A = props(p)
    E = {th: modulus(A, th) / 1e6 for th in (0, 45, 90)}
    print(f"material: Vf {VF:.3f} (from 40 wt%), aspect ratio {ASP:.1f}; fitted shell orientation a_xx = {p:.3f} (from E0/E90 = {target:.2f})")
    print(f"[K1] modulus [MPa]  model vs measured: 0 deg {E[0]:.0f} vs {MEAS[0][0]:.0f} ({E[0] / MEAS[0][0] - 1:+.0%}), "
          f"45 deg {E[45]:.0f} vs {MEAS[45][0]:.0f} (ratio E45/E0 {E[45] / E[0]:.3f} vs {MEAS[45][0] / MEAS[0][0]:.3f}), 90 deg {E[90]:.0f} vs {MEAS[90][0]:.0f}")
    # criterion A calibrated at 0 deg
    e1 = np.linalg.solve(A, load_vec(0)); ec = MEAS[0][1] * 1e6 * H * float(sr.max_principal(e1))
    SA = {th: strength_A(A, th, ec) / 1e6 for th in (0, 45, 90)}
    XL = VF * sr.TAU * 0.226 / 0.012 + (1 - VF) * sr.SIGM
    SB = {th: strength_B(A, ax, ay, th, XL) / 1e6 for th in (0, 45, 90)}
    print(f"[K2] criterion A (critical strain {ec * 100:.2f} %, calibrated at 0 deg): 45 deg {SA[45]:.0f} vs {MEAS[45][1]:.1f} MPa "
          f"({SA[45] / MEAS[45][1] - 1:+.0%}), 90 deg {SA[90]:.0f} vs {MEAS[90][1]:.1f} MPa ({SA[90] / MEAS[90][1] - 1:+.0%}); "
          f"same failure strain at all angles, measured {MEAS[0][2]} / {MEAS[45][2]} / {MEAS[90][2]} %")
    print(f"[K3] criterion B (X_L {XL / 1e6:.0f} MPa Kelly-Tyson, Y_T {sr.YT / 1e6:.0f}, S {sr.SS / 1e6:.0f} MPa, examples): "
          + ", ".join(f"{th} deg {SB[th]:.0f} vs {MEAS[th][1]:.1f} MPa ({SB[th] / MEAS[th][1] - 1:+.0%})" for th in (0, 45, 90)))
    rB = {th: SB[th] / SB[0] for th in (45, 90)}; rM = {th: MEAS[th][1] / MEAS[0][1] for th in (45, 90)}
    print(f"     ratios to 0 deg: measured 45 {rM[45]:.2f}, 90 {rM[90]:.2f} | A {SA[45] / SA[0]:.2f}, {SA[90] / SA[0]:.2f} | B {rB[45]:.2f}, {rB[90]:.2f}")
    # K4 calibration of B
    from scipy.optimize import minimize
    LF, DF = 0.226, 0.012

    def predB(x):
        tau, yt, ss = np.exp(x); sr.YT, sr.SS = yt, ss
        return {th: strength_B(A, ax, ay, th, VF * tau * LF / DF + (1 - VF) * sr.SIGM) / 1e6 for th in (0, 45, 90)}
    obj = lambda x: sum(np.log(v / MEAS[k][1]) ** 2 for k, v in predB(x).items())  # noqa: E731
    r = minimize(obj, np.log([35e6, 60e6, 45e6]), method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-10, maxiter=600))
    tau, yt, ss = np.exp(r.x); SBc = predB(r.x)
    print(f"[K4] B calibrated (best): Y_T {yt / 1e6:.1f} MPa, tau and S unbounded; "
          + ", ".join(f"{th} deg {SBc[th]:.1f} vs {MEAS[th][1]:.1f}" for th in (0, 45, 90)))
    sr.YT, sr.SS = 1e12, 1e12
    sr.YT = yt
    cap0y = strength_B(A, ax, ay, 0, 1e12) / 1e6
    print(f"     -> FAILED: the fit drives tau and S to unbounded values; with X_L and S unbounded the 0 deg strength is capped at "
          f"{cap0y:.0f} MPa by the transverse pseudo-plies (Y_T = {yt / 1e6:.0f} MPa), below the measured {MEAS[0][1]:.1f}. "
          "First-pseudo-ply failure cannot represent the data with any constants.")
    out = dict(source="Ricotta, Sorgato, Zappalorto, Procedia Struct. Integr. 33 (2021) 695-703, Table 2",
               measured={str(k): dict(E_MPa=v[0], UTS_MPa=v[1], strain_pct=v[2]) for k, v in MEAS.items()},
               Vf=VF, aspect=ASP, p_shell=p, E_model_MPa={str(k): v for k, v in E.items()}, eps_c_A=ec,
               strength_A_MPa={str(k): v for k, v in SA.items()}, XL_MPa=XL / 1e6, strength_B_MPa={str(k): v for k, v in SB.items()},
               B_calibrated_best=dict(YT_MPa=yt / 1e6, strength_MPa={str(k): v for k, v in SBc.items()}),
               B_calibration_failed=True, B_0deg_cap_MPa=cap0y)
    dst = os.path.join(ROOT, "results", "denso_strength", "pps_gf40_angle_check.json")
    json.dump(out, open(dst, "w"), indent=1); print("wrote", dst)


if __name__ == "__main__":
    main()
