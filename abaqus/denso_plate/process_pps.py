"""abaqus/denso_plate/process_pps.py -- the process analysis of process_reference.py (heat transfer ->
crystallisation-coupled viscoelastic stress, the validated Python port of denso_cryst_umat.f) with the
constants switched from the PEEK-like ILLUSTRATIVE set to GF/PPS values where literature values exist.
The PEEK-like analysis (process_reference.py, Abaqus-checked on fifa) is kept unchanged as the reference.

Switched (source):
  plate 3.08 mm, orientation a_xx, a_yy through the thickness   Tsuda et al. 2026, Fig. 8a (read by eye)
  UD stiffness at 20 C, Mori-Tanaka, GF 18.2 vol%, s = 16.8       Tsuda et al. 2026 Table 1 + text
  CTE: secant over 20-180 C of the instantaneous CTE (UMAT takes  Tsuda et al. 2026 Table 1
       one constant; above 180 C the 180 C value is implied)
  crystallisation: Hoffman-Lauritzen with U*, T_inf, Tm0 of       Lovinger et al. 1985; scale from Nohara
       Lovinger et al.; ONE (K0, Kg) fitted over 150-290 C to the  et al. 2006 (validation/
       two-regime Nakamura rate, n = 3                            pps_crystallization_kinetics.py)
  WLF reference temperature = Tg = 90 C                           Tsuda et al. 2026
  melt 320 C, mould 150 C                                         Lopez & Wilkes 1988; Tsuda et al. 2026
Kept ILLUSTRATIVE (no PPS data in the repository): crystallisation shrinkage B_L, B_T and X_inf, the
Prony series and WLF constants C1, C2, the crystallinity shift BX, film coefficient, thermal diffusivity.
Honest scope: the UMAT has elastic constants independent of temperature (20 C values here) and one
regime of crystallisation kinetics; fibre nucleation is not in the neat-PPS kinetics; no anneal step.
This is NOT a prediction of the paper's annealed plate; it is the as-moulded state with these constants.
Result to read with care (2026-11-03 run): mould 150/150 C and 140/160 C give the same residual state to
3 decimals. With the WLF reference at Tg = 90 C and the illustrative Prony series, the stress relaxes
almost fully while the part is above Tg, so the mould history is forgotten and the residual state forms
in the (symmetric) air cooling below Tg; the sag then comes from the up/down asymmetry of the read
orientation profile. Rough consistency (not a validation): surface sxx -9.1 MPa vs the elastic
180 -> 20 C value -24.4 MPa scaled by (90 - 25)/(180 - 20) = 0.41, i.e. about -10 MPa.
VE sweep (printed; relaxation times x 1e-3..1e3, WLF reference 70..110 C, GINF 1e-3..1e-1): surface sxx
-6.7 .. -12.4 MPa, sag of the 150/150 mould -2.849 .. -2.860 mm, effect of the 140/160 mould asymmetry
<= 0.001 mm. Both mould walls are above Tg, so within the WLF form the conclusion holds over these
ranges; it would change for a mould wall below Tg or a non-WLF (e.g. crystalline-phase) relaxation.
The sag comes from the up/down asymmetry of the orientation profile read from the paper's figure;
orientation check (printed): the symmetrised profile gives zero sag (so the asymmetry is the whole
cause), and random reading errors of +-0.03 on a symmetric profile give sag_y sd 0.20 mm, max 0.46 mm
in 20 draws, against -2.85 mm: the asymmetry of the read profile is far larger than the reading error,
i.e. it is in the paper's figure (whether the plate itself warps so is not reported there).

Run:  python3 abaqus/denso_plate/process_pps.py   (writes abaqus/denso_plate/process_results/pps_summary.json)
"""
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (HERE, ROOT, os.path.join(ROOT, "validation")):
    sys.path.insert(0, p)
import process_reference as pr  # noqa: E402
import denso_tsuda2026_compare as tc  # noqa: E402
import pps_crystallization_kinetics as kin  # noqa: E402

OUT = os.path.join(HERE, "process_results")
H_PPS = 3.08e-3


def ud_constants():
    C, aL, aT = tc.ud(20.0)
    S = np.linalg.inv(C)
    return dict(EL=1 / S[0, 0], ET=1 / S[1, 1], nuLT=-S[0, 1] / S[0, 0], nuTT=-S[1, 2] / S[1, 1], GLT=1 / S[5, 5])


def secant_cte():
    T = np.linspace(20.0, 180.0, 161)
    aL = [tc.ud(t)[1] for t in T]; aT = [tc.ud(t)[2] for t in T]
    return float(np.trapezoid(aL, T) / 160.0), float(np.trapezoid(aT, T) / 160.0)


def fit_hl():
    """One (K0, Kg) for the UMAT form, least squares in ln K over 150-290 C, U*/R, T_inf, Tm0 fixed."""
    c = json.load(open(os.path.join(ROOT, "results", "pps_kinetics", "kinetics.json")))["constants"]
    T = np.linspace(150.0, 290.0, 141)
    lnK = np.log(c["scale_c_per_cm"] * kin.growth(T))
    TK = T + kin.TABS; Tm0K = kin.TM0 + kin.TABS; TinfK = kin.TINF + kin.TABS
    f = 2 * TK / (Tm0K + TK)
    x = -1.0 / (TK * (Tm0K - TK) * f)
    y = lnK + (kin.USTAR / kin.R) / (TK - TinfK)
    Kg, lnK0 = np.polyfit(x, y, 1)
    resid = y - (Kg * x + lnK0)
    return dict(K0=float(math.exp(lnK0)), USTAR_R=kin.USTAR / kin.R, KG=float(Kg), TM0=kin.TM0, TINF=kin.TINF,
                fit_rms_lnK=float(np.sqrt(np.mean(resid ** 2))), fit_max_lnK=float(np.max(np.abs(resid))))


def orientation_pps():
    d = json.load(open(os.path.join(ROOT, "results", "denso_tsuda2026", "compare.json")))["C3_stress_3mm"]
    dep = np.array(d["depth_mm"]); axx = np.array(d["axx"]); ayy = np.array(d["ayy"])
    def fn(zm):                                  # zm in [-1, 1]; +1 = drilled / top face
        depth = (1 - np.asarray(zm)) / 2 * H_PPS * 1e3
        a11 = np.interp(depth, dep, axx); a22 = np.interp(depth, dep, ayy)
        return a11, a22, np.clip(1 - a11 - a22, 0.0, 1.0)
    return fn


def main():
    os.makedirs(OUT, exist_ok=True)
    ud = ud_constants(); aL, aT = secant_cte(); hl = fit_hl()
    CP = replace(pr.C, UD=ud, A_L=aL, A_T=aT, H=H_PPS, T_MELT=320.0)
    pr.C, pr.H = CP, H_PPS
    pr.K_COND = CP.DIFF * pr.RHO * pr.CP
    pr.HL = {k: hl[k] for k in ("K0", "USTAR_R", "KG", "TM0", "TINF")}
    pr.N_AVRAMI = kin.N_AV
    pr.VE = dict(pr.VE, WLF_TREF=90.0)
    print(f"UD at 20 C: EL {ud['EL']/1e9:.2f} GPa, ET {ud['ET']/1e9:.2f} GPa, nuLT {ud['nuLT']:.3f}, "
          f"nuTT {ud['nuTT']:.3f}, GLT {ud['GLT']/1e9:.2f} GPa; secant CTE 20-180 C: aL {aL*1e6:.1f}, aT {aT*1e6:.1f} ppm/C")
    print(f"single-regime HL fit 150-290 C: K0 {hl['K0']:.3e} 1/s, Kg {hl['KG']:.3e} K^2; "
          f"ln K residual rms {hl['fit_rms_lnK']:.2f}, max {hl['fit_max_lnK']:.2f} (factor {math.exp(hl['fit_max_lnK']):.1f})")
    orient = orientation_pps()
    sag = lambda k: float(k * 0.10 ** 2 / 8 * 1e3)            # over 100 mm, mm
    rng = lambda v: float((np.max(v) - np.min(v)) * 1e3)
    res = dict(constants=dict(UD=ud, A_L=aL, A_T=aT, H=H_PPS, T_MELT=320.0, HL=hl, n=kin.N_AV, VE=pr.VE,
                              illustrative=["B_L", "B_T", "X_INF", "Prony", "WLF_C1", "WLF_C2", "BX", "H_FILM", "DIFF"]),
               cases={})
    for case, (Tb, Tt) in dict(mould_150=(150.0, 150.0), asym_140_160=(140.0, 160.0)).items():
        d = {}
        for part in ("total", "thermal", "cryst"):
            P, _ = pr.layer_props(orient, c=CP, part=part)
            t, Tn = pr.heat_1d(Tb, Tt)
            r = pr.plate_history(P, t, Tn, record=True)
            d[part] = dict(range_xx=rng(r["eres_layer"][:, 0]), range_yy=rng(r["eres_layer"][:, 1]),
                           sag_x=sag(r["kappa"][0]), sag_y=sag(r["kappa"][1]),
                           sxx_surface_MPa=float(r["sig_layer"][-1, 0] / 1e6), syy_surface_MPa=float(r["sig_layer"][-1, 1] / 1e6))
            if part == "total":
                a_m = r["steps"]["MOULD"]["alpha"]
                d["alpha_end_mould"] = [float(a_m.min()), float(a_m.max())]
                d["alpha_end"] = [float(r["alpha_layer"].min()), float(r["alpha_layer"].max())]
            x = d[part]
            print(f"{case:13s} {part:7s}: e_res range xx {x['range_xx']:.3f} yy {x['range_yy']:.3f} x1e-3 | "
                  f"surface sxx {x['sxx_surface_MPa']:+.1f} syy {x['syy_surface_MPa']:+.1f} MPa | "
                  f"sag x {x['sag_x']:+.3f} y {x['sag_y']:+.3f} mm")
        sup = abs(d["total"]["sag_y"] - d["thermal"]["sag_y"] - d["cryst"]["sag_y"])
        print(f"{case:13s} relative crystallinity: end of mould {d['alpha_end_mould'][0]:.3f}..{d['alpha_end_mould'][1]:.3f}, "
              f"end {d['alpha_end'][0]:.3f}..{d['alpha_end'][1]:.3f}; [check] sag total - thermal - cryst = {sup:.1e} mm")
        d["superposition_sag_mm"] = sup
        res["cases"][case] = d
    res["ve_sweep"] = ve_sweep(orient, CP, sag)
    res["orientation_check"] = orientation_check(orient, CP, sag)
    json.dump(res, open(os.path.join(OUT, "pps_summary.json"), "w"), indent=1)
    print(f"wrote {OUT}/pps_summary.json")


def orientation_check(orient, CP, sag, u=0.03, n_rand=20, seed=20261103):
    """Where the sag comes from. (i) the read profile made symmetric (mean of profile and its mirror):
    sag must vanish if the up/down asymmetry of the profile is its cause. (ii) random reading errors:
    independent +-u (uniform; u = 0.03 is the reading uncertainty stated for Fig. 8a) on a_xx and a_yy at
    the 28 read depths, of a profile that is symmetric: the sag these alone produce, to compare with the
    sag of the read profile."""
    d = json.load(open(os.path.join(ROOT, "data", "literature", "tsuda2026_jscm52_88.json"), encoding="utf-8"))["fig8_orientation_3mm"]
    dep = np.array(d["depth_mm"]); Hm = H_PPS * 1e3
    def mk(axx_f, ayy_f):
        def fn(zm):
            depth = (1 - np.asarray(zm)) / 2 * Hm
            a11, a22 = axx_f(depth), ayy_f(depth)
            return a11, a22, np.clip(1 - a11 - a22, 0.0, 1.0)
        return fn
    def run(fn):
        P, _ = pr.layer_props(fn, c=CP, part="total")
        t, Tn = pr.heat_1d(150.0, 150.0)
        r = pr.plate_history(P, t, Tn)
        return sag(r["kappa"][0]), sag(r["kappa"][1])
    sym = lambda f: (lambda x: 0.5 * (f(x) + f(Hm - x)))
    fx = lambda x: np.interp(x, dep, d["a_xx"]); fy = lambda x: np.interp(x, dep, d["a_yy"])
    read = run(mk(fx, fy)); symm = run(mk(sym(fx), sym(fy)))
    print(f"[orientation] sag x, y [mm]: read profile {read[0]:+.3f}, {read[1]:+.3f} | symmetrised {symm[0]:+.3f}, {symm[1]:+.3f}")
    rng = np.random.default_rng(seed); sy = []
    for _ in range(n_rand):
        ex = rng.uniform(-u, u, dep.size); ey = rng.uniform(-u, u, dep.size)
        gx = lambda x, ex=ex: sym(fx)(x) + np.interp(x, dep, ex)
        gy = lambda x, ey=ey: sym(fy)(x) + np.interp(x, dep, ey)
        sy.append(run(mk(gx, gy))[1])
    sy = np.array(sy)
    print(f"[orientation] symmetric profile + reading errors +-{u} ({n_rand} draws): sag_y mean {sy.mean():+.3f}, "
          f"sd {sy.std():.3f}, max |.| {np.abs(sy).max():.3f} mm (read profile: {read[1]:+.3f} mm)")
    return dict(read=read, symmetrised=symm, reading_error_u=u, sag_y_draws=sy.tolist(),
                sag_y_sd=float(sy.std()), sag_y_maxabs=float(np.abs(sy).max()))


def ve_sweep(orient, CP, sag):
    """Sensitivity to the ILLUSTRATIVE viscoelastic constants, over general ranges (not PPS data):
    relaxation times x 1e-3 .. 1e3, WLF reference Tg +- 20 K (WLF_TREF 70..110 C), equilibrium fraction
    GINF 1e-3 .. 1e-1. Read: surface stress and sag of the symmetric 150/150 mould, and how much of the
    mould asymmetry (140/160 minus 150/150) survives in the sag."""
    base = dict(pr.VE)
    def run(ve, Tb, Tt):
        pr.VE = ve
        P, _ = pr.layer_props(orient, c=CP, part="total")
        t, Tn = pr.heat_1d(Tb, Tt)
        r = pr.plate_history(P, t, Tn)
        return float(r["sig_layer"][-1, 0] / 1e6), sag(r["kappa"][1])
    rows = []
    variants = [("base", {})]
    variants += [(f"tau x {k:g}", dict(TAU=tuple(t * k for t in base["TAU"]))) for k in (1e-3, 1e-1, 1e1, 1e3)]
    variants += [(f"Tref {tr:.0f} C", dict(WLF_TREF=tr)) for tr in (70.0, 110.0)]
    for gi in (1e-3, 1e-1):
        g = tuple(x / sum(base["G"]) * (1 - gi) for x in base["G"])
        variants.append((f"GINF {gi:g}", dict(GINF=gi, G=g)))
    for name, ch in variants:
        ve = dict(base, **ch)
        s0, k0 = run(ve, 150.0, 150.0); s1, k1 = run(ve, 140.0, 160.0)
        rows.append(dict(variant=name, sxx_surface_MPa=s0, sag_y_sym=k0, sag_y_asym=k1, asym_effect=k1 - k0))
        print(f"[VE sweep] {name:12s}: surface sxx {s0:+6.1f} MPa | sag_y 150/150 {k0:+.3f} mm, "
              f"140/160 {k1:+.3f} mm, mould-asymmetry effect {k1 - k0:+.3f} mm")
    pr.VE = base
    return rows


if __name__ == "__main__":
    main()
