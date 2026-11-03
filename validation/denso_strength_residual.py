"""validation/denso_strength_residual.py -- stage 2 (strength), first step: how much of the tensile strength margin of the
GF/PPS plate does the residual stress take, and how does the identified stress-free temperature change that?

Plate: the 3.08 mm plate of the paper (orientation Fig. 8a, read by eye), the laminate chain of
validation/denso_tsuda2026_compare.py (Mori-Tanaka + orientation averaging, reduced stiffness Q_k(z) at 20 C).
Load: uniaxial in-plane tension in MD (x) or TD (y), membrane strain only (grips hold the plate flat):
    eps_load = lambda A^-1 [N, 0, 0]^T  (or [0, N, 0]),  sigma_k = Q_k (eps_res,k + eps_load)                (1)
Residual mechanical strain per layer: eps_res,k = Q_k^-1 sigma_res,k, sigma_res from the free plate cooled from
T0 (180 C as in the paper, or T0_eff = 143 C identified from the measured released strain,
validation/denso_rescue_released.py) to 20 C.
Failure (first layer): maximum principal in-plane mechanical strain of any layer reaches eps_c              (2)
eps_c is set so that the stress-free plate fails at the datasheet tensile strength in MD: 155 MPa
(DURAFIDE 1130T6, ISO 527-2, a sibling grade: the 1130A6 sheet is not public) -> EXAMPLE, not calibrated.
Reported: failure stress (load / thickness) in MD and TD for no residual stress, T0 = 180 C and T0_eff = 143 C,
the change, and where through the thickness the first failure occurs.
Honest scope: linear elastic to failure (the real curves are slightly non-linear, paper Fig. 9); a strain
criterion with one critical strain for all layers and directions (fibre- and matrix-dominated failure are not
distinguished); no directional strength data to check the TD prediction; the datasheet specimen itself
contains residual stress, so eps_c already absorbs part of it. The result is the RELATIVE effect.
Run:  python3 validation/denso_strength_residual.py      (writes results/denso_strength/strength.json)
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import denso_tsuda2026_compare as tc  # noqa: E402

OUT = os.path.join(ROOT, "results", "denso_strength")
SIG_MD_DATASHEET = 155e6        # Pa, DURAFIDE 1130T6 (proxy grade), example
T0_ID = json.load(open(os.path.join(ROOT, "results", "denso_rescue", "results.json")))["R4"]["T0 only"]["theta"][0]


def plate():
    z = np.linspace(-tc.H3 / 2, tc.H3 / 2, 155)
    d = (z + tc.H3 / 2) * 1e3
    ax = np.interp(d, tc.PROF_3MM["d"], tc.PROF_3MM["axx"]); ay = np.interp(d, tc.PROF_3MM["d"], tc.PROF_3MM["ayy"])
    return z, d, *tc.layers(z, ax, ay)


def max_principal(e):
    """in-plane strain (exx, eyy, gxy) -> max principal (engineering shear)."""
    c = 0.5 * (e[..., 0] + e[..., 1]); r = np.sqrt((0.5 * (e[..., 0] - e[..., 1])) ** 2 + (0.5 * e[..., 2]) ** 2)
    return c + r


def failure(Q, A, H, e_res, direction, ec):
    n = np.zeros(3); n[direction] = 1.0
    e1 = np.linalg.solve(A, n)                              # membrane strain per unit resultant [1/(N/m)]
    lo, hi = 0.0, 10.0 * ec / abs(e1[direction])
    for _ in range(80):                                     # bisection on the load resultant
        mid = 0.5 * (lo + hi)
        if np.max(max_principal(e_res + mid * e1[None, :])) >= ec:
            hi = mid
        else:
            lo = mid
    k = int(np.argmax(max_principal(e_res + hi * e1[None, :])))
    return hi / H, k


def main():
    os.makedirs(OUT, exist_ok=True)
    z, d, ax, ay, az = plate()
    Q, _ = tc.layer_props(tc.T_RT, ax, ay, az)
    A, B, D, dz = tc.abd(z, Q)
    H = z[-1] - z[0]
    # eps_c from the stress-free MD strength (2)
    e1 = np.linalg.solve(A, np.array([1.0, 0, 0]))
    ec = float(np.max(max_principal(SIG_MD_DATASHEET * H * e1[None, :].repeat(len(z), 0))))
    cases = {"no residual stress": None, "T0 = 180 C (paper)": 180.0, f"T0_eff = {T0_ID:.0f} C (identified)": T0_ID}
    out = dict(eps_c=ec, sigma_md_datasheet_MPa=SIG_MD_DATASHEET / 1e6, T0_identified=T0_ID, cases={})
    for name, T0 in cases.items():
        if T0 is None:
            e_res = np.zeros((len(z), 3)); sres = np.zeros((len(z), 3))
        else:
            tc.T0 = T0
            sres, _ = tc.residual_stress(z, ax, ay, az, "total")
            e_res = np.linalg.solve(Q, sres[..., None])[..., 0]
        r = {}
        for lab, dr in (("MD", 0), ("TD", 1)):
            s, k = failure(Q, A, H, e_res, dr, ec)
            r[lab] = dict(strength_MPa=s / 1e6, first_failure_depth_mm=float(d[k]))
        r["max_residual_sxx_MPa"] = float(sres[:, 0].max() / 1e6); r["max_residual_syy_MPa"] = float(sres[:, 1].max() / 1e6)
        out["cases"][name] = r
        print(f"{name:30s}: MD {r['MD']['strength_MPa']:6.1f} MPa (first failure at depth {r['MD']['first_failure_depth_mm']:.2f} mm), "
              f"TD {r['TD']['strength_MPa']:6.1f} MPa (depth {r['TD']['first_failure_depth_mm']:.2f} mm); "
              f"max residual tension sxx {r['max_residual_sxx_MPa']:+.1f}, syy {r['max_residual_syy_MPa']:+.1f} MPa")
    base = out["cases"]["no residual stress"]
    for name in list(cases)[1:]:
        c = out["cases"][name]
        c["change_MD"] = c["MD"]["strength_MPa"] / base["MD"]["strength_MPa"] - 1
        c["change_TD"] = c["TD"]["strength_MPa"] / base["TD"]["strength_MPa"] - 1
        print(f"  {name}: strength change MD {c['change_MD'] * 100:+.1f} %, TD {c['change_TD'] * 100:+.1f} %")
    print(f"eps_c (example, from {SIG_MD_DATASHEET / 1e6:.0f} MPa MD, stress-free) = {ec * 100:.2f} %")
    json.dump(out, open(os.path.join(OUT, "strength.json"), "w"), indent=1)
    print("wrote", os.path.join(OUT, "strength.json"))


if __name__ == "__main__":
    main()
