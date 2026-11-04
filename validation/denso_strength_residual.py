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
Criterion B (fibre / matrix split, laminate analogy, Advani & Tucker style): each layer is split into pseudo-plies of
aligned fibre at angles theta with weights w(theta) ~ exp(kappa cos 2 theta), kappa chosen so that the weighted
<cos^2 theta> equals the layer's a_xx / (a_xx + a_yy). Same strain in all pseudo-plies of a layer (iso-strain).
Each pseudo-ply: UD stress from the aligned Mori-Tanaka stiffness, Tsai-Hill index with
    X_L (Kelly-Tyson, sub-critical fibres): eta_0 V_f tau l / d + (1 - V_f) sigma_m'                         (3)
    Y_T, S (matrix-dominated): general values for GF thermoplastics.
The first pseudo-ply with weight above 2 % of the layer's maximum reaching index 1 fails the plate (first-ply).
Constants (ALL EXAMPLE): tau = 35 MPa, eta_0 = 1 (fibres in the ply direction), sigma_m' = 60 MPa, Y_T = 60 MPa,
S = 45 MPa, fibre length 170.6 um, diameter 10.1 um (paper). X_L from (3) is printed and compared with the
datasheet 155 MPa (not fitted).
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


from fiber_orientation_cte import VOIGT  # noqa: E402  (orientation module, 11,22,33,23,13,12)
sys.path.insert(0, os.path.join(ROOT, "viz"))
from denso_plate_features import reduced_Q  # noqa: E402

TAU, ETA0, SIGM, YT, SS = 35e6, 1.0, 60e6, 60e6, 45e6          # example constants (criterion B)
VF, LF, DF = tc.VF, tc._m["fibre_length_mean_um"], tc._m["fibre_diameter_mean_um"]
XL = ETA0 * VF * TAU * LF / DF + (1 - VF) * SIGM                 # (3)
THETA = np.radians(np.linspace(-90, 90, 72, endpoint=False))


def rot_strain(e, th):
    """global in-plane (exx, eyy, gxy) -> ply axes (e11, e22, g12)."""
    c, s_ = np.cos(th), np.sin(th)
    return np.stack([c * c * e[..., 0] + s_ * s_ * e[..., 1] + c * s_ * e[..., 2],
                     s_ * s_ * e[..., 0] + c * c * e[..., 1] - c * s_ * e[..., 2],
                     -2 * c * s_ * e[..., 0] + 2 * c * s_ * e[..., 1] + (c * c - s_ * s_) * e[..., 2]], -1)


def weights(frac):
    """kappa with <cos^2> = frac for w ~ exp(kappa cos 2 theta) on THETA."""
    lo, hi = -40.0, 40.0
    for _ in range(80):
        k = 0.5 * (lo + hi); w = np.exp(k * np.cos(2 * THETA) - abs(k)); m = np.sum(w * np.cos(THETA) ** 2) / w.sum()
        lo, hi = (k, hi) if m < frac else (lo, k)
    return w / w.max()


def tsai_hill_failure(A, H, e_res, ax, ay, direction):
    Qud = reduced_Q(tc.ud(tc.T_RT)[0])
    W = np.array([weights(x / (x + y)) for x, y in zip(ax, ay)])          # (layers, angles)
    act = W > 0.02
    n = np.zeros(3); n[direction] = 1.0
    e1 = np.linalg.solve(A, n)

    def index(lam):
        e = e_res + lam * e1[None, :]                                   # (layers, 3)
        el = rot_strain(e[:, None, :], THETA[None, :])                   # (layers, angles, 3)
        sg = el @ Qud.T
        s1, s2, t = sg[..., 0], sg[..., 1], sg[..., 2]
        X = np.where(s1 >= 0, XL, 2 * XL); Y = np.where(s2 >= 0, YT, 2.5 * YT)       # compression stronger (example)
        f = s1 ** 2 / X ** 2 - s1 * s2 / X ** 2 + s2 ** 2 / Y ** 2 + t ** 2 / SS ** 2
        return np.where(act, f, 0.0)
    lo, hi = 0.0, 2e3 * H * 1e6
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if index(mid).max() >= 1 else (mid, hi)
    k = np.unravel_index(np.argmax(index(hi)), index(hi).shape)
    return hi / H, int(k[0]), float(np.degrees(THETA[k[1]]))


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
            sb, kb, thb = tsai_hill_failure(A, H, e_res, ax, ay, dr)
            r[lab] = dict(strength_MPa=s / 1e6, first_failure_depth_mm=float(d[k]),
                          B_strength_MPa=sb / 1e6, B_depth_mm=float(d[kb]), B_ply_angle_deg=thb)
        r["max_residual_sxx_MPa"] = float(sres[:, 0].max() / 1e6); r["max_residual_syy_MPa"] = float(sres[:, 1].max() / 1e6)
        out["cases"][name] = r
        print(f"{name:30s}: [A strain] MD {r['MD']['strength_MPa']:6.1f} MPa (first failure at depth {r['MD']['first_failure_depth_mm']:.2f} mm), "
              f"TD {r['TD']['strength_MPa']:6.1f} MPa (depth {r['TD']['first_failure_depth_mm']:.2f} mm); "
              f"max residual tension sxx {r['max_residual_sxx_MPa']:+.1f}, syy {r['max_residual_syy_MPa']:+.1f} MPa")
        print(f"   {'':30s} [B fibre/matrix] MD {r['MD']['B_strength_MPa']:6.1f} MPa (depth {r['MD']['B_depth_mm']:.2f} mm, ply {r['MD']['B_ply_angle_deg']:+.0f} deg), "
              f"TD {r['TD']['B_strength_MPa']:6.1f} MPa (depth {r['TD']['B_depth_mm']:.2f} mm, ply {r['TD']['B_ply_angle_deg']:+.0f} deg)")
    base = out["cases"]["no residual stress"]
    for name in list(cases)[1:]:
        c = out["cases"][name]
        c["change_MD"] = c["MD"]["strength_MPa"] / base["MD"]["strength_MPa"] - 1
        c["change_TD"] = c["TD"]["strength_MPa"] / base["TD"]["strength_MPa"] - 1
        c["B_change_MD"] = c["MD"]["B_strength_MPa"] / base["MD"]["B_strength_MPa"] - 1
        c["B_change_TD"] = c["TD"]["B_strength_MPa"] / base["TD"]["B_strength_MPa"] - 1
        print(f"  {name}: strength change [A] MD {c['change_MD'] * 100:+.1f} %, TD {c['change_TD'] * 100:+.1f} % | "
              f"[B] MD {c['B_change_MD'] * 100:+.1f} %, TD {c['B_change_TD'] * 100:+.1f} %")
    print(f"[B] X_L (Kelly-Tyson, example constants) = {XL / 1e6:.0f} MPa (datasheet 155 MPa, proxy grade, not fitted)")
    out["B_XL_MPa"] = XL / 1e6
    print(f"eps_c (example, from {SIG_MD_DATASHEET / 1e6:.0f} MPa MD, stress-free) = {ec * 100:.2f} %")
    json.dump(out, open(os.path.join(OUT, "strength.json"), "w"), indent=1)
    print("wrote", os.path.join(OUT, "strength.json"))


if __name__ == "__main__":
    main()
