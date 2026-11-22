"""validation/flow_freeze_thermal.py -- the thermal part of the two-zone orientation rule (research/MEMO_flow_orientation.md)
without solving the flow: 1-D transient conduction through the thickness after the melt front has passed a section.

    rho c dT/dt = k d2T/dz2,  T(z, 0) = T_melt,  -k dT/dz = h (T - T_wall) at both walls (h -> inf: fixed wall)      (1)
A layer is "frozen" when T < T_nf (no-flow temperature). Contact time at a section x of a cavity of length L filled at a
constant front speed in the time t_fill:  t_c = t_fill (1 - x / L).                                                   (2)
The earlier check of the rule (denso_flow_coupling.py D1-D3) used the semi-infinite erf solution with t_c = t_fill = 1 s
(ASSUMED). This script asks what survives when (2), the finite thickness and a finite wall coefficient are used.
Checks and results (printed, written to results/denso_flow/freeze_thermal.json):
  F1 numerics: fixed wall, short time, against the erf solution;
  F2 skin thickness for the 3 mm plate: erf vs finite slab vs finite wall coefficient h = 5e3 W/m2K (the CFD's value),
     at the same contact time;
  F3 the rule with (2) at the paper's site (50 mm of 80 mm): the fill time is the ONE free number, set from the 3 mm
     plate's core (1.60 mm); the 1 mm plate (core 0.19 mm) is then PREDICTED with the same volumetric flow rate
     (t_fill proportional to the thickness). Done for a fixed wall and for h = 5e3;
  F4 one-sided comparison with the fifa 2-D filling run (stage4_strip): at 50 mm the cell at 0.144 mm depth had not
     frozen 0.134 s after the front passed. Thermal constants of that run's tables are not in the repository:
     typical HDPE melt values are used (k 0.25 W/mK, rho c 2.0e6 J/m3K, ASSUMED).
Constants (ASSUMED, as denso_flow_orientation.py): PPS/GF diffusivity 1.8e-7 m2/s, k = 0.30 W/mK, T_melt 320 C,
T_wall 150 C, T_nf 280 C.
Honest scope: no convection of hot melt along the flow, no shear heating, no latent heat (all three make the real skin
thinner); constant front speed; the paper gives no fill time, so F3 tests the RATIO between the two plates only.
Run:  python3 validation/flow_freeze_thermal.py
"""
import json
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D, K = 1.8e-7, 0.30
TM, TW, TNF = 320.0, 150.0, 280.0


def skin(H, t, h=None, D=D, k=K, Tm=TM, Tw=TW, Tnf=TNF, n=400):
    """frozen thickness per wall [m] after contact time t; implicit finite differences on half the thickness."""
    dz = (H / 2) / n; z = (np.arange(n) + 0.5) * dz                # cell centres from the wall to the mid-plane
    T = np.full(n, Tm); nt = max(int(t / 2e-4), 200); dt = t / nt; r = D * dt / dz ** 2
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = 1 + 2 * r
        if i > 0: A[i, i - 1] = -r
        if i < n - 1: A[i, i + 1] = -r
    A[-1, -1] = 1 + r                                               # symmetry at the mid-plane
    if h is None:
        g = 2.0                                                     # fixed wall temperature at the face (half cell)
    else:
        g = 1.0 / (0.5 + k / (h * dz))                              # series: half cell + wall film
    A[0, 0] = 1 + r + r * g; b0 = r * g * Tw
    Ai = np.linalg.inv(A)
    for _ in range(nt):
        rhs = T.copy(); rhs[0] += b0; T = Ai @ rhs
    fz = T < Tnf
    if not fz.any(): return 0.0
    if fz.all(): return H / 2
    i = int(np.argmin(fz))                                          # first unfrozen cell
    if i == 0: return 0.0
    return float(z[i - 1] + (Tnf - T[i - 1]) / (T[i] - T[i - 1]) * dz)


def erf_skin(t, D=D):
    rr = (TNF - TW) / (TM - TW); lo, hi = 0.0, 6.0
    for _ in range(60):
        m = 0.5 * (lo + hi); lo, hi = (m, hi) if math.erf(m) < rr else (lo, m)
    return 2 * lo * math.sqrt(D * t)


def main():
    out = {}
    H3, H1 = 3.08e-3, 1.08e-3
    a, b = skin(H3, 0.05) * 1e3, erf_skin(0.05) * 1e3
    out["F1"] = dict(fd_mm=a, erf_mm=b); print(f"[F1] fixed wall, 0.05 s: finite differences {a:.3f} mm, erf {b:.3f} mm")
    out["F2"] = {}
    for t in (0.375, 1.0):
        e, f, g = erf_skin(t) * 1e3, skin(H3, t) * 1e3, skin(H3, t, h=5e3) * 1e3
        out["F2"][str(t)] = dict(erf=e, slab=f, slab_h5e3=g)
        print(f"[F2] 3 mm plate, contact {t:.3f} s: skin erf {e:.2f}, finite slab {f:.2f}, with h = 5e3 W/m2K {g:.2f} mm")
    out["F3"] = {}
    for lab, h in (("fixed wall", None), ("h = 5e3", 5e3)):
        lo, hi = 0.05, 30.0                                         # contact time giving the 3 mm core 1.60 mm
        for _ in range(40):
            m = 0.5 * (lo + hi); lo, hi = (m, hi) if H3 - 2 * skin(H3, m, h) > 1.60e-3 else (lo, m)
        tc3 = 0.5 * (lo + hi); tf3 = tc3 / (1 - 50 / 80); tc1 = tc3 * H1 / H3
        core1 = (H1 - 2 * skin(H1, tc1, h)) * 1e3
        lo, hi = 0.01, 5.0                                          # contact time the 1 mm core 0.19 mm needs
        for _ in range(40):
            m = 0.5 * (lo + hi); lo, hi = (m, hi) if H1 - 2 * skin(H1, m, h) > 0.19e-3 else (lo, m)
        tc1n = 0.5 * (lo + hi)
        lo, hi = 0.01, 5.0                                          # ... and the time at which the 1 mm plate is frozen through
        for _ in range(40):
            m = 0.5 * (lo + hi); lo, hi = (m, hi) if H1 - 2 * skin(H1, m, h) > 1e-6 else (lo, m)
        tfull = 0.5 * (lo + hi)
        out["F3"][lab] = dict(contact_3mm_s=tc3, fill_3mm_s=tf3, contact_1mm_same_flow_rate_s=tc1, core_1mm_pred_mm=core1,
                              contact_1mm_needed_s=tc1n, frozen_through_1mm_s=tfull, ratio_needed=tc1n / tc3, ratio_thickness=H1 / H3)
        print(f"[F3] {lab}: 3 mm core 1.60 mm needs contact {tc3:.2f} s -> fill time {tf3:.2f} s (site 50 of 80 mm); "
              f"1 mm plate at the same flow rate: contact {tc1:.2f} s -> core {core1:.2f} mm (paper 0.19 mm). "
              f"The 1 mm core needs contact {tc1n:.3f} s (frozen through at {tfull:.3f} s): ratio to the 3 mm plate "
              f"{tc1n / tc3:.2f} against the thickness ratio {H1 / H3:.2f}")
    # F4: CFD, typical HDPE values
    Dh, kh = 0.25 / 2.0e6, 0.25
    s = skin(H3, 0.134, h=5e3, D=Dh, k=kh, Tnf=270.0) * 1e3; s2 = skin(H3, 0.134, h=None, D=Dh, k=kh, Tnf=270.0) * 1e3
    out["F4"] = dict(skin_h5e3_mm=s, skin_fixed_mm=s2, cfd_unfrozen_depth_mm=0.144)
    print(f"[F4] fifa 2-D run, 0.134 s after the front (typical HDPE constants, ASSUMED): skin {s:.3f} mm with h = 5e3 "
          f"({s2:.3f} mm fixed wall); CFD: not frozen at 0.144 mm -> "
          + ("consistent" if s < 0.13 else "borderline, not decided (within the uncertainty of the assumed constants)" if s < 0.16 else "NOT consistent"))
    dst = os.path.join(ROOT, "results", "denso_flow", "freeze_thermal.json")
    json.dump(out, open(dst, "w"), indent=1); print("wrote", dst)


if __name__ == "__main__":
    main()
