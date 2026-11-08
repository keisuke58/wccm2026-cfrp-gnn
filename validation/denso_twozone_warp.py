"""validation/denso_twozone_warp.py -- mould-wall temperatures -> through-thickness orientation (two-zone rule of
validation/denso_flow_coupling.py) -> warp of the PPS process model. Replaces the asymmetry estimate of
validation/denso_flow_orientation.py, whose 1-D shear model failed its check.

Two-zone rule: the skin that freezes during filling (thickness d from 1-D conduction into each wall, eq. 1 of
denso_flow_orientation.py, fill time 1 s ASSUMED) is aligned with the flow (orientation of the Folgar-Tucker trajectory
at large strain); the channel still flowing at the end of filling keeps the gate state a0 (ASSUMED 0.20 / 0.70 / 0.10).
Checked on the paper's 3 mm and 1 mm plates (core 1.64 vs 1.60 mm, 0.24 vs 0.19 mm; denso_flow_coupling.py D2, D3).
Warp: the process model of design/denso_mould_pps_bo.py (abaqus/denso_plate/process_reference.py: 1-D cooling between
the walls, crystallisation, viscoelastic relaxation, laminate), the plate cooled at the same wall temperatures;
sag of a 100 mm span (kappa_y L^2 / 8), as denso_flow_orientation.py.
Reported:
  W1 symmetric walls 150 / 150 C: the rule gives a symmetric profile -> no warp (consistency);
  W2 the paper's measured shells are unequal (0.79 top / 0.60 bottom mm): the wall-temperature difference that the rule
     needs for that, and the warp it gives, compared with the warp of the read orientation (process model, -2.85 mm);
  W3 a sweep of the wall difference (design handle).
Honest scope: the rule's mechanism is not explained; assumed fill time, no-flow temperature, diffusivity and gate state;
the orientation set by the FILL is assumed to depend on the same wall temperatures as the cooling; one flow length;
the mould temperatures of the paper's plate are nominal (150 C both), so W2 is an interpretation, not a check.
Result (2026-11-08): W1 sag 0.000 mm (rule) vs -2.854 mm (read orientation). W2 the paper's unequal shells would need
walls 194 / 106 C (88 C apart, nominal 150 / 150): implausible, so that asymmetry is not explained by the walls; the rule
then gives -1.79 mm, 63 % of the read orientation's -2.85 mm. W3 sag is close to linear in the wall difference:
+-0.38 / 0.76 / 1.69 mm for 20 / 40 / 80 C (about 0.02 mm per C), a design handle. (Skin fraction averaged per layer;
a first version without it gave a staircase, because the skin changes 0.02 mm per 10 C against 0.077 mm layers.)
Run:  python3 validation/denso_twozone_warp.py   (writes results/denso_flow/twozone_warp.json)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "design"))
import denso_flow_coupling as fc  # noqa: E402
import denso_flow_orientation as fo  # noqa: E402

OUT = os.path.join(ROOT, "results", "denso_flow")
BASE = dict(fo.BASE, L=50e-3)
AL = fc.trajectory(BASE["a0"], BASE["Ci"], 1.0)[1][-1]
A0 = BASE["a0"]
HMM = fo.H * 1e3


def skins(Tb, Tt):
    return fo.frozen(Tb, BASE) * 1e3, fo.frozen(Tt, BASE) * 1e3       # bottom, top [mm]


def orient_fn(Tb, Tt):
    db, dt = skins(Tb, Tt)

    def fn(zm):
        zm = np.asarray(zm); depth = (1 - zm) / 2 * HMM                 # zm = +1 top face
        h = HMM / len(zm) / 2                                            # layer half-thickness (equal layers)
        f = np.clip((dt - (depth - h)) / (2 * h), 0, 1) + np.clip(((depth + h) - (HMM - db)) / (2 * h), 0, 1)
        f = np.clip(f, 0, 1)                                             # skin fraction of each layer (no staircase)
        a11 = f * AL[0, 0] + (1 - f) * A0[0]; a22 = f * AL[1, 1] + (1 - f) * A0[1]
        return a11, a22, np.clip(1 - a11 - a22, 0, 1)
    return fn


def main():
    os.makedirs(OUT, exist_ok=True)
    import denso_mould_pps_bo as bo
    import process_reference as pr
    P_read = bo.setup()
    CP = pr.C

    def sag(P, Tb, Tt):
        t, Tn = pr.heat_1d(Tb, Tt); o = pr.plate_history(P, t, Tn)
        return float(o["kappa"][1] * 0.10 ** 2 / 8 * 1e3)

    def sag_rule(Tb, Tt):
        P, _ = pr.layer_props(orient_fn(Tb, Tt), c=CP)
        return sag(P, Tb, Tt)
    s_read = sag(P_read, 150.0, 150.0)
    s_sym = sag_rule(150.0, 150.0)
    print(f"[W1] symmetric walls 150/150 C: two-zone rule sag {s_sym:+.3f} mm (symmetric profile, expected ~0); "
          f"read orientation (paper Fig. 8a) sag {s_read:+.3f} mm")
    # W2: wall difference giving the paper's shells 0.79 (top) / 0.60 (bottom), mean wall 150 C
    target = 0.79 - 0.60
    lo, hi = 0.0, 60.0
    for _ in range(50):
        m = 0.5 * (lo + hi); db, dt = skins(150 + m, 150 - m)
        lo, hi = (m, hi) if dt - db < target else (lo, m)
    dT = 0.5 * (lo + hi); db, dt = skins(150 + dT, 150 - dT)
    s_w2 = sag_rule(150 + dT, 150 - dT)
    print(f"[W2] the paper's unequal shells (top 0.79 / bottom 0.60 mm) need walls bottom {150 + dT:.0f} / top {150 - dT:.0f} C "
          f"(skins {dt:.2f} / {db:.2f} mm); sag {s_w2:+.3f} mm vs read orientation {s_read:+.3f} mm")
    rows = []
    for d in (-40.0, -20.0, -10.0, 0.0, 10.0, 20.0, 40.0):
        Tb, Tt = 150.0 + d, 150.0 - d
        db, dt = skins(Tb, Tt); s = sag_rule(Tb, Tt)
        rows.append(dict(Tb=Tb, Tt=Tt, skin_bottom_mm=db, skin_top_mm=dt, sag_mm=s))
        print(f"[W3] walls bottom {Tb:.0f} / top {Tt:.0f} C: skins {db:.2f} / {dt:.2f} mm, sag {s:+.3f} mm", flush=True)
    json.dump(dict(W1=dict(sag_rule=s_sym, sag_read=s_read), W2=dict(dT=dT, skin_top=dt, skin_bottom=db, sag=s_w2), W3=rows),
              open(os.path.join(OUT, "twozone_warp.json"), "w"), indent=1)
    print("wrote", os.path.join(OUT, "twozone_warp.json"))


if __name__ == "__main__":
    main()
