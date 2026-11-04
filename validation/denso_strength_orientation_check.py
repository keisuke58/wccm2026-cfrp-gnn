"""validation/denso_strength_orientation_check.py -- an independent check of the strain criterion of
validation/denso_strength_residual.py against published strengths of GF30/PPS with two orientation states.

Data (different grades, examples of the orientation effect, not of 1130A6):
  aligned: datasheet ISO 527 bar of DURAFIDE 1130T6, 155 MPa (fibres mostly along the bar);
  balanced: Poellinger et al., Polymers 16(3), 412, 2024 (PMC10857418), PPS/GF30 (MOCOM) specimens cut from moulded
            plates with "equal fractions of the parallel and perpendicular-oriented fibers", 113.2 +- 1.8 MPa.
Measured ratio balanced / aligned = 0.73.
Model: uniform in-plane orientation through the thickness, a_zz = 0.05 (assumed); aligned bar a_xx = 0.80
(assumed, typical of an ISO bar skin-dominated state), balanced plate a_xx = a_yy = 0.475. The strain criterion
(one critical strain) gives strength proportional to the membrane modulus in the load direction, so the ratio
needs no critical strain and no strength value. Residual stress is ignored here (both specimens contain some).
Honest scope: two grades from two suppliers, assumed orientation states; a check of the orientation effect only.
Run:  python3 validation/denso_strength_orientation_check.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import denso_tsuda2026_compare as tc  # noqa: E402

MEAS = dict(aligned=155.0, balanced=113.2)


def modulus_x(axx, ayy, n=21):
    z = np.linspace(-1.5e-3, 1.5e-3, n)
    ax, ay, az = tc.layers(z, np.full(n, axx), np.full(n, ayy))
    Q, _ = tc.layer_props(tc.T_RT, ax, ay, az)
    A, _, _, _ = tc.abd(z, Q)
    return 1 / (np.linalg.inv(A)[0, 0] * (z[-1] - z[0]))


out = {}
for a_al in (0.70, 0.80, 0.90):
    r = modulus_x(0.475, 0.475) / modulus_x(a_al, 0.95 - a_al)
    out[f"a_xx_bar_{a_al}"] = r
    print(f"aligned bar a_xx = {a_al:.2f}: predicted strength ratio balanced / aligned = {r:.2f}")
print(f"measured (two grades): {MEAS['balanced'] / MEAS['aligned']:.2f}")
out["measured_ratio"] = MEAS["balanced"] / MEAS["aligned"]
os.makedirs(os.path.join(ROOT, "results", "denso_strength"), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, "results", "denso_strength", "orientation_check.json"), "w"), indent=1)
