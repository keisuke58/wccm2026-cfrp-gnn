"""validation/greif3d_crystal_factor.py -- why is the 3-D correction factor about 2? A physics estimate from the
crystallisation shrinkage of PP, which the prior of validation/greif3d_rescue.py leaves out.

The prior uses a matrix strain alpha_m dT = 1e-4 x 100 K = 1.0 % (general values, example). The data (Greif &
Meyer) were run WITHOUT holding pressure, so the volume lost on crystallisation is not fed back by packing.
Linear crystallisation shrinkage of the matrix (isotactic PP, textbook densities rho_c = 0.946, rho_a = 0.855
g/cm3, general values; not checked against the Celanese grade):
    eps_c = X_c (1 - rho_a / rho_c) / 3                                                                    (1)
Added to the matrix strain, it passes through the same Mori-Tanaka / Levin chain (fibres constrain it), so the
composite eigenstrain grows by a factor close to (alpha_m dT + eps_c) / (alpha_m dT). Checked here per axis for
one orientation state (not exactly uniform: the fibre direction grows less).
Then the error of the uncorrected-but-crystallisation-scaled prediction follows exactly from the stored
per-geometry values (as validation/greif3d_transfer_factor.py, eq. 1), for X_c = 0.4 and 0.5.
Honest scope: one scalar factor approximates an axis-dependent ratio (about 8 % apart); X_c and how much of the
crystallisation happens before the gate freezes are unknown; general constants; Moldflow data, not measurements.
Run:  python3 validation/greif3d_crystal_factor.py abaqus/denso_plate/abaqus_results/greif3d_rescue/per_geometry.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import greif3d_rescue as g  # noqa: E402

RHO_C, RHO_A = 0.946, 0.855
src = sys.argv[1]
A = np.diag([0.6, 0.3, 0.1])
base = np.diag(g.cluster_props(np.array([A]))[1][0])
R = json.load(open(src))
c = np.array([r["cos_prior_data"] for r in R]); e0 = np.array([r["prior"] for r in R]); afe = np.array([r["field_1f_a"][0] for r in R])
sq = np.sqrt(np.maximum(c ** 2 - 1 + e0 ** 2, 0)); rr = np.where(np.abs(c / (c - sq) - afe) < np.abs(c / (c + sq) - afe), c - sq, c + sq)
E = lambda a: np.sqrt(np.maximum(a ** 2 * rr ** 2 - 2 * a * rr * c + 1, 0))
out = dict(n=len(R), observed_factor_median=float(np.median(c / rr)), cases=[])
am = g.PP["alpha"]
for Xc in (0.4, 0.5):
    ec = Xc * (1 - RHO_A / RHO_C) / 3
    g.PP["alpha"] = am + ec / abs(g.DT)
    new = np.diag(g.cluster_props(np.array([A]))[1][0]); g.PP["alpha"] = am
    ratio = new / base; a = float(np.mean(ratio))
    out["cases"].append(dict(Xc=Xc, eps_c=ec, ratio_per_axis=ratio.tolist(), factor=a, median_error=float(np.median(E(a)))))
    print(f"X_c {Xc}: eps_c {ec * 100:.2f} %, composite eigenstrain ratio per axis {np.round(ratio, 2).tolist()} (mean {a:.2f}); "
          f"median error with this factor and no data {np.median(E(a)):.3f}")
print(f"observed best factor median {out['observed_factor_median']:.2f}; no correction {np.median(e0):.3f}; "
      f"other parts' factor (LOO) see transfer_factor.json")
dst = os.path.join(os.path.dirname(os.path.abspath(src)), "crystal_factor.json")
json.dump(out, open(dst, "w"), indent=1); print("wrote", dst)
