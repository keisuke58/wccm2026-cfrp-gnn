"""validation/greif3d_transfer_factor.py -- can the correction factor found on OTHER parts predict a NEW part with no
measurement at all? Post-processing of validation/greif3d_rescue.py results (per_geometry.json, fifa §19).

The 1-factor error is a quadratic in the factor a, so it follows exactly from three numbers already stored per
geometry. With p, d the rigid-removed prior and data fields (rigid removal is a linear projection):
    E(a)^2 = |a p - d|^2 / |d|^2 = a^2 r^2 - 2 a r c + 1,   r = |p| / |d|,  c = cos(p, d)            (1)
At a = 1 (the stored 'prior' error e0):  r^2 - 2 r c + 1 = e0^2  ->  r = c -+ sqrt(c^2 - 1 + e0^2)           (2)
Both roots fit e0; the stored data alone cannot tell them apart, so each geometry takes the root whose best
factor c / r matches the stored field_1f_a (the FE post-processing). A first version took the larger root
everywhere; T1 failed (factor 0.68 vs 1.93: the prior UNDER-predicts, |p| < |d|) and the choice was fixed.
Checks:
  T1 the best factor a* = c / r and best error sqrt(1 - c^2) from (1)-(2) reproduce the stored field_1f and
     field_1f_a (both computed by the FE post-processing directly);
  T2 leave-one-out: a_LOO = median of a* over the OTHER geometries; error E(a_LOO) of the held-out geometry
     with NO measurement on it, compared with the prior (a = 1) and with RESCUE using 5 / 10 nodes.
Honest scope: the "data" are Moldflow results without holding pressure (not measurements), one material, the
same simulation setup for all parts; transfer across materials or processes is not tested.
Run:  python3 validation/greif3d_transfer_factor.py abaqus/denso_plate/abaqus_results/greif3d_rescue/per_geometry.json
"""
import json
import os
import sys

import numpy as np

src = sys.argv[1]
R = json.load(open(src))
c = np.array([r["cos_prior_data"] for r in R]); e0 = np.array([r["prior"] for r in R])
sq = np.sqrt(np.maximum(c ** 2 - 1 + e0 ** 2, 0.0)); a_fe = np.array([x["field_1f_a"][0] for x in R])
r = np.where(np.abs(c / (c - sq) - a_fe) < np.abs(c / (c + sq) - a_fe), c - sq, c + sq)
astar = c / r
E = lambda a: np.sqrt(np.maximum(a ** 2 * r ** 2 - 2 * a * r * c + 1, 0.0))
t1a = np.max(np.abs(astar - np.array([x["field_1f_a"][0] for x in R])))
t1e = np.max(np.abs(np.sqrt(1 - c ** 2) - np.array([x["field_1f"] for x in R])))
print(f"[T1] from (1)-(2): best factor max |diff| {t1a:.1e}, best error max |diff| {t1e:.1e} (vs FE post-processing)")
loo = np.array([np.median(np.delete(astar, i)) for i in range(len(R))])
eL = E(loo)
out = dict(n=len(R), T1_factor=float(t1a), T1_error=float(t1e), a_star_median=float(np.median(astar)),
           a_star_q=[float(q) for q in np.percentile(astar, [25, 75])])
for k, v in (("prior (a = 1)", e0), ("LOO factor, no measurement", eL),
             ("RESCUE 5 nodes, 1 factor", np.array([x["rescue_5_1f"] for x in R])),
             ("RESCUE 10 nodes, 2 factors", np.array([x["rescue_10_2f"] for x in R])),
             ("whole field, 1 factor", np.sqrt(1 - c ** 2))):
    q = np.percentile(v, [25, 50, 75])
    out[k] = dict(q25=float(q[0]), median=float(q[1]), q75=float(q[2]), max=float(v.max()))
    print(f"[T2] {k:28s} median {q[1]:.3f}  (q25 {q[0]:.3f}, q75 {q[2]:.3f}, max {v.max():.3f})")
worse = int(np.sum(eL > e0))
print(f"     best factor a* median {np.median(astar):.2f} (q25-q75 {np.percentile(astar, 25):.2f}-{np.percentile(astar, 75):.2f}); "
      f"LOO factor worse than the prior on {worse} of {len(R)} geometries")
out["n_loo_worse_than_prior"] = worse
dst = os.path.join(os.path.dirname(os.path.abspath(src)), "transfer_factor.json")
json.dump(out, open(dst, "w"), indent=1); print("wrote", dst)
