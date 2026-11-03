"""validation/greif3d_gate_ranking.py -- a first look at stage 3 (moulding design) on 3-D parts: does the physics
prior choose the same gate position as the data? Post-processing of fifa §19 (per_geometry.json), no new FE.

In Greif & Meyer (Zenodo 17831586) geometries that share a hash are the same base part with different gate
positions (the .yml holds inlet_location / inlet_normal; 33 of the 110 base parts have 2-10 gate variants).
For each base part with >= 2 variants already computed, the gates are ranked by the maximum warp
(rigid motion removed) of (a) the data (Moldflow) and (b) the physics prior; one common factor would not
change a ranking within a part, so this is also the ranking of the corrected prediction.
Reported: whether the best gate agrees, the Spearman rank correlation per part.
Honest scope: the prior uses the data's fibre orientation FOR EACH GATE (from Moldflow's filling analysis);
choosing a gate that was not simulated needs the orientation for that gate (a flow analysis or a learned
model), which this check does not test. The "data" are Moldflow results, not measurements. Interim set.
Run:  python3 validation/greif3d_gate_ranking.py abaqus/denso_plate/abaqus_results/greif3d_rescue/per_geometry.json
"""
import collections
import json
import os
import sys

import numpy as np

src = sys.argv[1]
R = json.load(open(src))
h = collections.defaultdict(list)
for r in R:
    h[r["vtu"].split("_")[1]].append(r)
rows = []
for k, v in sorted(h.items()):
    if len(v) < 2:
        continue
    d = np.array([x["max_data_mm"] for x in v]); p = np.array([x["max_prior_mm"] for x in v])
    rd, rp = np.argsort(np.argsort(d)), np.argsort(np.argsort(p))
    rho = float(np.corrcoef(rd, rp)[0, 1]) if len(v) > 2 else float(np.sign((d[1] - d[0]) * (p[1] - p[0])))
    gap = float(np.sort(d)[1] / np.sort(d)[0] - 1)                       # how distinct the best gate is in the data
    rows.append(dict(base=k, n_gates=len(v), ids=[x["id"] for x in v], max_data_mm=d.round(3).tolist(),
                     max_prior_mm=p.round(3).tolist(), best_agrees=bool(np.argmin(d) == np.argmin(p)),
                     spearman=rho, data_gap_best_to_second=gap))
    print(f"{k[:8]}: {len(v)} gates, data {d.round(2).tolist()}, prior {p.round(2).tolist()}, best gate agrees "
          f"{rows[-1]['best_agrees']}, rank corr {rho:+.2f}, data gap best/second {gap * 100:.0f} %")
n = len(rows); ok = sum(r["best_agrees"] for r in rows)
print(f"best gate agrees in {ok} of {n} base parts; the disagreeing ones have data gaps "
      f"{[round(r['data_gap_best_to_second'] * 100) for r in rows if not r['best_agrees']]} %")
dst = os.path.join(os.path.dirname(os.path.abspath(src)), "gate_ranking.json")
json.dump(dict(n_parts=n, best_agrees=ok, parts=rows), open(dst, "w"), indent=1); print("wrote", dst)
