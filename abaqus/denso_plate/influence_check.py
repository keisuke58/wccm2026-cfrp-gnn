"""influence_check.py -- assemble and check the hole-drilling influence functions written by
`run_hole_drilling.sh influence` (gen_hole_drilling.py --eig {xx,yy} --band b): the released strain at
each gauge and drilling step for a UNIT eigenstrain (1e-3) of one in-plane component in one 0.2 mm
depth band. Released strain is linear in the eigenstrain, so for any eigenstrain profile e*(z)

    y[gauge, step] = sum_c sum_b  G_c[gauge, step, b] * ebar*_c,b / 1e-3,                         (1)

with ebar* the band average: the operator that turns a predicted eigenstrain (or stress) profile into
the quantity actually measured, for profiles other than the paper's. The stiffness is the paper's
(the operator is exact only for that stiffness profile; for other orientations it is an approximation).

Checks (printed):
  I1 linearity: sum over the 16 bands vs the run with unit eigenstrain at all depths (should agree to
     round-off, as the basis runs did, 2.5e-7);
  I2 band averaging: (1) with the paper's eigenstrain averaged per band vs the full production run
     denso_hole_released.json (abaqus_results): the error of using 0.2 mm bands.
Writes <jobs>/influence_operator.json (G and the checks).

Run:  python3 abaqus/denso_plate/influence_check.py abaqus/denso_plate/jobs
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CH = ["GCH1", "GCH2", "GCH3"]
NB = 16
UNIT = 1.0e-3


def load(d, tag):
    r = json.load(open(os.path.join(d, f"denso_hole_{tag}_released.json")))
    return np.array([r["released"][g] for g in CH]), r["depth_mm"]        # (3, steps+1)


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "jobs")
    G = {}; out = dict(unit=UNIT, band_mm=0.2, n_bands=NB, gauges=CH, checks={})
    for c in ("xx", "yy"):
        G[c] = np.stack([load(d, f"inf_{c}_{b:02d}")[0] for b in range(NB)], axis=-1)   # (3, steps+1, NB)
        allr, depth = load(d, f"inf_{c}_all")
        err = float(np.max(np.abs(G[c].sum(-1) - allr)) / np.max(np.abs(allr)))
        out["checks"][f"I1_linearity_{c}"] = err
        print(f"[I1] {c}: sum of {NB} bands vs all-depth run, max rel. difference {err:.1e}")
    out["depth_mm"] = depth
    # I2: paper eigenstrain, band-averaged
    import gen_hole_drilling as g
    mat = g.layer_materials(154)
    dep = np.asarray(mat["depth"])                    # mm, layer centres from the drilled face
    eth = np.asarray(mat["eth"])                      # (154, 3) integrated eigenstrain 180 -> 20 C
    band = np.minimum((dep // 0.2).astype(int), NB - 1)
    ebar = {c: np.array([eth[band == b, i].mean() for b in range(NB)]) for i, c in enumerate(("xx", "yy"))}
    y = sum(G[c] @ (ebar[c] / UNIT) for c in ("xx", "yy"))                        # (3, steps+1)
    ref_path = os.path.join(HERE, "abaqus_results", "denso_hole_released.json")
    if os.path.exists(ref_path):
        ref = json.load(open(ref_path))
        R = np.array([ref["released"][gg] for gg in CH])
        e2 = float(np.max(np.abs(y - R)) / np.max(np.abs(R)))
        out["checks"]["I2_band_average_vs_full"] = e2
        for k, gg in enumerate(CH):
            print(f"[I2] {gg}: operator {np.round(y[k, 1:]).astype(int).tolist()}")
            print(f"      full run {np.round(R[k, 1:]).astype(int).tolist()}")
        print(f"[I2] max rel. difference, band-averaged paper eigenstrain vs full production run: {e2:.1e}")
    out["G"] = {c: G[c].tolist() for c in G}
    json.dump(out, open(os.path.join(d, "influence_operator.json"), "w"))
    print(f"wrote {os.path.join(d, 'influence_operator.json')}")


if __name__ == "__main__":
    main()
