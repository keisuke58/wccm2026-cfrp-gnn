"""extract_denso_plate.py -- read the Denso plate .odb files and write <job>_abaqus.json
(centre-column elastic strain EE11, EE22 per layer; mid-plane U3 along the x and y axes).
Run with Abaqus' Python on the Abaqus host, from abaqus/denso_plate/jobs/:

    abaqus python ../extract_denso_plate.py                 # every <job>.odb that has a _ref.json
    abaqus python ../extract_denso_plate.py denso_asymmetric

Labels are taken from <job>_ref.json (written by gen_denso_plate.py), so no set lookup in the
odb is needed. EE is averaged over the 8 integration points of each C3D8I element; for the
linear-in-z strain of a bent plate this equals the centroid value.
Works with the Python 2.7 (Abaqus <= 2023) and 3.x (2024) interpreters.
Not run in Abaqus yet (written in an environment without a licence).
"""
from __future__ import print_function
import glob
import json
import os
import sys

try:
    from odbAccess import openOdb
except Exception:                                   # pragma: no cover
    print("Run with Abaqus' Python:  abaqus python extract_denso_plate.py")
    raise


def extract(job):
    ref = json.load(open(job + "_ref.json"))
    odb = openOdb(job + ".odb", readOnly=True)
    step = odb.steps[list(odb.steps.keys())[-1]]
    fr = step.frames[-1]
    col = set(ref["col_elements"])
    acc = {}
    for v in fr.fieldOutputs["EE"].values:
        if v.elementLabel in col:
            acc.setdefault(v.elementLabel, []).append((float(v.data[0]), float(v.data[1])))
    missing = [e for e in ref["col_elements"] if e not in acc]
    if missing:
        raise RuntimeError("EE missing for centre-column elements %s" % missing[:5])
    mean = lambda l, i: sum(t[i] for t in l) / float(len(l))
    exx = [mean(acc[e], 0) for e in ref["col_elements"]]
    eyy = [mean(acc[e], 1) for e in ref["col_elements"]]
    want = set(ref["midx_nodes"]) | set(ref["midy_nodes"])
    u3 = {}
    for v in fr.fieldOutputs["U"].values:
        if v.nodeLabel in want:
            u3[v.nodeLabel] = float(v.data[2])
    out = dict(job=job, solver="abaqus", ee_xx=exx, ee_yy=eyy,
               u3_midx=[u3[n] for n in ref["midx_nodes"]],
               u3_midy=[u3[n] for n in ref["midy_nodes"]])
    odb.close()
    with open(job + "_abaqus.json", "w") as f:
        json.dump(out, f)
    print("wrote %s_abaqus.json" % job)


if __name__ == "__main__":
    jobs = sys.argv[1:] or sorted(p[:-4] for p in glob.glob("*.odb")
                                  if os.path.exists(p[:-4] + "_ref.json"))
    for j in jobs:
        extract(j)
