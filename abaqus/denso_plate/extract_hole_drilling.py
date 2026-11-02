"""extract_hole_drilling.py -- read denso_hole<tag>.odb (gen_hole_drilling.py) and write
denso_hole<tag>_abaqus.json: for every step (COOL, DRILL01..DRILL10) the strain E of each gauge
element, averaged over its integration points. Run with Abaqus' Python in abaqus/denso_plate/jobs/:

    abaqus python ../extract_hole_drilling.py denso_hole
Python 2.7 / 3.x. Not run in Abaqus yet.
"""
from __future__ import print_function
import json
import sys

try:
    from odbAccess import openOdb
except Exception:                                   # pragma: no cover
    print("Run with Abaqus' Python:  abaqus python extract_hole_drilling.py denso_hole")
    raise


def main(job):
    meta = json.load(open(job + "_meta.json"))
    want = set()
    for g in meta["gauges"].values():
        want.update(g["elements"])
    odb = openOdb(job + ".odb", readOnly=True)
    out = dict(job=job, solver="abaqus", shear="engineering", steps=[])
    for name in odb.steps.keys():
        fr = odb.steps[name].frames[-1]
        acc = {}
        for v in fr.fieldOutputs["E"].values:
            if v.elementLabel in want:
                acc.setdefault(v.elementLabel, []).append([float(x) for x in v.data])
        mean = dict((str(e), [sum(r[i] for r in rows) / len(rows) for i in range(6)]) for e, rows in acc.items())
        out["steps"].append(dict(name=name, E=mean))
    odb.close()
    json.dump(out, open(job + "_abaqus.json", "w"))
    print("wrote %s_abaqus.json (%d steps)" % (job, len(out["steps"])))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "denso_hole")
