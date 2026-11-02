"""extract_denso_process.py -- read the Denso PROCESS odbs (heat + stress) and write
denso_proc_<case>_abaqus.json for compare_denso_process.py. Run with Abaqus' Python from
abaqus/denso_plate/jobs/:

    abaqus python ../extract_denso_process.py asymmetric        # or: symmetric
    abaqus python ../extract_denso_process.py                   # every case in manifest_process.txt

Extracted:
  heat   NT of the centre-column nodes (NCOL) at every increment, with total time;
  stress S11, S22, S33 and SDV1 (relative crystallinity) of the centre-column elements (ECOL),
         averaged over the 8 integration points, at the end of MOULD and at the end of AIR;
         U3 of the mid-plane nodes along both symmetry axes at the end of AIR.
Labels come from denso_proc_<case>_ref.json. Python 2.7 / 3.x. Not run in Abaqus yet.
"""
from __future__ import print_function
import json
import sys

try:
    from odbAccess import openOdb
except Exception:                                   # pragma: no cover
    print("Run with Abaqus' Python:  abaqus python extract_denso_process.py <case>")
    raise


def column_mean(field, labels, ncomp):
    want = set(labels); acc = {}
    for v in field.values:
        if v.elementLabel in want:
            d = v.data
            row = [float(d[i]) for i in range(ncomp)] if ncomp > 1 else [float(d)]
            acc.setdefault(v.elementLabel, []).append(row)
    out = []
    for e in labels:
        rows = acc[e]
        out.append([sum(r[i] for r in rows) / float(len(rows)) for i in range(len(rows[0]))])
    return out


def nodal(field, labels, comp):
    want = set(labels); got = {}
    for v in field.values:
        if v.nodeLabel in want:
            got[v.nodeLabel] = float(v.data[comp]) if comp is not None else float(v.data)
    return [got[n] for n in labels]


def extract(case):
    ref = json.load(open("denso_proc_%s_ref.json" % case))
    out = dict(case=case, solver="abaqus")
    # ---- heat: centre-column temperature history
    odb = openOdb(ref["heat_job"] + ".odb", readOnly=True)
    region = None
    try:
        region = odb.rootAssembly.instances["PART-1-1"].nodeSets["NCOL"]
    except Exception:
        pass
    times, T = [], []
    for sname in odb.steps.keys():
        st = odb.steps[sname]
        for fr in st.frames:
            if sname != list(odb.steps.keys())[0] and fr.incrementNumber == 0:
                continue                              # step start duplicates previous end
            f = fr.fieldOutputs["NT11"]
            if region is not None:
                f = f.getSubset(region=region)
            times.append(st.totalTime + fr.frameValue)
            T.append(nodal(f, ref["col_nodes"], None))
    odb.close()
    out["times"] = times; out["T_col"] = T
    # ---- stress
    odb = openOdb(ref["proc_job"] + ".odb", readOnly=True)
    for sname, key in (("MOULD", "mould"), ("AIR", "end")):
        fr = odb.steps[sname].frames[-1]
        out["sdv1_" + key] = [r[0] for r in column_mean(fr.fieldOutputs["SDV1"], ref["col_elements"], 1)]
        if key == "end":
            s = column_mean(fr.fieldOutputs["S"], ref["col_elements"], 3)
            out["s11"] = [r[0] for r in s]; out["s22"] = [r[1] for r in s]; out["s33"] = [r[2] for r in s]
            out["u3_midx"] = nodal(fr.fieldOutputs["U"], ref["midx_nodes"], 2)
            out["u3_midy"] = nodal(fr.fieldOutputs["U"], ref["midy_nodes"], 2)
    odb.close()
    with open("denso_proc_%s_abaqus.json" % case, "w") as f:
        json.dump(out, f)
    print("wrote denso_proc_%s_abaqus.json (%d heat frames)" % (case, len(times)))


if __name__ == "__main__":
    cases = sys.argv[1:] or [l.strip() for l in open("manifest_process.txt") if l.strip()]
    for c in cases:
        extract(c)
