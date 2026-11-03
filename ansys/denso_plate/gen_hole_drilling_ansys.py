"""ansys/denso_plate/gen_hole_drilling_ansys.py -- the hole-drilling deck of
abaqus/denso_plate/gen_hole_drilling.py (Tsuda et al. 2026 inputs) converted to an Ansys Mechanical APDL
batch input, node for node and element for element, so that the hole-drilling result (Abaqus 2024:
released strain 1.54 x measured; CalculiX within 1 %) is checked in the solver DENSO uses.

Conversion (from the Abaqus .inp, which is read, not regenerated):
  C3D8I -> SOLID185, KEYOPT(2) = 3 (simplified enhanced strain); same node order;
  *ELASTIC, TYPE=ORTHOTROPIC (D1111 .. D2323) -> MP EX, EY, EZ, PRXY, PRYZ, PRXZ, GXY, GYZ, GXZ
      (major Poisson ratios, as gen_denso_plate_ansys.py; rebuilt stiffness checked here, A1);
  *EXPANSION, TYPE=ORTHO -> MP ALPX, ALPY, ALPZ (secant), TREF = initial temperature, BFUNIF = final;
  XSYM / YSYM / NFIX -> D, UX / UY / UZ = 0;
  step COOL -> SOLVE; step DRILLbb (*MODEL CHANGE, REMOVE) -> EKILL of the same elements, SOLVE;
  output per load step: total strain EPTO X, Y, Z, XY of every element of GAUGES (ETABLE, element
      average) -> <job>_ansys_strain.csv; csv_to_json() turns it into the JSON that
      abaqus/denso_plate/compare_hole_drilling.py reads (--result).
Validated here (no Ansys in this environment): A1 the MP constants rebuild every layer stiffness to
round-off; A2 the CSV -> JSON -> compare path on a CSV written from the CalculiX coarse result.
NOT validated: the APDL input itself has not been run in Ansys.

Usage:
  python3 abaqus/denso_plate/gen_hole_drilling.py --layers 77 --na 8 --nc 20 --tag coarse
  python3 ansys/denso_plate/gen_hole_drilling_ansys.py abaqus/denso_plate/jobs/denso_hole_coarse.inp
  -> ansys/denso_plate/jobs/denso_hole_coarse.dat  (+ _meta.json copied)
  after the run:  python3 ansys/denso_plate/gen_hole_drilling_ansys.py --csv2json jobs/denso_hole_coarse
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_inp(path):
    L = open(path).read().splitlines()
    nodes, elems, elsets, mats, sections, steps = [], [], {}, {}, {}, []
    T0 = T1 = None
    i = 0; cur = None
    def data(i):
        out = []
        while i < len(L) and not L[i].startswith("*"):
            if L[i].strip() and not L[i].startswith("**"):
                out.append(L[i])
            i += 1
        return out, i
    while i < len(L):
        s = L[i]; u = s.upper()
        if u.startswith("*NODE"):
            d, i = data(i + 1); nodes += [[float(v) for v in r.split(",")] for r in d]; continue
        if u.startswith("*ELEMENT,") and "TYPE=" in u:
            d, i = data(i + 1); elems += [[int(v) for v in r.split(",")] for r in d]; continue
        if u.startswith("*ELSET"):
            name = re.search(r"ELSET=([^,\s]+)", s).group(1); gen = "GENERATE" in u
            d, i = data(i + 1)
            vals = [int(v) for r in d for v in r.split(",") if v.strip()]
            elsets[name] = list(range(vals[0], vals[1] + 1, vals[2])) if gen else vals; continue
        if u.startswith("*NSET"):
            name = re.search(r"NSET=([^,\s]+)", s).group(1); d, i = data(i + 1)
            elsets["N:" + name] = [int(v) for r in d for v in r.split(",") if v.strip()]; continue
        if u.startswith("*SOLID SECTION"):
            sections[re.search(r"ELSET=([^,\s]+)", s).group(1)] = re.search(r"MATERIAL=([^,\s]+)", s).group(1); i += 1; continue
        if u.startswith("*MATERIAL"):
            cur = re.search(r"NAME=([^,\s]+)", s).group(1); mats[cur] = {}; i += 1; continue
        if u.startswith("*ELASTIC"):
            d, i = data(i + 1); mats[cur]["D"] = [float(v) for r in d for v in r.split(",") if v.strip()]; continue
        if u.startswith("*EXPANSION"):
            d, i = data(i + 1); mats[cur]["A"] = [float(v) for v in d[0].split(",")]; continue
        if u.startswith("*INITIAL CONDITIONS"):
            d, i = data(i + 1); T0 = float(d[0].split(",")[1]); continue
        if u.startswith("*STEP"):
            steps.append(dict(name=re.search(r"NAME=([^,\s]+)", s).group(1), remove=None)); i += 1; continue
        if u.startswith("*TEMPERATURE"):
            d, i = data(i + 1); T1 = float(d[0].split(",")[1]); continue
        if u.startswith("*MODEL CHANGE"):
            d, i = data(i + 1); steps[-1]["remove"] = d[0].strip().rstrip(","); continue
        i += 1
    return dict(nodes=nodes, elems=elems, elsets=elsets, mats=mats, sections=sections, steps=steps, T0=T0, T1=T1)


def mp_from_D(D):
    """Abaqus ORTHOTROPIC D1111,D1122,D2222,D1133,D2233,D3333,D1212,D1313,D2323 -> engineering constants."""
    d11, d12, d22, d13, d23, d33, g12, g13, g23 = D
    C = np.array([[d11, d12, d13], [d12, d22, d23], [d13, d23, d33]])
    S = np.linalg.inv(C)
    EX, EY, EZ = 1 / S[0, 0], 1 / S[1, 1], 1 / S[2, 2]
    return dict(EX=EX, EY=EY, EZ=EZ, PRXY=-S[0, 1] * EX, PRYZ=-S[1, 2] * EY, PRXZ=-S[0, 2] * EX, GXY=g12, GYZ=g23, GXZ=g13)


def rebuild(mp):
    S = np.array([[1 / mp["EX"], -mp["PRXY"] / mp["EX"], -mp["PRXZ"] / mp["EX"]],
                  [-mp["PRXY"] / mp["EX"], 1 / mp["EY"], -mp["PRYZ"] / mp["EY"]],
                  [-mp["PRXZ"] / mp["EX"], -mp["PRYZ"] / mp["EY"], 1 / mp["EZ"]]])
    C = np.linalg.inv(S)
    return [C[0, 0], C[0, 1], C[1, 1], C[0, 2], C[1, 2], C[2, 2], mp["GXY"], mp["GXZ"], mp["GYZ"]]


def write_apdl(m, path, job):
    out = [f"/TITLE,{job}: hole drilling after Tsuda et al. (2026), converted from the Abaqus deck",
           "/PREP7", "ET,1,SOLID185", "KEYOPT,1,2,3            ! simplified enhanced strain (~ C3D8I)"]
    for n in m["nodes"]:
        out.append(f"N,{int(n[0])},{n[1]:.9e},{n[2]:.9e},{n[3]:.9e}")
    mat_id = {name: k + 1 for k, name in enumerate(m["mats"])}
    err = 0.0
    for name, k in mat_id.items():
        mp = mp_from_D(m["mats"][name]["D"]); Db = rebuild(mp); D = m["mats"][name]["D"]
        err = max(err, max(abs(a - b) for a, b in zip(D, Db)) / max(abs(x) for x in D))
        for key, v in mp.items():
            out.append(f"MP,{key},{k},{v:.10e}")
        for key, v in zip(("ALPX", "ALPY", "ALPZ"), m["mats"][name]["A"]):
            out.append(f"MP,{key},{k},{v:.10e}")
    el_mat = {}
    for es, mname in m["sections"].items():
        for e in m["elsets"][es]:
            el_mat[e] = mat_id[mname]
    cur = None
    for e in m["elems"]:
        if el_mat[e[0]] != cur:
            cur = el_mat[e[0]]; out.append(f"MAT,{cur}")
        out.append(f"EN,{e[0]}," + ",".join(str(v) for v in e[1:9]))
    def comp(name, ids, kind):
        out.append(f"{'ESEL' if kind == 'E' else 'NSEL'},NONE")
        for s in range(0, len(ids), 1):
            out.append(f"{'ESEL' if kind == 'E' else 'NSEL'},A,{'ELEM' if kind == 'E' else 'NODE'},,{ids[s]}")
        out.append(f"CM,{name},{'ELEM' if kind == 'E' else 'NODE'}")
    for nm in ("XSYM", "YSYM", "NFIX"):
        comp(nm, m["elsets"]["N:" + nm], "N")
    comp("GAUGES", m["elsets"]["GAUGES"], "E")
    for st in m["steps"]:
        if st["remove"]:
            comp(st["remove"], m["elsets"][st["remove"]], "E")
    out += ["ALLSEL,ALL", "CMSEL,S,XSYM", "D,ALL,UX,0", "CMSEL,S,YSYM", "D,ALL,UY,0", "CMSEL,S,NFIX", "D,ALL,UZ,0",
            "ALLSEL,ALL", "FINISH", "/SOLU", "ANTYPE,STATIC", "NLGEOM,OFF", f"TREF,{m['T0']:.4f}", f"BFUNIF,TEMP,{m['T1']:.4f}",
            f"*CFOPEN,{job}_ansys_strain,csv", "*CFCLOS"]
    ng = len(m["elsets"]["GAUGES"])
    out.insert(out.index("ANTYPE,STATIC") + 1, "OUTRES,ALL,ALL")
    for st in m["steps"]:                              # load step 1 = COOL, then one per drilling band
        if st["remove"]:
            out += [f"CMSEL,S,{st['remove']}", "EKILL,ALL", "ALLSEL,ALL"]
        out += ["SOLVE"]
    out += ["FINISH", "/POST1"]
    for k in range(len(m["steps"])):
        out += [f"SET,{k + 1},LAST", "CMSEL,S,GAUGES",
                "ETABLE,TX,EPTO,X", "ETABLE,TY,EPTO,Y", "ETABLE,TZ,EPTO,Z", "ETABLE,TXY,EPTO,XY",
                f"*CFOPEN,{job}_ansys_strain,csv,,APPEND", "*GET,EN,ELEM,0,NUM,MIN",
                f"*DO,I,1,{ng}", "*GET,A1,ELEM,EN,ETAB,TX", "*GET,A2,ELEM,EN,ETAB,TY", "*GET,A3,ELEM,EN,ETAB,TZ",
                "*GET,A4,ELEM,EN,ETAB,TXY", f"*VWRITE,{k},EN,A1,A2,A3,A4", "(F4.0,',',F10.0,4(',',E24.15))",
                "*GET,EN,ELEM,EN,NXTH", "*ENDDO", "*CFCLOS", "ALLSEL,ALL"]
    out += ["FINISH"]
    open(path, "w").write("\n".join(out) + "\n")
    return err


def csv_to_json(job):
    steps = {}
    for r in csv.reader(open(job + "_ansys_strain.csv")):
        if not r:
            continue
        k, e = int(float(r[0])), str(int(float(r[1])))
        ex, ey, ez, gxy = (float(v) for v in r[2:6])
        steps.setdefault(k, {})[e] = [ex, ey, ez, gxy, 0.0, 0.0]
    res = dict(solver="ansys", shear="engineering", steps=[dict(name=f"LS{k + 1}", E=steps[k]) for k in sorted(steps)])
    json.dump(res, open(job + "_ansys.json", "w"))
    return job + "_ansys.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("inp", nargs="?")
    ap.add_argument("--csv2json")
    ap.add_argument("--out", default=os.path.join(HERE, "jobs"))
    a = ap.parse_args()
    if a.csv2json:
        print("wrote", csv_to_json(a.csv2json)); return
    m = parse_inp(a.inp)
    job = os.path.splitext(os.path.basename(a.inp))[0]
    os.makedirs(a.out, exist_ok=True)
    err = write_apdl(m, os.path.join(a.out, job + ".dat"), job)
    shutil.copy(os.path.splitext(a.inp)[0] + "_meta.json", os.path.join(a.out, job + "_meta.json"))
    print(f"{job}: {len(m['nodes'])} nodes, {len(m['elems'])} SOLID185, {len(m['mats'])} materials, "
          f"{len(m['steps'])} load steps (TREF {m['T0']}, BFUNIF {m['T1']}); [A1] stiffness rebuilt from MP to {err:.1e} (rel)")
    print(f"wrote {os.path.join(a.out, job + '.dat')}")


if __name__ == "__main__":
    main()
