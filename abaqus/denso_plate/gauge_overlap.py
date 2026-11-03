"""gauge_overlap.py -- released strain of the hole-drilling simulation averaged over a gauge grid by
EXACT element/grid overlap areas, for grids moved radially by any shift. Needs a deck written by
gen_hole_drilling.py with the ZONE element set (drilled-face elements within 1.0 mm of each grid),
so one Abaqus/CalculiX run serves every shift. Plain python3.

    cd abaqus/denso_plate/jobs
    python3 ../gauge_overlap.py denso_hole                                   # <job>_abaqus.json
    python3 ../gauge_overlap.py denso_hole_coarse --calculix denso_hole_coarse_ccx.dat

Grid k (Ch.3 at 0 deg, Ch.1 at 90 deg, Ch.2 at 45 deg) after shift d: the rectangle
r in [2.5 + d, 4.5 + d] mm along its axis, |t| <= 0.55 mm across it (TML FRAS-2, see gen_hole_drilling.py).
Released strain of the grid = sum_e A_e * (strain_e(step) - strain_e(cool)) / sum_e A_e, with
A_e the overlap area of element e (a quad of the drilled-face layer) with the rectangle, by
Sutherland-Hodgman clipping. Element strain = mean over its integration points (as written by
extract_hole_drilling.py); the strain field inside an element is not reconstructed.

Check (printed): for every shift, sum_e A_e = the part of the grid inside the quarter model: 1.1 mm^2 for
Ch.1/Ch.3 (the grids straddle a symmetry plane, so half lies in the model and, by symmetry, has the same
mean) and 2.2 mm^2 for Ch.2 (on the diagonal, whole). A smaller sum means the grid leaves the ZONE set:
the shift is then rejected.
Honest scope: piecewise-constant strain per element; the remaining error is the strain variation
inside an element, which shrinks with the mesh. Shifts are bounded by the 1.0 mm zone padding (+-0.4 mm checked).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from compare_hole_drilling import read_calculix  # noqa: E402

R0, R1, W = 2.5e-3, 4.5e-3, 1.1e-3
AXES = {"GCH3": 0.0, "GCH1": math.pi / 2, "GCH2": math.pi / 4}


def clip(poly, a, b, c):
    """Keep the part of polygon poly with a*x + b*y <= c (Sutherland-Hodgman, one edge)."""
    out = []
    for i in range(len(poly)):
        p, q = poly[i], poly[(i + 1) % len(poly)]
        fp, fq = a * p[0] + b * p[1] - c, a * q[0] + b * q[1] - c
        if fp <= 0:
            out.append(p)
        if fp * fq < 0:
            t = fp / (fp - fq)
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return out


def area(poly):
    if len(poly) < 3:
        return 0.0
    x = np.array([p[0] for p in poly]); y = np.array([p[1] for p in poly])
    return abs(0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


def overlap(poly, ang, d):
    c, s = math.cos(ang), math.sin(ang)          # u = c x + s y (along the grid), t = -s x + c y
    P = [tuple(p) for p in poly]
    for a, b, k in ((-c, -s, -(R0 + d)), (c, s, R1 + d), (s, -c, W / 2), (-s, c, W / 2)):
        P = clip(P, a, b, k)
        if not P:
            return 0.0
    return area(P)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("job")
    ap.add_argument("--calculix")
    ap.add_argument("--shifts", default="-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4", help="radial shifts [mm]")
    a = ap.parse_args()
    meta = json.load(open(a.job + "_meta.json"))
    if "ZONE" not in meta["gauges"]:
        sys.exit("deck has no ZONE set: regenerate it with the current gen_hole_drilling.py")
    res = read_calculix(a.calculix) if a.calculix else json.load(open(a.job + "_abaqus.json"))
    gamma = (lambda e: e[3]) if res["shear"] == "engineering" else (lambda e: 2 * e[3])
    comp = {"GCH3": lambda e: e[0], "GCH1": lambda e: e[1], "GCH2": lambda e: 0.5 * (e[0] + e[1]) + 0.5 * gamma(e)}
    Z = meta["gauges"]["ZONE"]; els = [str(e) for e in Z["elements"]]; polys = Z["polygons"]
    steps = res["steps"]; E0 = steps[0]["E"]
    f12 = meta["paper_fig12"]
    shifts = [float(v) for v in a.shifts.split(",")]
    out = dict(job=a.job, solver=res["solver"], shifts_mm=shifts, ch={})
    print(f"{a.job} [{res['solver']}]: grid averaged by element overlap areas; peak |released strain| [micro-strain]")
    print(" shift   Ch3 peak (x meas)   Ch1 peak (x meas)   grid-area check (Ch3, Ch1, Ch2) [mm^2]")
    for d in shifts:
        row = {}
        for g, ang in AXES.items():
            A = np.array([overlap(p, ang, d * 1e-3) for p in polys])
            tot = A.sum()
            rel = [0.0] + [float(sum(A[i] * (comp[g](st["E"][e]) - comp[g](E0[e])) for i, e in enumerate(els) if A[i] > 0)
                                 / tot) * 1e6 for st in steps[1:]] if tot > 0 else []
            row[g] = dict(area_mm2=tot * 1e6, released=rel)
        ok = all(abs(row[g]["area_mm2"] - (2.2 if g == "GCH2" else 1.1)) < 1e-6 for g in AXES)
        pk = {g: max(abs(v) for v in row[g]["released"]) for g in ("GCH3", "GCH1")}
        ex = {g: max(abs(v) for v in f12[f"Ch{g[-1]}_exp"]) for g in ("GCH3", "GCH1")}
        print(f" {d:+5.2f}   {pk['GCH3']:5.0f} (x{pk['GCH3']/ex['GCH3']:.2f})      {pk['GCH1']:5.0f} (x{pk['GCH1']/ex['GCH1']:.2f})"
              f"      {row['GCH3']['area_mm2']:.6f}, {row['GCH1']['area_mm2']:.6f}, {row['GCH2']['area_mm2']:.6f}"
              + ("" if ok else "   <- grid leaves the zone: REJECTED"))
        out["ch"][f"{d:+.2f}"] = dict(ok=ok, peak=pk, ratio_to_exp={g: pk[g] / ex[g] for g in pk}, grids=row)
    json.dump(out, open(a.job + "_overlap.json", "w"), indent=1)
    print(f"wrote {a.job}_overlap.json")


if __name__ == "__main__":
    main()
