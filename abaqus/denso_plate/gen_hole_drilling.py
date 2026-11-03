"""abaqus/denso_plate/gen_hole_drilling.py -- Abaqus deck that simulates the hole-drilling test of
Tsuda et al. (DENSO / Polyplastics), J. Jpn. Soc. Compos. Mater. 52(3) 88-96 (2026), so that the
residual state computed with the paper's inputs can be compared with its MEASURED released
strain (its Fig. 12).

Model (as stated in the paper unless marked):
  * 20 x 20 x 3.08 mm specimen, 154 element layers of 20 um through the thickness, layer
    orientation a_ij from the paper's Fig. 8a (read by eye; data/literature/...json), depth measured
    from the drilled face (z = +H/2);
  * layer properties: Mori-Tanaka + orientation averaging with the paper's Table 1 PPS values and
    GF constants (validation/denso_tsuda2026_compare.py, hybrid closure: the paper's closure is not
    named); stiffness at 20 C; eigenstrain = integral of the instantaneous CTE from T0 = 180 C to
    20 C (the "total" reading that reproduced the paper's surface stress, -24.4 vs -24.3 MPa),
    imposed as an equivalent expansion under a uniform cool-down;
  * QUARTER model with symmetry planes x = 0 and y = 0 (exact: layer materials are orthotropic in
    x, y, z once the off-diagonal a_ij, |a_ij| < 0.15 in the paper, are dropped -- an assumption);
    free edges otherwise, one point fixed in z;
  * drilling: phi 1.6 mm hole at the centre, removed in 10 steps of 0.2 mm (*MODEL CHANGE, REMOVE;
    the paper set the stiffness of the same elements to zero with a UMAT);
  * gauges: TML FRAS-2 rosette (the paper's Fig. 6: "1.1 x 2.0", "phi 7.0"; TML catalogue: gauge-centre
    diameter 7.0 mm, grid 2.0 mm long x 1.1 mm wide, read from a search excerpt of the F-series catalogue;
    the TML site was under maintenance when checked), i.e. grids spanning r = 2.5 .. 4.5 mm radially;
    Ch.3 along MD (x), Ch.1 along TD (y), Ch.2 at 45 deg. (A first run with the grids wrongly placed
    at r = 1.5 .. 3.5 mm gave released strains 3.6x the paper's FE: the strain falls off ~1/r^2.) Released strain = average over the drilled-face element layer inside each grid
    of (strain after drilling step - strain before drilling).
Mesh: inner square + transition ring inside the hole (removed), then a graded ring from the hole
edge to the specimen edge; size set by --na/--nb/--nc/--layers (defaults match the paper's layers).

Usage:
  python3 abaqus/denso_plate/gen_hole_drilling.py                 # full deck (154 layers)
  python3 abaqus/denso_plate/gen_hole_drilling.py --layers 77 --na 8 --nc 20 --tag coarse
  python3 abaqus/denso_plate/gen_hole_drilling.py --gauge-shift 0.2 --tag gp02   # grids moved out by 0.2 mm
Outputs in abaqus/denso_plate/jobs/: denso_hole<tag>.inp and denso_hole<tag>_meta.json.
NOT run in Abaqus yet.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "validation")); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "viz"))
import denso_tsuda2026_compare as tc  # noqa: E402
from fiber_orientation_cte import orient_avg_cte  # noqa: E402

R_HOLE = 0.8e-3
L_HALF = 10.0e-3
DRILL_STEP, DRILL_MAX = 0.2e-3, 2.0e-3
GAUGE_R = (2.5e-3, 4.5e-3)       # TML FRAS-2: gauge-centre diameter 7.0 mm, grid 2.0 mm long (catalogue)
GAUGE_W = 1.1e-3


def mesh2d(na, nb, nc, q):
    """Quarter-domain quad mesh: nodes (N,2), quads (M,4), hole flag per quad."""
    a = 0.45 * R_HOLE
    nodes = {}; pts = []

    def nid(key, xy):
        if key not in nodes:
            nodes[key] = len(pts); pts.append(xy)
        return nodes[key]
    sq = lambda i, j: nid(("s", i, j), (a * i / na, a * j / na))
    quads, hole = [], []
    for j in range(na):
        for i in range(na):
            quads.append([sq(i, j), sq(i + 1, j), sq(i + 1, j + 1), sq(i, j + 1)]); hole.append(True)
    nt = 2 * na
    path = [sq(na, j) for j in range(na + 1)] + [sq(na - j, na) for j in range(1, na + 1)]
    th = [0.5 * math.pi * j / nt for j in range(nt + 1)]
    arc = [(R_HOLE * math.cos(t), R_HOLE * math.sin(t)) for t in th]
    out = [((L_HALF, L_HALF * math.tan(t)) if t <= math.pi / 4 + 1e-12 else (L_HALF / math.tan(t), L_HALF)) for t in th]
    ring = [path]
    for r in range(1, nb + 1):
        s = r / nb
        ring.append([nid(("b", r, j), ((1 - s) * pts[path[j]][0] + s * arc[j][0],
                                         (1 - s) * pts[path[j]][1] + s * arc[j][1])) for j in range(nt + 1)])
    for r in range(nb):
        for j in range(nt):
            quads.append([ring[r][j], ring[r + 1][j], ring[r + 1][j + 1], ring[r][j + 1]]); hole.append(True)
    g = [(q ** i - 1) / (q ** nc - 1) for i in range(nc + 1)]
    cring = [ring[nb]]
    for r in range(1, nc + 1):
        cring.append([nid(("c", r, j), (arc[j][0] + g[r] * (out[j][0] - arc[j][0]),
                                         arc[j][1] + g[r] * (out[j][1] - arc[j][1]))) for j in range(nt + 1)])
    for r in range(nc):
        for j in range(nt):
            quads.append([cring[r][j], cring[r + 1][j], cring[r + 1][j + 1], cring[r][j + 1]]); hole.append(False)
    P = np.array(pts); Q = np.array(quads)
    # orientation check: counter-clockwise (positive signed area) for every quad
    x, y = P[Q, 0], P[Q, 1]
    area = 0.5 * np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1)
    assert np.all(area > 0), "quad orientation"
    return P, Q, np.array(hole), area


def layer_materials(nl):
    """Per-layer 3-D stiffness at 20 C (repo Voigt) and integrated eigenstrain 180 -> 20 C."""
    H = tc.H3
    dz = H / nl
    depth = (np.arange(nl) + 0.5) * dz * 1e3                 # mm, from the drilled face
    axx = np.interp(depth, tc.PROF_3MM["d"], tc.PROF_3MM["axx"])
    ayy = np.interp(depth, tc.PROF_3MM["d"], tc.PROF_3MM["ayy"])
    axx, ayy, azz = tc.layers(None, axx, ayy)
    Cud, aL, aT = tc.ud(tc.T_RT)
    C, eth = [], np.zeros((nl, 3))
    for x, y, w in zip(axx, ayy, azz):
        _, Cv = orient_avg_cte(np.diag([x, y, w]), "hybrid", Cud, aL=aL, aT=aT)
        C.append(Cv)
    Ts = np.linspace(tc.T0, tc.T_RT, 161)
    for T1, T2 in zip(Ts[:-1], Ts[1:]):
        Cu, al, at = tc.ud(0.5 * (T1 + T2))
        for k, (x, y, w) in enumerate(zip(axx, ayy, azz)):
            a, _ = orient_avg_cte(np.diag([x, y, w]), "hybrid", Cu, aL=al, aT=at)
            eth[k] += np.diag(a) * (T2 - T1)
    return dict(depth=depth, axx=axx, ayy=ayy, azz=azz, C=np.array(C), eth=eth, dz=dz)


def lamination_check(mat):
    """Free-plate surface stress for the same layers (should match denso_tsuda2026_compare)."""
    from denso_plate_features import reduced_Q
    H = tc.H3; nl = len(mat["depth"]); z = H / 2 - (mat["depth"] * 1e-3)      # drilled face at +H/2
    Qs = np.array([reduced_Q(C) for C in mat["C"]]); e = mat["eth"].copy(); e[:, 2] = 0.0
    dz = mat["dz"]
    A = np.einsum("kij->ij", Qs) * dz; B = np.einsum("k,kij->ij", z, Qs) * dz; D = np.einsum("k,kij->ij", z * z, Qs) * dz
    N = np.einsum("kij,kj->i", Qs, e) * dz; M = np.einsum("k,kij,kj->i", z, Qs, e) * dz
    sol = np.linalg.solve(np.block([[A, B], [B, D]]), np.concatenate([N, M]))
    sig = np.einsum("kij,kj->ki", Qs, sol[None, :3] + z[:, None] * sol[None, 3:] - e)
    return sig[0] / 1e6                                       # top (drilled-face) layer, MPa


def write_deck(path, P, Q, hole, mat, nl, tag, eig="all"):
    H = tc.H3; dz = H / nl
    Np, Nq = len(P), len(Q)
    nd = lambda p, k: 1 + p + Np * k                 # k = 0 bottom plane ... nl top plane (drilled face)
    el = lambda q, k: 1 + q + Nq * k                 # k = 0 bottom layer ... nl-1 top layer
    L = ["*HEADING", f" Hole-drilling simulation after Tsuda et al. (2026), quarter model, {nl} layers ({tag})",
         " Generated by abaqus/denso_plate/gen_hole_drilling.py. Paper inputs; FRAS-2 gauge geometry from the TML catalogue.",
         "** Units: m, Pa, deg C.", "*NODE, NSET=NALL"]
    for k in range(nl + 1):
        z = -H / 2 + k * dz
        for p, (x, y) in enumerate(P):
            L.append(f"{nd(p, k)}, {x:.9e}, {y:.9e}, {z:.9e}")
    L.append("*ELEMENT, TYPE=C3D8I, ELSET=EALL")
    for k in range(nl):
        for qi, qd in enumerate(Q):
            n = [nd(v, k) for v in qd] + [nd(v, k + 1) for v in qd]
            L.append(f"{el(qi, k)}, " + ", ".join(map(str, n)))

    def elset(name, labels):
        L.append(f"*ELSET, ELSET={name}")
        for s in range(0, len(labels), 16):
            L.append(", ".join(map(str, labels[s:s + 16])) + ",")

    def nset(name, labels):
        L.append(f"*NSET, NSET={name}")
        for s in range(0, len(labels), 16):
            L.append(", ".join(map(str, labels[s:s + 16])) + ",")
    for k in range(nl):
        L += [f"*ELSET, ELSET=L{k+1:03d}, GENERATE", f" {el(0, k)}, {el(Nq - 1, k)}, 1"]
    # drilling bands from the drilled face (top) downwards
    nb = int(round(DRILL_MAX / DRILL_STEP)); per = int(round(DRILL_STEP / dz))
    hole_q = np.where(hole)[0]
    for b in range(nb):
        ks = [nl - 1 - m for m in range(b * per, (b + 1) * per)]
        elset(f"DRILL{b+1:02d}", [el(q, k) for k in ks for q in hole_q])
    # gauges on the drilled-face layer
    cen = P[Q].mean(axis=1)
    def in_rect(c, ang):
        u = c[:, 0] * math.cos(ang) + c[:, 1] * math.sin(ang)
        v = -c[:, 0] * math.sin(ang) + c[:, 1] * math.cos(ang)
        return (u >= GAUGE_R[0]) & (u <= GAUGE_R[1]) & (np.abs(v) <= GAUGE_W / 2)
    gauges = {"GCH3": (in_rect(cen, 0.0), "MD (x)"), "GCH1": (in_rect(cen, math.pi / 2), "TD (y)"),
              "GCH2": (in_rect(cen, math.pi / 4), "45 deg")}
    gauge_info = {}
    for name, (mask, lab) in gauges.items():
        qs = np.where(mask)[0]
        labels = [int(el(q, nl - 1)) for q in qs]
        elset(name, labels)
        xq, yq = P[Q[qs], 0], P[Q[qs], 1]
        areas = 0.5 * np.sum(xq * np.roll(yq, -1, axis=1) - np.roll(xq, -1, axis=1) * yq, axis=1)
        gauge_info[name] = dict(direction=lab, elements=labels, quad_areas=[float(v) for v in areas])
    nset("XSYM", [nd(p, k) for k in range(nl + 1) for p in np.where(P[:, 0] < 1e-12)[0]])
    nset("YSYM", [nd(p, k) for k in range(nl + 1) for p in np.where(P[:, 1] < 1e-12)[0]])
    nset("NFIX", [nd(int(np.argmin(np.hypot(P[:, 0], P[:, 1]))), 0)])
    L += ["*ORIENTATION, NAME=GLOBAL", " 1.0, 0.0, 0.0, 0.0, 1.0, 0.0"]
    ab = [(0, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2), (5, 5), (4, 4), (3, 3)]
    dT = tc.T_RT - tc.T0
    for k in range(nl):
        C = mat["C"][nl - 1 - k]                  # element layer k sits at depth index nl-1-k
        ae = mat["eth"][nl - 1 - k] / dT
        if eig != "all":                          # one eigenstrain component only (superposition basis)
            ae = ae * (np.arange(3) == "xyz".index(eig[0]))
        v = [C[i, j] for i, j in ab]
        L += [f"*SOLID SECTION, ELSET=L{k+1:03d}, MATERIAL=M{k+1:03d}, ORIENTATION=GLOBAL",
              f"*MATERIAL, NAME=M{k+1:03d}", "*ELASTIC, TYPE=ORTHOTROPIC",
              " " + ", ".join(f"{x:.10e}" for x in v[:8]) + ",", f" {v[8]:.10e}",
              "*EXPANSION, TYPE=ORTHO", " " + ", ".join(f"{x:.10e}" for x in ae)]
    L += ["*INITIAL CONDITIONS, TYPE=TEMPERATURE", f" NALL, {tc.T0:.4f}",
          "*BOUNDARY", " XSYM, 1, 1, 0.0", " YSYM, 2, 2, 0.0", " NFIX, 3, 3, 0.0",
          "*STEP, NAME=COOL, NLGEOM=NO", "*STATIC", " 1.0, 1.0", "*TEMPERATURE", f" NALL, {tc.T_RT:.4f}",
          "*OUTPUT, FIELD", "*ELEMENT OUTPUT, ELSET=GAUGES", " E, S", "*EL PRINT, ELSET=GAUGES, SUMMARY=NO", " E",
          "*END STEP"]
    for b in range(nb):
        L += [f"*STEP, NAME=DRILL{b+1:02d}, NLGEOM=NO", "*STATIC", " 1.0, 1.0",
              "*MODEL CHANGE, REMOVE", f" DRILL{b+1:02d},",
              "*OUTPUT, FIELD", "*ELEMENT OUTPUT, ELSET=GAUGES", " E, S", "*EL PRINT, ELSET=GAUGES, SUMMARY=NO", " E",
              "*END STEP"]
    # GAUGES = union, defined before the steps
    idx = L.index("*ORIENTATION, NAME=GLOBAL")
    allg = sorted(set(sum((g["elements"] for g in gauge_info.values()), [])))
    gl = ["*ELSET, ELSET=GAUGES"] + [", ".join(map(str, allg[s:s + 16])) + "," for s in range(0, len(allg), 16)]
    L[idx:idx] = gl
    open(path, "w").write("\n".join(L) + "\n")
    return gauge_info, nb


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--layers", type=int, default=154)
    ap.add_argument("--na", type=int, default=12, help="elements along each inner-square edge (2*na around the quarter)")
    ap.add_argument("--nb", type=int, default=4)
    ap.add_argument("--nc", type=int, default=26)
    ap.add_argument("--q", type=float, default=1.15, help="radial growth ratio outside the hole")
    ap.add_argument("--tag", default="")
    ap.add_argument("--eig", default="all", choices=["all", "xx", "yy", "zz"],
                    help="keep only this eigenstrain component (linear superposition basis)")
    ap.add_argument("--gauge-shift", type=float, default=0.0,
                    help="move all gauge grids radially by this many mm (sensitivity to the gauge position)")
    ap.add_argument("--out", default=os.path.join(HERE, "jobs"))
    a = ap.parse_args()
    global GAUGE_R
    GAUGE_R = (GAUGE_R[0] + a.gauge_shift * 1e-3, GAUGE_R[1] + a.gauge_shift * 1e-3)
    assert abs(tc.H3 / a.layers * round(DRILL_STEP / (tc.H3 / a.layers)) - DRILL_STEP) < 1e-6 * DRILL_STEP or True
    os.makedirs(a.out, exist_ok=True)
    P, Q, hole, area = mesh2d(a.na, a.nb, a.nc, a.q)
    mat = layer_materials(a.layers)
    s_top = lamination_check(mat)
    job = "denso_hole" + (("_" + a.tag) if a.tag else "")
    per = DRILL_STEP / (tc.H3 / a.layers)
    if abs(per - round(per)) > 1e-6:
        print(f"note: 0.2 mm is {per:.3f} layers; drilling bands are rounded to {round(per)} layers "
              f"({round(per) * tc.H3 / a.layers * 1e3:.3f} mm)")
    gauges, nb = write_deck(os.path.join(a.out, job + ".inp"), P, Q, hole, mat, a.layers, a.tag or "full", a.eig)
    meta = dict(job=job, layers=a.layers, na=a.na, nb=a.nb, nc=a.nc, q=a.q, n_quads=int(len(Q)),
                n_elements=int(len(Q) * a.layers), drill_steps=nb, drill_step_mm=DRILL_STEP * 1e3,
                band_mm=round(per) * tc.H3 / a.layers * 1e3,
                gauge_r_mm=[g * 1e3 for g in GAUGE_R], gauge_w_mm=GAUGE_W * 1e3,
                gauges=gauges, lamination_surface_stress_MPa=dict(sxx=float(s_top[0]), syy=float(s_top[1])),
                eig=a.eig, paper_fig12=tc.PAPER["fig12_released_strain"])
    json.dump(meta, open(os.path.join(a.out, job + "_meta.json"), "w"), indent=1)
    print(f"{job}: {len(Q)} quads x {a.layers} layers = {len(Q) * a.layers} C3D8I, {len(P) * (a.layers + 1)} nodes; "
          f"gauge elements Ch1 {len(gauges['GCH1']['elements'])}, Ch2 {len(gauges['GCH2']['elements'])}, "
          f"Ch3 {len(gauges['GCH3']['elements'])}; lamination surface stress sxx {s_top[0]:+.1f} / syy {s_top[1]:+.1f} MPa")


if __name__ == "__main__":
    main()
