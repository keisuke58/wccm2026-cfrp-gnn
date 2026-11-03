"""validation/greif3d_diag_incavity.py -- why does the RESCUE correction fail on some 3-D parts? A diagnostic that
separates the FE (stiffness, mesh, hold) from the eigenstrain estimate.

The data carry Moldflow's in-cavity residual stress per element (Pa, 6 components, order assumed 11,22,33,12,23,13
as for the orientation). Moldflow's warp is the release of this stress. Here it is put into the SAME FE as the
prior (validation/greif3d_rescue.py: clusters, Mori-Tanaka stiffness, C3D10, 3-2-1 hold) as an initial stress
(CalculiX *INITIAL CONDITIONS, TYPE=STRESS, the same value at the 4 integration points) and released:
  D1 cos(u_release, u_data) high  -> the FE and the stress order are fine; the prior's failure lies in its
                                     eigenstrain (uniform dT, no pressure/cooling history);
  D1 low                          -> the FE side (stiffness, which differs from Moldflow's) or the assumed order.
  D2 the same with the off-diagonal order permuted (5 other orders): tests the assumed component order.
cos and the relative error after the best single factor are reported (rigid motion removed throughout).
  D0 (check, tiny plate): an initial stress -C:alpha dT reproduces the thermal-load warp to 4.8e-8.
Result (2026-11-04): 00010768 cos 0.999, error 0.035 (FE reproduces Moldflow); 00010077 cos 0.92, error 0.39 for
all 6 orders (0.36-0.40): the order of the off-diagonals is NOT the cause, and about two thirds of that part's
failure (0.57 with the prior) remains even with Moldflow's own stress -> stiffness (ours vs Moldflow's) or
effects outside the in-cavity stress; the rest is the prior's eigenstrain.
Honest scope: our stiffness, not Moldflow's (G4: about 0.57 of the data's modulus), so a common factor is
expected; one value per element at all integration points. Moldflow data, not measurements.
Run:  python3 validation/greif3d_diag_incavity.py DATA WORK id [id ...] [--perm]
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import greif3d_rescue as g  # noqa: E402


def load_stress(d):
    import meshio
    vtu = sorted(f for f in os.listdir(d) if f.endswith("_sim.vtu"))[0]
    m = meshio.read(os.path.join(d, vtu))
    k = [k for k in m.cell_data if k.startswith("In-cavity")][0]
    return np.concatenate([np.asarray(a, float) for a in m.cell_data[k]]) / 1e6     # MPa


def release(X, T, lab, Cs, sig, work, tag):
    """sig (n_el, 6) in order 11,22,33,12,13,23 [MPa]."""
    Xs, Ts = g.to_c3d10(X, T)
    hold = g.hold_321(Xs)
    job = os.path.join(work, tag)
    zero = np.zeros((len(Cs), 3, 3))
    g.write_deck(job + ".inp", Xs, Ts, lab, Cs, zero, hold)
    txt = open(job + ".inp").read()
    ini = ["*INITIAL CONDITIONS, TYPE=STRESS"]
    for e in range(len(T)):
        v = ",".join(f"{x:.6g}" for x in sig[e])
        for ip in (1, 2, 3, 4):
            ini.append(f"{e + 1},{ip},{v}")
    txt = txt.replace("*INITIAL CONDITIONS, TYPE=TEMPERATURE", "\n".join(ini) + "\n*INITIAL CONDITIONS, TYPE=TEMPERATURE")
    txt = txt.replace(f"*TEMPERATURE\nNALL, {g.DT}\n", "*TEMPERATURE\nNALL, 0.\n")   # no temperature change (ccx needs a final one)
    open(job + ".inp", "w").write(txt)
    return g.run_ccx(job, len(Xs), 4)[: len(X)]


def score(X, u, data):
    p, d = g.remove_rigid(X, u).reshape(-1), g.remove_rigid(X, data).reshape(-1)
    c = float(p @ d / np.linalg.norm(p) / np.linalg.norm(d))
    a = float(p @ d / (p @ p))
    return c, g.rel_err(X, a * u, data), a


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    perm = "--perm" in sys.argv
    DATA, WORK, ids = args[0], args[1], args[2:]
    os.makedirs(WORK, exist_ok=True)
    out = []
    for gid in ids:
        d = os.path.join(DATA, gid)
        _, X, T, data, An, _ = g.load_geometry(d)
        T, _ = g.fix_orientation(X, T)
        A = g.clean_orientation(An[T].mean(1))
        lab, cen = g.kmeans(g.vec6(A), 32)
        Cs, _ = g.cluster_props(g.clean_orientation(g.mat6(cen)))
        s = load_stress(d)                                      # file order 11,22,33,12,23,13 (assumed)
        orders = list(itertools.permutations([3, 4, 5])) if perm else [(3, 4, 5)]
        for o in orders:
            sig = np.column_stack([s[:, 0], s[:, 1], s[:, 2], s[:, o[0]], s[:, o[2]], s[:, o[1]]])  # -> 11,22,33,12,13,23
            u = release(X, T, lab, Cs, sig, WORK, f"{gid}_ic{''.join(map(str, o))}")
            c, e, a = score(X, u, data)
            r = dict(id=gid, order=["11", "22", "33"] + [{3: "a", 4: "b", 5: "c"}[k] for k in o], cos=c, err_best_factor=e, factor=a)
            out.append(r)
            print(f"[D{'2' if o != (3, 4, 5) else '1'}] {gid} off-diagonal file columns {o} as (12, 23, 13): cos {c:+.3f}, "
                  f"error after best factor {e:.3f} (factor {a:.2f})", flush=True)
    dst = os.path.join(ROOT, "results", "greif3d_rescue", "diag_incavity.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    json.dump(out, open(dst, "w"), indent=1); print("wrote", dst)


if __name__ == "__main__":
    main()
