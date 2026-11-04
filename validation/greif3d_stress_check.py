"""validation/greif3d_stress_check.py -- does a factor fitted to a FEW WARP POINTS also give the RESIDUAL STRESS level
that a strength check needs? 3-D parts of Greif & Meyer (Moldflow data, not measurements).

Reference stress: the data's in-cavity residual stress released in our FE (as validation/greif3d_diag_incavity.py):
sigma_ref = sigma_0 + C eps(u), i.e. the stress left in the free part after warping.
Prior stress: our thermal load cases (isotropic / deviatoric CTE, validation/greif3d_rescue.py), stresses of the two
runs combined with the factors fitted on 10 surface points of the warp (the same fit as RESCUE, eq. 2).
Compared per element (integration-point average): the 95th percentile of the von Mises stress and of the maximum
principal stress, prior (a = 1) and corrected, against the reference; and the element-wise correlation.
Honest scope: the reference is Moldflow's stress through OUR stiffness (not Moldflow's warp stresses); 10 points,
one draw; two parts; linear elasticity.
Run:  python3 validation/greif3d_stress_check.py DATA WORK id [id ...]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import greif3d_diag_incavity as dg  # noqa: E402
import greif3d_rescue as g  # noqa: E402


def with_stress_output(path, n_sets):
    txt = open(path).read()
    names = [f"E{j}" for j in range(n_sets) if f"ELSET=E{j}\n" in txt]
    sets = "\n".join(",".join(names[i:i + 8]) for i in range(0, len(names), 8))     # ccx: at most 16 entries per line
    txt = txt.replace("*STEP\n", f"*ELSET, ELSET=EALL\n{sets}\n*STEP\n", 1)
    txt = txt.replace("*NODE PRINT, NSET=NALL\nU\n", "*NODE PRINT, NSET=NALL\nU\n*EL PRINT, ELSET=EALL\nS\n")
    open(path, "w").write(txt)


def run_with_stress(job, n_nodes, n_el):
    env = dict(os.environ, OMP_NUM_THREADS="4")
    import subprocess
    d, j = os.path.split(job)
    r = subprocess.run([g.CCX, "-i", j], cwd=d or ".", env=env, capture_output=True, text=True)
    if not os.path.exists(job + ".dat") or "*ERROR" in r.stdout:
        raise RuntimeError(r.stdout[-1500:])
    U = np.zeros((n_nodes, 3)); S = np.zeros((n_el, 6)); cnt = np.zeros(n_el); mode = None
    for line in open(job + ".dat"):
        if "displacements" in line:
            mode = "u"; continue
        if "stresses" in line:
            mode = "s"; continue
        p = line.split()
        if mode == "u" and len(p) == 4:
            U[int(p[0]) - 1] = [float(v) for v in p[1:]]
        elif mode == "s" and len(p) == 8:
            e = int(p[0]) - 1; S[e] += [float(v) for v in p[2:]]; cnt[e] += 1
    for ext in (".dat", ".frd", ".sta", ".cvg", ".12d", ".inp"):
        if os.path.exists(job + ext):
            os.remove(job + ext)
    return U, S / np.maximum(cnt, 1)[:, None]


def vm(S):
    sx, sy, sz, txy, txz, tyz = S.T
    return np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3 * (txy ** 2 + txz ** 2 + tyz ** 2))


def smax(S):
    M = np.zeros((len(S), 3, 3))
    M[:, 0, 0], M[:, 1, 1], M[:, 2, 2] = S[:, 0], S[:, 1], S[:, 2]
    M[:, 0, 1] = M[:, 1, 0] = S[:, 3]; M[:, 0, 2] = M[:, 2, 0] = S[:, 4]; M[:, 1, 2] = M[:, 2, 1] = S[:, 5]
    return np.linalg.eigvalsh(M)[:, -1]


def main():
    DATA, WORK, ids = sys.argv[1], sys.argv[2], sys.argv[3:]
    os.makedirs(WORK, exist_ok=True)
    rng = np.random.default_rng(3)
    rows = []
    for gid in ids:
        d = os.path.join(DATA, gid)
        _, X, T, data, An, _ = g.load_geometry(d)
        T, _ = g.fix_orientation(X, T)
        A = g.clean_orientation(An[T].mean(1))
        lab, cen = g.kmeans(g.vec6(A), 32)
        Cs, als = g.cluster_props(g.clean_orientation(g.mat6(cen)))
        Xs, Ts = g.to_c3d10(X, T)
        hold = g.hold_321(Xs)
        tr = np.trace(als, axis1=1, axis2=2) / 3; aiso = tr[:, None, None] * np.eye(3)[None]
        Ub, Sb = [], []
        for nm, al in (("iso", aiso), ("dev", als - aiso)):
            job = os.path.join(WORK, f"{gid}_s{nm}")
            g.write_deck(job + ".inp", Xs, Ts, lab, Cs, al, hold); with_stress_output(job + ".inp", len(Cs))
            U, S = run_with_stress(job, len(Xs), len(T)); Ub.append(U[: len(X)]); Sb.append(S)
        # reference: in-cavity stress released
        s0 = dg.load_stress(d)
        sig = np.column_stack([s0[:, 0], s0[:, 1], s0[:, 2], s0[:, 3], s0[:, 5], s0[:, 4]])   # -> 11,22,33,12,13,23
        job = os.path.join(WORK, f"{gid}_sref")
        zero = np.zeros((len(Cs), 3, 3))
        g.write_deck(job + ".inp", Xs, Ts, lab, Cs, zero, hold)
        txt = open(job + ".inp").read()
        ini = ["*INITIAL CONDITIONS, TYPE=STRESS"] + [f"{e + 1},{ip},{','.join(f'{x:.6g}' for x in sig[e])}" for e in range(len(T)) for ip in (1, 2, 3, 4)]
        txt = txt.replace("*INITIAL CONDITIONS, TYPE=TEMPERATURE", "\n".join(ini) + "\n*INITIAL CONDITIONS, TYPE=TEMPERATURE")
        txt = txt.replace(f"*TEMPERATURE\nNALL, {g.DT}\n", "*TEMPERATURE\nNALL, 0.\n")
        open(job + ".inp", "w").write(txt); with_stress_output(job + ".inp", len(Cs))
        Uref, Sref = run_with_stress(job, len(Xs), len(T))
        idx = rng.choice(g.surface_nodes(T), size=10, replace=False)
        a, _ = g.fit(X, Ub, data + rng.normal(0, g.SY, data.shape), idx, True)
        Sp, Sc = Sb[0] + Sb[1], a[0] * Sb[0] + a[1] * Sb[1]
        r = dict(id=gid, factors=a.tolist())
        for nm, f in (("vm", vm), ("smax", smax)):
            ref, pr, co = f(Sref), f(Sp), f(Sc)
            r[f"{nm}_p95_ref"], r[f"{nm}_p95_prior"], r[f"{nm}_p95_corrected"] = [float(np.percentile(v, 95)) for v in (ref, pr, co)]
            r[f"{nm}_corr"] = float(np.corrcoef(ref, co)[0, 1])
        rows.append(r)
        print(f"{gid}: factors {np.round(a, 2).tolist()} | von Mises p95 [MPa]: reference {r['vm_p95_ref']:.1f}, prior {r['vm_p95_prior']:.1f}, "
              f"corrected {r['vm_p95_corrected']:.1f} (element corr {r['vm_corr']:.2f}) | max principal p95: reference "
              f"{r['smax_p95_ref']:.1f}, prior {r['smax_p95_prior']:.1f}, corrected {r['smax_p95_corrected']:.1f} (corr {r['smax_corr']:.2f})", flush=True)
    dst = os.path.join(ROOT, "results", "greif3d_rescue", "stress_check.json")
    json.dump(rows, open(dst, "w"), indent=1); print("wrote", dst)


if __name__ == "__main__":
    main()
