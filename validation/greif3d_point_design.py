"""validation/greif3d_point_design.py -- where to measure: choosing the K surface points for the RESCUE correction
of a 3-D part from the physics prior alone (before any measurement), against random points.

Model (validation/greif3d_rescue.py, eq. 2): y = a_iso U_iso + a_dev U_dev + t + w x (X - Xc) + noise, all
three displacement components measured at each point (noise s_y). The rigid motion (6) is a nuisance. For a set
S of points the information on the two factors, with the rigid motion eliminated (Schur complement), is
    J_a(S) = A^T A - A^T R (R^T R)^-1 R^T A,   A = [U_iso, U_dev](S), R = rigid basis(S)                    (1)
and the design maximises log det(J_a / s_y^2 + I / s_a^2) (D-optimal; the prior keeps it finite). Points are
chosen greedily, one at a time, from the surface nodes (sub-sampled), using only U_iso, U_dev (no data).
Evaluation: the relative error of the corrected warp field over all nodes (as greif3d_rescue.py) for
  - D-optimal points (one set per part),
  - random points: median and 90th percentile of 50 draws (the 90th percentile is the "unlucky" case),
  - the whole-field limit (all nodes, 2 factors).
Honest scope: the design targets the noise-driven uncertainty of the factors; most of the residual error after
correction is the SHAPE mismatch of the prior, which no choice of points removes. All three components per
point (a CMM usually gives the surface-normal one). Moldflow data, not measurements; parts taken from the
interim fifa set (the VTUs are not in the repository).
Run:  python3 validation/greif3d_point_design.py DATA WORK id [id ...]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import greif3d_rescue as g  # noqa: E402

OUT = os.path.join(ROOT, "results", "greif3d_rescue")


def prior(DATA, WORK, gid):
    cache = os.path.join(WORK, f"{gid}_prior.npz")
    if os.path.exists(cache):
        z = np.load(cache); return [z["Ui"], z["Ud"]]
    vtu, X, T, data, An, E1 = g.load_geometry(os.path.join(DATA, gid))
    T, _ = g.fix_orientation(X, T)
    A = g.clean_orientation(An[T].mean(1))
    lab, cen = g.kmeans(g.vec6(A), 32)
    Cs, als = g.cluster_props(g.clean_orientation(g.mat6(cen)))
    Xs, Ts = g.to_c3d10(X, T)
    U = g.solve(Xs, Ts, lab, Cs, als, WORK, gid, threads=4)
    Ub = [U[0][: len(X)], U[1][: len(X)]]
    np.savez(cache, Ui=Ub[0], Ud=Ub[1]); return Ub


def schur_logdet(A, R, eps=1e-9):
    J = A.T @ A
    RR = R.T @ R + eps * np.trace(R.T @ R + np.eye(6)) * np.eye(6)
    Ja = J - A.T @ R @ np.linalg.solve(RR, R.T @ A)
    return np.linalg.slogdet(Ja / g.SY ** 2 + np.eye(2) / g.SA ** 2)[1]


def greedy(X, Ub, cand, K):
    Rall = g.rigid_basis(X).reshape(len(X), 3, 6)
    A3 = np.stack([Ub[0], Ub[1]], -1)                      # (n, 3, 2)
    chosen = []
    for _ in range(K):
        best, bv = None, -np.inf
        for c in cand:
            if c in chosen:
                continue
            S = chosen + [c]
            v = schur_logdet(A3[S].reshape(-1, 2), Rall[S].reshape(-1, 6))
            if v > bv:
                best, bv = c, v
        chosen.append(best)
    return np.array(chosen)


def main():
    DATA, WORK, ids = sys.argv[1], sys.argv[2], sys.argv[3:]
    os.makedirs(WORK, exist_ok=True); os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(20261104)
    rows = []
    for gid in ids:
        _, X, T, data, _, _ = g.load_geometry(os.path.join(DATA, gid))
        Ub = prior(DATA, WORK, gid)
        S = g.surface_nodes(T)
        cand = rng.choice(S, size=min(600, len(S)), replace=False)
        r = dict(id=gid, field_2f=g.fit(X, Ub, data, np.arange(len(X)), True, prior=False)[1])
        r["field_2f"] = g.rel_err(X, r["field_2f"], data)
        r["prior"] = g.rel_err(X, Ub[0] + Ub[1], data)
        for K in (3, 5, 10):
            idx = greedy(X, Ub, cand, K)
            y = data + rng.normal(0, g.SY, data.shape)
            r[f"dopt_{K}"] = g.rel_err(X, g.fit(X, Ub, y, idx, True)[1], data)
            es = []
            for _ in range(50):
                ii = rng.choice(S, size=K, replace=False)
                y = data + rng.normal(0, g.SY, data.shape)
                es.append(g.rel_err(X, g.fit(X, Ub, y, ii, True)[1], data))
            r[f"rand_{K}_median"], r[f"rand_{K}_p90"] = float(np.median(es)), float(np.percentile(es, 90))
        rows.append(r)
        print(f"{gid}: prior {r['prior']:.2f}, field {r['field_2f']:.2f} | " + " | ".join(
            f"K={K}: D-opt {r[f'dopt_{K}']:.2f}, random median {r[f'rand_{K}_median']:.2f} / p90 {r[f'rand_{K}_p90']:.2f}"
            for K in (3, 5, 10)), flush=True)
    summ = {k: float(np.median([r[k] for r in rows])) for k in rows[0] if k != "id"}
    print("median over parts:", {k: round(v, 3) for k, v in summ.items()})
    json.dump(dict(parts=rows, median=summ), open(os.path.join(OUT, "point_design.json"), "w"), indent=1)
    print("wrote", os.path.join(OUT, "point_design.json"))


if __name__ == "__main__":
    main()
