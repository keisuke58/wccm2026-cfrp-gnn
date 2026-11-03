"""validation/greif3d_rescue.py -- RESCUE on 3-D parts: a simple physics prior for the warpage of the Greif & Meyer
geometries (Zenodo 17831586, Moldflow 2023, PP/GF30, holding pressure removed), corrected with a few "measured"
nodes. Same structure as the flat plate (validation/denso_tsuda2026_transfer.py), in 3-D.

Prior (per geometry):
  data fibre orientation a_ij (nodes; element value = mean of its 4 corner nodes; trace normalised to 1)
  -> K clusters of a (k-means) -> per cluster Mori-Tanaka UD + Advani-Tucker orientation average (hybrid closure):
     stiffness <C> and CTE <alpha> (fiber_orientation_cte.orient_avg_cte)
  -> linear-elastic FE (CalculiX, C3D10 from the data's tetrahedra with straight mid-side nodes, or C3D4), free part
     held 3-2-1, uniform temperature change dT. Two load cases by superposition: the isotropic part of <alpha>
     (tr/3 I) and the deviatoric part.                                                                (1)
  u_prior = U_iso + U_dev.
Measurement stand-in: the data's Moldflow deflection (metres in the VTU, x1000 to mm), at a few surface nodes
(K_meas = 5, 10, 20, random, 10 draws), with noise s_y (0.01 mm, example).
Correction (L2-SP, linear, closed form): u = a_iso U_iso + a_dev U_dev + t + w x (X - Xc)                 (2)
  with prior a = 1 (s_a = 0.5) and the six rigid-body terms free; 1-factor version a_iso = a_dev = a.
Error measure: rms over ALL corner nodes of (prediction - data) after removing the best rigid-body motion of
the difference, divided by the rms of the rigid-removed data deflection (relative shape error).
Reported per geometry: prior, whole-field best 1- and 2-factor (the most a factor can do), RESCUE with K_meas
nodes, the cosine between the prior and the data field.

Checks:
  G1 (selftest) patch test: homogeneous isotropic material, uniform alpha: the 3-2-1 held part expands freely,
     u = alpha dT (X - X_A) up to a rigid motion (machine precision expected);
  G2 (selftest) synthetic recovery: data := 0.7 U_iso + 1.3 U_dev + rigid motion + noise on a plate with an
     asymmetric skin-core orientation; RESCUE with 10 nodes must recover the factors;
  G3 (selftest) C3D4 vs C3D10 on the same plate (stiffness locking of linear tetrahedra in bending), and
     G3b C3D10 on the mesh refined 2x (4 -> 8 elements through the thickness, as in the data's 6-8);
  G4 (data) modulus check: the prior's Young's modulus along the principal orientation direction vs the data's
     "Tensile_modulus_in_first_principal_direction" (median ratio per geometry);
  G5 (data, option --k) cluster count: K = 32 vs 64 changes the prior field by (printed).

Honest scope: material constants are GENERAL PP/GF30 values (not Celanese Factor PP GF 30 data; example, not
calibrated); one effective dT (example -100 K, absorbed by the factors); no crystallisation shrinkage model (PP
shrinks isotropically on crystallisation: in the prior only through the isotropic factor); linear elasticity, no
relaxation; the off-diagonal order of the data's orientation tensor (xy, yz, xz) is assumed, not documented;
the "measurement" is Moldflow without holding pressure, not a real measurement (the same limit as Greif & Meyer).

Run:
  python3 validation/greif3d_rescue.py --selftest                     (G1-G3, here; needs ccx)
  python3 validation/greif3d_rescue.py --data ~/data/greif2025_warp --ids 00010006 00010010 --out DIR
  python3 validation/greif3d_rescue.py --data ~/data/greif2025_warp --smallest 20 --out DIR
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
import denso_tsuda2026_compare as tc  # noqa: E402
from fiber_orientation_cte import orient_avg_cte, VOIGT  # noqa: E402

# GENERAL PP/GF30 values (example, not calibrated)
PP = dict(E=1.5e9, nu=0.40, alpha=1.0e-4)          # isotact. PP, room temperature, typical
GFP = dict(E=72e9, nu=0.22, alpha=5.0e-6)
VF_PP = (0.30 / 2.55) / (0.30 / 2.55 + 0.70 / 0.905)  # 30 wt%, densities 2.55 / 0.905 -> 0.131
ASPECT_PP = 20.0                                    # assumed
DT = -100.0                                         # example effective cooling [K]
SY, SA = 0.01, 0.5                                  # measurement noise [mm] (example), prior width of factors
CCX = os.environ.get("CCX", "ccx")


# ----------------------------------------------------------------------------- material
def ud_pp():
    Cm = tc.iso_C_mandel(PP["E"], PP["nu"]); Cf = tc.iso_C_mandel(GFP["E"], GFP["nu"])
    C = tc.mori_tanaka(Cm, Cf, tc.eshelby_prolate(ASPECT_PP, PP["nu"]), VF_PP)
    a = tc.levin_alpha(C, Cm, Cf, PP["alpha"], GFP["alpha"])
    return tc.mandel_to_voigt(C), a[0], a[1]


def cluster_props(A):
    """A (K,3,3) orientation tensors -> stiffness (K,6,6) [MPa, Voigt 11,22,33,23,13,12, engineering shear]
    and alpha (K,3,3)."""
    Cud, aL, aT = ud_pp()
    Cs, als = [], []
    for a in A:
        al, C = orient_avg_cte(a, "hybrid", Cud, aL=aL, aT=aT)
        Cs.append(C / 1e6); als.append(al)
    return np.array(Cs), np.array(als)


def kmeans(X, k, seed=0, iters=40):
    rng = np.random.default_rng(seed)
    k = min(k, len(X))
    c = [X[rng.integers(len(X))]]
    for _ in range(1, k):                      # k-means++
        d2 = np.min(((X[:, None, :] - np.array(c)[None]) ** 2).sum(-1), 1)
        if d2.sum() == 0:
            break
        c.append(X[rng.choice(len(X), p=d2 / d2.sum())])
    c = np.array(c)
    for _ in range(iters):
        lab = np.argmin(((X[:, None, :] - c[None]) ** 2).sum(-1), 1)
        c = np.array([X[lab == j].mean(0) if np.any(lab == j) else c[j] for j in range(len(c))])
    lab = np.argmin(((X[:, None, :] - c[None]) ** 2).sum(-1), 1)
    return lab, c


def vec6(A):   # (n,3,3) -> (n,6) with sqrt2 on shear, for distances
    s = np.sqrt(2)
    return np.stack([A[:, 0, 0], A[:, 1, 1], A[:, 2, 2], s * A[:, 0, 1], s * A[:, 1, 2], s * A[:, 0, 2]], 1)


def mat6(v):
    s = np.sqrt(2)
    A = np.zeros((len(v), 3, 3))
    A[:, 0, 0], A[:, 1, 1], A[:, 2, 2] = v[:, 0], v[:, 1], v[:, 2]
    A[:, 0, 1] = A[:, 1, 0] = v[:, 3] / s; A[:, 1, 2] = A[:, 2, 1] = v[:, 4] / s; A[:, 0, 2] = A[:, 2, 0] = v[:, 5] / s
    return A


def clean_orientation(A):
    w, V = np.linalg.eigh(0.5 * (A + A.transpose(0, 2, 1)))
    w = np.clip(w, 0.0, None); w /= np.maximum(w.sum(1, keepdims=True), 1e-12)
    return np.einsum("nij,nj,nkj->nik", V, w, V)


# --------------------------------------------------------------------------------- mesh
def fix_orientation(X, T):
    v = np.einsum("ij,ij->i", np.cross(X[T[:, 1]] - X[T[:, 0]], X[T[:, 2]] - X[T[:, 0]]), X[T[:, 3]] - X[T[:, 0]])
    T = T.copy(); neg = v < 0
    T[neg, 1], T[neg, 2] = T[neg, 2].copy(), T[neg, 1].copy()
    return T, np.abs(v) / 6


def to_c3d10(X, T):
    E = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]], T[:, [0, 3]], T[:, [1, 3]], T[:, [2, 3]]])
    Es = np.sort(E, 1)
    uniq, inv = np.unique(Es, axis=0, return_inverse=True)
    mid = 0.5 * (X[uniq[:, 0]] + X[uniq[:, 1]])
    ids = len(X) + inv.reshape(6, -1).T
    return np.vstack([X, mid]), np.hstack([T, ids])


def surface_nodes(T):
    F = np.concatenate([T[:, [0, 1, 2]], T[:, [0, 1, 3]], T[:, [0, 2, 3]], T[:, [1, 2, 3]]])
    Fs = np.sort(F, 1)
    u, c = np.unique(Fs, axis=0, return_counts=True)
    return np.unique(u[c == 1])


def hold_321(X):
    """Three nodes and the dofs that hold a free body statically determinate."""
    A = int(np.argmin(X[:, 0] + X[:, 1] + X[:, 2]))
    B = int(np.argmax(((X - X[A]) ** 2).sum(1)))
    e = (X[B] - X[A]) / np.linalg.norm(X[B] - X[A])
    r = X - X[A]; dperp = r - np.outer(r @ e, e)
    C = int(np.argmax((dperp ** 2).sum(1)))
    n = np.cross(e, X[C] - X[A]); n /= np.linalg.norm(n)
    ib = int(np.argmax(np.abs(e)))
    ic = int(np.argmax(np.abs(n)))
    return [(A, d) for d in (1, 2, 3)] + [(B, d + 1) for d in range(3) if d != ib] + [(C, ic + 1)]


# ------------------------------------------------------------------------------ CalculiX
def aniso21(C):
    """Voigt (11,22,33,23,13,12) -> CalculiX ANISO order D1111,D1122,D2222,D1133,D2233,D3333,D1112,D2212,D3312,
    D1212,D1113,D2213,D3313,D1213,D1313,D1123,D2223,D3323,D1223,D1323,D2323."""
    ix = {p: i for i, p in enumerate(VOIGT)}
    def c(a, b):
        return C[ix[tuple(sorted(a))], ix[tuple(sorted(b))]]
    pairs = [((0, 0), (0, 0)), ((0, 0), (1, 1)), ((1, 1), (1, 1)), ((0, 0), (2, 2)), ((1, 1), (2, 2)), ((2, 2), (2, 2)),
             ((0, 0), (0, 1)), ((1, 1), (0, 1)), ((2, 2), (0, 1)), ((0, 1), (0, 1)),
             ((0, 0), (0, 2)), ((1, 1), (0, 2)), ((2, 2), (0, 2)), ((0, 1), (0, 2)), ((0, 2), (0, 2)),
             ((0, 0), (1, 2)), ((1, 1), (1, 2)), ((2, 2), (1, 2)), ((0, 1), (1, 2)), ((0, 2), (1, 2)), ((1, 2), (1, 2))]
    return [c(a, b) for a, b in pairs]


def write_deck(path, X, T, lab, Cs, als, hold):
    q = T.shape[1] == 10
    with open(path, "w") as f:
        f.write("*NODE, NSET=NALL\n")
        for i, x in enumerate(X):
            f.write(f"{i + 1},{x[0]:.9g},{x[1]:.9g},{x[2]:.9g}\n")
        for j in range(len(Cs)):
            el = np.where(lab == j)[0]
            if len(el) == 0:
                continue
            f.write(f"*ELEMENT, TYPE={'C3D10' if q else 'C3D4'}, ELSET=E{j}\n")
            for e in el:
                f.write(f"{e + 1}," + ",".join(str(n + 1) for n in T[e]) + "\n")
        for j in range(len(Cs)):
            if not np.any(lab == j):
                continue
            d = aniso21(Cs[j]); a = als[j]
            f.write(f"*MATERIAL, NAME=M{j}\n*ELASTIC, TYPE=ANISO\n")
            for k in range(0, 21, 8):
                f.write(",".join(f"{v:.9g}" for v in d[k:k + 8]) + "\n")
            f.write("*EXPANSION, TYPE=ANISO, ZERO=0.\n")
            f.write(",".join(f"{v:.9g}" for v in (a[0, 0], a[1, 1], a[2, 2], a[0, 1], a[0, 2], a[1, 2])) + "\n")
            f.write(f"*SOLID SECTION, ELSET=E{j}, MATERIAL=M{j}\n")
        f.write("*INITIAL CONDITIONS, TYPE=TEMPERATURE\nNALL, 0.\n*BOUNDARY\n")
        for n, d in hold:
            f.write(f"{n + 1},{d},{d},0.\n")
        f.write(f"*STEP\n*STATIC\n*TEMPERATURE\nNALL, {DT}\n*NODE PRINT, NSET=NALL\nU\n*END STEP\n")


def run_ccx(job, nnode, threads=2):
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    d, j = os.path.split(job)
    r = subprocess.run([CCX, "-i", j], cwd=d or ".", env=env, capture_output=True, text=True)
    dat = job + ".dat"
    if not os.path.exists(dat):
        raise RuntimeError(r.stdout[-2000:] + r.stderr[-2000:])
    U = np.zeros((nnode, 3)); on = False
    for line in open(dat):
        if "displacements" in line:
            on = True; continue
        if on:
            p = line.split()
            if len(p) == 4:
                U[int(p[0]) - 1] = [float(v) for v in p[1:]]
    for ext in (".frd", ".sta", ".cvg", ".12d", ".spool", ".dat", ".inp"):   # decks of large parts are 100+ MB
        if os.path.exists(job + ext):
            os.remove(job + ext)
    return U


def solve(X, T, lab, Cs, als, work, tag, threads=2):
    """Two load cases (isotropic / deviatoric part of alpha). Returns U_iso, U_dev at all nodes [mm]."""
    hold = hold_321(X)
    tr = np.trace(als, axis1=1, axis2=2) / 3
    aiso = tr[:, None, None] * np.eye(3)[None]
    out = []
    for nm, al in (("iso", aiso), ("dev", als - aiso)):
        job = os.path.join(work, f"{tag}_{nm}")
        write_deck(job + ".inp", X, T, lab, Cs, al, hold)
        out.append(run_ccx(job, len(X), threads))
    return out


# ------------------------------------------------------------------------------- RESCUE
def rigid_basis(X):
    Xc = X - X.mean(0)
    n = len(X); R = np.zeros((n, 3, 6))
    R[:, 0, 0] = R[:, 1, 1] = R[:, 2, 2] = 1.0
    # w x r : rows of the cross-product matrix
    R[:, 0, 4], R[:, 0, 5] = Xc[:, 2], -Xc[:, 1]
    R[:, 1, 3], R[:, 1, 5] = -Xc[:, 2], Xc[:, 0]
    R[:, 2, 3], R[:, 2, 4] = Xc[:, 1], -Xc[:, 0]
    return R.reshape(3 * n, 6)


def remove_rigid(X, u):
    R = rigid_basis(X); y = u.reshape(-1)
    c, *_ = np.linalg.lstsq(R, y, rcond=None)
    return (y - R @ c).reshape(-1, 3)


def rel_err(X, pred, data):
    e = remove_rigid(X, pred - data); d = remove_rigid(X, data)
    return float(np.sqrt((e ** 2).sum(1).mean()) / np.sqrt((d ** 2).sum(1).mean()))


def fit(X, Ub, y, idx, two=True, prior=True):
    """L2-SP least squares (2): factors with prior 1 (s_a) and free rigid motion, on nodes idx."""
    Ucols = [Ub[0][idx].reshape(-1), Ub[1][idx].reshape(-1)] if two else [(Ub[0] + Ub[1])[idx].reshape(-1)]
    M = np.column_stack(Ucols + [rigid_basis(X)[np.concatenate([3 * idx[:, None] + k for k in range(3)], 1).reshape(-1)]])
    # rigid_basis rows are node-major (x,y,z per node): reorder to match idx
    na = len(Ucols)
    W = np.vstack([M / SY] + ([np.hstack([np.eye(na) / SA, np.zeros((na, 6))])] if prior else []))
    b = np.concatenate([y[idx].reshape(-1) / SY] + ([np.ones(na) / SA] if prior else []))
    p, *_ = np.linalg.lstsq(W, b, rcond=None)
    a = p[:na]
    pred = a[0] * Ub[0] + a[1] * Ub[1] if two else a[0] * (Ub[0] + Ub[1])
    return a, pred


def evaluate(X, Xs_idx, Ub, data, rng, kmeas=(5, 10, 20), draws=10):
    prior = Ub[0] + Ub[1]
    allidx = np.arange(len(X))
    r = dict(prior=rel_err(X, prior, data))
    pd, dd = remove_rigid(X, prior).reshape(-1), remove_rigid(X, data).reshape(-1)
    r["cos_prior_data"] = float(pd @ dd / np.linalg.norm(pd) / np.linalg.norm(dd))
    for two in (False, True):
        a, pred = fit(X, Ub, data, allidx, two, prior=False)
        r[f"field_{'2f' if two else '1f'}"] = rel_err(X, pred, data); r[f"field_{'2f' if two else '1f'}_a"] = a.tolist()
    for k in kmeas:
        for two in (False, True):
            es, As = [], []
            for _ in range(draws):
                idx = rng.choice(Xs_idx, size=min(k, len(Xs_idx)), replace=False)
                y = data + rng.normal(0, SY, data.shape)
                a, pred = fit(X, Ub, y, idx, two)
                es.append(rel_err(X, pred, data)); As.append(a.tolist())
            r[f"rescue_{k}_{'2f' if two else '1f'}"] = float(np.median(es))
            r[f"rescue_{k}_{'2f' if two else '1f'}_a_median"] = np.median(np.array(As), 0).tolist()
    return r


# ----------------------------------------------------------------------------- selftest
def plate_mesh(L=40.0, W=20.0, H=2.0, nx=20, ny=10, nz=4):
    xs, ys, zs = np.linspace(0, L, nx + 1), np.linspace(0, W, ny + 1), np.linspace(0, H, nz + 1)
    X = np.array([[x, y, z] for z in zs for y in ys for x in xs])
    nid = lambda i, j, k: k * (ny + 1) * (nx + 1) + j * (nx + 1) + i
    T = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                c = [nid(i, j, k), nid(i + 1, j, k), nid(i + 1, j + 1, k), nid(i, j + 1, k),
                     nid(i, j, k + 1), nid(i + 1, j, k + 1), nid(i + 1, j + 1, k + 1), nid(i, j + 1, k + 1)]
                for t in ((0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)):
                    T.append([c[m] for m in t])
    return X, np.array(T)


def selftest(work):
    os.makedirs(work, exist_ok=True)
    X, T = plate_mesh(); T, vol = fix_orientation(X, T)
    res = {}
    # G1
    Ciso = tc.mandel_to_voigt(tc.iso_C_mandel(3000.0, 0.35)); al = 5e-5 * np.eye(3)
    lab = np.zeros(len(T), int)
    X10, T10 = to_c3d10(X, T)
    U = solve(X10, T10, lab, np.array([Ciso]), np.array([al]), work, "g1")
    u = (U[0] + U[1])[: len(X)]
    exact = 5e-5 * DT * (X - X[0])
    g1 = float(np.max(np.abs(remove_rigid(X, u - exact))) / np.max(np.abs(exact)))
    res["G1_patch"] = g1
    print(f"[G1] patch test, isotropic free expansion: max |u - alpha dT x| (rigid removed) {g1:.1e} of max |u|")
    # synthetic asymmetric skin-core orientation (in x), plate
    zc = X[T].mean(1)[:, 2] / 2.0
    a11 = np.clip(0.45 + 0.45 * zc ** 2 + 0.15 * zc, 0.1, 0.9)
    A = np.zeros((len(T), 3, 3)); A[:, 0, 0] = a11; A[:, 1, 1] = 0.9 - a11 + 0.05; A[:, 2, 2] = 1 - A[:, 0, 0] - A[:, 1, 1]
    A = clean_orientation(A)
    lab, cen = kmeans(vec6(A), 16)
    Cs, als = cluster_props(clean_orientation(mat6(cen)))
    U10 = solve(X10, T10, lab, Cs, als, work, "g2")
    Ub = [U10[0][: len(X)], U10[1][: len(X)]]
    rng = np.random.default_rng(1)
    rig = (rigid_basis(X) @ rng.normal(0, 1e-3, 6)).reshape(-1, 3)
    data = 0.7 * Ub[0] + 1.3 * Ub[1] + rig
    S = surface_nodes(T)
    ev = evaluate(X, S, Ub, data, rng, kmeas=(10,), draws=10)
    res["G2"] = ev
    print(f"[G2] synthetic data 0.7 U_iso + 1.3 U_dev: prior error {ev['prior']:.3f}, RESCUE 10 nodes 2-factor "
          f"{ev['rescue_10_2f']:.3f} (factors {np.round(ev['rescue_10_2f_a_median'], 3).tolist()}), "
          f"1-factor {ev['rescue_10_1f']:.3f}; max |u| {np.abs(remove_rigid(X, data)).max():.3f} mm")
    # G3
    U4 = solve(X, T, lab, Cs, als, work, "g3")
    p4, p10 = U4[0] + U4[1], Ub[0] + Ub[1]
    g3 = rel_err(X, p4, p10)
    res["G3_c3d4_vs_c3d10"] = g3
    print(f"[G3] C3D4 vs C3D10 (same tetrahedra), relative difference of the warp field {g3:.3f}; "
          f"max out-of-plane C3D4 {np.abs(remove_rigid(X, p4)[:, 2]).max():.4f}, C3D10 {np.abs(remove_rigid(X, p10)[:, 2]).max():.4f} mm")
    # G3b: C3D10 with the mesh refined 2x in every direction (same synthetic orientation law)
    Xf, Tf = plate_mesh(nx=40, ny=20, nz=8); Tf, _ = fix_orientation(Xf, Tf)
    zf = Xf[Tf].mean(1)[:, 2] / 2.0
    af = np.clip(0.45 + 0.45 * zf ** 2 + 0.15 * zf, 0.1, 0.9)
    Af = np.zeros((len(Tf), 3, 3)); Af[:, 0, 0] = af; Af[:, 1, 1] = 0.9 - af + 0.05; Af[:, 2, 2] = 1 - Af[:, 0, 0] - Af[:, 1, 1]
    labf = np.argmin(((vec6(clean_orientation(Af))[:, None, :] - cen[None]) ** 2).sum(-1), 1)
    Xf10, Tf10 = to_c3d10(Xf, Tf)
    Uf = solve(Xf10, Tf10, labf, Cs, als, work, "g3b")
    wf = np.abs(remove_rigid(Xf, (Uf[0] + Uf[1])[: len(Xf)])[:, 2]).max()
    w10 = np.abs(remove_rigid(X, p10)[:, 2]).max()
    res["G3b_c3d10_refined_change"] = float(abs(wf - w10) / wf)
    print(f"[G3b] C3D10, mesh refined 2x: max out-of-plane {wf:.4f} mm vs {w10:.4f} mm (change {abs(wf - w10) / wf:.3f})")
    return res


# --------------------------------------------------------------------------------- data
def load_geometry(d):
    import meshio
    vtu = sorted(f for f in os.listdir(d) if f.endswith("_sim.vtu"))[0]
    m = meshio.read(os.path.join(d, vtu))
    X = np.asarray(m.points, float)
    T = np.vstack([c.data for c in m.cells if c.type == "tetra"]).astype(int)
    key = [k for k in m.point_data if k.startswith("Deflection")][0]
    u = np.asarray(m.point_data[key], float)[:, :3] * 1e3                        # m -> mm
    ok = [k for k in m.point_data if k.startswith("Fiber_orientation_tensor")][0]
    o = np.asarray(m.point_data[ok], float)                                      # 11,22,33,12,23,13 (assumed)
    An = np.zeros((len(X), 3, 3))
    An[:, 0, 0], An[:, 1, 1], An[:, 2, 2] = o[:, 0], o[:, 1], o[:, 2]
    An[:, 0, 1] = An[:, 1, 0] = o[:, 3]; An[:, 1, 2] = An[:, 2, 1] = o[:, 4]; An[:, 0, 2] = An[:, 2, 0] = o[:, 5]
    E1 = None
    for k, v in m.cell_data.items():
        if k.startswith("Tensile_modulus_in_first"):
            E1 = np.concatenate([np.asarray(a, float).reshape(-1) for a in v])
    return vtu, X, T, u, An, E1


def directional_E(C, n):
    S = np.linalg.inv(C)
    v = np.array([n[0] ** 2, n[1] ** 2, n[2] ** 2, n[1] * n[2], n[0] * n[2], n[0] * n[1]])  # engineering
    return 1.0 / (v @ S @ v)


def run_geometry(d, work, k, quad, threads, rng):
    t0 = time.time()
    vtu, X, T, data, An, E1 = load_geometry(d)
    T, vol = fix_orientation(X, T)
    A = clean_orientation(An[T].mean(1))
    lab, cen = kmeans(vec6(A), k)
    Ac = clean_orientation(mat6(cen))
    Cs, als = cluster_props(Ac)
    qerr = float(np.sqrt(((vec6(A) - cen[lab]) ** 2).sum(1).mean()))
    Xs, Ts = to_c3d10(X, T) if quad else (X, T)
    U = solve(Xs, Ts, lab, Cs, als, work, os.path.basename(d), threads)
    Ub = [U[0][: len(X)], U[1][: len(X)]]
    r = dict(id=os.path.basename(d), vtu=vtu, n_tet=len(T), n_node=len(X), k=k, element="C3D10" if quad else "C3D4",
             cluster_rms=qerr, max_data_mm=float(np.linalg.norm(remove_rigid(X, data), axis=1).max()),
             max_prior_mm=float(np.linalg.norm(remove_rigid(X, Ub[0] + Ub[1]), axis=1).max()))
    if E1 is not None and len(E1) == len(T):
        Ep = np.array([directional_E(Cs[j], np.linalg.eigh(Ac[j])[1][:, -1]) for j in range(len(Cs))])
        r["G4_E1_ratio_median"] = float(np.median(Ep[lab] / (E1 / 1e6)))
    r.update(evaluate(X, surface_nodes(T), Ub, data, rng))
    r["seconds"] = time.time() - t0
    return r, X, Ub


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--data")
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--smallest", type=int, default=0, help="the N geometries with the fewest tetrahedra (survey CSV)")
    ap.add_argument("--survey", default=os.path.join(ROOT, "abaqus", "denso_plate", "abaqus_results", "greif3d", "greif_survey.csv"))
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--k-check", type=int, default=0, help="G5: also run with this K and report the change")
    ap.add_argument("--c3d4", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "greif3d_rescue"))
    ap.add_argument("--seed", type=int, default=20261104)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    work = os.path.join(a.out, "work"); os.makedirs(work, exist_ok=True)
    if a.selftest:
        res = selftest(work)
        json.dump(res, open(os.path.join(a.out, "selftest.json"), "w"), indent=1)
        print("wrote", os.path.join(a.out, "selftest.json")); return
    ids = list(a.ids or [])
    if a.smallest:
        rows = list(csv.DictReader(open(a.survey)))
        ids += [r["id"] for r in sorted(rows, key=lambda r: int(r["n_cells"]))[: a.smallest]]
    rng = np.random.default_rng(a.seed)
    rows = []
    for gid in ids:
        r, X, Ub = run_geometry(os.path.join(a.data, gid), work, a.k, not a.c3d4, a.threads, rng)
        if a.k_check:
            r2, _, Ub2 = run_geometry(os.path.join(a.data, gid), work, a.k_check, not a.c3d4, a.threads, rng)
            r["G5_k_change"] = rel_err(X, Ub2[0] + Ub2[1], Ub[0] + Ub[1])
        rows.append(r)
        print(f"{gid}: tets {r['n_tet']}, prior {r['prior']:.2f}, cos {r['cos_prior_data']:+.2f}, field 1f/2f "
              f"{r['field_1f']:.2f}/{r['field_2f']:.2f}, RESCUE 10 nodes 1f/2f {r['rescue_10_1f']:.2f}/{r['rescue_10_2f']:.2f}, "
              f"E1 ratio {r.get('G4_E1_ratio_median', float('nan')):.2f}, {r['seconds']:.0f} s", flush=True)
        json.dump(rows, open(os.path.join(a.out, "per_geometry.json"), "w"), indent=1)
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float)) and not isinstance(rows[0][k], bool)]
    with open(os.path.join(a.out, "per_geometry.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id"] + keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    summ = {k: dict(median=float(np.median([r[k] for r in rows])), q25=float(np.percentile([r[k] for r in rows], 25)),
                    q75=float(np.percentile([r[k] for r in rows], 75)))
            for k in keys if k.startswith(("prior", "field_", "rescue_", "cos_", "G4", "G5")) and not k.endswith("_a")}
    json.dump(dict(n=len(rows), k=a.k, summary=summ), open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
