"""
Zero-shot physics anomaly detection: harmonicity violation of the stress sum.

In 2D isotropic elasticity (plane stress/strain, no body force) the first
stress invariant is HARMONIC: ∇²(σ1+σ2) = 0 (Beltrami–Michell compatibility).
A defect locally violates compatibility → |∇²DSPSS| spikes there.

NO reference data, NO training — a single measured field suffices. CFRP is
orthotropic so harmonicity holds only approximately; the question is whether
the violation signal still separates defect nodes.

Score: graph-Laplacian magnitude on the node mesh,
  L v (i) = v_i − Σ_j w_ij v_j ,  w_ij ∝ 1/d_ij over k spatial neighbours.

Run: python3 harmonic_anomaly.py
"""
import os
import glob
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

BASE      = "/home/nishioka/CFRP/CFRP_hole/hole_data_inp"
DIR_2X2   = f"{BASE}/Defect_hole_2x2_Region1_21_npy"
DIR_4X4   = f"{BASE}/Defect_hole_4x4_Region1_21_npy"
LBL_2X2   = f"{BASE}/Def2x2_19class_label"
LBL_4X4   = f"{BASE}/Def4x4_19class_label"
COORD_DIR = f"{BASE}/basicdata_for_holegnn"
N_NODES   = 13942
K         = 8
NOISE_LEVELS = [0.0, 0.1, 0.3, 0.5]


def load_field(fpath):
    v = np.load(fpath).astype(np.float32)[:N_NODES]
    return (v - v.mean()) / (v.std() + 1e-8)


def load_label(label_dir, fpath):
    base = os.path.splitext(os.path.basename(fpath))[0]
    for cand in (f"{base}.npy", f"{base}_19label.npy"):
        lf = os.path.join(label_dir, cand)
        if os.path.exists(lf):
            lbl = np.load(lf)
            return (lbl[:, 1:].sum(axis=1) > 0).astype(np.float32)
    return None


def build_laplacian():
    x = np.load(f"{COORD_DIR}/normalized_x_2layer.npy")[:N_NODES]
    y = np.load(f"{COORD_DIR}/normalized_y_2layer.npy")[:N_NODES]
    P = np.stack([x, y], 1)
    nn_ = NearestNeighbors(n_neighbors=K + 1).fit(P)
    dist, idx = nn_.kneighbors(P)
    nb, d = idx[:, 1:], dist[:, 1:] + 1e-9            # exclude self
    w = 1.0 / d
    w = w / w.sum(1, keepdims=True)                    # row-normalised weights
    return nb, w.astype(np.float32)


def laplacian_score(v, nb, w):
    return np.abs(v - (v[nb] * w).sum(1))


def evaluate(nb, w, raw_dir, label_dir, n_samples, noise, nrng):
    files = sorted(glob.glob(os.path.join(raw_dir, "*.npy")))[:n_samples]
    S, L = [], []
    for f in files:
        lbl = load_label(label_dir, f)
        if lbl is None:
            continue
        v = load_field(f)
        if noise > 0:
            v = v + nrng.normal(0, noise, v.shape).astype(np.float32)
        S.append(laplacian_score(v, nb, w))
        L.append(lbl)
    return roc_auc_score(np.concatenate(L), np.concatenate(S))


def main():
    nb, w = build_laplacian()
    print("zero-shot harmonicity detector (no reference, no training)")
    print(f"{'noise':>6} {'2x2':>8} {'4x4':>8}")
    nrng = np.random.default_rng(7)
    for noise in NOISE_LEVELS:
        a2 = evaluate(nb, w, DIR_2X2, LBL_2X2, 200, noise, nrng)
        a4 = evaluate(nb, w, DIR_4X4, LBL_4X4, 100, noise, nrng)
        print(f"{noise:>6.1f} {a2:>8.4f} {a4:>8.4f}", flush=True)
    print("HARMONIC-DONE")


if __name__ == "__main__":
    main()
