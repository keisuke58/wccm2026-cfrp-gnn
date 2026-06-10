"""
Per-node Gaussian / PCA-residual anomaly detection — the right baseline for
fixed-geometry data.

All 13942-node DSPSS fields share ONE specimen geometry, so the healthy
manifold is a point cloud around a single template. No deep model needed:
  score 1 (Mahalanobis): per-node z = |x_i − μ_i| / σ_i over the 1x1 reference set
  score 2 (PCA residual): project field onto top-k healthy PCs, residual per node

Reference set = 1x1-defect fields (defect occupies ~3/13942 nodes ≈ noise).
Eval identical to diffusion_anomaly: node-level AUROC on 2x2 / 4x4.
Runs on CPU in minutes — direct comparison against diffusion (best 4x4: 0.663).

Run: python3 mahalanobis_anomaly.py
"""
import os
import glob
import numpy as np
from sklearn.metrics import roc_auc_score

BASE      = "/home/nishioka/CFRP/CFRP_hole/hole_data_inp"
DIR_1X1   = f"{BASE}/Defect_hole_1x1_Random_npy"
DIR_2X2   = f"{BASE}/Defect_hole_2x2_Region1_21_npy"
DIR_4X4   = f"{BASE}/Defect_hole_4x4_Region1_21_npy"
LBL_2X2   = f"{BASE}/Def2x2_19class_label"
LBL_4X4   = f"{BASE}/Def4x4_19class_label"
N_NODES   = 13942
N_REF     = 2000     # reference samples (same budget as diffusion training)
PCA_K     = 32


def load_field(fpath):
    v = np.load(fpath).astype(np.float32)[:N_NODES]
    return (v - v.mean()) / (v.std() + 1e-8)        # per-sample z-score (plain)


def load_label(label_dir, fpath):
    base = os.path.splitext(os.path.basename(fpath))[0]
    for cand in (f"{base}.npy", f"{base}_19label.npy"):
        lf = os.path.join(label_dir, cand)
        if os.path.exists(lf):
            lbl = np.load(lf)
            return (lbl[:, 1:].sum(axis=1) > 0).astype(np.float32)
    return None


def evaluate(score_fn, raw_dir, label_dir, tag, n_samples):
    files = sorted(glob.glob(os.path.join(raw_dir, "*.npy")))[:n_samples]
    scores, labels = [], []
    for f in files:
        lbl = load_label(label_dir, f)
        if lbl is None:
            continue
        scores.append(score_fn(load_field(f)))
        labels.append(lbl)
    s, l = np.concatenate(scores), np.concatenate(labels)
    auroc = roc_auc_score(l, s)
    print(f"  [{tag}] AUROC={auroc:.4f}  (n={len(files)}, pos_frac={l.mean():.5f})")
    return auroc


def main():
    rng   = np.random.default_rng(42)
    files = sorted(glob.glob(os.path.join(DIR_1X1, "*.npy")))
    idx   = rng.choice(len(files), min(N_REF, len(files)), replace=False)
    print(f"reference set: {len(idx)} 1x1 fields")
    X = np.stack([load_field(files[i]) for i in idx])       # (N_REF, 13942)

    # ── method 1: per-node Gaussian (Mahalanobis, diagonal) ──────────────────
    mu, sd = X.mean(0), X.std(0) + 1e-6

    def maha(v):
        return np.abs(v - mu) / sd

    print("\n── per-node Mahalanobis ──")
    evaluate(maha, DIR_2X2, LBL_2X2, "2x2", 200)
    evaluate(maha, DIR_4X4, LBL_4X4, "4x4", 100)

    # ── method 2: PCA residual ───────────────────────────────────────────────
    Xc = X - mu
    # economy SVD on (N_REF, 13942)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt[:PCA_K]                                          # (k, 13942)
    ev = (S[:PCA_K] ** 2).sum() / (S ** 2).sum()
    print(f"\n── PCA residual (k={PCA_K}, explained={ev:.3f}) ──")

    def pca_resid(v):
        c = v - mu
        rec = (c @ V.T) @ V
        return np.abs(c - rec) / sd                          # whitened residual

    evaluate(pca_resid, DIR_2X2, LBL_2X2, "2x2", 200)
    evaluate(pca_resid, DIR_4X4, LBL_4X4, "4x4", 100)

    print("\nreference: diffusion FM plain g128 best — 2x2 0.503 / 4x4 0.663")


if __name__ == "__main__":
    main()
