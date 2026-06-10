"""
Anomaly-detection zoo on fixed-geometry DSPSS fields — with a noise-robustness
axis where methods can actually separate.

Methods (all node-space, no grid projection):
  maha      per-node diagonal Gaussian                 (μ_i, σ_i)
  pca       PCA residual (k=32), whitened
  padim     PaDiM-style: per-node LOCAL feature (node + 8 spatial NN values),
            full-covariance Gaussian per node
  patchcore per-node nonparametric: distance to nearest reference local-feature
  flow      FastFlow-lite: RealNVP density on local features (shared across nodes)

Protocol: reference = 2000 plain z-scored 1x1 fields. Test = 2x2 (200) / 4x4 (100),
node-level AUROC. Noise sweep: additive Gaussian σ ∈ {0, 0.1, 0.3, 0.5} on TEST
fields only (reference stays clean = FEM; noisy test = measurement scenario).

Run: python3 anomaly_zoo.py            (CPU, ~30 min)
     python3 anomaly_zoo.py --quick    (reduced test sizes)
"""
import os
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

BASE      = "/home/nishioka/CFRP/CFRP_hole/hole_data_inp"
DIR_1X1   = f"{BASE}/Defect_hole_1x1_Random_npy"
DIR_2X2   = f"{BASE}/Defect_hole_2x2_Region1_21_npy"
DIR_4X4   = f"{BASE}/Defect_hole_4x4_Region1_21_npy"
LBL_2X2   = f"{BASE}/Def2x2_19class_label"
LBL_4X4   = f"{BASE}/Def4x4_19class_label"
COORD_DIR = f"{BASE}/basicdata_for_holegnn"
N_NODES   = 13942
N_REF     = 2000
K_LOCAL   = 8          # spatial neighbours per node for local features
PCA_K     = 32
NOISE_LEVELS = [0.0, 0.1, 0.3, 0.5]

rng = np.random.default_rng(42)


# ─── data ────────────────────────────────────────────────────────────────────
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


def neighbour_index():
    """(N_NODES, K_LOCAL+1) indices: node itself + K spatial nearest neighbours."""
    x = np.load(f"{COORD_DIR}/normalized_x_2layer.npy")[:N_NODES]
    y = np.load(f"{COORD_DIR}/normalized_y_2layer.npy")[:N_NODES]
    nn_ = NearestNeighbors(n_neighbors=K_LOCAL + 1).fit(np.stack([x, y], 1))
    _, idx = nn_.kneighbors(np.stack([x, y], 1))
    return idx                                            # includes self at [:,0]


def local_features(V, nb_idx):
    """V: (..., N_NODES) → (..., N_NODES, K_LOCAL+1) local patch vectors."""
    return V[..., nb_idx]                                  # fancy-index gather


# ─── methods (each: fit(X_ref) → score(v) per-node) ─────────────────────────
class Maha:
    name = "maha"
    def fit(self, X, nb_idx):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
    def score(self, v):
        return np.abs(v - self.mu) / self.sd


class PCAResid:
    name = "pca"
    def fit(self, X, nb_idx):
        self.mu = X.mean(0); self.sd = X.std(0) + 1e-6
        Xc = X - self.mu
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.V = Vt[:PCA_K]
    def score(self, v):
        c = v - self.mu
        rec = (c @ self.V.T) @ self.V
        return np.abs(c - rec) / self.sd


class PaDiM:
    name = "padim"
    def fit(self, X, nb_idx):
        F = local_features(X, nb_idx)                      # (R, N, D)
        self.mu = F.mean(0)                                # (N, D)
        D = F.shape[2]
        cov = np.einsum("rnd,rne->nde", F - self.mu, F - self.mu) / (len(X) - 1)
        cov += 1e-3 * np.eye(D)[None]
        self.icov = np.linalg.inv(cov)                     # (N, D, D)
        self.nb_idx = nb_idx
    def score(self, v):
        f = local_features(v[None], self.nb_idx)[0] - self.mu   # (N, D)
        return np.sqrt(np.einsum("nd,nde,ne->n", f, self.icov, f))


class PatchCore:
    name = "patchcore"
    def fit(self, X, nb_idx):
        self.bank = local_features(X, nb_idx)              # (R, N, D)
        self.nb_idx = nb_idx
    def score(self, v):
        f = local_features(v[None], self.nb_idx)[0]        # (N, D)
        # per-node nearest reference distance, chunked over nodes
        out = np.empty(N_NODES, dtype=np.float32)
        B = 2000
        for i in range(0, N_NODES, B):
            d = ((self.bank[:, i:i+B, :] - f[None, i:i+B, :]) ** 2).sum(-1)  # (R, b)
            out[i:i+B] = np.sqrt(d.min(0))
        return out


class CouplingLayer(nn.Module):
    def __init__(self, dim, hidden=64, flip=False):
        super().__init__()
        self.flip = flip
        half = dim // 2
        self.net = nn.Sequential(
            nn.Linear(dim - half, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * half))
        self.half = half
    def forward(self, x):
        # conditioning part a has (dim - half) features, transformed part b has half
        if self.flip:
            a, b = x[:, self.half:], x[:, :self.half]
        else:
            a, b = x[:, :x.shape[1] - self.half], x[:, x.shape[1] - self.half:]
        st   = self.net(a)
        s, t = st.chunk(2, dim=1)
        s    = torch.tanh(s)
        b2   = b * torch.exp(s) + t
        out  = torch.cat([b2, a] if self.flip else [a, b2], dim=1)
        return out, s.sum(1)


class FlowLite:
    """RealNVP on local features, shared across nodes (FastFlow spirit)."""
    name = "flow"
    def fit(self, X, nb_idx, epochs=5, lr=1e-3):
        self.nb_idx = nb_idx
        F = local_features(X, nb_idx).reshape(-1, K_LOCAL + 1)
        self.f_mu = F.mean(0); self.f_sd = F.std(0) + 1e-6
        F = (F - self.f_mu) / self.f_sd
        sub = F[rng.choice(len(F), min(400_000, len(F)), replace=False)]
        D = K_LOCAL + 1
        self.layers = nn.ModuleList([CouplingLayer(D, flip=bool(i % 2)) for i in range(4)])
        opt = torch.optim.Adam(self.layers.parameters(), lr=lr)
        data = torch.from_numpy(sub)
        for ep in range(epochs):
            perm = torch.randperm(len(data))
            tot = 0
            for i in range(0, len(data), 4096):
                x = data[perm[i:i+4096]]
                logdet = 0
                for L in self.layers:
                    x, ld = L(x); logdet = logdet + ld
                nll = (0.5 * (x ** 2).sum(1) - logdet).mean()
                opt.zero_grad(); nll.backward(); opt.step()
                tot += nll.item()
            print(f"    flow epoch {ep+1}/{epochs} nll={tot/(len(data)//4096+1):.3f}")
    @torch.no_grad()
    def score(self, v):
        f = local_features(v[None], self.nb_idx)[0]
        f = (f - self.f_mu) / self.f_sd
        x = torch.from_numpy(f.astype(np.float32))
        logdet = 0
        for L in self.layers:
            x, ld = L(x); logdet = logdet + ld
        return (0.5 * (x ** 2).sum(1) - logdet).numpy()    # NLL per node


# ─── evaluation ──────────────────────────────────────────────────────────────
def evaluate(method, raw_dir, label_dir, tag, n_samples, noise):
    files = sorted(glob.glob(os.path.join(raw_dir, "*.npy")))[:n_samples]
    scores, labels = [], []
    nrng = np.random.default_rng(7)
    for f in files:
        lbl = load_label(label_dir, f)
        if lbl is None:
            continue
        v = load_field(f)
        if noise > 0:
            v = v + nrng.normal(0, noise, v.shape).astype(np.float32)
        scores.append(method.score(v))
        labels.append(lbl)
    s, l = np.concatenate(scores), np.concatenate(labels)
    return roc_auc_score(l, s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    n2, n4 = (50, 30) if args.quick else (200, 100)

    files = sorted(glob.glob(os.path.join(DIR_1X1, "*.npy")))
    idx = rng.choice(len(files), min(N_REF, len(files)), replace=False)
    X = np.stack([load_field(files[i]) for i in idx])
    nb_idx = neighbour_index()
    print(f"reference: {len(X)} 1x1 fields, local patch D={K_LOCAL+1}")

    methods = [Maha(), PCAResid(), PaDiM(), PatchCore(), FlowLite()]
    results = {}
    for m in methods:
        print(f"\n=== {m.name}: fitting ===")
        m.fit(X, nb_idx)
        for noise in NOISE_LEVELS:
            a2 = evaluate(m, DIR_2X2, LBL_2X2, "2x2", n2, noise)
            a4 = evaluate(m, DIR_4X4, LBL_4X4, "4x4", n4, noise)
            results[(m.name, noise)] = (a2, a4)
            print(f"  noise={noise:.1f}  2x2={a2:.4f}  4x4={a4:.4f}", flush=True)

    print("\n" + "=" * 64)
    print(f"{'method':10s}" + "".join(f"  σ={n:<4.1f}(2x2/4x4) " for n in NOISE_LEVELS))
    for m in methods:
        row = f"{m.name:10s}"
        for n in NOISE_LEVELS:
            a2, a4 = results[(m.name, n)]
            row += f"  {a2:.3f}/{a4:.3f}    "
        print(row)
    print("\ndiffusion FM plain g128 (σ=0 ref): 2x2 0.503 / 4x4 0.663")


if __name__ == "__main__":
    main()
