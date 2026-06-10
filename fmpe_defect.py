"""
FMPE — Flow Matching Posterior Estimation for defect characterisation.

Amortised Bayesian inverse problem on existing data (no new FEM):
  θ = (cx, cy, layer, log2 size)   ← label-mask centroid + filename L / H
  x = DSPSS field                  ← conditioned via PCA(64) of deviation field
  learn q(θ | x) with conditional flow matching; sample posterior by ODE.

Output per measurement: full posterior over defect position, DEPTH (layer)
and SIZE — "how severe is it", with uncertainty. Stage-2 of the pipeline
(Stage-0 maha screening → Stage-1 GNN classification → Stage-2 this).

Train on mixed 1x1/2x2/4x4 (80/20 split by sample).
Metrics: posterior-mean errors + 90%-interval coverage (calibration).

Run (vancouver): python3 fmpe_defect.py
"""
import os
import re
import glob
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

BASE      = "/home/nishioka/CFRP/CFRP_hole/hole_data_inp"
SETS = [  # (data_dir, label_dir, n_max)
    (f"{BASE}/Defect_hole_1x1_Random_npy",     f"{BASE}/Def1x1_19class_label", 4000),
    (f"{BASE}/Defect_hole_2x2_Region1_21_npy", f"{BASE}/Def2x2_19class_label", 4000),
    (f"{BASE}/Defect_hole_4x4_Region1_21_npy", f"{BASE}/Def4x4_19class_label", 3942),
]
COORD_DIR = f"{BASE}/basicdata_for_holegnn"
import socket
if socket.gethostname().startswith("vancouver"):
    for i, (d, l, n) in enumerate(SETS):
        SETS[i] = (d, "/home/nishioka/GNN/GNN_hole_2026/all_19class_label", n)
    COORD_DIR = "/home/nishioka/GNN/GNN_hole/GNN_hole_data"

N_NODES = 13942
PCA_K   = 64
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
RE_META = re.compile(r"_L(?P<L>\d+)_.*_H(?P<H>\d+)_W")
rng = np.random.default_rng(42)


# ─── data: build (θ, embedding) pairs ────────────────────────────────────────
def load_field(fpath):
    v = np.load(fpath).astype(np.float32)[:N_NODES]
    return (v - v.mean()) / (v.std() + 1e-8)


def load_mask(label_dir, fpath):
    base = os.path.splitext(os.path.basename(fpath))[0]
    for cand in (f"{base}.npy", f"{base}_19label.npy"):
        lf = os.path.join(label_dir, cand)
        if os.path.exists(lf):
            lbl = np.load(lf)
            return lbl[:, 1:].sum(axis=1) > 0
    return None


def build_dataset():
    x_c = np.load(f"{COORD_DIR}/normalized_x_2layer.npy")[:N_NODES]
    y_c = np.load(f"{COORD_DIR}/normalized_y_2layer.npy")[:N_NODES]

    fields, thetas = [], []
    for data_dir, label_dir, n_max in SETS:
        files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        idx = rng.choice(len(files), min(n_max, len(files)), replace=False)
        kept = 0
        for i in idx:
            f = files[i]
            m = RE_META.search(os.path.basename(f))
            mask = load_mask(label_dir, f)
            if m is None or mask is None or mask.sum() == 0:
                continue
            cx, cy = x_c[mask].mean(), y_c[mask].mean()
            L  = float(m.group("L"))
            Hs = float(m.group("H"))
            fields.append(load_field(f))
            thetas.append([cx, cy, L, np.log2(Hs)])
            kept += 1
        print(f"  {os.path.basename(data_dir)}: kept {kept}")
    return np.stack(fields), np.array(thetas, dtype=np.float32)


# ─── conditional flow matching in θ-space ────────────────────────────────────
class VelocityNet(nn.Module):
    def __init__(self, theta_dim=4, cond_dim=PCA_K, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(theta_dim + 1 + cond_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, theta_dim))

    def forward(self, th, t, c):
        return self.net(torch.cat([th, t[:, None], c], dim=1))


@torch.no_grad()
def sample_posterior(model, c, n_samples=256, steps=64):
    """c: (cond_dim,) → (n_samples, 4) posterior draws via Euler ODE 1→0."""
    cB = c[None].repeat(n_samples, 1)
    th = torch.randn(n_samples, 4, device=DEVICE)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((n_samples,), 1.0 - i * dt, device=DEVICE)
        th = th - model(th, t, cB) * dt
    return th


def main():
    print("[fmpe] building dataset…")
    fields, thetas = build_dataset()
    n = len(fields)
    print(f"  total pairs: {n}")

    # field embedding: PCA on deviation from global mean field
    mu_f = fields.mean(0)
    Xc = fields - mu_f
    _, _, Vt = np.linalg.svd(Xc[rng.choice(n, min(3000, n), replace=False)],
                             full_matrices=False)
    V = Vt[:PCA_K]
    embed = Xc @ V.T                                   # (n, 64)
    e_mu, e_sd = embed.mean(0), embed.std(0) + 1e-6
    embed = (embed - e_mu) / e_sd

    # θ standardisation
    t_mu, t_sd = thetas.mean(0), thetas.std(0) + 1e-6
    th_n = (thetas - t_mu) / t_sd

    idx = rng.permutation(n); n_tr = int(0.8 * n)
    tr, te = idx[:n_tr], idx[n_tr:]

    C  = torch.from_numpy(embed.astype(np.float32)).to(DEVICE)
    TH = torch.from_numpy(th_n.astype(np.float32)).to(DEVICE)

    model = VelocityNet().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)

    print("[fmpe] training conditional flow…")
    BS = 512
    for ep in range(200):
        perm = torch.randperm(n_tr, device=DEVICE)
        tot = 0
        for i in range(0, n_tr, BS):
            b   = torch.from_numpy(tr).to(DEVICE)[perm[i:i+BS]]
            th0 = TH[b]; c = C[b]
            t   = torch.rand(len(b), device=DEVICE)
            eps = torch.randn_like(th0)
            tht = (1 - t)[:, None] * th0 + t[:, None] * eps
            v   = model(tht, t, c)
            loss = ((v - (eps - th0)) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        sched.step()
        if (ep + 1) % 20 == 0:
            print(f"  epoch {ep+1}/200 loss={tot/(n_tr//BS+1):.4f}", flush=True)

    # ── evaluation on held-out test ──────────────────────────────────────────
    print("[fmpe] evaluating posterior on test set…")
    names = ["cx", "cy", "layer", "log2size"]
    means, lo90, hi90, truths = [], [], [], []
    for j in te[:500]:
        post = sample_posterior(model, C[j]).cpu().numpy() * t_sd + t_mu
        means.append(post.mean(0))
        lo90.append(np.percentile(post, 5, axis=0))
        hi90.append(np.percentile(post, 95, axis=0))
        truths.append(thetas[j])
    means, lo90, hi90, truths = map(np.array, (means, lo90, hi90, truths))

    print(f"\n{'param':>9} {'RMSE':>8} {'R2':>7} {'cov90':>7}")
    for k, name in enumerate(names):
        rmse = np.sqrt(((means[:, k] - truths[:, k]) ** 2).mean())
        r2   = r2_score(truths[:, k], means[:, k])
        cov  = ((truths[:, k] >= lo90[:, k]) & (truths[:, k] <= hi90[:, k])).mean()
        print(f"{name:>9} {rmse:>8.4f} {r2:>7.3f} {cov:>7.3f}")

    # size as classification (1/2/4)
    sz_pred = np.round(means[:, 3]).clip(0, 2)
    sz_true = np.round(truths[:, 3])
    print(f"\nsize accuracy (1x1/2x2/4x4): {(sz_pred == sz_true).mean():.3f}")
    lay_pred = np.round(means[:, 2])
    print(f"layer exact-match: {(lay_pred == truths[:, 2]).mean():.3f}, "
          f"|Δlayer|≤1: {(np.abs(lay_pred - truths[:, 2]) <= 1).mean():.3f}")
    print("FMPE-DONE")


if __name__ == "__main__":
    main()
