"""
CFRP Defect Detection via DDPM Anomaly Scoring
------------------------------------------------
Train a denoising diffusion model on near-normal (1x1 defect, ~100% class-0) samples.
At test time, partial-noise a sample and reconstruct; the pixelwise MSE becomes an anomaly map.
Large 2x2/4x4 defects produce high reconstruction error in the defect region.

Usage:
  python diffusion_anomaly.py --mode train
  python diffusion_anomaly.py --mode eval --ckpt results/ddpm_best.pt
"""

import os
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import griddata
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── paths (auto-detect local vs Vancouver) ─────────────────────────────────
import socket
_HOST = socket.gethostname()
_ON_VANCOUVER = _HOST.startswith("vancouver")

if _ON_VANCOUVER:
    DATA_BASE = "/home/nishioka/GNN/GNN_hole_2026"
    COORD_DIR = "/home/nishioka/GNN/GNN_hole/GNN_hole_data"
    NORM_1X1  = f"{DATA_BASE}/all_sub_hole_defect_zscore/train"
    RAW_2X2   = f"{DATA_BASE}/all_sub_hole_defect_zscore/test"   # mixed sizes → filtered by label
    RAW_4X4   = f"{DATA_BASE}/all_sub_hole_defect_zscore/test"
    LABEL_1X1 = f"{DATA_BASE}/all_19class_label/train"
    LABEL_2X2 = f"{DATA_BASE}/all_19class_label/test"
    LABEL_4X4 = f"{DATA_BASE}/all_19class_label/test"
    NODEFECT  = f"{DATA_BASE}/all_sub_hole_defect_zscore/train/hole_no_defect_original.npy"
else:
    DATA_BASE = "/home/nishioka/CFRP/CFRP_hole/hole_data_inp"
    COORD_DIR = f"{DATA_BASE}/basicdata_for_holegnn"
    NORM_1X1  = f"{DATA_BASE}/Defect_hole_1x1_subtracted_zscore"
    RAW_2X2   = f"{DATA_BASE}/Defect_hole_2x2_Region1_21_npy"
    RAW_4X4   = f"{DATA_BASE}/Defect_hole_4x4_Region1_21_npy"
    LABEL_1X1 = f"{DATA_BASE}/Def1x1_19class_label"
    LABEL_2X2 = f"{DATA_BASE}/Def2x2_19class_label"
    LABEL_4X4 = f"{DATA_BASE}/Def4x4_19class_label"
    NODEFECT  = f"{DATA_BASE}/hole_no_defect_data/hole_no_defect_original.npy"
RESULTS_DIR = "results/diffusion"

GRID_SIZE   = 64   # spatial grid resolution
T_MAX       = 1000 # diffusion timesteps
T_PARTIAL   = 300  # partial noising depth for anomaly scoring
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── spatial projection ──────────────────────────────────────────────────────
def load_coords():
    x = np.load(f"{COORD_DIR}/normalized_x_2layer.npy")
    y = np.load(f"{COORD_DIR}/normalized_y_2layer.npy")
    return x, y


def nodes_to_grid(dspss: np.ndarray, x: np.ndarray, y: np.ndarray,
                  grid_size: int = GRID_SIZE) -> np.ndarray:
    """Scatter 13942 node values onto a (grid_size x grid_size) 2D image."""
    gx = np.linspace(0, 1, grid_size)
    gy = np.linspace(0, 1, grid_size)
    gx2d, gy2d = np.meshgrid(gx, gy)                  # (G,G)
    grid = griddata((x, y), dspss, (gx2d, gy2d), method="linear", fill_value=0.0)
    return grid.astype(np.float32)                     # (G,G)


def compute_nodefect_grid(x, y, grid_size=GRID_SIZE):
    """Compute z-score params from the no-defect sample for later normalisation."""
    raw = np.load(NODEFECT).astype(np.float32)
    mu  = raw.mean()
    sig = raw.std() + 1e-8
    nd_norm = (raw - mu) / sig
    return nodes_to_grid(nd_norm, x, y, grid_size), mu, sig


# ─── dataset ─────────────────────────────────────────────────────────────────
class CFRPGridDataset(Dataset):
    """
    Load pre-normalised (subtracted + z-scored) .npy files and project to grid.
    Returns (1, G, G) tensors.
    """
    def __init__(self, data_dir: str, x, y, max_samples: int = None):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        if max_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(self.files), min(max_samples, len(self.files)), replace=False)
            self.files = [self.files[i] for i in sorted(idx)]
        self.x, self.y = x, y

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        dspss = np.load(self.files[i]).astype(np.float32)[:13942]
        grid  = nodes_to_grid(dspss, self.x, self.y)          # (G,G)
        # clip to [-5,5] for training stability
        grid  = np.clip(grid, -5.0, 5.0) / 5.0                # → [-1,1]
        return torch.from_numpy(grid).unsqueeze(0)             # (1,G,G)

    def filename(self, i):
        return self.files[i]


# ─── UNet with sinusoidal time embedding ────────────────────────────────────
class SinusoidalEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(half) / half)
        self.register_buffer("freqs", freqs)

    def forward(self, t):
        emb = t[:, None] * self.freqs[None]           # (B, half)
        return torch.cat([emb.sin(), emb.cos()], dim=1)  # (B, dim)


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_c)
        self.res = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x, emb):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = h + self.emb_proj(F.silu(emb))[:, :, None, None]
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return h + self.res(x)


class UNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=64, emb_dim=128, levels=4):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalEmbed(emb_dim), nn.Linear(emb_dim, emb_dim * 4),
            nn.SiLU(), nn.Linear(emb_dim * 4, emb_dim)
        )
        chs = [base_ch * (2 ** i) for i in range(levels)]   # 64,128,256,512

        # encoder
        self.enc_in  = nn.Conv2d(in_ch, chs[0], 3, padding=1)
        self.enc_blk = nn.ModuleList([ConvBlock(chs[i], chs[i], emb_dim) for i in range(levels)])
        self.downs   = nn.ModuleList([nn.Conv2d(chs[i], chs[i+1], 3, stride=2, padding=1)
                                      for i in range(levels - 1)])

        # bottleneck
        self.mid = ConvBlock(chs[-1], chs[-1], emb_dim)

        # decoder
        self.ups     = nn.ModuleList([nn.ConvTranspose2d(chs[i+1], chs[i], 2, stride=2)
                                      for i in reversed(range(levels - 1))])
        self.dec_blk = nn.ModuleList([ConvBlock(chs[i] * 2, chs[i], emb_dim)
                                      for i in reversed(range(levels - 1))])
        self.out     = nn.Conv2d(chs[0], in_ch, 1)

    def forward(self, x, t):
        emb  = self.time_embed(t.float())
        h    = self.enc_in(x)
        skips = []
        for blk, down in zip(self.enc_blk[:-1], self.downs):
            h = blk(h, emb); skips.append(h); h = down(h)
        h = self.enc_blk[-1](h, emb)
        h = self.mid(h, emb)
        for up, blk, skip in zip(self.ups, self.dec_blk, reversed(skips)):
            h = up(h); h = torch.cat([h, skip], dim=1); h = blk(h, emb)
        return self.out(h)


# ─── DDPM noise schedule ─────────────────────────────────────────────────────
def cosine_schedule(T, s=0.008):
    steps = T + 1
    t     = np.linspace(0, T, steps) / T
    f     = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    alphas_cumprod = f / f[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return np.clip(betas, 0, 0.999).astype(np.float32)


class DDPM:
    def __init__(self, T=T_MAX, device=DEVICE):
        self.T      = T
        self.device = device
        betas       = torch.from_numpy(cosine_schedule(T)).to(device)
        alphas      = 1 - betas
        self.alpha_bar = torch.cumprod(alphas, dim=0)   # ᾱ_t

    def q_sample(self, x0, t, noise=None):
        """Forward process: x_t = sqrt(ᾱ_t) * x0 + sqrt(1 - ᾱ_t) * ε"""
        if noise is None:
            noise = torch.randn_like(x0)
        ab = self.alpha_bar[t][:, None, None, None]
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise, noise

    def training_loss(self, model, x0):
        t     = torch.randint(0, self.T, (x0.shape[0],), device=self.device)
        xt, noise = self.q_sample(x0, t)
        pred  = model(xt, t)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def partial_denoise(self, model, x0, t_start=T_PARTIAL):
        """
        Anomaly scoring: add noise up to t_start, then fully denoise.
        MSE(original, denoised) = anomaly map.
        """
        model.eval()
        t_tensor = torch.full((x0.shape[0],), t_start - 1, device=self.device, dtype=torch.long)
        xt = self.q_sample(x0, t_tensor)[0]
        x  = xt.clone()
        for t in range(t_start - 1, -1, -1):
            tb   = torch.full((x.shape[0],), t, device=self.device, dtype=torch.long)
            ab   = self.alpha_bar[t]
            ab_p = self.alpha_bar[t - 1] if t > 0 else torch.tensor(1.0)
            beta = 1 - ab / ab_p
            eps  = model(x, tb)
            x0_hat = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
            x0_hat = x0_hat.clamp(-1.5, 1.5)
            mean   = ab_p.sqrt() * x0_hat + (1 - ab_p).sqrt() * eps
            noise  = torch.randn_like(x) if t > 0 else 0
            x      = mean + beta.sqrt() * noise if t > 0 else mean
        return x   # denoised


# ─── training ────────────────────────────────────────────────────────────────
def train(args):
    print(f"[train] device={DEVICE}, grid={GRID_SIZE}x{GRID_SIZE}, T={T_MAX}")
    x_c, y_c = load_coords()

    ds     = CFRPGridDataset(NORM_1X1, x_c, y_c, max_samples=args.max_train)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    print(f"  training samples: {len(ds)}")

    model  = UNet(base_ch=args.base_ch).to(DEVICE)
    ddpm   = DDPM(T=T_MAX, device=DEVICE)
    optim  = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0
        for x0 in loader:
            x0   = x0.to(DEVICE)
            loss = ddpm.training_loss(model, x0)
            optim.zero_grad(); loss.backward(); optim.step()
            total += loss.item()
        sched.step()
        avg = total / len(loader)
        if epoch % 10 == 0:
            print(f"  epoch {epoch:4d}/{args.epochs}  loss={avg:.4f}")
        if avg < best_loss:
            best_loss = avg
            torch.save({"model": model.state_dict(), "epoch": epoch, "loss": avg},
                       f"{RESULTS_DIR}/ddpm_best.pt")

    print(f"[train] done, best loss={best_loss:.4f}")


# ─── evaluation ──────────────────────────────────────────────────────────────
def load_labels_onehot(label_dir, filename):
    """Return binary defect mask (1 = any defect class, 0 = class-0) per node."""
    base = os.path.splitext(os.path.basename(filename))[0]
    lf   = os.path.join(label_dir, f"{base}.npy")
    if not os.path.exists(lf):
        return None
    lbl = np.load(lf)          # (13942, 19)
    return (lbl[:, 0] == 0).astype(np.float32)   # 1 if defect (not class-0)


def eval_dataset(model, ddpm, raw_dir, label_dir, x_c, y_c, tag, n_samples=100):
    """Evaluate AUROC on a raw (not z-scored) dataset against node-level labels."""
    # Load no-defect for subtraction
    nd_raw = np.load(NODEFECT).astype(np.float32)
    mu     = nd_raw.mean(); sig = nd_raw.std() + 1e-8

    files = sorted(glob.glob(os.path.join(raw_dir, "*.npy")))[:n_samples]
    all_scores, all_labels = [], []

    for fpath in files:
        lbl = load_labels_onehot(label_dir, fpath)
        if lbl is None:
            continue
        raw    = np.load(fpath).astype(np.float32)[:13942]
        zscore = (raw - mu) / sig
        grid   = np.clip(nodes_to_grid(zscore, x_c, y_c), -5.0, 5.0) / 5.0
        x0     = torch.from_numpy(grid).unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            x_rec = ddpm.partial_denoise(model, x0, t_start=T_PARTIAL)

        # Per-pixel anomaly score
        score_map = (x0 - x_rec).abs().squeeze().cpu().numpy()   # (G,G)

        # Re-project grid score back to nodes using nearest neighbour
        gx  = np.linspace(0, 1, GRID_SIZE)
        gy  = np.linspace(0, 1, GRID_SIZE)
        xi  = (x_c * (GRID_SIZE - 1)).astype(int).clip(0, GRID_SIZE - 1)
        yi  = (y_c * (GRID_SIZE - 1)).astype(int).clip(0, GRID_SIZE - 1)
        node_scores = score_map[yi, xi]

        all_scores.append(node_scores)
        all_labels.append(lbl)

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    if labels.sum() == 0:
        print(f"  [{tag}] no positive labels found, skipping AUROC")
        return None
    auroc = roc_auc_score(labels, scores)
    print(f"  [{tag}] AUROC={auroc:.4f}  (n_samples={len(files)}, pos_frac={labels.mean():.5f})")
    return auroc


def visualise_anomaly(model, ddpm, fpath, label_dir, x_c, y_c, out_path):
    nd_raw = np.load(NODEFECT).astype(np.float32)
    mu = nd_raw.mean(); sig = nd_raw.std() + 1e-8

    raw    = np.load(fpath).astype(np.float32)[:13942]
    zscore = (raw - mu) / sig
    grid   = np.clip(nodes_to_grid(zscore, x_c, y_c), -5.0, 5.0) / 5.0
    x0     = torch.from_numpy(grid).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        x_rec = ddpm.partial_denoise(model, x0, t_start=T_PARTIAL)

    score_map = (x0 - x_rec).abs().squeeze().cpu().numpy()
    orig_map  = x0.squeeze().cpu().numpy()

    lbl = load_labels_onehot(label_dir, fpath)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(orig_map,  cmap="RdBu_r", origin="lower"); axes[0].set_title("DSPSS (z-score)")
    axes[1].imshow(score_map, cmap="hot",    origin="lower"); axes[1].set_title("Anomaly score")
    if lbl is not None:
        xi = (x_c * (GRID_SIZE-1)).astype(int).clip(0, GRID_SIZE-1)
        yi = (y_c * (GRID_SIZE-1)).astype(int).clip(0, GRID_SIZE-1)
        lbl_grid = np.zeros((GRID_SIZE, GRID_SIZE))
        lbl_grid[yi[lbl>0], xi[lbl>0]] = 1
        axes[2].imshow(lbl_grid, cmap="hot", origin="lower"); axes[2].set_title("Ground truth")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  saved → {out_path}")


def evaluate(args):
    ckpt  = torch.load(args.ckpt, map_location=DEVICE)
    model = UNet(base_ch=args.base_ch).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    ddpm  = DDPM(T=T_MAX, device=DEVICE)
    x_c, y_c = load_coords()

    print(f"[eval] checkpoint: {args.ckpt}  (epoch {ckpt['epoch']}, loss {ckpt['loss']:.4f})")
    eval_dataset(model, ddpm, RAW_2X2, LABEL_2X2, x_c, y_c, "2x2", n_samples=200)
    eval_dataset(model, ddpm, RAW_4X4, LABEL_4X4, x_c, y_c, "4x4", n_samples=100)

    # Visualise one 4x4 sample
    files_4x4 = sorted(glob.glob(os.path.join(RAW_4X4, "*.npy")))
    if files_4x4:
        visualise_anomaly(model, ddpm, files_4x4[0], LABEL_4X4, x_c, y_c,
                          f"{RESULTS_DIR}/anomaly_4x4_example.png")


# ─── entrypoint ──────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",       default="train", choices=["train", "eval"])
    p.add_argument("--ckpt",       default=f"{RESULTS_DIR}/ddpm_best.pt")
    p.add_argument("--epochs",     type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--base_ch",    type=int, default=32, help="UNet base channels (32 or 64)")
    p.add_argument("--max_train",  type=int, default=500, help="max training samples (None=all)")
    args = p.parse_args()

    if args.mode == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
