"""
STFPM (student-teacher feature pyramid matching) on DSPSS grids — the deep
grid-space entry of the anomaly zoo.

Teacher = ImageNet ResNet18 (frozen). Student = same arch, random init, trained
on healthy (1x1) grids to match teacher features at layer1-3. Anomaly map =
per-pixel feature discrepancy, upsampled, projected back to nodes.

Grid-space deep methods are expected to underperform node-space statistics on
this fixed-geometry data (cf. diffusion 0.503/0.663 vs maha 0.999/0.995) —
included to quantify the representation effect, plus the noise axis.

Run (vancouver): python3 stfpm_anomaly.py
"""
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from scipy.interpolate import griddata

import torchvision

BASE      = "/home/nishioka/CFRP/CFRP_hole/hole_data_inp"
DIR_1X1   = f"{BASE}/Defect_hole_1x1_Random_npy"
DIR_2X2   = f"{BASE}/Defect_hole_2x2_Region1_21_npy"
DIR_4X4   = f"{BASE}/Defect_hole_4x4_Region1_21_npy"
import socket
if socket.gethostname().startswith("vancouver"):
    LBL_2X2 = LBL_4X4 = "/home/nishioka/GNN/GNN_hole_2026/all_19class_label"
    COORD_DIR = "/home/nishioka/GNN/GNN_hole/GNN_hole_data"
else:
    LBL_2X2 = f"{BASE}/Def2x2_19class_label"
    LBL_4X4 = f"{BASE}/Def4x4_19class_label"
    COORD_DIR = f"{BASE}/basicdata_for_holegnn"

N_NODES = 13942
GRID    = 128
N_REF   = 2000
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
NOISE_LEVELS = [0.0, 0.1, 0.3, 0.5]
rng = np.random.default_rng(42)


def load_coords():
    x = np.load(f"{COORD_DIR}/normalized_x_2layer.npy")[:N_NODES]
    y = np.load(f"{COORD_DIR}/normalized_y_2layer.npy")[:N_NODES]
    return x, y


def field_to_grid(v, x, y):
    gx = np.linspace(0, 1, GRID); gy = np.linspace(0, 1, GRID)
    gx2, gy2 = np.meshgrid(gx, gy)
    g = griddata((x, y), v, (gx2, gy2), method="linear", fill_value=0.0)
    return np.clip(g, -5, 5).astype(np.float32) / 5.0


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


class FeatureExtractor(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        r = torchvision.models.resnet18(weights=w)
        self.stem   = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.layer1, self.layer2, self.layer3 = r.layer1, r.layer2, r.layer3

    def forward(self, x):
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        return [f1, f2, f3]


def to3ch(g):
    t = torch.from_numpy(g).unsqueeze(0)        # (1, G, G)
    return t.repeat(3, 1, 1)


def main():
    x_c, y_c = load_coords()
    files = sorted(glob.glob(os.path.join(DIR_1X1, "*.npy")))
    idx   = rng.choice(len(files), min(N_REF, len(files)), replace=False)
    print(f"[stfpm] device={DEVICE}, refs={len(idx)}, grid={GRID}")

    teacher = FeatureExtractor(pretrained=True).to(DEVICE).eval()
    student = FeatureExtractor(pretrained=False).to(DEVICE)
    for p in teacher.parameters():
        p.requires_grad = False
    opt = torch.optim.SGD(student.parameters(), lr=0.4, momentum=0.9, weight_decay=1e-4)

    # pre-grid all reference fields once
    refs = np.stack([field_to_grid(load_field(files[i]), x_c, y_c) for i in idx])

    EPOCHS, BS = 20, 32
    for ep in range(EPOCHS):
        perm = rng.permutation(len(refs)); tot = 0
        student.train()
        for i in range(0, len(refs), BS):
            batch = torch.stack([to3ch(refs[j]) for j in perm[i:i+BS]]).to(DEVICE)
            with torch.no_grad():
                tf = teacher(batch)
            sf = student(batch)
            loss = sum(F.mse_loss(F.normalize(s, dim=1), F.normalize(t, dim=1))
                       for s, t in zip(sf, tf))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        print(f"  epoch {ep+1}/{EPOCHS} loss={tot/(len(refs)//BS+1):.4f}", flush=True)

    @torch.no_grad()
    def score_field(v):
        g = field_to_grid(v, x_c, y_c)
        b = to3ch(g).unsqueeze(0).to(DEVICE)
        tf, sf = teacher(b), student(b)
        amap = None
        for s, t in zip(sf, tf):
            d = 1 - F.cosine_similarity(s, t, dim=1, eps=1e-6)   # (1, h, w)
            d = F.interpolate(d.unsqueeze(1), size=GRID, mode="bilinear",
                              align_corners=False).squeeze()
            amap = d if amap is None else amap * d               # STFPM: product
        amap = amap.cpu().numpy()
        xi = (x_c * (GRID - 1)).astype(int).clip(0, GRID - 1)
        yi = (y_c * (GRID - 1)).astype(int).clip(0, GRID - 1)
        return amap[yi, xi]

    nrng = np.random.default_rng(7)
    print(f"\n{'noise':>6} {'2x2':>8} {'4x4':>8}")
    for noise in NOISE_LEVELS:
        aucs = []
        for raw_dir, lbl_dir, n in [(DIR_2X2, LBL_2X2, 200), (DIR_4X4, LBL_4X4, 100)]:
            fs = sorted(glob.glob(os.path.join(raw_dir, "*.npy")))[:n]
            S, L = [], []
            for f in fs:
                lbl = load_label(lbl_dir, f)
                if lbl is None:
                    continue
                v = load_field(f)
                if noise > 0:
                    v = v + nrng.normal(0, noise, v.shape).astype(np.float32)
                S.append(score_field(v)); L.append(lbl)
            aucs.append(roc_auc_score(np.concatenate(L), np.concatenate(S)))
        print(f"{noise:>6.1f} {aucs[0]:>8.4f} {aucs[1]:>8.4f}", flush=True)
    print("STFPM-DONE")


if __name__ == "__main__":
    main()
