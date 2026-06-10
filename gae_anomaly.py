"""
Graph autoencoder anomaly detection — the deep NODE-space entry of the zoo.

GCN encoder-decoder trained on healthy (1x1) fields to reconstruct node values
from (value, x, y); anomaly score = per-node reconstruction error. Works on the
actual mesh graph (hole_edges), so no grid-projection signal loss — the fair
deep competitor to per-node statistics.

Run (vancouver): python3 gae_anomaly.py
"""
import os
import glob
import socket
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import SAGEConv

BASE    = "/home/nishioka/CFRP/CFRP_hole/hole_data_inp"
DIR_1X1 = f"{BASE}/Defect_hole_1x1_Random_npy"
DIR_2X2 = f"{BASE}/Defect_hole_2x2_Region1_21_npy"
DIR_4X4 = f"{BASE}/Defect_hole_4x4_Region1_21_npy"
if socket.gethostname().startswith("vancouver"):
    LBL_2X2 = LBL_4X4 = "/home/nishioka/GNN/GNN_hole_2026/all_19class_label"
    COORD_DIR = "/home/nishioka/GNN/GNN_hole/GNN_hole_data"
    EDGE_FILE = "/home/nishioka/GNN/GNN_hole/GNN_hole_data/hole_edges_2layer_best.npy"
else:
    LBL_2X2 = f"{BASE}/Def2x2_19class_label"
    LBL_4X4 = f"{BASE}/Def4x4_19class_label"
    COORD_DIR = f"{BASE}/basicdata_for_holegnn"
    EDGE_FILE = None   # local fallback: knn graph

N_NODES = 13942
N_REF   = 2000
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
NOISE_LEVELS = [0.0, 0.1, 0.3, 0.5]
MASK_P  = 0.3          # input value dropout → denoising objective
rng = np.random.default_rng(42)


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


def edge_index():
    if EDGE_FILE and os.path.exists(EDGE_FILE):
        e = np.load(EDGE_FILE)
        if e.shape[0] != 2:
            e = e.T
        e = e[:, (e[0] < N_NODES) & (e[1] < N_NODES)]
        return torch.from_numpy(np.ascontiguousarray(e)).long()
    from sklearn.neighbors import NearestNeighbors
    x = np.load(f"{COORD_DIR}/normalized_x_2layer.npy")[:N_NODES]
    y = np.load(f"{COORD_DIR}/normalized_y_2layer.npy")[:N_NODES]
    nn_ = NearestNeighbors(n_neighbors=7).fit(np.stack([x, y], 1))
    _, idx = nn_.kneighbors(np.stack([x, y], 1))
    src = np.repeat(np.arange(N_NODES), 6)
    dst = idx[:, 1:].reshape(-1)
    return torch.from_numpy(np.stack([src, dst])).long()


class GAE(nn.Module):
    def __init__(self, hidden=64, layers=4):
        super().__init__()
        self.convs = nn.ModuleList(
            [SAGEConv(3, hidden)] +
            [SAGEConv(hidden, hidden) for _ in range(layers - 2)] +
            [SAGEConv(hidden, 1)])

    def forward(self, x, ei):
        for c in self.convs[:-1]:
            x = F.relu(c(x, ei))
        return self.convs[-1](x, ei).squeeze(-1)


def main():
    x_c = np.load(f"{COORD_DIR}/normalized_x_2layer.npy")[:N_NODES].astype(np.float32)
    y_c = np.load(f"{COORD_DIR}/normalized_y_2layer.npy")[:N_NODES].astype(np.float32)
    coords = torch.from_numpy(np.stack([x_c, y_c], 1)).to(DEVICE)
    ei = edge_index().to(DEVICE)
    print(f"[gae] device={DEVICE}, edges={ei.shape[1]}")

    files = sorted(glob.glob(os.path.join(DIR_1X1, "*.npy")))
    idx   = rng.choice(len(files), min(N_REF, len(files)), replace=False)
    refs  = np.stack([load_field(files[i]) for i in idx])
    refs_t = torch.from_numpy(refs).to(DEVICE)

    model = GAE().to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)

    EPOCHS = 30
    for ep in range(EPOCHS):
        perm = torch.randperm(len(refs_t)); tot = 0
        model.train()
        for j in perm[:400]:                                    # 400 graphs/epoch
            v    = refs_t[j]
            mask = (torch.rand(N_NODES, device=DEVICE) < MASK_P)
            vin  = v.clone(); vin[mask] = 0.0
            x    = torch.cat([vin.unsqueeze(1), coords], dim=1)  # (N, 3)
            out  = model(x, ei)
            loss = F.mse_loss(out[mask], v[mask])                # denoise masked nodes
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        if (ep + 1) % 5 == 0:
            print(f"  epoch {ep+1}/{EPOCHS} loss={tot/400:.4f}", flush=True)

    @torch.no_grad()
    def score_field(v):
        """Average masked-reconstruction error over K random masks."""
        v_t = torch.from_numpy(v).to(DEVICE)
        acc = torch.zeros(N_NODES, device=DEVICE)
        cnt = torch.zeros(N_NODES, device=DEVICE)
        for _ in range(8):
            mask = (torch.rand(N_NODES, device=DEVICE) < MASK_P)
            vin  = v_t.clone(); vin[mask] = 0.0
            x    = torch.cat([vin.unsqueeze(1), coords], dim=1)
            out  = model(x, ei)
            acc[mask] += (out[mask] - v_t[mask]).abs()
            cnt[mask] += 1
        return (acc / cnt.clamp(min=1)).cpu().numpy()

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
    print("GAE-DONE")


if __name__ == "__main__":
    main()
