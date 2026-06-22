"""
selfsuper_baseline.py — Idea C: self-supervised baseline estimation.

Problem
-------
Diff-norm requires a pristine baseline.  Bootstrap (Idea A) waits one flight.
For truly one-shot deployment (no first-flight data, no FEM), we need to
estimate the baseline from the raw signal itself.

Approach
--------
Train a GNN autoencoder on healthy-only specimens:
    Encoder: raw DSPSS → latent z  (graph-level)
    Decoder: latent z  → reconstructed healthy DSPSS  (node-level)

At inference on an unknown specimen:
    pseudo_diff = raw_dspss - AE_decoder(AE_encoder(raw_dspss))
             ≈ raw_dspss - healthy_prediction
             = residual (anomaly)

The residual should be near-zero for healthy nodes and non-zero at defects.
This is a "baseline-free" approximation to diff-norm.

Architecture
------------
    GNN Encoder: 3 SAGEConv layers, pool → 128-d latent
    GNN Decoder: 3 SAGEConv layers, decode to (N_nodes, 1)
    Loss: MSE on healthy specimens only (never train on defective data)

Training data
-------------
    Use oned_1x1_subtracted_zscore/ (1×1 healthy reference, no defect)
    as the healthy set.  Optionally augment with Stage-0 healthy detections.

Limitations (honest)
--------------------
- The AE reconstructs average healthy field, not specimen-specific geometry.
  Hole-edge stress concentration is structure-specific → the residual still
  has hole-edge artefacts (same problem as raw, but reduced).
- Performance gap vs true diff-norm expected to be ~5-10 F1 points.
- Best used as a fallback when neither FEM nor first-flight baseline exists.

Author: Nishioka (2026-06)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool

N_NODES   = 13942
LATENT_DIM = 128
HIDDEN     = 256

GD = Path("/home/nishioka/GNN")
DIR_HEALTHY = GD / "GNN_hole_2026/oned_1x1_subtracted_zscore"
DIR_COORD   = GD / "GNN_hole/GNN_hole_data"
CK_OUT      = GD / "GNN_hole_2026/GNN_model/selfsuper_ae"


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class GNNEncoder(nn.Module):
    """Node features → graph-level latent z."""

    def __init__(self, in_ch: int = 4, hidden: int = HIDDEN, latent: int = LATENT_DIM):
        super().__init__()
        self.conv1 = SAGEConv(in_ch,   hidden)
        self.conv2 = SAGEConv(hidden,  hidden)
        self.conv3 = SAGEConv(hidden,  latent)
        self.bn1   = nn.BatchNorm1d(hidden)
        self.bn2   = nn.BatchNorm1d(hidden)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = self.conv3(x, edge_index)
        return global_mean_pool(x, batch)   # (B, latent)


class GNNDecoder(nn.Module):
    """Graph-level latent z → per-node DSPSS reconstruction."""

    def __init__(self, latent: int = LATENT_DIM, hidden: int = HIDDEN, n_nodes: int = N_NODES):
        super().__init__()
        # Broadcast latent to each node, then refine with GNN
        self.node_proj = nn.Linear(latent, hidden)
        self.conv1     = SAGEConv(hidden, hidden)
        self.conv2     = SAGEConv(hidden, hidden)
        self.head      = nn.Linear(hidden, 1)
        self.n_nodes   = n_nodes

    def forward(self, z, edge_index, batch):
        # z: (B, latent) → broadcast to (B*N, hidden)
        # batch: (B*N,) mapping each node to its graph index
        z_per_node = self.node_proj(z[batch])          # (B*N, hidden)
        h = F.relu(self.conv1(z_per_node, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        return self.head(h).squeeze(-1)                 # (B*N,)


class SelfSupervisedAE(nn.Module):
    """Full autoencoder: encoder + decoder."""

    def __init__(self):
        super().__init__()
        # in_ch=4: [x, y, z, raw_dspss]
        self.encoder = GNNEncoder(in_ch=4, hidden=HIDDEN, latent=LATENT_DIM)
        self.decoder = GNNDecoder(latent=LATENT_DIM, hidden=HIDDEN, n_nodes=N_NODES)

    def forward(self, data: Data):
        z   = self.encoder(data.x, data.edge_index, data.batch)
        rec = self.decoder(z, data.edge_index, data.batch)
        return rec

    def pseudo_diff(self, data: Data) -> np.ndarray:
        """
        Compute pseudo-diff-norm = raw - AE_reconstruction.
        Returns per-node residual (N_nodes,).
        """
        self.eval()
        with torch.no_grad():
            rec = self.forward(data)
        raw = data.x[:, 3].cpu().numpy()   # column 3 = raw DSPSS
        residual = raw - rec.cpu().numpy()
        # per-node z-score
        s = residual.std()
        return (residual - residual.mean()) / (s + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def load_healthy(noise_std: float = 0.02):
    """Load healthy 1×1 specimens as training data (raw DSPSS, no labels needed)."""
    import os
    cx = np.load(DIR_COORD / "normalized_x_2layer.npy")[:N_NODES]
    cy = np.load(DIR_COORD / "normalized_y_2layer.npy")[:N_NODES]
    cz = np.load(DIR_COORD / "normalized_z_2layer.npy")[:N_NODES]
    edges = np.load(DIR_COORD / "hole_edges_2layer_best.npy")
    edge_index = torch.tensor(edges, dtype=torch.long)

    data_list = []
    for fname in sorted(os.listdir(DIR_HEALTHY)):
        if not fname.endswith(".npy"):
            continue
        vals = np.load(DIR_HEALTHY / fname)[:N_NODES].astype(np.float32)
        if noise_std > 0:
            vals = vals + np.random.normal(0, noise_std, N_NODES).astype(np.float32)
        feats = np.column_stack([cx, cy, cz, vals]).astype(np.float32)
        d = Data(
            x=torch.tensor(feats),
            edge_index=edge_index,
            y=torch.tensor(vals),   # target = clean raw (noise-free target)
        )
        data_list.append(d)
    return data_list


def train_ae(epochs: int = 100, lr: float = 1e-3, noise_std: float = 0.05) -> None:
    """Train the self-supervised autoencoder on healthy specimens."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data   = load_healthy(noise_std=noise_std)
    loader = DataLoader(data, batch_size=16, shuffle=True, num_workers=4)

    model = SelfSupervisedAE().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)

    CK_OUT.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            rec   = model(batch)
            # target: original (noise-free) raw stored in batch.y
            loss  = F.mse_loss(rec, batch.y)
            loss.backward()
            opt.step()
            total += loss.item()
        avg = total / len(loader)
        if epoch % 10 == 0:
            print(f"  epoch {epoch:3d}  MSE={avg:.5f}")
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), CK_OUT / "ae_best.pth")

    print(f"[AE] training done.  best_MSE={best_loss:.5f}")
    print("[AE] Use pseudo_diff() at inference to get baseline-free diff-norm.")


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests
# ─────────────────────────────────────────────────────────────────────────────

def _tests() -> None:
    print("[test] selfsuper_baseline unit tests...")

    # Encoder output shape
    N = N_NODES
    B = 2
    x   = torch.randn(N * B, 4)
    ei  = torch.zeros(2, 0, dtype=torch.long)   # no edges (isolation check)
    bat = torch.repeat_interleave(torch.arange(B), N)
    enc = GNNEncoder()
    z   = enc(x, ei, bat)
    assert z.shape == (B, LATENT_DIM), f"encoder output: {z.shape}"

    # Decoder output shape
    dec = GNNDecoder()
    out = dec(z, ei, bat)
    assert out.shape == (N * B,), f"decoder output: {out.shape}"

    # Full AE forward
    ae = SelfSupervisedAE()
    from torch_geometric.data import Batch
    d1 = Data(x=torch.randn(N, 4), edge_index=ei,
               y=torch.randn(N), batch=torch.zeros(N, dtype=torch.long))
    d2 = Data(x=torch.randn(N, 4), edge_index=ei,
               y=torch.randn(N), batch=torch.ones(N, dtype=torch.long))
    batch = Batch.from_data_list([
        Data(x=torch.randn(N, 4), edge_index=ei, y=torch.randn(N)),
        Data(x=torch.randn(N, 4), edge_index=ei, y=torch.randn(N)),
    ])
    rec = ae(batch)
    assert rec.shape == (N * 2,), f"AE output: {rec.shape}"

    print("[test] ALL PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--test",  action="store_true")
    ap.add_argument("--epochs",    type=int,   default=100)
    ap.add_argument("--noise_std", type=float, default=0.05)
    args = ap.parse_args()

    if args.test:
        _tests()
    else:
        train_ae(epochs=args.epochs, noise_std=args.noise_std)
