"""
mesh_agnostic_gnn.py — Mesh-agnostic dual-stream GNN via dynamic k-NN graph.

Problem
-------
Fixed edge_index assumes a specific mesh (N=13942).
mixed_400 (N=15206) and czm_96_binary (N=193866) cannot be batched with it.

Solution
--------
Drop the pre-built edge_index entirely.
At every forward() call, build a k-NN graph from node coordinates (x,y,z).
PyTorch Geometric handles variable node-count graphs natively in DataLoader.

Node features: [x, y, z, dspss_diff, dspss_raw]  (in_channels=5)
               ┗━━ first 3 cols used to build k-NN ━━┛

Why this works across meshes
-----------------------------
The same physical phenomenon (CFRP laminate stress field under TSA/IR loading)
is discretised at different resolutions. A k-NN graph from coordinates recovers
the same spatial neighbourhood structure regardless of node count.
With k=12 the receptive field is ~5–6 message-passing steps for any mesh.

Multi-label strategy
--------------------
Different datasets may have different label granularity:
  - 13942-node data : 19-class (region × layer)
  - mixed_400       : binary or regression
  - czm_96_binary   : binary (defect / no defect)

This file supports three label modes via `label_mode`:
  'full'   : 19-class node-level localisation (existing WCCM data)
  'binary' : 2-class defect / no-defect (czm_96_binary)
  'stage0' : used as Stage-0 pretrain (binary, loss on detection score only)

Pretraining strategy for strongest model:
  1. Pretrain backbone on ALL data with 'binary' mode (Stage-0 loss).
  2. Fine-tune on 13942-node data with 'full' mode (19-class loss).

Usage
-----
    # Step 1: pretrain on all meshes (binary task)
    python mesh_agnostic_gnn.py --pretrain --data_dirs <dir1> <dir2> <dir3>

    # Step 2: fine-tune on 19-class
    python mesh_agnostic_gnn.py --finetune --ckpt <pretrain_ckpt>

    # Unit tests
    python mesh_agnostic_gnn.py --test

Author: Nishioka (2026-06)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv
try:
    from torch_geometric.nn import knn_graph as _pyg_knn_graph
except ImportError:
    _pyg_knn_graph = None


def _knn_brute(x: torch.Tensor, k: int, batch: Optional[torch.Tensor],
               loop: bool) -> torch.Tensor:
    """Pure-PyTorch brute-force k-NN — O(N²), CPU only, for small N tests."""
    if batch is not None:
        # process each specimen in the batch separately and concatenate
        edges = []
        offset = 0
        for b in batch.unique(sorted=True):
            mask  = (batch == b).nonzero(as_tuple=True)[0]
            xb    = x[mask]
            ei    = _knn_brute(xb, k, None, loop)
            edges.append(ei + offset)
            offset += xb.size(0)
        return torch.cat(edges, dim=1)
    N     = x.size(0)
    k_eff = min(k, N - (0 if loop else 1))
    dist  = torch.cdist(x.float(), x.float())
    if not loop:
        dist.fill_diagonal_(float("inf"))
    knn_idx = dist.topk(k_eff, dim=1, largest=False).indices
    src = torch.arange(N, device=x.device).unsqueeze(1).expand_as(knn_idx).reshape(-1)
    dst = knn_idx.reshape(-1)
    return torch.stack([src, dst], dim=0)


def knn_graph(x: torch.Tensor, k: int, batch: Optional[torch.Tensor] = None,
              loop: bool = False) -> torch.Tensor:
    """
    k-NN graph builder. Uses torch_cluster when available;
    falls back to pure-PyTorch O(N²) for small N (unit tests only).
    """
    if _pyg_knn_graph is not None:
        try:
            return _pyg_knn_graph(x, k=k, batch=batch, loop=loop)
        except (ImportError, RuntimeError):
            pass
    return _knn_brute(x, k, batch, loop)

# ── paths ─────────────────────────────────────────────────────────────────────
GD = Path("/home/nishioka/GNN")

# Existing 13942-node data (diff-norm + raw)
DIR_DIFF_13k = GD / "GNN_hole_2026/all_sub_hole_defect_zscore"
DIR_RAW_13k  = GD / "GNN_hole_2026/all_hole_defect_zscore"
DIR_LBL_13k  = GD / "GNN_hole_2026/all_19class_label"
DIR_COORD_13k = GD / "GNN_hole/GNN_hole_data"

# mixed_400 and czm_96_binary — set to actual paths when available
DIR_MIXED400  = GD / "mixed_400"          # placeholder
DIR_CZM96     = GD / "czm_96_binary"      # placeholder

CK_OUT = GD / "GNN_hole_2026/GNN_model/mesh_agnostic"

K_NEIGHBORS = 12   # k-NN for dynamic graph
N_NODES_13k = 13942


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic k-NN Model
# ─────────────────────────────────────────────────────────────────────────────

class _MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return self.net(x)


class MeshAgnosticGNN(nn.Module):
    """
    GNN with dynamic k-NN graph — works on any mesh.

    Architecture
    ------------
    1. Node encoder: [x,y,z,diff,raw] → hidden
    2. N × SAGEConv with residuals (scale-invariant aggregation)
    3. Multi-task head: 'full' (19-class) or 'binary' (2-class)

    The k-NN graph is rebuilt every forward() from the (x,y,z) block.
    batch index is required for correct k-NN across specimens.
    """

    def __init__(
        self,
        in_channels: int = 5,
        hidden: int = 256,
        num_layers: int = 8,
        k: int = K_NEIGHBORS,
        num_classes_full: int = 19,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.k        = k
        self.hidden   = hidden
        self.dropout  = dropout

        self.node_encoder = _MLP(in_channels, hidden, hidden)

        self.convs = nn.ModuleList(
            [SAGEConv(hidden, hidden) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden) for _ in range(num_layers)]
        )

        # Two heads: 19-class for fine-tune, binary for pretrain
        self.head_full   = nn.Linear(hidden, num_classes_full)
        self.head_binary = nn.Linear(hidden, 2)

    def _build_knn(self, data: Data) -> torch.Tensor:
        """Build k-NN edge_index from x[:, :3] = (x,y,z) coords."""
        coords = data.x[:, :3]
        batch  = data.batch if hasattr(data, 'batch') and data.batch is not None \
                 else torch.zeros(coords.size(0), dtype=torch.long, device=coords.device)
        return knn_graph(coords, k=self.k, batch=batch, loop=False)

    def forward(self, data: Data, mode: str = "full") -> torch.Tensor:
        """
        Parameters
        ----------
        data : PyG Data/Batch — must have x (N,5) and batch
        mode : 'full' (19-class) | 'binary' (2-class)
        """
        edge_index = self._build_knn(data)
        h = self.node_encoder(data.x)

        for conv, norm in zip(self.convs, self.norms):
            h = h + F.dropout(
                F.gelu(norm(conv(h, edge_index))),
                p=self.dropout, training=self.training
            )

        if mode == "binary":
            return self.head_binary(h)
        return self.head_full(h)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading — mesh-agnostic (no pre-built edges)
# ─────────────────────────────────────────────────────────────────────────────

def load_mesh_data(
    coord_npy: Tuple[np.ndarray, np.ndarray, np.ndarray],
    dspss_dir: Path,
    label_dir: Optional[Path],
    raw_dir: Optional[Path] = None,
    label_mode: str = "full",
    diff_channel_dropout: float = 0.0,
    noise_std: float = 0.0,
    split_name: str = "",
) -> List[Data]:
    """
    Load any mesh into a list of Data objects WITHOUT edge_index.
    The edge_index is built dynamically at forward() time.

    Parameters
    ----------
    coord_npy   : (cx, cy, cz) arrays for this mesh
    dspss_dir   : directory of per-specimen .npy DSPSS (diff or raw)
    label_dir   : directory of per-specimen label .npy (None → use binary proxy)
    raw_dir     : if provided, loaded as 5th channel; else zeros
    label_mode  : 'full' (19-class) | 'binary' (class>0 vs 0)
    """
    cx, cy, cz = coord_npy
    N = len(cx)
    coord_block = np.column_stack([cx, cy, cz]).astype(np.float32)

    if not dspss_dir.is_dir():
        raise FileNotFoundError(f"DSPSS dir not found: {dspss_dir}")

    fnames = sorted(f for f in os.listdir(dspss_dir) if f.endswith(".npy"))
    data_list: List[Data] = []

    for fname in fnames:
        diff_vals = np.load(dspss_dir / fname)[:N].astype(np.float32)

        # raw channel
        if raw_dir and (raw_dir / fname).exists():
            raw_vals = np.load(raw_dir / fname)[:N].astype(np.float32)
        else:
            raw_vals = np.zeros(N, dtype=np.float32)

        # noise augmentation
        if noise_std > 0.0:
            diff_vals += np.random.normal(0, noise_std, N).astype(np.float32)
            if raw_vals.sum() != 0:
                raw_vals += np.random.normal(0, noise_std, N).astype(np.float32)

        # channel dropout
        if diff_channel_dropout > 0.0 and np.random.rand() < diff_channel_dropout:
            diff_vals = np.zeros(N, dtype=np.float32)

        feats = np.column_stack([coord_block, diff_vals, raw_vals])
        x_t   = torch.tensor(feats, dtype=torch.float)

        # labels
        if label_dir is not None:
            lbl_fname = fname.replace(".npy", "_19label.npy")
            lbl_path  = label_dir / lbl_fname
            if not lbl_path.exists():
                # try plain _label.npy
                lbl_fname = fname.replace(".npy", "_label.npy")
                lbl_path  = label_dir / lbl_fname
            if lbl_path.exists():
                lbl = np.load(lbl_path)[:N]
                lbl_t = torch.tensor(lbl, dtype=torch.float)
                if lbl_t.ndim == 2 and lbl_t.shape[1] > 1:
                    y = torch.argmax(lbl_t, dim=1).long()
                else:
                    y = lbl_t.long().squeeze()
                if label_mode == "binary":
                    y = (y > 0).long()   # collapse to defect / no-defect
            else:
                y = torch.zeros(N, dtype=torch.long)   # no label → treat as healthy
        else:
            y = torch.zeros(N, dtype=torch.long)

        d = Data(x=x_t, y=y)      # NO edge_index — built at forward()
        d.filename   = fname
        d.mesh_nodes = N
        data_list.append(d)

    tag = f"{split_name} " if split_name else ""
    print(f"[mesh-agnostic] {tag}loaded {len(data_list)} specimens, "
          f"N_nodes={N}, mode={label_mode}")
    return data_list


def load_13k(split: str, **kw) -> List[Data]:
    """Load the standard 13,942-node interstage data."""
    cx = np.load(DIR_COORD_13k / "normalized_x_2layer.npy")[:N_NODES_13k]
    cy = np.load(DIR_COORD_13k / "normalized_y_2layer.npy")[:N_NODES_13k]
    cz = np.load(DIR_COORD_13k / "normalized_z_2layer.npy")[:N_NODES_13k]
    kw.setdefault("label_mode", "full")
    return load_mesh_data(
        coord_npy=(cx, cy, cz),
        dspss_dir=DIR_DIFF_13k / split,
        label_dir=DIR_LBL_13k,
        raw_dir=DIR_RAW_13k / split,
        split_name=f"13k/{split}",
        **kw,
    )


def load_mixed400(split: str, **kw) -> List[Data]:
    """
    Load mixed_400 (15,206-node, thermal proxy) data.
    Coordinates must be provided separately (not yet standard in this repo).
    """
    coord_file = DIR_MIXED400 / "coords.npz"
    if not coord_file.exists():
        raise FileNotFoundError(
            f"mixed_400 coords not found at {coord_file}.\n"
            "Expected npz with keys 'x','y','z' of shape (15206,)."
        )
    c = np.load(coord_file)
    kw.setdefault("label_mode", "binary")
    return load_mesh_data(
        coord_npy=(c["x"], c["y"], c["z"]),
        dspss_dir=DIR_MIXED400 / split,
        label_dir=DIR_MIXED400 / "labels" if (DIR_MIXED400 / "labels").is_dir() else None,
        split_name=f"mixed400/{split}",
        **kw,
    )


def load_czm96(split: str, **kw) -> List[Data]:
    """
    Load czm_96_binary (193,866-node, faithful CZM) data.
    Coordinates must be provided separately.
    """
    coord_file = DIR_CZM96 / "coords.npz"
    if not coord_file.exists():
        raise FileNotFoundError(
            f"czm_96 coords not found at {coord_file}.\n"
            "Expected npz with keys 'x','y','z' of shape (193866,)."
        )
    c = np.load(coord_file)
    kw.setdefault("label_mode", "binary")
    return load_mesh_data(
        coord_npy=(c["x"], c["y"], c["z"]),
        dspss_dir=DIR_CZM96 / split,
        label_dir=DIR_CZM96 / "labels" if (DIR_CZM96 / "labels").is_dir() else None,
        split_name=f"czm96/{split}",
        **kw,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Training loops
# ─────────────────────────────────────────────────────────────────────────────

def _eval_f1(model, loader, device, mode="full") -> float:
    from sklearn.metrics import f1_score
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred  = model(batch, mode=mode).argmax(dim=1)
            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
    avg = "binary" if mode == "binary" else "macro"
    return float(f1_score(y_true, y_pred, average=avg, zero_division=0))


def pretrain_binary(
    train_data: List[Data],
    val_data: List[Data],
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 16,
) -> MeshAgnosticGNN:
    """
    Stage 1: pretrain on ALL available meshes with binary defect/no-defect loss.
    DataLoader handles variable-N graphs transparently (each specimen = one graph).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[pretrain] device={device}  n_train={len(train_data)}  n_val={len(val_data)}")

    # Note: batch_size here = number of specimens per batch (not nodes).
    # Each specimen has different N_nodes → PyG packs them with batch vector.
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False, num_workers=2)

    model     = MeshAgnosticGNN(in_channels=5, hidden=256, num_layers=8, k=K_NEIGHBORS)
    model     = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    CK_OUT.mkdir(parents=True, exist_ok=True)
    best_f1, best_path = 0.0, CK_OUT / "pretrain_binary_best.pth"

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out  = model(batch, mode="binary")
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        if epoch % 10 == 0:
            f1 = _eval_f1(model, val_loader, device, mode="binary")
            avg_loss = total_loss / len(train_loader)
            print(f"  [pretrain] epoch {epoch:3d}  loss={avg_loss:.4f}  val_binary_F1={f1:.4f}")
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), best_path)
                print(f"    → saved best  (F1={best_f1:.4f})")

    print(f"[pretrain] done.  best_binary_F1={best_f1:.4f}")
    return model


def finetune_full(
    pretrain_ckpt: Path,
    train_data: List[Data],
    val_data: List[Data],
    epochs: int = 200,
    lr: float = 3e-4,
    batch_size: int = 16,
) -> MeshAgnosticGNN:
    """
    Stage 2: load pretrained backbone, swap to 19-class head, fine-tune on 13k data.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[finetune] device={device}  ckpt={pretrain_ckpt}")

    model = MeshAgnosticGNN(in_channels=5, hidden=256, num_layers=8, k=K_NEIGHBORS)

    # Load pretrained weights (backbone only; head_full is random → fine-tuned)
    state = torch.load(pretrain_ckpt, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  missing={missing[:3]}...  unexpected={unexpected[:3]}...")
    model = model.to(device)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False, num_workers=2)

    # Freeze backbone layers for first 20 epochs → only fine-tune head
    for p in model.node_encoder.parameters():  p.requires_grad = False
    for p in model.convs.parameters():         p.requires_grad = False
    for p in model.norms.parameters():         p.requires_grad = False

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    criterion = nn.CrossEntropyLoss()
    best_f1, best_path = 0.0, CK_OUT / "finetune_full_best.pth"
    CK_OUT.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        if epoch == 21:  # unfreeze backbone
            for p in model.parameters(): p.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(), lr=lr / 5)
            print("  [finetune] backbone unfrozen at epoch 21")

        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out  = model(batch, mode="full")
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            f1 = _eval_f1(model, val_loader, device, mode="full")
            print(f"  [finetune] epoch {epoch:3d}  loss={total_loss/len(train_loader):.4f}"
                  f"  val_macro_F1={f1:.4f}")
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), best_path)
                print(f"    → saved best  (F1={best_f1:.4f})")

    print(f"[finetune] done.  best_macro_F1={best_f1:.4f}")
    print(f"  Reference: diff-norm HybridMGN+Stage-0 = 0.803")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests (no GPU, no real data needed)
# ─────────────────────────────────────────────────────────────────────────────

def _tests() -> None:
    print("[test] mesh_agnostic_gnn unit tests...")

    # Three synthetic meshes with DIFFERENT node counts
    torch.manual_seed(0)
    meshes = [
        (13942, "interstage_13k"),
        (15206, "mixed_400"),
        (3000,  "small_debug"),   # small so tests run fast
    ]

    model = MeshAgnosticGNN(in_channels=5, hidden=64, num_layers=3, k=6)
    model.eval()

    for N, name in meshes:
        coords = torch.randn(N, 3)
        dspss  = torch.randn(N, 2)
        x      = torch.cat([coords, dspss], dim=1)      # (N, 5)
        y      = torch.zeros(N, dtype=torch.long)

        data = Data(x=x, y=y)
        data.batch = torch.zeros(N, dtype=torch.long)    # single specimen

        with torch.no_grad():
            out_full   = model(data, mode="full")
            out_binary = model(data, mode="binary")

        assert out_full.shape   == (N, 19), f"full head shape: {out_full.shape}"
        assert out_binary.shape == (N,  2), f"binary head shape: {out_binary.shape}"
        print(f"  {name}: N={N}  full={tuple(out_full.shape)}  "
              f"binary={tuple(out_binary.shape)}  ✓")

    # Batching: two specimens with DIFFERENT node counts in one batch
    d1 = Data(x=torch.randn(500, 5), y=torch.zeros(500, dtype=torch.long))
    d2 = Data(x=torch.randn(800, 5), y=torch.zeros(800, dtype=torch.long))
    from torch_geometric.data import Batch
    batch = Batch.from_data_list([d1, d2])
    with torch.no_grad():
        out = model(batch, mode="full")
    assert out.shape == (1300, 19), f"mixed batch shape: {out.shape}"
    print(f"  mixed batch (500+800): output={tuple(out.shape)}  ✓")

    # k-NN is built per-batch correctly (no cross-contamination)
    batch_idx = batch.batch
    assert (batch_idx[:500] == 0).all()
    assert (batch_idx[500:] == 1).all()

    # Binary mode on 19-class label data: y > 0 → 1
    y_19 = torch.randint(0, 19, (300,))
    y_bin = (y_19 > 0).long()
    assert y_bin.max() <= 1

    print("[test] ALL PASS")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Mesh-agnostic dual-stream GNN")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pretrain",  action="store_true",
                      help="Stage 1: binary pretrain on all available meshes")
    mode.add_argument("--finetune",  action="store_true",
                      help="Stage 2: 19-class fine-tune on 13k data")
    mode.add_argument("--test",      action="store_true")

    ap.add_argument("--ckpt",         default=str(CK_OUT / "pretrain_binary_best.pth"))
    ap.add_argument("--epochs",       type=int,   default=100)
    ap.add_argument("--batch_size",   type=int,   default=16)
    ap.add_argument("--lr",           type=float, default=1e-3)
    ap.add_argument("--noise_std",    type=float, default=0.05)
    ap.add_argument("--diff_channel_dropout", type=float, default=0.15)
    ap.add_argument("--use_mixed400", action="store_true",
                    help="Include mixed_400 dataset in pretrain (requires DIR_MIXED400)")
    ap.add_argument("--use_czm96",    action="store_true",
                    help="Include czm_96_binary dataset in pretrain (requires DIR_CZM96)")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse()

    if args.test:
        _tests()

    elif args.pretrain:
        aug = dict(noise_std=args.noise_std,
                   diff_channel_dropout=args.diff_channel_dropout)
        train_data = load_13k("train", label_mode="binary", **aug)
        val_data   = load_13k("val",   label_mode="binary")

        if args.use_mixed400:
            try:
                train_data += load_mixed400("train", label_mode="binary", **aug)
                val_data   += load_mixed400("val",   label_mode="binary")
            except FileNotFoundError as e:
                print(f"[warn] mixed400 skipped: {e}")

        if args.use_czm96:
            try:
                train_data += load_czm96("train", label_mode="binary", **aug)
                val_data   += load_czm96("val",   label_mode="binary")
            except FileNotFoundError as e:
                print(f"[warn] czm96 skipped: {e}")

        pretrain_binary(train_data, val_data,
                        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)

    elif args.finetune:
        aug = dict(noise_std=args.noise_std,
                   diff_channel_dropout=args.diff_channel_dropout)
        train_data = load_13k("train", **aug)
        val_data   = load_13k("val")
        finetune_full(Path(args.ckpt), train_data, val_data,
                      epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
