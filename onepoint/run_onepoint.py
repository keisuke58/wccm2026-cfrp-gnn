#!/usr/bin/env python
"""
One-point relative-coordinate GNN, headless port of GNN_onepoint-3-4.ipynb
(Sosuke Asano's notebook) adapted to the WCCM CFRP hole-defect dataset.

Approach (unchanged from the notebook):
  pick a seed node -> NeighborLoader builds a disjoint subgraph around it
  -> recenter coords to relative (x[:, :3] - seed_xyz) concat the DSPSS value
  -> 5-layer GATv2 (DeepGATModel) predicts the seed node's 19-class label.
  FocalLoss + class weights, per-epoch random seed sampling (defect / bg).

Adaptations vs notebook:
  * removed the undefined debug_check_seed_structure(...) call in validation
    (and the structure_ok_* variables it fed).
  * generalized data<->label pairing: key on the full file basename so it works
    for our names Defect_L{L}_B{B}_el{el}_H{H}_W{W}.npy (and the el-less variants
    Defect_L{L}_B{B}_H{H}_W{W}.npy), pairing data <-> Defect_..._19label.npy.
  * replaced Asano's k-fold / white-noise folders with our train/val folders.
  * coords files renamed to ours: normalized_{x,y,z}_2layer.npy.
  * argparse + headless logging.
"""

import os
import re
import sys
import random
import argparse
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data, Batch
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import f1_score

# Ensure the repo dir (which holds Loss/) is importable when run from the repo root.
_REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)

from Loss.focal_loss import FocalLoss  # noqa: E402

NUM_NODES = 13942
NUM_CLASSES = 19


# ==============================================================================
# ユーティリティ / utilities
# ==============================================================================
def extract_key(file_name):
    """Return a canonical key from a data or label file name.

    Data : Defect_L10_B100_el1165_H2_W2.npy        -> Defect_L10_B100_el1165_H2_W2
    Label: Defect_L10_B100_el1165_H2_W2_19label.npy -> Defect_L10_B100_el1165_H2_W2
    Also handles el-less variants: Defect_L10_B114_H4_W8(.npy).
    """
    name = file_name
    if name.endswith("_19label.npy"):
        name = name[: -len("_19label.npy")]
    elif name.endswith(".npy"):
        name = name[: -len(".npy")]
    # sanity: must look like a Defect_L..._B... descriptor
    if not re.match(r"Defect_L\d+_B\d+", name):
        return None
    return name


def create_data_label_pairs(data_files, label_files):
    data_label_pairs = {}
    unmatched_labels = set(label_files)

    for data_file in data_files:
        key = extract_key(data_file)
        if key is not None:
            data_label_pairs[key] = {"data": data_file}

    for label_file in label_files:
        if not label_file.endswith("_19label.npy"):
            continue
        key = extract_key(label_file)
        if key is not None and key in data_label_pairs:
            data_label_pairs[key]["label"] = label_file
            unmatched_labels.discard(label_file)

    no_label_counter = 0
    for _, v in data_label_pairs.items():
        if "label" not in v and no_label_counter < 3:
            print(f"No label found for data file: {v['data']}")
            no_label_counter += 1

    valid_pairs = [
        (v["data"], v["label"])
        for _, v in data_label_pairs.items()
        if "label" in v
    ]

    print(f"{len(valid_pairs)} matched pairs created. "
          f"{len(unmatched_labels)} labels were unused in this split.")
    return valid_pairs


def compute_class_weights(labels, num_classes=NUM_CLASSES):
    class_counts = np.bincount(labels, minlength=num_classes)
    class_counts = np.maximum(class_counts, 1)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.mean()
    return torch.tensor(class_weights, dtype=torch.float)


def prepare_data(pairs, normalized_data_folder, label_data_folder,
                 x_coords, y_coords, z_coords, edge_index):
    data_list = []
    labels = []
    pair_counter = 0

    for data_file, label_file in pairs:
        data_file_path = os.path.join(normalized_data_folder, data_file)
        label_file_path = os.path.join(label_data_folder, label_file)

        if not os.path.exists(data_file_path) or not os.path.exists(label_file_path):
            continue

        try:
            values = np.load(data_file_path)[:NUM_NODES]
            label = np.load(label_file_path)[:NUM_NODES]
        except Exception as e:
            print(f"データ読み込みエラー / data load error: {e}")
            continue

        try:
            node_features = np.vstack((x_coords, y_coords, z_coords, values)).T
            x = torch.tensor(node_features, dtype=torch.float)
            y = torch.argmax(torch.tensor(label, dtype=torch.float), dim=1).long()
        except Exception as e:
            print(f"特徴量作成エラー / feature build error: {e}")
            continue

        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
        labels.extend(y.tolist())

        if pair_counter < 1:
            print(f"Pair {pair_counter + 1}:")
            print(f"  Data file:  {data_file_path}")
            print(f"  Label file: {label_file_path}")
            print(f"  x shape: {x.shape}, y shape: {y.shape}")
            pair_counter += 1

    if len(labels) == 0:
        print("ラベルが存在しません / no labels; class-weight computation failed.")
        return None, None

    class_weights = compute_class_weights(np.array(labels), num_classes=NUM_CLASSES)
    return data_list, class_weights


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==============================================================================
# モデル / model
# ==============================================================================
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def edge_dropout(edge_index, drop_prob=0.0001):
    if drop_prob == 0:
        return edge_index
    mask = torch.rand(edge_index.size(1), device=edge_index.device) > drop_prob
    return edge_index[:, mask]


class DeepGATModel(torch.nn.Module):
    def __init__(self, hidden_channels=64, num_classes=NUM_CLASSES, num_layers=5, heads=4):
        super().__init__()
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.projs = nn.ModuleList()

        # --- first layer ---
        self.convs.append(GATv2Conv(4, hidden_channels, heads=heads, concat=True))
        self.norms.append(nn.BatchNorm1d(hidden_channels * heads))
        self.projs.append(nn.Linear(4, hidden_channels * heads))

        # --- middle layers ---
        for _ in range(num_layers - 2):
            self.convs.append(
                GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, concat=True)
            )
            self.norms.append(nn.BatchNorm1d(hidden_channels * heads))
            self.projs.append(nn.Identity())

        # --- last layer ---
        self.convs.append(
            GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, concat=True)
        )
        self.norms.append(nn.BatchNorm1d(hidden_channels * heads))
        self.projs.append(nn.Identity())

        self.fc = nn.Linear(hidden_channels * heads, num_classes)
        self.dropout = nn.Dropout(p=0.05)

        self.apply(initialize_weights)

    def forward(self, x, edge_index, target_indices=None):
        if self.training:
            edge_index = edge_dropout(edge_index, drop_prob=0.0001)

        for i in range(self.num_layers):
            x_residual = x

            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)

            if x_residual.size(1) != x.size(1):
                x_residual = self.projs[i](x_residual)

            x = x + x_residual

        if target_indices is not None:
            x = x[target_indices]

        x = self.fc(x)
        x = F.softmax(x, dim=-1)
        return x


# ==============================================================================
# Disjoint neighbor sampling
#
# The notebook uses torch_geometric.loader.NeighborLoader(..., disjoint=True),
# whose backend (pyg-lib / torch-sparse) is NOT installed in the available conda
# envs on this host. We therefore provide a pure-PyTorch fallback that produces
# the SAME batch contract the training loop relies on:
#   - batch.x          : node features of all nodes in the disjoint subgraphs
#   - batch.edge_index : edges, re-indexed to local node ids
#   - batch.y          : node labels (same order as batch.x)
#   - batch.batch      : subgraph id per node
#   - batch.batch_size : number of seeds; seeds occupy local ids 0..batch_size-1
# i.e. for each seed we build an independent (disjoint) subgraph of its sampled
# k-hop neighborhood, with the seed remapped to the front. This matches the
# behaviour the relative-coordinate recentering + target_indices logic expects.
# ==============================================================================
class DisjointNeighborLoader:
    def __init__(self, data, input_nodes, num_neighbors, batch_size,
                 shuffle=False, device="cpu"):
        self.x = data.x
        self.y = data.y
        self.num_neighbors = num_neighbors
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device

        if input_nodes.dtype == torch.bool:
            self.seeds = torch.nonzero(input_nodes, as_tuple=False).view(-1)
        else:
            self.seeds = input_nodes.view(-1)

        # build CSR-like adjacency (sorted by source) for fast neighbor lookup
        ei = data.edge_index
        order = torch.argsort(ei[0])
        self.src_sorted = ei[0][order]
        self.dst_sorted = ei[1][order]
        n = data.x.size(0)
        # ptr[i]..ptr[i+1] = slice of dst_sorted that are neighbors of node i
        counts = torch.bincount(self.src_sorted, minlength=n)
        self.ptr = torch.zeros(n + 1, dtype=torch.long)
        self.ptr[1:] = torch.cumsum(counts, dim=0)

    def _neighbors(self, node, k):
        s = int(self.ptr[node])
        e = int(self.ptr[node + 1])
        if e <= s:
            return torch.empty(0, dtype=torch.long)
        nbrs = self.dst_sorted[s:e]
        if k < 0 or nbrs.numel() <= k:
            return nbrs
        idx = torch.randperm(nbrs.numel())[:k]
        return nbrs[idx]

    def _sample_subgraph(self, seed):
        # BFS expansion: frontier per hop, sampling num_neighbors[hop] each hop.
        # nodes: ordered list with seed first; node_set maps global->present.
        nodes = [int(seed)]
        present = {int(seed)}
        frontier = [int(seed)]
        for k in self.num_neighbors:
            next_frontier = []
            for node in frontier:
                for nb in self._neighbors(node, k).tolist():
                    if nb not in present:
                        present.add(nb)
                        nodes.append(nb)
                        next_frontier.append(nb)
            frontier = next_frontier
        return nodes

    def __iter__(self):
        seeds = self.seeds
        if self.shuffle:
            seeds = seeds[torch.randperm(seeds.numel())]
        for start in range(0, seeds.numel(), self.batch_size):
            batch_seeds = seeds[start:start + self.batch_size]
            yield self._build_batch(batch_seeds)

    def _build_batch(self, batch_seeds):
        bs = batch_seeds.numel()
        # seeds first (local ids 0..bs-1), then the rest of each subgraph appended
        global_ids = []          # local -> global node id
        batch_vec = []           # local -> subgraph id
        # reserve seed slots
        for sg, seed in enumerate(batch_seeds.tolist()):
            global_ids.append(seed)
            batch_vec.append(sg)
        # per-subgraph local mapping for edge reindexing
        local_maps = [{int(s): i} for i, s in enumerate(batch_seeds.tolist())]
        for sg, seed in enumerate(batch_seeds.tolist()):
            sub_nodes = self._sample_subgraph(seed)
            for nb in sub_nodes[1:]:  # skip seed (already placed)
                local_maps[sg][nb] = len(global_ids)
                global_ids.append(nb)
                batch_vec.append(sg)

        global_ids_t = torch.tensor(global_ids, dtype=torch.long)
        batch_vec_t = torch.tensor(batch_vec, dtype=torch.long)

        # build local edge_index: for each subgraph, edges among its nodes
        e_src, e_dst = [], []
        for sg in range(bs):
            lm = local_maps[sg]
            members = list(lm.keys())
            mset = set(members)
            for gnode in members:
                lsrc = lm[gnode]
                for nb in self._neighbors(gnode, -1).tolist():
                    if nb in mset:
                        e_src.append(lsrc)
                        e_dst.append(lm[nb])
        if e_src:
            edge_index = torch.tensor([e_src, e_dst], dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        sub = Data(
            x=self.x[global_ids_t],
            edge_index=edge_index,
            y=self.y[global_ids_t],
        )
        sub.batch = batch_vec_t
        sub.batch_size = bs
        return sub


def make_loader(data, input_nodes, num_neighbors, batch_size, shuffle, use_pyg):
    if use_pyg:
        return NeighborLoader(
            data, input_nodes=input_nodes, num_neighbors=num_neighbors,
            batch_size=batch_size, shuffle=shuffle, disjoint=True, num_workers=0)
    return DisjointNeighborLoader(
        data, input_nodes, num_neighbors, batch_size, shuffle=shuffle)


def pyg_sampler_available():
    """True if NeighborLoader's C++ backend (pyg-lib/torch-sparse) is importable."""
    try:
        import pyg_lib  # noqa: F401
        return True
    except Exception:
        pass
    try:
        import torch_sparse  # noqa: F401
        return True
    except Exception:
        return False


# ==============================================================================
# main
# ==============================================================================
def list_npy(folder):
    return [f for f in os.listdir(folder) if f.endswith(".npy")]


def main():
    p = argparse.ArgumentParser(description="One-point relative-coord GNN (headless)")
    p.add_argument("--data_base",
                   default="/home/nishioka/GNN/GNN_hole_2026/all_sub_hole_defect_zscore")
    p.add_argument("--label_dir",
                   default="/home/nishioka/GNN/GNN_hole_2026/all_19class_label")
    p.add_argument("--coords_dir",
                   default="/home/nishioka/GNN/GNN_hole/GNN_hole_data")
    p.add_argument("--edges",
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "hole_edges_2layer_bidirectional.npy"))
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--device", default="cpu")
    p.add_argument("--num_neighbors", default="10,10")
    p.add_argument("--node_batch_size", type=int, default=16)
    p.add_argument("--val_batch_size", type=int, default=64)
    p.add_argument("--train_sampling", type=float, default=0.05)
    p.add_argument("--limit_files", type=int, default=60,
                   help="cap #train and #val files (0 = all)")
    p.add_argument("--hidden_channels", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=2.0)
    args = p.parse_args()

    set_seed(args.seed)

    device = torch.device(args.device)
    num_neighbors = [int(x) for x in args.num_neighbors.split(",") if x.strip() != ""]
    num_layers = len(num_neighbors)

    print("=" * 70)
    print(f"device:          {device}")
    print(f"num_neighbors:   {num_neighbors} (-> num_layers={num_layers})")
    print(f"node_batch_size: {args.node_batch_size}  val_batch_size: {args.val_batch_size}")
    print(f"train_sampling:  {args.train_sampling}  limit_files: {args.limit_files}")
    print(f"hidden_channels: {args.hidden_channels}  epochs: {args.epochs}")
    print(f"started: {datetime.datetime.now().isoformat(timespec='seconds')}")
    print("=" * 70)

    train_folder = os.path.join(args.data_base, "train")
    val_folder = os.path.join(args.data_base, "val")

    # coords + edges
    x_coords = np.load(os.path.join(args.coords_dir, "normalized_x_2layer.npy"))[:NUM_NODES]
    y_coords = np.load(os.path.join(args.coords_dir, "normalized_y_2layer.npy"))[:NUM_NODES]
    z_coords = np.load(os.path.join(args.coords_dir, "normalized_z_2layer.npy"))[:NUM_NODES]
    edges = np.load(args.edges)
    # edges expected as (E, 2) -> edge_index (2, E); guard if already (2, E)
    if edges.shape[0] == 2 and edges.shape[1] != 2:
        edge_index = torch.tensor(edges, dtype=torch.long)
    else:
        edge_index = torch.tensor(edges.T, dtype=torch.long)
    print(f"coords nodes: {x_coords.shape[0]}  "
          f"edge_index: {tuple(edge_index.shape)}  "
          f"edge max idx: {int(edge_index.max())}")
    assert x_coords.shape[0] == NUM_NODES, f"coords have {x_coords.shape[0]} nodes, expected {NUM_NODES}"
    assert int(edge_index.max()) < NUM_NODES, "edge index exceeds node count"

    # file lists
    train_data_files = sorted(list_npy(train_folder))
    val_data_files = sorted(list_npy(val_folder))
    label_files = [f for f in os.listdir(args.label_dir) if f.endswith("_19label.npy")]

    if args.limit_files and args.limit_files > 0:
        train_data_files = train_data_files[:args.limit_files]
        val_data_files = val_data_files[:args.limit_files]

    train_pairs = create_data_label_pairs(train_data_files, label_files)
    val_pairs = create_data_label_pairs(val_data_files, label_files)
    print(f"#train pairs: {len(train_pairs)}   #val pairs: {len(val_pairs)}")

    train_dataset_list, class_weights = prepare_data(
        train_pairs, train_folder, args.label_dir,
        x_coords, y_coords, z_coords, edge_index)
    val_dataset_list, _ = prepare_data(
        val_pairs, val_folder, args.label_dir,
        x_coords, y_coords, z_coords, edge_index)

    if not train_dataset_list or not val_dataset_list:
        print("Dataset preparation failed (empty). Aborting.")
        sys.exit(2)

    # one big combined graph each
    tb = Batch.from_data_list(train_dataset_list)
    train_data_combined = Data(x=tb.x, edge_index=tb.edge_index, y=tb.y)
    vb = Batch.from_data_list(val_dataset_list)
    val_data_combined = Data(x=vb.x, edge_index=vb.edge_index, y=vb.y)

    y_train = train_data_combined.y
    y_val = val_data_combined.y

    val_defect_mask = (y_val != 0) & (torch.rand(y_val.size(0)) < args.train_sampling)
    val_bg_mask = (y_val == 0) & (torch.rand(y_val.size(0)) < args.train_sampling)
    val_input_mask = val_defect_mask | val_bg_mask
    if val_input_mask.sum().item() == 0:
        print("Validation input mask empty. Aborting.")
        sys.exit(2)
    print(f"[Val Sampling] Used: {val_input_mask.sum().item()} / {y_val.size(0)}")

    use_pyg = pyg_sampler_available()
    print(f"sampler backend: {'pyg NeighborLoader' if use_pyg else 'pure-python DisjointNeighborLoader (pyg-lib/torch-sparse not installed)'}")

    val_loader = make_loader(
        val_data_combined, val_input_mask, num_neighbors,
        args.val_batch_size, shuffle=False, use_pyg=use_pyg)

    model = DeepGATModel(
        hidden_channels=args.hidden_channels,
        num_classes=NUM_CLASSES,
        num_layers=num_layers,
        heads=4,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    cw_tensor = class_weights.to(device) if class_weights is not None else None
    loss_fn = FocalLoss(gamma=args.gamma, weights=cw_tensor, reduction="mean").to(device)

    final_f1 = float("nan")
    for epoch in range(1, args.epochs + 1):
        # per-epoch random seed sampling (defect / bg) -- as in the notebook
        train_defect_mask = (y_train != 0) & (torch.rand(y_train.size(0)) < args.train_sampling)
        train_bg_mask = (y_train == 0) & (torch.rand(y_train.size(0)) < args.train_sampling)
        train_input_mask = train_defect_mask | train_bg_mask
        if train_input_mask.sum().item() == 0:
            print(f"[epoch {epoch}] train input mask empty, skipping.")
            continue

        train_loader = make_loader(
            train_data_combined, train_input_mask, num_neighbors,
            args.node_batch_size, shuffle=True, use_pyg=use_pyg)

        model.train()
        total_loss, total_batches = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            bs = batch.batch_size
            seed_indices = torch.arange(bs, device=device)
            seed_coords = batch.x[seed_indices, :3]
            nodes_center_coords = seed_coords[batch.batch]

            rel_coords = batch.x[:, :3] - nodes_center_coords
            other_features = batch.x[:, 3:]
            x_new = torch.cat([rel_coords, other_features], dim=1)

            out_target = model(x_new, batch.edge_index, target_indices=seed_indices)
            y_target = batch.y[seed_indices]

            loss = loss_fn(out_target, y_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += float(loss.item())
            total_batches += 1

        avg_train_loss = total_loss / total_batches if total_batches else 0.0

        # validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.inference_mode():
            for batch in val_loader:
                batch = batch.to(device)
                bs = batch.batch_size
                seed_indices = torch.arange(bs, device=device)
                seed_coords = batch.x[seed_indices, :3]
                nodes_center_coords = seed_coords[batch.batch]

                rel_coords = batch.x[:, :3] - nodes_center_coords
                other_features = batch.x[:, 3:]
                x_new = torch.cat([rel_coords, other_features], dim=1)

                out_target = model(x_new, batch.edge_index, target_indices=seed_indices)
                y_target = batch.y[seed_indices]
                pred = out_target.argmax(dim=1)
                all_preds.extend(pred.cpu().tolist())
                all_labels.extend(y_target.cpu().tolist())

        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0) if all_preds else 0.0
        final_f1 = macro_f1
        print(f"[epoch {epoch}/{args.epochs}] train_loss={avg_train_loss:.4f}  "
              f"val_macro_f1={macro_f1:.4f}  (train_batches={total_batches}, "
              f"val_samples={len(all_labels)})")

    print("=" * 70)
    print(f"SMOKE SUMMARY: epochs={args.epochs}  "
          f"train_pairs={len(train_pairs)}  val_pairs={len(val_pairs)}  "
          f"final_val_macro_f1={final_f1:.4f}  "
          f"finished={datetime.datetime.now().isoformat(timespec='seconds')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
