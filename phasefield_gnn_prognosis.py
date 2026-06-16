"""phasefield_gnn_prognosis.py — a GNN that learns PROGNOSIS from phase-field damage fields [idea #6].

This is the Keio Phase-2 bridge prototype: couple the AT2 phase-field fracture model (computational
solid mechanics, Murakami-lab territory) with a graph neural network (the structure-agnostic SHM
side) so the GNN learns to PREDICT remaining life directly from a damage field — a fast, learned
prognosis surrogate of the phase-field dynamics.

What the GNN learns. In this normalised laminate the snap LOAD is nearly defect-independent (matrix-
dominated transverse failure goes unstable at a near-fixed applied strain), but the post-snap crack
PATTERN is strongly seed-dependent — its through-thickness position tracks the defect ply and its
extent tracks the defect size. So the well-posed, useful learnable map is the FORWARD OPERATOR:
seed defect field -> final (post-snap) damage field. The GNN becomes a fast emulator of the AT2
phase-field solver — exactly the computational-solid-mechanics × ML bridge for the Keio thesis.

Pipeline
--------
  1. DATA from physics: draw many seeded defects (varied location / size / ply). For each, run the
     AT2 phase-field to a fixed super-critical load and record the final damage field.
  2. GRAPH: the seed state is a mesh graph — grid nodes, 4-neighbour edges, node features
     [seed damage d0, local toughness Gc, x, y]. No torch_geometric: a 3-round mean-aggregation
     message passing implemented directly in torch.
  3. LEARN: the GNN regresses the per-node final damage field, holding out WHOLE defects
     (leave-defect-out) so the test measures emulation of unseen defects, not memorisation. We report
     node-level R^2 and the crack-region IoU.

Honest scope: a prototype on a normalised 2-D laminate with ~30 defects — a proof that the
phase-field -> GNN forward-emulation loop trains and generalises across defects, to be scaled
(3-D, real Gc, fatigue cycles, UQ propagation, then inverse prognosis) for the Keio thesis.
See PHASEFIELD_GNN_PLAN.md.

Run:  python3 phasefield_gnn_prognosis.py   (writes phasefield_gnn_prognosis.png)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

import cfrp_phasefield_2d as pf

SEED = 20270614
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)


# ----------------------------- physics data generation -----------------------------
def make_cfg():
    # smaller grid for a fast prototype; still resolves the length scale (ell ~ 2.2 h)
    return pf.LaminateConfig(nx=40, ny=32, n_load_steps=10)


FIXED_LOAD = 0.16          # fixed super-critical load (above the ~0.095 snap) for the forward map


def forward_pair(cfg, cx, cy, layer, log2size):
    """Run AT2 to FIXED_LOAD; return (seed field d0, final damage field d_final)."""
    d0 = pf.seed_defect(cx, cy, layer, log2size, cfg)
    res = pf.simulate_growth(d0, FIXED_LOAD, cfg)
    return d0, res.d_final


def grid_edges(nx, ny):
    idx = np.arange(nx * ny).reshape(ny, nx)
    e = []
    for j in range(ny):
        for i in range(nx):
            if i + 1 < nx:
                e += [(idx[j, i], idx[j, i + 1]), (idx[j, i + 1], idx[j, i])]
            if j + 1 < ny:
                e += [(idx[j, i], idx[j + 1, i]), (idx[j + 1, i], idx[j, i])]
    return np.array(e, dtype=np.int64).T          # (2, E)


def build_dataset():
    cfg = make_cfg()
    nx, ny = cfg.nx, cfg.ny
    Gc_map, *_ = pf.build_laminate_maps(cfg)
    Gc_flat = (Gc_map / Gc_map.mean()).reshape(-1).astype(np.float32)
    xs = np.tile(np.linspace(0, 1, nx), ny).astype(np.float32)
    ys = np.repeat(np.linspace(0, 1, ny), nx).astype(np.float32)
    edges = grid_edges(nx, ny)

    seeds = []
    for _ in range(30):                           # 30 defects (leave-defect-out later)
        seeds.append(dict(cx=rng.uniform(0.25, 0.75), cy=0.5,
                          layer=rng.integers(1, 18), log2size=rng.uniform(-1.0, 1.5)))
    samples = []                                  # (node_feat[N,6], target_field[N], seed_id, meta)
    for sid, s in enumerate(seeds):
        d0, dfin = forward_pair(cfg, s["cx"], s["cy"], s["layer"], s["log2size"])
        seed_flat = d0.reshape(-1)
        sm = seed_flat > 0.5
        # broadcast seed centroid (sx, sy) to every node: tells each node where the crack will band
        sx_b = np.full(nx * ny, float(xs[sm].mean()) if sm.any() else 0.5, np.float32)
        sy_b = np.full(nx * ny, float(ys[sm].mean()) if sm.any() else 0.5, np.float32)
        nf = np.stack([seed_flat.astype(np.float32), Gc_flat, xs, ys, sx_b, sy_b], axis=1)
        tgt = dfin.reshape(-1).astype(np.float32)
        samples.append((nf, tgt, sid, dict(layer=s["layer"], size=2.0 ** s["log2size"],
                                           area=float((dfin > 0.5).mean()))))
        print(f"  defect {sid:2d}: cx={s['cx']:.2f} layer={s['layer']:2d} "
              f"size={2.0**s['log2size']:.2f}  crack_frac={ (dfin>0.5).mean():.3f}")
    return samples, edges, len(seeds), (nx, ny)


# ----------------------------- GNN -----------------------------
class MeshGNN(nn.Module):
    def __init__(self, in_dim=6, hid=48, rounds=8):
        super().__init__()
        self.enc = nn.Linear(in_dim, hid)
        # message input = [self, neighbour-mean, GLOBAL-mean] -> 3*hid (global gives long-range
        # propagation cheaply, so a domain-spanning crack band is reachable without many rounds)
        self.msg = nn.ModuleList([nn.Linear(3 * hid, hid) for _ in range(rounds)])
        self.head = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 1))
        self.rounds = rounds

    def forward(self, x, edge_index):
        h = torch.relu(self.enc(x))
        src, dst = edge_index
        ones = torch.ones(src.size(0), 1, device=h.device)
        for layer in self.msg:
            agg = torch.zeros_like(h)
            deg = torch.zeros(h.size(0), 1, device=h.device)
            agg.index_add_(0, dst, h[src])
            deg.index_add_(0, dst, ones)
            agg = agg / deg.clamp(min=1)
            glob = h.mean(0, keepdim=True).expand_as(h)
            h = h + torch.relu(layer(torch.cat([h, agg, glob], dim=1)))
        return torch.sigmoid(self.head(h)).squeeze(-1)    # per-node damage in [0,1]


def main():
    print("Generating phase-field forward pairs (AT2 staggered solver)...")
    samples, edges, n_seed, (nx, ny) = build_dataset()
    ei = torch.tensor(edges)

    # leave-defect-out: hold out the last 8 defects (unseen at training)
    test_ids = set(range(n_seed - 8, n_seed))
    train = [s for s in samples if s[2] not in test_ids]
    test = [s for s in samples if s[2] in test_ids]
    print(f"\ntrain defects={len(train)}  test(held-out defects)={len(test)}  nodes/graph={nx*ny}")

    model = MeshGNN()
    opt = torch.optim.Adam(model.parameters(), lr=4e-3, weight_decay=1e-5)
    lossf = nn.BCELoss()                           # soft targets in [0,1]; sharpens crack boundary
    for ep in range(300):
        model.train()
        perm = rng.permutation(len(train))
        tot = 0.0
        for k in perm:
            nf, tgt, _, _ = train[k]
            pred = model(torch.tensor(nf), ei).clamp(1e-6, 1 - 1e-6)
            loss = lossf(pred, torch.tensor(tgt).clamp(0, 1))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        if (ep + 1) % 60 == 0:
            print(f"  epoch {ep+1:3d}  train node-BCE={tot/len(train):.4f}")

    model.eval()
    def evaluate(split):
        r2s, ious = [], []
        preds = []
        with torch.no_grad():
            for nf, tgt, _, meta in split:
                p = model(torch.tensor(nf), ei).numpy()
                ss = ((tgt - p) ** 2).sum() / max(((tgt - tgt.mean()) ** 2).sum(), 1e-9)
                r2s.append(1 - ss)
                a, b = p > 0.5, tgt > 0.5
                iou = (a & b).sum() / max((a | b).sum(), 1)
                ious.append(iou)
                preds.append(p)
        return np.array(r2s), np.array(ious), preds

    r2_tr, iou_tr, _ = evaluate(train)
    r2_te, iou_te, preds_te = evaluate(test)
    print(f"\nTRAIN  node-R2={r2_tr.mean():.2f}  crack-IoU={iou_tr.mean():.2f}")
    print(f"TEST (held-out defects)  node-R2={r2_te.mean():.2f}  crack-IoU={iou_te.mean():.2f}")

    # ---- figure: example held-out emulations + area parity ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(13.5, 5.2))
    # pick 2 held-out defects to show true vs predicted crack
    show = test[:2]
    for col, (nf, tgt, sid, meta) in enumerate(show):
        p = preds_te[col]
        for row, (field, lab) in enumerate([(tgt, "AT2 truth"), (p, "GNN emulation")]):
            ax = fig.add_subplot(2, 4, row * 4 + col + 1)
            ax.imshow(field.reshape(ny, nx), origin="lower", cmap="inferno", vmin=0, vmax=1,
                      aspect="auto")
            ax.set_title(f"defect {sid} (ly={meta['layer']}, sz={meta['size']:.1f})\n{lab}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    # area parity (held-out)
    ax = fig.add_subplot(1, 2, 2)
    at = np.array([(s[1] > 0.5).mean() for s in test])
    ap = np.array([(pp > 0.5).mean() for pp in preds_te])
    att = np.array([(s[1] > 0.5).mean() for s in train])
    apt = np.array([(model(torch.tensor(s[0]), ei).detach().numpy() > 0.5).mean() for s in train])
    ax.scatter(att, apt, s=28, c="#999999", ec="k", lw=.3, label="train")
    ax.scatter(at, ap, s=70, c="#c0392b", ec="k", lw=.4, label="held-out defects")
    lim = [0, max(at.max(), ap.max(), att.max()) * 1.1]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set(xlabel="true crack-area fraction (AT2)", ylabel="GNN-emulated crack-area fraction",
           xlim=lim, ylim=lim,
           title=f"Crack-extent parity\nheld-out node-R²={r2_te.mean():.2f}  IoU={iou_te.mean():.2f}")
    ax.legend(fontsize=9); ax.grid(alpha=.3)

    fig.suptitle("Phase-field × GNN forward emulation (Keio Phase-2 prototype): a GNN reproduces the "
                 "AT2 post-snap crack field of UNSEEN defects", fontweight="bold")
    fig.tight_layout()
    out = "/home/nishioka/cfrp_datasets/phasefield_gnn_prognosis.png"
    fig.savefig(out, dpi=140)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
