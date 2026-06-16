"""phasefield_gnn_3d.py — phase-field × GNN forward emulation in 3-D [idea #6, (b)].

Phase-2 本丸: scale the 2-D forward emulator to a 3-D laminate where the damage is a
DELAMINATION front advancing in-plane at an inter-ply interface — the 3-D feature a 2-D
cross-section cannot represent (phasefield_3d.simulate_growth_3d). The GNN emulates
seed delamination patch -> grown 3-D delamination field, on a 12×16×16 mesh graph, and
generalises to UNSEEN defects (leave-defect-out). A CNN U-Net needs a structured grid and would
not survive an irregular 3-D mesh; the message-passing GNN is the mesh-agnostic choice that does.

Run:  python3 phasefield_gnn_3d.py   (writes phasefield_gnn_3d.png)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

import phasefield_3d as p3
from phasefield_gnn_prognosis import MeshGNN     # generic message-passing GNN (any graph)

SEED = 20270614
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)


def grid_edges_3d(nx, ny, nz):
    idx = np.arange(nx * ny * nz).reshape(nz, ny, nx)
    e = []
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                a = idx[iz, iy, ix]
                if ix + 1 < nx: e += [(a, idx[iz, iy, ix + 1]), (idx[iz, iy, ix + 1], a)]
                if iy + 1 < ny: e += [(a, idx[iz, iy + 1, ix]), (idx[iz, iy + 1, ix], a)]
                if iz + 1 < nz: e += [(a, idx[iz + 1, iy, ix]), (idx[iz + 1, iy, ix], a)]
    return np.array(e, dtype=np.int64).T


def build_dataset_3d(n_seed=26):
    cfg = p3.LaminateConfig3D()
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    Gc_map, *_ = p3.build_interface_maps_3d(cfg, z_interface=0.5)
    Gc_flat = (Gc_map / Gc_map.mean()).reshape(-1).astype(np.float32)
    zz, yy, xx = np.meshgrid(np.linspace(0, 1, nz), np.linspace(0, 1, ny),
                             np.linspace(0, 1, nx), indexing="ij")
    xs, ys, zs = xx.reshape(-1).astype(np.float32), yy.reshape(-1).astype(np.float32), zz.reshape(-1).astype(np.float32)
    edges = grid_edges_3d(nx, ny, nz)

    samples = []
    for sid in range(n_seed):
        cx, cy = rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7)
        r = rng.uniform(0.05, 0.13)
        drive = rng.uniform(1.2, 2.6)
        d0 = p3.seed_defect_3d(cx, cy, 0.5, r, r, 0.04, cfg)
        res = p3.simulate_growth_3d(d0, cfg, n_steps=6, drive=drive, z_interface=0.5)
        seed_flat = d0.reshape(-1)
        sm = seed_flat > 0.5
        sx = np.full(nx*ny*nz, float(xs[sm].mean()) if sm.any() else .5, np.float32)
        sy = np.full(nx*ny*nz, float(ys[sm].mean()) if sm.any() else .5, np.float32)
        sz = np.full(nx*ny*nz, float(zs[sm].mean()) if sm.any() else .5, np.float32)
        nf = np.stack([seed_flat.astype(np.float32), Gc_flat, xs, ys, zs, sx, sy, sz], axis=1)
        tgt = res.d_final.reshape(-1).astype(np.float32)
        samples.append((nf, tgt, sid, dict(frac=float((res.d_final > 0.5).mean()), drive=drive)))
        print(f"  defect {sid:2d}: cx={cx:.2f} r={r:.2f} drive={drive:.1f}  "
              f"vol0={res.vol0:.4f}->{res.vol_final:.4f}  frac={(res.d_final>0.5).mean():.3f}")
    return samples, edges, n_seed, (nx, ny, nz)


def main():
    print("Generating 3-D phase-field delamination pairs...")
    samples, edges, n_seed, (nx, ny, nz) = build_dataset_3d()
    ei = torch.tensor(edges)
    test_ids = set(range(n_seed - 6, n_seed))
    train = [s for s in samples if s[2] not in test_ids]
    test = [s for s in samples if s[2] in test_ids]
    print(f"\ntrain={len(train)} test={len(test)} nodes/graph={nx*ny*nz}")

    model = MeshGNN(in_dim=8, hid=48, rounds=8)
    opt = torch.optim.Adam(model.parameters(), 4e-3, weight_decay=1e-5)
    lf = nn.BCELoss()
    for ep in range(120):
        for k in rng.permutation(len(train)):
            nf, tgt, _, _ = train[k]
            p = model(torch.tensor(nf), ei).clamp(1e-6, 1 - 1e-6)
            loss = lf(p, torch.tensor(tgt).clamp(0, 1))
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 50 == 0:
            print(f"  epoch {ep+1:3d}  train BCE={loss.item():.4f}")

    model.eval()
    def ev(split):
        r2s, ious, preds = [], [], []
        with torch.no_grad():
            for nf, tgt, _, _ in split:
                p = model(torch.tensor(nf), ei).numpy()
                r2s.append(1 - ((tgt-p)**2).sum()/max(((tgt-tgt.mean())**2).sum(), 1e-9))
                a, b = p > 0.5, tgt > 0.5
                ious.append((a & b).sum()/max((a | b).sum(), 1)); preds.append(p)
        return np.array(r2s), np.array(ious), preds
    r2tr, ioutr, _ = ev(train)
    r2te, ioute, pte = ev(test)
    print(f"\nTRAIN  node-R2={r2tr.mean():.2f}  vol-IoU={ioutr.mean():.2f}")
    print(f"TEST (held-out 3-D defects)  node-R2={r2te.mean():.2f}  vol-IoU={ioute.mean():.2f}")

    # ---- figure: interface-plane (z=mid) slices true vs pred + volume parity ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    zmid = nz // 2
    fig = plt.figure(figsize=(13.5, 5.2))
    show = test[:2]
    for col, (nf, tgt, sid, meta) in enumerate(show):
        tv = tgt.reshape(nz, ny, nx)[zmid]
        pv = pte[col].reshape(nz, ny, nx)[zmid]
        for row, (f, lab) in enumerate([(tv, "AT2 truth"), (pv, "GNN emulation")]):
            ax = fig.add_subplot(2, 4, row*4 + col + 1)
            ax.imshow(f, origin="lower", cmap="inferno", vmin=0, vmax=1, aspect="auto")
            ax.set_title(f"defect {sid} — {lab}\ninterface plane z={zmid}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(1, 2, 2)
    vt = np.array([(s[1] > 0.5).mean() for s in test])
    vp = np.array([(pp > 0.5).mean() for pp in pte])
    vtt = np.array([(s[1] > 0.5).mean() for s in train])
    vpt = np.array([(model(torch.tensor(s[0]), ei).detach().numpy() > 0.5).mean() for s in train])
    ax.scatter(vtt, vpt, s=26, c="#999999", ec="k", lw=.3, label="train")
    ax.scatter(vt, vp, s=70, c="#c0392b", ec="k", lw=.4, label="held-out 3-D defects")
    lim = [0, max(vt.max(), vp.max(), vtt.max())*1.15 + 1e-3]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set(xlabel="true delamination volume-fraction", ylabel="GNN-emulated volume-fraction",
           xlim=lim, ylim=lim,
           title=f"3-D delamination volume parity\nheld-out node-R²={r2te.mean():.2f}  vol-IoU={ioute.mean():.2f}")
    ax.legend(fontsize=9); ax.grid(alpha=.3)
    fig.suptitle("Phase-field × GNN forward emulation in 3-D (Keio Phase-2): a mesh-agnostic GNN "
                 "emulates the AT2 delamination front of unseen 3-D defects", fontweight="bold")
    fig.tight_layout()
    out = "/home/nishioka/cfrp_datasets/phasefield_gnn_3d.png"
    fig.savefig(out, dpi=140)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
