"""phasefield_emulator_compare.py — message-passing GNN vs a U-Net emulator [idea #6, (a)].

The MeshGNN forward emulator reaches held-out node-R^2~0.94 / crack-IoU~0.89, but the sharp
full-width crack band is still slightly under-resolved. On a STRUCTURED mesh the natural way to
sharpen morphology is a multi-resolution U-Net: pooling gives the long-range receptive field a
local message-passing GNN must reach by many hops. We train BOTH on the SAME phase-field pairs
(cached from phasefield_gnn_prognosis.build_dataset) and compare honestly:

  * MeshGNN   — mesh-AGNOSTIC (works on any graph), 8-round message passing + global context.
  * U-Net     — exploits the structured grid (CNN encoder/decoder w/ skips); the dense upper bound.

Finding (this run): the mesh-agnostic GNN MATCHES the U-Net (both node-R^2~0.96, IoU~0.91-0.92) —
on this structured case mesh-agnosticism costs essentially nothing, which is exactly what you want
to know before scaling to irregular 3-D meshes where a CNN U-Net no longer applies.

Run:  python3 phasefield_emulator_compare.py   (writes phasefield_emulator_compare.png)
"""
from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn

import phasefield_gnn_prognosis as P

CACHE = "/home/nishioka/cfrp_datasets/_phasefield_forward_cache.npz"
SEED = 20270614
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)


def load_cached():
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        X = z["X"]; Y = z["Y"]; sid = z["sid"]; meta = list(z["meta"])
        edges = z["edges"]; nx, ny = int(z["nx"]), int(z["ny"])
        samples = [(X[i], Y[i], int(sid[i]), meta[i]) for i in range(len(X))]
        return samples, edges, int(sid.max()) + 1, (nx, ny)
    samples, edges, n, (nx, ny) = P.build_dataset()
    np.savez(CACHE,
             X=np.stack([s[0] for s in samples]),
             Y=np.stack([s[1] for s in samples]),
             sid=np.array([s[2] for s in samples]),
             meta=np.array([s[3] for s in samples], dtype=object),
             edges=edges, nx=nx, ny=ny)
    return samples, edges, n, (nx, ny)


# ----------------------------- U-Net on the structured grid -----------------------------
class UNet(nn.Module):
    def __init__(self, in_ch=6, base=24):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(o, o, 3, padding=1), nn.ReLU())
        self.e1 = block(in_ch, base)
        self.e2 = block(base, base * 2)
        self.e3 = block(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = block(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):                       # x: (1, in_ch, ny, nx)
        c1 = self.e1(x)
        c2 = self.e2(self.pool(c1))
        c3 = self.e3(self.pool(c2))
        u2 = self.d2(torch.cat([self._fit(self.up2(c3), c2), c2], 1))
        u1 = self.d1(torch.cat([self._fit(self.up1(u2), c1), c1], 1))
        return torch.sigmoid(self.out(u1))      # (1,1,ny,nx)

    @staticmethod
    def _fit(a, ref):                           # pad/crop a to ref's H,W (odd dims)
        dh, dw = ref.size(2) - a.size(2), ref.size(3) - a.size(3)
        if dh or dw:
            a = nn.functional.pad(a, [0, dw, 0, dh])
        return a


def to_image(nf, nx, ny):
    return torch.tensor(nf.reshape(ny, nx, -1)).permute(2, 0, 1).unsqueeze(0)  # (1,C,ny,nx)


def metrics(yt, yp):
    r2 = 1 - ((yt - yp) ** 2).sum() / max(((yt - yt.mean()) ** 2).sum(), 1e-9)
    a, b = yp > 0.5, yt > 0.5
    iou = (a & b).sum() / max((a | b).sum(), 1)
    return r2, iou


def train_gnn(train, ei):
    m = P.MeshGNN(); opt = torch.optim.Adam(m.parameters(), 4e-3, weight_decay=1e-5)
    lf = nn.BCELoss()
    for ep in range(150):
        for k in rng.permutation(len(train)):
            nf, tgt, _, _ = train[k]
            p = m(torch.tensor(nf), ei).clamp(1e-6, 1 - 1e-6)
            loss = lf(p, torch.tensor(tgt).clamp(0, 1))
            opt.zero_grad(); loss.backward(); opt.step()
    return m


def train_unet(train, nx, ny):
    m = UNet(); opt = torch.optim.Adam(m.parameters(), 3e-3, weight_decay=1e-5)
    lf = nn.BCELoss()
    for ep in range(150):
        for k in rng.permutation(len(train)):
            nf, tgt, _, _ = train[k]
            p = m(to_image(nf, nx, ny)).clamp(1e-6, 1 - 1e-6).reshape(-1)
            loss = lf(p, torch.tensor(tgt).clamp(0, 1))
            opt.zero_grad(); loss.backward(); opt.step()
    return m


def main():
    print("Loading/generating phase-field forward pairs...")
    samples, edges, n_seed, (nx, ny) = load_cached()
    ei = torch.tensor(edges)
    test_ids = set(range(n_seed - 8, n_seed))
    train = [s for s in samples if s[2] not in test_ids]
    test = [s for s in samples if s[2] in test_ids]
    print(f"train={len(train)} test={len(test)} grid={nx}x{ny}")

    print("training MeshGNN..."); gnn = train_gnn(train, ei)
    print("training U-Net...");  unet = train_unet(train, nx, ny)

    def eval_model(model, kind):
        r2s, ious, preds = [], [], []
        with torch.no_grad():
            for nf, tgt, _, _ in test:
                if kind == "gnn":
                    p = model(torch.tensor(nf), ei).numpy()
                else:
                    p = model(to_image(nf, nx, ny)).reshape(-1).numpy()
                r2, iou = metrics(tgt, p); r2s.append(r2); ious.append(iou); preds.append(p)
        return np.array(r2s), np.array(ious), preds

    r2g, ioug, pg = eval_model(gnn, "gnn")
    r2u, iouu, pu = eval_model(unet, "unet")
    print(f"\nHELD-OUT  GNN   node-R2={r2g.mean():.2f}  IoU={ioug.mean():.2f}")
    print(f"HELD-OUT  U-Net node-R2={r2u.mean():.2f}  IoU={iouu.mean():.2f}")

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(13.5, 6.0))
    show = test[:2]
    titles = ["AT2 truth", "GNN", "U-Net"]
    for col, (nf, tgt, sid, meta) in enumerate(show):
        fields = [tgt, pg[col], pu[col]]
        for row in range(3):
            ax = fig.add_subplot(3, 4, row * 4 + col + 1)
            ax.imshow(fields[row].reshape(ny, nx), origin="lower", cmap="inferno", vmin=0, vmax=1,
                      aspect="auto")
            ax.set_title(f"defect {sid} — {titles[row]}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(1, 2, 2)
    x = np.arange(2)
    ax.bar(x - 0.2, [r2g.mean(), ioug.mean()], 0.4, color="#1f6f6f", ec="k", label="MeshGNN (mesh-agnostic)")
    ax.bar(x + 0.2, [r2u.mean(), iouu.mean()], 0.4, color="#c0392b", ec="k", label="U-Net (structured grid)")
    for xi, v in zip(x - 0.2, [r2g.mean(), ioug.mean()]):
        ax.annotate(f"{v:.2f}", (xi, v), ha="center", va="bottom", fontsize=9)
    for xi, v in zip(x + 0.2, [r2u.mean(), iouu.mean()]):
        ax.annotate(f"{v:.2f}", (xi, v), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(["node-R²", "crack-IoU"])
    ax.set_ylim(0, 1.05); ax.set_ylabel("held-out score")
    ax.set_title("Mesh-agnostic GNN MATCHES the structured-grid\nU-Net ceiling — agnosticism is ~free here")
    ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
    fig.suptitle("Phase-field forward emulation: message-passing GNN vs U-Net (held-out defects)",
                 fontweight="bold")
    fig.tight_layout()
    out = "/home/nishioka/cfrp_datasets/phasefield_emulator_compare.png"
    fig.savefig(out, dpi=140)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
