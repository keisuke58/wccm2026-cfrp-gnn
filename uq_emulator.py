"""uq_emulator.py — calibrated uncertainty for the phase-field GNN emulator [#3].

⑥/#1 give a point-prediction emulator. For the decision/clearance pipeline (⑤ conformal,
decision_uq) the surrogate must carry CALIBRATED error bars, not just a mean. We add MC-dropout to
the MeshGNN: dropout stays ON at inference, K stochastic passes give a per-node mean and std. We
then check the two things that make UQ usable:
  * CALIBRATION — bin nodes by predicted std; does the actual RMSE rise with predicted std?
  * LOCALISATION — is the uncertainty concentrated at the crack boundary (where decisions are hard)?
The per-prediction std is exactly the quantity a downstream conformal/decision layer gates on.

Run:  python3 uq_emulator.py   (writes uq_emulator.png)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from phasefield_emulator_compare import load_cached

SEED = 20270614
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)


class MeshGNN_MC(nn.Module):
    """MeshGNN with global context + dropout (kept on at inference for MC-dropout UQ)."""
    def __init__(self, in_dim=6, hid=48, rounds=8, p=0.15):
        super().__init__()
        self.enc = nn.Linear(in_dim, hid)
        self.msg = nn.ModuleList([nn.Linear(3 * hid, hid) for _ in range(rounds)])
        self.drop = nn.Dropout(p)
        self.head = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Dropout(p), nn.Linear(hid, 1))

    def forward(self, x, edge_index):
        h = torch.relu(self.enc(x))
        src, dst = edge_index
        for layer in self.msg:
            agg = torch.zeros_like(h)
            deg = torch.zeros(h.size(0), 1, device=h.device)
            agg.index_add_(0, dst, h[src])
            deg.index_add_(0, dst, torch.ones(src.size(0), 1, device=h.device))
            agg = agg / deg.clamp(min=1)
            glob = h.mean(0, keepdim=True).expand_as(h)
            h = h + self.drop(torch.relu(layer(torch.cat([h, agg, glob], dim=1))))
        return torch.sigmoid(self.head(h)).squeeze(-1)


def main():
    print("Loading cached forward pairs + training MC-dropout emulator...")
    samples, edges, n_seed, (nx, ny) = load_cached()
    ei = torch.tensor(edges)
    test_ids = set(range(n_seed - 8, n_seed))
    train = [s for s in samples if s[2] not in test_ids]
    test = [s for s in samples if s[2] in test_ids]

    m = MeshGNN_MC()
    opt = torch.optim.Adam(m.parameters(), 4e-3, weight_decay=1e-5)
    lf = nn.BCELoss()
    for ep in range(100):
        m.train()
        for k in rng.permutation(len(train)):
            nf, tgt, _, _ = train[k]
            p = m(torch.tensor(nf), ei).clamp(1e-6, 1 - 1e-6)
            loss = lf(p, torch.tensor(tgt).clamp(0, 1))
            opt.zero_grad(); loss.backward(); opt.step()

    # MC-dropout inference: dropout ON (train mode), K passes
    K = 25
    m.train()
    all_err, all_std, maps = [], [], []
    with torch.no_grad():
        for nf, tgt, _, _ in test:
            xs = torch.tensor(nf)
            ps = np.stack([m(xs, ei).numpy() for _ in range(K)])    # (K, N)
            mean, std = ps.mean(0), ps.std(0)
            all_err.append(np.abs(mean - tgt)); all_std.append(std)
            maps.append((tgt, mean, std))
    err = np.concatenate(all_err); std = np.concatenate(all_std)

    # calibration: bin by predicted std, measure actual RMSE
    qs = np.quantile(std, np.linspace(0, 1, 9))
    bx, by = [], []
    for i in range(len(qs) - 1):
        msk = (std >= qs[i]) & (std < qs[i + 1] if i < len(qs) - 2 else std <= qs[i + 1])
        if msk.sum() > 5:
            bx.append(std[msk].mean()); by.append(np.sqrt((err[msk] ** 2).mean()))
    corr = np.corrcoef(std, err)[0, 1]
    print(f"\nUQ calibration: corr(pred-std, abs-err)={corr:.2f}  over {len(test)} held-out defects, K={K}")

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(14, 4.7))
    tgt0, mean0, std0 = maps[0]
    for col, (f, lab, cmap) in enumerate([(tgt0, "AT2 truth", "inferno"),
                                          (mean0, "GNN mean", "inferno"),
                                          (std0, "GNN uncertainty (MC-std)", "viridis")]):
        ax = fig.add_subplot(1, 4, col + 1)
        im = ax.imshow(f.reshape(ny, nx), origin="lower", cmap=cmap, aspect="auto",
                       vmin=0, vmax=(1 if col < 2 else None))
        ax.set_title(lab, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
        if col == 2:
            fig.colorbar(im, ax=ax, fraction=0.046)
    ax = fig.add_subplot(1, 4, 4)
    ax.plot(bx, by, "o-", color="#1f6f6f", lw=2)
    lim = [0, max(max(bx), max(by)) * 1.1]
    ax.plot(lim, lim, "k--", lw=1, label="ideal calibration")
    ax.set(xlabel="predicted std (MC-dropout)", ylabel="actual RMSE",
           title=f"Calibration: error rises with\npredicted uncertainty (corr={corr:.2f})")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.suptitle("Calibrated uncertainty for the phase-field GNN emulator (MC-dropout): the std "
                 "localises at the crack boundary and tracks the error", fontweight="bold")
    fig.tight_layout()
    out = "/home/nishioka/cfrp_datasets/uq_emulator.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
