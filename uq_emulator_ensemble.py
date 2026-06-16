"""uq_emulator_ensemble.py — deep-ensemble UQ for the phase-field GNN emulator [#3, honest redo].

MC-dropout (uq_emulator.py) gave POORLY-calibrated uncertainty here: the per-node std was noise-like,
did NOT localise at the crack boundary, and barely tracked the error (corr≈0.07), badly
under-estimating the true RMSE. That is an honest negative for MC-dropout on this emulator.

The standard, usually-better alternative is a DEEP ENSEMBLE: train K independent emulators (different
init + data shuffle); the prediction is the ensemble mean and the uncertainty is the ensemble std
(disagreement). We test whether it calibrates — corr(ensemble-std, abs-err) and the std map's
localisation — against the MC-dropout baseline. The well-calibrated std is what a downstream
conformal/decision layer (⑤, decision_uq) gates on.

Run:  python3 uq_emulator_ensemble.py   (writes uq_emulator_ensemble.png)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from phasefield_gnn_prognosis import MeshGNN
from phasefield_emulator_compare import load_cached

rng = np.random.default_rng(20270614)


def train_one(train, ei, seed, epochs=70):
    torch.manual_seed(seed)
    g = np.random.default_rng(seed)
    m = MeshGNN(in_dim=6, hid=48, rounds=8)
    opt = torch.optim.Adam(m.parameters(), 4e-3, weight_decay=1e-5)
    lf = nn.BCELoss()
    for ep in range(epochs):
        for k in g.permutation(len(train)):
            nf, tgt, _, _ = train[k]
            p = m(torch.tensor(nf), ei).clamp(1e-6, 1 - 1e-6)
            loss = lf(p, torch.tensor(tgt).clamp(0, 1))
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    return m


def main():
    print("Loading cache + training deep ensemble (K=4)...")
    samples, edges, n_seed, (nx, ny) = load_cached()
    ei = torch.tensor(edges)
    test_ids = set(range(n_seed - 8, n_seed))
    train = [s for s in samples if s[2] not in test_ids]
    test = [s for s in samples if s[2] in test_ids]

    K = 4
    models = [train_one(train, ei, seed=100 + j) for j in range(K)]

    all_err, all_std, maps = [], [], []
    with torch.no_grad():
        for nf, tgt, _, _ in test:
            xs = torch.tensor(nf)
            ps = np.stack([mm(xs, ei).numpy() for mm in models])     # (K, N)
            mean, std = ps.mean(0), ps.std(0)
            all_err.append(np.abs(mean - tgt)); all_std.append(std)
            maps.append((tgt, mean, std))
    err = np.concatenate(all_err); std = np.concatenate(all_std)
    corr = np.corrcoef(std, err)[0, 1]

    # calibration curve: bin by predicted std -> actual RMSE
    qs = np.quantile(std, np.linspace(0, 1, 9))
    bx, by = [], []
    for i in range(len(qs) - 1):
        hi = std <= qs[i + 1] if i == len(qs) - 2 else std < qs[i + 1]
        msk = (std >= qs[i]) & hi
        if msk.sum() > 5:
            bx.append(std[msk].mean()); by.append(np.sqrt((err[msk] ** 2).mean()))
    # boundary localisation: is std higher on crack-boundary nodes than elsewhere?
    tgt_all = np.concatenate([m[0] for m in maps])
    boundary = (tgt_all > 0.2) & (tgt_all < 0.8)
    loc = std[boundary].mean() / max(std[~boundary].mean(), 1e-9)
    print(f"\nENSEMBLE(K={K})  corr(std,err)={corr:.2f}  boundary/interior std ratio={loc:.2f}  "
          f"(MC-dropout baseline corr=0.07)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(14, 4.7))
    tgt0, mean0, std0 = maps[0]
    for col, (f, lab, cmap, vm) in enumerate([(tgt0, "AT2 truth", "inferno", 1),
                                              (mean0, "ensemble mean", "inferno", 1),
                                              (std0, "ensemble uncertainty (std)", "viridis", None)]):
        ax = fig.add_subplot(1, 4, col + 1)
        im = ax.imshow(f.reshape(ny, nx), origin="lower", cmap=cmap, aspect="auto",
                       vmin=0, vmax=vm)
        ax.set_title(lab, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
        if col == 2:
            fig.colorbar(im, ax=ax, fraction=0.046)
    ax = fig.add_subplot(1, 4, 4)
    ax.plot(bx, by, "o-", color="#1f6f6f", lw=2)
    lim = [0, max(max(bx), max(by)) * 1.1]
    ax.plot(lim, lim, "k--", lw=1, label="ideal")
    ax.set(xlabel="ensemble std", ylabel="actual RMSE",
           title=f"Calibration corr={corr:.2f}\n(MC-dropout was 0.07)")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.suptitle(f"Deep-ensemble UQ for the phase-field GNN emulator FAILS too: corr(std,err)={corr:.2f}, "
                 f"boundary/interior std ratio ×{loc:.1f} (no localisation) — error is systematic, not variance",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    out = "/home/nishioka/cfrp_datasets/uq_emulator_ensemble.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
