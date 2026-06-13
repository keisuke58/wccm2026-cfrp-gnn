"""inverse_rigor.py — noise robustness + inversion uncertainty for the differentiable inverse [#1+].

inverse_defect_gnn.py showed the differentiable emulator recovers a defect's location (MAE ~0.05).
For the forward-inverse PAPER the inverse half needs two more things a referee will ask for:
  (1) NOISE ROBUSTNESS — invert from observed fields corrupted by Gaussian noise of growing σ;
      report recovery error vs noise (graceful degradation, not a cliff).
  (2) INVERSION UNCERTAINTY — from K random initialisations the spread of recovered θ is a
      cheap posterior proxy. HYPOTHESIS (tested): spread should GROW with noise and bracket the
      actual error. RESULT: spread is same-order (0.04–0.06) as the error but NON-MONOTONIC in σ
      → only a coarse, non-calibrated proxy. We report both honestly.

Run on frontale (idle).  Writes inverse_rigor.png.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

import cfrp_phasefield_2d as pf
from phasefield_gnn_prognosis import MeshGNN
from phasefield_emulator_compare import load_cached

torch.manual_seed(20270614)
rng = np.random.default_rng(20270614)


def train_emulator(train, ei, epochs=100):
    m = MeshGNN(in_dim=6, hid=48, rounds=8)
    opt = torch.optim.Adam(m.parameters(), 4e-3, weight_decay=1e-5)
    lf = nn.BCELoss()
    for _ in range(epochs):
        for k in rng.permutation(len(train)):
            nf, tgt, _, _ = train[k]
            p = m(torch.tensor(nf), ei).clamp(1e-6, 1 - 1e-6)
            lf(p, torch.tensor(tgt).clamp(0, 1)).backward()
            opt.step(); opt.zero_grad()
    for p in m.parameters():
        p.requires_grad_(False)
    m.eval()
    return m


def soft_seed(cx, cy, logr, xs, ys, Gc, temp=0.01):
    r = torch.exp(logr); rx = r.clamp(0.02, 0.25); ry = (0.4 * rx).clamp(0.015, 0.1)
    seed = torch.sigmoid((1.0 - ((xs - cx) / rx) ** 2 - ((ys - cy) / ry) ** 2) / temp)
    return torch.stack([seed, Gc, xs, ys, cx.expand_as(xs), cy.expand_as(ys)], dim=1)


def invert(emu, obs, xs, ys, Gc, ei, init, iters=250):
    cx = torch.tensor(init[0], requires_grad=True)
    cy = torch.tensor(init[1], requires_grad=True)
    logr = torch.tensor(np.log(0.08), dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([cx, cy, logr], lr=0.03)
    for _ in range(iters):
        loss = ((emu(soft_seed(cx, cy, logr, xs, ys, Gc), ei) - obs) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            cx.clamp_(0.1, 0.9); cy.clamp_(0.05, 0.95); logr.clamp_(np.log(0.02), np.log(0.25))
    return float(cx), float(cy)


def main():
    samples, edges, n_seed, (nx, ny) = load_cached()
    ei = torch.tensor(edges)
    emu = train_emulator(samples, ei)
    cfg = pf.LaminateConfig(nx=40, ny=32)
    Gc_map, *_ = pf.build_laminate_maps(cfg)
    Gc = torch.tensor((Gc_map / Gc_map.mean()).reshape(-1).astype(np.float32))
    xs = torch.tensor(np.tile(np.linspace(0, 1, nx), ny).astype(np.float32))
    ys = torch.tensor(np.repeat(np.linspace(0, 1, ny), nx).astype(np.float32))

    truths = [(0.45, 5), (0.6, 11), (0.35, 3), (0.55, 14), (0.5, 8)]
    noises = [0.0, 0.05, 0.10, 0.20]
    K = 6                                          # restarts for inversion-UQ
    err_by_noise, spread_by_noise = {s: [] for s in noises}, {s: [] for s in noises}
    grng = np.random.default_rng(7)
    for (cx_t, layer_t) in truths:
        d0 = pf.seed_defect(cx_t, 0.5, layer_t, 0.0, cfg)
        clean = pf.simulate_growth(d0, 0.16, cfg).d_final.reshape(-1).astype(np.float32)
        cy_t = layer_t / (cfg.n_layers_data - 1)
        for s in noises:
            obs = torch.tensor(np.clip(clean + grng.normal(0, s, clean.shape), 0, 1).astype(np.float32))
            recs = np.array([invert(emu, obs, xs, ys, Gc, ei,
                                    init=(grng.uniform(0.2, 0.8), grng.uniform(0.2, 0.8)))
                             for _ in range(K)])
            mean_rec = recs.mean(0)
            err = np.hypot(mean_rec[0] - cx_t, mean_rec[1] - cy_t)     # location error
            spread = np.hypot(recs[:, 0].std(), recs[:, 1].std())      # inversion uncertainty
            err_by_noise[s].append(err); spread_by_noise[s].append(spread)

    print("noise  mean-loc-err  mean-inversion-spread")
    me, ms = [], []
    for s in noises:
        e, sp = np.mean(err_by_noise[s]), np.mean(spread_by_noise[s])
        me.append(e); ms.append(sp)
        print(f"  σ={s:.2f}    {e:.3f}        {sp:.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    ax.plot(noises, me, "o-", color="#1f6f6f", lw=2, label="location error")
    ax.plot(noises, ms, "s--", color="#c0392b", lw=2, label="inversion spread (UQ)")
    ax.set(xlabel="observation noise σ", ylabel="location [domain units]",
           title="(a) Location error ~flat in noise (robust); spread grows but non-monotonic")
    ax.legend(fontsize=9); ax.grid(alpha=.3)
    ax = axes[1]
    allx = [spread_by_noise[s][i] for s in noises for i in range(len(truths))]
    ally = [err_by_noise[s][i] for s in noises for i in range(len(truths))]
    ax.scatter(allx, ally, s=50, c="#34495e", ec="k")
    lim = [0, max(max(allx), max(ally)) * 1.1 + 1e-3]
    ax.plot(lim, lim, "k--", lw=1, label="error = spread")
    ax.set(xlabel="inversion spread (restart std)", ylabel="actual location error",
           xlim=lim, ylim=lim, title="(b) Spread is same-order as error — a coarse proxy, not calibrated")
    ax.legend(fontsize=9); ax.grid(alpha=.3)
    fig.suptitle("Differentiable inversion — robustness & uncertainty: error degrades gracefully "
                 "with noise; restart-spread is a coarse (non-calibrated) uncertainty proxy", fontweight="bold")
    fig.tight_layout()
    out = "/home/nishioka/cfrp_datasets/inverse_rigor.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
