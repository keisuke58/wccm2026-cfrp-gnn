"""inverse_defect_gnn.py — gradient-based defect identification through the differentiable GNN [#1].

⑥ gave a differentiable GNN that emulates the AT2 forward map (seed defect field -> crack field).
Because it is differentiable, we can INVERT it: given an OBSERVED crack field, recover the defect
that produced it by gradient descent through the emulator — the inverse-problem core of the Keio
thesis (and the fast, differentiable counterpart of FMPE's posterior sampling, fmpe_defect.py).

Pipeline
--------
  1. Train the forward emulator (MeshGNN) on cached AT2 forward pairs.
  2. Pick a TRUE defect (cx*, cy*, size*); run the REAL AT2 solver -> observed crack field
     (so the inversion is tested against true physics, not the emulator's own output).
  3. Parametrise a DIFFERENTIABLE soft seed s(θ), θ=(cx, cy, log r); minimise
     ||emulator(s(θ)) - observed||² by Adam -> recovered θ̂.
  4. Report recovered vs true and the reconstructed field. Repeat over several true defects.

This closes the loop: forward emulation (⑥) + differentiable inversion (#1) = a fast defect-ID
surrogate, with the honest caveat that location/through-thickness (cx, cy) invert far better than
absolute size from a single field.

Run:  python3 inverse_defect_gnn.py   (writes inverse_defect_gnn.png)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

import cfrp_phasefield_2d as pf
from phasefield_gnn_prognosis import MeshGNN
from phasefield_emulator_compare import load_cached

SEED = 20270614
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)


def train_emulator(train, ei, epochs=100):
    m = MeshGNN(in_dim=6, hid=48, rounds=8)
    opt = torch.optim.Adam(m.parameters(), 4e-3, weight_decay=1e-5)
    lf = nn.BCELoss()
    for ep in range(epochs):
        for k in rng.permutation(len(train)):
            nf, tgt, _, _ = train[k]
            p = m(torch.tensor(nf), ei).clamp(1e-6, 1 - 1e-6)
            loss = lf(p, torch.tensor(tgt).clamp(0, 1))
            opt.zero_grad(); loss.backward(); opt.step()
    for p in m.parameters():
        p.requires_grad_(False)        # freeze emulator for inversion
    return m


def soft_seed_features(cx, cy, logr, xs, ys, Gc, temp=0.01):
    """Differentiable 6-feature node tensor for a soft elliptical seed at (cx,cy), size exp(logr)."""
    r = torch.exp(logr)
    rx = r.clamp(0.02, 0.25)
    ry = (0.4 * rx).clamp(0.015, 0.1)
    q = 1.0 - ((xs - cx) / rx) ** 2 - ((ys - cy) / ry) ** 2
    seed = torch.sigmoid(q / temp)                              # soft ellipse in [0,1]
    sx = cx.expand_as(xs); sy = cy.expand_as(ys)
    return torch.stack([seed, Gc, xs, ys, sx, sy], dim=1)


def main():
    print("Loading cached AT2 forward pairs + training emulator...")
    samples, edges, n_seed, (nx, ny) = load_cached()
    ei = torch.tensor(edges)
    emu = train_emulator(samples, ei)        # train on all (we test inversion on fresh AT2 solves)

    cfg = pf.LaminateConfig(nx=40, ny=32)
    Gc_map, *_ = pf.build_laminate_maps(cfg)
    Gc = torch.tensor((Gc_map / Gc_map.mean()).reshape(-1).astype(np.float32))
    xs = torch.tensor(np.tile(np.linspace(0, 1, nx), ny).astype(np.float32))
    ys = torch.tensor(np.repeat(np.linspace(0, 1, ny), nx).astype(np.float32))

    # several TRUE defects; invert each from its REAL AT2 crack field
    truths = [(0.45, 5, -0.2), (0.6, 11, 0.3), (0.35, 3, 0.0), (0.55, 14, 0.1)]
    results = []
    for (cx_t, layer_t, logsz_t) in truths:
        d0 = pf.seed_defect(cx_t, 0.5, layer_t, logsz_t, cfg)
        obs = torch.tensor(pf.simulate_growth(d0, 0.16, cfg).d_final.reshape(-1).astype(np.float32))
        cy_t = (layer_t / (cfg.n_layers_data - 1))             # true normalised through-thickness

        # invert: optimise (cx, cy, logr)
        cx = torch.tensor(0.5, requires_grad=True)
        cy = torch.tensor(0.5, requires_grad=True)
        logr = torch.tensor(np.log(0.08), dtype=torch.float32, requires_grad=True)
        opt = torch.optim.Adam([cx, cy, logr], lr=0.03)
        for it in range(300):
            nf = soft_seed_features(cx, cy, logr, xs, ys, Gc)
            pred = emu(nf, ei)
            loss = ((pred - obs) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                cx.clamp_(0.1, 0.9); cy.clamp_(0.05, 0.95); logr.clamp_(np.log(0.02), np.log(0.25))
        rec = (float(cx), float(cy), float(torch.exp(logr)))
        results.append(dict(true=(cx_t, cy_t, 2 ** logsz_t), rec=rec, obs=obs.numpy(),
                            recon=emu(soft_seed_features(cx, cy, logr, xs, ys, Gc), ei).detach().numpy()))
        print(f"  true (cx={cx_t:.2f}, cy={cy_t:.2f})  ->  recovered (cx={rec[0]:.2f}, cy={rec[1]:.2f})  "
              f"|Δcx|={abs(rec[0]-cx_t):.3f} |Δcy|={abs(rec[1]-cy_t):.3f}")

    dcx = np.mean([abs(r["rec"][0] - r["true"][0]) for r in results])
    dcy = np.mean([abs(r["rec"][1] - r["true"][1]) for r in results])
    print(f"\nmean |Δcx|={dcx:.3f}  mean |Δcy(through-thickness)|={dcy:.3f}  (domain spans [0,1])")

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(14, 5.6))
    for col, r in enumerate(results[:2]):
        for row, (f, lab) in enumerate([(r["obs"], "observed (AT2)"), (r["recon"], "GNN-inverted recon")]):
            ax = fig.add_subplot(2, 4, row * 4 + col + 1)
            ax.imshow(f.reshape(ny, nx), origin="lower", cmap="inferno", vmin=0, vmax=1, aspect="auto")
            ax.set_title(f"true cy={r['true'][1]:.2f}→rec {r['rec'][1]:.2f}\n{lab}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(1, 2, 2)
    tcx = [r["true"][0] for r in results]; rcx = [r["rec"][0] for r in results]
    tcy = [r["true"][1] for r in results]; rcy = [r["rec"][1] for r in results]
    ax.scatter(tcx, rcx, s=90, c="#1f6f6f", ec="k", label=f"in-plane cx (MAE {dcx:.2f})")
    ax.scatter(tcy, rcy, s=90, c="#c0392b", marker="s", ec="k", label=f"through-thickness cy (MAE {dcy:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="true location", ylabel="GNN-inverted location", xlim=(0, 1), ylim=(0, 1),
           title="Differentiable inversion recovers the defect location\n(gradient descent through the emulator)")
    ax.legend(fontsize=9); ax.grid(alpha=.3)
    fig.suptitle("Defect identification by gradient descent through the differentiable phase-field "
                 "GNN emulator (inverse problem, Keio Phase-2)", fontweight="bold")
    fig.tight_layout()
    out = "/home/nishioka/cfrp_datasets/inverse_defect_gnn.png"
    fig.savefig(out, dpi=140)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
