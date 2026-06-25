"""pi_emulator_ensemble.py — PHYSICS-INFORMED deep-ensemble UQ for the phase-field GNN emulator [#3b].

uq_emulator_ensemble.py showed a plain deep ensemble (K independent MeshGNNs, std = disagreement)
calibrates far better than MC-dropout here. This asks the next question: does a PHYSICS prior on the
emulator help — both the prediction AND the uncertainty?

The AT2 (Ambrosio–Tortorelli) phase-field energy that generated the data carries a gradient term
~ (ell**2/2)|grad d|**2 that regularises the damage field over a length scale ell. Its exact discrete
analogue on the mesh graph is the graph-Dirichlet energy  sum_{(i,j) in E} (d_i - d_j)**2. We add a
small multiple of it to the per-emulator BCE loss, so every ensemble member is nudged toward fields
that respect that physical smoothness — except where the data genuinely demands a sharp crack.

We then run the SAME evaluation as the plain ensemble, head-to-head (lambda=0 vs lambda>0):
  * test RMSE                              — accuracy
  * corr(ensemble-std, abs-err)            — does disagreement track the real error?
  * crack-boundary / interior std ratio    — does uncertainty localise on the crack?
and print an honest verdict from the numbers (it may or may not help — that is the experiment).

Run:   python3 pi_emulator_ensemble.py                 (writes pi_emulator_ensemble.png)
Knobs: PI_K=4  PI_EPOCHS=70  PI_LAM=5e-3               (override ensemble size / epochs / prior weight)
       PI_SMOKE=1                                       (tiny CPU smoke: K=1, 2 epochs, few samples)
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn

from phasefield_gnn_prognosis import MeshGNN
from phasefield_emulator_compare import load_cached

rng = np.random.default_rng(20270615)


def graph_dirichlet_energy(pred: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Mean graph-Dirichlet energy of a per-node field — discrete AT2 gradient term.

    Parameters
    ----------
    pred : torch.Tensor, shape (N,)
        Per-node predicted damage in [0, 1].
    edge_index : torch.Tensor, shape (2, E)
        COO connectivity [src, dst].

    Returns
    -------
    torch.Tensor, scalar
        ``mean_{(i,j) in E} (pred_i - pred_j)**2``. Minimising it promotes the spatial
        coherence imposed by the phase-field regularisation length.
    """
    diff = pred[edge_index[0]] - pred[edge_index[1]]
    return (diff * diff).mean()


def train_one(train, ei, seed, lam_phys, epochs):
    """Train one MeshGNN with BCE + lam_phys * graph-Dirichlet prior (lam_phys=0 -> plain)."""
    torch.manual_seed(seed)
    g = np.random.default_rng(seed)
    m = MeshGNN(in_dim=6, hid=48, rounds=8)
    opt = torch.optim.Adam(m.parameters(), 4e-3, weight_decay=1e-5)
    lf = nn.BCELoss()
    for _ in range(epochs):
        for k in g.permutation(len(train)):
            nf, tgt, _, _ = train[k]
            p = m(torch.tensor(nf), ei).clamp(1e-6, 1 - 1e-6)
            loss = lf(p, torch.tensor(tgt).clamp(0, 1))
            if lam_phys > 0:
                loss = loss + lam_phys * graph_dirichlet_energy(p, ei)
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    return m


def _safe_corr(a, b):
    """Pearson corr, NaN-safe for degenerate (constant / tiny) inputs."""
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def eval_ensemble(models, test, ei):
    """Ensemble mean/std over `models`; return metrics dict + first-sample maps for plotting."""
    all_err, all_std, maps = [], [], []
    with torch.no_grad():
        for nf, tgt, _, _ in test:
            xs = torch.tensor(nf)
            ps = np.stack([mm(xs, ei).numpy() for mm in models])      # (K, N)
            mean, std = ps.mean(0), ps.std(0)
            all_err.append(np.abs(mean - tgt)); all_std.append(std)
            maps.append((tgt, mean, std))
    err = np.concatenate(all_err); std = np.concatenate(all_std)
    tgt_all = np.concatenate([m[0] for m in maps])
    boundary = (tgt_all > 0.2) & (tgt_all < 0.8)
    interior_std = std[~boundary].mean() if (~boundary).any() else float("nan")
    loc = std[boundary].mean() / max(interior_std, 1e-9) if boundary.any() else float("nan")
    return {
        "rmse": float(np.sqrt((err ** 2).mean())),
        "corr": _safe_corr(std, err),
        "loc": float(loc),
        "maps": maps,
    }


def main():
    smoke = os.environ.get("PI_SMOKE") == "1"
    K = 1 if smoke else int(os.environ.get("PI_K", 4))
    epochs = 2 if smoke else int(os.environ.get("PI_EPOCHS", 70))
    lam = float(os.environ.get("PI_LAM", 5e-3))

    print(f"Loading cache + training ensembles (K={K}, epochs={epochs}, lambda_phys={lam}, "
          f"smoke={smoke})...")
    samples, edges, n_seed, (nx, ny) = load_cached()
    ei = torch.tensor(edges)
    test_ids = set(range(n_seed - 8, n_seed))
    train = [s for s in samples if s[2] not in test_ids]
    test = [s for s in samples if s[2] in test_ids]
    if smoke:
        train, test = train[:4], test[:2]
    print(f"train={len(train)} test={len(test)} grid={nx}x{ny}")

    base = [train_one(train, ei, seed=100 + j, lam_phys=0.0, epochs=epochs) for j in range(K)]
    phys = [train_one(train, ei, seed=200 + j, lam_phys=lam, epochs=epochs) for j in range(K)]

    rb = eval_ensemble(base, test, ei)
    rp = eval_ensemble(phys, test, ei)

    print("\n                  RMSE      corr(std,err)   boundary/interior std")
    print(f"  baseline (l=0)  {rb['rmse']:.4f}   {rb['corr']:>6.2f}          {rb['loc']:.2f}")
    print(f"  phys (l={lam:g})   {rp['rmse']:.4f}   {rp['corr']:>6.2f}          {rp['loc']:.2f}")

    d_rmse = rb["rmse"] - rp["rmse"]
    better_acc = d_rmse > 0
    better_cal = (rp["corr"] > rb["corr"]) if np.isfinite(rp["corr"] + rb["corr"]) else False
    verdict = {
        (True, True): "physics prior HELPS both accuracy and calibration",
        (True, False): "physics prior helps accuracy but not calibration",
        (False, True): "physics prior helps calibration but not accuracy",
        (False, False): "physics prior does NOT help here (honest negative)",
    }[(better_acc, better_cal)]
    print(f"\nVerdict: {verdict}  (dRMSE={d_rmse:+.4f}, dcorr={rp['corr'] - rb['corr']:+.2f})")

    if smoke:
        print("[smoke] skipping figure"); return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tgt0, mean0, std0 = rp["maps"][0]
    fig = plt.figure(figsize=(14, 4.7))
    panels = [(tgt0, "AT2 truth", "inferno", 1.0),
              (mean0, "phys-ensemble mean", "inferno", 1.0),
              (std0, "phys-ensemble std (UQ)", "viridis", None),
              (np.abs(mean0 - tgt0), "abs error", "magma", None)]
    for col, (f, lab, cmap, vm) in enumerate(panels):
        ax = fig.add_subplot(1, 4, col + 1)
        im = ax.imshow(f.reshape(ny, nx), origin="lower", cmap=cmap, aspect="auto",
                       vmin=0, vmax=vm)
        ax.set_title(lab, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Physics-informed deep ensemble (K={K}, lambda={lam:g})  "
                 f"corr(std,err)={rp['corr']:.2f}  bnd/int={rp['loc']:.2f}", fontsize=11)
    fig.tight_layout()
    fig.savefig("pi_emulator_ensemble.png", dpi=130)
    print("wrote pi_emulator_ensemble.png")


if __name__ == "__main__":
    main()
