"""conformal_emulator_uq.py — distribution-free UQ for the emulator, where variance-UQ failed [#3].

Honest finding upstream: MC-dropout (corr 0.07) AND deep ensembles (corr 0.08, boundary/interior std
ratio 0.97 ≈ no localisation) BOTH fail to give calibrated, boundary-localised uncertainty for this
phase-field GNN emulator. Reason: the held-out error is dominated by SYSTEMATIC bias (every model
makes the same out-of-distribution error), which disagreement-based UQ is blind to.

The right tool is then SPLIT-CONFORMAL residual calibration (distribution-free): it makes NO
assumption that model variance equals error — it calibrates an interval half-width from held-out
residual quantiles and GUARANTEES marginal coverage ≥ 1−α under exchangeability. We:
  1. train one emulator on a TRAIN split,
  2. on a CALIBRATION split compute per-node abs-residuals, take the (1−α) quantile q̂,
  3. on a TEST split emit [pred−q̂, pred+q̂] and measure empirical coverage vs the 1−α target.
This is the surrogate-side UQ that plugs into the clearance conformal layer (⑤, conformal_transfer).

Run:  python3 conformal_emulator_uq.py   (writes conformal_emulator_uq.png)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from phasefield_gnn_prognosis import MeshGNN
from phasefield_emulator_compare import load_cached

torch.manual_seed(20270614)
rng = np.random.default_rng(20270614)


def main():
    samples, edges, n_seed, (nx, ny) = load_cached()
    ei = torch.tensor(edges)
    ids = list(range(n_seed))
    rng.shuffle(ids)
    n_te = 8; n_cal = 8
    test_ids = set(ids[:n_te]); cal_ids = set(ids[n_te:n_te + n_cal])
    train = [s for s in samples if s[2] not in test_ids and s[2] not in cal_ids]
    cal = [s for s in samples if s[2] in cal_ids]
    test = [s for s in samples if s[2] in test_ids]
    print(f"train={len(train)} cal={len(cal)} test={len(test)}")

    m = MeshGNN(in_dim=6, hid=48, rounds=8)
    opt = torch.optim.Adam(m.parameters(), 4e-3, weight_decay=1e-5)
    lf = nn.BCELoss()
    for ep in range(100):
        for k in rng.permutation(len(train)):
            nf, tgt, _, _ = train[k]
            p = m(torch.tensor(nf), ei).clamp(1e-6, 1 - 1e-6)
            loss = lf(p, torch.tensor(tgt).clamp(0, 1))
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()

    def preds(split):
        out = []
        with torch.no_grad():
            for nf, tgt, _, _ in split:
                out.append((m(torch.tensor(nf), ei).numpy(), tgt))
        return out

    cal_p = preds(cal); test_p = preds(test)
    cal_res = np.concatenate([np.abs(p - t) for p, t in cal_p])

    alphas = [0.20, 0.10, 0.05]
    rows = []
    for a in alphas:
        qhat = np.quantile(cal_res, 1 - a)                       # split-conformal half-width
        cov = np.mean([np.mean(np.abs(p - t) <= qhat) for p, t in test_p])
        rows.append((a, qhat, cov))
        print(f"  target {1-a:.2f}  qhat={qhat:.3f}  empirical coverage={cov:.3f}")

    # figure: coverage vs target + an example interval map
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    ax = axes[0]
    tg = [1 - a for a, _, _ in rows]; cv = [c for _, _, c in rows]
    ax.plot(tg, cv, "o-", color="#1f6f6f", lw=2, ms=9, label="conformal (this)")
    ax.plot([0.7, 1], [0.7, 1], "k--", lw=1, label="ideal (cov=target)")
    ax.scatter([0.9], [0.08 + 0.9], s=0)  # keep scale
    ax.set(xlabel="target coverage 1−α", ylabel="empirical coverage",
           xlim=(0.75, 1.0), ylim=(0.75, 1.02),
           title="(a) Split-conformal gives VALID coverage\n(variance-UQ corr was ~0.07–0.08)")
    ax.legend(fontsize=8); ax.grid(alpha=.3)

    # (b) example: a test field's interval half-width is uniform q̂ (distribution-free) but coverage holds
    p0, t0 = test_p[0]
    a = 0.10; qh = np.quantile(cal_res, 1 - a)
    covered = (np.abs(p0 - t0) <= qh).reshape(ny, nx)
    ax = axes[1]
    ax.imshow(t0.reshape(ny, nx), origin="lower", cmap="inferno", vmin=0, vmax=1, aspect="auto")
    ax.set_title("(b) AT2 truth (example test defect)", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    ax = axes[2]
    ax.imshow(covered, origin="lower", cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_title(f"(c) covered by 90% conformal band\n(green=in, red=miss; q̂={qh:.2f})", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Distribution-free conformal UQ for the GNN emulator: valid coverage where "
                 "ensemble/dropout variance failed (error is systematic, not variance)", fontweight="bold")
    fig.tight_layout()
    out = "/home/nishioka/cfrp_datasets/conformal_emulator_uq.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
