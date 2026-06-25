"""pi_lambda_sweep.py — sweep the physics-prior weight lambda for the PI deep-ensemble emulator [#3c].

pi_emulator_ensemble.py found the AT2 graph-Dirichlet prior cuts test RMSE ~12% at lambda=5e-3 but
HURTS ensemble calibration: corr(std,err) fell 0.13 -> -0.02. The reason is structural — imposing the
SAME smoothness prior on every member makes them agree, collapsing the disagreement a deep ensemble
turns into uncertainty. So accuracy and calibration pull in opposite directions in lambda.

This sweeps lambda to map that trade-off and pick the sweet spot: the lambda with the best (lowest)
RMSE among those that do NOT degrade calibration below the plain ensemble (lambda=0). It reuses the
exact training / evaluation used in pi_emulator_ensemble.py so the numbers are directly comparable.

Run:   python3 pi_lambda_sweep.py                       (writes pi_lambda_sweep.png)
Knobs: PI_K=4  PI_EPOCHS=40  PI_THREADS=4  PI_LAMS=0,1e-3,3e-3,5e-3,1e-2,3e-2
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch

from phasefield_emulator_compare import load_cached
from pi_emulator_ensemble import eval_ensemble, train_ensemble


def main():
    K = int(os.environ.get("PI_K", 4))
    epochs = int(os.environ.get("PI_EPOCHS", 40))
    threads = int(os.environ.get("PI_THREADS", "4"))
    if threads > 0:
        torch.set_num_threads(threads)
    lams = [float(x) for x in os.environ.get("PI_LAMS", "0,1e-3,3e-3,5e-3,1e-2,3e-2").split(",")]

    samples, edges, n_seed, (nx, ny) = load_cached()
    ei = torch.tensor(edges)
    test_ids = set(range(n_seed - 8, n_seed))
    train = [s for s in samples if s[2] not in test_ids]
    test = [s for s in samples if s[2] in test_ids]
    print(f"sweep lambdas={lams}  K={K} epochs={epochs} threads={threads} "
          f"train={len(train)} test={len(test)} grid={nx}x{ny}", flush=True)

    rows = []  # (lam, rmse, corr, loc)
    for li, lam in enumerate(lams):
        t0 = time.time()
        models = train_ensemble(f"l={lam:g}", train, ei, 300 + 10 * li, lam, K, epochs)
        r = eval_ensemble(models, test, ei)
        rows.append((lam, r["rmse"], r["corr"], r["loc"]))
        print(f"  lambda={lam:<7g} RMSE={r['rmse']:.4f} corr={r['corr']:+.2f} "
              f"bnd/int={r['loc']:.2f} ({time.time() - t0:.0f}s)", flush=True)

    base_rmse, base_corr = rows[0][1], rows[0][2]
    print("\n  lambda     RMSE      corr(std,err)   bnd/int    dRMSE vs base")
    for lam, rmse, corr, loc in rows:
        print(f"  {lam:<8g}  {rmse:.4f}    {corr:+.2f}          {loc:.2f}      {base_rmse - rmse:+.4f}")

    # Sweet spot: minimum RMSE among lambdas that keep calibration >= the plain ensemble.
    keep = [r for r in rows if np.isfinite(r[2]) and r[2] >= base_corr - 1e-9]
    if keep:
        best = min(keep, key=lambda r: r[1])
        print(f"\nSweet spot (min RMSE s.t. corr >= baseline {base_corr:+.2f}): "
              f"lambda={best[0]:g}  RMSE={best[1]:.4f} ({base_rmse - best[1]:+.4f})  corr={best[2]:+.2f}")
    else:
        print(f"\nNo lambda>0 preserves baseline calibration ({base_corr:+.2f}) — "
              f"the accuracy gain genuinely costs ensemble diversity here (honest trade-off).")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = list(range(len(rows)))
    fig, ax1 = plt.subplots(figsize=(7.2, 4.6))
    ax1.plot(xs, [r[1] for r in rows], "o-", color="tab:blue", label="test RMSE")
    ax1.set_xlabel("lambda_phys (physics-prior weight)")
    ax1.set_ylabel("test RMSE", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_xticks(xs)
    ax1.set_xticklabels([f"{r[0]:g}" for r in rows])
    ax2 = ax1.twinx()
    ax2.plot(xs, [r[2] for r in rows], "s--", color="tab:red", label="corr(std,err)")
    ax2.axhline(base_corr, color="tab:red", ls=":", alpha=0.5)
    ax2.set_ylabel("corr(ensemble-std, abs-err)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    fig.suptitle(f"Physics-prior sweep: accuracy vs calibration (K={K}, {epochs} ep)")
    fig.tight_layout()
    fig.savefig("pi_lambda_sweep.png", dpi=130)
    print("wrote pi_lambda_sweep.png")


if __name__ == "__main__":
    main()
