"""pi_rigor.py — repeated k-fold CV with error bars: does the AT2 physics prior really help? [#3d]

The lambda sweep (pi_lambda_sweep.py) gave differences within noise (n=8 test, one ensemble each).
This adds statistical power and a paired design:

  * REPEATED K-FOLD CV over all 30 phase-field seeds -> every seed is held out once, so each repeat
    pools test predictions over the FULL dataset (test n=30), not 8.
  * PAIRED: within a repeat+fold, the baseline (lambda=0) and the physics ensemble are trained on the
    SAME split with the SAME member initialisations and data ordering -- the ONLY difference is the
    prior. So per-repeat dRMSE / dcorr isolate the prior's effect with minimal variance.
  * R independent repeats (fresh splits + seeds) give an error bar; we report mean +/- std and flag a
    difference as "within noise" unless |mean| exceeds ~2 standard errors.

Run:   python3 pi_rigor.py                              (writes pi_rigor.png)
Knobs: PI_R=3  PI_FOLDS=4  PI_K=3  PI_EPOCHS=30  PI_THREADS=4  PI_LAM=5e-3
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch

from phasefield_emulator_compare import load_cached
from pi_emulator_ensemble import train_one


def _safe_corr(a, b):
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def run_cv(samples, ei, lam, K, epochs, F, split_seed, member_seed0):
    """One full K-fold CV; return (rmse, corr) on out-of-fold predictions pooled over all seeds.

    The fold partition is fixed by `split_seed` and the K member initialisations by `member_seed0`,
    so calling this with the same seeds but different `lam` yields a paired baseline/physics pair.
    """
    n = len(samples)
    idx = np.arange(n)
    np.random.default_rng(split_seed).shuffle(idx)
    folds = [idx[i::F] for i in range(F)]
    err_all, std_all = [], []
    for test_idx in folds:
        test_set = set(int(i) for i in test_idx)
        train = [samples[i] for i in range(n) if i not in test_set]
        models = [train_one(train, ei, member_seed0 + j, lam_phys=lam, epochs=epochs)
                  for j in range(K)]
        with torch.no_grad():
            for i in test_idx:
                nf, tgt, _, _ = samples[int(i)]
                ps = np.stack([mm(torch.tensor(nf), ei).numpy() for mm in models])  # (K, N)
                mean, std = ps.mean(0), ps.std(0)
                err_all.append(np.abs(mean - tgt)); std_all.append(std)
    err = np.concatenate(err_all); std = np.concatenate(std_all)
    return float(np.sqrt((err ** 2).mean())), _safe_corr(std, err)


def _mean_se(x):
    x = np.asarray(x, dtype=np.float64)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def main():
    R = int(os.environ.get("PI_R", 3))
    F = int(os.environ.get("PI_FOLDS", 4))
    K = int(os.environ.get("PI_K", 3))
    epochs = int(os.environ.get("PI_EPOCHS", 30))
    threads = int(os.environ.get("PI_THREADS", "4"))
    if threads > 0:
        torch.set_num_threads(threads)
    lam = float(os.environ.get("PI_LAM", 5e-3))

    samples, edges, n_seed, (nx, ny) = load_cached()
    ei = torch.tensor(edges)
    print(f"rigor: R={R} folds={F} K={K} epochs={epochs} lambda={lam} threads={threads} "
          f"seeds={len(samples)} grid={nx}x{ny}", flush=True)

    rmse_b, rmse_p, corr_b, corr_p = [], [], [], []
    for r in range(R):
        split_seed = 1000 + r
        member0 = 2000 + 100 * r
        t0 = time.time()
        rb, cb = run_cv(samples, ei, 0.0, K, epochs, F, split_seed, member0)   # baseline
        rp, cp = run_cv(samples, ei, lam, K, epochs, F, split_seed, member0)   # physics (paired)
        rmse_b.append(rb); rmse_p.append(rp); corr_b.append(cb); corr_p.append(cp)
        print(f"  repeat {r + 1}/{R}: base RMSE={rb:.4f} corr={cb:+.2f} | "
              f"phys RMSE={rp:.4f} corr={cp:+.2f} | dRMSE={rp - rb:+.4f} dcorr={cp - cb:+.2f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    d_rmse = np.array(rmse_p) - np.array(rmse_b)
    d_corr = np.array(corr_p) - np.array(corr_b)
    mb_r, se_b_r = _mean_se(rmse_b); mp_r, se_p_r = _mean_se(rmse_p)
    mb_c, se_b_c = _mean_se(corr_b); mp_c, se_p_c = _mean_se(corr_p)
    md_r, se_d_r = _mean_se(d_rmse); md_c, se_d_c = _mean_se(d_corr)

    print(f"\n  metric        baseline(l=0)        physics(l={lam:g})")
    print(f"  RMSE          {mb_r:.4f} +/- {se_b_r:.4f}     {mp_r:.4f} +/- {se_p_r:.4f}")
    print(f"  corr(std,err) {mb_c:+.3f} +/- {se_b_c:.3f}     {mp_c:+.3f} +/- {se_p_c:.3f}")
    print(f"\n  paired delta (physics - baseline), over R={R} repeats:")
    print(f"    dRMSE = {md_r:+.4f} +/- {se_d_r:.4f}  (negative = physics better)")
    print(f"    dcorr = {md_c:+.3f} +/- {se_d_c:.3f}  (positive = physics better)")

    def _verdict(mean, se, good_sign):
        if not np.isfinite(se) or abs(mean) < 2 * se:
            return "within noise"
        return "physics BETTER" if (mean * good_sign > 0) else "physics WORSE"
    print(f"\n  Verdict  accuracy : {_verdict(md_r, se_d_r, -1)}")
    print(f"  Verdict  calibration: {_verdict(md_c, se_d_c, +1)}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.4))
    a1.bar([0, 1], [mb_r, mp_r], yerr=[se_b_r, se_p_r], capsize=6,
           color=["tab:gray", "tab:blue"]); a1.set_xticks([0, 1])
    a1.set_xticklabels(["baseline", f"phys l={lam:g}"]); a1.set_ylabel("test RMSE (mean +/- SE)")
    a1.set_title(f"Accuracy  (dRMSE={md_r:+.4f}+/-{se_d_r:.4f})")
    a2.bar([0, 1], [mb_c, mp_c], yerr=[se_b_c, se_p_c], capsize=6,
           color=["tab:gray", "tab:red"]); a2.set_xticks([0, 1])
    a2.set_xticklabels(["baseline", f"phys l={lam:g}"]); a2.set_ylabel("corr(std,err) (mean +/- SE)")
    a2.set_title(f"Calibration  (dcorr={md_c:+.3f}+/-{se_d_c:.3f})")
    fig.suptitle(f"Repeated {F}-fold CV, R={R}, K={K}, {epochs} ep — paired baseline vs physics")
    fig.tight_layout()
    fig.savefig("pi_rigor.png", dpi=130)
    print("\nwrote pi_rigor.png")


if __name__ == "__main__":
    main()
