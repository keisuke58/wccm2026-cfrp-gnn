"""denso_data_size.py — how much data does the residual-strain model need? Two learning curves
on the synthetic source/target setup of denso_residual_strain_ml.py:

  (A) simulated (source) plates n_s in {25, 50, 100, 200, 400, 800, 1600}: test error in the
      source domain, and on the target domain without (source-only) and with fine-tuning on
      n_t = 2 and 5 measured plates (5 points each);
  (B) the measured (target) side is in experiment 2 of the main script and in
      denso_ft_variants.py; (A) adds how it depends on the size of the source set.

Training budget: the number of optimiser STEPS is held fixed (as for n_s = 1600 at 1500 epochs,
batch 4096), so that small source sets are not merely under-trained. Fine-tuning uses L2-SP
with lambda = 1e-2 (fixed a priori, see denso_ft_variants.py). The source test set is a fixed
400 plates not used for training; target test as in the main script (200 plates).

Honest scope: one plate = one moulding condition with a 41-point through-thickness profile;
a real simulation gives a 3-D field, so "plates" here map to simulation RUNS (conditions),
not to data points. Synthetic target; constants illustrative. The numbers say how the error
scales in this model problem, not how many Ansys runs the real part needs.

Run:  python3 denso_data_size.py --workers 4 --out results/denso_data_size
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import sys
import time

import matplotlib
matplotlib.use("Agg")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import denso_residual_strain_ml as ml  # noqa: E402
from denso_ft_variants import fit_ft  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-source", default="25,50,100,200,400,800,1600")
    ap.add_argument("--n-target", default="2,5")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--points", type=int, default=5)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--ft-epochs", type=int, default=600)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="results/denso_data_size")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ns_list = [int(v) for v in a.n_source.split(",")]
    nt_list = [int(v) for v in a.n_target.split(",")]
    rng = np.random.default_rng(ml.SEED + 7)                    # independent of the main run
    t0 = time.time()
    src_all = ml.generate(ml.sample_params(rng, max(ns_list)), ml.SOURCE, a.workers)
    src_test = ml.generate(ml.sample_params(rng, 400), ml.SOURCE, a.workers)
    tgt_test = ml.generate(ml.sample_params(rng, 200), ml.TARGET, a.workers)
    tgt_pool = ml.generate(ml.sample_params(rng, max(nt_list) * a.repeats), ml.TARGET, a.workers)
    print(f"data in {time.time()-t0:.0f} s")
    Xst, Yst = ml.features(src_test, True); Xtt, Ytt = ml.features(tgt_test, True)
    steps_total = 1500 * math.ceil(max(ns_list) * ml.NZ / 4096)
    out = dict(args=vars(a), steps=steps_total, rows=[])
    for ns in ns_list:
        tr = src_all[:ns]
        Xtr, Ytr = ml.features(tr, True); sc = ml.Scaler(Xtr, Ytr)
        epochs = max(1, steps_total // math.ceil(len(Xtr) / 4096))
        m = ml.make_model(Xtr.shape[1], torch).to(dev)
        t1 = time.time(); ml.fit(m, Xtr, Ytr, sc, torch, dev, epochs, 2e-3, seed=ml.SEED)
        row = dict(n_source=ns, epochs=epochs, train_s=time.time() - t1,
                   source_rmse=ml.rmse(ml.predict(m, Xst, sc, torch, dev), Yst),
                   source_r2=ml.r2(ml.predict(m, Xst, sc, torch, dev), Yst),
                   target_source_only=ml.rmse(ml.predict(m, Xtt, sc, torch, dev), Ytt))
        for nt in nt_list:
            errs = []
            for r in range(a.repeats):
                plates = tgt_pool[r * max(nt_list): r * max(nt_list) + nt]
                rr = np.random.default_rng(ml.SEED + 1000 * r + nt)
                idx = [rr.choice(ml.NZ, size=a.points, replace=False) for _ in plates]
                Xs, Ys = ml.features(plates, True, idx)
                Ys = Ys + rr.normal(0, a.noise, Ys.shape).astype(np.float32)
                mf = fit_ft(copy.deepcopy(m), Xs, Ys, sc, torch, dev, a.ft_epochs, 3e-4, lam=1e-2, seed=ml.SEED + r)
                errs.append(ml.rmse(ml.predict(mf, Xtt, sc, torch, dev), Ytt))
            row[f"target_ft_{nt}"] = dict(mean=float(np.mean(errs)), sd=float(np.std(errs)))
        out["rows"].append(row)
        print(f"n_source {ns:5d}: source RMSE {row['source_rmse']:.3f} (R^2 {row['source_r2']:.4f}) | target "
              f"source-only {row['target_source_only']:.3f} | " +
              " | ".join(f"ft n_t={nt} {row[f'target_ft_{nt}']['mean']:.3f}" for nt in nt_list) +
              f"  ({row['train_s']:.0f} s)")
    out["target_std"] = float(np.std(Ytt))
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(out, f, indent=1)
    _plot(os.path.join(a.out, "denso_data_size.png"), out, nt_list)
    print(f"wrote {a.out}/results.json and denso_data_size.png")


def _plot(path, out, nt_list):
    sys.path.insert(0, os.path.join(HERE, "slides", "figure_sources"))
    try:                                   # repo convention: thesis_style, graceful fallback
        from thesis_style import use
        use(width_frac=0.6, aspect=0.75)
        ok = shutil.which("latex") is not None
        if ok and matplotlib.rcParams["text.usetex"]:
            try:
                from matplotlib.texmanager import TexManager
                TexManager().get_text_width_height_descent("lp", 10)
            except Exception:
                ok = False
        if not ok:
            matplotlib.rcParams["text.usetex"] = False
            matplotlib.rcParams["font.family"] = "serif"
            matplotlib.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "STIXGeneral"]
            matplotlib.rcParams["mathtext.fontset"] = "stix"
    except Exception:
        pass
    import matplotlib.pyplot as plt
    rows = out["rows"]; ns = [r["n_source"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.plot(ns, [r["source_rmse"] for r in rows], "k-o", ms=4, lw=1, label="source test (in-domain)")
    ax.plot(ns, [r["target_source_only"] for r in rows], "-s", color="0.55", ms=4, lw=1, label="target, source-only")
    for nt, ls in zip(nt_list, ("--", "-.")):
        ax.errorbar(ns, [r[f"target_ft_{nt}"]["mean"] for r in rows], yerr=[r[f"target_ft_{nt}"]["sd"] for r in rows],
                    color="k", ls=ls, marker="^", ms=4, lw=1, capsize=2, label=f"target, fine-tune on {nt} plates")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("simulated (source) plates = moulding conditions")
    ax.set_ylabel(r"RMSE [$10^{-3}$]")
    ax.set_title("learning curves (synthetic, illustrative constants)", fontsize=8)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
