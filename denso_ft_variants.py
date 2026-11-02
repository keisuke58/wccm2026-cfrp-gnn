"""denso_ft_variants.py — why does fine-tuning on ONE measured plate do worse than using the
pre-trained model as is (0.456 vs 0.294 x1e-3 in denso_residual_strain_ml.py), and which
restriction of the fine-tune fixes it?

Same data, same pre-trained "context" model and the same measured points as experiment 2 of
denso_residual_strain_ml.py (identical seeds and generation order), so that the baseline row
reproduces that experiment. Fine-tune variants (all start from the pre-trained weights):

  full-600     baseline: all layers, Adam lr 3e-4, weight decay 1e-4 (towards 0), 600 epochs
  full-100     the same with 100 epochs (fewer steps = implicit pull towards the start)
  last         only the output layer is updated (600 epochs, lr 3e-4)
  affine       no gradient training: per output, y = s f(x) + b fitted by ridge least squares
               towards s = 1, b = 0 (mu = 1e-2 on normalised outputs)
  l2sp-fixed   all layers, loss + lambda ||theta - theta_0||^2 (Li et al., ICML 2018), lambda = 1e-2
  l2sp-cv      as l2sp, lambda chosen from {1e-4 .. 1} by cross-validation on the MEASURED points
               only (leave-one-point-out for one plate, leave-one-plate-out otherwise, <= 5 folds)
Fixed hyper-parameters were set before looking at the test error and are not tuned on it;
only l2sp-cv selects anything, and it selects from the measured points alone.

Honest scope: synthetic target (uniformly shifted constants), so a good variant here is a
candidate for the real data, not evidence that it works there. Same caveats as
denso_residual_strain_ml.py.

Run:  python3 denso_ft_variants.py --workers 4 --out results/denso_ft_variants
      python3 denso_ft_variants.py --quick --workers 4         (smoke test)
"""
from __future__ import annotations

import argparse
import copy
import json
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

LAM_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
VARIANTS = ["full-600", "full-100", "last", "affine", "l2sp-fixed", "l2sp-cv"]


def fit_ft(model, X, Y, sc, torch, dev, epochs, lr, lam=0.0, last_only=False, seed=0):
    """Full-batch fine-tune with an optional L2-SP pull towards the starting weights."""
    torch.manual_seed(seed)
    Xt = torch.tensor((X - sc.mx) / sc.sx, device=dev)
    Yt = torch.tensor((Y - sc.my) / sc.sy, device=dev)
    theta0 = [p.detach().clone() for p in model.parameters()]
    params = list(model[-1].parameters()) if last_only else list(model.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    model.train()
    for _ in range(epochs):
        loss = torch.mean((model(Xt) - Yt) ** 2)
        if lam > 0:
            loss = loss + lam * sum(torch.sum((p - p0) ** 2) for p, p0 in zip(model.parameters(), theta0))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return model


def affine(base, X, Y, sc, torch, dev, mu=1e-2):
    """Per-output ridge fit of y = s f + b (normalised units), shrunk towards s = 1, b = 0."""
    F = (ml.predict(base, X, sc, torch, dev) - sc.my) / sc.sy
    Yn = (Y - sc.my) / sc.sy
    coef = []
    for j in range(Y.shape[1]):
        A = np.stack([F[:, j], np.ones(len(F))], axis=1)
        prior = np.array([1.0, 0.0])
        coef.append(np.linalg.solve(A.T @ A + mu * np.eye(2), A.T @ Yn[:, j] + mu * prior))
    coef = np.array(coef)

    def pred(Xq):
        f = (ml.predict(base, Xq, sc, torch, dev) - sc.my) / sc.sy
        return (f * coef[:, 0] + coef[:, 1]) * sc.sy + sc.my
    return pred, coef


def cv_lambda(base, Xs, Ys, groups, sc, torch, dev, epochs, lr, seed):
    """Choose lambda for L2-SP using only the measured points (grouped K-fold, K <= 5)."""
    ug = np.unique(groups)
    folds = [np.array([g]) for g in ug] if len(ug) > 1 else [np.array([i]) for i in range(len(Ys))]
    if len(ug) == 1:
        keys = np.arange(len(Ys))
    else:
        keys = groups
    folds = folds[:5]
    scores = []
    for lam in LAM_GRID:
        err = []
        for f in folds:
            te = np.isin(keys, f); tr = ~te
            m = fit_ft(copy.deepcopy(base), Xs[tr], Ys[tr], sc, torch, dev, epochs, lr, lam=lam, seed=seed)
            err.append(np.mean((ml.predict(m, Xs[te], sc, torch, dev) - Ys[te]) ** 2))
        scores.append(float(np.mean(err)))
    return LAM_GRID[int(np.argmin(scores))], scores


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-source", type=int, default=2000)
    ap.add_argument("--n-target-test", type=int, default=200)
    ap.add_argument("--n-target", default="1,2,5,10")
    ap.add_argument("--pool-max", type=int, default=50, help="as --n-target max of the main run, "
                    "so that the measured plates are the same")
    ap.add_argument("--points", type=int, default=5)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--ft-epochs", type=int, default=600)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results/denso_ft_variants")
    a = ap.parse_args()
    if a.quick:
        a.n_source, a.n_target_test, a.epochs, a.ft_epochs, a.repeats = 120, 40, 400, 300, 3
        a.n_target, a.pool_max = "1,2,5", 10
    os.makedirs(a.out, exist_ok=True)
    import torch
    dev = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available())
                       else ("cpu" if a.device == "auto" else a.device))
    print(f"device {dev} | torch {torch.__version__}")

    # ---- identical data and pre-training to denso_residual_strain_ml.py ----------------
    rng = np.random.default_rng(ml.SEED); t0 = time.time()
    src = ml.generate(ml.sample_params(rng, a.n_source), ml.SOURCE, a.workers)
    tgt_test = ml.generate(ml.sample_params(rng, a.n_target_test), ml.TARGET, a.workers)
    tgt_pool = ml.generate(ml.sample_params(rng, a.pool_max * a.repeats), ml.TARGET, a.workers)
    print(f"data in {time.time()-t0:.0f} s")
    ntr = int(0.8 * len(src)); tr = src[:ntr]
    Xtr, Ytr = ml.features(tr, True); sc = ml.Scaler(Xtr, Ytr)
    base = ml.make_model(Xtr.shape[1], torch).to(dev)
    t1 = time.time(); ml.fit(base, Xtr, Ytr, sc, torch, dev, a.epochs, 2e-3, seed=ml.SEED)
    print(f"pre-training in {time.time()-t1:.0f} s")
    Xtt, Ytt = ml.features(tgt_test, True)
    src_only = ml.rmse(ml.predict(base, Xtt, sc, torch, dev), Ytt)
    print(f"source-only on target: {src_only:.3f} x1e-3")

    n_list = [int(v) for v in a.n_target.split(",")]
    curve = {v: {n: [] for n in n_list} for v in VARIANTS}
    lam_chosen = {n: [] for n in n_list}
    for r in range(a.repeats):
        for n in n_list:
            plates = tgt_pool[r * a.pool_max: r * a.pool_max + n]
            rr = np.random.default_rng(ml.SEED + 1000 * r + n)
            idx = [rr.choice(ml.NZ, size=a.points, replace=False) for _ in plates]
            Xs, Ys = ml.features(plates, True, idx)
            Ys = Ys + rr.normal(0, a.noise, Ys.shape).astype(np.float32)
            groups = np.repeat(np.arange(n), a.points)
            seed = ml.SEED + r
            # baseline: exactly the call of the main script
            m = copy.deepcopy(base); ml.fit(m, Xs, Ys, sc, torch, dev, a.ft_epochs, 3e-4, wd=1e-4, seed=seed)
            curve["full-600"][n].append(ml.rmse(ml.predict(m, Xtt, sc, torch, dev), Ytt))
            m = copy.deepcopy(base); ml.fit(m, Xs, Ys, sc, torch, dev, 100, 3e-4, wd=1e-4, seed=seed)
            curve["full-100"][n].append(ml.rmse(ml.predict(m, Xtt, sc, torch, dev), Ytt))
            m = fit_ft(copy.deepcopy(base), Xs, Ys, sc, torch, dev, a.ft_epochs, 3e-4, last_only=True, seed=seed)
            curve["last"][n].append(ml.rmse(ml.predict(m, Xtt, sc, torch, dev), Ytt))
            pred, _ = affine(base, Xs, Ys, sc, torch, dev)
            curve["affine"][n].append(ml.rmse(pred(Xtt), Ytt))
            m = fit_ft(copy.deepcopy(base), Xs, Ys, sc, torch, dev, a.ft_epochs, 3e-4, lam=1e-2, seed=seed)
            curve["l2sp-fixed"][n].append(ml.rmse(ml.predict(m, Xtt, sc, torch, dev), Ytt))
            lam, _ = cv_lambda(base, Xs, Ys, groups, sc, torch, dev, a.ft_epochs, 3e-4, seed)
            lam_chosen[n].append(lam)
            m = fit_ft(copy.deepcopy(base), Xs, Ys, sc, torch, dev, a.ft_epochs, 3e-4, lam=lam, seed=seed)
            curve["l2sp-cv"][n].append(ml.rmse(ml.predict(m, Xtt, sc, torch, dev), Ytt))
        print(f"repeat {r+1}/{a.repeats} done ({time.time()-t0:.0f} s)")

    summ = {v: {n: dict(mean=float(np.mean(x)), sd=float(np.std(x)),
                        worse_than_source_only=int(np.sum(np.array(x) > src_only)))
                for n, x in d.items()} for v, d in curve.items()}
    res = dict(args=vars(a), device=str(dev), source_only=src_only, variants=summ,
               lambda_cv={n: [float(v) for v in l] for n, l in lam_chosen.items()},
               raw={v: {n: [float(t) for t in x] for n, x in d.items()} for v, d in curve.items()})
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(f"\ntarget RMSE [x1e-3], mean +- sd over {a.repeats} repeats; source-only {src_only:.3f}; "
          "(k) = repeats worse than source-only")
    print("variant      " + "".join(f"{'n=' + str(n):>20s}" for n in n_list))
    for v in VARIANTS:
        print(f"{v:12s} " + "".join(f"{summ[v][n]['mean']:>9.3f} ±{summ[v][n]['sd']:.3f} ({summ[v][n]['worse_than_source_only']:2d})"
                                    for n in n_list))
    print("lambda chosen by CV: " + "; ".join(f"n={n}: {sorted(set(l))}" for n, l in lam_chosen.items()))
    _plot(os.path.join(a.out, "denso_ft_variants.png"), summ, n_list, src_only, a.repeats)
    print(f"wrote {a.out}/results.json and denso_ft_variants.png")


def _plot(path, summ, n_list, src_only, repeats):
    sys.path.insert(0, os.path.join(HERE, "slides", "figure_sources"))
    try:                                   # repo convention: thesis_style, graceful fallback
        from thesis_style import use
        use(width_frac=0.6, aspect=0.75)
        ok = shutil.which("latex") is not None
        if ok and matplotlib.rcParams["text.usetex"]:
            try:                           # latex may exist without cm-super (copaam)
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
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    styles = {"full-600": ("k", "o", "-"), "full-100": ("0.45", "s", "--"), "last": ("0.45", "^", ":"),
              "affine": ("0.2", "D", "-."), "l2sp-fixed": ("0.6", "v", "--"), "l2sp-cv": ("k", "P", "--")}
    for v, (c, mk, ls) in styles.items():
        y = [summ[v][n]["mean"] for n in n_list]; e = [summ[v][n]["sd"] for n in n_list]
        ax.errorbar(n_list, y, yerr=e, color=c, marker=mk, ls=ls, ms=4, capsize=2, lw=1, label=v)
    ax.axhline(src_only, color="0.5", lw=0.8, ls=":", label="source-only")
    ax.set_xscale("log"); ax.set_xlabel("measured target plates (5 points each)")
    ax.set_ylabel(r"target RMSE [$10^{-3}$]")
    ax.set_title(f"fine-tune variants, mean $\\pm$ sd over {repeats} repeats (synthetic, illustrative)", fontsize=8)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
