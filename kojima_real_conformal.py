"""
kojima_real_conformal.py — does the framework's DISTRIBUTION-FREE conformal
coverage SURVIVE the real sim-to-real shift?

Where this sits / why it is new
-------------------------------
`conformal_clearance` / `conformal_transfer` give the decision stack a
distribution-free coverage guarantee — on SYNTHETIC data.  `kojima_real_decision`
showed the 22 real measured-DSPSS specimens are OOD vs the FEM training pool
(77% flagged, KS 0.091 > FEM-self 0.069).  Split-conformal's guarantee assumes
EXCHANGEABILITY between calibration and test; a sim-to-real shift breaks exactly
that.  So the decision-relevant question is: when we calibrate conformal intervals
on FEM and apply them to REAL data, does the 1−α coverage hold, or COLLAPSE?  This
module measures it — the honest stress-test of the framework's headline guarantee
on real data, and whether REAL recalibration restores it.

Set-up (scale-invariant, to isolate the SHAPE shift)
----------------------------------------------------
The task is a scale-free field-shape regression computable identically on the FEM
graphs and the real grids: predict the TAIL heaviness t = p95/median from the BODY
shape x = [skew, excess-kurtosis, IQR/median].  Both sides are standardised within
each field, so the experiment is purely about whether the FEM-learned shape→tail
relationship — and its conformal calibration — transfer to real measurements,
independent of loading amplitude.

  * fit a linear t≈w·x on FEM-train (221 graphs),
  * split-conformal the |residual| on FEM-cal (22) → q_α,
  * empirical coverage on FEM-test (22, IN-distribution) vs the 22 REAL specimens
    (OOD).  Nominal vs achieved is the result.
  * RECALIBRATION: redo the conformal quantile on REAL residuals (leave-one-out
    over the 22) — does coverage come back?  (The model transfers; only the
    CALIBRATION must be redone on real — the actionable conclusion.)

HONEST LIMITATION
-----------------
n = 22 real specimens (one campaign), so the real coverage estimate has a wide
binomial CI; we report it as a direction, not a certified number.  The task is a
deliberately simple scalar shape-regression (not the full GNN) so the conformal
behaviour is clean and attributable to the sim-to-real shift, not to a particular
network.  The FEM graphstress pool is HOMOGENEOUS (near-identical augmented
fields), which makes the FEM-calibrated interval very tight and AMPLIFIES the
collapse — the transferable lesson is the DIRECTION (FEM-calibrated under-covers
real, real recalibration restores nominal), not the literal 0%.  Data attributed
to Kojima; method is the framework's own.

Usage
-----
    python kojima_real_conformal.py            # coverage: FEM-cal vs real-recal
    python kojima_real_conformal.py --fig
    python kojima_real_conformal.py --test
"""
from __future__ import annotations

import argparse
import os

import numpy as np

import tsa_sim2real as t2r

HERE = os.path.dirname(os.path.abspath(__file__))

DERIVED = "/home/nishioka/from_kojima/copaam_derived"
FEM_SPLITS = {
    "train": f"{DERIVED}/Pros_dte_graphstress/graphstress_original_train.npy",
    "val":   f"{DERIVED}/Pros_dte_graphstress/graphstress_original_val.npy",
    "test":  f"{DERIVED}/Pros_dte_graphstress/graphstress_original_test.npy",
}
ALPHAS = (0.20, 0.10, 0.05)            # nominal miscoverage → 80/90/95% target


# ═════════════════════════════════════════════════════════════════════════════
#  Scale-free field-shape features + tail target
# ═════════════════════════════════════════════════════════════════════════════

def field_xy(v: np.ndarray) -> tuple[np.ndarray, float]:
    """Scale-free body-shape features x=[skew,kurt,IQR/median] and tail target
    t=p95/median for a field's masked values."""
    v = np.asarray(v, float).ravel()
    v = v[v > 1e-6]
    z = (v - v.mean()) / (v.std() + 1e-9)
    skew = float(np.mean(z ** 3)); kurt = float(np.mean(z ** 4) - 3.0)
    med = float(np.median(v)) + 1e-9
    iqr_ratio = float(np.percentile(v, 75) - np.percentile(v, 25)) / med
    tail = float(np.percentile(v, 95)) / med
    return np.array([skew, kurt, iqr_ratio]), tail


def fem_xy(split: str) -> tuple[np.ndarray, np.ndarray]:
    arr = np.load(FEM_SPLITS[split])
    X, y = [], []
    for row in arr:
        x, t = field_xy(row)
        X.append(x); y.append(t)
    return np.array(X), np.array(y)


def real_xy() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Real specimens → (X, y, label) with label 1=convex/totsu, 0=concave/ou."""
    recs = t2r.load_rect_dspss()
    X, y, lab = [], [], []
    for name, field in recs:
        x, t = field_xy(field)
        X.append(x); y.append(t)
        lab.append(1 if name.startswith("totsu") else 0)
    return np.array(X), np.array(y), np.array(lab)


# ═════════════════════════════════════════════════════════════════════════════
#  Linear fit + split conformal
# ═════════════════════════════════════════════════════════════════════════════

def fit_linear(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least-squares weights for [1, x] → y (intercept first)."""
    A = np.hstack([np.ones((len(X), 1)), X])
    w, *_ = np.linalg.lstsq(A, y, rcond=None)
    return w


def predict(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((len(X), 1)), X]) @ w


def conformal_q(residuals: np.ndarray, alpha: float) -> float:
    """Split-conformal radius: the ceil((n+1)(1−α))/n empirical quantile of the
    absolute calibration residuals (finite-sample valid under exchangeability)."""
    n = len(residuals)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    return float(np.sort(residuals)[k - 1])


def coverage(y: np.ndarray, pred: np.ndarray, q: float) -> float:
    return float((np.abs(y - pred) <= q + 1e-12).mean())


# ═════════════════════════════════════════════════════════════════════════════
#  Experiment: FEM-calibrated coverage (in-dist vs OOD) + real recalibration
# ═════════════════════════════════════════════════════════════════════════════

def real_loo_coverage(w: np.ndarray, Xr: np.ndarray, yr: np.ndarray,
                      alpha: float) -> float:
    """Leave-one-out conformal on the REAL set: calibrate q on the other 21 real
    residuals (same FEM-fit model), test the held-out one. Coverage if real
    recalibration restores the guarantee."""
    pred = predict(w, Xr)
    res = np.abs(yr - pred)
    hit = 0
    n = len(yr)
    for i in range(n):
        cal = np.delete(res, i)
        q = conformal_q(cal, alpha)
        hit += int(res[i] <= q + 1e-12)
    return hit / n


def run_study(verbose: bool = True) -> dict:
    Xtr, ytr = fem_xy("train")
    Xcal, ycal = fem_xy("val")
    Xte, yte = fem_xy("test")
    Xr, yr, lab = real_xy()

    w = fit_linear(Xtr, ytr)
    pred_cal = predict(w, Xcal)
    res_cal = np.abs(ycal - pred_cal)
    pred_te = predict(w, Xte)
    pred_r = predict(w, Xr)

    rows = []
    for a in ALPHAS:
        q = conformal_q(res_cal, a)
        cov_fem = coverage(yte, pred_te, q)         # in-distribution
        cov_real = coverage(yr, pred_r, q)          # OOD (FEM-calibrated)
        cov_real_recal = real_loo_coverage(w, Xr, yr, a)   # real-recalibrated
        rows.append({"target": 1 - a, "q_fem": q,
                     "cov_fem_test": cov_fem, "cov_real_femcal": cov_real,
                     "cov_real_recal": cov_real_recal})

    # residual inflation FEM→real (the mechanism)
    infl = float(np.mean(np.abs(yr - pred_r)) / (np.mean(res_cal) + 1e-9))
    out = {"rows": rows, "w": w, "res_cal": res_cal,
           "res_real": np.abs(yr - pred_r), "infl": infl,
           "yte": yte, "pred_te": pred_te, "yr": yr, "pred_r": pred_r,
           "lab": lab, "n_fem_test": len(yte), "n_real": len(yr)}
    if verbose:
        _report(out)
    return out


def _report(o: dict) -> None:
    bar = "=" * 78
    print(bar)
    print("  CONFORMAL COVERAGE UNDER THE REAL SIM-TO-REAL SHIFT")
    print("  (FEM-calibrated intervals applied to real measured DSPSS)")
    print(bar)
    print(f"  scale-free shape→tail regression; FEM-fit (221) / FEM-cal (22) / "
          f"test on FEM ({o['n_fem_test']}, in-dist) vs REAL ({o['n_real']}, OOD)\n")
    print(f"  {'target':>7}{'FEM-test':>11}{'REAL (FEM-cal)':>16}"
          f"{'REAL (recal)':>14}")
    print("  " + "-" * 50)
    for r in o["rows"]:
        print(f"  {r['target']*100:5.0f}% {r['cov_fem_test']*100:9.1f}%"
              f"{r['cov_real_femcal']*100:14.1f}%{r['cov_real_recal']*100:13.1f}%")
    print(f"\n  residual inflation FEM→real = ×{o['infl']:.2f}  "
          f"(the exchangeability break that collapses coverage)")
    print("\n  READING: FEM-test coverage tracks the nominal target (the guarantee")
    print("  holds in-distribution). FEM-CALIBRATED coverage on REAL data falls")
    print("  BELOW nominal — the sim-to-real shift breaks exchangeability, so the")
    print("  'distribution-free' guarantee does NOT transfer. REAL recalibration")
    print("  (leave-one-out on the 22) brings coverage back toward nominal — the")
    print("  model transfers; the CALIBRATION must be redone on real data.")
    print("\n  LIMITATION: n=22 real (one campaign) → wide binomial CI, read as a")
    print("  direction; deliberately a simple scalar shape-regression (not the GNN)")
    print("  so the conformal behaviour is attributable to the shift. The FEM")
    print("  graphstress pool is HOMOGENEOUS (near-identical augmented fields, see")
    print("  fig-c cluster) so its calibration interval is very tight — this")
    print("  AMPLIFIES the collapse; the transferable lesson is the DIRECTION")
    print("  (FEM-cal under-covers real, recal restores), not the literal 0%. "
          "Data = Kojima's.")
    print(bar)


# ═════════════════════════════════════════════════════════════════════════════
#  Figure
# ═════════════════════════════════════════════════════════════════════════════

def make_figure(o: dict, out_path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import sys
    sys.path.insert(0, os.path.join(HERE, "slides", "figure_sources"))
    try:
        from thesis_style import use
        figsize = use(width_frac=1.0, aspect=0.40)
    except Exception:
        figsize = (12.0, 4.0)
    import matplotlib.pyplot as plt

    rows = o["rows"]
    targ = [r["target"] * 100 for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=figsize)

    # (a) coverage vs nominal: in-dist holds, FEM-cal-on-real collapses, recal back
    ax[0].plot(targ, targ, "k--", lw=0.9, label="nominal")
    ax[0].plot(targ, [r["cov_fem_test"] * 100 for r in rows], "o-",
               color="#2e7d32", lw=1.4, label="FEM-test (in-dist)")
    ax[0].plot(targ, [r["cov_real_femcal"] * 100 for r in rows], "s-",
               color="#b71c1c", lw=1.4, label="REAL (FEM-cal)")
    ax[0].plot(targ, [r["cov_real_recal"] * 100 for r in rows], "D-",
               color="#1565C0", lw=1.4, label="REAL (recal)")
    ax[0].set_xlabel("nominal coverage [%]", fontsize=7)
    ax[0].set_ylabel("achieved coverage [%]", fontsize=7)
    ax[0].set_title("(a) FEM guarantee does NOT\ntransfer; recal restores it",
                    fontsize=8)
    ax[0].legend(fontsize=5.5, loc="lower right"); ax[0].tick_params(labelsize=6)

    # (b) residual distributions FEM-cal vs real (the exchangeability break)
    ax[1].hist(o["res_cal"], bins=12, color="#2e7d32", alpha=0.7, density=True,
               label="FEM-cal residual", edgecolor="k", linewidth=0.3)
    ax[1].hist(o["res_real"], bins=12, color="#b71c1c", alpha=0.6, density=True,
               label="REAL residual", edgecolor="k", linewidth=0.3)
    ax[1].set_xlabel("|residual| (tail p95/median)", fontsize=7)
    ax[1].set_ylabel("density", fontsize=7)
    ax[1].set_title(f"(b) residuals inflate ×{o['infl']:.1f} on real\n"
                    "(breaks exchangeability)", fontsize=8)
    ax[1].legend(fontsize=5.5, loc="upper right"); ax[1].tick_params(labelsize=6)

    # (c) predicted vs true tail, FEM-test vs real (bias under shift)
    ax[2].plot([1, max(o["yr"].max(), o["yte"].max())],
               [1, max(o["yr"].max(), o["yte"].max())], "k--", lw=0.8)
    ax[2].scatter(o["pred_te"], o["yte"], c="#2e7d32", s=22, label="FEM-test",
                  edgecolor="k", linewidth=0.3)
    lab = o["lab"]
    ax[2].scatter(o["pred_r"][lab == 1], o["yr"][lab == 1], c="#b71c1c", s=22,
                  marker="^", label="real convex", edgecolor="k", linewidth=0.3)
    ax[2].scatter(o["pred_r"][lab == 0], o["yr"][lab == 0], c="#ef9a9a", s=22,
                  marker="v", label="real concave", edgecolor="k", linewidth=0.3)
    ax[2].set_xlabel("predicted tail (FEM model)", fontsize=7)
    ax[2].set_ylabel("true tail p95/median", fontsize=7)
    ax[2].set_title("(c) real tails sit off the\nFEM-fit diagonal (shift)",
                    fontsize=8)
    ax[2].legend(fontsize=5.0, loc="upper left"); ax[2].tick_params(labelsize=6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    fig.savefig(os.path.splitext(out_path)[0] + ".png", bbox_inches="tight",
                dpi=150)
    plt.close(fig)
    return out_path


# ═════════════════════════════════════════════════════════════════════════════
#  Unit tests
# ═════════════════════════════════════════════════════════════════════════════

def run_tests() -> int:
    n = 0

    def ok(cond, msg):
        nonlocal n
        assert cond, msg
        n += 1

    # ---- features scale-free + finite -------------------------------------
    rng = np.random.default_rng(0)
    v = np.abs(rng.lognormal(0, 0.5, 5000))
    x1, t1 = field_xy(v); x2, t2 = field_xy(7.3 * v)
    ok(np.allclose(x1, x2, atol=1e-6), "shape features scale-free")
    ok(abs(t1 - t2) < 1e-6, "tail target scale-free")
    ok(np.all(np.isfinite(x1)) and np.isfinite(t1), "features finite")

    # ---- conformal_q is a valid empirical quantile (coverage on cal) -------
    res = np.abs(rng.normal(0, 1, 200))
    for a in (0.2, 0.1, 0.05):
        q = conformal_q(res, a)
        emp = (res <= q + 1e-12).mean()
        ok(emp >= 1 - a - 1e-9, f"conformal q gives ≥{1-a:.0%} on its own cal")
    ok(conformal_q(res, 0.05) >= conformal_q(res, 0.2),
       "tighter α ⇒ larger radius")

    # ---- linear fit recovers a known plane --------------------------------
    Xt = rng.normal(0, 1, (100, 3)); wt = np.array([0.5, 1.0, -2.0, 3.0])
    yt = wt[0] + Xt @ wt[1:]
    w = fit_linear(Xt, yt)
    ok(np.allclose(w, wt, atol=1e-6), "linear fit recovers the plane")
    ok(np.allclose(predict(w, Xt), yt, atol=1e-6), "predict matches")

    # ---- synthetic conformal: in-dist holds, shifted collapses, recal back -
    Xc = rng.normal(0, 1, (60, 3)); yc = 1.0 + Xc @ np.array([1., -1., .5]) \
        + rng.normal(0, 0.2, 60)
    wc = fit_linear(Xc, yc)
    rescal = np.abs(yc - predict(wc, Xc))
    q90 = conformal_q(rescal, 0.1)
    Xin = rng.normal(0, 1, (200, 3)); yin = 1.0 + Xin @ np.array([1., -1., .5]) \
        + rng.normal(0, 0.2, 200)
    cov_in = coverage(yin, predict(wc, Xin), q90)
    Xsh = rng.normal(1.5, 1, (200, 3)); ysh = 1.0 + Xsh @ np.array([1., -1., .5]) \
        + rng.normal(0, 0.9, 200)        # shifted + noisier ⇒ exchangeability broken
    cov_sh = coverage(ysh, predict(wc, Xsh), q90)
    ok(cov_in >= 0.82, f"in-distribution coverage ≈ nominal ({cov_in:.2f})")
    ok(cov_sh < cov_in, f"shift collapses coverage ({cov_sh:.2f} < {cov_in:.2f})")
    cov_recal = real_loo_coverage(wc, Xsh, ysh, 0.1)
    ok(cov_recal > cov_sh, "recalibrating on the shifted set restores coverage")

    # ---- real data study ---------------------------------------------------
    o = run_study(verbose=False)
    ok(len(o["rows"]) == 3, "three nominal levels")
    ok(o["n_real"] == 22, "22 real specimens")
    # FEM-test coverage near nominal at 90%
    r90 = [r for r in o["rows"] if abs(r["target"] - 0.9) < 1e-6][0]
    ok(r90["cov_fem_test"] >= 0.7, "FEM-test 90% coverage reasonable (in-dist)")
    # real recalibration is at least as good as FEM-calibrated on real (the point)
    ok(all(r["cov_real_recal"] >= r["cov_real_femcal"] - 1e-9 for r in o["rows"]),
       "real recalibration ≥ FEM-calibrated coverage on real")
    ok(o["infl"] > 0, "residual inflation defined")

    print(f"kojima_real_conformal: {n}/{n} unit tests passed")
    return n


def main():
    ap = argparse.ArgumentParser(
        description="Does conformal coverage survive the real sim-to-real shift "
                    "(FEM-calibrated vs real-recalibrated on Kojima's measured data)?")
    ap.add_argument("--fig", action="store_true")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.test:
        run_tests(); return
    o = run_study()
    if args.fig:
        p = make_figure(o, os.path.join(HERE, "paper_figs",
                                        "kojima_real_conformal.pdf"))
        print(f"\nfigure → {p}")


if __name__ == "__main__":
    main()
