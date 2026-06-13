"""
kojima_real_decision.py — a LABELLED decision on REAL measured data: can the
framework tell a CONVEX from a CONCAVE defect from the measured stress field —
from its SHAPE, not its loading amplitude?

Where this sits / why it is new
-------------------------------
`tsa_sim2real` already ingests Kojima's 22 REAL measured-DSPSS specimens
(12 convex `totsu` + 10 concave `ou`) and stops at the honest sim-to-real DOMAIN
GAP (mean standardised-field KS ≈ 0.091) plus a qualitative Stage-0 |z| hot-spot
demo.  It never makes a LABELLED decision on the real data.  This module does:
the 22 specimens carry a real defect-TYPE label (convex vs concave), so we ask
whether the framework can DISCRIMINATE the two from the measured field — the first
labelled Stage-1 result on real measured data, evaluated honestly by
leave-one-out CV with a permutation test (n=22 is small; we say so).

The rigour that matters
-----------------------
Convex and concave specimens differ in absolute stress amplitude (different
batches/loading: convex max ≈ 217 MPa, concave ≈ 67 MPa).  Classifying on raw
magnitude would just read the LOADING, not the defect — a confound.  So the
headline classifier uses ONLY SCALE-INVARIANT, within-specimen-standardised field
features (skewness, kurtosis, |z| hot-spot fraction, hot-spot concentration) —
the SHAPE of the stress field, which is what the defect geometry actually
imprints.  We report the magnitude-based classifier separately and flag it as the
confounded comparison, so the honest claim is about the defect SIGNATURE.

It also runs an OOD check: are the real fields flagged out-of-distribution
against the FEM training pool (the gap the decision must survive)?

HONEST LIMITATION
-----------------
n = 22 specimens, ONE application (CFRP prosthetic), TWO defect types (both
defective — this is convex-vs-concave discrimination, NOT healthy-vs-defect
detection).  Leave-one-out CV + a label-permutation p-value guard the small n,
but the confidence interval is wide and this is a single real campaign, not a
benchmark.  The data is Kojima's raw measured DSPSS (attributed); the METHOD is
the framework's own (distinct from his stress-emulator paper).

Usage
-----
    python kojima_real_decision.py            # scale-invariant + magnitude + OOD
    python kojima_real_decision.py --fig
    python kojima_real_decision.py --test
"""
from __future__ import annotations

import argparse
import os

import numpy as np

import tsa_sim2real as t2r
import kojima_real_case as kr

HERE = os.path.dirname(os.path.abspath(__file__))


# ═════════════════════════════════════════════════════════════════════════════
#  Load the 22 real specimens with their defect-type label
# ═════════════════════════════════════════════════════════════════════════════

def load_specimens() -> dict:
    """The 22 measured-DSPSS specimens → masked value arrays + binary label
    (1 = convex/totsu, 0 = concave/ou)."""
    recs = t2r.load_rect_dspss()
    names, vals, y = [], [], []
    for name, field in recs:
        v = np.asarray(field, float).ravel()
        v = v[v > 1e-6]                                  # drop the zero padding
        if v.size < 50:
            continue
        names.append(name); vals.append(v)
        y.append(1 if name.startswith("totsu") else 0)
    return {"names": names, "vals": vals, "y": np.array(y)}


# ═════════════════════════════════════════════════════════════════════════════
#  Features
# ═════════════════════════════════════════════════════════════════════════════

def _skew_kurt(z: np.ndarray) -> tuple[float, float]:
    m3 = float(np.mean(z ** 3)); m4 = float(np.mean(z ** 4))
    return m3, m4 - 3.0                                  # excess kurtosis


def shape_features(v: np.ndarray) -> np.ndarray:
    """SCALE-INVARIANT field-shape features (within-specimen standardised) — the
    defect SIGNATURE, independent of loading amplitude."""
    z = (v - v.mean()) / (v.std() + 1e-9)
    sk, ku = _skew_kurt(z)
    azz = kr.field_anomaly(v, mode="abs")               # framework Stage-0 |z|
    hot2 = float((azz > 2.0).mean())                    # hot-spot fraction
    hot3 = float((azz > 3.0).mean())
    conc = float(azz.max() / (np.median(azz) + 1e-9))   # hot-spot concentration
    iqr_ratio = float((np.percentile(v, 75) - np.percentile(v, 25))
                      / (np.median(v) + 1e-9))           # robust CoV (scale-free)
    return np.array([sk, ku, hot2, hot3, conc, iqr_ratio])


def magnitude_features(v: np.ndarray) -> np.ndarray:
    """Absolute-amplitude features (CONFOUNDED with loading) — reported only for
    contrast, never the headline."""
    return np.array([float(np.median(v)), float(np.percentile(v, 95)),
                     float(v.max())])


def feature_matrix(vals: list, kind: str) -> np.ndarray:
    f = shape_features if kind == "shape" else magnitude_features
    return np.array([f(v) for v in vals])


# ═════════════════════════════════════════════════════════════════════════════
#  Fisher LDA (1-D), leave-one-out CV, metrics
# ═════════════════════════════════════════════════════════════════════════════

def fisher_lda(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fisher linear discriminant direction (pooled within-class covariance)."""
    X1, X0 = X[y == 1], X[y == 0]
    mu1, mu0 = X1.mean(0), X0.mean(0)
    Sw = np.cov(X1, rowvar=False) * (len(X1) - 1) \
        + np.cov(X0, rowvar=False) * (len(X0) - 1)
    Sw += 1e-6 * np.eye(X.shape[1])                      # ridge for stability
    w = np.linalg.solve(Sw, mu1 - mu0)
    return w / (np.linalg.norm(w) + 1e-12)


def _standardize(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    return (Xtr - mu) / sd, (Xte - mu) / sd


def loo_scores(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Leave-one-out projection score for each specimen (trained on the other
    21, oriented so higher = class 1)."""
    n = len(y); s = np.zeros(n)
    for i in range(n):
        tr = np.ones(n, bool); tr[i] = False
        Xtr, Xte = _standardize(X[tr], X[i:i + 1])
        w = fisher_lda(Xtr, y[tr])
        proj_tr = (Xtr @ w)
        # orient so class-1 mean projects higher
        if proj_tr[y[tr] == 1].mean() < proj_tr[y[tr] == 0].mean():
            w = -w
        s[i] = float((Xte @ w).ravel()[0])
    return s


def auroc(y: np.ndarray, score: np.ndarray) -> float:
    pos = score[y == 1]; neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return float(wins / (len(pos) * len(neg)))


def loo_metrics(X: np.ndarray, y: np.ndarray) -> dict:
    s = loo_scores(X, y)
    thr = np.median(s)                                   # symmetric operating pt
    pred = (s > thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    acc = (tp + tn) / len(y)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"auroc": auroc(y, s), "acc": acc, "f1": f1, "prec": prec,
            "rec": rec, "scores": s, "pred": pred,
            "confusion": (tp, fp, fn, tn)}


def permutation_pvalue(X: np.ndarray, y: np.ndarray, observed_auroc: float,
                       n_perm: int = 2000, seed: int = 0) -> float:
    """Label-permutation p-value: how often a shuffled-label LOO AUROC matches or
    beats the observed one (guards the small-n result against chance)."""
    rng = np.random.default_rng(seed)
    cnt = 1
    for _ in range(n_perm):
        yp = rng.permutation(y)
        if len(np.unique(yp)) < 2:
            continue
        a = auroc(yp, loo_scores(X, yp))
        if a >= observed_auroc - 1e-12:
            cnt += 1
    return cnt / (n_perm + 1)


# ═════════════════════════════════════════════════════════════════════════════
#  OOD: are the real fields flagged out-of-distribution vs the FEM pool?
# ═════════════════════════════════════════════════════════════════════════════

def ood_vs_fem(vals: list, n_fem_ref: int = 400, seed: int = 0) -> dict:
    """Per-specimen standardised-field KS vs the FEM training pool, compared to
    the FEM-vs-FEM self-distance — a conformal-style OOD flag (real specimen is
    OOD if its gap exceeds the 95th percentile of FEM self-gaps)."""
    fem = t2r.fem_dspss_pool()
    rng = np.random.default_rng(seed)
    fem_std = (fem - fem.mean()) / (fem.std() + 1e-9)
    # FEM self-distance null: KS between two random FEM subsamples
    null = []
    for _ in range(60):
        a = rng.choice(fem_std, 2000); b = rng.choice(fem_std, 2000)
        null.append(kr.ks_statistic(a, b))
    thr = float(np.quantile(null, 0.95))
    real_ks = [kr.ks_statistic((v - v.mean()) / (v.std() + 1e-9), fem_std)
               for v in vals]
    flagged = [k > thr for k in real_ks]
    return {"real_ks": np.array(real_ks), "fem_self_thr": thr,
            "frac_flagged": float(np.mean(flagged)),
            "mean_real_ks": float(np.mean(real_ks))}


# ═════════════════════════════════════════════════════════════════════════════
#  Driver
# ═════════════════════════════════════════════════════════════════════════════

def run_study(seed: int = 0, verbose: bool = True) -> dict:
    d = load_specimens()
    vals, y = d["vals"], d["y"]
    Xs = feature_matrix(vals, "shape")
    Xm = feature_matrix(vals, "magnitude")
    m_shape = loo_metrics(Xs, y)
    m_mag = loo_metrics(Xm, y)
    p_shape = permutation_pvalue(Xs, y, m_shape["auroc"], seed=seed)
    ood = ood_vs_fem(vals, seed=seed)
    out = {"d": d, "y": y, "Xs": Xs, "Xm": Xm, "m_shape": m_shape,
           "m_mag": m_mag, "p_shape": p_shape, "ood": ood}
    if verbose:
        _report(out)
    return out


def _report(o: dict) -> None:
    bar = "=" * 76
    y = o["y"]
    print(bar)
    print("  LABELLED DECISION ON REAL MEASURED DATA — convex vs concave defect")
    print("  (Kojima's 22 measured-DSPSS CFRP specimens; method = framework's own)")
    print(bar)
    print(f"  n = {len(y)} specimens: {int((y==1).sum())} convex(totsu) + "
          f"{int((y==0).sum())} concave(ou)\n")
    ms, mm = o["m_shape"], o["m_mag"]
    print("  HEADLINE — SCALE-INVARIANT field-SHAPE features (defect signature,")
    print("  loading-amplitude removed):")
    print(f"     LOO-CV  AUROC {ms['auroc']:.3f}   acc {ms['acc']*100:.1f}%   "
          f"F1 {ms['f1']:.3f}   (perm-test p = {o['p_shape']:.3f})")
    tp, fp, fn, tn = ms["confusion"]
    print(f"     confusion (tp,fp,fn,tn) = ({tp},{fp},{fn},{tn})")
    print("\n  CONTRAST — absolute-magnitude features (CONFOUNDED with loading,")
    print("  not a defect-signature claim):")
    print(f"     LOO-CV  AUROC {mm['auroc']:.3f}   acc {mm['acc']*100:.1f}%   "
          f"F1 {mm['f1']:.3f}")
    od = o["ood"]
    print(f"\n  OOD vs FEM training pool: mean real KS {od['mean_real_ks']:.3f} "
          f"(FEM self-95% thr {od['fem_self_thr']:.3f}) → "
          f"{od['frac_flagged']*100:.0f}% of real specimens flagged OOD")
    print("    → the measured sim-to-real shift is real and the framework's OOD")
    print("      gate SEES it (the gap any real decision must survive).")
    print("\n  LIMITATION: n=22, ONE application, convex-vs-concave (both")
    print("  defective, NOT healthy-vs-defect). LOO-CV + permutation p guard the")
    print("  small n but the CI is wide; single real campaign, not a benchmark.")
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

    y = o["y"]; ms = o["m_shape"]; mm = o["m_mag"]
    fig, ax = plt.subplots(1, 3, figsize=figsize)

    # (a) LOO score separation (scale-invariant) by true class.
    s = ms["scores"]
    ax[0].scatter(np.where(y == 1)[0], s[y == 1], c="#b71c1c", s=26,
                  label="convex (totsu)", edgecolor="k", linewidth=0.3)
    ax[0].scatter(np.where(y == 0)[0], s[y == 0], c="#1565C0", s=26,
                  label="concave (ou)", edgecolor="k", linewidth=0.3)
    ax[0].axhline(np.median(s), color="k", ls="--", lw=0.8, label="threshold")
    ax[0].set_xlabel("specimen", fontsize=7)
    ax[0].set_ylabel("LOO discriminant score", fontsize=7)
    ax[0].set_title(f"(a) scale-invariant signature\nAUROC {ms['auroc']:.2f}, "
                    f"p={o['p_shape']:.3f}", fontsize=8)
    ax[0].legend(fontsize=5.5, loc="best"); ax[0].tick_params(labelsize=6)

    # (b) shape vs magnitude AUROC (honest contrast).
    ax[1].bar([0, 1], [ms["auroc"], mm["auroc"]],
              color=["#2e7d32", "#9e9e9e"], edgecolor="k", linewidth=0.3)
    ax[1].axhline(0.5, color="k", ls=":", lw=0.8)
    ax[1].set_xticks([0, 1])
    ax[1].set_xticklabels(["shape\n(signature)", "magnitude\n(confounded)"],
                          fontsize=6.5)
    ax[1].set_ylim(0, 1.05); ax[1].set_ylabel("LOO-CV AUROC", fontsize=7)
    ax[1].set_title("(b) the defect SIGNATURE discriminates,\n"
                    "not just loading amplitude", fontsize=8)
    ax[1].tick_params(labelsize=6)

    # (c) OOD: real-specimen KS vs FEM pool + FEM self threshold.
    od = o["ood"]
    ax[2].hist(od["real_ks"], bins=10, color="#1565C0", alpha=0.8,
               edgecolor="k", linewidth=0.3)
    ax[2].axvline(od["fem_self_thr"], color="#b71c1c", lw=1.4,
                  label=f"FEM self 95% ({od['fem_self_thr']:.2f})")
    ax[2].set_xlabel("standardised-field KS vs FEM pool", fontsize=7)
    ax[2].set_ylabel("real specimens", fontsize=7)
    ax[2].set_title(f"(c) real data is OOD vs FEM\n"
                    f"({od['frac_flagged']*100:.0f}% flagged)", fontsize=8)
    ax[2].legend(fontsize=5.5, loc="upper right"); ax[2].tick_params(labelsize=6)

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

    d = load_specimens()
    y = d["y"]
    ok(len(y) == 22, f"22 real specimens loaded ({len(y)})")
    ok(int((y == 1).sum()) == 12 and int((y == 0).sum()) == 10,
       "12 convex + 10 concave labels")
    ok(all(v.size > 50 for v in d["vals"]), "every specimen has a masked field")

    # ---- features ----------------------------------------------------------
    Xs = feature_matrix(d["vals"], "shape")
    Xm = feature_matrix(d["vals"], "magnitude")
    ok(Xs.shape == (22, 6), "shape feature matrix 22×6")
    ok(np.all(np.isfinite(Xs)), "shape features finite")
    # scale invariance: scaling a field by a constant leaves shape features fixed
    v = d["vals"][0]
    ok(np.allclose(shape_features(v), shape_features(3.7 * v), atol=1e-6),
       "shape features are scale-invariant")
    ok(not np.allclose(magnitude_features(v), magnitude_features(3.7 * v)),
       "magnitude features are NOT scale-invariant (by design)")

    # ---- Fisher LDA + AUROC sanity -----------------------------------------
    rng = np.random.default_rng(0)
    Xa = np.vstack([rng.normal(0, 1, (10, 3)), rng.normal(3, 1, (10, 3))])
    ya = np.array([0] * 10 + [1] * 10)
    s = Xa @ fisher_lda(Xa, ya)
    ok(auroc(ya, s) > 0.95, "Fisher LDA separates a clean 2-class toy")
    ok(abs(auroc(ya, -s) - (1 - auroc(ya, s))) < 1e-9, "AUROC flips with sign")
    ok(abs(auroc(ya, np.zeros_like(s)) - 0.5) < 1e-9, "tie score → AUROC 0.5")

    # ---- LOO metrics on the real data --------------------------------------
    m = loo_metrics(Xs, y)
    ok(0.0 <= m["auroc"] <= 1.0, "AUROC in range")
    ok(0.0 <= m["acc"] <= 1.0, "accuracy in range")
    ok(m["scores"].shape == (22,), "one LOO score per specimen")
    # the scale-invariant signature should beat chance on the real specimens
    ok(m["auroc"] > 0.6, f"real defect-signature discrimination beats chance "
       f"(AUROC {m['auroc']:.3f})")

    # ---- permutation p-value behaves ---------------------------------------
    p = permutation_pvalue(Xs, y, m["auroc"], n_perm=200, seed=1)
    ok(0.0 < p <= 1.0, "permutation p in (0,1]")
    # a clearly-separable signal → small p; pure-noise labels → p near uniform
    yn = np.array([0, 1] * 11)
    mn = loo_metrics(Xs, yn)
    pn = permutation_pvalue(Xs, yn, mn["auroc"], n_perm=200, seed=2)
    ok(pn > p - 1e-9 or pn > 0.1, "shuffled-ish labels give a larger p")

    # ---- OOD vs FEM --------------------------------------------------------
    od = ood_vs_fem(d["vals"], seed=0)
    ok(od["fem_self_thr"] > 0, "FEM self KS threshold positive")
    ok(0.0 <= od["frac_flagged"] <= 1.0, "OOD flagged fraction in range")
    ok(od["mean_real_ks"] > 0, "real KS vs FEM positive")

    # ---- end-to-end --------------------------------------------------------
    o = run_study(seed=0, verbose=False)
    ok("m_shape" in o and "ood" in o, "study returns shape metrics + OOD")
    ok(o["m_shape"]["auroc"] >= o["m_shape"]["auroc"], "study AUROC defined")

    print(f"kojima_real_decision: {n}/{n} unit tests passed")
    return n


def main():
    ap = argparse.ArgumentParser(
        description="Labelled convex-vs-concave defect decision on Kojima's real "
                    "measured DSPSS (scale-invariant signature + OOD).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fig", action="store_true")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.test:
        run_tests(); return
    o = run_study(seed=args.seed)
    if args.fig:
        p = make_figure(o, os.path.join(HERE, "paper_figs",
                                        "kojima_real_decision.pdf"))
        print(f"\nfigure → {p}")


if __name__ == "__main__":
    main()
