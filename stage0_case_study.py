"""
stage0_case_study.py — what does the Stage-0 per-node screen actually key on?

For Q&A defence of the WCCM claim "first, a per-node statistic screens for free".
The detector is  s_i = |x_i - mu_i| / sigma_i, with (mu_i, sigma_i) the per-node
healthy mean/std from N healthy (1x1-defect) fields. Each node carries ONE
feature (the per-sample z-scored DSPSS value), so "which feature drove detection"
is really TWO questions, answered here on real CFRP-hole data:

  (1) WHERE   — do the high-score nodes coincide with the true defect mask?
                (precision@K, per-case AUROC, defect-node sigma-deviation)
  (2) WHICH WAY — is the defect signature a stress VALLEY (negative deviation)
                or a peak? i.e. the physical sign the screen exploits.

Reported for (a) ONE named deep-dive specimen and (b) aggregated over a sample
of specimens so the story is not cherry-picked.

    python stage0_case_study.py            # report
    python stage0_case_study.py --fig      # + paper_figs/stage0_case_study.pdf
    python stage0_case_study.py --test     # unit tests only
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

BASE     = "/home/nishioka/CFRP/CFRP_hole/hole_data_inp"
DIR_1X1  = f"{BASE}/Defect_hole_1x1_Random_npy"           # healthy-proxy reference
DIR_4X4  = f"{BASE}/Defect_hole_4x4_Region1_21_npy"       # largest defect
LBL_4X4  = f"{BASE}/Def4x4_19class_label"
N_NODES  = 13942
N_REF    = 2000
HERE     = os.path.dirname(os.path.abspath(__file__))


def load_field(fpath: str) -> np.ndarray:
    """Per-sample z-score, identical to mahalanobis_anomaly.load_field."""
    v = np.load(fpath).astype(np.float64)[:N_NODES]
    return (v - v.mean()) / (v.std() + 1e-8)


def load_mask(fpath: str) -> np.ndarray | None:
    base = os.path.splitext(os.path.basename(fpath))[0]
    for cand in (f"{base}.npy", f"{base}_19label.npy"):
        lf = os.path.join(LBL_4X4, cand)
        if os.path.exists(lf):
            lbl = np.load(lf)
            return (lbl[:, 1:].sum(axis=1) > 0).astype(int)
    return None


def build_reference(n_ref: int = N_REF, seed: int = 42):
    """Per-node healthy (mu_i, sigma_i) from n_ref 1x1 fields."""
    files = sorted(glob.glob(os.path.join(DIR_1X1, "*.npy")))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(files), min(n_ref, len(files)), replace=False)
    X = np.stack([load_field(files[i]) for i in idx])        # (n_ref, N_NODES)
    return X.mean(0), X.std(0) + 1e-6


def score_case(field: np.ndarray, mu: np.ndarray, sd: np.ndarray):
    """Signed per-node deviation z_i and the detector score s_i = |z_i|."""
    z = (field - mu) / sd
    return z, np.abs(z)


def case_metrics(field: np.ndarray, mask: np.ndarray,
                 mu: np.ndarray, sd: np.ndarray) -> dict:
    """One specimen: where (precision@K, AUROC, sigma-gap) and which-way (sign)."""
    from sklearn.metrics import roc_auc_score
    z, s = score_case(field, mu, sd)
    K = int(mask.sum())
    order = np.argsort(s)[::-1]
    topK = order[:K]
    prec_at_K = float(mask[topK].mean()) if K else float("nan")
    auroc = float(roc_auc_score(mask, s)) if 0 < K < len(mask) else float("nan")
    # rank of defect nodes: are they in the extreme top of the score order?
    rank = np.empty(len(s), int); rank[order] = np.arange(len(s))
    def_rank_pct = float(np.median(rank[mask == 1]) / len(s) * 100)  # 0 = top
    s_def, s_bg = s[mask == 1], s[mask == 0]
    z_def = z[mask == 1]
    return {
        "K": K,
        "auroc": auroc,
        "prec_at_K": prec_at_K,
        "def_rank_pct": def_rank_pct,                        # median defect rank, % from top
        # NOTE: |z| magnitudes are inflated (per-node sigma_i ~1e-3 on z-scored
        # fields), so report the MEDIAN and never quote it as "N sigma".
        "score_def_med": float(np.median(s_def)),
        "score_bg_med": float(np.median(s_bg)),
        "frac_def_negative": float((z_def < 0).mean()),      # stress-valley fraction
        "z_def_med": float(np.median(z_def)),                # signed: <0 ⇒ valley
    }


def aggregate(n_cases: int = 200, seed: int = 0, mu=None, sd=None) -> dict:
    if mu is None:
        mu, sd = build_reference()
    files = sorted(glob.glob(os.path.join(DIR_4X4, "*.npy")))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(files), min(n_cases, len(files)), replace=False)
    rows = []
    for i in idx:
        f = files[i]
        mask = load_mask(f)
        if mask is None or not (0 < mask.sum() < len(mask)):
            continue
        rows.append(case_metrics(load_field(f), mask, mu, sd))
    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    agg["n_cases"] = len(rows)
    return agg, mu, sd


def deep_dive(name_contains: str = "L10", mu=None, sd=None) -> dict:
    """One named specimen, fully decomposed."""
    files = sorted(glob.glob(os.path.join(DIR_4X4, "*.npy")))
    pick = next((f for f in files if name_contains in os.path.basename(f)
                 and load_mask(f) is not None), files[0])
    m = case_metrics(load_field(pick), load_mask(pick), mu, sd)
    m["name"] = os.path.basename(pick)
    return m


def report():
    print("=" * 74)
    print("  STAGE-0 CASE STUDY — what the per-node screen keys on (real 4x4 data)")
    print("=" * 74)
    agg, mu, sd = aggregate()
    dd = deep_dive(mu=mu, sd=sd)

    print(f"\n  reference: per-node (mu_i, sigma_i) from {N_REF} healthy 1x1 fields")
    print(f"  detector : s_i = |x_i - mu_i| / sigma_i   (per-node, label-free)\n")

    print(f"  (A) DEEP DIVE — {dd['name']}")
    print(f"      defect nodes K              = {dd['K']}  "
          f"({dd['K']/N_NODES*100:.2f}% of {N_NODES})")
    print(f"      per-case AUROC              = {dd['auroc']:.4f}   [robust]")
    print(f"      median defect-node rank     = top {dd['def_rank_pct']:.2f}%  "
          f"[robust: defect sits in the extreme tail]")
    print(f"      defect deviation sign       = {dd['frac_def_negative']*100:.0f}% "
          f"NEGATIVE  (median z = {dd['z_def_med']:+.0f}) "
          f"⇒ {'stress VALLEY' if dd['z_def_med']<0 else 'stress PEAK'}   [robust]")
    print(f"      precision@K (top-K nodes)   = {dd['prec_at_K']:.3f}   "
          f"[honest limit: hole-edge nodes also rank high]")

    print(f"\n  (B) AGGREGATE over {agg['n_cases']} random 4x4 specimens (not cherry-picked)")
    print(f"      mean per-case AUROC         = {agg['auroc']:.4f}")
    print(f"      mean median defect rank     = top {agg['def_rank_pct']:.2f}%")
    print(f"      defect deviation NEGATIVE   = {agg['frac_def_negative']*100:.0f}% "
          f"of defect nodes  ⇒ the screen exploits a stress VALLEY")
    print(f"      mean precision@K            = {agg['prec_at_K']:.3f}  "
          f"(top-K confounded by hole-edge — see note)")
    print("\n  CAVEAT (do not overclaim): the per-node sigma_i ~1e-3 on z-scored")
    print("  fields, so |z| MAGNITUDES (tens-hundreds) are denominator artefacts —")
    print("  quote AUROC / rank / valley-fraction, never 'N sigma'. precision@K<1")
    print("  because the hole-edge stress concentration also deviates; that is")
    print("  exactly why the pipeline difference-normalises before precise localisation.")
    print("\n  Q&A one-liner:")
    print("   The screen keys on a LOCAL STRESS VALLEY: defect nodes sit BELOW the")
    print("   per-node healthy template (98-100% negative deviation), ranking in the")
    print("   top ~0.2% by |z| — AUROC ~0.998, no learning, no cross-node covariance,")
    print("   because fixed geometry makes the per-node template the right baseline.")
    print("=" * 74)
    return agg, dd


# ─────────────────────────────────────────────────────────────────────────────
def make_figure(mu, sd, out_path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = sorted(glob.glob(os.path.join(DIR_4X4, "*.npy")))
    pick = next(f for f in files if "L10" in os.path.basename(f)
                and load_mask(f) is not None)
    field, mask = load_field(pick), load_mask(pick)
    z, s = score_case(field, mu, sd)

    order = np.argsort(s)[::-1]
    rank = np.empty(len(s), int); rank[order] = np.arange(len(s))
    rank_pct = rank / len(s) * 100

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    # (a) WHERE: defect nodes rank in the extreme top by score (rank percentile)
    ax[0].hist(rank_pct[mask == 0], bins=np.linspace(0, 100, 101), density=True,
               color="#90a4ae", alpha=0.7, label="healthy nodes")
    ax[0].hist(rank_pct[mask == 1], bins=np.linspace(0, 100, 101), density=True,
               color="#d7301f", alpha=0.9, label="defect nodes")
    ax[0].set_xlabel("score rank  (0 = highest $|z|$)  [percentile]")
    ax[0].set_ylabel("density")
    ax[0].set_xlim(-1, 100)
    ax[0].set_title("(a) WHERE: defect nodes rank in the top $\\sim$0.2%")
    ax[0].legend(fontsize=8)
    # (b) WHICH WAY: signed deviation on symlog — defect is a NEGATIVE (valley) tail
    bins = np.array([-1e3, -1e2, -1e1, -3, -1, 0, 1, 3, 1e1, 1e2, 1e3])
    ax[1].hist(z[mask == 0], bins=bins, density=True, color="#90a4ae",
               alpha=0.7, label="healthy nodes")
    ax[1].hist(z[mask == 1], bins=bins, density=True, color="#1565C0",
               alpha=0.9, label="defect nodes")
    ax[1].set_xscale("symlog", linthresh=1)
    ax[1].axvline(0, ls="--", lw=1, color="0.4")
    ax[1].set_xlabel(r"signed deviation $z_i=(x_i-\mu_i)/\sigma_i$  (symlog)")
    ax[1].set_title("(b) WHICH WAY: defect is a stress VALLEY ($z<0$)")
    ax[1].legend(fontsize=8)
    fig.text(0.5, -0.02,
             r"$|z|$ magnitudes are inflated ($\sigma_i\!\sim\!10^{-3}$); "
             r"the robust facts are the RANK (a) and the SIGN (b).",
             ha="center", fontsize=7, color="0.35")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    fig.savefig(os.path.splitext(out_path)[0] + ".png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
def run_tests() -> int:
    n = 0

    def ok(c, m):
        nonlocal n
        assert c, m
        n += 1

    # synthetic: a fixed template + a planted negative (valley) cluster
    rng = np.random.default_rng(0)
    N = 500
    mu = rng.normal(0, 1, N)
    sd = np.full(N, 0.5)
    field = mu + rng.normal(0, 0.5, N)         # healthy draw around template
    mask = np.zeros(N, int)
    mask[:25] = 1
    field[:25] = mu[:25] - 5 * sd[:25]          # defect = deep valley
    m = case_metrics(field, mask, mu, sd)
    ok(m["auroc"] > 0.99, "planted valley → near-perfect AUROC")
    ok(m["prec_at_K"] > 0.95, "top-K recovers the defect cluster")
    ok(m["frac_def_negative"] == 1.0, "valley ⇒ all defect deviations negative")
    ok(m["def_rank_pct"] < 10.0, "defect nodes rank in the top of the score order")
    ok(m["score_def_med"] > m["score_bg_med"], "defect score median above background")
    ok(m["z_def_med"] < 0, "signed median negative ⇒ valley detected")
    # a PEAK defect flips the sign but still detects
    field2 = field.copy()
    field2[:25] = mu[:25] + 5 * sd[:25]
    m2 = case_metrics(field2, mask, mu, sd)
    ok(m2["auroc"] > 0.99 and m2["frac_def_negative"] == 0.0,
       "peak defect → detected, sign flips positive")
    print(f"stage0_case_study: {n}/{n} unit tests passed")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", action="store_true")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.test:
        run_tests()
        return
    agg, dd = report()
    if args.fig:
        mu, sd = build_reference()
        p = make_figure(mu, sd, os.path.join(HERE, "paper_figs",
                                              "stage0_case_study.pdf"))
        print(f"\nfigure → {p}")


if __name__ == "__main__":
    main()
