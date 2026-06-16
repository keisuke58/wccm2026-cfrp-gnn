"""conformal_transfer.py — how a clearance coverage guarantee DEGRADES across structures [idea #5].

`conformal_clearance.py` gives a split-conformal guarantee that the OK/REPAIR/RETIRE prediction
set covers the oracle action with prob >= 1-alpha — but ONLY under exchangeability, i.e. when the
calibration scenarios come from the SAME structure you deploy on. The honest operational question
for a structure-agnostic pipeline is: if I calibrate on structures I HAVE and deploy on a NEW one,
how much does the guarantee erode, and can I restore it?

This module quantifies that, reusing the exact decision core (`decision_uq.decide`) and the
conformal machinery (`conformal_clearance.nonconformity / conformal_quantile / prediction_sets`):

  1. COMMON RULE. We fix one deployed decision rule (alpha_band, beta_band) so the action bands —
     and therefore the conformal scores — are identical across structures; coverage then measures
     transfer, not a change of rule.
  2. IN-DISTRIBUTION baseline: split-conformal within each structure -> coverage ~ 1-alpha (sanity).
  3. CROSS-STRUCTURE: calibrate qhat on a SOURCE structure (or the pooled others = LOSO), deploy on
     a held-out TARGET. Empirical coverage on the target is the transferred guarantee.
  4. STRUCTURE DISTANCE: d = 1-Wasserstein distance between the SOURCE and TARGET distributions of
     the true-action conformal score. Coverage erodes as d grows (the exchangeability break is
     bounded by this shift — cf. weighted/covariate-shift conformal, Tibshirani et al. 2019).
  5. DISTANCE-INFLATED CORRECTION: inflate the quantile qhat -> min(1, qhat + c*d). We fit the
     single constant c so the WORST cross-structure coverage is restored to >= 1-alpha, and report
     the price in mean prediction-set size (efficiency lost to honesty).

Run:  python3 conformal_transfer.py   (prints the transfer table; writes conformal_transfer.png)
"""
from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance

import conformal_clearance as cc
import decision_uq as duq
import system_baseline as sb
from system_baseline import DECISIONS

A_BAND, B_BAND = 0.02, 0.48        # the single deployed decision rule (common across structures)
ALPHA = 0.10                       # nominal miss budget -> target coverage 1-alpha = 0.90
SEED = 20270614


def oracle_actions(sc):
    return np.array([duq.decide(float(p), A_BAND, B_BAND) for p in sc.p_oracle])


def true_action_scores(sc):
    """Per-point nonconformity of the TRUE (oracle) action under the common rule."""
    s = cc.nonconformity(np.asarray(sc.p_pipeline, float), A_BAND, B_BAND)   # (M,3)
    oa = oracle_actions(sc)
    j = np.array([DECISIONS.index(a) for a in oa])
    return s[np.arange(len(j)), j], oa


def deploy(qhat, test_sc):
    sets = cc.prediction_sets(np.asarray(test_sc.p_pipeline, float), A_BAND, B_BAND, qhat)
    oa = oracle_actions(test_sc)
    cov = float(np.mean([oa[i] in sets[i] for i in range(len(oa))]))
    size = float(np.mean([len(s) for s in sets]))
    return cov, size


def in_distribution(sc, alpha, seed=SEED):
    """Split-conformal WITHIN one structure (exchangeable) -> baseline coverage."""
    cal_scores, oa = true_action_scores(sc)
    M = len(oa)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(M)
    n_cal = max(1, M // 2)
    cal, test = idx[:n_cal], idx[n_cal:]
    qhat = cc.conformal_quantile(cal_scores[cal], alpha)
    sets = cc.prediction_sets(np.asarray(sc.p_pipeline, float)[test], A_BAND, B_BAND, qhat)
    cov = float(np.mean([oa[t] in sets[i] for i, t in enumerate(test)]))
    size = float(np.mean([len(s) for s in sets]))
    return qhat, cov, size


def structure_distance(src_sc, tgt_sc):
    s_src, _ = true_action_scores(src_sc)
    s_tgt, _ = true_action_scores(tgt_sc)
    return float(wasserstein_distance(s_src, s_tgt))


def main():
    per = cc.structure_scenarios(seed=SEED, M=80)
    names = [n for n in per if n != "pooled"]
    print(f"structures: {names}   common rule (alpha,beta)=({A_BAND},{B_BAND})  target cov={1-ALPHA:.2f}\n")

    # 1) in-distribution baseline
    print("=== in-distribution (exchangeable) split-conformal ===")
    for n in names:
        _, cov, size = in_distribution(per[n], ALPHA)
        print(f"  {n:16s} coverage={cov:.2f}  set_size={size:.2f}")

    # 2) cross-structure pairs + LOSO (pooled others -> held-out)
    rows = []   # (label, distance, coverage, size, qhat)
    print("\n=== cross-structure transfer (calibrate SOURCE -> deploy TARGET) ===")
    for src in names:
        cal_scores, _ = true_action_scores(per[src])
        qhat = cc.conformal_quantile(cal_scores, ALPHA)
        for tgt in names:
            if tgt == src:
                continue
            d = structure_distance(per[src], per[tgt])
            cov, size = deploy(qhat, per[tgt])
            rows.append((f"{src[:4]}->{tgt[:4]}", d, cov, size, qhat))
    # LOSO: pooled others -> held out
    loso = []
    for tgt in names:
        others = cc.pool_scenarios([per[n] for n in names if n != tgt])
        cal_scores, _ = true_action_scores(others)
        qhat = cc.conformal_quantile(cal_scores, ALPHA)
        d = structure_distance(others, per[tgt])
        cov, size = deploy(qhat, per[tgt])
        loso.append((f"LOSO->{tgt[:4]}", d, cov, size, qhat))

    for lab, d, cov, size, q in rows + loso:
        flag = "" if cov >= 1 - ALPHA else "  <-- UNDER-COVERS"
        print(f"  {lab:14s} dist={d:.3f}  coverage={cov:.2f}  set_size={size:.2f}{flag}")

    # 3) distance-inflated correction: qhat -> min(1, qhat + c*d); fit c on the cross pairs
    allp = rows + loso
    ds = np.array([r[1] for r in allp])
    # worst under-coverage drives c: need qhat+c*d large enough. Search c.
    cs = np.linspace(0, 5, 400)
    def worst_cov(c):
        covs = []
        for lab, d, _, _, q in allp:
            # recompute target from label
            tgt = [n for n in names if n[:4] == lab.split("->")[1]][0]
            cov, _ = deploy(min(1.0, q + c * d), per[tgt])
            covs.append(cov)
        return min(covs)
    c_fit = next((c for c in cs if worst_cov(c) >= 1 - ALPHA), cs[-1])
    print(f"\n=== distance-inflated correction  qhat -> min(1, qhat + {c_fit:.2f} * dist) ===")
    corr = []
    for lab, d, cov, size, q in allp:
        tgt = [n for n in names if n[:4] == lab.split("->")[1]][0]
        cov2, size2 = deploy(min(1.0, q + c_fit * d), per[tgt])
        corr.append((lab, d, cov2, size2))
        print(f"  {lab:14s} dist={d:.3f}  coverage {cov:.2f}->{cov2:.2f}  set_size {size:.2f}->{size2:.2f}")
    print(f"\nworst coverage  raw={min(r[2] for r in allp):.2f}  corrected={min(r[2] for r in corr):.2f}"
          f"  (target {1-ALPHA:.2f});  mean set_size {np.mean([r[3] for r in allp]):.2f}"
          f"->{np.mean([r[3] for r in corr]):.2f}")

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7))
    ax = axes[0]
    ax.scatter(ds, [r[2] for r in allp], s=70, c="#c0392b", ec="k", zorder=3, label="cross-structure")
    for lab, d, cov, size, q in allp:
        ax.annotate(lab, (d, cov), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.axhline(1 - ALPHA, color="green", ls="--", lw=1.5, label=f"target {1-ALPHA:.2f}")
    ax.set(xlabel="structure distance  (Wasserstein of conformal score)",
           ylabel="empirical coverage on target",
           title="Coverage erodes with structure distance\n(exchangeability breaks across structures)")
    ax.legend(fontsize=8); ax.grid(alpha=.3)

    ax = axes[1]
    x = np.arange(len(allp))
    ax.bar(x - 0.2, [r[2] for r in allp], 0.4, color="#c0392b", ec="k", label="raw transfer")
    ax.bar(x + 0.2, [r[2] for r in corr], 0.4, color="#1f6f6f", ec="k", label="distance-inflated")
    ax.axhline(1 - ALPHA, color="green", ls="--", lw=1.5, label=f"target {1-ALPHA:.2f}")
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in allp], rotation=40, ha="right", fontsize=7)
    ax.set(ylabel="coverage", ylim=(0, 1.05),
           title=f"Distance inflation restores the guarantee\n(c={c_fit:.2f}; "
                 f"set size {np.mean([r[3] for r in allp]):.2f}->{np.mean([r[3] for r in corr]):.2f})")
    ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
    fig.suptitle("Distribution-free clearance guarantee across structures: it degrades with "
                 "distance, and a distance-inflated quantile restores it", fontweight="bold")
    fig.tight_layout()
    out = "/home/nishioka/cfrp_datasets/conformal_transfer.png"
    fig.savefig(out, dpi=140)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
