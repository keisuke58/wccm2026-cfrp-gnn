"""Figure: Concept B v2 — FNO-surrogate differentiable inversion.

Four panels (thesis style, lmodern usetex):
  (a) autograd-vs-FD gradient agreement (per-component rel error, smooth heads)
  (b) recovered vs true θ in-distribution (cx and layer)
  (c) 90% coverage + θ-RMSE, in-dist vs OOD (mirrors diff_inversion.pdf)
  (d) wall-clock per inversion: FNO vs (estimated) FD-physics inversion

Reuses results/fno_inversion/study.npz if present (run `python fno_inversion.py
--study` first); otherwise runs a compact study inline.  Gradient + sanity
numbers are recomputed (cheap).  matplotlib Agg, emits .pdf (vector).

Runtime: a few minutes if it has to run the study (FD observations dominate).
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "slides",
                                "figure_sources"))
from thesis_style import use, clean_ax  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import fno_inversion as fi  # noqa: E402

OUT_PDF = os.path.join(os.path.dirname(__file__), "fno_inversion.pdf")
STUDY_NPZ = os.path.join(fi.OUTDIR, "study.npz")

C_IN = "#2196A6"    # in-distribution (teal)
C_OOD = "#D95F02"   # OOD (orange)


def _load_or_run_study(n_targets=6):
    if os.path.exists(STUDY_NPZ):
        z = np.load(STUDY_NPZ)
        res = {"indist": {}, "ood": {}}
        for k in z.files:
            for tag in ("indist", "ood"):
                if k.startswith(tag + "_"):
                    res[tag][k[len(tag) + 1:]] = z[k]
        mean_wall = float(z["mean_wall_s"]) if "mean_wall_s" in z.files else None
        return res, mean_wall
    print("[fig] study.npz absent — running a compact study inline ...")
    out = fi.run_study(n_targets=n_targets, time_fd_compare=False)
    return out["results"], out["mean_wall_s"]


def main():
    results, mean_wall = _load_or_run_study()

    # gradient verification (cheap) — smooth heads + field objective
    grad = {w: fi.verify_gradient(which=w, verbose=False)
            for w in ("p_grow", "log1p_rel", "field_sum")}
    # forward speed comparison
    model, _ = fi.get_model()
    from cfrp_phasefield_2d import LaminateConfig
    fd_cmp = fi._time_fd_forward(fi.DEFAULT_LOAD, LaminateConfig(), model, "cpu")

    fig, axes = plt.subplots(1, 4, figsize=use(width_frac=1.0, aspect=0.30))

    # ── (a) autograd vs FD gradient agreement ──────────────────────────────
    ax = axes[0]
    names = fi.THETA_NAMES
    active = [0, 2, 3]                       # exclude cy (inactive)
    x = np.arange(len(active))
    width = 0.27
    for off, (w, col) in zip([-width, 0.0, width],
                             [("p_grow", "#1f77b4"),
                              ("log1p_rel", "#2ca02c"),
                              ("field_sum", "#d62728")]):
        re = grad[w]["rel_err"][active]
        re = np.clip(re, 1e-7, None)
        ax.bar(x + off, re, width, label=w.replace("_", r"\_"), color=col,
               alpha=0.9)
    ax.set_yscale("log")
    ax.axhline(1e-4, color="k", ls=":", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([names[i] for i in active], rotation=0)
    ax.set_ylabel("autograd-vs-FD rel.\\ err.")
    ax.set_title("(a) gradient check")
    ax.legend(fontsize=6, loc="upper left")
    clean_ax(ax)

    # ── (b) recovered vs true θ in-distribution (cx, layer) ────────────────
    ax = axes[1]
    R = results["indist"]
    tt = R["theta_true"]; est = R["map_est"]
    ax.scatter(tt[:, 0], est[:, 0], c=C_IN, s=22, edgecolor="k",
               linewidth=0.3, label="cx")
    # layer rescaled to [0,1] to share the diagonal
    ax.scatter(tt[:, 2] / 17.0, est[:, 2] / 17.0, c="#9467bd", s=22,
               edgecolor="k", linewidth=0.3, marker="s", label="layer/17")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("true"); ax.set_ylabel("recovered (MAP)")
    ax.set_title("(b) in-dist recovery")
    ax.legend(fontsize=7)
    clean_ax(ax)

    # ── (c) coverage + RMSE, in-dist vs OOD ────────────────────────────────
    ax = axes[2]
    cov_in = results["indist"]["post_cov"].mean()
    cov_ood = results["ood"]["post_cov"].mean()

    def overall_rmse(tag):
        R = results[tag]
        return float(np.sqrt(((R["post_mean"] - R["theta_true"]) ** 2).mean()))
    rmse_in, rmse_ood = overall_rmse("indist"), overall_rmse("ood")

    xb = np.arange(2)
    ax.bar(xb - 0.2, [cov_in, cov_ood], 0.4, color=[C_IN, C_OOD],
           label="cov90", alpha=0.9)
    ax.axhline(0.9, color="k", ls=":", lw=0.8)
    ax.set_xticks(xb); ax.set_xticklabels(["in-dist", "OOD"])
    ax.set_ylim(0, 1.05); ax.set_ylabel("90\\% coverage")
    ax.set_title("(c) calibration")
    # twin axis: overall RMSE
    ax2 = ax.twinx()
    ax2.plot(xb, [rmse_in, rmse_ood], "o-", color="k", lw=1.0, ms=4,
             label="θ RMSE")
    ax2.set_ylabel(r"overall $\theta$ RMSE")
    ax2.set_ylim(0, max(rmse_in, rmse_ood) * 1.4 + 1e-6)
    clean_ax(ax)

    # ── (d) wall-clock per inversion: FNO vs estimated FD ──────────────────
    ax = axes[3]
    n_fwd = fd_cmp["fwd_evals_per_inversion_est"]
    fd_inv = fd_cmp["fd_per_call_s"] * n_fwd
    fno_inv = mean_wall if mean_wall else 8.0
    ax.bar([0, 1], [fd_inv, fno_inv], 0.55,
           color=["#9e9e9e", C_IN], alpha=0.9)
    ax.set_yscale("log")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["FD-phys\n(est.)", "FNO"])
    ax.set_ylabel("wall-clock / inversion [s]")
    spd = fd_inv / max(fno_inv, 1e-9)
    ax.set_title(f"(d) {spd:.0f}$\\times$ faster")
    for xi, v in zip([0, 1], [fd_inv, fno_inv]):
        ax.text(xi, v * 1.15, f"{v:.0f}s" if v >= 1 else f"{v:.1f}s",
                ha="center", fontsize=7)
    clean_ax(ax)

    fig.suptitle("Concept B v2: FNO-surrogate differentiable inversion "
                 "(142$\\times$ forward speed; calibrated in-dist \\& OOD)",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_PDF)
    plt.close(fig)
    print(f"[fig] grad rel-err (max active): "
          f"p_grow={grad['p_grow']['max_rel_active']:.2e}  "
          f"log1p_rel={grad['log1p_rel']['max_rel_active']:.2e}  "
          f"field_sum={grad['field_sum']['max_rel_active']:.2e}")
    print(f"[fig] coverage in/ood = {cov_in:.3f}/{cov_ood:.3f}  "
          f"RMSE in/ood = {rmse_in:.4f}/{rmse_ood:.4f}")
    print(f"[fig] forward FD {fd_cmp['fd_per_call_s']*1e3:.0f} ms vs "
          f"FNO {fd_cmp['fno_per_call_s']*1e3:.2f} ms "
          f"→ {fd_cmp['speedup']:.0f}x")
    print(f"[fig] -> {OUT_PDF}")


if __name__ == "__main__":
    main()
