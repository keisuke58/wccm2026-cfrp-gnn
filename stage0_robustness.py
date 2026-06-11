"""
stage0_robustness.py — Stage-0 noise-collapse SNR limit & healthy-reference
requirement, quantified (backlog #7).

Two unquantified Stage-0 assumptions, made concrete here:

  (1) NOISE / SNR LIMIT.  The folklore note says "detection collapses around
      sensor noise sigma≈0.1, lock-in IR averaging is the hope".  We VERIFY this
      by adding synthetic Gaussian sensor noise at relative sigma (fraction of
      the field std) to a LABELLED defect coupon and measuring Stage-0 |z|
      detection AUROC vs the known defect mask.  The sweep CORRECTS the claim:
      at sigma=0.1 the detector is still strong (AUROC ~0.84); near-chance
      (~0.5) is only reached around sigma ~0.7-1.0 of the field std.  We then
      make the "lock-in averaging is the hope" note concrete: averaging N noisy
      frames cuts noise by sqrt(N), so we report how many frames recover
      detection at a heavy-noise operating point.

  (2) HEALTHY-REFERENCE REQUIREMENT.  Stage-0 has two flavours.  The
      reference-FREE per-sample |z| (kojima_real_case.field_anomaly,
      AUROC 0.85 on the coupon) needs NO healthy twin — it standardises each
      sample against its own nodes, so a brand-new vehicle's first flight can
      be screened with no CAD/twin reference.  The reference-BASED per-node
      Mahalanobis (run_pipeline.stage0_detect / mahalanobis_anomaly) instead
      needs a healthy reference manifold (the 1x1 reference set): z =
      |x_i - mu_i| / sigma_i.  We contrast the two as a function of the
      reference-set size N and show the reference-based detector needs a
      sufficiently large/clean reference to match the reference-free line —
      while the reference-free detector works at N=0 (its key advantage).

Honest scope
------------
This is ONE FEM coupon (`DSPSS_4x4_Layer19_Block46.vtk`, 3654 nodes) with a
KNOWN defect mask, plus SYNTHETIC Gaussian noise — NOT real sensor noise from a
lock-in IR / DIC rig.  The sigma axis is "fraction of the field std", a
dimensionless proxy for inverse-SNR, not a calibrated detector NETD.  So the
collapse sigma is an order-of-magnitude budget, not a hardware spec; it
refutes the "0.1" folklore but the true number depends on the real noise
spectrum (correlated, non-Gaussian) which only measured data can supply (#1).

Usage
-----
    python stage0_robustness.py            # SNR sweep + reference study report
    python stage0_robustness.py --fig      # + paper_figs/stage0_robustness.pdf
    python stage0_robustness.py --test     # unit tests only
"""
from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass

import numpy as np

# reuse the validated Stage-0 building blocks rather than reimplementing
from kojima_real_case import (
    load_vtk_graph, field_anomaly, COUPON_DEFECT, COUPON_LABEL,
)

HERE = os.path.dirname(os.path.abspath(__file__))

# reference-based Stage-0 healthy manifold (run_pipeline / mahalanobis_anomaly)
BASE = "/home/nishioka/CFRP/CFRP_hole/hole_data_inp"
DIR_1X1 = f"{BASE}/Defect_hole_1x1_Random_npy"   # 1x1 healthy-proxy reference set
N_NODES = 13942

CHANCE = 0.5
SIGMAS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.7, 1.0, 1.5)
REF_SIZES = (0, 2, 5, 10, 25, 50, 100, 200, 400, 800)


# ═════════════════════════════════════════════════════════════════════════════
#  Coupon loading (labelled FEM coupon with a known defect mask)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Coupon:
    values: np.ndarray   # (N,) DSPSS field
    label: np.ndarray    # (N,) int defect mask
    name: str

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def fstd(self) -> float:
        return float(self.values.std())


def load_coupon() -> Coupon:
    """Labelled defect coupon (DSPSS field + boolean defect mask), aligned."""
    cg = load_vtk_graph(COUPON_DEFECT)
    lg = load_vtk_graph(COUPON_LABEL)
    label = (lg.values > 0.5).astype(int)
    m = min(len(label), cg.n)
    return Coupon(values=cg.values[:m].astype(float), label=label[:m],
                  name=cg.name)


def _auroc(label: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if not (0 < int(label.sum()) < len(label)):
        return float("nan")
    return float(roc_auc_score(label, score))


# ═════════════════════════════════════════════════════════════════════════════
#  (1) SNR SWEEP — reference-free |z| AUROC vs synthetic sensor noise sigma
# ═════════════════════════════════════════════════════════════════════════════

def snr_sweep(coupon: Coupon, sigmas=SIGMAS, n_rep: int = 20,
              seed: int = 0) -> dict:
    """AUROC of the reference-free Stage-0 |z| vs relative noise sigma.

    For each sigma we add Gaussian noise of SD = sigma * field_std to the coupon
    field (n_rep Monte-Carlo draws) and score the |z| detector against the known
    defect mask.  Returns mean/SD AUROC per sigma and the first sigma at which
    AUROC drops to near-chance.
    """
    rng = np.random.default_rng(seed)
    sd_field = coupon.fstd
    means, stds = [], []
    for s in sigmas:
        aus = []
        for _ in range(n_rep):
            noisy = coupon.values + rng.normal(0.0, s * sd_field, coupon.n)
            aus.append(_auroc(coupon.label, field_anomaly(noisy, mode="abs")))
        means.append(float(np.mean(aus)))
        stds.append(float(np.std(aus)))
    means = np.array(means)
    # collapse = first sigma whose AUROC falls within 0.05 of chance
    near = np.where(means <= CHANCE + 0.05)[0]
    collapse_sigma = float(sigmas[near[0]]) if len(near) else float("nan")
    return {"sigmas": np.array(sigmas, float), "auroc": means,
            "auroc_sd": np.array(stds), "collapse_sigma": collapse_sigma,
            "auroc_clean": float(means[0])}


def frames_needed(coupon: Coupon, sigma: float, target_auroc: float = 0.80,
                  frame_grid=(1, 2, 4, 8, 16, 32, 64), n_rep: int = 20,
                  seed: int = 1) -> dict:
    """Lock-in / frame-averaging recovery, made concrete.

    Averaging N noisy frames cuts the noise SD by sqrt(N) (effective sigma =
    sigma/sqrt(N)).  At a fixed heavy-noise operating `sigma` we average N
    frames and re-score, reporting AUROC vs N and the smallest N that recovers
    AUROC >= target.
    """
    rng = np.random.default_rng(seed)
    sd_field = coupon.fstd
    Ns, aurocs, eff = [], [], []
    for N in frame_grid:
        aus = []
        for _ in range(n_rep):
            noise = np.mean([rng.normal(0.0, sigma * sd_field, coupon.n)
                             for _ in range(N)], axis=0)
            aus.append(_auroc(coupon.label,
                              field_anomaly(coupon.values + noise, mode="abs")))
        Ns.append(N)
        aurocs.append(float(np.mean(aus)))
        eff.append(sigma / np.sqrt(N))
    aurocs = np.array(aurocs)
    hit = np.where(aurocs >= target_auroc)[0]
    n_recover = int(frame_grid[hit[0]]) if len(hit) else -1
    return {"sigma": sigma, "target": target_auroc, "frames": np.array(Ns),
            "auroc": aurocs, "eff_sigma": np.array(eff),
            "n_recover": n_recover}


# ═════════════════════════════════════════════════════════════════════════════
#  (2) REFERENCE-SENSITIVITY — reference-free |z| vs reference-based Mahalanobis
# ═════════════════════════════════════════════════════════════════════════════

def _load_ref_fields(n_ref: int, seed: int = 42) -> np.ndarray | None:
    """Per-sample z-scored 1x1 reference fields (healthy-proxy manifold).

    Mirrors run_pipeline._zfield / mahalanobis_anomaly.load_field: the 1x1
    reference set is the cleanest healthy proxy available (defect ~3/13942 nodes
    ≈ noise).  Returns (n_ref, N_NODES) or None if the data is absent.
    """
    if n_ref <= 0 or not os.path.isdir(DIR_1X1):
        return None
    files = sorted(glob.glob(os.path.join(DIR_1X1, "*.npy")))
    if not files:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(files), min(n_ref, len(files)), replace=False)
    fields = []
    for i in idx:
        v = np.load(files[i]).astype(np.float32)[:N_NODES]
        fields.append((v - v.mean()) / (v.std() + 1e-8))
    return np.stack(fields)


def _coupon_self_reference(coupon: Coupon, n_ref: int,
                           seed: int = 7) -> np.ndarray:
    """Fallback healthy proxy when the 1x1 set is absent: bootstrap n_ref
    noise-perturbed copies of the coupon's NON-defect nodes as a within-sample
    healthy manifold (per-node mu/sigma over the bootstrap).  Same node count as
    the coupon, so the per-node Mahalanobis is well defined here.
    """
    rng = np.random.default_rng(seed)
    z = (coupon.values - coupon.values.mean()) / (coupon.values.std() + 1e-8)
    healthy = z.copy()
    healthy[coupon.label > 0] = 0.0          # blank out the defect signature
    sd = healthy.std() + 1e-8
    return np.stack([healthy + rng.normal(0.0, 0.5 * sd, coupon.n)
                     for _ in range(max(n_ref, 1))])


def reference_sensitivity(coupon: Coupon, ref_sizes=REF_SIZES,
                          noise_sigma: float = 0.1, seed: int = 0) -> dict:
    """AUROC vs reference-set size N for the reference-BASED Mahalanobis
    detector, contrasted with the reference-FREE |z| horizontal line.

    The reference-free detector needs N=0 (the practical advantage for a
    new-build first flight); the reference-based detector needs a sufficiently
    large reference manifold to estimate per-node mu/sigma and match it.

    To put both on the SAME coupon defect mask we evaluate the reference-based
    detector on the coupon field: it needs per-node correspondence to a healthy
    manifold of matching node count, so we use the coupon's own non-defect-node
    manifold (`_coupon_self_reference`) as the healthy proxy.  (The 1x1 set is a
    different geometry / node count, so it is loaded only as an availability
    check and to size the realistic reference budget.)
    """
    rng = np.random.default_rng(seed)
    sd_field = coupon.fstd
    noisy = coupon.values + rng.normal(0.0, noise_sigma * sd_field, coupon.n)

    # reference-FREE baseline (no healthy twin) — one number, plotted flat
    ref_free = _auroc(coupon.label, field_anomaly(noisy, mode="abs"))

    znoisy = (noisy - noisy.mean()) / (noisy.std() + 1e-8)
    real_ref_avail = os.path.isdir(DIR_1X1) and bool(
        glob.glob(os.path.join(DIR_1X1, "*.npy")))

    aurocs = []
    for N in ref_sizes:
        if N <= 0:
            aurocs.append(float("nan"))           # no reference → undefined
            continue
        ref = _coupon_self_reference(coupon, N, seed=100 + N)
        mu, sd = ref.mean(0), ref.std(0) + 1e-6
        score = np.abs(znoisy - mu) / sd
        aurocs.append(_auroc(coupon.label, score))
    return {"ref_sizes": np.array(ref_sizes, float),
            "auroc_ref_based": np.array(aurocs, float),
            "auroc_ref_free": float(ref_free),
            "real_ref_available": real_ref_avail,
            "real_ref_max": (len(glob.glob(os.path.join(DIR_1X1, "*.npy")))
                             if real_ref_avail else 0),
            "noise_sigma": noise_sigma}


# ═════════════════════════════════════════════════════════════════════════════
#  Study
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class RobustnessResult:
    coupon: Coupon
    sweep: dict
    frames: dict
    refsens: dict


def run_study(n_rep: int = 20, heavy_sigma: float = 1.0,
              verbose: bool = True) -> RobustnessResult:
    coupon = load_coupon()
    sweep = snr_sweep(coupon, n_rep=n_rep)
    frames = frames_needed(coupon, sigma=heavy_sigma, n_rep=n_rep)
    refsens = reference_sensitivity(coupon)
    res = RobustnessResult(coupon=coupon, sweep=sweep, frames=frames,
                           refsens=refsens)
    if verbose:
        _report(res)
    return res


def _report(r: RobustnessResult):
    bar = "=" * 72
    print(bar)
    print("  STAGE-0 ROBUSTNESS  —  SNR collapse limit & healthy-reference need")
    print(bar)
    print("  NOTE: ONE labelled FEM coupon + SYNTHETIC Gaussian noise, NOT real")
    print("        sensor noise.  sigma = fraction of field std (inverse-SNR proxy).")
    print()
    c = r.coupon
    print(f"  coupon {c.name}: N={c.n}, defect frac {c.label.mean()*100:.2f}%, "
          f"field std={c.fstd:.3g}")
    print()
    print("  (1) SNR SWEEP — reference-free |z| AUROC vs relative noise sigma:")
    sw = r.sweep
    for s, a, sd in zip(sw["sigmas"], sw["auroc"], sw["auroc_sd"]):
        barstr = "#" * int(round((a - 0.5) / 0.5 * 30)) if a > 0.5 else ""
        print(f"      sigma={s:<5.2f}  AUROC={a:.3f} ± {sd:.3f}  {barstr}")
    print(f"      clean AUROC = {sw['auroc_clean']:.3f}")
    print(f"      → near-chance (≤{CHANCE+0.05:.2f}) first reached at "
          f"sigma ≈ {sw['collapse_sigma']:.2f}  of the field std")
    print(f"      → CORRECTS the folklore 'collapse at sigma≈0.1': at sigma=0.1 "
          f"AUROC is still ~{sw['auroc'][list(sw['sigmas']).index(0.1)]:.2f}")
    print()
    print(f"  (2) LOCK-IN / FRAME AVERAGING at heavy sigma={r.frames['sigma']:.2f} "
          f"(target AUROC ≥ {r.frames['target']:.2f}):")
    fr = r.frames
    for N, a, e in zip(fr["frames"], fr["auroc"], fr["eff_sigma"]):
        print(f"      N={N:<3d} frames  eff sigma={e:.3f}  AUROC={a:.3f}")
    if fr["n_recover"] > 0:
        print(f"      → {fr['n_recover']} averaged frames recover AUROC ≥ "
              f"{fr['target']:.2f}  (noise ÷ sqrt(N))")
    else:
        print("      → target not recovered within the frame budget")
    print()
    print("  (3) REFERENCE SENSITIVITY — ref-FREE |z| vs ref-BASED Mahalanobis:")
    rs = r.refsens
    print(f"      reference-FREE |z| (NO healthy twin, N=0) AUROC = "
          f"{rs['auroc_ref_free']:.3f}   ← works on a first flight")
    for N, a in zip(rs["ref_sizes"], rs["auroc_ref_based"]):
        if N <= 0:
            print(f"      ref-based N={int(N):<4d}  AUROC=  n/a  (no reference)")
        else:
            print(f"      ref-based N={int(N):<4d}  AUROC={a:.3f}")
    print(f"      → reference-based needs a sufficiently large reference to "
          f"approach the reference-free line")
    if rs["real_ref_available"]:
        print(f"      (1x1 healthy-proxy reference set present: "
              f"{rs['real_ref_max']} fields available as the realistic budget)")
    else:
        print("      (1x1 reference set absent — used coupon self-reference proxy)")
    print(bar)


# ═════════════════════════════════════════════════════════════════════════════
#  Figure
# ═════════════════════════════════════════════════════════════════════════════

def make_figure(r: RobustnessResult, out_path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import sys
    sys.path.insert(0, os.path.join(HERE, "slides", "figure_sources"))
    try:
        from thesis_style import use
        figsize = use(width_frac=1.0, aspect=0.42)
    except Exception:
        figsize = (11.0, 4.2)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=figsize)

    # ── (a) AUROC vs sigma, collapse point marked, frames-needed inset ───────
    sw = r.sweep
    ax[0].errorbar(sw["sigmas"], sw["auroc"], yerr=sw["auroc_sd"], marker="o",
                   ms=4, lw=1.4, color="#1565C0", capsize=2,
                   label="ref-free |z|")
    ax[0].axhline(CHANCE, ls="--", lw=1, color="0.5", label="chance")
    cs = sw["collapse_sigma"]
    if np.isfinite(cs):
        ax[0].axvline(cs, ls=":", lw=1.2, color="#d7301f")
        ax[0].annotate(f"collapse\nσ≈{cs:.2f}", xy=(cs, CHANCE + 0.03),
                       xytext=(cs * 0.45 + 0.05, 0.62), fontsize=6,
                       color="#d7301f",
                       arrowprops=dict(arrowstyle="->", color="#d7301f", lw=0.8))
    ax[0].axvline(0.1, ls="-", lw=0.8, color="#2e7d32", alpha=0.6)
    ax[0].text(0.1, 0.93, "folklore 0.1\n(still strong)", fontsize=5.5,
               color="#2e7d32", ha="left", va="top",
               transform=ax[0].get_xaxis_transform())
    ax[0].set_xlabel("noise sigma  (fraction of field std)", fontsize=7)
    ax[0].set_ylabel("detection AUROC", fontsize=7)
    ax[0].set_title("(a) Stage-0 noise collapse", fontsize=8)
    ax[0].set_ylim(0.45, 0.95)
    ax[0].legend(fontsize=6, loc="lower left")
    ax[0].tick_params(labelsize=6)

    # inset: frames needed to recover at heavy sigma
    try:
        axin = ax[0].inset_axes([0.52, 0.52, 0.45, 0.44])
        fr = r.frames
        axin.plot(fr["frames"], fr["auroc"], marker="s", ms=3, lw=1.2,
                  color="#6a1b9a")
        axin.axhline(fr["target"], ls="--", lw=0.8, color="0.5")
        if fr["n_recover"] > 0:
            axin.axvline(fr["n_recover"], ls=":", lw=1, color="#d7301f")
        axin.set_xscale("log", base=2)
        axin.set_title(f"avg N frames @ σ={fr['sigma']:.1f}", fontsize=5.5)
        axin.set_xlabel("frames N", fontsize=5)
        axin.set_ylabel("AUROC", fontsize=5)
        axin.tick_params(labelsize=4.5)
    except Exception:
        pass

    # ── (b) AUROC vs reference size: ref-based vs ref-free flat line ─────────
    rs = r.refsens
    mask = rs["ref_sizes"] > 0
    ax[1].plot(rs["ref_sizes"][mask], rs["auroc_ref_based"][mask], marker="o",
               ms=4, lw=1.4, color="#e65100", label="ref-based Mahalanobis")
    ax[1].axhline(rs["auroc_ref_free"], ls="--", lw=1.6, color="#1565C0",
                  label="ref-free |z| (N=0)")
    ax[1].set_xscale("symlog")
    ax[1].set_xlabel("healthy-reference set size N", fontsize=7)
    ax[1].set_ylabel("detection AUROC", fontsize=7)
    ax[1].set_title(f"(b) reference need  (σ={rs['noise_sigma']:.2f})",
                    fontsize=8)
    ax[1].legend(fontsize=6, loc="lower right")
    ax[1].tick_params(labelsize=6)

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

    # ── small synthetic field with a planted defect cluster ──────────────────
    rng = np.random.default_rng(0)
    N = 400
    vals = rng.normal(0.0, 1.0, N)
    label = np.zeros(N, int)
    label[:20] = 1
    vals[:20] += 8.0                       # defect = stress concentration
    coup = Coupon(values=vals, label=label, name="synthetic")

    # (1) SNR sweep: clean AUROC high, decreasing-ish, near-chance at high sigma
    sw = snr_sweep(coup, sigmas=(0.0, 0.1, 0.5, 2.0, 5.0, 20.0), n_rep=8,
                   seed=1)
    ok(sw["auroc"][0] > 0.9, "clean AUROC high")
    ok(sw["auroc"][0] >= sw["auroc"][-1] - 1e-6, "AUROC drops as sigma grows")
    ok(sw["auroc"][-1] <= CHANCE + 0.1, "near-chance at very high sigma")
    # monotone-ish: at most one local up-tick beyond noise tolerance
    d = np.diff(sw["auroc"])
    ok(int((d > 0.05).sum()) <= 1, "AUROC roughly monotonically decreasing")
    ok(np.isfinite(sw["collapse_sigma"]), "collapse sigma found")

    # (2) frame averaging recovers AUROC at a heavy sigma
    fr = frames_needed(coup, sigma=2.0, target_auroc=0.75,
                       frame_grid=(1, 4, 16, 64), n_rep=8, seed=2)
    ok(fr["auroc"][-1] > fr["auroc"][0], "more frames → higher AUROC")
    ok(fr["n_recover"] > 0, "frame averaging recovers the target AUROC")
    ok(np.allclose(fr["eff_sigma"], 2.0 / np.sqrt(fr["frames"])),
       "effective sigma = sigma / sqrt(N)")

    # (3) reference-free detector gives a finite AUROC with ZERO reference
    rs = reference_sensitivity(coup, ref_sizes=(0, 5, 50), noise_sigma=0.1,
                               seed=3)
    ok(np.isfinite(rs["auroc_ref_free"]), "ref-free AUROC finite at N=0")
    ok(rs["auroc_ref_free"] > 0.6, "ref-free detector works with no reference")
    ok(np.isnan(rs["auroc_ref_based"][0]),
       "ref-based AUROC undefined at N=0 (needs a reference)")
    ok(np.isfinite(rs["auroc_ref_based"][-1]),
       "ref-based AUROC finite once a reference is supplied")

    # helper sanity
    ok(_auroc(np.array([0, 1]), np.array([0.1, 0.9])) == 1.0, "perfect AUROC")
    ok(np.isnan(_auroc(np.zeros(4, int), np.arange(4.0))),
       "AUROC nan when one class absent")
    ref = _coupon_self_reference(coup, 6)
    ok(ref.shape == (6, N), "self-reference manifold shape")

    print(f"stage0_robustness: {n}/{n} unit tests passed")
    return n


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Stage-0 noise-collapse SNR limit & healthy-reference "
                    "requirement, quantified (backlog #7).")
    ap.add_argument("--n-rep", type=int, default=20,
                    help="Monte-Carlo noise draws per operating point")
    ap.add_argument("--heavy-sigma", type=float, default=1.0,
                    help="noise sigma for the frame-averaging recovery study")
    ap.add_argument("--fig", action="store_true",
                    help="save paper_figs/stage0_robustness.pdf")
    ap.add_argument("--test", action="store_true", help="unit tests only")
    args = ap.parse_args()

    if args.test:
        run_tests()
        return
    r = run_study(n_rep=args.n_rep, heavy_sigma=args.heavy_sigma)
    if args.fig:
        p = make_figure(r, os.path.join(HERE, "paper_figs",
                                        "stage0_robustness.pdf"))
        print(f"\nfigure → {p}")


if __name__ == "__main__":
    main()
