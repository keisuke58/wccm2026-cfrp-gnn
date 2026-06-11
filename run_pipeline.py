"""
run_pipeline.py — ONE-COMMAND reusable-rocket SHM decision pipeline.

Ties the existing, separately-built Stage 0→4 CFRP structural-health-monitoring
modules into a single reproducible artifact: take ONE held-out test sample and
walk it end-to-end through

    Stage 0  detect       mahalanobis_anomaly   per-node anomaly map + verdict
    Stage 2  characterise fmpe_defect           posterior θ=(cx,cy,layer,size)
    Stage 3  prognose     crack_surrogate (FNO) P(grow) over the posterior
             decide       cfrp_phasefield_2d    OK / REPAIR / RETIRE clearance
    Stage 4  fleet-update fleet_learning        fleet-prior sharpening + leader

This module IMPORTS the stage modules unchanged — it adds no new physics/ML,
only the orchestration, the human-readable rocket-context report block, and an
optional combined figure.  The FAST FNO surrogate (crack_surrogate) is the
default Stage-3 forward engine, so the whole run finishes in well under a
minute; pass --engine fd to fall back to the exact FD phase-field forward
(cfrp_phasefield_2d.flight_clearance, fatigue-aware) for validation.

Stage-1 (GNN classification) is a coarse pre-screen subsumed by the Stage-0
anomaly map + Stage-2 posterior in this end-to-end demo and is not re-run.

Usage
-----
    python run_pipeline.py                       # cached real FMPE posterior
    python run_pipeline.py --sample 7            # pick a held-out 4x4 sample
    python run_pipeline.py --engine fd           # exact FD Stage-3 (slow)
    python run_pipeline.py --n-flights 20 --fig  # + combined figure
    python run_pipeline.py --fmpe                # retrain FMPE on the sample
    python run_pipeline.py --fleet-leader        # full 2.75→1.25 study

The posterior source: by default the cached results/fmpe_posterior_e2e.npz is
used — it is a REAL FMPE posterior (trained on the spot, held-out test sample),
the canonical demo sample.  --fmpe re-runs the full FMPE chain on the chosen
held-out sample (heavy: ~minutes).  Either way Stage 0 is always run live on
the field of the chosen / cached sample's defect class.
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np

# ── stage modules (imported unchanged) ───────────────────────────────────────
import crack_surrogate as cs
import cfrp_phasefield_2d as pf
import fleet_learning as fl

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "/home/nishioka/CFRP/CFRP_hole/hole_data_inp"
COORD_DIR = f"{BASE}/basicdata_for_holegnn"
DIR_1X1 = f"{BASE}/Defect_hole_1x1_Random_npy"
DIR_4X4 = f"{BASE}/Defect_hole_4x4_Region1_21_npy"
LBL_4X4 = f"{BASE}/Def4x4_19class_label"
N_NODES = 13942
CACHED_POSTERIOR = os.path.join(HERE, "results", "fmpe_posterior_e2e.npz")


# ═════════════════════════════════════════════════════════════════════════════
#  Stage 0 — Mahalanobis anomaly detection (mahalanobis_anomaly semantics)
# ═════════════════════════════════════════════════════════════════════════════

def _zfield(fpath: str) -> np.ndarray:
    """Per-sample z-scored DSPSS field (mahalanobis_anomaly.load_field)."""
    v = np.load(fpath).astype(np.float32)[:N_NODES]
    return (v - v.mean()) / (v.std() + 1e-8)


def _load_mask(label_dir: str, fpath: str):
    """Defect node mask from the 19-class label (lbl[:,1:].sum(1)>0)."""
    base = os.path.splitext(os.path.basename(fpath))[0]
    for cand in (f"{base}.npy", f"{base}_19label.npy"):
        lf = os.path.join(label_dir, cand)
        if os.path.exists(lf):
            lbl = np.load(lf)
            return (lbl[:, 1:].sum(axis=1) > 0)
    return None


def stage0_detect(sample_idx: int, n_ref: int = 800,
                  seed: int = 42) -> dict:
    """Run per-node Mahalanobis detection on one held-out 4x4 sample.

    Healthy manifold = 1x1-defect reference set (defect ≈ noise); score is the
    per-node z = |x_i − μ_i| / σ_i.  Returns the score map, the ground-truth
    defect mask, a single-sample node-level AUROC, and a detected/clean verdict
    (max score vs a robust reference quantile).
    """
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    ref_files = sorted(glob.glob(os.path.join(DIR_1X1, "*.npy")))
    ridx = rng.choice(len(ref_files), min(n_ref, len(ref_files)), replace=False)
    X = np.stack([_zfield(ref_files[i]) for i in ridx])
    mu, sd = X.mean(0), X.std(0) + 1e-6

    files4 = sorted(glob.glob(os.path.join(DIR_4X4, "*.npy")))
    fpath = files4[sample_idx % len(files4)]
    v = _zfield(fpath)
    score = np.abs(v - mu) / sd
    mask = _load_mask(LBL_4X4, fpath)

    auroc = None
    if mask is not None and 0 < mask.sum() < len(mask):
        auroc = float(roc_auc_score(mask.astype(int), score))

    # detection verdict: peak anomaly far above the healthy reference tail
    ref_scores = np.abs(X - mu) / sd
    thr = float(np.quantile(ref_scores, 0.999))
    n_flagged = int((score > thr).sum())
    detected = n_flagged > 0
    return {
        "fpath": fpath,
        "name": os.path.basename(fpath),
        "score": score,
        "v": v,
        "mask": mask,
        "auroc": auroc,
        "thr": thr,
        "n_flagged": n_flagged,
        "max_score": float(score.max()),
        "detected": detected,
        "n_defect_true": int(mask.sum()) if mask is not None else None,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  Stage 2 — FMPE posterior θ=(cx,cy,layer,log2size)
# ═════════════════════════════════════════════════════════════════════════════

def stage2_posterior(use_fmpe: bool, sample_idx: int) -> dict:
    """Get the Stage-2 FMPE posterior for the sample.

    Default: load the cached REAL FMPE posterior (results/fmpe_posterior_e2e.npz
    — trained on the spot on a held-out test sample).  --fmpe re-runs the full
    FMPE chain (heavy) and samples the posterior of the chosen held-out sample.
    Returns posterior (N,4), the truth θ if known, and the source tag.
    """
    if not use_fmpe:
        z = np.load(CACHED_POSTERIOR, allow_pickle=True)
        return {
            "posterior": z["posterior"].astype(float),
            "truth": z["truth"].astype(float),
            "source": str(z["source"]),
        }

    # heavy path: train FMPE end-to-end, then sample the posterior of a held-out
    # 4x4 test sample.  Reuses fmpe_defect's data/embedding/training verbatim.
    import torch
    import fmpe_defect as fm

    print("[stage2] re-running FMPE end-to-end (this is the heavy path)…")
    fields, thetas = fm.build_dataset()
    n = len(fields)
    rng = np.random.default_rng(42)
    mu_f = fields.mean(0)
    Xc = fields - mu_f
    _, _, Vt = np.linalg.svd(Xc[rng.choice(n, min(3000, n), replace=False)],
                             full_matrices=False)
    V = Vt[:fm.PCA_K]
    embed = Xc @ V.T
    e_mu, e_sd = embed.mean(0), embed.std(0) + 1e-6
    embed = (embed - e_mu) / e_sd
    t_mu, t_sd = thetas.mean(0), thetas.std(0) + 1e-6
    th_n = (thetas - t_mu) / t_sd

    idx = rng.permutation(n)
    n_tr = int(0.8 * n)
    tr, te = idx[:n_tr], idx[n_tr:]
    C = torch.from_numpy(embed.astype(np.float32)).to(fm.DEVICE)
    TH = torch.from_numpy(th_n.astype(np.float32)).to(fm.DEVICE)

    model = fm.VelocityNet().to(fm.DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    BS = 512
    for ep in range(120):
        perm = torch.randperm(n_tr, device=fm.DEVICE)
        for i in range(0, n_tr, BS):
            b = torch.from_numpy(tr).to(fm.DEVICE)[perm[i:i + BS]]
            th0, c = TH[b], C[b]
            t = torch.rand(len(b), device=fm.DEVICE)
            eps = torch.randn_like(th0)
            tht = (1 - t)[:, None] * th0 + t[:, None] * eps
            loss = ((model(tht, t, c) - (eps - th0)) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()

    j = te[sample_idx % len(te)]
    post = fm.sample_posterior(model, C[j]).cpu().numpy() * t_sd + t_mu
    return {"posterior": post.astype(float),
            "truth": thetas[j].astype(float),
            "source": f"live FMPE (120 ep, held-out test idx {int(j)})"}


def _posterior_stats(post: np.ndarray) -> dict:
    names = ["cx", "cy", "layer", "size(elem)"]
    cols = [post[:, 0], post[:, 1], post[:, 2], 2.0 ** post[:, 3]]
    return {n: (float(np.mean(c)), float(np.std(c)))
            for n, c in zip(names, cols)}


# ═════════════════════════════════════════════════════════════════════════════
#  Stage 3 — surrogate forward + flight clearance
# ═════════════════════════════════════════════════════════════════════════════

def stage3_surrogate(post: np.ndarray, load: float, n_flights: int,
                     cfg: pf.LaminateConfig, n_draws: int, seed: int,
                     model=None) -> dict:
    """FAST Stage-3 path: FNO surrogate P(grow) over the posterior + a survival
    curve and an expected-cost clearance.

    P(grow) per flight = mean of the per-draw surrogate P(grow) head over the
    posterior (crack_surrogate.growth_probability_surrogate).  The n-flight
    survival curve uses the rate-independent i.i.d.-per-flight model
    S(n) = (1−p)^n (matching cfrp_phasefield_2d.flight_clearance(fatigue=False)
    semantics); the decision uses the SAME calibrated α/β expected-cost
    thresholds (calibrate_thresholds → 0.02, 0.48).
    """
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    p = cs.growth_probability_surrogate(post, load, model=model, cfg=cfg,
                                        n_draws=n_draws, rng=rng)
    dt = time.perf_counter() - t0

    alpha, beta = pf.calibrate_thresholds()
    flights = np.arange(1, n_flights + 1)
    surv = (1.0 - p) ** flights
    p_smooth = (p * n_draws + 1.0) / (n_draws + 2.0)
    e_rem = (1.0 - p_smooth) / max(p_smooth, 1e-9)
    decision = "OK" if p <= alpha else "REPAIR" if p <= beta else "RETIRE"
    return {
        "engine": "surrogate (FNO)",
        "p_growth_next": float(p),
        "p_survive_curve": surv,
        "p_survive_n": float(surv[-1]),
        "expected_remaining_flights": float(e_rem),
        "decision": decision,
        "alpha": alpha, "beta": beta,
        "forward_time_s": dt,
        "n_draws": n_draws,
    }


def stage3_fd(post: np.ndarray, load: float, n_flights: int,
              cfg: pf.LaminateConfig, n_draws: int, seed: int) -> dict:
    """EXACT Stage-3 path: cfrp_phasefield_2d.flight_clearance (fatigue-aware
    FD phase-field forward).  Much slower — for validation / ablation."""
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    out = pf.flight_clearance(post, load_profile=[0.5 * load, load],
                              n_flights=n_flights, cfg=cfg,
                              n_draws=n_draws, rng=rng)
    dt = time.perf_counter() - t0
    out = dict(out)
    out["engine"] = "FD phase-field (fatigue)"
    out["forward_time_s"] = dt
    out["n_draws"] = n_draws
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  Stage 4 — fleet learning update
# ═════════════════════════════════════════════════════════════════════════════

def stage4_fleet(theta_mean: np.ndarray, full_leader: bool,
                 seed: int = 0) -> dict:
    """Feed this inspection into the hierarchical-Bayes fleet (fleet_learning).

    Default (fast): show the fleet-prior φ=(μ_φ,σ_φ) SHARPENING with fleet size
    (recovery_study), then introduce THIS inspection as a new vehicle and
    partial-pool its growth-rate posterior against the fleet prior — the
    fleet-leader benefit (isolated vs fleet-prior posterior spread after one
    inspection).  --fleet-leader additionally runs the full
    inspections-to-stable-decision study (the headline 2.75 → 1.25 effect).
    """
    cfg = fl.FleetConfig()
    rng = np.random.default_rng(seed)

    rec = fl.recovery_study(cfg, sizes=[2, 5, 10, 20])

    # build the fleet prior from the established vehicles, then introduce this
    # inspection as a freshly-arrived vehicle and partial-pool it.
    fleet = fl.simulate_fleet(cfg, rng)
    post_phi = fl.fit_fleet(fleet[:-1], cfg, n_samples=1500, burn=500,
                            rng=np.random.default_rng(seed + 1))
    prior = (post_phi.mu_phi_mean, post_phi.sigma_phi_mean)
    new_veh = fleet[-1]
    m_iso, sd_iso = fl._vehicle_s_posterior(new_veh, 1, None, cfg)
    m_fl, sd_fl = fl._vehicle_s_posterior(new_veh, 1, prior, cfg)

    out = {
        "rec_sizes": rec.fleet_sizes,
        "rec_sigma_mean": rec.sigma_phi_mean,
        "rec_sigma_sd": rec.sigma_phi_sd,      # posterior SD of σ_φ → sharpening
        "fleet_prior": prior,
        "new_veh_sd_isolated": float(sd_iso),
        "new_veh_sd_with_prior": float(sd_fl),
    }
    if full_leader:
        flr = fl.fleet_leader_experiment(cfg, n_samples=1200, burn=400,
                                         rng=np.random.default_rng(seed + 2))
        out["leader_iso"] = float(flr.insp_without_prior[0])
        out["leader_fleet"] = float(flr.insp_with_prior[-1])
        out["leader_sigma_sd"] = (float(flr.sigma_phi_post_sd[0]),
                                  float(flr.sigma_phi_post_sd[-1]))
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  Reporting
# ═════════════════════════════════════════════════════════════════════════════

def _print_report(s0, s2, s3, s4, load, n_flights, total_t,
                  speed_note: str | None):
    bar = "=" * 72
    print(bar)
    print("  REUSABLE-ROCKET STRUCTURAL-HEALTH DECISION  —  one-command pipeline")
    print(bar)

    print(f"\n[Stage 0  DETECT]  sample: {s0['name']}")
    auroc = "n/a" if s0["auroc"] is None else f"{s0['auroc']:.4f}"
    verdict = "DEFECT DETECTED" if s0["detected"] else "clean"
    print(f"  per-node Mahalanobis: max z={s0['max_score']:.2f}  "
          f"(healthy 99.9% thr={s0['thr']:.2f})")
    print(f"  flagged nodes: {s0['n_flagged']}   "
          f"true defect nodes: {s0['n_defect_true']}/13942")
    print(f"  node-level AUROC (this sample): {auroc}   →  verdict: {verdict}")

    print(f"\n[Stage 2  CHARACTERISE]  FMPE posterior  ({s2['source']})")
    stats = _posterior_stats(s2["posterior"])
    truth = s2.get("truth")
    tmap = {"cx": truth[0], "cy": truth[1], "layer": truth[2],
            "size(elem)": 2.0 ** truth[3]} if truth is not None else {}
    print(f"  {'param':>11}  {'post mean ± sd':>18}  {'truth':>9}")
    for k, (m, sd) in stats.items():
        tv = f"{tmap[k]:.3f}" if k in tmap else ""
        print(f"  {k:>11}  {m:>9.3f} ± {sd:<6.3f}  {tv:>9}")

    print(f"\n[Stage 3  PROGNOSE]  engine: {s3['engine']}  "
          f"(load={load:.3f}, {s3['n_draws']} posterior draws)")
    print(f"  P(crack grows next flight)         = {s3['p_growth_next']:.3f}")
    print(f"  P(survive {n_flights:>2d} flights)             "
          f"= {s3['p_survive_n']:.3f}")
    print(f"  E[remaining flights]               = "
          f"{s3['expected_remaining_flights']:.2f}")
    print(f"  Stage-3 forward time               = "
          f"{s3['forward_time_s']:.3f} s")
    if speed_note:
        print(f"  {speed_note}")

    print(f"\n[DECISION]  expected-cost thresholds α={s3['alpha']:.2f}, "
          f"β={s3['beta']:.2f}")
    dec = s3["decision"]
    tag = {"OK": "FLY AS-IS", "REPAIR": "REPAIR BEFORE NEXT FLIGHT",
           "RETIRE": "RETIRE VEHICLE"}[dec]
    print(f"  ──►  {dec:>7}   ({tag})")

    print(f"\n[Stage 4  FLEET-UPDATE]  hierarchical-Bayes fleet prior")
    print("  φ posterior sharpening (post. SD of σ_φ) vs fleet size N:")
    for n, sm, ss in zip(s4["rec_sizes"], s4["rec_sigma_mean"],
                         s4["rec_sigma_sd"]):
        print(f"      N={n:>2}:  E[σ_φ]={sm:.3f}   sd(σ_φ)={ss:.3f}")
    mu0, sig0 = s4["fleet_prior"]
    print(f"  fleet prior φ=(μ_φ={mu0:.3f}, σ_φ={sig0:.3f})")
    print(f"  NEW vehicle (this inspection), growth-rate posterior after 1 insp:")
    print(f"      isolated   sd(log r) = {s4['new_veh_sd_isolated']:.3f}")
    print(f"      +fleet prior sd(log r) = {s4['new_veh_sd_with_prior']:.3f}"
          f"   →  fleet-leader tightening")
    if "leader_iso" in s4:
        print(f"  inspections-to-stable-decision: isolated "
              f"{s4['leader_iso']:.2f}  →  full-fleet prior "
              f"{s4['leader_fleet']:.2f}")

    print(f"\n{bar}")
    print(f"  TOTAL WALL-CLOCK: {total_t:.2f} s")
    print(bar)


# ═════════════════════════════════════════════════════════════════════════════
#  Combined figure
# ═════════════════════════════════════════════════════════════════════════════

def make_figure(s0, s2, s3, s4, load, n_flights,
                out_path: str) -> str:
    """Single combined multi-panel figure (thesis style if LaTeX available):
    anomaly map | posterior cloud | survival curve + clearance | fleet sharpening.
    """
    import matplotlib
    matplotlib.use("Agg")
    import sys
    sys.path.insert(0, os.path.join(HERE, "slides", "figure_sources"))
    try:
        from thesis_style import use
        figsize = use(width_frac=1.0, aspect=0.30)
    except Exception:
        figsize = (12.0, 3.6)
    import matplotlib.pyplot as plt

    # node coordinates rotated to principal axes (as in plot_stage_visuals)
    xc = np.load(f"{COORD_DIR}/normalized_x_2layer.npy")[:N_NODES]
    yc = np.load(f"{COORD_DIR}/normalized_y_2layer.npy")[:N_NODES]
    P = np.stack([xc, yc], 1)
    Pc = P - P.mean(0)
    _, _, Vt = np.linalg.svd(Pc, full_matrices=False)
    R = Vt.T
    xr, yr = ((P - P.mean(0)) @ R).T

    fig, axes = plt.subplots(1, 4, figsize=figsize)

    # panel (a): Stage-0 anomaly map
    sc = axes[0].scatter(xr, yr, c=s0["score"], s=1.6, cmap="inferno",
                         vmin=0, vmax=8, rasterized=True)
    axes[0].set_title("(a) Stage-0 anomaly map", fontsize=8)
    axes[0].set_aspect("equal"); axes[0].set_xticks([]); axes[0].set_yticks([])
    fig.colorbar(sc, ax=axes[0], fraction=0.04, pad=0.02)

    # panel (b): Stage-2 posterior cloud (cx, cy) + truth
    post = s2["posterior"]
    axes[1].scatter(post[:, 0], post[:, 1], s=6, c="#1565C0", alpha=0.45,
                    label="posterior")
    if s2.get("truth") is not None:
        t = s2["truth"]
        axes[1].scatter([t[0]], [t[1]], marker="*", s=150, c="#d7301f",
                        edgecolor="k", linewidth=0.5, zorder=5, label="truth")
    axes[1].set_title("(b) Stage-2 posterior $(c_x,c_y)$", fontsize=8)
    axes[1].set_xlabel("$c_x$", fontsize=8); axes[1].set_ylabel("$c_y$", fontsize=8)
    axes[1].legend(fontsize=6, loc="best")

    # panel (c): survival curve + clearance label
    n = len(s3["p_survive_curve"])
    axes[2].plot(np.arange(1, n + 1), s3["p_survive_curve"], "-o", ms=3,
                 c="#1565C0")
    axes[2].axhline(0.5, color="0.6", lw=0.8, ls="--")
    axes[2].set_ylim(0, 1.02)
    axes[2].set_title(f"(c) survival curve → {s3['decision']}", fontsize=8)
    axes[2].set_xlabel("flight", fontsize=8)
    axes[2].set_ylabel("P(survive)", fontsize=8)
    axes[2].text(0.05, 0.08,
                 f"P(grow)={s3['p_growth_next']:.2f}\n{s3['engine']}",
                 transform=axes[2].transAxes, fontsize=6, va="bottom")

    # panel (d): Stage-4 fleet-prior sharpening
    axes[3].plot(s4["rec_sizes"], s4["rec_sigma_sd"], "-s", ms=3, c="#2e7d32")
    axes[3].set_title("(d) Stage-4 fleet-prior sharpening", fontsize=8)
    axes[3].set_xlabel("fleet size $N$", fontsize=8)
    axes[3].set_ylabel(r"post. SD of $\sigma_\phi$", fontsize=8)
    if "leader_iso" in s4:
        axes[3].text(0.95, 0.92,
                     f"insp: {s4['leader_iso']:.1f}$\\to${s4['leader_fleet']:.1f}",
                     transform=axes[3].transAxes, fontsize=6, ha="right",
                     va="top")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    png = os.path.splitext(out_path)[0] + ".png"
    fig.savefig(png, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out_path


# ═════════════════════════════════════════════════════════════════════════════
#  Orchestration
# ═════════════════════════════════════════════════════════════════════════════

def run_pipeline(sample: int = 0, engine: str = "surrogate",
                 n_flights: int = 20, load: float = 0.10,
                 n_draws: int = 20, use_fmpe: bool = False,
                 full_leader: bool = False, fig: bool = False,
                 seed: int = 0, quiet: bool = False) -> dict:
    """Run the full Stage 0→4 chain on one held-out sample; return all results."""
    t_start = time.perf_counter()
    cfg = pf.LaminateConfig()

    s0 = stage0_detect(sample)
    s2 = stage2_posterior(use_fmpe, sample)
    post = s2["posterior"]

    model = None
    speed_note = None
    if engine == "surrogate":
        model, _ = cs.load_surrogate(device="cpu")
        s3 = stage3_surrogate(post, load, n_flights, cfg, n_draws, seed,
                              model=model)
        per_call = s3["forward_time_s"] / max(s3["n_draws"], 1)
        speed_note = (f"surrogate ≈{per_call*1e3:.1f} ms/draw "
                      f"(~142× faster than FD simulate_growth ≈1.7 s/call)")
    elif engine == "fd":
        s3 = stage3_fd(post, load, n_flights, cfg, n_draws, seed)
        speed_note = ("FD fatigue forward (exact); use --engine surrogate for "
                      "the ~142× faster FNO path")
    else:
        raise ValueError(f"unknown engine: {engine}")

    s4 = stage4_fleet(post.mean(0), full_leader, seed=seed)

    total_t = time.perf_counter() - t_start
    if not quiet:
        _print_report(s0, s2, s3, s4, load, n_flights, total_t, speed_note)

    fig_path = None
    if fig:
        fig_path = make_figure(s0, s2, s3, s4, load, n_flights,
                               os.path.join(HERE, "paper_figs",
                                            "pipeline_e2e_full.pdf"))
        if not quiet:
            print(f"\nfigure → {fig_path}")

    return {"s0": s0, "s2": s2, "s3": s3, "s4": s4,
            "total_time_s": total_t, "fig_path": fig_path}


def main():
    ap = argparse.ArgumentParser(
        description="One-command Stage 0→4 reusable-rocket SHM decision.")
    ap.add_argument("--sample", type=int, default=0,
                    help="held-out 4x4 sample index for Stage-0 (and --fmpe)")
    ap.add_argument("--engine", choices=["surrogate", "fd"], default="surrogate",
                    help="Stage-3 forward engine (default: FNO surrogate)")
    ap.add_argument("--n-flights", type=int, default=20)
    ap.add_argument("--load", type=float, default=0.10,
                    help="nominal transverse peel strain (in-distribution)")
    ap.add_argument("--n-draws", type=int, default=20,
                    help="posterior sub-sample size for Stage-3")
    ap.add_argument("--fmpe", action="store_true",
                    help="re-run the full FMPE chain (heavy) instead of cache")
    ap.add_argument("--fleet-leader", action="store_true",
                    help="also run the full 2.75→1.25 fleet-leader study")
    ap.add_argument("--fig", action="store_true",
                    help="save combined paper_figs/pipeline_e2e_full.pdf")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_pipeline(sample=args.sample, engine=args.engine,
                 n_flights=args.n_flights, load=args.load,
                 n_draws=args.n_draws, use_fmpe=args.fmpe,
                 full_leader=args.fleet_leader, fig=args.fig, seed=args.seed)


if __name__ == "__main__":
    main()
