#!/usr/bin/env python3
"""e2e_stage0to3_demo.py — full NDT pipeline on ONE held-out 4x4 sample.

  Stage 0  per-node Mahalanobis anomaly map (mahalanobis_anomaly.py logic,
           reference = 1x1 fields whose 3/13942-node defects are ~noise)
  Stage 1  size-class estimate from the hot-node count (the trained GNN
           classifier is not re-run here — heuristic stand-in, documented)
  Stage 2  FMPE posterior over theta = (cx, cy, layer, log2size):
           PREFERRED  saved posterior npz (results/fmpe_posterior*.npz)
           ELSE       train fmpe_defect's flow on the spot, HOLDING OUT the
                      demo sample (full budget on CUDA; reduced data +
                      ~30 epochs on CPU — posterior is real but coarse)
           FALLBACK   mock posterior centred on the Stage-0 centroid with
                      FMPE-like spreads (--mock or if data/torch missing)
  Stage 3  fatigue-aware flight clearance (Carrara-2020 phase-field fatigue,
           cfrp_phasefield_2d.py) over a 20-flight horizon at nominal and
           max-Q loads, fatigue ON vs OFF, expected-cost thresholds.

Prints the Stage 0->3 table and saves a 4-panel figure to
results/e2e_stage0to3.png.  CPU-only friendly; ~10-15 min end to end.

Run:  python3 scripts/e2e_stage0to3_demo.py [--mock] [--epochs N]
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from cfrp_phasefield_2d import LaminateConfig, flight_clearance  # noqa: E402

BASE = "/home/nishioka/CFRP/CFRP_hole/hole_data_inp"
DIR_1X1 = f"{BASE}/Defect_hole_1x1_Random_npy"
DIR_2X2 = f"{BASE}/Defect_hole_2x2_Region1_21_npy"
DIR_4X4 = f"{BASE}/Defect_hole_4x4_Region1_21_npy"
LBL_1X1 = f"{BASE}/Def1x1_19class_label"
LBL_2X2 = f"{BASE}/Def2x2_19class_label"
LBL_4X4 = f"{BASE}/Def4x4_19class_label"
COORD_DIR = f"{BASE}/basicdata_for_holegnn"
N_NODES = 13942
RE_META = re.compile(r"_L(?P<L>\d+)_.*_H(?P<H>\d+)_W")

OUT_PNG = os.path.join(REPO, "results", "e2e_stage0to3.png")
OUT_NPZ = os.path.join(REPO, "results", "fmpe_posterior_e2e.npz")

# Stage-3 setup (normalised load scale of cfrp_phasefield_2d, see docstring)
LOAD_CASES = {"nominal ascent": 0.06, "max-Q gust": 0.10}
N_FLIGHTS = 20
N_DRAWS = 10
SEED = 42


def load_field(fpath: str) -> np.ndarray:
    v = np.load(fpath).astype(np.float32)[:N_NODES]
    return (v - v.mean()) / (v.std() + 1e-8)


def load_mask(label_dir: str, fpath: str):
    base = os.path.splitext(os.path.basename(fpath))[0]
    for cand in (f"{base}.npy", f"{base}_19label.npy"):
        lf = os.path.join(label_dir, cand)
        if os.path.exists(lf):
            lbl = np.load(lf)
            return lbl[:, 1:].sum(axis=1) > 0
    return None


# ───────────────────────── Stage 0: Mahalanobis ─────────────────────────────
def stage0_maha(test_field: np.ndarray, x_c: np.ndarray, y_c: np.ndarray,
                n_ref: int = 400) -> dict:
    """Per-node Mahalanobis map (mahalanobis_anomaly.py 'method 1')."""
    import glob
    rng = np.random.default_rng(SEED)
    files = sorted(glob.glob(os.path.join(DIR_1X1, "*.npy")))
    idx = rng.choice(len(files), min(n_ref, len(files)), replace=False)
    X = np.stack([load_field(files[i]) for i in idx])
    mu, sd = X.mean(0), X.std(0) + 1e-6

    z = np.abs(test_field - mu) / sd
    hot = z > 4.0
    if hot.sum() < 5:                       # fall back to top-0.5% nodes
        hot = z > np.percentile(z, 99.5)
    cx = float(x_c[hot].mean())
    cy = float(y_c[hot].mean())
    return {"z": z, "hot": hot, "cx": cx, "cy": cy,
            "z_max": float(z.max()), "n_hot": int(hot.sum()),
            "mu": mu, "sd": sd}


# ───────────────────────── Stage 1: size-class heuristic ────────────────────
def stage1_sizeclass(n_hot: int, maha: dict) -> float:
    """log2(defect size in elements) from the hot-node count.

    Hot-node count scales ~ defect area ~ (2**log2size)^2; calibrated against
    held-out 1x1 samples (H=1 → log2size=0).  Stand-in for the Stage-1 GNN
    classifier (not re-run here); only used by the MOCK Stage-2 path.
    """
    import glob
    files = sorted(glob.glob(os.path.join(DIR_1X1, "*.npy")))[-8:]
    counts = []
    for f in files:
        z = np.abs(load_field(f) - maha["mu"]) / maha["sd"]
        counts.append(max((z > 4.0).sum(), 1))
    c1 = float(np.median(counts))
    return float(np.clip(0.5 * np.log2(max(n_hot, 1) / c1), -0.5, 2.5))


# ───────────────────────── Stage 2: FMPE posterior ──────────────────────────
def stage2_posterior(test_file: str, test_field: np.ndarray, maha: dict,
                     size_hint: float, mock: bool, epochs: int | None
                     ) -> tuple[np.ndarray, str]:
    """Posterior (N,4) over theta and a provenance string."""
    import glob
    hits = sorted(glob.glob(os.path.join(REPO, "results",
                                         "fmpe_posterior*.npz")))
    hits = [h for h in hits if os.path.abspath(h) != os.path.abspath(OUT_NPZ)]
    if hits and not mock:
        with np.load(hits[0]) as z:
            arr = z[z.files[0]]
        if arr.ndim == 2 and arr.shape[1] == 4:
            return arr.astype(float), f"saved FMPE posterior ({hits[0]})"

    if not mock:
        try:
            return _train_fmpe_and_sample(test_file, test_field, epochs)
        except Exception as exc:                          # noqa: BLE001
            print(f"[stage2] FMPE training failed ({exc!r}) -> mock fallback")

    rng = np.random.default_rng(SEED)
    n = 256
    post = np.column_stack([
        rng.normal(maha["cx"], 0.04, n),     # cx  <- Stage-0 centroid
        rng.normal(maha["cy"], 0.04, n),     # cy  <- Stage-0 centroid
        rng.normal(9.0, 2.5, n),             # layer: broad mid-laminate prior
        rng.normal(size_hint, 0.35, n),      # log2size <- Stage-1 heuristic
    ])
    return post, ("MOCK posterior (maha centroid + Stage-1 size heuristic; "
                  "spreads ~ FMPE test RMSE)")


def _train_fmpe_and_sample(test_file: str, test_field: np.ndarray,
                           epochs: int | None) -> tuple[np.ndarray, str]:
    """Train fmpe_defect's conditional flow from scratch, holding out
    `test_file`, then sample the posterior for it.

    On CPU the budget is reduced (n<=1200 per set, ~30 epochs, SVD on 1500
    rows): the posterior is REAL but coarser than the full GPU run.
    """
    import glob
    import torch
    from fmpe_defect import VelocityNet, sample_posterior

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_max = 4000 if device == "cuda" else 1200
    epochs = epochs or (200 if device == "cuda" else 30)
    rng = np.random.default_rng(SEED)

    x_c = np.load(f"{COORD_DIR}/normalized_x_2layer.npy")[:N_NODES]
    y_c = np.load(f"{COORD_DIR}/normalized_y_2layer.npy")[:N_NODES]

    fields, thetas = [], []
    for data_dir, label_dir in ((DIR_1X1, LBL_1X1), (DIR_2X2, LBL_2X2),
                                (DIR_4X4, LBL_4X4)):
        files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        files = [f for f in files
                 if os.path.abspath(f) != os.path.abspath(test_file)]
        idx = rng.choice(len(files), min(n_max, len(files)), replace=False)
        kept = 0
        for i in idx:
            f = files[i]
            m = RE_META.search(os.path.basename(f))
            mask = load_mask(label_dir, f)
            if m is None or mask is None or mask.sum() == 0:
                continue
            fields.append(load_field(f))
            thetas.append([x_c[mask].mean(), y_c[mask].mean(),
                           float(m.group("L")), np.log2(float(m.group("H")))])
            kept += 1
        print(f"  [stage2] {os.path.basename(data_dir)}: kept {kept}")
    fields = np.stack(fields)
    thetas = np.array(thetas, dtype=np.float32)
    n = len(fields)

    # PCA(64) embedding of the deviation field (as in fmpe_defect.main)
    pca_k = 64
    mu_f = fields.mean(0)
    Xc = fields - mu_f
    sub = rng.choice(n, min(1500, n), replace=False)
    _, _, Vt = np.linalg.svd(Xc[sub], full_matrices=False)
    V = Vt[:pca_k]
    embed = Xc @ V.T
    e_mu, e_sd = embed.mean(0), embed.std(0) + 1e-6
    embed = (embed - e_mu) / e_sd
    t_mu, t_sd = thetas.mean(0), thetas.std(0) + 1e-6
    th_n = (thetas - t_mu) / t_sd

    C = torch.from_numpy(embed.astype(np.float32)).to(device)
    TH = torch.from_numpy(th_n.astype(np.float32)).to(device)
    model = VelocityNet(cond_dim=pca_k).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    print(f"  [stage2] training FMPE flow: {n} pairs, {epochs} epochs "
          f"on {device}")
    bs = 512
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        tot, nb = 0.0, 0
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            th0, c = TH[b], C[b]
            t = torch.rand(len(b), device=device)
            eps = torch.randn_like(th0)
            tht = (1 - t)[:, None] * th0 + t[:, None] * eps
            loss = ((model(tht, t, c) - (eps - th0)) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        if (ep + 1) % 10 == 0:
            print(f"    epoch {ep+1}/{epochs} loss={tot/nb:.4f}", flush=True)

    # condition on the held-out field, sample the posterior
    import fmpe_defect
    fmpe_defect.DEVICE = device       # sample_posterior reads module global
    c_test = (test_field - mu_f) @ V.T
    c_test = (c_test - e_mu) / e_sd
    c_t = torch.from_numpy(c_test.astype(np.float32)).to(device)
    post = sample_posterior(model, c_t, n_samples=256).cpu().numpy()
    post = post * t_sd + t_mu
    return post.astype(float), (f"REAL FMPE posterior (trained on the spot: "
                                f"{n} pairs, {epochs} epochs, {device}, "
                                f"held-out test sample)")


# ───────────────────────── Stage 3 + report ─────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true",
                    help="skip FMPE training, use the mock posterior")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    import glob
    t_start = time.perf_counter()

    # held-out 4x4 test sample (deterministic pick with a valid label)
    files = sorted(glob.glob(os.path.join(DIR_4X4, "*.npy")))
    rng = np.random.default_rng(7)
    test_file = None
    for i in rng.permutation(len(files)):
        if (RE_META.search(os.path.basename(files[i]))
                and load_mask(LBL_4X4, files[i]) is not None):
            test_file = files[i]
            break
    assert test_file is not None, "no labelled 4x4 sample found"

    x_c = np.load(f"{COORD_DIR}/normalized_x_2layer.npy")[:N_NODES]
    y_c = np.load(f"{COORD_DIR}/normalized_y_2layer.npy")[:N_NODES]
    test_field = load_field(test_file)
    mask_true = load_mask(LBL_4X4, test_file)
    m = RE_META.search(os.path.basename(test_file))
    truth = np.array([x_c[mask_true].mean(), y_c[mask_true].mean(),
                      float(m.group("L")), np.log2(float(m.group("H")))])

    print("=" * 76)
    print("End-to-end Stage 0->3 demo: NDT field -> anomaly -> posterior -> "
          "clearance")
    print("=" * 76)
    print(f"held-out sample : {os.path.basename(test_file)}")
    print(f"ground truth    : cx={truth[0]:.3f} cy={truth[1]:.3f} "
          f"layer={truth[2]:.0f} log2size={truth[3]:.0f} (4x4)")

    # Stage 0
    maha = stage0_maha(test_field, x_c, y_c)
    print(f"\n[Stage 0] Mahalanobis screening (400 reference 1x1 fields)")
    print(f"  z_max={maha['z_max']:.1f}, hot nodes (z>4): {maha['n_hot']}, "
          f"centroid=({maha['cx']:.3f}, {maha['cy']:.3f}) "
          f"[truth ({truth[0]:.3f}, {truth[1]:.3f})]")

    # Stage 1
    size_hint = stage1_sizeclass(maha["n_hot"], maha)
    print(f"[Stage 1] size-class heuristic: log2size~{size_hint:.2f} "
          f"(GNN classifier not re-run; heuristic only feeds the MOCK path)")

    # Stage 2
    posterior, source = stage2_posterior(test_file, test_field, maha,
                                         size_hint, args.mock, args.epochs)
    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
    np.savez(OUT_NPZ, posterior=posterior, truth=truth,
             source=np.array(source))
    pm = posterior.mean(0)
    ps = posterior.std(0)
    print(f"[Stage 2] {source}")
    print(f"  posterior mean+-sd: cx={pm[0]:.3f}+-{ps[0]:.3f} "
          f"cy={pm[1]:.3f}+-{ps[1]:.3f} layer={pm[2]:.1f}+-{ps[2]:.1f} "
          f"log2size={pm[3]:.2f}+-{ps[3]:.2f}")
    print(f"  saved: {OUT_NPZ}")

    # Stage 3
    cfg = LaminateConfig()
    print(f"\n[Stage 3] fatigue-aware flight clearance "
          f"({N_FLIGHTS}-flight horizon, K={cfg.cycles_per_flight} "
          f"cycles/flight, {N_DRAWS} posterior draws, "
          f"thresholds alpha/beta from expected cost 100:1)")
    header = (f"{'load case':>15} {'load':>6} {'fatigue':>8} "
              f"{'P(grow next)':>13} {'P(survive %d)' % N_FLIGHTS:>14} "
              f"{'E[flights]':>11} {'decision':>9}")
    print(header)
    print("-" * len(header))
    results = {}
    for name, load in LOAD_CASES.items():
        for fat in (True, False):
            t0 = time.perf_counter()
            out = flight_clearance(posterior, load_profile=[load],
                                   n_flights=N_FLIGHTS, fatigue=fat,
                                   cfg=cfg, n_draws=N_DRAWS,
                                   rng=np.random.default_rng(SEED))
            dt = time.perf_counter() - t0
            results[(name, fat)] = out
            print(f"{name:>15} {load:>6.3f} {'ON' if fat else 'off':>8} "
                  f"{out['p_growth_next']:>13.2f} {out['p_survive_n']:>14.2f} "
                  f"{out['expected_remaining_flights']:>11.1f} "
                  f"{out['decision']:>9}   [{dt:.0f}s]")
    a, b = (results[next(iter(results))][k] for k in ("alpha", "beta"))
    print(f"thresholds: alpha={a:.3f} (OK), beta={b:.3f} (REPAIR), "
          f"else RETIRE")
    print("\nNote: normalised load proxy (no absolute strains); fatigue "
          "cycle counts are\nrelative severity (alpha_T not yet anchored to "
          "S-N data).")

    _figure(maha, posterior, truth, results, x_c, y_c, source)
    print(f"\nfigure: {OUT_PNG}")
    print(f"total {time.perf_counter() - t_start:.0f}s")


def _figure(maha, posterior, truth, results, x_c, y_c, source) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    dec_color = {"OK": "#2a9d3a", "REPAIR": "#e9a800", "RETIRE": "#d62728"}
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))

    # (a) Stage-0 anomaly map
    ax = axes[0]
    z = np.clip(maha["z"], 0, np.percentile(maha["z"], 99.9))
    order = np.argsort(z)
    sc = ax.scatter(x_c[order], y_c[order], c=z[order], s=3, cmap="magma")
    ax.scatter([truth[0]], [truth[1]], marker="*", s=180, ec="cyan",
               fc="none", lw=1.5, label="true centroid")
    ax.scatter([maha["cx"]], [maha["cy"]], marker="x", s=90, c="lime",
               label="maha centroid")
    fig.colorbar(sc, ax=ax, label="|z|")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title(f"(a) Stage-0 Mahalanobis map\n"
                 f"z_max={maha['z_max']:.0f}, hot={maha['n_hot']} nodes",
                 fontsize=9)

    # (b) Stage-2 posterior
    ax = axes[1]
    sc = ax.scatter(posterior[:, 0], posterior[:, 1], c=posterior[:, 3],
                    s=12, alpha=0.6, cmap="viridis")
    ax.scatter([truth[0]], [truth[1]], marker="*", s=220, c="red",
               label="truth")
    fig.colorbar(sc, ax=ax, label="log2size")
    ax.set_xlabel("cx"); ax.set_ylabel("cy")
    ax.legend(fontsize=7)
    kind = "REAL FMPE" if "REAL" in source else \
        ("saved FMPE" if "saved" in source else "MOCK")
    ax.set_title(f"(b) Stage-2 posterior ({kind})\n"
                 f"layer={posterior[:,2].mean():.1f}+-"
                 f"{posterior[:,2].std():.1f} (true {truth[2]:.0f})",
                 fontsize=9)

    # (c) survival curves
    ax = axes[2]
    styles = {True: "-", False: "--"}
    colors = {"nominal ascent": "#1f77b4", "max-Q gust": "#d62728"}
    for (name, fat), out in results.items():
        ax.plot(np.arange(1, len(out["p_survive_curve"]) + 1),
                out["p_survive_curve"], styles[fat], color=colors[name],
                label=f"{name}, fatigue {'ON' if fat else 'off'}")
    ax.set_xlabel("flight n"); ax.set_ylabel("P(survive n)")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=7)
    ax.set_title("(c) Stage-3 survival vs flights\n"
                 "(fatigue decays even at constant load)", fontsize=9)

    # (d) decision
    ax = axes[3]
    keys = list(results)
    ps = [results[k]["p_growth_next"] for k in keys]
    decs = [results[k]["decision"] for k in keys]
    labels = [f"{n.split()[0]}\nfat {'ON' if f else 'off'}" for n, f in keys]
    bars = ax.bar(range(len(keys)), ps, color=[dec_color[d] for d in decs],
                  width=0.6)
    for bar, d, p in zip(bars, decs, ps):
        ax.text(bar.get_x() + bar.get_width() / 2, p + 0.03, d, ha="center",
                fontsize=8, fontweight="bold", color=dec_color[d])
    out0 = results[keys[0]]
    ax.axhline(out0["alpha"], ls=":", c="gray", lw=1)
    ax.axhline(out0["beta"], ls="--", c="gray", lw=1)
    ax.text(len(keys) - 0.4, out0["alpha"] + 0.01,
            f"alpha={out0['alpha']:.2f}", fontsize=7, color="gray")
    ax.text(len(keys) - 0.4, out0["beta"] + 0.01,
            f"beta={out0['beta']:.2f}", fontsize=7, color="gray")
    ax.set_xticks(range(len(keys))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("P(growth next flight)")
    ax.set_title("(d) expected-cost decision\n(loss:repair = 100:1)",
                 fontsize=9)

    fig.suptitle("Stage 0->3: anomaly screening -> amortised posterior -> "
                 "fatigue phase-field prognosis -> clearance", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
