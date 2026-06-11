"""
crack_surrogate.py — Fourier-Neural-Operator surrogate for the Stage-3 CFRP
phase-field CRACK-GROWTH forward model (`cfrp_phasefield_2d.simulate_growth`).

Operator learned
----------------
    G : (d0 seeded-damage field, load) ──► (d_final damage field,
                                            rel_growth, P(grow))

One FNO inference (~ms on CPU) replaces one ~1.7 s FD `simulate_growth`
call, enabling fleet-scale Monte-Carlo over the Stage-2 FMPE posterior and a
fast forward map inside differentiable inversion (Concept B).

Pipeline position (CFRP NDT, WCCM2026 / thesis):
  Stage-2 FMPE posterior θ=(cx,cy,layer,log2size) → seed_defect → d0
  → Stage-3 FD phase-field growth (slow)   ← THIS surrogate accelerates it.

Architecture provenance
------------------------
The FNO blocks (`SpectralConv2d`, `FNOBlock`) and the field-surrogate style
are copied/adapted from
  /home/nishioka/git/wafer-proc-sim/ml/surrogate_fno.py   (SiC static stress)
(wafer-proc-sim is read-only reference; this module is self-contained).
Adaptations for crack growth:
  * input is a FIELD (d0) stacked with a broadcast scalar `load` + coord grid,
    not just broadcast scalars → 4 input channels (d0, load, x, y);
  * a dual head: a per-pixel field head (d_final) AND a global scalar head
    (rel_growth regression + P(grow) logit) pooled from the FNO features.

Honest scope (see `simulate_growth` docstring + module-level limitations):
  coarse FD grid (nx=60, ny=48), synthetic in-distribution defects/loads;
  the FD rel_growth has a SHARP super-critical transition (≈0.6 → tens once
  the crack spans a ply), which the smooth surrogate regresses in log-space
  but cannot reproduce exactly — P(grow) (the decision-relevant output) is the
  robust target, raw rel_growth magnitude above the threshold is not.

CPU torch; trains in < 1 min on a few hundred coarse samples.
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

import numpy as np

# torch is imported lazily inside functions that need it so that the data
# generator / pure-numpy helpers run on a bare box without torch.

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "results", "crack_surrogate")
DATA_NPZ = os.path.join(DATA_DIR, "data.npz")
MODEL_PT = os.path.join(DATA_DIR, "fno_crack.pt")

# ── default sampling ranges (the "in-distribution" envelope) ──────────────────
LOAD_LO, LOAD_HI = 0.04, 0.14        # sub- to super-critical transverse strain
LOG2SIZE_LO, LOG2SIZE_HI = 0.0, 2.0  # defect size 2**log2size elements (1–4)
CX_LO, CX_HI = 0.2, 0.8              # keep seed away from grips/edges
LAYER_LO, LAYER_HI = 3.0, 15.0       # interior plies (n_layers_data = 19)


# ═════════════════════════════════════════════════════════════════════════════
#  Deliverable 1.1 — Data generation (the slow part; multiprocessing)
# ═════════════════════════════════════════════════════════════════════════════

def _sim_one(args) -> dict:
    """Worker: seed a defect, run ONE FD simulate_growth, return arrays.

    Top-level (picklable) so it works with multiprocessing.Pool.
    """
    from cfrp_phasefield_2d import LaminateConfig, seed_defect, simulate_growth

    cx, cy, layer, log2size, load, nx, ny = args
    cfg = LaminateConfig(nx=int(nx), ny=int(ny))
    d0 = seed_defect(cx, cy, layer, log2size, cfg)
    res = simulate_growth(d0, float(load), cfg)
    return {
        "theta": np.array([cx, cy, layer, log2size], dtype=np.float32),
        "load": np.float32(load),
        "d0": d0.astype(np.float32),
        "d_final": res.d_final.astype(np.float32),
        "rel_growth": np.float32(res.rel_growth),
        "grown": np.bool_(res.grown),
        "area0": np.float32(res.area0),
    }


def generate_data(n_samples: int = 800, nx: int = 60, ny: int = 48,
                  seed: int = 0, n_workers: int | None = None,
                  out_path: str = DATA_NPZ) -> dict:
    """Sample (defect, load) → run FD simulate_growth → cache to .npz.

    Loads are drawn on a grid spanning sub- to super-critical so the held-out
    set probes the whole growth/no-growth transition.  Defects (cx, layer,
    log2size) are sampled uniformly in the in-distribution envelope; cy only
    fixes the section plane (unused by the 2D geometry) so it is sampled but
    carried for API compatibility.

    Parallelised with multiprocessing (default: all cores).  On frontale use
    PY=/home/nishioka/IKM_Hiwi/.venv_jax/bin/python3 under qsub; locally it
    just uses os.cpu_count().
    """
    import multiprocessing as mp

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rng = np.random.default_rng(seed)

    # load grid: evenly tile [LOAD_LO, LOAD_HI] then jitter so every draw is
    # unique but the transition band is densely covered.
    n_load_levels = 11
    load_grid = np.linspace(LOAD_LO, LOAD_HI, n_load_levels)
    loads = rng.choice(load_grid, size=n_samples)
    loads = np.clip(loads + rng.normal(0, 0.004, n_samples), LOAD_LO, LOAD_HI)

    cx = rng.uniform(CX_LO, CX_HI, n_samples)
    cy = rng.uniform(0.0, 1.0, n_samples)              # section plane only
    layer = rng.uniform(LAYER_LO, LAYER_HI, n_samples)
    log2size = rng.uniform(LOG2SIZE_LO, LOG2SIZE_HI, n_samples)

    jobs = [(cx[i], cy[i], layer[i], log2size[i], loads[i], nx, ny)
            for i in range(n_samples)]

    n_workers = n_workers or os.cpu_count() or 1
    t0 = time.perf_counter()
    print(f"[data] {n_samples} FD sims, grid {nx}x{ny}, {n_workers} workers ...")
    if n_workers > 1:
        with mp.Pool(n_workers) as pool:
            recs = pool.map(_sim_one, jobs, chunksize=4)
    else:
        recs = [_sim_one(j) for j in jobs]
    dt = time.perf_counter() - t0

    data = {
        "theta": np.stack([r["theta"] for r in recs]),
        "load": np.array([r["load"] for r in recs], dtype=np.float32),
        "d0": np.stack([r["d0"] for r in recs]),
        "d_final": np.stack([r["d_final"] for r in recs]),
        "rel_growth": np.array([r["rel_growth"] for r in recs], dtype=np.float32),
        "grown": np.array([r["grown"] for r in recs], dtype=bool),
        "area0": np.array([r["area0"] for r in recs], dtype=np.float32),
        "nx": np.int64(nx), "ny": np.int64(ny),
    }
    np.savez_compressed(out_path, **data)
    frac = data["grown"].mean()
    print(f"[data] done in {dt:.1f}s ({dt/n_samples:.2f}s/sim); "
          f"grown fraction = {frac:.2f}; cached → {out_path}")
    return data


def load_data(path: str = DATA_NPZ) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ═════════════════════════════════════════════════════════════════════════════
#  Deliverable 1.2 — FNO surrogate (field + scalar heads)
#  Spectral blocks copied from wafer-proc-sim/ml/surrogate_fno.py (provenance).
# ═════════════════════════════════════════════════════════════════════════════

def _torch():
    import torch
    return torch


class SpectralConv2d:  # placeholder replaced below once torch.nn is available
    pass


def _build_modules():
    """Define the torch nn.Modules. Done lazily so importing this file without
    torch (e.g. for data generation on a bare node) does not fail."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class SpectralConv2d(nn.Module):
        """2D Fourier layer (copied from wafer-proc-sim surrogate_fno.py)."""

        def __init__(self, in_ch, out_ch, modes1, modes2):
            super().__init__()
            self.in_ch, self.out_ch = in_ch, out_ch
            self.modes1, self.modes2 = modes1, modes2
            scale = 1 / (in_ch * out_ch)
            self.weights1 = nn.Parameter(
                scale * torch.randn(in_ch, out_ch, modes1, modes2,
                                    dtype=torch.cfloat))
            self.weights2 = nn.Parameter(
                scale * torch.randn(in_ch, out_ch, modes1, modes2,
                                    dtype=torch.cfloat))

        @staticmethod
        def _compl_mul2d(x, w):
            return torch.einsum("bixy,ioxy->boxy", x, w)

        def forward(self, x):
            B, _, H, W = x.shape
            x_ft = torch.fft.rfft2(x, norm="ortho")
            out_ft = torch.zeros(B, self.out_ch, H, W // 2 + 1,
                                 dtype=torch.cfloat, device=x.device)
            m1 = min(self.modes1, H)
            m2 = min(self.modes2, W // 2 + 1)
            out_ft[:, :, :m1, :m2] = self._compl_mul2d(
                x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2])
            out_ft[:, :, -m1:, :m2] = self._compl_mul2d(
                x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2])
            return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")

    class FNOBlock(nn.Module):
        """Spectral conv + 1×1 bypass + GELU (from surrogate_fno.py)."""

        def __init__(self, channels, modes1, modes2):
            super().__init__()
            self.spectral = SpectralConv2d(channels, channels, modes1, modes2)
            self.bypass = nn.Conv2d(channels, channels, 1)

        def forward(self, x):
            return F.gelu(self.spectral(x) + self.bypass(x))

    class CrackFNO(nn.Module):
        """FNO operator G:(d0, load) → (d_final field, rel_growth, P(grow)).

        Input channels (4): [d0 field, load broadcast, x grid, y grid].
        Field head → d_final in [0,1] (sigmoid).  A global average pool of the
        last FNO feature map feeds a small MLP scalar head producing
        (log1p rel_growth regression, P(grow) logit).
        """

        def __init__(self, modes1=12, modes2=12, width=32, n_layers=4,
                     grid_h=48, grid_w=60):
            super().__init__()
            self.grid_h, self.grid_w = grid_h, grid_w
            self.input_proj = nn.Conv2d(4, width, 1)
            self.blocks = nn.ModuleList(
                [FNOBlock(width, modes1, modes2) for _ in range(n_layers)])
            self.field_head = nn.Sequential(
                nn.Conv2d(width, 128, 1), nn.GELU(), nn.Conv2d(128, 1, 1))
            self.scalar_head = nn.Sequential(
                nn.Linear(width, 64), nn.GELU(),
                nn.Linear(64, 2))   # [log1p_rel_growth, grow_logit]

        def _grid(self, B, device):
            xs = torch.linspace(0, 1, self.grid_w, device=device)
            ys = torch.linspace(0, 1, self.grid_h, device=device)
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            g = torch.stack([gx, gy], 0)              # (2,H,W)
            return g.unsqueeze(0).expand(B, -1, -1, -1)

        def forward(self, d0, load):
            """d0: (B,1,H,W);  load: (B,1) → returns dict of outputs."""
            B = d0.shape[0]
            H, W = self.grid_h, self.grid_w
            lp = load.view(B, 1, 1, 1).expand(B, 1, H, W)
            g = self._grid(B, d0.device)
            x = torch.cat([d0, lp, g], dim=1)         # (B,4,H,W)
            x = self.input_proj(x)
            for blk in self.blocks:
                x = blk(x)
            field = torch.sigmoid(self.field_head(x))  # (B,1,H,W) in [0,1]
            pooled = x.mean(dim=(-1, -2))              # (B,width)
            s = self.scalar_head(pooled)               # (B,2)
            return {
                "d_final": field,
                "log1p_rel": s[:, 0],
                "grow_logit": s[:, 1],
            }

    return CrackFNO


# ── training ──────────────────────────────────────────────────────────────────

@dataclass
class SplitData:
    """Train/test tensors + the raw held-out numpy for evaluation."""
    Xtr_d0: object; Xtr_load: object
    Ytr_field: object; Ytr_logrel: object; Ytr_grow: object
    Xte_d0: object; Xte_load: object
    Yte_field: object; Yte_logrel: object; Yte_grow: object
    raw_test: dict


def _prepare(data: dict, test_frac: float = 0.2, seed: int = 0):
    import torch
    n = len(data["load"])
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_te = max(1, int(round(test_frac * n)))
    te, tr = idx[:n_te], idx[n_te:]

    def to(arr, i):
        return torch.tensor(arr[i])

    def fields(i):
        return torch.tensor(data["d_final"][i]).unsqueeze(1)  # (N,1,H,W)

    def d0(i):
        return torch.tensor(data["d0"][i]).unsqueeze(1)

    logrel = np.log1p(np.clip(data["rel_growth"], 0, None)).astype(np.float32)
    grow = data["grown"].astype(np.float32)
    load = data["load"].astype(np.float32)

    raw_test = {k: data[k][te] for k in
                ("theta", "load", "d0", "d_final", "rel_growth", "grown",
                 "area0")}
    return SplitData(
        d0(tr), to(load, tr).unsqueeze(1), fields(tr), to(logrel, tr), to(grow, tr),
        d0(te), to(load, te).unsqueeze(1), fields(te), to(logrel, te), to(grow, te),
        raw_test), tr, te


def train(data: dict | None = None, epochs: int = 300, lr: float = 2e-3,
          batch: int = 16, width: int = 32, modes: int = 12, n_layers: int = 4,
          test_frac: float = 0.2, seed: int = 0, device: str | None = None,
          out_path: str = MODEL_PT, verbose: bool = True) -> dict:
    """Train the FNO surrogate; save checkpoint; return held-out metrics."""
    import torch
    import torch.nn.functional as F

    data = data or load_data()
    nx, ny = int(data["nx"]), int(data["ny"])
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    split, tr_idx, te_idx = _prepare(data, test_frac, seed)
    CrackFNO = _build_modules()
    model = CrackFNO(modes1=modes, modes2=modes, width=width,
                     n_layers=n_layers, grid_h=ny, grid_w=nx).to(device)

    Xd0 = split.Xtr_d0.to(device); Xl = split.Xtr_load.to(device)
    Yf = split.Ytr_field.to(device); Yr = split.Ytr_logrel.to(device)
    Yg = split.Ytr_grow.to(device)
    n = Xd0.shape[0]

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bce = torch.nn.BCEWithLogitsLoss()

    # log1p_rel normalisation for a balanced regression loss
    rel_mu, rel_sd = float(Yr.mean()), float(Yr.std()) + 1e-6

    if verbose:
        npar = sum(p.numel() for p in model.parameters())
        print(f"[train] {device}  n_train={n} n_test={len(te_idx)}  "
              f"{npar:,} params  grid {nx}x{ny}")

    model.train()
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for b in range(0, n, batch):
            bi = perm[b:b + batch]
            out = model(Xd0[bi], Xl[bi])
            l_field = F.mse_loss(out["d_final"], Yf[bi])
            l_rel = F.mse_loss((out["log1p_rel"] - rel_mu) / rel_sd,
                               (Yr[bi] - rel_mu) / rel_sd)
            l_grow = bce(out["grow_logit"], Yg[bi])
            loss = l_field + 0.3 * l_rel + 0.5 * l_grow
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(bi)
        sched.step()
        if verbose and (ep % 50 == 0 or ep == 1):
            print(f"  ep {ep:4d}/{epochs}  loss={tot/n:.5f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}")

    torch.save({
        "model_state": model.state_dict(),
        "modes": modes, "width": width, "n_layers": n_layers,
        "grid_h": ny, "grid_w": nx,
        "rel_mu": rel_mu, "rel_sd": rel_sd,
        "load_lo": LOAD_LO, "load_hi": LOAD_HI,
    }, out_path)
    if verbose:
        print(f"[train] checkpoint → {out_path}")

    metrics = evaluate(model, split, device=device)
    metrics["test_idx"] = te_idx
    return metrics


# ── load a trained surrogate for inference ────────────────────────────────────

def load_surrogate(path: str = MODEL_PT, device: str | None = None):
    import torch
    device = device or "cpu"
    ckpt = torch.load(path, map_location=device, weights_only=False)
    CrackFNO = _build_modules()
    model = CrackFNO(modes1=ckpt["modes"], modes2=ckpt["modes"],
                     width=ckpt["width"], n_layers=ckpt["n_layers"],
                     grid_h=ckpt["grid_h"], grid_w=ckpt["grid_w"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def predict(model, d0: np.ndarray, load: float, device: str = "cpu") -> dict:
    """Single-sample inference: d0 (ny,nx) + scalar load → surrogate outputs."""
    import torch
    model.eval()
    with torch.no_grad():
        x = torch.tensor(d0[None, None], dtype=torch.float32, device=device)
        l = torch.tensor([[load]], dtype=torch.float32, device=device)
        out = model(x, l)
        return {
            "d_final": out["d_final"][0, 0].cpu().numpy(),
            "rel_growth": float(np.expm1(out["log1p_rel"][0].cpu().numpy())),
            "p_grow": float(torch.sigmoid(out["grow_logit"][0]).cpu().numpy()),
        }


def predict_batch(model, d0: np.ndarray, load: np.ndarray,
                  device: str = "cpu") -> dict:
    """Vectorised inference over a batch of (d0, load)."""
    import torch
    model.eval()
    with torch.no_grad():
        x = torch.tensor(d0[:, None], dtype=torch.float32, device=device)
        l = torch.tensor(np.asarray(load)[:, None], dtype=torch.float32,
                         device=device)
        out = model(x, l)
        return {
            "d_final": out["d_final"][:, 0].cpu().numpy(),
            "rel_growth": np.expm1(out["log1p_rel"].cpu().numpy()),
            "p_grow": torch.sigmoid(out["grow_logit"]).cpu().numpy(),
        }


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, split: SplitData, device: str = "cpu") -> dict:
    """Held-out: field MSE / relative-L2, grow-flag accuracy, P(grow)
    calibration (Brier + reliability bins) vs FD ground truth."""
    import torch
    model.eval()
    with torch.no_grad():
        out = model(split.Xte_d0.to(device), split.Xte_load.to(device))
        pred_field = out["d_final"].cpu().numpy()[:, 0]
        p_grow = torch.sigmoid(out["grow_logit"]).cpu().numpy()
        pred_rel = np.expm1(out["log1p_rel"].cpu().numpy())

    true_field = split.raw_test["d_final"]
    true_rel = split.raw_test["rel_growth"]
    true_grow = split.raw_test["grown"].astype(bool)

    field_mse = float(np.mean((pred_field - true_field) ** 2))
    num = np.linalg.norm((pred_field - true_field).reshape(len(true_field), -1),
                         axis=1)
    den = np.linalg.norm(true_field.reshape(len(true_field), -1), axis=1) + 1e-8
    rel_l2 = float(np.mean(num / den))

    pred_grow = p_grow >= 0.5
    grow_acc = float((pred_grow == true_grow).mean())
    brier = float(np.mean((p_grow - true_grow.astype(float)) ** 2))

    # rel_growth correlation in log space (robust to the super-critical tail)
    lr_t = np.log1p(np.clip(true_rel, 0, None))
    lr_p = np.log1p(np.clip(pred_rel, 0, None))
    if lr_t.std() > 1e-8 and lr_p.std() > 1e-8:
        rel_corr = float(np.corrcoef(lr_t, lr_p)[0, 1])
    else:
        rel_corr = float("nan")

    return {
        "n_test": int(len(true_rel)),
        "field_mse": field_mse,
        "field_rel_l2": rel_l2,
        "grow_acc": grow_acc,
        "brier": brier,
        "rel_growth_logcorr": rel_corr,
        "grown_frac_true": float(true_grow.mean()),
        "p_grow": p_grow,
        "true_grow": true_grow,
        "pred_rel": pred_rel,
        "true_rel": true_rel,
        "pred_field": pred_field,
        "true_field": true_field,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  Deliverable 1.3 — Speed benchmark (FD vs surrogate)
# ═════════════════════════════════════════════════════════════════════════════

def benchmark_speed(n: int = 16, nx: int = 60, ny: int = 48,
                    model=None, device: str = "cpu") -> dict:
    """Wall-time of FD simulate_growth vs surrogate inference over n samples.

    Returns per-call seconds for each and the speedup factor (headline number).
    """
    from cfrp_phasefield_2d import LaminateConfig, seed_defect, simulate_growth

    cfg = LaminateConfig(nx=nx, ny=ny)
    rng = np.random.default_rng(123)
    cx = rng.uniform(CX_LO, CX_HI, n)
    layer = rng.uniform(LAYER_LO, LAYER_HI, n)
    l2s = rng.uniform(LOG2SIZE_LO, LOG2SIZE_HI, n)
    loads = rng.uniform(LOAD_LO, LOAD_HI, n)
    d0s = np.stack([seed_defect(cx[i], 0.5, layer[i], l2s[i], cfg)
                    for i in range(n)])

    # FD timing (serial — this is what the surrogate replaces per call)
    t0 = time.perf_counter()
    for i in range(n):
        simulate_growth(d0s[i], float(loads[i]), cfg)
    fd_total = time.perf_counter() - t0

    if model is None:
        model, _ = load_surrogate(device=device)
    # warm-up (first call pays import/JIT)
    predict_batch(model, d0s[:1], loads[:1], device=device)
    t0 = time.perf_counter()
    predict_batch(model, d0s, loads, device=device)
    surr_batch = time.perf_counter() - t0
    t0 = time.perf_counter()
    for i in range(n):
        predict(model, d0s[i], float(loads[i]), device=device)
    surr_serial = time.perf_counter() - t0

    return {
        "n": n,
        "fd_per_call_s": fd_total / n,
        "surr_per_call_s": surr_serial / n,
        "surr_per_call_batched_s": surr_batch / n,
        "speedup_serial": (fd_total / n) / (surr_serial / n),
        "speedup_batched": (fd_total / n) / (surr_batch / n),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  Deliverable 1.4 — Drop-in surrogate growth_probability
# ═════════════════════════════════════════════════════════════════════════════

def growth_probability_surrogate(posterior_samples: np.ndarray, load: float,
                                 model=None, cfg=None, n_draws: int = 20,
                                 rng: np.random.Generator | None = None,
                                 device: str = "cpu") -> float:
    """Surrogate twin of cfrp_phasefield_2d.growth_probability.

    Same signature/semantics — MC over the Stage-2 FMPE posterior θ=(cx,cy,
    layer,log2size) — but each FD simulate_growth is replaced by one FNO
    inference, so the whole estimate is one batched forward pass (ms) instead
    of n_draws × ~1.7 s.

    P(grow) here is the MEAN of the per-draw surrogate P(grow) head (a soft
    probability), which both estimates the FD growth fraction and smooths the
    Monte-Carlo noise of the n_draws sub-sample.
    """
    from cfrp_phasefield_2d import LaminateConfig, seed_defect

    cfg = cfg or LaminateConfig()
    rng = rng or np.random.default_rng(0)
    samples = np.atleast_2d(np.asarray(posterior_samples, dtype=float))
    if samples.shape[1] != 4:
        raise ValueError("posterior_samples must be (N, 4): "
                         "(cx, cy, layer, log2size)")
    if len(samples) > n_draws:
        idx = rng.choice(len(samples), n_draws, replace=False)
        samples = samples[idx]

    if model is None:
        model, _ = load_surrogate(device=device)
    d0s = np.stack([seed_defect(cx, cy, layer, l2s, cfg)
                    for cx, cy, layer, l2s in samples])
    loads = np.full(len(samples), float(load), dtype=np.float32)
    out = predict_batch(model, d0s, loads, device=device)
    return float(out["p_grow"].mean())


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generate", action="store_true", help="run FD data gen")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--n-samples", type=int, default=800)
    ap.add_argument("--nx", type=int, default=60)
    ap.add_argument("--ny", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    if args.generate:
        generate_data(n_samples=args.n_samples, nx=args.nx, ny=args.ny,
                      n_workers=args.workers)
    if args.train:
        m = train(epochs=args.epochs)
        print("[eval] held-out:",
              {k: (round(v, 4) if isinstance(v, float) else v)
               for k, v in m.items()
               if k in ("n_test", "field_mse", "field_rel_l2", "grow_acc",
                        "brier", "rel_growth_logcorr", "grown_frac_true")})
    if args.benchmark:
        print("[bench]", benchmark_speed())


if __name__ == "__main__":
    main()
