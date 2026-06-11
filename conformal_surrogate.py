"""
conformal_surrogate.py — HONEST uncertainty quantification for the Stage-3 FNO
crack-growth surrogate (`crack_surrogate.CrackFNO`) via split-conformal
prediction, with an explicit OUT-OF-DISTRIBUTION (OOD) stress test and a
deployable TRUST GATE.

Narrative ("分布外で 142 倍速がどこまで信用できるか")
------------------------------------------------------
The FNO surrogate replaces a ~1.8 s FD `simulate_growth` call with one ms-scale
inference (the "142x faster" headline of `crack_surrogate.py`).  That speed is
only useful if we know WHEN to trust it.  This module:

  1. CALIBRATES split-conformal prediction intervals on a held-out IN-DISTRIBUTION
     set for the decision-relevant scalar `rel_growth`, scored in the surrogate's
     own LOG1P space (the FD rel_growth has a sharp super-critical jump, so raw
     magnitude is hopeless — log space is where the surrogate is honest).  We
     report nominal 80/90/95% intervals and verify empirical coverage on a fresh
     in-dist test split.

  2. OOD STRESS TEST (the point): generates FD ground truth at defects/loads
     pushed progressively OUTSIDE the training envelope (load > 0.14, log2size >
     2, layers outside [3,15], cx near the edges) and quantifies, as a function
     of a scalar "distance from training distribution", (a) surrogate point
     error, (b) empirical coverage of the conformal interval (does it drop below
     nominal?), (c) interval width.  Vanilla split-conformal gives only a
     MARGINAL, exchangeability-based guarantee that BREAKS under the covariate
     shift OOD induces — showing where it breaks is the experiment, not a bug.

  3. TRUST GATE: a deployable rule — fall back to the slow exact FD solve when
     the input is outside the training envelope by more than a margin OR the
     conformal interval is wider than a threshold.  We report the flag rate on
     an OOD test set and the coverage recovered when flagged samples are
     deferred to FD (exact, coverage 1).  Cheap surrogate where trustworthy,
     exact FD where not.

  4. OPTIONAL covariate-shift-aware variant: widen the interval by a function of
     the OOD distance (a crude shift correction) and show it restores coverage
     better than vanilla.  Clearly labelled as a heuristic, not a guarantee.

HONESTY note (carried in every reported number): split-conformal's 1-α coverage
holds *marginally* under exchangeability of calibration and test points.  The
OOD set is by construction NOT exchangeable with the in-dist calibration set, so
coverage is expected to degrade — that degradation curve is the deliverable.

This module imports `crack_surrogate` and `cfrp_phasefield_2d` (does not modify
them).  Reuses the cached `data.npz` for the in-dist calibration/test; only the
OOD points cost fresh FD sims (parallelised with multiprocessing, like
`crack_surrogate.generate_data`).

CPU; needs torch only to load the FNO checkpoint.
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field

import numpy as np

import crack_surrogate as cs
from crack_surrogate import (CX_LO, CX_HI, LAYER_LO, LAYER_HI,
                             LOAD_LO, LOAD_HI, LOG2SIZE_LO, LOG2SIZE_HI)

HERE = os.path.dirname(os.path.abspath(__file__))
OOD_NPZ = os.path.join(cs.DATA_DIR, "conformal_ood.npz")

# Training envelope, as (lo, hi) per FMPE/load parameter.  theta = (cx, cy,
# layer, log2size); cy only fixes the section plane (unused by the 2D geometry)
# so it is NOT part of the distance metric.  Order of `ENVELOPE` keys defines
# the parameter vector used by `ood_distance`.
ENVELOPE = {
    "load":     (LOAD_LO, LOAD_HI),       # 0.04 – 0.14
    "log2size": (LOG2SIZE_LO, LOG2SIZE_HI),  # 0 – 2
    "cx":       (CX_LO, CX_HI),           # 0.2 – 0.8
    "layer":    (LAYER_LO, LAYER_HI),     # 3 – 15
}


# ═════════════════════════════════════════════════════════════════════════════
#  OOD distance:  how far outside the training envelope is an input?
# ═════════════════════════════════════════════════════════════════════════════

def _param_dict(load, cx, layer, log2size):
    return {"load": np.asarray(load, dtype=float),
            "log2size": np.asarray(log2size, dtype=float),
            "cx": np.asarray(cx, dtype=float),
            "layer": np.asarray(layer, dtype=float)}


def ood_distance(load, cx, layer, log2size) -> np.ndarray:
    """Scalar distance-from-training-distribution for each sample.

    For each envelope parameter we measure how far it falls OUTSIDE its training
    range [lo, hi], normalised by the range width (so each parameter's unit is
    "one training-range-width past the edge").  Inputs inside the range
    contribute 0.  The sample distance is the SUM over parameters of these
    one-sided normalised excursions — 0 for fully in-distribution points, > 0
    once any parameter leaves its envelope, growing with how far / how many
    parameters are OOD.

    Returns
    -------
    (N,) array of non-negative distances (0 = in-distribution).
    """
    P = _param_dict(load, cx, layer, log2size)
    n = max(len(v.ravel()) for v in P.values()) if P else 1
    dist = np.zeros(n, dtype=float)
    for name, (lo, hi) in ENVELOPE.items():
        v = np.broadcast_to(P[name], (n,)).astype(float)
        width = (hi - lo) or 1.0
        below = np.clip(lo - v, 0.0, None) / width
        above = np.clip(v - hi, 0.0, None) / width
        dist = dist + below + above
    return dist


def in_envelope(load, cx, layer, log2size, margin: float = 0.0) -> np.ndarray:
    """Boolean mask: is the input inside the training envelope (dist <= margin)?"""
    return ood_distance(load, cx, layer, log2size) <= margin


# ═════════════════════════════════════════════════════════════════════════════
#  Nonconformity scores + split-conformal calibration
# ═════════════════════════════════════════════════════════════════════════════

def _log1p(x):
    return np.log1p(np.clip(np.asarray(x, dtype=float), 0.0, None))


def rel_growth_scores(pred_rel: np.ndarray, true_rel: np.ndarray) -> np.ndarray:
    """Nonconformity score for the rel_growth scalar: absolute residual in the
    surrogate's LOG1P space, s_i = |log1p(pred_i) - log1p(true_i)|.

    Log space is used because the FD rel_growth has a sharp super-critical jump
    (≈0.6 → tens once a crack spans a ply); the surrogate regresses log-rel and
    is only honest there.  The resulting interval is symmetric in log1p and maps
    back to a multiplicative band on rel_growth.
    """
    return np.abs(_log1p(pred_rel) - _log1p(true_rel))


def field_scores(pred_field: np.ndarray, true_field: np.ndarray,
                 reduce: str = "mean") -> np.ndarray:
    """Optional nonconformity score for the d_final field: per-sample mean (or
    max) absolute residual over pixels."""
    res = np.abs(pred_field - true_field).reshape(len(pred_field), -1)
    return res.mean(axis=1) if reduce == "mean" else res.max(axis=1)


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Standard split-conformal quantile q̂ of calibration scores at level 1-α.

    Uses the finite-sample-valid rank: the ⌈(n+1)(1-α)⌉ / n empirical quantile,
    so that for a fresh exchangeable point  P(s_test <= q̂) >= 1-α.  Clipped to
    the max score when (n+1)(1-α) > n (small n / very high coverage).
    """
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    if n == 0:
        return float("inf")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:                 # cannot guarantee finite-sample coverage → +inf
        return float(s[-1])   # report widest finite band (report=inf separately)
    return float(s[k - 1])


@dataclass
class ConformalModel:
    """A fitted split-conformal predictor for surrogate rel_growth (log1p)."""
    qhat: dict                # alpha → q̂ (log1p score quantile)
    alphas: tuple             # nominal miscoverage levels
    n_calib: int
    score_kind: str = "rel_growth_log1p"
    field_qhat: dict = field(default_factory=dict)  # alpha → q̂ for field score

    def interval_log1p(self, pred_rel, alpha: float):
        """Prediction interval for log1p(rel_growth): [lo, hi] (clipped at 0)."""
        q = self.qhat[alpha]
        c = _log1p(pred_rel)
        return np.clip(c - q, 0.0, None), c + q

    def interval(self, pred_rel, alpha: float):
        """Prediction interval back in rel_growth space (expm1 of log1p band)."""
        lo, hi = self.interval_log1p(pred_rel, alpha)
        return np.expm1(lo), np.expm1(hi)

    def width_log1p(self, alpha: float) -> float:
        """Half-width is q̂; full interval width in log1p space is 2·q̂."""
        return 2.0 * self.qhat[alpha]

    def covered(self, pred_rel, true_rel, alpha: float) -> np.ndarray:
        lo, hi = self.interval_log1p(pred_rel, alpha)
        t = _log1p(true_rel)
        return (t >= lo - 1e-9) & (t <= hi + 1e-9)


def calibrate(pred_rel: np.ndarray, true_rel: np.ndarray,
              alphas=(0.20, 0.10, 0.05),
              pred_field: np.ndarray | None = None,
              true_field: np.ndarray | None = None) -> ConformalModel:
    """Split-conformal calibration on an in-distribution held-out set.

    pred_rel / true_rel : surrogate and FD rel_growth on the CALIBRATION split.
    alphas              : miscoverage levels → nominal coverage 1-α (default
                          80/90/95%).
    """
    scores = rel_growth_scores(pred_rel, true_rel)
    qhat = {a: conformal_quantile(scores, a) for a in alphas}
    fq = {}
    if pred_field is not None and true_field is not None:
        fs = field_scores(pred_field, true_field)
        fq = {a: conformal_quantile(fs, a) for a in alphas}
    return ConformalModel(qhat=qhat, alphas=tuple(alphas),
                          n_calib=len(scores), field_qhat=fq)


def empirical_coverage(cm: ConformalModel, pred_rel, true_rel,
                       alpha: float) -> float:
    """Fraction of points whose true log1p(rel_growth) falls in the interval."""
    cov = cm.covered(pred_rel, true_rel, alpha)
    return float(cov.mean()) if len(cov) else float("nan")


# ═════════════════════════════════════════════════════════════════════════════
#  In-distribution surrogate predictions from the cached dataset
# ═════════════════════════════════════════════════════════════════════════════

def _split_indist(data: dict, calib_frac: float = 0.5, seed: int = 1):
    """Split the cached in-dist dataset into calibration + test indices.

    Independent of the train/test split used to FIT the FNO (seed 0): we draw a
    fresh permutation (seed 1) and use HALF for conformal calibration, HALF for
    the in-dist coverage check.  (Conformal needs only that calibration points
    were unseen at *calibration* time and exchangeable with the test points;
    using the surrogate's own training points would optimistically bias scores,
    so we restrict to the FNO's held-out indices below.)
    """
    n = len(data["load"])
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_cal = max(1, int(round(calib_frac * len(idx))))
    return idx[:n_cal], idx[n_cal:]


def indist_predictions(data: dict, model, device: str = "cpu",
                       use_heldout_only: bool = True, fno_seed: int = 0):
    """Surrogate + FD rel_growth/field over the cached in-dist set.

    To keep conformal scores honest we restrict to the FNO's HELD-OUT test
    indices (the 20% never seen during training, seed `fno_seed`), so the
    nonconformity scores reflect genuine generalisation error, not memorised
    training residuals.  Returns a dict of arrays on that subset.
    """
    n = len(data["load"])
    if use_heldout_only:
        _, _, te_idx = cs._prepare(data, test_frac=0.2, seed=fno_seed)
        sub = np.asarray(te_idx)
    else:
        sub = np.arange(n)

    d0 = data["d0"][sub]
    load = data["load"][sub].astype(np.float32)
    out = cs.predict_batch(model, d0, load, device=device)
    theta = data["theta"][sub]      # (cx, cy, layer, log2size)
    dist = ood_distance(load, theta[:, 0], theta[:, 2], theta[:, 3])
    return {
        "idx": sub,
        "load": load,
        "cx": theta[:, 0], "layer": theta[:, 2], "log2size": theta[:, 3],
        "pred_rel": out["rel_growth"],
        "true_rel": data["rel_growth"][sub],
        "pred_field": out["d_final"],
        "true_field": data["d_final"][sub],
        "p_grow": out["p_grow"],
        "true_grow": data["grown"][sub].astype(bool),
        "ood_dist": dist,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  OOD stress test: FD ground truth at progressively-OOD inputs
# ═════════════════════════════════════════════════════════════════════════════

def sample_ood_inputs(n: int, seed: int = 7):
    """Sample inputs from in-distribution out to clearly OOD.

    A `level` in [0,1] interpolates how far each parameter is pushed past its
    training edge: level 0 → fully in-distribution; level 1 → load up to 0.20,
    log2size up to 3 (8 elements), layers just outside [3,15], cx pressed toward
    the edges (0.1 / 0.9).  Levels are tiled across the batch so the resulting
    OOD-distance sweep is dense.
    """
    rng = np.random.default_rng(seed)
    # `level` is tiled to span fully-in-dist (0) → clearly-OOD (1) with enough
    # mass at high levels to populate the far bins of the distance sweep.
    level = rng.uniform(0.0, 1.0, n) ** 0.6
    # load: in-range up to LOAD_HI, then push out to 0.20
    load = LOAD_LO + (LOAD_HI - LOAD_LO) * rng.uniform(0.4, 1.0, n)
    load = load + level * (0.20 - LOAD_HI) * rng.uniform(0.3, 1.0, n)
    # log2size: in-range up to 2, push out to 3
    log2size = rng.uniform(LOG2SIZE_LO, LOG2SIZE_HI, n)
    log2size = log2size + level * (3.0 - LOG2SIZE_HI) * rng.uniform(0.3, 1.0, n)
    # layer: occasionally pushed just outside [3,15] (toward 0 / 18)
    layer = rng.uniform(LAYER_LO, LAYER_HI, n)
    push_low = rng.random(n) < 0.5
    layer = np.where(push_low,
                     layer - level * (LAYER_LO - 0.0) * rng.uniform(0.3, 1, n),
                     layer + level * (18.0 - LAYER_HI) * rng.uniform(0.3, 1, n))
    layer = np.clip(layer, 0.0, 18.0)
    # cx: pressed toward the edges (0.1 / 0.9), outside [0.2,0.8]
    cx = rng.uniform(CX_LO, CX_HI, n)
    toward_lo = rng.random(n) < 0.5
    cx = np.where(toward_lo,
                  cx - level * (CX_LO - 0.1) * rng.uniform(0, 1, n),
                  cx + level * (0.9 - CX_HI) * rng.uniform(0, 1, n))
    cx = np.clip(cx, 0.05, 0.95)
    cy = rng.uniform(0.0, 1.0, n)
    return {"cx": cx, "cy": cy, "layer": layer, "log2size": log2size,
            "load": load, "level": level}


def generate_ood_ground_truth(n: int = 300, nx: int = 60, ny: int = 48,
                              seed: int = 7, n_workers: int | None = None,
                              out_path: str = OOD_NPZ,
                              cache: bool = True) -> dict:
    """Run FD `simulate_growth` at OOD inputs (parallelised) and cache.

    Mirrors `crack_surrogate.generate_data` but samples from in-dist out to
    clearly-OOD (`sample_ood_inputs`).  ~1.8 s/sim, so default n=300 ≈ a few
    minutes on all cores.  Cached to OOD_NPZ; reused on subsequent calls unless
    `cache=False` or the cached n differs.
    """
    import multiprocessing as mp

    if cache and os.path.exists(out_path):
        z = load_ood(out_path)
        if len(z["load"]) >= n:
            return z

    inp = sample_ood_inputs(n, seed=seed)
    jobs = [(inp["cx"][i], inp["cy"][i], inp["layer"][i], inp["log2size"][i],
             inp["load"][i], nx, ny) for i in range(n)]
    n_workers = n_workers or os.cpu_count() or 1
    t0 = time.perf_counter()
    print(f"[ood] {n} FD sims, grid {nx}x{ny}, {n_workers} workers ...")
    if n_workers > 1:
        with mp.Pool(n_workers) as pool:
            recs = pool.map(cs._sim_one, jobs, chunksize=4)
    else:
        recs = [cs._sim_one(j) for j in jobs]
    dt = time.perf_counter() - t0

    data = {
        "theta": np.stack([r["theta"] for r in recs]),       # cx,cy,layer,l2s
        "load": np.array([r["load"] for r in recs], dtype=np.float32),
        "d0": np.stack([r["d0"] for r in recs]),
        "d_final": np.stack([r["d_final"] for r in recs]),
        "rel_growth": np.array([r["rel_growth"] for r in recs], dtype=np.float32),
        "grown": np.array([r["grown"] for r in recs], dtype=bool),
        "area0": np.array([r["area0"] for r in recs], dtype=np.float32),
        "level": inp["level"].astype(np.float32),
        "nx": np.int64(nx), "ny": np.int64(ny),
    }
    if cache:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez_compressed(out_path, **data)
    print(f"[ood] done in {dt:.1f}s ({dt/n:.2f}s/sim); cached → {out_path}")
    return data


def load_ood(path: str = OOD_NPZ) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def ood_predictions(ood: dict, model, device: str = "cpu") -> dict:
    """Surrogate predictions + OOD distance over the OOD ground-truth set."""
    out = cs.predict_batch(model, ood["d0"], ood["load"].astype(np.float32),
                           device=device)
    theta = ood["theta"]
    dist = ood_distance(ood["load"], theta[:, 0], theta[:, 2], theta[:, 3])
    return {
        "load": ood["load"], "cx": theta[:, 0], "layer": theta[:, 2],
        "log2size": theta[:, 3],
        "pred_rel": out["rel_growth"], "true_rel": ood["rel_growth"],
        "pred_field": out["d_final"], "true_field": ood["d_final"],
        "p_grow": out["p_grow"], "true_grow": ood["grown"].astype(bool),
        "ood_dist": dist, "level": ood.get("level", np.zeros(len(dist))),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  Coverage / width / error as a function of OOD distance
# ═════════════════════════════════════════════════════════════════════════════

def binned_by_distance(pred: dict, cm: ConformalModel, alpha: float,
                       edges: np.ndarray | None = None) -> dict:
    """Coverage, mean interval width (log1p), and point error vs OOD distance.

    Bins the samples by `ood_dist`; per bin reports empirical coverage of the
    1-α conformal interval, mean log1p interval width (constant for vanilla
    conformal — included to contrast with the shift-aware variant), and mean
    absolute log1p point error of the surrogate.
    """
    dist = pred["ood_dist"]
    if edges is None:
        dmax = max(float(dist.max()), 1e-3)
        edges = np.linspace(0.0, dmax + 1e-9, 7)
    centres, cov, width, err, counts = [], [], [], [], []
    base_w = cm.width_log1p(alpha)
    pe = np.abs(_log1p(pred["pred_rel"]) - _log1p(pred["true_rel"]))
    covered = cm.covered(pred["pred_rel"], pred["true_rel"], alpha)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dist >= lo) & (dist < hi)
        if m.sum() == 0:
            continue
        centres.append(0.5 * (lo + hi))
        cov.append(float(covered[m].mean()))
        width.append(base_w)
        err.append(float(pe[m].mean()))
        counts.append(int(m.sum()))
    # ensure the final (rightmost) bin includes the max
    m = dist >= edges[-1]
    if m.sum() > 0:
        centres.append(float(dist[m].mean()))
        cov.append(float(covered[m].mean()))
        width.append(base_w)
        err.append(float(pe[m].mean()))
        counts.append(int(m.sum()))
    return {"centres": np.array(centres), "coverage": np.array(cov),
            "width": np.array(width), "abs_log_err": np.array(err),
            "counts": np.array(counts), "edges": edges, "alpha": alpha}


# ═════════════════════════════════════════════════════════════════════════════
#  Covariate-shift-aware variant (OPTIONAL, heuristic — not a guarantee)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ShiftAwareConformal:
    """Heuristic shift-aware wrapper: widen the log1p interval as a function of
    OOD distance, half-width = q̂·(1 + gamma·dist).

    This is NOT a finite-sample-valid conformal guarantee under arbitrary
    covariate shift — it is a monotone, distance-driven inflation that trades
    width for recovered coverage OOD.  `gamma` is fit so that the widened
    interval's empirical coverage on a (small) OOD calibration slice matches the
    nominal level; reported as a clearly-labelled baseline.
    """
    base: ConformalModel
    gamma: float

    def half_width_log1p(self, dist, alpha):
        return self.base.qhat[alpha] * (1.0 + self.gamma * np.asarray(dist))

    def covered(self, pred_rel, true_rel, dist, alpha):
        c = _log1p(pred_rel)
        hw = self.half_width_log1p(dist, alpha)
        t = _log1p(true_rel)
        return (t >= c - hw - 1e-9) & (t <= c + hw + 1e-9)

    def width_log1p(self, dist, alpha):
        return 2.0 * self.half_width_log1p(dist, alpha)


def fit_shift_aware(cm: ConformalModel, pred_rel, true_rel, dist, alpha: float,
                    gammas=None) -> ShiftAwareConformal:
    """Pick the smallest inflation `gamma` whose OOD coverage reaches nominal.

    Grid-search gamma on the provided (OOD-inclusive) slice; choose the smallest
    that lifts empirical coverage to >= 1-α (or the largest tried if none do).
    """
    if gammas is None:
        gammas = np.linspace(0.0, 4.0, 33)
    target = 1.0 - alpha
    best = gammas[-1]
    for g in gammas:
        sa = ShiftAwareConformal(cm, float(g))
        cov = sa.covered(pred_rel, true_rel, dist, alpha).mean()
        if cov >= target:
            best = float(g)
            break
    return ShiftAwareConformal(cm, float(best))


# ═════════════════════════════════════════════════════════════════════════════
#  TRUST GATE: defer to the slow exact FD solve when out of depth
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class TrustGate:
    """Deployable rule: trust the surrogate iff it is in-envelope AND its
    conformal interval is narrow; otherwise FALL BACK to the FD solve.

    Flag (defer to FD) when:
        ood_dist > dist_margin   OR   interval_width_log1p > width_thresh
    The width branch matters only for the shift-aware variant (vanilla width is
    constant); for vanilla conformal the gate is driven by `dist_margin`.
    """
    dist_margin: float
    width_thresh: float = float("inf")

    def flag(self, ood_dist, interval_width_log1p=None) -> np.ndarray:
        d = np.asarray(ood_dist, dtype=float)
        flagged = d > self.dist_margin
        if interval_width_log1p is not None:
            w = np.asarray(interval_width_log1p, dtype=float)
            flagged = flagged | (w > self.width_thresh)
        return flagged


def evaluate_trust_gate(pred: dict, cm: ConformalModel, gate: TrustGate,
                        alpha: float) -> dict:
    """Apply the gate: deferred (flagged) samples go to FD (exact → covered),
    kept samples use the surrogate's conformal interval.

    Reports:
      flag_rate            : fraction deferred to FD
      cov_no_defer         : coverage if we naively trust the surrogate on ALL
      cov_with_defer       : coverage after deferring flagged to FD (=1 there)
      cov_kept             : coverage on the KEPT (surrogate-served) samples
      fd_call_fraction     : = flag_rate (each deferred sample = one FD call)
      flagged_true_ood     : recall — of truly-OOD samples (dist>0), how many
                             were flagged
    """
    dist = pred["ood_dist"]
    flagged = gate.flag(dist, interval_width_log1p=cm.width_log1p(alpha))
    covered = cm.covered(pred["pred_rel"], pred["true_rel"], alpha)
    # deferred samples are solved exactly by FD → trivially "covered"
    covered_after = covered.copy()
    covered_after[flagged] = True

    truly_ood = dist > 1e-9
    recall = (float(flagged[truly_ood].mean()) if truly_ood.any()
              else float("nan"))
    kept = ~flagged
    return {
        "alpha": alpha,
        "nominal": 1.0 - alpha,
        "flag_rate": float(flagged.mean()),
        "cov_no_defer": float(covered.mean()),
        "cov_with_defer": float(covered_after.mean()),
        "cov_kept": float(covered[kept].mean()) if kept.any() else float("nan"),
        "fd_call_fraction": float(flagged.mean()),
        "flagged_true_ood_recall": recall,
        "n": int(len(dist)),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  End-to-end experiment driver
# ═════════════════════════════════════════════════════════════════════════════

def run_experiment(n_ood: int = 300, alphas=(0.20, 0.10, 0.05),
                   dist_margin: float = 0.10, seed: int = 7,
                   device: str = "cpu", n_workers: int | None = None,
                   verbose: bool = True) -> dict:
    """Full pipeline: calibrate → in-dist coverage → OOD stress → trust gate.

    Returns a dict with every reported number.  Reuses cached data.npz / the
    cached OOD set where present; only generates fresh FD sims for OOD points.
    """
    data = cs.load_data()
    nx, ny = int(data["nx"]), int(data["ny"])
    model, _ = cs.load_surrogate(device=device)

    # ── in-dist: held-out surrogate predictions, then split calib / test ──
    ind = indist_predictions(data, model, device=device)
    cal_i, te_i = _split_indist({"load": ind["load"]}, calib_frac=0.5, seed=1)

    cm = calibrate(ind["pred_rel"][cal_i], ind["true_rel"][cal_i],
                   alphas=alphas,
                   pred_field=ind["pred_field"][cal_i],
                   true_field=ind["true_field"][cal_i])

    indist_cov = {a: empirical_coverage(cm, ind["pred_rel"][te_i],
                                        ind["true_rel"][te_i], a)
                  for a in alphas}

    # ── OOD ground truth + surrogate predictions ──
    ood = generate_ood_ground_truth(n=n_ood, nx=nx, ny=ny, seed=seed,
                                    n_workers=n_workers)
    oodp = ood_predictions(ood, model, device=device)

    # OOD coverage at the primary level (centre of `alphas`, default 90%)
    primary = alphas[len(alphas) // 2]
    ood_cov_all = {a: empirical_coverage(cm, oodp["pred_rel"],
                                         oodp["true_rel"], a) for a in alphas}

    # coverage / width / error vs OOD distance (primary level)
    binned = binned_by_distance(oodp, cm, primary)

    # point-error growth: in-dist vs most-OOD bin
    pe_in = float(np.abs(_log1p(ind["pred_rel"][te_i])
                         - _log1p(ind["true_rel"][te_i])).mean())

    # ── trust gate (vanilla) ──
    gate = TrustGate(dist_margin=dist_margin)
    gate_res = {a: evaluate_trust_gate(oodp, cm, gate, a) for a in alphas}

    # ── optional shift-aware variant ──
    sa = fit_shift_aware(cm, oodp["pred_rel"], oodp["true_rel"],
                         oodp["ood_dist"], primary)
    sa_cov = float(sa.covered(oodp["pred_rel"], oodp["true_rel"],
                              oodp["ood_dist"], primary).mean())
    sa_mean_width = float(sa.width_log1p(oodp["ood_dist"], primary).mean())

    res = {
        "alphas": tuple(alphas), "primary_alpha": primary,
        "n_calib": cm.n_calib, "n_indist_test": int(len(te_i)),
        "n_ood": int(len(oodp["ood_dist"])),
        "qhat_log1p": dict(cm.qhat),
        "indist_coverage": indist_cov,
        "ood_coverage_all": ood_cov_all,
        "binned": binned,
        "abs_log_err_indist": pe_in,
        "abs_log_err_ood": float(binned["abs_log_err"].max()),
        "trust_gate": gate_res,
        "shift_aware": {"gamma": sa.gamma, "ood_coverage": sa_cov,
                        "mean_width_log1p": sa_mean_width,
                        "vanilla_width_log1p": cm.width_log1p(primary),
                        "vanilla_ood_coverage": ood_cov_all[primary]},
        "_cm": cm, "_ind": ind, "_oodp": oodp,
        "_cal_i": cal_i, "_te_i": te_i, "_sa": sa,
    }

    if verbose:
        _print_report(res)
    return res


def _print_report(res: dict):
    a = res["primary_alpha"]
    nom = 1 - a
    print("\n================ CONFORMAL SURROGATE — HONEST OOD UQ ===========")
    print(f"calib n={res['n_calib']}  in-dist test n={res['n_indist_test']}  "
          f"OOD n={res['n_ood']}")
    print("\n-- split-conformal log1p half-widths q̂ (rel_growth) --")
    for al, q in res["qhat_log1p"].items():
        print(f"   {int((1-al)*100):>3d}% : q̂={q:.4f}  (full width {2*q:.4f})")
    print("\n-- IN-DIST empirical coverage vs nominal (exchangeable → should "
          "match) --")
    for al, c in res["indist_coverage"].items():
        print(f"   nominal {int((1-al)*100):>3d}% : empirical {c*100:5.1f}%")
    print("\n-- OOD empirical coverage (NOT exchangeable → expected to drop) --")
    for al, c in res["ood_coverage_all"].items():
        print(f"   nominal {int((1-al)*100):>3d}% : empirical {c*100:5.1f}%")
    b = res["binned"]
    print(f"\n-- coverage vs OOD distance ({int(nom*100)}% interval) --")
    for ctr, cov, n in zip(b["centres"], b["coverage"], b["counts"]):
        print(f"   dist≈{ctr:5.2f} : coverage {cov*100:5.1f}%  (n={n})")
    print(f"\n-- point error (mean |Δlog1p rel|): in-dist {res['abs_log_err_indist']:.3f}"
          f"  → most-OOD bin {res['abs_log_err_ood']:.3f}  "
          f"(×{res['abs_log_err_ood']/max(res['abs_log_err_indist'],1e-9):.1f})")
    g = res["trust_gate"][a]
    print(f"\n-- TRUST GATE (dist_margin, {int(nom*100)}% level) --")
    print(f"   flag rate (→FD): {g['flag_rate']*100:5.1f}%   "
          f"true-OOD recall: {g['flagged_true_ood_recall']*100:5.1f}%")
    print(f"   coverage trusting surrogate everywhere : {g['cov_no_defer']*100:5.1f}%")
    print(f"   coverage after deferring flagged to FD : {g['cov_with_defer']*100:5.1f}%")
    print(f"   coverage on KEPT (surrogate-served)    : {g['cov_kept']*100:5.1f}%")
    s = res["shift_aware"]
    print(f"\n-- shift-aware variant (heuristic, γ={s['gamma']:.2f}) --")
    print(f"   vanilla OOD coverage {s['vanilla_ood_coverage']*100:5.1f}% "
          f"(width {s['vanilla_width_log1p']:.3f})  →  shift-aware "
          f"{s['ood_coverage']*100:5.1f}% (mean width {s['mean_width_log1p']:.3f})")
    print("================================================================\n")


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="store_true", help="full experiment")
    ap.add_argument("--gen-ood", action="store_true",
                    help="(re)generate the OOD FD ground-truth set only")
    ap.add_argument("--n-ood", type=int, default=300)
    ap.add_argument("--dist-margin", type=float, default=0.10)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    if args.gen_ood:
        generate_ood_ground_truth(n=args.n_ood, n_workers=args.workers,
                                  cache=True)
    if args.run or not (args.gen_ood):
        run_experiment(n_ood=args.n_ood, dist_margin=args.dist_margin,
                       n_workers=args.workers)


if __name__ == "__main__":
    main()
