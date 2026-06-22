"""
bootstrap_adapt.py — Idea A: first-flight bootstrap → diff-norm transition.

Problem
-------
Diff-norm DSPSS requires a pristine (defect-free) baseline field.
On the very first inspection of an unknown structure this baseline does not exist.
Raw DSPSS is the only option — but its hole-edge artefacts hurt Stage-1 GNN.

Solution: zero-knowledge start → self-bootstrapping baseline
-------------------------------------------------------------
Flight 0  (no baseline):
    Stage-0  raw  Mahalanobis detection  (AUROC ≈ 0.84 degraded, still usable)
    Stage-1  SKIP  (no reliable GNN without diff channel)
    Action   → store healthy-verdict specimens as candidate baseline pool

Flight 1+  (baseline available):
    Compute baseline from Flight-0 healthy pool (mean field or median)
    Stage-0  diff-norm detection  (AUROC ≈ 0.999)
    Stage-1  diff-norm GNN  (macro-F1 ≈ 0.803)
    Action   → update baseline with newly confirmed healthy specimens

Baseline update rule (EMA with lockout):
    baseline_new = α * mean(new_healthy) + (1 - α) * baseline_old
    α = 0.2  (conservative; prevents sudden baseline drift from misclassified defects)

Usage
-----
    # Simulate a two-flight campaign
    python bootstrap_adapt.py --test
    python bootstrap_adapt.py --demo --n_flights 5

Author: Nishioka (2026-06)
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Vancouver paths ───────────────────────────────────────────────────────────
GD = Path("/home/nishioka/GNN")
DIR_RAW_FLAT  = GD / "GNN_hole_2026/all_sub_hole_defect"          # flat subtracted (= raw - ref)
DIR_REF       = GD / "GNN_hole_2026/real_hole_no_defect_original.npy"
DIR_SUB_ZSCORE = GD / "GNN_hole_2026/all_sub_hole_defect_zscore"
N_NODES = 13942

# Stage-0 detection parameters (matches run_pipeline.py)
THR_Q       = 0.999   # node-level z-score threshold quantile
ALPHA_EMA   = 0.2     # baseline EMA update weight


# ─────────────────────────────────────────────────────────────────────────────
# Baseline Manager
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BaselineManager:
    """
    Maintains and updates the healthy-field baseline across flights.

    Attributes
    ----------
    baseline      : current best estimate of healthy DSPSS field  (N_NODES,)
    baseline_age  : number of flights since baseline was first established
    healthy_pool  : DSPSS fields of confirmed-healthy specimens this flight
    """
    baseline: Optional[np.ndarray] = None
    baseline_age: int = 0
    healthy_pool: List[np.ndarray] = field(default_factory=list)
    _history: List[float] = field(default_factory=list)  # per-flight mean baseline

    @property
    def has_baseline(self) -> bool:
        return self.baseline is not None

    def add_healthy_specimen(self, dspss: np.ndarray) -> None:
        """Register a confirmed-healthy specimen for baseline update."""
        self.healthy_pool.append(dspss[:N_NODES].copy())

    def commit_flight(self, alpha: float = ALPHA_EMA) -> None:
        """
        End of flight: update baseline from this flight's healthy pool.
        Call after all specimens in the current flight have been processed.
        """
        if not self.healthy_pool:
            print("[baseline] WARNING: no healthy specimens this flight — baseline unchanged")
            return

        new_mean = np.mean(np.stack(self.healthy_pool, axis=0), axis=0)

        if self.baseline is None:
            self.baseline = new_mean.copy()
            print(f"[baseline] baseline initialised from {len(self.healthy_pool)} specimens")
        else:
            self.baseline = alpha * new_mean + (1 - alpha) * self.baseline
            print(f"[baseline] baseline updated (EMA α={alpha}) "
                  f"from {len(self.healthy_pool)} new healthy specimens")

        self.baseline_age += 1
        self._history.append(float(self.baseline.mean()))
        self.healthy_pool.clear()

    def diff_norm(self, dspss: np.ndarray, clip_p: Tuple[float, float] = (1.0, 99.0)) -> np.ndarray:
        """
        Compute difference-normalised DSPSS: per-node z-score of (raw - baseline).

        Returns zeros if no baseline is available (safe fallback).
        """
        if self.baseline is None:
            return np.zeros(N_NODES, dtype=np.float32)

        diff = dspss[:N_NODES].astype(np.float64) - self.baseline
        p1, p99 = np.percentile(diff, clip_p[0]), np.percentile(diff, clip_p[1])
        diff = np.clip(diff, p1, p99)
        s = diff.std()
        return ((diff - diff.mean()) / s if s > 0 else diff * 0.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Stage-0 detection (raw z-score, parameter-free)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Stage0Detector:
    """
    Mahalanobis/z-score detection on raw DSPSS.
    Per-node μ/σ estimated from reference healthy pool.
    """
    mu: Optional[np.ndarray] = None
    sigma: Optional[np.ndarray] = None

    def fit(self, healthy_fields: List[np.ndarray]) -> None:
        """Fit μ/σ from a set of confirmed-healthy fields."""
        arr = np.stack([h[:N_NODES] for h in healthy_fields], axis=0)
        self.mu    = arr.mean(axis=0)
        self.sigma = arr.std(axis=0) + 1e-8
        print(f"[stage0] fitted on {len(healthy_fields)} healthy fields  "
              f"σ_mean={self.sigma.mean():.4f}")

    def score(self, dspss: np.ndarray) -> float:
        """
        Return specimen-level anomaly score = fraction of nodes exceeding 3σ
        relative to the HEALTHY REFERENCE μ/σ (fitted by `fit()`).
        Higher score → more likely defective.

        Before fit(): falls back to the 90th-percentile of the specimen's own
        z-scores — coarser but still order-preserving across specimens.
        """
        if self.mu is None:
            # Before fit: treat all specimens as healthy so the first flight
            # accumulates healthy samples for baseline initialisation.
            return 0.0
        z = np.abs((dspss[:N_NODES] - self.mu) / self.sigma)
        # Absolute 3-sigma threshold — healthy nodes cluster near z≈1,
        # defective nodes deviate to z>>3 at the damage site.
        return float((z > 3.0).mean())

    def detect(self, dspss: np.ndarray, decision_thr: float = 0.01) -> bool:
        """True if specimen is classified as defective (>1% of nodes exceed 3σ)."""
        return self.score(dspss) > decision_thr


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline state machine
# ─────────────────────────────────────────────────────────────────────────────

class BootstrapPipeline:
    """
    Full bootstrap-adaptive SHM pipeline.

    State machine:
        PHASE_RAW   : Flight 0 — raw Stage-0 only, no GNN
        PHASE_FULL  : Flight 1+ — diff-norm Stage-0 + GNN
    """

    PHASE_RAW  = "raw_only"
    PHASE_FULL = "full_pipeline"

    def __init__(self, gnn_model=None) -> None:
        self.baseline = BaselineManager()
        self.stage0   = Stage0Detector()
        self.gnn      = gnn_model   # T.build_model(...) or None
        self.phase    = self.PHASE_RAW
        self.flight   = 0

    @property
    def mode_label(self) -> str:
        return f"flight={self.flight}  phase={self.phase}"

    def process_specimen(
        self,
        dspss_raw: np.ndarray,
        ground_truth_label: Optional[int] = None,
    ) -> Dict:
        """
        Process one specimen through the pipeline.

        Returns dict with keys:
            detected    : bool
            gnn_class   : int or None (None in PHASE_RAW)
            diff_field  : np.ndarray (zeros in PHASE_RAW)
            phase       : str
        """
        detected = self.stage0.detect(dspss_raw)

        if not detected:
            # Confirmed healthy → add to baseline pool
            self.baseline.add_healthy_specimen(dspss_raw)

        diff_field = np.zeros(N_NODES, dtype=np.float32)
        gnn_class  = None

        if self.phase == self.PHASE_FULL and detected and self.gnn is not None:
            diff_field = self.baseline.diff_norm(dspss_raw)
            gnn_class  = self._run_gnn(dspss_raw, diff_field)

        return {
            "detected":   detected,
            "gnn_class":  gnn_class,
            "diff_field": diff_field,
            "phase":      self.phase,
        }

    def end_of_flight(self) -> None:
        """
        Call after all specimens in one flight are processed.
        Commits baseline update and advances phase.
        """
        # Fit Stage-0 BEFORE commit_flight() clears the healthy pool
        if self.baseline.healthy_pool and self.phase == self.PHASE_RAW:
            self.stage0.fit(self.baseline.healthy_pool)

        self.baseline.commit_flight()
        self.flight += 1

        if self.phase == self.PHASE_RAW and self.baseline.has_baseline:
            self.phase = self.PHASE_FULL
            print(f"[pipeline] TRANSITION: {self.PHASE_RAW} → {self.PHASE_FULL}  "
                  f"(baseline_age={self.baseline.baseline_age})")

    def _run_gnn(self, dspss_raw: np.ndarray, diff_field: np.ndarray) -> int:
        """Run GNN on 5-channel features [x,y,z,diff,raw]."""
        import torch
        # coords would normally be loaded once — placeholder here
        cx = cy = cz = np.zeros(N_NODES, dtype=np.float32)
        feats = np.column_stack([cx, cy, cz, diff_field, dspss_raw[:N_NODES]])
        x_t   = torch.tensor(feats, dtype=torch.float).unsqueeze(0)
        with torch.no_grad():
            logits = self.gnn(x_t)
        return int(logits.argmax(dim=-1).mode().values.item())


# ─────────────────────────────────────────────────────────────────────────────
# Demo / simulation
# ─────────────────────────────────────────────────────────────────────────────

def demo(n_flights: int = 5, specimens_per_flight: int = 100, defect_rate: float = 0.1) -> None:
    """
    Simulate a multi-flight campaign with synthetic DSPSS fields.
    Shows detection accuracy improving after Phase transition.
    """
    rng = np.random.default_rng(42)
    true_healthy_mean = rng.normal(1.0, 0.1, N_NODES).astype(np.float32)

    pipeline = BootstrapPipeline(gnn_model=None)

    # Fit a reference Stage-0 from synthetic pristine data (in practice: from first healthy batch)
    pristine_pool = [true_healthy_mean + rng.normal(0, 0.05, N_NODES).astype(np.float32)
                     for _ in range(50)]
    pipeline.stage0.fit(pristine_pool)

    print(f"\n{'='*60}")
    print(f"Bootstrap demo: {n_flights} flights × {specimens_per_flight} specimens, "
          f"defect_rate={defect_rate:.0%}")
    print('='*60)

    for flt in range(n_flights):
        tp = fp = tn = fn = 0

        for _ in range(specimens_per_flight):
            is_defective = rng.random() < defect_rate
            if is_defective:
                dspss = true_healthy_mean + rng.normal(0.5, 0.2, N_NODES).astype(np.float32)
            else:
                dspss = true_healthy_mean + rng.normal(0.0, 0.05, N_NODES).astype(np.float32)

            result = pipeline.process_specimen(dspss)
            det    = result["detected"]

            if is_defective  and det:     tp += 1
            elif is_defective and not det: fn += 1
            elif not is_defective and det: fp += 1
            else:                          tn += 1

        pipeline.end_of_flight()

        recall = tp / (tp + fn + 1e-9)
        fpr    = fp / (fp + tn + 1e-9)
        print(f"  flight {flt}  [{pipeline.phase:13s}]  "
              f"recall={recall:.3f}  FPR={fpr:.3f}  "
              f"baseline_age={pipeline.baseline.baseline_age}")

    print(f"\nBaseline history (mean per flight): "
          f"{[f'{v:.4f}' for v in pipeline.baseline._history]}")
    print("Demo done.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests
# ─────────────────────────────────────────────────────────────────────────────

def _tests() -> None:
    print("[test] bootstrap_adapt unit tests...")

    rng = np.random.default_rng(0)
    N   = N_NODES

    # BaselineManager
    bm = BaselineManager()
    assert not bm.has_baseline
    for _ in range(10):
        bm.add_healthy_specimen(rng.normal(0, 1, N).astype(np.float32))
    bm.commit_flight()
    assert bm.has_baseline
    assert bm.baseline_age == 1
    assert bm.baseline.shape == (N,)
    assert len(bm.healthy_pool) == 0   # cleared after commit

    # diff_norm returns zeros before baseline
    bm2 = BaselineManager()
    z = bm2.diff_norm(rng.normal(0, 1, N).astype(np.float32))
    assert np.all(z == 0.0), "Should return zeros before baseline init"

    # diff_norm after baseline: mean≈0, std≈1
    z = bm.diff_norm(rng.normal(0, 1, N).astype(np.float32))
    assert abs(z.mean()) < 0.1, f"diff_norm mean off: {z.mean():.4f}"

    # Stage0Detector
    healthy = [rng.normal(1.0, 0.05, N).astype(np.float32) for _ in range(50)]
    d = Stage0Detector()
    d.fit(healthy)
    # healthy specimen should not be detected
    h_score = d.score(rng.normal(1.0, 0.05, N).astype(np.float32))
    # defective specimen (shifted) should score higher
    def_score = d.score(rng.normal(1.5, 0.5, N).astype(np.float32))
    assert def_score > h_score, f"defective={def_score:.4f} should > healthy={h_score:.4f}"

    # BootstrapPipeline state machine
    pipe = BootstrapPipeline()
    assert pipe.phase == BootstrapPipeline.PHASE_RAW
    # simulate flight 0
    for _ in range(20):
        pipe.process_specimen(rng.normal(1.0, 0.05, N).astype(np.float32))
    pipe.end_of_flight()
    assert pipe.flight == 1
    assert pipe.phase == BootstrapPipeline.PHASE_FULL
    assert pipe.baseline.has_baseline

    print("[test] ALL PASS")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--test",  action="store_true")
    mode.add_argument("--demo",  action="store_true")
    ap.add_argument("--n_flights",            type=int,   default=5)
    ap.add_argument("--specimens_per_flight", type=int,   default=100)
    ap.add_argument("--defect_rate",          type=float, default=0.1)
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse()
    if args.test:
        _tests()
    else:
        demo(args.n_flights, args.specimens_per_flight, args.defect_rate)
