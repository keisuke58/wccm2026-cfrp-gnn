"""
uq_baseline_propagation.py — Idea D: baseline uncertainty propagation.

Problem
-------
The diff-norm depends on: diff(x) = raw(x) - baseline(x)
When the baseline is estimated from a finite number of healthy flights,
there is per-node uncertainty:
    baseline(x) ~ N(μ_bl(x),  σ_bl(x)²)
where σ_bl(x) ∝ 1/√n_flights.

This uncertainty propagates into the diff-norm and ultimately into the GNN
class probabilities — but the current model ignores it and outputs a single
point prediction.

Approach: Monte Carlo uncertainty propagation
---------------------------------------------
1. Estimate baseline uncertainty  σ_bl(x)  from historical healthy flights.
2. At inference, sample K baseline perturbations:
       diff_k(x) = raw(x) - (μ_bl(x) + ε_k(x)),   ε_k ~ N(0, σ_bl²)
3. Run GNN on each perturbed diff → K class probability vectors.
4. Output: mean class probability + epistemic uncertainty (std across K runs).

This gives prediction intervals and a "confidence" flag:
    if std_class_prob > threshold → "uncertain, re-inspect"

Connection to Keio thesis
-------------------------
Village: Muramatsu lab (computational solid mechanics, UQ, phase-field).
This is the bridge:
    LUH CFRP-GNN SHM  →  Keio: UQ propagation in physics-aware inference
The phase-field connection: σ_bl uncertainty ~ crack-opening field noise →
propagate into AT2 damage field uncertainty → inspection decision with UQ.
Publishable at IWSHM 2027 (deadline ~Jan 2027).

Author: Nishioka (2026-06)
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

N_NODES    = 13942
GD         = Path("/home/nishioka/GNN")
DIR_COORD  = GD / "GNN_hole/GNN_hole_data"


# ─────────────────────────────────────────────────────────────────────────────
# Baseline uncertainty estimator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BaselineUncertainty:
    """
    Tracks per-node mean and variance of the healthy DSPSS field
    across multiple flights (online Welford update).

    σ_bl(x) ∝ 1/√n  as expected from sampling theory.
    """
    n: int = 0
    mean: Optional[np.ndarray] = None
    M2: Optional[np.ndarray] = None        # Welford's sum of squared deviations

    def update(self, field: np.ndarray) -> None:
        """Add one new healthy field observation (Welford online algorithm)."""
        x = field[:N_NODES].astype(np.float64)
        self.n += 1
        if self.mean is None:
            self.mean = x.copy()
            self.M2   = np.zeros(N_NODES, dtype=np.float64)
        else:
            delta     = x - self.mean
            self.mean = self.mean + delta / self.n
            delta2    = x - self.mean
            self.M2   = self.M2 + delta * delta2

    @property
    def variance(self) -> np.ndarray:
        """Per-node variance (Bessel-corrected, returns zeros if n < 2)."""
        if self.n < 2 or self.M2 is None:
            return np.zeros(N_NODES, dtype=np.float64)
        return self.M2 / (self.n - 1)

    @property
    def std(self) -> np.ndarray:
        """
        Standard error of the baseline mean estimate = σ / √n.
        Decreases as more healthy flights are observed (∝ 1/√n).
        This is the uncertainty on the MEAN, not the spread of individual values.
        """
        population_std = np.sqrt(self.variance + 1e-16)
        return (population_std / np.sqrt(max(self.n, 1))).astype(np.float32)

    def sample_baseline(self, rng: np.random.Generator) -> np.ndarray:
        """Sample one plausible baseline realisation from N(μ_bl, σ_bl²)."""
        if self.mean is None:
            return np.zeros(N_NODES, dtype=np.float32)
        eps = rng.normal(0.0, 1.0, N_NODES).astype(np.float32)
        return (self.mean + self.std * eps).astype(np.float32)

    def coverage_interval(self, alpha: float = 0.05) -> tuple:
        """(lower, upper) point-wise (1-α) interval for the baseline."""
        z = float(np.abs(np.quantile(np.random.standard_normal(10000), alpha / 2)))
        lo = (self.mean - z * self.std).astype(np.float32)
        hi = (self.mean + z * self.std).astype(np.float32)
        return lo, hi


# ─────────────────────────────────────────────────────────────────────────────
# Monte Carlo uncertainty propagation
# ─────────────────────────────────────────────────────────────────────────────

class MCBaselineUQ:
    """
    Monte Carlo propagation of baseline uncertainty through the GNN.

    Parameters
    ----------
    model       : trained dual-stream GNN (in_channels=5)
    bl_uq       : BaselineUncertainty fitted from healthy flights
    K           : number of MC samples
    device      : torch device
    """

    def __init__(self, model, bl_uq: BaselineUncertainty, K: int = 50,
                 device: Optional[torch.device] = None):
        self.model  = model
        self.bl_uq  = bl_uq
        self.K      = K
        self.device = device or torch.device("cpu")
        self._rng   = np.random.default_rng(0)

    def _make_data(self, raw: np.ndarray, diff: np.ndarray,
                   edge_index: torch.Tensor, coords: np.ndarray) -> Data:
        feats = np.column_stack([coords, diff[:N_NODES], raw[:N_NODES]]).astype(np.float32)
        return Data(
            x=torch.tensor(feats, dtype=torch.float, device=self.device),
            edge_index=edge_index.to(self.device),
        )

    @torch.no_grad()
    def predict(
        self,
        raw: np.ndarray,
        edge_index: torch.Tensor,
        coords: np.ndarray,
    ) -> dict:
        """
        Run K MC samples and return:
            mean_prob    : (N_nodes, n_class) mean class probabilities
            std_prob     : (N_nodes, n_class) std of class probabilities
            pred_class   : (N_nodes,) argmax of mean_prob
            confidence   : (N_nodes,) 1 - H(mean_prob) / log(n_class)  ∈ [0,1]
            uncertain    : (N_nodes,) bool, True if std > uncertainty_thr
        """
        self.model.eval()
        probs_list = []

        for _ in range(self.K):
            bl_sample  = self.bl_uq.sample_baseline(self._rng)
            diff       = self._compute_diff(raw, bl_sample)
            data       = self._make_data(raw, diff, edge_index, coords)
            logits     = self.model(data)   # (N, n_class)
            probs      = F.softmax(logits, dim=-1).cpu().numpy()
            probs_list.append(probs)

        probs_arr  = np.stack(probs_list, axis=0)   # (K, N, C)
        mean_prob  = probs_arr.mean(axis=0)          # (N, C)
        std_prob   = probs_arr.std(axis=0)           # (N, C)
        pred_class = mean_prob.argmax(axis=-1)        # (N,)

        # Shannon entropy normalised → confidence in [0,1]
        eps = 1e-12
        H   = -(mean_prob * np.log(mean_prob + eps)).sum(axis=-1)  # (N,)
        n_class    = mean_prob.shape[-1]
        confidence = 1.0 - H / np.log(n_class)

        # flag nodes where std of the dominant class > threshold
        dom_std   = std_prob[np.arange(len(pred_class)), pred_class]
        uncertain = dom_std > 0.1   # 10% threshold (tune per application)

        return {
            "mean_prob":  mean_prob,
            "std_prob":   std_prob,
            "pred_class": pred_class,
            "confidence": confidence,
            "uncertain":  uncertain,
        }

    @staticmethod
    def _compute_diff(raw: np.ndarray, baseline: np.ndarray) -> np.ndarray:
        diff = raw[:N_NODES].astype(np.float64) - baseline.astype(np.float64)
        s    = diff.std()
        return ((diff - diff.mean()) / (s + 1e-8)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Inspection decision gate
# ─────────────────────────────────────────────────────────────────────────────

def inspection_decision(result: dict, uncertain_fraction_thr: float = 0.05) -> dict:
    """
    Map MC UQ result to a flight-safety decision.

    Returns:
        action     : 'fly' | 'monitor' | 're-inspect'
        defect_prob: fraction of nodes predicted as defective (non-class-0)
        unc_frac   : fraction of nodes flagged as uncertain
        justification: human-readable string
    """
    pred     = result["pred_class"]
    conf     = result["confidence"]
    unc      = result["uncertain"]

    defect_prob = float((pred > 0).mean())
    unc_frac    = float(unc.mean())
    mean_conf   = float(conf.mean())

    if defect_prob < 0.01 and unc_frac < uncertain_fraction_thr:
        action = "fly"
        justification = (
            f"Defect probability {defect_prob:.1%} < 1%, "
            f"uncertainty fraction {unc_frac:.1%} < {uncertain_fraction_thr:.0%}. "
            f"Mean confidence {mean_conf:.2f}."
        )
    elif unc_frac >= uncertain_fraction_thr:
        action = "re-inspect"
        justification = (
            f"High baseline uncertainty ({unc_frac:.1%} nodes flagged). "
            f"Acquire more healthy flights to reduce σ_bl before deciding."
        )
    else:
        action = "monitor"
        justification = (
            f"Defect probability {defect_prob:.1%} detected. "
            f"Uncertainty {unc_frac:.1%}. "
            f"Initiate prognosis (Stage 3)."
        )

    return {"action": action, "defect_prob": defect_prob,
            "unc_frac": unc_frac, "justification": justification}


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests
# ─────────────────────────────────────────────────────────────────────────────

def _tests() -> None:
    print("[test] uq_baseline_propagation unit tests...")
    rng = np.random.default_rng(42)

    # BaselineUncertainty Welford
    bu = BaselineUncertainty()
    for _ in range(30):
        bu.update(rng.normal(0, 1, N_NODES))
    assert bu.n == 30
    assert bu.mean.shape == (N_NODES,)
    assert bu.std.shape  == (N_NODES,)
    assert (bu.std > 0).all(), "std should be positive after 30 updates"

    # std decreases as n grows
    bu2 = BaselineUncertainty()
    for _ in range(300):
        bu2.update(rng.normal(0, 1, N_NODES))
    assert bu2.std.mean() < bu.std.mean(), "std should shrink with more data"

    # sample_baseline returns correct shape and is noisy (not identical)
    s1 = bu.sample_baseline(rng)
    s2 = bu.sample_baseline(rng)
    assert s1.shape == (N_NODES,)
    assert not np.allclose(s1, s2), "different samples should differ"

    # coverage_interval shape
    lo, hi = bu.coverage_interval()
    assert lo.shape == hi.shape == (N_NODES,)
    assert (hi > lo).all()

    # inspection_decision
    result_safe = {
        "pred_class": np.zeros(N_NODES, dtype=int),
        "confidence": np.ones(N_NODES),
        "uncertain":  np.zeros(N_NODES, dtype=bool),
    }
    dec = inspection_decision(result_safe)
    assert dec["action"] == "fly"

    result_unc = {
        "pred_class": np.zeros(N_NODES, dtype=int),
        "confidence": np.zeros(N_NODES),
        "uncertain":  np.ones(N_NODES, dtype=bool),   # all uncertain
    }
    dec2 = inspection_decision(result_unc)
    assert dec2["action"] == "re-inspect"

    print("[test] ALL PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.test:
        _tests()
    else:
        print("Usage: python uq_baseline_propagation.py --test")
        print("For full pipeline, integrate MCBaselineUQ into dual_stream_gnn.py --eval")
