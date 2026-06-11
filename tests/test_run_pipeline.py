"""Tests for run_pipeline — the one-command Stage 0→4 reusable-rocket SHM chain.

Smoke-level: the full chain must run end-to-end on one held-out sample and
return a valid OK/REPAIR/RETIRE decision with finite survival / remaining-flight
numbers.  These require the trained FNO checkpoint + cached FMPE posterior +
the held-out CFRP data on disk; if any are missing the tests skip with a clear
message.  torch is required (FNO + FMPE) and import-skipped if absent.

CPU-only and fast: the surrogate path finishes in a few seconds; the FD-engine
test uses a tiny sub-sample and is marked slow.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")

import crack_surrogate as cs  # noqa: E402
import run_pipeline as rp  # noqa: E402


def _have_artifacts() -> bool:
    return (os.path.exists(cs.MODEL_PT)
            and os.path.exists(rp.CACHED_POSTERIOR)
            and os.path.isdir(rp.DIR_4X4)
            and os.path.isdir(rp.DIR_1X1))


requires_artifacts = pytest.mark.skipif(
    not _have_artifacts(),
    reason="needs trained FNO (fno_crack.pt), cached FMPE posterior "
           "(fmpe_posterior_e2e.npz) and the CFRP held-out data on disk")

VALID = {"OK", "REPAIR", "RETIRE"}


@requires_artifacts
def test_full_chain_surrogate_returns_valid_decision():
    """Full Stage 0→4 chain on one sample, surrogate engine → valid decision
    with finite P(survive) and E[remaining flights]."""
    out = rp.run_pipeline(sample=3, engine="surrogate", n_flights=20,
                          n_draws=12, quiet=True)
    s3 = out["s3"]
    assert s3["decision"] in VALID
    assert 0.0 <= s3["p_growth_next"] <= 1.0
    assert 0.0 <= s3["p_survive_n"] <= 1.0
    assert math.isfinite(s3["expected_remaining_flights"])
    assert s3["expected_remaining_flights"] >= 0.0
    # survival curve is a proper, monotone-non-increasing probability curve
    surv = s3["p_survive_curve"]
    assert len(surv) == 20
    assert np.all(np.isfinite(surv))
    assert np.all(surv[1:] <= surv[:-1] + 1e-9)
    # Stage-2 posterior is the expected (N,4) shape
    assert out["s2"]["posterior"].shape[1] == 4
    # Stage-4 produced the fleet-prior sharpening sequence
    assert len(out["s4"]["rec_sizes"]) >= 2
    assert math.isfinite(out["total_time_s"])


@requires_artifacts
def test_stage1_screen_well_formed_and_agrees_with_stage2():
    """Stage-1 19-class screen runs in the e2e chain and is shape-consistent:
    a defect-present bool, a 0..18 class / layer bucket, a finite (cx,cy) region
    estimate, a confidence in [0,1], and a per-node map over class-0/pred-class.
    On the cached defective sample it should agree with Stage-2 (same defect-
    present call and same layer bucket)."""
    out = rp.run_pipeline(sample=0, engine="surrogate", n_draws=8,
                          quiet=True, stage1=True)
    s1, s2 = out["s1"], out["s2"]
    assert s1 is not None
    assert isinstance(s1["defect_present"], bool)
    assert 0 <= s1["pred_class"] <= rp.N_CLASSES - 1
    assert s1["layer_bucket"] in ("defect-free", "layer-1", "layer-2")
    assert len(s1["region_xy"]) == 2 and all(np.isfinite(v) for v in s1["region_xy"])
    assert 0.0 <= s1["confidence"] <= 1.0
    assert s1["node_pred"].shape == (rp.N_NODES,)
    assert set(np.unique(s1["node_pred"])) <= {0, s1["pred_class"]}
    cmp = rp._stage1_vs_stage2(s1, s2)
    assert isinstance(cmp["agree"], bool)
    # the canonical cached sample is defective → Stage-1 and Stage-2 concur
    assert cmp["agree"] is True


@requires_artifacts
def test_stage1_can_be_disabled():
    """--no-stage1 path leaves s1 as None (backward-compatible Stage 0,2,3,4)."""
    out = rp.run_pipeline(sample=0, engine="surrogate", n_draws=8,
                          quiet=True, stage1=False)
    assert out["s1"] is None


@requires_artifacts
def test_stage0_detects_planted_defect():
    """Stage-0 Mahalanobis flags the planted defect on a known defective 4x4
    sample (high single-sample AUROC, non-zero flagged nodes)."""
    s0 = rp.stage0_detect(sample_idx=3)
    assert s0["mask"] is not None and s0["mask"].sum() > 0  # defective sample
    assert s0["detected"] is True
    assert s0["n_flagged"] > 0
    assert s0["auroc"] is not None and s0["auroc"] > 0.9


@requires_artifacts
def test_surrogate_pgrow_in_unit_interval():
    """The surrogate Stage-3 P(grow) is a probability for several loads."""
    import cfrp_phasefield_2d as pf
    z = np.load(rp.CACHED_POSTERIOR, allow_pickle=True)
    post = z["posterior"].astype(float)
    cfg = pf.LaminateConfig()
    model, _ = cs.load_surrogate(device="cpu")
    for load in (0.06, 0.10, 0.14):
        r = rp.stage3_surrogate(post, load, n_flights=10, cfg=cfg,
                                n_draws=8, seed=0, model=model)
        assert 0.0 <= r["p_growth_next"] <= 1.0
        assert r["decision"] in VALID


@requires_artifacts
@pytest.mark.slow
def test_full_chain_fd_engine_runs():
    """The exact FD phase-field engine also runs end-to-end (tiny sub-sample,
    short horizon) and returns a valid decision."""
    out = rp.run_pipeline(sample=3, engine="fd", n_flights=4,
                          n_draws=3, quiet=True)
    s3 = out["s3"]
    assert s3["decision"] in VALID
    assert 0.0 <= s3["p_survive_n"] <= 1.0
    assert math.isfinite(s3["expected_remaining_flights"])
    assert s3["engine"].startswith("FD")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
