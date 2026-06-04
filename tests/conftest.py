"""Shared pytest fixtures for the expkit pure-python test suite.

NO torch / CUDA: every fixture and test imports only expkit.config / .metrics /
.aggregate / .sweep so the suite runs on a bare CPU box.
"""
from __future__ import annotations

import os
import textwrap

import pytest

from expkit.config import ExpConfig


# --------------------------------------------------------------------------- #
# config fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def base_cfg() -> ExpConfig:
    """A default ExpConfig (defaults == balanced recipe per the SPEC)."""
    return ExpConfig()


@pytest.fixture
def named_cfg() -> ExpConfig:
    """A config with non-default meta + a couple toggles flipped."""
    return ExpConfig(
        name="unit",
        gpu=2,
        master_port=29515,
        conv_type="sage",
        seed=7,
        use_onecycle=False,
        use_amp=True,
        include_ndf=False,
        ndf_count=0,
        defect_cap=0,
    )


# --------------------------------------------------------------------------- #
# synthetic train.log strings (match train.py print formats verbatim)
# --------------------------------------------------------------------------- #
@pytest.fixture
def done_log_text() -> str:
    """A complete, successful run log (early-stopped) with a test section."""
    return textwrap.dedent(
        """\
        Seed: 42

        === Epoch 1/100 ===
        Train Loss: 2.5000, Val Loss: 2.4000, Val Acc: 0.1000, Macro F1: 0.0500

        === Epoch 2/100 ===
        Train Loss: 1.9000, Val Loss: 1.8000, Val Acc: 0.3000, Macro F1: 0.4200

        === Epoch 3/100 ===
        Train Loss: 1.2000, Val Loss: 1.1000, Val Acc: 0.5000, Macro F1: 0.7654

        ============================================================
        Early stopping triggered at epoch 63
          - Macro F1 hasn't improved for 60 epochs (monitoring Macro F1 only)
          - Best Val Loss: 1.1000
          - Best Macro F1: 0.7654
          - Best model was at epoch 3
        ============================================================

        [INFO] Test set result:
         - Test Loss: 1.0500
         - Test Accuracy: 0.5200
         - Weighted Precision: 0.6000, Weighted Recall: 0.5200, Weighted F1-Score: 0.5500
         - Macro Precision: 0.7000, Macro Recall: 0.6800, Macro F1-Score: 0.7123
         - Balanced Accuracy: 0.6800

        [INFO] Macro F1 Score: 0.7123 (logged for reference)
        """
    )


@pytest.fixture
def running_log_text() -> str:
    """An in-progress run: epochs only, no early-stop / best / test section."""
    return textwrap.dedent(
        """\
        Seed: 7

        === Epoch 1/500 ===
        Train Loss: 2.5000, Val Loss: 2.4000, Val Acc: 0.1000, Macro F1: 0.0500

        === Epoch 2/500 ===
        Train Loss: 1.9000, Val Loss: 1.8000, Val Acc: 0.3000, Macro F1: 0.4200
        """
    )


@pytest.fixture
def failed_log_text() -> str:
    """A crashed run: a python traceback, no final/best line."""
    return textwrap.dedent(
        """\
        Seed: 42

        === Epoch 1/100 ===
        Train Loss: 2.5000, Val Loss: 2.4000, Val Acc: 0.1000, Macro F1: 0.0500
        Traceback (most recent call last):
          File "train.py", line 2056, in main
            raise RuntimeError("CUDA out of memory")
        RuntimeError: CUDA out of memory
        """
    )


# --------------------------------------------------------------------------- #
# fake run-dir builders for aggregate tests
# --------------------------------------------------------------------------- #
@pytest.fixture
def make_run_dir(tmp_path):
    """Factory: create {root}/{name}/{run_id}/ with a train.log and optional
    metrics.json. Returns the run-dir path.
    """
    from expkit.metrics import write_metrics_json

    def _make(run_id, name="exp", log_text="", metrics=None, root=None):
        root = root or (tmp_path / "runs")
        run_dir = os.path.join(str(root), name, run_id)
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "train.log"), "w") as fh:
            fh.write(log_text)
        if metrics is not None:
            write_metrics_json(run_dir, metrics)
        return run_dir

    return _make
