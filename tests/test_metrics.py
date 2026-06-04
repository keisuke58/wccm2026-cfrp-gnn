"""Tests for expkit.metrics.parse_log / write/read metrics json (pure-python)."""
from __future__ import annotations

import os

from expkit.metrics import parse_log, read_metrics_json, write_metrics_json


def _write_log(tmp_path, text, name="train.log"):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


# --------------------------------------------------------------------------- #
# parse_log — successful run
# --------------------------------------------------------------------------- #
def test_parse_done_log(tmp_path, done_log_text):
    log = _write_log(tmp_path, done_log_text)
    d = parse_log(log)

    assert d["best_macro_f1"] == 0.7654
    assert d["best_epoch"] == 3
    assert d["last_epoch"] == 3
    assert d["test_macro_f1"] == 0.7123
    assert d["status"] == "done"
    assert d["error"] is None
    # parse_log leaves meta (seed/run_id/name/conv_type) None; runner fills them
    assert d["seed"] is None
    assert d["log_path"] == log


def test_parse_running_log(tmp_path, running_log_text):
    log = _write_log(tmp_path, running_log_text)
    d = parse_log(log)

    assert d["status"] == "running"
    # no "Best Macro F1:" summary line printed yet
    assert d["best_macro_f1"] is None
    # no test section while still running
    assert d["test_macro_f1"] is None
    assert d["last_epoch"] == 2
    assert d["seed"] is None
    assert d["error"] is None


def test_parse_failed_log(tmp_path, failed_log_text):
    log = _write_log(tmp_path, failed_log_text)
    d = parse_log(log)

    assert d["status"] == "failed"
    assert d["error"] is not None
    assert "CUDA out of memory" in d["error"]


# --------------------------------------------------------------------------- #
# parse_log — robustness
# --------------------------------------------------------------------------- #
def test_parse_schema_keys_present(tmp_path, done_log_text):
    log = _write_log(tmp_path, done_log_text)
    d = parse_log(log)
    for key in ("best_macro_f1", "best_epoch", "last_epoch", "test_macro_f1",
                "status", "error", "log_path", "seed"):
        assert key in d


def test_parse_empty_log(tmp_path):
    log = _write_log(tmp_path, "")
    d = parse_log(log)
    assert d["best_macro_f1"] is None
    assert d["last_epoch"] is None
    # an empty/never-started log is not 'done'
    assert d["status"] in ("running", "failed")


def test_parse_picks_last_epoch_max(tmp_path):
    text = (
        "=== Epoch 1/100 ===\nMacro F1: 0.1000\n"
        "=== Epoch 2/100 ===\nMacro F1: 0.2000\n"
        "=== Epoch 10/100 ===\nMacro F1: 0.3000\n"
    )
    log = _write_log(tmp_path, text)
    d = parse_log(log)
    assert d["last_epoch"] == 10


# --------------------------------------------------------------------------- #
# write / read metrics json
# --------------------------------------------------------------------------- #
def test_write_then_read_metrics_json(tmp_path):
    run_dir = str(tmp_path)
    d = {
        "run_id": "gat_s42_x",
        "name": "exp",
        "conv_type": "gat",
        "seed": 42,
        "best_macro_f1": 0.77,
        "best_epoch": 5,
        "last_epoch": 9,
        "test_macro_f1": 0.71,
        "status": "done",
        "error": None,
        "log_path": os.path.join(run_dir, "train.log"),
    }
    path = write_metrics_json(run_dir, d)
    assert os.path.exists(path)
    assert path.endswith("metrics.json")

    back = read_metrics_json(run_dir)
    assert back["run_id"] == "gat_s42_x"
    assert back["best_macro_f1"] == 0.77
    assert back["status"] == "done"


def test_read_missing_metrics_json_returns_none(tmp_path):
    assert read_metrics_json(str(tmp_path)) is None
