"""Tests for expkit.runner TensorBoard wiring (pure-python, no torch).

Uses dry_run=True so no subprocess / GPU is needed.
"""
from __future__ import annotations

import os

from expkit import runner
from expkit.config import ExpConfig


# --------------------------------------------------------------------------- #
# tensorboard_cmd helper
# --------------------------------------------------------------------------- #
def test_tensorboard_cmd_default_port():
    cmd = runner.tensorboard_cmd("/some/logdir")
    assert cmd == ["tensorboard", "--logdir", "/some/logdir", "--port", "6006"]


def test_tensorboard_cmd_custom_port():
    cmd = runner.tensorboard_cmd("/out", port=6010)
    assert cmd == ["tensorboard", "--logdir", "/out", "--port", "6010"]


# --------------------------------------------------------------------------- #
# per-run tb_dir wiring
# --------------------------------------------------------------------------- #
def test_run_sets_per_run_tb_dir(tmp_path):
    cfg = ExpConfig(name="exp")  # empty tb_dir -> runner fills it in
    res = runner.run(cfg, str(tmp_path), "run0", dry_run=True)
    expected = os.path.join(res["run_dir"], "tb")
    assert cfg.tb_dir == expected
    # the cmd carries the per-run tb_dir
    assert "--tb_dir" in res["cmd"]
    assert res["cmd"][res["cmd"].index("--tb_dir") + 1] == expected


def test_run_respects_explicit_tb_dir(tmp_path):
    explicit = "/explicit/tb/path"
    cfg = ExpConfig(name="exp", tb_dir=explicit)
    res = runner.run(cfg, str(tmp_path), "run1", dry_run=True)
    assert cfg.tb_dir == explicit  # not overwritten
    assert res["cmd"][res["cmd"].index("--tb_dir") + 1] == explicit
