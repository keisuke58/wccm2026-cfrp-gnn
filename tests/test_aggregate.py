"""Tests for expkit.aggregate.collect_runs / ranking / serialization."""
from __future__ import annotations

from expkit.aggregate import collect_runs, to_csv, to_markdown


def test_collect_runs_finds_all(make_run_dir, tmp_path):
    root = tmp_path / "runs"
    make_run_dir("gat_s42_a", name="archA", root=root,
                 metrics={"run_id": "gat_s42_a", "name": "archA",
                          "conv_type": "gat", "seed": 42,
                          "best_macro_f1": 0.70, "best_epoch": 3,
                          "last_epoch": 3, "test_macro_f1": 0.68,
                          "status": "done", "error": None,
                          "log_path": ""})
    make_run_dir("sage_s42_b", name="archB", root=root,
                 metrics={"run_id": "sage_s42_b", "name": "archB",
                          "conv_type": "sage", "seed": 42,
                          "best_macro_f1": 0.81, "best_epoch": 7,
                          "last_epoch": 7, "test_macro_f1": 0.79,
                          "status": "done", "error": None,
                          "log_path": ""})
    rows = collect_runs(str(root))
    assert len(rows) == 2
    ids = {r["run_id"] for r in rows}
    assert ids == {"gat_s42_a", "sage_s42_b"}


def test_collect_runs_ranks_by_best_macro_f1_desc(make_run_dir, tmp_path):
    root = tmp_path / "runs"
    for rid, f1 in [("low", 0.30), ("high", 0.90), ("mid", 0.60)]:
        make_run_dir(rid, name="n", root=root,
                     metrics={"run_id": rid, "name": "n", "conv_type": "gat",
                              "seed": 42, "best_macro_f1": f1, "best_epoch": 1,
                              "last_epoch": 1, "test_macro_f1": f1,
                              "status": "done", "error": None, "log_path": ""})
    rows = collect_runs(str(root))
    f1s = [r["best_macro_f1"] for r in rows]
    assert f1s == sorted(f1s, reverse=True)
    assert rows[0]["run_id"] == "high"
    assert rows[-1]["run_id"] == "low"


def test_collect_runs_falls_back_to_parse_log(make_run_dir, tmp_path, done_log_text):
    # No metrics.json -> aggregate must parse train.log instead.
    root = tmp_path / "runs"
    make_run_dir("parsed", name="n", root=root, log_text=done_log_text, metrics=None)
    rows = collect_runs(str(root))
    assert len(rows) == 1
    assert rows[0]["best_macro_f1"] == 0.7654
    assert rows[0]["status"] == "done"


def test_collect_runs_handles_none_f1_in_ranking(make_run_dir, tmp_path, running_log_text):
    root = tmp_path / "runs"
    make_run_dir("running", name="n", root=root, log_text=running_log_text, metrics=None)
    make_run_dir("done", name="n", root=root,
                 metrics={"run_id": "done", "name": "n", "conv_type": "gat",
                          "seed": 42, "best_macro_f1": 0.5, "best_epoch": 1,
                          "last_epoch": 1, "test_macro_f1": 0.5,
                          "status": "done", "error": None, "log_path": ""})
    rows = collect_runs(str(root))  # must not raise on None f1
    assert len(rows) == 2
    # the run with a real f1 ranks above the None one
    assert rows[0]["run_id"] == "done"


def test_to_markdown_contains_runs(make_run_dir, tmp_path):
    root = tmp_path / "runs"
    make_run_dir("gat_s42_a", name="n", root=root,
                 metrics={"run_id": "gat_s42_a", "name": "n", "conv_type": "gat",
                          "seed": 42, "best_macro_f1": 0.70, "best_epoch": 3,
                          "last_epoch": 3, "test_macro_f1": 0.68,
                          "status": "done", "error": None, "log_path": ""})
    rows = collect_runs(str(root))
    md = to_markdown(rows)
    assert isinstance(md, str)
    assert "gat_s42_a" in md


def test_to_csv_writes_file(make_run_dir, tmp_path):
    root = tmp_path / "runs"
    make_run_dir("gat_s42_a", name="n", root=root,
                 metrics={"run_id": "gat_s42_a", "name": "n", "conv_type": "gat",
                          "seed": 42, "best_macro_f1": 0.70, "best_epoch": 3,
                          "last_epoch": 3, "test_macro_f1": 0.68,
                          "status": "done", "error": None, "log_path": ""})
    rows = collect_runs(str(root))
    out = tmp_path / "summary.csv"
    to_csv(rows, str(out))
    assert out.exists()
    content = out.read_text()
    assert "gat_s42_a" in content


def test_collect_runs_empty_root(tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    rows = collect_runs(str(empty))
    assert rows == []
