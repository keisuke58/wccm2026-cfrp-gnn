"""expkit.aggregate — collect per-run metrics and render leaderboards.

PURE PYTHON (no torch / CUDA). pandas is imported lazily (only ``to_dataframe``
needs it) so that ``collect_runs`` / ``to_markdown`` / ``to_csv`` work on a bare
CPU box without pandas installed.

A "run-dir" is any directory containing either:
  * ``metrics.json``  (preferred — written by the runner via metrics.write_metrics_json), or
  * ``train.log``     (fallback — parsed on the fly via metrics.parse_log).

The metrics dict schema (see expkit.metrics) is::

    {run_id, name, conv_type, seed, best_macro_f1, best_epoch, last_epoch,
     test_macro_f1, status, error, log_path}

``collect_runs`` walks ``root`` recursively, gathers one dict per run-dir and
returns them ranked by ``best_macro_f1`` (descending, None last).

CLI::

    python -m expkit.aggregate <root>     # prints the ranked markdown table
"""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Optional

# Column order for tabular output (dataframe / markdown / csv).
COLUMNS: List[str] = [
    "run_id",
    "name",
    "conv_type",
    "seed",
    "best_macro_f1",
    "best_epoch",
    "last_epoch",
    "test_macro_f1",
    "status",
    "error",
    "log_path",
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _f1_sort_key(row: Dict[str, Any]) -> float:
    """Sort key for ranking: higher best_macro_f1 first, None/invalid last."""
    v = row.get("best_macro_f1")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("-inf")


def _is_run_dir(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "metrics.json")) or os.path.isfile(
        os.path.join(path, "train.log")
    )


def _row_for_run_dir(run_dir: str) -> Optional[Dict[str, Any]]:
    """Build one metrics row for a single run-dir.

    Prefers an existing ``metrics.json``; otherwise parses ``train.log``.
    Returns ``None`` if the directory is not actually a run-dir.
    """
    # Imported lazily so that merely importing expkit.aggregate does not require
    # expkit.metrics to exist yet (Contract/Build phase ordering).
    from . import metrics as _metrics

    metrics_json = os.path.join(run_dir, "metrics.json")
    if os.path.isfile(metrics_json):
        d = _metrics.read_metrics_json(run_dir)
        if d:
            return d

    log_path = os.path.join(run_dir, "train.log")
    if os.path.isfile(log_path):
        return _metrics.parse_log(log_path)

    return None


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def collect_runs(root: str) -> List[Dict[str, Any]]:
    """Scan ``root`` recursively for run-dirs and return ranked metric rows.

    A run-dir is one holding ``metrics.json`` (preferred) or ``train.log``
    (parsed via ``metrics.parse_log``). Rows are ranked by ``best_macro_f1``
    descending (None last).
    """
    rows: List[Dict[str, Any]] = []
    if not root or not os.path.isdir(root):
        return rows

    for dirpath, _dirnames, _filenames in os.walk(root):
        if _is_run_dir(dirpath):
            row = _row_for_run_dir(dirpath)
            if row is not None:
                rows.append(row)

    rows.sort(key=_f1_sort_key, reverse=True)
    return rows


def to_dataframe(rows: List[Dict[str, Any]]):
    """Return a pandas DataFrame of the rows (ranked order preserved).

    Imported lazily; only this function pulls in pandas.
    """
    import pandas as pd

    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.DataFrame(rows)
    # Stable, known column order; keep any extra keys at the end.
    ordered = [c for c in COLUMNS if c in df.columns]
    extras = [c for c in df.columns if c not in COLUMNS]
    return df[ordered + extras]


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        # Compact but informative for F1 scores.
        return f"{value:.4f}"
    return str(value)


def to_markdown(rows: List[Dict[str, Any]], columns: Optional[List[str]] = None) -> str:
    """Render rows as a GitHub-flavoured markdown table, ranked by F1 desc.

    Adds a leading ``rank`` column. Pure string formatting (no pandas).
    """
    cols = columns or COLUMNS
    ranked = sorted(rows, key=_f1_sort_key, reverse=True)

    header = ["rank"] + list(cols)
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for i, row in enumerate(ranked, start=1):
        cells = [str(i)] + [_fmt(row.get(c)) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def to_csv(rows: List[Dict[str, Any]], path: str) -> str:
    """Write rows to ``path`` as CSV (ranked by F1 desc). Returns ``path``."""
    ranked = sorted(rows, key=_f1_sort_key, reverse=True)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in ranked:
            writer.writerow({c: row.get(c) for c in COLUMNS})
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _main(argv: Optional[List[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m expkit.aggregate",
        description="Scan an experiment output root and print a ranked markdown leaderboard.",
    )
    p.add_argument("root", help="output root to scan for run-dirs")
    p.add_argument("--csv", default=None, help="also write the ranked rows to this CSV path")
    args = p.parse_args(argv)

    rows = collect_runs(args.root)
    print(to_markdown(rows))
    if args.csv:
        to_csv(rows, args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
