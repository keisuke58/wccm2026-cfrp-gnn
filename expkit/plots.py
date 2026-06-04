"""expkit.plots — publication figures for the WCCM CFRP-GNN experiments.

Uses the matplotlib ``Agg`` backend (no display / no GUI) so it runs headless on
the compute nodes and in CI. Three entry points:

    f1_curve(log_paths, out_png)      training Macro-F1 vs epoch (one line per run)
    arch_bars(rows, out_png)          best Macro-F1 per run as a ranked bar chart
    confusion_from_csv(csv, out_png)  best-effort confusion-matrix heatmap

Parsing of train.log files is delegated to ``expkit.metrics`` (no torch needed).
"""

from __future__ import annotations

import csv as _csv
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")  # headless backend; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402

from . import metrics as _metrics  # noqa: E402

LogPath = Union[str, "os.PathLike[str]"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _ensure_parent(out_png: str) -> None:
    parent = os.path.dirname(os.path.abspath(out_png))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _label_for(log_path: str) -> str:
    """Human-friendly series label: prefer the run-dir name over the file name."""
    run_dir = os.path.dirname(os.path.abspath(log_path))
    name = os.path.basename(run_dir)
    return name or os.path.basename(log_path)


def _parse_epoch_f1(text: str) -> List[Tuple[int, float]]:
    """Return ordered (epoch, macro_f1) pairs from a train.log body.

    Reuses the compiled regexes from :mod:`expkit.metrics` (single source of
    truth for the print formats) so plots stay in sync with the parser. "Best
    Macro F1:" / "Test ... Macro F1:" summary lines are skipped; the first
    per-epoch F1 inside an "=== Epoch N ===" block is taken for that epoch.
    """
    pairs: List[Tuple[int, float]] = []
    cur_epoch: Optional[int] = None
    seen_for_epoch = False
    for line in text.splitlines():
        em = _metrics._RE_EPOCH.search(line)
        if em:
            cur_epoch = int(em.group(1))
            seen_for_epoch = False
            continue
        # skip summary / test lines (same guard metrics._infer_best_epoch uses)
        if "best" in line.lower() or "test" in line.lower():
            continue
        fm = _metrics._RE_EPOCH_F1.search(line)
        if fm and cur_epoch is not None and not seen_for_epoch:
            f1 = _metrics._to_float(fm.group(1))
            if f1 is not None:
                pairs.append((cur_epoch, f1))
                seen_for_epoch = True
    return pairs


# --------------------------------------------------------------------------- #
# f1_curve
# --------------------------------------------------------------------------- #
def f1_curve(
    log_paths: Union[LogPath, Sequence[LogPath]],
    out_png: str,
    *,
    labels: Optional[Sequence[str]] = None,
    title: str = "Training Macro-F1 vs Epoch",
    dpi: int = 150,
) -> str:
    """Plot Macro-F1 vs epoch for one or more train.log files.

    Epoch/F1 pairs are parsed via :func:`expkit.metrics.parse_epoch_f1`. Runs
    with no parseable curve are skipped (but still leave an empty-but-valid
    figure if *none* parse). Returns the output path.
    """
    if isinstance(log_paths, (str, os.PathLike)):
        paths: List[str] = [os.fspath(log_paths)]
    else:
        paths = [os.fspath(p) for p in log_paths]

    _ensure_parent(out_png)
    fig, ax = plt.subplots(figsize=(8, 5))

    plotted = 0
    for i, path in enumerate(paths):
        try:
            with open(path, "r", errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue
        pairs = _parse_epoch_f1(text)
        if not pairs:
            continue
        xs = [e for e, _ in pairs]
        ys = [f for _, f in pairs]
        if labels is not None and i < len(labels):
            lbl = labels[i]
        else:
            lbl = _label_for(path)
        ax.plot(xs, ys, marker="", linewidth=1.4, label=lbl)
        # mark the best point
        b = max(range(len(ys)), key=lambda k: ys[k])
        ax.scatter([xs[b]], [ys[b]], s=18, zorder=5)
        plotted += 1

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Macro F1")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend(fontsize=8, loc="lower right")
    else:
        ax.text(
            0.5, 0.5, "no parseable F1 curves",
            ha="center", va="center", transform=ax.transAxes,
        )
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------------- #
# arch_bars
# --------------------------------------------------------------------------- #
def _row_get(row: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return default


def arch_bars(
    rows: Sequence[Mapping[str, Any]],
    out_png: str,
    *,
    value_key: str = "best_macro_f1",
    title: str = "Best Macro-F1 by run",
    dpi: int = 150,
) -> str:
    """Ranked horizontal bar chart of best Macro-F1 per run.

    ``rows`` are metrics dicts (as produced by :mod:`expkit.metrics` /
    :mod:`expkit.aggregate`). Each bar is labelled by conv_type+run_id when
    available. Rows missing the value are dropped. Returns the output path.
    """
    items: List[Tuple[str, float]] = []
    for r in rows:
        v = _row_get(r, value_key)
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        conv = _row_get(r, "conv_type", default="")
        ident = _row_get(r, "run_id", "name", default="run")
        label = f"{conv}:{ident}" if conv else str(ident)
        items.append((label, fv))

    items.sort(key=lambda t: t[1], reverse=True)

    _ensure_parent(out_png)
    height = max(2.5, 0.45 * max(1, len(items)) + 1.0)
    fig, ax = plt.subplots(figsize=(9, height))

    if items:
        labels = [t[0] for t in items]
        vals = [t[1] for t in items]
        ypos = list(range(len(items)))
        bars = ax.barh(ypos, vals, color="#3b6ea5")
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()  # best (highest) on top
        ax.set_xlabel("Best Macro F1")
        for rect, v in zip(bars, vals):
            ax.text(
                rect.get_width() + 0.005,
                rect.get_y() + rect.get_height() / 2,
                f"{v:.4f}",
                va="center", fontsize=7,
            )
        ax.set_xlim(0, max(1.0, max(vals) * 1.12))
    else:
        ax.text(
            0.5, 0.5, "no rows with a value",
            ha="center", va="center", transform=ax.transAxes,
        )

    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------------- #
# confusion_from_csv (best-effort)
# --------------------------------------------------------------------------- #
def confusion_from_csv(
    csv_path: str,
    out_png: str,
    *,
    title: str = "Confusion matrix",
    normalize: bool = False,
    dpi: int = 150,
    annotate_max: int = 25,
) -> Optional[str]:
    """Best-effort confusion-matrix heatmap from a CSV.

    Accepts a square numeric CSV (with or without a header row / index column).
    Header / leading-label cells that are non-numeric are auto-detected and used
    as tick labels. Returns the output path, or None when the CSV is missing /
    unparseable (best-effort: never raises on bad input).
    """
    if not os.path.isfile(csv_path):
        return None

    try:
        with open(csv_path, "r", newline="", errors="replace") as fh:
            raw_rows = [r for r in _csv.reader(fh) if r and any(c.strip() for c in r)]
    except OSError:
        return None
    if not raw_rows:
        return None

    def _is_num(s: str) -> bool:
        try:
            float(s)
            return True
        except (TypeError, ValueError):
            return False

    col_labels: Optional[List[str]] = None
    row_labels: List[str] = []

    # detect a header row (first row has any non-numeric cell beyond col 0)
    first = raw_rows[0]
    if any(not _is_num(c) for c in first[1:]):
        col_labels = [c.strip() for c in first[1:]]
        body = raw_rows[1:]
    else:
        body = raw_rows

    matrix: List[List[float]] = []
    for r in body:
        cells = list(r)
        # detect a leading index/label column
        if cells and not _is_num(cells[0]) and len(cells) > 1:
            row_labels.append(cells[0].strip())
            cells = cells[1:]
        else:
            row_labels.append(str(len(matrix)))
        try:
            matrix.append([float(c) for c in cells])
        except (TypeError, ValueError):
            return None  # not a clean numeric matrix -> bail (best-effort)

    if not matrix or not matrix[0]:
        return None
    ncols = len(matrix[0])
    if any(len(row) != ncols for row in matrix):
        return None  # ragged

    import numpy as np

    arr = np.array(matrix, dtype=float)
    if normalize:
        with np.errstate(invalid="ignore", divide="ignore"):
            sums = arr.sum(axis=1, keepdims=True)
            arr = np.divide(arr, sums, out=np.zeros_like(arr), where=sums != 0)

    if col_labels is None:
        col_labels = [str(i) for i in range(ncols)]

    _ensure_parent(out_png)
    fig, ax = plt.subplots(figsize=(max(5, 0.5 * ncols + 2), max(5, 0.5 * len(arr) + 2)))
    im = ax.imshow(arr, cmap="Blues", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(ncols))
    ax.set_yticks(range(len(arr)))
    ax.set_xticklabels(col_labels[:ncols], rotation=90, fontsize=7)
    ax.set_yticklabels(row_labels[: len(arr)], fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    # annotate cells only for small matrices to keep figures legible
    if max(arr.shape) <= annotate_max:
        thresh = arr.max() / 2.0 if arr.size else 0.0
        fmt = "{:.2f}" if normalize else "{:.0f}"
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                ax.text(
                    j, i, fmt.format(arr[i, j]),
                    ha="center", va="center", fontsize=6,
                    color="white" if arr[i, j] > thresh else "black",
                )

    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return out_png
