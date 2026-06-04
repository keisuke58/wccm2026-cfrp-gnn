"""expkit.metrics — parse train.py logs into a machine-readable metrics dict.

PURE PYTHON: no torch / CUDA import here so this loads & runs on CPU (tests too).

train.py does NOT emit a summary json, so we regex its known print formats:
  * "Best Macro F1: X.XXXX"                      -> best_macro_f1
  * "=== Epoch N ==="                            -> epoch markers (last_epoch)
  * per-epoch "Macro F1: X.XXXX"                 -> tracks epoch with best F1
  * checkpoint filename "...MacroF1BestX.XXXX..." -> best_macro_f1 fallback + best_epoch
  * a Traceback / "Error" -> status='failed' with the error text
  * a final "Test ... Macro F1: X.XXXX"          -> test_macro_f1

Status logic:
  * 'failed'  if a Python Traceback appears in the log
  * 'done'    if a "Best Macro F1:" summary line was printed (training finished)
  * 'running' otherwise (log exists but no terminal summary / traceback)
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

METRICS_FILENAME = "metrics.json"

# ------------------------------------------------------------------ #
# regex patterns matching the train.py print formats
# ------------------------------------------------------------------ #
_RE_BEST_F1 = re.compile(r"Best\s+Macro\s+F1:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
# train.py prints "=== Epoch N/total ===" (the "/total" suffix is optional here)
_RE_EPOCH = re.compile(r"===\s*Epoch\s+(\d+)(?:\s*/\s*\d+)?\s*===", re.IGNORECASE)
# per-epoch line: "Macro F1: 0.1234" but NOT "Best Macro F1:" / "Test ... Macro F1:"
_RE_EPOCH_F1 = re.compile(r"(?<![A-Za-z])Macro\s+F1:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
# checkpoint filename token: ..._Best_Epoch12_...MacroF1Best0.8123_....pth
_RE_CKPT_EPOCH = re.compile(r"Best_Epoch(\d+)", re.IGNORECASE)
_RE_CKPT_F1 = re.compile(r"MacroF1Best([0-9]*\.?[0-9]+)", re.IGNORECASE)
# test-section macro F1. train.py's test block prints
#   " - Macro Precision: .., Macro Recall: .., Macro F1-Score: 0.7123"
# and "[INFO] Macro F1 Score: 0.7123 (logged for reference)". Both use
# "Macro F1" followed by an optional "-Score"/" Score" then the value.
_RE_TEST_F1 = re.compile(
    r"Macro\s+F1[-\s]Score:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE
)
_RE_TRACEBACK = re.compile(r"^Traceback \(most recent call last\):", re.MULTILINE)


def _to_float(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _extract_error(text: str) -> Optional[str]:
    """Return the last traceback block / final error line, trimmed."""
    if not text:
        return None
    matches = list(_RE_TRACEBACK.finditer(text))
    if matches:
        block = text[matches[-1].start():]
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if lines:
            # last non-empty line is usually "ExceptionType: message"
            tail = lines[-1].strip()
            return tail[:500]
    return None


def parse_log(log_path: str) -> Dict[str, Any]:
    """Parse a train.log into the metrics dict (partial; runner fills meta).

    Returns keys: best_macro_f1, best_epoch, last_epoch, test_macro_f1,
    status, error, log_path. Meta keys (run_id/name/conv_type/seed) are left
    None here and merged by the runner/aggregate layer.
    """
    out: Dict[str, Any] = {
        "run_id": None,
        "name": None,
        "conv_type": None,
        "seed": None,
        "best_macro_f1": None,
        "best_epoch": None,
        "last_epoch": None,
        "test_macro_f1": None,
        "status": "running",
        "error": None,
        "log_path": log_path,
    }

    if not log_path or not os.path.isfile(log_path):
        out["status"] = "failed"
        out["error"] = f"log not found: {log_path}"
        return out

    with open(log_path, "r", errors="replace") as fh:
        text = fh.read()

    # --- last epoch marker ---
    epochs: List[int] = [int(m.group(1)) for m in _RE_EPOCH.finditer(text)]
    if epochs:
        out["last_epoch"] = epochs[-1]

    # --- best macro F1 (prefer explicit summary, else checkpoint filename) ---
    best_matches = _RE_BEST_F1.findall(text)
    has_best_summary = bool(best_matches)
    best_val: Optional[float] = None
    if best_matches:
        best_val = _to_float(best_matches[-1])

    ckpt_f1 = _RE_CKPT_F1.findall(text)
    ckpt_epoch = _RE_CKPT_EPOCH.findall(text)
    if best_val is None and ckpt_f1:
        best_val = _to_float(ckpt_f1[-1])
    out["best_macro_f1"] = best_val

    # --- best epoch: from checkpoint filename if present, else infer from
    #     the per-epoch Macro F1 line that equals best_val ---
    if ckpt_epoch:
        out["best_epoch"] = int(ckpt_epoch[-1])
    else:
        # walk epochs + per-epoch F1 together to find argmax epoch
        best_epoch = _infer_best_epoch(text)
        if best_epoch is not None:
            out["best_epoch"] = best_epoch

    # --- test macro F1 ---
    test_matches = _RE_TEST_F1.findall(text)
    if test_matches:
        out["test_macro_f1"] = _to_float(test_matches[-1])

    # --- status / error ---
    error = _extract_error(text)
    if error is not None:
        out["status"] = "failed"
        out["error"] = error
    elif has_best_summary:
        out["status"] = "done"
    else:
        out["status"] = "running"

    return out


def _infer_best_epoch(text: str) -> Optional[int]:
    """Scan interleaved '=== Epoch N ===' and 'Macro F1: x' lines; return the
    epoch number with the maximum per-epoch macro F1."""
    cur_epoch: Optional[int] = None
    best_f1 = float("-inf")
    best_ep: Optional[int] = None
    for line in text.splitlines():
        em = _RE_EPOCH.search(line)
        if em:
            cur_epoch = int(em.group(1))
            continue
        if "best" in line.lower() or re.search(r"Test", line, re.IGNORECASE):
            continue
        fm = _RE_EPOCH_F1.search(line)
        if fm and cur_epoch is not None:
            f1 = _to_float(fm.group(1))
            if f1 is not None and f1 > best_f1:
                best_f1 = f1
                best_ep = cur_epoch
    return best_ep


# ------------------------------------------------------------------ #
# json IO
# ------------------------------------------------------------------ #
def write_metrics_json(run_dir: str, d: Dict[str, Any]) -> str:
    """Write the metrics dict to {run_dir}/metrics.json and return the path."""
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, METRICS_FILENAME)
    with open(path, "w") as fh:
        json.dump(d, fh, indent=2, sort_keys=False)
        fh.write("\n")
    return path


def read_metrics_json(run_dir: str) -> Optional[Dict[str, Any]]:
    """Read {run_dir}/metrics.json -> dict, or None if absent/unreadable."""
    path = os.path.join(run_dir, METRICS_FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None
