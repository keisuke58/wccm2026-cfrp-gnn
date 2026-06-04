"""expkit.runner — launch (or dry-run) one experiment and capture its outputs.

run(cfg, output_root, run_id, train_py, dry_run=False) manages a single run dir:

    {output_root}/{cfg.name}/{run_id}/
        config.yaml   snapshot of the resolved ExpConfig
        git.txt       `git rev-parse HEAD` + dirty/clean state
        cmd.txt       the full launch command + env overrides
        train.log     captured stdout/stderr of the torchrun subprocess
        metrics.json  parse_log(train.log) result (run_id/name/conv_type/seed stamped)

The torchrun command is built from cfg.torchrun_cmd(train_py) and the launch env
from cfg.env() (merged over os.environ). dry_run=True builds & snapshots the cmd
but does NOT launch a subprocess (so tests run on CPU with no GPU), returning the
cmd in the result dict with metrics=None.

This module imports subprocess but NOT torch — the heavy work happens in the
spawned train.py process, not here.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any, Dict, List, Optional

from .config import ExpConfig
from . import metrics as _metrics

# default train.py location (repo root next to the expkit package)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TRAIN_PY = os.path.join(_REPO_ROOT, "train.py")

CONFIG_FILENAME = "config.yaml"
GIT_FILENAME = "git.txt"
CMD_FILENAME = "cmd.txt"
LOG_FILENAME = "train.log"


# ------------------------------------------------------------------ #
# snapshot helpers
# ------------------------------------------------------------------ #
def _git_snapshot(repo_dir: str) -> str:
    """Return 'HEAD <sha> (dirty|clean)\\n<status>' for the repo, best-effort."""
    lines: List[str] = []
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        sha = head.stdout.strip() or "UNKNOWN"
    except (OSError, ValueError):
        sha = "UNKNOWN"

    dirty = False
    status_txt = ""
    try:
        st = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        status_txt = st.stdout
        dirty = bool(status_txt.strip())
    except (OSError, ValueError):
        status_txt = ""

    lines.append(f"HEAD {sha} ({'dirty' if dirty else 'clean'})")
    if status_txt.strip():
        lines.append("--- git status --porcelain ---")
        lines.append(status_txt.rstrip("\n"))
    return "\n".join(lines) + "\n"


def _write_cmd_txt(path: str, cmd: List[str], env_overrides: Dict[str, str]) -> None:
    parts: List[str] = []
    parts.append("# env overrides")
    for k, v in env_overrides.items():
        parts.append(f"{k}={v}")
    parts.append("")
    parts.append("# command")
    parts.append(" ".join(cmd))
    parts.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(parts))


def tensorboard_cmd(logdir: str, port: int = 6006) -> List[str]:
    """Build the `tensorboard` launch command for a given log dir.

    Point ``logdir`` at a single run's ``tb/`` or at ``output_root`` to
    aggregate all runs in one TensorBoard instance.
    """
    return ["tensorboard", "--logdir", logdir, "--port", str(port)]


# ------------------------------------------------------------------ #
# main entry point
# ------------------------------------------------------------------ #
def run(
    cfg: ExpConfig,
    output_root: str,
    run_id: str,
    train_py: str = DEFAULT_TRAIN_PY,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Launch (or dry-run) one experiment.

    Parameters
    ----------
    cfg : ExpConfig
        Fully resolved config; drives the train.py CLI, env, and torchrun cmd.
    output_root : str
        Base directory for all runs; the run dir is {output_root}/{cfg.name}/{run_id}.
    run_id : str
        Caller-supplied run id (e.g. f"{conv_type}_s{seed}_{stamp}"). The framework
        does NOT call wall-clock itself — the caller bakes the timestamp into run_id.
    train_py : str
        Path to train.py to launch (defaults to the repo's train.py).
    dry_run : bool
        If True, snapshot config/git/cmd and return the cmd WITHOUT launching a
        subprocess (no GPU needed); metrics is None.

    Returns
    -------
    dict with keys: run_dir, cmd, metrics (dict or None).
    """
    # --- create run dir ---
    run_dir = os.path.join(output_root, cfg.name, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # --- per-run TensorBoard log dir (don't overwrite an explicit tb_dir) ---
    if not cfg.tb_dir:
        cfg.tb_dir = os.path.join(run_dir, "tb")

    # --- build cmd + env ---
    cmd = cfg.torchrun_cmd(train_py)
    env_overrides = cfg.env()

    # --- snapshots ---
    cfg.to_yaml(os.path.join(run_dir, CONFIG_FILENAME))

    train_py_dir = os.path.dirname(os.path.abspath(train_py)) or os.getcwd()
    repo_dir = train_py_dir if os.path.isdir(train_py_dir) else _REPO_ROOT
    try:
        git_txt = _git_snapshot(repo_dir)
    except Exception as exc:  # pragma: no cover - defensive
        git_txt = f"git snapshot failed: {exc}\n"
    with open(os.path.join(run_dir, GIT_FILENAME), "w") as fh:
        fh.write(git_txt)

    cmd_path = os.path.join(run_dir, CMD_FILENAME)
    _write_cmd_txt(cmd_path, cmd, env_overrides)

    result: Dict[str, Any] = {"run_dir": run_dir, "cmd": cmd, "metrics": None}

    # --- dry run: stop before launching ---
    if dry_run:
        return result

    # --- launch subprocess, capturing stdout+stderr -> train.log ---
    env = dict(os.environ)
    env.update(env_overrides)
    log_path = os.path.join(run_dir, LOG_FILENAME)

    status: Optional[str] = None
    error: Optional[str] = None
    with open(log_path, "w") as logf:
        try:
            proc = subprocess.run(
                cmd,
                cwd=repo_dir,
                env=env,
                stdout=logf,
                stderr=subprocess.STDOUT,
                check=False,
            )
            returncode = proc.returncode
        except (OSError, ValueError) as exc:
            logf.write(f"\n[runner] failed to launch: {exc}\n")
            returncode = -1
            status = "failed"
            error = f"launch error: {exc}"

    # --- parse log -> metrics.json (stamp identity fields) ---
    md = _metrics.parse_log(log_path)
    md["run_id"] = run_id
    md["name"] = cfg.name
    md["conv_type"] = cfg.conv_type
    if md.get("seed") is None:
        md["seed"] = cfg.seed

    # let an explicit launch failure / nonzero exit override the parsed status
    if status == "failed":
        md["status"] = "failed"
        if md.get("error") is None:
            md["error"] = error
    elif returncode != 0 and md.get("status") != "failed":
        md["status"] = "failed"
        if md.get("error") is None:
            md["error"] = f"train.py exited with code {returncode}"

    _metrics.write_metrics_json(run_dir, md)
    result["metrics"] = md
    return result
