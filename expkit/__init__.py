"""expkit — config-driven experiment framework for the WCCM CFRP-GNN train.py.

Public API:
    ExpConfig    - dataclass holding all train.py flags + meta/NDF/balance fields
    run          - launch (or dry-run) one experiment
    parse_log    - parse a train.log into a metrics dict
    collect_runs - scan an output root and gather per-run metrics

The config/metrics/aggregate/sweep layers are PURE PYTHON (no torch); only the
runner actually launches the GPU subprocess. Heavier submodules (runner, metrics,
aggregate) are imported lazily so that ``from expkit import ExpConfig`` works even
before those files exist or without their optional deps (pandas, etc.).
"""

from .config import ExpConfig

__all__ = ["ExpConfig", "run", "parse_log", "collect_runs"]


def run(*args, **kwargs):
    """Lazy proxy to :func:`expkit.runner.run`."""
    from .runner import run as _run

    return _run(*args, **kwargs)


def parse_log(*args, **kwargs):
    """Lazy proxy to :func:`expkit.metrics.parse_log`."""
    from .metrics import parse_log as _parse_log

    return _parse_log(*args, **kwargs)


def collect_runs(*args, **kwargs):
    """Lazy proxy to :func:`expkit.aggregate.collect_runs`."""
    from .aggregate import collect_runs as _collect_runs

    return _collect_runs(*args, **kwargs)
