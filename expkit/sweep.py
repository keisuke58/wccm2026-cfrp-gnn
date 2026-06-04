"""expkit.sweep — cartesian-product expansion of ExpConfigs + port planning.

PURE PYTHON: no torch / CUDA import here so sweeps (and tests) run on CPU.

A *grid* is a plain dict with two optional sections:

    base: {<ExpConfig field>: value, ...}     # baseline overrides atop ExpConfig()
    grid: {<ExpConfig field>: [v1, v2, ...]}  # fields to vary (cartesian product)

Plus two optional knobs that live alongside the grid mapping:

    name:        base name; per-combo names are derived as ``<name>__<field>-<val>...``
    master_port: starting master port; successive combos get successive ports.

``expand(grid)`` returns a list of fully-materialised :class:`ExpConfig` objects,
one per point in the cartesian product of the varying fields, each merged atop the
base. Non-varying fields fall back to the base (then to ExpConfig defaults).

Special-case: a per-combo override map keyed ``per_conv_type`` (or, generically,
``per_<field>``) lets a sweep set field values that depend on the swept value, e.g.
batch_size per conv_type. See ``configs/ablation_arch.yaml``.

CLI::

    python -m expkit.sweep --config configs/ablation_arch.yaml --dry-run
"""

from __future__ import annotations

import argparse
import itertools
from typing import Any, Dict, List

import yaml

from .config import ExpConfig

# meta keys in the grid file that are NOT ExpConfig fields / grid axes
_META_KEYS = {"base", "grid", "name", "master_port"}


def plan_ports(base_port: int, n: int) -> List[int]:
    """Return ``n`` successive master ports starting at ``base_port``."""
    return [base_port + i for i in range(n)]


def _combo_suffix(combo: Dict[str, Any]) -> str:
    """Compact, filesystem-safe suffix describing a varying-field combination."""
    parts = []
    for k in sorted(combo):
        v = combo[k]
        sval = str(v).replace("/", "-").replace(" ", "")
        parts.append(f"{k}-{sval}")
    return "__".join(parts)


def expand(grid: Dict[str, Any]) -> List[ExpConfig]:
    """Expand a grid mapping into a list of ExpConfig (cartesian over ``grid``).

    Parameters
    ----------
    grid : dict
        May contain ``base`` (dict of overrides), ``grid`` (dict field->list),
        ``name`` (str), ``master_port`` (int) and any ``per_<field>`` override
        maps. Bare top-level ExpConfig fields are also accepted and folded into
        ``base`` for convenience.
    """
    grid = dict(grid or {})

    base_overrides: Dict[str, Any] = dict(grid.get("base") or {})
    varying: Dict[str, List[Any]] = dict(grid.get("grid") or {})
    base_name: str = grid.get("name", base_overrides.get("name", "exp"))
    start_port: int = int(
        grid.get("master_port", base_overrides.get("master_port", 29500))
    )

    # per-<field> conditional override maps: {field: {swept_value: {f: v, ...}}}
    per_field_maps: Dict[str, Dict[Any, Dict[str, Any]]] = {}
    for k, v in grid.items():
        if k.startswith("per_") and isinstance(v, dict):
            per_field_maps[k[len("per_"):]] = v

    # bare top-level ExpConfig fields (not meta, not per_*, not varying) -> base
    known = set(ExpConfig._field_names())
    for k, v in grid.items():
        if k in _META_KEYS or k.startswith("per_"):
            continue
        if k in varying:
            continue
        if k in known:
            base_overrides.setdefault(k, v)

    # cartesian product over the varying axes (deterministic key order)
    axis_keys = list(varying.keys())
    axis_vals = [list(varying[k]) for k in axis_keys]
    combos = list(itertools.product(*axis_vals)) if axis_keys else [()]

    ports = plan_ports(start_port, len(combos))

    configs: List[ExpConfig] = []
    for i, combo_vals in enumerate(combos):
        combo = dict(zip(axis_keys, combo_vals))

        merged: Dict[str, Any] = dict(base_overrides)
        merged.update(combo)

        # apply per-<field> conditional overrides for the swept values
        for field_name, table in per_field_maps.items():
            swept = merged.get(field_name)
            if swept is not None and swept in table:
                merged.update(table[swept])

        # name + port (combo values win over any base-supplied name/port)
        if combo:
            merged["name"] = f"{base_name}__{_combo_suffix(combo)}"
        else:
            merged["name"] = base_name
        merged["master_port"] = ports[i]

        configs.append(ExpConfig.from_dict(merged))

    return configs


def load_grid(path: str) -> Dict[str, Any]:
    """Load a grid/config YAML into a plain dict."""
    with open(path, "r") as fh:
        return yaml.safe_load(fh) or {}


def _format_plan(configs: List[ExpConfig]) -> str:
    lines = [f"Sweep plan: {len(configs)} run(s)"]
    for i, cfg in enumerate(configs):
        lines.append(
            f"  [{i:02d}] name={cfg.name}  conv_type={cfg.conv_type}  "
            f"gpu={cfg.gpu}  port={cfg.master_port}  "
            f"batch_size={cfg.batch_size}  epochs={cfg.epochs}  "
            f"include_ndf={cfg.include_ndf}  ndf_count={cfg.ndf_count}  "
            f"defect_cap={cfg.defect_cap}  seed={cfg.seed}"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m expkit.sweep",
        description="Expand a sweep config and (dry-run) print the plan.",
    )
    p.add_argument("--config", required=True, help="path to a sweep YAML")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print the expanded plan without launching anything",
    )
    args = p.parse_args(argv)

    grid = load_grid(args.config)
    configs = expand(grid)

    if args.dry_run:
        print(_format_plan(configs))
        return 0

    # Without --dry-run the sweep CLI still only plans; actual launching is the
    # runner's job (kept out of the pure-python core / tests).
    print(_format_plan(configs))
    print("\n(--dry-run not set: launching is delegated to expkit.runner.run)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
