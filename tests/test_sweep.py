"""Tests for expkit.sweep.expand / plan_ports (pure-python, no torch).

The real ``expand(grid)`` takes a single grid mapping whose ``base`` key holds
baseline overrides and whose ``grid`` key holds the field->list axes to vary
(cartesian product). ``name``/``master_port`` are meta knobs alongside them.
"""
from __future__ import annotations

from expkit.config import ExpConfig
from expkit.sweep import expand, plan_ports


# --------------------------------------------------------------------------- #
# expand — cartesian product over varying fields atop a base
# --------------------------------------------------------------------------- #
def test_expand_single_axis():
    grid = {"base": {"batch_size": 64}, "grid": {"conv_type": ["gat", "sage", "pna"]}}
    cfgs = expand(grid)
    assert len(cfgs) == 3
    assert {c.conv_type for c in cfgs} == {"gat", "sage", "pna"}
    # non-varying base fields are preserved
    assert all(c.batch_size == 64 for c in cfgs)
    assert all(isinstance(c, ExpConfig) for c in cfgs)


def test_expand_cartesian_two_axes():
    grid = {"grid": {"conv_type": ["gat", "sage"], "seed": [1, 2, 3]}}
    cfgs = expand(grid)
    assert len(cfgs) == 6
    combos = {(c.conv_type, c.seed) for c in cfgs}
    assert combos == {
        ("gat", 1), ("gat", 2), ("gat", 3),
        ("sage", 1), ("sage", 2), ("sage", 3),
    }


def test_expand_empty_grid_returns_base_copy():
    grid = {"base": {"conv_type": "pna", "seed": 9}}
    cfgs = expand(grid)
    assert len(cfgs) == 1
    assert cfgs[0].conv_type == "pna"
    assert cfgs[0].seed == 9


def test_expand_empty_mapping_returns_one_default():
    cfgs = expand({})
    assert len(cfgs) == 1
    assert cfgs[0].hidden_channels == 16


def test_expand_bare_field_folded_into_base():
    # bare top-level ExpConfig field (not under 'base') is accepted as a base override
    grid = {"hidden_channels": 32, "grid": {"conv_type": ["gat", "gatv2"]}}
    cfgs = expand(grid)
    assert len(cfgs) == 2
    assert all(c.hidden_channels == 32 for c in cfgs)


def test_expand_assigns_unique_consecutive_ports():
    grid = {"master_port": 30000, "grid": {"conv_type": ["gat", "sage", "pna"]}}
    cfgs = expand(grid)
    ports = [c.master_port for c in cfgs]
    assert ports == [30000, 30001, 30002]


def test_expand_per_field_conditional_override():
    # per_conv_type lets batch_size depend on the swept conv_type
    grid = {
        "grid": {"conv_type": ["gat", "pna", "meshgraphnet"]},
        "per_conv_type": {
            "pna": {"batch_size": 8},
            "meshgraphnet": {"batch_size": 16},
        },
    }
    cfgs = {c.conv_type: c for c in expand(grid)}
    assert cfgs["pna"].batch_size == 8
    assert cfgs["meshgraphnet"].batch_size == 16
    assert cfgs["gat"].batch_size == 64  # default


def test_expand_derives_per_combo_names():
    grid = {"name": "arch", "grid": {"conv_type": ["gat", "sage"]}}
    names = [c.name for c in expand(grid)]
    assert all("arch" in n for n in names)
    assert len(set(names)) == 2  # unique per combo


# --------------------------------------------------------------------------- #
# plan_ports — deterministic, unique master ports
# --------------------------------------------------------------------------- #
def test_plan_ports_count_and_uniqueness():
    ports = plan_ports(29500, 5)
    assert len(ports) == 5
    assert len(set(ports)) == 5
    assert ports[0] == 29500


def test_plan_ports_consecutive():
    assert plan_ports(30000, 3) == [30000, 30001, 30002]


def test_plan_ports_zero():
    assert plan_ports(29500, 0) == []
