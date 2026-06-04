"""Tests that ExpConfig defaults reproduce the BALANCED 5000:5000 NDF recipe.

This is the central reproducibility contract of the framework: with no overrides,
a config must emit the balanced-training CLI (include_ndf + ndf_count 5000 +
defect_cap 5000) — the data-recipe that the ablation found dominates architecture.
"""
from __future__ import annotations

from expkit.config import ExpConfig


def test_defaults_are_balanced_recipe(base_cfg):
    assert base_cfg.include_ndf is True
    assert base_cfg.ndf_count == 5000
    assert base_cfg.defect_cap == 5000
    assert base_cfg.seed == 42


def test_balanced_recipe_in_cli(base_cfg):
    a = base_cfg.to_cli_args()
    assert "--include_ndf" in a
    # 5000:5000 balance encoded as explicit caps
    assert "--ndf_count" in a and a[a.index("--ndf_count") + 1] == "5000"
    assert "--defect_cap" in a and a[a.index("--defect_cap") + 1] == "5000"
    assert "--seed" in a and a[a.index("--seed") + 1] == "42"


def test_recipe_off_drops_caps_and_flag():
    # The "architecture-only / no-recipe" baseline: NDF disabled, caps cleared.
    off = ExpConfig(include_ndf=False, ndf_count=0, defect_cap=0)
    a = off.to_cli_args()
    assert "--include_ndf" not in a
    assert "--ndf_count" not in a
    assert "--defect_cap" not in a
    # seed still always emitted for reproducibility
    assert "--seed" in a


def test_seed_always_emitted_for_reproducibility():
    for s in (0, 1, 42, 2026):
        a = ExpConfig(seed=s).to_cli_args()
        assert "--seed" in a
        assert a[a.index("--seed") + 1] == str(s)


def test_recipe_survives_yaml_roundtrip(tmp_path, base_cfg):
    p = tmp_path / "recipe.yaml"
    base_cfg.to_yaml(str(p))
    loaded = ExpConfig.from_yaml(str(p))
    assert loaded.include_ndf is True
    assert loaded.ndf_count == 5000
    assert loaded.defect_cap == 5000
