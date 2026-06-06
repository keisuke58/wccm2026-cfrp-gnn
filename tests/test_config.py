"""Tests for expkit.config.ExpConfig (pure-python, no torch)."""
from __future__ import annotations

from expkit.config import (
    DEFAULT_DATA_BASE,
    LD_LIBRARY_PATH,
    TORCHRUN_BIN,
    ExpConfig,
)


# --------------------------------------------------------------------------- #
# defaults
# --------------------------------------------------------------------------- #
def test_defaults_match_spec(base_cfg):
    c = base_cfg
    assert c.hidden_channels == 16
    assert c.learning_rate == 0.002
    assert c.batch_size == 64
    assert c.epochs == 100
    assert c.patience == 60
    assert c.dropout == 0.1
    assert c.edge_drop_prob == 0.01
    assert c.conv_type == "gat"
    assert c.use_class_frequency_sampler is True
    assert c.group_purge_eval is True
    assert c.data_base == DEFAULT_DATA_BASE
    assert c.seed == 42
    # meta
    assert c.name == "exp"
    assert c.gpu == 0
    assert isinstance(c.master_port, int)


# --------------------------------------------------------------------------- #
# (de)serialization
# --------------------------------------------------------------------------- #
def test_from_dict_merges_over_defaults():
    c = ExpConfig.from_dict({"conv_type": "pna", "batch_size": 8})
    assert c.conv_type == "pna"
    assert c.batch_size == 8
    # untouched fields keep defaults
    assert c.hidden_channels == 16
    assert c.seed == 42


def test_from_dict_ignores_unknown_keys():
    c = ExpConfig.from_dict({"conv_type": "sage", "bogus_field": 123})
    assert c.conv_type == "sage"
    assert not hasattr(c, "bogus_field")


def test_from_dict_none_yields_defaults():
    c = ExpConfig.from_dict(None)
    assert c == ExpConfig()


def test_yaml_roundtrip(tmp_path, named_cfg):
    p = tmp_path / "cfg.yaml"
    named_cfg.to_yaml(str(p))
    assert p.exists()
    loaded = ExpConfig.from_yaml(str(p))
    assert loaded == named_cfg


def test_to_dict_is_flat_and_complete(base_cfg):
    d = base_cfg.to_dict()
    for key in ("name", "gpu", "master_port", "conv_type", "seed",
                "include_ndf", "ndf_count", "defect_cap", "data_base"):
        assert key in d


# --------------------------------------------------------------------------- #
# to_cli_args — value flags
# --------------------------------------------------------------------------- #
def _pairs(args):
    """Map every '--flag value' pair into a dict (ignores bare store_true flags)."""
    out = {}
    i = 0
    while i < len(args):
        tok = args[i]
        if tok.startswith("--") and i + 1 < len(args) and not args[i + 1].startswith("--"):
            out[tok] = args[i + 1]
            i += 2
        else:
            i += 1
    return out


def test_cli_value_flags(base_cfg):
    args = base_cfg.to_cli_args()
    p = _pairs(args)
    assert p["--hidden_channels"] == "16"
    assert p["--learning_rate"] == "0.002"
    assert p["--batch_size"] == "64"
    assert p["--epochs"] == "100"
    assert p["--conv_type"] == "gat"
    assert p["--data_base"] == DEFAULT_DATA_BASE
    assert p["--seed"] == "42"


def test_cli_meta_not_emitted(base_cfg):
    args = base_cfg.to_cli_args()
    assert "--name" not in args
    assert "--gpu" not in args
    assert "--master_port" not in args


# --------------------------------------------------------------------------- #
# to_cli_args — store_true logic
# --------------------------------------------------------------------------- #
def test_store_true_emitted_only_when_true():
    on = ExpConfig(use_amp=True, use_minority_sampler=True, edge_geo_features=True)
    a = on.to_cli_args()
    assert "--use_amp" in a
    assert "--use_minority_sampler" in a
    assert "--edge_geo_features" in a

    off = ExpConfig(use_amp=False, use_minority_sampler=False, edge_geo_features=False)
    a2 = off.to_cli_args()
    assert "--use_amp" not in a2
    assert "--use_minority_sampler" not in a2
    assert "--edge_geo_features" not in a2


def test_default_true_flags_emit_negation_when_false():
    c = ExpConfig(
        use_logit_adjust=False,
        use_onecycle=False,
        use_layer_constraint=False,
        save_on_best=False,
        best_report=False,
    )
    a = c.to_cli_args()
    assert "--no_logit_adjust" in a
    assert "--no_onecycle" in a
    assert "--no_layer_constraint" in a
    assert "--no_save_on_best" in a
    assert "--no_best_report" in a


def test_default_true_flags_no_negation_when_true(base_cfg):
    a = base_cfg.to_cli_args()
    # default-True paired flags -> their --no_* form must NOT appear
    for neg in ("--no_onecycle", "--no_layer_constraint",
                "--no_save_on_best", "--no_best_report"):
        assert neg not in a


def test_logit_adjust_off_by_default(base_cfg):
    # use_logit_adjust default is False (fixes the over-prediction collapse) ->
    # --no_logit_adjust IS emitted to disable train.py's aggressive LogitAdjustLoss.
    assert base_cfg.use_logit_adjust is False
    assert "--no_logit_adjust" in base_cfg.to_cli_args()
    assert "--no_logit_adjust" not in ExpConfig(use_logit_adjust=True).to_cli_args()


def test_class_frequency_sampler_store_true(base_cfg):
    # default True -> emitted as store_true
    assert "--use_class_frequency_sampler" in base_cfg.to_cli_args()
    off = ExpConfig(use_class_frequency_sampler=False)
    assert "--use_class_frequency_sampler" not in off.to_cli_args()


# --------------------------------------------------------------------------- #
# to_cli_args — cap-skip logic
# --------------------------------------------------------------------------- #
def test_caps_skipped_when_zero():
    c = ExpConfig(ndf_count=0, defect_cap=0)
    a = c.to_cli_args()
    assert "--ndf_count" not in a
    assert "--defect_cap" not in a


def test_caps_emitted_when_nonzero(base_cfg):
    a = base_cfg.to_cli_args()
    p = _pairs(a)
    assert p["--ndf_count"] == "5000"
    assert p["--defect_cap"] == "5000"


def test_onecycle_max_lr_skipped_when_none():
    assert "--onecycle_max_lr" not in ExpConfig(onecycle_max_lr=None).to_cli_args()
    a = ExpConfig(onecycle_max_lr=0.01).to_cli_args()
    assert "--onecycle_max_lr" in a


def test_empty_path_overrides_skipped():
    c = ExpConfig(label_dir="", mirror_perm_path="")
    a = c.to_cli_args()
    assert "--label_dir" not in a
    assert "--mirror_perm_path" not in a


def test_include_ndf_store_true():
    assert "--include_ndf" in ExpConfig(include_ndf=True).to_cli_args()
    assert "--include_ndf" not in ExpConfig(include_ndf=False).to_cli_args()


# --------------------------------------------------------------------------- #
# to_cli_args — TensorBoard
# --------------------------------------------------------------------------- #
def test_tensorboard_defaults(base_cfg):
    # default config has TensorBoard on and no explicit tb_dir
    assert base_cfg.tensorboard is True
    assert base_cfg.tb_dir == ""
    a = base_cfg.to_cli_args()
    assert "--tensorboard" in a
    assert "--tb_dir" not in a  # empty tb_dir is omitted


def test_tensorboard_store_true():
    assert "--tensorboard" in ExpConfig(tensorboard=True).to_cli_args()
    assert "--tensorboard" not in ExpConfig(tensorboard=False).to_cli_args()


def test_tb_dir_emitted_when_set():
    a = ExpConfig(tb_dir="/runs/exp1/tb").to_cli_args()
    p = _pairs(a)
    assert p["--tb_dir"] == "/runs/exp1/tb"


def test_tb_dir_omitted_when_empty():
    assert "--tb_dir" not in ExpConfig(tb_dir="").to_cli_args()


def test_tensorboard_off_but_tb_dir_set():
    # tb_dir is independent of the --tensorboard toggle
    a = ExpConfig(tensorboard=False, tb_dir="/runs/x/tb").to_cli_args()
    assert "--tensorboard" not in a
    assert _pairs(a)["--tb_dir"] == "/runs/x/tb"


# --------------------------------------------------------------------------- #
# launch helpers
# --------------------------------------------------------------------------- #
def test_env_contract(named_cfg):
    e = named_cfg.env()
    assert e["CUDA_VISIBLE_DEVICES"] == "2"
    assert e["LD_LIBRARY_PATH"] == LD_LIBRARY_PATH


def test_torchrun_cmd(named_cfg):
    cmd = named_cfg.torchrun_cmd("/some/train.py")
    assert cmd[0] == TORCHRUN_BIN
    assert "--nproc_per_node=1" in cmd
    assert "--master_port=29515" in cmd
    assert "/some/train.py" in cmd
    # the train.py args follow the script path
    assert cmd.index("/some/train.py") < len(cmd) - 1
    assert "--conv_type" in cmd
