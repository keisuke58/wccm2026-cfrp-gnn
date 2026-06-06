"""expkit.config — ExpConfig dataclass for the WCCM CFRP-GNN experiment framework.

PURE PYTHON: no torch / CUDA import here so configs (and tests) load on CPU.

ExpConfig holds every train.py flag the framework drives plus the new NDF/balance
flags and a few meta fields (name, gpu, master_port). It can be (de)serialized to
YAML and turned into the exact train.py CLI via ``to_cli_args()``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional

import yaml

# torchrun / env constants (single-GPU launch contract from the SPEC)
TORCHRUN_BIN = "/home/nishioka/miniconda3/envs/interstage_gnn_seconda/bin/torchrun"
LD_LIBRARY_PATH = "/home/nishioka/miniconda3/lib"
DEFAULT_DATA_BASE = (
    "/home/nishioka/GNN/GNN_hole_2026/all_sub_hole_defect_zscore_noise"
)


@dataclass
class ExpConfig:
    # ----- meta (not passed to train.py) -----
    name: str = "exp"
    gpu: int = 0
    master_port: int = 29500

    # ----- core hyperparameters -----
    hidden_channels: int = 16
    learning_rate: float = 0.002
    batch_size: int = 64
    epochs: int = 100
    weight_decay: float = 5e-4
    patience: int = 60
    gamma: float = 3.0
    class_weight_multiplier: float = 5.0

    # ----- loss / logit adjustment -----
    # IMPORTANT: logit-adjust default was True; with tau=1.5 on this extreme imbalance
    # (class-0 prior ~0.998) it over-boosts minority classes -> over-prediction collapse
    # (high recall / ~0 precision, macro-F1 stuck ~0.16). Verified 2026-06-06: disabling it
    # recovers learning (gat climbs past 0.30 like the old script -> 0.70). Default OFF.
    use_logit_adjust: bool = False         # --no_logit_adjust when False
    logit_adjust_tau: float = 1.5
    use_log_softmax: bool = False          # store_true
    label_smoothing: float = 0.0

    # ----- training infra -----
    use_amp: bool = False                  # store_true
    use_onecycle: bool = True              # --no_onecycle when False
    onecycle_max_lr: Optional[float] = None
    onecycle_final_div_factor: float = 10000.0

    # ----- samplers -----
    use_minority_sampler: bool = False     # store_true
    minority_weight_power: float = 1.0
    use_class_frequency_sampler: bool = True   # store_true

    # ----- regularization / augmentation -----
    dropout: float = 0.1
    edge_drop_prob: float = 0.01
    data_usage_ratio: float = 1.0
    train_noise_std: float = 0.0
    train_noise_curriculum: bool = False   # store_true
    mirror_augment: bool = False           # store_true
    mirror_perm_path: str = ""

    # ----- layer constraint -----
    use_layer_constraint: bool = True      # --no_layer_constraint when False
    layer_constraint_weight: float = 1.0

    # ----- architecture -----
    conv_type: str = "gat"
    mgn_blocks: int = 10
    edge_geo_features: bool = False        # store_true
    extra_geo_features: bool = False       # store_true

    # ----- data sources -----
    data_base: str = DEFAULT_DATA_BASE
    label_dir: str = ""

    # ----- leakage / grouping -----
    group_key: str = "LBel"
    group_purge_eval: bool = True          # store_true
    group_purge_target: str = "valtest"

    # ----- checkpoint / reporting -----
    save_on_best: bool = True              # --no_save_on_best when False
    save_every: int = 100
    best_report: bool = True               # --no_best_report when False
    best_report_topk: int = 5
    best_report_confusion_topk: int = 3

    # ----- NEW: NDF / balanced recipe -----
    include_ndf: bool = True               # store_true
    ndf_count: int = 5000                  # 0 = all; skipped when 0
    defect_cap: int = 5000                 # 0 = no cap; skipped when 0
    seed: int = 42

    # ----- TensorBoard logging -----
    tensorboard: bool = True               # store_true; emit --tensorboard when True
    tb_dir: str = ""                       # empty => runner sets {run_dir}/tb

    # ------------------------------------------------------------------ #
    # (de)serialization
    # ------------------------------------------------------------------ #
    @classmethod
    def _field_names(cls) -> List[str]:
        return [f.name for f in fields(cls)]

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "ExpConfig":
        """Merge an arbitrary mapping over the defaults (unknown keys ignored)."""
        d = dict(d or {})
        known = set(cls._field_names())
        kwargs = {k: v for k, v in d.items() if k in known}
        unknown = set(d) - known
        if unknown:
            # Surface typos loudly but do not crash config loading.
            import warnings

            warnings.warn(f"ExpConfig.from_dict: ignoring unknown keys {sorted(unknown)}")
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str) -> "ExpConfig":
        with open(path, "r") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def to_yaml(self, path: str) -> str:
        with open(path, "w") as fh:
            yaml.safe_dump(self.to_dict(), fh, default_flow_style=False, sort_keys=False)
        return path

    # ------------------------------------------------------------------ #
    # CLI generation
    # ------------------------------------------------------------------ #
    def to_cli_args(self) -> List[str]:
        """Build the exact train.py CLI argument list.

        Rules:
          * value flags -> ['--flag', str(value)]
          * store_true flags -> emit '--flag' only when True
          * paired (default-True) flags -> emit the '--no_*' counterpart when False
          * --ndf_count / --defect_cap skipped when 0 (meaning all / no-cap)
          * empty-string path overrides skipped (train.py treats '' as default)
          * onecycle_max_lr skipped when None
        """
        a: List[str] = []

        def val(flag: str, value: Any) -> None:
            a.extend([flag, str(value)])

        # core hyperparameters
        val("--hidden_channels", self.hidden_channels)
        val("--learning_rate", self.learning_rate)
        val("--batch_size", self.batch_size)
        val("--epochs", self.epochs)
        val("--weight_decay", self.weight_decay)
        val("--patience", self.patience)
        val("--gamma", self.gamma)
        val("--class_weight_multiplier", self.class_weight_multiplier)

        # loss / logit adjustment (default True -> emit negation when False)
        if not self.use_logit_adjust:
            a.append("--no_logit_adjust")
        val("--logit_adjust_tau", self.logit_adjust_tau)
        if self.use_log_softmax:
            a.append("--use_log_softmax")
        val("--label_smoothing", self.label_smoothing)

        # training infra
        if self.use_amp:
            a.append("--use_amp")
        if not self.use_onecycle:
            a.append("--no_onecycle")
        if self.onecycle_max_lr is not None:
            val("--onecycle_max_lr", self.onecycle_max_lr)
        val("--onecycle_final_div_factor", self.onecycle_final_div_factor)

        # samplers
        if self.use_minority_sampler:
            a.append("--use_minority_sampler")
        val("--minority_weight_power", self.minority_weight_power)
        if self.use_class_frequency_sampler:
            a.append("--use_class_frequency_sampler")

        # regularization / augmentation
        val("--dropout", self.dropout)
        val("--edge_drop_prob", self.edge_drop_prob)
        val("--data_usage_ratio", self.data_usage_ratio)
        val("--train_noise_std", self.train_noise_std)
        if self.train_noise_curriculum:
            a.append("--train_noise_curriculum")
        if self.mirror_augment:
            a.append("--mirror_augment")
        if self.mirror_perm_path:
            val("--mirror_perm_path", self.mirror_perm_path)

        # layer constraint (default True -> emit negation when False)
        if not self.use_layer_constraint:
            a.append("--no_layer_constraint")
        val("--layer_constraint_weight", self.layer_constraint_weight)

        # architecture
        val("--conv_type", self.conv_type)
        val("--mgn_blocks", self.mgn_blocks)
        if self.edge_geo_features:
            a.append("--edge_geo_features")
        if self.extra_geo_features:
            a.append("--extra_geo_features")

        # data sources (empty string => use train.py default)
        if self.data_base:
            val("--data_base", self.data_base)
        if self.label_dir:
            val("--label_dir", self.label_dir)

        # leakage / grouping
        val("--group_key", self.group_key)
        if self.group_purge_eval:
            a.append("--group_purge_eval")
        val("--group_purge_target", self.group_purge_target)

        # checkpoint / reporting (default True -> emit negation when False)
        if not self.save_on_best:
            a.append("--no_save_on_best")
        val("--save_every", self.save_every)
        if not self.best_report:
            a.append("--no_best_report")
        val("--best_report_topk", self.best_report_topk)
        val("--best_report_confusion_topk", self.best_report_confusion_topk)

        # NEW: NDF / balanced recipe
        if self.include_ndf:
            a.append("--include_ndf")
        if self.ndf_count:  # skip 0 (= all)
            val("--ndf_count", self.ndf_count)
        if self.defect_cap:  # skip 0 (= no cap)
            val("--defect_cap", self.defect_cap)
        val("--seed", self.seed)

        # TensorBoard (store_true; tb_dir only when explicitly set)
        if self.tensorboard:
            a.append("--tensorboard")
        if self.tb_dir:
            val("--tb_dir", self.tb_dir)

        return a

    # ------------------------------------------------------------------ #
    # launch helpers
    # ------------------------------------------------------------------ #
    def env(self) -> Dict[str, str]:
        """Environment overrides for the launch (merged onto os.environ by runner)."""
        return {
            "CUDA_VISIBLE_DEVICES": str(self.gpu),
            "LD_LIBRARY_PATH": LD_LIBRARY_PATH,
        }

    def torchrun_cmd(self, train_py_path: str) -> List[str]:
        """Full torchrun command (single GPU) for this config."""
        return [
            TORCHRUN_BIN,
            "--nproc_per_node=1",
            f"--master_port={self.master_port}",
            train_py_path,
            *self.to_cli_args(),
        ]
