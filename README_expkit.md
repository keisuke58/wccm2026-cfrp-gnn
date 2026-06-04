# expkit — config-driven experiments for the WCCM CFRP-GNN

`expkit` turns the ~50-flag `train.py` into a **reproducible, config-driven**
experiment framework. You describe an experiment in YAML (or a Python
`ExpConfig`), the runner snapshots config + git + the exact launch command,
captures the log, and parses a machine-readable `metrics.json`. Sweeps,
aggregation, and plots are built on top.

The config / metrics / aggregate / sweep layers are **pure Python** (no
`torch`, no CUDA) so configs load and tests run on a plain CPU. Only
`expkit.runner` actually launches the GPU subprocess.

```
expkit/
  config.py     ExpConfig dataclass  -> CLI, YAML, env, torchrun cmd
  runner.py     run()  -> create run-dir, launch, capture log, parse metrics
  metrics.py    parse_log() / write_metrics_json() / read_metrics_json()
  aggregate.py  collect_runs() / to_dataframe() / to_markdown() / to_csv()
  plots.py      f1_curve() / arch_bars() / confusion_from_csv()
  sweep.py      expand() / plan_ports() + `python -m expkit.sweep` CLI
configs/
  base.yaml             single balanced-recipe reference config
  ablation_arch.yaml    sweep conv_type with the BALANCED recipe
  recipe_ablation.yaml  toggle the recipe to isolate its effect
```

---

## Why this exists: data-recipe + balance dominates architecture

The headline ablation finding that motivated `expkit`:

> **The data recipe (balanced NDF negatives) dominates the GNN architecture.**
> The old hand-run script trained on a **balanced 5000 : 5000** mix
> (5000 `Defect_L*` positives : 5000 `NoiseDefectFree` negatives) and reached
> **Macro-F1 ≈ 0.70** with **ARM1**. The newer `train.py` arch-sweep, run on the
> **imbalanced** full dataset, **capped around ~0.13** no matter which
> `conv_type` it used.

The architecture was never the bottleneck — the **data recipe and class
balance** were. So the first thing `expkit` does is promote that recipe from an
`INCLUDE_NDF` environment-variable hack into **real, reproducible flags**, and
make `--defect_cap` / `--ndf_count` first-class so any architecture is
benchmarked on the *same balanced footing*.

### The recipe flags

These are the new `train.py` flags (mirrored as `ExpConfig` fields):

| Flag | Field | Default | Meaning |
|------|-------|---------|---------|
| `--include_ndf` | `include_ndf` | `True` | Include `NoiseDefectFree_*` class-0 negatives in **TRAIN only** (val/test have none). Replaces the old `INCLUDE_NDF=1` env hack. |
| `--ndf_count N` | `ndf_count` | `5000` | Cap the number of NDF negatives. `0` = use all. Selection is **deterministic** (sorted, first N). Skipped from the CLI when `0`. |
| `--defect_cap N` | `defect_cap` | `5000` | Cap the number of `Defect_L*` TRAIN positives (deterministic, seeded sample) so training can be **balanced**. `0` = no cap. Skipped from the CLI when `0`. |
| `--seed N` | `seed` | `42` | Master seed (torch / numpy / random / cuda). Printed at startup for reproducibility. |

### The balanced 5000 : 5000 rationale

`--defect_cap 5000 --ndf_count 5000` reproduces the **old-script balance**:
5000 defect positives against 5000 noise-only negatives. This matters because:

- **The minority class stops being drowned.** On the full imbalanced set the
  network can score well on accuracy while collapsing macro-F1 to ~0.13.
  Balancing the two pools is what unlocked the ~0.70 macro-F1.
- **It is the controlled baseline for arch comparisons.** When you sweep
  `conv_type`, every architecture sees the *same* 5000:5000 footing, so any
  difference is attributable to the model — not to who happened to get a luckier
  class ratio.
- **It is deterministic.** Both caps select deterministically (NDF sorted +
  first-N; defects seeded sample under `--seed`), so a given config + seed
  reproduces the exact training set.

These are the `ExpConfig` **defaults** (`include_ndf=True`, `ndf_count=5000`,
`defect_cap=5000`, `seed=42`) — i.e. *out of the box* `expkit` runs the
balanced recipe that gives the strong result, not the imbalanced one that
caps at ~0.13.

---

## 1. Write a config

A config is just an `ExpConfig`. Start from the defaults and override what you
need. In Python:

```python
from expkit import ExpConfig

cfg = ExpConfig(
    name="gat_balanced",
    gpu=0,
    master_port=29500,
    conv_type="gat",
    epochs=500,
    patience=120,
    use_onecycle=True,
    # balanced recipe (these four are already the defaults; shown for clarity)
    include_ndf=True,
    ndf_count=5000,
    defect_cap=5000,
    seed=42,
)
cfg.to_yaml("configs/gat_balanced.yaml")
```

Or write the YAML directly (any omitted field falls back to the `ExpConfig`
default; unknown keys are warned about, not fatal):

```yaml
# configs/gat_balanced.yaml
name: gat_balanced
gpu: 0
master_port: 29500
conv_type: gat
epochs: 500
patience: 120
use_onecycle: true
include_ndf: true
ndf_count: 5000
defect_cap: 5000
seed: 42
```

Load it back and inspect the exact CLI it will produce:

```python
cfg = ExpConfig.from_yaml("configs/gat_balanced.yaml")
print(" ".join(cfg.to_cli_args()))
# ...--conv_type gat ... --include_ndf --ndf_count 5000 --defect_cap 5000 --seed 42
```

`to_cli_args()` rules worth knowing:

- value flags emit `--flag VALUE`;
- `store_true` flags (e.g. `--include_ndf`, `--use_amp`) appear **only when
  True**;
- default-True paired flags emit their negation when False
  (`use_onecycle=False` -> `--no_onecycle`,
  `use_logit_adjust=False` -> `--no_logit_adjust`,
  `save_on_best=False` -> `--no_save_on_best`, etc.);
- `--ndf_count` / `--defect_cap` are **skipped when 0** (0 means "all" /
  "no cap");
- empty-string path overrides (`data_base=""`, `label_dir=""`,
  `mirror_perm_path=""`) are skipped so `train.py` uses its own default;
- `onecycle_max_lr` is skipped when `None`.

---

## 2. Run one experiment

The runner creates the run-dir, snapshots everything, launches the single-GPU
`torchrun`, tails stdout/stderr into `train.log`, and parses metrics:

```python
from expkit import ExpConfig
from expkit.runner import run

cfg = ExpConfig.from_yaml("configs/base.yaml")

result = run(
    cfg,
    output_root="runs",
    run_id="gat_s42_20260604_1200",   # caller supplies the stamp; core never reads the wall clock
    # train_py defaults to <repo>/train.py
)
print(result["run_dir"])   # runs/<name>/gat_s42_20260604_1200/
print(result["metrics"])   # parsed metrics dict (or None on dry_run / failure)
```

`run_id` convention: `f"{conv_type}_s{seed}_{stamp}"` where **`stamp` is a
parameter you pass in** — the framework core deliberately never calls the wall
clock, so runs are reproducible from the outside.

### Dry run (no GPU) — see the command without launching

```python
result = run(cfg, output_root="runs", run_id="gat_s42_dry", dry_run=True)
print(" ".join(result["cmd"]))
```

This prints the full `torchrun` invocation (built by
`cfg.torchrun_cmd(train_py)`) with the env from `cfg.env()`
(`CUDA_VISIBLE_DEVICES`, `LD_LIBRARY_PATH`) — handy in CI / on a CPU box.

### What lands in the run-dir

```
runs/<name>/<run_id>/
  config.yaml    # snapshot of the ExpConfig actually used
  git.txt        # git rev-parse HEAD + dirty/clean flag
  cmd.txt        # full launch command + env
  train.log      # captured stdout/stderr
  metrics.json   # parsed summary (schema below)
  tb/            # TensorBoard event files (when tensorboard is enabled)
```

`metrics.json` schema (produced by `expkit.metrics`):

```json
{
  "run_id": "...", "name": "...", "conv_type": "gat", "seed": 42,
  "best_macro_f1": 0.70, "best_epoch": 318, "last_epoch": 500,
  "test_macro_f1": null,
  "status": "done", "error": null, "log_path": "runs/.../train.log"
}
```

`parse_log(log_path)` does the regex parsing of the known `train.py` print
formats (`Best Macro F1: X.XXXX`, `=== Epoch N ===`, per-epoch
`Macro F1: X`), since `train.py` itself does not emit a summary JSON.

### Running from the shell (without Python)

Every config can also be launched by hand using the CLI it generates — useful
for one-offs:

```bash
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/home/nishioka/miniconda3/lib \
  /home/nishioka/miniconda3/envs/interstage_gnn_seconda/bin/torchrun \
  --nproc_per_node=1 --master_port=29500 \
  train.py $(python -c "from expkit import ExpConfig; \
                        print(' '.join(ExpConfig.from_yaml('configs/base.yaml').to_cli_args()))")
```

(The runner does all of this for you; the snippet just shows the moving parts.)

### TensorBoard

`train.py` logs per-epoch scalars and final hparams to TensorBoard when enabled
(rank 0 only). Enable it via the config field `tensorboard: true` (the default)
or, for a hand-launched run, the `--tensorboard` flag. If the `tensorboard`
package is not installed in the env, `train.py` prints a warning and continues
without it — training never crashes on a missing TensorBoard.

Logged each epoch: `loss/train`, `loss/val`, `f1/macro_val`, `acc/val`, `lr`,
and per-class `f1_class/<i>`; at the end, `add_hparams` records the key
hyperparameters against `best_macro_f1`.

The runner points each run's log dir at `{run_dir}/tb/`, so every run's events
live inside its own run-dir (unless you set `tb_dir` explicitly in the config,
which is honored as-is). View one run, or aggregate every run under an
`output_root`, with:

```bash
# all runs at once (each run shows up as a separate TensorBoard run)
tensorboard --logdir runs

# a single run
tensorboard --logdir runs/<name>/<run_id>/tb
```

Or build the command from Python:

```python
from expkit import runner
cmd = runner.tensorboard_cmd("runs", port=6006)
# -> ['tensorboard', '--logdir', 'runs', '--port', '6006']
```

---

## 3. Sweep architectures

A sweep is a base config plus a grid of fields to vary; `expand()` takes the
cartesian product and yields one `ExpConfig` per cell. `plan_ports()` assigns a
distinct `master_port` to each so concurrent runs don't collide.

```python
from expkit.sweep import expand, plan_ports

grid = {
    "conv_type": ["gat", "gatv2", "sage", "pna", "meshgraphnet"],
    "seed": [42, 43],
}
base = ExpConfig.from_yaml("configs/ablation_arch.yaml")
cfgs = expand(grid)                      # base read from the config in the yaml
ports = plan_ports(base.master_port, len(cfgs))
```

Preview a sweep plan from the shell **without touching a GPU**:

```bash
python -m expkit.sweep --config configs/ablation_arch.yaml --dry-run
```

This prints the per-cell plan (name, conv_type, port, full CLI) so you can
sanity-check before committing GPU time.

### The shipped sweep config: `configs/ablation_arch.yaml`

This sweeps `conv_type` over `[gat, gatv2, sage, pna, meshgraphnet]` **with the
balanced recipe held fixed** (`include_ndf=true`, `ndf_count=5000`,
`defect_cap=5000`, `epochs=500`, `patience=120`, `use_onecycle=true`).
Per-arch memory differences are handled with `batch_size` overrides
(`pna=8`, `meshgraphnet=16`, others `64`). Because the recipe is identical
across cells, this sweep is exactly the apples-to-apples architecture
comparison the ablation finding calls for.

### Isolating the recipe: `configs/recipe_ablation.yaml`

To *prove* the recipe effect rather than the architecture, toggle the recipe
itself — flip `include_ndf` true/false and vary `defect_cap` — while holding
`conv_type` fixed. This is the experiment that reproduces "balanced -> ~0.70 vs
imbalanced -> ~0.13".

---

## 4. Aggregate results

`collect_runs(root)` scans every run-dir under `root`, preferring `metrics.json`
and falling back to parsing `train.log`, and returns a list of metric dicts
ranked by `best_macro_f1` (descending).

```python
from expkit.aggregate import collect_runs, to_dataframe, to_markdown, to_csv

rows = collect_runs("runs")
print(to_markdown(rows))          # ranked Markdown table for the paper/notes
to_csv(rows, "runs/summary.csv")  # flat CSV
df = to_dataframe(rows)           # pandas DataFrame for further analysis
```

`to_markdown(rows)` is what you paste into the thesis / WCCM notes; the top row
is the best `best_macro_f1`, so a correct balanced run should sit near ~0.70 and
any imbalanced control near ~0.13 — making the recipe effect obvious at a
glance.

---

## 5. Plot

Plots use the matplotlib **Agg** backend (no display needed, safe over SSH /
headless):

```python
from expkit.plots import f1_curve, arch_bars

# Macro-F1 vs epoch, one line per run
f1_curve(
    ["runs/gat_balanced/gat_s42_.../train.log",
     "runs/gat_imbalanced/gat_s42_.../train.log"],
    out_png="runs/f1_curve.png",
)

# Best Macro-F1 per architecture (from aggregated rows)
arch_bars(collect_runs("runs"), out_png="runs/arch_bars.png")
```

`f1_curve` parses the per-epoch `Macro F1` lines straight out of the logs;
`arch_bars` renders the ranked best-F1 per `conv_type`. `confusion_from_csv(...)`
is a best-effort helper for a confusion matrix when one has been exported.

---

## End-to-end (the typical loop)

```bash
# 1. preview the balanced arch sweep (no GPU)
python -m expkit.sweep --config configs/ablation_arch.yaml --dry-run

# 2. run it (each cell -> its own run-dir with config/git/cmd/log/metrics)
python -m expkit.sweep --config configs/ablation_arch.yaml --output-root runs

# 3. rank + export
python -c "from expkit.aggregate import collect_runs,to_markdown,to_csv; \
           r=collect_runs('runs'); print(to_markdown(r)); to_csv(r,'runs/summary.csv')"

# 4. plot
python -c "from expkit.aggregate import collect_runs; from expkit.plots import arch_bars; \
           arch_bars(collect_runs('runs'),'runs/arch_bars.png')"
```

---

## Reproducibility notes

- **Seeding.** `--seed` (default 42) seeds torch / numpy / random / cuda and is
  printed at startup. The deterministic `--defect_cap` sample is drawn under
  this seed, so `(config, seed)` fully determines the training set.
- **No hidden wall-clock.** The framework core never reads the clock; the `stamp`
  in `run_id` is supplied by the caller, so runs are reproducible and
  re-namable.
- **Snapshots.** Each run-dir keeps the exact `config.yaml`, `git.txt`
  (HEAD + dirty flag), and `cmd.txt`, so any number can be traced back to the
  precise code + flags that produced it.
- **No more env hacks.** The retired `INCLUDE_NDF=1` environment variable is
  replaced by `--include_ndf` (+ `--ndf_count` / `--defect_cap`), so the
  balanced recipe is captured in the config, not in your shell history.
