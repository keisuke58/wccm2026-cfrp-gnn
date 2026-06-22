# Raw × Diff-Norm Integration Plan

**Goal**: combine raw DSPSS and difference-normalised DSPSS to build the strongest
practical SHM model, covering both known-baseline and unknown-baseline scenarios.

---

## Why combining raw and diff matters

| | raw DSPSS | diff-norm DSPSS |
|---|---|---|
| Prerequisite | none | FEM or 1st-flight baseline |
| Hole-edge artefact | dominates (holes mask defects) | cancelled (hole cancels out) |
| Defect contrast | ~1× | ~10× |
| Depth discrimination | near-impossible | enabled |
| Stage-0 AUROC | 0.999 (clean) / 0.84 (noisy) | N/A (same field) |
| GNN macro-F1 | 0.61 (hole excl.) | 0.803 (HybridMGN+Stage-0) |
| Unknown structure | ✓ only option | ✗ |

**Key insight**: each channel carries information the other lacks.
The GNN can learn "trust diff near holes; trust raw far from holes."

---

## Four complementary ideas

### A — Bootstrap transition (`bootstrap_adapt.py`)
**Practical deployment. Implement first.**

- Flight 0: raw Stage-0 only; collect healthy specimens → build baseline
- Flight 1+: automatic switch to diff-norm Stage-0 + dual-stream GNN
- EMA update (α=0.2) prevents baseline drift from misclassified defects
- Run: `python bootstrap_adapt.py --demo --n_flights 5`

### B — Dual-stream GNN (`dual_stream_gnn.py`)
**Highest research ROI. Run on Vancouver after generating raw data.**

- Node features: `[x, y, z, dspss_diff, dspss_raw]`  (in_channels=5)
- `--no_baseline` mode: zero both DSPSS channels → model degrades gracefully
- Partial-baseline mode: diff channel only → equivalent to current best
- Run: see §Prerequisites below

### C — Self-supervised baseline (`selfsuper_baseline.py`)
**Research direction. For truly baseline-free deployment.**

- GNN autoencoder trained on healthy specimens only
- Pseudo-diff = raw − AE_reconstruction  ≈ anomaly residual
- Expected gap vs true diff-norm: ~5-10 F1 points (hole-edge still present)
- Worth testing if AE-pseudo-diff reaches ~0.70 F1

### D — UQ propagation (`uq_baseline_propagation.py`)
**Keio thesis / IWSHM 2027. Build after B is validated.**

- Baseline uncertainty: σ_bl ∝ 1/√n_flights (Welford estimator)
- K=50 MC samples → per-node class probabilities + std
- Inspection gate: fly / monitor / re-inspect based on defect_prob + unc_frac
- Bridge to Muramatsu lab: UQ × phase-field prognosis

---

## Prerequisites on Vancouver

```bash
# Step 1: generate raw z-score data (one-time, ~30 min)
cd /home/nishioka/GNN/wccm2026-cfrp-gnn
python data_prep/prepare_plain_zscore.py
# Output: GNN_hole_2026/all_hole_defect_zscore/{train,val,test}/

# Step 2: verify
ls GNN_hole_2026/all_hole_defect_zscore/train/ | wc -l
# Should match: ls GNN_hole_2026/all_sub_hole_defect_zscore/train/ | wc -l

# Step 3: run unit tests (local, no GPU needed)
python dual_stream_gnn.py --test
python bootstrap_adapt.py --test
python selfsuper_baseline.py --test
python uq_baseline_propagation.py --test
```

---

## Training the strongest model (Idea B + noise augmentation)

```bash
# On Vancouver (GPU required)
python dual_stream_gnn.py --train \
    --arch hybridmgn \
    --epochs 300 \
    --noise_std 0.05 \
    --noise_healthy_std 0.03 \
    --baseline_noise_std 0.02 \
    --diff_channel_dropout 0.15
```

### Noise augmentation taxonomy (for "最強モデル")

| Augmentation | Flag | Target | Purpose |
|---|---|---|---|
| Input noise (defect) | `--noise_std` | diff + raw channels | simulate measurement noise on damaged specimen |
| Healthy noise | `--noise_healthy_std` | raw channel of healthy specimens | make Stage-0 robust to noisy healthy reference |
| Baseline noise | `--baseline_noise_std` | perturb baseline before diff = raw-bl | simulate imperfect 1st-flight baseline |
| Channel dropout | `--diff_channel_dropout` | zero diff channel randomly | force model to learn from raw alone (baseline-free robustness) |

All four together: the model sees every combination of noise level and baseline
quality during training → robust at all deployment phases (Flight 0 through N).

---

## Expected performance targets

| Experiment | Expected macro-F1 | Data requirement |
|---|---|---|
| Current (diff-norm, 4-ch) | 0.803 | FEM baseline |
| Dual-stream (B), full | 0.810–0.830 | FEM baseline + raw generated |
| Dual-stream (B), no_baseline | 0.65–0.70 | none (raw only) |
| Self-supervised AE (C) | 0.68–0.73 | healthy specimens only |
| Bootstrap Flight 1+ (A) | → B full | 1 healthy flight |

---

## Publication roadmap

| Paper | Venue | Key contribution |
|---|---|---|
| Current WCCM abstract | WCCM 2026 (Jul 22) | diff-norm GNN, Stage-0/1 |
| Composites B extension | 2026 Q4 | B (dual-stream) ablation + OOD |
| Full 3-structure paper | IWSHM 2027 | A (bootstrap) + B + D (UQ) |
| Keio master thesis | 2028 Mar | D (UQ) × phase-field prognosis |

---

## File map

```
dual_stream_gnn.py          ← Idea B: 5-channel model, run first
bootstrap_adapt.py          ← Idea A: deployment state machine
selfsuper_baseline.py       ← Idea C: AE-based pseudo-diff (no baseline)
uq_baseline_propagation.py  ← Idea D: MC UQ propagation (Keio bridge)
data_prep/prepare_plain_zscore.py   ← generates raw z-score data (prerequisite)
```
