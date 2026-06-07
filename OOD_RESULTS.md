# Size-OOD generalization: architecture cross-comparison

Train on defect sizes **2×2 / 4×4 / 8×8** (+ NDF negatives); evaluate on **unseen 1×1**
(8916 specimens, `oned_1x1_subtracted_zscore` + `oned_1x1_19class_label`).
Metric: macro-F1 (support-only). Eval code: `ood_eval_arch.py` / `ood_eval_1x1.py`.

| architecture | in-dist (2/4/8) | OOD 1×1 | drop |
|---|---:|---:|---:|
| **HybridMGN** (front/back cross-layer edges) | **0.792** | 0.333 | 0.460 |
| MeshGraphNet | 0.761 | 0.332 | 0.429 |
| BiStrideMGN | 0.741 | 0.332 | 0.409 |
| GATv2 | 0.705 | 0.337 | 0.369 |
| GAT (Frontiers baseline) | 0.685 | 0.333 | 0.352 |
| GraphSAGE | 0.668 | 0.332 | 0.335 |
| MGN-Transformer | 0.552 | 0.333 | 0.219 |

## Key findings
1. **HybridMGN is the new in-distribution SOTA (0.792)** — adding front/back cross-layer
   ("world") edges between the two plies beats plain MeshGraphNet (0.761) and the
   published GAT baseline (0.61).
2. **Architecture is essentially irrelevant to out-of-distribution (unseen-size)
   generalization.** Despite in-dist spanning 0.55–0.79, OOD-1×1 is **0.332–0.337 for
   every architecture** (within noise). The best in-dist model (Hybrid) and the worst
   (MGN-T) reach the *same* OOD score.
3. ⇒ The size-generalization gap is a **data / representation / normalization** problem,
   **not** an architecture problem. Architectural gains do not transfer to unseen defect
   sizes. Closing the OOD gap requires non-architectural levers (size-aware features,
   scale normalization, augmentation, domain generalization), which is the next study.

## Paper framing (Composites B / CMAME)
- Headline 1: HybridMGN — physically-motivated cross-ply edges set a new in-dist SOTA.
- Headline 2 (non-obvious): "more expressive architectures do not generalize to unseen
  defect sizes" — a cautionary, rigorous OOD result the field needs.

## Richer metrics (2026-06-07): macro-F1 hides the truth

macro-F1(support-only) is inflated by class 0 (defect-free, 99.97% of nodes). Breaking it down:

| arch | set | macroF1 | **defF1** | detRec | detFPR | AUPRC | grpAcc | **exact** |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| HybridMGN | in-dist | 0.792 | 0.781 | 0.990 | 0.000 | 0.988 | 0.990 | 0.852 |
| MeshGraphNet | in-dist | 0.761 | 0.748 | 0.969 | 0.001 | 0.970 | 0.969 | 0.839 |
| GAT | in-dist | 0.685 | 0.667 | 0.972 | 0.000 | 0.986 | 0.972 | 0.701 |
| HybridMGN | OOD-1×1 | 0.333 | **0.001** | 0.727 | 0.007 | 0.040 | 0.727 | **0.001** |
| MeshGraphNet | OOD-1×1 | 0.332 | **0.000** | 0.643 | 0.007 | 0.018 | 0.643 | **0.000** |
| GAT | OOD-1×1 | 0.333 | **0.001** | **0.958** | 0.008 | 0.036 | 0.958 | **0.001** |

defF1 = macro-F1 over the 18 defect classes only; detRec/FPR = node-level defect-vs-healthy
detection; grpAcc = correct surface-group (1–9 vs 10–18) among true-defect nodes; exact =
exact 19-class accuracy among true-defect nodes.

### Findings (richer)
1. **The OOD macro-F1 ≈0.33 is entirely class-0**: defect-only F1 and exact accuracy are **≈0
   for every architecture** on unseen size — fine defect classification collapses.
2. **Detection partially generalizes**: the model still flags defect *nodes* (GAT 0.96 recall,
   HybridMGN 0.73, MeshGraphNet 0.64) at low FPR, and gets the surface group right at the same
   rate — but cannot assign the correct class/region.
3. **In-dist→OOD reversal**: HybridMGN is best in-distribution, but the *simplest* model (GAT)
   detects unseen-size defects best (0.96 recall). Expressive mesh-physics models overfit
   in-distribution defect signatures.
4. ⇒ The right story is **"detection generalizes, fine localization does not"** on unseen
   defect sizes — invisible to macro-F1, exposed by defect-only F1 + detection/localization split.
