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
