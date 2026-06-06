# figure_sources — reproducible WCCM paper-deck figures

Clean, presentation-grade figures regenerated from FEA data (not pptx screenshots).

- `gen_clean_figs.py` — generator. On the GPU server it reads from
  `/home/nishioka/GNN/GNN_hole_2026/` (structured 57×125 grid, 2 layers × 6971
  nodes = 13942; hole region masked). Produces:
  `diff_clean.png` (no-defect / with-defect / difference),
  `noise_clean.png` (Z-score vs Z-score+noise input),
  `ndf_clean.png` (defect-free + 10% noise sample),
  `label_clean.png` (19-class ground-truth target).
- `visualize_normalized_zscore_spatial.py` — original server script the
  grid-reshape logic was taken from.
- `data/` — minimal source arrays for specimen `Defect_L10_B100_el2515_H4_W4`
  (difference, zscore+noise, 19-label) + `real_hole_no_defect_original.npy`
  (no-defect reference) so the figures can be regenerated offline.

Output PNGs live in `../figures_paper/` and are used by `wccm_beamer_paper.tex`.
