# One-point relative-coordinate GNN (Asano contribution)

Source: Sosuke Asano (collaborator), shared 2026-06-05. Original from `Payload2026/asano/`.

## Idea
Per-node defect classification using a **local neighborhood + relative coordinates**, instead of
classifying the whole 13942-node graph at once.

1. Pick a node (seed). 2. Take its n-hop neighborhood via `NeighborLoader` (disjoint subgraph per seed,
`num_neighbors=[-1]*5` = 5 full hops). 3. Re-center coordinates: `rel = x[:,:3] - seed_xyz`, concat with
the DSPSS feature → input `[rel_x, rel_y, rel_z, DSPSS]`. 4. Predict the **seed node's** 19-class label.

Motivation: absolute coordinates made the GNN memorize absolute positions (overfit to location). Relative
coordinates remove that → should generalize better. This is a complementary formulation to the full-graph
`train.py` (per-graph 19-class) approach.

Model: 5-layer GATv2 (heads=4) + BatchNorm + residual, FocalLoss(+class weights, gamma=2), AdamW,
ReduceLROnPlateau, k-fold (only fold 0 run), per-epoch random seed sampling (5% defect + 5% bg).

## Files
- `GNN_onepoint-3-4.ipynb` — the notebook (as received).
- `hole_edges_2layer_bidirectional.npy` — **(54896, 2)** = the 2-layer hole graph edges made **bidirectional**
  (2× the old `hole_edges_2layer_best.npy` (27448,2)). Use THESE edges (per Asano).

## 表裏 (front/back) status — the open problem
The 2-layer mesh has a front and a back layer. Previously edges connected both layers, so a seed's
neighborhood spanned front+back and the model could tell whether a defect was front or back. In the current
state the two layers are **not yet combined**, so front-vs-back is indistinguishable. The bidirectional
edges file is the proposed fix (connect the layers so the local subgraph includes both sides). **Asano is
still fixing this.**

## Adaptation TODO before running in this repo
- [ ] **Bug**: `debug_check_seed_structure(...)` is called in validation (epoch 1, batch 0) but is **not defined**
      in the notebook → would raise NameError. Define it or remove the call.
- [ ] Hardcoded paths point to `/home/asano/asano_handover/...` (data `npy_datas_H8_W8_*`, k-fold folders,
      white-noise intensities, xyz/edge paths, output dir). Rewire to our data layout.
- [ ] `from Loss.focal_loss import FocalLoss` — present in this repo's `Loss/`, OK.
- [ ] Data: uses the same 13942-node 2-layer hole mesh, 19-class labels, H8_W8 defect blocks. Confirm our
      datasets match (node order / label format `_19label.npy`, one-hot → argmax).
- [ ] Consider integrating as a `--mode onepoint` path in the expkit framework, or keep as a standalone NB.

## Assessment
Approach is sound and a genuine, complementary research angle (relative-coord local-subgraph generalization).
Worth adopting. Needs the debug-fn fix + path rewiring before it runs here.
