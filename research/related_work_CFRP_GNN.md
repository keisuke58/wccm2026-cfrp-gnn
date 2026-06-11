# Related work — CFRP × GNN damage detection (for the Composites B paper)

Curated literature notes for positioning the planned Composites Part B paper
(surface-stress DSPSS + mesh-physics GNN + OOD generalization + detection/
localization split on perforated interstage CFRP). Two anchors: Kojima et al.
(the direct surface-stress lineage, to BUILD ON) and PIGMID (an adjacent
guided-wave GNN, to DIFFERENTIATE FROM).

---

## [1] PIGMID / "PIGMind" — Li et al. 2026 (adjacent, different modality)

- **Title**: Interpretable and physics-informed graph neural modeling for
  intelligent damage localization in CFRP composites
- **Authors**: Xinming Li, Jinrui Zhang (equal), Kehui Zhu, Qingrui Hu,
  Lingyu Sun, Jiawei Gu, Wenhan Lyu, Yanxue Wang
- **Venue**: Composites Part B: Engineering, Vol. 315, 113469, 15 Apr 2026
- **DOI**: 10.1016/j.compositesb.2026.113469
- **Funding/affil**: Chinese (NSFC 52275079/12204031/62503500, Guangxi, BUCEA) —
  unrelated group.
- **Name note**: highlights say "PIGMind", body says "PIGMID" — same framework.

**What it does**
- Input modality = **ultrasonic guided waves (Lamb waves)** from an 8-sensor PZT
  array (time-series signals along propagation paths). NOT surface stress.
- Graph: nodes = propagation paths; edges = **physics-derived energy indicators**
  (peak amplitude × area coupling) → physics-constrained adjacency encoding the
  directional dependence of guided-wave energy.
- Innovations: (1) physics-guided amplitude–area edge weighting; (2) node
  personalized embeddings via amortized learning (beats the global-parameter-
  sharing "one-size-fits-all" of standard STGNNs, capturing path heterogeneity);
  (3) isotropic spatiotemporal message passing on the physics-weighted
  adjacency; (4) clustering-regularized decoding for latent geometric/physical
  consistency.
- Result: mean localization error **10.18 mm** (8-sensor CFRP plate), +7.8% over
  graph baselines, robust to anisotropy/noise.
- Honest limits (stated): baseline-available lab conditions, localization within
  the array-covered region only; future = baseline drift (temp/load), reduced
  sensor density, irregular layouts.

**Why it matters to us**
- DIFFERENT sensing modality (guided-wave signals vs full-field surface stress)
  → not a direct competitor; cite & differentiate in one line: "Unlike
  guided-wave GNNs (PIGMID), we operate on full-field surface stress (DSPSS) on
  the structural FE mesh itself."
- The "physics-informed graph" banner overlaps conceptually → make our "physics"
  explicit: FE-mesh structure + DSPSS, not wave-energy propagation.
- Signals "physics-informed GNN for CFRP damage" is hot in CPB 2026 (already 12
  citations) — tailwind, not headwind.
- **PIGMID cites Kojima et al. [ref 20]** as related work (surface principal
  stress → 3D internal defect via transfer-learning CNN) → confirms the Kojima
  surface-stress lineage is internationally recognized prior art.

---

## [2] Kojima et al. — the direct surface-stress lineage (BUILD ON, cite)

The data and methodology our sim-to-real pillar extends. Use of the measured TSA
dataset is cleared with Kojima (co-authorship understood).

- **Kojima, Hirayama, Endo, Harada, Muramatsu (2025)** — "Transfer-learning-aided
  defect prediction in simply shaped CFRP specimens based on stress distribution
  obtained from finite element analysis and infrared stress measurement",
  *Composites Part B: Engineering* 291, 111958.
  DOI 10.1016/j.compositesb.2024.111958 (SSRN preprint 10.2139/ssrn.4767114).
  → FEM stress → IR/TSA stress **transfer learning** (sim-to-real) on simple
  shapes; the source of the measured TSA/DSPSS data. **Most directly overlapping
  prior work — cite and differentiate hard.**
- **Kojima, Muramatsu, … (2025)** — "Development of defect localization method for
  perforated CFRP specimens using FEM and graph neural network", *Frontiers in
  Materials*. DOI 10.3389/fmats.2025.1652484. → DSPSS-based FEM+GNN 3D defect
  localization; the WCCM/this-paper baseline ("先行研究").
- **Kojima, Hirayama, Harada, Muramatsu (2024)** — "Discussion on infrared stress
  measurements based on finite element analysis of transient heat conduction",
  *Mechanical Engineering Journal* 11(4). DOI 10.1299/mej.23-00571. → the IR→DSPSS
  transient-heat-conduction processing (the fourier_* / heat_conduction pipeline).
- **Kojima, Muramatsu, … (2022)** — "Inverse estimation method for internal
  defects based on surface stress of CFRP using machine learning", *Advanced
  Composite Materials*. DOI 10.1080/09243046.2022.2052786.

---

## Our differentiation (three-way table for the related-work section)

| axis | Kojima 2025 (CPB) | PIGMID (Li 2026, CPB) | **Ours (planned CPB)** |
|---|---|---|---|
| modality | surface stress (FEM→IR) | guided waves (PZT array) | surface stress (DSPSS) |
| model | transfer-learning CNN | physics GNN (wave energy) | **mesh-physics GNN** |
| geometry | simply shaped coupons | instrumented plate | **perforated interstage 3-D** |
| novelty | sim2real transfer | path heterogeneity + interpretable | **OOD generalization + detection/localization split** |
| physics | noise-augmented sim pretrain | wave-energy adjacency | **FE-mesh topology + DSPSS** |

Our distinct contributions vs both: (i) MeshGraphNet-class architecture sweep on
perforated interstage; (ii) OOD size/position generalization quantified
(detection generalizes, precise localization collapses — a finding invisible to
macro-F1); (iii) noise-robust recipe with no-defect negatives; (iv) sim-to-real
on measured DSPSS (extending Kojima 2025, with co-authorship).

---

## [3] Other related work — landscape (DOIs marked [verify] need a check)

### (A) GNN / graph learning for composite & structural damage — guided-wave lineage
- **Li et al. 2026 (PIGMID)** — see [1] above. Guided-wave physics GNN, CPB.
- **Sun et al. — P-GCN + 1D-CNN STGNN with an elliptical-propagation-law graph**
  (PIGMID ref [25]); one of the earliest STGNN guided-wave localization frameworks.
  [verify]
- **Graph-in-Graph (G-GCN)** — splits each guided-wave signal into time–frequency
  segments → segment-relation graph → location (PIGMID ref [24]). [verify]
- **Baseline-free assisted Lamb-wave damage detection in CFRP using graph
  convolutional networks + Transformer** — *Composite Structures* ~2024,
  pii S026322412402044X. Guided-wave GCN+Transformer.
- **Real-time damage detection & localization on aerospace structures using GNNs**
  — *Journal of Sensor and Actuator Networks* (MDPI) 14(5):89, 2025. Nodes =
  strain-measurement points, strain mode shapes as features; binary + localized
  spatial-probability outputs. Closest in spirit to a *structural* (not
  wave-signal) graph, but uses strain modes, not full-field DSPSS.

### (B) Surface / full-field stress ML surrogates (closest to our DSPSS input)
- **Composite U-Net surrogate for stress & damage in CFRP deformation** — arXiv
  2504.14143, 2025. Auto-regressive U-Net predicting full-field stress AND
  damage (crack init/propagation). Field-based like us, but CNN/grid, no graph,
  no OOD study.
- **Deep learning for multi-component stress fields in fiber-reinforced
  composites under different load paths** — *Composites Sci. & Tech.* 2025,
  pii S0266353825001666. Full-field stress prediction. [verify]
- **Kojima et al. (Frontiers 2025 / CPB 2025)** — see [2]; DSPSS surface stress,
  the direct lineage.

### (C) Sim-to-real / transfer learning for composite damage
- **Kojima et al. 2025 (CPB)** — see [2]; FEM→IR(TSA) transfer learning. THE
  reference for our sim2real pillar.
- **Deep transfer learning for localization of damage area in composite laminates
  using acoustic-emission signals** — *Sensors* 2023, PMC10053609. CNN +
  fine-tuning; 96.4% with 900 samples (≈ direct-1800), 17.7% of the training
  time. Transfer-learning-for-composite-NDT precedent (different modality, AE).
- **Deep transfer learning fusing monitoring data with physical mechanism** —
  *Eng. Appl. of AI* 2023, pii S0952197623004293. [verify]

### (D) Physics-guided / physics-constrained DL (our "physics" framing peers)
- **Physics-guided deep learning for damage detection in CFRP structures** —
  *Composite Structures* 2024, pii S0263822324000175.
- **Symmetry-constrained neural networks for detection & localization of damage in
  metal plates** — *APL Machine Learning* 3(2):026106, 2025. Physics-constrained,
  metal (not CFRP).
- **Unsupervised Kolmogorov–Arnold autoencoder, baseline-free, guided waves** —
  arXiv 2508.01081, 2025. Unsupervised damage detection/localization.

### (E) Thermoelastic stress analysis (TSA) for composite damage (our measured modality)
- **De Finis et al. 2020** — "Evaluation of damage in composites by using
  thermoelastic stress analysis", *Fatigue Fract. Eng. Mater. Struct.* 43,
  DOI 10.1111/ffe.13285. TSA → stiffness-degradation assessment.
- **Dulieu-Barton et al.** — foundational TSA on CFRP laminates for damage
  quantification (BSSM). [verify exact ref] — the classical TSA-NDT anchor.

---

## Expanded differentiation (where each axis sits)

| work | sensing | model | structure | physics prior | sim2real | OOD study |
|---|---|---|---|---|---|---|
| Kojima 2025 (CPB) | surface stress (FEM→IR) | transfer-learning CNN | simple coupons | noise-aug sim pretrain | ✅ FEM→IR | – |
| PIGMID (Li 2026) | guided waves (PZT×8) | physics STGNN | instrumented plate | wave-energy adjacency | – | – (lab, in-array) |
| GNN-aerospace (MDPI 2025) | strain mode shapes | GNN | aerospace panel | spatial graph | – | – |
| U-Net surrogate (2025) | full-field stress | CNN U-Net | coupon | – | – | – |
| AE-transfer (2023) | acoustic emission | CNN + fine-tune | laminate | – | partial | – |
| **Ours (planned CPB)** | **full-field surface stress (DSPSS)** | **mesh-physics GNN (MGN-class)** | **perforated interstage 3-D** | **FE-mesh topology + DSPSS** | **✅ measured DSPSS (22 spec.)** | **✅ size/position; detect vs localize split** |

**Single-sentence positioning**: *"Whereas prior CFRP GNNs operate on guided-wave
sensor signals (PIGMID) or simple-coupon surface stress with transfer-learning
CNNs (Kojima 2025), we learn directly on the structural FE mesh from full-field
surface stress (DSPSS), and are the first to quantify OOD size/position
generalization — showing detection transfers while precise localization
collapses — and to validate against measured TSA-derived DSPSS on a 22-specimen
set."*
