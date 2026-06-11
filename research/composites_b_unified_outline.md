# Composites B — unified 2-structure decision-framework paper (B) — outline

Strategic decision (2026-06-12): **two papers**.
- **Paper A** = interstage surface-stress GNN (WCCM 2026 / Frontiers 2025, already
  written). Stays standalone; **cited** by B as the interstage Stage-1 detector.
- **Paper B (this outline)** = the Stage 0-5 **decision framework**, demonstrated on
  **two real spacecraft CFRP structures**: a perforated **interstage** and an H3
  **satellite fairing**. **Absorbs the Payload2026 fairing work** — Payload is NOT
  published standalone (impact concentrated into one strong paper).

## Working title
"A structural-health **decision framework** for reusable CFRP spacecraft
structures: from GNN detection to fleet-level reuse decisions, demonstrated on a
perforated interstage and an H3 fairing"

## Headline contributions (keep to 4 — depth over breadth)
1. **Structure-agnostic decision pipeline**: detect → classify → characterise →
   prognose → clear (OK/REPAIR/RETIRE) → fleet-learn → design, with an
   expected-cost decision rule anchored to launch economics.
2. **Two real detection front-ends**: interstage = surface-stress (DSPSS) GNN
   (cite A); fairing = guided-wave GNN (Payload, LGSTA best F1 0.86).
3. **End-to-end decision UQ + verification**: oracle-scored clearance accuracy
   85% / dangerous-miss 0% / Platt-recalibrated confidence (ECE 0.18→0.06);
   FD solver MMS-verified 2nd order in 2-D and 3-D.
4. **Sim-to-real with domain adaptation, on BOTH structures**: interstage =
   measured TSA-derived DSPSS + quantile DA (97% gap closed); fairing = real OGW
   long-term **temperature** shift, quantile DA recovers FPR 14%→5% (z-score
   makes it worse) — the cross-structure real-data spine.

## Section structure (~12-14 pp; figures in [], SI = supplementary)
1. **Introduction** — reusable CFRP spacecraft structures (interstage + fairing),
   the fly-again / how-many-flights-left decision; gap = no end-to-end,
   uncertainty-propagated, real-data-grounded decision pipeline.
2. **Two structures & data** — interstage (perforated, static surface DSPSS) +
   fairing (honeycomb skin, dynamic guided waves); FEM + real data (Kojima TSA;
   OGW long-term). [fig: the two structures + the pipeline schematic]
3. **Detection front-ends (Stage 1)** — interstage surface-stress GNN (cite A,
   1 result fig); fairing guided-wave GNN (LGSTA 0.86, 1 result fig). The two
   structures enter here; downstream is shared. [fig: two detection panels]
4. **Anomaly + characterisation (Stage 0/2)** — geometry-agnostic |z| screen
   (validated AUROC 0.85, ties/beats ML baselines, AUPRC-leading); FMPE
   posterior θ. [fig: stage-0 / baselines → maybe SI]
5. **Prognosis & clearance (Stage 3)** — FD anisotropic AT2 phase-field (interstage
   delamination) → flight clearance; **honest line: fairing honeycomb debond =
   different physics, detection+decision applied, prognosis structure-specific /
   future**. Surrogate (FNO 142×) for speed. [fig: crack_surrogate → SI]
6. **End-to-end decision UQ** — oracle decomposition (85% / 0% dangerous-miss),
   Platt recalibration (ECE 0.18→0.06), α/β anchored to launch economics
   (verdict fragile at P~0.3-0.5). [fig: decision_uq + cost_calibration]
7. **Fleet learning & design feedback (Stage 4/5)** — hierarchical Bayes
   fleet-leader (2.75→1.25); multi-objective design (off-axis 75°→15° interior
   optimum). [fig: fleet + design_pareto]
8. **Solver verification** — MMS 2-D 2.04 / 3-D 2.05, mesh 0.4%; load spectrum
   (proxy under-predicts life 3.4×) + S-N anchoring. [fig: physics_validation]
9. **Sim-to-real & domain adaptation (BOTH structures)** — interstage TSA DSPSS +
   DA (97%); **fairing OGW temperature DA (FPR 14%→5%, z-score worsens to 30%)**.
   The cross-structure real-data result. [fig: tsa_sim2real + payload_da_longterm]
10. **Discussion** — structure-agnostic framework vs structure-specific physics;
    when learning is needed; honest limitations.
11. **Conclusion**.

## Figure budget (~9 main; rest SI)
KEEP (main): structures+schematic · two-detection panel · decision_uq ·
cost_calibration (or fold into decision_uq) · fleet · design_feedback_pareto ·
physics_validation · tsa_sim2real · **payload_da_longterm** (the real-data
cross-structure highlight).
SI / fold: crack_surrogate, fno_inversion, conformal_surrogate, phasefield_3d,
flight_load_spectrum, baselines_comparison, kojima_real_case, payload_da_gw
(the negative FFT demo → one sentence + SI).

## Assets to pull from Payload2026 (into B)
- Fairing GNN-SHM results: LGSTA node_cls F1 0.86, SAGE 0.79, multitask + reg
  (project_payload_link). Detection panel + the architecture note (condensed).
- Real-data: OGW long-term temperature DA (`scripts/payload_da_longterm.py`,
  FPR 14%→5% by quantile; z-score worsens). The headline cross-structure sim2real.
- Honest negatives kept brief: cross-frequency FFT demo (no defect signal → why
  proper features), structure-mismatch sim-to-real (FEM↔OGW).

## ✅ Pre-writing gap closure — ALL addressed (2026-06-12)
Before writing, the honest gaps were worked through:
- **#1 fairing end-to-end** ✅ `fairing_pipeline.py` (26/26): decision-UQ 92.5% /
  dangerous-miss 0% / ECE 0.075, fleet sharpening ×2.73, design lever P(grow)
  0.69→0 — the whole framework runs on structure 2, not just clearance.
- **#2 prognosis validation** ✅ `fairing_prognosis_validation.py` (14/14):
  rank-validated vs the Payload guided-wave FEM (Spearman 0.87, monotone in
  radius, position-sensitive); honest limit = static-detection FEM, not growth.
- **#3 interstage measured detection** ⏳ `interstage_measured_detection.py` (6/6):
  harness ready (coupon AUROC 0.851 proven), measured AUROC BLOCKED on Kojima
  masks — `detection_auroc(field,mask)` one call from closing.
- **#4 fairing Stage-2** ✅ `Payload2026/src/fairing_stage2.py`: debond severity
  posterior (mean+sd) from GW features → bridges to Stage-3 (rank corr 0.95).
- **#5 unification analysis** ✅ `composites_b_unification_analysis.md`: the
  three-layer honest spine (sensing NOT unified; decision core IS).
- **#6 system baseline** ✅ `system_baseline.py` (51/51): framework lowest expected
  cost (15.15 vs 16.45–21.25, oracle floor 13.45), tied 0% dangerous-miss.
Only blocker left = #3's measured masks (data, from Kojima). Otherwise ready.

## Pluggable prognosis interface (the layer-2 seam)
The framework is END-TO-END on BOTH structures via a PLUGGABLE prognosis
interface (`fairing_debond_prognosis.py`):
- interstage Stage-3 = FD phase-field delamination (InterstagePrognosis);
- fairing Stage-3 = honeycomb skin-core DEBOND growth (FairingPrognosis:
  interfacial ERR G=kappa*load^2*a^2 vs Gc + Paris da/dN -> facesheet instability);
- both implement a tiny `Prognosis` protocol; `flight_clearance_generic` consumes
  ANY prognosis with the SHARED calibrate_thresholds -> structure-agnostic decision.
- => the "fairing = detection only" caveat is removed; the fairing runs through the
  same Stage-3->clear (small debond OK, large RETIRE). [fig: fairing_debond_prognosis]
- **Future CFRP parts** (shaft, panel, tank, prosthetic ...) plug in by adding ONE
  prognosis model and reusing detect/characterise/decide/fleet/design — the
  framework's extensibility is now structural, not just aspirational.
- ✅ **DEMONSTRATED — 3rd structure SRB-3 motor case** (`srb3_motorcase.py`, 23/23,
  2026-06-12): a filament-wound CFRP pressure vessel added with a NEW sensing
  modality (acoustic emission) AND new physics (internal-pressure burst), proving
  extensibility concretely. Stage-1 detection on REAL experimental AE data (4TU
  CC0 "AE monitoring in CFRP compression tests", .pridb Vallen, ~48k hits):
  group-aware pristine-vs-damaged RF → per-hit AUROC 0.888 / per-specimen 1.000
  (linear baseline ~0.52); + amplitude–duration damage-mode clustering (matrix /
  delam / fibre-break), reported honestly (compression pristine coupons also emit
  high-energy AE → the multivariate detector, not one cluster fraction, is the
  read-out). Stage-3 `SRB3Prognosis` (hoop σ=pR/t, residual burst
  p_burst0(1−(a/a_crit)^n), Paris growth, unsafe when p_burst/p_op<SF=2) clears
  small OK / medium REPAIR / large RETIRE through the SAME flight_clearance_generic.
  Honest: compression AE = real-data PROXY for the motor-case AE front-end (same
  hit-feature physics, not a burst test); lumped representative constants. fig
  srb3_motorcase.pdf. → optional 3rd column in the paper's structure schematic,
  or the headline "extensible to a new modality+physics" Discussion result.

## Honesty lines (reviewer-critical — do NOT overclaim)
- Stage 0/2/4/5 + decision UQ + the clearance rule are structure-agnostic (both).
- BOTH structures now have a Stage-3 prognosis (interstage phase-field /
  fairing debond), but each is a REPRESENTATIVE model (lumped constants, trends
  not certification values) — be explicit; the contribution is the end-to-end
  pluggable framework, not calibrated fairing certification numbers.
- Interstage measured data (Kojima TSA) use = Kojima co-author (CPB 2025 owner).
- DANN is a synthetic prototype; the real DA result is the unsupervised
  quantile/CORAL temperature recovery, not a DANN head-to-head.

## Next actions (when writing starts — after exams)
1. Restructure `paper/composites_b_draft.tex` into the two-structure frame:
   add §2 (two structures), §3 (two detection front-ends), §9 (fairing OGW DA).
2. Pull the fairing detection + temperature-DA figures from Payload2026.
3. Trim the current Stage-by-Stage prose to the 4-contribution spine; push
   surrogate/inversion/conformal/3D/spectrum/baselines to SI.
4. Reconcile authorship: A authors + Payload (fairing) contributors + Kojima
   (interstage TSA) + Muramatsu/JAXA.
