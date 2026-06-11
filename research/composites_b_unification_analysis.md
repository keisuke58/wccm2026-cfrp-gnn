# What unifies the two structures — the honest conceptual spine (paper B)

The reviewer-critical question for the 2-structure paper: *interstage sensing is a
STATIC surface-stress field; fairing sensing is DYNAMIC guided-wave signals — these
are fundamentally different. In what sense is this "one framework"?* This note
fixes the honest answer so the writing does not overclaim.

## Three layers — only the third is unified

| layer | interstage (段間) | fairing (フェアリング) | shared? |
|---|---|---|---|
| **(1) Sensing + detection (front-end)** | static surface DSPSS + mesh-physics GNN (Paper A) | dynamic guided waves (Lamb) + GNN (Payload, LGSTA 0.86) | **NO — modality-specific, by design** |
| **(2) Prognosis (Stage 3)** | FD anisotropic AT2 phase-field, ply delamination | interfacial debond growth (ERR + Paris), skin–core | **NO — different physics; SAME interface** |
| **(3) Decision core** | anomaly |z| · characterisation posterior · expected-cost clearance · end-to-end UQ · hierarchical fleet · design feedback | identical | **YES — this is the unification** |

**We do NOT claim to unify sensing/detection.** The two front-ends are genuinely
different (static stress vs dynamic waves) and stay structure-specific — that is
correct engineering, not a gap. What we unify and contribute is the **decision
layer**: take any per-structure detector + prognosis, and the SAME machinery
propagates uncertainty to an OK/REPAIR/RETIRE call, scores it against an exact
oracle, learns across a fleet, and feeds design.

## The unification is made concrete by a pluggable interface

Layer (2) is the seam: both structures implement a one-method `Prognosis`
protocol (`growth_probability(posterior, load) -> p`). `flight_clearance_generic`
consumes ANY prognosis through the shared expected-cost thresholds. So:
- interstage: phase-field → P(grow) → clearance;
- fairing: debond growth → P(grow) → clearance;
- future CFRP part: add ONE prognosis model, reuse everything downstream.

The decision-UQ result is the proof the layer-3 claim is real, not nominal:
- interstage: 85% decision accuracy, 0% dangerous-miss, ECE 0.18→0.06 (Platt).
- fairing: 92.5% decision accuracy, 0% dangerous-miss, ECE 0.075.
Same UQ story, two physically distinct structures, no re-engineering of the core.

## What genuinely transfers vs what is rebuilt per structure

- **Transfers (built once, validated once, reused):** the geometry-agnostic |z|
  anomaly screen; the expected-cost OK/REPAIR/RETIRE rule + its economic
  anchoring; the end-to-end decision-UQ + Platt recalibration; the
  hierarchical-Bayes fleet (leader effect); the design-feedback loop; the
  system-baseline value (lowest expected cost vs naive strategies).
- **Rebuilt per structure (and honestly so):** the sensor modality, the GNN
  detector, the Stage-2 characterisation features, and the Stage-3 prognosis
  physics.

## The one-sentence framing (for the abstract)
> "We do not unify CFRP damage *sensing* — interstage surface stress and fairing
> guided waves are different modalities — we unify the *reuse decision*: a
> structure-agnostic detect→characterise→prognose→clear→fleet→design core, with
> uncertainty propagated to the clearance call and validated against an exact
> oracle, demonstrated end-to-end on a perforated interstage and an H3 fairing
> through a pluggable prognosis interface, and extensible to further CFRP parts."

## Honesty guardrails (do not cross)
- Both prognoses are REPRESENTATIVE models (lumped constants, verified for
  numerical order / monotone-consistent with FEM, NOT calibrated certification
  values). The fairing prognosis is rank-validated vs the Payload guided-wave FEM
  (Spearman 0.87), not against a debond-growth test (future).
- The fairing front-end detector is Payload's GNN; in this paper the fairing
  Stage-0/2 fed to the decision core are representative/characterised, not the
  full GNN re-run inside B.
- Interstage measured-data DETECTION AUROC still awaits Kojima's per-specimen
  masks (the sim-to-real DETECTION number, distinct from the domain-gap +
  domain-adaptation results already in hand).
