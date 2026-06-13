# Stage 0-5 SHM decision framework — results of record

Single source of truth for the framework's headline numbers, every figure, and
the honest open items. Regenerate everything with:

```bash
python run_all.py            # all module --tests (22 modules)
python run_all.py --fig      # + regenerate every figure under paper_figs/
python run_all.py --quick    # skip the slow FD/FMPE modules
```

**Status (2026-06-12): 27 modules, 671/671 checks pass** (651 unit tests +
20 `cost_calibration` checks). Near-term venue = JSCES 2027 (May);
`nishioka_jsces2027.tex` carries the 3-structure story.

The framework is **structure-agnostic in the decision core, structure-specific
in sensing + prognosis**. It is demonstrated on **three real CFRP structures
spanning three sensing modalities**:

| structure | sensing modality | Stage-3 prognosis physics | real data |
|---|---|---|---|
| perforated interstage | surface stress (DSPSS) GNN | anisotropic AT2 phase-field delamination | Kojima TSA (measured) |
| H3 satellite fairing | guided-wave GNN | honeycomb skin-core debond (ERR+Paris) | OGW long-term (measured) |
| SRB-3 motor case | acoustic emission (AE) | filament-wound internal-pressure burst | 4TU AE .pridb (measured, CC0) |

---

## ⚠️ VALIDATION STATUS — read this before believing any number below
An honest, self-critical audit of what is and is NOT validated (2026-06-12). The
framework is, today, **a well-built SIMULATION demonstration, not an experimentally
validated capability.** Every result falls into exactly one of three tiers:

- **[R] real data, externally meaningful** — only the DETECTION front-ends touch
  real measurements: SRB-3 AE pristine-vs-damaged (4TU, real AE), the FEM-coupon
  detection harness, OGW temperature DA. *Even these validate detection, not
  prognosis.* The interstage **measured** detection AUROC is still BLOCKED (Kojima
  masks); the AE is a compression-coupon **proxy**, not the motor case itself.
- **[S] self-consistency, NOT ground truth** — every "oracle", oracle-decomposition
  clearance accuracy, decision-UQ, LOSO transfer, conformal, system_baseline number
  uses the SAME FEM/surrogate model as both the predictor and the "truth." These are
  **internal-consistency checks** (does UQ propagate correctly?), not validations
  against real failure. The CZM-surrogate R²=0.994 is a model approximating a model
  (half self-circular). Do not read "85% clearance accuracy" as "85% correct vs
  reality."
- **[U] representative / uncalibrated** — ALL Stage-3 prognosis physics (phase-field,
  debond ERR+Paris, burst hoop-stress, CZM p_cr), every cost/value constant, every
  fleet growth rate. Lumped, trend-level, NOT fitted to any real fracture/fatigue
  test. Absolute numbers are illustrative only.

**The core gap (concede openly):** no result has been compared to the real failure
of the actual interstage / fairing / SRB-3, nor to a real CFRP fracture/fatigue
coupon. Highest-leverage fixes, in order: (1) one structure end-to-end on real
measured data (interstage Kojima — external dependency); (2) calibrate ONE prognosis
case against a real run-to-failure coupon dataset (NASA PCoE composite is on disk,
real RUL validation NOT yet done — only loaded for detection). Depth > breadth:
better to validate one thread than to add another breadth module.

**Update (2026-06-12): fix (2) DONE for the Paris law (positive [R]).**
`composite_fatigue_calibration` fits the Paris delamination law to **real CFRP
Mode-I fatigue** (4TU DCB, 37 specimens). Within-specimen R²=0.98; Bayesian
population **m = 16.7 [CI 15.0–18.3]**, so the framework's representative **m=3 is
unsupported (~6× too shallow)**, and a single global law fails to generalise
(fibre-bridging R-curve). This is a genuine real-data calibration AND critique of
the Stage-3 prognosis. (The earlier `nasa_rul_validation` RUL attempt via raw
strain was an honest negative — that proxy is a dead-end; the Paris route via
clean tabular fracture data is the one that worked.) Remaining: real run-to-failure
of the actual structures, and the interstage Kojima measured end-to-end (fix 1).

Tier tags **[R]/[S]/[U]** are applied per row in the tables below.

---

## Stage 0 — anomaly / robustness
| module | tests | headline |
|---|---|---|
| `stage0_robustness` | 15/15 | geometry-agnostic \|z\| screen; refutes a σ=0.1 "collapse", true breakdown σ≈0.7–1.0 |

## Stage 1 — detection front-ends (3 structures, 3 modalities)
| module | tests | headline |
|---|---|---|
| `kojima_real_case` | 14/14 | real measured TSA DSPSS field-anomaly detector |
| `interstage_measured_detection` | 6/6 | harness proven on FEM coupon **AUROC 0.851**; measured AUROC BLOCKED on Kojima masks |
| `baselines_comparison` | 43/43 | physics \|z\| detector: AUROC #2 / **AUPRC #1** vs ML baselines |
| `srb3_motorcase` | 23/23 | **real AE** (4TU CC0, ~48k hits) pristine-vs-damaged RF **per-hit AUROC 0.888 / per-specimen 1.000** (linear ~0.52); + amplitude–duration damage-mode clusters |

- **interstage GNN (separate WCCM/Frontiers line, cited):** HybridMGN (cross-ply
  edges) in-dist macro-F1 **0.792** (defect-F1 0.781, AUPRC 0.988) > MeshGraphNet
  0.761 > GAT 0.685. OOD to unseen 1×1 size collapses to ≈0.33 across all archs
  (detection partially generalises, precise localisation does not).
- **fairing GNN (from Payload2026):** guided-wave LGSTA node-F1 **0.86** > SAGE 0.79.

## Stage 2 — characterisation / domain adaptation
| module | tests | headline |
|---|---|---|
| `tsa_sim2real` | 15/15 | measured-vs-FEM DSPSS gap KS≈0.09; quantile DA closes **97%** |
| `domain_adapt` | 12/12 | none/standardize/CORAL/quantile aligners; MMD² + proxy-A-distance gap report |

- **fairing OGW temperature DA (Payload):** quantile DA recovers FPR **14%→5%**
  (z-score worsens to 30%) — the cross-structure real-data sim2real spine.

## Stage 3 — prognosis + validation
| module | tests | headline |
|---|---|---|
| `fairing_debond_prognosis` | 14/14 | skin-core debond: ERR G=κσ²a² vs Gc + Paris growth → facesheet instability |
| `fairing_prognosis_validation` | 14/14 | vs Payload guided-wave FEM, severity Spearman **0.87** (honest: static-detection FEM) |
| `srb3_prognosis_validation` | 21/21 | vs **CZM 495-case critical-pressure FEM**: pcr↓ with a0 ρ=**−0.994**; prognosis severity ρ=**+0.994** (p_burst) / +0.803 (P(grow)); location+interface sensitive |
| `phasefield_3d` | 12/12 | 3-D AT2 phase-field, MMS convergence **2.05** |
| `flight_load_spectrum` | 12/12 | simple load proxy under-predicts life **3.4×** (S-N anchored) |
| `physics_validation` | 12/12 | MMS 2-D **2.04** / 3-D **2.05**, mesh sensitivity 0.4% |
| `nasa_rul_validation` **[R, honest negative]** | 16/16 | **real-data confrontation** — RUL vs NASA PCoE composite **run-to-failure** coupons. Method sound on controlled data (noiseless → α-λ≈1.0) but **FAILS on real coupons (α-λ ≈ 0%)**: only 3/13 give a usable raw-strain trajectory, no calibrated common failure level. Honest finding: RUL via that proxy is unvalidated (raw-strain is a dead-end; gauge debonds). |
| `composite_fatigue_calibration` **[R, POSITIVE]** | 14/14 | **first real-data prognosis calibration** — fits the Paris delamination law to **real CFRP Mode-I fatigue** (4TU Yao/Alderliesten DCB, 37 specimens, 1270 (ΔG,da/dN) pts). **Within-specimen Paris fits excellently (median R²=0.98)**; Bayesian population posterior gives **real m = 16.7 [95% CI 15.0–18.3]** — the framework's representative **m=3 is unsupported (~6× too shallow)**. A single global law fails to generalise (LOSO R²<0; fibre-bridging R-curve). Concrete real-data fix for the Stage-3 Paris constant + an honest critique of the single-(C,m) assumption |

## Decision UQ — 3 structures, symmetric audited form
oracle decomposition + 95% bootstrap CI + dangerous-miss + Platt ECE recalibration.

| structure | module | clearance acc | dangerous-miss | ECE raw→cal |
|---|---|---|---|---|
| interstage | `decision_uq` (21/21) | **85.4% [95% CI 75–94]** | **0%** | 0.13→0.03 |
| fairing | `fairing_pipeline` (26/26) | **92.5%** | **0%** | →0.075 |
| SRB-3 | `srb3_decision_uq` (24/24) | **65% [95% CI 50–80]** | **0%** | 0.335→0.118 |

- SRB-3 reads lower because the burst margin is physically near-binary (keeps the
  safety factor or loses it) → OK/RETIRE-dominated, no contrived REPAIR band; the
  symmetry is in the **audited UQ form**, not the headline percentage.
- `cost_calibration` (20 checks): decision thresholds use the conservative baseline
  **α=0.02 / β=0.48**; economic anchoring gives α≈0.035 [0.021,0.055] /
  β≈0.53 [0.276,0.927] — CIs contain/abut the baseline (consistent, documented).

## Decision-core rigor — the scientific spine
| module | tests | headline |
|---|---|---|
| `loso_decision_transfer` | 49/49 | **LOSO cross-structure transfer**: cost-optimal thresholds transfer zero-shot (0 pp acc gap, **0% dangerous-miss**; RAW fixed-0.5 = 65–79% acc, 15–38% dangerous-miss); cross-structure Platt cuts held-out ECE (SRB-3 0.131→0.074, fairing 0.110→0.103) |
| `conformal_clearance` | 91/91 | distribution-free split-conformal clearance set; pooled coverage 0.95→**97%**, mean set size 1.5–2.3; one-sided RETIRE-omission guarantee with **K-fold cross-conformal shore-up** (pooled omission within budget at every α_danger: 19.7/9.4/3.4% ≤ 20/10/5%) + Mondrian cross-structure |
| `voi_inspection` | 28/28 | Stage-2.5 value of information: EVPI/EVSI peak at **P≈0.48**, inspect-band **P∈[0.38,0.58]** = quantifies the abstract's "fragile at P~0.3–0.5" |
| `lifetime_policy` | 20/20 | **sequential retire/repair/fly optimal-stopping DP** ("how many flights left?"): the lifetime policy extracts **+37% more lifetime value** than the framework's myopic per-flight clearance (220.8 vs 160.9) at ~0% failures, E[flights to retire]≈21.7; resolving growth-rate uncertainty (SD 0.45→0.05) **buys back 13.5 lifetime value** = the lifetime payoff of Stage-4 fleet learning |
| `pomdp_inspection` | 31/31 | **belief-state POMDP over the PARTIALLY-observed crack** (fuses `lifetime_policy`+`voi_inspection`): adds an INSPECT action (costly noisy read, sharpens the belief) to fly/repair/retire, solved by assumed-density backward induction. **Headline — state uncertainty dominates**: act-on-the-mean (point crack read) **fails ~30% and loses ⅓ of value** (133 vs 200) vs the belief-state policy (0.3% fail). **Active inspection = modest, κ-gated increment** (POMDP−never-inspect ≈ +0.25 at baseline NDT noise, up to +5 for sharper reads) — not a free lunch; INSPECT is bought only in the **uncertainty-gated mid-band** = `voi_inspection`'s VoI result made sequential/endogenous. Honest: Gaussian moment-projection belief, known growth rate (crack-only state) |

## Stage 4/5 — fleet learning, design, system baseline
| module | tests | headline |
|---|---|---|
| `system_baseline` | 76/76 | 3-structure framework vs naive: aggregate **18.11 cost / 0% dangerous-miss = rank #1** (next point-est 23.15/14.7%, oracle floor 15.18); per-structure 13.83 / 22.07 / 18.43, all 0% |
| `cross_structure_fleet` | 16/16 | **3-level "fleet of fleets"** (Stage-4 deepened; learning-layer analogue of LOSO): a cold-start structure (3 vehicles × 4 flights) borrows from the other fleets → growth-rate posterior **×2.96 sharper** (and more accurate: \|err\| 0.32→0.12) than within-structure pooling; global M0 recovered −3.09 (true −3.0); a brand-new (0-vehicle) structure starts with a **×4.4 tighter** prior than uninformative |
| `cross_structure_design` | 20/20 | **portfolio toughening-budget allocation** (Stage-5 cross-structure; chains Stage-4): optimal split of a shared budget = lowest total risk + **18% better worst-case** vs uniform, ≫ greedy-worst/no-budget (12.7 vs 33/55); **warm-start payoff** — using the cross-structure (vs within-structure) risk estimate **lowers allocation regret** (mean −0.07; at 2 cold flights 0.26→0.12), i.e. fleet-of-fleets learning concretely improves the design decision |
| `design_feedback` | 37/37 | hierarchical-Bayes fleet sharpening (σ-ratio 2.75→1.25); multi-objective design optimum (off-axis 75°→15°) |

---

## Honest open items (the things NOT to overclaim)
1. **Representative constants.** Every Stage-3 prognosis (phase-field / debond /
   burst) uses lumped representative constants — trends, not certification values.
   The contribution is the end-to-end pluggable framework, not calibrated numbers.
2. **Interstage measured AUROC — BLOCKED.** `interstage_measured_detection` is
   harness-ready; the measured detection AUROC awaits per-specimen defect masks
   (Kojima TSA campaign). One `detection_auroc(field, mask)` call from closing.
3. **Kojima TSA data = co-author pending.** Use is verbally OK'd; formal
   co-authorship/scope agreement with Kojima (CPB 2025 data owner) still pending.
4. **Conformal RETIRE-omission — shored up (2026-06-12).** The single-split
   one-sided guarantee was high-variance on small real splits (pooled breached:
   33/17/6% vs 20/10/5% budget). Fixed with **K-fold cross-conformal** (every
   RETIRE point tested once, tighter q̂): the **pooled** guarantee now holds at
   every budget (19.7/9.4/3.4% ≤ 20/10/5%); per-structure only ~1–2 pp finite-
   sample noise remains at the αd=0.20 small-N corner. **Mondrian** cross-
   structure borrowing holds for interstage↔fairing but FAILS for srb3 (its
   near-binary burst gives a different RETIRE-score distribution) — an honest
   bound on cross-structure exchangeability. Coverage is still MARGINAL, not
   conditional/per-input.
5. **LOSO N=3 folds.** The decision-layer transfer is shown across 3 structures /
   physics / modalities — strong evidence it is shared, not fleet-scale proof; and
   the cost thresholds are partly cost-anchored so part of their transfer is
   by-design (the emergent finding is the shared miscalibration pattern).

## SRB-3 AE raw data
The 4TU `.pridb` AE files (~27 MB) are git-ignored (`data/srb3_ae/`); they are
public-domain (CC0), re-downloadable from 4TU article 21621381. Compression AE is
used as a real-data **proxy** for the motor-case AE front-end (same hit-feature
physics — matrix crack / fibre break / delamination — not a pressure-vessel burst test).
