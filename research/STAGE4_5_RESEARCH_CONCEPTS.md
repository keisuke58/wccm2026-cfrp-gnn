# Stage 4--5 Research Concepts — Fleet Learning, Differentiable Inversion, Semiconductor Transfer

Forward-looking research concepts built on the implemented Stage 0--3 SHM stack
(per-node Mahalanobis detection → GNN classification → FMPE calibrated posterior
→ phase-field + Carrara fatigue prognosis → flight clearance). Drafted 2026-06-11.
Intended for: Keio (Muramatsu lab) research proposal, PhD direction, startup thesis.

Context anchors: SpaceX condition-based monitoring + staged 20→40-flight
certification + fleet-leader practice (formalised here); Muramatsu lab =
computational solid mechanics / phase-field / multiscale / quantum-annealing-ML.

---

## Concept A — Hierarchical Bayesian Fleet Learning (Stage 4)
*Target phase: Keio M1 (2027). Venue: Reliability Eng. & System Safety / SHM journals.*

### Problem
Stage 0--3 reasons about **one vehicle, one inspection**. A reusable fleet
generates a stream of (measurement, posterior, observed growth, decision)
tuples across many tail numbers and flights. Each booster has its own
manufacturing/usage idiosyncrasy, yet they share a common physics. Treating
each vehicle independently wastes the fleet; pooling them naively ignores
individual variation.

### Formulation
Hierarchical Bayesian model over the fleet:
- **Global (fleet) prior** `φ` on damage-growth parameters (e.g. effective
  fracture toughness Gc, fatigue rate ᾱ_T, defect-incidence rate).
- **Vehicle-level** parameters `θ_v ~ p(θ_v | φ)` — manufacturing/lot variation.
- **Flight-level** observations: each inspection's FMPE posterior is the
  likelihood term; observed crack growth between flights updates `θ_v` and,
  through it, `φ`.

`p(φ, {θ_v} | all data) ∝ p(φ) ∏_v p(θ_v|φ) ∏_{f} p(measurement_{v,f} | θ_v)`.

Inference: TMCMC / SMC (direct line from the biofilm thesis) or amortised
variational. As flights accumulate, `φ` sharpens → the per-vehicle prior at the
*next* inspection is tighter → fewer measurements needed for the same clearance
confidence. This is exactly the SpaceX "fleet-leader" loop made explicit: the
lead vehicle's deep-dive inspections are high-information likelihood draws that
update the fleet prior `φ`, certifying the rest.

### Novelty / why it's open
Hierarchical Bayes is standard; its use as the *certification engine* for a
reusable-vehicle fleet, coupling per-inspection amortised posteriors (Stage 2)
into a fleet-level life model, is not in the SHM literature. Bridges
digital-twin and reliability-growth communities.

### Prerequisites
Multi-vehicle, multi-flight data (synthetic first: simulate a fleet with the
Stage-3 model + lot variation; real later). Reuse TMCMC/GPU machinery.

### Muramatsu-lab connection
Phase-field forward model supplies the growth likelihood; the lab's
multiscale/UQ interest covers the hierarchical structure. Quantum-annealing
angle (optional): inspection-scheduling as a combinatorial expected-information
maximisation = QUBO.

---

## Concept B — Differentiable FMPE × Phase-Field Inversion (PhD-grade)
*Target phase: Keio M2 / PhD (2028+). Venue: CMAME / IJNME / JMPS.*

### Problem
Stage 2 (FMPE) and Stage 3 (phase-field) are currently a **one-way pipe**:
posterior → seed → forward simulate. The inverse map (measurement → defect
parameters) is learned amortised and never sees the forward physics; the forward
model never informs the inference. We want a single differentiable object where
the gradient of the *physical* misfit flows back into the *probabilistic*
inference.

### Formulation
Make the phase-field forward solver **differentiable** (adjoint or
autodiff-through-solver; the existing FD AT2 is already amenable). Then:
- **Physics-consistent posterior**: train the FMPE/flow so that sampled
  parameters θ, pushed through the differentiable PF forward `F(θ)`, reproduce
  the observed field — add a forward-consistency loss `‖F(θ) − x_obs‖` whose
  gradient ∂F/∂θ is available. The amortised posterior is regularised by the
  PDE, not just by labelled examples.
- **Diffusion-posterior-sampling (DPS) variant**: use the differentiable
  forward as the likelihood score in a score-based / flow posterior sampler →
  `p(θ | x) ∝ p(x | F(θ)) p(θ)` sampled by guided ODE. Replaces the current
  label-trained FMPE with a *simulation-grounded* posterior — needs no labelled
  (θ, x) pairs, only the forward model.

### Novelty / why it's open
Literature has (i) PINN-based PF forward/inverse [Manav 2024, CMAME], (ii)
Bayesian PF parameter estimation [Khodadadian 2020], (iii) differentiable PF
contact/fracture [2026]. The specific combination — *amortised flow-matching
posterior whose training is closed through a differentiable phase-field
likelihood* (one gradient loop linking inverse and forward) — is not done.
High-IF target.

### Prerequisites
Differentiable PF (JAX rewrite of the AT2 FD solver — wafer-proc-sim already
uses JAX for the surrogate, so the kernel exists). Validation against the
current label-trained FMPE (Stage 2) as baseline.

### Muramatsu-lab connection
This is the lab's core: variational phase-field + computational inverse
mechanics, now made differentiable and Bayesian. The thesis spine.

---

## Concept C — Semiconductor Cross-Domain Transfer (startup seed)
*Target phase: parallel / post-graduation. Venue: product, plus J. Manuf.
Process / IEEE Trans. Semicond. Manuf.*

### Problem
The same Stage 0--5 stack should govern any *physical asset re-inspected over a
lifecycle*. Wafer dicing / grinding (DISCO domain) is structurally identical:
fixed-geometry stress fields, anisotropic brittle fracture (SiC, Si), defect
detection → characterisation → growth prognosis → "rework or scrap" decision.

### Mapping
| Rocket (Payload/WCCM) | Semiconductor (wafer-proc-sim) |
|---|---|
| DSPSS thermoelastic field | wafer stress / DSPSS field |
| ply / interface delamination | dicing-induced chipping / subsurface crack |
| flight load cycles | process steps (grind, dice, anneal) |
| flight clearance OK/REPAIR/RETIRE | wafer pass / rework / scrap |
| anisotropic AT2 (β = fiber dir) | anisotropic AT2 (β = crystal orientation) — **same code** |
| FMPE defect posterior | KABRA GP + same FMPE on process params |
| fleet learning | per-lot / per-tool hierarchical learning |

The 2D AT2 anisotropic simulator was *originally built for SiC* — the rocket
work is the lateral transfer, not the other way round. wafer-proc-sim already
has the KABRA GP + TuRBO BO (+71% throughput) and a JAX surrogate (1.3% error).

### Novelty / why it's a moat
The cross-domain "operational intelligence for physical assets" framing —
identical Bayesian-physics stack across aerospace and semiconductor — is the
startup thesis (Physics-Informed AI Simulation platform). Domain knowledge from
DISCO is the barrier to entry; the rocket work de-risks the method publicly.

### Prerequisites
Real wafer NDT data (DISCO employment unlocks this). Until then, the simulator +
synthetic study carries it.

### Strategic note
Keep quiet (per startup-plan memory). Publish the *method* via rocket/CFRP;
apply it commercially via semiconductor. Same Stage 0--5 figure serves the
research proposal and the business plan.

---

## Sequencing
1. **Now → 2027**: finish Composites B (method paper); credits; Masterarbeit.
2. **Keio M1 (2027)**: Concept A (fleet learning) — buildable on synthetic
   fleet data with existing TMCMC machinery; lowest external-data dependence.
3. **Keio M2 / PhD (2028+)**: Concept B (differentiable inversion) — the
   high-IF thesis spine; needs the JAX-differentiable PF first.
4. **Parallel / shadow**: Concept C accrues via DISCO; surfaces post-graduation.

## Cross-cutting risks
- All three need data that does not yet exist (fleet histories, real wafer NDT,
  measured full-field). Mitigate with simulator-generated studies first; the
  Stage-3 model is the data generator.
- Scope discipline: each is a thesis on its own. Do not start B before A ships.
