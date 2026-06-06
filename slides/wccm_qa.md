# WCCM 2026 — Anticipated Q&A (MS090E)

Session **MS090E "Machine Learning for Computational Mechanics Across Scales V"**, 22 Jul, slot 4/6.
Other talks: Cueto group (thermodynamics-preserving / GENERIC GNNs), Sharma & Fink (momentum-conserving
physics-informed GNNs), Liu & Yi (Fourier-embedded PINN × DIC). Position our talk as the only
**real-structure NDT pipeline + industrial application**.

---

## A. Method / model

**Q: Why a GNN (GAT) rather than a CNN on the stress image?**
A: The interstage panel is perforated and curved, so the surface is an irregular FEM mesh, not a
clean image grid. A CNN needs resampling and handles holes badly. A GNN operates directly on the mesh
graph; GAT additionally weights neighbours by attention, which suits the highly non-uniform stress
field around holes and defects.

**Q: Why 19 classes instead of regression of defect coordinates?**
A: We need both in-plane region and insertion layer; framing it as per-node classification (region ×
layer, class 0 = healthy) lets one model output the full 3-D location and naturally handles "no defect."

**Q: How do you handle the extreme class imbalance (<3% defect nodes)?**
A: Focal loss down-weights the abundant healthy nodes, plus — in the new work — 5000 defect-free
"NDF" negatives so the model calibrates against false positives. We report macro-F1, not accuracy,
precisely because accuracy is meaningless at a 0.998 healthy prior.

**Q: Is the edge set / mesh the same across specimens?**
A: Yes — fixed FEM connectivity reused across all graphs, so only node features (coordinates + DSPSS)
vary. This is what makes batching and the difference-normalization well-defined.

**Q: What exactly is DSPSS and why is it physically meaningful as a node feature?**
A: Sum of principal stresses on the surface, σ₁+σ₂+σ₃ = tr(σ). It is directly measurable by
infrared thermoelastic stress analysis (ΔT = −kTΔσ_sum), so the input is experimentally obtainable,
not just an FEA artefact.

## B. Session-specific (differentiation)

**Q (Cueto-style, the one to nail): "Your GNN has no physics / thermodynamic structure — why not a
structure-preserving or GENERIC-based GNN?"**

*What the question means.* The Cueto/Hernández line builds GNNs whose architecture **hard-codes physical
laws** — energy conservation and non-negative entropy production (the GENERIC formalism), or symmetries
(equivariance). The network *cannot* violate the law by construction, which helps generalization and
trust in **forward simulation** of a dynamical system. "No physics-conservation structure" means our GAT
is a generic function approximator: nothing in its layers forces it to obey a conservation law.

*Why that is fine for us (layered answer — give 1, add 2–3 if pressed).*
1. **Different problem class.** They *solve/advance* a physical state and must conserve energy/momentum
   along a trajectory. We do **static inverse inference**: given one measured stress field, classify
   which node is defective. There is no time evolution and no conserved quantity to preserve along — so a
   GENERIC bracket has nothing to act on here.
2. **Physics is already in the pipeline.** The input is not raw pixels: it is the **thermoelastic sum of
   principal stresses** (ΔT = −kTΔσ_sum), and the fields come from **FEA that already enforces
   equilibrium and the constitutive law**. The conservation physics lives in the data generation; the
   network only has to read the residual signature.
3. **The right inductive bias for *this* task is locality, not conservation.** The defect signature is a
   local stress-valley whose shape encodes the ply layer. Attention over mesh neighbours captures that
   directly. A conservation constraint would not improve *classification* of a discrete label.
4. **It is our explicit future work — and a bridge to your method.** Making the model physically
   consistent (structure-preserving / equivariant GNNs, Hernández & Cueto) is exactly the direction we
   cite for the sim-to-real and forward-field stages. So we see their work as complementary, not a gap.

*One-sentence version if time is short:* "We do static defect inference, not forward dynamics — there's
no conserved trajectory to preserve; the physics enters through the FEA data and the thermoelastic
feature, and structure-preserving GNNs are our planned next step."

*Do NOT say:* "physics doesn't matter" or "we didn't think about it." Frame it as a deliberate scoping
choice with a clear bridge to their methods.

**Q (Sharma/Fink-style): Could a physics-informed / conservation-based loss help?**
A: Possibly for the forward stress field, but our quantity of interest is a discrete defect label, not
a dynamic conserved field. A momentum/energy residual doesn't directly constrain "which layer." We see
PI losses as more relevant once we move to time-resolved or guided-wave SHM (the payload-fairing
direction).

**Q (Liu/Yi-style): You use difference normalization; have you considered Fourier features / spectral
methods, given the stress-concentration is high-frequency near the hole?**
A: Good point — the stress-concentration field is sharply varying. Difference normalization removes the
*static* hole field; for the residual high-frequency content, Fourier/spectral encoders are exactly one
of the architectural directions we are exploring (preliminary MeshGraphNet/Fourier-KAN variants). Their
Fourier-PINN × DIC line is also why we are interested in sim-to-real with experimental full-field data.

## C. Results / validation

**Q: 61% macro-F1 — is that good enough for a safety-critical application?**
A: It is a feasibility result on simulation, and the operationally important number is the defect
false-negative rate, which is under 3% — we rarely miss a defect. The remaining macro-F1 loss is mostly
*adjacent-layer* confusion, i.e. right region, neighbouring layer. For go/no-go inspection that is far
less critical than a missed defect.

**Q: Where do the errors come from?**
A: Two places: adjacent-layer confusion (the ply-angle signature between neighbouring layers is similar),
and false positives near the hole edge where stress gradients are steepest. The NDF negatives in the new
work specifically target the false-positive side.

**Q: How many specimens / how is the split done? Any leakage?**
A: Synthetic FEA dataset with many defect placements; we split by specimen and (in the extended
pipeline) group-purge so the same geometry/defect group never spans train and test.

## D. Sim-to-real / generalization

**Q: This is all FEA. Will it work on real infrared measurements?**
A: That is the central next step. The noise-robust training here — structured line+white noise plus the
NDF negatives — is explicitly designed to narrow the sim-to-real gap. We expect a domain shift and plan
calibration/fine-tuning on a small set of real TSA measurements.

**Q: Does it generalize to unseen defect sizes / hole geometries?**
A: Within the simulated range it does; out-of-distribution size generalization is a known limitation we
are actively working on (geometry-aware features and the architecture study).

## E. The extension (if pushed on the "outlook")

**Q: You mention MeshGraphNet variants — what do they give you?**
A: Preliminary and not part of the published results, so I'll be brief: a mesh-physics encode-process-
decode architecture that updates edge features along the mesh appears to improve over the attention-only
GAT under a balanced training recipe. We are validating it with our co-authors before claiming numbers.

---

### One-liners to keep ready
- "Operationally, the false-negative rate (<3%) matters more than the 61% macro-F1."
- "Physics enters through the FEA data and the thermoelastic DSPSS feature; structure-preserving GNNs are our future-work bridge to the methods in this session."
- "Difference normalization removes the static hole field; the residual defect signal is what the GNN learns."
- "The noise + NDF training is our sim-to-real down-payment."
