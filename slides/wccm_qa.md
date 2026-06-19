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

## F. Stage-0 detection / Mahalanobis screening (new — narrated in the talk)

**Q: If a per-node z-score detects defects at AUROC ≈ 1.0 with no learning, why do you need the GNN at all?**
A: Detection and localization are different jobs. The screen answers "is there a defect, and roughly
where" — a binary/coarse question that the shared mesh geometry makes trivial. The GNN does the hard
part: the 19-class *region × layer* semantic identification (which ply, which in-plane region) under
noise. The screen can't tell you the insertion layer; that's the whole point of the talk.

**Q: You claim the per-node statistic beats a diffusion model and every learned anomaly detector. On what, exactly?**
A: Node-level AUROC on clean fields: Mahalanobis/PCA reach 0.999/0.995 (2×2 / 4×4). The reason is
structural, not a tuning win — every specimen shares one FEM mesh, so healthy fields are a tight point
cloud around a single template and a simple Gaussian distance is near-optimal. A learned detector has
nothing extra to exploit on clean data; it only helps if the healthy manifold is complex, which here it
isn't. So I frame it as "the cheap baseline is the right tool for detection," not "deep learning failed."

**Q: You say the σ=0.1 collapse is "wrong" — that contradicts the literature. What's your evidence?**
A: On a *labelled coupon* the detector still scores ≈0.84 AUROC at noise = 0.1× the field std, and only
falls to chance near 0.7–1.0× the field std. The often-quoted collapse assumes noise comparable to the
*defect signal*, not to the field; on our scale 0.1 is mild. And lock-in IR thermography averages noise
down as 1/√K frames, so practice has even more margin. I'm careful to call this a coupon-level result,
not a universal claim.

**Q: How big are the z-scores / how many sigma is a defect?**
A: I deliberately don't quote N-sigma. On a z-scored field the per-node σ is tiny, which inflates the
absolute magnitudes, so the number would be misleading. I only quote **rank** (defect nodes in the top
~0.2%, per-case AUROC ≈0.998 over 200 specimens) and **sign** (98–100% of defect deviations are
negative — a local stress *valley*, the thermoelastic signature of a sub-surface defect). There's a
backup appendix slide on exactly this.

**Q: Then why difference-normalize the input before the GNN, if the raw z-score already detects?**
A: Top-K precision of the raw screen is <1 because the hole-edge stress concentration also deviates
strongly. Differencing removes that static hole field so the GNN sees the residual defect signal — good
enough for go/no-go screening, not for precise per-layer localization.

---

### One-liners to keep ready
- "Detection is free and needs no learning; the GNN earns its keep on the 19-class layer-by-region localization."
- "The screen beats a diffusion model on clean data because one shared mesh makes healthy fields a single tight template — the simple baseline is the right tool."
- "I quote rank and sign, never N-sigma — z-scored fields inflate the magnitudes."
- "σ=0.1 collapse assumes noise on the defect-signal scale; on the field scale we hold ~0.84 AUROC, and lock-in IR averages noise as 1/√K."
- "Operationally, the false-negative rate (<3%) matters more than the 61% macro-F1."
- "Physics enters through the FEA data and the thermoelastic DSPSS feature; structure-preserving GNNs are our future-work bridge to the methods in this session."
- "Difference normalization removes the static hole field; the residual defect signal is what the GNN learns."
- "The noise + NDF training is our sim-to-real down-payment."
