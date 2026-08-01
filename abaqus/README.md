# Abaqus skeletons for the CFRTP (Daikin/NEDO) residual-stress & delamination work

Production-grade counterparts of the self-contained Python weak-form-FE seeds in this
repo. The Python demos (`cfrp_cure_residual_stress_fe.py` ⑳, `cfrtp_residual_stress_fe.py`,
`cfrtp_cohesive_mixedmode.py`, `cfrtp_delamination_2d_fe.py`, …) are transparent,
license-free prototypes for understanding the physics and building the ML / surrogate /
inverse-design pipeline. For a real project the physics moves to a commercial solver —
**Abaqus** (composites: cohesive elements, UMAT/USDFLD, B-K, VCCT/XFEM) or **Ansys**
(ACP + Mechanical CZM). These files are the Abaqus starting points.

> ✅ **VERIFIED IN ABAQUS 2024** (2026-07-28, on a licensed box). All three decks run
> and converge cleanly (`THE ANALYSIS HAS COMPLETED SUCCESSFULLY`, no severe
> discontinuity iterations, no cutbacks). Summary below ("Verified results"); see that
> section for one known cross-check gap (a `BETA` cure-shrinkage property mismatch
> against the Python seed) that is flagged but not yet reconciled.
>
> ✅ **UPDATE (2026-08-01)**: all seven decks (`sanity`, `cure`, `ve`, `crystve`,
> `crystpeek`, `delam`, `delam3d`) verified. Fixed two real bugs found along the way:
> (1) `gen_inp.py`'s crystallization-coupled decks under-specified `*INITIAL
> CONDITIONS, TYPE=SOLUTION` (1 value against `*DEPVAR, 23`), which Abaqus 2024
> rejects outright; (2) `postprocess.py`'s delamination-front element count summed
> raw `SDEG` field values instead of distinct elements, over-counting damaged
> elements 2×/4× for COH2D4/COH3D8. See "Verified results (2026-08-01)" below.

## Files

| file | purpose | Python counterpart |
|---|---|---|
| `cfrtp_cure_umat.f` | UMAT: orthotropic **CHILE** (cure-hardening) + cure kinetics (STATEV α) + cure/crystallization shrinkage + thermal eigenstrain, incremental residual-stress build-up | `cfrp_cure_residual_stress_fe.py` (⑳), `cfrtp_residual_stress_fe.py` |
| `cfrtp_cure_residual.inp` | **complete** 3D [0/90] laminate (105 nodes, 48 C3D8) residual stress driven by a cure cycle, using the UMAT | same |
| `cfrtp_delamination_mixedmode.inp` | **complete** 2D plane-strain bilayer (366 nodes, 240 CPE4 + 45 COH2D4) with **built-in cohesive** + **Benzeggagh-Kenane** mixed-mode, pre-crack + mixed-mode loading | `cfrtp_cohesive_mixedmode.py`, `cfrtp_delamination_2d_fe.py` |
| `cfrtp_1elem_sanity.inp` | **complete** single-C3D8 UMAT free-contraction sanity test (3-2-1 support, same cure cycle) | the `[validate]` line each Python seed prints |
| `cfrtp_cure_umat_ve.f` | UMAT: CHILE **+ thermo-viscoelastic** (generalized-Maxwell / Prony + WLF shift). Hot → fast relaxation, cold → frozen residual stress. 30 constants, 22 STATEV | `cfrtp_viscoelastic_residual_stress.py` |
| `cfrtp_cure_residual_ve.inp` | **complete** 3D [0/90] (105 nodes, 48 C3D8) driven by the **viscoelastic** UMAT | same |
| `cfrtp_cryst_umat_ve.f` | UMAT: viscoelastic **+ non-isothermal crystallization** (Nakamura). α = relative crystallinity develops on melt→cool and drives stiffness, shrinkage and the relaxation shift `a_X(α)`. 30 constants, 23 STATEV | `cfrtp_residual_stress_fe.py` (cooling-rate → crystallinity → residual) |
| `cfrtp_cryst_residual_ve.inp` | **complete** 3D [0/90] (105 nodes, 48 C3D8) driven by the **crystallization-coupled VE** UMAT on a melt→cool cycle | same |
| `cfrtp_cryst_peek_validation.inp` | **complete** carbon/**PEEK** validation case (105 nodes, 48 C3D8) — literature-typical AS4/PEEK values so the method can be checked against **public** crystallization + residual-stress data (fluoropolymer-CF data are proprietary) | same |
| `cfrtp_delamination_3d.inp` | **complete** 3D bilayer (656 nodes, 240 C3D8 + 90 COH3D8), built-in cohesive + B-K; a curved delamination **front** develops across the width | `cfrtp_delamination_2d_fe.py` (3D extension) |
| `gen_inp.py` | regenerates **all** decks (mesh size / geometry / layup / mixity / Prony); pure-Python, no Abaqus/numpy needed | — |

## Physics mapping (Python → Abaqus)

- **Cure / CFRTP residual stress** — the Python incremental CHILE update
  `sigma += g(α) C (dε − dε_eig)` with `dε_eig = α_CTE dT + β dα` becomes the UMAT above
  (`DDSDDE = g*C`, incremental `STRESS` update, α in `STATEV(1)`). The cure cycle T(t) is
  supplied as a predefined `*TEMPERATURE` field with an `*AMPLITUDE`. Validate with a
  1-element free-contraction test → ~0 residual stress (the composite free-expansion check
  the Python demos also use).
  - Thermoplastic CFRTP (Daikin): reinterpret α as **solidification/crystallinity**, β as
    **crystallization shrinkage**, and drive with a **melt → cool** cycle. For
    measurement-grade magnitudes add viscoelastic stress relaxation — implemented in
    **`cfrtp_cure_umat_ve.f`** (`cfrtp_cure_residual_ve.inp`): a generalized-Maxwell / Prony
    series with a WLF temperature shift `a_T(T)` (hot → fast relaxation, cold → frozen), the
    Abaqus counterpart of `cfrtp_viscoelastic_residual_stress.py`. Prony `τ_k` are in the
    deck's time unit — co-calibrate them with the cure-cycle duration (and re-tune the
    kinetics `AK`) before trusting magnitudes.
  - **Semi-crystalline upgrade** — for a fluoropolymer-matrix (semi-crystalline)
    CFRTP the most defensible model couples the viscoelasticity with
    **crystallization kinetics**: **`cfrtp_cryst_umat_ve.f`** (`cfrtp_cryst_residual_ve.inp`)
    makes α the **relative crystallinity** from a Nakamura non-isothermal law
    (`dα/dt = n K(T)(1−α)[−ln(1−α)]^((n−1)/n)`, bell-shaped `K(T)` window),
    developed on a **melt→cool** cycle, driving stiffness `g(α)`, crystallization
    shrinkage `β·dα` and a crystallinity relaxation shift `a_X = 10^{BX·α}` (crystals
    freeze relaxation as α→1). This generates the cooling-rate → crystallinity →
    residual-stress pathway of `cfrtp_residual_stress_fe.py`. The Nakamura integrator
    is checked against the closed-form isothermal Avrami `α = 1 − exp(−(K t)^n)`.
  - **Validation on public data** — the Daikin fluoropolymer-CF system is proprietary,
    but the *method* can be validated on **carbon/PEEK**, for which crystallization
    kinetics and residual-stress data are public. `cfrtp_cryst_peek_validation.inp`
    uses literature-typical AS4/PEEK values (Avrami `n≈2.5`; bell `K(T)` centred in the
    PEEK crystallization window, `Tg≈143`/`Tm≈343 °C`; universal WLF `C1=17.4,
    C2=51.6` at `Tref=Tg`; `KMAX` calibrated by simulation so α→1 over the cool-down,
    dt-converged). These are literature-typical values to confirm against the sources,
    **not** extracted from one table: Parlevliet et al., *Composites Part A* (2006–07,
    residual stresses in TP composites, 3-part review); Tierney & Gillespie,
    *Composites Part A* (2004, PEEK non-isothermal kinetics); MDPI *Polymers* (2025,
    transient PEEK crystallization). Swap in the Daikin system once data are available.
- **Mixed-mode delamination** — Abaqus has this built in, so it is mostly an `.inp`:
  `*COHESIVE SECTION` + `*DAMAGE INITIATION, CRITERION=QUADS` + `*DAMAGE EVOLUTION,
  TYPE=ENERGY, MIXED MODE BEHAVIOR=BK, POWER=η`. This reproduces the Camanho-Davila
  bilinear + Benzeggagh-Kenane law of `cfrtp_cohesive_mixedmode.py` and the propagating
  front of `cfrtp_delamination_2d_fe.py`. A small viscosity on `*SECTION CONTROLS`
  (`CONTROLS=` on the `*COHESIVE SECTION`) helps convergence through softening — note
  `*DAMAGE STABILIZATION` on the material itself is silently ignored for
  `TRACTION SEPARATION` cohesive elements in Abaqus 2024 (see "Verified results" below);
  the Python demo uses a secant scheme + small steps for the same stabilization purpose.
  VCCT (`*DEBOND`, `*FRACTURE CRITERION, TYPE=VCCT`) is the alternative.
  **`cfrtp_delamination_3d.inp`** is the 3D extension (COH3D8 between two C3D8
  sublaminates), where a curved delamination **front** can develop across the width — the
  next step toward the 3D mixed-mode front noted in the roadmap.
- **Impregnation / voids (開繊)** — Darcy/Gebart flow (`cfrtp_impregnation_void.py`) is not
  a structural-FE job; use a resin-flow tool (Moldflow, PAM-RTM) or an Abaqus/Ansys porous-
  flow model. No skeleton here — noted for completeness.

## ML / surrogate / inverse design

The surrogate (`electrothermal_operator.py`, `cfrtp_process_surrogate.py`), calibration
(`cfrtp_calibration.py`) and inverse design (`electrothermal_inverse_design.py`) wrap
around **either** solver: run Abaqus/Ansys to generate the dataset (via the Python driver
calling the solver in batch), then train and optimize exactly as in the Python demos, with
the commercial FE as the accuracy authority.

## Run notes (Abaqus)

```
abaqus job=cfrtp_cure_residual  user=cfrtp_cure_umat.f  interactive
abaqus job=cfrtp_delamination_mixedmode  interactive
```
Requires a compatible Fortran compiler linked to Abaqus for the UMAT. Check
`ABA_PARAM.INC`, the UMAT argument list, and stress/strain component ordering
(3D: 11,22,33,12,13,23) for your Abaqus version before trusting results.

**Units** are SI (Pa, m, s, K). Temperature is supplied in °C; the UMAT adds
`TABS=273.15` internally for the Arrhenius kinetics — keep that consistent if you edit.

**First check — UMAT free-contraction sanity test.** `cfrtp_1elem_sanity.inp`
(one C3D8 element, the UMAT, 3-2-1 constraint, same cure cycle): a single uniform ply
that can contract freely must end at ~0 residual stress — the composite free-expansion
check the Python demos also use. If it is not ~0, the eigenstrain sign / props /
ordering is off. **Verified 2026-07-28**: max |σ| over every increment/component is
~1 Pa against E1=135 GPa (relative ~1e-11, i.e. solver round-off) — confirms the
eigenstrain sign and BC set are correct.

**Resizing / editing.** `python3 gen_inp.py` regenerates all three decks; change mesh
density, geometry, layup, pre-crack length `a0`, or the loading angle `theta_deg` at the
top of `gen_cure` / `gen_delam`.

## One-shot run + summary (on your own server)

Everything runs on **your** Abaqus box over **your** SSH session — this sandbox cannot
reach your server (outbound is HTTPS-proxy-only, and it should not hold your credentials).
Get the repo onto the server (`git clone …` or `rsync -av abaqus/ user@host:~/cfrtp/`),
then:

```
cd abaqus
bash run_all.sh              # both jobs + summary   (ABAQUS=abq2023 bash run_all.sh to pick a version)
bash run_all.sh cure        # cure job only
bash run_all.sh delam       # delamination job only
```

- `run_all.sh` submits `cfrtp_cure_residual` (with the UMAT) and
  `cfrtp_delamination_mixedmode`, then runs `postprocess.py`.
- `postprocess.py` (Abaqus Python / odbAccess) prints metrics to compare with the Python
  seeds: cure → residual σ₁₁ range + warpage + degree-of-cure; delam → peak tip reaction
  + delamination front. Untested here (no license) — if a field/step/instance name
  differs in your Abaqus version, adjust the small helpers at the top of the file.

Send me the printed summary (or the `.dat`/`.msg` tail on a non-convergence) and I'll
tune the UMAT constants, cohesive parameters, or increment controls.

## Verified results (2026-07-28, Abaqus 2024)

**1-element sanity** (`cfrtp_1elem_sanity.inp`) — converged; max |σ| ~1 Pa vs
E1=135 GPa (solver round-off). Eigenstrain sign / BC set confirmed correct.

**Cure `[0/90]` laminate** (`cfrtp_cure_residual.inp`) — converged (53 increments, no
cutbacks). `postprocess.py` output:
```
residual sigma_11 range: [-71.7, 13.1] MPa
warpage max|U3|: 0.053 mm
degree of cure alpha: [0.988, 0.988]
```
A 2-ply `[0/90]` is *unsymmetric* in classical laminate theory (bending-extension
coupling), so non-zero warpage is expected, not a defect. 0.053 mm over a 20×12 mm,
1.2 mm-thick plate (~0.26% of span) is modest for the CTE/cure-shrinkage mismatch size.

**Mixed-mode delamination** (`cfrtp_delamination_mixedmode.inp`) — converged (55
increments, no cutbacks, equilibrium iterations peak at 9-10 through the softening
region). **Found and fixed a real bug during this run**: Abaqus 2024 silently ignores
`*DAMAGE STABILIZATION` on cohesive-material `TRACTION SEPARATION` response
(`***WARNING: ... VISCOSITY SHOULD BE DEFINED BY USING *SECTION CONTROLS`) — the
intended viscous regularization was not actually active. Fixed in `gen_delam()` by
moving it to `*SECTION CONTROLS, VISCOSITY=1.0e-5` + `CONTROLS=` on the
`*COHESIVE SECTION`; re-ran and confirmed identical results (the warning-only silent
failure wasn't perturbing the physics, just wasn't wired up):
```
peak reaction |RF| at TIP: 119631.0 (frame 55)
delaminated cohesive elements (SDEG>0.5): 2 / 45
delamination front x: 17.0 mm   (from pre-crack a0=15 mm, i.e. Δa=2 mm)
```
(the element count was originally misreported as 4 — `postprocess.py` was counting
raw `SDEG` field values rather than distinct elements; COH2D4 reports 2 integration
points per element. Fixed 2026-08-01, see below.)

**Cohesive mesh resolution** — the Camanho-Davila/Turon estimate
`lcz ≈ E·Gc/σmax²` (M=1) with this deck's props (E=60 GPa, GIc/GIIc=200/600 N/m,
σmax=12/18 MPa) gives `lcz ≈ 83-111 mm` — **83-111 elements** could fit across it at
the current 1 mm element size, i.e. mesh resolution is not the constraint (Turon
recommends ≥3). But `lcz` exceeds even the bonded ligament (45 mm), so the process zone
can never fully form within this specimen: expect diffuse softening spread across most
of the ligament rather than a sharp, steadily-advancing crack tip. That is the likely
reason only 2 mm of front advance showed up at `SDEG>0.5` despite the load already
being past the softening onset — not a meshing or convergence problem.

**Cross-check against the Python seeds** — same directory tree, run with `python3
<script>.py`:
- `cfrp_cure_residual_stress_fe.py` (direct counterpart of `cfrtp_cure_residual.inp`):
  single-ply free contraction ~0 (matches); `[0/90]` σ_xx range **[-138.7, 77.0] MPa**
  vs Abaqus's **[-71.7, 13.1] MPa** — same sign pattern, same order of magnitude, but
  ~2-2.5× wider in the Python seed. Root causes found: (1) **`BETA` (cure-shrinkage
  coefficient) mismatch** — Python uses `BETA_T=-4e-3`, the Abaqus UMAT `*USER
  MATERIAL` uses `BETA=-3.0e-3` (a ~33% property difference, not yet reconciled —
  pick one and edit the other); (2) ply thickness differs 2× (Python: 0.3 mm/ply,
  0.6 mm laminate; Abaqus: 0.6 mm/ply, 1.2 mm laminate); (3) 2D plane-stress CST vs
  full 3D C3D8 + different mesh density. `cfrtp_residual_stress_fe.py` is a
  *different* cycle (fluoropolymer melt→RT at 340 °C) and is not the matching
  counterpart for this cure deck.
- `cfrtp_cohesive_mixedmode.py` (cohesive-law point check, no BVP): properties match
  the deck exactly (GIc/GIIc=200/600 N/m, tn/ts=12/18 MPa, B-K η=1.6).
- `cfrtp_delamination_2d_fe.py` (direct counterpart of the mixedmode deck): **same**
  E, Gc, strengths, η as the Abaqus deck, and the **same** front metric definition
  (`damage > 0.5`, i.e. `SDEG>0.5`). Geometry is self-similar (precrack/domain = 0.25
  in both) but its domain is 3× smaller (20 mm vs 60 mm) with a comparable tip
  displacement (600 µm vs 500 µm); it shows the front sweeping most of its short
  domain (Δa 14.8 mm of a 15 mm ligament) where Abaqus's 3×-longer domain only shows
  the early stage (Δa 2 mm of 45 mm) of the same regime — consistent with the `lcz`
  finding above, not a discrepancy.

**Open item**: reconcile the `BETA` cure-shrinkage constant between the UMAT
(`gen_cure()` in `gen_inp.py`, currently `-3.0e-3`) and `cfrp_cure_residual_stress_fe.py`
(currently `-4e-3`) if exact quantitative parity with the Python seed is wanted.

## Verified results (2026-08-01, Abaqus 2024) — remaining four decks

All four decks not covered above (`ve`, `crystve`, `crystpeek`, `delam3d`) converged
(53 increments, no cutbacks for the cure-family jobs). **Two real bugs found and
fixed during this pass**:

1. `gen_inp.py`'s `gen_cryst_ve()` / `gen_cryst_peek()` declared `*DEPVAR, 23` but
   supplied only 1 value (`EALL, 1.0e-3`) in `*INITIAL CONDITIONS, TYPE=SOLUTION`.
   Abaqus 2024 rejects this outright (`***ERROR: THERE ARE INSUFFICIENT DATA CARDS
   TO DEFINE THE SOLUTION-DEPENDENT VARIABLES`) rather than zero-filling as older
   versions do. Fixed by padding the remaining 22 state variables with `0.0`.
2. `postprocess.py`'s `delam()` counted raw `SDEG` field *values*, not distinct
   elements — COH2D4/COH3D8 report 2/4 integration-point values per element, so the
   printed damaged-element count was 2×/4× too high (see correction above).

**`cfrtp_cure_residual_ve.inp` (viscoelastic CHILE cure)**:
```
[ve] residual sigma_11 range: [-44.5, 8.1] MPa   (vs elastic cure: [-71.7, 13.1] MPa)
     warpage max|U3|: 0.053 mm                     (same as elastic cure — geometry-driven)
     degree of cure alpha: [0.988, 0.988]
```
Peak magnitude relaxes from 71.7 → 44.5 MPa (~38%) relative to the elastic `cure`
job. The Python counterpart in the mapping table, `cfrtp_viscoelastic_residual_stress.py`,
runs a *different* base cycle (fluoropolymer melt→RT, 340→25 °C, crystallization
kinetics) rather than this deck's Arrhenius/CHILE cure kinetics (same family as the
elastic `cure`/`sanity` jobs) — so magnitudes aren't expected to match 1:1. At the
default 5 °C/s rate it reports elastic 500 → viscoelastic 131 MPa (74% relaxed).
Both show the same qualitative story (VE relaxation cuts peak residual stress by
roughly a third to three-quarters), but exact parity would need the same underlying
kinetics model and cure/melt cycle on both sides — not yet reconciled, same spirit as
the open `BETA` item above.

**`cfrtp_cryst_residual_ve.inp` / `cfrtp_cryst_peek_validation.inp` (crystallization + VE)**:
```
[crystve]   residual sigma_11 range: [-1.0, 0.2] MPa   relative crystallinity: 0.230
[crystpeek] residual sigma_11 range: [-52.0, 7.5] MPa  relative crystallinity: 1.000
```
`crystve` (illustrative fluoropolymer proxy, `TCRYST=120 °C`, melt cycle from 260 °C)
only reaches partial crystallization (α=0.23) over the cool-down, so stiffness
development `g(α)` stays low and locked-in stress is correspondingly tiny (~1 MPa) —
physically consistent, but worth a second look given this is the exact deck where the
`*DEPVAR` bug above was found; flagging as an open item rather than asserting it's
correct. `crystpeek` (AS4/PEEK, `TCRYST=290 °C`, melt 380→RT) reaches full
crystallization (α=1.0, as `KMAX` was calibrated to do) and the largest warpage of the
cure-family jobs (0.269 mm), with σ₁₁≈52 MPa. Cross-check: the *elastic*, no-relaxation
Python melt-crystallize model (`cfrtp_residual_stress_fe.py`, melt 340→RT @5 °C/s)
predicts crystallinity 0.363 and σ_xx range **[-499.7, 292.0] MPa** — an order of
magnitude above `crystpeek`'s VE-relaxed 52 MPa, which is expected since that Python
seed has no relaxation. Its viscoelastic counterpart
(`cfrtp_viscoelastic_residual_stress.py`) brings the elastic 500 MPa down to 131 MPa
(74% relaxed) at the same rate, and down to 88 MPa at a slow 0.5 °C/s cool — the same
order of magnitude as `crystpeek`'s 52 MPa. Not a strict validation (different
material system, parameter set, and crystallization temperature window), but the same
qualitative conclusion: crystallization + VE relaxation brings residual stress down
from a purely-elastic hundreds-of-MPa prediction to a measurement-grade tens-of-MPa
range.

**`cfrtp_delamination_3d.inp` (3D mixed-mode front, COH3D8 + B-K)**:
```
[delam3d] peak reaction |RF| at TIP: 7835.5 N (frame 142)
          delaminated cohesive elements (SDEG>0.5): 30 / 90
          delamination front x: 30.0 mm   (from pre-crack a0=15 mm, i.e. Δa=15 mm)
```
Same `Lx=60 mm`, `a0=15 mm` domain as the 2D `delam` deck above (θ=30° vs 25°),
extended across a 20 mm width with 90 COH3D8 elements. 30/90 (33%) of the cohesive
elements are damaged and the front advances 15 mm of the 45 mm ligament — a much
larger fraction than the 2D job's Δa=2 mm/45 mm (4%) despite the identical `Lx`/`a0`.
This is consistent with the true 3D crack front (curving across the width) relieving
the constraint that the plane-strain 2D idealization imposes, and matches the
direction (if not magnitude) of `cfrtp_delamination_2d_fe.py`'s own front sweep
(Δa 14.8 mm of a 15 mm ligament, 99% — see the `lcz` discussion above for why the
short-ligament Python/3D domains show more front advance than the longer 2D deck).

### Ansys equivalents (pointers)
- Cure/CHILE residual stress: `USERMAT`/`USERMATTH` (or the Ansys Composite Cure Simulation
  ACT extension), thermal + field-driven eigenstrain.
- Mixed-mode delamination: **CZM** (`TB,CZM` with bilinear/exponential, B-K mixed-mode) on
  `INTER`/contact elements, or VCCT (`CINT`).
