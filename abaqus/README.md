# Abaqus skeletons for the CFRTP (Daikin/NEDO) residual-stress & delamination work

Production-grade counterparts of the self-contained Python weak-form-FE seeds in this
repo. The Python demos (`cfrp_cure_residual_stress_fe.py` ⑳, `cfrtp_residual_stress_fe.py`,
`cfrtp_cohesive_mixedmode.py`, `cfrtp_delamination_2d_fe.py`, …) are transparent,
license-free prototypes for understanding the physics and building the ML / surrogate /
inverse-design pipeline. For a real project the physics moves to a commercial solver —
**Abaqus** (composites: cohesive elements, UMAT/USDFLD, B-K, VCCT/XFEM) or **Ansys**
(ACP + Mechanical CZM). These files are the Abaqus starting points.

> ⚠️ **COMPLETE decks, but NOT VERIFIED IN ABAQUS.** The `.inp` files now carry real,
> generated meshes and consistent node/element sets (validated: no dangling node or set
> references), so they are meant to submit directly. But Abaqus is not available in this
> environment (no license), so **nothing was run or compiled here** — check the card
> syntax / UMAT argument list against your Abaqus version, and run the 1-element
> free-contraction sanity test (below) before trusting results.

## Files

| file | purpose | Python counterpart |
|---|---|---|
| `cfrtp_cure_umat.f` | UMAT: orthotropic **CHILE** (cure-hardening) + cure kinetics (STATEV α) + cure/crystallization shrinkage + thermal eigenstrain, incremental residual-stress build-up | `cfrp_cure_residual_stress_fe.py` (⑳), `cfrtp_residual_stress_fe.py` |
| `cfrtp_cure_residual.inp` | **complete** 3D [0/90] laminate (105 nodes, 48 C3D8) residual stress driven by a cure cycle, using the UMAT | same |
| `cfrtp_delamination_mixedmode.inp` | **complete** 2D plane-strain bilayer (366 nodes, 240 CPE4 + 45 COH2D4) with **built-in cohesive** + **Benzeggagh-Kenane** mixed-mode, pre-crack + mixed-mode loading | `cfrtp_cohesive_mixedmode.py`, `cfrtp_delamination_2d_fe.py` |
| `gen_inp.py` | regenerates both decks (change mesh size / geometry / layup / mixity here); pure-Python, no Abaqus/numpy needed | — |

## Physics mapping (Python → Abaqus)

- **Cure / CFRTP residual stress** — the Python incremental CHILE update
  `sigma += g(α) C (dε − dε_eig)` with `dε_eig = α_CTE dT + β dα` becomes the UMAT above
  (`DDSDDE = g*C`, incremental `STRESS` update, α in `STATEV(1)`). The cure cycle T(t) is
  supplied as a predefined `*TEMPERATURE` field with an `*AMPLITUDE`. Validate with a
  1-element free-contraction test → ~0 residual stress (the composite free-expansion check
  the Python demos also use).
  - Thermoplastic CFRTP (Daikin): reinterpret α as **solidification/crystallinity**, β as
    **crystallization shrinkage**, and drive with a **melt → cool** cycle. Add viscoelastic
    stress relaxation (→ measurement-grade magnitudes, cf. `cfrtp_viscoelastic_residual_stress.py`)
    via a UMAT with a Prony series / `*VISCOELASTIC` + WLF shift, or a state-variable
    relaxation in the UMAT.
- **Mixed-mode delamination** — Abaqus has this built in, so it is mostly an `.inp`:
  `*COHESIVE SECTION` + `*DAMAGE INITIATION, CRITERION=QUADS` + `*DAMAGE EVOLUTION,
  TYPE=ENERGY, MIXED MODE BEHAVIOR=BK, POWER=η`. This reproduces the Camanho-Davila
  bilinear + Benzeggagh-Kenane law of `cfrtp_cohesive_mixedmode.py` and the propagating
  front of `cfrtp_delamination_2d_fe.py`. `*DAMAGE STABILIZATION` (small viscosity) helps
  convergence through softening (the Python demo uses a secant scheme + small steps for
  the same reason). VCCT (`*DEBOND`, `*FRACTURE CRITERION, TYPE=VCCT`) is the alternative.
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

**First check — UMAT free-contraction sanity test.** Before trusting the [0/90]
result, run one C3D8 element with the UMAT, the 3-2-1 constraint (statically
determinate, free to contract), and the cure cycle. A single uniform ply that can
contract freely must end at ~0 residual stress — the composite free-expansion check the
Python demos also use. If it is not ~0, the eigenstrain sign / props / ordering is off.

**Resizing / editing.** `python3 gen_inp.py` regenerates both decks; change mesh
density, geometry, layup, pre-crack length `a0`, or the loading angle `theta_deg` at the
top of `gen_cure` / `gen_delam`.

### Ansys equivalents (pointers)
- Cure/CHILE residual stress: `USERMAT`/`USERMATTH` (or the Ansys Composite Cure Simulation
  ACT extension), thermal + field-driven eigenstrain.
- Mixed-mode delamination: **CZM** (`TB,CZM` with bilinear/exponential, B-K mixed-mode) on
  `INTER`/contact elements, or VCCT (`CINT`).
