# Abaqus skeletons for the CFRTP (Daikin/NEDO) residual-stress & delamination work

Production-grade counterparts of the self-contained Python weak-form-FE seeds in this
repo. The Python demos (`cfrp_cure_residual_stress_fe.py` ⑳, `cfrtp_residual_stress_fe.py`,
`cfrtp_cohesive_mixedmode.py`, `cfrtp_delamination_2d_fe.py`, …) are transparent,
license-free prototypes for understanding the physics and building the ML / surrogate /
inverse-design pipeline. For a real project the physics moves to a commercial solver —
**Abaqus** (composites: cohesive elements, UMAT/USDFLD, B-K, VCCT/XFEM) or **Ansys**
(ACP + Mechanical CZM). These files are the Abaqus starting points.

> ⚠️ **SKELETONS — NOT VERIFIED.** Abaqus is not available in this environment (no
> license), so none of these were run. They are structured, commented templates with
> placeholder meshes/props to adapt in Abaqus/CAE, then validate. Treat every number and
> the exact card syntax as a starting point to check against the Abaqus Keywords/User
> Subroutines manuals for your version.

## Files

| file | purpose | Python counterpart |
|---|---|---|
| `cfrtp_cure_umat.f` | UMAT: orthotropic **CHILE** (cure-hardening) + cure kinetics (STATEV α) + cure/crystallization shrinkage + thermal eigenstrain, incremental residual-stress build-up | `cfrp_cure_residual_stress_fe.py` (⑳), `cfrtp_residual_stress_fe.py` |
| `cfrtp_cure_residual.inp` | [0/90] laminate residual stress driven by a cure cycle, using the UMAT | same |
| `cfrtp_delamination_mixedmode.inp` | bilayer + **built-in cohesive** elements with **Benzeggagh-Kenane** mixed-mode delamination | `cfrtp_cohesive_mixedmode.py`, `cfrtp_delamination_2d_fe.py` |

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

### Ansys equivalents (pointers)
- Cure/CHILE residual stress: `USERMAT`/`USERMATTH` (or the Ansys Composite Cure Simulation
  ACT extension), thermal + field-driven eigenstrain.
- Mixed-mode delamination: **CZM** (`TB,CZM` with bilinear/exponential, B-K mixed-mode) on
  `INTER`/contact elements, or VCCT (`CINT`).
