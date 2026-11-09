# GUIDE section 20 on fifa: cause of the stage 3 divergence (steps 1 and 2), 9 Nov 2026

The stage 3 strip case (`../stage3_strip/FIFA_STAGE3.md`) diverged in its first time steps. Section 20 asks to compare
its boundary conditions with the tutorial `dogbone_fiber` (which runs, stage 2) and to change one thing at a time.
Material: the stage 3 illustrative values (the example's HDPE with the temperature constants shifted by +160 K, not
PPS), unchanged; no constant was changed.

## Step 1: boundary conditions, tutorial against strip

| field | inlet | outlet | difference |
| --- | --- | --- | --- |
| U | `flowRateInletVelocity` | `zeroGradient` | only the flow rate (2e-5 against 4.928e-7 m^3/s) |
| p | `calculated` 1e5 | `calculated` 1e5 | none |
| p_rgh | `zeroGradient` | `fixedValue` 1e5 | none |
| alpha.poly | `fixedValue` 1 | `zeroGradient` | none |
| T | `fixedValue` | `externalWallHeatFluxTemperature`, h = 1 | only the temperatures (melt 593.15 / 493.15 K; the strip's outlet Ta is still the example's 323.15 K, almost adiabatic with h = 1) |

The initial fields are of the same kind (U = 0, alpha = 0, p = 1e5, T uniform at the melt temperature). `fvSchemes`
and `fvSolution` are identical; `controlDict` differs only in endTime and writeInterval (maxCo 0.2, maxAlphaCo 0.05,
initial deltaT 1e-6 in both).

Openings, from the meshes (`cases/patch_geometry.py`): the tutorial's inlet and outlet are **also whole end faces**
(4 mm x 20 mm, 8.0e-5 m^2 each, 1847 / 1844 faces) that meet the walls at the edges. So candidates 1 (outlet velocity)
and 2 (gate meeting the walls) are not differences from the tutorial. What differs is the mesh:

| | tutorial | strip |
| --- | --- | --- |
| mesh | snappyHexMesh, near-cubic cells (face size about 0.21 mm) | blockMesh 320 x 4 x 32, cells 0.25 x 0.5 x 0.096 mm (x:z = 2.6, y:z = 5.2) |
| mean inlet velocity | 0.25 m/s | 0.08 m/s |

## Step 2: one change at a time (serial, 150 s each, `cases/strip_E.pbs`)

Criterion of section 20: at least 10 steps and max(U) within some tens of the mean inlet velocity 0.08 m/s.

| run | change from the strip | steps | max(U) [m/s] | result |
| --- | --- | --- | --- | --- |
| (a) | outlet velocity as the tutorial | - | - | already the tutorial's condition: no change, not run |
| (b) E_b | gate on the middle 2/3 of the thickness only (22 of 32 cells; the rest of the x = 0 face wall) | 2 | 337 | diverges (FPE, exit 136) |
| (c) E_c | vent on the middle 2/3 only | 3 | 6424 | diverges |
| (d) E_d | (b) and (c) | 2 | 337 | diverges |
| M1 | x cells 320 -> 833 (dx = 0.096 mm, cubic in x-z), 106 624 cells | 4 (stopped at 150 s) | 0.107 | does not diverge |
| M2 | thickness cells 32 -> 16 (dz = 0.19 mm, x:z = 1.3), 20 480 cells | 19 (stopped at 150 s) | 0.122 | does not diverge |

(b)-(d) split the thickness into three blocks of 5 / 22 / 5 cells (dz 0.103 / 0.093 mm). The two runs that do not
diverge both lower the cell ratio x:z (flow direction to thickness) from 2.6 to 1-1.3; M1 keeps y:z = 5.2, so it is the
x:z ratio that matters. M2 does not meet "30 or more elements through the thickness" of section 17; M1 does
(32), so M1 is the mesh for step 3.

## Step 3: time estimate (filling run not started)

M1 on 8 ranks for a fixed 30 min (`cases/strip_M1p.pbs`, `cases/Atrial`): 203 steps, no divergence, T = 1.92e-2 s at the
end (liquid fraction 0.0192, consistent with the fill time 1 s), max(U) 0.14 m/s. The step grew to 1.23e-4 s and is held
there by maxCo 0.2 (max Courant 0.200), so the rate is steady: 1.24e-5 s of filling per wall second over the last 9 min.
**Estimate for the filling run to T = 0.98 s: about 21.5 h on 8 ranks**, above the 12 h of section 20, so it was not
started (`cases/M1_trial_estimate.txt`). (The trial job did not exit after its 30 min `timeout`: the solver's log
stopped at 30 min but the job stayed running and was deleted at 45 min.)

Options, not done: (i) run it anyway (about 1 day, 8 cores of a compute node); (ii) more ranks (the strip has
833 x 4 x 32 cells, so 16 ranks along x is still 52 cells per rank); (iii) a shorter strip or a coarser x (x:z between 1
and 1.3 was stable, M2 used dx 0.25 with dz 0.19), which changes the mesh again.

Files: `cases/<run>/blockMeshDict`, `trial_tail.txt` (last log lines), `checkMesh_summary.txt`;
`cases/make_blockmesh.py` writes the variants. The OpenFOAM case directories are on fifa (`~/flow_sim/runs/strip_*`).

## Step 3 again, in 2-D (the user's request, 9 Nov): M1 with one cell across (sides empty), 833 x 1 x 32

Same x and z cells as M1 (0.096 mm, cubic in x-z), same material, constants and inlet flow rate (mean 0.08 m/s).
30 min trial on 8 ranks: 676 steps, no divergence, max(U) 0.27 m/s (3.4 times the mean inlet velocity), T = 0.0765 s
(liquid fraction 0.076), deltaT 1.20e-4 s held by maxCo 0.2; rate 4.19e-5 s per wall s over the last 10 min, so
**about 6.0 h for the filling run** (12 h limit of section 20): started (`cases/2D/fill_2D.pbs`, with `-l mem=16gb` and
a memory guard `cases/2D/memguard.sh`).

Filling run: exit 136, 25160 s wall, 6603 steps, last T = 0.7677128885 s, liquid fraction -0.41180897, max(U) over the run
8.65e+03 m/s. Last log lines: `cases/2D/fill_log_tail.txt`.

## Step 4: probes 50 mm from the gate (2-D filling run)

`orientation_50mm_2D.csv` (written by `step4_probes.py` from the probes of `cases/2D/controlDict0`): per depth from
the top face (32 cell centres), a_xx, a_yy, a_zz at the end of filling (x flow, y width, z thickness), the time the
melt front passed (first alpha.poly >= 0.5) and the time the layer froze (first T < TnoFlow = 543.15 K after the front
passed; empty if it did not freeze before the end of filling). end of the probe record T = 0.76769 s; front passed at 32/32 probes; frozen at 2/32; TnoFlow 543.15 K
No comparison or fitting done here; constants unchanged.
