# GUIDE section 17, stage 3 on fifa: strip from the film gate (stopped, the user, 4 Nov 2026)

Stage 3 was stopped before any production run, as asked: the trial diverged in its first time steps, and the cause
was not searched further. This records the mesh and the diagnostic runs.

## Case (files in `case/`)
- Strip from the film gate: 80 mm (flow, x) x 2 mm (width, y) x 3.08 mm (thickness, z); blockMesh 320 x 4 x 32 =
  40 960 cells (0.25 x 0.5 x 0.096 mm); `checkMesh`: Mesh OK (`fifa_strip_checkMesh.txt`).
- Patches: `inlet` = the whole x = 0 face (film gate as thick as the plate, assumption), `outlet` = the whole x = 80 mm
  face (vent), `walls` = top and bottom, `sym0` / `sym1` = the two side faces as separate symmetryPlane patches
  (one patch over both faces is rejected by OpenFOAM as not planar; that was the first, failed submission).
- Melt 320 C (593.15 K, inlet and initial), mould 150 C (Ta = 423.15 K) with the example's wall coefficient
  h = 5e3 W/(m2 K); fill time 1 s, i.e. inlet flow rate 80 x 2 x 3.08 mm3 / 1 s = 4.928e-7 m3/s.
- Material (illustrative, NOT PPS; the user's choice): the example's HDPE viscosity and PVT, with the
  temperature constants shifted by +160 K so that the no-flow temperature is 270 C: D2 153.15 -> 313.15 K,
  TnoFlow 383.15 -> 543.15 K, Tait b5 390.65 -> 550.65 K. The cp and kappa tables are the example's, unshifted
  (cp keeps the HDPE latent-heat peak at about 92-107 C).
- Fibres: aspect ratio 16.8 and FiberVolfrac 0.182 (paper), so CI = 0.0184 exp(-0.7148 * 3.058) = 0.00207
  (Bay-Phan-Thien, as in the solver); FiberSuspension off and the hybrid closure as in the example.
- Solver settings (`system/`) are the example's; 8 MPI ranks on frontale03 (qsub), podman image on the node's disk.

## Result: diverges in the first steps
The velocity grows over the PIMPLE outer iterations (50 per step) from 0.1 m/s to several hundred m/s within 2-3
steps, and the run stops with a floating-point exception in exp() (exit 136). The mean inlet velocity is 0.08 m/s.
Last lines: `fifa_strip_trial_tail.txt`.

Short diagnostic copies (1-2 min each, not results):

| run | change from the case | steps | last time [s] | last max(U) [m/s] |
|---|---|---|---|---|
| D1 | the example's material constants (no shift) | 3 | 3.15e-6 | 902 |
| D2 | shifted constants, the example's temperatures (melt 220 C, mould 50 C) | 1 | 1.20e-6 | 48 |
| D3 | 2-D: one cell across, side faces empty | 3 | 3.00e-6 | 859 |
| D4 | side faces as no-slip walls | 3 | 2.95e-6 | 1073 |
| D5 | D3 run serially (no decomposition) | 3 | 3.00e-6 | 859 |

All five diverge in the same way, so the material shift, the temperatures, the symmetry planes and the parallel
decomposition are not the cause. What differs from the example (which runs, stage 2) is the mesh and the openings:
a blockMesh box with the whole end faces as gate and vent, against the example's snappyHexMesh mesh with small
gate and vent patches. Not investigated further (the user, 4 Nov).
