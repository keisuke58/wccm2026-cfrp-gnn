# GUIDE section 18 on fifa: contents of Greif & Meyer, Zenodo 17831586 (no computation)

Downloaded `data.zip` (2 031 797 362 bytes, 18 min) to `~/data/greif2025_warp/` on fifa and unpacked it (4.5 GB). The
data was only read. `greif_survey.csv` has one row per geometry (written by `fifa_survey.py`, 37 s for all 196).

## 1. Files
588 files: one folder per geometry (196 folders, 8-digit id), each with 3 files of the same stem
`<id>_<32-hex geometry hash>_step_<nnn>`:
- `..._sim.vtu`: mesh and results (2.7-27 MB);
- `....log`: the Moldflow 2023 log (`mhb3d -fill -pack`, then `warp3d`);
- `....yml`: `inlet_location` (3 coordinates, mm) and `inlet_normal` (unit vector).
Folders that share a hash are steps of the same base geometry (e.g. `..._75f31cb4..._step_000` to `step_017`).

## 2. One VTU (00010006)
- Elements: 4-node tetrahedra only (`tetra`), 63 426 elements, 11 926 nodes. All 196 files: tetra only,
  24 810-365 100 elements (median 118 100), 4 746-66 650 nodes.
- Coordinates in **mm**: bounding box 57.0 x 57.0 x 2.514 mm (the log's volume 7.976e-7 m3 = 797.6 mm3 fits mm).
- Size over all 196: bounding-box diagonal 17-354 mm (median 80), smallest box edge 1.75-118 mm (median 18).

## 3. Fields (the same names in the 5 files checked)
Point data:
- `Fill_time` [s]; `Freeze_time` [s].
- `Deflection,_all_effects[X_Component/Y_Component/Z_Component/Deflection]`: 3 components (x, y, z) in **metres**.
  The log reports the same values in mm (00010006: Trans-X min -1.4143e-01 mm; VTU -1.4143e-04).
  The coordinates are in mm, so the deflection must be multiplied by 1000 to be in the units of the coordinates.
- `Fiber_orientation_tensor_on_nodes_(3D)`: 6 components; components 0, 1, 2 are the diagonal (sum exactly 1.0,
  standard deviation 0). By the usual VTK/Moldflow order the rest are xy, yz, xz; not confirmed from a document.
  Closure in the log: "Orthotropic 3".
Cell data:
- `In-cavity_residual_stress_tensor`: 6 components in **Pa** (00010006: diagonal means 139, 139, 145 MPa;
  off-diagonal |max| about 7 MPa). Same component order as the orientation tensor (assumed, not documented).
- `Tensile_modulus_in_{first,second,third}_principal_direction_(3D_fiber)` [Pa] (3.7-9.2 / 2.3-4.4 / 2.2-3.3 GPa in
  00010006), `Shear_modulus_(G12|G13|G23)_(3D_fiber)` [Pa], `Poisson's_ratio_(v12|v13|v23)_(3D_fiber)` [-].
  These refer to the local principal directions of orientation; the directions themselves are not stored.
No field data. The Zenodo page gives neither units nor component orders; the units above come from the data and the logs.

## 4. Size of the warpage (all 196, `max_deflection_mm`, `max_defl_over_diag`)
Maximum deflection magnitude: 0.081-5.34 mm, median 0.53 mm (quartiles 0.37 / 0.90 mm).
Counts: < 0.1 mm: 2; 0.1-0.2: 16; 0.2-0.5: 69; 0.5-1: 67; 1-2: 34; 2-5: 6; > 5 mm: 2.
Divided by the bounding-box diagonal: 0.0020-0.035, median 0.0066 (quartiles 0.0052 / 0.0090).

## 5. Elements through the thickness
Not measured element by element. The Moldflow logs report it: all runs requested 10 tetrahedral refinement layers,
for which Moldflow expects node layer 6 at the part centre line (about 10 elements across). Warnings in the logs:
- node layer at the centre line 5 or less (8 elements or fewer across): median 36 % of the part, up to 100 %
  (100 % in 2 geometries, e.g. 00010006, smallest box edge 2.5 mm);
- 4 or less (6 elements or fewer): median 8 %, up to 85 %.
So the thin geometries have about 6-8 tetrahedra across, not 1-2; per geometry in the CSV
(`pct_centerline_layers_le5`, `pct_centerline_layers_le4`).
