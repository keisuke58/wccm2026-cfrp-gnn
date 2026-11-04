# DENSO data intake report

input: `/home/user/wccm2026-cfrp-gnn/results/denso_intake/selftest`; files: hole_drilling.csv, orientation_profile.csv, specimens.csv, warpage_points.csv

## Checks

- no problems found

## Results

### PAPER
- orientation: given
- H1 T0_eff 143 C; RMSE 57.5 -> 15.0 micro-strain
- H2 common factor 0.67 at T0 = 180 C; RMSE 15.1
- H3 MD only: T0 144 C; TD/45 RMSE 46.0 -> 10.3
- warpage: 25 points, measured curvature kx -0.627, ky +1.061 1/m (surface fit rms 0.000 mm)
  - predicted (T0 = 180 C, 180 C): kx -0.628, ky +1.062 1/m
  - predicted (T0_eff from hole drilling, 143 C): kx -0.430, ky +0.699 1/m
