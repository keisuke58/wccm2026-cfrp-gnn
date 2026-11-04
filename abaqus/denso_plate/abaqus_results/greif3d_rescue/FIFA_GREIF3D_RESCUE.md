# GUIDE section 19 on fifa: RESCUE on Greif & Meyer 3-D parts (interim: 142 of 196 geometries)

**Interim push** (142 geometries; the user allowed pushing what is done). The user first set the scope to the 100
smallest geometries, then (4 Nov) allowed all 196 since the compute nodes were free; the jobs for the rest are still
running, and the final numbers will replace these.

## Setup
- Code: `validation/greif3d_rescue.py` at `dffce92` (PR #54 branch), run with the venv `~/IKM_Hiwi/.venv_render`
  (numpy, scipy, meshio) and CalculiX 2.23 from a separate conda env (`CCX=~/miniconda3/envs/calculix/bin/ccx`); CPU only.
- PBS jobs, 4 threads each: the 20 smallest and the 3-geometry K check (1 job, 36 min), then geometries 21-60 (2 jobs)
  and 61-100 (4 jobs) in parallel on the compute nodes.
- Constants as in the script (PP/GF30 generic values, illustrative); nothing tuned.

## Self-check (same as the guide)
| check | fifa | guide |
| --- | --- | --- |
| G1 patch test | 6.8e-12 | 6.8e-12 |
| G2 factors recovered (10 nodes, 2 factors) | 0.704 / 1.298, error 0.370 -> 0.021 | 0.704 / 1.298, 0.37 -> 0.021 |
| G3 C3D4 / C3D10 out-of-plane | 0.1287 / 0.3084 mm (0.42) | 0.42 |
| G3b C3D10 refined 2x | change 0.017 | 1.7 % |

## G5: K = 32 against K = 64 (3 smallest geometries)
`G5_k_change` (relative change of the prior warp field): 00010017 0.113, 00010520 0.088, 00010768 0.024.

## Summary (142 geometries; median, quartiles in parentheses)
```
  G4_E1_ratio_median   0.567 (0.557-0.576)
  prior                0.538 (0.487-0.570)
  cos_prior_data       0.981 (0.969-0.988)
  field_1f             0.194 (0.153-0.247)
  field_2f             0.162 (0.132-0.219)
  rescue_5_1f          0.201 (0.162-0.252)
  rescue_5_2f          0.191 (0.154-0.253)
  rescue_10_1f         0.198 (0.157-0.250)
  rescue_10_2f         0.173 (0.138-0.228)
  rescue_20_1f         0.195 (0.155-0.248)
  rescue_20_2f         0.169 (0.136-0.221)
```
- G4: the prior's first principal modulus is about 0.57 of the data's `Tensile_modulus_in_first_principal_direction`.
- The shape of the prior matches the data closely (cos median 0.98, no geometry below 0.5), but its size is about
  half: the fitted factors are near 2. A few measured points (5-20) bring the error close to the whole-field limit.

## Examples (by rescue_10_2f / prior)
```
works well:
  00010768 n_tet 46907 max_data 0.89 mm prior 0.51 -> rescue_10_2f 0.03 (field_2f 0.02) cos 1.00
  00010941 n_tet 61490 max_data 1.55 mm prior 0.54 -> rescue_10_2f 0.04 (field_2f 0.04) cos 1.00
  00010463 n_tet 165725 max_data 1.14 mm prior 0.61 -> rescue_10_2f 0.07 (field_2f 0.07) cos 0.99
works poorly:
  00010077 n_tet 73263 max_data 1.44 mm prior 0.76 -> rescue_10_2f 0.61 (field_2f 0.57) cos 0.82
  00010041 n_tet 76041 max_data 0.77 mm prior 0.59 -> rescue_10_2f 0.48 (field_2f 0.48) cos 0.88
  00010656 n_tet 240040 max_data 1.61 mm prior 0.38 -> rescue_10_2f 0.33 (field_2f 0.32) cos 0.94
cos < 0.5: 0 of 142
```
