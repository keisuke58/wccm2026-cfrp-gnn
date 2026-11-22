"""validation/flow_cfd_vs_1d.py -- the 1-D flow-orientation model (validation/denso_flow_orientation.py, as tabulated in
denso_flow_coupling.py) against the 2-D openInjMoldSimF filling run on fifa (GUIDE §20, stage4_strip, 50 mm from the gate).

CFD: strip 80 x 3.08 mm, one cell across (2-D), cells 0.096 mm; melt 320 C, mould 150 C (h = 5e3 W/m2K); material the
example's HDPE with temperature constants shifted +160 K (illustrative, NOT PPS), Cross-WLF n = 0.394; Folgar-Tucker
with Ci = 0.00207 (Bay), aspect ratio 16.8, hybrid closure; ISOTROPIC orientation at the gate. Diverged at 74 % filled
(T = 0.767 s, a gate-corner instability, FIFA_STAGE4.md); the 50 mm section filled at 0.62-0.73 s, the CSV is from
the records up to 0.766 s. Only the outermost cell froze, so the CFD has (almost) no frozen skin at that time.
1-D model with the same inputs: isotropic a0, Ci = 0.00207, kappa = 1, L = 50 mm, NO frozen skin; the power-law index n
is swept over the material's range (Cross-WLF: n = 0.394 at high stress, -> 1 at low stress), NOT fitted.
Reported: RMSE of a_xx, a_yy against the CFD, the mid-plane a_xx, and the width where a_xx < 0.65 (the less-aligned core).
Honest scope: 2-D, illustrative material, 74 % filled, one site; the CFD cannot test the two-zone rule (no frozen skin
yet); this checks the SHEAR part of the 1-D model only.
Run:  python3 validation/flow_cfd_vs_1d.py   (writes results/denso_flow/cfd_vs_1d.json)
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import denso_flow_coupling as fc  # noqa: E402
import denso_flow_orientation as fo  # noqa: E402

CSV = os.path.join(ROOT, "abaqus", "denso_plate", "abaqus_results", "flow", "stage4_strip", "orientation_50mm_2D.csv")


def main():
    rows = list(csv.DictReader(open(CSV)))
    d = np.array([float(r["depth_from_top_mm"]) for r in rows])
    ax = np.array([float(r["a_xx"]) for r in rows]); ay = np.array([float(r["a_yy"]) for r in rows])
    dz = d[1] - d[0]
    width = lambda a: float(np.ptp(d[a < 0.65]) + dz) if (a < 0.65).any() else 0.0  # noqa: E731
    iso = (1 / 3, 1 / 3, 1 / 3)
    tr = fc.trajectory(iso, 0.00207, 1.0)
    out = dict(cfd=dict(mid_axx=float(ax[len(ax) // 2]), width_mm=width(ax)), model=[])
    print(f"CFD (2-D, 74 % filled): mid-plane a_xx {ax[len(ax) // 2]:.2f}, width a_xx < 0.65 {width(ax):.2f} mm, "
          f"a_yy > a_xx at {int((ay > ax).sum())} of {len(ax)} depths")
    for n in (0.394, 0.5, 0.7, 1.0):
        p = dict(fo.BASE, L=50e-3, n=n, T_nf=fo.BASE["Tb"], a0=iso)        # T_nf = wall temperature -> no frozen skin
        r = fc.profile(p, 0.0, tr)
        mx = np.interp(d, r["depth"], r["axx"]); my = np.interp(d, r["depth"], r["ayy"])
        row = dict(n=n, rmse_axx=float(np.sqrt(np.mean((mx - ax) ** 2))), rmse_ayy=float(np.sqrt(np.mean((my - ay) ** 2))),
                   mid_axx=float(mx[len(mx) // 2]), width_mm=width(mx))
        out["model"].append(row)
        print(f"1-D n {n:.3f}: RMSE a_xx {row['rmse_axx']:.3f}, a_yy {row['rmse_ayy']:.3f}; mid a_xx {row['mid_axx']:.2f}; "
              f"width {row['width_mm']:.2f} mm")
    dst = os.path.join(ROOT, "results", "denso_flow", "cfd_vs_1d.json")
    json.dump(out, open(dst, "w"), indent=1); print("wrote", dst)


if __name__ == "__main__":
    main()
