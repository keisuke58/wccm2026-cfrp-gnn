"""survey.py data_dir out.csv -- one row per Greif & Meyer geometry: mesh size, bounding box, max deflection,
through-thickness layer warnings from the Moldflow log. Reads only; the data is not changed."""
import csv
import glob
import re
import sys
from pathlib import Path

import meshio
import numpy as np

D, OUT = Path(sys.argv[1]), sys.argv[2]
rows = []
for d in sorted(p for p in D.iterdir() if p.is_dir()):
    vtu = sorted(d.glob("*_sim.vtu"))
    log = sorted(d.glob("*.log"))
    if not vtu:
        continue
    m = meshio.read(vtu[0])
    types = {c.type: len(c.data) for c in m.cells}
    lo, hi = m.points.min(0), m.points.max(0)
    ext = hi - lo
    key = [k for k in m.point_data if k.startswith("Deflection")][0]
    u = np.linalg.norm(m.point_data[key], axis=1) * 1e3          # m -> mm (log: "unit: mm" equals VTU * 1e3)
    txt = log[0].read_text(errors="replace") if log else ""
    layers = re.search(r"Tetrahedral refinement layers \(specified minimum\)\s*=\s*(\d+)", txt)
    p5 = re.search(r"centerline is 5 or less for\s+([\d.]+) percent", txt)
    p4 = re.search(r"\(i\.e\. 4 or less\) for\s+([\d.]+) percent", txt)
    lmax = re.search(r"Trans-X.*\n.*Trans-Y.*\n.*Trans-Z", txt)
    rows.append(dict(id=d.name, file=vtu[0].name, cell_types=";".join(f"{k}:{v}" for k, v in types.items()),
                     n_cells=sum(types.values()), n_points=len(m.points),
                     ext_x_mm=round(float(ext[0]), 3), ext_y_mm=round(float(ext[1]), 3), ext_z_mm=round(float(ext[2]), 3),
                     min_ext_mm=round(float(ext.min()), 3), diag_mm=round(float(np.linalg.norm(ext)), 3),
                     max_deflection_mm=round(float(u.max()), 5), max_defl_over_diag=float(u.max() / np.linalg.norm(ext)),
                     layers_requested=int(layers.group(1)) if layers else "",
                     pct_centerline_layers_le5=float(p5.group(1)) if p5 else 0.0,
                     pct_centerline_layers_le4=float(p4.group(1)) if p4 else 0.0))
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print(len(rows), "geometries")
