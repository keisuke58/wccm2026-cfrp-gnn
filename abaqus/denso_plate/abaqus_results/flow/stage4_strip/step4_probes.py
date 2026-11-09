"""step4_probes.py -- GUIDE section 20 step 4: from the probes 50 mm from the gate (32 cell centres through the
thickness), write per depth from the top face: a_xx, a_yy, a_zz at the end of filling, the time the melt front passed
(first alpha.poly >= 0.5) and the time the layer froze (first T < TnoFlow after the front passed).

Axes of the strip: x flow direction, y width (one empty cell in the 2-D case), z thickness.

usage: python step4_probes.py <case dir> <out csv> [TnoFlow K]
"""
import csv
import re
import sys
from pathlib import Path

H = 3.08e-3        # plate thickness [m]


def read(path):
    """rows of (time, [value per probe]); tensors as tuples of floats."""
    pos, rows = [], []
    for ln in Path(path).read_text().splitlines():
        m = re.match(r"# Probe \d+ \(([^)]*)\)", ln)
        if m:
            pos.append(tuple(map(float, m.group(1).split())))
            continue
        if not ln.strip() or ln.startswith("#"):
            continue
        t, rest = ln.split(None, 1)
        vals = re.findall(r"\(([^)]*)\)", rest)
        vals = [tuple(map(float, v.split())) for v in vals] if vals else list(map(float, rest.split()))
        rows.append((float(t), vals))
    return pos, rows


def main():
    case, out = Path(sys.argv[1]), sys.argv[2]
    tnf = float(sys.argv[3]) if len(sys.argv) > 3 else 543.15
    dirs = sorted((case / "postProcessing/probes50mm").iterdir(), key=lambda p: float(p.name))
    get = lambda f: [r for d in dirs if (d / f).exists() for r in read(d / f)[1]]
    pos = read(dirs[0] / "alpha.poly")[0]
    al, T, A = get("alpha.poly"), get("T"), get("A2")
    n = len(pos)
    t_front = [next((t for t, v in al if v[k] >= 0.5), None) for k in range(n)]
    t_freeze = []
    for k in range(n):
        tf = t_front[k]
        t_freeze.append(None if tf is None else next((t for t, v in T if t >= tf and v[k] < tnf), None))
    t_end, a_end = A[-1]
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["depth_from_top_mm", "z_mm", "a_xx", "a_yy", "a_zz", "t_front_s", "t_freeze_s"])
        for k in sorted(range(n), key=lambda k: -pos[k][2]):
            xx, xy, xz, yy, yz, zz = a_end[k]
            w.writerow([f"{1e3 * (H - pos[k][2]):.4f}", f"{1e3 * pos[k][2]:.4f}", f"{xx:.5f}", f"{yy:.5f}", f"{zz:.5f}",
                        "" if t_front[k] is None else f"{t_front[k]:.5f}", "" if t_freeze[k] is None else f"{t_freeze[k]:.5f}"])
    filled = sum(v is not None for v in t_front)
    print(f"end of the probe record T = {t_end:.5f} s; front passed at {filled}/{n} probes; "
          f"frozen at {sum(v is not None for v in t_freeze)}/{n}; TnoFlow {tnf} K")


if __name__ == "__main__":
    main()
