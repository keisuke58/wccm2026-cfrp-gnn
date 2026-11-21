import re, sys
import numpy as np
d = sys.argv[1] + "/constant/polyMesh/"
def body(fn):
    s = open(d + fn).read()
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    s = s[s.index("FoamFile"):]; s = s[s.index("}") + 1:]
    return s
s = body("points"); n = int(re.search(r"(\d+)\s*\(", s).group(1))
P = np.array(re.findall(r"\(([-\deE.+]+) ([-\deE.+]+) ([-\deE.+]+)\)", s), float)
s = body("faces"); F = [list(map(int, m.split())) for m in re.findall(r"\d+\(([\d ]+)\)", s)]
b = body("boundary")
print("bbox", P.min(0), P.max(0))
for name in ("inlet", "outlet", "walls"):
    m = re.search(name + r"\s*\{[^}]*nFaces\s+(\d+);[^}]*startFace\s+(\d+);", b)
    if not m: continue
    nf, sf = int(m.group(1)), int(m.group(2))
    A = 0; ctr = []; nrm = np.zeros(3)
    for f in F[sf:sf + nf]:
        p = P[f]; c = p.mean(0); a = 0.5 * sum(np.cross(p[i] - c, p[(i + 1) % len(p)] - c) for i in range(len(p)))
        A += np.linalg.norm(a); nrm += a; ctr.append(c)
    ctr = np.array(ctr)
    print(name, "faces", nf, "area m2 %.4e" % A, "n", np.round(nrm / np.linalg.norm(nrm), 3), "centre bbox", ctr.min(0), ctr.max(0),
          "mean face %.3e m" % np.sqrt(A / nf))
