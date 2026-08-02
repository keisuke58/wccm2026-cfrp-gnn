"""springin_angle.py -- measure the L-bracket SPRING-IN from the deformed ODB.

Run inside Abaqus Python (odbAccess):
    abaqus python springin_angle.py            # reads cfrtp_lbracket_springin.odb

Spring-in is the process-induced closure of the corner of a curved composite
part: the two flanges, initially at 90 deg, rotate toward each other as residual
stresses (CTE mismatch + crystallization shrinkage, locked in through the corner)
relax on cool-down. It is THE classic, directly-measurable validation quantity
for a residual-stress model -- you compare the predicted angle change with a
protractor/CMM measurement on a molded part.

Method: from the last frame, take the deformed coordinates (X + U) of the flange
node sets written by gen_inp.gen_lbracket_springin --
    A_TIP, A_ROOT   (leg A, the initially-vertical flange)
    B_ROOT, B_TIP   (leg B, the initially-horizontal flange)
form the flange axis vectors vA = A_TIP - A_ROOT and vB = B_TIP - B_ROOT, and
report the enclosed angle between them. Spring-in = 90 deg - enclosed_angle
(positive = corner closed, as expected physically). Undeformed enclosed angle is
90 deg by construction, so the whole change is the process-induced spring-in.
"""
from __future__ import print_function
import math
try:
    from odbAccess import openOdb
except ImportError:
    raise SystemExit("run with:  abaqus python springin_angle.py  (needs odbAccess)")

ODB = "cfrtp_lbracket_springin.odb"
INST = "PART-1-1"        # default instance name for a model with no *PART/*INSTANCE
SETS = ("A_TIP", "A_ROOT", "B_ROOT", "B_TIP")


def _node_of_set(odb, inst, name):
    """Return the single node label in node set `name`."""
    ra = odb.rootAssembly
    if name in ra.nodeSets and ra.nodeSets[name].nodes:
        grp = ra.nodeSets[name].nodes
        return grp[0][0].label if isinstance(grp[0], (list, tuple)) else grp[0].label
    ns = odb.rootAssembly.instances[inst].nodeSets[name]
    return ns.nodes[0].label


def _instance(odb):
    ra = odb.rootAssembly
    if INST in ra.instances:
        return INST
    return list(ra.instances.keys())[0]


def main():
    odb = openOdb(ODB, readOnly=True)
    inst = _instance(odb)
    instobj = odb.rootAssembly.instances[inst]

    # undeformed coordinates, keyed by node label
    coord = {n.label: n.coordinates for n in instobj.nodes}
    labels = {s: _node_of_set(odb, inst, s) for s in SETS}

    step = odb.steps[list(odb.steps.keys())[-1]]
    frame = step.frames[-1]
    U = frame.fieldOutputs["U"]
    Usub = U.getSubset(region=instobj)
    disp = {v.nodeLabel: v.data for v in Usub.values}

    def defo(setname):
        lab = labels[setname]
        c = coord[lab]
        d = disp.get(lab, (0.0, 0.0, 0.0))
        return [c[i] + d[i] for i in range(3)]

    pA_tip, pA_root = defo("A_TIP"), defo("A_ROOT")
    pB_root, pB_tip = defo("B_ROOT"), defo("B_TIP")

    def vec(p, q):
        return [p[i] - q[i] for i in range(3)]

    def angle(u, v):
        du = math.sqrt(sum(c * c for c in u))
        dv = math.sqrt(sum(c * c for c in v))
        cos = sum(u[i] * v[i] for i in range(3)) / (du * dv + 1e-30)
        cos = max(-1.0, min(1.0, cos))
        return math.degrees(math.acos(cos))

    vA = vec(pA_tip, pA_root)          # along flange A, corner -> tip
    vB = vec(pB_tip, pB_root)          # along flange B, corner -> tip
    enclosed = angle(vA, vB)
    springin = 90.0 - enclosed

    print("=" * 60)
    print("L-bracket spring-in (from %s, step '%s', last frame)" % (ODB, step.name))
    print("  enclosed flange angle : %.3f deg  (undeformed = 90.000)" % enclosed)
    print("  SPRING-IN             : %+.3f deg  (+ = corner closed)" % springin)
    print("=" * 60)
    print("  NB: PLACEHOLDER Daikin-PFA params -- magnitude illustrative until")
    print("      calibrated to DSC/DMA + a measured molded-part angle.")
    odb.close()


if __name__ == "__main__":
    main()
