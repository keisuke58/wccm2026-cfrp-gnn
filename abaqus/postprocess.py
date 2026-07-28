"""postprocess.py -- extract key metrics from the CFRTP Abaqus .odb files and print a
summary comparable to the Python seed demos. Run with Abaqus' Python:

    abaqus python postprocess.py            # both odbs if present
    abaqus python postprocess.py cure       # cfrtp_cure_residual.odb only
    abaqus python postprocess.py delam      # cfrtp_delamination_mixedmode.odb only

Uses odbAccess (Abaqus Python 2.7 interpreter). Runs on YOUR Abaqus box, not the
sandbox. Untested here (no Abaqus license in this environment) -- if a field/step/
instance name differs in your version, adjust the small helpers below.

Metrics:
  cure  : residual sigma_11 range [MPa], warpage max|U3| [mm], degree-of-cure range
          -> compare with cfrp_cure_residual_stress_fe.py / cfrtp_residual_stress_fe.py
  delam : peak reaction at the loaded tip, delamination front (max x with SDEG>0.5)
          -> compare with cfrtp_delamination_2d_fe.py
"""
import sys

try:
    from odbAccess import openOdb
except Exception:                                  # pragma: no cover
    print("This must be run with Abaqus' Python:  abaqus python postprocess.py")
    raise


def last_step_frame(odb):
    stepname = list(odb.steps.keys())[-1]
    return odb.steps[stepname], odb.steps[stepname].frames[-1]


def node_coords(odb):
    """label -> (x,y,z) over all instances."""
    xyz = {}
    for inst in odb.rootAssembly.instances.values():
        for n in inst.nodes:
            xyz[(inst.name, n.label)] = tuple(n.coordinates)
    return xyz


def cure(odbname="cfrtp_cure_residual.odb"):
    odb = openOdb(odbname, readOnly=True)
    step, fr = last_step_frame(odb)
    s11 = [v.data[0] for v in fr.fieldOutputs["S"].values]      # 11-component
    u3 = [abs(v.data[2]) for v in fr.fieldOutputs["U"].values]  # out-of-plane warp
    print("[cure] step=%s frame=%d" % (step.name, len(step.frames) - 1))
    print("  residual sigma_11 range: [%.1f, %.1f] MPa" %
          (min(s11) / 1.0e6, max(s11) / 1.0e6))
    print("  warpage max|U3|: %.3f mm" % (max(u3) * 1.0e3))
    if "SDV_alpha" in fr.fieldOutputs.keys():
        a = [v.data for v in fr.fieldOutputs["SDV_alpha"].values]
        print("  degree of cure alpha: [%.3f, %.3f]" % (min(a), max(a)))
    odb.close()


def delam(odbname="cfrtp_delamination_mixedmode.odb"):
    odb = openOdb(odbname, readOnly=True)
    stepname = list(odb.steps.keys())[-1]; step = odb.steps[stepname]
    # peak reaction magnitude at the TIP node set (sum RF over the set) across frames
    tip = None
    for nm, ns in odb.rootAssembly.nodeSets.items():
        if nm.upper() == "TIP":
            tip = ns
    peak = 0.0; peak_f = -1
    for fi, fr in enumerate(step.frames):
        rf = fr.fieldOutputs["RF"]
        vals = rf.getSubset(region=tip).values if tip is not None else rf.values
        tot = 0.0
        for v in vals:
            tot += (v.data[0] ** 2 + v.data[1] ** 2) ** 0.5
        if tot > peak:
            peak = tot; peak_f = fi
    # delamination front at the last frame: max x among COH elements with SDEG>0.5
    fr = step.frames[-1]
    xyz = node_coords(odb)
    conn = {}
    for inst in odb.rootAssembly.instances.values():
        for e in inst.elements:
            conn[(inst.name, e.label)] = [(inst.name, l) for l in e.connectivity]
    front = 0.0; damaged = 0
    if "SDEG" in fr.fieldOutputs.keys():
        for v in fr.fieldOutputs["SDEG"].values:
            if v.data > 0.5:
                damaged += 1
                key = (v.instance.name if v.instance else list(odb.rootAssembly.instances.keys())[0], v.elementLabel)
                xs = [xyz[k][0] for k in conn.get(key, []) if k in xyz]
                if xs:
                    front = max(front, max(xs))
    print("[delam] step=%s" % stepname)
    print("  peak reaction |RF| at TIP: %.1f (at frame %d)" % (peak, peak_f))
    print("  delaminated cohesive elements (SDEG>0.5): %d" % damaged)
    print("  delamination front x: %.1f mm" % (front * 1.0e3))
    odb.close()


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    import os
    if which in ("cure", "all") and os.path.exists("cfrtp_cure_residual.odb"):
        cure()
    if which in ("delam", "all") and os.path.exists("cfrtp_delamination_mixedmode.odb"):
        delam()
