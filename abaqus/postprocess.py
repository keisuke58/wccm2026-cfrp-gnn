"""postprocess.py -- extract key metrics from the CFRTP Abaqus .odb files and print a
summary comparable to the Python seed demos. Run with Abaqus' Python:

    abaqus python postprocess.py            # every odb present, from JOBS below
    abaqus python postprocess.py cure       # cfrtp_cure_residual.odb only
    abaqus python postprocess.py delam      # cfrtp_delamination_mixedmode.odb only
    abaqus python postprocess.py sanity     # cfrtp_1elem_sanity.odb only
    abaqus python postprocess.py ve         # cfrtp_cure_residual_ve.odb only
    abaqus python postprocess.py crystve    # cfrtp_cryst_residual_ve.odb only
    abaqus python postprocess.py crystpeek  # cfrtp_cryst_peek_validation.odb only
    abaqus python postprocess.py crystpeekhl # cfrtp_cryst_peek_hl.odb only
    abaqus python postprocess.py daikinpfa  # cfrtp_daikin_pfa_hl.odb only
    abaqus python postprocess.py delam3d    # cfrtp_delamination_3d.odb only

Uses odbAccess (Abaqus Python 2.7 interpreter). Runs on YOUR Abaqus box, not the
sandbox.

Metrics:
  cure/sanity/ve/crystve/crystpeek/crystpeekhl/daikinpfa : residual sigma_11
          range [MPa], warpage max|U3| [mm], SDV_alpha range (degree of cure /
          relative crystallinity)
          -> compare with cfrp_cure_residual_stress_fe.py / cfrtp_residual_stress_fe.py
  crystve/crystpeek/crystpeekhl/daikinpfa additionally report an FE-derived
          crystallization peak temperature Tp: the temperature at which
          d(alpha)/dt -- finite-differenced across output frames, using the
          *AMPLITUDE, NAME=MELTCOOL table in the matching .inp for the
          frame-time -> temperature lookup -- is maximal. This is the FE
          analogue of Tp in validation/cfrtp_peek_validation.py, and is
          comparable across decks even when their time axes are pseudo- vs.
          physical-time, since the melt->cool amplitude is monotonic: the
          (alpha, T) location of the rate peak doesn't depend on the time
          scaling, only on temperature.
          -> compare with validation/cfrtp_peek_validation.py
  delam/delam3d : peak reaction at the loaded tip, delamination front
          (max x with SDEG>0.5)
          -> compare with cfrtp_delamination_2d_fe.py
"""
import os
import sys

try:
    from odbAccess import openOdb
except Exception:                                  # pragma: no cover
    print("This must be run with Abaqus' Python:  abaqus python postprocess.py")
    raise


def last_step_frame(odb):
    stepname = list(odb.steps.keys())[-1]
    return odb.steps[stepname], odb.steps[stepname].frames[-1]


def _amplitude_table(inpname, ampname="MELTCOOL"):
    """Parse a `*AMPLITUDE, NAME=<ampname>, TIME=TOTAL TIME` (time, value)
    table out of an .inp file. Returns a time-sorted list of (time, value)
    pairs, or [] if the file/amplitude isn't found."""
    if not os.path.exists(inpname):
        return []
    pairs = []
    in_block = False
    with open(inpname) as f:
        for line in f:
            s = line.strip()
            if s.upper().startswith("*AMPLITUDE"):
                in_block = ("NAME=%s" % ampname).upper() in s.upper()
                continue
            if in_block:
                if s.startswith("*"):
                    break
                nums = [float(x) for x in s.split(",") if x.strip()]
                pairs.extend(zip(nums[0::2], nums[1::2]))
    return sorted(pairs)


def _temp_at(t, table):
    """Piecewise-linear lookup of an amplitude (time, value) table at time t."""
    if t <= table[0][0]:
        return table[0][1]
    if t >= table[-1][0]:
        return table[-1][1]
    for (t0, v0), (t1, v1) in zip(table, table[1:]):
        if t0 <= t <= t1:
            return v0 if t1 == t0 else v0 + (t - t0) / (t1 - t0) * (v1 - v0)
    return table[-1][1]


def crystallization_tp(step, inpname, ampname="MELTCOOL"):
    """FE analogue of Tp in validation/cfrtp_peek_validation.py: the
    temperature at which d(alpha)/dt -- finite-differenced across this step's
    output frames -- is maximal. Returns None if SDV_alpha or the amplitude
    table isn't available."""
    table = _amplitude_table(inpname, ampname)
    if len(table) < 2:
        return None
    ts, alphas = [], []
    for fr in step.frames:
        if "SDV_alpha" not in fr.fieldOutputs.keys():
            continue
        a = [v.data for v in fr.fieldOutputs["SDV_alpha"].values]
        ts.append(fr.frameValue)
        alphas.append(sum(a) / len(a))
    if len(ts) < 3:
        return None
    best_rate, best_t = -1.0, None
    for i in range(1, len(ts)):
        dt = ts[i] - ts[i - 1]
        if dt <= 0:
            continue
        rate = (alphas[i] - alphas[i - 1]) / dt
        tmid = 0.5 * (ts[i] + ts[i - 1])
        if rate > best_rate:
            best_rate, best_t = rate, tmid
    return _temp_at(best_t, table) if best_t is not None else None


def node_coords(odb):
    """label -> (x,y,z) over all instances."""
    xyz = {}
    for inst in odb.rootAssembly.instances.values():
        for n in inst.nodes:
            xyz[(inst.name, n.label)] = tuple(n.coordinates)
    return xyz


def cure(odbname="cfrtp_cure_residual.odb", label="cure", alpha_desc="degree of cure",
         find_tp=False, inpname=None, ampname="MELTCOOL"):
    odb = openOdb(odbname, readOnly=True)
    step, fr = last_step_frame(odb)
    s11 = [v.data[0] for v in fr.fieldOutputs["S"].values]      # 11-component
    u3 = [abs(v.data[2]) for v in fr.fieldOutputs["U"].values]  # out-of-plane warp
    print("[%s] step=%s frame=%d" % (label, step.name, len(step.frames) - 1))
    print("  residual sigma_11 range: [%.1f, %.1f] MPa" %
          (min(s11) / 1.0e6, max(s11) / 1.0e6))
    print("  warpage max|U3|: %.3f mm" % (max(u3) * 1.0e3))
    if "SDV_alpha" in fr.fieldOutputs.keys():
        a = [v.data for v in fr.fieldOutputs["SDV_alpha"].values]
        print("  %s alpha: [%.3f, %.3f]" % (alpha_desc, min(a), max(a)))
    if find_tp:
        tp = crystallization_tp(step, inpname or odbname.replace(".odb", ".inp"), ampname)
        if tp is not None:
            print("  FE crystallization Tp (peak d-alpha/dt): %.1f C  "
                  "-- cf. validation/cfrtp_peek_validation.py" % tp)
    odb.close()


def delam(odbname="cfrtp_delamination_mixedmode.odb", label="delam"):
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
    front = 0.0; damaged_elems = set()
    if "SDEG" in fr.fieldOutputs.keys():
        # COH2D4/COH3D8 report SDEG per integration point (2 / 4 per element), so
        # dedupe by element -- an element counts as damaged if ANY IP exceeds 0.5.
        for v in fr.fieldOutputs["SDEG"].values:
            if v.data > 0.5:
                key = (v.instance.name if v.instance else list(odb.rootAssembly.instances.keys())[0], v.elementLabel)
                damaged_elems.add(key)
                xs = [xyz[k][0] for k in conn.get(key, []) if k in xyz]
                if xs:
                    front = max(front, max(xs))
    damaged = len(damaged_elems)
    print("[%s] step=%s" % (label, stepname))
    print("  peak reaction |RF| at TIP: %.1f (at frame %d)" % (peak, peak_f))
    print("  delaminated cohesive elements (SDEG>0.5): %d" % damaged)
    print("  delamination front x: %.1f mm" % (front * 1.0e3))
    odb.close()


# (key, odb filename, handler, extra kwargs for the handler) -- order = print order
JOBS = [
    ("sanity",    "cfrtp_1elem_sanity.odb", cure, {"label": "sanity"}),
    ("cure",      "cfrtp_cure_residual.odb", cure, {}),
    ("ve",        "cfrtp_cure_residual_ve.odb", cure, {"label": "ve"}),
    ("crystve",   "cfrtp_cryst_residual_ve.odb", cure,
     {"label": "crystve", "alpha_desc": "relative crystallinity", "find_tp": True}),
    ("crystpeek", "cfrtp_cryst_peek_validation.odb", cure,
     {"label": "crystpeek", "alpha_desc": "relative crystallinity", "find_tp": True}),
    ("crystpeekhl", "cfrtp_cryst_peek_hl.odb", cure,
     {"label": "crystpeekhl", "alpha_desc": "relative crystallinity", "find_tp": True}),
    ("daikinpfa", "cfrtp_daikin_pfa_hl.odb", cure,
     {"label": "daikinpfa", "alpha_desc": "relative crystallinity", "find_tp": True}),
    ("delam",     "cfrtp_delamination_mixedmode.odb", delam, {}),
    ("delam3d",   "cfrtp_delamination_3d.odb", delam, {"label": "delam3d"}),
]

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    import os
    for key, odbname, handler, kwargs in JOBS:
        if which in (key, "all") and os.path.exists(odbname):
            handler(odbname, **kwargs)
