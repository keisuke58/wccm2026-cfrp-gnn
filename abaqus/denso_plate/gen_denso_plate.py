"""abaqus/denso_plate/gen_denso_plate.py -- Abaqus decks for the Denso residual-strain
plate, plus the matching lamination-theory reference, so that the Python model used for the
ML rehearsal (denso_residual_strain_ml.py, viz/denso_plate_features.py) can be checked against
3-D solid finite elements.

Model written to each deck:
  * quarter of a free 100 mm x 100 mm x 3 mm plate (symmetry planes x = 0 and y = 0),
    NL = 40 element layers through the thickness, C3D8I (incompatible modes, for bending);
  * one material per layer: the orientation-averaged stiffness <C> (hybrid closure,
    fiber_orientation_cte.py), orthotropic in the global axes because a_ij is diagonal;
  * the eigenstrain of each layer, e* = <alpha> dT + <beta> X_c, imposed as an orthotropic
    expansion alpha_eff = e* / dT with the whole plate cooled uniformly by
    dT = T_room - T_lock. For a linear elastic model this reproduces the eigenstrain exactly;
    it is a device for loading, not a claim that crystallisation is thermal;
  * layer properties are evaluated at the layer mid-depth (orientation from its profile
    function, X_c linearly interpolated between the 41 depth points of the 1-D cooling run).

Reference written next to each deck (<job>_ref.json): the classical-lamination solution for
the SAME piecewise-constant layers, integrated exactly (A, B, D by layer). The centre column
lies 33 plate thicknesses from the free edges, so edge effects should not reach it; the pass
criterion in compare_denso_plate.py is 5e-3 relative on strain and sag.

Validated here (no Abaqus in this environment):
  G1 the layer stiffness is orthotropic in x, y, z (normal-shear couplings vanish), so the
     *ELASTIC, TYPE=ORTHOTROPIC card loses nothing;
  G2 force and moment balance of the piecewise-constant reference (machine precision);
  G3 the piecewise-constant reference differs from the 41-point trapezoid model used by the ML
     script by 2-4 % of the strain range (layer midpoints vs nodes across the steep skin; a
     discretisation difference, printed, not asserted);
  G4 the same deck, converted to CalculiX 2.21 syntax (to_calculix.sh: keyword spellings only),
     agrees with the reference: strain error 6.9e-4 (asymmetric) / 5.5e-4 (symmetric), sag_y
     -1.3619 vs -1.3618 mm (calculix_check/).
NOT validated: the decks have not been run in Abaqus itself. Run them on the Abaqus host
(run_denso_plate.sh) and check with compare_denso_plate.py before using them for anything.

Honest scope: the physics is the same linear elastic eigenstrain model as the Python seed
(no heat transfer, no crystallisation kinetics or viscoelasticity inside Abaqus; X_c comes
from the Python 1-D cooling run). Running it in Abaqus checks the lamination model and builds
the solver-to-dataset pipeline; it adds no new physics except finite-plate edge effects. The
next step is a sequential heat-transfer + HL-UMAT deck (../cfrtp_cryst_umat_hl.f). All
constants are ILLUSTRATIVE (uncalibrated), as in the Python seeds.

Usage (any machine with python3 + numpy + matplotlib; Abaqus is NOT needed for this step):
  python3 abaqus/denso_plate/gen_denso_plate.py                       # symmetric + asymmetric
  python3 abaqus/denso_plate/gen_denso_plate.py --part thermal        # thermal part only
  python3 abaqus/denso_plate/gen_denso_plate.py --sample 20 --seed 1  # 20 random source plates
Decks go to abaqus/denso_plate/jobs/ (gitignored).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (ROOT, os.path.join(ROOT, "validation"), os.path.join(ROOT, "viz")):
    sys.path.insert(0, p)
from fiber_orientation_cte import ud_stiffness, orient_avg_cte, VOIGT  # noqa: E402
from denso_plate_features import orientation_profile, reduced_Q, CASES  # noqa: E402
import denso_residual_strain_ml as ml  # noqa: E402

NL = 40                 # element layers through the thickness (= ml.NZ - 1)
NX = NY = 25            # in-plane elements of the quarter plate (2 mm)
L_PLATE = 0.10          # full plate side [m]; the quarter model is L/2 x L/2
assert ml.NZ == NL + 1


# ------------------------------------------------------------------ layers --------
def layer_data(orient_fn, Xc_nodes, c, part="total"):
    """Per-layer <C> (6x6, repo Voigt 11,22,33,23,13,12), eigenstrain e* (3 normal comps)."""
    zn = np.linspace(-1.0, 1.0, NL + 1)
    zm = 0.5 * (zn[1:] + zn[:-1])
    a11, a22, a33 = orient_fn(zm)
    Xc = 0.5 * (Xc_nodes[1:] + Xc_nodes[:-1])
    Cud = ud_stiffness(**c.UD)
    dT = c.T_ROOM - c.T_LOCK
    Cs, es = [], []
    for x, y, w, X in zip(a11, a22, a33, Xc):
        a = np.diag([x, y, w])
        al, Cavg = orient_avg_cte(a, "hybrid", Cud, aL=c.A_L, aT=c.A_T)
        be, _ = orient_avg_cte(a, "hybrid", Cud, aL=c.B_L, aT=c.B_T)
        eT = np.diag(al) * dT
        eX = np.diag(be) * X
        es.append({"total": eT + eX, "thermal": eT, "cryst": eX}[part])
        Cs.append(Cavg)
    return dict(zm=zm, a11=a11, a22=a22, a33=a33, Xc=Xc, C=np.array(Cs), estar=np.array(es),
                dT=dT)


def orthotropy_defect(Cs):
    """G1: largest normal-shear or shear-shear coupling relative to max |C|."""
    bad = [(i, j) for i in range(6) for j in range(6)
           if i != j and (i >= 3 or j >= 3)]
    return max(abs(C[i, j]) for C in Cs for (i, j) in bad) / np.max(np.abs(Cs))


def lamination_pc(zm_unit, Cs, estar, H):
    """Exact lamination for piecewise-constant layers; returns e0, kappa, mech strain at mids."""
    h = H / NL
    z0 = -H / 2 + h * np.arange(NL); z1 = z0 + h
    Qs = np.array([reduced_Q(C) for C in Cs])
    e = np.zeros((NL, 3)); e[:, 0] = estar[:, 0]; e[:, 1] = estar[:, 1]   # xx, yy, xy=0
    A = np.einsum("k,kij->ij", z1 - z0, Qs)
    B = np.einsum("k,kij->ij", (z1 ** 2 - z0 ** 2) / 2, Qs)
    D = np.einsum("k,kij->ij", (z1 ** 3 - z0 ** 3) / 3, Qs)
    Ns = np.einsum("k,kij,kj->i", z1 - z0, Qs, e)
    Ms = np.einsum("k,kij,kj->i", (z1 ** 2 - z0 ** 2) / 2, Qs, e)
    sol = np.linalg.solve(np.block([[A, B], [B, D]]), np.concatenate([Ns, Ms]))
    e0, kap = sol[:3], sol[3:]
    zmid = zm_unit * H / 2
    em = e0[None, :] + zmid[:, None] * kap[None, :] - e
    sig = np.einsum("kij,kj->ki", Qs, em)
    # exact layer integrals of the (linear-in-z) stress Q (e0 - e* + z kappa)
    s0 = np.einsum("kij,kj->ki", Qs, e0[None, :] - e); s1 = np.einsum("kij,j->ki", Qs, kap)
    F = np.einsum("k,ki->i", z1 - z0, s0) + np.einsum("k,ki->i", (z1 ** 2 - z0 ** 2) / 2, s1)
    M = (np.einsum("k,ki->i", (z1 ** 2 - z0 ** 2) / 2, s0)
         + np.einsum("k,ki->i", (z1 ** 3 - z0 ** 3) / 3, s1))
    smag = np.sum((z1 - z0) * np.abs(sig[:, 0])) + 1e-30
    return dict(e0=e0, kappa=kap, em=em, sig=sig,
                balance_F=float(np.max(np.abs(F[:2])) * H / smag),
                balance_M=float(np.max(np.abs(M[:2])) / (smag * H / 2)))


# ------------------------------------------------------------------ deck ----------
def nid(i, j, k):
    return 1 + i + (NX + 1) * (j + (NY + 1) * k)


def eid(i, j, k):
    return 1 + i + NX * (j + NY * k)


def _labels(lines, labels, per=16):
    for s in range(0, len(labels), per):
        lines.append(", ".join(str(v) for v in labels[s:s + per]) + ",")


def write_deck(path, job, title, lay, c):
    H = c.H; a = L_PLATE / 2
    L = ["*HEADING", f" Denso residual-strain plate ({title}) -- quarter of a free "
         f"{L_PLATE*1e3:.0f} mm square plate, H = {H*1e3:.1f} mm",
         " Generated by abaqus/denso_plate/gen_denso_plate.py. ILLUSTRATIVE constants.",
         "** Per-layer orientation-averaged orthotropic stiffness; eigenstrain e* = <alpha> dT +",
         "** <beta> X_c imposed as alpha_eff = e*/dT under a uniform cool-down T_lock -> T_room.",
         "** Units: m, Pa, deg C.", "*NODE, NSET=NALL"]
    xs = np.linspace(0.0, a, NX + 1); ys = np.linspace(0.0, a, NY + 1)
    zs = np.linspace(-H / 2, H / 2, NL + 1)
    for k in range(NL + 1):
        for j in range(NY + 1):
            for i in range(NX + 1):
                L.append(f"{nid(i, j, k)}, {xs[i]:.9e}, {ys[j]:.9e}, {zs[k]:.9e}")
    L.append("*ELEMENT, TYPE=C3D8I, ELSET=EALL")
    for k in range(NL):
        for j in range(NY):
            for i in range(NX):
                n = [nid(i, j, k), nid(i + 1, j, k), nid(i + 1, j + 1, k), nid(i, j + 1, k),
                     nid(i, j, k + 1), nid(i + 1, j, k + 1), nid(i + 1, j + 1, k + 1),
                     nid(i, j + 1, k + 1)]
                L.append(f"{eid(i, j, k)}, " + ", ".join(map(str, n)))
    for k in range(NL):
        L += [f"*ELSET, ELSET=L{k+1:02d}, GENERATE", f" {eid(0, 0, k)}, {eid(NX-1, NY-1, k)}, 1"]
    col_e = [eid(0, 0, k) for k in range(NL)]
    L.append("*ELSET, ELSET=ECOL"); _labels(L, col_e)
    sets = {
        "XSYM": [nid(0, j, k) for k in range(NL + 1) for j in range(NY + 1)],
        "YSYM": [nid(i, 0, k) for k in range(NL + 1) for i in range(NX + 1)],
        "NFIX": [nid(0, 0, 0)],
        "MIDX": [nid(i, 0, NL // 2) for i in range(NX + 1)],
        "MIDY": [nid(0, j, NL // 2) for j in range(NY + 1)],
    }
    for nm, lab in sets.items():
        L.append(f"*NSET, NSET={nm}"); _labels(L, lab)
    ab = {"D1111": (0, 0), "D1122": (0, 1), "D2222": (1, 1), "D1133": (0, 2),
          "D2233": (1, 2), "D3333": (2, 2), "D1212": (5, 5), "D1313": (4, 4), "D2323": (3, 3)}
    # Abaqus requires a local orientation for anisotropic materials (fifa, 3 Nov 2026: "Anisotropic material
    # properties without a local orientation system"); this one equals the global axes, so nothing else changes
    L += ["*ORIENTATION, NAME=GLOBAL", " 1.0, 0.0, 0.0, 0.0, 1.0, 0.0"]
    for k in range(NL):
        C = lay["C"][k]; ae = lay["estar"][k] / lay["dT"]
        v = [C[ij] for ij in ab.values()]
        L += [f"*SOLID SECTION, ELSET=L{k+1:02d}, MATERIAL=M{k+1:02d}, ORIENTATION=GLOBAL",
              f"*MATERIAL, NAME=M{k+1:02d}",
              f"** z/(H/2) = {lay['zm'][k]:+.4f}: a11 {lay['a11'][k]:.4f}, a22 {lay['a22'][k]:.4f}, "
              f"a33 {lay['a33'][k]:.4f}, X_c {lay['Xc'][k]:.4f}",
              "*ELASTIC, TYPE=ORTHOTROPIC",
              " " + ", ".join(f"{x:.10e}" for x in v[:8]) + ",",
              f" {v[8]:.10e}",
              "*EXPANSION, TYPE=ORTHO",
              " " + ", ".join(f"{x:.10e}" for x in ae)]
    L += ["*INITIAL CONDITIONS, TYPE=TEMPERATURE", f" NALL, {c.T_LOCK:.4f}",
          "*BOUNDARY", " XSYM, 1, 1, 0.0", " YSYM, 2, 2, 0.0", " NFIX, 3, 3, 0.0",
          "*STEP, NAME=COOL, NLGEOM=NO", "*STATIC", " 1.0, 1.0",
          "*TEMPERATURE", f" NALL, {c.T_ROOM:.4f}",
          "*OUTPUT, FIELD", "*NODE OUTPUT", " U", "*ELEMENT OUTPUT", " S, E, EE, THE",
          "*EL PRINT, ELSET=ECOL, SUMMARY=NO", " EE", "*END STEP"]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    return dict(col_elements=col_e, midx_nodes=sets["MIDX"], midy_nodes=sets["MIDY"],
                x_mid=xs.tolist(), y_mid=ys.tolist())


# ------------------------------------------------------------------ main ----------
def build(job, title, orient_fn, Xc_nodes, c, part, outdir, extra=None):
    lay = layer_data(orient_fn, Xc_nodes, c, part)
    g1 = orthotropy_defect(lay["C"])
    ref = lamination_pc(lay["zm"], lay["C"], lay["estar"], c.H)
    info = write_deck(os.path.join(outdir, job + ".inp"), job, title, lay, c)
    out = dict(job=job, title=title, part=part, NL=NL, NX=NX, NY=NY, L_plate=L_PLATE, H=c.H,
               T_lock=c.T_LOCK, T_room=c.T_ROOM, zm=lay["zm"].tolist(),
               a11=lay["a11"].tolist(), a22=lay["a22"].tolist(), a33=lay["a33"].tolist(),
               Xc=lay["Xc"].tolist(),
               ref_exx=ref["em"][:, 0].tolist(), ref_eyy=ref["em"][:, 1].tolist(),
               ref_e0=ref["e0"].tolist(), ref_kappa=ref["kappa"].tolist(),
               orthotropy_defect=g1, balance_F=ref["balance_F"], balance_M=ref["balance_M"],
               **info, **(extra or {}))
    with open(os.path.join(outdir, job + "_ref.json"), "w") as f:
        json.dump(out, f, separators=(",", ":"))
    return out, ref


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--part", default="total", choices=["total", "thermal", "cryst"])
    ap.add_argument("--sample", type=int, default=0,
                    help="instead of the two named cases, write N random SOURCE plates "
                         "(same sampler as denso_residual_strain_ml.py)")
    ap.add_argument("--seed", type=int, default=ml.SEED)
    ap.add_argument("--out", default=os.path.join(HERE, "jobs"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    c = ml.SOURCE

    jobs = []
    if args.sample:
        rng = np.random.default_rng(args.seed)
        for n, (Tb, Tt, op) in enumerate(ml.sample_params(rng, args.sample)):
            jobs.append((f"denso_sample_{n:04d}", f"sample {n}, mould {Tb:.1f}/{Tt:.1f} C",
                         (lambda z, op=op: ml.orientation(z, op)), Tb, Tt,
                         dict(T_bottom=Tb, T_top=Tt, orient_params=list(op), seed=args.seed)))
    else:
        for name, (Tb, Tt) in CASES.items():
            sfx = "" if args.part == "total" else "_" + args.part
            jobs.append((f"denso_{name}{sfx}", f"{name}, mould {Tb:.0f}/{Tt:.0f} C, {args.part}",
                         orientation_profile, Tb, Tt, dict(T_bottom=Tb, T_top=Tt)))

    manifest = []
    for job, title, ofn, Tb, Tt, extra in jobs:
        Xc_nodes = ml.crystallise(Tb, Tt, c)
        out, ref = build(job, title, ofn, Xc_nodes, c, args.part, args.out, extra)
        # G3: compare with the 41-point trapezoid model used by the ML script / viewer
        z = np.linspace(-1.0, 1.0, ml.NZ)
        Cud = ud_stiffness(**c.UD); dT = c.T_ROOM - c.T_LOCK
        a11, a22, a33 = ofn(z)
        Qs, es = [], []
        for x, y, w, X in zip(a11, a22, a33, Xc_nodes):
            al, Cavg = orient_avg_cte(np.diag([x, y, w]), "hybrid", Cud, aL=c.A_L, aT=c.A_T)
            be, _ = orient_avg_cte(np.diag([x, y, w]), "hybrid", Cud, aL=c.B_L, aT=c.B_T)
            eT = np.diag(al) * dT; eX = np.diag(be) * X
            e = {"total": eT + eX, "thermal": eT, "cryst": eX}[args.part]
            Qs.append(reduced_Q(Cavg)); es.append([e[0], e[1], 0.0])
        from denso_plate_features import lamination
        _, kap_t, em_t, *_ = lamination(z * c.H / 2, np.array(Qs), np.array(es))
        exx_t = np.interp(out["zm"], z, em_t[:, 0])
        g3 = float(np.max(np.abs(exx_t - ref["em"][:, 0])) / np.max(np.abs(ref["em"][:, 0])))
        sag = ref["kappa"][1] * L_PLATE ** 2 / 8 * 1e3
        print(f"{job:28s} G1 orthotropy {out['orthotropy_defect']:.1e} | G2 balance F "
              f"{out['balance_F']:.1e} M {out['balance_M']:.1e} | G3 vs 41-pt model "
              f"{g3:.1e} | exx {ref['em'][:,0].min()*1e3:+.3f}..{ref['em'][:,0].max()*1e3:+.3f}"
              f" x1e-3, sag_y {sag:+.3f} mm")
        assert out["orthotropy_defect"] < 1e-12, "layer stiffness not orthotropic in xyz"
        assert max(out["balance_F"], out["balance_M"]) < 1e-10, "reference not balanced"
        manifest.append(job)
    with open(os.path.join(args.out, "manifest.txt"), "w") as f:
        f.write("\n".join(manifest) + "\n")
    print(f"wrote {len(manifest)} deck(s) + references to {args.out}")


if __name__ == "__main__":
    main()
