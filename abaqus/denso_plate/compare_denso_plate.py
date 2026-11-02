"""compare_denso_plate.py -- compare the solid-FE result of a Denso plate deck with its
lamination-theory reference (<job>_ref.json from gen_denso_plate.py). Plain python3 + numpy.

    cd abaqus/denso_plate/jobs
    python3 ../compare_denso_plate.py                        # every <job>_abaqus.json here
    python3 ../compare_denso_plate.py denso_asymmetric --fig  # also write <job>_compare.png
    python3 ../compare_denso_plate.py denso_asymmetric --calculix denso_asymmetric_ccx.dat
    cd ansys/denso_plate/jobs && python3 ../../../abaqus/denso_plate/compare_denso_plate.py --ansys --fig

Quantities compared:
  * residual (elastic) strain e_xx, e_yy of the centre-column elements vs the reference at the
    layer mid-depths; error = max |FE - ref| / max |ref| over both components;
  * curvatures kappa_x, kappa_y from a least-squares fit w = c0 + c2 s^2 to the mid-plane U3
    on the inner half of each symmetry axis (kappa = -2 c2), reported as the sag over the full
    plate, kappa L^2 / 8.
Pass: strain error < 5e-3 and |sag_FE - sag_ref| < max(5e-3 |sag_ref|, 1e-3 mm).
Recorded check without Abaqus: CalculiX 2.21 on the same deck (keyword spellings adapted;
research/GUIDE_fifa_abaqus.md) gives strain error 7e-4 and kappa_y error 6e-5 (asymmetric).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np

TOL_STRAIN = 5e-3
TOL_SAG_REL, TOL_SAG_MM = 5e-3, 1e-3


def read_calculix(path, ref):
    """ME (mechanical strain) per integration point and U for sets MIDX/MIDY from a ccx .dat."""
    L = open(path).read().splitlines()

    def block(key):
        i = next(n for n, l in enumerate(L) if key in l)
        rows = []
        for l in L[i + 1:]:
            t = l.split()
            if not t:
                if rows:
                    break
                continue
            if not t[0].lstrip("-").isdigit():
                break
            rows.append([float(x) for x in t])
        return np.array(rows)

    me = block("mechanical strains")
    acc = {}
    for r in me:
        acc.setdefault(int(r[0]), []).append(r[2:4])
    m = np.array([np.mean(acc[e], axis=0) for e in ref["col_elements"]])
    ux = {int(r[0]): r[3] for r in block("set MIDX")}
    uy = {int(r[0]): r[3] for r in block("set MIDY")}
    return dict(job=ref["job"], solver="calculix", ee_xx=m[:, 0].tolist(), ee_yy=m[:, 1].tolist(),
                u3_midx=[ux[n] for n in ref["midx_nodes"]], u3_midy=[uy[n] for n in ref["midy_nodes"]])


def read_ansys(job, ref):
    """CSV files written by the APDL input of ansys/denso_plate/gen_denso_plate_ansys.py."""
    def rows(name):
        return [[float(v) for v in l.split(",")] for l in open(f"{job}_ansys_{name}.csv") if l.strip()]
    col = {int(r[0]): r[1:] for r in rows("col")}
    ux = {int(r[0]): r[1] for r in rows("midx")}; uy = {int(r[0]): r[1] for r in rows("midy")}
    return dict(job=job, solver="ansys", ee_xx=[col[e][0] for e in ref["col_elements"]],
                ee_yy=[col[e][1] for e in ref["col_elements"]],
                u3_midx=[ux[n] for n in ref["midx_nodes"]], u3_midy=[uy[n] for n in ref["midy_nodes"]])


def kappa_fit(s, w, half):
    s = np.asarray(s); w = np.asarray(w); sel = s <= half / 2 + 1e-12
    c2 = np.polyfit(s[sel] ** 2, w[sel], 1)[0]
    return -2.0 * c2


def compare(ref, fe):
    rx, ry = np.array(ref["ref_exx"]), np.array(ref["ref_eyy"])
    fx, fy = np.array(fe["ee_xx"]), np.array(fe["ee_yy"])
    scale = max(np.abs(rx).max(), np.abs(ry).max())
    err = max(np.abs(fx - rx).max(), np.abs(fy - ry).max()) / scale
    half = ref["L_plate"] / 2
    kx = kappa_fit(ref["x_mid"], fe["u3_midx"], half)
    ky = kappa_fit(ref["y_mid"], fe["u3_midy"], half)
    s = lambda k: k * ref["L_plate"] ** 2 / 8 * 1e3
    sag_fe = np.array([s(kx), s(ky)]); sag_ref = np.array([s(k) for k in ref["ref_kappa"][:2]])
    tol = np.maximum(TOL_SAG_REL * np.abs(sag_ref), TOL_SAG_MM)
    ok = err < TOL_STRAIN and bool(np.all(np.abs(sag_fe - sag_ref) < tol))
    return dict(strain_err=float(err), sag_fe=sag_fe.tolist(), sag_ref=sag_ref.tolist(), ok=ok)


def figure(ref, fe, path):
    import matplotlib
    matplotlib.use("Agg")
    try:                                   # repo convention: thesis_style, graceful fallback
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "..", "slides", "figure_sources"))
        from thesis_style import use
        use(width_frac=0.6, aspect=0.8)
        if shutil.which("latex") is None:  # no TeX: Times-compatible text + STIX math
            matplotlib.rcParams["text.usetex"] = False
            matplotlib.rcParams["font.family"] = "serif"
            matplotlib.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif",
                                                 "STIXGeneral", "DejaVu Serif"]
            matplotlib.rcParams["mathtext.fontset"] = "stix"
    except Exception:
        pass
    import matplotlib.pyplot as plt
    z = np.array(ref["zm"])
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    for key, fk, lab, ls, mk in (("ref_exx", "ee_xx", r"$\varepsilon_{xx}$", "-", "o"),
                                 ("ref_eyy", "ee_yy", r"$\varepsilon_{yy}$", "--", "s")):
        ax.plot(np.array(ref[key]) * 1e3, z, "k" + ls, lw=1.2, label=lab + " lamination theory")
        ax.plot(np.array(fe[fk]) * 1e3, z, mk, ms=3.0, mfc="none", mec="0.35",
                label=lab + " solid FE (" + fe.get("solver", "FE") + ")")
    ax.set_xlabel(r"residual (elastic) strain $\varepsilon_{\mathrm{res}}$ [$10^{-3}$]")
    ax.set_ylabel(r"normalised thickness coordinate $2z/H$")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title(ref["title"].replace(" C,", " \u00b0C,") + " (illustrative constants)", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("jobs", nargs="*")
    ap.add_argument("--calculix", help="read this CalculiX .dat instead of <job>_abaqus.json")
    ap.add_argument("--ansys", action="store_true",
                    help="read <job>_ansys_{col,midx,midy}.csv (ansys/denso_plate) instead")
    ap.add_argument("--fig", action="store_true")
    a = ap.parse_args()
    pat = "_ansys_col.csv" if a.ansys else "_abaqus.json"
    jobs = a.jobs or sorted(p[:-len(pat)] for p in glob.glob("*" + pat))
    if not jobs:
        sys.exit("no <job>_abaqus.json found (run extract_denso_plate.py first)")
    nfail = 0
    for job in jobs:
        ref = json.load(open(job + "_ref.json"))
        if a.calculix:
            fe = read_calculix(a.calculix, ref)
        elif a.ansys:
            fe = read_ansys(job, ref)
        else:
            fe = json.load(open(job + "_abaqus.json"))
        r = compare(ref, fe)
        nfail += not r["ok"]
        print(f"{job:28s} [{fe.get('solver')}] strain err {r['strain_err']:.1e} | sag x "
              f"{r['sag_fe'][0]:+.4f} (ref {r['sag_ref'][0]:+.4f}) mm, y {r['sag_fe'][1]:+.4f} "
              f"(ref {r['sag_ref'][1]:+.4f}) mm | {'PASS' if r['ok'] else 'FAIL'}")
        if a.fig:
            figure(ref, fe, job + "_compare.png")
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
