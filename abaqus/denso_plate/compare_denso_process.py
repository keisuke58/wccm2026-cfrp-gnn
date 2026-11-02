"""compare_denso_process.py -- check the Abaqus PROCESS analysis against process_reference.py,
heat and stress separately. Plain python3 + numpy (+ matplotlib for --fig).

    cd abaqus/denso_plate/jobs
    python3 ../compare_denso_process.py asymmetric --fig          # uses denso_proc_<case>_abaqus.json
    python3 ../compare_denso_process.py asymmetric --calculix-heat denso_heat_asymmetric_ccx.dat

  (H) heat: centre-column temperature history of Abaqus vs the 1-D FE reference (lumped and
      consistent capacity both shown; the better one is used for the verdict);
  (M) stress: the Python reference is RERUN with the Abaqus temperatures, so that only the
      UMAT + equilibrium are compared: sigma_xx, sigma_yy per layer, alpha per layer at the end
      of MOULD and of AIR, sag over the full plate from the mid-plane U3.
Pass: (H) max |dT| < 0.5 K for t >= 1 s (the first increments after the melt meets the 170-200 C
walls depend on how the element capacity is integrated: CalculiX differs from the 1-D reference by
up to 7 K at the wall-adjacent node at t = 0.05 s, 0.3 K at 1 s; the full-history maximum is
printed as well); (M) stress error < 2e-2 of max |sigma|, alpha error < 1e-2,
|d sag| < max(2e-2 |sag|, 0.01 mm).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import process_reference as pr  # noqa: E402
from denso_plate_features import orientation_profile  # noqa: E402
from compare_denso_plate import kappa_fit  # noqa: E402

T_SKIP = 1.0                      # s; start of the heat verdict window
TOL_T, TOL_S, TOL_A, TOL_SAG_REL, TOL_SAG_MM = 0.5, 2e-2, 1e-2, 2e-2, 0.01


def read_ccx_heat(path, ref):
    L = open(path).read().splitlines()
    times, T = [], []
    i = 0
    while i < len(L):
        if "temperatures for set NCOL" in L[i]:
            t = float(L[i].split()[-1]); vals = {}
            i += 1
            while i < len(L) and (not L[i].strip() or L[i].split()[0].isdigit()):
                p = L[i].split()
                if p:
                    vals[int(p[0])] = float(p[1])
                elif vals:
                    break
                i += 1
            times.append(t); T.append([vals[n] for n in ref["col_nodes"]])
        else:
            i += 1
    return np.array(times), np.array(T)


def heat_check(ref, times, T):
    """max |dT| for t >= T_SKIP per capacity form, and the full-history maximum."""
    out = {}
    times = np.asarray(times); T = np.asarray(T); late = times >= T_SKIP
    for lumped in (True, False):
        tp, Tp = pr.heat_1d(ref["T_bottom"], ref["T_top"], lumped=lumped)
        Ti = np.array([np.interp(times, tp, Tp[:, k]) for k in range(Tp.shape[1])]).T
        d = np.abs(Ti - T)
        key = "lumped" if lumped else "consistent"
        out[key] = float(d[late].max()) if late.any() else float("nan")
        out[key + "_all"] = float(d.max())
    return out


def stress_check(ref, ab):
    P, _ = pr.layer_props(orientation_profile)
    times = np.array(ab["times"]); T = np.array(ab["T_col"])
    r = pr.plate_history(P, times, T, record=True)
    sx, sy = r["sig_layer"][:, 0], r["sig_layer"][:, 1]
    scale = max(np.abs(sx).max(), np.abs(sy).max())
    es = max(np.abs(np.array(ab["s11"]) - sx).max(), np.abs(np.array(ab["s22"]) - sy).max()) / scale
    a_m = r["steps"]["MOULD"]["alpha"].reshape(pr.NL, 2).mean(1)
    ea = max(np.abs(np.array(ab["sdv1_mould"]) - a_m).max(),
             np.abs(np.array(ab["sdv1_end"]) - r["alpha_layer"]).max())
    half = ref["L_plate"] / 2
    sag = lambda k: k * ref["L_plate"] ** 2 / 8 * 1e3
    sag_ab = [sag(kappa_fit(ref["xs"], ab["u3_midx"], half)), sag(kappa_fit(ref["ys"], ab["u3_midy"], half))]
    sag_py = [sag(r["kappa"][0]), sag(r["kappa"][1])]
    tol = [max(TOL_SAG_REL * abs(s), TOL_SAG_MM) for s in sag_py]
    ok = es < TOL_S and ea < TOL_A and all(abs(a - b) < t for a, b, t in zip(sag_ab, sag_py, tol))
    return dict(stress_err=float(es), alpha_err=float(ea), sag_abaqus=sag_ab, sag_python=sag_py,
                s33_max=float(np.max(np.abs(ab["s33"]))), ok=ok, r=r)


def figure(ref, ab, r, path):
    import matplotlib
    matplotlib.use("Agg")
    try:                                   # repo convention: thesis_style, graceful fallback
        sys.path.insert(0, os.path.join(HERE, "..", "..", "slides", "figure_sources"))
        from thesis_style import use
        use(width_frac=1.0, aspect=0.45)
        if shutil.which("latex") is None:
            matplotlib.rcParams["text.usetex"] = False
            matplotlib.rcParams["font.family"] = "serif"
            matplotlib.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "STIXGeneral"]
            matplotlib.rcParams["mathtext.fontset"] = "stix"
    except Exception:
        pass
    import matplotlib.pyplot as plt
    z = np.array(ref["zm"])
    fig, ax = plt.subplots(1, 2, figsize=(8.0, 3.6), sharey=True)
    ax[0].plot(r["sig_layer"][:, 0] / 1e6, z, "k-", lw=1.2, label="Python reference")
    ax[0].plot(np.array(ab["s11"]) / 1e6, z, "o", ms=3, mfc="none", mec="0.35", label="Abaqus")
    ax[0].set_xlabel(r"residual stress $\sigma_{xx}$ [MPa]")
    ax[0].set_ylabel(r"normalised thickness coordinate $2z/H$")
    ax[1].plot(r["alpha_layer"], z, "k-", lw=1.2)
    ax[1].plot(ab["sdv1_end"], z, "o", ms=3, mfc="none", mec="0.35")
    ax[1].set_xlabel(r"relative crystallinity $\alpha$ [$-$]")
    ax[0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"process analysis, {ref['case']} (illustrative constants)", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cases", nargs="*")
    ap.add_argument("--calculix-heat", help="check only the heat history from a CalculiX .dat")
    ap.add_argument("--fig", action="store_true")
    a = ap.parse_args()
    cases = a.cases or [l.strip() for l in open("manifest_process.txt") if l.strip()]
    nfail = 0
    for case in cases:
        ref = json.load(open(f"denso_proc_{case}_ref.json"))
        if a.calculix_heat:
            t, T = read_ccx_heat(a.calculix_heat, ref)
            h = heat_check(ref, t, T)
            ok = min(h["lumped"], h["consistent"]) < TOL_T
            print(f"{case:10s} [calculix] (H) {len(t)} increments to t = {t[-1]:.1f} s, max |dT| (t >= "
                  f"{T_SKIP:.0f} s) lumped {h['lumped']:.3f} K, consistent {h['consistent']:.3f} K; whole "
                  f"history {h['lumped_all']:.2f} / {h['consistent_all']:.2f} K | {'PASS' if ok else 'FAIL'}")
            nfail += not ok
            continue
        ab = json.load(open(f"denso_proc_{case}_abaqus.json"))
        h = heat_check(ref, np.array(ab["times"]), np.array(ab["T_col"]))
        m = stress_check(ref, ab)
        okh = min(h["lumped"], h["consistent"]) < TOL_T
        print(f"{case:10s} (H) max |dT| (t >= {T_SKIP:.0f} s) lumped {h['lumped']:.3f} K, consistent "
              f"{h['consistent']:.3f} K; whole history {h['lumped_all']:.2f} / {h['consistent_all']:.2f} K "
              f"| {'PASS' if okh else 'FAIL'}")
        print(f"{'':10s} (M) stress err {m['stress_err']:.1e}, alpha err {m['alpha_err']:.1e}, sag x "
              f"{m['sag_abaqus'][0]:+.3f} (py {m['sag_python'][0]:+.3f}) mm, y {m['sag_abaqus'][1]:+.3f} "
              f"(py {m['sag_python'][1]:+.3f}) mm, max|S33| {m['s33_max']/1e6:.2f} MPa "
              f"| {'PASS' if m['ok'] else 'FAIL'}")
        nfail += (not okh) + (not m["ok"])
        if a.fig:
            figure(ref, ab, m["r"], f"denso_proc_{case}_compare.png")
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
