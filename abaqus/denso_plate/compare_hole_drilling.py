"""compare_hole_drilling.py -- released strain against drilling depth from the hole-drilling
simulation (gen_hole_drilling.py), next to the MEASURED and the SIMULATED curves of Tsuda et al.
(2026) Fig. 12 (read by eye, data/literature/tsuda2026_jscm52_88.json). Plain python3.

    cd abaqus/denso_plate/jobs
    python3 ../compare_hole_drilling.py denso_hole                            # <job>_abaqus.json
    python3 ../compare_hole_drilling.py denso_hole_coarse --calculix denso_hole_coarse_ccx.dat

Released strain of a gauge after drilling step b = area-weighted mean over its elements of
(strain after step b - strain after the cool-down step). Ch.3 = e_xx (MD), Ch.1 = e_yy (TD),
Ch.2 = e along 45 deg = (e_xx + e_yy)/2 + gamma_xy/2. Reported: the curves, and the ratio of the
peak |released strain| to the measured and to the paper's simulated peak per gauge.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def read_calculix(path):
    """Strain blocks per step from a ccx .dat (*EL PRINT, ELSET=GAUGES, E): tensor shear."""
    L = open(path).read().splitlines(); steps = []; i = 0
    while i < len(L):
        if L[i].strip().startswith("strains (elem"):
            rows = {}; i += 1
            while i < len(L):
                t = L[i].split()
                if not t:
                    if rows:
                        break
                    i += 1; continue
                if not t[0].isdigit():
                    break
                rows.setdefault(int(t[0]), []).append([float(v) for v in t[2:8]]); i += 1
            steps.append({str(e): list(np.mean(v, axis=0)) for e, v in rows.items()})
        else:
            i += 1
    return dict(solver="calculix", shear="tensor", steps=[dict(name=f"step{k}", E=E) for k, E in enumerate(steps)])


def released(meta, res):
    steps = res["steps"]; E0 = steps[0]["E"]
    gamma = (lambda e: e[3]) if res["shear"] == "engineering" else (lambda e: 2 * e[3])
    comp = {"GCH3": lambda e: e[0], "GCH1": lambda e: e[1],
            "GCH2": lambda e: 0.5 * (e[0] + e[1]) + 0.5 * gamma(e)}
    out = {}
    for g, f in comp.items():
        els = [str(e) for e in meta["gauges"][g]["elements"]]; w = np.array(meta["gauges"][g]["quad_areas"])
        val = [0.0]
        for st in steps[1:]:
            d = [f(st["E"][e]) - f(E0[e]) for e in els]
            val.append(float(np.sum(w * np.array(d)) / np.sum(w)) * 1e6)
        out[g] = val
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("job")
    ap.add_argument("--calculix")
    ap.add_argument("--no-fig", action="store_true")
    a = ap.parse_args()
    meta = json.load(open(a.job + "_meta.json"))
    res = read_calculix(a.calculix) if a.calculix else json.load(open(a.job + "_abaqus.json"))
    rel = released(meta, res)
    f12 = meta["paper_fig12"]
    depth = [meta["band_mm"] * b for b in range(len(rel["GCH1"]))]
    print(f"{a.job} [{res['solver']}]: released strain [micro-strain] at depth [mm]; paper Fig. 12 read by eye")
    print(" depth   Ch1 this / exp / paper-FE     Ch2 this / exp / paper-FE     Ch3 this / exp / paper-FE")
    for k, d in enumerate(depth):
        row = []
        for ch, g in (("1", "GCH1"), ("2", "GCH2"), ("3", "GCH3")):
            e = np.interp(d, f12["depth_mm"], f12[f"Ch{ch}_exp"]); s = np.interp(d, f12["depth_mm"], f12[f"Ch{ch}_sim"])
            row.append(f"{rel[g][k]:7.0f} / {e:5.0f} / {s:5.0f}")
        print(f" {d:4.2f}   " + "    ".join(row))
    summ = {}
    for ch, g in (("1", "GCH1"), ("3", "GCH3")):
        mine = np.max(np.abs(rel[g])); ex = np.max(np.abs(f12[f"Ch{ch}_exp"])); si = np.max(np.abs(f12[f"Ch{ch}_sim"]))
        summ[g] = dict(peak=float(mine), peak_exp=float(ex), peak_paper_fe=float(si), ratio_to_exp=float(mine / ex),
                       ratio_to_paper_fe=float(mine / si), depth_of_peak_mm=float(depth[int(np.argmax(np.abs(rel[g])))]))
        print(f"Ch{ch}: peak |e| {mine:.0f} (at {summ[g]['depth_of_peak_mm']:.2f} mm) vs measured {ex:.0f} "
              f"(x{mine/ex:.2f}) and paper FE {si:.0f} (x{mine/si:.2f}); paper FE / measured = x{si/ex:.2f}")
    json.dump(dict(job=a.job, solver=res["solver"], depth_mm=depth, released=rel, summary=summ),
              open(a.job + "_released.json", "w"), indent=1)
    if not a.no_fig:
        _plot(a.job, depth, rel, f12, res["solver"])


def _plot(job, depth, rel, f12, solver):
    import matplotlib
    matplotlib.use("Agg")
    try:                                   # repo convention: thesis_style, graceful fallback
        sys.path.insert(0, os.path.join(ROOT, "slides", "figure_sources"))
        from thesis_style import use
        use(width_frac=0.6, aspect=0.75)
        if shutil.which("latex") is None:
            matplotlib.rcParams["text.usetex"] = False
    except Exception:
        pass
    matplotlib.rcParams.update({"text.usetex": False, "font.family": "serif",
                                "font.serif": ["Times New Roman", "Liberation Serif", "STIXGeneral"],
                                "mathtext.fontset": "stix"})
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    for ch, g, c in (("3", "GCH3", "k"), ("1", "GCH1", "0.5")):
        ax.plot(f12["depth_mm"], f12[f"Ch{ch}_exp"], "-o", color=c, ms=3, lw=1.0, label=f"Ch.{ch} measured (paper)")
        ax.plot(f12["depth_mm"], f12[f"Ch{ch}_sim"], ":s", color=c, ms=3, mfc="white", lw=1.0, label=f"Ch.{ch} FE (paper)")
        ax.plot(depth, rel[g], "--^", color=c, ms=4, lw=1.4, label=f"Ch.{ch} this work ({solver})")
    ax.axhline(0, color="0.85", lw=0.6)
    ax.set_xlabel("drilling depth [mm]"); ax.set_ylabel(r"released strain [$\mu\varepsilon$]")
    ax.set_title("hole drilling: paper inputs, paper Fig. 12 read by eye", fontsize=8)
    ax.legend(frameon=False, fontsize=6.5, ncol=2)
    fig.tight_layout(); fig.savefig(job + "_released.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
