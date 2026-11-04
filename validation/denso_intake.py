"""validation/denso_intake.py -- intake of DENSO data: check the files, then run the calibration chain in one go, so that
results exist on the day the data arrive. File formats: data/denso_templates/*.csv, described in
research/DATA_SPEC_denso.md.

Steps per specimen:
  V  checks: required columns, numbers, ranges (|strain| < 5000 micro-strain, 0 <= a_ij <= 1, a_xx + a_yy <= 1.02,
     thickness 0.5-10 mm), depth increasing, specimen ids consistent across files. Problems are listed, not fixed.
  H  hole drilling (if given): prediction through the influence operator of fifa §16 (16 bands of 0.2 mm, 3 gauges,
     10 steps; validated 0.8 % on the paper's plate) with the specimen's orientation profile (or the paper's Fig. 8a
     if none, with a warning):
       y(T0) = G_xx ebar_xx(T0) + G_yy ebar_yy(T0),  ebar = band average of int_{T0}^{20 C} alpha(T) dT          (1)
     H1 the stress-free temperature T0_eff that minimises the RMSE to the measurement (golden section, 80-260 C);
     H2 one common factor a at T0 = 180 C (the plate-paper calibration);
     H3 the transfer check: T0_eff from the MD gauge only, prediction of TD and 45 deg.
  W  warpage (if given): least-squares quadratic surface w = c0 + c1 x + c2 y + (kx x^2 + ky y^2 + kxy xy) / 2 per
     specimen; measured curvatures compared with the laminate prediction (free plate, validation/
     denso_tsuda2026_compare.residual_stress) at T0 = 180 C and at the specimen's T0_eff from H1.
Outputs: <out>/report.json, <out>/report.md, <out>/hole_drilling.png.

Honest scope: the influence operator G holds the paper's plate (3.08 mm), gauge (FRAS-2, hole 1.5 mm) and stiffness;
for another thickness, gauge or material it is an approximation and a new operator must be computed on fifa (§16)
-- the script warns. Elastic chain, relaxation only through T0_eff. Warp: the free-plate laminate prediction ignores
the in-plane shape of the part. No result here is a DENSO result until real files are given.
Run:  python3 validation/denso_intake.py --dir DATA_DIR --out results/denso_intake/RUN
      python3 validation/denso_intake.py --selftest      (synthetic files from the paper's values; must give
                                                        T0_eff close to the 143 C of validation/denso_rescue_released.py)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import denso_tsuda2026_compare as tc  # noqa: E402

INF = json.load(open(os.path.join(ROOT, "abaqus", "denso_plate", "abaqus_results", "influence", "influence_operator.json")))
G = {c: np.array(INF["G"][c])[:, 1:, :] for c in ("xx", "yy")}        # (3 gauges GCH1, GCH2, GCH3; 10 steps; 16 bands)
STEPS = np.arange(1, 11) * 0.2
NB, H_OP = 16, 3.08
PAPER = json.load(open(os.path.join(ROOT, "data", "literature", "tsuda2026_jscm52_88.json"), encoding="utf-8"))
COLS = {"hole_drilling.csv": ["specimen", "depth_mm", "ch1_td", "ch2_45", "ch3_md"],
        "orientation_profile.csv": ["specimen", "depth_mm", "a_xx", "a_yy"],
        "warpage_points.csv": ["specimen", "x_mm", "y_mm", "w_mm"],
        "specimens.csv": ["specimen", "thickness_mm"]}


# ------------------------------------------------------------------------------------------- reading / checks
def read(path):
    with open(path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(line for line in f if line.strip() and not line.lstrip().startswith("#"))]
    return rows


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def check(d):
    files, problems = {}, []
    for name, req in COLS.items():
        p = os.path.join(d, name)
        if not os.path.exists(p):
            continue
        rows = read(p)
        miss = [c for c in req if rows and c not in rows[0]]
        if miss:
            problems.append(f"{name}: missing columns {miss}")
            continue
        for i, r in enumerate(rows):
            for c in req[1:]:
                if num(r[c]) is None:
                    problems.append(f"{name} row {i + 2}: '{c}' is not a number ({r[c]!r})")
        files[name] = [r for r in rows if r.get("specimen", "").upper() != "EXAMPLE"] or rows
    hd = files.get("hole_drilling.csv", [])
    for r in hd:
        for c in ("ch1_td", "ch2_45", "ch3_md"):
            v = num(r[c])
            if v is not None and abs(v) > 5000:
                problems.append(f"hole_drilling: specimen {r['specimen']} depth {r['depth_mm']}: {c} = {v} (> 5000 micro-strain)")
    for r in files.get("orientation_profile.csv", []):
        ax, ay = num(r["a_xx"]), num(r["a_yy"])
        if ax is not None and ay is not None and not (0 <= ax <= 1 and 0 <= ay <= 1 and ax + ay <= 1.02):
            problems.append(f"orientation: specimen {r['specimen']} depth {r['depth_mm']}: a_xx {ax}, a_yy {ay} out of range")
    for r in files.get("specimens.csv", []):
        t = num(r["thickness_mm"])
        if t is not None and not 0.5 <= t <= 10:
            problems.append(f"specimens: {r['specimen']} thickness {t} mm out of 0.5-10")
    for name in ("hole_drilling.csv", "orientation_profile.csv"):
        by = {}
        for r in files.get(name, []):
            by.setdefault(r["specimen"], []).append(num(r["depth_mm"]))
        for s, dd in by.items():
            if any(b <= a for a, b in zip(dd, dd[1:]) if a is not None and b is not None):
                problems.append(f"{name}: specimen {s}: depth not increasing")
    known = {r["specimen"] for r in files.get("specimens.csv", [])}
    for name in ("hole_drilling.csv", "orientation_profile.csv", "warpage_points.csv"):
        for s in {r["specimen"] for r in files.get(name, [])} - known:
            if known:
                problems.append(f"{name}: specimen {s} not in specimens.csv")
    return files, problems


# ------------------------------------------------------------------------------------------------- physics
def profile_of(files, spec, H):
    rows = [r for r in files.get("orientation_profile.csv", []) if r["specimen"] == spec]
    if rows:
        d = np.array([num(r["depth_mm"]) for r in rows]); ax = np.array([num(r["a_xx"]) for r in rows])
        ay = np.array([num(r["a_yy"]) for r in rows]); return d, ax, ay, "given"
    f8 = PAPER["fig8_orientation_3mm"]
    return np.array(f8["depth_mm"]) * H / H_OP, np.array(f8["a_xx"]), np.array(f8["a_yy"]), "paper Fig. 8a (none given)"


def eigenstrain(prof, H, T0, nz=155, nT=33):
    d, axx, ayy = prof[:3]
    zq = np.linspace(0.0, H, nz)
    ax = np.interp(zq, d, axx); ay = np.interp(zq, d, ayy)
    a1, a2, a3 = tc.layers((zq - H / 2) * 1e-3, ax, ay)
    Ts = np.linspace(T0, tc.T_RT, nT); eth = np.zeros((nz, 3))
    for T1, T2 in zip(Ts[:-1], Ts[1:]):
        _, al = tc.layer_props(0.5 * (T1 + T2), a1, a2, a3); eth += al * (T2 - T1)
    return zq, eth


def released(prof, H, T0):
    zq, eth = eigenstrain(prof, H, T0)
    band = np.minimum((zq // 0.2).astype(int), NB - 1)
    ex = np.array([eth[band == b, 0].mean() if np.any(band == b) else 0.0 for b in range(NB)])
    ey = np.array([eth[band == b, 1].mean() if np.any(band == b) else 0.0 for b in range(NB)])
    return G["xx"] @ (ex / 1e-3) + G["yy"] @ (ey / 1e-3)              # (3, 10): Ch1 TD, Ch2 45, Ch3 MD


def golden(f, lo, hi, n=40):
    g = (np.sqrt(5) - 1) / 2; a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a); fc, fd = f(c), f(d)
    for _ in range(n):
        if fc < fd:
            b, d, fd = d, c, fc; c = b - g * (b - a); fc = f(c)
        else:
            a, c, fc = c, d, fd; d = a + g * (b - a); fd = f(d)
    return 0.5 * (a + b)


def hole_drilling(files, spec, H):
    rows = [r for r in files["hole_drilling.csv"] if r["specimen"] == spec]
    dep = np.array([num(r["depth_mm"]) for r in rows])
    Y = np.array([[num(r[c]) for r in rows] for c in ("ch1_td", "ch2_45", "ch3_md")])     # (3, n)
    ok = (STEPS >= dep.min() - 1e-9) & (STEPS <= dep.max() + 1e-9)
    y = np.array([np.interp(STEPS[ok], dep, Y[k]) for k in range(3)])                    # (3, m)
    prof = profile_of(files, spec, H)
    pred = lambda T0: released(prof, H, T0)[:, ok]
    rmse = lambda p, k=slice(None): float(np.sqrt(np.mean((p[k] - y[k]) ** 2)))
    p180 = pred(180.0)
    T0 = golden(lambda t: rmse(pred(t)), 80.0, 260.0)
    a = float(np.sum(p180 * y) / np.sum(p180 * p180))
    T0md = golden(lambda t: rmse(pred(t), [2]), 80.0, 260.0)
    pT, pmd = pred(T0), pred(T0md)
    r = dict(orientation=prof[3], depths_used_mm=STEPS[ok].tolist(),
             H1=dict(T0_eff=T0, rmse_T180=rmse(p180), rmse_T0eff=rmse(pT)),
             H2=dict(factor=a, rmse=rmse(a * p180)),
             H3=dict(T0_from_MD=T0md, rmse_TD_45_before=rmse(p180, [0, 1]), rmse_TD_45_after=rmse(pmd, [0, 1])),
             measured=y.tolist(), pred_T180=p180.tolist(), pred_T0eff=pT.tolist())
    if abs(H - H_OP) > 0.1:
        r["warning"] = f"thickness {H} mm differs from the operator's {H_OP} mm: approximate, recompute G on fifa (§16)"
    return r


def warpage(files, spec, H, T0s):
    rows = [r for r in files["warpage_points.csv"] if r["specimen"] == spec]
    x = np.array([num(r["x_mm"]) for r in rows]); y = np.array([num(r["y_mm"]) for r in rows]); w = np.array([num(r["w_mm"]) for r in rows])
    M = np.column_stack([np.ones_like(x), x, y, x * x / 2, y * y / 2, x * y / 2])
    c, *_ = np.linalg.lstsq(M, w, rcond=None)
    res = float(np.sqrt(np.mean((M @ c - w) ** 2)))
    out = dict(n_points=len(w), kx_meas_per_m=c[3] * 1e3, ky_meas_per_m=c[4] * 1e3, fit_rms_mm=res, pred={})
    prof = profile_of(files, spec, H)
    z = np.linspace(-H / 2, H / 2, 155) * 1e-3
    ax = np.interp(z * 1e3 + H / 2, prof[0], prof[1]); ay = np.interp(z * 1e3 + H / 2, prof[0], prof[2])
    a1, a2, a3 = tc.layers(z, ax, ay)
    for name, T0 in T0s.items():
        tc.T0 = T0
        _, kap = tc.residual_stress(z, a1, a2, a3, "total")
        out["pred"][name] = dict(T0=T0, kx_per_m=float(kap[0]), ky_per_m=float(kap[1]))
    tc.T0 = 180.0
    return out


# ------------------------------------------------------------------------------------------------- report
def run(d, out):
    os.makedirs(out, exist_ok=True)
    files, problems = check(d)
    specs = sorted({r["specimen"] for name in files for r in files[name]})
    thick = {r["specimen"]: num(r["thickness_mm"]) for r in files.get("specimens.csv", [])}
    rep = dict(input_dir=os.path.abspath(d), files=sorted(files), problems=problems, specimens={})
    for s in specs:
        H = thick.get(s) or H_OP
        r = {}
        if any(x["specimen"] == s for x in files.get("hole_drilling.csv", [])):
            r["hole_drilling"] = hole_drilling(files, s, H)
        if any(x["specimen"] == s for x in files.get("warpage_points.csv", [])):
            T0s = {"T0 = 180 C": 180.0}
            if "hole_drilling" in r:
                T0s["T0_eff from hole drilling"] = r["hole_drilling"]["H1"]["T0_eff"]
            r["warpage"] = warpage(files, s, H, T0s)
        rep["specimens"][s] = r
    json.dump(rep, open(os.path.join(out, "report.json"), "w"), indent=1)
    lines = ["# DENSO data intake report", "", f"input: `{rep['input_dir']}`; files: {', '.join(rep['files'])}", "",
             "## Checks", ""] + ([f"- {p}" for p in problems] or ["- no problems found"]) + ["", "## Results", ""]
    for s, r in rep["specimens"].items():
        lines.append(f"### {s}")
        if "hole_drilling" in r:
            h = r["hole_drilling"]
            lines += [f"- orientation: {h['orientation']}" + (f"; WARNING: {h['warning']}" if "warning" in h else ""),
                      f"- H1 T0_eff {h['H1']['T0_eff']:.0f} C; RMSE {h['H1']['rmse_T180']:.1f} -> {h['H1']['rmse_T0eff']:.1f} micro-strain",
                      f"- H2 common factor {h['H2']['factor']:.2f} at T0 = 180 C; RMSE {h['H2']['rmse']:.1f}",
                      f"- H3 MD only: T0 {h['H3']['T0_from_MD']:.0f} C; TD/45 RMSE {h['H3']['rmse_TD_45_before']:.1f} -> {h['H3']['rmse_TD_45_after']:.1f}"]
        if "warpage" in r:
            w = r["warpage"]
            lines.append(f"- warpage: {w['n_points']} points, measured curvature kx {w['kx_meas_per_m']:+.3f}, ky {w['ky_meas_per_m']:+.3f} 1/m "
                         f"(surface fit rms {w['fit_rms_mm']:.3f} mm)")
            for k, p in w["pred"].items():
                lines.append(f"  - predicted ({k}, {p['T0']:.0f} C): kx {p['kx_per_m']:+.3f}, ky {p['ky_per_m']:+.3f} 1/m")
        lines.append("")
    open(os.path.join(out, "report.md"), "w").write("\n".join(lines))
    plot(rep, out)
    print("\n".join(lines))
    return rep


def plot(rep, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.family": ["Times New Roman", "Liberation Serif", "IPAPMincho"], "font.size": 9})
    hs = [(s, r["hole_drilling"]) for s, r in rep["specimens"].items() if "hole_drilling" in r]
    if not hs:
        return
    fig, axs = plt.subplots(1, len(hs), figsize=(4.2 * len(hs), 3.2), squeeze=False)
    for a, (s, h) in zip(axs[0], hs):
        d = np.array(h["depths_used_mm"])
        for k, lab in ((2, "MD"), (0, "TD"), (1, "45°")):
            a.plot(d, h["measured"][k], "s", ms=4, mfc="white", color="k")
            a.plot(d, h["pred_T180"][k], "--", color="0.55", lw=1.2)
            a.plot(d, h["pred_T0eff"][k], "-", color="#1F3A5F", lw=1.8)
            a.text(d[-1] + 0.03, h["pred_T0eff"][k][-1], lab, fontsize=8, va="center")
        a.set_title(f"{s}：T0,eff {h['H1']['T0_eff']:.0f} °C", fontsize=10)
        a.set_xlabel("穴の深さ [mm]"); a.set_ylabel("解放ひずみ [µε]")
    fig.text(0.5, 0.005, "□ 実測　破線 T0 = 180 °C　実線 T0,eff（当てはめ）", ha="center", fontsize=8, color="0.35")
    fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(os.path.join(out, "hole_drilling.png"), dpi=200); plt.close(fig)


def selftest(tmp):
    """Synthetic files from the paper (Fig. 12 measured, Fig. 8a orientation) -> must reproduce T0_eff ~ 143 C."""
    os.makedirs(tmp, exist_ok=True)
    f12, f8 = PAPER["fig12_released_strain"], PAPER["fig8_orientation_3mm"]
    with open(os.path.join(tmp, "specimens.csv"), "w") as f:
        f.write("specimen,thickness_mm,mould_T_C,melt_T_C,annealed,anneal_T_C,note\nPAPER,3.08,,,1,180,Tsuda et al. 2026 read by eye\n")
    with open(os.path.join(tmp, "hole_drilling.csv"), "w") as f:
        f.write("specimen,depth_mm,ch1_td,ch2_45,ch3_md\n")
        for i, dd in enumerate(f12["depth_mm"]):
            if dd > 0:
                f.write(f"PAPER,{dd},{f12['Ch1_exp'][i]},{f12['Ch2_exp'][i]},{f12['Ch3_exp'][i]}\n")
    with open(os.path.join(tmp, "orientation_profile.csv"), "w") as f:
        f.write("specimen,depth_mm,a_xx,a_yy,a_zz\n")
        for dd, ax, ay in zip(f8["depth_mm"], f8["a_xx"], f8["a_yy"]):
            f.write(f"PAPER,{dd},{ax},{ay},\n")
    with open(os.path.join(tmp, "warpage_points.csv"), "w") as f:          # synthetic plate bent to the laminate curvature
        f.write("specimen,x_mm,y_mm,w_mm\n")
        z = np.linspace(-H_OP / 2, H_OP / 2, 155) * 1e-3
        ax = np.interp(z * 1e3 + H_OP / 2, f8["depth_mm"], f8["a_xx"]); ay = np.interp(z * 1e3 + H_OP / 2, f8["depth_mm"], f8["a_yy"])
        tc.T0 = 180.0; _, kap = tc.residual_stress(z, *tc.layers(z, ax, ay), "total")
        rng = np.random.default_rng(0)
        for x in np.linspace(-10, 10, 5):
            for y in np.linspace(-10, 10, 5):
                w = 0.5 * (kap[0] * x * x + kap[1] * y * y) * 1e-3 + rng.normal(0, 1e-4)
                f.write(f"PAPER,{x},{y},{w:.6f}\n")
    rep = run(tmp, os.path.join(tmp, "out"))
    h = rep["specimens"]["PAPER"]["hole_drilling"]
    ref = json.load(open(os.path.join(ROOT, "results", "denso_rescue", "results.json")))["R4"]["T0 only"]["theta"][0]
    w = rep["specimens"]["PAPER"]["warpage"]
    k180 = w["pred"]["T0 = 180 C"]
    print(f"[S1] T0_eff from the intake {h['H1']['T0_eff']:.1f} C vs validation/denso_rescue_released.py {ref:.1f} C "
          f"(difference {h['H1']['T0_eff'] - ref:+.1f} C; expected within ~2 C, the surrogate fit vs direct physics)")
    print(f"[S2] warpage curvature from a synthetic plate made at T0 = 180 C: measured kx {w['kx_meas_per_m']:+.4f} vs "
          f"predicted {k180['kx_per_m']:+.4f} 1/m, ky {w['ky_meas_per_m']:+.4f} vs {k180['ky_per_m']:+.4f} 1/m")
    print(f"[S3] problems found in the synthetic files: {len(rep['problems'])}")
    return h["H1"]["T0_eff"], ref


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir"); ap.add_argument("--out", default=os.path.join(ROOT, "results", "denso_intake", "run"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        t, ref = selftest(os.path.join(ROOT, "results", "denso_intake", "selftest"))
        json.dump(dict(T0_eff_intake=t, T0_eff_reference=ref), open(os.path.join(ROOT, "results", "denso_intake", "selftest.json"), "w"), indent=1)
        return
    run(a.dir, a.out)


if __name__ == "__main__":
    main()
