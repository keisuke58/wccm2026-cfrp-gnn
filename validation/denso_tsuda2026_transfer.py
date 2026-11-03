"""validation/denso_tsuda2026_transfer.py — transfer of the physics prediction onto the MEASURED
hole-drilling data of Tsuda et al. (2026) (their Fig. 12, read by eye), in the linear-in-parameters
form that one measured plate allows.

Source ("pre-trained") model: the hole-drilling simulation with the paper's inputs
(abaqus/denso_plate/gen_hole_drilling.py). The released strain is linear in the eigenstrain, so it
splits EXACTLY into the responses to the xx, yy and zz eigenstrain components, h_xx + h_yy + h_zz
(three basis runs; the split is checked against the full run). Correction = component weights a:

    y_pred = a_xx h_xx + a_yy h_yy + a_zz h_zz,        prior a = (1, 1, 1)  (the physics as is)

fitted to the measured released strain y (Ch.1, Ch.2, Ch.3 at 10 depths) by L2-SP / ridge towards
the prior (Li et al., ICML 2018; here the closed-form Gaussian posterior):

    min_a  |H a - y|^2 / s_y^2 + |a - 1|^2 / s_a^2,    s_y = 10 micro-strain (reading uncertainty),
                                                        s_a = 0.5 (prior: +-50 % on each component).

Held-out checks on the single plate (nothing used in a fit is used in its score):
  H1 fit on depths 0.2-1.0 mm (all gauges), predict 1.2-2.0 mm;
  H2 fit on Ch.3 (MD) only, predict Ch.1 (TD) and Ch.2.
Reported: RMSE of the physics as is vs after the correction, and the plain least-squares fit (no
prior) for comparison. With a common factor a, the eigenstrain integral from T0 = 180 C scales as
a; the equivalent stress-free temperature T0_eff = 20 + a (180 - 20) is given only as a reading of
the result, next to the paper's own estimate (~120 C).

Honest scope: ONE measured plate, one moulding condition; three gauges; values read from a figure
(+-10 micro-strain); coarse-mesh CalculiX basis runs (the full mesh runs on fifa). The correction
has three parameters, so it shows the mechanics of calibrating a physics prior on point data, not
the neural-network transfer learning of the main study, which needs many conditions.

Run:  python3 validation/denso_tsuda2026_transfer.py --basis-dir <dir with denso_hole_coarse_{xx,yy,zz}_released.json>
      full mesh (fifa):  ... --basis-dir abaqus/denso_plate/abaqus_results --prefix denso_hole \
                             --full abaqus/denso_plate/abaqus_results/denso_hole_released.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "denso_tsuda2026")
S_Y, S_A = 10.0, 0.5
CH = ["GCH1", "GCH2", "GCH3"]


def load(basis_dir, full_json, prefix="denso_hole_coarse"):
    B = {c: json.load(open(os.path.join(basis_dir, f"{prefix}_{c}_released.json"))) for c in ("xx", "yy", "zz")}
    full = json.load(open(full_json))
    paper = json.load(open(os.path.join(ROOT, "data", "literature", "tsuda2026_jscm52_88.json"), encoding="utf-8"))
    f12 = paper["fig12_released_strain"]
    depth = np.array(full["depth_mm"])
    # stack observations: gauge-major, depths 0.2..2.0 (skip the zero at depth 0)
    idx = np.arange(1, len(depth))
    H = np.stack([np.concatenate([np.array(B[c]["released"][g])[idx] for g in CH]) for c in ("xx", "yy", "zz")], axis=1)
    full_vec = np.concatenate([np.array(full["released"][g])[idx] for g in CH])
    y = np.concatenate([np.interp(depth[idx], f12["depth_mm"], f12[f"Ch{g[-1]}_exp"]) for g in CH])
    yfe = np.concatenate([np.interp(depth[idx], f12["depth_mm"], f12[f"Ch{g[-1]}_sim"]) for g in CH])
    gauge = np.repeat(np.arange(3), len(idx)); dd = np.tile(depth[idx], 3)
    return H, full_vec, y, yfe, gauge, dd, depth


def fit(H, y, prior=True):
    a0 = np.ones(H.shape[1])
    if prior:
        A = H.T @ H / S_Y ** 2 + np.eye(len(a0)) / S_A ** 2
        b = H.T @ y / S_Y ** 2 + a0 / S_A ** 2
    else:
        A = H.T @ H / S_Y ** 2; b = H.T @ y / S_Y ** 2
    a = np.linalg.solve(A, b)
    return a, np.sqrt(np.diag(np.linalg.inv(A)))


rmse = lambda p, y: float(np.sqrt(np.mean((p - y) ** 2)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--basis-dir", required=True)
    ap.add_argument("--full", default=os.path.join(OUT, "hole_drilling_coarse_calculix.json"))
    ap.add_argument("--prefix", default="denso_hole_coarse", help="basis file prefix; full mesh from fifa: denso_hole")
    a_ = ap.parse_args()
    H, full_vec, y, yfe, gauge, dd, depth = load(a_.basis_dir, a_.full, a_.prefix)
    sup = float(np.max(np.abs(H.sum(1) - full_vec)) / np.max(np.abs(full_vec)))
    print(f"[check] superposition h_xx + h_yy + h_zz vs full run: {sup:.1e} (rel)")
    print(f"        column norms [micro-strain]: xx {np.linalg.norm(H[:,0]):.0f}, yy {np.linalg.norm(H[:,1]):.0f}, zz {np.linalg.norm(H[:,2]):.0f}")
    res = dict(superposition_err=sup, s_y=S_Y, s_a=S_A, checks={})
    p0 = H.sum(1)
    print(f"physics as is: RMSE vs measured {rmse(p0, y):.1f} micro-strain (paper FE {rmse(yfe, y):.1f})")
    for name, tr in (("H1 fit depth<=1.0, predict 1.2-2.0", dd <= 1.0 + 1e-9), ("H2 fit Ch.3 (MD), predict Ch.1/Ch.2", gauge == 2)):
        te = ~tr
        a_p, sd = fit(H[tr], y[tr], True); a_l, _ = fit(H[tr], y[tr], False)
        r = dict(a_l2sp=a_p.tolist(), a_sd=sd.tolist(), a_lsq=a_l.tolist(),
                 rmse_test_physics=rmse(p0[te], y[te]), rmse_test_l2sp=rmse(H[te] @ a_p, y[te]),
                 rmse_test_lsq=rmse(H[te] @ a_l, y[te]), rmse_test_paper_fe=rmse(yfe[te], y[te]),
                 n_fit=int(tr.sum()), n_test=int(te.sum()))
        res["checks"][name] = r
        print(f"[{name}] a(L2-SP) = {np.round(a_p,3)} +- {np.round(sd,3)}; a(LSQ) = {np.round(a_l,3)}")
        print(f"     test RMSE [micro-strain]: physics as is {r['rmse_test_physics']:.1f} | L2-SP {r['rmse_test_l2sp']:.1f} | "
              f"LSQ {r['rmse_test_lsq']:.1f} | paper FE {r['rmse_test_paper_fe']:.1f}  ({r['n_fit']} fit / {r['n_test']} test points)")
    a_all, sd_all = fit(H, y, True)
    common = float(np.linalg.lstsq(p0[:, None], y, rcond=None)[0][0])
    res.update(a_all=a_all.tolist(), a_all_sd=sd_all.tolist(), common_factor=common, T0_eff=20 + common * 160,
               rmse_all_after=rmse(H @ a_all, y), rmse_all_physics=rmse(p0, y))
    print(f"[all data] a(L2-SP) = {np.round(a_all,3)} +- {np.round(sd_all,3)}; RMSE {rmse(p0,y):.1f} -> {rmse(H @ a_all, y):.1f}")
    print(f"           single common factor {common:.3f} -> equivalent T0_eff = {20 + common*160:.0f} C (paper's estimate ~120 C)")
    json.dump(res, open(os.path.join(OUT, "transfer_hole_drilling.json"), "w"), indent=1)
    _plot(H, y, yfe, p0, depth, dd, gauge, res)
    print(f"wrote {OUT}/transfer_hole_drilling.json and .png")


def _plot(H, y, yfe, p0, depth, dd, gauge, res):
    sys.path.insert(0, os.path.join(ROOT, "slides", "figure_sources"))
    try:                                   # repo convention: thesis_style, graceful fallback
        from thesis_style import use
        use(width_frac=1.0, aspect=0.42)
    except Exception:
        pass
    matplotlib.rcParams.update({"text.usetex": False, "font.family": "serif",
                                "font.serif": ["Times New Roman", "Liberation Serif", "STIXGeneral"], "mathtext.fontset": "stix"})
    import matplotlib.pyplot as plt
    a1 = np.array(res["checks"]["H1 fit depth<=1.0, predict 1.2-2.0"]["a_l2sp"])
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for k, (g, c, lab) in enumerate(((2, "k", "Ch.3 (MD)"), (0, "0.5", "Ch.1 (TD)"))):
        m = gauge == g; d = np.concatenate([[0], dd[m]])
        f = lambda v: np.concatenate([[0], v[m]])
        ax[0].plot(d, f(y), "-o", color=c, ms=3, lw=1.0, label=f"{lab} measured" if k == 0 else None)
        ax[0].plot(d, f(p0), ":", color=c, lw=1.2, label="physics as is" if k == 0 else None)
        ax[0].plot(d, f(H @ a1), "--^", color=c, ms=3.5, lw=1.3, label="corrected (fit on depth <= 1.0 mm)" if k == 0 else None)
    ax[0].axvspan(1.1, 2.05, color="0.93", zorder=0); ax[0].text(1.55, 330, "held out", ha="center", fontsize=8)
    ax[0].axhline(0, color="0.85", lw=0.6); ax[0].set_xlim(0, 2.05); ax[0].set_ylim(-330, 360)
    ax[0].set_xlabel("drilling depth [mm]"); ax[0].set_ylabel(r"released strain [$\mu\varepsilon$]")
    ax[0].legend(frameon=False, fontsize=7, loc="center right")
    ax[0].set_title("(a) correction fitted on the shallow half, predicted on the deep half", fontsize=8)
    names = ["physics\nas is", "paper FE", "LSQ", "L2-SP"]
    for j, (key, lab) in enumerate((("H1 fit depth<=1.0, predict 1.2-2.0", "H1: shallow -> deep"), ("H2 fit Ch.3 (MD), predict Ch.1/Ch.2", "H2: MD -> TD, 45 deg"))):
        r = res["checks"][key]
        v = [r["rmse_test_physics"], r["rmse_test_paper_fe"], r["rmse_test_lsq"], r["rmse_test_l2sp"]]
        ax[1].bar(np.arange(4) + 0.38 * j - 0.19, v, width=0.36, color=["0.35", "0.75"][j], label=lab)
    ax[1].set_xticks(range(4)); ax[1].set_xticklabels(names, fontsize=8)
    ax[1].set_ylabel(r"held-out RMSE [$\mu\varepsilon$]"); ax[1].legend(frameon=False, fontsize=7)
    ax[1].set_title("(b) error on measurements not used in the fit", fontsize=8)
    fig.suptitle("Calibrating the physics prediction on Tsuda et al. (2026) Fig. 12 (read by eye); one plate, coarse mesh", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "transfer_hole_drilling.png"), dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
