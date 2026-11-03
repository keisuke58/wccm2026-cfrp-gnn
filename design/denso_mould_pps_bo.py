"""design/denso_mould_pps_bo.py -- stage 3 with the GF/PPS process model: choose the two mould-wall
temperatures (T_bottom, T_top) in [60, 160] C that minimise the sag of the as-moulded plate, subject
to full crystallisation (minimum relative crystallinity >= 0.9 through the thickness: an amorphous
skin would crystallise later in service and change the shape). Bayesian optimisation (2-D GP +
expected improvement, design/denso_mould_temperature_bo.py) over the process model of
abaqus/denso_plate/process_pps.py (heat -> Nakamura/HL -> viscoelastic UMAT port, PPS constants of
Tsuda et al. 2026 where available); every proposal is evaluated by the physics.

Objective: |sag_y| over 100 mm + 10 mm x max(0, 0.9 - min alpha)   (penalty for the constraint).
Validated (printed): the BO optimum against a brute-force grid of the same model.
Honest scope: as process_pps.py -- viscoelastic constants, crystallisation shrinkage and heat
diffusivity are ILLUSTRATIVE; neat-PPS kinetics; the orientation is the paper's read profile and is
NOT a design variable (it would need a flow simulation); flat plate, mould time fixed at 40 s.

Result (2026-11-03): only 70 of 441 grid points crystallise fully (lowest symmetric mould 125 C).
Within them the sag stays at 2.83 .. 2.85 mm (grid best 2.830 mm at 150 / 120 C; BO 2.854 mm, the
landscape is flat): with full crystallisation required, the mould temperatures cannot remove the warp,
which is set by the up/down asymmetry of the fibre orientation. Below Tg the mould temperatures do
change the sag strongly (e.g. 80/80 C -0.59 mm), but only with an amorphous skin. A negative result:
the design variable for this plate has to act on the orientation (gate, flow), or on a post-mould
anneal, not on the mould temperatures alone.

Run:  python3 design/denso_mould_pps_bo.py   (writes results/denso_mould_pps_bo/)
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "abaqus", "denso_plate")); sys.path.insert(0, HERE)
import process_pps as pp  # noqa: E402
import process_reference as pr  # noqa: E402
from denso_mould_temperature_bo import GP2, ei  # noqa: E402

LO, HI = 60.0, 160.0
A_MIN = 0.9
OUT = os.path.join(ROOT, "results", "denso_mould_pps_bo")


def setup():
    ud = pp.ud_constants(); aL, aT = pp.secant_cte(); hl = pp.fit_hl()
    CP = replace(pr.C, UD=ud, A_L=aL, A_T=aT, H=pp.H_PPS, T_MELT=320.0)
    pr.C, pr.H = CP, pp.H_PPS; pr.K_COND = CP.DIFF * pr.RHO * pr.CP
    pr.HL = {k: hl[k] for k in ("K0", "USTAR_R", "KG", "TM0", "TINF")}; pr.N_AVRAMI = 3.0
    pr.VE = dict(pr.VE, WLF_TREF=90.0)
    P, _ = pr.layer_props(pp.orientation_pps(), c=CP)
    return P


def physics(P, Tb, Tt):
    t, Tn = pr.heat_1d(Tb, Tt)
    r = pr.plate_history(P, t, Tn)
    sag = float(r["kappa"][1] * 0.10 ** 2 / 8 * 1e3)
    amin = float(r["alpha_layer"].min())
    return sag, amin, abs(sag) + 10.0 * max(0.0, A_MIN - amin)


def main():
    os.makedirs(OUT, exist_ok=True)
    P = setup()
    t0 = time.time()
    g = np.linspace(LO, HI, 21)
    grid = np.array([[physics(P, tb, tt) for tb in g] for tt in g])          # [i_top, i_bottom, 3]
    feas = grid[..., 1] >= A_MIN
    obj = np.where(feas, np.abs(grid[..., 0]), np.inf)
    it, ib = np.unravel_index(np.argmin(obj), obj.shape)
    print(f"grid 21x21 ({g.size**2} physics calls, {time.time()-t0:.0f} s): feasible (min alpha >= {A_MIN}) "
          f"{int(feas.sum())} points; best |sag| {obj[it, ib]:.3f} mm at T_bottom {g[ib]:.0f} / T_top {g[it]:.0f} C; "
          f"symmetric 150/150 sag {physics(P, 150., 150.)[0]:+.3f} mm")
    lowest_feasible = float(min(g[ib2] for ib2 in range(g.size) if feas[ib2, ib2]))
    print(f"     lowest symmetric mould temperature that crystallises fully: {lowest_feasible:.0f} C")
    rng = np.random.default_rng(20261103)
    U = list(rng.uniform(0, 1, (6, 2))); to_T = lambda u: LO + (HI - LO) * np.asarray(u)
    Y = []; S = []
    for u in U:
        s, a, f = physics(P, *to_T(u)); Y.append(f); S.append((s, a))
    cand = np.array([(x, y) for x in np.linspace(0, 1, 81) for y in np.linspace(0, 1, 81)])
    hist = [min(Y)]
    for _ in range(24):
        gp = GP2().fit(np.array(U), np.log(np.array(Y) + 1e-3))
        mu, sd = gp.predict(cand)
        u = cand[int(np.argmax(ei(mu, sd, np.log(min(Y) + 1e-3))))]
        s, a, f = physics(P, *to_T(u)); U.append(u); Y.append(f); S.append((s, a)); hist.append(min(Y))
    k = int(np.argmin(Y)); Tb, Tt = to_T(U[k])
    print(f"BO (6 + 24 = {len(Y)} physics calls): best T_bottom {Tb:.1f} / T_top {Tt:.1f} C, sag {S[k][0]:+.3f} mm, "
          f"min alpha {S[k][1]:.3f} (objective {Y[k]:.3f})")
    res = dict(bounds=[LO, HI], alpha_min=A_MIN, grid=dict(T=g.tolist(), sag=grid[..., 0].tolist(), alpha_min=grid[..., 1].tolist(),
               best=[float(g[ib]), float(g[it])], best_abs_sag=float(obj[it, ib]), n_feasible=int(feas.sum())),
               bo=dict(T=[to_T(u).tolist() for u in U], objective=Y, best=[float(Tb), float(Tt)], sag=S[k][0], alpha=S[k][1],
                       history=hist), symmetric_150=physics(P, 150., 150.)[0], lowest_feasible_symmetric=lowest_feasible)
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    _plot(g, grid, feas, np.array([to_T(u) for u in U]), k)
    print(f"wrote {OUT}/results.json and denso_mould_pps_bo.png")


def _plot(g, grid, feas, T, k):
    sys.path.insert(0, os.path.join(ROOT, "slides", "figure_sources"))
    try:                                   # repo convention: thesis_style, graceful fallback
        from thesis_style import use
        use(width_frac=0.6, aspect=0.85)
    except Exception:
        pass
    matplotlib.rcParams.update({"text.usetex": False, "font.family": "serif",
                                "font.serif": ["Times New Roman", "Liberation Serif", "STIXGeneral"],
                                "mathtext.fontset": "stix", "font.size": 9})
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    cs = ax.contourf(g, g, grid[..., 0], levels=21, cmap="Greys")
    fig.colorbar(cs, ax=ax, label="sag over 100 mm [mm]")
    ax.contour(g, g, grid[..., 0], levels=[0.0], colors="k", linewidths=1.2)
    ax.contour(g, g, grid[..., 1], levels=[A_MIN], colors="k", linewidths=1.0, linestyles="--")
    ax.plot(T[:, 0], T[:, 1], "^", mfc="white", mec="k", ms=4, label="BO samples")
    ax.plot(T[k, 0], T[k, 1], "*", color="k", ms=11, label="BO best")
    ax.set_xlabel(r"$T_\mathrm{bottom}$ [°C]"); ax.set_ylabel(r"$T_\mathrm{top}$ [°C]")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    ax.text(0.98, 0.02, "solid: flat; dashed: min. rel. crystallinity 0.9", transform=ax.transAxes, ha="right", fontsize=6.5)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "denso_mould_pps_bo.png"), dpi=300); plt.close(fig)


if __name__ == "__main__":
    main()
