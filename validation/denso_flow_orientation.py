"""validation/denso_flow_orientation.py -- through-thickness fibre orientation from a simple filling model,
to link moulding conditions (mould-wall temperatures, flow-length, melt shear thinning) to a_ij(z), and
a_ij(z) to the warp of the PPS process model (abaqus/denso_plate/process_pps.py).

Flow (fully developed, at a distance L from the gate, in the gap of thickness H):
  frozen skins of thickness d_b, d_t from 1-D conduction into each wall (erf solution),
      erf(d / (2 sqrt(D t_f))) = (T_nf - T_w) / (T_melt - T_w),                                   (1)
  between them a power-law melt (index n) under a pressure gradient: velocity u(z) and shear rate g(z).
Orientation (Folgar-Tucker; Jeffery with shape factor lam = (s^2 - 1)/(s^2 + 1), s = 16.8, hybrid closure):
  da/dgamma = W.a - a.W + lam (D.a + a.D - 2 A4:D) + 2 Ci (I - 3a),   in shear-strain units,        (2)
  integrated from the gate orientation a0 to the accumulated strain of each layer,
      gamma(z) = g(z) L / u(z)   (residence time L/u at shear rate g).                               (3)
  Frozen skins take the orientation of the flowing layer next to them (they froze from it).
Inputs and status: s, H, T_w = 150 C, T_melt = 320 C from Tsuda et al. 2026 / Lopez & Wilkes 1988;
  Ci = 0.005, n = 0.5, L = 30 mm, no-flow temperature T_nf = 280 C, D = 1.8e-7 m^2/s, fill time 1 s and
  the gate orientation a0 = diag(0.20, 0.70, 0.10) (fibres across the flow from the diverging gate flow)
  are ASSUMED typical values, not fitted.
Check (printed): the computed a_xx(z), a_yy(z) against the paper's Fig. 8a (read by eye) -- not fitted,
  so only the skin/shell/core structure and the core width are compared, with the RMSE stated.
Then (printed): the mould-wall asymmetry T_b != T_t moves the core off centre; the resulting a_ij(z) is
  passed to the PPS process model and its sag reported, i.e. a design handle on the warp.
Honest scope: no fountain flow (the outermost skin), no 2-D/3-D gate flow, isothermal melt viscosity,
  steady frozen-layer thickness at the fill time, one location. A qualitative link, NOT a flow simulation.

Result (2026-11-03): the check FAILS. With the assumed values the core (a_yy > a_xx) is 0.21 mm wide
against 1.60 mm in the paper (RMSE a_xx 0.46, a_yy 0.40); the shell maximum a_xx is right (0.93 vs
0.90). Sensitivity (flow length L 3-30 mm, fill time 0.3-1 s, n 0.3-0.7; not a fit): the core width
is 0.00-1.39 mm, never 1.60 mm; it is widest close to the gate and for strong shear thinning. Shear
alone (eq. 3) aligns too much of the thickness; the wide transverse core of the paper needs the
extensional / diverging flow at the gate and flow front, which this 1-D model does not contain.
The mould-wall asymmetry moves the core and gives a sag (110/190 C: +0.48 mm, fully crystallised),
i.e. a design handle exists in principle, but from a model that fails its check: qualitative only.
RSC (slow orientation, kappa < 1) at the paper's flow length L = 50 mm (film gate; Tsuda et al. 2026):
core 0.31 / 0.62 / 0.77 / 0.92 mm for kappa 1 / 0.2 / 0.1 / 0.05 (n = 0.3), RMSE a_yy 0.38 -> 0.25;
still short of 1.60 mm. Remaining suspects: growth of the frozen layer DURING filling and the flow-front
(fountain) region, the gate orientation a0 -- a 3-D filling simulation (openInjMoldSim, fifa guide §17).

Run:  python3 validation/denso_flow_orientation.py   (writes results/denso_flow/)
"""
from __future__ import annotations

import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "denso_flow")
H = 3.08e-3
S_AR = 16.8
LAM = (S_AR ** 2 - 1) / (S_AR ** 2 + 1)
BASE = dict(Ci=0.005, n=0.5, L=30e-3, T_nf=280.0, T_melt=320.0, D=1.8e-7, t_fill=1.0,
            a0=(0.20, 0.70, 0.10), Tb=150.0, Tt=150.0)


def hybrid_A4D(a, D):
    """A4:D with the hybrid closure (Advani & Tucker 1990)."""
    f = 1.0 - 27.0 * np.linalg.det(a)
    I = np.eye(3)
    quad = a * np.tensordot(a, D)
    tr = np.trace(a @ D)
    lin = (-1 / 35) * (2 * D) + (1 / 7) * (2 * (a @ D + D @ a) + I * tr)     # tr D = 0 for shear
    return (1 - f) * lin + f * quad


def rsc_terms(a, A4D, kappa):
    """RSC correction (Wang, O'Gara & Tucker, J. Rheol. 52, 2008): closure term A4:D is replaced by
    A4:D + (1 - kappa)(L:D - M:(A4:D)), L = sum_i lam_i e_i e_i e_i e_i, M = sum_i e_i e_i e_i e_i."""
    lam, E = np.linalg.eigh(a)
    LD = sum(lam[i] * (E[:, i] @ D_G @ E[:, i]) * np.outer(E[:, i], E[:, i]) for i in range(3))
    MAD = sum((E[:, i] @ A4D @ E[:, i]) * np.outer(E[:, i], E[:, i]) for i in range(3))
    return A4D + (1 - kappa) * (LD - MAD)


_LG = np.zeros((3, 3)); _LG[0, 2] = 1.0
W_G = 0.5 * (_LG - _LG.T); D_G = 0.5 * (_LG + _LG.T)


def orient_after_strain(a0, gamma, Ci, dg=0.05, kappa=1.0):
    Lg = np.zeros((3, 3)); Lg[0, 2] = 1.0                      # u_x(z): dU_x/dz, unit shear rate
    W = 0.5 * (Lg - Lg.T); D = 0.5 * (Lg + Lg.T)
    a = np.diag(a0).astype(float); g = 0.0; I = np.eye(3)
    while g < gamma:
        h = min(dg, gamma - g)
        def rhs(a):
            A4D = hybrid_A4D(a, D)
            if kappa != 1.0:
                A4D = rsc_terms(a, A4D, kappa)
            return W @ a - a @ W + LAM * (D @ a + a @ D - 2 * A4D) + 2 * kappa * Ci * (I - 3 * a)
        k1 = rhs(a); k2 = rhs(a + h / 2 * k1); k3 = rhs(a + h / 2 * k2); k4 = rhs(a + h * k3)
        a = a + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4); a = 0.5 * (a + a.T); a = a / np.trace(a); g += h
    return a


def frozen(Tw, p):
    r = (p["T_nf"] - Tw) / (p["T_melt"] - Tw)
    if r <= 0:
        return 0.0
    # invert erf by bisection
    lo, hi = 0.0, 6.0
    for _ in range(60):
        m = 0.5 * (lo + hi)
        lo, hi = (m, hi) if math.erf(m) < r else (lo, m)
    return 2 * lo * math.sqrt(p["D"] * p["t_fill"])


def profile(p, nz=61, gmax=400.0):
    db, dt = frozen(p["Tb"], p), frozen(p["Tt"], p)
    zb, zt = -H / 2 + db, H / 2 - dt; hc = 0.5 * (zt - zb); zc = 0.5 * (zt + zb)
    z = np.linspace(-H / 2, H / 2, nz)
    n = p["n"]
    xi = np.clip((z - zc) / hc, -1, 1)                           # -1..1 in the flowing gap
    u = 1 - np.abs(xi) ** (1 + 1 / n)                            # power-law shape, max 1 at the centre
    g = (1 + 1 / n) * np.abs(xi) ** (1 / n) / hc                 # |du/dz| per unit max velocity
    gamma = np.where(u > 1e-6, g * p["L"] / np.maximum(u, 1e-6), gmax)
    gamma = np.minimum(gamma, gmax)
    A = np.array([orient_after_strain(p["a0"], gm, p["Ci"], kappa=p.get("kappa", 1.0)) for gm in gamma])
    inflow = (z > zb) & (z < zt)
    # frozen skins: orientation of the adjacent flowing layer
    ib = np.argmax(inflow); it = len(z) - 1 - np.argmax(inflow[::-1])
    A[~inflow & (z <= zb)] = A[ib]; A[~inflow & (z >= zt)] = A[it]
    depth = (H / 2 - z) * 1e3                                    # from the top (drilled) face, mm
    return dict(depth=depth[::-1], axx=A[::-1, 0, 0], ayy=A[::-1, 1, 1], azz=A[::-1, 2, 2], d_b=db, d_t=dt)


def main():
    os.makedirs(OUT, exist_ok=True)
    pap = json.load(open(os.path.join(ROOT, "data", "literature", "tsuda2026_jscm52_88.json"), encoding="utf-8"))["fig8_orientation_3mm"]
    r = profile(BASE)
    ax_i = np.interp(pap["depth_mm"], r["depth"], r["axx"]); ay_i = np.interp(pap["depth_mm"], r["depth"], r["ayy"])
    rm_x = float(np.sqrt(np.mean((ax_i - np.array(pap["a_xx"])) ** 2)))
    rm_y = float(np.sqrt(np.mean((ay_i - np.array(pap["a_yy"])) ** 2)))
    core = lambda d, ay: (lambda m: (float(d[m][0]), float(d[m][-1])) if m.any() else (None, None))(np.asarray(ay) > np.asarray(1 - np.asarray(ay) - 0.1))
    cm = r["ayy"] > r["axx"]; cp = np.array(pap["a_yy"]) > np.array(pap["a_xx"])
    cw_m = float(r["depth"][cm].max() - r["depth"][cm].min()) if cm.any() else 0.0
    cw_p = float(np.array(pap["depth_mm"])[cp].max() - np.array(pap["depth_mm"])[cp].min()) if cp.any() else 0.0
    print(f"frozen skins at Tw 150 C: {r['d_b']*1e3:.3f} / {r['d_t']*1e3:.3f} mm")
    print(f"[check vs paper Fig. 8a, NOT fitted] RMSE a_xx {rm_x:.3f}, a_yy {rm_y:.3f}; "
          f"core width (a_yy > a_xx): model {cw_m:.2f} mm, paper {cw_p:.2f} mm; "
          f"shell a_xx max: model {r['axx'].max():.2f}, paper {max(pap['a_xx']):.2f}")
    res = dict(base=BASE, check=dict(rmse_axx=rm_x, rmse_ayy=rm_y, core_width_model=cw_m, core_width_paper=cw_p),
               profile_base={k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in r.items()}, asym=[])
    # mould-wall asymmetry -> core shift -> sag (PPS process model)
    sys.path.insert(0, os.path.join(ROOT, "design"))
    import denso_mould_pps_bo as bo
    import process_reference as pr
    bo.setup()
    from dataclasses import replace  # noqa: F401
    CP = pr.C
    for Tb, Tt in ((150.0, 150.0), (130.0, 170.0), (110.0, 190.0), (170.0, 130.0)):
        p = dict(BASE, Tb=Tb, Tt=Tt); q = profile(p)
        def fn(zm, q=q):
            depth = (1 - np.asarray(zm)) / 2 * H * 1e3
            a11 = np.interp(depth, q["depth"], q["axx"]); a22 = np.interp(depth, q["depth"], q["ayy"])
            return a11, a22, np.clip(1 - a11 - a22, 0, 1)
        P, _ = pr.layer_props(fn, c=CP)
        # orientation is set by the FILL wall temperatures; the moulded part is then cooled at the same walls
        t, Tn = pr.heat_1d(Tb, Tt); out = pr.plate_history(P, t, Tn)
        sag = float(out["kappa"][1] * 0.10 ** 2 / 8 * 1e3); amin = float(out["alpha_layer"].min())
        cmask = q["ayy"] > q["axx"]; cc = float(q["depth"][cmask].mean()) if cmask.any() else float("nan")
        res["asym"].append(dict(Tb=Tb, Tt=Tt, d_b=q["d_b"], d_t=q["d_t"], core_centre_depth_mm=cc, sag_y_mm=sag, alpha_min=amin))
        print(f"walls {Tb:.0f}/{Tt:.0f} C: skins {q['d_b']*1e3:.3f}/{q['d_t']*1e3:.3f} mm, core centre at depth "
              f"{cc:.2f} mm (mid 1.54), sag_y {sag:+.3f} mm, min rel. crystallinity {amin:.2f}")
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    _plot(r, pap)
    print(f"wrote {OUT}/results.json and flow_orientation.png")


def _plot(r, pap):
    sys.path.insert(0, os.path.join(ROOT, "slides", "figure_sources"))
    try:                                   # repo convention: thesis_style, graceful fallback
        from thesis_style import use
        use(width_frac=0.5, aspect=1.2)
    except Exception:
        pass
    matplotlib.rcParams.update({"text.usetex": False, "font.family": "serif",
                                "font.serif": ["Times New Roman", "Liberation Serif", "STIXGeneral"],
                                "mathtext.fontset": "stix", "font.size": 9})
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.6, 3.8))
    ax.plot(pap["a_xx"], pap["depth_mm"], "ko", ms=3, label=r"$a_{xx}$ paper (read)")
    ax.plot(pap["a_yy"], pap["depth_mm"], "s", color="0.5", ms=3, mfc="white", label=r"$a_{yy}$ paper (read)")
    ax.plot(r["axx"], r["depth"], "k-", lw=1.2, label=r"$a_{xx}$ model")
    ax.plot(r["ayy"], r["depth"], "--", color="0.5", lw=1.2, label=r"$a_{yy}$ model")
    ax.invert_yaxis(); ax.set_xlim(0, 1)
    ax.set_xlabel("orientation tensor component [$-$]"); ax.set_ylabel("depth from the top face [mm]")
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "flow_orientation.png"), dpi=300); plt.close(fig)


if __name__ == "__main__":
    main()
