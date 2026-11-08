"""validation/denso_flow_coupling.py -- does fibre-flow coupling widen the transverse core of the 1-D flow-orientation
model (validation/denso_flow_orientation.py, whose check FAILED: core 0.21-0.92 mm against 1.60 mm in the paper)?

Coupling (Lipscomb et al. 1988 / Dinh & Armstrong form, as used by Tseng et al. for "flow-fibre coupling"):
    tau = 2 eta (D + N_p A4:D),   so in the gap shear  tau_xz = eta(gdot) gdot F(a),
    F(a) = 1 + 2 N_p (A4:D_hat)_xz   (D_hat: unit shear rate, hybrid closure)                          (1)
Gap flow at the evaluation site (fully developed, frozen skins as the base model, power-law melt index n):
    tau_xz = G (z - z_c)  ->  gdot(z) = (|z - z_c| / F(a(z)))^(1/n) (up to a constant),
    u(z) = int_{wall}^{z} gdot dz',  accumulated strain gamma(z) = gdot(z) L / u(z)   (scale-free)       (2)
Orientation: all layers start from the same gate state a0, so a(gamma) is ONE trajectory of Folgar-Tucker (RSC
optional), integrated once and interpolated; (1)-(2) are iterated to a fixed point (F from the current a(z)).
Approximation: F uses the final orientation of each layer, not its history.
Base values as denso_flow_orientation.py (ASSUMED typical values) except L = 50 mm (the paper's evaluation site,
50 mm from the film gate). N_p is a coupling number, 0 = the decoupled base model; values 5-40 are the range quoted
for short-glass-fibre melts (assumed, not measured for PPS).
Check (not fitted): core width (a_yy > a_xx) and RMSE of a_xx, a_yy against the paper's Fig. 8a (read by eye).
Honest scope: same 1-D, steady, no-fountain-flow model; one assumed gate state; the coupling number is not known for
this material; a sweep, not a fit -- the width is reported for every value, and the result is a failure unless the
paper's core is reached for a physically plausible N_p without other changes.
Result (2026-11-08):
  coupling: FAILS. The core widens only 0.40 -> 0.52 mm (kappa 1, n 0.3, N_p 0 -> 40) and not at all with RSC
  (kappa 0.1: 0.80 -> 0.74 mm), against 1.60 mm. Coupling is not the missing piece.
  diagnosis D1: the paper's shells (0.79 / 0.60 mm, a_xx = a_yy crossings) match the frozen-skin thickness of the
  conduction model at the assumed fill time (0.71 mm); i.e. the core is the whole channel still flowing at the end
  of filling, which the shear model over-shears.
  D2: the two-zone rule (skins aligned, channel at the gate state; no number fitted to orientation data) gives core
  1.64 mm, RMSE a_xx 0.21, a_yy 0.19 (best shear model 0.35, 0.29).
  D3: the 1 mm plate (Fig. 7a): core 0.24 mm (two-zone, fill time scaled with thickness, ASSUMED) vs 0.19 mm.
  D4 (third plate, Ricotta et al. 2021, PPS/GF40 1.8 mm, mould 80 C, fill 1.8 s, core ~0.15 mm): the rule needs a
  contact time of 0.83 s (46 % of the fill; a section at about 54 % of the flow path). Consistent, not a check.
  Not explained: WHY the channel keeps the gate orientation (candidates: flow-front / packing reorientation).
  The fill time (1 s at 3 mm), T_nf and D are assumed; the two plates test the RULE, not these values.
Run:  python3 validation/denso_flow_coupling.py   (writes results/denso_flow/coupling.json)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import denso_flow_orientation as fo  # noqa: E402

OUT = os.path.join(ROOT, "results", "denso_flow")
GMAX = 400.0


def trajectory(a0, Ci, kappa, dg=0.02):
    """a(gamma) for gamma in [0, GMAX] from a0 (Folgar-Tucker / RSC, hybrid closure), and F-term (A4:D_hat)_xz."""
    gs = [0.0]; As = [np.diag(a0).astype(float)]
    a = As[0].copy(); g = 0.0; I = np.eye(3); W, D = fo.W_G, fo.D_G

    def rhs(a):
        A4D = fo.hybrid_A4D(a, D)
        if kappa != 1.0:
            A4D = fo.rsc_terms(a, A4D, kappa)
        return W @ a - a @ W + fo.LAM * (D @ a + a @ D - 2 * A4D) + 2 * kappa * Ci * (I - 3 * a)
    while g < GMAX:
        h = dg if g < 20 else 0.2
        k1 = rhs(a); k2 = rhs(a + h / 2 * k1); k3 = rhs(a + h / 2 * k2); k4 = rhs(a + h * k3)
        a = a + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4); a = 0.5 * (a + a.T); a /= np.trace(a); g += h
        gs.append(g); As.append(a.copy())
    gs = np.array(gs); As = np.array(As)
    fx = np.array([fo.hybrid_A4D(x, D)[0, 2] for x in As])
    return gs, As, fx


def profile(p, Np, traj, nz=241, iters=40):
    gs, As, fx = traj
    db, dt = fo.frozen(p["Tb"], p), fo.frozen(p["Tt"], p)
    zb, zt = -fo.H / 2 + db, fo.H / 2 - dt; zc = 0.5 * (zt + zb)
    z = np.linspace(zb, zt, nz)
    F = np.ones(nz)
    for _ in range(iters):
        hc = 0.5 * (zt - zb)
        gd = (np.abs(z - zc) / hc / F) ** (1 / p["n"]) / hc                  # O(1/hc): u = O(1), no clamp effect
        # u from each wall towards the centre (symmetric walls: the two halves meet at z_c)
        seg = 0.5 * (gd[1:] + gd[:-1]) * np.diff(z)
        ub = np.concatenate([[0.0], np.cumsum(seg)]); ut = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
        u = np.where(z <= zc, ub, ut)
        u = np.maximum(u, 1e-12)
        gam = np.minimum(gd * p["L"] / u, GMAX)
        Fn = 1 + 2 * Np * np.interp(gam, gs, fx)
        if np.max(np.abs(Fn - F)) < 1e-6:
            break
        F = 0.5 * F + 0.5 * Fn
    A = np.array([[np.interp(gm, gs, As[:, i, j]) for i in range(3) for j in range(3)] for gm in gam]).reshape(-1, 3, 3)
    zf = np.linspace(-fo.H / 2, fo.H / 2, 309)
    Af = np.empty((len(zf), 3, 3))
    for i in range(3):
        for j in range(3):
            Af[:, i, j] = np.interp(zf, z, A[:, i, j])               # skins: orientation of the adjacent flowing layer
    depth = (fo.H / 2 - zf) * 1e3
    return dict(depth=depth[::-1], axx=Af[::-1, 0, 0], ayy=Af[::-1, 1, 1], d_b=db, d_t=dt, F_max=float(F.max()), F_min=float(F.min()))


def check(r, pap):
    dp = np.array(pap["depth_mm"])
    ax = np.interp(dp, r["depth"], r["axx"]); ay = np.interp(dp, r["depth"], r["ayy"])
    rx = float(np.sqrt(np.mean((ax - np.array(pap["a_xx"])) ** 2))); ry = float(np.sqrt(np.mean((ay - np.array(pap["a_yy"])) ** 2)))
    cm = r["ayy"] > r["axx"]
    cw = float(r["depth"][cm].max() - r["depth"][cm].min()) if cm.any() else 0.0
    return cw, rx, ry


def main():
    os.makedirs(OUT, exist_ok=True)
    pap = json.load(open(os.path.join(ROOT, "data", "literature", "tsuda2026_jscm52_88.json"), encoding="utf-8"))["fig8_orientation_3mm"]
    cp = np.array(pap["a_yy"]) > np.array(pap["a_xx"]); dp = np.array(pap["depth_mm"])
    cw_p = float(dp[cp].max() - dp[cp].min())
    base = dict(fo.BASE, L=50e-3)
    rows = []
    # consistency: N_p = 0 reproduces the base model's profile at the same inputs
    tr1 = trajectory(base["a0"], base["Ci"], 1.0)
    r0 = profile(base, 0.0, tr1); rb = fo.profile(base)
    d0 = float(np.max(np.abs(np.interp(rb["depth"], r0["depth"], r0["ayy"]) - rb["ayy"])))
    print(f"[C0] N_p = 0 vs denso_flow_orientation.profile at L = 50 mm: max |a_yy difference| {d0:.3f} "
          f"(different z-grid and tabulated trajectory; small expected)")
    print(f"paper core width (a_yy > a_xx): {cw_p:.2f} mm; frozen skins {r0['d_b'] * 1e3:.2f} / {r0['d_t'] * 1e3:.2f} mm")
    for kappa in (1.0, 0.1):
        tr = tr1 if kappa == 1.0 else trajectory(base["a0"], base["Ci"], kappa)
        for n in (0.5, 0.3):
            for Np in (0.0, 5.0, 10.0, 20.0, 40.0):
                p = dict(base, n=n)
                r = profile(p, Np, tr); cw, rx, ry = check(r, pap)
                rows.append(dict(kappa=kappa, n=n, Np=Np, core_mm=cw, rmse_axx=rx, rmse_ayy=ry, F_min=r["F_min"], F_max=r["F_max"]))
                print(f"kappa {kappa:4.2f}  n {n:.1f}  N_p {Np:5.1f}: core {cw:.2f} mm (paper {cw_p:.2f}), RMSE a_xx {rx:.2f}, a_yy {ry:.2f}; "
                      f"F {r['F_min']:.2f}-{r['F_max']:.2f}", flush=True)
    # diagnosis: shell thickness of the paper (a_xx = a_yy crossings) vs the frozen skins of the conduction model
    ax_p, ay_p = np.array(pap["a_xx"]), np.array(pap["a_yy"]); dd = ay_p - ax_p
    xs = [float(dp[i] - dd[i] * (dp[i + 1] - dp[i]) / (dd[i + 1] - dd[i])) for i in range(len(dp) - 1) if dd[i] * dd[i + 1] < 0]
    top, bot = xs[0], float(dp[-1] - xs[-1])
    print(f"[D1] paper: a_yy > a_xx between depth {xs[0]:.2f} and {xs[-1]:.2f} mm -> shells {top:.2f} (top) / {bot:.2f} (bottom) mm; "
          f"frozen skins at the fill time (eq. 1 of the base model, not fitted) {r0['d_t'] * 1e3:.2f} / {r0['d_b'] * 1e3:.2f} mm")
    # two-zone limit: frozen skins aligned (orientation at large strain), the whole flowing channel at the gate state
    two = dict(depth=r0["depth"], axx=np.where((r0["depth"] < r0["d_t"] * 1e3) | (r0["depth"] > (fo.H - r0["d_b"]) * 1e3), tr1[1][-1, 0, 0], base["a0"][0]),
               ayy=np.where((r0["depth"] < r0["d_t"] * 1e3) | (r0["depth"] > (fo.H - r0["d_b"]) * 1e3), tr1[1][-1, 1, 1], base["a0"][1]))
    cw2, rx2, ry2 = check(two, pap)
    print(f"[D2] two-zone limit (skins aligned, channel unsheared, no free number): core {cw2:.2f} mm, RMSE a_xx {rx2:.2f}, a_yy {ry2:.2f} "
          f"(best coupled 1-D: {min(r['rmse_axx'] for r in rows):.2f}, {min(r['rmse_ayy'] for r in rows):.2f})")
    # D3: the 1 mm plate (Fig. 7a, mean of 4, symmetric) with the same two-zone rule. Fill time unknown for both plates:
    # ASSUMED the same volumetric flow rate, so t_fill scales with the cavity thickness (1.08 / 3.08).
    f7 = json.load(open(os.path.join(ROOT, "data", "literature", "tsuda2026_jscm52_88.json"), encoding="utf-8"))["fig7_orientation_1mm"]
    z7, ax7, ay7 = map(np.array, (f7["z_mm"], f7["a_xx"], f7["a_yy"])); d7 = ay7 - ax7
    zc7 = [float(z7[i] - d7[i] * (z7[i + 1] - z7[i]) / (d7[i + 1] - d7[i])) for i in range(len(z7) - 1) if d7[i] * d7[i + 1] < 0 and z7[i] > -0.45]
    core7_p = 2 * abs(zc7[-1]) if zc7 else 0.0
    H1 = 1.08e-3; p1 = dict(base, t_fill=base["t_fill"] * H1 / fo.H)
    d1 = fo.frozen(150.0, p1)
    core7_m = max(H1 - 2 * d1, 0.0) * 1e3
    print(f"[D3] 1 mm plate: paper core (a_yy > a_xx) {core7_p:.2f} mm, shells {(1.08 - core7_p) / 2:.2f} mm; two-zone with fill time "
          f"x {H1 / fo.H:.2f} (same flow rate, ASSUMED): skins {d1 * 1e3:.2f} mm -> core {core7_m:.2f} mm")
    # D4: a third plate, Ricotta, Sorgato, Zappalorto, Procedia Struct. Integr. 33 (2021) 695-703: PPS/GF40, 200 x 200 x
    # 1.8 mm, mould 80 C, melt 310 C, flow rate 40 cm3/s -> fill time 1.8 s; SEM: core about 0.15 mm (8.3 %). The site
    # of the section is not given, so the contact time is unknown: report the contact time the rule needs.
    p4 = dict(base, T_melt=310.0); H4, tf4 = 1.8, 0.2 * 0.2 * 1.8e-3 * 1e6 / 40.0
    lo, hi = 0.01, 3.0
    for _ in range(60):
        m = 0.5 * (lo + hi); lo, hi = (m, hi) if H4 - 2 * fo.frozen(80.0, dict(p4, t_fill=m)) * 1e3 > 0.15 else (lo, m)
    print(f"[D4] Ricotta et al. 2021 plate (1.8 mm, mould 80 C, fill {tf4:.1f} s, core ~0.15 mm): the rule needs a contact time "
          f"of {m:.2f} s ({m / tf4:.0%} of the fill, i.e. a section at about {1 - m / tf4:.0%} of the flow path at constant "
          f"front speed); full-fill contact would freeze the whole thickness. Consistent, NOT a check (site unknown).")
    best = max(rows, key=lambda x: x["core_mm"])
    ok = best["core_mm"] >= 0.8 * cw_p
    print(f"widest core {best['core_mm']:.2f} mm at kappa {best['kappa']}, n {best['n']}, N_p {best['Np']} -> "
          + ("reaches 80 % of the paper's core" if ok else "FAILS: below 80 % of the paper's core for every value"))
    json.dump(dict(paper_core_mm=cw_p, C0=d0, rows=rows, reached=ok, D1=dict(paper_shell_top=top, paper_shell_bottom=bot,
                                                                                frozen_skin_mm=r0["d_t"] * 1e3),
                   D2=dict(core_mm=cw2, rmse_axx=rx2, rmse_ayy=ry2),
                   D3=dict(core_1mm_paper=core7_p, core_1mm_model=core7_m, skin_1mm_model=d1 * 1e3),
                   D4=dict(fill_s=tf4, contact_s_needed=m)), open(os.path.join(OUT, "coupling.json"), "w"), indent=1)
    print("wrote", os.path.join(OUT, "coupling.json"))


if __name__ == "__main__":
    main()
