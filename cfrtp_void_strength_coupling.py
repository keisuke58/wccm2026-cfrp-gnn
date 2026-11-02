"""cfrtp_void_strength_coupling.py — couple the IMPREGNATION/VOID model
(cfrtp_impregnation_void.py) to the INTERLAMINAR STRENGTH model
(cfrtp_ilss_interface.py), closing the loop process -> void -> ILSS. This is the
"開繊/含浸のボイド→力学特性 連成" next step listed in research/CFRTP_DAIKIN_SUMMARY.md
and research/MEMO_cfrp_residual_stress_collab.md. Daikin/NEDO theme.

Why: the two Daikin issues are currently modelled in isolation. cfrtp_impregnation_void.py
predicts how much VOID a given process (tow thickness, spreading, time above melt,
viscosity, pressure) leaves behind; cfrtp_ilss_interface.py predicts ILSS from interface
strength/toughness. But voids are not a cosmetic defect: they remove bonded area at the
ply interface, so they directly degrade the interlaminar properties. Chaining the two
turns "how much void" into the number that actually matters for the part — "how much
interlaminar strength is left" — and lets the fibre-SPREADING (開繊) benefit be stated in
MECHANICAL terms rather than in impregnation-time terms.

Void -> interface property knockdown (stereology + net-section):
  * areal void fraction on the interface plane. By Delesse's principle the areal fraction
    on a RANDOM section equals the volume fraction v. Real interlaminar voids are not
    random: they are flattened and segregate to the ply interfaces, so they intercept the
    interface plane MORE than a random cut would. A segregation factor kappa >= 1 carries
    this (kappa = 1 recovers the random/Delesse lower bound):
        phi = 1 - min(kappa * v, 1)         (bonded area fraction left)
  * interface STRENGTH: net-section area loss, plus stress concentration in the narrowing
    ligaments between voids, as a power law
        tau_max_eff = tau_max * phi**N_SCF   (N_SCF = 1 -> pure area loss, >1 -> SCF penalty)
  * interface TOUGHNESS: dissipation happens only in bonded ligaments, so it scales with
    the surviving area, linearly and with no SCF term
        GIIc_eff = GIIc * phi
Then ILSS follows from the SAME cohesive shear-lag solver as cfrtp_ilss_interface.py,
called with the degraded (tau_max_eff, GIIc_eff) — no new mechanics, just the coupling.

Validated:
  * v = 0 -> phi = 1 -> ILSS reproduces the uncoupled baseline to machine precision;
  * the small-void slope is checked against the classic composites rule of thumb that
    ILSS falls by roughly 6-8% per 1% voids (up to ~4% voids) — an independent sanity
    anchor, since no fluoropolymer-CFRTP data is available here.

*** WHAT THE SANITY CHECK SAYS (a real, reported mismatch, not tuned away) ***
With the argued-for parameters (kappa=2, N_SCF=1.5) the model predicts only ~2.3% ILSS
loss per 1% void — roughly 3x SHALLOWER than the empirical 6-8%/1% band. The mechanism is
transparent: in this configuration the shear-lag ILSS is STRENGTH-controlled, not
toughness-controlled (checked directly: ILSS tracks tau_max almost linearly, while a 40%
cut in GIIc moves it ~1%; the strength limit tau_max=15 MPa binds well below the LEFM
limit sqrt(2 EA GIIc)/L = 27 MPa). So ILSS ~ tau_max * phi**N_SCF and the knockdown slope
is set by the product kappa*N_SCF, which is 3 here. Reaching the empirical band needs
kappa*N_SCF ~ 7-9 (reported by the script). Read physically, matching real ILSS-vs-void
data requires EITHER strong interlaminar segregation of flattened voids (kappa ~ 4-6,
plausible: interlaminar voids lie IN the bond plane rather than being randomly cut) OR
voids doing more than removing area — acting as crack INITIATORS / notches, which a
scalar area-fraction model cannot represent. This is a genuine, useful negative result: a
pure area-loss picture under-predicts how badly voids hurt ILSS, and it tells the
experimentalist exactly what to measure (void morphology and segregation at the interface
plane, not just total void volume).

Shows: the ILSS knockdown law vs void content (against the empirical band, with the
required-kappa*N_SCF sensitivity made explicit); void AND the resulting ILSS vs tow
thickness (where the process window really closes); the fibre-spreading (開繊) benefit
propagated all the way to ILSS, at two melt viscosities (so the THRESHOLD nature of the
benefit is visible); and an ILSS map over the (tow thickness, time above melt) process
window — the same window as cfrtp_impregnation_void.py, but read in strength rather than
void units. Also reports the triple coupling with residual stress (residual interfacial
shear eats interface capacity, cf. cfrtp_ilss_interface.py).

Honest scope: the knockdown law is a REDUCED stereological/net-section model with two
shape parameters (kappa, N_SCF) that are argued-for but NOT fitted to data — and, as
above, with those parameters it does NOT reproduce the empirical slope; the band is
reported as a check, never used to tune. Voids are a scalar fraction (no size
distribution, no explicit void geometry/location, no crack-initiation mechanism); the
underlying 1-D Darcy void model and mode-II shear-lag ILSS model keep all their own
limitations, and the ILSS regime conclusion is specific to this overlap/property set.
Illustrative fluoropolymer-CFRTP values throughout. Physics leads; no ML.

Run:  python3 cfrtp_void_strength_coupling.py     (writes cfrtp_void_strength_coupling.png)
      python3 cfrtp_void_strength_coupling.py --help
"""
from __future__ import annotations

import argparse

import numpy as np

from cfrtp_impregnation_void import void_content, t_impregnate, H0, T_PROC, MU, DP, VF
from cfrtp_ilss_interface import solve as ilss_solve

SEED = 20260726
# void -> interface knockdown shape parameters (argued-for, NOT fitted)
KAPPA_SEG = 2.0        # interlaminar segregation: voids intercept the interface plane
                       # kappa x more than a random section would (1.0 = Delesse/random)
N_SCF = 1.5            # strength exponent: 1.0 = pure net-section, >1 = stress concentration
# uncoupled baseline interface (the "weak / fluoropolymer" case of cfrtp_ilss_interface.py)
TAU_MAX_0 = 15e6       # [Pa]
GIIC_0 = 0.3e3         # [J/m^2]
# classic composites rule of thumb: ILSS falls ~6-8% per 1% voids up to ~4% voids
EMPIRICAL_SLOPE = (0.06, 0.08)   # fractional ILSS loss per 1% void


def bonded_fraction(v, kappa=KAPPA_SEG):
    """Areal bonded fraction left on the interface plane, from void volume fraction."""
    return np.clip(1.0 - kappa * np.asarray(v, float), 0.0, 1.0)


def degraded_interface(v, tau_max=TAU_MAX_0, GIIc=GIIC_0, kappa=KAPPA_SEG, n_scf=N_SCF):
    """Void-degraded interface strength and toughness."""
    phi = bonded_fraction(v, kappa)
    return tau_max * phi ** n_scf, GIIc * phi, phi


def ilss_of_void(v, tau_max=TAU_MAX_0, GIIc=GIIC_0, kappa=KAPPA_SEG, n_scf=N_SCF):
    """ILSS [Pa] for a given void volume fraction (0 -> uncoupled baseline)."""
    te, ge, phi = degraded_interface(v, tau_max, GIIc, kappa, n_scf)
    if te <= 0 or ge <= 0:
        return 0.0
    return ilss_solve(tau_max=te, GIIc=ge)[4]


def ilss_of_process(h, t_proc=T_PROC, vf=VF, mu=MU, dp=DP, **kw):
    """Full chain: process (tow half-thickness, time above melt, ...) -> void -> ILSS."""
    v = float(void_content(h, t_proc=t_proc, vf=vf, mu=mu, dp=dp))
    return ilss_of_void(v, **kw), v


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kappa", type=float, default=KAPPA_SEG, help="void interlaminar segregation factor")
    ap.add_argument("--nscf", type=float, default=N_SCF, help="strength knockdown exponent")
    ap.add_argument("--nmap", type=int, default=18, help="process-map resolution")
    ap.add_argument("--out", type=str, default="cfrtp_void_strength_coupling.png")
    args = ap.parse_args()
    np.random.seed(SEED)

    kw = dict(kappa=args.kappa, n_scf=args.nscf)
    ilss0 = ilss_of_void(0.0, **kw)
    ilss0_ref = ilss_solve(tau_max=TAU_MAX_0, GIIc=GIIC_0)[4]
    print(f"[validate] void=0 -> ILSS {ilss0/1e6:.6f} MPa vs uncoupled baseline "
          f"{ilss0_ref/1e6:.6f} MPa (rel err {abs(ilss0-ilss0_ref)/ilss0_ref:.2e}, should be ~0)")

    # knockdown law + comparison with the empirical rule of thumb
    vs = np.linspace(0, 0.20, 25)
    ilss_v = np.array([ilss_of_void(v, **kw) for v in vs])
    rel = ilss_v / ilss0
    # small-void slope: fractional ILSS loss per 1% void, over the 0..4% band the rule covers
    m = vs <= 0.04
    slope = -np.polyfit(vs[m] * 100.0, rel[m], 1)[0]
    lo, hi = EMPIRICAL_SLOPE
    verdict = "within" if lo <= slope <= hi else "OUTSIDE"
    print(f"[validate] small-void slope {slope*100:.1f}% ILSS loss per 1% void "
          f"({verdict} the classic {lo*100:.0f}-{hi*100:.0f}%/1% band; "
          f"kappa={args.kappa}, N_SCF={args.nscf}, product={args.kappa*args.nscf:.1f})")

    # what product kappa*N_SCF would be needed to reach the empirical band? (reported,
    # NOT applied -- the defaults stay at the argued-for values)
    def slope_for(product):
        kap = product / args.nscf
        il = np.array([ilss_of_void(v, kappa=kap, n_scf=args.nscf) for v in vs[m]])
        return -np.polyfit(vs[m] * 100.0, il / il[0], 1)[0]

    prods = np.linspace(1.0, 14.0, 27)
    sl = np.array([slope_for(p) for p in prods])
    need = prods[(sl >= lo) & (sl <= hi)]
    if verdict == "OUTSIDE" and len(need):
        print(f"[validate]   -> reaching the band needs kappa*N_SCF ≈ {need[0]:.1f}–{need[-1]:.1f} "
              f"(vs {args.kappa*args.nscf:.1f} used): i.e. strong interlaminar void segregation "
              "and/or a crack-initiation effect this area-loss model does not represent")
    # regime check that explains the shallow slope
    lefm = np.sqrt(2 * 100e9 * 0.3e-3 * GIIC_0) / 5e-3
    print(f"[validate] ILSS regime: strength limit {TAU_MAX_0/1e6:.0f} MPa vs LEFM limit "
          f"{lefm/1e6:.0f} MPa -> STRENGTH-controlled (so the knockdown follows tau_max*phi^N_SCF)")

    print(f"\nCFRTP void -> interlaminar strength coupling (baseline ILSS {ilss0/1e6:.1f} MPa):")
    for v in (0.01, 0.02, 0.05, 0.10):
        print(f"  void {v*100:4.1f}%  -> bonded area {bonded_fraction(v, args.kappa)*100:5.1f}%"
              f"  -> ILSS {ilss_of_void(v, **kw)/1e6:5.2f} MPa "
              f"({100*(1-ilss_of_void(v, **kw)/ilss0):4.1f}% lost)")

    # the spreading (開繊) benefit, carried through to ILSS, at two melt viscosities
    # (the harder mu=6000 case makes the benefit a gradient rather than a step)
    spreads = np.array([1, 1.5, 2, 3, 5])
    sp = {}
    for mu_i in (MU, 2 * MU):
        print(f"\nfibre spreading (開繊) benefit in MECHANICAL terms "
              f"(h0={H0*1e6:.0f} um, t_proc={T_PROC:.0f} s, mu={mu_i:.0f} Pa.s):")
        vv, ii = [], []
        for s in spreads:
            il, v = ilss_of_process(H0 / s, mu=mu_i, **kw)
            vv.append(v); ii.append(il)
            print(f"  spread x{s:<4g} h={H0/s*1e6:5.1f} um  t_imp {t_impregnate(H0/s, mu=mu_i):6.1f} s  "
                  f"void {v*100:5.2f}%  -> ILSS {il/1e6:5.2f} MPa")
        sp[mu_i] = (np.array(vv), np.array(ii))
    sp_void, sp_ilss = sp[MU]
    print(f"  -> at mu={MU:.0f} the benefit is a THRESHOLD: spreading x1.5 already completes "
          f"impregnation (ILSS {sp_ilss[0]/1e6:.2f} -> {sp_ilss[1]/1e6:.2f} MPa, "
          f"{sp_ilss[1]/sp_ilss[0]:.1f}x) and further spreading buys nothing, because the "
          "residual void sits at the pressure-reducible micro-void floor")
    v2, i2 = sp[2 * MU]
    print(f"  -> at mu={2*MU:.0f} (harder to impregnate) the threshold moves out to x{spreads[int(np.argmax(i2 >= 0.99*i2.max()))]:g}: "
          "viscosity sets how much spreading is enough")

    # void vs tow thickness, and the resulting ILSS
    hs = np.linspace(0.25, 1.6, 22) * H0
    v_h = np.array([float(void_content(h, t_proc=T_PROC)) for h in hs])
    ilss_h = np.array([ilss_of_void(v, **kw) for v in v_h])

    # triple coupling: residual interfacial shear ALSO eats interface capacity
    # (cf. cfrtp_ilss_interface.py) -- combine it with the void knockdown
    tau_res = 11e6
    v_ex = float(void_content(H0, t_proc=T_PROC))
    te_v, ge_v, phi_v = degraded_interface(v_ex, **kw)
    ilss_void_only = ilss_of_void(v_ex, **kw)
    ilss_res_only = ilss_solve(tau_max=TAU_MAX_0 - tau_res, GIIc=GIIC_0)[4]
    exhausted = te_v <= tau_res            # residual alone exceeds the degraded strength
    ilss_both = 0.0 if exhausted else ilss_solve(tau_max=te_v - tau_res, GIIc=ge_v)[4]
    print(f"\ntriple coupling at the unspread tow (void {v_ex*100:.1f}%, residual shear "
          f"{tau_res/1e6:.0f} MPa):")
    print(f"  pristine {ilss0/1e6:5.2f} | void only {ilss_void_only/1e6:5.2f} | "
          f"residual only {ilss_res_only/1e6:5.2f} | BOTH "
          + (f"{ilss_both/1e6:5.2f} MPa ({100*(1-ilss_both/ilss0):.0f}% lost)" if not exhausted
             else "0 (DEGENERATE)"))
    if exhausted:
        print(f"  -> degenerate, not a solver failure: the void-degraded interface strength "
              f"({te_v/1e6:.1f} MPa) is BELOW the residual interfacial shear "
              f"({tau_res/1e6:.0f} MPa), so no capacity is left for applied load. Physically "
              "this predicts spontaneous interlaminar failure on cool-down, i.e. this "
              "(void, residual) combination is outside the feasible process window.")

    # process-window map: ILSS over (tow thickness, time above melt)
    n = args.nmap
    hh = np.linspace(0.25, 1.6, n) * H0
    tt = np.linspace(5, 120, n)
    Z = np.zeros((n, n))
    for i, t_p in enumerate(tt):
        for j, h in enumerate(hh):
            Z[i, j] = ilss_of_process(h, t_proc=t_p, **kw)[0] / 1e6

    _plot(args.out, vs, rel, slope, ilss0, hs, v_h, ilss_h, spreads, sp,
          hh, tt, Z, args.kappa, args.nscf, prods, sl,
          (ilss0, ilss_void_only, ilss_res_only, ilss_both, exhausted), v_ex, tau_res)
    print(f"\nwrote {args.out}")


def _plot(out, vs, rel, slope, ilss0, hs, v_h, ilss_h, spreads, sp,
          hh, tt, Z, kappa, n_scf, prods, sl, quad, v_ex, tau_res):
    import matplotlib
    matplotlib.use("Agg")
    import os, shutil, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "slides", "figure_sources"))
    try:                                   # repo convention: thesis_style, graceful fallback
        from thesis_style import use
        use(width_frac=1.0, aspect=0.45)
        if shutil.which("latex") is None:  # style turns usetex on; no TeX here -> plain text
            matplotlib.rcParams["text.usetex"] = False
            matplotlib.rcParams["font.family"] = "sans-serif"   # lmodern is unavailable too
    except Exception:
        pass
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    # (a) knockdown law vs the empirical band
    a = ax[0, 0]
    lo, hi = EMPIRICAL_SLOPE
    vv = np.linspace(0, 0.04, 20)
    a.fill_between(vv * 100, 1 - hi * vv * 100, 1 - lo * vv * 100, color="#999999", alpha=0.25,
                   label=f"empirical {lo*100:.0f}–{hi*100:.0f}% ILSS loss / 1% void")
    a.plot(vs * 100, rel, "-o", ms=3, color="#d62728",
           label=f"model (k={kappa:g}, N_SCF={n_scf:g}): {slope*100:.1f}%/1%  -- TOO SHALLOW")
    a.set_xlabel("void content v [%]"); a.set_ylabel("ILSS / ILSS(v=0)")
    a.set_title("void -> interlaminar strength knockdown\n"
                "area loss alone UNDER-predicts the empirical sensitivity")
    a.legend(fontsize=8); a.grid(True, alpha=0.3); a.set_ylim(0, 1.05)
    # inset: what product k*N_SCF would be needed to enter the band
    ai = a.inset_axes([0.52, 0.16, 0.44, 0.34])
    ai.axhspan(lo, hi, color="#999999", alpha=0.3)
    ai.plot(prods, sl, "-", color="#d62728", lw=1.2)
    ai.axvline(kappa * n_scf, color="k", ls=":", lw=1.0)
    ai.set_xlabel("k x N_SCF", fontsize=7); ai.set_ylabel("slope /1%", fontsize=7)
    ai.tick_params(labelsize=6); ai.set_title("required knockdown product", fontsize=7)

    # (b) tow thickness -> void AND the resulting ILSS
    b = ax[0, 1]; b2 = b.twinx()
    b.plot(hs * 1e6, v_h * 100, "-", color="#7f7f7f", label="void content")
    b2.plot(hs * 1e6, ilss_h / 1e6, "-o", ms=3, color="#1f77b4", label="ILSS")
    b.axvline(H0 * 1e6, color="k", ls=":", alpha=0.6)
    b.annotate("unspread tow", (H0 * 1e6, b.get_ylim()[1] * 0.9), fontsize=8,
               ha="right", rotation=90, va="top")
    b.set_xlabel("tow half-thickness h [µm]"); b.set_ylabel("void content [%]", color="#7f7f7f")
    b2.set_ylabel("ILSS [MPa]", color="#1f77b4")
    b.set_title("the process window closes in STRENGTH,\nnot just in void content")
    b.grid(True, alpha=0.3)

    # (c) spreading benefit carried through to ILSS, at two melt viscosities
    c = ax[1, 0]
    xpos = np.arange(len(spreads)); w = 0.38
    mus = sorted(sp.keys())
    for k, (mu_i, col) in enumerate(zip(mus, ("#2ca02c", "#8c564b"))):
        vv, ii = sp[mu_i]
        c.bar(xpos + (k - 0.5) * w, ii / 1e6, w, color=col, alpha=0.85,
              label=f"mu = {mu_i:.0f} Pa.s")
        for j, (il, v) in enumerate(zip(ii, vv)):
            c.annotate(f"{v*100:.1f}%", (xpos[j] + (k - 0.5) * w, il / 1e6),
                       ha="center", va="bottom", fontsize=6.5, rotation=90)
    c.set_xticks(xpos); c.set_xticklabels([f"x{s:g}" for s in spreads])
    c.set_xlabel("fibre spreading ratio s (kaisen)"); c.set_ylabel("ILSS [MPa]")
    c.set_title("spreading benefit is a THRESHOLD, set by viscosity\n(bar labels = void content)")
    c.legend(fontsize=8); c.grid(True, alpha=0.3, axis="y")
    c.set_ylim(0, max(sp[m_][1].max() for m_ in mus) / 1e6 * 1.3)

    # (d) ILSS process-window map
    d = ax[1, 1]
    im = d.contourf(hh * 1e6, tt, Z, levels=16, cmap="viridis")
    fig.colorbar(im, ax=d, label="ILSS [MPa]")
    cs = d.contour(hh * 1e6, tt, Z, levels=[0.5 * ilss0 / 1e6, 0.9 * ilss0 / 1e6],
                   colors="w", linewidths=1.2)
    d.clabel(cs, fmt={0.5 * ilss0 / 1e6: "50% ILSS", 0.9 * ilss0 / 1e6: "90% ILSS"}, fontsize=8)
    d.set_xlabel("tow half-thickness h [µm]"); d.set_ylabel("time above melt t_proc [s]")
    d.set_title("process window read in STRENGTH units\n(same window as the void model, now actionable)")

    p0, pv, pr, pb, exh = quad
    both = "interface capacity fully consumed" if exh else f"{pb/1e6:.1f} MPa"
    fig.suptitle("CFRTP void -> interlaminar-strength coupling: impregnation defects become a "
                 f"strength knockdown (pristine {p0/1e6:.1f} | void-only {pv/1e6:.1f} | "
                 f"+ residual {tau_res/1e6:.0f} MPa: {both}) -- Daikin theme", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main()
