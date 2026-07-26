"""gaa_wfm_vth.py — work-function-metal (WFM) gate tuning of the threshold voltage
on a cylindrical GAA nanosheet cross-section. Maps the pilot-line WFM deposition
step (e.g. Applied Endura 3 E2CC, research/SEMICON_PILOT_LINE_GAA_PROCESS.md #12)
to a computable Vth, inside the same weak-form FE nonlinear-Poisson framework as
gaa_material_sweep_warmstart.py.

Physics (scaled equilibrium Poisson-Boltzmann on the disk, P1 FE weak form):

    -div(eps_r grad u) + kappa*sinh(u) = f_dop ,   u = u_gate_eff on the gate ring

The gate work-function metal sets a flat-band offset phi_ms = Phi_m - Phi_semi, so
the potential the semiconductor sees is u_gate_eff = (V_g - phi_ms)/Vt'. Different
WFM (different Phi_m) therefore shift the whole Q-V_g curve horizontally — i.e. they
shift Vth. n-type WFM (low Phi_m) lower Vth; p-type WFM (high Phi_m) raise it.

We sweep V_g for five representative WFM, extract Vth as the gate voltage at which
the integrated mobile (inversion) charge crosses a fixed onset Q*, and recover the
textbook ideal relation  dVth = dPhi_WFM  (slope 1) on the real GAA cross-section.

Honest scope: ideal flat-band-shift model — no interface traps, no poly depletion,
no quantum/degenerate statistics. For solver stability the demo uses an illustrative
thermal scale Vt' = 0.1 V (deep inversion under Boltzmann sinh is otherwise extreme);
the linear Vth-vs-WFM law is scaling-independent, which is the physical point.
Depends on gaa_material_sweep_warmstart.py (reuses its mesh/assembly/Newton).

Run:  python3 gaa_wfm_vth.py            (writes gaa_wfm_vth.png)
      python3 gaa_wfm_vth.py --help
"""
from __future__ import annotations

import argparse

import numpy as np

from gaa_material_sweep_warmstart import (
    MATERIALS, EPS_OX, C_DOP, kappa_of, build_disk_mesh, assemble, newton,
)

SEED = 20260725
VT_DEMO = 0.1          # illustrative volts-per-scaled-unit (see docstring)
PHI_SEMI = 4.60        # reference semiconductor work function [eV]
SI = 2                 # Si index in MATERIALS

# Representative replacement-metal-gate work-function metals (illustrative).
WFMS = [
    #  label                              Phi_m [eV]
    ("nMOS WFM ~4.2 eV (TiAl-rich)",       4.20),
    ("~4.4 eV",                            4.40),
    ("mid-gap 4.6 eV",                     4.60),
    ("~4.8 eV",                            4.80),
    ("pMOS WFM ~5.0 eV (TiN/Mo)",          5.00),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nr", type=int, default=16)
    ap.add_argument("--nt", type=int, default=48)
    ap.add_argument("--nvg", type=int, default=17, help="gate-voltage points")
    ap.add_argument("--vgmax", type=float, default=1.2, help="max gate voltage [V]")
    ap.add_argument("--out", type=str, default="gaa_wfm_vth.png")
    args = ap.parse_args()
    if args.nvg < 3:
        ap.error("--nvg must be >= 3 to extract a threshold")

    nodes, tris, gate, elem_semi = build_disk_mesh(args.nr, args.nt)
    free = ~gate
    N = len(nodes)

    # Si channel operator (eps in core, oxide ring; screening from Si n_i)
    name, eps_r, n_i = MATERIALS[SI]
    eps_elem = np.where(elem_semi, eps_r, EPS_OX)
    K, ML = assemble(nodes, tris, eps_elem, elem_semi)
    K = K.tocsr(); fdop = ML * C_DOP; kap = kappa_of(n_i)

    Vg = np.linspace(0.0, args.vgmax, args.nvg)          # volts
    charge = np.zeros((len(WFMS), args.nvg))             # integrated mobile charge
    fields = {}
    for m, (lab, phi_m) in enumerate(WFMS):
        phi_ms = phi_m - PHI_SEMI                        # flat-band offset [V]
        u0 = np.zeros(N)
        for k, vg in enumerate(Vg):
            u_gate = (vg - phi_ms) / VT_DEMO             # scaled effective gate potential
            u, _, _ = newton(K, ML, kap, fdop, u_gate, free, gate, u0)
            u0 = u                                       # bias continuation (idea C)
            charge[m, k] = float(np.sum(kap * ML * np.sinh(u)))
            fields[(m, k)] = u

    # onset threshold Q*: a fixed fraction of the mid-gap WFM's max charge
    mid = 2
    Qstar = 0.10 * charge[mid].max()

    def vth_of(qcurve):
        # first V_g where Q crosses Qstar (linear interpolation)
        for k in range(1, len(Vg)):
            if qcurve[k] >= Qstar:
                q0, q1 = qcurve[k - 1], qcurve[k]
                t = (Qstar - q0) / (q1 - q0 + 1e-30)
                return Vg[k - 1] + t * (Vg[k] - Vg[k - 1])
        return np.nan

    vth = np.array([vth_of(charge[m]) for m in range(len(WFMS))])
    phis = np.array([w[1] for w in WFMS])
    ok = np.isfinite(vth)
    slope, intercept = np.polyfit(phis[ok], vth[ok], 1)

    print(f"\nGAA WFM -> Vth (Si nanosheet cross-section, {len(WFMS)} WFM):")
    for m, (lab, phi_m) in enumerate(WFMS):
        print(f"  Phi_m {phi_m:.2f} eV  Vth {vth[m]:.3f} V   {lab}")
    print(f"  fit: dVth/dPhi_WFM = {slope:.3f} V/eV  (ideal flat-band = 1.000), "
          f"R offset {intercept:+.2f}")

    _plot(args.out, nodes, tris, elem_semi, Vg, charge, vth, phis, slope,
          fields, [w[0] for w in WFMS], [w[1] for w in WFMS], Qstar, args.nvg)
    print(f"wrote {args.out}")


def _plot(out, nodes, tris, elem_semi, Vg, charge, vth, phis, slope,
          fields, labels, phivals, Qstar, nvg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    triang = Triangulation(nodes[:, 0], nodes[:, 1], tris)
    fig, ax = plt.subplots(2, 3, figsize=(17, 10))

    # (0,0) cross-section material map
    ax[0, 0].tripcolor(triang, facecolors=elem_semi.astype(float), cmap="Pastel1",
                       edgecolors="k", linewidth=0.15)
    ax[0, 0].set_aspect("equal")
    ax[0, 0].set_title("GAA nanosheet cross-section\n(Si core = channel, ring = high-k/oxide)")

    # (0,1)(0,2) potential for nMOS-WFM vs pMOS-WFM at a mid-high V_g
    kk = min(nvg - 3, nvg - 1)
    for a, m, tt in ((ax[0, 1], 0, "n-type WFM (low Φ)"), (ax[0, 2], 4, "p-type WFM (high Φ)")):
        tp = a.tripcolor(triang, fields[(m, kk)], shading="gouraud", cmap="viridis")
        a.set_aspect("equal"); a.set_title(f"potential u  —  {tt}")
        fig.colorbar(tp, ax=a, fraction=0.046)

    # (1,0) Q-Vg family (transfer-like), one curve per WFM
    for m, lab in enumerate(labels):
        ax[1, 0].plot(Vg, charge[m], "-o", ms=3, label=lab)
    ax[1, 0].axhline(Qstar, color="0.4", ls="--", lw=1)
    ax[1, 0].text(Vg[0], Qstar, " Q* (onset)", va="bottom", fontsize=9, color="0.3")
    ax[1, 0].set_title("inversion charge vs gate voltage (per WFM)")
    ax[1, 0].set_xlabel("gate voltage V_g [V]"); ax[1, 0].set_ylabel("integrated mobile charge")
    ax[1, 0].legend(fontsize=8); ax[1, 0].grid(True, alpha=0.3)

    # (1,1) extracted Vth vs Phi_WFM (linear, slope ~1)
    ax[1, 1].plot(phis, vth, "o", ms=9, color="#6a3d9a")
    xs = np.linspace(min(phis) - 0.05, max(phis) + 0.05, 20)
    ax[1, 1].plot(xs, slope * xs + (vth[2] - slope * phis[2]), "-", color="#33a02c",
                  label=f"fit slope {slope:.2f} V/eV")
    ax[1, 1].set_title("WFM tunes threshold: ΔVth = ΔΦ_WFM  (ideal, slope≈1)")
    ax[1, 1].set_xlabel("WFM work function Φ_m [eV]"); ax[1, 1].set_ylabel("extracted Vth [V]")
    ax[1, 1].legend(fontsize=9); ax[1, 1].grid(True, alpha=0.3)

    # (1,2) design view: Vth window from nMOS to pMOS WFM
    ax[1, 2].barh(range(len(labels)), vth, color=["#1f77b4", "#5591c0", "#8f8f8f", "#c0714f", "#d62728"])
    ax[1, 2].set_yticks(range(len(labels)))
    ax[1, 2].set_yticklabels([f"{p:.1f} eV" for p in phivals], fontsize=10)
    ax[1, 2].invert_yaxis()
    ax[1, 2].set_xlabel("Vth [V]")
    ax[1, 2].set_title(f"Vth design window\n(single WFM choice sets Vth; span {np.nanmax(vth)-np.nanmin(vth):.2f} V)")
    for i, v in enumerate(vth):
        ax[1, 2].text(v, i, f" {v:.2f}", va="center", fontsize=9)

    fig.suptitle("WFM gate tuning of Vth on a GAA nanosheet cross-section "
                 "(maps the Endura-3 WFM step to a computed threshold; weak-form FE)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main()
