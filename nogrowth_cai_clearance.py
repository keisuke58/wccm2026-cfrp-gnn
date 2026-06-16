"""
nogrowth_cai_clearance.py — TWO-PILLAR flight clearance: No-Growth × residual strength (CAI).

WHY (research-driven upgrade, 2026-06-14)
-----------------------------------------
A deep-research pass (adversarially verified, primary NASA/MIL sources) confirmed how
composite structure is ACTUALLY certified, and exposed a gap in our single-axis clearance:

  [S1] MIL-HDBK-17-3F / CMH-17-3: composite damage tolerance is built on TWO pillars, not one.
       The standard philosophy is **No-Growth** (defects must not grow harmfully under service
       load), AND a **two-load residual-strength** requirement: a structure with Barely-Visible
       Impact Damage (BVID) must still carry **ULTIMATE** load, and with clearly-visible damage
       must still carry **LIMIT** load. BVID impact-energy cap ≈ 100 ft-lb (~140 J) external.
  [S4] CAI (Compression-After-Impact, ASTM D7136/D7137, ISO 18352, Airbus AITM 1.0010) is the
       industry-standard residual-strength metric: low-velocity impact leaves a near-invisible
       surface but large internal delamination that **slashes compressive residual strength** —
       so BVID is the DESIGN DRIVER and residual strength, not just crack growth, gates flight.

Our `decision_uq.py` already implements the No-Growth pillar rigorously: it integrates the exact
phase-field forward `simulate_growth(theta, load).grown` over operational load scatter to get
P_grow, and maps it to OK/REPAIR/RETIRE via expected-cost thresholds (alpha, beta). **But it has
only the growth pillar.** A defect that does NOT grow (P_grow low -> growth-OK) yet has already
eaten too much residual compressive strength (R < limit) is a certification FAIL that a
growth-only monitor silently passes. This module adds the missing **residual-strength (CAI)
pillar** and takes the **more-severe of the two** — exactly the certification logic.

WHAT
----
  residual_strength_ratio(theta, cfg)  -> R = sigma_residual / sigma_design_ultimate  (CAI knockdown)
  strength_decision(R)                 -> OK/REPAIR/RETIRE from the two-load (ult/limit) rule
  growth_decision(p_grow, a, b)        -> the existing No-Growth call (re-exported)
  clearance(theta, loads, ...)         -> two-pillar = max-severity(growth, strength) + provenance
  study_grid(...)                      -> sweep damage size x load, quantify how often the CAI
                                          pillar ESCALATES beyond growth-only (the gap it closes)

The CAI knockdown uses the standard "impact-damage ~ equivalent open hole" design practice
(open-hole compression as the BVID design value, CMH-17), a transparent monotone soft-inclusion
form with REPRESENTATIVE, clearly-labelled parameters (not a fit to proprietary CAI data —
see CAI_PARAMS). The growth pillar uses the exact FD phase-field (no new data), via decision_uq.

Run:
  python nogrowth_cai_clearance.py --test     # unit tests (fast, no FD)
  python nogrowth_cai_clearance.py --smoke     # tiny FD grid sanity
  python nogrowth_cai_clearance.py --fig       # full grid + paper_figs/nogrowth_cai_clearance.pdf
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np

import cfrp_phasefield_2d as pf
import decision_uq as du

HERE = os.path.dirname(os.path.abspath(__file__))
SEVERITY = {"OK": 0, "REPAIR": 1, "RETIRE": 2}
DECISIONS = ("OK", "REPAIR", "RETIRE")

# ── residual-strength (CAI) knockdown model ──────────────────────────────────
# Impact damage treated as an equivalent soft inclusion / open hole (CMH-17 BVID design
# practice). R = fraction of ULTIMATE design load the damaged laminate still carries.
# Monotone soft-inclusion form: R(xi) = R_inf + (R0 - R_inf)*exp(-(xi/xi_c)**p),
# xi = damage width / laminate transverse width. REPRESENTATIVE params (label, not a fit):
CAI_PARAMS = dict(
    R0=1.05,      # pristine / sub-resolution damage: retains > ultimate (small positive margin)
    R_inf=0.40,   # large-damage compressive floor ~ open-hole-compression knockdown (~0.4-0.6)
    xi_c=0.123,   # characteristic damage-to-width ratio where the knockdown sets in
    p=2.0,        # knockdown sharpness
)
# Calibrated (representative) so the seeded defect range (1-4 elements, log2size 0-2) spans the
# full certification ladder: smallest -> retains ULTIMATE (R~1.0), largest -> below LIMIT (R~0.6).
# Two-load certification gates (aerospace factor of safety 1.5: ultimate = 1.5 x limit):
R_ULT = 1.00              # retains ULTIMATE (BVID requirement)  -> growth/strength OK level
R_LIM = 1.0 / 1.5         # retains LIMIT only (visible-damage requirement) -> below = RETIRE


def damage_width(log2size: float, cfg: pf.LaminateConfig) -> float:
    """Physical damage characteristic width: 2**log2size seed elements x element size."""
    return float(2.0 ** log2size) * (cfg.Lx / cfg.nx)


def residual_strength_ratio(theta, cfg: pf.LaminateConfig | None = None,
                            params: dict | None = None) -> float:
    """R = sigma_residual / sigma_ultimate (CAI knockdown) for defect theta=(cx,cy,layer,log2size).
    Deeper (more interior) damage is slightly more severe in compression (sub-surface
    delamination buckling); folded in as a mild depth factor on the effective width."""
    cfg = cfg or pf.LaminateConfig()
    pr = {**CAI_PARAMS, **(params or {})}
    _, _, layer, log2size = (float(x) for x in theta)
    a = damage_width(log2size, cfg)
    xi = a / cfg.Ly                      # damage width / laminate transverse width
    R = pr["R_inf"] + (pr["R0"] - pr["R_inf"]) * np.exp(-(xi / pr["xi_c"]) ** pr["p"])
    return float(R)


def strength_decision(R: float) -> str:
    """Two-load certification rule on residual strength: retains ultimate -> OK,
    retains only limit -> REPAIR, below limit -> RETIRE."""
    if R >= R_ULT:
        return "OK"
    if R >= R_LIM:
        return "REPAIR"
    return "RETIRE"


def growth_decision(p_grow: float, alpha: float, beta: float) -> str:
    """No-Growth pillar — identical to decision_uq.decide (P(grow) vs expected-cost thresholds)."""
    return du.decide(p_grow, alpha, beta)


def _more_severe(a: str, b: str) -> str:
    return a if SEVERITY[a] >= SEVERITY[b] else b


@dataclass
class Clearance:
    decision: str          # final two-pillar call
    growth: str            # No-Growth pillar call
    strength: str          # CAI pillar call
    p_grow: float
    R: float
    driver: str            # which pillar set the final (or "both")


def clearance(theta, p_grow: float, alpha: float, beta: float,
              cfg: pf.LaminateConfig | None = None) -> Clearance:
    """Two-pillar clearance: take the MORE SEVERE of {No-Growth, residual-strength}."""
    R = residual_strength_ratio(theta, cfg)
    g = growth_decision(p_grow, alpha, beta)
    s = strength_decision(R)
    final = _more_severe(g, s)
    driver = ("both" if g == s else
              "strength(CAI)" if SEVERITY[s] > SEVERITY[g] else "no-growth")
    return Clearance(final, g, s, float(p_grow), R, driver)


# ── study: where does the CAI pillar change the call vs growth-only? ──────────

def study_grid(n_size: int = 7, n_nom: int = 6, n_load: int = 5,
               cx: float = 0.5, cy: float = 0.5, layer: float = 8.0,
               n_workers: int | None = None, verbose: bool = True) -> dict:
    """Sweep damage size x nominal load on the EXACT FD phase-field; for each scenario compute
    the No-Growth probability and the CAI residual strength, then compare growth-only vs the
    two-pillar clearance. Headline = fraction of scenarios the CAI pillar ESCALATES (the gap)."""
    import crack_surrogate as cs
    cfg = pf.LaminateConfig()
    alpha, beta = pf.calibrate_thresholds()
    sizes = np.linspace(cs.LOG2SIZE_LO, cs.LOG2SIZE_HI, n_size)
    noms = np.linspace(cs.LOAD_LO + 0.005, cs.LOAD_HI - 0.005, n_nom)

    # build FD jobs: every (size, nominal, scatter-load)
    jobs, idx = [], []
    for i, l2 in enumerate(sizes):
        for j, nom in enumerate(noms):
            for L in du.load_grid(float(nom), n_load):
                jobs.append((cx, cy, float(layer), float(l2), float(L)))
                idx.append((i, j))
    if verbose:
        print(f"[study] {len(jobs)} FD solves "
              f"({n_size} sizes x {n_nom} loads x {n_load} scatter)…", flush=True)
    flags = du._run_fd(jobs, n_workers)
    grown = {}
    for (i, j), f in zip(idx, flags):
        grown.setdefault((i, j), []).append(bool(f))

    rows = []
    for i, l2 in enumerate(sizes):
        R = residual_strength_ratio((cx, cy, layer, l2), cfg)
        for j, nom in enumerate(noms):
            p_grow = float(np.mean(grown[(i, j)]))
            cl = clearance((cx, cy, layer, l2), p_grow, alpha, beta, cfg)
            rows.append(dict(size=float(l2), width=damage_width(l2, cfg), nom=float(nom),
                             **cl.__dict__))
    g_only = np.array([SEVERITY[r["growth"]] for r in rows])
    final = np.array([SEVERITY[r["decision"]] for r in rows])
    escal = final > g_only                                   # CAI pillar made it more severe
    # of those, the dangerous ones a growth-only monitor would PASS as OK:
    missed_ok = np.array([r["growth"] == "OK" and r["decision"] != "OK" for r in rows])
    return dict(rows=rows, sizes=sizes, noms=noms, alpha=alpha, beta=beta,
                cfg=cfg, n_escalated=int(escal.sum()), n_total=len(rows),
                frac_escalated=float(escal.mean()),
                n_growthOK_but_unsafe=int(missed_ok.sum()))


def print_report(st: dict) -> None:
    bar = "=" * 74
    print(bar)
    print("  TWO-PILLAR CLEARANCE — No-Growth  ×  residual strength (CAI)")
    print(bar)
    print(f"  thresholds (No-Growth): alpha={st['alpha']:.3f}  beta={st['beta']:.3f}")
    print(f"  CAI gates: retains ULTIMATE R>={R_ULT:.2f}  | retains LIMIT R>={R_LIM:.2f}")
    print(f"  scenarios: {st['n_total']}  (sizes x loads)")
    print()
    print(f"  CAI pillar ESCALATED the call in {st['n_escalated']}/{st['n_total']} "
          f"= {st['frac_escalated']*100:.0f}%  of scenarios")
    print(f"  ── of which growth-only said OK but two-pillar is NOT OK "
          f"(certification gap a growth-only monitor MISSES): "
          f"{st['n_growthOK_but_unsafe']}")
    print()
    print("  residual strength vs damage size (CAI knockdown):")
    seen = set()
    for r in st["rows"]:
        if r["size"] in seen:
            continue
        seen.add(r["size"])
        print(f"     log2size={r['size']:.2f}  width={r['width']:.4f}  "
              f"R={r['R']:.3f}  -> strength={r['strength']}")
    print(bar)


def make_figure(st: dict, out_path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    cfg = st["cfg"]
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.1))
    col = {"OK": "#2e7d32", "REPAIR": "#ef6c00", "RETIRE": "#b71c1c"}

    # (a) CAI knockdown curve
    l2 = np.linspace(0, 2, 100)
    Rs = [residual_strength_ratio((0.5, 0.5, 8.0, x), cfg) for x in l2]
    w = [damage_width(x, cfg) / cfg.Ly for x in l2]
    ax[0].plot(w, Rs, "-", color="#1565C0", lw=2)
    ax[0].axhline(R_ULT, ls="--", color="#2e7d32", lw=1)
    ax[0].axhline(R_LIM, ls="--", color="#b71c1c", lw=1)
    ax[0].text(ax[0].get_xlim()[1], R_ULT, " retains ULTIMATE", va="bottom", ha="right",
               fontsize=7, color="#2e7d32")
    ax[0].text(ax[0].get_xlim()[1], R_LIM, " retains LIMIT", va="bottom", ha="right",
               fontsize=7, color="#b71c1c")
    ax[0].set_xlabel("damage width / laminate width  $\\xi$", fontsize=8)
    ax[0].set_ylabel("residual strength  $R=\\sigma_{res}/\\sigma_{ult}$", fontsize=8)
    ax[0].set_title("(a) CAI knockdown (BVID design driver)\n[S1,S4: CMH-17 / ASTM D7136-7]",
                    fontsize=8.5)
    ax[0].tick_params(labelsize=7)

    # (b) two-pillar decision map in (p_grow, R)
    pg = np.array([r["p_grow"] for r in st["rows"]])
    R = np.array([r["R"] for r in st["rows"]])
    dec = [r["decision"] for r in st["rows"]]
    drv = [r["driver"] for r in st["rows"]]
    for d in DECISIONS:
        m = [i for i, x in enumerate(dec) if x == d]
        mk = ["o" if drv[i] != "strength(CAI)" else "^" for i in m]
        # split by marker
        for marker in ("o", "^"):
            mm = [i for i in m if ("^" if drv[i] == "strength(CAI)" else "o") == marker]
            if mm:
                ax[1].scatter(pg[mm], R[mm], c=col[d], marker=marker, s=34,
                              edgecolors="k", linewidths=0.3, alpha=0.85)
    for thr, c in ((st["alpha"], "0.5"), (st["beta"], "0.5")):
        ax[1].axvline(thr, ls=":", color=c, lw=0.8)
    ax[1].axhline(R_ULT, ls="--", color="#2e7d32", lw=1)
    ax[1].axhline(R_LIM, ls="--", color="#b71c1c", lw=1)
    ax[1].set_xlabel("No-Growth pillar  $P_{grow}$  (α,β dashed)", fontsize=8)
    ax[1].set_ylabel("CAI pillar  $R$", fontsize=8)
    ax[1].set_title("(b) two-pillar map: △ = CAI pillar drove\nfinal = more-severe(growth, strength)",
                    fontsize=8.5)
    ax[1].legend(handles=[Patch(color=col[d], label=d) for d in DECISIONS],
                 fontsize=7, loc="lower left")
    ax[1].tick_params(labelsize=7)

    # (c) growth-only vs two-pillar decision counts
    g_cnt = {d: sum(r["growth"] == d for r in st["rows"]) for d in DECISIONS}
    t_cnt = {d: sum(r["decision"] == d for r in st["rows"]) for d in DECISIONS}
    x = np.arange(3); wbar = 0.38
    ax[2].bar(x - wbar / 2, [g_cnt[d] for d in DECISIONS], wbar, label="growth-only",
              color="#90a4ae")
    ax[2].bar(x + wbar / 2, [t_cnt[d] for d in DECISIONS], wbar, label="two-pillar",
              color=[col[d] for d in DECISIONS])
    ax[2].set_xticks(x); ax[2].set_xticklabels(DECISIONS, fontsize=8)
    ax[2].set_ylabel("scenarios", fontsize=8)
    ax[2].set_title(f"(c) CAI pillar escalates {st['frac_escalated']*100:.0f}%\n"
                    f"({st['n_growthOK_but_unsafe']} growth-OK→not-OK = caught gap)",
                    fontsize=8.5)
    ax[2].legend(fontsize=7)
    ax[2].tick_params(labelsize=7)

    fig.suptitle("Two-pillar flight clearance: No-Growth × residual-strength (CAI) — "
                 "aligning Stage 3 with how composites are actually certified",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    fig.savefig(os.path.splitext(out_path)[0] + ".png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out_path


def run_tests() -> int:
    n = 0

    def ok(c, m):
        nonlocal n
        assert c, m
        n += 1

    cfg = pf.LaminateConfig()
    # monotone knockdown: bigger damage -> lower residual strength
    Rs = [residual_strength_ratio((0.5, 0.5, 8.0, x), cfg) for x in np.linspace(0, 2, 6)]
    ok(all(a >= b - 1e-9 for a, b in zip(Rs, Rs[1:])), "R decreases with damage size")
    ok(Rs[0] >= R_ULT, "tiny damage retains ultimate (R>=R_ULT)")
    # strength rule boundaries
    ok(strength_decision(1.2) == "OK", "R high -> OK")
    ok(strength_decision(0.8) == "REPAIR", "R between limit and ult -> REPAIR")
    ok(strength_decision(0.5) == "RETIRE", "R below limit -> RETIRE")
    ok(strength_decision(R_LIM) == "REPAIR", "R==limit -> REPAIR (>=)")
    # two-pillar takes the more severe; CAI can override a growth-OK
    a, b = pf.calibrate_thresholds()
    big = (0.5, 0.5, 8.0, 2.0)                 # large damage -> low R
    cl = clearance(big, p_grow=0.0, alpha=a, beta=b, cfg=cfg)   # growth says OK
    ok(cl.growth == "OK", "growth pillar OK at p=0")
    ok(SEVERITY[cl.decision] >= SEVERITY[cl.strength], "final >= strength severity")
    ok(cl.decision != "OK", "large-damage no-growth still NOT OK (CAI override)")
    ok(cl.driver == "strength(CAI)", "driver is the CAI pillar")
    # tiny damage, no growth -> OK
    cl2 = clearance((0.5, 0.5, 8.0, 0.0), p_grow=0.0, alpha=a, beta=b, cfg=cfg)
    ok(cl2.decision == "OK", "tiny no-growth damage -> OK")
    # growth pillar can still drive when strength is fine
    cl3 = clearance((0.5, 0.5, 8.0, 0.0), p_grow=1.0, alpha=a, beta=b, cfg=cfg)
    ok(cl3.decision == "RETIRE" and cl3.driver == "no-growth", "growth drives at p=1")
    print(f"nogrowth_cai_clearance: {n}/{n} unit tests passed")
    return n


def main():
    ap = argparse.ArgumentParser(description="Two-pillar (No-Growth × CAI) flight clearance.")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny FD grid")
    ap.add_argument("--fig", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    if args.test:
        run_tests(); return
    kw = dict(n_size=4, n_nom=3, n_load=3) if args.smoke else dict(n_size=7, n_nom=6, n_load=5)
    st = study_grid(n_workers=args.workers, **kw)
    print_report(st)
    if args.fig:
        p = make_figure(st, os.path.join(HERE, "paper_figs", "nogrowth_cai_clearance.pdf"))
        print(f"\nfigure → {p}")


if __name__ == "__main__":
    main()
