"""
srb3_prognosis_validation.py — VALIDATE the SRB-3 Stage-3 burst prognosis
(`srb3_motorcase.py`: SRB3Prognosis / residual_burst_pressure / burst_margin /
burst_growth_probability, MotorCaseConfig) against a REAL CZM finite-element
dataset, so the burst prognosis is not just a lumped-constant toy.

This is the direct analogue of `fairing_prognosis_validation.py`, but STRONGER:
the fairing FEM is a STATIC guided-wave DETECTION simulation at a fixed debond,
whereas the SRB-3 CZM dataset is a debond-GROWTH CRITICAL-PRESSURE dataset — for
each geometry it reports `pcr`, the cohesive-zone critical internal pressure at
which an initial debond `a0` grows UNSTABLY. So unlike the fairing, this DOES
touch growth-criticality (the onset pressure), not just detection.

GROUND-TRUTH CZM DATA
---------------------
`.../JSCES/2027/abaqus_srb3/srb3_dataset.csv` (495 rows). Columns:
  layup, location, a0_mm, interface_m, Ex_MPa, Ey_MPa, pcr_MPa, MEOP_MPa,
  margin (= pcr/MEOP), decision, status.
Here `a0_mm` = initial delamination/debond size, `pcr_MPa` = CZM critical
pressure for UNSTABLE growth (the FEM ground-truth SEVERITY: lower pcr = more
severe), `location` (junction/cyl/dome), `interface_m` (interface-depth ply) and
`layup` are the other GROUND-TRUTH geometry knobs. Loaded with numpy/csv and
guarded by os.path.exists + a synthetic fallback so --test runs without the file.

What this validates (mirrors the fairing's 4 points, honestly)
--------------------------------------------------------------
  1. FEM SEVERITY SIGNATURE — from the CZM data, FEM severity is DECREASING in
     pcr: s_fem = -pcr (equivalently 1/margin). Lower critical pressure = a more
     severe / closer-to-growth configuration.

  2. MONOTONICITY + GEOMETRY SENSITIVITY — holding location+interface+layup
     fixed and sweeping a0_mm, pcr DECREASES as a0 grows (more initial damage ->
     lower critical pressure). Verified on the dominant representative group.
     SENSITIVITY: pcr/margin also differ by `location` (junction vs cyl vs dome)
     and by `interface_m` (interface depth) — the analogue of fairing position
     sensitivity.

  3. PROGNOSIS CONSISTENCY (the headline) — map the CZM a0_mm onto the prognosis
     damage variable `a` (linear, order-preserving calibration into the
     MotorCaseConfig a range), compute the prognosis severity proxies —
     residual_burst_pressure(a) (DECREASING) and burst_growth_probability(a,p_op)
     (INCREASING) — and rank-correlate the FEM pcr ordering against the prognosis
     severity ordering across the a0 sweep (one representative group fixed). The
     CZM severity (-pcr) and the prognosis severity (P(grow)) should be strongly
     POSITIVELY rank-correlated (both rise as a0 grows). Report Spearman rho +
     pairwise agreement.

  4. LIGHT CALIBRATION — report the implied prognosis `a` range for the CZM
     a0_mm range.

LIMITATION (honest): this validates that the burst-prognosis driving force is
monotone-CONSISTENT with the CZM critical-pressure ordering (right trend, right
ordering, sensitivity to geometry). The CZM gives the growth-ONSET pressure, so
this DOES touch growth-criticality (stronger than the fairing's static-detection
validation). But the absolute Paris/ERR rate constants in the burst model are
still REPRESENTATIVE (lumped, not coupon-calibrated): the CZM certifies the
ordering of growth onset, not the cyclic da/dN rate. That residual calibration
needs a cyclic burst-growth test / rate-resolved CZM (future work).

Usage
-----
    python srb3_prognosis_validation.py            # validation report
    python srb3_prognosis_validation.py --fig
    python srb3_prognosis_validation.py --test
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

import srb3_motorcase as srb3
from srb3_motorcase import (MotorCaseConfig, residual_burst_pressure,
                            burst_growth_probability)
# reuse the fairing rank utilities (no SciPy) to avoid duplication
from fairing_prognosis_validation import (_rankdata, spearman, rank_agreement)

HERE = os.path.dirname(os.path.abspath(__file__))
CZM_CSV = ("/home/nishioka/LUHsummer26/40_Academic/JSCES/2027/"
           "abaqus_srb3/srb3_dataset.csv")

# the representative controlled-comparison group: fix layup/location/interface/
# MEOP, sweep a0_mm only. (Chosen because every a0 level is present here — the
# clean a0-monotonicity group, the analogue of the fairing center-debond sweep.)
REP_GROUP = dict(layup="L0", location="junction", interface_m=3, MEOP_MPa=9.0)


# ═════════════════════════════════════════════════════════════════════════════
#  CZM dataset I/O (numpy/csv only) + synthetic fallback
# ═════════════════════════════════════════════════════════════════════════════

def _load_czm(path: str = CZM_CSV) -> list[dict]:
    """Load the SRB-3 CZM dataset CSV -> list of row dicts with typed fields.
    Numeric columns are floats; layup/location/decision/status stay strings."""
    num = {"a0_mm", "interface_m", "Ex_MPa", "Ey_MPa", "pcr_MPa", "MEOP_MPa",
           "margin"}
    rows: list[dict] = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            row = {}
            for k, v in r.items():
                if k in num:
                    try:
                        row[k] = float(v)
                    except (TypeError, ValueError):
                        row[k] = np.nan
                else:
                    row[k] = v
            rows.append(row)
    return rows


def synthetic_czm(n_a0: int = 10, rng: np.random.Generator | None = None
                  ) -> list[dict]:
    """CZM-LIKE fallback dataset for when the JSCES csv is absent (so --test
    runs). Across a single representative group, pcr DECREASES with a0_mm (more
    initial debond -> lower unstable-growth pressure), with location/interface
    offsets to mimic the geometry sensitivity. margin = pcr/MEOP."""
    rng = rng or np.random.default_rng(0)
    a0_grid = np.linspace(40.0, 300.0, n_a0)
    meop = 9.0
    loc_off = {"junction": 0.0, "cyl": 4.0, "dome": -1.0}    # cyl strongest
    int_off = {2: -4.0, 3: 0.0, 4: 7.0}                      # deeper -> tougher
    rows: list[dict] = []
    for loc in ("junction", "cyl", "dome"):
        for iface in (2, 3, 4):
            for a0 in a0_grid:
                # pcr decreasing in a0 (~1/sqrt(a)-like), + geometry offsets
                pcr = (28.0 * (40.0 / a0) ** 0.45
                       + loc_off[loc] + int_off[iface]
                       + rng.normal(0, 0.05))
                pcr = float(max(pcr, 1.0))
                rows.append(dict(layup="L0", location=loc, a0_mm=float(a0),
                                 interface_m=float(iface), Ex_MPa=58564.0,
                                 Ey_MPa=73237.0, pcr_MPa=pcr, MEOP_MPa=meop,
                                 margin=pcr / meop,
                                 decision="OK", status="grew"))
    return rows


def _filter_group(rows: list[dict], group: dict) -> list[dict]:
    """Select rows matching the fixed-knob group, sorted by a0_mm ascending."""
    def match(r):
        for k, v in group.items():
            rv = r.get(k)
            if isinstance(v, str):
                if rv != v:
                    return False
            else:
                if not np.isclose(float(rv), float(v)):
                    return False
        return True
    sub = [r for r in rows if match(r)]
    sub.sort(key=lambda r: r["a0_mm"])
    return sub


# ═════════════════════════════════════════════════════════════════════════════
#  Geometry sensitivity: pcr/margin by location and by interface depth
# ═════════════════════════════════════════════════════════════════════════════

def geometry_sensitivity(rows: list[dict]) -> dict:
    """Mean pcr (and margin) grouped by `location` and by `interface_m`, holding
    layup fixed (L0). The analogue of the fairing's position sensitivity: does
    the CZM critical pressure depend on WHERE the debond is and HOW DEEP?"""
    base = [r for r in rows if r.get("layup") == "L0"]

    def _by(key):
        agg: dict = {}
        for r in base:
            agg.setdefault(r[key], []).append(r["pcr_MPa"])
        return {k: float(np.mean(v)) for k, v in sorted(agg.items(),
                                                        key=lambda kv: str(kv[0]))}

    by_loc = _by("location")
    by_iface = _by("interface_m")
    # span (max-min) = strength of the sensitivity
    return {"pcr_by_location": by_loc, "pcr_by_interface": by_iface,
            "location_span": (max(by_loc.values()) - min(by_loc.values())
                              if by_loc else 0.0),
            "interface_span": (max(by_iface.values()) - min(by_iface.values())
                               if by_iface else 0.0)}


# ═════════════════════════════════════════════════════════════════════════════
#  Calibration: map CZM a0_mm -> prognosis damage variable a (order-preserving)
# ═════════════════════════════════════════════════════════════════════════════

def calibrate_a0(a0_mm: np.ndarray, cfg: MotorCaseConfig | None = None) -> dict:
    """Linearly map CZM a0_mm onto the prognosis damage variable a in
    [a_lo, a_hi] within the MotorCaseConfig a range, preserving ordering, so the
    prognosis severity ordering matches the CZM a0 sweep. Reports the implied a
    range only — absolute Paris/ERR rate constants are NOT calibrated (see module
    LIMITATION)."""
    cfg = cfg or MotorCaseConfig()
    a0 = np.asarray(a0_mm, float)
    lo, hi = float(a0.min()), float(a0.max())
    a_lo, a_hi = 0.05, 0.9 * cfg.a_max          # stay inside [0, a_max]
    if hi - lo < 1e-9:
        a = np.full_like(a0, 0.5 * (a_lo + a_hi))
    else:
        a = a_lo + (a0 - lo) / (hi - lo) * (a_hi - a_lo)
    return {"a": a, "a_min": float(a.min()), "a_max": float(a.max()),
            "a0_mm_min": lo, "a0_mm_max": hi}


# ═════════════════════════════════════════════════════════════════════════════
#  Driver: extract CZM severity, run the prognosis, rank-correlate
# ═════════════════════════════════════════════════════════════════════════════

def run_validation(verbose: bool = True) -> dict:
    cfg = MotorCaseConfig()
    have_czm = os.path.exists(CZM_CSV)
    rows = _load_czm(CZM_CSV) if have_czm else synthetic_czm()

    # 1) representative controlled group: fix knobs, sweep a0_mm ................
    rep = _filter_group(rows, REP_GROUP)
    if len(rep) < 3:    # fallback group can't be matched -> relax to synthetic
        rows = synthetic_czm()
        rep = _filter_group(rows, REP_GROUP)
        have_czm = False
    a0_mm = np.array([r["a0_mm"] for r in rep], float)
    pcr = np.array([r["pcr_MPa"] for r in rep], float)
    margin = np.array([r["margin"] for r in rep], float)
    meop = float(rep[0]["MEOP_MPa"])

    # FEM severity = decreasing in pcr (lower critical pressure = more severe)
    s_fem = -pcr                                  # equivalently 1/margin ordering

    # 2) monotonicity: pcr DECREASES as a0 grows (within the rep group) .........
    order = np.argsort(a0_mm)
    pcr_sorted = pcr[order]
    # rank-correlation pcr vs a0 should be strongly NEGATIVE
    mono_rho = spearman(a0_mm, pcr)
    pcr_decreases = mono_rho < 0
    geo = geometry_sensitivity(rows)

    # 3) prognosis consistency: calibrate a0 -> a, severity proxies, rank agree .
    cal = calibrate_a0(a0_mm, cfg)
    a_cal = cal["a"]
    p_op = cfg.p_op
    prog_pburst = residual_burst_pressure(a_cal, cfg)          # DECREASING in a
    prog_Pgrow = np.array([burst_growth_probability(np.full(8, ai), p_op, cfg)
                           for ai in a_cal])                   # INCREASING in a
    # prognosis SEVERITY = decreasing burst pressure / increasing P(grow).
    # Present everything as SEVERITY orderings so agreement is POSITIVE:
    #   FEM severity  = -pcr                    (rises with a0)
    #   prog severity = -p_burst  and  +P(grow) (both rise with a0)
    ra_pburst = rank_agreement(s_fem, -prog_pburst)
    ra_Pgrow = rank_agreement(s_fem, prog_Pgrow)
    # raw pcr-vs-P(grow) rho (expected strongly NEGATIVE: pcr down, P(grow) up)
    rho_pcr_Pgrow = spearman(pcr, prog_Pgrow)

    out = dict(have_czm=have_czm, rep_group=REP_GROUP, n_rep=len(rep),
               a0_mm=a0_mm, pcr=pcr, margin=margin, meop=meop, s_fem=s_fem,
               mono_rho=mono_rho, pcr_decreases=bool(pcr_decreases),
               pcr_sorted=pcr_sorted, geo=geo, cal=cal, a_cal=a_cal,
               prog_pburst=prog_pburst, prog_Pgrow=prog_Pgrow,
               ra_pburst=ra_pburst, ra_Pgrow=ra_Pgrow,
               rho_pcr_Pgrow=rho_pcr_Pgrow, cfg=cfg)
    if verbose:
        _report(out)
    return out


def _report(d: dict):
    bar = "=" * 76
    print(bar)
    print("  SRB-3 BURST PROGNOSIS — VALIDATION vs REAL CZM critical-pressure FEM")
    print(bar)
    print(f"  CZM source : {CZM_CSV}")
    print(f"  CZM found  : {d['have_czm']}"
          + ("" if d["have_czm"] else "  (FALLBACK to synthetic CZM-like data)"))
    g = d["rep_group"]
    print(f"  rep. group : layup={g['layup']} location={g['location']} "
          f"interface={g['interface_m']} MEOP={g['MEOP_MPa']} MPa  "
          f"(n={d['n_rep']} a0 levels)")

    print("\n  (1) FEM severity signature  s_fem = -pcr  (lower pcr = more severe):")
    for a0, pc, mg in zip(d["a0_mm"], d["pcr"], d["margin"]):
        print(f"    a0={a0:5.0f}mm   pcr={pc:6.2f} MPa   margin={mg:5.3f}")

    print("\n  (2) Monotonicity + geometry sensitivity (the CZM trend):")
    print(f"    pcr DECREASES as a0 grows -> {d['pcr_decreases']}  "
          f"(Spearman rho(a0,pcr)={d['mono_rho']:+.3f})")
    print(f"      pcr[a0_min]={d['pcr_sorted'][0]:.2f} -> "
          f"pcr[a0_max]={d['pcr_sorted'][-1]:.2f} MPa")
    geo = d["geo"]
    loc_s = "  ".join(f"{k}={v:.2f}" for k, v in geo["pcr_by_location"].items())
    int_s = "  ".join(f"d{int(k)}={v:.2f}" for k, v in geo["pcr_by_interface"].items())
    print(f"    location sensitivity (mean pcr): {loc_s}   "
          f"span={geo['location_span']:.2f} MPa")
    print(f"    interface sensitivity (mean pcr): {int_s}   "
          f"span={geo['interface_span']:.2f} MPa")

    print("\n  (3) CZM severity ordering  vs  prognosis severity ordering:")
    print(f"    calibrated a0(mm) sweep -> prognosis a sweep "
          f"[{d['cal']['a_min']:.3f}, {d['cal']['a_max']:.3f}]")
    print(f"    p_burst(a) : severity rho={d['ra_pburst']['spearman']:+.3f}  "
          f"pairwise agree={d['ra_pburst']['pairwise_agree']*100:.0f}% "
          f"({d['ra_pburst']['n_pairs']} pairs)")
    print(f"    P(grow)    : severity rho={d['ra_Pgrow']['spearman']:+.3f}  "
          f"pairwise agree={d['ra_Pgrow']['pairwise_agree']*100:.0f}%")
    print(f"    raw rho(pcr, P(grow)) = {d['rho_pcr_Pgrow']:+.3f}  "
          f"(expected strongly NEGATIVE: pcr down as P(grow) up)")

    print("\n  (4) Light calibration CZM a0(mm) -> prognosis a:")
    print(f"    a0 in [{d['cal']['a0_mm_min']:.0f}, {d['cal']['a0_mm_max']:.0f}] mm"
          f"  ->  a in [{d['cal']['a_min']:.3f}, {d['cal']['a_max']:.3f}]")

    print("\n  LIMITATION: the CZM gives the growth-ONSET critical pressure, so this")
    print("  DOES touch growth-criticality (stronger than the fairing's static")
    print("  detection). It validates monotone CONSISTENCY of the burst-prognosis")
    print("  driving force with the CZM pcr ordering (right trend, ordering, geometry")
    print("  sensitivity). The absolute Paris/ERR cyclic-rate constants remain")
    print("  representative (not coupon-calibrated) — future work.")
    print(bar)


# ═════════════════════════════════════════════════════════════════════════════
#  Figure
# ═════════════════════════════════════════════════════════════════════════════

def make_figure(d: dict, out_path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import sys
    sys.path.insert(0, os.path.join(HERE, "slides", "figure_sources"))
    try:
        from thesis_style import use
        figsize = use(width_frac=1.0, aspect=0.40)
    except Exception:
        figsize = (11.0, 4.0)
    import matplotlib.pyplot as plt

    a0 = d["a0_mm"]
    order = np.argsort(a0)
    a0o = a0[order]

    def _norm(v):
        v = np.asarray(v, float); rng = v.max() - v.min()
        return (v - v.min()) / rng if rng > 1e-30 else np.zeros_like(v)

    pcr_n = _norm(d["pcr"])[order]
    pb_n = _norm(d["prog_pburst"])[order]
    pg_n = _norm(d["prog_Pgrow"])[order]

    fig, ax = plt.subplots(1, 2, figsize=figsize)

    # (a) CZM pcr vs a0 overlaid with prognosis residual-burst / P(grow)
    ax[0].plot(a0o, pcr_n, "-o", ms=6, lw=1.8, c="#1565C0",
               label="CZM $p_{cr}$ (FEM)")
    ax[0].plot(a0o, pb_n, "--s", ms=5, lw=1.5, c="#2e7d32",
               label="prognosis $p_{burst}(a)$")
    ax[0].plot(a0o, pg_n, ":^", ms=5, lw=1.5, c="#b71c1c",
               label="prognosis $P$(grow)")
    ax[0].set_xlabel("initial debond $a_0$ (mm)", fontsize=8)
    ax[0].set_ylabel("normalised", fontsize=8)
    ax[0].set_title("(a) CZM $p_{cr}$ vs prognosis (a$_0$ sweep)", fontsize=9)
    ax[0].legend(fontsize=6); ax[0].tick_params(labelsize=7)
    ax[0].grid(alpha=0.3)

    # (b) rank-agreement scatter: FEM severity rank vs prognosis severity rank
    fr = _rankdata(d["s_fem"]); pr = _rankdata(d["prog_Pgrow"])
    n = len(fr)
    ax[1].plot([1, n], [1, n], "k--", lw=0.8, alpha=0.6)
    ax[1].scatter(fr, pr, s=70, c="#1565C0", zorder=3)
    for i, a0v in enumerate(a0):
        ax[1].annotate(f"{a0v:.0f}", (fr[i], pr[i]), fontsize=6,
                       xytext=(3, 3), textcoords="offset points")
    ax[1].set_xlabel("CZM severity rank ($-p_{cr}$)", fontsize=8)
    ax[1].set_ylabel("prognosis $P$(grow) rank", fontsize=8)
    ax[1].set_title(f"(b) rank agreement  $\\rho$="
                    f"{d['ra_Pgrow']['spearman']:+.2f}, "
                    f"{d['ra_Pgrow']['pairwise_agree']*100:.0f}% pairs", fontsize=9)
    ax[1].tick_params(labelsize=7); ax[1].grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    fig.savefig(os.path.splitext(out_path)[0] + ".png", bbox_inches="tight",
                dpi=150)
    plt.close(fig)
    return out_path


# ═════════════════════════════════════════════════════════════════════════════
#  Unit tests
# ═════════════════════════════════════════════════════════════════════════════

def run_tests() -> int:
    n = 0

    def ok(cond, msg):
        nonlocal n
        assert cond, msg
        n += 1

    cfg = MotorCaseConfig()

    # synthetic-fallback loader returns finite pcr/a0 arrays ...................
    rows = synthetic_czm()
    ok(len(rows) > 0, "synthetic CZM returns rows")
    pcr = np.array([r["pcr_MPa"] for r in rows]); a0 = np.array([r["a0_mm"] for r in rows])
    ok(np.all(np.isfinite(pcr)) and np.all(np.isfinite(a0)),
       "synthetic pcr/a0 finite")
    ok(np.all(pcr > 0), "synthetic pcr positive")

    # filtering the representative group returns an a0 sweep ...................
    rep = _filter_group(rows, REP_GROUP)
    ok(len(rep) >= 3, "rep group has >=3 a0 levels")
    a0r = np.array([r["a0_mm"] for r in rep])
    ok(np.all(np.diff(a0r) >= 0), "rep group sorted ascending by a0")

    # within the synthetic rep group pcr decreases as a0 grows ................
    pcr_r = np.array([r["pcr_MPa"] for r in rep])
    ok(spearman(a0r, pcr_r) < 0, "synthetic: pcr decreases with a0")

    # rank utilities (imported from fairing) behave ...........................
    ok(abs(spearman([1, 2, 3], [3, 2, 1]) + 1.0) < 1e-9, "spearman reversal=-1")
    ra = rank_agreement(np.array([1.0, 2, 3]), np.array([1.0, 2, 3]))
    ok(abs(ra["spearman"] - 1.0) < 1e-9 and ra["pairwise_agree"] == 1.0,
       "perfect rank agreement")

    # calibration preserves a0 ordering + stays within MotorCaseConfig a range .
    cal = calibrate_a0(np.array([40.0, 100.0, 300.0]), cfg)
    ok(cal["a_min"] >= 0.0 and cal["a_max"] <= cfg.a_max, "calibrated a in range")
    ok(cal["a"][0] < cal["a"][-1], "calibration preserves a0 ordering")

    # prognosis proxies monotone in a .........................................
    a_grid = np.array([0.05, 0.2, 0.4, 0.6])
    pb = residual_burst_pressure(a_grid, cfg)
    ok(np.all(np.diff(pb) < 0), "residual burst pressure DECREASES in a")
    pg = np.array([burst_growth_probability(np.full(8, ai), cfg.p_op, cfg)
                   for ai in a_grid])
    ok(np.all(np.diff(pg) >= 0), "P(grow) non-decreasing in a")

    # geometry sensitivity returns finite spans ...............................
    geo = geometry_sensitivity(rows)
    ok(geo["location_span"] >= 0 and geo["interface_span"] >= 0,
       "geometry sensitivity spans finite/non-negative")
    ok(len(geo["pcr_by_location"]) >= 2, "multiple locations present")

    # full validation runs (synthetic fallback if csv absent) + sane outputs ..
    v = run_validation(verbose=False)
    ok(-1.0 <= v["ra_Pgrow"]["spearman"] <= 1.0, "P(grow) rank corr in [-1,1]")
    ok(-1.0 <= v["ra_pburst"]["spearman"] <= 1.0, "p_burst rank corr in [-1,1]")
    ok(np.all(np.isfinite(v["s_fem"])) and np.all(np.isfinite(v["prog_Pgrow"])),
       "FEM + prognosis severities finite")
    ok(v["a_cal"].min() >= 0.0 and v["a_cal"].max() <= cfg.a_max,
       "calibrated a sweep within range")

    # on the REAL csv (if present): pcr decreases with a0 in rep group + high
    # severity rank agreement .................................................
    if v["have_czm"]:
        ok(v["pcr_decreases"], "REAL csv: pcr decreases with a0 in rep group")
        ok(abs(v["ra_Pgrow"]["spearman"]) > 0.8,
           "REAL csv: severity rank agreement |rho|>0.8")
        ok(v["rho_pcr_Pgrow"] < -0.8,
           "REAL csv: pcr vs P(grow) strongly negative")

    tag = "" if v["have_czm"] else "  (NOTE: CZM csv absent -> synthetic fallback)"
    print(f"srb3_prognosis_validation: {n}/{n} unit tests passed{tag}")
    return n


def main():
    ap = argparse.ArgumentParser(
        description="Validate the SRB-3 burst prognosis vs the real CZM "
                    "critical-pressure FEM dataset.")
    ap.add_argument("--fig", action="store_true")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.test:
        run_tests(); return
    d = run_validation()
    if args.fig:
        p = make_figure(d, os.path.join(HERE, "paper_figs",
                                        "srb3_prognosis_validation.pdf"))
        print(f"\nfigure -> {p}")


if __name__ == "__main__":
    main()
