"""
run_all.py — single reproducible driver for the Stage 0-5 SHM decision framework.

Runs every module's `--test` (and optionally `--fig`), parses the
"N/N unit tests passed" line, and prints a stage-grouped summary table with
total test counts and wall-clock. Exit code is non-zero if ANY module fails,
so this doubles as CI. The module list is the curated framework surface,
ordered by pipeline stage so the output reads like the paper's structure.

Usage
-----
    python run_all.py                 # run all --test, summary table
    python run_all.py --fig           # also regenerate every module's figure
    python run_all.py --quick         # skip the slow (FD/FMPE) modules
    python run_all.py --only srb3     # only modules whose name contains 'srb3'
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# Curated framework surface, grouped by pipeline stage. `slow` modules touch the
# finite-difference solver / FMPE and take longer; --quick skips them.
GROUPS: list[tuple[str, list[tuple[str, bool]]]] = [
    ("Stage 0 — anomaly / robustness", [
        ("stage0_robustness", False),
    ]),
    ("Stage 1 — detection front-ends (3 structures)", [
        ("kojima_real_case", False),
        ("interstage_measured_detection", False),
        ("baselines_comparison", False),
        ("srb3_motorcase", False),            # AE detection + burst prognosis
    ]),
    ("Stage 2 — characterisation / domain adaptation", [
        ("tsa_sim2real", False),
        ("kojima_real_decision", False),      # labelled convex/concave on REAL data
        ("domain_adapt", False),
    ]),
    ("Stage 3 — prognosis + validation", [
        ("fairing_debond_prognosis", False),
        ("fairing_prognosis_validation", False),
        ("srb3_prognosis_validation", False),
        ("phasefield_3d", True),
        ("flight_load_spectrum", False),
        ("physics_validation", False),
        ("nasa_rul_validation", False),       # REAL run-to-failure (honest neg.)
        ("composite_fatigue_calibration", False),  # REAL Paris calibration [R]
    ]),
    ("Stage 3/clear — decision UQ (3 structures, symmetric)", [
        ("decision_uq", True),
        ("fairing_pipeline", False),
        ("srb3_decision_uq", False),
        ("cost_calibration", False),
    ]),
    ("Decision-core rigor (transfer / guarantee / VoI / lifetime)", [
        ("loso_decision_transfer", False),
        ("conformal_clearance", False),
        ("voi_inspection", False),
        ("lifetime_policy", False),           # sequential optimal-stopping
        ("pomdp_inspection", False),          # belief-state POMDP (+inspect action)
    ]),
    ("Stage 4/5 — fleet learning + design + system baseline", [
        ("system_baseline", False),
        ("cross_structure_fleet", False),     # 3-level fleet of fleets
        ("cross_structure_design", False),    # portfolio budget allocation
        ("qubo_maintenance", True),           # discrete fleet schedule as a QUBO (QA-ready)
        ("design_feedback", False),
    ]),
]

# two in-repo conventions: "N/N unit tests passed" and "N checks passed"
_PASS_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s+unit tests passed")
_CHECK_RE = re.compile(r"(\d+)\s+checks passed")


def _run_one(mod: str, do_fig: bool, timeout: int) -> dict:
    """Run `python3 <mod>.py --test` (+ --fig), parse the pass line."""
    rec = {"mod": mod, "passed": 0, "total": 0, "ok": False,
           "dt": 0.0, "note": "", "fig_ok": None}
    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, f"{mod}.py", "--test"],
                           cwd=HERE, capture_output=True, text=True,
                           timeout=timeout)
        m = c = None
        for line in r.stdout.splitlines():
            mm = _PASS_RE.search(line)
            if mm:
                m = mm
            cc = _CHECK_RE.search(line)
            if cc:
                c = cc
        if m:
            rec["passed"], rec["total"] = int(m.group(1)), int(m.group(2))
            rec["ok"] = (rec["passed"] == rec["total"] and r.returncode == 0)
        elif c:                                  # "N checks passed" convention
            rec["passed"] = rec["total"] = int(c.group(1))
            rec["ok"] = (r.returncode == 0)
        else:
            rec["note"] = "no pass-line" + (
                f" (rc={r.returncode})" if r.returncode else "")
            tail = (r.stderr or r.stdout).strip().splitlines()
            if tail:
                rec["note"] += f": {tail[-1][:60]}"
    except subprocess.TimeoutExpired:
        rec["note"] = f"TIMEOUT >{timeout}s"
    rec["dt"] = time.time() - t0

    if do_fig and rec["ok"]:
        try:
            rf = subprocess.run([sys.executable, f"{mod}.py", "--fig"],
                               cwd=HERE, capture_output=True, text=True,
                               timeout=timeout)
            rec["fig_ok"] = (rf.returncode == 0)
        except subprocess.TimeoutExpired:
            rec["fig_ok"] = False
    return rec


def main():
    ap = argparse.ArgumentParser(description="Run all SHM framework module tests.")
    ap.add_argument("--fig", action="store_true", help="also regenerate figures")
    ap.add_argument("--quick", action="store_true", help="skip slow FD/FMPE modules")
    ap.add_argument("--only", default=None, help="substring filter on module name")
    ap.add_argument("--timeout", type=int, default=300, help="per-module timeout (s)")
    args = ap.parse_args()

    bar = "=" * 76
    print(bar)
    print("  Stage 0-5 SHM decision framework — full test sweep")
    print(f"  fig={args.fig}  quick={args.quick}"
          + (f"  only~'{args.only}'" if args.only else ""))
    print(bar)

    grand_pass = grand_total = n_mod = n_fail = 0
    fig_made = fig_failed = 0
    t_start = time.time()
    fails: list[str] = []

    for title, mods in GROUPS:
        shown = [(m, slow) for (m, slow) in mods
                 if not (args.quick and slow)
                 and (args.only is None or args.only.lower() in m.lower())]
        if not shown:
            continue
        print(f"\n▸ {title}")
        for mod, slow in shown:
            if not os.path.exists(os.path.join(HERE, f"{mod}.py")):
                print(f"    {mod:34s}  MISSING")
                continue
            rec = _run_one(mod, args.fig, args.timeout)
            n_mod += 1
            grand_pass += rec["passed"]
            grand_total += rec["total"]
            tag = "ok " if rec["ok"] else "FAIL"
            slow_tag = " ·slow" if slow else ""
            cnt = f"{rec['passed']}/{rec['total']}" if rec["total"] else "  - "
            extra = f"  {rec['note']}" if rec["note"] else ""
            figtag = ""
            if rec["fig_ok"] is not None:
                figtag = "  fig:ok" if rec["fig_ok"] else "  fig:FAIL"
                fig_made += int(bool(rec["fig_ok"]))
                fig_failed += int(not rec["fig_ok"])
            print(f"    {mod:34s}  {tag}  {cnt:>9s}  {rec['dt']:5.1f}s"
                  f"{slow_tag}{figtag}{extra}")
            if not rec["ok"]:
                n_fail += 1
                fails.append(mod)

    dt = time.time() - t_start
    print("\n" + bar)
    print(f"  modules: {n_mod}   tests: {grand_pass}/{grand_total} passed   "
          f"failed modules: {n_fail}   wall: {dt:.1f}s")
    if args.fig:
        print(f"  figures: {fig_made} regenerated, {fig_failed} failed")
    if fails:
        print(f"  FAILED: {', '.join(fails)}")
    print(bar)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
