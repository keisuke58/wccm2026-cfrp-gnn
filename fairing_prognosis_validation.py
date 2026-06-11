"""
fairing_prognosis_validation.py — VALIDATE the fairing Stage-3 debond prognosis
(`fairing_debond_prognosis.py`) against the Payload2026 fairing guided-wave FEM,
so the prognosis is not just a lumped-constant toy (the #1 reviewer target).

What this validates (and what it honestly cannot)
-------------------------------------------------
The Payload2026 FEM (`Job-GW-*_sensors.csv`) is a STATIC guided-wave DETECTION
simulation at a FIXED debond — NOT a debond-GROWTH simulation. Therefore we
CANNOT validate the Paris growth law or the absolute strain-energy release rate
(ERR) constant `kappa`. We do NOT pretend to. What we CAN do, and do here:

  1. SEVERITY SIGNATURE — extract a scalar debond-severity signature from each
     FEM config's sensor signals. Two complementary forms:
       (a) transmission-loss signature  s = -log10( E_far / E_source )
           (wave energy reaching sensors PAST the debond, normalised to the
           source sensor) — baseline-free, defined for every config;
       (b) matched-baseline signal-difference RMS  s = ||U_dmg - U_healthy|| / ||U_healthy||
           where a layout-matched Healthy baseline exists (Healthy↔Debond pair).
     A larger / closer debond scatters more energy => stronger signature.

  2. MONOTONICITY — show the FEM signature increases with debond RADIUS
     (D2-r15 < D3-r40) and is sensitive to POSITION (D4-near > D5-edge).

  3. PROGNOSIS CONSISTENCY — the prognosis's ERR G(a) and P(grow) are monotone
     increasing in debond radius a. Rank-correlate the FEM severity ordering
     against the prognosis severity ordering across the shared debond configs
     (Spearman). Report the rank agreement.

  4. LIGHT CALIBRATION — map the FEM debond radii (r15, r40 mm) onto the
     prognosis radius `a` so the prognosis severity ordering matches the FEM;
     report the implied `a` range. The absolute Paris/ERR constants STILL need a
     debond-GROWTH experiment or growth-FEM — stated as the honest limitation /
     future work (see LIMITATION below).

LIMITATION (honest): this validation establishes that the prognosis driving
force is monotone-CONSISTENT with FEM debond severity (right trend, right
ordering, right position sensitivity). It does NOT certify absolute ERR or
Paris-law growth rates; those require cyclic debond-growth FEM / coupon tests.

Usage
-----
    python fairing_prognosis_validation.py            # validation report
    python fairing_prognosis_validation.py --fig
    python fairing_prognosis_validation.py --test
"""
from __future__ import annotations

import argparse
import os

import numpy as np

import fairing_debond_prognosis as fdp

HERE = os.path.dirname(os.path.abspath(__file__))
PAYLOAD_DIR = "/home/nishioka/Payload2026/abaqus_work"

# FEM configs we validate against. radius_mm / position encode the GROUND TRUTH
# debond geometry baked into each Abaqus job (None radius = qualitative only).
FEM_CONFIGS = {
    # name              file stem                radius_mm position
    "Healthy":  dict(stem="Job-GW-Healthy",  radius_mm=0.0,  position="none",  baseline=None),
    "Debond":   dict(stem="Job-GW-Debond",   radius_mm=25.0, position="center", baseline="Job-GW-Healthy"),
    "D2-r15":   dict(stem="Job-GW-D2-r15",   radius_mm=15.0, position="center", baseline=None),
    "D3-r40":   dict(stem="Job-GW-D3-r40",   radius_mm=40.0, position="center", baseline=None),
    "D4-near":  dict(stem="Job-GW-D4-near",  radius_mm=25.0, position="near",   baseline=None),
    "D5-edge":  dict(stem="Job-GW-D5-edge",  radius_mm=25.0, position="edge",   baseline=None),
}
# the size sweep used for the monotonicity / rank test (shared 5-sensor layout)
SWEEP_KEYS = ["D2-r15", "D3-r40"]                  # radius monotonicity
POSITION_KEYS = ["D4-near", "D5-edge"]             # position sensitivity


# ═════════════════════════════════════════════════════════════════════════════
#  FEM signal I/O + debond severity signature extraction
# ═════════════════════════════════════════════════════════════════════════════

def _load_fem(stem: str):
    """Load a Payload2026 sensor CSV: returns (t, U[ntime, nsensor], x_mm).
    Header line 1 = column names, line 2 = '# x_mm,<positions>'."""
    path = os.path.join(PAYLOAD_DIR, f"{stem}_sensors.csv")
    with open(path) as f:
        f.readline()                                   # column-name header
        xs = f.readline().strip().lstrip("# ").split(",")
    x_mm = np.array([float(v) for v in xs[1:]], dtype=float)
    d = np.genfromtxt(path, delimiter=",", skip_header=2)
    return d[:, 0], d[:, 1:], x_mm


def transmission_severity(U: np.ndarray) -> float:
    """Baseline-free debond-severity signature from a guided-wave record.

    s = -log10( E_far / E_source ),  E_source = energy at the source/near sensor
    (column 0), E_far = summed energy at sensors PAST the debond. A larger /
    closer debond scatters & blocks more energy => smaller transmitted fraction
    => LARGER s. Monotone-increasing in debond severity. Always finite."""
    E = np.sum(np.asarray(U, dtype=float) ** 2, axis=0)
    e_src = float(max(E[0], 1e-30))
    e_far = float(max(np.sum(E[1:]), 1e-30))
    trans = e_far / e_src
    return float(-np.log10(max(trans, 1e-30)))


def baseline_diff_severity(U: np.ndarray, U_base: np.ndarray) -> float:
    """Matched-baseline signal-difference RMS (presence-of-debond signature),
    used where a layout-matched Healthy baseline exists (Healthy↔Debond).
    s = ||U - U_base|| / ||U_base||. Larger residual => stronger debond."""
    U = np.asarray(U, float); U_base = np.asarray(U_base, float)
    num = float(np.sqrt(np.mean((U - U_base) ** 2)))
    den = float(max(np.sqrt(np.mean(U_base ** 2)), 1e-30))
    return num / den


def synthetic_fem_signal(radius_mm: float, n_t: int = 600, n_s: int = 5,
                         rng: np.random.Generator | None = None) -> np.ndarray:
    """FEM-LIKE fallback guided-wave record for when the Payload CSVs are absent
    (so --test still runs). A larger debond radius attenuates the transmitted
    wave more strongly: far-sensor amplitude decays as exp(-radius)."""
    rng = rng or np.random.default_rng(0)
    t = np.linspace(0, 1, n_t)
    pulse = np.sin(2 * np.pi * 5 * t) * np.exp(-((t - 0.2) / 0.05) ** 2)
    atten = np.exp(-0.04 * radius_mm)                  # more radius -> less transmission
    U = np.zeros((n_t, n_s))
    U[:, 0] = pulse                                    # source sensor (unattenuated)
    for j in range(1, n_s):
        U[:, j] = pulse * atten ** j * (0.6 ** j)
    return U


# ═════════════════════════════════════════════════════════════════════════════
#  Monotonicity / rank-agreement utilities
# ═════════════════════════════════════════════════════════════════════════════

def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-rank (ties shared), no SciPy dependency."""
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(1, len(a) + 1)
    # average tied ranks
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts)); np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation between two severity orderings. +1 = perfect
    agreement of the FEM and prognosis severity orderings."""
    rx, ry = _rankdata(x), _rankdata(y)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    den = np.sqrt(np.sum(rx ** 2) * np.sum(ry ** 2))
    if den < 1e-30:
        return 0.0
    return float(np.sum(rx * ry) / den)


def rank_agreement(fem_sev: np.ndarray, prog_sev: np.ndarray) -> dict:
    """Compare the FEM severity ordering vs the prognosis severity ordering.
    Returns Spearman rho and the fraction of pairwise orderings that agree."""
    fem_sev = np.asarray(fem_sev, float); prog_sev = np.asarray(prog_sev, float)
    rho = spearman(fem_sev, prog_sev)
    n = len(fem_sev); agree = tot = 0
    for i in range(n):
        for j in range(i + 1, n):
            tot += 1
            if np.sign(fem_sev[i] - fem_sev[j]) == np.sign(prog_sev[i] - prog_sev[j]):
                agree += 1
    return {"spearman": rho, "pairwise_agree": agree / max(tot, 1),
            "n_pairs": tot, "n_configs": n}


# ═════════════════════════════════════════════════════════════════════════════
#  Calibration: map FEM debond radii (mm) -> prognosis radius a
# ═════════════════════════════════════════════════════════════════════════════

def calibrate_radii(radii_mm: np.ndarray, cfg: fdp.FairingConfig | None = None) -> dict:
    """Linearly map FEM debond radii (mm) onto the prognosis radius a in
    [a_min, a_max], preserving ordering, so the prognosis severity ordering
    matches the FEM size sweep. Reports the implied a range only — absolute
    ERR/Paris constants are NOT calibrated (see module LIMITATION)."""
    cfg = cfg or fdp.FairingConfig()
    r = np.asarray(radii_mm, float)
    r_lo, r_hi = float(r.min()), float(r.max())
    a_lo, a_hi = 0.02, 0.9 * cfg.a_max
    if r_hi - r_lo < 1e-9:
        a = np.full_like(r, 0.5 * (a_lo + a_hi))
    else:
        a = a_lo + (r - r_lo) / (r_hi - r_lo) * (a_hi - a_lo)
    return {"a": a, "a_min": float(a.min()), "a_max": float(a.max()),
            "radius_mm_min": r_lo, "radius_mm_max": r_hi}


# ═════════════════════════════════════════════════════════════════════════════
#  Driver: extract FEM signatures, run the prognosis, rank-correlate
# ═════════════════════════════════════════════════════════════════════════════

def run_validation(load: float = 0.10, verbose: bool = True) -> dict:
    cfg = fdp.FairingConfig()
    have_fem = os.path.isdir(PAYLOAD_DIR) and os.path.exists(
        os.path.join(PAYLOAD_DIR, f"{FEM_CONFIGS['D2-r15']['stem']}_sensors.csv"))

    # 1) extract the FEM severity signature for every config we can find ........
    fem = {}
    for name, c in FEM_CONFIGS.items():
        path = os.path.join(PAYLOAD_DIR, f"{c['stem']}_sensors.csv")
        if have_fem and os.path.exists(path):
            _, U, _ = _load_fem(c["stem"])
            sev = transmission_severity(U)
            diff = None
            if c["baseline"]:
                bp = os.path.join(PAYLOAD_DIR, f"{c['baseline']}_sensors.csv")
                if os.path.exists(bp):
                    _, Ub, _ = _load_fem(c["baseline"])
                    if Ub.shape == U.shape:
                        diff = baseline_diff_severity(U, Ub)
            fem[name] = dict(severity=sev, diff_rms=diff, radius_mm=c["radius_mm"],
                             position=c["position"], synthetic=False)
        else:
            U = synthetic_fem_signal(c["radius_mm"])    # fallback
            fem[name] = dict(severity=transmission_severity(U), diff_rms=None,
                             radius_mm=c["radius_mm"], position=c["position"],
                             synthetic=True)

    # 2) size-sweep monotonicity (radius) + position sensitivity ...............
    mono_radius = fem["D3-r40"]["severity"] > fem["D2-r15"]["severity"]
    pos_sensitive = fem["D4-near"]["severity"] > fem["D5-edge"]["severity"]

    # 3) rank-agreement: FEM severity vs prognosis severity across debond configs.
    #    The prognosis is a function of RADIUS only, so the fair, controlled rank
    #    test fixes POSITION (center) and varies radius: D2-r15 < Debond(r25) <
    #    D3-r40. (Position is validated SEPARATELY in (2): D4-near > D5-edge.)
    rank_keys = ["D2-r15", "Debond", "D3-r40"]   # center-debond radius sweep
    rank_keys = [k for k in rank_keys if k in fem]
    fem_sev = np.array([fem[k]["severity"] for k in rank_keys])
    radii_mm = np.array([fem[k]["radius_mm"] for k in rank_keys])
    cal = calibrate_radii(radii_mm, cfg)
    a_cal = cal["a"]
    # prognosis severity proxies at the calibrated radii (both monotone in a)
    prog_G = np.array([fdp.energy_release_rate(ai, load, cfg) for ai in a_cal])
    prog_P = np.array([fdp.debond_growth_probability(np.full(8, ai), load, cfg)
                       for ai in a_cal])
    ra_G = rank_agreement(fem_sev, prog_G)
    ra_P = rank_agreement(fem_sev, prog_P)

    out = dict(load=load, have_fem=have_fem, fem=fem, rank_keys=rank_keys,
               fem_sev=fem_sev, radii_mm=radii_mm, a_cal=a_cal, cal=cal,
               prog_G=prog_G, prog_P=prog_P, ra_G=ra_G, ra_P=ra_P,
               mono_radius=bool(mono_radius), pos_sensitive=bool(pos_sensitive))
    if verbose:
        _report(out)
    return out


def _report(d: dict):
    bar = "=" * 74
    print(bar)
    print("  FAIRING DEBOND PROGNOSIS — VALIDATION vs Payload2026 guided-wave FEM")
    print(bar)
    print(f"  FEM source: {PAYLOAD_DIR}")
    print(f"  Payload CSVs found: {d['have_fem']}"
          + ("" if d["have_fem"] else "  (FALLBACK to synthetic FEM-like signals)"))
    print("\n  (1) FEM debond severity signature  s = -log10(E_far/E_source):")
    for name, c in FEM_CONFIGS.items():
        if name not in d["fem"]:
            continue
        f = d["fem"][name]
        extra = f"  diffRMS={f['diff_rms']:.3f}" if f["diff_rms"] is not None else ""
        tag = " [synthetic]" if f["synthetic"] else ""
        print(f"    {name:9s} r={f['radius_mm']:5.1f}mm {f['position']:7s} "
              f"s={f['severity']:.4f}{extra}{tag}")
    print("\n  (2) Monotonicity / position sensitivity (the FEM trend):")
    print(f"    radius:   D3-r40 > D2-r15  ->  {d['mono_radius']}  "
          f"({d['fem']['D3-r40']['severity']:.4f} vs {d['fem']['D2-r15']['severity']:.4f})")
    print(f"    position: D4-near > D5-edge ->  {d['pos_sensitive']}  "
          f"({d['fem']['D4-near']['severity']:.4f} vs {d['fem']['D5-edge']['severity']:.4f})")
    print("\n  (3) FEM severity ordering  vs  prognosis severity ordering:")
    print(f"    configs (by FEM radius): {d['rank_keys']}")
    print(f"    G(a)   : Spearman rho={d['ra_G']['spearman']:+.3f}  "
          f"pairwise agree={d['ra_G']['pairwise_agree']*100:.0f}% "
          f"({d['ra_G']['n_pairs']} pairs)")
    print(f"    P(grow): Spearman rho={d['ra_P']['spearman']:+.3f}  "
          f"pairwise agree={d['ra_P']['pairwise_agree']*100:.0f}%")
    print("\n  (4) Light calibration FEM radius(mm) -> prognosis a:")
    print(f"    r in [{d['cal']['radius_mm_min']:.0f}, {d['cal']['radius_mm_max']:.0f}] mm"
          f"  ->  a in [{d['cal']['a_min']:.3f}, {d['cal']['a_max']:.3f}]")
    print("\n  LIMITATION: FEM is STATIC detection at a FIXED debond — this validates")
    print("  monotone CONSISTENCY of the prognosis driving force with FEM severity,")
    print("  NOT absolute ERR/Paris growth. Those need a debond-GROWTH FEM/coupon")
    print("  test (future work).")
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

    keys = d["rank_keys"]
    fem_sev = d["fem_sev"]
    # normalise both orderings to [0,1] for overlay readability
    def _norm(v):
        v = np.asarray(v, float); rng = v.max() - v.min()
        return (v - v.min()) / rng if rng > 1e-30 else np.zeros_like(v)
    order = np.argsort(d["radii_mm"])
    ko = [keys[i] for i in order]
    fem_n = _norm(fem_sev)[order]
    G_n = _norm(d["prog_G"])[order]
    P_n = _norm(d["prog_P"])[order]
    xpos = np.arange(len(ko))

    fig, ax = plt.subplots(1, 2, figsize=figsize)
    # (a) FEM severity vs prognosis severity, ordered by debond size
    ax[0].plot(xpos, fem_n, "-o", ms=6, lw=1.8, c="#1565C0", label="FEM severity (transm. loss)")
    ax[0].plot(xpos, G_n, "--s", ms=5, lw=1.5, c="#2e7d32", label="prognosis ERR $G(a)$")
    ax[0].plot(xpos, P_n, ":^", ms=5, lw=1.5, c="#b71c1c", label="prognosis $P$(grow)")
    ax[0].set_xticks(xpos)
    ax[0].set_xticklabels([f"{k}\n{d['radii_mm'][keys.index(k)]:.0f}mm" for k in ko],
                          fontsize=7)
    ax[0].set_ylabel("severity (normalised)", fontsize=8)
    ax[0].set_title("(a) FEM vs prognosis severity ordering", fontsize=9)
    ax[0].legend(fontsize=6); ax[0].tick_params(labelsize=7)
    ax[0].grid(alpha=0.3)

    # (b) rank-agreement scatter: FEM rank vs prognosis rank
    fr = _rankdata(fem_sev); pr = _rankdata(d["prog_P"])
    ax[1].plot([1, len(keys)], [1, len(keys)], "k--", lw=0.8, alpha=0.6)
    ax[1].scatter(fr, pr, s=70, c="#1565C0", zorder=3)
    for i, k in enumerate(keys):
        ax[1].annotate(k, (fr[i], pr[i]), fontsize=6,
                       xytext=(3, 3), textcoords="offset points")
    ax[1].set_xlabel("FEM severity rank", fontsize=8)
    ax[1].set_ylabel("prognosis $P$(grow) rank", fontsize=8)
    ax[1].set_title(f"(b) rank agreement  $\\rho$={d['ra_P']['spearman']:+.2f}, "
                    f"{d['ra_P']['pairwise_agree']*100:.0f}% pairs", fontsize=9)
    ax[1].tick_params(labelsize=7); ax[1].grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    fig.savefig(os.path.splitext(out_path)[0] + ".png", bbox_inches="tight", dpi=150)
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

    # signature extractor returns finite scalars ...............................
    U_small = synthetic_fem_signal(10.0)
    U_big = synthetic_fem_signal(40.0)
    s_small = transmission_severity(U_small)
    s_big = transmission_severity(U_big)
    ok(np.isfinite(s_small) and np.isfinite(s_big), "severity is finite scalar")
    ok(np.isscalar(s_small) or np.ndim(s_small) == 0, "severity is scalar")

    # larger (synthetic) perturbation => larger signature ......................
    ok(s_big > s_small, "bigger debond -> larger transmission-loss severity")

    # baseline-diff severity finite + larger for bigger residual ...............
    base = synthetic_fem_signal(0.0)
    d_small = baseline_diff_severity(synthetic_fem_signal(10.0), base)
    d_big = baseline_diff_severity(synthetic_fem_signal(40.0), base)
    ok(np.isfinite(d_small) and d_big >= d_small,
       "baseline-diff severity finite + monotone in perturbation")

    # rank utilities on a toy ordering .........................................
    fem_toy = np.array([0.1, 0.2, 0.3, 0.4])
    prog_toy = np.array([1.0, 2.0, 3.0, 4.0])          # perfectly co-monotone
    ra = rank_agreement(fem_toy, prog_toy)
    ok(abs(ra["spearman"] - 1.0) < 1e-9, "toy perfect agreement -> rho=1")
    ok(ra["pairwise_agree"] == 1.0, "toy perfect pairwise agreement")
    ra_rev = rank_agreement(fem_toy, prog_toy[::-1])
    ok(abs(ra_rev["spearman"] + 1.0) < 1e-9, "reversed ordering -> rho=-1")
    ok(abs(spearman([1, 2, 3], [1, 2, 3]) - 1.0) < 1e-9, "spearman identity")

    # calibration preserves order + stays within prognosis a range ............
    cfg = fdp.FairingConfig()
    cal = calibrate_radii(np.array([15.0, 25.0, 40.0]), cfg)
    ok(cal["a_min"] >= 0.0 and cal["a_max"] <= cfg.a_max, "calibrated a in range")
    ok(cal["a"][0] < cal["a"][-1], "calibration preserves radius ordering")

    # prognosis severity proxies are monotone in a (consistency premise) .......
    a_grid = np.array([0.02, 0.06, 0.12, 0.18])
    G = np.array([fdp.energy_release_rate(a, 0.1, cfg) for a in a_grid])
    ok(np.all(np.diff(G) > 0), "prognosis G(a) monotone increasing in a")

    # full validation runs (synthetic fallback if FEM absent) + sane outputs ...
    v = run_validation(verbose=False)
    ok(set(v["fem"]) >= {"D2-r15", "D3-r40", "D4-near", "D5-edge"},
       "validation extracts the sweep configs")
    ok(-1.0 <= v["ra_P"]["spearman"] <= 1.0, "rank corr in [-1,1]")
    ok(np.all(np.isfinite(v["fem_sev"])), "all FEM severities finite")

    tag = "" if v["have_fem"] else "  (NOTE: Payload CSVs absent -> synthetic fallback)"
    print(f"fairing_prognosis_validation: {n}/{n} unit tests passed{tag}")
    return n


def main():
    ap = argparse.ArgumentParser(
        description="Validate the fairing debond prognosis vs Payload2026 FEM.")
    ap.add_argument("--load", type=float, default=0.10)
    ap.add_argument("--fig", action="store_true")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.test:
        run_tests(); return
    d = run_validation(args.load)
    if args.fig:
        p = make_figure(d, os.path.join(HERE, "paper_figs",
                                        "fairing_prognosis_validation.pdf"))
        print(f"\nfigure → {p}")


if __name__ == "__main__":
    main()
