"""
fairing_debond_prognosis.py — Stage-3 prognosis for the SECOND structure
(satellite fairing honeycomb skin-core DEBOND), and a pluggable, structure-
agnostic prognosis interface so the SAME decision rule serves any CFRP part.

Why this exists
---------------
The Stage 0-5 decision framework was end-to-end only on the perforated interstage
(matrix-crack delamination, FD anisotropic AT2 phase-field in cfrp_phasefield_2d).
The fairing is a DIFFERENT physics — facesheet/core interfacial DEBOND, not a
ply-level matrix crack — so the interstage phase-field does not apply. Rather than
caveat the fairing as "detection only", we give it its own prognosis (interfacial
fracture mechanics + Paris debond growth) that produces the SAME contract
(grown / rel_growth -> growth_probability -> OK/REPAIR/RETIRE via the shared
calibrate_thresholds). The framework then runs END-TO-END on BOTH structures.

Extensibility (future CFRP parts)
---------------------------------
Both structures implement a tiny `Prognosis` protocol
(`growth_probability(posterior, load) -> p`); `flight_clearance_generic` consumes
ANY prognosis. Adding a new CFRP component (shaft, panel, tank ...) = add one
prognosis model; the detect/characterise/decide/fleet/design machinery is reused.

Honest scope
------------
A representative interfacial-debond model: the strain-energy release rate for a
circular facesheet debond of radius a under a transverse load uses the clamped-
plate scaling G = kappa * load^2 * a^4 (a lumped kappa, trends-not-absolute — the
same stance as the interstage load proxy). Cyclic growth is Paris-law
da/dN = C (G/Gc)^m. Numbers are illustrative; the contribution is the
end-to-end, pluggable framework, not calibrated fairing certification values.

Usage
-----
    python fairing_debond_prognosis.py            # fairing clearance demo
    python fairing_debond_prognosis.py --fig
    python fairing_debond_prognosis.py --test
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Protocol

import numpy as np

import cfrp_phasefield_2d as pf   # reuse the SHARED decision rule

HERE = os.path.dirname(os.path.abspath(__file__))


# ═════════════════════════════════════════════════════════════════════════════
#  Fairing honeycomb skin-core debond model
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class FairingConfig:
    """Representative H3-class CFRP-facesheet / honeycomb-core debond config."""
    Gc_interface: float = 0.6     # interfacial fracture toughness (kJ/m^2, illustr.)
    kappa: float = 8.0e3          # lumped ERR constant: G = kappa * load^2 * a^a_exp
    a_exp: float = 2.0            # ERR exponent in debond radius (representative)
    paris_C: float = 1.0e-3       # Paris coefficient da/dN = C (G/Gc)^m
    paris_m: float = 3.0          # Paris exponent (default = representative)
    paris_m_sigma: float = 0.0    # >0: sample m~N(paris_m,sigma) per draw (real-data spread)
    cycles_per_flight: int = 100  # acoustic/inertial load cycles per flight
    a_init_frac: float = 1.0      # scales the seeded debond radius
    growth_rel_threshold: float = 0.5   # rel area growth per flight = "grown"
    a_max: float = 0.20           # facesheet patch half-size (debond saturates)

    @classmethod
    def from_calibration(cls, path: str | None = None, **kw):
        """Build a config whose Paris exponent is the REAL-DATA calibrated
        population posterior (4TU DCB mode-I; composite_fatigue_calibration.py).
        Uses the hierarchical mean as paris_m and the between-specimen sigma as
        paris_m_sigma, so the prognosis PROPAGATES the bridging spread rather than
        assuming a single point exponent (which fails leave-one-specimen-out)."""
        import json, os
        if path is None:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "results", "composite_fatigue_4tu", "calibration.json")
        a = json.load(open(path))
        return cls(paris_m=float(a["mu_m"]), paris_m_sigma=float(a["sigma_m"]), **kw)


def energy_release_rate(a: np.ndarray | float, load: float,
                        cfg: FairingConfig) -> np.ndarray | float:
    """Strain-energy release rate G at the debond front (clamped-facesheet
    blister scaling G = kappa * load^2 * a^a_exp). Monotone in a and load."""
    a = np.clip(a, 1e-6, cfg.a_max)
    return cfg.kappa * load ** 2 * a ** cfg.a_exp


def simulate_debond_growth(a0: float, load: float,
                           cfg: FairingConfig | None = None,
                           m_override: float | None = None) -> dict:
    """One flight (K cycles) of Paris interfacial-debond growth from radius a0.

    da/dN = C (G(a,load)/Gc)^m, integrated over K cycles (forward Euler in cycle
    count; a clipped to a_max). Returns the interstage-compatible contract:
    grown / rel_growth (debond AREA growth) plus a_final.
    `m_override` lets a caller inject a sampled Paris exponent (real-data posterior).
    """
    cfg = cfg or FairingConfig()
    m = cfg.paris_m if m_override is None else float(m_override)
    a = float(np.clip(a0, 1e-6, cfg.a_max))
    area0 = np.pi * a ** 2
    for _ in range(cfg.cycles_per_flight):
        G = energy_release_rate(a, load, cfg)
        da = cfg.paris_C * (G / cfg.Gc_interface) ** m
        a = min(a + da, cfg.a_max)
    area_final = np.pi * a ** 2
    rel = (area_final - area0) / max(area0, 1e-12)
    unstable = a >= 0.99 * cfg.a_max          # ran away to facesheet failure
    return {"grown": bool(unstable or rel > cfg.growth_rel_threshold),
            "rel_growth": float(rel), "a0": float(a0), "a_final": float(a),
            "G0": float(energy_release_rate(a0, load, cfg)),
            "unstable": bool(unstable)}


def debond_growth_probability(posterior_a: np.ndarray, load: float,
                              cfg: FairingConfig | None = None,
                              n_draws: int = 20,
                              rng: np.random.Generator | None = None) -> float:
    """P(debond grows next flight) over a Stage-2 posterior of debond radii —
    the fairing twin of cfrp_phasefield_2d.growth_probability."""
    cfg = cfg or FairingConfig()
    rng = rng or np.random.default_rng(0)
    a = np.atleast_1d(np.asarray(posterior_a, dtype=float)).ravel()
    if len(a) > n_draws:
        a = a[rng.choice(len(a), n_draws, replace=False)]
    # real-data calibrated exponent: sample m~N(mu,sigma) per draw to propagate
    # the between-specimen (fibre-bridging) spread; else fixed cfg.paris_m.
    if cfg.paris_m_sigma > 0:
        ms = np.clip(rng.normal(cfg.paris_m, cfg.paris_m_sigma, size=len(a)), 0.5, None)
        flags = [simulate_debond_growth(ai * cfg.a_init_frac, load, cfg, m_override=mi)["grown"]
                 for ai, mi in zip(a, ms)]
    else:
        flags = [simulate_debond_growth(ai * cfg.a_init_frac, load, cfg)["grown"]
                 for ai in a]
    return float(np.mean(flags))


# ═════════════════════════════════════════════════════════════════════════════
#  Pluggable, structure-agnostic prognosis interface  (extensibility)
# ═════════════════════════════════════════════════════════════════════════════

class Prognosis(Protocol):
    name: str

    def growth_probability(self, posterior: np.ndarray, load: float,
                           n_draws: int = 20) -> float: ...


@dataclass
class FairingPrognosis:
    """Stage-3 prognosis for the fairing (honeycomb debond)."""
    cfg: FairingConfig = None
    name: str = "fairing-debond"

    def growth_probability(self, posterior, load, n_draws: int = 20) -> float:
        return debond_growth_probability(posterior, load,
                                         self.cfg or FairingConfig(), n_draws)


@dataclass
class InterstagePrognosis:
    """Stage-3 prognosis for the interstage (FD phase-field delamination) —
    wraps the existing solver behind the same interface."""
    cfg: object = None
    name: str = "interstage-phasefield"

    def growth_probability(self, posterior, load, n_draws: int = 20) -> float:
        return pf.growth_probability(np.atleast_2d(posterior), load,
                                     cfg=self.cfg, n_draws=n_draws)


def flight_clearance_generic(prognosis: Prognosis, posterior: np.ndarray,
                             load: float, n_flights: int = 20,
                             costs: dict | None = None,
                             n_draws: int = 20) -> dict:
    """Structure-AGNOSTIC clearance: take ANY prognosis, get P(grow), apply the
    SHARED expected-cost decision rule. Identical decision semantics for the
    interstage, the fairing, and any future CFRP part."""
    alpha, beta = pf.calibrate_thresholds(**(costs or {}))
    p = prognosis.growth_probability(posterior, load, n_draws=n_draws)
    flights = np.arange(1, n_flights + 1)
    surv = (1.0 - p) ** flights
    p_s = (p * n_draws + 1.0) / (n_draws + 2.0)
    decision = "OK" if p <= alpha else "REPAIR" if p <= beta else "RETIRE"
    return {"structure": prognosis.name, "p_growth_next": float(p),
            "decision": decision, "alpha": alpha, "beta": beta,
            "p_survive_curve": surv, "p_survive_n": float(surv[-1]),
            "expected_remaining_flights": float((1.0 - p_s) / max(p_s, 1e-9))}


# ═════════════════════════════════════════════════════════════════════════════
#  Demo: the fairing, end-to-end through the shared decision rule
# ═════════════════════════════════════════════════════════════════════════════

def run_demo(load: float = 0.10, verbose: bool = True) -> dict:
    cfg = FairingConfig()
    prog = FairingPrognosis(cfg)
    rng = np.random.default_rng(0)
    # three debond severities (small / medium / large initial radius)
    cases = {"small a0=0.02": 0.02, "medium a0=0.06": 0.06, "large a0=0.11": 0.11}
    out = {}
    for name, a0 in cases.items():
        post = np.clip(rng.normal(a0, 0.1 * a0, 40), 1e-3, cfg.a_max)
        out[name] = flight_clearance_generic(prog, post, load)
    # a debond-radius sweep -> decision boundary
    radii = np.linspace(0.01, 0.15, 25)
    sweep_p = [debond_growth_probability(np.full(8, a), load, cfg) for a in radii]
    if verbose:
        _report(out, radii, sweep_p, load)
    return {"cases": out, "radii": radii, "sweep_p": sweep_p, "load": load}


def _report(out, radii, sweep_p, load):
    bar = "=" * 72
    print(bar)
    print("  FAIRING DEBOND PROGNOSIS  —  Stage-3 for the 2nd structure (end-to-end)")
    print(bar)
    a, b = pf.calibrate_thresholds()
    print(f"  honeycomb skin-core interfacial debond, Paris growth; load={load}")
    print(f"  SHARED decision rule: alpha={a:.2f}, beta={b:.2f} (same as interstage)\n")
    for name, r in out.items():
        print(f"  {name:>16}:  P(grow)={r['p_growth_next']:.3f}  "
              f"E[remaining]={r['expected_remaining_flights']:.1f}  "
              f"-> {r['decision']}")
    # decision boundary radii
    bnd = None
    for a_r, p in zip(radii, sweep_p):
        if p > a and bnd is None:
            bnd = a_r
    print(f"\n  P(grow) crosses the REPAIR threshold near debond radius "
          f"a≈{bnd:.3f}" if bnd else "\n  (no threshold crossing in sweep)")
    print("  => the fairing now runs END-TO-END through the same Stage-3->clear")
    print("     machinery as the interstage, via the pluggable Prognosis interface.")
    print(bar)


def make_figure(d: dict, out_path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import sys
    sys.path.insert(0, os.path.join(HERE, "slides", "figure_sources"))
    try:
        from thesis_style import use
        figsize = use(width_frac=1.0, aspect=0.32)
    except Exception:
        figsize = (12.0, 3.6)
    import matplotlib.pyplot as plt
    cfg = FairingConfig()
    a, b = pf.calibrate_thresholds()

    fig, ax = plt.subplots(1, 3, figsize=figsize)
    # (a) ERR G(a) vs Gc for a few loads
    radii = np.linspace(0.005, cfg.a_max, 100)
    for ld in (0.06, 0.10, 0.14):
        ax[0].plot(radii, [energy_release_rate(r, ld, cfg) for r in radii],
                   label=f"load {ld}")
    ax[0].axhline(cfg.Gc_interface, color="k", ls="--", lw=0.9, label="$G_c$")
    ax[0].set_xlabel("debond radius $a$", fontsize=8)
    ax[0].set_ylabel("$G$ (ERR)", fontsize=8); ax[0].set_yscale("log")
    ax[0].set_title("(a) interfacial driving force", fontsize=9)
    ax[0].legend(fontsize=6); ax[0].tick_params(labelsize=7)
    # (b) P(grow) vs debond radius with alpha/beta bands
    ax[1].plot(d["radii"], d["sweep_p"], "-o", ms=3, c="#1565C0")
    ax[1].axhline(a, color="#2e7d32", ls="--", lw=0.9, label=r"$\alpha$ (OK|REPAIR)")
    ax[1].axhline(b, color="#b71c1c", ls="--", lw=0.9, label=r"$\beta$ (REPAIR|RETIRE)")
    ax[1].set_xlabel("debond radius $a$", fontsize=8)
    ax[1].set_ylabel("P(grow next flight)", fontsize=8)
    ax[1].set_title("(b) clearance decision boundary", fontsize=9)
    ax[1].legend(fontsize=6); ax[1].tick_params(labelsize=7)
    # (c) survival curves for the three cases
    for name, r in d["cases"].items():
        ax[2].plot(np.arange(1, len(r["p_survive_curve"]) + 1),
                   r["p_survive_curve"], "-o", ms=2,
                   label=f"{name.split()[0]} → {r['decision']}")
    ax[2].set_xlabel("flight", fontsize=8); ax[2].set_ylabel("P(survive)", fontsize=8)
    ax[2].set_ylim(0, 1.02); ax[2].set_title("(c) survival per severity", fontsize=9)
    ax[2].legend(fontsize=6); ax[2].tick_params(labelsize=7)

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

    cfg = FairingConfig()
    # ERR monotonicity
    ok(energy_release_rate(0.1, 0.1, cfg) > energy_release_rate(0.05, 0.1, cfg),
       "G increases with debond radius")
    ok(energy_release_rate(0.1, 0.14, cfg) > energy_release_rate(0.1, 0.08, cfg),
       "G increases with load")

    # growth: a larger debond / higher load reaches a larger final radius
    small = simulate_debond_growth(0.02, 0.10, cfg)
    large = simulate_debond_growth(0.12, 0.10, cfg)
    ok(large["a_final"] >= small["a_final"], "larger debond → larger final radius")
    ok(not small["grown"] or large["grown"], "monotone grow flag")
    ok(simulate_debond_growth(0.06, 0.16, cfg)["a_final"] >=
       simulate_debond_growth(0.06, 0.06, cfg)["a_final"], "load drives growth")
    ok(simulate_debond_growth(0.05, 0.0, cfg)["rel_growth"] == 0.0,
       "no load → no growth")

    # probability + decision monotone in severity (P(grow) rises with radius)
    p_small = debond_growth_probability(np.full(8, 0.02), 0.10, cfg)
    p_large = debond_growth_probability(np.full(8, 0.12), 0.10, cfg)
    ok(0 <= p_small <= 1 and 0 <= p_large <= 1, "prob in range")
    ok(p_large >= p_small, "P(grow) rises with debond size")

    # pluggable interface: BOTH prognoses satisfy the protocol + shared rule
    fp = FairingPrognosis(cfg)
    r = flight_clearance_generic(fp, np.full(8, 0.12), 0.12)
    ok(r["decision"] in ("OK", "REPAIR", "RETIRE"), "fairing clearance verdict")
    ok(r["structure"] == "fairing-debond", "structure tag")
    ok(r["alpha"] == pf.calibrate_thresholds()[0], "shared alpha threshold")
    # a tiny debond at low load clears OK; a big one at high load does not
    ok(flight_clearance_generic(fp, np.full(8, 0.01), 0.05)["decision"] == "OK",
       "small debond / low load → OK")
    big = flight_clearance_generic(fp, np.full(8, 0.14), 0.14)["decision"]
    ok(big in ("REPAIR", "RETIRE"), "large debond / high load → not OK")

    # extensibility: the SAME generic clearance accepts a different prognosis
    class _Toy:
        name = "toy"
        def growth_probability(self, posterior, load, n_draws=20):
            return float(np.clip(np.mean(posterior), 0, 1))
    rt = flight_clearance_generic(_Toy(), np.array([0.9]), 0.1)
    ok(rt["decision"] == "RETIRE" and rt["structure"] == "toy",
       "generic clearance is prognosis-agnostic (future CFRP parts)")

    print(f"fairing_debond_prognosis: {n}/{n} unit tests passed")
    return n


def main():
    ap = argparse.ArgumentParser(
        description="Fairing honeycomb-debond Stage-3 prognosis (2nd structure).")
    ap.add_argument("--load", type=float, default=0.10)
    ap.add_argument("--fig", action="store_true")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.test:
        run_tests(); return
    d = run_demo(args.load)
    if args.fig:
        p = make_figure(d, os.path.join(HERE, "paper_figs",
                                        "fairing_debond_prognosis.pdf"))
        print(f"\nfigure → {p}")


if __name__ == "__main__":
    main()
