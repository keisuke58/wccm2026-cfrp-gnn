"""
lifetime_policy.py — the SEQUENTIAL (lifetime) retire/repair/fly decision as an
optimal-stopping dynamic program: "may the part fly again, AND FOR HOW MANY
FLIGHTS?" answered over the whole life, not one flight at a time.

Where this sits / why it is new
-------------------------------
Every other decision module makes a ONE-SHOT clearance call (OK/REPAIR/RETIRE)
for the NEXT flight from the current damage posterior (`decision_uq`,
`fairing_pipeline`, `srb3_decision_uq`).  A reusable vehicle, though, flies a
SEQUENCE of those decisions; "how many flights are left" is an OPTIMAL-STOPPING
problem — keep flying (earning mission value, growing the crack), refurbish, or
retire for salvage — solved by dynamic programming over the remaining life.  No
existing module does the time axis; this one does, and it turns the framework's
myopic per-flight clearance into the GROUND-TRUTH-comparable lifetime policy.

The Markov decision process (finite-horizon backward induction)
---------------------------------------------------------------
State  = (flight index f, crack size a).  Crack grows with the SAME surrogate
Paris/Carrara law as Stage-4 (`fleet_learning`): a' = a + r·Δσ^m·a^p, Δσ ~
N(load_amp, load_amp_sd).  Per flight the operator chooses:
  * FLY    — earn FLY_VALUE if it survives; with prob p_fail(a) the crack
             reaches a_fail mid-flight → catastrophic −C_FAIL and the life ends.
  * REPAIR — pay −C_REPAIR, reset the crack to A_REPAIR, fly on.
  * RETIRE — take salvage +S_RETIRE and stop (the optimal-stopping action).
Backward induction gives V_f(a) and the optimal action map, hence the
RETIREMENT BOUNDARY a*(f) and the expected number of flights to retirement
E[N*] — the quantitative "how many flights left".

What we show
------------
  1. The DP lifetime policy EXTRACTS MORE expected lifetime value at ~0 failures
     than (a) the framework's myopic per-flight clearance (`clearance_decision`),
     (b) the best fixed crack-threshold, and (c) fly-to-horizon — on Monte-Carlo
     rollouts.
  2. GROWTH-RATE UNCERTAINTY COSTS LIFETIME VALUE.  Planning with only a
     posterior on r (must hedge with a conservative high quantile) retires
     earlier and earns less than planning with the true r; the gap shrinks as
     the r posterior sharpens — directly linking Stage-4 fleet learning
     (`cross_structure_fleet`) to lifetime value (value of resolving r).

HONEST LIMITATION
-----------------
The growth law, a_fail, and the value/cost constants (FLY_VALUE, C_FAIL,
C_REPAIR, S_RETIRE) are REPRESENTATIVE (trends, not certified); the contribution
is the lifetime DECISION STRUCTURE and the two qualitative results (optimal
stopping beats myopic clearance; resolving growth-rate uncertainty buys lifetime
value), not certified retirement schedules. The MDP is fully observed in the
crack state (inspection noise folded into the growth-rate belief, not a POMDP
over a); the belief-state POMDP over a — with an explicit costly INSPECT action —
is realised in `pomdp_inspection.py`.

Usage
-----
    python lifetime_policy.py            # policy comparison + uncertainty study
    python lifetime_policy.py --fig
    python lifetime_policy.py --test
"""
from __future__ import annotations

import argparse
import os

import numpy as np

import fleet_learning as fl

HERE = os.path.dirname(os.path.abspath(__file__))

# Representative lifetime economics (trends, not certified — see LIMITATION).
# Repair is a SUBSTANTIAL refurbishment cost (not a free reset), so retirement is
# a genuine optimal-stopping trade-off rather than "always refurbish".
FLY_VALUE = 10.0          # value of one successful mission
C_FAIL = 200.0            # catastrophic in-flight failure
C_REPAIR = 25.0           # refurbishment turn (resets the crack) — a real cost
S_RETIRE = 30.0           # salvage value of a retired-but-intact vehicle
A_REPAIR = 0.10           # crack size after a refurbishment
F_MAX = 24                # design-life horizon (flights)
N_LOAD = 96               # load-scatter draws per DP cell
N_GRID = 80               # crack-size grid resolution


# ═════════════════════════════════════════════════════════════════════════════
#  Growth law (reuse the Stage-4 surrogate) + one-step transition
# ═════════════════════════════════════════════════════════════════════════════

def grow_step(a, r: float, loads: np.ndarray, cfg: fl.FleetConfig) -> np.ndarray:
    """Vectorised one-flight crack growth a' = a + r·load^m·a^p over load draws."""
    a = np.asarray(a, float)
    return a + r * (loads ** cfg.paris_m) * np.maximum(a, 0.0) ** cfg.paris_p


def _load_draws(cfg: fl.FleetConfig, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(cfg.load_amp + cfg.load_amp_sd * rng.standard_normal(N_LOAD),
                   1e-6, None)


# ═════════════════════════════════════════════════════════════════════════════
#  Dynamic program: backward induction → value + optimal action map
# ═════════════════════════════════════════════════════════════════════════════

ACTIONS = ("FLY", "REPAIR", "RETIRE")


def solve_dp(r: float, cfg: fl.FleetConfig | None = None, seed: int = 0) -> dict:
    """Finite-horizon backward induction for growth rate r. Returns the value
    table V[f, a], the optimal action map, the retirement boundary a*(f), and
    the per-grid crack axis."""
    cfg = cfg or fl.FleetConfig()
    a_grid = np.linspace(0.0, cfg.a_fail, N_GRID)
    loads = _load_draws(cfg, seed)
    V = np.zeros((F_MAX + 1, N_GRID))
    act = np.zeros((F_MAX, N_GRID), dtype=int)
    V[F_MAX, :] = S_RETIRE                      # terminal: retire for salvage

    def interp_next(vec, a_query):
        return np.interp(np.clip(a_query, 0.0, cfg.a_fail), a_grid, vec)

    for f in range(F_MAX - 1, -1, -1):
        Vnext = V[f + 1]
        v_repair = -C_REPAIR + float(np.interp(A_REPAIR, a_grid, Vnext))
        for i, a in enumerate(a_grid):
            a_next = grow_step(a, r, loads, cfg)
            fail = a_next >= cfg.a_fail
            p_fail = float(fail.mean())
            if (~fail).any():
                v_surv = float(np.mean(interp_next(Vnext, a_next[~fail])))
            else:
                v_surv = 0.0
            v_fly = (1 - p_fail) * (FLY_VALUE + v_surv) + p_fail * (-C_FAIL)
            v_retire = S_RETIRE
            choices = (v_fly, v_repair, v_retire)
            best = int(np.argmax(choices))
            V[f, i] = choices[best]
            act[f, i] = best
    boundary = _retire_boundary(act, a_grid)
    return {"V": V, "act": act, "a_grid": a_grid, "boundary": boundary,
            "r": r, "cfg": cfg, "loads": loads}


def _retire_boundary(act: np.ndarray, a_grid: np.ndarray) -> np.ndarray:
    """Smallest crack size at which RETIRE is optimal, per flight (a_fail if
    never)."""
    out = np.full(act.shape[0], a_grid[-1])
    for f in range(act.shape[0]):
        retire = np.where(act[f] == ACTIONS.index("RETIRE"))[0]
        if retire.size:
            out[f] = a_grid[retire[0]]
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  Policies (callable: (f, a, p_fail_next) -> action index) for MC rollout
# ═════════════════════════════════════════════════════════════════════════════

def dp_policy(dp: dict):
    a_grid = dp["a_grid"]; act = dp["act"]
    def pol(f, a, p_next):
        i = int(np.clip(np.searchsorted(a_grid, a), 0, len(a_grid) - 1))
        return act[min(f, F_MAX - 1), i]
    return pol


def clearance_policy(cfg: fl.FleetConfig, r_assumed: float, seed: int = 0):
    """The framework's MYOPIC per-flight clearance as a lifetime policy: act on
    the next-flight failure probability via clearance_decision (OK→FLY,
    REPAIR→REPAIR, RETIRE→RETIRE)."""
    loads = _load_draws(cfg, seed)
    a_grid = np.linspace(0.0, cfg.a_fail, N_GRID)
    # precompute next-flight p_fail(a) under the assumed rate
    pmap = np.array([float((grow_step(a, r_assumed, loads, cfg) >= cfg.a_fail).mean())
                     for a in a_grid])
    amap = {"OK": ACTIONS.index("FLY"), "REPAIR": ACTIONS.index("REPAIR"),
            "RETIRE": ACTIONS.index("RETIRE")}
    def pol(f, a, p_next):
        p = float(np.interp(min(a, cfg.a_fail), a_grid, pmap))
        return amap[fl.clearance_decision(p)]
    return pol


def fixed_threshold_policy(cfg: fl.FleetConfig, a_thr: float):
    def pol(f, a, p_next):
        return ACTIONS.index("RETIRE") if a >= a_thr else ACTIONS.index("FLY")
    return pol


def fly_to_horizon_policy(cfg: fl.FleetConfig):
    def pol(f, a, p_next):
        return ACTIONS.index("FLY")
    return pol


# ═════════════════════════════════════════════════════════════════════════════
#  Monte-Carlo policy evaluation
# ═════════════════════════════════════════════════════════════════════════════

def rollout(policy, r_true: float, cfg: fl.FleetConfig, n_vehicles: int = 4000,
            seed: int = 0) -> dict:
    """Simulate n_vehicles lives under `policy` at the TRUE growth rate; report
    mean lifetime value, failure rate, and mean flights flown."""
    rng = np.random.default_rng(seed)
    a0 = np.exp(cfg.log_a0_mean + cfg.log_a0_sd * rng.standard_normal(n_vehicles))
    vals = np.zeros(n_vehicles); flights = np.zeros(n_vehicles)
    failed = np.zeros(n_vehicles, dtype=bool)
    for v in range(n_vehicles):
        a = float(a0[v]); val = 0.0; nfl = 0
        for f in range(F_MAX):
            act = policy(f, a, None)
            if act == ACTIONS.index("RETIRE"):
                val += S_RETIRE; break
            if act == ACTIONS.index("REPAIR"):
                val -= C_REPAIR; a = A_REPAIR; continue
            # FLY
            load = max(cfg.load_amp + cfg.load_amp_sd * rng.standard_normal(), 1e-6)
            a_next = a + r_true * (load ** cfg.paris_m) * max(a, 0.0) ** cfg.paris_p
            if a_next >= cfg.a_fail:
                val -= C_FAIL; failed[v] = True; break
            val += FLY_VALUE; a = a_next; nfl += 1
        else:
            val += S_RETIRE                     # reached horizon intact → salvage
        vals[v] = val; flights[v] = nfl
    return {"value": float(vals.mean()), "value_sd": float(vals.std()),
            "fail_rate": float(failed.mean()), "flights": float(flights.mean())}


# ═════════════════════════════════════════════════════════════════════════════
#  Driver: compare policies + growth-rate-uncertainty cost
# ═════════════════════════════════════════════════════════════════════════════

def best_fixed_threshold(r_true: float, cfg: fl.FleetConfig, seed: int = 0
                         ) -> tuple:
    """Grid-search the single best fixed crack-retirement threshold."""
    best_thr, best_val = cfg.a_fail, -np.inf
    for thr in np.linspace(0.3, cfg.a_fail, 12):
        rv = rollout(fixed_threshold_policy(cfg, thr), r_true, cfg,
                     n_vehicles=1500, seed=seed)
        v = rv["value"] - (1e6 if rv["fail_rate"] > 0.02 else 0.0)
        if v > best_val:
            best_val, best_thr = v, thr
    return best_thr, best_val


def uncertainty_study(cfg: fl.FleetConfig, mu_r: float = -2.1,
                      post_sds=(0.45, 0.30, 0.18, 0.10, 0.05), seed: int = 0
                      ) -> dict:
    """Cost of growth-rate uncertainty: plan with a CONSERVATIVE high quantile of
    the r posterior (hedge), evaluate at the true r. As the posterior sharpens
    (smaller SD — what fleet learning delivers), the plan approaches the
    known-r optimum and earns more lifetime value."""
    r_true = float(np.exp(mu_r))
    known = solve_dp(r_true, cfg, seed=seed)
    v_known = rollout(dp_policy(known), r_true, cfg, n_vehicles=2500, seed=seed)["value"]
    vals = []
    for sd in post_sds:
        r_hi = float(np.exp(mu_r + 1.2816 * sd))      # ~90th percentile (hedge)
        dp_hi = solve_dp(r_hi, cfg, seed=seed)
        v = rollout(dp_policy(dp_hi), r_true, cfg, n_vehicles=2500, seed=seed)["value"]
        vals.append(v)
    return {"post_sds": list(post_sds), "value": vals, "v_known": v_known,
            "r_true": r_true}


def run_study(mu_r: float = -2.1, seed: int = 0, verbose: bool = True) -> dict:
    cfg = fl.FleetConfig()
    r_true = float(np.exp(mu_r))
    dp = solve_dp(r_true, cfg, seed=seed)

    policies = {
        "DP-optimal": dp_policy(dp),
        "myopic-clearance": clearance_policy(cfg, r_true, seed=seed),
        "fly-to-horizon": fly_to_horizon_policy(cfg),
    }
    thr, _ = best_fixed_threshold(r_true, cfg, seed=seed)
    policies["best-fixed-thr"] = fixed_threshold_policy(cfg, thr)

    evals = {name: rollout(pol, r_true, cfg, seed=seed)
             for name, pol in policies.items()}
    unc = uncertainty_study(cfg, mu_r=mu_r, seed=seed)
    out = {"dp": dp, "evals": evals, "uncertainty": unc, "r_true": r_true,
           "fixed_thr": thr, "cfg": cfg}
    if verbose:
        _report(out)
    return out


def _report(d: dict) -> None:
    bar = "=" * 78
    print(bar)
    print("  LIFETIME RETIREMENT POLICY — optimal-stopping DP "
          "(how many flights left?)")
    print(bar)
    print(f"  growth rate r={d['r_true']:.4f}/flight,  a_fail={d['cfg'].a_fail}, "
          f"horizon={F_MAX} flights")
    print(f"  economics: FLY +{FLY_VALUE}, FAIL -{C_FAIL}, REPAIR -{C_REPAIR}, "
          f"RETIRE +{S_RETIRE}\n")
    print(f"  {'policy':<20}{'life value':>12}{'fail rate':>11}{'flights':>9}")
    print("  " + "-" * 60)
    for name in ("DP-optimal", "myopic-clearance", "best-fixed-thr",
                 "fly-to-horizon"):
        e = d["evals"][name]
        print(f"  {name:<20}{e['value']:12.2f}{e['fail_rate']*100:10.2f}%"
              f"{e['flights']:9.1f}")
    opt = d["evals"]["DP-optimal"]["value"]
    myo = d["evals"]["myopic-clearance"]["value"]
    gain = (opt - myo) / abs(myo) * 100 if myo else 0.0
    print(f"\n  DP-optimal lifetime value {gain:+.1f}% vs the framework's myopic "
          f"per-flight clearance,")
    print(f"  at {d['evals']['DP-optimal']['fail_rate']*100:.2f}% failures; "
          f"E[flights to retire] ≈ {d['evals']['DP-optimal']['flights']:.1f}.")
    u = d["uncertainty"]
    cost0 = u["v_known"] - u["value"][0]
    cost1 = u["v_known"] - u["value"][-1]
    print(f"\n  COST OF GROWTH-RATE UNCERTAINTY (plan with a conservative hedge):")
    print(f"    known r           : life value {u['v_known']:.2f}")
    print(f"    posterior SD {u['post_sds'][0]:.2f} : {u['value'][0]:.2f}  "
          f"(−{cost0:.2f} vs known)")
    print(f"    posterior SD {u['post_sds'][-1]:.2f} : {u['value'][-1]:.2f}  "
          f"(−{cost1:.2f} vs known)")
    print(f"  → sharpening the growth-rate posterior (fleet learning) recovers "
          f"{cost0 - cost1:.2f} of")
    print(f"    lifetime value — the lifetime payoff of Stage-4.")
    print("\n  LIMITATION: representative growth law + value/cost constants; the "
          "result is the")
    print("  decision STRUCTURE (optimal stopping beats myopic clearance; "
          "resolving r buys")
    print("  lifetime value), not a certified retirement schedule. Fully-observed "
          "crack state.")
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
        figsize = (12.0, 4.0)
    import matplotlib.pyplot as plt

    dp = d["dp"]; cfg = d["cfg"]
    fig, ax = plt.subplots(1, 3, figsize=figsize)

    # (a) optimal action map + retirement boundary in (flight, crack) space.
    cmap = plt.matplotlib.colors.ListedColormap(["#2e7d32", "#f9a825", "#b71c1c"])
    ax[0].imshow(dp["act"].T, aspect="auto", origin="lower", cmap=cmap,
                 extent=[0, F_MAX, 0, cfg.a_fail], vmin=0, vmax=2,
                 interpolation="nearest")
    ax[0].plot(np.arange(F_MAX) + 0.5, dp["boundary"], "k-", lw=1.4,
               label="retire boundary $a^*(f)$")
    ax[0].set_xlabel("flight index $f$", fontsize=7)
    ax[0].set_ylabel("crack size $a$", fontsize=7)
    ax[0].set_title("(a) optimal policy map\n(green FLY / amber REPAIR / red "
                    "RETIRE)", fontsize=8)
    ax[0].legend(fontsize=5.5, loc="lower left"); ax[0].tick_params(labelsize=6)

    # (b) lifetime value by policy (fail rate annotated).
    order = ["DP-optimal", "myopic-clearance", "best-fixed-thr", "fly-to-horizon"]
    vals = [d["evals"][k]["value"] for k in order]
    frs = [d["evals"][k]["fail_rate"] for k in order]
    cols = ["#2e7d32", "#5c93c4", "#c4a35c", "#b71c1c"]
    x = np.arange(len(order))
    bars = ax[1].bar(x, vals, color=cols)
    for xi, v, fr in zip(x, vals, frs):
        ax[1].text(xi, v + (2 if v >= 0 else -6), f"{fr*100:.1f}%\nfail",
                   ha="center", fontsize=5.5)
    ax[1].set_xticks(x); ax[1].set_xticklabels(order, fontsize=5.5, rotation=15)
    ax[1].set_ylabel("expected lifetime value", fontsize=7)
    ax[1].set_title("(b) DP-optimal extracts the most\nvalue at ~0 failures",
                    fontsize=8)
    ax[1].tick_params(labelsize=6); ax[1].axhline(0, color="k", lw=0.5)

    # (c) cost of growth-rate uncertainty vs posterior SD.
    u = d["uncertainty"]
    ax[2].axhline(u["v_known"], color="k", ls="--", lw=0.9, label="known r")
    ax[2].plot(u["post_sds"], u["value"], "D-", color="#b71c1c", ms=5, lw=1.3,
               label="plan under uncertainty")
    ax[2].invert_xaxis()
    ax[2].set_xlabel("growth-rate posterior SD (← fleet sharpens)", fontsize=7)
    ax[2].set_ylabel("expected lifetime value", fontsize=7)
    ax[2].set_title("(c) resolving growth-rate\nuncertainty buys lifetime value",
                    fontsize=8)
    ax[2].legend(fontsize=5.5, loc="lower left"); ax[2].tick_params(labelsize=6)

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

    cfg = fl.FleetConfig()

    # ---- growth step monotone in crack + rate -------------------------------
    loads = _load_draws(cfg, 0)
    ok(grow_step(0.5, 0.05, loads, cfg).mean() > 0.5, "crack grows")
    ok(grow_step(0.5, 0.10, loads, cfg).mean()
       > grow_step(0.5, 0.05, loads, cfg).mean(), "faster rate → more growth")

    # ---- DP structure: value table + action map shapes ----------------------
    dp = solve_dp(0.05, cfg, seed=0)
    ok(dp["V"].shape == (F_MAX + 1, N_GRID), "value table shape")
    ok(dp["act"].shape == (F_MAX, N_GRID), "action map shape")
    ok(np.all(dp["V"][F_MAX] == S_RETIRE), "terminal value = salvage")

    # ---- FLY at the smallest crack; RETIRE optimal at end-of-life high crack -
    ok(dp["act"][0, 0] == ACTIONS.index("FLY"), "fly when crack ≈ 0")
    ok(dp["act"][F_MAX - 1, -1] == ACTIONS.index("RETIRE"),
       "retire at the last flight with a near-critical crack")
    ok((dp["act"] == ACTIONS.index("RETIRE")).any(), "RETIRE is used somewhere")
    ok((dp["act"] == ACTIONS.index("REPAIR")).any(), "REPAIR is used somewhere")
    # ---- retirement boundary is a valid crack level -------------------------
    ok(np.all((dp["boundary"] > 0) & (dp["boundary"] <= cfg.a_fail)),
       "retire boundary within (0, a_fail]")

    # ---- CORE: DP-optimal ≥ every other policy in lifetime value ------------
    r_true = 0.12
    dp = solve_dp(r_true, cfg, seed=0)
    e_dp = rollout(dp_policy(dp), r_true, cfg, n_vehicles=1500, seed=1)
    e_fly = rollout(fly_to_horizon_policy(cfg), r_true, cfg, n_vehicles=1500, seed=1)
    e_myo = rollout(clearance_policy(cfg, r_true, seed=0), r_true, cfg,
                    n_vehicles=1500, seed=1)
    ok(e_dp["value"] >= e_fly["value"] - 1e-6, "DP ≥ fly-to-horizon value")
    ok(e_dp["value"] >= e_myo["value"] - 1.0,
       "DP ≥ myopic-clearance value (optimal-stopping advantage)")
    # ---- DP keeps failures low (it protects against the catastrophe) --------
    ok(e_dp["fail_rate"] <= 0.03, f"DP fail rate low ({e_dp['fail_rate']:.3f})")
    ok(e_fly["fail_rate"] > e_dp["fail_rate"],
       "never-retire fails more than DP")

    # ---- rollout sanity: flights ≥ 0, value finite --------------------------
    ok(0 <= e_dp["flights"] <= F_MAX, "flights within horizon")
    ok(np.isfinite(e_dp["value"]), "value finite")

    # ---- uncertainty study: known-r ≥ planning under uncertainty, and -------
    #      sharper posterior → more value (monotone-ish recovery) -------------
    u = uncertainty_study(cfg, mu_r=-2.1, post_sds=(0.45, 0.10), seed=0)
    ok(u["v_known"] >= u["value"][0] - 1.0, "known r ≥ hedged-uncertain value")
    ok(u["value"][-1] >= u["value"][0] - 1.0,
       "sharper posterior recovers (≥) lifetime value")

    # ---- end-to-end run_study returns the ranking ---------------------------
    d = run_study(seed=0, verbose=False)
    ok(d["evals"]["DP-optimal"]["value"]
       >= d["evals"]["fly-to-horizon"]["value"] - 1e-6,
       "end-to-end DP ≥ fly-to-horizon")
    ok(0 < d["fixed_thr"] <= cfg.a_fail, "best fixed threshold in range")

    print(f"lifetime_policy: {n}/{n} unit tests passed")
    return n


def main():
    ap = argparse.ArgumentParser(
        description="Lifetime retire/repair/fly optimal-stopping DP.")
    ap.add_argument("--mu-r", type=float, default=-2.1,
                    help="E[log growth rate] of the vehicle")
    ap.add_argument("--fig", action="store_true")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.test:
        run_tests(); return
    d = run_study(mu_r=args.mu_r)
    if args.fig:
        p = make_figure(d, os.path.join(HERE, "paper_figs", "lifetime_policy.pdf"))
        print(f"\nfigure → {p}")


if __name__ == "__main__":
    main()
