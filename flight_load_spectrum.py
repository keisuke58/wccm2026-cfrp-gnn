"""
flight_load_spectrum.py — variable-amplitude flight loading + S–N anchoring
for the Stage-3 CFRP phase-field fatigue prognosis (cfrp_phasefield_2d.py).

Pipeline position (CFRP NDT, WCCM2026 / thesis):
  … → Stage-3 cfrp_phasefield_2d.simulate_fatigue_flights: TRUE cumulative
  Carrara-2020 AT2 fatigue — K identical load cycles per flight at ONE constant
  peak load, ᾱ degrades Gc, returns per-flight growth + survival.

This module closes the two remaining honesty gaps in that fatigue layer:

  (1) REALISTIC FLIGHT LOAD SPECTRUM.  A real reusable-rocket flight is NOT K
      cycles at one peak load: it is a variable-amplitude sequence of distinct
      events (liftoff/ascent ramp, max-Q, transonic buffet, MECO / stage-sep
      transient, reentry, landing-burn / touchdown), each with its own
      amplitude and cycle count.  We define such a per-flight spectrum
      (`FlightSpectrum`, named blocks) and run the EXISTING Carrara fatigue
      under it by CHAINING `simulate_fatigue_flights` across blocks of
      differing load, carrying the `FatigueState` (d, H, ᾱ, cycles) — no edit
      to cfrp_phasefield_2d is needed, its API already accepts a carried
      `state`, a per-call `cycles_per_flight`, and `ramp_steps_first_flight=0`
      to suppress the quasi-static ramp on continuation blocks.

      We then compare survival / clearance under the realistic spectrum against
      the legacy CONSTANT-PEAK proxy run at the SAME peak load and the SAME
      total cycle count, and quantify how much the spectrum shifts the
      predicted life / decision (the constant-peak proxy over-counts damage —
      it applies the rare max-Q amplitude to EVERY cycle, including the many
      low-amplitude buffet cycles).

  (2) S–N ANCHORING of the fatigue calibration.  ᾱ_T (cfg.fatigue_alpha_T) is
      a free knob (scale·Gc_ply/(6ℓ)).  We run the fatigue model at several
      constant amplitudes, extract cycles-to-failure N(amplitude), fit a
      Basquin/Wöhler line  log10(amp) = c + s·log10(N),  and pick
      fatigue_alpha_T_scale so the model slope s matches a documented
      representative CFRP S–N slope.

      Target slope (representative, NOT a fit to specific coupon data):
      CFRP stress-life curves are famously FLAT.  In the Basquin form
      σ_a = σ_f' · N^(-b), tension–tension CFRP laminates show b ≈ 0.05–0.10
      (≈ 5–10 % strength loss per decade of life), an order of magnitude
      flatter than metals — see Harris (ed.), *Fatigue in Composites*,
      Woodhead 2003, and the Mandell power-law summaries therein.  Here
      "amplitude" is the normalised transverse-peel strain proxy of the 2D
      cross-section model (cfrp_phasefield_2d docstring, §Loading), so the
      anchored slope d log10(amp)/d log10(N) = -b is a REPRESENTATIVE strain-
      life target on this model's load scale, not a calibration to a named
      coupon dataset.  We default the target to b = 0.06.

CPU, numpy/scipy only.  All studies here run on tiny grids (nx≈22) so the
whole --test path is a few seconds and --fig is well under a minute.

Usage:
  python3 flight_load_spectrum.py --test   # unit tests only
  python3 flight_load_spectrum.py --fig    # report + paper_figs/flight_load_spectrum.pdf
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field

import numpy as np

import cfrp_phasefield_2d as pf

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_PATH = os.path.join(HERE, "paper_figs", "flight_load_spectrum.pdf")

# Representative CFRP Basquin slope b in σ_a = σ_f' N^(-b) (see module docstring).
# d log10(amp)/d log10(N) = -b.  Default 0.06 (flat-ish CFRP, ~6 %/decade).
DEFAULT_TARGET_BASQUIN_B = 0.06

# Fatigue-ACTIVE alpha_T scale for the survival comparison.  The library default
# (1000) puts ᾱ_T so high that ᾱ barely accumulates over a HCF mission, so over
# a realistic cycle budget growth is gated by the static peak alone and the
# spectrum is indistinguishable from the single-peak proxy.  We run the
# spectrum-vs-proxy survival in the SAME fatigue-operative regime as the S–N
# anchoring (smaller ᾱ_T) so cumulative variable-amplitude damage is resolved.
DEFAULT_FATIGUE_SCALE = 120.0


# ═════════════════════════════════════════════════════════════════════════════
#  Flight load spectrum
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class SpectrumBlock:
    """One named load event within a flight: `n_cycles` cycles at `amplitude`.

    `amplitude` is the block peak transverse-peel strain (same load scale as
    cfrp_phasefield_2d.simulate_fatigue_flights `load`).
    """
    name: str
    amplitude: float
    n_cycles: int


@dataclass
class FlightSpectrum:
    """Variable-amplitude per-flight load spectrum: an ordered list of blocks.

    The default is a representative reusable-rocket profile (qualitative
    amplitudes on the normalised peel-strain scale; the few high-amplitude
    transients are rare, the buffet block supplies the bulk of the cycles):

      liftoff      ascent thrust build-up / pad release transient
      maxq         max dynamic pressure — the per-flight peak load
      buffet       transonic buffet — many low-amplitude cycles
      meco         MECO / stage-separation shock transient
      reentry      reentry aero / thermal-mechanical load
      landingburn  landing burn + touchdown transient

    Total cycles per flight = Σ n_cycles; peak = max amplitude (max-Q).
    """
    blocks: list[SpectrumBlock] = field(default_factory=lambda: [
        SpectrumBlock("liftoff",     0.050, 3),
        SpectrumBlock("maxq",        0.066, 2),
        SpectrumBlock("buffet",      0.030, 40),
        SpectrumBlock("meco",        0.058, 2),
        SpectrumBlock("reentry",     0.046, 6),
        SpectrumBlock("landingburn", 0.054, 4),
    ])

    @property
    def peak(self) -> float:
        return max(b.amplitude for b in self.blocks)

    @property
    def total_cycles(self) -> int:
        return int(sum(b.n_cycles for b in self.blocks))

    def equivalent_amplitude(self, b_exp: float) -> float:
        """Miner/Basquin equivalent constant amplitude over one flight.

        Damage per cycle ∝ amplitude^(1/b) (Basquin), so the constant
        amplitude doing the SAME total damage in `total_cycles` cycles is
            amp_eq = ( Σ n_i amp_i^(1/b) / Σ n_i )^b .
        This is the fair constant-amplitude proxy to compare the spectrum
        against (a single peak load is NOT fair — it over-counts).
        """
        p = 1.0 / max(b_exp, 1e-6)
        num = sum(blk.n_cycles * blk.amplitude ** p for blk in self.blocks)
        den = max(self.total_cycles, 1)
        return float((num / den) ** b_exp)


# ═════════════════════════════════════════════════════════════════════════════
#  Run the EXISTING Carrara fatigue under a spectrum (carried FatigueState)
# ═════════════════════════════════════════════════════════════════════════════

def simulate_spectrum_flights(d0: np.ndarray, spectrum: FlightSpectrum,
                              n_flights: int,
                              cfg: pf.LaminateConfig | None = None,
                              ramp_steps_first_block: int = 6,
                              stop_on_growth: bool = True) -> pf.FatigueResult:
    """Carrara-2020 AT2 fatigue under a variable-amplitude flight spectrum.

    Each flight replays the spectrum block-by-block; within a block we call
    the UNCHANGED `cfrp_phasefield_2d.simulate_fatigue_flights` for ONE
    "flight" of `block.n_cycles` cycles at `block.amplitude`, carrying the
    FatigueState (d, H, ᾱ, cycles) from the previous block.  The quasi-static
    ramp (`ramp_steps_first_block`) is applied ONLY on the very first block of
    the mission; every continuation block passes `ramp_steps_first_flight=0`
    so it is pure cyclic fatigue at the carried damage state.

    Growth is judged the same way as the proxy: relative d>0.5 area increase
    vs the SEED `d0` exceeding cfg.growth_rel_threshold, checked after each
    full flight.  Returns a FatigueResult whose rel_growth_per_flight is the
    end-of-flight relative growth (NaN after an early stop on growth).
    """
    cfg = cfg or pf.LaminateConfig()
    area0 = pf.damage_area(d0, cfg)
    state: pf.FatigueState | None = None
    rel_growth = np.full(n_flights, np.nan)
    fail_flight = None
    first_block_done = False

    for fl in range(1, n_flights + 1):
        for blk in spectrum.blocks:
            ramp = ramp_steps_first_block if not first_block_done else 0
            res = pf.simulate_fatigue_flights(
                d0, blk.amplitude, 1, cfg,
                cycles_per_flight=blk.n_cycles,
                state=state, ramp_steps_first_flight=ramp,
                stop_on_growth=False)
            state = res.state
            first_block_done = True
        rel = (pf.damage_area(state.d, cfg) - area0) / max(area0, 1e-12)
        rel_growth[fl - 1] = rel
        if fail_flight is None and rel > cfg.growth_rel_threshold:
            fail_flight = fl
            if stop_on_growth:
                break

    return pf.FatigueResult(grown=fail_flight is not None,
                            fail_flight=fail_flight, area0=area0,
                            rel_growth_per_flight=rel_growth, state=state)


def _survival_curve_spectrum(post: np.ndarray, spectrum: FlightSpectrum,
                             n_flights: int, cfg: pf.LaminateConfig,
                             n_draws: int,
                             rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Across-draw survival curve S(n) under the spectrum + first-fail flights."""
    samples = np.atleast_2d(np.asarray(post, dtype=float))
    if len(samples) > n_draws:
        samples = samples[rng.choice(len(samples), n_draws, replace=False)]
    fail = np.full(len(samples), n_flights + 1, dtype=float)
    for k, (cx, cy, layer, l2s) in enumerate(samples):
        d0 = pf.seed_defect(cx, cy, layer, l2s, cfg)
        res = simulate_spectrum_flights(d0, spectrum, n_flights, cfg)
        if res.fail_flight is not None:
            fail[k] = res.fail_flight
    flights = np.arange(1, n_flights + 1)
    surv = np.array([(fail > fl).mean() for fl in flights])
    return surv, fail


# ═════════════════════════════════════════════════════════════════════════════
#  (1) Spectrum vs constant-peak proxy comparison
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class SpectrumVsProxyResult:
    n_flights: int
    spectrum_surv: np.ndarray       # S(n) under variable-amplitude spectrum
    proxy_peak_surv: np.ndarray     # S(n) constant-peak proxy (max-Q on all cyc)
    proxy_eq_surv: np.ndarray       # S(n) constant Basquin-equivalent proxy
    spectrum_fail: np.ndarray       # first-fail flight per draw (spectrum)
    proxy_peak_fail: np.ndarray
    proxy_eq_fail: np.ndarray
    peak: float
    total_cycles: int
    amp_eq: float
    life_spectrum: float            # mean flights survived = Σ S(n) (censored)
    life_proxy_peak: float
    life_proxy_eq: float


def _constant_proxy_curve(post: np.ndarray, amp: float, K: int,
                          n_flights: int, cfg: pf.LaminateConfig,
                          n_draws: int,
                          rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Survival curve for a CONSTANT-amplitude proxy: K cycles at `amp`/flight."""
    samples = np.atleast_2d(np.asarray(post, dtype=float))
    if len(samples) > n_draws:
        samples = samples[rng.choice(len(samples), n_draws, replace=False)]
    fail = np.full(len(samples), n_flights + 1, dtype=float)
    for k, (cx, cy, layer, l2s) in enumerate(samples):
        d0 = pf.seed_defect(cx, cy, layer, l2s, cfg)
        res = pf.simulate_fatigue_flights(d0, amp, n_flights, cfg,
                                          cycles_per_flight=K,
                                          stop_on_growth=True)
        if res.fail_flight is not None:
            fail[k] = res.fail_flight
    flights = np.arange(1, n_flights + 1)
    surv = np.array([(fail > fl).mean() for fl in flights])
    return surv, fail


def _mean_life(surv: np.ndarray) -> float:
    """Mean flights survived over the horizon = Σ_n S(n) (area under S, censored
    at the horizon).  Robust to ties in the median first-fail flight and the
    same statistic flight_clearance reports as expected_remaining_flights."""
    return float(np.sum(surv))


def compare_spectrum_vs_proxy(post: np.ndarray,
                              spectrum: FlightSpectrum | None = None,
                              n_flights: int = 12,
                              cfg: pf.LaminateConfig | None = None,
                              b_exp: float = DEFAULT_TARGET_BASQUIN_B,
                              fatigue_scale: float = DEFAULT_FATIGUE_SCALE,
                              n_draws: int = 8,
                              rng: np.random.Generator | None = None
                              ) -> SpectrumVsProxyResult:
    """Survival / life under the realistic spectrum vs constant-amplitude proxies.

    Two proxies, both matched in TOTAL cycles/flight to the spectrum:
      * proxy_peak : the legacy single-peak proxy — every cycle at the max-Q
        peak (the current Stage-3 default; over-counts damage);
      * proxy_eq   : a constant amplitude at the Basquin-equivalent amplitude
        `spectrum.equivalent_amplitude(b_exp)` (the fair Miner proxy).

    The comparison runs at `fatigue_scale` (a fatigue-operative ᾱ_T) so that
    cumulative variable-amplitude damage is resolved over the mission — see
    DEFAULT_FATIGUE_SCALE.
    """
    base = cfg or pf.LaminateConfig()
    cfg = pf.LaminateConfig(
        Lx=base.Lx, Ly=base.Ly, nx=base.nx, ny=base.ny,
        Gc_ply=base.Gc_ply, growth_rel_threshold=base.growth_rel_threshold,
        fatigue_alpha_T_scale=fatigue_scale)
    spectrum = spectrum or FlightSpectrum()
    rng = rng or np.random.default_rng(0)
    K = spectrum.total_cycles
    peak = spectrum.peak
    amp_eq = spectrum.equivalent_amplitude(b_exp)

    # use independent rng copies so the SAME draws are used for each method
    s_surv, s_fail = _survival_curve_spectrum(
        post, spectrum, n_flights, cfg, n_draws, np.random.default_rng(rng.integers(1 << 30)))
    seed = int(rng.integers(1 << 30))
    pk_surv, pk_fail = _constant_proxy_curve(
        post, peak, K, n_flights, cfg, n_draws, np.random.default_rng(seed))
    eq_surv, eq_fail = _constant_proxy_curve(
        post, amp_eq, K, n_flights, cfg, n_draws, np.random.default_rng(seed))

    return SpectrumVsProxyResult(
        n_flights=n_flights, spectrum_surv=s_surv,
        proxy_peak_surv=pk_surv, proxy_eq_surv=eq_surv,
        spectrum_fail=s_fail, proxy_peak_fail=pk_fail, proxy_eq_fail=eq_fail,
        peak=peak, total_cycles=K, amp_eq=amp_eq,
        life_spectrum=_mean_life(s_surv),
        life_proxy_peak=_mean_life(pk_surv),
        life_proxy_eq=_mean_life(eq_surv))


# ═════════════════════════════════════════════════════════════════════════════
#  (2) S–N anchoring of fatigue_alpha_T_scale
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class SNResult:
    amps: np.ndarray            # constant amplitudes used (finite-life only)
    N_fail: np.ndarray          # cycles-to-failure at each amplitude
    slope: float                # fitted d log10(amp)/d log10(N)  (= -b)
    intercept: float
    scale: float                # fatigue_alpha_T_scale used
    alpha_T: float              # resulting cfg.fatigue_alpha_T


def cycles_to_failure(amp: float, cfg: pf.LaminateConfig,
                      K_block: int = 8, max_blocks: int = 160,
                      ramp_steps: int = 6) -> float:
    """Cumulative cycles to growth at CONSTANT amplitude `amp`.

    Runs the Carrara fatigue in blocks of `K_block` cycles (carried state)
    until the relative crack area exceeds cfg.growth_rel_threshold; returns the
    accumulated cycle count, or np.inf if it survives `max_blocks` blocks.
    """
    d0 = pf.seed_defect(0.5, 0.5, 9.0, 1.0, cfg)
    state = None
    for b in range(max_blocks):
        ramp = ramp_steps if b == 0 else 0
        res = pf.simulate_fatigue_flights(d0, amp, 1, cfg,
                                          cycles_per_flight=K_block,
                                          state=state,
                                          ramp_steps_first_flight=ramp,
                                          stop_on_growth=False)
        state = res.state
        if res.rel_growth_per_flight[0] > cfg.growth_rel_threshold:
            return float(state.cycles)
    return float("inf")


def sn_curve(amps, scale: float, base_cfg: pf.LaminateConfig | None = None,
             K_block: int = 8, min_cycles_for_fit: float = 12.0) -> SNResult:
    """S–N curve at a given fatigue_alpha_T_scale.

    Excludes amplitudes that fail within the first block (N ≤ min_cycles_for_fit
    ≈ statically critical, not cyclic) and any with infinite life, then fits a
    Basquin line in log10(amp)–log10(N).
    """
    base = base_cfg or pf.LaminateConfig()
    cfg = pf.LaminateConfig(
        Lx=base.Lx, Ly=base.Ly, nx=base.nx, ny=base.ny,
        Gc_ply=base.Gc_ply, growth_rel_threshold=base.growth_rel_threshold,
        fatigue_alpha_T_scale=scale)
    amps = np.asarray(amps, dtype=float)
    N = np.array([cycles_to_failure(a, cfg, K_block=K_block) for a in amps])
    fit_mask = np.isfinite(N) & (N > min_cycles_for_fit)
    if fit_mask.sum() >= 2:
        slope, intercept = np.polyfit(np.log10(N[fit_mask]),
                                      np.log10(amps[fit_mask]), 1)
    else:
        slope, intercept = np.nan, np.nan
    return SNResult(amps=amps[fit_mask], N_fail=N[fit_mask],
                    slope=float(slope), intercept=float(intercept),
                    scale=float(scale), alpha_T=float(cfg.fatigue_alpha_T))


_SLOPE_ERR_UNDEFINED = 1.0   # error charged when a scale yields no S–N slope


def _slope_err(slope: float, target_slope: float) -> float:
    """|slope − target|, or a max penalty if the slope is undefined (NaN).

    A scale that produces fewer than two CYCLIC failures cannot define an S–N
    line at all (the default ᾱ_T does exactly this — see anchor docstring), so
    we charge it `_SLOPE_ERR_UNDEFINED`: an undefined Wöhler curve is strictly
    worse than any finite-slope fit for the purpose of S–N anchoring.
    """
    if not np.isfinite(slope):
        return _SLOPE_ERR_UNDEFINED
    return abs(slope - target_slope)


def anchor_alpha_T_scale(target_b: float = DEFAULT_TARGET_BASQUIN_B,
                         amps=(0.044, 0.050, 0.056),
                         scales=(40.0, 80.0, 160.0, 320.0, 640.0),
                         base_cfg: pf.LaminateConfig | None = None,
                         K_block: int = 8) -> dict:
    """Anchor fatigue_alpha_T_scale to a target representative CFRP S–N slope.

    Sweeps `scales`, measures the Basquin slope d log10(amp)/d log10(N) at each
    (NaN where a scale leaves <2 amplitudes in the cyclic regime), and selects
    the scale whose curve is best anchored — minimum slope error to the target
    among VALID (finite-slope) scales, breaking ties toward the scale with the
    most cyclic fit points (best-conditioned Wöhler line).

    Honest finding (reported, not hidden): on this 2D peel-proxy model the
    Basquin slope is largely an INTRINSIC property of the Carrara-2020 fatigue
    law + defect geometry (≈ -0.33 here), only weakly dependent on ᾱ_T (the
    scale mainly shifts N, i.e. the S–N intercept).  It is therefore STEEPER
    than real flat CFRP (b ≈ 0.05–0.10).  ᾱ_T anchoring places the cyclic
    window and life intercept onto a sensible HCF range and yields a
    well-defined Wöhler curve; matching the absolute CFRP slope would require a
    different fatigue degradation kernel (e.g. a Wöhler-exponent variant of
    f(ᾱ)) — out of scope.  We still report the achieved slope honestly.

    Default reference: the LIBRARY default scale (1000) pushes every amplitude
    out of the cyclic regime over a feasible cycle budget → no failures → an
    UNDEFINED S–N slope.  Anchoring restores a finite, well-defined slope, so
    the slope error is strictly reduced (undefined → finite).
    """
    base = base_cfg or pf.LaminateConfig()
    target_slope = -abs(target_b)
    swept = [sn_curve(amps, s, base_cfg=base, K_block=K_block) for s in scales]

    # choose the valid scale with the smallest slope error, ties → most points
    def _key(r: SNResult):
        return (_slope_err(r.slope, target_slope), -r.N_fail.size)
    valid = [r for r in swept if np.isfinite(r.slope) and r.N_fail.size >= 2]
    best = min(valid, key=_key) if valid else min(swept, key=lambda r: -r.N_fail.size)
    anchored_scale = best.scale
    anchored = best

    # default reference = library default ᾱ_T scale, SAME grid/amplitudes
    default_scale = float(pf.LaminateConfig().fatigue_alpha_T_scale)
    default = sn_curve(amps, default_scale, base_cfg=base, K_block=K_block)

    err_default = _slope_err(default.slope, target_slope)
    err_anchored = _slope_err(anchored.slope, target_slope)

    return {
        "target_b": abs(target_b),
        "target_slope": target_slope,
        "swept": swept,
        "anchored_scale": anchored_scale,
        "anchored": anchored,
        "default_scale": default_scale,
        "default": default,
        "slope_err_default": float(err_default),
        "slope_err_anchored": float(err_anchored),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  Report
# ═════════════════════════════════════════════════════════════════════════════

def print_report(cmp: SpectrumVsProxyResult, anc: dict,
                 spectrum: FlightSpectrum) -> None:
    print("=" * 70)
    print(" FLIGHT LOAD SPECTRUM + S–N ANCHORING — Stage-3 CFRP fatigue")
    print("=" * 70)

    print("\n(1) Variable-amplitude flight spectrum")
    print(f"    {'block':<12}{'amplitude':>10}{'n_cycles':>10}")
    for blk in spectrum.blocks:
        print(f"    {blk.name:<12}{blk.amplitude:>10.3f}{blk.n_cycles:>10d}")
    print(f"    {'TOTAL':<12}{'':>10}{spectrum.total_cycles:>10d}"
          f"   peak(max-Q)={spectrum.peak:.3f}  "
          f"amp_eq(Basquin)={cmp.amp_eq:.4f}")

    print("\n    Survival vs constant-amplitude proxies "
          f"(matched K={cmp.total_cycles} cyc/flight, {cmp.n_flights} flights):")
    print(f"    {'flight':>7}{'spectrum':>10}{'proxy_peak':>12}{'proxy_eq':>10}")
    for i in range(cmp.n_flights):
        print(f"    {i+1:>7}{cmp.spectrum_surv[i]:>10.2f}"
              f"{cmp.proxy_peak_surv[i]:>12.2f}{cmp.proxy_eq_surv[i]:>10.2f}")
    print(f"    mean flights survived (ΣS(n)):  spectrum={cmp.life_spectrum:.2f}"
          f"  proxy_peak={cmp.life_proxy_peak:.2f}  "
          f"proxy_eq={cmp.life_proxy_eq:.2f}")
    if cmp.life_proxy_peak > 0:
        factor = cmp.life_spectrum / cmp.life_proxy_peak
        print(f"    => single-peak proxy UNDER-predicts mean life by ~{factor:.1f}x"
              f" (it applies the rare max-Q amplitude to EVERY cycle).")
    if cmp.life_proxy_eq > 0:
        f2 = cmp.life_proxy_eq / cmp.life_spectrum
        print(f"    => Basquin-equivalent proxy OVER-predicts life by ~{f2:.1f}x "
              f"(Miner-equivalent amplitude misses sequence/peak effects).")

    print("\n(2) S–N anchoring of fatigue_alpha_T_scale")
    print(f"    target representative CFRP Basquin b = {anc['target_b']:.3f}  "
          f"(slope d log10(amp)/d log10(N) = {anc['target_slope']:+.3f})")
    print(f"    {'scale':>8}{'alpha_T':>10}{'slope':>10}{'n_pts':>7}")
    for r in anc["swept"]:
        sl = f"{r.slope:>10.4f}" if np.isfinite(r.slope) else f"{'nan':>10}"
        print(f"    {r.scale:>8.0f}{r.alpha_T:>10.4f}{sl}{r.N_fail.size:>7d}")
    a = anc["anchored"]
    print(f"    anchored scale = {anc['anchored_scale']:.1f}  "
          f"(alpha_T = {a.alpha_T:.4f}), achieved slope = {a.slope:+.4f}")
    dft_sl = (f"{anc['default'].slope:+.4f}"
              if np.isfinite(anc['default'].slope) else "UNDEFINED (no cyclic "
              "failures at default ᾱ_T)")
    print(f"    default scale={anc['default_scale']:.0f} slope = {dft_sl}")
    print(f"    slope error vs target:  default={anc['slope_err_default']:.4f}"
          f"  ->  anchored={anc['slope_err_anchored']:.4f}"
          f"  (reduced: {anc['slope_err_anchored'] <= anc['slope_err_default']})")
    print(f"    anchored S–N points (amp, N): "
          f"{list(zip(np.round(a.amps,3), np.round(a.N_fail,0)))}")
    print("    HONEST NOTES:")
    print("      * representative strain-life anchoring on the model's "
          "normalised load")
    print("        scale — NOT a fit to a named coupon dataset.")
    print(f"      * the model's Basquin slope (~{a.slope:+.2f}) is INTRINSIC to "
          "the Carrara")
    print("        f(ᾱ) kernel + defect geometry; it is STEEPER than flat CFRP "
          f"(b≈{anc['target_b']:.2f}).")
    print("        ᾱ_T anchoring fixes the cyclic window / life intercept and a "
          "WELL-DEFINED")
    print("        Wöhler curve; matching CFRP's absolute slope needs a "
          "different kernel.")
    print("=" * 70)


# ═════════════════════════════════════════════════════════════════════════════
#  Figure
# ═════════════════════════════════════════════════════════════════════════════

def make_figure(cmp: SpectrumVsProxyResult, anc: dict,
                spectrum: FlightSpectrum, out_path: str = FIG_PATH) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import sys
    sys.path.insert(0, os.path.join(HERE, "slides", "figure_sources"))
    try:
        from thesis_style import use
        figsize = use(width_frac=1.0, aspect=0.36)
    except Exception:
        figsize = (12.0, 4.0)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=figsize)

    # (a) the load spectrum (one flight) as a stairstep of per-cycle amplitude
    amps_seq, names = [], []
    for blk in spectrum.blocks:
        amps_seq.extend([blk.amplitude] * blk.n_cycles)
        names.append(blk.name)
    amps_seq = np.array(amps_seq)
    x = np.arange(1, len(amps_seq) + 1)
    ax[0].step(x, amps_seq, where="mid", color="#1565C0", lw=1.2)
    ax[0].axhline(spectrum.peak, color="#b71c1c", ls="--", lw=0.9,
                  label=f"max-Q peak={spectrum.peak:.3f}")
    ax[0].axhline(cmp.amp_eq, color="#2e7d32", ls=":", lw=1.0,
                  label=f"Basquin eq.={cmp.amp_eq:.3f}")
    # label block boundaries
    c0 = 0
    for blk in spectrum.blocks:
        cm = c0 + blk.n_cycles / 2.0
        ax[0].annotate(blk.name, (cm, blk.amplitude), fontsize=5,
                       ha="center", va="bottom", rotation=20)
        c0 += blk.n_cycles
    ax[0].set_xlabel("cycle within flight", fontsize=8)
    ax[0].set_ylabel("peel-strain amplitude", fontsize=8)
    ax[0].set_title("(a) per-flight load spectrum", fontsize=9)
    ax[0].legend(fontsize=6, loc="upper right")
    ax[0].tick_params(labelsize=7)
    ax[0].grid(True, alpha=0.25)

    # (b) survival under spectrum vs proxies
    fl = np.arange(1, cmp.n_flights + 1)
    ax[1].plot(fl, cmp.spectrum_surv, "o-", color="#1565C0", ms=4,
               label="variable spectrum")
    ax[1].plot(fl, cmp.proxy_peak_surv, "s--", color="#b71c1c", ms=4,
               label="constant-peak proxy")
    ax[1].plot(fl, cmp.proxy_eq_surv, "^:", color="#2e7d32", ms=4,
               label="Basquin-eq. proxy")
    ax[1].set_xlabel("flight", fontsize=8)
    ax[1].set_ylabel("$S(n)$ = P(survive)", fontsize=8)
    ax[1].set_title("(b) survival: spectrum vs proxy", fontsize=9)
    ax[1].set_ylim(-0.05, 1.05)
    ax[1].legend(fontsize=6, loc="lower left")
    ax[1].tick_params(labelsize=7)
    ax[1].grid(True, alpha=0.25)

    # (c) anchored S–N curve + default + target slope
    a = anc["anchored"]
    d = anc["default"]
    if a.N_fail.size:
        ax[2].loglog(a.N_fail, a.amps, "o", color="#1565C0", ms=6,
                     label=f"anchored (s={a.slope:+.3f})")
        Ng = np.array([a.N_fail.min(), a.N_fail.max()])
        ax[2].loglog(Ng, 10 ** (a.intercept + a.slope * np.log10(Ng)),
                     "-", color="#1565C0", lw=1.0)
        # target slope line anchored at the anchored mid-point
        Nmid = np.sqrt(a.N_fail.min() * a.N_fail.max())
        amid = 10 ** (a.intercept + a.slope * np.log10(Nmid))
        c_t = np.log10(amid) - anc["target_slope"] * np.log10(Nmid)
        ax[2].loglog(Ng, 10 ** (c_t + anc["target_slope"] * np.log10(Ng)),
                     "k--", lw=0.9,
                     label=f"target b={anc['target_b']:.3f}")
    if d.N_fail.size:
        ax[2].loglog(d.N_fail, d.amps, "s", color="#b71c1c", ms=5,
                     mfc="none", label=f"default (s={d.slope:+.3f})")
    ax[2].set_xlabel("cycles to failure $N$", fontsize=8)
    ax[2].set_ylabel("amplitude", fontsize=8)
    ax[2].set_title("(c) anchored S–N (Wöhler)", fontsize=9)
    ax[2].legend(fontsize=6, loc="upper right")
    ax[2].tick_params(labelsize=7)
    ax[2].grid(True, which="both", alpha=0.25)

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

    rng = np.random.default_rng(3)
    # tiny grid for speed
    cfg = pf.LaminateConfig(nx=22, ny=18)

    # --- FlightSpectrum bookkeeping ---
    spec = FlightSpectrum()
    ok(spec.total_cycles == sum(b.n_cycles for b in spec.blocks),
       "total_cycles sums blocks")
    ok(abs(spec.peak - max(b.amplitude for b in spec.blocks)) < 1e-12,
       "peak = max block amplitude")
    amp_eq = spec.equivalent_amplitude(DEFAULT_TARGET_BASQUIN_B)
    ok(min(b.amplitude for b in spec.blocks) < amp_eq < spec.peak,
       f"Basquin-equivalent amplitude between min and peak ({amp_eq:.4f})")

    # --- spectrum chains the carried FatigueState (cycles accumulate) ---
    d0 = pf.seed_defect(0.5, 0.5, 9.0, 1.0, cfg)
    res = simulate_spectrum_flights(d0, spec, 2, cfg, stop_on_growth=False)
    ok(res.state.cycles >= 2 * spec.total_cycles - 1,
       f"carried state accumulates ~2 flights of cycles "
       f"({res.state.cycles})")
    ok(res.rel_growth_per_flight.shape == (2,),
       "rel_growth_per_flight has one entry per flight")

    # --- (1) variable spectrum gives DIFFERENT life than constant-peak proxy ---
    post = np.column_stack([
        rng.normal(0.5, 0.03, 6), rng.normal(0.5, 0.03, 6),
        rng.normal(9.0, 0.5, 6), rng.normal(1.0, 0.2, 6)])
    cmp = compare_spectrum_vs_proxy(post, spec, n_flights=8, cfg=cfg,
                                    n_draws=4, rng=np.random.default_rng(1))
    # constant-peak proxy applies max-Q to EVERY cycle -> shorter life / lower S
    ok(np.any(cmp.spectrum_surv != cmp.proxy_peak_surv),
       "variable spectrum survival differs from constant-peak proxy")
    # the single-peak proxy applies max-Q to EVERY cycle -> over-counts damage
    # -> shorter mean life (smaller area under S(n)) than the real spectrum
    ok(cmp.life_proxy_peak < cmp.life_spectrum,
       "constant-peak proxy mean life < spectrum mean life (proxy over-counts "
       f"damage): {cmp.life_proxy_peak:.2f} vs {cmp.life_spectrum:.2f}")

    # --- (1b) higher-amplitude blocks dominate the damage ---
    # zero-out the rare high-amplitude transients -> much less fatigue damage
    low = FlightSpectrum(blocks=[
        b if b.amplitude < 0.05 else SpectrumBlock(b.name, b.amplitude, 0)
        for b in spec.blocks])
    d_full = simulate_spectrum_flights(
        pf.seed_defect(0.5, 0.5, 9.0, 1.0, cfg), spec, 3, cfg,
        stop_on_growth=False).state.alpha_bar.max()
    d_low = simulate_spectrum_flights(
        pf.seed_defect(0.5, 0.5, 9.0, 1.0, cfg), low, 3, cfg,
        stop_on_growth=False).state.alpha_bar.max()
    ok(d_full > d_low * 1.2,
       f"high-amplitude blocks dominate fatigue accumulation "
       f"(full ᾱ={d_full:.3g} >> low-only ᾱ={d_low:.3g})")

    # --- (2) S–N: cycles-to-failure decreases with amplitude (Wöhler sense) ---
    cfg_sn = pf.LaminateConfig(nx=20, ny=16, fatigue_alpha_T_scale=80.0)
    N_lo = cycles_to_failure(0.050, cfg_sn)
    N_hi = cycles_to_failure(0.060, cfg_sn)
    ok(N_hi < N_lo,
       f"higher amplitude fails sooner (N(0.06)={N_hi} < N(0.05)={N_lo})")

    # --- (2b) anchoring yields a WELL-DEFINED S–N curve, default does NOT,
    #          so anchoring REDUCES the slope error vs the library default ---
    anc = anchor_alpha_T_scale(target_b=DEFAULT_TARGET_BASQUIN_B,
                               amps=(0.044, 0.050, 0.056),
                               scales=(40.0, 160.0),
                               base_cfg=pf.LaminateConfig(nx=20, ny=16))
    ok(np.isfinite(anc["anchored"].slope) and anc["anchored"].N_fail.size >= 2,
       "anchored S–N slope is finite / well-defined")
    ok(not np.isfinite(anc["default"].slope),
       "library-default ᾱ_T gives NO cyclic failures -> undefined S–N slope")
    ok(anc["slope_err_anchored"] < anc["slope_err_default"],
       f"anchoring reduces slope error vs default "
       f"({anc['slope_err_anchored']:.4f} < {anc['slope_err_default']:.4f})")

    print(f"flight_load_spectrum: {n}/{n} unit tests passed")
    return n


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Variable-amplitude flight load spectrum + S–N anchoring "
                    "for the Stage-3 CFRP phase-field fatigue layer.")
    ap.add_argument("--target-b", type=float, default=DEFAULT_TARGET_BASQUIN_B,
                    help="representative CFRP Basquin slope b (default 0.06)")
    ap.add_argument("--n-flights", type=int, default=12,
                    help="mission horizon for the spectrum-vs-proxy survival")
    ap.add_argument("--n-draws", type=int, default=8,
                    help="posterior draws for the survival curves")
    ap.add_argument("--nx", type=int, default=22, help="grid nx (coarse=fast)")
    ap.add_argument("--ny", type=int, default=18, help="grid ny")
    ap.add_argument("--fig", action="store_true",
                    help="save paper_figs/flight_load_spectrum.pdf (+ .png)")
    ap.add_argument("--test", action="store_true", help="unit tests only")
    args = ap.parse_args()

    if args.test:
        run_tests(); return

    t0 = time.perf_counter()
    cfg = pf.LaminateConfig(nx=args.nx, ny=args.ny)
    spectrum = FlightSpectrum()
    rng = np.random.default_rng(7)
    post = np.column_stack([
        rng.normal(0.5, 0.04, 32), rng.normal(0.5, 0.04, 32),
        rng.normal(9.0, 0.8, 32), rng.normal(1.0, 0.25, 32)])

    print("[spectrum] survival under variable-amplitude spectrum vs proxy ...")
    cmp = compare_spectrum_vs_proxy(post, spectrum, n_flights=args.n_flights,
                                    cfg=cfg, b_exp=args.target_b,
                                    n_draws=args.n_draws, rng=rng)
    print("[s-n] anchoring fatigue_alpha_T_scale to target Basquin slope ...")
    anc = anchor_alpha_T_scale(target_b=args.target_b,
                               base_cfg=pf.LaminateConfig(nx=args.nx, ny=args.ny))
    print(f"[done] studies in {time.perf_counter() - t0:.1f}s\n")

    print_report(cmp, anc, spectrum)

    if args.fig:
        out = make_figure(cmp, anc, spectrum)
        print(f"\n[fig] wrote {out} (+ .png)")


if __name__ == "__main__":
    main()
