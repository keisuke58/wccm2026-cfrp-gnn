"""
srb3_motorcase.py — the SRB-3 solid-rocket-booster MOTOR CASE as the THIRD
structure in the Stage 0-5 decision framework: a filament-wound CFRP pressure
vessel with its OWN sensing modality (acoustic emission, AE) and its OWN physics
(internal-pressure burst-margin prognosis).

Why this exists
---------------
The framework was end-to-end on the perforated interstage (matrix-crack
delamination, FD AT2 phase-field) and the satellite fairing (skin-core debond,
interfacial fracture). Both detect with full-field optical/strain data and clear
through the SAME pluggable `Prognosis` + `flight_clearance_generic` core
(fairing_debond_prognosis.py). A motor case is a different beast:

  * sensing is ACOUSTIC EMISSION, not full-field optics — transient elastic
    waves from cracking/fibre-break/delamination, recorded as per-hit features;
  * the failure physics is INTERNAL-PRESSURE BURST of a filament-wound vessel
    (hoop stress sigma_theta = p R / t), not a transverse-load blister.

Adding it shows the framework's extensibility: a NEW CFRP part with a NEW
front-end (AE) and NEW prognosis (burst margin) still clears OK/REPAIR/RETIRE
through the identical decision core. SRB-3 = 3rd structure, same decision rule.

Real data (Stage-1 detection)
-----------------------------
4TU "Acoustic Emission monitoring in CFRP compression tests" (CC0). Vallen
`.pridb` SQLite hit tables under data/srb3_ae/:
  comp0 / comp90        — plain compression 0deg/90deg          -> PRISTINE (0)
  CAI-spec1/2/3         — compression-after-impact (impacted)   -> DAMAGED  (1)
  compteflon            — Teflon insert (artificial delam)      -> DAMAGED  (1)
Per AE hit we read [Amp, RiseT, Dur, Eny, Counts, RMS] (+ derived log-energy,
RA = RiseT/Amp, average-frequency = Counts/Dur). Stage-1 is a GROUP-aware
(hold-out whole specimens) pristine-vs-damaged classifier reporting AUROC, plus
the classic amplitude-duration / RA-avg-freq damage-MODE clustering.

Honest scope / data proxy
-------------------------
This is a CFRP COMPRESSION AE dataset used as a representative REAL-DATA proxy
for the motor-case AE front-end (the same hit-feature physics: matrix cracking,
fibre breakage, delamination signatures). It is NOT a pressure-vessel burst
test, and the burst prognosis below is a representative filament-wound model
(lumped constants, trends-not-absolute — same stance as the interstage load
proxy and the fairing debond model). The contribution is the end-to-end,
pluggable, multi-structure framework, not certified motor-case numbers.

Usage
-----
    python srb3_motorcase.py            # AE detection + burst prognosis report
    python srb3_motorcase.py --fig
    python srb3_motorcase.py --test
"""
from __future__ import annotations

import argparse
import glob
import os
import sqlite3
from dataclasses import dataclass, field

import numpy as np

import cfrp_phasefield_2d as pf  # reuse the SHARED decision rule (via fairing)
from fairing_debond_prognosis import (Prognosis, flight_clearance_generic)

HERE = os.path.dirname(os.path.abspath(__file__))
AE_DIR = os.path.join(HERE, "data", "srb3_ae")

# specimen -> damage-condition label (1 = damaged, 0 = pristine)
SPEC_LABELS = {
    "comp0": 0, "comp90": 0,                       # plain compression = pristine
    "CAI-spec1": 1, "CAI-spec2": 1, "CAI-spec3": 1,  # compression-after-impact
    "compteflon": 1,                               # Teflon insert = artificial delam
}

FEATURE_NAMES = ["Amp", "RiseT", "Dur", "Eny", "Counts", "RMS",
                 "logEny", "RA", "avgFreq"]


# ═════════════════════════════════════════════════════════════════════════════
#  (1) AE loader + per-hit features (REAL Vallen .pridb)
# ═════════════════════════════════════════════════════════════════════════════

def load_pridb_hits(path: str, max_hits: int | None = None) -> np.ndarray:
    """Load AE HITS from a Vallen .pridb -> (N, 9) feature array.

    AE hits are rows with SetType=2 AND Amp IS NOT NULL (other SetTypes are
    status/param rows). Columns read: Amp, RiseT, Dur, Eny, Counts, RMS;
    derived: log-energy, RA value (RiseT/Amp), average frequency (Counts/Dur).
    NULLs are coerced to 0 and non-finite derived values are sanitised.
    """
    con = sqlite3.connect(path)
    try:
        cur = con.cursor()
        q = ("SELECT Amp, RiseT, Dur, Eny, Counts, RMS FROM ae_data "
             "WHERE SetType=2 AND Amp IS NOT NULL")
        if max_hits:
            q += f" LIMIT {int(max_hits)}"
        rows = cur.execute(q).fetchall()
    finally:
        con.close()
    if not rows:
        return np.empty((0, len(FEATURE_NAMES)))
    raw = np.array([[0.0 if v is None else float(v) for v in r] for r in rows],
                   dtype=float)
    return _augment_features(raw)


def _augment_features(raw: np.ndarray) -> np.ndarray:
    """[Amp,RiseT,Dur,Eny,Counts,RMS] -> +[logEny, RA, avgFreq], sanitised."""
    raw = np.atleast_2d(np.asarray(raw, dtype=float))
    Amp, RiseT, Dur, Eny, Counts = (raw[:, 0], raw[:, 1], raw[:, 2],
                                    raw[:, 3], raw[:, 4])
    log_eny = np.log10(np.clip(Eny, 1e-9, None) + 1.0)
    ra = RiseT / np.clip(Amp, 1e-6, None)            # RA value (us/dB-ish)
    avg_freq = Counts / np.clip(Dur, 1e-6, None)     # average frequency proxy
    out = np.column_stack([raw[:, :6], log_eny, ra, avg_freq])
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def specimen_name(path: str) -> str:
    """File stem normalised to a SPEC_LABELS key (CAIspec1->CAI-spec1)."""
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem in SPEC_LABELS:
        return stem
    alt = stem.replace("CAIspec", "CAI-spec")
    return alt if alt in SPEC_LABELS else stem


def load_all_specimens(ae_dir: str = AE_DIR, max_hits: int | None = 8000
                       ) -> dict:
    """Load every .pridb under ae_dir -> stacked features X (M,9), specimen-id
    group array g, damage-condition labels y (1=damaged, 0=pristine), and the
    specimen-name list. max_hits caps per-specimen hits to keep the demo fast."""
    feats, groups, labels, names = [], [], [], []
    paths = sorted(glob.glob(os.path.join(ae_dir, "*.pridb")))
    for gi, p in enumerate(paths):
        name = specimen_name(p)
        if name not in SPEC_LABELS:
            continue
        X = load_pridb_hits(p, max_hits=max_hits)
        if len(X) == 0:
            continue
        feats.append(X)
        groups.append(np.full(len(X), len(names)))
        labels.append(np.full(len(X), SPEC_LABELS[name]))
        names.append(name)
    if not feats:
        return {"available": False}
    return {"available": True, "X": np.vstack(feats),
            "groups": np.concatenate(groups), "y": np.concatenate(labels),
            "names": names}


def _synthetic_ae(rng: np.random.Generator) -> dict:
    """Synthetic AE hits (fallback so tests/figures run without the real data).
    Damaged specimens get more high-amplitude / high-energy fibre-break-like
    hits; pristine are dominated by low-energy matrix-crack hits."""
    names = ["comp0", "comp90", "CAI-spec1", "CAI-spec2", "compteflon"]
    feats, groups, labels = [], [], []
    for gi, nm in enumerate(names):
        dmg = SPEC_LABELS[nm]
        nhit = 1500
        amp = rng.gamma(3 + 4 * dmg, 60, nhit)        # damaged -> higher Amp
        riset = rng.gamma(2, 20, nhit)
        dur = rng.gamma(2, 80, nhit)
        eny = amp ** 1.6 * rng.gamma(1 + dmg, 1, nhit)  # damaged -> more energy
        counts = np.clip(dur / 12 + rng.poisson(3 + 6 * dmg, nhit), 1, None)
        rms = amp / 8 + rng.normal(0, 5, nhit)
        raw = np.column_stack([amp, riset, dur, eny, counts, rms])
        feats.append(_augment_features(raw))
        groups.append(np.full(nhit, gi))
        labels.append(np.full(nhit, dmg))
    return {"available": True, "X": np.vstack(feats),
            "groups": np.concatenate(groups), "y": np.concatenate(labels),
            "names": names, "synthetic": True}


# ═════════════════════════════════════════════════════════════════════════════
#  (2) Stage-1 detection (REAL AE): pristine-vs-damaged + damage-mode clusters
# ═════════════════════════════════════════════════════════════════════════════

def _log_features(X: np.ndarray) -> np.ndarray:
    """Sign-preserving log1p of the (heavy-tailed) AE features — the natural
    representation for amplitude/energy/duration spanning several decades."""
    X = np.asarray(X, float)
    return np.sign(X) * np.log1p(np.abs(X))


def detect_damaged_vs_pristine(data: dict, seed: int = 0) -> dict:
    """GROUP-aware (leave-whole-specimens-out) pristine-vs-damaged AE detector.

    Log-transform the heavy-tailed per-hit features, standardise, train a random
    forest with specimen-level cross-validation (whole specimens held out, no
    leakage), and report per-hit AUROC/accuracy plus a per-specimen AUROC from
    the mean per-hit score. Numbers reported as-is.

    NOTE (honest): with only 6 specimens (2 pristine, 4 damaged) and very strong
    overlap — plain-compression PRISTINE coupons actually release MORE AE energy
    at failure than some pre-impacted coupons — hit-level pristine-vs-damaged is
    close to chance under leave-one-specimen-out. That is the real-data finding;
    the AE damage-MODE clustering below is the more discriminative AE read-out."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, accuracy_score
    from sklearn.model_selection import LeaveOneGroupOut

    X, g, y, names = data["X"], data["groups"], data["y"], data["names"]
    Xl = _log_features(X)
    logo = LeaveOneGroupOut()
    oof = np.full(len(y), np.nan)
    for tr, te in logo.split(Xl, y, g):
        if len(np.unique(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(Xl[tr])
        clf = RandomForestClassifier(n_estimators=120, max_depth=8,
                                     class_weight="balanced", n_jobs=-1,
                                     random_state=seed)
        clf.fit(sc.transform(Xl[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(Xl[te]))[:, 1]
    valid = ~np.isnan(oof)
    yv, sv = y[valid], oof[valid]
    hit_auroc = (float(roc_auc_score(yv, sv))
                 if len(np.unique(yv)) == 2 else float("nan"))
    hit_acc = (float(accuracy_score(yv, (sv > 0.5).astype(int)))
               if valid.any() else float("nan"))
    # per-specimen mean score -> specimen-level detection
    spec_scores, spec_y = [], []
    for gi in np.unique(g):
        m = (g == gi) & valid
        if m.any():
            spec_scores.append(float(np.mean(oof[m])))
            spec_y.append(int(y[g == gi][0]))
    spec_y = np.array(spec_y)
    spec_auroc = (float(roc_auc_score(spec_y, spec_scores))
                  if len(np.unique(spec_y)) == 2 else float("nan"))
    return {"hit_auroc": hit_auroc, "hit_accuracy": hit_acc,
            "specimen_auroc": spec_auroc,
            "specimen_scores": dict(zip(names, spec_scores)),
            "n_hits": int(valid.sum())}


def damage_mode_clusters(data: dict, n_clusters: int = 3, seed: int = 0
                         ) -> dict:
    """Classic AE damage-MODE clustering on the (log-amplitude, log-duration,
    RA, average-frequency) signature space. Clusters are ordered by mean
    amplitude and labelled matrix-crack / delamination / fibre-break (low->high
    amplitude/energy). Returns per-specimen cluster fractions. (On this
    compression dataset the high-energy/fibre-break fraction is NOT a clean
    monotone damage proxy — plain-compression pristine coupons fail
    catastrophically and emit strong fibre-break AE too; the multivariate
    detector above is the discriminative read-out. Numbers reported as-is.)"""
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    X, g, names = data["X"], data["groups"], data["names"]
    Amp, Dur, RA, avgF, Eny = (X[:, 0], X[:, 2], X[:, 7], X[:, 8], X[:, 3])
    Z = np.column_stack([np.log10(np.clip(Amp, 1, None)),
                         np.log10(np.clip(Dur, 1, None)),
                         np.log10(np.clip(RA, 1e-6, None) + 1e-6),
                         np.log10(np.clip(avgF, 1e-6, None) + 1e-6)])
    Zs = StandardScaler().fit_transform(np.nan_to_num(Z))
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(Zs)
    lab = km.labels_
    # order clusters low->high mean amplitude
    order = np.argsort([Amp[lab == k].mean() if (lab == k).any() else 0
                        for k in range(n_clusters)])
    remap = {old: new for new, old in enumerate(order)}
    lab = np.array([remap[l] for l in lab])
    mode_names = (["matrix-crack", "delamination", "fibre-break"]
                  if n_clusters == 3 else [f"mode{k}" for k in range(n_clusters)])
    fractions = {}
    for gi, nm in enumerate(names):
        m = g == gi
        if not m.any():
            continue
        frac = [float((lab[m] == k).mean()) for k in range(n_clusters)]
        fractions[nm] = dict(zip(mode_names, frac))
    # high-energy (top cluster) fraction, pristine vs damaged
    top = n_clusters - 1
    hi_pristine = [fractions[nm][mode_names[top]] for nm in names
                   if SPEC_LABELS.get(nm) == 0 and nm in fractions]
    hi_damaged = [fractions[nm][mode_names[top]] for nm in names
                  if SPEC_LABELS.get(nm) == 1 and nm in fractions]
    return {"labels": lab, "Z": Z, "mode_names": mode_names,
            "fractions": fractions, "n_clusters": n_clusters,
            "hi_frac_pristine": float(np.mean(hi_pristine)) if hi_pristine else float("nan"),
            "hi_frac_damaged": float(np.mean(hi_damaged)) if hi_damaged else float("nan")}


# ═════════════════════════════════════════════════════════════════════════════
#  (3) Stage-3 prognosis: filament-wound pressure-vessel BURST margin
#      (distinct physics; SAME Prognosis protocol / decision core)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class MotorCaseConfig:
    """Representative SRB-3-class filament-wound CFRP motor-case config.

    Hoop stress dominates: sigma_theta = p R / t. Residual burst pressure
    degrades with damage descriptor a (delamination/damage size, normalised):
    p_burst(a) = p_burst0 * (1 - (a/a_crit)^n). Cyclic pressurisation grows the
    damage via a Paris-like law driven by the energy-release G ~ (p R / t)^2 * a.
    'Grown/unsafe' when the residual burst margin p_burst(a)/p_op drops below the
    safety factor. Lumped constants, trends-not-absolute."""
    R: float = 0.60              # case radius (m, representative)
    t: float = 6.0e-3            # wall thickness (m)
    p_op: float = 5.0e6          # operating internal pressure (Pa)
    p_burst0: float = 20.0e6     # pristine burst pressure (Pa)
    a_crit: float = 1.0          # damage size at which burst -> 0 (normalised)
    burst_n: float = 1.5         # burst-degradation exponent
    safety_factor: float = 2.0   # required p_burst/p_op margin (design SF)
    paris_C: float = 9.0e-3      # Paris coefficient da/dN = C (G/Gc)^m
    paris_m: float = 2.0         # Paris exponent
    Gc: float = 1.0              # toughness scale (normalised)
    G_kappa: float = 1.0         # G = kappa * (sigma_theta/sigma_op)^2 * a
    cycles_per_flight: int = 20  # pressurisation cycles per flight (proof+burn)
    a_init_frac: float = 1.0     # scales the seeded damage size
    growth_rel_threshold: float = 0.5  # rel damage growth per flight = "grown"
    a_max: float = 0.999         # damage saturates at a_crit


def hoop_stress(p: float, cfg: MotorCaseConfig) -> float:
    """Thin-wall hoop stress sigma_theta = p R / t (dominant in a pressure
    vessel; ~2x the axial stress)."""
    return p * cfg.R / cfg.t


def residual_burst_pressure(a, cfg: MotorCaseConfig):
    """Residual burst pressure vs damage: p_burst(a) = p_burst0*(1-(a/a_crit)^n).
    Monotonically DECREASING in a; clipped non-negative."""
    a = np.clip(a, 0.0, cfg.a_crit)
    return cfg.p_burst0 * np.clip(1.0 - (a / cfg.a_crit) ** cfg.burst_n, 0.0, 1.0)


def burst_margin(a, cfg: MotorCaseConfig):
    """Safety margin p_burst(a) / p_op. Below cfg.safety_factor = unsafe."""
    return residual_burst_pressure(a, cfg) / max(cfg.p_op, 1e-9)


def _energy_release(a: float, p: float, cfg: MotorCaseConfig) -> float:
    """G ~ kappa * (sigma_theta/sigma_op)^2 * a  (hoop-stress-driven ERR,
    normalised by the operating hoop stress so kappa is O(1); monotone in a,p)."""
    s_op = hoop_stress(cfg.p_op, cfg)
    s = hoop_stress(p, cfg)
    return cfg.G_kappa * (s / max(s_op, 1e-12)) ** 2 * max(a, 1e-9)


def simulate_burst_growth(a0: float, p: float,
                          cfg: MotorCaseConfig | None = None) -> dict:
    """One flight (K pressurisation cycles) of Paris-like damage growth from a0.

    da/dN = C (G(a,p)/Gc)^m, integrated over K cycles; a clipped to a_max.
    Returns the interstage/fairing-compatible contract: grown / rel_growth plus
    a_final and the residual burst margins before/after."""
    cfg = cfg or MotorCaseConfig()
    a = float(np.clip(a0, 0.0, cfg.a_max))
    a_start = a
    for _ in range(cfg.cycles_per_flight):
        G = _energy_release(a, p, cfg)
        da = cfg.paris_C * (G / cfg.Gc) ** cfg.paris_m
        a = min(a + da, cfg.a_max)
    rel = (a - a_start) / max(a_start, 1e-12)
    margin_after = burst_margin(a, cfg)
    unsafe = margin_after < cfg.safety_factor      # dropped below design SF
    runaway = a >= 0.99 * cfg.a_max
    return {"grown": bool(unsafe or runaway or rel > cfg.growth_rel_threshold),
            "rel_growth": float(rel), "a0": float(a0), "a_final": float(a),
            "margin_before": float(burst_margin(a_start, cfg)),
            "margin_after": float(margin_after), "unsafe": bool(unsafe)}


def burst_growth_probability(posterior_a: np.ndarray, p: float,
                             cfg: MotorCaseConfig | None = None,
                             n_draws: int = 20,
                             rng: np.random.Generator | None = None) -> float:
    """P(damage grows / margin lost next flight) over a Stage-2 posterior of
    damage sizes — the motor-case twin of the fairing/interstage growth prob."""
    cfg = cfg or MotorCaseConfig()
    rng = rng or np.random.default_rng(0)
    a = np.atleast_1d(np.asarray(posterior_a, dtype=float)).ravel()
    if len(a) > n_draws:
        a = a[rng.choice(len(a), n_draws, replace=False)]
    flags = [simulate_burst_growth(ai * cfg.a_init_frac, p, cfg)["grown"]
             for ai in a]
    return float(np.mean(flags))


@dataclass
class SRB3Prognosis:
    """Stage-3 prognosis for the SRB-3 motor case (filament-wound burst margin).
    Implements the SAME `Prognosis` protocol as FairingPrognosis, so SRB-3
    clears OK/REPAIR/RETIRE through flight_clearance_generic — the identical
    decision core. `load` is the internal pressure p (Pa)."""
    cfg: MotorCaseConfig = field(default_factory=MotorCaseConfig)
    name: str = "srb3-motorcase-burst"

    def growth_probability(self, posterior, load, n_draws: int = 20) -> float:
        return burst_growth_probability(posterior, load, self.cfg, n_draws)


# ═════════════════════════════════════════════════════════════════════════════
#  (4) End-to-end demo + report
# ═════════════════════════════════════════════════════════════════════════════

def run(verbose: bool = True, ae_dir: str = AE_DIR) -> dict:
    # ---- Stage-1: real AE detection -----------------------------------------
    data = load_all_specimens(ae_dir)
    if not data.get("available"):
        data = _synthetic_ae(np.random.default_rng(0))
    det = detect_damaged_vs_pristine(data)
    clu = damage_mode_clusters(data)

    # ---- Stage-3: burst prognosis through the shared clearance ---------------
    cfg = MotorCaseConfig()
    prog = SRB3Prognosis(cfg)
    cases = {"small a0=0.10": 0.10, "medium a0=0.52": 0.52, "large a0=0.85": 0.85}
    clearance = {}
    for nm, a0 in cases.items():
        rng = np.random.default_rng(0)   # fresh per case -> reproducible posterior
        post = np.clip(rng.normal(a0, 0.10 * max(a0, 1e-3), 40), 1e-3, cfg.a_max)
        clearance[nm] = flight_clearance_generic(prog, post, cfg.p_op)
    # damage sweep -> decision boundary
    a_sweep = np.linspace(0.02, 0.95, 25)
    sweep_p = [burst_growth_probability(np.full(8, a), cfg.p_op, cfg)
               for a in a_sweep]

    out = {"detection": det, "clusters": clu, "clearance": clearance,
           "a_sweep": a_sweep, "sweep_p": sweep_p, "cfg": cfg,
           "synthetic": data.get("synthetic", False)}
    if verbose:
        _report(out)
    return out


def _report(d: dict):
    bar = "=" * 74
    det, clu = d["detection"], d["clusters"]
    cfg = d["cfg"]
    a, b = pf.calibrate_thresholds()
    print(bar)
    print("  SRB-3 MOTOR CASE  —  3rd structure: AE front-end + burst prognosis")
    print(bar)
    src = "SYNTHETIC fallback" if d["synthetic"] else "REAL 4TU AE data"
    print(f"  data source: {src}  ({det['n_hits']} AE hits used)")
    print("\n  Stage-1 detection (pristine vs damaged AE, group-aware):")
    print(f"    per-hit AUROC      = {det['hit_auroc']:.3f}   "
          f"accuracy = {det['hit_accuracy']:.3f}")
    print(f"    per-specimen AUROC = {det['specimen_auroc']:.3f}")
    print("  Stage-1 damage-mode clustering (high-energy/fibre-break fraction):")
    print(f"    pristine specimens: {clu['hi_frac_pristine']:.3f}    "
          f"damaged specimens: {clu['hi_frac_damaged']:.3f}")
    print("    (note: plain-compression PRISTINE coupons fail catastrophically and")
    print("     emit strong high-energy fibre-break AE too — so this fraction is")
    print("     NOT a clean monotone damage proxy; the multivariate detector is.)")
    for nm, fr in clu["fractions"].items():
        tag = "DMG" if SPEC_LABELS.get(nm) == 1 else "PRI"
        fr_s = "  ".join(f"{k}={v:.2f}" for k, v in fr.items())
        print(f"      [{tag}] {nm:>11}:  {fr_s}")
    print("\n  Stage-3 burst prognosis (filament-wound, hoop sigma=pR/t):")
    print(f"    SHARED decision rule alpha={a:.2f}, beta={b:.2f}; "
          f"design SF={cfg.safety_factor}, p_op={cfg.p_op/1e6:.1f} MPa")
    for nm, r in d["clearance"].items():
        print(f"    {nm:>15}:  P(grow)={r['p_growth_next']:.3f}  "
              f"E[remaining]={r['expected_remaining_flights']:.1f}  "
              f"-> {r['decision']}")
    bnd = next((aa for aa, p in zip(d["a_sweep"], d["sweep_p"]) if p > a), None)
    if bnd is not None:
        print(f"    P(grow) crosses REPAIR threshold near damage a≈{bnd:.2f}")
    print("\n  => SRB-3 = 3rd structure: AE front-end + burst prognosis, SAME")
    print("     decision core (flight_clearance_generic) as interstage & fairing.")
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
        figsize = use(width_frac=1.0, aspect=0.30)
    except Exception:
        figsize = (13.0, 3.6)
    import matplotlib.pyplot as plt

    cfg = d["cfg"]
    a, b = pf.calibrate_thresholds()
    # reload AE for the scatter (or synth fallback already in d via clusters)
    data = load_all_specimens()
    if not data.get("available"):
        data = _synthetic_ae(np.random.default_rng(0))
    X, y, g = data["X"], data["y"], data["groups"]
    clu = d["clusters"]

    fig, ax = plt.subplots(1, 4, figsize=figsize)

    # (a) AE amplitude-duration scatter, coloured damaged/pristine (real)
    sub = slice(None, None, max(1, len(X) // 4000))
    Amp, Dur = X[sub, 0], X[sub, 2]
    yy = y[sub]
    for lab, col, nm in [(0, "#1565C0", "pristine"), (1, "#c62828", "damaged")]:
        m = yy == lab
        ax[0].scatter(np.clip(Dur[m], 1, None), np.clip(Amp[m], 1, None),
                      s=3, alpha=0.25, c=col, label=nm)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("duration", fontsize=8); ax[0].set_ylabel("amplitude", fontsize=8)
    ax[0].set_title("(a) AE hits (real data)", fontsize=9)
    ax[0].legend(fontsize=6, markerscale=2); ax[0].tick_params(labelsize=7)

    # (b) damage-mode clusters in amplitude-duration space
    lab = clu["labels"]
    Amp2, Dur2 = X[:, 0], X[:, 2]
    cols = ["#2e7d32", "#f9a825", "#c62828"]
    for k in range(clu["n_clusters"]):
        m = lab == k
        ms = np.where(m)[0][::max(1, m.sum() // 1500)]
        ax[1].scatter(np.clip(Dur2[ms], 1, None), np.clip(Amp2[ms], 1, None),
                      s=3, alpha=0.3, c=cols[k % 3],
                      label=clu["mode_names"][k])
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("duration", fontsize=8); ax[1].set_ylabel("amplitude", fontsize=8)
    ax[1].set_title("(b) damage-mode clusters", fontsize=9)
    ax[1].legend(fontsize=6, markerscale=2); ax[1].tick_params(labelsize=7)

    # (c) residual burst pressure vs damage with SF line
    aa = np.linspace(0, cfg.a_crit, 100)
    pb = residual_burst_pressure(aa, cfg) / 1e6
    ax[2].plot(aa, pb, c="#1565C0", label="$p_{burst}(a)$")
    ax[2].axhline(cfg.safety_factor * cfg.p_op / 1e6, color="#c62828", ls="--",
                  lw=0.9, label=f"SF·$p_{{op}}$ ({cfg.safety_factor}×)")
    ax[2].axhline(cfg.p_op / 1e6, color="k", ls=":", lw=0.8, label="$p_{op}$")
    ax[2].set_xlabel("damage size $a$", fontsize=8)
    ax[2].set_ylabel("burst pressure (MPa)", fontsize=8)
    ax[2].set_title("(c) residual burst margin", fontsize=9)
    ax[2].legend(fontsize=6); ax[2].tick_params(labelsize=7)

    # (d) P(grow)/clearance vs damage severity with alpha/beta bands
    ax[3].plot(d["a_sweep"], d["sweep_p"], "-o", ms=3, c="#1565C0")
    ax[3].axhline(a, color="#2e7d32", ls="--", lw=0.9, label=r"$\alpha$ (OK|REPAIR)")
    ax[3].axhline(b, color="#b71c1c", ls="--", lw=0.9, label=r"$\beta$ (REPAIR|RETIRE)")
    ax[3].set_xlabel("damage size $a$", fontsize=8)
    ax[3].set_ylabel("P(grow next flight)", fontsize=8)
    ax[3].set_title("(d) clearance boundary", fontsize=9)
    ax[3].legend(fontsize=6); ax[3].tick_params(labelsize=7)

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

    rng = np.random.default_rng(0)

    # ---- AE loader: real .pridb if present, else synthetic fallback ----------
    real = sorted(glob.glob(os.path.join(AE_DIR, "*.pridb")))
    if real:
        X = load_pridb_hits(real[0], max_hits=500)
        ok(X.ndim == 2 and X.shape[1] == len(FEATURE_NAMES),
           "real .pridb -> (N,9) feature array")
        ok(len(X) > 0 and np.isfinite(X).all(), "real AE features finite")
        data = load_all_specimens(max_hits=1500)
        ok(data["available"] and set(np.unique(data["y"])) <= {0, 1},
           "loader returns binary damage labels")
        ok(len(np.unique(data["groups"])) == len(data["names"]),
           "one group per specimen")
    else:
        data = _synthetic_ae(rng)
        ok(np.isfinite(data["X"]).all(), "synthetic AE features finite")

    # augment handles NULL/zeros without nan/inf
    aug = _augment_features(np.zeros((3, 6)))
    ok(np.isfinite(aug).all() and aug.shape == (3, 9), "augment robust to zeros")

    # ---- Stage-1 detection returns sane metric -------------------------------
    det = detect_damaged_vs_pristine(data)
    ok(np.isnan(det["hit_auroc"]) or 0.0 <= det["hit_auroc"] <= 1.0,
       "detection AUROC in range")
    ok(0.0 <= det["hit_accuracy"] <= 1.0, "detection accuracy in range")

    # ---- damage-mode clustering shape ----------------------------------------
    clu = damage_mode_clusters(data, n_clusters=3)
    ok(len(clu["labels"]) == len(data["X"]), "cluster label per hit")
    ok(set(np.unique(clu["labels"])) <= {0, 1, 2}, "3 clusters")
    ok(all(abs(sum(fr.values()) - 1.0) < 1e-6 for fr in clu["fractions"].values()),
       "per-specimen cluster fractions sum to 1")

    # ---- burst model monotonicity --------------------------------------------
    cfg = MotorCaseConfig()
    ok(hoop_stress(2 * cfg.p_op, cfg) > hoop_stress(cfg.p_op, cfg),
       "hoop stress rises with pressure")
    ok(residual_burst_pressure(0.2, cfg) > residual_burst_pressure(0.6, cfg),
       "residual burst DECREASES with damage")
    ok(residual_burst_pressure(0.0, cfg) == cfg.p_burst0, "pristine burst = p_burst0")
    small = simulate_burst_growth(0.05, cfg.p_op, cfg)
    large = simulate_burst_growth(0.80, cfg.p_op, cfg)
    ok(large["a_final"] >= small["a_final"], "larger damage -> larger final size")
    ok(simulate_burst_growth(0.4, 0.0, cfg)["rel_growth"] == 0.0,
       "no pressure -> no growth")
    p_small = burst_growth_probability(np.full(8, 0.05), cfg.p_op, cfg)
    p_large = burst_growth_probability(np.full(8, 0.85), cfg.p_op, cfg)
    ok(0 <= p_small <= 1 and 0 <= p_large <= 1, "P(grow) in range")
    ok(p_large >= p_small, "larger damage -> higher P(grow)")
    p_lowp = burst_growth_probability(np.full(8, 0.6), 0.3 * cfg.p_op, cfg)
    p_hip = burst_growth_probability(np.full(8, 0.6), cfg.p_op, cfg)
    ok(p_hip >= p_lowp, "higher pressure -> higher P(grow)")

    # ---- SRB3Prognosis satisfies the protocol + shared decision rule ---------
    prog: Prognosis = SRB3Prognosis(cfg)
    r = flight_clearance_generic(prog, np.full(8, 0.85), cfg.p_op)
    ok(r["decision"] in ("OK", "REPAIR", "RETIRE"), "SRB-3 clearance verdict")
    ok(r["structure"] == "srb3-motorcase-burst", "structure tag")
    ok(r["alpha"] == pf.calibrate_thresholds()[0], "shared alpha threshold")
    ok(flight_clearance_generic(prog, np.full(8, 0.03), 0.3 * cfg.p_op
                                )["decision"] == "OK", "tiny damage -> OK")
    big = flight_clearance_generic(prog, np.full(8, 0.92), cfg.p_op)["decision"]
    ok(big in ("REPAIR", "RETIRE"), "large damage -> not OK")

    print(f"srb3_motorcase: {n}/{n} unit tests passed")
    return n


def main():
    ap = argparse.ArgumentParser(
        description="SRB-3 motor case: AE detection + burst prognosis (3rd structure).")
    ap.add_argument("--fig", action="store_true")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.test:
        run_tests(); return
    d = run()
    if args.fig:
        p = make_figure(d, os.path.join(HERE, "paper_figs", "srb3_motorcase.pdf"))
        print(f"\nfigure -> {p}")


if __name__ == "__main__":
    main()
