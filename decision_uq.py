"""
decision_uq.py — END-TO-END DECISION-LEVEL uncertainty quantification.

The Stage 0-5 stack calibrates every stage in isolation (FMPE posterior
coverage, conformal surrogate intervals, fleet posterior spread), but nowhere
does it answer the one question an operator actually asks:

    "The pipeline just said OK / REPAIR / RETIRE — how often is that final
     call right, and where does the uncertainty come from?"

This module propagates uncertainty all the way to the CLEARANCE DECISION and
turns it into a single auditable number, using the exact FD phase-field forward
(`cfrp_phasefield_2d.simulate_growth`) as ground-truth physics — so NO new
measured data is needed.

Ground truth
------------
For a known defect θ_true the next-flight growth is deterministic at a fixed
load, so a single FD solve gives only {grow, no-grow}.  The genuine aleatoric
source is OPERATIONAL LOAD SCATTER: the next flight's peak transverse strain
varies about the nominal.  Integrating the exact FD forward over that load
distribution yields a real growth PROBABILITY P_true ∈ [0,1], and the SAME
expected-cost rule (`calibrate_thresholds`, α=0.02, β=0.48) maps it to the
ORACLE decision — the best call anyone could make knowing θ exactly.

Four decisions per scenario (the error decomposition)
-----------------------------------------------------
    oracle        θ_true  + FD          → the label
    surr-oracle   θ_true  + surrogate   → isolates SURROGATE forward error
    fd-posterior  FMPE θ  + FD          → isolates Stage-2 POSTERIOR error
    pipeline      FMPE θ  + surrogate   → the actual shipped product (total)

Disagreement(·, oracle) for each isolates a contribution; pipeline-vs-oracle is
the end-to-end decision error.  The posteriors are REAL FMPE posteriors of
held-out test samples (built once via --build, cached), so this is the true
end-to-end chain, not an emulation.

Headline numbers
----------------
  * end-to-end decision accuracy  P(pipeline = oracle)
  * DANGEROUS-MISS rate           P(pipeline less severe than oracle) and in
                                  particular P(pipeline = OK | oracle = RETIRE)
  * decomposition                 surrogate-only / posterior-only / total flips
  * per-decision confidence       bootstrap the (posterior × load) draws → a
                                  distribution over {OK,REPAIR,RETIRE} for a
                                  SINGLE vehicle ("this OK call is 87 % reliable")
                                  — then validated against the oracle (reliability
                                  diagram), closing the UQ loop.

Usage
-----
    python decision_uq.py --build            # train FMPE once, cache test posteriors
    python decision_uq.py --smoke            # tiny fast sanity run (no cache needed*)
    python decision_uq.py                    # full study (uses cached posteriors + FD)
    python decision_uq.py --fig              # + paper_figs/decision_uq.pdf
    python decision_uq.py --test             # unit tests only

(* --smoke falls back to a Gaussian posterior emulation if the FMPE cache is
   absent, so it runs anywhere; the headline study requires --build first.)
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

import numpy as np

import cfrp_phasefield_2d as pf
import crack_surrogate as cs

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "results", "decision_uq")
FMPE_CACHE = os.path.join(CACHE_DIR, "fmpe_testset.npz")
STUDY_CACHE = os.path.join(CACHE_DIR, "study.npz")

# in-distribution envelope (crack_surrogate); nominal loads spread across it so
# the scenario set spans all three decision classes.
LOAD_LO, LOAD_HI = cs.LOAD_LO, cs.LOAD_HI
SEVERITY = {"OK": 0, "REPAIR": 1, "RETIRE": 2}
DECISIONS = ("OK", "REPAIR", "RETIRE")


# ═════════════════════════════════════════════════════════════════════════════
#  Decision rule (identical semantics to run_pipeline / flight_clearance)
# ═════════════════════════════════════════════════════════════════════════════

def decide(p: float, alpha: float, beta: float) -> str:
    return "OK" if p <= alpha else "REPAIR" if p <= beta else "RETIRE"


def load_grid(nominal: float, n_load: int, delta: float = 0.12) -> np.ndarray:
    """Operational peak-load scatter: nominal·(1±delta), clipped to envelope."""
    if n_load <= 1:
        return np.array([nominal], dtype=float)
    g = np.linspace(-1.0, 1.0, n_load)
    return np.clip(nominal * (1.0 + delta * g), LOAD_LO, LOAD_HI)


# ═════════════════════════════════════════════════════════════════════════════
#  Exact FD forward — one big picklable job pool
# ═════════════════════════════════════════════════════════════════════════════

def _fd_grow(args) -> bool:
    """Worker: one FD simulate_growth grow flag for (θ, load).  Top-level."""
    from cfrp_phasefield_2d import LaminateConfig, seed_defect, simulate_growth
    cx, cy, layer, l2s, load = args
    cfg = LaminateConfig()
    d0 = seed_defect(cx, cy, layer, l2s, cfg)
    return bool(simulate_growth(d0, float(load), cfg).grown)


def _run_fd(jobs: list, n_workers: int | None) -> np.ndarray:
    """Evaluate a flat list of (cx,cy,layer,l2s,load) FD jobs → bool array."""
    if not jobs:
        return np.zeros(0, dtype=bool)
    if n_workers is None:
        n_workers = min(len(jobs), max(1, (os.cpu_count() or 2) - 2))
    if n_workers > 1 and len(jobs) > 1:
        import multiprocessing as mp
        with mp.Pool(n_workers) as pool:
            out = pool.map(_fd_grow, jobs, chunksize=4)
    else:
        out = [_fd_grow(j) for j in jobs]
    return np.array(out, dtype=bool)


# ═════════════════════════════════════════════════════════════════════════════
#  Surrogate forward (free) — P(grow) over (draws × loads)
# ═════════════════════════════════════════════════════════════════════════════

def _surrogate_probs(thetas: np.ndarray, loads: np.ndarray, model,
                     cfg: pf.LaminateConfig) -> np.ndarray:
    """Per-(θ,load) surrogate P(grow), shape (len(thetas), len(loads))."""
    thetas = np.atleast_2d(thetas)
    nL = len(loads)
    d0s, ld = [], []
    for cx, cy, layer, l2s in thetas:
        d0 = pf.seed_defect(cx, cy, layer, l2s, cfg)
        for L in loads:
            d0s.append(d0)
            ld.append(float(L))
    out = cs.predict_batch(model, np.stack(d0s), np.asarray(ld), device="cpu")
    return out["p_grow"].reshape(len(thetas), nL)


# ═════════════════════════════════════════════════════════════════════════════
#  Stage-2 posteriors: REAL FMPE (built once) or Gaussian emulation fallback
# ═════════════════════════════════════════════════════════════════════════════

def build_fmpe_testset(n_keep: int = 48, epochs: int = 120,
                       seed: int = 42) -> dict:
    """Train FMPE once on the real CFRP data, cache (truth, posterior) pairs for
    `n_keep` held-out test samples.  Mirrors fmpe_defect / run_pipeline --fmpe.
    """
    import torch
    import fmpe_defect as fm

    print("[build] FMPE end-to-end (heavy, one-off)…", flush=True)
    fields, thetas = fm.build_dataset()
    n = len(fields)
    rng = np.random.default_rng(seed)
    mu_f = fields.mean(0)
    Xc = fields - mu_f
    _, _, Vt = np.linalg.svd(Xc[rng.choice(n, min(3000, n), replace=False)],
                             full_matrices=False)
    V = Vt[:fm.PCA_K]
    embed = Xc @ V.T
    e_mu, e_sd = embed.mean(0), embed.std(0) + 1e-6
    embed = (embed - e_mu) / e_sd
    t_mu, t_sd = thetas.mean(0), thetas.std(0) + 1e-6
    th_n = (thetas - t_mu) / t_sd

    idx = rng.permutation(n)
    n_tr = int(0.8 * n)
    tr, te = idx[:n_tr], idx[n_tr:]
    C = torch.from_numpy(embed.astype(np.float32)).to(fm.DEVICE)
    TH = torch.from_numpy(th_n.astype(np.float32)).to(fm.DEVICE)

    model = fm.VelocityNet().to(fm.DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    BS = 512
    for ep in range(epochs):
        perm = torch.randperm(n_tr, device=fm.DEVICE)
        for i in range(0, n_tr, BS):
            b = torch.from_numpy(tr).to(fm.DEVICE)[perm[i:i + BS]]
            th0, c = TH[b], C[b]
            t = torch.rand(len(b), device=fm.DEVICE)
            eps = torch.randn_like(th0)
            tht = (1 - t)[:, None] * th0 + t[:, None] * eps
            loss = ((model(tht, t, c) - (eps - th0)) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 20 == 0:
            print(f"  ep {ep+1}/{epochs} loss={loss.item():.4f}", flush=True)

    keep = te[:n_keep]
    posteriors, truths = [], []
    for j in keep:
        post = fm.sample_posterior(model, C[j]).cpu().numpy() * t_sd + t_mu
        posteriors.append(post.astype(np.float32))
        truths.append(thetas[j].astype(np.float32))
    posteriors = np.stack(posteriors)        # (M, N, 4)
    truths = np.stack(truths)                # (M, 4)

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(FMPE_CACHE, truths=truths, posteriors=posteriors,
                        source="real FMPE, held-out test samples")
    print(f"[build] cached {len(truths)} posteriors → {FMPE_CACHE}")
    return {"truths": truths, "posteriors": posteriors}


def emulated_testset(n_keep: int, n_draws: int = 200, seed: int = 0) -> dict:
    """Fallback when the FMPE cache is absent (--smoke anywhere): θ_true from the
    surrogate's in-distribution envelope + a Gaussian posterior whose per-param
    spread matches the cached single real FMPE posterior if available, else a
    conservative default.  Clearly NOT the real posterior — smoke use only.
    """
    rng = np.random.default_rng(seed)
    # per-parameter posterior SD, taken from the real cached posterior if present
    sd = np.array([0.06, 0.08, 1.6, 0.35])
    cp = os.path.join(HERE, "results", "fmpe_posterior_e2e.npz")
    if os.path.exists(cp):
        z = np.load(cp, allow_pickle=True)
        sd = z["posterior"].astype(float).std(0)
    truths = np.column_stack([
        rng.uniform(cs.CX_LO, cs.CX_HI, n_keep),
        rng.uniform(0.0, 1.0, n_keep),
        rng.uniform(cs.LAYER_LO, cs.LAYER_HI, n_keep),
        rng.uniform(cs.LOG2SIZE_LO, cs.LOG2SIZE_HI, n_keep),
    ]).astype(np.float32)
    posteriors = (truths[:, None, :]
                  + rng.normal(0, 1, (n_keep, n_draws, 4)) * sd[None, None, :]
                  ).astype(np.float32)
    return {"truths": truths, "posteriors": posteriors,
            "source": "EMULATED Gaussian posterior (smoke only)"}


def load_testset(n_keep: int) -> dict:
    if os.path.exists(FMPE_CACHE):
        z = np.load(FMPE_CACHE, allow_pickle=True)
        return {"truths": z["truths"][:n_keep].astype(float),
                "posteriors": z["posteriors"][:n_keep].astype(float),
                "source": str(z["source"])}
    return emulated_testset(n_keep)


# ═════════════════════════════════════════════════════════════════════════════
#  The study
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Study:
    nominal: np.ndarray                  # (M,) nominal load per scenario
    p_oracle: np.ndarray                 # (M,) FD prob at θ_true
    p_surr_oracle: np.ndarray            # (M,) surrogate prob at θ_true
    p_fd_post: np.ndarray                # (M,) FD prob over posterior
    p_pipeline: np.ndarray               # (M,) surrogate prob over posterior
    conf: np.ndarray                     # (M, 3) bootstrap decision confidence
    alpha: float
    beta: float
    source: str

    def dec(self, which: str) -> np.ndarray:
        p = getattr(self, f"p_{which}")
        return np.array([decide(pi, self.alpha, self.beta) for pi in p])


def run_study(n_keep: int = 40, n_load: int = 5, n_draws_fd: int = 6,
              n_draws_surr: int = 64, n_boot: int = 400,
              n_workers: int | None = None, seed: int = 0,
              verbose: bool = True) -> Study:
    """Full end-to-end decision-UQ study over `n_keep` scenarios.

    All FD jobs (oracle + fd-posterior, across every scenario and load) are
    flattened into ONE pool.map; the surrogate paths are batched and free.
    """
    rng = np.random.default_rng(seed)
    cfg = pf.LaminateConfig()
    alpha, beta = pf.calibrate_thresholds()
    ts = load_testset(n_keep)
    truths, posteriors = ts["truths"], ts["posteriors"]
    M = len(truths)

    # nominal loads tiled across the envelope so all three classes appear
    nominal = np.linspace(LOAD_LO + 0.01, LOAD_HI - 0.01, M)
    nominal = nominal[rng.permutation(M)]
    loads = [load_grid(nominal[s], n_load) for s in range(M)]

    model, _ = cs.load_surrogate(device="cpu")

    # ── surrogate paths (free) ───────────────────────────────────────────────
    p_surr_oracle = np.empty(M)
    p_pipeline = np.empty(M)
    conf = np.empty((M, 3))
    fd_draw_idx = []                      # which posterior draws go to FD
    for s in range(M):
        Ls = loads[s]
        # surr-oracle: θ_true × loads
        po = _surrogate_probs(truths[s], Ls, model, cfg)        # (1, nL)
        p_surr_oracle[s] = float(po.mean())
        # pipeline: n_draws_surr posterior draws × loads
        post = posteriors[s]
        di = rng.choice(len(post), min(n_draws_surr, len(post)), replace=False)
        pp = _surrogate_probs(post[di], Ls, model, cfg)         # (nd, nL)
        p_pipeline[s] = float(pp.mean())
        # per-sample decision confidence: bootstrap the (draw,load) grid
        flat = pp.reshape(-1)
        bidx = rng.integers(0, len(flat), (n_boot, len(flat)))
        bp = flat[bidx].mean(1)                                  # (n_boot,)
        cnt = np.zeros(3)
        for bpi in bp:
            cnt[SEVERITY[decide(bpi, alpha, beta)]] += 1
        conf[s] = cnt / n_boot
        # pick the FD-posterior draws (subset, exact forward is expensive)
        fdi = di[:n_draws_fd]
        fd_draw_idx.append(fdi)

    # ── exact FD paths: one big pool ─────────────────────────────────────────
    jobs, tags = [], []                   # tag = (s, 'oracle'|'post', k)
    for s in range(M):
        Ls = loads[s]
        for L in Ls:
            cx, cy, ly, l2 = truths[s]
            jobs.append((cx, cy, ly, l2, L)); tags.append((s, "oracle", 0))
        for k in fd_draw_idx[s]:
            cx, cy, ly, l2 = posteriors[s][k]
            for L in Ls:
                jobs.append((cx, cy, ly, l2, L)); tags.append((s, "post", k))

    if verbose:
        print(f"[study] M={M} scenarios, {len(jobs)} FD solves "
              f"(n_load={n_load}, n_draws_fd={n_draws_fd})…", flush=True)
    t0 = time.perf_counter()
    flags = _run_fd(jobs, n_workers)
    dt = time.perf_counter() - t0

    p_oracle = np.zeros(M)
    post_acc = [[] for _ in range(M)]
    o_acc = [[] for _ in range(M)]
    for (s, kind, _k), f in zip(tags, flags):
        (o_acc if kind == "oracle" else post_acc)[s].append(f)
    p_oracle = np.array([np.mean(o_acc[s]) for s in range(M)])
    p_fd_post = np.array([np.mean(post_acc[s]) for s in range(M)])

    if verbose:
        print(f"[study] FD done in {dt:.1f}s "
              f"({dt/max(len(jobs),1)*1e3:.0f} ms/solve eff.)")
    return Study(nominal=nominal, p_oracle=p_oracle,
                 p_surr_oracle=p_surr_oracle, p_fd_post=p_fd_post,
                 p_pipeline=p_pipeline, conf=conf, alpha=alpha, beta=beta,
                 source=ts["source"])


# ═════════════════════════════════════════════════════════════════════════════
#  Metrics
# ═════════════════════════════════════════════════════════════════════════════

def confusion(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """3×3 confusion (rows=truth/oracle, cols=pred) over DECISIONS."""
    C = np.zeros((3, 3), dtype=int)
    for p, t in zip(pred, truth):
        C[SEVERITY[t], SEVERITY[p]] += 1
    return C


def severity(dec: np.ndarray) -> np.ndarray:
    return np.array([SEVERITY[d] for d in dec])


def metrics(study: Study) -> dict:
    o = study.dec("oracle")
    so = study.dec("surr_oracle")
    fp = study.dec("fd_post")
    pl = study.dec("pipeline")
    M = len(o)
    s_o, s_pl = severity(o), severity(pl)

    miss = float(np.mean(s_pl < s_o))           # under-call (UNSAFE)
    alarm = float(np.mean(s_pl > s_o))          # over-call (conservative)
    n_retire = int((o == "RETIRE").sum())
    dmiss = (float(np.mean(pl[o == "RETIRE"] == "OK")) if n_retire else
             float("nan"))                      # OK while oracle says RETIRE
    return {
        "M": M,
        "acc_pipeline": float(np.mean(pl == o)),
        "acc_surr_oracle": float(np.mean(so == o)),
        "acc_fd_post": float(np.mean(fp == o)),
        "flip_surrogate_only": float(np.mean(so != o)),
        "flip_posterior_only": float(np.mean(fp != o)),
        "flip_total": float(np.mean(pl != o)),
        "unsafe_miss_rate": miss,
        "conservative_alarm_rate": alarm,
        "dangerous_miss_OK_given_RETIRE": dmiss,
        "n_retire_oracle": n_retire,
        "oracle_class_counts": {d: int((o == d).sum()) for d in DECISIONS},
        "confusion_pipeline_vs_oracle": confusion(pl, o),
    }


def reliability(study: Study, n_bins: int = 5) -> dict:
    """Validate the per-sample decision confidence: bin scenarios by the
    confidence the pipeline assigned to ITS OWN chosen decision, then measure
    how often that decision actually matched the oracle.  Well-calibrated → the
    empirical agreement tracks the assigned confidence (diagonal)."""
    pl = study.dec("pipeline")
    o = study.dec("oracle")
    own_conf = np.array([study.conf[s, SEVERITY[pl[s]]] for s in range(len(pl))])
    correct = (pl == o).astype(float)
    edges = np.linspace(1.0 / 3.0, 1.0, n_bins + 1)   # confidence ∈ [1/3, 1]
    mids, emp, cnt = [], [], []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        m = (own_conf >= lo) & (own_conf <= hi if b == n_bins - 1
                                else own_conf < hi)
        if m.sum() == 0:
            continue
        mids.append(float(own_conf[m].mean()))
        emp.append(float(correct[m].mean()))
        cnt.append(int(m.sum()))
    ece = float(np.sum(np.array(cnt) / len(pl)
                       * np.abs(np.array(mids) - np.array(emp)))) if cnt else \
        float("nan")
    return {"conf_mid": mids, "emp_acc": emp, "count": cnt, "ece": ece,
            "own_conf": own_conf, "correct": correct}


# ═════════════════════════════════════════════════════════════════════════════
#  Report + figure
# ═════════════════════════════════════════════════════════════════════════════

def print_report(study: Study, m: dict, rel: dict):
    bar = "=" * 72
    print(bar)
    print("  END-TO-END DECISION-LEVEL UQ  —  is the final OK/REPAIR/RETIRE right?")
    print(bar)
    print(f"  posterior source : {study.source}")
    print(f"  scenarios        : M={m['M']}   "
          f"thresholds α={study.alpha:.2f}, β={study.beta:.2f}")
    print(f"  oracle classes   : " +
          "  ".join(f"{d}={m['oracle_class_counts'][d]}" for d in DECISIONS))
    print()
    print(f"  END-TO-END decision accuracy (pipeline = oracle): "
          f"{m['acc_pipeline']*100:5.1f}%")
    print(f"  ── unsafe under-call  (less severe than oracle) : "
          f"{m['unsafe_miss_rate']*100:5.1f}%")
    print(f"  ── conservative alarm (more severe than oracle) : "
          f"{m['conservative_alarm_rate']*100:5.1f}%")
    dm = m["dangerous_miss_OK_given_RETIRE"]
    print(f"  ── DANGEROUS miss  P(OK | oracle=RETIRE)        : "
          + ("n/a" if dm != dm else f"{dm*100:5.1f}%")
          + f"   (n_retire={m['n_retire_oracle']})")
    print()
    print("  ── error decomposition (decision flips vs oracle) ──")
    print(f"     surrogate-only (θ_true, FNO forward) : "
          f"{m['flip_surrogate_only']*100:5.1f}%")
    print(f"     posterior-only (FMPE θ, FD forward)  : "
          f"{m['flip_posterior_only']*100:5.1f}%")
    print(f"     TOTAL pipeline                       : "
          f"{m['flip_total']*100:5.1f}%")
    print()
    print("  ── confusion (rows=oracle, cols=pipeline) ──")
    C = m["confusion_pipeline_vs_oracle"]
    print(f"           {'OK':>6}{'REPAIR':>8}{'RETIRE':>8}")
    for i, d in enumerate(DECISIONS):
        print(f"     {d:>7}{C[i,0]:>6}{C[i,1]:>8}{C[i,2]:>8}")
    print()
    print(f"  ── per-decision confidence calibration ──")
    print(f"     ECE(own-decision confidence) = {rel['ece']:.3f}")
    for cm, ea, ct in zip(rel["conf_mid"], rel["emp_acc"], rel["count"]):
        print(f"     conf≈{cm:.2f}:  empirical agree {ea*100:5.1f}%  (n={ct})")
    print(bar)


def make_figure(study: Study, m: dict, rel: dict, out_path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import sys
    sys.path.insert(0, os.path.join(HERE, "slides", "figure_sources"))
    try:
        from thesis_style import use
        figsize = use(width_frac=1.0, aspect=0.30)
    except Exception:
        figsize = (12.0, 3.4)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 4, figsize=figsize)

    # (a) confusion matrix
    C = m["confusion_pipeline_vs_oracle"]
    ax[0].imshow(C, cmap="Blues")
    ax[0].set_xticks(range(3)); ax[0].set_yticks(range(3))
    ax[0].set_xticklabels(DECISIONS, fontsize=6, rotation=30)
    ax[0].set_yticklabels(DECISIONS, fontsize=6)
    for i in range(3):
        for j in range(3):
            ax[0].text(j, i, C[i, j], ha="center", va="center", fontsize=8,
                       color="k" if C[i, j] < C.max() * 0.6 else "w")
    ax[0].set_title(f"(a) decision confusion\nacc={m['acc_pipeline']*100:.0f}%",
                    fontsize=8)
    ax[0].set_xlabel("pipeline", fontsize=7)
    ax[0].set_ylabel("oracle (FD truth)", fontsize=7)

    # (b) error decomposition
    labs = ["surrogate\nonly", "posterior\nonly", "TOTAL"]
    vals = [m["flip_surrogate_only"], m["flip_posterior_only"], m["flip_total"]]
    ax[1].bar(labs, np.array(vals) * 100,
              color=["#1565C0", "#2e7d32", "#b71c1c"])
    ax[1].set_ylabel("decision flips vs oracle (%)", fontsize=7)
    ax[1].set_title("(b) where the error comes from", fontsize=8)
    ax[1].tick_params(labelsize=6)

    # (c) oracle vs pipeline P(grow), thresholds, colored by agreement
    o = study.dec("oracle"); pl = study.dec("pipeline")
    agree = pl == o
    ax[2].scatter(study.p_oracle[agree], study.p_pipeline[agree], s=14,
                  c="#2e7d32", label="agree", alpha=0.8)
    ax[2].scatter(study.p_oracle[~agree], study.p_pipeline[~agree], s=18,
                  c="#b71c1c", marker="x", label="flip")
    for thr in (study.alpha, study.beta):
        ax[2].axvline(thr, color="0.6", lw=0.7, ls="--")
        ax[2].axhline(thr, color="0.6", lw=0.7, ls="--")
    ax[2].plot([0, 1], [0, 1], color="0.7", lw=0.6)
    ax[2].set_xlim(-0.02, 1.02); ax[2].set_ylim(-0.02, 1.02)
    ax[2].set_xlabel("oracle $P_{grow}$", fontsize=7)
    ax[2].set_ylabel("pipeline $P_{grow}$", fontsize=7)
    ax[2].set_title("(c) decisions vs truth", fontsize=8)
    ax[2].legend(fontsize=6, loc="upper left")
    ax[2].tick_params(labelsize=6)

    # (d) confidence reliability
    ax[3].plot([1/3, 1], [1/3, 1], color="0.7", lw=0.7, ls="--")
    if rel["conf_mid"]:
        ax[3].plot(rel["conf_mid"], rel["emp_acc"], "-o", ms=4, c="#1565C0")
    ax[3].set_xlim(0.3, 1.02); ax[3].set_ylim(0.3, 1.02)
    ax[3].set_xlabel("assigned decision confidence", fontsize=7)
    ax[3].set_ylabel("empirical oracle agreement", fontsize=7)
    ax[3].set_title(f"(d) confidence calibration\nECE={rel['ece']:.2f}",
                    fontsize=8)
    ax[3].tick_params(labelsize=6)

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

    a, b = pf.calibrate_thresholds()
    ok(decide(0.0, a, b) == "OK", "p=0 → OK")
    ok(decide(a + 1e-9, a, b) == "REPAIR", "just above α → REPAIR")
    ok(decide(1.0, a, b) == "RETIRE", "p=1 → RETIRE")
    ok(decide(b, a, b) == "REPAIR", "p=β → REPAIR (≤β)")

    g = load_grid(0.10, 5)
    ok(len(g) == 5 and g.min() >= LOAD_LO and g.max() <= LOAD_HI, "load grid clip")
    ok(np.isclose(load_grid(0.10, 1)[0], 0.10), "n_load=1 returns nominal")
    ok(np.all(np.diff(g) >= -1e-9), "load grid sorted ascending")

    C = confusion(np.array(["OK", "RETIRE", "REPAIR"]),
                  np.array(["OK", "REPAIR", "REPAIR"]))
    ok(C.sum() == 3 and C[0, 0] == 1, "confusion counts")
    ok(C[1, 2] == 1, "oracle REPAIR→pipeline RETIRE off-diagonal")

    ok(np.array_equal(severity(np.array(["OK", "REPAIR", "RETIRE"])),
                      np.array([0, 1, 2])), "severity map")

    # emulated tiny study end-to-end (serial FD, no cache needed)
    st = run_study(n_keep=4, n_load=2, n_draws_fd=2, n_draws_surr=6,
                   n_boot=50, n_workers=1, seed=1, verbose=False)
    ok(st.p_oracle.shape == (4,), "oracle shape")
    ok(np.all((st.conf >= 0) & (st.conf <= 1)), "confidence in [0,1]")
    ok(np.allclose(st.conf.sum(1), 1.0), "confidence rows sum to 1")
    ok(np.all((st.p_pipeline >= 0) & (st.p_pipeline <= 1)), "pipeline prob range")
    m = metrics(st)
    ok(0.0 <= m["acc_pipeline"] <= 1.0, "accuracy in range")
    ok(m["confusion_pipeline_vs_oracle"].sum() == 4, "confusion total = M")
    ok(0.0 <= m["unsafe_miss_rate"] <= 1.0, "miss rate range")
    r = reliability(st, n_bins=3)
    ok(r["ece"] == r["ece"], "ece is finite-or-nan but computed")

    print(f"decision_uq: {n}/{n} unit tests passed")
    return n


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="End-to-end decision-level UQ for the SHM clearance call.")
    ap.add_argument("--build", action="store_true",
                    help="train FMPE once and cache held-out test posteriors")
    ap.add_argument("--n-keep", type=int, default=40,
                    help="number of scenarios (test samples)")
    ap.add_argument("--n-load", type=int, default=5,
                    help="operational load-scatter grid size (aleatoric)")
    ap.add_argument("--n-draws-fd", type=int, default=6,
                    help="posterior draws sent through the EXACT FD forward")
    ap.add_argument("--n-draws-surr", type=int, default=64,
                    help="posterior draws sent through the surrogate")
    ap.add_argument("--n-boot", type=int, default=400,
                    help="bootstrap reps for per-decision confidence")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny fast run (works without the FMPE cache)")
    ap.add_argument("--fig", action="store_true",
                    help="save paper_figs/decision_uq.pdf")
    ap.add_argument("--test", action="store_true", help="unit tests only")
    args = ap.parse_args()

    if args.test:
        run_tests(); return
    if args.build:
        build_fmpe_testset(n_keep=max(args.n_keep, 48)); return

    if args.smoke:
        kw = dict(n_keep=6, n_load=3, n_draws_fd=3, n_draws_surr=16,
                  n_boot=100)
    else:
        kw = dict(n_keep=args.n_keep, n_load=args.n_load,
                  n_draws_fd=args.n_draws_fd, n_draws_surr=args.n_draws_surr,
                  n_boot=args.n_boot)
    study = run_study(n_workers=args.workers, seed=args.seed, **kw)
    m = metrics(study)
    rel = reliability(study)
    print_report(study, m, rel)
    if args.fig:
        p = make_figure(study, m, rel,
                        os.path.join(HERE, "paper_figs", "decision_uq.pdf"))
        print(f"\nfigure → {p}")


if __name__ == "__main__":
    main()
