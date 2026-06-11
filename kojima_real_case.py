"""
kojima_real_case.py — cross-geometry external-validity on Kojima's REAL CFRP
prosthetic structure (gaps #3/#8 of the Stage 0-5 backlog).

Honest scope
------------
The whole Stage 0-5 stack is trained/validated on the SYNTHETIC 2-D
plate-with-hole coupon (13 942-node grid, parametric defects).  Kojima's
prosthetic case (`CFRP/CFRP_kojima/`) is a genuinely DIFFERENT real-application
structure: a curved 3-D CFRP+PMMA socket, solved in Abaqus and exported as a
graph with the per-node DSPSS (sum-of-principal-stresses) field
(`graph_with_values_DSPSS_1.vtk`, 9 852 nodes / 19 303 edges).

It is NOT measured experimental data — it is FEM of the real application — so it
does NOT by itself close the sim-to-real gap (#1, still open until true
measurements arrive).  What it DOES give, for the first time, is:

  * a real, independent geometry to test whether the stack transfers at all
    (#3 cross-geometry transfer was previously unproven), and
  * an external reference case to quantify the domain gap (#8), the natural
    stepping stone before measured data.

The screening idea
------------------
Stage-0's per-node Mahalanobis needs node correspondence across samples, which
does NOT exist between a 13 942-node coupon grid and a 9 852-node curved socket.
So we use a GEOMETRY-AGNOSTIC detector that needs no correspondence: the
per-sample standardised DSPSS magnitude |z|, z=(v−μ)/σ over the sample's own
nodes — the same two-sided semantics as Stage-0's Mahalanobis, but with the
per-sample healthy reference replaced by the sample itself.

We validate on a LABELLED defect coupon: |z| reaches AUROC 0.85 (and 0.93
one-sided, since a stiffness-reducing defect is a stress CONCENTRATION —
higher DSPSS), then apply the identical detector to the real prosthetic and
quantify the coupon→socket domain gap.  An alternative graph-neighbourhood
z-score (local contrast over the VTK edges) was tried and REJECTED: it scores
AUROC 0.23 because the defect core is locally smooth — the signal lives in
absolute magnitude, not local gradient (an honest negative result we keep).

Usage
-----
    python kojima_real_case.py            # validate + transfer report
    python kojima_real_case.py --fig      # + paper_figs/kojima_real_case.pdf
    python kojima_real_case.py --test     # unit tests only
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CFRP = "/home/nishioka/CFRP"
KOJIMA_VTK = f"{CFRP}/CFRP_kojima/graph_with_values_DSPSS_1.vtk"
COUPON_HEALTHY = f"{CFRP}/DSPSS.vtk/dspss_nodefect.vtk"
COUPON_DEFECT = f"{CFRP}/DSPSS.vtk/DSPSS_4x4_Layer19_Block46.vtk"
COUPON_LABEL = f"{CFRP}/DSPSS.vtk/defectlabel_test_0_L19B46.vtk"


# ═════════════════════════════════════════════════════════════════════════════
#  VTK POLYDATA loader (POINTS / LINES / POINT_DATA SCALARS) — ASCII
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Graph:
    points: np.ndarray        # (N, 3)
    edges: np.ndarray         # (E, 2) int
    values: np.ndarray        # (N,)   DSPSS / label scalar
    name: str

    @property
    def n(self) -> int:
        return len(self.points)


def load_vtk_graph(path: str) -> Graph:
    """Parse an ASCII VTK POLYDATA graph (POINTS, LINES, POINT_DATA SCALARS).

    LINES are stored as polylines "<k> i0 i1 ... i_{k-1}"; we emit the
    consecutive index pairs as undirected edges.
    """
    with open(path) as fh:
        tok = fh.read().split("\n")
    i = 0
    points = edges = values = None
    while i < len(tok):
        line = tok[i].strip()
        parts = line.split()
        if not parts:
            i += 1
            continue
        key = parts[0].upper()
        if key == "POINTS":
            n = int(parts[1])
            vals = []
            i += 1
            while len(vals) < 3 * n:
                vals.extend(tok[i].split())
                i += 1
            points = np.array(vals[:3 * n], dtype=float).reshape(n, 3)
            continue
        if key == "LINES":
            n_lines = int(parts[1])
            ints = []
            i += 1
            # read until we've consumed the LINES connectivity block
            need = int(parts[2])
            while len(ints) < need:
                ints.extend(tok[i].split())
                i += 1
            ints = list(map(int, ints[:need]))
            ep, k = [], 0
            for _ in range(n_lines):
                cnt = ints[k]; idx = ints[k + 1:k + 1 + cnt]; k += 1 + cnt
                for a, b in zip(idx[:-1], idx[1:]):
                    ep.append((a, b))
            edges = np.array(ep, dtype=int) if ep else np.zeros((0, 2), int)
            continue
        if key == "SCALARS":
            # next line is LOOKUP_TABLE, then the values
            i += 2
            vals = []
            while i < len(tok) and len(vals) < (len(points) if points is not None
                                                else 1 << 30):
                s = tok[i].split()
                if s and not s[0][0].isalpha():
                    vals.extend(s); i += 1
                else:
                    break
            values = np.array(vals, dtype=float)
            continue
        i += 1
    if points is None:
        raise ValueError(f"no POINTS in {path}")
    if edges is None:
        edges = np.zeros((0, 2), int)
    if values is None:
        values = np.zeros(len(points))
    return Graph(points=points, edges=edges, values=values,
                 name=os.path.basename(path))


# ═════════════════════════════════════════════════════════════════════════════
#  Geometry-agnostic Stage-0: per-node graph-neighbourhood DSPSS z-score
# ═════════════════════════════════════════════════════════════════════════════

def field_anomaly(values: np.ndarray, mode: str = "abs") -> np.ndarray:
    """Geometry-agnostic Stage-0 score: per-sample standardised DSPSS.

    z = (v − μ)/σ over the sample's OWN nodes (no node correspondence, no
    healthy twin needed) — the two-sided |z| matches Stage-0's Mahalanobis
    semantics.  mode='abs' → |z| (two-sided, general); mode='signed' → z
    (one-sided, exploits that a stiffness-reducing defect is a stress
    concentration, higher DSPSS).
    """
    z = (values - values.mean()) / (values.std() + 1e-8)
    return np.abs(z) if mode == "abs" else z


def _adjacency(edges: np.ndarray, n: int) -> list:
    adj = [[] for _ in range(n)]
    for a, b in edges:
        if a != b:
            adj[a].append(b); adj[b].append(a)
    return adj


def graph_neighbor_anomaly(values: np.ndarray, edges: np.ndarray, n: int,
                           hops: int = 1) -> np.ndarray:
    """Per-node local outlier score |v_i − μ_i| / σ_i, where μ_i, σ_i are the
    mean/SD of v over node i's `hops`-neighbourhood (graph edges only).

    Needs NO node correspondence between samples — so it transfers to any mesh
    or structure.  Field is first per-sample standardised (pipeline convention).
    Isolated nodes (no neighbours) fall back to the global z-score.
    """
    v = (values - values.mean()) / (values.std() + 1e-8)
    adj = _adjacency(edges, n)
    if hops > 1:
        adj = _expand_hops(adj, hops)
    score = np.empty(n)
    for k in range(n):
        nb = adj[k]
        if not nb:
            score[k] = abs(v[k])
            continue
        nv = v[nb]
        score[k] = abs(v[k] - nv.mean()) / (nv.std() + 1e-6)
    return score


def _expand_hops(adj: list, hops: int) -> list:
    out = [set(a) for a in adj]
    frontier = [set(a) for a in adj]
    for _ in range(hops - 1):
        nxt = []
        for k in range(len(adj)):
            grow = set()
            for j in frontier[k]:
                grow.update(adj[j])
            grow.discard(k)
            nxt.append(grow - out[k])
            out[k].update(grow)
        frontier = nxt
    return [sorted(s) for s in out]


# ═════════════════════════════════════════════════════════════════════════════
#  Domain-gap metrics (coupon healthy vs prosthetic)
# ═════════════════════════════════════════════════════════════════════════════

def _moments(v: np.ndarray) -> dict:
    z = (v - v.mean()) / (v.std() + 1e-12)
    return {"mean": float(v.mean()), "sd": float(v.std()),
            "skew": float((z ** 3).mean()), "kurt": float((z ** 4).mean() - 3.0)}


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample KS distance between standardised fields (no SciPy)."""
    za = np.sort((a - a.mean()) / (a.std() + 1e-12))
    zb = np.sort((b - b.mean()) / (b.std() + 1e-12))
    grid = np.unique(np.concatenate([za, zb]))
    Fa = np.searchsorted(za, grid, side="right") / len(za)
    Fb = np.searchsorted(zb, grid, side="right") / len(zb)
    return float(np.max(np.abs(Fa - Fb)))


def domain_gap(coupon: np.ndarray, prosthetic: np.ndarray) -> dict:
    return {"coupon": _moments(coupon), "prosthetic": _moments(prosthetic),
            "ks_standardised": ks_statistic(coupon, prosthetic)}


# ═════════════════════════════════════════════════════════════════════════════
#  Study
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class CaseResult:
    coupon_auroc: float          # primary detector: per-sample |z|
    coupon_auroc_signed: float   # one-sided z (stress concentration)
    coupon_auroc_neigh: float    # rejected graph-neighbourhood variant
    coupon_label_frac: float
    proto: Graph
    proto_score: np.ndarray
    proto_flag_frac: float
    gap: dict


def run_case(flag_q: float = 0.95, hops: int = 1,
             verbose: bool = True) -> CaseResult:
    """Validate the geometry-agnostic detector on a labelled coupon (AUROC),
    then transfer it to the real prosthetic and quantify the domain gap.

    Primary detector = per-sample |z| (field_anomaly, Stage-0 Mahalanobis
    semantics).  The graph-neighbourhood z-score is also scored and reported as
    a tried-and-rejected alternative (the defect core is locally smooth).
    """
    from sklearn.metrics import roc_auc_score

    # ── (A) validate on labelled coupon ──────────────────────────────────────
    cg = load_vtk_graph(COUPON_DEFECT)
    lg = load_vtk_graph(COUPON_LABEL)
    label = (lg.values > 0.5).astype(int)
    m = min(len(label), cg.n)
    label = label[:m]

    def _au(score):
        return (float(roc_auc_score(label, score[:m]))
                if 0 < label.sum() < len(label) else float("nan"))

    auroc = _au(field_anomaly(cg.values[:m], mode="abs"))
    auroc_signed = _au(field_anomaly(cg.values[:m], mode="signed"))
    auroc_neigh = _au(graph_neighbor_anomaly(cg.values[:m], cg.edges, m,
                                             hops=hops))

    # ── (B) transfer the VALIDATED detector to the real prosthetic ───────────
    pg = load_vtk_graph(KOJIMA_VTK)
    pscore = field_anomaly(pg.values, mode="abs")
    thr = float(np.quantile(pscore, flag_q))
    flag_frac = float((pscore > thr).mean())

    # ── domain gap: healthy coupon vs prosthetic ─────────────────────────────
    hg = load_vtk_graph(COUPON_HEALTHY)
    gap = domain_gap(hg.values, pg.values)

    res = CaseResult(coupon_auroc=auroc, coupon_auroc_signed=auroc_signed,
                     coupon_auroc_neigh=auroc_neigh,
                     coupon_label_frac=float(label.mean()),
                     proto=pg, proto_score=pscore, proto_flag_frac=flag_frac,
                     gap=gap)
    if verbose:
        _report(res, cg, hg, flag_q)
    return res


def _report(r: CaseResult, cg: Graph, hg: Graph, flag_q: float):
    bar = "=" * 72
    print(bar)
    print("  CROSS-GEOMETRY EXTERNAL VALIDITY  —  Kojima real CFRP prosthetic")
    print(bar)
    print("  NOTE: FEM of the real application, NOT measured data — this is the")
    print("        cross-geometry external test (#3/#8), not sim-to-real (#1).")
    print()
    print(f"  (A) detector validation on a LABELLED coupon "
          f"({cg.name}, N={cg.n}, defect nodes {r.coupon_label_frac*100:.1f}%):")
    print(f"      per-sample |z|  (Stage-0 Mahalanobis sem.) AUROC = "
          f"{r.coupon_auroc:.3f}   ← PRIMARY")
    print(f"      one-sided z     (defect = stress concentr.) AUROC = "
          f"{r.coupon_auroc_signed:.3f}")
    print(f"      graph-neighbourhood z (local contrast)      AUROC = "
          f"{r.coupon_auroc_neigh:.3f}   ← rejected (defect core smooth)")
    print(f"      → per-sample |z| detects defects WITHOUT node correspondence")
    print()
    print(f"  (B) transfer to the REAL prosthetic ({r.proto.name}):")
    print(f"      nodes {r.proto.n}, edges {len(r.proto.edges)}   "
          f"DSPSS μ={r.gap['prosthetic']['mean']:.2f} "
          f"σ={r.gap['prosthetic']['sd']:.2f}")
    print(f"      flagged hotspots at q={flag_q:.2f}: "
          f"{r.proto_flag_frac*100:.1f}% of nodes "
          f"(max score {r.proto_score.max():.2f})")
    print()
    print(f"  (C) coupon→prosthetic DOMAIN GAP (healthy {hg.name}, N={hg.n}):")
    g = r.gap
    print(f"      {'':>10}{'mean':>9}{'sd':>9}{'skew':>8}{'kurt':>8}")
    for tag in ("coupon", "prosthetic"):
        d = g[tag]
        print(f"      {tag:>10}{d['mean']:>9.2f}{d['sd']:>9.2f}"
              f"{d['skew']:>8.2f}{d['kurt']:>8.2f}")
    print(f"      KS distance (standardised fields) = {g['ks_standardised']:.3f}"
          f"   → the measured cross-geometry domain gap")
    print(bar)


# ═════════════════════════════════════════════════════════════════════════════
#  Figure
# ═════════════════════════════════════════════════════════════════════════════

def make_figure(r: CaseResult, out_path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import sys
    sys.path.insert(0, os.path.join(HERE, "slides", "figure_sources"))
    try:
        from thesis_style import use
        figsize = use(width_frac=1.0, aspect=0.34)
    except Exception:
        figsize = (12.0, 3.8)
    import matplotlib.pyplot as plt

    pg = r.proto
    P = pg.points - pg.points.mean(0)
    _, _, Vt = np.linalg.svd(P, full_matrices=False)
    xy = P @ Vt.T[:, :2]

    fig, ax = plt.subplots(1, 3, figsize=figsize)

    sc0 = ax[0].scatter(xy[:, 0], xy[:, 1], c=pg.values, s=4, cmap="viridis",
                        rasterized=True)
    ax[0].set_title("(a) prosthetic DSPSS field", fontsize=8)
    ax[0].set_aspect("equal"); ax[0].set_xticks([]); ax[0].set_yticks([])
    fig.colorbar(sc0, ax=ax[0], fraction=0.046, pad=0.02)

    hg = load_vtk_graph(COUPON_HEALTHY)
    zc = (hg.values - hg.values.mean()) / (hg.values.std() + 1e-12)
    zp = (pg.values - pg.values.mean()) / (pg.values.std() + 1e-12)
    ax[1].hist(zc, bins=60, density=True, alpha=0.55, label="coupon (healthy)",
               color="#1565C0")
    ax[1].hist(zp, bins=60, density=True, alpha=0.55, label="prosthetic",
               color="#d7301f")
    ax[1].set_title(f"(b) domain gap  KS={r.gap['ks_standardised']:.2f}",
                    fontsize=8)
    ax[1].set_xlabel("standardised DSPSS", fontsize=7)
    ax[1].legend(fontsize=6); ax[1].tick_params(labelsize=6)

    thr = np.quantile(r.proto_score, 0.95)
    hot = r.proto_score > thr
    ax[2].scatter(xy[~hot, 0], xy[~hot, 1], c="0.8", s=3, rasterized=True)
    sc2 = ax[2].scatter(xy[hot, 0], xy[hot, 1], c=r.proto_score[hot], s=10,
                        cmap="inferno", rasterized=True)
    ax[2].set_title(f"(c) Stage-0 hotspots (AUROC {r.coupon_auroc:.2f} on coupon)",
                    fontsize=8)
    ax[2].set_aspect("equal"); ax[2].set_xticks([]); ax[2].set_yticks([])
    fig.colorbar(sc2, ax=ax[2], fraction=0.046, pad=0.02)

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

    # tiny synthetic graph: a line 0-1-2-3-4 with one spike at node 2
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    vals = np.array([1.0, 1.0, 9.0, 1.0, 1.0])
    sc = graph_neighbor_anomaly(vals, edges, 5)
    ok(np.argmax(sc) == 2, "spike node has the highest neighbourhood score")
    ok(sc.shape == (5,), "score shape")

    fa = field_anomaly(vals, mode="abs")
    ok(np.argmax(fa) == 2, "per-sample |z| peaks at the spike")
    ok(np.all(fa >= 0), "|z| non-negative")
    ok(field_anomaly(vals, mode="signed")[2] > 0, "signed z positive at spike")

    adj = _adjacency(edges, 5)
    ok(adj[2] == [1, 3], "adjacency of interior node")
    ok(_adjacency(np.array([[0, 0]]), 1)[0] == [], "self-loop dropped")

    a2 = _expand_hops(_adjacency(edges, 5), 2)
    ok(set(a2[2]) == {0, 1, 3, 4}, "2-hop neighbourhood")

    ks = ks_statistic(np.random.default_rng(0).normal(0, 1, 2000),
                      np.random.default_rng(1).normal(0, 1, 2000))
    ok(ks < 0.1, "KS small for same distribution")
    ks2 = ks_statistic(np.random.default_rng(0).normal(0, 1, 2000),
                       np.random.default_rng(1).normal(0, 1, 2000) ** 2)
    ok(ks2 > 0.2, "KS large for different distribution")

    m = _moments(np.array([0.0, 0.0, 0.0, 10.0]))
    ok(m["skew"] > 0, "right-skewed moment")

    # VTK round-trip on a tiny in-memory file
    import tempfile
    txt = ("# vtk DataFile Version 4.2\nx\nASCII\nDATASET POLYDATA\n"
           "POINTS 3 float\n0 0 0\n1 0 0\n2 0 0\n"
           "LINES 2 6\n2 0 1\n2 1 2\n"
           "POINT_DATA 3\nSCALARS NodeValues float\nLOOKUP_TABLE default\n"
           "1.0\n2.0\n3.0\n")
    with tempfile.NamedTemporaryFile("w", suffix=".vtk", delete=False) as fh:
        fh.write(txt); p = fh.name
    g = load_vtk_graph(p)
    os.unlink(p)
    ok(g.n == 3 and g.edges.shape == (2, 2), "vtk points+edges parsed")
    ok(np.allclose(g.values, [1, 2, 3]), "vtk scalars parsed")
    ok(tuple(g.edges[0]) == (0, 1), "vtk edge pair")

    print(f"kojima_real_case: {n}/{n} unit tests passed")
    return n


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Cross-geometry external validity on Kojima's real CFRP "
                    "prosthetic (gaps #3/#8).")
    ap.add_argument("--flag-q", type=float, default=0.95,
                    help="hotspot flag quantile on the prosthetic")
    ap.add_argument("--hops", type=int, default=1,
                    help="graph-neighbourhood radius for the z-score")
    ap.add_argument("--fig", action="store_true",
                    help="save paper_figs/kojima_real_case.pdf")
    ap.add_argument("--test", action="store_true", help="unit tests only")
    args = ap.parse_args()

    if args.test:
        run_tests(); return
    r = run_case(flag_q=args.flag_q, hops=args.hops)
    if args.fig:
        p = make_figure(r, os.path.join(HERE, "paper_figs",
                                        "kojima_real_case.pdf"))
        print(f"\nfigure → {p}")


if __name__ == "__main__":
    main()
