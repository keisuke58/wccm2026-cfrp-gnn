"""
tsa_sim2real.py — FIRST real measured full-field data in the loop (gap #1).

Kojima's master's-thesis campaign (Jan 2024) measured the full-field surface
stress of CFRP prosthetic coupons by Thermoelastic Stress Analysis (TSA, a
FLIR/Cedip IR camera + Altair lock-in), under 0.5 Hz / 2 kN / 200-cycle loading.
Each specimen's field is a 320×255 map of stress amplitude in MPa
(`No*_curve.txt`).  This is the EXPERIMENTAL analogue of the FEM DSPSS
(sum-of-principal-stresses) field on which the GNN/Stage-0 stack is trained — so
it is exactly the measured datum gap #1 (sim-to-real) needs, finally on disk.

This module
  * parses the TSA full-field `.txt` export into an (H,W) MPa array,
  * quantifies the SIM-TO-REAL domain gap (measured TSA vs FEM DSPSS): the same
    standardised-field KS / moment comparison used for the cross-geometry case,
  * runs the geometry-agnostic Stage-0 screen (per-sample |z|, validated AUROC
    0.85 on the labelled FEM coupon in kojima_real_case.py) on the REAL fields
    and flags stress hot-spots.

HONEST SCOPE: 3 specimens, no per-pixel defect labels on disk (the dissipated-
energy/defect labelling lives in the thesis), so this is the sim-to-real GAP
quantification + a qualitative real-data Stage-0 demo — not a labelled
detection benchmark.  It is, however, the first time the pipeline ingests
measured full-field stress, the stepping stone the backlog called for.

Usage
-----
    python tsa_sim2real.py            # parse + sim2real gap + Stage-0 report
    python tsa_sim2real.py --fig      # + paper_figs/tsa_sim2real.pdf
    python tsa_sim2real.py --test     # unit tests only
"""
from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass

import numpy as np

import kojima_real_case as kr   # reuse field_anomaly / ks_statistic / _moments

HERE = os.path.dirname(os.path.abspath(__file__))
TSA_DIR = "/home/nishioka/from_kojima/02_TSA_fullfield_stress_2401"
FEM_DSPSS_VTK = "/home/nishioka/CFRP/CFRP_kojima/graph_with_values_DSPSS_1.vtk"
# Kojima's processed measured DSPSS (the SAME quantity as the FEM DSPSS) — the
# clean apples-to-apples sim2real set: 22 specimens (12 convex + 10 concave).
DERIVED = "/home/nishioka/from_kojima/copaam_derived"
RECT_DSPSS = {  # measured DSPSS full-field grids, per defect geometry
    "totsu(convex)": f"{DERIVED}/exp_to_FEM_CNN/rect_dspss_exp_totsu.npy",
    "ou(concave)": f"{DERIVED}/exp_to_FEM_CNN/rect_dspss_exp_ou.npy",
}
# FEM DSPSS reference pool (the GNN's FEM training graphs on the 9852-node mesh;
# "original" = pre-augmentation real NODES, not measured — it is FEM).
FEM_GRAPHSTRESS = f"{DERIVED}/Pros_dte_graphstress/graphstress_original_train.npy"


# ═════════════════════════════════════════════════════════════════════════════
#  TSA full-field loader
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class TSAField:
    name: str
    field: np.ndarray        # (H, W) stress amplitude, MPa
    shape_hdr: tuple         # (W, H) as written in the "Format" header


def load_tsa_field(path: str) -> TSAField:
    """Parse an Altair/Cedip TSA full-field text export.

    Header block ("General information", File, Frame, Total frames, Format W H,
    …) followed by H tab-separated rows of W stress values (MPa, decimal comma
    or point).  Returns the first frame as an (H, W) float array.
    """
    rows, fmt = [], None
    with open(path, errors="ignore") as fh:
        for ln in fh:
            if ln.startswith("Format"):
                p = ln.split()
                fmt = (int(p[-2]), int(p[-1]))            # (W, H)
            toks = ln.replace(",", ".").split("\t")
            try:
                vals = [float(x) for x in toks if x.strip() != ""]
            except ValueError:
                vals = []
            if len(vals) >= 100:
                rows.append(vals)
    if not rows:
        raise ValueError(f"no data rows in {path}")
    w = max(len(r) for r in rows)
    rows = [r for r in rows if len(r) == w]
    field = np.array(rows, dtype=float)
    return TSAField(name=os.path.basename(path).split("_")[0],
                    field=field, shape_hdr=fmt or (field.shape[1],
                                                   field.shape[0]))


def load_all_tsa(tsa_dir: str = TSA_DIR) -> list:
    files = sorted(glob.glob(os.path.join(tsa_dir, "No*_curve.txt")))
    return [load_tsa_field(f) for f in files]


def fem_dspss_values(path: str = FEM_DSPSS_VTK) -> np.ndarray:
    """The FEM DSPSS nodal field of the prosthetic (sim side of sim2real)."""
    return kr.load_vtk_graph(path).values


def load_rect_dspss(paths: dict = RECT_DSPSS) -> list:
    """Measured DSPSS full-field grids → list of (name, (H,W) field).

    Each file is (n_specimens, H, W); a specimen's field carries zero-padding
    outside the specimen mask, which we drop (>0) before the distribution stats.
    """
    out = []
    for tag, p in paths.items():
        arr = np.load(p)
        for i, fld in enumerate(arr):
            out.append((f"{tag}#{i:02d}", np.asarray(fld, dtype=float)))
    return out


def fem_dspss_pool(path: str = FEM_GRAPHSTRESS) -> np.ndarray:
    """FEM DSPSS reference distribution: all nodal values over the FEM training
    graphs (richer than a single field).  Falls back to the single VTK field."""
    if os.path.exists(path):
        return np.load(path).ravel()
    return fem_dspss_values()


# ═════════════════════════════════════════════════════════════════════════════
#  Sim-to-real domain gap + Stage-0 on real fields
# ═════════════════════════════════════════════════════════════════════════════

def robust_stats(v: np.ndarray) -> dict:
    """Mean/SD plus robust median/IQR (TSA has grip/edge outliers)."""
    v = v[np.isfinite(v)]
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return {"mean": float(v.mean()), "sd": float(v.std()),
            "median": float(med), "iqr": float(q3 - q1),
            "max": float(v.max())}


def sim2real_gap(measured: np.ndarray, fem: np.ndarray) -> dict:
    """Standardised-field domain gap between a measured TSA field and the FEM
    DSPSS field (same comparison as the cross-geometry case, kojima_real_case).
    """
    m = measured[np.isfinite(measured)].ravel()
    return {"ks_standardised": kr.ks_statistic(m, fem),
            "measured": robust_stats(m), "fem": robust_stats(fem),
            "moments_measured": kr._moments(m), "moments_fem": kr._moments(fem)}


def stage0_on_field(field: np.ndarray, flag_q: float = 0.99) -> dict:
    """Geometry-agnostic Stage-0 |z| on a real measured field; flag the top
    quantile as stress hot-spots (the validated detector from kojima_real_case).
    """
    flat = field.ravel()
    score = kr.field_anomaly(flat, mode="abs").reshape(field.shape)
    thr = float(np.quantile(score, flag_q))
    return {"score": score, "thr": thr,
            "flag_frac": float((score > thr).mean()),
            "max_sigma": float(score.max())}


@dataclass
class TSAResult:
    fields: list                 # list[TSAField]
    gaps: list                   # per-specimen sim2real_gap dict
    stage0: list                 # per-specimen stage0 dict
    fem: np.ndarray


def run(flag_q: float = 0.99, verbose: bool = True) -> TSAResult:
    fields = load_all_tsa()
    if not fields:
        raise FileNotFoundError(f"no TSA No*_curve.txt under {TSA_DIR}")
    fem = fem_dspss_values()
    gaps = [sim2real_gap(f.field, fem) for f in fields]
    s0 = [stage0_on_field(f.field, flag_q) for f in fields]
    res = TSAResult(fields=fields, gaps=gaps, stage0=s0, fem=fem)
    if verbose:
        _report(res, flag_q)
    return res


def _report(r: TSAResult, flag_q: float):
    bar = "=" * 72
    print(bar)
    print("  SIM-TO-REAL  —  measured TSA full-field stress vs FEM DSPSS (gap #1)")
    print(bar)
    print("  FIRST measured full-field data in the loop: TSA (FLIR/Cedip + Altair),")
    print("  CFRP prosthetic coupons, 0.5 Hz / 2 kN / 200 cyc, Jan 2024.")
    print(f"  FEM DSPSS reference: {os.path.basename(FEM_DSPSS_VTK)} "
          f"(N={len(r.fem)} nodes)\n")
    for f, g, s in zip(r.fields, r.gaps, r.stage0):
        mm, fm = g["measured"], g["fem"]
        print(f"  ── {f.name}  ({f.field.shape[0]}×{f.field.shape[1]} px, MPa) ──")
        print(f"     measured: median {mm['median']:.1f}  IQR {mm['iqr']:.1f}  "
              f"max {mm['max']:.0f} MPa")
        print(f"     sim2real KS(standardised) vs FEM DSPSS = "
              f"{g['ks_standardised']:.3f}")
        print(f"     Stage-0 |z| hot-spots at q={flag_q:.2f}: "
              f"{s['flag_frac']*100:.1f}% of px  (max {s['max_sigma']:.1f}σ)")
    ks = np.mean([g["ks_standardised"] for g in r.gaps])
    print(f"\n  mean sim2real domain gap KS = {ks:.3f}  "
          f"→ the measured sim-to-real gap (the real #1 number)")
    print(bar)


# ═════════════════════════════════════════════════════════════════════════════
#  Figure
# ═════════════════════════════════════════════════════════════════════════════

def make_figure(r: TSAResult, out_path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import sys
    sys.path.insert(0, os.path.join(HERE, "slides", "figure_sources"))
    try:
        from thesis_style import use
        figsize = use(width_frac=1.0, aspect=0.52)
    except Exception:
        figsize = (12.0, 6.2)
    import matplotlib.pyplot as plt

    n = len(r.fields)
    fig, ax = plt.subplots(2, n + 1, figsize=figsize)

    # top row: measured fields (robust colour range)
    for j, f in enumerate(r.fields):
        v = f.field
        vmax = np.percentile(v, 99)
        im = ax[0, j].imshow(v, cmap="viridis", vmin=0, vmax=vmax,
                             aspect="auto")
        ax[0, j].set_title(f"{f.name} TSA stress (MPa)", fontsize=8)
        ax[0, j].set_xticks([]); ax[0, j].set_yticks([])
        fig.colorbar(im, ax=ax[0, j], fraction=0.046, pad=0.02)

    # top-right: sim2real domain-gap histogram (standardised)
    zf = (r.fem - r.fem.mean()) / (r.fem.std() + 1e-12)
    ax[0, n].hist(zf, bins=60, density=True, alpha=0.5, color="#1565C0",
                  label="FEM DSPSS")
    allm = np.concatenate([f.field.ravel() for f in r.fields])
    zm = (allm - allm.mean()) / (allm.std() + 1e-12)
    ax[0, n].hist(np.clip(zm, -3, 6), bins=60, density=True, alpha=0.5,
                  color="#d7301f", label="measured TSA")
    ks = np.mean([g["ks_standardised"] for g in r.gaps])
    ax[0, n].set_title(f"sim2real gap  KS={ks:.2f}", fontsize=8)
    ax[0, n].set_xlabel("standardised stress", fontsize=7)
    ax[0, n].legend(fontsize=6); ax[0, n].tick_params(labelsize=6)

    # bottom row: Stage-0 hot-spot maps on the real fields
    for j, (f, s) in enumerate(zip(r.fields, r.stage0)):
        im = ax[1, j].imshow(s["score"], cmap="inferno", vmin=0, vmax=6,
                             aspect="auto")
        ax[1, j].set_title(f"{f.name} Stage-0 |z| (max {s['max_sigma']:.0f}σ)",
                           fontsize=8)
        ax[1, j].set_xticks([]); ax[1, j].set_yticks([])
        fig.colorbar(im, ax=ax[1, j], fraction=0.046, pad=0.02)

    # bottom-right: per-specimen KS bars
    names = [f.name for f in r.fields]
    ax[1, n].bar(names, [g["ks_standardised"] for g in r.gaps], color="#2e7d32")
    ax[1, n].set_title("per-specimen sim2real KS", fontsize=8)
    ax[1, n].tick_params(labelsize=6); ax[1, n].set_ylim(0, 1)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    fig.savefig(os.path.splitext(out_path)[0] + ".png", bbox_inches="tight",
                dpi=150)
    plt.close(fig)
    return out_path


# ═════════════════════════════════════════════════════════════════════════════
#  Apples-to-apples: measured DSPSS vs FEM DSPSS (same quantity, 22 specimens)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class DSPSSResult:
    specimens: list              # list[(name, (H,W) field)]
    ks: list                     # per-specimen sim2real KS vs FEM pool
    stage0: list                 # per-specimen stage0 dict
    fem_pool: np.ndarray


def run_dspss(flag_q: float = 0.99, verbose: bool = True) -> DSPSSResult:
    """Clean sim2real: Kojima's MEASURED DSPSS (rect_dspss_exp, 22 specimens) vs
    the FEM DSPSS pool — identical physical quantity, so the standardised-field
    KS is a like-for-like sim-to-real domain gap over a real specimen set."""
    specs = load_rect_dspss()
    fem = fem_dspss_pool()
    ks, s0 = [], []
    for _, fld in specs:
        v = fld[fld > 0]                         # drop zero-padding outside mask
        ks.append(kr.ks_statistic(v, fem))
        s0.append(stage0_on_field(fld, flag_q))
    res = DSPSSResult(specimens=specs, ks=ks, stage0=s0, fem_pool=fem)
    if verbose:
        _report_dspss(res, flag_q)
    return res


def _report_dspss(r: DSPSSResult, flag_q: float):
    bar = "=" * 72
    print(bar)
    print("  SIM-TO-REAL (apples-to-apples)  —  measured DSPSS vs FEM DSPSS")
    print(bar)
    print(f"  measured: rect_dspss_exp, {len(r.specimens)} CFRP prosthetic "
          f"specimens (convex+concave)")
    print(f"  FEM ref : DSPSS pool, {r.fem_pool.size} nodal values "
          f"(mean {r.fem_pool.mean():.1f} MPa)\n")
    by = {}
    for (name, _), ks in zip(r.specimens, r.ks):
        by.setdefault(name.split("#")[0], []).append(ks)
    for tag, v in by.items():
        print(f"  {tag:>16}: n={len(v):2d}  sim2real KS = "
              f"{np.mean(v):.3f} ± {np.std(v):.3f}")
    print(f"\n  mean sim2real domain gap KS = {np.mean(r.ks):.3f} ± "
          f"{np.std(r.ks):.3f}  (over {len(r.ks)} measured specimens)")
    print(f"  → the apples-to-apples measured sim-to-real gap (gap #1, "
          f"same quantity)")
    mx = max(s["max_sigma"] for s in r.stage0)
    print(f"  Stage-0 |z| on the real fields: hot-spots up to {mx:.0f}σ "
          f"(q={flag_q:.2f} flag)")
    print(bar)


def make_figure_dspss(r: DSPSSResult, out_path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import sys
    sys.path.insert(0, os.path.join(HERE, "slides", "figure_sources"))
    try:
        from thesis_style import use
        figsize = use(width_frac=1.0, aspect=0.5)
    except Exception:
        figsize = (12.0, 6.0)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 3, figsize=figsize)
    # two example measured fields (one convex, one concave)
    pick = [0, len(r.specimens) - 1]
    for col, idx in enumerate(pick):
        name, fld = r.specimens[idx]
        disp = name.replace("#", " no.")            # '#' breaks LaTeX titles
        vmax = np.percentile(fld[fld > 0], 99)
        im = ax[0, col].imshow(fld, cmap="viridis", vmin=0, vmax=vmax,
                               aspect="auto")
        ax[0, col].set_title(f"{disp} measured DSPSS", fontsize=8)
        ax[0, col].set_xticks([]); ax[0, col].set_yticks([])
        fig.colorbar(im, ax=ax[0, col], fraction=0.046, pad=0.02)
        im2 = ax[1, col].imshow(r.stage0[idx]["score"], cmap="inferno",
                                vmin=0, vmax=6, aspect="auto")
        ax[1, col].set_title(f"Stage-0 |z| (max {r.stage0[idx]['max_sigma']:.0f}σ)",
                             fontsize=8)
        ax[1, col].set_xticks([]); ax[1, col].set_yticks([])
        fig.colorbar(im2, ax=ax[1, col], fraction=0.046, pad=0.02)

    # standardised distributions: FEM pool vs all measured
    zf = (r.fem_pool - r.fem_pool.mean()) / (r.fem_pool.std() + 1e-12)
    allm = np.concatenate([f[f > 0].ravel() for _, f in r.specimens])
    zm = (allm - allm.mean()) / (allm.std() + 1e-12)
    ax[0, 2].hist(zf, bins=60, density=True, alpha=0.5, color="#1565C0",
                  label="FEM DSPSS")
    ax[0, 2].hist(np.clip(zm, -3, 6), bins=60, density=True, alpha=0.5,
                  color="#d7301f", label="measured DSPSS")
    ax[0, 2].set_title(f"sim2real gap  KS={np.mean(r.ks):.2f}", fontsize=8)
    ax[0, 2].set_xlabel("standardised DSPSS", fontsize=7)
    ax[0, 2].legend(fontsize=6); ax[0, 2].tick_params(labelsize=6)

    # per-specimen KS, grouped by defect type
    names = [n.split("#")[0] for n, _ in r.specimens]
    colors = ["#2e7d32" if "totsu" in n else "#8e24aa" for n in names]
    ax[1, 2].bar(range(len(r.ks)), r.ks, color=colors)
    ax[1, 2].axhline(np.mean(r.ks), color="k", lw=0.8, ls="--")
    ax[1, 2].set_title("per-specimen sim2real KS", fontsize=8)
    ax[1, 2].set_xlabel("specimen (green=convex, purple=concave)", fontsize=6)
    ax[1, 2].set_ylim(0, 1); ax[1, 2].tick_params(labelsize=6)

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

    import tempfile
    # synthetic TSA export: Format 4 3, then 3 rows of 4 values (comma decimals)
    txt = ("General information :\n\nFile\tx.ptw\nFrame\t1\nTotal frames\t1\n"
           "Format\t4\t3\n"
           + "\n".join("\t".join(f"{v:.2f}".replace(".", ",")
                                 for v in row)
                       for row in [[1, 2, 3, 4] * 25,    # ≥100 toks → kept
                                   [2, 3, 4, 5] * 25,
                                   [3, 4, 5, 6] * 25]) + "\n")
    with tempfile.NamedTemporaryFile("w", suffix="No99_curve.txt",
                                     delete=False) as fh:
        fh.write(txt); p = fh.name
    f = load_tsa_field(p)
    os.unlink(p)
    ok(f.field.shape == (3, 100), "TSA parsed to (rows, cols)")
    ok(f.shape_hdr == (4, 3), "Format header W,H parsed")
    ok(abs(f.field[0, 0] - 1.0) < 1e-9, "comma-decimal parsed")

    g = sim2real_gap(np.array([[1.0, 2, 3], [4, 5, 6.0]]),
                     np.random.default_rng(0).normal(50, 10, 500))
    ok(0.0 <= g["ks_standardised"] <= 1.0, "KS in range")
    ok("median" in g["measured"], "robust stats present")

    s = stage0_on_field(np.array([[1.0, 1, 1], [1, 9, 1]]), flag_q=0.8)
    ok(s["score"].shape == (2, 3), "stage0 score shape matches field")
    ok(np.unravel_index(np.argmax(s["score"]), (2, 3)) == (1, 1),
       "hot-spot at the spike pixel")
    ok(s["max_sigma"] > 1.0, "spike is a clear outlier")

    rs = robust_stats(np.array([0.0, 1, 2, 3, 100]))
    ok(rs["median"] == 2.0, "median robust to the 100 outlier")

    # DSPSS path on a synthetic specimen set (no disk dependency).
    # NB ks_statistic standardises, so it measures the SHAPE gap: a measured
    # field with a different distribution shape (exponential) vs a Gaussian FEM
    # pool yields KS>0, while a pure mean/scale shift would not.
    rng = np.random.default_rng(0)
    specs = [(f"totsu(convex)#{i:02d}",
              rng.exponential(8, (12, 10)) + 1.0) for i in range(3)]
    fem = rng.normal(40, 12, 5000)
    ks = [kr.ks_statistic(f[f > 0], fem) for _, f in specs]
    ok(all(0 <= k <= 1 for k in ks), "DSPSS KS in range")
    ok(np.mean(ks) > 0.15, "shape-different measured vs FEM → sim2real gap")
    dr = DSPSSResult(specimens=specs, ks=ks,
                     stage0=[stage0_on_field(f) for _, f in specs], fem_pool=fem)
    ok(len(dr.stage0) == 3 and dr.stage0[0]["score"].shape == (12, 10),
       "DSPSSResult stage0 shapes")

    print(f"tsa_sim2real: {n}/{n} unit tests passed")
    return n


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Sim-to-real: measured TSA full-field stress vs FEM DSPSS.")
    ap.add_argument("--source", choices=["dspss", "tsa"], default="dspss",
                    help="dspss = measured DSPSS vs FEM DSPSS (22 specimens, "
                         "apples-to-apples); tsa = raw TSA stress (3 specimens)")
    ap.add_argument("--flag-q", type=float, default=0.99,
                    help="Stage-0 hot-spot flag quantile on the real field")
    ap.add_argument("--fig", action="store_true",
                    help="save the figure under paper_figs/")
    ap.add_argument("--test", action="store_true", help="unit tests only")
    args = ap.parse_args()

    if args.test:
        run_tests(); return
    if args.source == "dspss":
        r = run_dspss(flag_q=args.flag_q)
        if args.fig:
            p = make_figure_dspss(r, os.path.join(HERE, "paper_figs",
                                                  "tsa_sim2real.pdf"))
            print(f"\nfigure → {p}")
    else:
        r = run(flag_q=args.flag_q)
        if args.fig:
            p = make_figure(r, os.path.join(HERE, "paper_figs",
                                            "tsa_sim2real_raw.pdf"))
            print(f"\nfigure → {p}")


if __name__ == "__main__":
    main()
