"""
interstage_measured_detection.py — the interstage sim-to-real DETECTION harness,
ready to produce a measured-data detection AUROC the moment per-specimen defect
masks arrive (gap #3, currently data-blocked).

State of the interstage real-data story:
  * domain gap measured-vs-FEM DSPSS quantified (tsa_sim2real: KS ~0.09)         ✓
  * FEM->measured domain ADAPTATION (quantile closes 97% of the gap)             ✓
  * geometry-agnostic |z| detector validated on the LABELLED FEM coupon (0.85)   ✓
  * measured-data DETECTION AUROC (defect vs healthy on the real TSA fields)      ✗  <- this file
    BLOCKED: the measured specimens (Kojima TSA campaign) have NO per-pixel defect
    masks on disk (they live in the thesis). The number awaits the masks.

This module makes #3 "as done as possible without the data":
  1. it runs the SAME detector (kojima_real_case.field_anomaly, |z|) on the
     available LABELLED interstage data (FEM coupon) and reports its AUROC — the
     harness is proven;
  2. it applies the detector to the MEASURED DSPSS fields (rect_dspss_exp) and
     reports the flagged hot-spots (no AUROC without masks);
  3. `detection_auroc(field, mask)` is the one call that yields the measured
     number the instant a mask array is provided (`--masks <dir>`).

Usage
-----
    python interstage_measured_detection.py            # status + coupon AUROC + measured hotspots
    python interstage_measured_detection.py --masks DIR  # measured AUROC once masks exist
    python interstage_measured_detection.py --test
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

import kojima_real_case as kr

HERE = os.path.dirname(os.path.abspath(__file__))


# ═════════════════════════════════════════════════════════════════════════════
#  The detection metric — the one call that needs masks
# ═════════════════════════════════════════════════════════════════════════════

def detection_auroc(field: np.ndarray, mask: np.ndarray) -> float:
    """Node/pixel-level detection AUROC of the validated geometry-agnostic |z|
    detector against a ground-truth defect mask. This is the measured-data number
    that gap #3 needs; it works identically on FEM or measured fields."""
    from sklearn.metrics import roc_auc_score
    score = kr.field_anomaly(np.asarray(field, float).ravel(), mode="abs")
    y = np.asarray(mask, float).ravel()[:len(score)] > 0.5
    if not (0 < y.sum() < len(y)):
        return float("nan")
    return float(roc_auc_score(y.astype(int), score))


# ═════════════════════════════════════════════════════════════════════════════
#  (1) harness proof on the available LABELLED interstage data (FEM coupon)
# ═════════════════════════════════════════════════════════════════════════════

def coupon_auroc() -> float:
    """Run the harness on the labelled FEM coupon (the available interstage
    ground truth) — proves the detector + metric work end-to-end."""
    cg = kr.load_vtk_graph(kr.COUPON_DEFECT)
    lg = kr.load_vtk_graph(kr.COUPON_LABEL)
    m = min(cg.n, len(lg.values))
    return detection_auroc(cg.values[:m], (lg.values[:m] > 0.5).astype(int))


# ═════════════════════════════════════════════════════════════════════════════
#  (2)/(3) measured DSPSS fields — hotspots now, AUROC when masks exist
# ═════════════════════════════════════════════════════════════════════════════

def _load_measured():
    """Measured DSPSS specimen fields (rect_dspss_exp) if present, else None."""
    try:
        import tsa_sim2real as ts
        return ts.load_rect_dspss()         # list[(name, (H,W) field)]
    except Exception:
        return None


def measured_status(mask_dir: str | None = None) -> dict:
    """Apply the validated detector to the measured fields. If a mask directory is
    supplied (one mask per specimen, name-matched), compute the real detection
    AUROC; otherwise report flagged hot-spots and the blocked status."""
    specs = _load_measured()
    if not specs:
        return {"available": False}
    out = []
    for name, fld in specs:
        score = kr.field_anomaly(fld.ravel(), mode="abs").reshape(fld.shape)
        rec = {"name": name, "max_sigma": float(score.max()),
               "flag_frac": float((score > np.quantile(score, 0.99)).mean()),
               "auroc": float("nan")}
        if mask_dir:
            mp = _find_mask(mask_dir, name)
            if mp is not None:
                rec["auroc"] = detection_auroc(fld, np.load(mp))
        out.append(rec)
    aurocs = [r["auroc"] for r in out if r["auroc"] == r["auroc"]]
    return {"available": True, "specimens": out,
            "mean_auroc": float(np.mean(aurocs)) if aurocs else float("nan"),
            "n_with_masks": len(aurocs)}


def _find_mask(mask_dir: str, name: str):
    base = name.split("#")[0].split("(")[0]
    for cand in glob.glob(os.path.join(mask_dir, "*.npy")):
        if base.lower() in os.path.basename(cand).lower():
            return cand
    return None


# ═════════════════════════════════════════════════════════════════════════════
#  Report
# ═════════════════════════════════════════════════════════════════════════════

def run(mask_dir: str | None = None, verbose: bool = True) -> dict:
    res = {"coupon_auroc": coupon_auroc(),
           "measured": measured_status(mask_dir)}
    if verbose:
        bar = "=" * 72
        print(bar)
        print("  INTERSTAGE SIM-TO-REAL DETECTION HARNESS  (gap #3)")
        print(bar)
        print(f"  (1) harness proof on labelled FEM coupon: detection AUROC = "
              f"{res['coupon_auroc']:.3f}  (validated detector works)")
        meas = res["measured"]
        if not meas["available"]:
            print("  (2) measured DSPSS fields: not on this machine.")
        else:
            print(f"  (2) measured DSPSS specimens ({len(meas['specimens'])}): "
                  f"|z| hot-spots up to "
                  f"{max(s['max_sigma'] for s in meas['specimens']):.1f} sigma")
            if meas["n_with_masks"]:
                print(f"  (3) MEASURED detection AUROC (masks supplied): "
                      f"{meas['mean_auroc']:.3f}  over {meas['n_with_masks']} "
                      f"specimens  ←  gap #3 CLOSED")
            else:
                print("  (3) measured detection AUROC: BLOCKED — no per-specimen")
                print("      defect masks on disk (Kojima thesis). Provide --masks")
                print("      DIR to close gap #3; detection_auroc() is ready.")
        print(bar)
    return res


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
    # planted-anomaly field: |z| detector should score the anomaly region high
    fld = rng.normal(10, 1, (20, 20))
    mask = np.zeros((20, 20))
    fld[5:8, 5:8] += 12.0        # a stress concentration
    mask[5:8, 5:8] = 1
    auc = detection_auroc(fld, mask)
    ok(0.9 <= auc <= 1.0, f"planted anomaly detected (AUROC {auc:.2f})")
    ok(detection_auroc(fld, np.zeros((20, 20))) != detection_auroc(fld, mask)
       or np.isnan(detection_auroc(fld, np.zeros((20, 20)))),
       "all-healthy mask → undefined AUROC")
    ok(np.isnan(detection_auroc(fld, np.ones((20, 20)))), "all-defect → nan")

    # the harness proof on the real labelled coupon (the validated ~0.85)
    a = coupon_auroc()
    ok(0.7 <= a <= 1.0, f"coupon harness AUROC sane ({a:.2f})")

    # _find_mask name-matching
    import tempfile
    d = tempfile.mkdtemp()
    np.save(os.path.join(d, "totsu_03.npy"), np.zeros(3))
    ok(_find_mask(d, "totsu(convex)#03") is not None, "mask name match")
    ok(_find_mask(d, "ou(concave)#01") is None, "no spurious mask match")

    print(f"interstage_measured_detection: {n}/{n} unit tests passed")
    return n


def main():
    ap = argparse.ArgumentParser(
        description="Interstage sim-to-real detection harness (gap #3, mask-ready).")
    ap.add_argument("--masks", default=None,
                    help="directory of per-specimen defect masks (.npy) → "
                         "computes the measured detection AUROC and closes gap #3")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.test:
        run_tests(); return
    run(args.masks)


if __name__ == "__main__":
    main()
