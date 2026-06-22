"""
e2e_stage0_gnn.py — End-to-end Stage-0 → Stage-1 (GNN) evaluation.

Two-stage SHM pipeline:
  Stage 0 : per-node Mahalanobis z-score, specimen-level detection
             (parameter-free — μ/σ from healthy reference, no training)
  Stage 1 : HybridMGN 19-class node localisation (pre-trained checkpoint)

Evaluation design
-----------------
Defective test set : all_sub_hole_defect_zscore_noise/test  (4×4 / 2×2 / 8×8)
Healthy proxy set  : oned_1x1_subtracted_zscore             (1×1, treated as healthy)

  Step 1  Stage-0 detection on every specimen in both sets.
          → specimen-level verdict: detected (defective) | clean (healthy)
          → metrics: recall on defective set, FPR on healthy set

  Step 2  GNN inference ONLY on Stage-0-detected specimens from defective set.
          → 19-class node-level localisation
          → metrics: conditional macro-F1 (given detection), defect-only F1, exact acc

  Step 3  Combined / system-level metrics
          → system recall = Stage-0 recall × (GNN defect-class acc on detected)
          → FN rate = fraction of defective specimens Stage-0 misses entirely

Usage (on Vancouver, from repo root)
-------------------------------------
    python e2e_stage0_gnn.py                      # HybridMGN
    python e2e_stage0_gnn.py --arch gat           # old GAT baseline
    python e2e_stage0_gnn.py --n-def 500          # cap specimens for speed
    python e2e_stage0_gnn.py --test               # unit tests only

All paths are set for the Vancouver GNN server layout.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader

# ── repo path (auto-detect: Vancouver layout or local git clone) ─────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "/home/nishioka/GNN/wccm2026-cfrp-gnn"
if not os.path.isdir(REPO):
    REPO = _HERE   # local clone (e.g. /home/nishioka/git/wccm2026-cfrp-gnn)
sys.path.insert(0, REPO)
os.chdir(REPO)

import train as T  # noqa: E402

GD = "/home/nishioka/GNN"
CK = f"{GD}/GNN_hole_2026/GNN_model/19classmodel_hole_zscore"

# Defective test specimens (subtracted z-score, same as GNN training input)
DIR_TEST = f"{GD}/GNN_hole_2026/all_sub_hole_defect_zscore_noise/test"
DIR_LBL = f"{GD}/GNN_hole_2026/all_19class_label"

# Healthy proxy reference (1×1 specimens, subtracted z-score)
DIR_REF = f"{GD}/GNN_hole_2026/oned_1x1_subtracted_zscore"
DIR_REF_LBL = f"{GD}/GNN_hole_2026/oned_1x1_19class_label"

# Mesh coordinates and edges
COORD = f"{GD}/GNN_hole/GNN_hole_data"
N_NODES = 13942
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# Stage-0 detection threshold quantile (matches run_pipeline.py)
THR_Q = 0.999

# Checkpoint filenames (from ood_eval_arch.py)
CKPTS = {
    "hybridmgn": "HybridMeshGraphNetModel_20260607_041607_Best.pth",
    "meshgraphnet": "MeshGraphNetModel_20260606_181418_Best.pth",
    "gat": "GATModel_20260606_174244_Best_Epoch117_ValLossBest0.0517_MacroF1Best0.6740_MacroF1Cur0.6740_MacroModesupport_only.pth",
}


# ═════════════════════════════════════════════════════════════════════════════
#  Data helpers
# ═════════════════════════════════════════════════════════════════════════════

def _load_field(fpath: str) -> np.ndarray:
    """Load subtracted z-score DSPSS field (N_NODES,)."""
    return np.load(fpath).astype(np.float32)[:N_NODES]


def _load_label(lbl_dir: str, fpath: str) -> np.ndarray | None:
    """Load 19-class node label for a specimen."""
    base = os.path.splitext(os.path.basename(fpath))[0]
    for cand in (f"{base}_19label.npy", f"{base}.npy"):
        lf = os.path.join(lbl_dir, cand)
        if os.path.exists(lf):
            return np.load(lf)
    return None


@dataclass
class Specimen:
    fpath: str
    name: str
    field: np.ndarray   # (N,)
    label: np.ndarray   # (N,) int, 0=healthy, 1–18=defect class
    is_defective: bool  # ground-truth specimen label


def _collect(data_dir: str, lbl_dir: str, cap: int, is_defective: bool) -> list[Specimen]:
    files = sorted(
        f for f in glob.glob(os.path.join(data_dir, "*.npy"))
        if "_19label" not in f
    )[:cap]
    out = []
    for fp in files:
        lbl = _load_label(lbl_dir, fp)
        if lbl is None:
            continue
        out.append(Specimen(
            fpath=fp,
            name=os.path.basename(fp),
            field=_load_field(fp),
            label=lbl[:N_NODES].astype(np.int64),
            is_defective=is_defective,
        ))
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  Stage 0 — Mahalanobis z-score detection
# ═════════════════════════════════════════════════════════════════════════════

def build_reference_manifold(ref_specs: list[Specimen]) -> tuple[np.ndarray, np.ndarray, float]:
    """μ, σ from healthy-proxy reference fields; also compute threshold."""
    X = np.stack([s.field for s in ref_specs])   # (N_ref, N_nodes)
    mu = X.mean(0)
    sd = X.std(0) + 1e-6
    ref_scores = np.abs(X - mu) / sd            # (N_ref, N_nodes)
    thr = float(np.quantile(ref_scores, THR_Q))
    return mu, sd, thr


def stage0_score(spec: Specimen, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return np.abs(spec.field - mu) / sd         # (N_nodes,)


def stage0_detect(spec: Specimen, mu: np.ndarray, sd: np.ndarray, thr: float) -> dict:
    score = stage0_score(spec, mu, sd)
    detected = bool(score.max() > thr)
    defect_mask = (spec.label > 0)
    auroc = None
    if defect_mask.any() and not defect_mask.all():
        auroc = float(roc_auc_score(defect_mask.astype(int), score))
    return {"detected": detected, "score": score, "auroc": auroc,
            "max_score": float(score.max())}


# ═════════════════════════════════════════════════════════════════════════════
#  Stage 1 — GNN 19-class node classification
# ═════════════════════════════════════════════════════════════════════════════

def _load_mesh():
    xq = np.load(f"{COORD}/normalized_x_2layer.npy")[:N_NODES]
    yq = np.load(f"{COORD}/normalized_y_2layer.npy")[:N_NODES]
    zq = np.load(f"{COORD}/normalized_z_2layer.npy")[:N_NODES]
    edges = np.load(f"{COORD}/hole_edges_2layer_best.npy")
    edge_index = torch.tensor(edges.T, dtype=torch.long)
    return xq, yq, zq, edge_index


def _build_graph(spec: Specimen, xq, yq, zq, edge_index) -> "torch_geometric.data.Data":
    from torch_geometric.data import Data
    x_feat = np.stack([spec.field, xq, yq, zq], axis=1).astype(np.float32)
    return Data(
        x=torch.tensor(x_feat),
        edge_index=edge_index,
        y=torch.tensor(spec.label, dtype=torch.long),
    )


def _load_model(arch: str) -> torch.nn.Module:
    fn = CKPTS.get(arch)
    if fn is None:
        raise ValueError(f"Unknown arch '{arch}'. Choose from: {list(CKPTS)}")
    p = os.path.join(CK, fn)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    a = arch.lower()
    M = {
        "hybridmgn": T.HybridMeshGraphNetModel,
        "meshgraphnet": T.MeshGraphNetModel,
    }
    if a in M:
        model = M[a](hidden_channels=16, num_classes=19, in_channels=4, num_blocks=10)
    else:
        model = T.GATModel(hidden_channels=16, num_classes=19, in_channels=4, conv_type=a)
    ck = torch.load(p, map_location="cpu")
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    return model.to(DEV)


@torch.no_grad()
def stage1_infer(model: torch.nn.Module, specs: list[Specimen],
                 xq, yq, zq, edge_index) -> list[np.ndarray]:
    """Return per-node predicted class for each specimen."""
    from torch_geometric.loader import DataLoader as GeoLoader
    from torch_geometric.data import Data
    graphs = [_build_graph(s, xq, yq, zq, edge_index) for s in specs]
    preds = []
    for b in GeoLoader(graphs, batch_size=16, shuffle=False):
        b = b.to(DEV)
        out = model(b).argmax(1).cpu().numpy()
        # split batch back into per-specimen arrays
        sizes = [N_NODES] * (b.num_graphs)
        for chunk in np.split(out, np.cumsum(sizes)[:-1]):
            preds.append(chunk)
    return preds


# ═════════════════════════════════════════════════════════════════════════════
#  Metrics
# ═════════════════════════════════════════════════════════════════════════════

def specimen_level_metrics(detections: list[bool], gt_defective: list[bool]) -> dict:
    tp = sum(d and g for d, g in zip(detections, gt_defective))
    fn = sum(not d and g for d, g in zip(detections, gt_defective))
    fp = sum(d and not g for d, g in zip(detections, gt_defective))
    tn = sum(not d and not g for d, g in zip(detections, gt_defective))
    n_pos = tp + fn
    n_neg = fp + tn
    recall = tp / n_pos if n_pos else float("nan")
    fpr = fp / n_neg if n_neg else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    return {"recall": recall, "fpr": fpr, "precision": precision,
            "tp": tp, "fn": fn, "fp": fp, "tn": tn}


def node_level_metrics(all_pred: np.ndarray, all_true: np.ndarray) -> dict:
    metrics = T.macro_f1_variants(all_true, all_pred, 19)
    defect_mask = all_true > 0
    det_rec = float((all_pred[defect_mask] > 0).mean()) if defect_mask.any() else float("nan")
    exact = float((all_pred[defect_mask] == all_true[defect_mask]).mean()) if defect_mask.any() else float("nan")
    fp_rate = float((all_pred[~defect_mask] > 0).mean()) if (~defect_mask).any() else float("nan")
    return {
        "macroF1": metrics["macro_f1_support_only"],
        "defF1": metrics.get("macro_f1_defect_only", float("nan")),
        "detRec": det_rec,
        "detFPR": fp_rate,
        "exact": exact,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  Main evaluation
# ═════════════════════════════════════════════════════════════════════════════

def run(arch: str = "hybridmgn", n_def: int = 1418, n_ref: int = 800,
        seed: int = 42, verbose: bool = True):
    rng = np.random.default_rng(seed)

    print(f"[e2e] arch={arch}  n_def={n_def}  n_ref={n_ref}", flush=True)

    # collect specimens
    print("[e2e] loading defective test set ...", flush=True)
    def_specs = _collect(DIR_TEST, DIR_LBL, cap=n_def, is_defective=True)
    print(f"      {len(def_specs)} defective specimens", flush=True)

    print("[e2e] loading healthy proxy (reference) set ...", flush=True)
    ref_all = _collect(DIR_REF, DIR_REF_LBL, cap=99999, is_defective=False)
    idx = rng.choice(len(ref_all), min(n_ref, len(ref_all)), replace=False)
    ref_specs = [ref_all[i] for i in idx]
    print(f"      {len(ref_specs)} reference specimens (for manifold)", flush=True)

    # remaining 1×1 used as healthy evaluation set (held-out from manifold)
    held_idx = [i for i in range(len(ref_all)) if i not in set(idx)][:n_def]
    healthy_eval = [ref_all[i] for i in held_idx]
    print(f"      {len(healthy_eval)} held-out healthy-proxy evaluation specimens", flush=True)

    # ── Stage 0 ──────────────────────────────────────────────────────────────
    print("\n[Stage 0] building reference manifold ...", flush=True)
    mu, sd, thr = build_reference_manifold(ref_specs)
    print(f"          threshold (q={THR_Q:.3f}): {thr:.4f}", flush=True)

    def_results = [stage0_detect(s, mu, sd, thr) for s in def_specs]
    hlt_results = [stage0_detect(s, mu, sd, thr) for s in healthy_eval]

    def_detected = [r["detected"] for r in def_results]
    hlt_detected = [r["detected"] for r in hlt_results]

    sm = specimen_level_metrics(
        def_detected + hlt_detected,
        [True] * len(def_specs) + [False] * len(healthy_eval),
    )
    aurocs = [r["auroc"] for r in def_results if r["auroc"] is not None]
    mean_auroc = float(np.mean(aurocs)) if aurocs else float("nan")

    print(f"\n  Stage-0 specimen-level:")
    print(f"    recall (defect detected)  : {sm['recall']:.4f}  ({sm['tp']}/{sm['tp']+sm['fn']})")
    print(f"    FPR   (healthy flagged)   : {sm['fpr']:.4f}  ({sm['fp']}/{sm['fp']+sm['tn']})")
    print(f"    mean node-level AUROC     : {mean_auroc:.4f}")

    # ── Stage 1 — GNN on Stage-0-detected specimens only ────────────────────
    detected_specs = [s for s, d in zip(def_specs, def_detected) if d]
    missed_specs = [s for s, d in zip(def_specs, def_detected) if not d]
    print(f"\n[Stage 1] running GNN on {len(detected_specs)} Stage-0-detected specimens ...", flush=True)

    if not detected_specs:
        print("  No specimens passed Stage 0 — cannot evaluate Stage 1.")
        return {}

    xq, yq, zq, edge_index = _load_mesh()
    model = _load_model(arch)
    model.eval()

    preds = stage1_infer(model, detected_specs, xq, yq, zq, edge_index)
    all_pred = np.concatenate(preds)
    all_true = np.concatenate([s.label for s in detected_specs])

    nm = node_level_metrics(all_pred, all_true)
    print(f"\n  Stage-1 node-level (on Stage-0-detected specimens only):")
    print(f"    macro-F1 (support-only)   : {nm['macroF1']:.4f}")
    print(f"    defect-class F1           : {nm['defF1']:.4f}")
    print(f"    defect-node recall        : {nm['detRec']:.4f}")
    print(f"    defect-node FPR           : {nm['detFPR']:.4f}")
    print(f"    exact 19-class acc (defect): {nm['exact']:.4f}")

    # ── Combined / system-level ──────────────────────────────────────────────
    # System recall = specimen detected by S0 AND correctly localized by S1.
    # "Correct localization" = ≥1 defect node predicted in correct class.
    correct_loc = []
    for spec, pred in zip(detected_specs, preds):
        defect_nodes = spec.label > 0
        if not defect_nodes.any():
            correct_loc.append(False)
            continue
        correct_loc.append(bool((pred[defect_nodes] == spec.label[defect_nodes]).any()))

    system_recall = float(np.mean(correct_loc)) * sm["recall"]
    fn_rate = 1.0 - sm["recall"]   # fraction of defects Stage-0 misses entirely

    print(f"\n  Combined / system-level:")
    print(f"    Stage-0 recall            : {sm['recall']:.4f}")
    print(f"    Conditional localization  : {float(np.mean(correct_loc)):.4f}")
    print(f"    System recall (S0×S1)     : {system_recall:.4f}")
    print(f"    System FN rate (S0 miss)  : {fn_rate:.4f}")
    print(f"    Missed specimens (S0)     : {len(missed_specs)}/{len(def_specs)}")

    # Reference: GNN alone (no Stage-0 gate)
    print(f"\n[Reference] GNN-alone macro-F1 on same test set ...", flush=True)
    preds_all = stage1_infer(model, def_specs, xq, yq, zq, edge_index)
    nm_all = node_level_metrics(np.concatenate(preds_all),
                                np.concatenate([s.label for s in def_specs]))
    print(f"    macro-F1 (GNN alone)      : {nm_all['macroF1']:.4f}")
    print(f"    macro-F1 (GNN|S0-gated)   : {nm['macroF1']:.4f}")
    delta = nm['macroF1'] - nm_all['macroF1']
    print(f"    Δ macro-F1 (gated - alone) : {delta:+.4f}")

    return {
        "arch": arch,
        "stage0": {**sm, "mean_auroc": mean_auroc, "thr": thr},
        "stage1_conditional": nm,
        "stage1_alone": nm_all,
        "system_recall": system_recall,
        "fn_rate": fn_rate,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  Unit tests
# ═════════════════════════════════════════════════════════════════════════════

def _tests():
    print("running unit tests ...", flush=True)
    rng = np.random.default_rng(0)

    # test build_reference_manifold
    N = 5000
    ref = [type("S", (), {"field": rng.standard_normal(N).astype(np.float32)})()
           for _ in range(50)]
    mu, sd, thr = build_reference_manifold(ref)
    assert mu.shape == (N,)
    assert sd.shape == (N,)
    assert thr > 0
    print("  [PASS] build_reference_manifold")

    # test stage0_detect — defective specimen (large spike)
    spec_def = type("S", (), {
        "field": rng.standard_normal(N).astype(np.float32),
        "label": np.zeros(N, dtype=np.int64),
    })()
    spec_def.field[100] += 50.0  # large anomaly
    spec_def.label[100] = 3
    r = stage0_detect(spec_def, mu, sd, thr)
    assert r["detected"], "defective specimen not detected"
    assert r["auroc"] is not None
    print("  [PASS] stage0_detect (defective)")

    # test stage0_detect — healthy specimen (no spike)
    spec_hlt = type("S", (), {
        "field": rng.standard_normal(N).astype(np.float32) * 0.3,
        "label": np.zeros(N, dtype=np.int64),
    })()
    r2 = stage0_detect(spec_hlt, mu, sd, thr)
    assert not r2["detected"], "healthy specimen falsely detected"
    print("  [PASS] stage0_detect (healthy)")

    # test node_level_metrics
    y_true = np.array([0, 0, 1, 2, 3, 3], dtype=np.int64)
    y_pred = np.array([0, 0, 1, 2, 4, 3], dtype=np.int64)
    nm = node_level_metrics(y_pred, y_true)
    assert 0 < nm["macroF1"] < 1
    assert 0 <= nm["exact"] <= 1
    print("  [PASS] node_level_metrics")

    # test specimen_level_metrics
    sm = specimen_level_metrics([True, True, False, False], [True, False, True, False])
    assert sm["tp"] == 1 and sm["fn"] == 1 and sm["fp"] == 1 and sm["tn"] == 1
    assert abs(sm["recall"] - 0.5) < 1e-9
    assert abs(sm["fpr"] - 0.5) < 1e-9
    print("  [PASS] specimen_level_metrics")

    print(f"all tests PASSED (4/4)", flush=True)


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="hybridmgn",
                    choices=list(CKPTS), help="GNN architecture")
    ap.add_argument("--n-def", type=int, default=1418,
                    help="cap on defective test specimens")
    ap.add_argument("--n-ref", type=int, default=800,
                    help="reference manifold size for Stage 0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test", action="store_true", help="run unit tests only")
    args = ap.parse_args()

    if args.test:
        _tests()
    else:
        run(arch=args.arch, n_def=args.n_def, n_ref=args.n_ref, seed=args.seed)
