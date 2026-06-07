"""Noise-intensity OOD: evaluate one MGN on clean-test vs noise-test (a 2nd OOD axis,
orthogonal to size). Reports the multi-metric panel. Run on vancouver:
  python ood_eval_noise.py <checkpoint.pth>
"""
import os, sys
import numpy as np, torch
REPO = "/home/nishioka/GNN/wccm2026-cfrp-gnn"
sys.path.insert(0, REPO); os.chdir(REPO)
from torch_geometric.loader import DataLoader
from train import MeshGraphNetModel, prepare_data, create_data_label_pairs

dev = "cuda" if torch.cuda.is_available() else "cpu"
GD = "/home/nishioka/GNN"
xq = np.load(f"{GD}/GNN_hole/GNN_hole_data/normalized_x_2layer.npy")
yq = np.load(f"{GD}/GNN_hole/GNN_hole_data/normalized_y_2layer.npy")
zq = np.load(f"{GD}/GNN_hole/GNN_hole_data/normalized_z_2layer.npy")
edges = np.load(f"{GD}/GNN_hole/GNN_hole_data/hole_edges_2layer_best.npy")
edge_index = torch.tensor(edges.T, dtype=torch.long)

ck = torch.load(sys.argv[1], map_location="cpu")
sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
sd = {k.replace("module.", ""): v for k, v in sd.items()}
model = MeshGraphNetModel(hidden_channels=16, num_classes=19, in_channels=4, num_blocks=10).to(dev)
model.load_state_dict(sd); model.eval()
print(f"loaded {os.path.basename(sys.argv[1])} on {dev}")


def build(dfolder, lfolder, cap=None):
    dfiles = sorted(f for f in os.listdir(dfolder)
                    if f.startswith("Defect_") and f.endswith(".npy") and "_19label" not in f)
    lfiles = sorted(f for f in os.listdir(lfolder) if f.endswith("_19label.npy"))
    if cap:
        dfiles = dfiles[:cap]
    pairs = create_data_label_pairs(dfiles, lfiles)
    return prepare_data(pairs, dfolder, lfolder, xq, yq, zq, edge_index)[0]


def panel(Y, P):
    # macro-F1 over present classes + defect-only (class>0) detection metrics
    cls = [c for c in range(19) if (Y == c).any()]
    f1s = []
    for c in cls:
        tp = int(((P == c) & (Y == c)).sum()); fp = int(((P == c) & (Y != c)).sum()); fn = int(((P != c) & (Y == c)).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    macro = float(np.mean(f1s))
    dY, dP = (Y > 0), (P > 0)              # defect vs defect-free detection
    tp = int((dY & dP).sum()); fp = int((~dY & dP).sum()); fn = int((dY & ~dP).sum()); tn = int((~dY & ~dP).sum())
    detRec = tp / (tp + fn) if tp + fn else 0.0
    detFPR = fp / (fp + tn) if fp + tn else 0.0
    dpr = tp / (tp + fp) if tp + fp else 0.0
    defF1 = 2 * dpr * detRec / (dpr + detRec) if dpr + detRec else 0.0
    exact = float((Y == P).mean())
    return macro, defF1, detRec, detFPR, exact


@torch.no_grad()
def evalset(name, dfolder, lfolder, cap=None):
    ds = build(dfolder, lfolder, cap)
    P, Y = [], []
    for b in DataLoader(ds, batch_size=16, shuffle=False):
        b = b.to(dev); out = model(b)
        P.append(out.argmax(1).cpu().numpy()); Y.append(b.y.cpu().numpy())
    P, Y = np.concatenate(P), np.concatenate(Y)
    m, df, dr, fp, ex = panel(Y, P)
    print(f"[{name}] N={len(ds)} macroF1={m:.4f} defectF1={df:.4f} detRec={dr:.4f} detFPR={fp:.4f} exact={ex:.4f}", flush=True)


CL = f"{GD}/GNN_hole_2026/all_sub_hole_defect_zscore"
NZ = f"{GD}/GNN_hole_2026/all_sub_hole_defect_zscore_noise"
LB = f"{GD}/GNN_hole_2026/all_19class_label"
evalset("clean test", f"{CL}/test", LB, cap=1418)
evalset("noise test", f"{NZ}/test", LB, cap=1418)
