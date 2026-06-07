import os, sys, glob
import numpy as np
import torch
REPO="/home/nishioka/GNN/wccm2026-cfrp-gnn"
sys.path.insert(0, REPO); os.chdir(REPO)
from torch_geometric.loader import DataLoader
from train import MeshGraphNetModel, macro_f1_variants, prepare_data, create_data_label_pairs

dev = "cuda" if torch.cuda.is_available() else "cpu"
GD="/home/nishioka/GNN"
xq=np.load(f"{GD}/GNN_hole/GNN_hole_data/normalized_x_2layer.npy")
yq=np.load(f"{GD}/GNN_hole/GNN_hole_data/normalized_y_2layer.npy")
zq=np.load(f"{GD}/GNN_hole/GNN_hole_data/normalized_z_2layer.npy")
edges=np.load(f"{GD}/GNN_hole/GNN_hole_data/hole_edges_2layer_best.npy")
edge_index=torch.tensor(edges.T, dtype=torch.long)

ck_path=sys.argv[1]
ck=torch.load(ck_path, map_location="cpu")
sd=ck.get("model_state_dict", ck) if isinstance(ck,dict) else ck
sd={k.replace("module.",""):v for k,v in sd.items()}
model=MeshGraphNetModel(hidden_channels=16, num_classes=19, in_channels=4, num_blocks=10).to(dev)
model.load_state_dict(sd); model.eval()
print(f"loaded {os.path.basename(ck_path)} on {dev}")

def build(dfolder, lfolder, cap=None):
    dfiles=sorted(f for f in os.listdir(dfolder) if f.startswith("Defect_") and f.endswith(".npy") and "_19label" not in f)
    lfiles=sorted(f for f in os.listdir(lfolder) if f.endswith("_19label.npy"))
    if cap: dfiles=dfiles[:cap]
    pairs=create_data_label_pairs(dfiles, lfiles)
    return prepare_data(pairs, dfolder, lfolder, xq, yq, zq, edge_index)[0]

@torch.no_grad()
def evalset(name, dfolder, lfolder, cap=None):
    ds=build(dfolder, lfolder, cap)
    ld=DataLoader(ds, batch_size=16, shuffle=False)
    P,Y=[],[]
    for b in ld:
        b=b.to(dev); out=model(b)
        P.append(out.argmax(1).cpu().numpy()); Y.append(b.y.cpu().numpy())
    P=np.concatenate(P); Y=np.concatenate(Y)
    v=macro_f1_variants(Y,P,19)
    print(f"[{name}] N={len(ds)} macroF1 support_only={v['macro_f1_support_only']:.4f} sklearn_like={v['macro_f1_sklearn_like']:.4f}")
    return v

# in-distribution test (sizes 2/4/8) — baseline
evalset("in-dist test (2/4/8)", f"{GD}/GNN_hole_2026/all_sub_hole_defect_zscore_noise/test",
        f"{GD}/GNN_hole_2026/all_19class_label", cap=1418)
# OOD 1x1 (unseen smallest size)
evalset("OOD 1x1", f"{GD}/GNN_hole_2026/oned_1x1_subtracted_zscore",
        f"{GD}/GNN_hole_2026/oned_1x1_19class_label", cap=1500)
