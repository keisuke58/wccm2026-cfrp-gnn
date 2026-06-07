import os, sys, glob
import numpy as np, torch
REPO="/home/nishioka/GNN/wccm2026-cfrp-gnn"; sys.path.insert(0,REPO); os.chdir(REPO)
from torch_geometric.loader import DataLoader
import train as T
GD="/home/nishioka/GNN"; dev="cuda" if torch.cuda.is_available() else "cpu"
xq=np.load(f"{GD}/GNN_hole/GNN_hole_data/normalized_x_2layer.npy")
yq=np.load(f"{GD}/GNN_hole/GNN_hole_data/normalized_y_2layer.npy")
zq=np.load(f"{GD}/GNN_hole/GNN_hole_data/normalized_z_2layer.npy")
edges=np.load(f"{GD}/GNN_hole/GNN_hole_data/hole_edges_2layer_best.npy")
edge_index=torch.tensor(edges.T,dtype=torch.long)
CK="/home/nishioka/GNN/GNN_hole_2026/GNN_model/19classmodel_hole_zscore"

def build_ds(dfolder,lfolder,cap):
    df=sorted(f for f in os.listdir(dfolder) if f.startswith("Defect_") and f.endswith(".npy") and "_19label" not in f)[:cap]
    lf=sorted(f for f in os.listdir(lfolder) if f.endswith("_19label.npy"))
    pairs=T.create_data_label_pairs(df,lf)
    return T.prepare_data(pairs,dfolder,lfolder,xq,yq,zq,edge_index)[0]

print("building datasets ...", flush=True)
indist=build_ds(f"{GD}/GNN_hole_2026/all_sub_hole_defect_zscore_noise/test", f"{GD}/GNN_hole_2026/all_19class_label", 1418)
ood=build_ds(f"{GD}/GNN_hole_2026/oned_1x1_subtracted_zscore", f"{GD}/GNN_hole_2026/oned_1x1_19class_label", 1500)

def mk(arch):
    a=arch.lower()
    M={"meshgraphnet":T.MeshGraphNetModel,"hybridmgn":T.HybridMeshGraphNetModel,
       "bistridemgn":T.BiStrideMeshGraphNetModel,"mgntransformer":T.MGNTransformerModel,
       "meshgraphkan":T.MeshGraphKANModel}
    if a in M: return M[a](hidden_channels=16,num_classes=19,in_channels=4,num_blocks=10)
    return T.GATModel(hidden_channels=16,num_classes=19,in_channels=4,conv_type=a)

@torch.no_grad()
def ev(model,ds):
    model.eval(); P,Y=[],[]
    for b in DataLoader(ds,batch_size=16,shuffle=False):
        b=b.to(dev); o=model(b); P.append(o.argmax(1).cpu().numpy()); Y.append(b.y.cpu().numpy())
    P=np.concatenate(P);Y=np.concatenate(Y)
    return T.macro_f1_variants(Y,P,19)["macro_f1_support_only"]

MODELS=[
 ("hybridmgn","HybridMeshGraphNetModel_20260607_041607_Best.pth"),
 ("meshgraphnet","MeshGraphNetModel_20260606_181418_Best.pth"),
 ("bistridemgn","BiStrideMeshGraphNetModel_20260607_041607_Best.pth"),
 ("mgntransformer","MGNTransformerModel_20260607_042630_Best.pth"),
 ("gatv2","GATModel_20260606_181418_Best_Epoch117_ValLossBest0.0480_MacroF1Best0.6938_MacroF1Cur0.6938_MacroModesupport_only.pth"),
 ("gat","GATModel_20260606_174244_Best_Epoch117_ValLossBest0.0517_MacroF1Best0.6740_MacroF1Cur0.6740_MacroModesupport_only.pth"),
 ("sage","GATModel_20260606_181418_Best_Epoch116_ValLossBest0.0518_MacroF1Best0.6633_MacroF1Cur0.6633_MacroModesupport_only.pth"),
]
print(f"{'arch':16s} {'in-dist':>8s} {'OOD-1x1':>8s} {'drop':>7s}", flush=True)
for arch,fn in MODELS:
    p=os.path.join(CK,fn)
    if not os.path.exists(p): print(f"{arch:16s} (ckpt missing)"); continue
    ck=torch.load(p,map_location="cpu"); sd=ck.get("model_state_dict",ck) if isinstance(ck,dict) else ck
    sd={k.replace("module.",""):v for k,v in sd.items()}
    m=mk(arch).to(dev)
    try: m.load_state_dict(sd)
    except Exception as e: print(f"{arch:16s} load-fail {str(e)[:50]}"); continue
    di=ev(m,indist); oo=ev(m,ood)
    print(f"{arch:16s} {di:8.4f} {oo:8.4f} {di-oo:7.4f}", flush=True)
