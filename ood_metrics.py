import os,sys,glob,numpy as np,torch
REPO="/home/nishioka/GNN/wccm2026-cfrp-gnn";sys.path.insert(0,REPO);os.chdir(REPO)
from torch_geometric.loader import DataLoader
import train as T
from sklearn.metrics import average_precision_score
GD="/home/nishioka/GNN";dev="cuda" if torch.cuda.is_available() else "cpu"
xq=np.load(f"{GD}/GNN_hole/GNN_hole_data/normalized_x_2layer.npy");yq=np.load(f"{GD}/GNN_hole/GNN_hole_data/normalized_y_2layer.npy");zq=np.load(f"{GD}/GNN_hole/GNN_hole_data/normalized_z_2layer.npy")
edge_index=torch.tensor(np.load(f"{GD}/GNN_hole/GNN_hole_data/hole_edges_2layer_best.npy").T,dtype=torch.long)
CK=f"{GD}/GNN_hole_2026/GNN_model/19classmodel_hole_zscore"
def build(df,lf,cap):
    d=sorted(f for f in os.listdir(df) if f.startswith("Defect_") and f.endswith(".npy") and "_19label" not in f)[:cap]
    l=sorted(f for f in os.listdir(lf) if f.endswith("_19label.npy"))
    return T.prepare_data(T.create_data_label_pairs(d,l),df,lf,xq,yq,zq,edge_index)[0]
print("building...",flush=True)
indist=build(f"{GD}/GNN_hole_2026/all_sub_hole_defect_zscore_noise/test",f"{GD}/GNN_hole_2026/all_19class_label",1418)
ood=build(f"{GD}/GNN_hole_2026/oned_1x1_subtracted_zscore",f"{GD}/GNN_hole_2026/oned_1x1_19class_label",1500)
def mk(a):
    a=a.lower();M={"meshgraphnet":T.MeshGraphNetModel,"hybridmgn":T.HybridMeshGraphNetModel,"bistridemgn":T.BiStrideMeshGraphNetModel,"mgntransformer":T.MGNTransformerModel}
    return M[a](hidden_channels=16,num_classes=19,in_channels=4,num_blocks=10) if a in M else T.GATModel(hidden_channels=16,num_classes=19,in_channels=4,conv_type=a)
def grp(a): return np.where(a==0,0,np.where(a<=9,1,2))
def metrics(yt,yp,pdef):
    o={}
    o['macroF1']=T.macro_f1_variants(yt,yp,19)['macro_f1_support_only']
    cs=[c for c in range(1,19) if (yt==c).any()];f1=[]
    for c in cs:
        tp=((yp==c)&(yt==c)).sum();fp=((yp==c)&(yt!=c)).sum();fn=((yp!=c)&(yt==c)).sum()
        p=tp/(tp+fp) if tp+fp else 0;r=tp/(tp+fn) if tp+fn else 0;f1.append(2*p*r/(p+r) if p+r else 0)
    o['defF1']=float(np.mean(f1)) if f1 else 0
    dt=(yt>0).astype(int);dp=(yp>0).astype(int)
    tp=((dp==1)&(dt==1)).sum();fp=((dp==1)&(dt==0)).sum();fn=((dp==0)&(dt==1)).sum();tn=((dp==0)&(dt==0)).sum()
    o['detRec']=tp/(tp+fn) if tp+fn else 0;o['detFPR']=fp/(fp+tn) if fp+tn else 0
    o['detAUPRC']=average_precision_score(dt,pdef) if dt.any() and (dt==0).any() else float('nan')
    dm=dt==1
    o['grpAcc']=(grp(yp)[dm]==grp(yt)[dm]).mean() if dm.any() else 0
    o['exact']=(yp[dm]==yt[dm]).mean() if dm.any() else 0
    return o
@torch.no_grad()
def ev(m,ds):
    P,Y,PD=[],[],[]
    for b in DataLoader(ds,batch_size=16,shuffle=False):
        b=b.to(dev);o=m(b);sm=torch.softmax(o,1)
        P.append(o.argmax(1).cpu().numpy());Y.append(b.y.cpu().numpy());PD.append((1-sm[:,0]).cpu().numpy())
    return metrics(np.concatenate(Y),np.concatenate(P),np.concatenate(PD))
MODELS=[("hybridmgn","HybridMeshGraphNetModel_20260607_041607_Best.pth"),("meshgraphnet","MeshGraphNetModel_20260606_181418_Best.pth"),("gat","GATModel_20260606_174244_Best_Epoch117_ValLossBest0.0517_MacroF1Best0.6740_MacroF1Cur0.6740_MacroModesupport_only.pth")]
hdr=f"{'arch':13s}{'set':8s}{'macroF1':>8s}{'defF1':>7s}{'detRec':>7s}{'detFPR':>7s}{'AUPRC':>7s}{'grpAcc':>7s}{'exact':>7s}"
print(hdr,flush=True)
for arch,fn in MODELS:
    p=os.path.join(CK,fn)
    if not os.path.exists(p): print(arch,"missing");continue
    ck=torch.load(p,map_location="cpu");sd=ck.get("model_state_dict",ck) if isinstance(ck,dict) else ck
    m=mk(arch).to(dev);m.load_state_dict({k.replace("module.",""):v for k,v in sd.items()});m.eval()
    for nm,ds in [("in-dist",indist),("OOD-1x1",ood)]:
        r=ev(m,ds)
        print(f"{arch:13s}{nm:8s}{r['macroF1']:8.3f}{r['defF1']:7.3f}{r['detRec']:7.3f}{r['detFPR']:7.3f}{r['detAUPRC']:7.3f}{r['grpAcc']:7.3f}{r['exact']:7.3f}",flush=True)
