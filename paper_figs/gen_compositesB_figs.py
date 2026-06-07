import os,sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE,"..","slides","figure_sources"))
try:
    import thesis_style as ts; ts.use()
    matplotlib.rcParams.update({"font.size":11,"axes.titlesize":12,"axes.labelsize":11})
    NAVY="#142b54"
except Exception:
    plt.rcParams.update({"font.family":"serif","font.size":11}); NAVY="#142b54"
OUT=HERE

# data
arch=["hybridmgn","meshgraphnet","bistridemgn","gatv2","gat","sage","mgntransformer"]
indist=[0.792,0.761,0.741,0.705,0.685,0.668,0.552]
ood   =[0.333,0.332,0.332,0.337,0.333,0.332,0.333]
lab=["HybridMGN","MeshGraphNet","BiStrideMGN","GATv2","GAT","SAGE","MGN-T"]

# Fig A: in-dist vs OOD per architecture
fig,ax=plt.subplots(figsize=(6.3,3.4))
x=np.arange(len(arch)); w=0.38
ax.bar(x-w/2,indist,w,label="in-distribution (sizes 2/4/8)",color=NAVY)
ax.bar(x+w/2,ood,w,label="OOD (unseen 1$\\times$1)",color="#c0712c")
ax.axhline(np.mean(ood),ls="--",lw=0.8,color="#c0712c",alpha=0.7)
ax.set_xticks(x); ax.set_xticklabels(lab,rotation=30,ha="right")
ax.set_ylabel("macro-F1 (support-only)"); ax.set_ylim(0,0.85)
ax.set_title("Architecture helps in-distribution, not size-OOD")
ax.legend(frameon=False,fontsize=9,loc="upper right")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_arch_indist_vs_ood.png",dpi=300); plt.close(fig)

# Fig B: detection vs localization (panel), 3 archs, in-dist vs OOD
A3=["HybridMGN","MeshGraphNet","GAT"]
met=["defF1","detRec","exact","AUPRC"]
IN ={"HybridMGN":[0.781,0.990,0.852,0.988],"MeshGraphNet":[0.748,0.969,0.839,0.970],"GAT":[0.667,0.972,0.701,0.986]}
OO ={"HybridMGN":[0.001,0.727,0.001,0.040],"MeshGraphNet":[0.000,0.643,0.000,0.018],"GAT":[0.001,0.958,0.001,0.036]}
fig,ax=plt.subplots(1,2,figsize=(8.4,3.4),sharey=True)
cols=["#1b5e20","#142b54","#c0712c","#7b1fa2"]
for k,(title,D) in enumerate([("in-distribution",IN),("OOD (1$\\times$1)",OO)]):
    x=np.arange(len(A3)); w=0.2
    for j,mm in enumerate(met):
        ax[k].bar(x+(j-1.5)*w,[D[a][j] for a in A3],w,label=mm,color=cols[j])
    ax[k].set_xticks(x); ax[k].set_xticklabels(A3,rotation=20,ha="right")
    ax[k].set_title(title); ax[k].set_ylim(0,1.05)
    ax[k].spines["top"].set_visible(False); ax[k].spines["right"].set_visible(False)
ax[0].set_ylabel("score"); ax[0].legend(frameon=False,fontsize=8,ncol=2,loc="upper center")
fig.suptitle("OOD: detection (detRec) survives, fine localization (defF1/exact) collapses",fontsize=11)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_detection_vs_localization.png",dpi=300); plt.close(fig)
print("figs ->",OUT)
