import numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family']='serif'; rcParams['font.size']=15
rcParams['axes.titlesize']=16; rcParams['axes.labelsize']=13
os.chdir("/home/nishioka/GNN/GNN_hole_2026")
OUT="/tmp/fig_out"; os.makedirs(OUT, exist_ok=True)

num_cols,num_rows,vpl,total=57,125,6971,57*125
hcs,hrs1,hrs2=26,51,77; hsc,hsr1,hsr2=7,7,15
hole=[]
for r in range(hrs1,hrs1+hsr1):
    for c in range(hcs,hcs+hsc): hole.append(r*num_cols+c)
for r in range(hrs2,hrs2+hsr2):
    for c in range(hcs,hcs+hsc): hole.append(r*num_cols+c)
hole=set(n-1 for n in hole)

def grid(data, layer=1):
    seg=data[:vpl] if layer==1 else data[vpl:]
    full=np.full(total,np.nan); k=0
    for i in range(total):
        if i not in hole and k<vpl: full[i]=seg[k]; k+=1
    g=np.flipud(np.rot90(full.reshape((num_rows,num_cols)),k=1))
    return np.ma.masked_invalid(g)

def panel(ax,g,title,cmap,vmin=None,vmax=None,cblabel=""):
    im=ax.imshow(g,cmap=cmap,aspect="equal",interpolation="none",vmin=vmin,vmax=vmax)
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    cb=plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04); cb.set_label(cblabel,fontsize=11)
    cb.ax.tick_params(labelsize=9)
    return im

spec="Defect_L10_B100_el2515_H4_W4"
diff=np.load(f"all_sub_hole_defect/{spec}.npy")
ref=np.load("real_hole_no_defect_original.npy")
orig=ref+diff
noised=np.load(f"all_sub_hole_defect_zscore_noise/train/{spec}.npy")
zs=(diff-diff.mean())/diff.std()
rng=np.random.default_rng(0)
ndf=rng.normal(0.0,0.1,size=diff.shape)   # defect-free base (0) + 10% noise, z-scale
lab=np.load(f"all_19class_label/{spec}_19label.npy").argmax(1).astype(float)

L=1  # L10 -> bottom layer
# --- Fig 1: difference construction (raw stress jet; difference coolwarm) ---
rv=np.nanpercentile(np.r_[ref,orig],[2,98]); 
fig,ax=plt.subplots(1,3,figsize=(15,5))
panel(ax[0],grid(ref,L),"(a) No-defect reference","jet",rv[0],rv[1],"DSPSS")
panel(ax[1],grid(orig,L),"(b) With defect","jet",rv[0],rv[1],"DSPSS")
dv=np.nanpercentile(np.abs(diff),98)
panel(ax[2],grid(diff,L),"(c) Difference (b)$-$(a)","coolwarm",-dv,dv,r"$\Delta$DSPSS")
plt.tight_layout(); plt.savefig(f"{OUT}/diff_clean.png",dpi=200,bbox_inches="tight"); plt.close()

# --- Fig 2: noise comparison (z-score vs z-score+noise) ---
zv=np.nanpercentile(np.abs(np.r_[zs,noised]),98)
fig,ax=plt.subplots(1,2,figsize=(11,5))
panel(ax[0],grid(zs,L),"Z-score input (clean)","coolwarm",-zv,zv,"Z-score")
panel(ax[1],grid(noised,L),"Z-score input + noise","coolwarm",-zv,zv,"Z-score")
plt.tight_layout(); plt.savefig(f"{OUT}/noise_clean.png",dpi=200,bbox_inches="tight"); plt.close()

# --- Fig 3: NDF illustration ---
fig,ax=plt.subplots(figsize=(6,5.2))
panel(ax,grid(ndf,L),"Defect-free sample + 10% noise","coolwarm",-zv,zv,"Z-score")
plt.tight_layout(); plt.savefig(f"{OUT}/ndf_clean.png",dpi=200,bbox_inches="tight"); plt.close()

# --- Fig 4: ground-truth 19-class label (target) ---
from matplotlib.colors import ListedColormap
base=plt.cm.tab20(np.linspace(0,1,20))[:19]; base[0]=[0.92,0.92,0.92,1]  # class0 gray
cmap=ListedColormap(base)
fig,ax=plt.subplots(figsize=(6,5.2))
im=ax.imshow(grid(lab,L),cmap=cmap,aspect="equal",interpolation="none",vmin=0,vmax=18)
ax.set_title("Ground-truth defect class (target)"); ax.set_xticks([]); ax.set_yticks([])
cb=plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04,ticks=[0,6,12,18]); cb.set_label("class id",fontsize=11)
plt.tight_layout(); plt.savefig(f"{OUT}/label_clean.png",dpi=200,bbox_inches="tight"); plt.close()
print("done:", sorted(os.listdir(OUT)))
