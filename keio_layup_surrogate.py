"""keio_layup_surrogate.py — D: layup-CONDITIONED prognosis surrogate (Keio Phase-3 seed).

The shipped Stage-3 FNO surrogate (crack_surrogate.py) is layup-BLIND: its inputs
are (d0, load) only and every training sample shares the baseline layup, so any
design change is out-of-distribution and Stage-5 must fall back to the slow exact
FD phase-field. This prototype closes that gap: it generates a small
(off-axis angle, interface toughness, load) -> growth dataset with the exact FD
forward and fits a surrogate that TAKES THE LAYUP AS INPUT, so design search can
be amortised. This is the concrete seed of the Keio continuation's calibrated,
layup-conditioned forward (and the forward a TMCMC/Bayesian-inverse phase needs).

Honest scope: coarse grid, representative phase-field constants (the absolute
fatigue rate is still uncalibrated, cf. composite_fatigue_calibration). The
contribution is the layup-conditioned INTERFACE + a working amortised map, not a
certified life model.

Output: results/design_feedback/keio_layup_surrogate.{png,pdf,json}
"""
import os, json, itertools
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.serif":["Nimbus Roman","STIXGeneral","DejaVu Serif"],"mathtext.fontset":"stix"})
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, KFold

import design_feedback as df
from cfrp_phasefield_2d import seed_defect, simulate_growth

OFFAXIS=[15.,30.,45.,60.,75.]; GCINT=[0.3,0.6,0.9]; LOAD=[0.08,0.10,0.12]
DEFECTS=[(0.0,0.0,2,3.0),(0.0,0.0,3,3.5)]   # (cx,cy,layer,log2size) representative

def label_growth(offaxis,gcint,load,theta):
    cfg=df.make_config([offaxis,gcint,1.0],nx=df.GRID_NX,ny=df.GRID_NY)
    cx,cy,layer,l2s=theta
    d0=seed_defect(cx,cy,layer,l2s,cfg)
    r=simulate_growth(d0,float(load),cfg)
    return float(np.log1p(max(r.rel_growth,0.0)))

def main():
    rows=[]
    grid=list(itertools.product(OFFAXIS,GCINT,LOAD))
    print("D: layup-conditioned surrogate — generating FD dataset (%d configs x %d defects)"
          %(len(grid),len(DEFECTS)))
    for (oa,gc,ld) in grid:
        for th in DEFECTS:
            rows.append([oa,gc,ld,th[2],th[3],label_growth(oa,gc,ld,th)])
    A=np.array(rows); X=A[:,:5]; y=A[:,5]   # X = [offaxis, gcint, load, layer, log2size]
    reg=GradientBoostingRegressor(random_state=0,max_depth=3,n_estimators=300,learning_rate=0.05)
    pred=cross_val_predict(reg,X,y,cv=KFold(5,shuffle=True,random_state=0))
    r2=1-np.sum((y-pred)**2)/np.sum((y-y.mean())**2)
    # layup sensitivity captured? correlation of prediction with off-axis at fixed load/defect
    reg.fit(X,y)
    print("  layup-conditioned surrogate: 5-fold CV R2=%.3f over %d samples"%(r2,len(y)))
    # figure: (a) parity, (b) growth vs off-axis (FD vs surrogate) at gcint=0.6, load=0.10
    fig,ax=plt.subplots(1,2,figsize=(9,3.6))
    ax[0].scatter(y,pred,s=12,alpha=0.5,color="#1565c0")
    lim=[min(y.min(),pred.min())-0.1,max(y.max(),pred.max())+0.1]
    ax[0].plot(lim,lim,"k--",lw=0.8); ax[0].set_xlim(lim); ax[0].set_ylim(lim)
    ax[0].set_title("(a) layup-conditioned surrogate\nCV $R^2$=%.2f"%r2,fontsize=9)
    ax[0].set_xlabel("FD log-growth"); ax[0].set_ylabel("surrogate"); ax[0].grid(alpha=0.3)
    th=DEFECTS[0]
    fd=[label_growth(oa,0.6,0.10,th) for oa in OFFAXIS]
    sg=[reg.predict([[oa,0.6,0.10,th[2],th[3]]])[0] for oa in OFFAXIS]
    ax[1].plot(OFFAXIS,fd,"-o",color="k",label="exact FD")
    ax[1].plot(OFFAXIS,sg,"--s",color="#0f8b8d",label="surrogate (layup-aware)")
    ax[1].set_xlabel("off-axis ply angle [deg]"); ax[1].set_ylabel("log-growth")
    ax[1].set_title("(b) captures the design dependence\n(the shipped FNO is layup-blind)",fontsize=9)
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.suptitle("Keio Phase-3 seed: a layup-CONDITIONED prognosis surrogate (design search amortised)",fontsize=10,y=1.02)
    fig.tight_layout()
    d=os.path.join(os.path.dirname(os.path.abspath(__file__)),"results","design_feedback")
    os.makedirs(d,exist_ok=True)
    for e in ("png","pdf"): fig.savefig(os.path.join(d,"keio_layup_surrogate."+e),dpi=200,bbox_inches="tight")
    json.dump({"cv_r2":float(r2),"n":len(y),"offaxis":OFFAXIS,"gcint":GCINT,"load":LOAD},
              open(os.path.join(d,"keio_layup_surrogate.json"),"w"),indent=2)
    print("  => surrogate now consumes the layup -> Stage-5 design search amortisable;")
    print("     the seed of the calibrated layup-conditioned forward (Keio Phase-3 / TMCMC inverse).")
    print("  saved results/design_feedback/keio_layup_surrogate.{png,pdf,json}")

if __name__=="__main__": main()
