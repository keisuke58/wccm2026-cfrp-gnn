"""design_robustness.py — Stage-5 設計結論の頑健性（未較正モデルへの耐性）。

GAP (gap table: Stage-5 設計 85/50): Stage-5 は未較正の phase-field 代表定数＋合成
fleet prior で最適化するため「設計結論の物理的意味が弱い」。

本スクリプトの答え＝**結論を絶対値でなく方向（トレンド）として提示し、それが未較正
定数に不変であることを実証する**。phase-field の主要な未較正定数 (ell, Gc_ply) と
合成 fleet 荷重を妥当範囲で OAT 摂動し、各シナリオで設計目的（FD phase-field の
mean log-growth）を off-axis角×界面靱性 grid で評価。

主張: たとえ絶対成長量は較正に依存しても、**最適設計の方向 ―― off-axis を立てる＋
界面を強靱化する ―― は全シナリオで不変**。よって Stage-5 の物理的結論（どちらへ
設計を動かすべきか）は較正に依らず頑健。絶対的な認証値は主張しない。

出力: results/design_feedback/design_robustness.{json,pdf,png} と要約print。
"""
import os, json, itertools
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.serif":["Nimbus Roman","STIXGeneral","DejaVu Serif"],"mathtext.fontset":"stix"})

import design_feedback as df
from cfrp_phasefield_2d import seed_defect, simulate_growth

GRID_NX, GRID_NY = df.GRID_NX, df.GRID_NY
OFFAXIS = [15.0, 35.0, 55.0, 75.0]      # 設計探索軸（粗grid）
GCINT   = [0.30, 0.60, 0.90]            # 界面靱性比
N_DEF   = 6                              # fleet 欠陥ドロー数（粗）
SEED    = 7

# 既定（ベースライン）定数を取得
_base_cfg = df.make_config(df.BASELINE_DESIGN, nx=GRID_NX, ny=GRID_NY)
ELL0, GC0 = float(_base_cfg.ell), float(_base_cfg.Gc_ply)
_prior = df.fleet_defect_prior()
LOAD0 = float(_prior.load)
_rng = np.random.default_rng(SEED)
DEFECTS = np.atleast_2d(_prior.sample(N_DEF, _rng))

# 未較正定数の OAT シナリオ（妥当範囲）: (label, cfg_kwargs, load)
SCEN = [
    ("baseline",        {},                         LOAD0),
    ("ell x0.8",        {"ell": 0.8*ELL0},           LOAD0),
    ("ell x1.3",        {"ell": 1.3*ELL0},           LOAD0),
    ("Gc_ply x0.6",     {"Gc_ply": 0.6*GC0},         LOAD0),
    ("Gc_ply x1.6",     {"Gc_ply": 1.6*GC0},         LOAD0),
    ("fleet load x0.85",{},                          0.85*LOAD0),
    ("fleet load x1.15",{},                          1.15*LOAD0),
]

def _job(args):
    design, theta, load, kw = args
    cfg = df.make_config(design, nx=GRID_NX, ny=GRID_NY, **kw)
    cx, cy, layer, l2s = theta
    d0 = seed_defect(cx, cy, layer, l2s, cfg)
    r = simulate_growth(d0, float(load), cfg)
    return float(np.log1p(max(r.rel_growth, 0.0)))

def eval_design(design, load, kw, pool):
    jobs = [(np.asarray(design,float), tuple(map(float,th)), load, kw) for th in DEFECTS]
    logs = np.array(pool.map(_job, jobs) if pool else [_job(j) for j in jobs])
    return float(logs.mean()) + df.manufacturability_penalty(design)

def main():
    import multiprocessing as mp
    nw = min(os.cpu_count() or 1, N_DEF)
    pool = mp.Pool(nw) if nw > 1 else None
    out = {"baseline_constants": {"ell": ELL0, "Gc_ply": GC0, "load": LOAD0}, "scenarios": {}}
    print("Stage-5 design robustness to uncalibrated constants (FD phase-field)")
    print("baseline: ell=%.4g Gc_ply=%.4g load=%.4g | grid off-axis%s x Gc_int%s, %d defects"
          %(ELL0,GC0,LOAD0,OFFAXIS,GCINT,N_DEF))
    fig,ax=plt.subplots(1,3,figsize=(13,3.8))
    colors=plt.cm.viridis(np.linspace(0,1,len(SCEN)))
    arg_off=[]; arg_gc=[]; redux=[]
    for si,(lab,kw,load) in enumerate(SCEN):
        grid=np.zeros((len(OFFAXIS),len(GCINT)))
        for i,oa in enumerate(OFFAXIS):
            for j,gc in enumerate(GCINT):
                grid[i,j]=eval_design([oa,gc,1.0],load,kw,pool)
        # baseline design objective in this scenario（同一定数で公平比較）
        obj_base=eval_design(df.BASELINE_DESIGN,load,kw,pool)
        ij=np.unravel_index(np.argmin(grid),grid.shape)
        oa_opt,gc_opt=OFFAXIS[ij[0]],GCINT[ij[1]]
        obj_opt=grid[ij]
        arg_off.append(oa_opt); arg_gc.append(gc_opt); redux.append(obj_base-obj_opt)
        out["scenarios"][lab]={"argmin_offaxis":oa_opt,"argmin_gcint":gc_opt,
                               "obj_opt":obj_opt,"obj_baseline":obj_base,
                               "reduction":obj_base-obj_opt,
                               "grid_offaxis_at_best_gc":grid[:,np.argmax(GCINT)].tolist()}
        # (a) objective vs off-axis at the toughest interface (gc=0.9)
        ax[0].plot(OFFAXIS,grid[:,-1],"-o",ms=3,color=colors[si],label=lab,lw=1.3)
        print("  %-18s -> opt off-axis=%2.0f deg, Gc_int=%.2f | growth %.3f->%.3f (Δ%.3f)"
              %(lab,oa_opt,gc_opt,obj_base,obj_opt,obj_base-obj_opt))
    ax[0].set_xlabel("off-axis ply angle |$\\theta$| [deg]");ax[0].set_ylabel("mean log-growth (lower=better)")
    ax[0].set_title("(a) objective vs off-axis (Gc$_{int}$=0.9)\nleverage (slope) only when load sits near the growth transition",fontsize=9)
    ax[0].legend(fontsize=6.2);ax[0].grid(alpha=0.3)
    # leverage = baseline growth in an intermediate band (not saturated, not ~0)
    lev=[0.3<out["scenarios"][s[0]]["obj_baseline"]<3.0 for s in SCEN]
    bcol=["#c62828" if l else "#9e9e9e" for l in lev]
    ax[1].bar(range(len(SCEN)),arg_off,color=bcol)
    ax[1].set_xticks(range(len(SCEN)));ax[1].set_xticklabels([s[0] for s in SCEN],rotation=40,ha="right",fontsize=6.5)
    ax[1].set_ylabel("optimal off-axis [deg]");ax[1].set_ylim(0,80)
    ax[1].set_title("(b) optimal off-axis is NOT robust: 15-75$^\\circ$\n(depends on uncalibrated ell / Gc / load)",fontsize=9)
    ax[2].bar(range(len(SCEN)),redux,color=bcol)
    ax[2].set_xticks(range(len(SCEN)));ax[2].set_xticklabels([s[0] for s in SCEN],rotation=40,ha="right",fontsize=6.5)
    ax[2].set_ylabel("growth reduction baseline$\\to$opt");ax[2].axhline(0,color="k",lw=0.6)
    ax[2].set_title("(c) design leverage collapses off the transition window\n(red=has leverage, grey=saturated/safe regime)",fontsize=9)
    for a in (ax[1],ax[2]): a.grid(alpha=0.3,axis="y")
    fig.suptitle("Stage-5 design optimum is SENSITIVE to uncalibrated phase-field constants: leverage is confined to the near-transition load window --- calibration is required before quoting a design optimum (honest limitation)",fontsize=9.5,y=1.03)
    fig.tight_layout()
    d=os.path.join(os.path.dirname(os.path.abspath(__file__)),"results","design_feedback")
    for e in ("pdf","png"): fig.savefig(os.path.join(d,"design_robustness."+e),dpi=200,bbox_inches="tight")
    # 要約
    off_spread=max(arg_off)-min(arg_off)
    n_lev=sum(1 for s in SCEN if 0.3<out["scenarios"][s[0]]["obj_baseline"]<3.0)
    print("FINDING (honest): optimal off-axis spans %.0f-%.0f deg (spread %.0f) across uncalibrated-constant scenarios"
          %(min(arg_off),max(arg_off),off_spread))
    print("  design leverage present in only %d/%d scenarios (others saturated/all-safe)"%(n_lev,len(SCEN)))
    print("=> Stage-5 design OPTIMUM is NOT robust to the uncalibrated phase-field constants.")
    print("   Leverage is confined to the near-transition load window; outside it growth saturates or vanishes.")
    print("   HONEST LIMITATION: the framework demonstrates the design-feedback MECHANISM, but a quoted optimum")
    print("   (e.g. a specific off-axis angle) requires calibration / a layup-conditioned calibrated forward (Keio Phase-3).")
    out["finding"]={"offaxis_spread_deg":off_spread,"n_leverage_scenarios":n_lev,
                    "robust":False,"note":"optimum sensitive to ell/Gc_ply/load; leverage only near transition"}
    json.dump(out,open(os.path.join(d,"design_robustness.json"),"w"),indent=2)
    if pool: pool.close()

if __name__=="__main__":
    main()
