"""validation/greif3d_twozone_probe.py -- probe of the two-zone orientation rule (research/MEMO_flow_orientation.md) on
Greif & Meyer's Moldflow parts (Zenodo 17831586; Moldflow results, not measurements). INCONCLUSIVE (2026-11-08).
Per node: flow direction from the gradient of Fill_time (P1 gradient per tet, volume-averaged to nodes), flow component
a_ff = n.A.n, distance to the surface h, skin indicator Freeze_time <= max(Fill_time). Prints, per part, the share of
nodes frozen during filling and the rank correlations of a_ff with h and with Freeze_time. (The four plausible orders of
the off-diagonal components were compared separately, by a_nn at the surface and a_ff: all give a_ff 0.19-0.36.)
Result on 10 parts: frozen-during-fill share 0-17 % (fill 0.1-0.7 s); a_ff 0.19-0.36 for every component order, i.e. the
fill-time gradient may not give the flow direction; few elements through the thickness. Neither confirms nor rejects.
Run:  python3 validation/greif3d_twozone_probe.py DATA   (DATA: folders <id>/<...>_sim.vtu, not in the repository)
"""
import meshio,glob,numpy as np,os,sys
from scipy.stats import spearmanr
for d in sorted(glob.glob(sys.argv[1]+'/*')):
  f=sorted(glob.glob(d+'/*_sim.vtu'))[0]; m=meshio.read(f); X=np.asarray(m.points,float)  # coordinates already in mm
  T=np.vstack([c.data for c in m.cells if c.type=='tetra']).astype(int)
  ft=np.asarray(m.point_data['Fill_time'],float).ravel(); fz=np.asarray(m.point_data['Freeze_time'],float).ravel()
  o=np.asarray(m.point_data['Fiber_orientation_tensor_on_nodes_(3D)'],float)
  A=np.zeros((len(o),3,3));A[:,0,0],A[:,1,1],A[:,2,2]=o[:,0],o[:,1],o[:,2];A[:,0,1]=A[:,1,0]=o[:,3];A[:,1,2]=A[:,2,1]=o[:,4];A[:,0,2]=A[:,2,0]=o[:,5]
  J=X[T[:,1:]]-X[T[:,[0]]]; dt=ft[T[:,1:]]-ft[T[:,[0]]]
  vol=np.abs(np.linalg.det(J))/6; g=np.linalg.solve(J,dt[...,None])[...,0]
  G=np.zeros((len(X),3)); W=np.zeros(len(X))
  for k in range(4): np.add.at(G,T[:,k],g*vol[:,None]); np.add.at(W,T[:,k],vol)
  G/=W[:,None]; n=G/np.maximum(np.linalg.norm(G,axis=1,keepdims=True),1e-12)
  aff=np.einsum('ni,nij,nj->n',n,A,n)
  # surface nodes: faces used once
  F=np.sort(np.vstack([T[:,[0,1,2]],T[:,[0,1,3]],T[:,[0,2,3]],T[:,[1,2,3]]]),1)
  u,c=np.unique(F,axis=0,return_counts=True); sn=np.unique(u[c==1])
  from scipy.spatial import cKDTree
  h=cKDTree(X[sn]).query(X)[0]
  tf=ft.max(); sk=fz<=tf
  r=spearmanr(h,aff).correlation; r2=spearmanr(fz,aff).correlation
  hb=np.percentile(h[h>0],[50,90]) if (h>0).any() else [0,0]
  print(os.path.basename(d),'bbox %.0f mm'%np.ptp(X,0).max(),'surface frac %.2f'%(h==0).mean(),'h p50/p90 %.2f/%.2f'%tuple(hb),
        'aff surf %.2f interior %.2f'%(np.median(aff[h==0]),np.median(aff[h>0]) if (h>0).any() else np.nan),
        'rho(h,aff) %+.2f rho(freeze,aff) %+.2f skinfrac %.3f'%(r,r2,sk.mean()))
