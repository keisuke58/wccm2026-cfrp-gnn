"""denso_rescue3d_parts.py -- pictures of the 3-D parts for the RESCUE slides (Greif & Meyer, Zenodo 17831586).
  rescue3d_gallery.png   six of the parts, coloured by the data's warp (Moldflow, rigid motion removed);
  rescue3d_example.png   two parts (one that the correction fixes, one that it does not): the data's warp, the
                         prior (physics model, before correction) and the prior corrected with 10 surface nodes
                         (marked), same colour scale per part, deformation magnified.
The prior and the correction are recomputed here with validation/greif3d_rescue.py (CalculiX C3D10, K = 32,
the same code as fifa §19); the printed errors are compared with fifa's per_geometry.json (check P1).
The VTU files are NOT in the repository (CC BY 4.0, 2 GB): extract the few needed, e.g. with remotezip, into DATA.
Run:  python3 slides/figure_sources/denso_rescue3d_parts.py DATA [WORK]
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "validation"))
import greif3d_rescue as g  # noqa: E402

DATA = sys.argv[1]
WORK = sys.argv[2] if len(sys.argv) > 2 else os.path.join(DATA, "_work")
os.makedirs(WORK, exist_ok=True)
OUTD = os.path.join(ROOT, "slides", "figures_paper")
ACC = "#1F3A5F"
matplotlib.rcParams.update({"font.family": ["Times New Roman", "Liberation Serif", "IPAPMincho"], "font.size": 10})
FIFA = {r["id"]: r for r in json.load(open(os.path.join(ROOT, "abaqus", "denso_plate", "abaqus_results",
                                                            "greif3d_rescue", "per_geometry.json")))}
CMAP = plt.cm.cividis


def surface(T):
    F = np.concatenate([T[:, [0, 1, 2]], T[:, [0, 1, 3]], T[:, [0, 2, 3]], T[:, [1, 2, 3]]])
    Fs = np.sort(F, 1)
    _, idx, c = np.unique(Fs, axis=0, return_index=True, return_counts=True)
    return F[idx[c == 1]]


def draw(ax, X, F, u, scale, vmax, nodes=None, title=""):
    P = X + scale * u
    val = np.linalg.norm(u, axis=1)[F].mean(1)
    pc = Poly3DCollection(P[F], facecolors=CMAP(np.clip(val / vmax, 0, 1)), edgecolors="none", linewidths=0)
    ax.add_collection3d(pc)
    lo, hi = P.min(0), P.max(0); c = (lo + hi) / 2; r = (hi - lo).max() / 2
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1), zoom=1.45); ax.view_init(28, -55); ax.set_axis_off()
    if nodes is not None:
        ax.scatter(*P[nodes].T, s=28, c="#C0392B", depthshade=False, edgecolors="white", linewidths=0.6, zorder=10)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=0)


def geometry(gid):
    return g.load_geometry(os.path.join(DATA, gid))


def prior(gid):
    cache = os.path.join(WORK, f"{gid}_prior.npz")
    if os.path.exists(cache):
        z = np.load(cache); return [z["Ui"], z["Ud"]]
    vtu, X, T, data, An, E1 = geometry(gid)
    T, _ = g.fix_orientation(X, T)
    A = g.clean_orientation(An[T].mean(1))
    lab, cen = g.kmeans(g.vec6(A), 32)
    Cs, als = g.cluster_props(g.clean_orientation(g.mat6(cen)))
    Xs, Ts = g.to_c3d10(X, T)
    U = g.solve(Xs, Ts, lab, Cs, als, WORK, gid, threads=4)
    Ub = [U[0][: len(X)], U[1][: len(X)]]
    np.savez(cache, Ui=Ub[0], Ud=Ub[1]); return Ub


# ---------------------------------------------------------------- gallery
gal = ["00010768", "00010941", "00010642", "00010017", "00010010", "00010520"]
gal = [x for x in gal if os.path.isdir(os.path.join(DATA, x))]
fig = plt.figure(figsize=(12.0, 2.5))
for i, gid in enumerate(gal):
    _, X, T, data, _, _ = geometry(gid)
    u = g.remove_rigid(X, data); F = surface(T)
    ax = fig.add_axes([i / len(gal), 0.08, 1 / len(gal), 0.84], projection="3d")
    dg = np.linalg.norm(X.max(0) - X.min(0))
    draw(ax, X, F, u, 0.0, np.linalg.norm(u, axis=1).max(), title=f"{dg:.0f} mm、反り最大 {np.linalg.norm(u, axis=1).max():.2f} mm")
fig.text(0.5, 0.01, "公開データの部品の例（Greif & Meyer, Zenodo 17831586）。色はデータ（Moldflow 解析）の反りの大きさ（各部品で最大を明るく）。"
         "寸法は外接箱の対角。", ha="center", fontsize=8, color="0.35")
fig.savefig(os.path.join(OUTD, "rescue3d_gallery.png"), dpi=220); plt.close(fig)

# ---------------------------------------------------------------- example
ex = [("00010768", "補正がよく効く部品"), ("00010077", "補正が効きにくい部品")]
fig = plt.figure(figsize=(12.0, 5.4))
rng = np.random.default_rng(3)
for row, (gid, lab) in enumerate(ex):
    _, X, T, data, _, _ = geometry(gid)
    Ub = prior(gid)
    F = surface(T); S = g.surface_nodes(T)
    idx = rng.choice(S, size=10, replace=False)
    a, corr = g.fit(X, Ub, data, idx, two=True)
    pr = Ub[0] + Ub[1]
    e0, e1 = g.rel_err(X, pr, data), g.rel_err(X, corr, data)
    f = FIFA.get(gid, {})
    print(f"[P1] {gid}: prior error {e0:.3f} (fifa {f.get('prior', float('nan')):.3f}), 10-node 2-factor this draw {e1:.3f} "
          f"(fifa median of 10 draws {f.get('rescue_10_2f', float('nan')):.3f}), factors {np.round(a, 2).tolist()}")
    ud, up, uc = g.remove_rigid(X, data), g.remove_rigid(X, pr), g.remove_rigid(X, corr)
    vmax = np.linalg.norm(ud, axis=1).max()
    scale = 0.08 * np.linalg.norm(X.max(0) - X.min(0)) / vmax
    for col, (u, t, nodes) in enumerate(((ud, "データの反り（Moldflow）", None),
                                         (up, f"補正前の予測：誤差 {e0 * 100:.0f} %", None),
                                         (uc, f"10 点（赤）で補正：誤差 {e1 * 100:.0f} %", idx))):
        ax = fig.add_axes([0.13 + col * 0.29, 0.52 - row * 0.45, 0.28, 0.43], projection="3d")
        draw(ax, X, F, u, scale, vmax, nodes, t)
    fig.text(0.075, 0.735 - row * 0.45, f"{lab}\n（{gid}）", ha="center", va="center", fontsize=10.5, fontweight="bold",
             color=ACC if row == 0 else "k")
cax = fig.add_axes([0.40, 0.085, 0.25, 0.018])
cb = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP), cax=cax, orientation="horizontal")
cb.set_ticks([0, 1]); cb.set_ticklabels(["0", "データの最大"]); cb.ax.tick_params(labelsize=8)
fig.text(0.5, 0.005, "色は反りの大きさ（剛体運動を除く、各部品でデータの最大を 1）。変形は拡大して表示。"
         "補正前の予測は形が合い大きさが約半分、補正で大きさが直る。下の部品は形そのものが合わず、倍率では直らない。",
         ha="center", fontsize=8, color="0.35")
fig.savefig(os.path.join(OUTD, "rescue3d_example.png"), dpi=220); plt.close(fig)
print("wrote rescue3d_gallery.png, rescue3d_example.png")
