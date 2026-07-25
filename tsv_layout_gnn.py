"""tsv_layout_gnn.py — theme B main step (research/RESEARCH_THEMES_muramatsu.md):
a graph neural network predicting *per-via* delamination risk from a TSV layout,
learned from finite-element ground truth. It makes the "why GNN" argument
concrete and fair: a via's risk is raised by its NEIGHBOURS (nearby vias / the
die edge concentrate interface stress), so message passing over the via graph —
with pairwise distance as an edge feature — beats a per-via MLP that sees only
the via's own features.

Pipeline:
  1. FE ground truth (cheap, physical): one *linear* thermoelastic solve at unit
     thermal load for a random multi-via Cu-in-Si layout. Per via, the peak
     interface elastic-energy density around it = its delamination risk
     (log10 energy; higher = delaminates sooner). Neighbour vias raise it — an
     interaction the linear solve captures. Consistent with the nonlinear
     phase-field demo tsv_interface_fracture.py ⑭.
  2. Graph: vias = nodes (features x, y, r, distance-to-edge); message passing
     uses edge features (centre distance, surface gap) — the relational signal.
  3. Baseline: same-capacity per-via MLP on OWN features only (no neighbours).
     The R² gap = the value of relational information.

Honest scope: linear-elastic risk proxy (no nonlinear propagation), illustrative
scaled thermal load, small random layouts (2–5 vias), minimal MPNN (repo core
uses GATv2). Concept seed for the theme-B reliability surrogate. Reuses
build_mesh (⑬).

Run:  python3 tsv_layout_gnn.py        (writes tsv_layout_gnn.png)
      python3 tsv_layout_gnn.py --help
"""
from __future__ import annotations

import argparse

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import torch.nn as nn

from phasefield_fracture_warmstart import build_mesh

SEED = 20260725
NU = 0.3
E_CU, E_SI = 1.1, 1.6
A_CU, A_SI = 1.0, 0.153
ELL = 0.03
MAXV = 5


def cmat(Emod):
    lam = Emod * NU / ((1 + NU) * (1 - 2 * NU)); mu = Emod / (2 * (1 + NU))
    return np.array([[lam + 2 * mu, lam, 0], [lam, lam + 2 * mu, 0], [0, 0, mu]])


def fe_geometry(n):
    nodes, tris = build_mesh(n)
    nel = len(tris); N = len(nodes)
    B = np.zeros((nel, 3, 6)); area = np.zeros(nel)
    for e, t in enumerate(tris):
        p = nodes[t]; x, y = p[:, 0], p[:, 1]
        A = 0.5 * ((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))
        b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]]) / (2 * A)
        c = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]]) / (2 * A)
        area[e] = abs(A)
        for k in range(3):
            B[e, 0, 2 * k] = b[k]; B[e, 1, 2 * k + 1] = c[k]
            B[e, 2, 2 * k] = c[k]; B[e, 2, 2 * k + 1] = b[k]
    C1 = cmat(1.0)
    Ke0u = np.einsum("e,eki,kl,elj->eij", area, B, C1, B)
    fthu = np.einsum("e,eki,k->ei", area, B, C1 @ np.array([1.0, 1.0, 0.0]))
    udof = np.zeros((nel, 6), np.int64); udof[:, 0::2] = 2 * tris; udof[:, 1::2] = 2 * tris + 1
    rows = np.repeat(udof, 6, axis=1).ravel(); cols = np.tile(udof, (1, 6)).ravel()
    x, y = nodes[:, 0], nodes[:, 1]
    edge = np.where((x < 1e-9) | (x > 1 - 1e-9) | (y < 1e-9) | (y > 1 - 1e-9))[0]
    fixdof = np.concatenate([2 * edge, 2 * edge + 1]); free = np.setdiff1d(np.arange(2 * N), fixdof)
    cent = nodes[tris].mean(axis=1)
    return dict(N=N, B=B, Ke0u=Ke0u, fthu=fthu, udof=udof, rows=rows, cols=cols,
                free=free, cent=cent, C1=C1)


def risk_per_via(G, vias):
    cent = G["cent"]
    dc = np.stack([np.hypot(cent[:, 0] - cx, cent[:, 1] - cy) for (cx, cy, r) in vias])
    radii = np.array([r for (_, _, r) in vias])[:, None]
    incu = (dc < radii).any(axis=0)
    Emod = np.where(incu, E_CU, E_SI); alpha = np.where(incu, A_CU, A_SI)
    N = G["N"]
    K = sp.csr_matrix(((G["Ke0u"] * Emod[:, None, None]).ravel(), (G["rows"], G["cols"])),
                      shape=(2 * N, 2 * N))
    f = np.bincount(G["udof"].ravel(),
                    weights=((Emod * alpha)[:, None] * G["fthu"]).ravel(), minlength=2 * N)
    u = np.zeros(2 * N)
    u[G["free"]] = spla.spsolve(K[np.ix_(G["free"], G["free"])].tocsc(), f[G["free"]])
    eps = np.einsum("eij,ej->ei", G["B"], u[G["udof"]]) - alpha[:, None] * np.array([1.0, 1.0, 0.0])
    psi = 0.5 * Emod * np.einsum("ei,ij,ej->e", eps, G["C1"], eps)
    risk = []
    for i, (cx, cy, r) in enumerate(vias):
        band = np.abs(dc[i] - r) < 1.5 * ELL
        peak = np.quantile(psi[band], 0.9) if band.any() else psi.max()
        risk.append(float(np.log10(peak + 1e-30)))
    return np.array(risk)


def rand_layout(rng):
    n = rng.integers(2, MAXV + 1); vias = []
    for _ in range(200):
        if len(vias) == n:
            break
        r = rng.uniform(0.07, 0.13); cx, cy = rng.uniform(0.22, 0.78, size=2)
        if all(np.hypot(cx - vx, cy - vy) > (r + vr + 0.015) for vx, vy, vr in vias):
            vias.append((cx, cy, r))
    return vias


def feats(vias):
    v = np.array(vias)
    edge_dist = np.minimum.reduce([v[:, 0], 1 - v[:, 0], v[:, 1], 1 - v[:, 1]]) - v[:, 2]
    X = np.column_stack([v[:, 0], v[:, 1], v[:, 2], edge_dist])   # node features
    P = v[:, :3].copy()                                           # x,y,r for geometry
    return X, P


class MPNN(nn.Module):
    """Padded batched message-passing net; edge features = centre distance & gap."""
    def __init__(self, fin=4, h=32):
        super().__init__()
        self.enc = nn.Linear(fin, h)
        self.m1 = nn.Sequential(nn.Linear(2 * h + 2, h), nn.SiLU(), nn.Linear(h, h))
        self.u1 = nn.Sequential(nn.Linear(2 * h, h), nn.SiLU(), nn.Linear(h, h))
        self.m2 = nn.Sequential(nn.Linear(2 * h + 2, h), nn.SiLU(), nn.Linear(h, h))
        self.u2 = nn.Sequential(nn.Linear(2 * h, h), nn.SiLU(), nn.Linear(h, h))
        self.head = nn.Sequential(nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 1))

    def mp(self, H, E, adj, msg, upd):
        B, n, h = H.shape
        Hi = H[:, :, None, :].expand(B, n, n, h)
        Hj = H[:, None, :, :].expand(B, n, n, h)
        m = msg(torch.cat([Hi, Hj, E], -1)) * adj[..., None]     # (B,n,n,h), masked
        magg = m.sum(2) / (adj.sum(2, keepdim=True) + 1e-9)      # mean over valid neighbours
        return upd(torch.cat([H, magg], -1)) + H                 # residual

    def forward(self, X, P, mask):
        B, n, _ = X.shape
        dx = P[:, :, 0:1] - P[:, None, :, 0]
        dy = P[:, :, 1:2] - P[:, None, :, 1]
        dist = torch.sqrt(dx * dx + dy * dy + 1e-9)
        gap = dist - P[:, :, 2:3] - P[:, None, :, 2]
        E = torch.stack([dist, gap], -1)                          # (B,n,n,2)
        eye = torch.eye(n)[None]
        adj = mask[:, None, :] * (1 - eye)                        # valid neighbours, no self
        H = torch.relu(self.enc(X))
        H = self.mp(H, E, adj, self.m1, self.u1)
        H = self.mp(H, E, adj, self.m2, self.u2)
        return self.head(H).squeeze(-1)                           # (B,n)


def pack(dset, MAXV):
    B = len(dset)
    X = np.zeros((B, MAXV, 4)); P = np.zeros((B, MAXV, 3)); Y = np.zeros((B, MAXV))
    M = np.zeros((B, MAXV))
    for i, (v, y) in enumerate(dset):
        Xi, Pi = feats(v); k = len(v)
        X[i, :k] = Xi; P[i, :k] = Pi; Y[i, :k] = y; M[i, :k] = 1.0
    return X, P, Y, M


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=56, help="FE cells per side")
    ap.add_argument("--ntrain", type=int, default=150)
    ap.add_argument("--ntest", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--out", type=str, default="tsv_layout_gnn.png")
    args = ap.parse_args()

    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    G = fe_geometry(args.n)
    data = []
    while len(data) < args.ntrain + args.ntest:
        v = rand_layout(rng)
        if len(v) >= 2:
            data.append((v, risk_per_via(G, v)))
    tr, te = data[:args.ntrain], data[args.ntrain:]

    Xtr, Ptr, Ytr, Mtr = pack(tr, MAXV); Xte, Pte, Yte, Mte = pack(te, MAXV)
    # standardise from train (masked)
    mtr = Mtr.astype(bool)
    fmu = Xtr[mtr].mean(0); fsd = Xtr[mtr].std(0) + 1e-6
    pmu = Ptr[mtr].mean(0); psd = Ptr[mtr].std(0) + 1e-6
    ymu = Ytr[mtr].mean(); ysd = Ytr[mtr].std()
    def norm(X, P, Y):
        return ((X - fmu) / fsd), ((P - pmu) / psd), ((Y - ymu) / ysd)
    Xtr_n, Ptr_n, Ytr_n = norm(Xtr, Ptr, Ytr); Xte_n, Pte_n, _ = norm(Xte, Pte, Yte)
    tX, tP, tY, tM = (torch.tensor(a, dtype=torch.float32) for a in (Xtr_n, Ptr_n, Ytr_n, Mtr))
    eX, eP, eM = (torch.tensor(a, dtype=torch.float32) for a in (Xte_n, Pte_n, Mte))

    net = MPNN()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=5e-4)
    for ep in range(args.epochs):
        pred = net(tX, tP, tM)
        loss = (((pred - tY) ** 2) * tM).sum() / tM.sum()
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 300 == 0:
            print(f"[gnn] {ep+1}/{args.epochs}  mse {loss.item():.3e}")

    # baseline: per-via MLP on OWN features only
    Xf = torch.tensor(Xtr_n[mtr], dtype=torch.float32); Yf = torch.tensor(Ytr_n[mtr, None], dtype=torch.float32)
    mlp = nn.Sequential(nn.Linear(4, 48), nn.SiLU(), nn.Linear(48, 48), nn.SiLU(),
                        nn.Linear(48, 48), nn.SiLU(), nn.Linear(48, 1))
    opt2 = torch.optim.Adam(mlp.parameters(), lr=3e-3, weight_decay=5e-4)
    for ep in range(1500):
        loss = ((mlp(Xf) - Yf) ** 2).mean(); opt2.zero_grad(); loss.backward(); opt2.step()

    net.eval()
    with torch.no_grad():
        gp = net(eX, eP, eM).numpy()
    mte = Mte.astype(bool)
    gnn_p = gp[mte] * ysd + ymu
    with torch.no_grad():
        mlp_p = mlp(torch.tensor(Xte_n[mte], dtype=torch.float32)).numpy().ravel() * ysd + ymu
    y_true = Yte[mte]

    def r2(p, r):
        return float(1 - np.sum((p - r) ** 2) / np.sum((r - r.mean()) ** 2))

    r2g, r2m = r2(gnn_p, y_true), r2(mlp_p, y_true)
    print(f"\nper-via delamination risk (log energy)  "
          f"({len(tr)} train / {len(te)} test layouts, {mte.sum()} test vias)")
    print(f"  GNN (MPNN, sees neighbours+distance)  test R2 {r2g:.3f}")
    print(f"  MLP (per-via, own features only)      test R2 {r2m:.3f}")
    print(f"  -> relational info gain = {r2g - r2m:+.3f} R2")

    _plot(args.out, te, net, feats, fmu, fsd, pmu, psd, ymu, ysd,
          gnn_p, mlp_p, y_true, r2g, r2m)
    print(f"wrote {args.out}")


def _plot(out, te, net, feats_fn, fmu, fsd, pmu, psd, ymu, ysd, gnn_p, mlp_p, y_true, r2g, r2m):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.4))

    # pick a held-out layout with the most vias for the example
    vex, yex = max(te, key=lambda d: len(d[0]))
    Xe, Pe = feats_fn(vex); k = len(vex)
    Xp = np.zeros((1, MAXV, 4)); Pp = np.zeros((1, MAXV, 3)); Mp = np.zeros((1, MAXV))
    Xp[0, :k] = (Xe - fmu) / fsd; Pp[0, :k] = (Pe - pmu) / psd; Mp[0, :k] = 1
    with torch.no_grad():
        pv = net(torch.tensor(Xp, dtype=torch.float32), torch.tensor(Pp, dtype=torch.float32),
                 torch.tensor(Mp, dtype=torch.float32)).numpy()[0, :k] * ysd + ymu
    th = np.linspace(0, 2 * np.pi, 60)
    vmin, vmax = min(pv.min(), yex.min()), max(pv.max(), yex.max())
    cmap = plt.cm.RdYlBu_r
    for kk, (cx, cy, r) in enumerate(vex):
        ax[0].fill(cx + r * np.cos(th), cy + r * np.sin(th),
                   color=cmap((pv[kk] - vmin) / (vmax - vmin + 1e-9)))
        ax[0].text(cx, cy, f"{pv[kk]:.1f}", ha="center", va="center", fontsize=8)
    ax[0].set_xlim(0, 1); ax[0].set_ylim(0, 1); ax[0].set_aspect("equal")
    ax[0].set_title("held-out layout — GNN per-via risk\n(red = delaminates sooner; numbers = pred)")

    lo, hi = y_true.min(), y_true.max()
    ax[1].plot([lo, hi], [lo, hi], "k--", lw=1)
    ax[1].scatter(y_true, gnn_p, s=16, color="#1f77b4", label=f"GNN (R²={r2g:.2f})")
    ax[1].scatter(y_true, mlp_p, s=16, color="#d62728", marker="x", label=f"per-via MLP (R²={r2m:.2f})")
    ax[1].set_xlabel("FE per-via risk (truth, log energy)"); ax[1].set_ylabel("predicted risk")
    ax[1].set_title("parity (per via): neighbours matter"); ax[1].legend(); ax[1].grid(True, alpha=0.3)

    ax[2].bar([0, 1], [r2g, r2m], color=["#1f77b4", "#d62728"])
    ax[2].set_xticks([0, 1]); ax[2].set_xticklabels(["GNN\n(neighbours+distance)", "MLP\n(own features)"])
    ax[2].set_ylabel("test R²"); ax[2].set_ylim(min(0, r2m) - 0.05, 1.0)
    ax[2].set_title("why GNN: relational info lifts R²")
    for i, v in enumerate([r2g, r2m]):
        ax[2].text(i, v, f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=11)

    fig.suptitle("TSV layout → per-via delamination-risk GNN (theme B): message passing with pairwise "
                 "distance beats a structure-blind per-via MLP", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main()
