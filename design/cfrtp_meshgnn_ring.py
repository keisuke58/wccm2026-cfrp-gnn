"""cfrtp_meshgnn_field.py -- mesh field prediction: a MeshGraphNet-style GNN
(pure PyTorch, no torch_geometric) that predicts the residual-stress field on a
RING-SHAPED CFRTP part from the process (cooling rate R).

The ring is one of Daikin's demonstrated fluoropolymer/CF CFRTP molded shapes
(deep-draw box, hemisphere-with-rib, and ring; product forms are UD [0/90] sheet
and chopped sheet -- Daikin Chemicals CFRTP development report, 2021). Its inner
and outer rims are free surfaces that cool fast (local quench -> low residual)
while the mid-radius lags (locks in); mechanical equilibrium then couples
neighbours (a few Laplacian-smoothing iterations on the mesh). It reports a node-wise MLP baseline alongside the GNN. NOTE (honest): with the
distance-to-free-surface node feature at FIXED geometry this field is largely
pointwise, so the node-MLP is competitive/better here; the GNN's message-passing
advantage is expected for VARYING geometry or SPARSE node inputs (propagating
boundary/sensor information via connectivity) -- the setting of the main GAT
defect-localization work. This is a mesh-native field surrogate either way.

Physics-first / ML-subordinate: the accuracy AUTHORITY is the FE-verified 0D
crystallization+VE model (design/cfrtp_inverse_design.process), applied per node
with the free-surface conduction-lag cooling field.

  encoder  : node feats [x,y,d_free,R] and edge feats [dx,dy,dist] -> H
  processor: M message-passing steps (edge MLP -> sum-aggregate -> node MLP), residual
  decoder  : node H -> sigma

    python3 design/cfrtp_meshgnn_field.py

torch + numpy + matplotlib. Magnitudes illustrative (uncalibrated).
"""
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cfrtp_inverse_design import process     # FE authority

torch.manual_seed(0); np.random.seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NG = 16                        # background grid resolution (per axis)
CX, CY = 0.5, 0.5             # ring centre
RO, RI = 0.46, 0.22          # outer / inner radius (ring)
R_LO, R_HI = 200.0, 600.0    # cooling-rate range [C/min]
AMP, LAM, SMOOTH = 4.0, 0.06, 6   # free-surface speedup, decay length, smoothing iters


def build_mesh():
    xs = np.linspace(0, 1, NG); ys = np.linspace(0, 1, NG)
    idx = -np.ones((NG, NG), int); coords = []
    for i in range(NG):
        for j in range(NG):
            r = np.hypot(xs[i] - CX, ys[j] - CY)
            if RI <= r <= RO:                       # keep annulus nodes only
                idx[i, j] = len(coords); coords.append((xs[i], ys[j]))
    coords = np.array(coords); edges = []
    for i in range(NG):
        for j in range(NG):
            if idx[i, j] < 0:
                continue
            for di, dj in ((1, 0), (0, 1)):
                ni, nj = i + di, j + dj
                if ni < NG and nj < NG and idx[ni, nj] >= 0:
                    a, b = idx[i, j], idx[ni, nj]
                    edges += [(a, b), (b, a)]
    return coords, np.array(edges).T


def geom_features(coords):
    x, y = coords[:, 0], coords[:, 1]
    rr = np.hypot(x - CX, y - CY)
    d_free = np.minimum(rr - RI, RO - rr)           # distance to nearest free rim
    return x, y, d_free


def laplacian_smooth(vals, edges, n_nodes, iters):
    src, dst = edges
    deg = np.bincount(dst, minlength=n_nodes).astype(float); deg[deg == 0] = 1
    v = vals.copy()
    for _ in range(iters):
        acc = np.bincount(dst, weights=v[src], minlength=n_nodes)
        v = 0.5 * v + 0.5 * acc / deg
    return v


def fe_field(R, coords, edges, d_free):
    r = R * (1.0 + AMP * np.exp(-d_free / LAM))      # fast at the free rims
    sig = np.array([process(max(ri, 5.0))[0] for ri in r])
    return laplacian_smooth(sig, edges, len(coords), SMOOTH)


def mlp(din, dout, h=64):
    return nn.Sequential(nn.Linear(din, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
                         nn.Linear(h, dout))


class MeshGNN(nn.Module):
    def __init__(self, fn=4, fe=3, H=32, steps=3):
        super().__init__()
        self.nenc = mlp(fn, H); self.eenc = mlp(fe, H)
        self.msg = nn.ModuleList([mlp(3 * H, H) for _ in range(steps)])
        self.upd = nn.ModuleList([mlp(2 * H, H) for _ in range(steps)])
        self.dec = mlp(H, 1); self.steps = steps

    def forward(self, nf, ef, ei):
        h = self.nenc(nf); e = self.eenc(ef); src, dst = ei
        for k in range(self.steps):
            m = self.msg[k](torch.cat([h[src], h[dst], e], 1))
            agg = torch.zeros_like(h).index_add_(0, dst, m)
            h = h + self.upd[k](torch.cat([h, agg], 1))
            e = e + m
        return self.dec(h).squeeze(-1)


class NodeMLP(nn.Module):                 # baseline: no message passing
    def __init__(self, fn=4, H=64):
        super().__init__(); self.net = mlp(fn, 1, H)

    def forward(self, nf, ef, ei):
        return self.net(nf).squeeze(-1)


def batch_graphs(Rlist, coords, edges, feats):
    x, y, d_free = feats; Nn = len(coords)
    NF, EF, EI, Y = [], [], [], []
    src, dst = edges
    ef_base = np.stack([coords[dst, 0] - coords[src, 0], coords[dst, 1] - coords[src, 1],
                        np.linalg.norm(coords[dst] - coords[src], axis=1)], 1)
    for b, R in enumerate(Rlist):
        Rn = (np.log10(R) - np.log10(R_LO)) / (np.log10(R_HI) - np.log10(R_LO))
        NF.append(np.stack([x, y, d_free / 0.2, np.full(Nn, Rn)], 1))
        EF.append(ef_base); EI.append(edges + b * Nn)
        Y.append(fe_field(R, coords, edges, d_free))
    return (torch.tensor(np.concatenate(NF), dtype=torch.float32),
            torch.tensor(np.concatenate(EF), dtype=torch.float32),
            torch.tensor(np.concatenate(EI, 1), dtype=torch.long),
            torch.tensor(np.concatenate(Y), dtype=torch.float32))


def train(model, nf, ef, ei, y, ymu, ysd, epochs=400, lr=3e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr); lf = nn.MSELoss()
    yn = (y - ymu) / ysd
    for _ in range(epochs):
        opt.zero_grad(); lf(model(nf, ef, ei), yn).backward(); opt.step()
    return model


def relL2(model, nf, ef, ei, y, ymu, ysd):
    model.eval()
    with torch.no_grad():
        p = model(nf, ef, ei) * ysd + ymu
    return float(torch.norm(p - y) / torch.norm(y)), p.cpu().numpy()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    coords, edges = build_mesh(); feats = geom_features(coords)
    print("mesh: %d nodes, %d directed edges (ring-shaped part)" % (len(coords), edges.shape[1]))
    Rtr = 10 ** np.random.uniform(np.log10(R_LO), np.log10(R_HI), 14)
    Rte = 10 ** np.random.uniform(np.log10(R_LO), np.log10(R_HI), 6)
    nf, ef, ei, y = [t.to(DEV) for t in batch_graphs(Rtr, coords, edges, feats)]
    nfe, efe, eie, ye = [t.to(DEV) for t in batch_graphs(Rte, coords, edges, feats)]
    ymu, ysd = float(y.mean()), float(y.std() + 1e-9)

    gnn = train(MeshGNN().to(DEV), nf, ef, ei, y, ymu, ysd)
    mlpb = train(NodeMLP().to(DEV), nf, ef, ei, y, ymu, ysd)
    rg, pg = relL2(gnn, nfe, efe, eie, ye, ymu, ysd)
    rm, pm = relL2(mlpb, nfe, efe, eie, ye, ymu, ysd)
    print("test rel-L2:  MeshGNN = %.3f   node-MLP baseline = %.3f  (GNN better by %.0f%%)"
          % (rg, rm, 100 * (1 - rg / rm)))

    # ---- figure: FE vs GNN field for one test case + error ----
    Nn = len(coords); k = 0
    ytrue = ye[k * Nn:(k + 1) * Nn].cpu().numpy(); ypred = pg[k * Nn:(k + 1) * Nn]
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.0), dpi=120)
    fig.suptitle("Mesh GNN residual-stress field on a ring-shaped CFRTP part "
                 "(a Daikin CFRTP molded shape)\nR=%.0f °C/min; test rel-L2: "
                 "GNN %.3f vs node-MLP %.3f" % (Rte[k], rg, rm), fontweight="bold")
    vmin, vmax = ytrue.min(), ytrue.max()
    for ax, val, ttl, err in [(axs[0], ytrue, "FE (authority)", False),
                              (axs[1], ypred, "MeshGNN prediction", False),
                              (axs[2], np.abs(ypred - ytrue), "|error|", True)]:
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=val, s=26, marker="s", cmap="magma",
                        vmin=(0 if err else vmin),
                        vmax=(max(0.15 * (vmax - vmin), 1e-6) if err else vmax))
        ax.set_aspect("equal"); ax.set_title(ttl, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([]); fig.colorbar(sc, ax=ax, fraction=0.046)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = os.path.join(here, "cfrtp_meshgnn_ring.png"); fig.savefig(out, dpi=140); plt.close(fig)
    print("wrote %s" % out)
    return dict(gnn=rg, mlp=rm)


if __name__ == "__main__":
    main()
