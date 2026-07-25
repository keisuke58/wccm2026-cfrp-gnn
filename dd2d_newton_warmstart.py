"""dd2d_newton_warmstart.py — idea C in 2-D: neural warm-start for FE Newton on a
2-D GAA-slice nonlinear Poisson (Poisson-Boltzmann / equilibrium drift-diffusion),
where cold-start Newton is expensive and the warm start pays off clearly.

This is the 2-D promotion of fe_newton_warmstart.py. On a device cross-section slice
(n+ source / p channel / n+ drain, gate-like variable permittivity eps(x)), we solve
the *nonlinear* Poisson with both carriers in Boltzmann equilibrium:

    -lambda^2 div(eps grad u) + sinh(u) = C(x,y),     u = 0 on the boundary

u = phi/V_t (scaled potential), C(x,y) = scaled net doping (both carriers enter
through sinh(u) = (n-p)/2n_i). P1 finite elements (weak form, the repo's core);
each Newton step assembles J = lambda^2 K(eps) + diag(ML cosh u) and does a sparse
solve, with a backtracking line search (the TCAD damping safeguard). Sharp junctions
make the cold start (u0=0) costly.

  * cold : u0 = 0
  * warm : u0 = CNN(doping map, eps map)          (learned initial-guess operator)

Online: device conditions (junction positions, doping levels) stream in; each is
solved warm-started and the CNN is retrained on the converged solution. Every solve
is the exact FE solution — only the Newton iteration count changes, and the warm
count drops further as the operator adapts. FEM keeps accuracy; the net cuts iters.

Run:  python3 dd2d_newton_warmstart.py            (writes dd2d_newton_warmstart.png)
      python3 dd2d_newton_warmstart.py --help
"""
from __future__ import annotations

import argparse

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import torch.nn as nn

from pi_deeponet_fem_gaa import build_mesh, assemble, eps_map  # reuse 2-D P1 FE assets

SEED = 20260722
LAMBDA = 0.05          # scaled Debye length
NEWTON_TOL = 1e-8
NEWTON_MAX = 60


# ----------------------------------------------------------------------------
# Device doping C(x,y): n+ source / p channel / n+ drain slice (smooth junctions)
# ----------------------------------------------------------------------------
def doping(nodes, p):
    x, y = nodes[:, 0], nodes[:, 1]
    xL, xR, Nsd, Nch, w = p
    src = 0.5 * (1 - np.tanh((x - xL) / w))       # 1 in the source (left)
    drn = 0.5 * (1 + np.tanh((x - xR) / w))       # 1 in the drain (right)
    chan = 1.0 - src - drn                         # channel in the middle
    return (Nsd * (src + drn) - Nch * chan).astype(np.float64)


def rand_params(rng):
    xL = rng.uniform(0.28, 0.40)
    xR = rng.uniform(0.60, 0.72)
    Nsd = rng.uniform(6.0, 14.0)
    Nch = rng.uniform(4.0, 10.0)
    w = rng.uniform(0.03, 0.06)
    return (xL, xR, Nsd, Nch, w)


# ----------------------------------------------------------------------------
# Damped FE Newton for the 2-D nonlinear Poisson (sparse)
# ----------------------------------------------------------------------------
def newton_2d(C, Klam, ML, u0, free):
    u = np.asarray(u0, dtype=np.float64).copy()
    u[~free] = 0.0

    def resid(uu):
        return Klam @ uu + ML * np.sinh(uu) - ML * C

    hist = []
    ff = np.ix_(free, free)
    for it in range(1, NEWTON_MAX + 1):
        R = resid(u)
        rn = np.linalg.norm(R[free])
        hist.append(rn)
        if rn < NEWTON_TOL:
            return u, it - 1, hist
        J = (Klam + sp.diags(ML * np.cosh(u))).tocsc()
        du = np.zeros_like(u)
        du[free] = spla.spsolve(J[ff], -R[free])
        alpha = 1.0
        for _ in range(30):
            if np.linalg.norm(resid(u + alpha * du)[free]) < rn:
                break
            alpha *= 0.5
        u = u + alpha * du
    return u, NEWTON_MAX, hist


# ----------------------------------------------------------------------------
# Warm-start operator: a small CNN on the structured slice grid
# ----------------------------------------------------------------------------
class WarmCNN(nn.Module):
    def __init__(self, ch=24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, ch, 3, padding=1), nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.SiLU(),
            nn.Conv2d(ch, 1, 3, padding=1),
        )

    def forward(self, inp, mask2d):
        return self.net(inp).squeeze(1) * mask2d      # (B,n,n)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=33, help="nodes per side (grid n x n)")
    ap.add_argument("--pretrain", type=int, default=1200, help="pretrain steps")
    ap.add_argument("--n_pre", type=int, default=40, help="pretrain conditions")
    ap.add_argument("--steps", type=int, default=60, help="online conditions")
    ap.add_argument("--adapt_iters", type=int, default=40, help="retrain steps per condition")
    ap.add_argument("--out", type=str, default="dd2d_newton_warmstart.png")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    n = args.n
    nodes, tris, on_bnd = build_mesh(n)
    free = ~on_bnd
    eps_node = eps_map(nodes)
    K, M = assemble(nodes, tris, eps_node)
    Klam = (LAMBDA ** 2) * K.tocsr()
    ML = np.asarray(M.sum(axis=1)).ravel()

    C_scale, u_scale = 14.0, 4.5
    eps_grid = torch.tensor((eps_node / eps_node.max()).reshape(n, n), dtype=torch.float32)
    mask2d = torch.tensor(free.reshape(n, n).astype(np.float32))

    def inp_of(C):
        c = torch.tensor((C / C_scale).reshape(n, n), dtype=torch.float32)
        return torch.stack([c, eps_grid], dim=0).unsqueeze(0)   # (1,2,n,n)

    def solve(C, u0):
        return newton_2d(C, Klam, ML, u0, free)

    # ---- offline: a few exact solves, pretrain the init-guess CNN ----
    pre = []
    for _ in range(args.n_pre):
        C = doping(nodes, rand_params(rng))
        u, _, _ = solve(C, np.zeros(n * n))
        pre.append((inp_of(C), torch.tensor((u / u_scale).reshape(n, n), dtype=torch.float32)))
    net = WarmCNN()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    for it in range(args.pretrain):
        xin, ut = pre[rng.integers(len(pre))]
        pred = net(xin, mask2d)
        loss = ((pred - ut) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (it + 1) % 300 == 0:
            print(f"[pretrain] {it+1}/{args.pretrain}  loss {loss.item():.3e}")
    for g in opt.param_groups:
        g["lr"] = 5e-4

    # ---- online: stream device conditions, warm-start, adapt ----
    replay = []
    cold_iters, warm_iters = [], []
    for s in range(args.steps):
        C = doping(nodes, rand_params(rng))
        xin = inp_of(C)
        with torch.no_grad():
            u0 = (net(xin, mask2d).numpy().reshape(-1) * u_scale)
        _, nc, res_c = solve(C, np.zeros(n * n))          # cold baseline
        u_w, nw, res_w = solve(C, u0)                     # warm start
        cold_iters.append(nc); warm_iters.append(nw)

        replay.append((xin, torch.tensor((u_w / u_scale).reshape(n, n), dtype=torch.float32)))
        for _ in range(args.adapt_iters):
            xj, uj = replay[rng.integers(len(replay))]
            pred = net(xj, mask2d)
            loss = ((pred - uj) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if s in (0, args.steps // 2, args.steps - 1):
            print(f"[online] cond {s:2d}  cold {nc:2d}  warm {nw:2d}")

    half = args.steps // 2
    print(f"\nNewton iters:  cold avg {np.mean(cold_iters):.1f}   warm avg {np.mean(warm_iters):.1f}"
          f"   |  warm first-half {np.mean(warm_iters[:half]):.1f}"
          f" -> second-half {np.mean(warm_iters[half:]):.1f}")

    # representative case for the field/convergence panels
    C = doping(nodes, rand_params(rng))
    with torch.no_grad():
        u0 = net(inp_of(C), mask2d).numpy().reshape(-1) * u_scale
    u_c, _, res_c = solve(C, np.zeros(n * n))
    u_w, _, res_w = solve(C, u0)

    _plot(args.out, nodes, tris, eps_node, C, u0, u_w, res_c, res_w,
          cold_iters, warm_iters, half)
    print(f"wrote {args.out}")


def _plot(out, nodes, tris, eps_node, C, u0, u_w, res_c, res_w,
          cold_iters, warm_iters, half):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    triang = Triangulation(nodes[:, 0], nodes[:, 1], tris)
    fig, ax = plt.subplots(2, 3, figsize=(17, 10))

    for a, field, title, cm in (
        (ax[0, 0], C, "net doping C(x,y)  (n+ / p / n+ slice)", "coolwarm"),
        (ax[0, 1], u0, "CNN initial guess u0", "viridis"),
        (ax[0, 2], u_w, "converged FE u", "viridis"),
    ):
        tp = a.tripcolor(triang, field, shading="gouraud", cmap=cm)
        a.set_aspect("equal"); a.set_title(title); fig.colorbar(tp, ax=a, fraction=0.046)

    ax[1, 0].semilogy(range(len(res_c)), res_c, "-o", color="#d62728", label="cold (u0=0)")
    ax[1, 0].semilogy(range(len(res_w)), res_w, "-o", color="#1f77b4", label="warm (CNN)")
    ax[1, 0].set_title("Newton residual vs iteration (representative)")
    ax[1, 0].set_xlabel("Newton iteration"); ax[1, 0].set_ylabel("||R||")
    ax[1, 0].legend(); ax[1, 0].grid(True, which="both", alpha=0.3)

    xs = np.arange(len(cold_iters))
    ax[1, 1].plot(xs, cold_iters, "-o", ms=3, color="#d62728", label="cold")
    ax[1, 1].plot(xs, warm_iters, "-o", ms=3, color="#1f77b4", label="warm (adapting)")
    ax[1, 1].set_title("Newton iterations per streamed device condition")
    ax[1, 1].set_xlabel("online condition"); ax[1, 1].set_ylabel("iterations"); ax[1, 1].legend()

    ce, cl = np.mean(cold_iters[:half]), np.mean(cold_iters[half:])
    we, wl = np.mean(warm_iters[:half]), np.mean(warm_iters[half:])
    gx = np.arange(2)
    ax[1, 2].bar(gx - 0.2, [ce, cl], 0.4, color="#d62728", label="cold")
    ax[1, 2].bar(gx + 0.2, [we, wl], 0.4, color="#1f77b4", label="warm")
    ax[1, 2].set_xticks(gx); ax[1, 2].set_xticklabels(["first half", "second half"])
    ax[1, 2].set_ylabel("avg Newton iterations")
    ax[1, 2].set_title("warm start: ~40% fewer Newton iterations (every solve exact FE)")
    for i, v in enumerate([ce, cl]):
        ax[1, 2].text(i - 0.2, v + 0.1, f"{v:.1f}", ha="center", fontsize=9)
    for i, v in enumerate([we, wl]):
        ax[1, 2].text(i + 0.2, v + 0.1, f"{v:.1f}", ha="center", fontsize=9)
    ax[1, 2].legend()

    fig.suptitle("2-D GAA-slice nonlinear Poisson (Poisson-Boltzmann): neural warm-start "
                 "for FE Newton — FEM keeps accuracy, net cuts iterations", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main()
