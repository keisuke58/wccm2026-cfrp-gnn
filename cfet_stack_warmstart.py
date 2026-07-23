"""cfet_stack_warmstart.py — idea C on a CFET stacked cross-section: neural warm-start
for FE Newton on a vertically stacked complementary-FET slice.

Where the roadmap goes next after GAA (FinFET -> GAA -> Forksheet -> CFET -> 2D-CFET
-> 3D monolithic), the device cross-section stops being a single tier: a CFET stacks
an nFET on a pFET separated by a dielectric isolation band, so the doping AND the
permittivity vary in *both* x and y. That is exactly the regime where the repo's two
pillars pay off — weak-form FE assembly on a multi-material section (variable eps with
a horizontal dielectric interface) and a learned Newton warm-start whose value grows
with problem complexity.

This reuses the 2-D nonlinear-Poisson (Poisson-Boltzmann) machinery of
dd2d_newton_warmstart.py unchanged (damped P1-FE Newton, CNN warm-start operator,
online adaptation) and only swaps the geometry:

  * bottom tier  (y < 0.4)   : nFET slice   -> net doping  +profile(x)
  * dielectric   (0.4..0.6)  : isolation    -> C = 0, low-k eps
  * top tier     (y > 0.6)   : pFET slice   -> net doping  -profile(x)  (complementary)

with profile(x) an n+/channel/n+ source-drain step. Every solve is the exact FE
solution; only the Newton iteration count changes, and it drops with a warm start.

Run:  python3 cfet_stack_warmstart.py            (writes cfet_stack_warmstart.png)
      python3 cfet_stack_warmstart.py --help
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from pi_deeponet_fem_gaa import build_mesh, assemble        # 2-D P1 FE assets
from dd2d_newton_warmstart import newton_2d, WarmCNN        # reuse solver + warm-start net

SEED = 20260722
LAMBDA = 0.05
Y_BOT, Y_TOP, Y_W = 0.40, 0.60, 0.04     # tier boundaries + transition width
EPS_SI, EPS_DIE = 11.7, 3.9              # Si tiers vs dielectric isolation band


# ----------------------------------------------------------------------------
# CFET stacked geometry: complementary doping over two tiers + dielectric band
# ----------------------------------------------------------------------------
def eps_cfet(nodes):
    """High-k Si tiers, low-k dielectric isolation band in the middle (interface in y)."""
    y = nodes[:, 1]
    die = np.abs(y - 0.5) < (0.5 * (Y_TOP - Y_BOT))
    return np.where(die, EPS_DIE, EPS_SI).astype(np.float64)


def doping(nodes, p):
    x, y = nodes[:, 0], nodes[:, 1]
    xL, xR, Nsd, Nch, w = p
    src = 0.5 * (1 - np.tanh((x - xL) / w))
    drn = 0.5 * (1 + np.tanh((x - xR) / w))
    chan = 1.0 - src - drn
    profile = Nsd * (src + drn) - Nch * chan          # n+ S/D, p channel (bottom nFET)
    wbot = 0.5 * (1 - np.tanh((y - Y_BOT) / Y_W))     # 1 in bottom tier
    wtop = 0.5 * (1 + np.tanh((y - Y_TOP) / Y_W))     # 1 in top tier
    return (profile * wbot - profile * wtop).astype(np.float64)   # top complementary


def rand_params(rng):
    return (rng.uniform(0.28, 0.40), rng.uniform(0.60, 0.72),
            rng.uniform(6.0, 14.0), rng.uniform(4.0, 10.0), rng.uniform(0.03, 0.06))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=33, help="nodes per side (grid n x n)")
    ap.add_argument("--pretrain", type=int, default=1200, help="pretrain steps")
    ap.add_argument("--n_pre", type=int, default=40, help="pretrain conditions")
    ap.add_argument("--steps", type=int, default=60, help="online conditions")
    ap.add_argument("--adapt_iters", type=int, default=40, help="retrain steps per condition")
    ap.add_argument("--out", type=str, default="cfet_stack_warmstart.png")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    n = args.n
    nodes, tris, on_bnd = build_mesh(n)
    free = ~on_bnd
    eps_node = eps_cfet(nodes)
    K, M = assemble(nodes, tris, eps_node)
    Klam = (LAMBDA ** 2) * K.tocsr()
    ML = np.asarray(M.sum(axis=1)).ravel()

    C_scale, u_scale = 14.0, 4.5
    eps_grid = torch.tensor((eps_node / eps_node.max()).reshape(n, n), dtype=torch.float32)
    mask2d = torch.tensor(free.reshape(n, n).astype(np.float32))

    def inp_of(C):
        c = torch.tensor((C / C_scale).reshape(n, n), dtype=torch.float32)
        return torch.stack([c, eps_grid], dim=0).unsqueeze(0)

    def solve(C, u0):
        return newton_2d(C, Klam, ML, u0, free)

    # ---- offline pretrain ----
    pre = []
    for _ in range(args.n_pre):
        C = doping(nodes, rand_params(rng))
        u, _, _ = solve(C, np.zeros(n * n))
        pre.append((inp_of(C), torch.tensor((u / u_scale).reshape(n, n), dtype=torch.float32)))
    net = WarmCNN()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    for it in range(args.pretrain):
        xin, ut = pre[rng.integers(len(pre))]
        loss = ((net(xin, mask2d) - ut) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (it + 1) % 300 == 0:
            print(f"[pretrain] {it+1}/{args.pretrain}  loss {loss.item():.3e}")
    for g in opt.param_groups:
        g["lr"] = 5e-4

    # ---- online: stream CFET conditions, warm-start, adapt ----
    replay = []
    cold_iters, warm_iters = [], []
    for s in range(args.steps):
        C = doping(nodes, rand_params(rng))
        xin = inp_of(C)
        with torch.no_grad():
            u0 = net(xin, mask2d).numpy().reshape(-1) * u_scale
        _, nc, res_c = solve(C, np.zeros(n * n))
        u_w, nw, res_w = solve(C, u0)
        cold_iters.append(nc); warm_iters.append(nw)
        replay.append((xin, torch.tensor((u_w / u_scale).reshape(n, n), dtype=torch.float32)))
        for _ in range(args.adapt_iters):
            xj, uj = replay[rng.integers(len(replay))]
            loss = ((net(xj, mask2d) - uj) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if s in (0, args.steps // 2, args.steps - 1):
            print(f"[online] cond {s:2d}  cold {nc:2d}  warm {nw:2d}")

    half = args.steps // 2
    print(f"\nNewton iters:  cold avg {np.mean(cold_iters):.1f}   warm avg {np.mean(warm_iters):.1f}"
          f"   |  warm first-half {np.mean(warm_iters[:half]):.1f}"
          f" -> second-half {np.mean(warm_iters[half:]):.1f}")

    C = doping(nodes, rand_params(rng))
    with torch.no_grad():
        u0 = net(inp_of(C), mask2d).numpy().reshape(-1) * u_scale
    _, _, res_c = solve(C, np.zeros(n * n))
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

    def tier_lines(a):
        for yv in (Y_BOT, Y_TOP):
            a.axhline(yv, color="w", ls="--", lw=1.0, alpha=0.8)

    panels = [
        (ax[0, 0], eps_node, "permittivity eps (Si tiers / dielectric band)", "viridis"),
        (ax[0, 1], C, "net doping C(x,y)  (nFET bottom / pFET top)", "coolwarm"),
        (ax[0, 2], u_w, "converged FE u (stacked potential)", "magma"),
    ]
    for a, field, title, cm in panels:
        tp = a.tripcolor(triang, field, shading="gouraud", cmap=cm)
        tier_lines(a); a.set_aspect("equal"); a.set_title(title)
        fig.colorbar(tp, ax=a, fraction=0.046)

    ax[1, 0].semilogy(range(len(res_c)), res_c, "-o", color="#d62728", label="cold (u0=0)")
    ax[1, 0].semilogy(range(len(res_w)), res_w, "-o", color="#1f77b4", label="warm (CNN)")
    ax[1, 0].set_title("Newton residual vs iteration (representative)")
    ax[1, 0].set_xlabel("Newton iteration"); ax[1, 0].set_ylabel("||R||")
    ax[1, 0].legend(); ax[1, 0].grid(True, which="both", alpha=0.3)

    xs = np.arange(len(cold_iters))
    ax[1, 1].plot(xs, cold_iters, "-o", ms=3, color="#d62728", label="cold")
    ax[1, 1].plot(xs, warm_iters, "-o", ms=3, color="#1f77b4", label="warm (adapting)")
    ax[1, 1].set_title("Newton iterations per streamed CFET condition")
    ax[1, 1].set_xlabel("online condition"); ax[1, 1].set_ylabel("iterations"); ax[1, 1].legend()

    ce, cl = np.mean(cold_iters[:half]), np.mean(cold_iters[half:])
    we, wl = np.mean(warm_iters[:half]), np.mean(warm_iters[half:])
    gx = np.arange(2)
    ax[1, 2].bar(gx - 0.2, [ce, cl], 0.4, color="#d62728", label="cold")
    ax[1, 2].bar(gx + 0.2, [we, wl], 0.4, color="#1f77b4", label="warm")
    ax[1, 2].set_xticks(gx); ax[1, 2].set_xticklabels(["first half", "second half"])
    ax[1, 2].set_ylabel("avg Newton iterations")
    ax[1, 2].set_title("warm start: fewer Newton iterations (every solve exact FE)")
    for i, v in enumerate([ce, cl]):
        ax[1, 2].text(i - 0.2, v + 0.1, f"{v:.1f}", ha="center", fontsize=9)
    for i, v in enumerate([we, wl]):
        ax[1, 2].text(i + 0.2, v + 0.1, f"{v:.1f}", ha="center", fontsize=9)
    ax[1, 2].legend()

    fig.suptitle("CFET stacked cross-section (nonlinear Poisson): neural warm-start for "
                 "FE Newton — weak-form FE on a multi-material stack, net cuts iterations",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main()
