"""fe_newton_warmstart.py — neural warm-start for FE Newton on the nonlinear
(Poisson-Boltzmann) semiconductor equation. Idea "C": the FE solver stays the
authority on accuracy; the operator net only supplies a good Newton initial guess,
so it *accelerates* rather than replaces the physics — the framing industrial ML-TCAD
work values ("ML fixes/accelerates TCAD convergence"), here married to the repo's
weak-form line.

Equation (1-D, scaled equilibrium nonlinear Poisson / Poisson-Boltzmann):

    -lambda^2 u'' + sinh(u) = d(x),     u(0) = u(1) = 0

u = phi / V_t (scaled potential), d(x) = scaled net doping, lambda = Debye length.
The sinh(u) space-charge term makes it nonlinear -> solved by Newton: each step
assembles the tangent  J = lambda^2 K + diag(ML * cosh(u))  and solves  J du = -R,
with the P1 weak form (K stiffness, ML lumped mass). Newton's iteration count to a
fixed tolerance depends strongly on the initial guess.

  * cold start : u0 = 0                      (baseline)
  * warm start : u0 = net(d)                 (learned operator d -> u)

Online: doping profiles stream in; each is solved warm-started, the converged
(d, u) pair is buffered, and the net is retrained on the fly. The warm-start Newton
count falls below the (flat) cold-start baseline and keeps dropping as the operator
adapts — while every solution is the exact FE solution (accuracy is never traded).

Run:  python3 fe_newton_warmstart.py            (writes fe_newton_warmstart.png)
      python3 fe_newton_warmstart.py --help
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn

SEED = 20260722
LAMBDA = 0.05          # scaled Debye length (small -> boundary/junction layers)
NEWTON_TOL = 1e-8
NEWTON_MAX = 50


# ----------------------------------------------------------------------------
# 1-D P1 finite-element operators on [0,1]
# ----------------------------------------------------------------------------
def fe_1d(n):
    x = np.linspace(0.0, 1.0, n)
    h = x[1] - x[0]
    K = np.zeros((n, n))          # stiffness  int u' v'
    M = np.zeros((n, n))          # consistent mass  int u v
    for e in range(n - 1):
        K[e, e] += 1 / h;   K[e + 1, e + 1] += 1 / h
        K[e, e + 1] -= 1 / h;   K[e + 1, e] -= 1 / h
        M[e, e] += h / 3;   M[e + 1, e + 1] += h / 3
        M[e, e + 1] += h / 6;   M[e + 1, e] += h / 6
    ML = M.sum(axis=1)            # lumped mass (row sums) for the nonlinear term
    return x, K, M, ML


def doping(x, rng):
    """Random pn-junction-like scaled net doping d(x): step + smooth ripple."""
    xj = rng.uniform(0.35, 0.65)
    amp = rng.uniform(3.0, 8.0)
    d = amp * np.tanh((x - xj) / rng.uniform(0.03, 0.09))
    d += rng.uniform(-2, 2) * np.sin(2 * np.pi * rng.uniform(1, 3) * x)
    return d


def newton_solve(d, K, ML, u0, free):
    """Damped FE Newton for -lambda^2 u'' + sinh(u) = d. Returns (u, n_iters, residuals).

    A backtracking line search (halve the step until the residual decreases) keeps
    the sinh nonlinearity from blowing up when the initial guess is imperfect — the
    standard TCAD safeguard, and what makes a warm start help instead of diverge.
    """
    u = np.asarray(u0, dtype=np.float64).copy()   # float64: a float32 net guess would
    u[~free] = 0.0                                 # floor the Newton correction near tol
    lamK = LAMBDA ** 2 * K

    def resid(uu):
        return lamK @ uu + ML * np.sinh(uu) - ML * d

    res_hist = []
    for it in range(1, NEWTON_MAX + 1):
        R = resid(u)
        rn = np.linalg.norm(R[free])
        res_hist.append(rn)
        if rn < NEWTON_TOL:
            return u, it - 1, res_hist
        J = lamK + np.diag(ML * np.cosh(u))              # consistent tangent
        du = np.zeros_like(u)
        du[free] = np.linalg.solve(J[np.ix_(free, free)], -R[free])
        alpha = 1.0
        for _ in range(30):                              # backtracking line search
            u_try = u + alpha * du
            if np.linalg.norm(resid(u_try)[free]) < rn:
                break
            alpha *= 0.5
        u = u + alpha * du
    return u, NEWTON_MAX, res_hist


# ----------------------------------------------------------------------------
# Operator net: learned initial guess  d(nodes) -> u(nodes)
# ----------------------------------------------------------------------------
class InitGuessNet(nn.Module):
    def __init__(self, n, hidden=160):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, n),
        )

    def forward(self, d, mask):
        return self.net(d) * mask       # hard u=0 on the boundary nodes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=81, help="FE nodes")
    ap.add_argument("--pretrain", type=int, default=2500, help="cold-start pretrain steps")
    ap.add_argument("--n_pre", type=int, default=64, help="pretrain doping profiles")
    ap.add_argument("--steps", type=int, default=80, help="online conditions")
    ap.add_argument("--adapt_iters", type=int, default=60, help="retrain steps per condition")
    ap.add_argument("--out", type=str, default="fe_newton_warmstart.png")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    x, K, M, ML = fe_1d(args.n)
    free = np.ones(args.n, bool); free[0] = free[-1] = False
    mask = torch.tensor(free.astype(np.float32))
    u_scale = 4.0                                    # typical |u|, for net normalisation

    def solve(d, u0):
        return newton_solve(d, K, ML, u0, free)

    # ---- offline: build a small set of exact solves, pretrain the init-guess net ----
    pre = []
    for _ in range(args.n_pre):
        d = doping(x, rng)
        u, _, _ = solve(d, np.zeros(args.n))
        pre.append((torch.tensor(d, dtype=torch.float32),
                    torch.tensor(u / u_scale, dtype=torch.float32)))
    net = InitGuessNet(args.n)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    for it in range(args.pretrain):
        d_t, u_t = pre[rng.integers(len(pre))]
        pred = net(d_t, mask)
        loss = ((pred - u_t) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (it + 1) % 400 == 0:
            print(f"[pretrain] {it+1}/{args.pretrain}  loss {loss.item():.3e}")
    for g in opt.param_groups:
        g["lr"] = 5e-4

    # ---- online: stream conditions, warm-start Newton, adapt the net ----
    replay = []
    cold_iters, warm_iters = [], []
    for s in range(args.steps):
        d = doping(x, rng)
        d_t = torch.tensor(d, dtype=torch.float32)
        with torch.no_grad():
            u0 = (net(d_t, mask).numpy() * u_scale)
        u_c, nc, res_c = solve(d, np.zeros(args.n))       # cold baseline (for comparison)
        u_w, nw, res_w = solve(d, u0)                     # warm start
        cold_iters.append(nc); warm_iters.append(nw)

        # buffer the exact solution and retrain the init-guess operator on the fly
        replay.append((d_t, torch.tensor(u_w / u_scale, dtype=torch.float32)))
        for _ in range(args.adapt_iters):
            dj, uj = replay[rng.integers(len(replay))]
            pred = net(dj, mask)
            loss = ((pred - uj) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()

        if s in (0, args.steps // 2, args.steps - 1):
            print(f"[online] cond {s:2d}  cold {nc:2d} iters   warm {nw:2d} iters")

    half = args.steps // 2
    print(f"\nNewton iters (cold vs warm):  cold avg {np.mean(cold_iters):.1f}"
          f"   warm avg {np.mean(warm_iters):.1f}"
          f"   |  warm first-half {np.mean(warm_iters[:half]):.1f}"
          f" -> second-half {np.mean(warm_iters[half:]):.1f}")

    # a representative case for the convergence-curve panel
    d = doping(x, rng)
    u0 = net(torch.tensor(d, dtype=torch.float32), mask).detach().numpy() * u_scale
    u_c, _, res_c = solve(d, np.zeros(args.n))
    u_w, _, res_w = solve(d, u0)

    _plot(args.out, x, d, u0, u_w, res_c, res_w, cold_iters, warm_iters, half)
    print(f"wrote {args.out}")


def _plot(out, x, d, u0, u_w, res_c, res_w, cold_iters, warm_iters, half):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    ax[0, 0].plot(x, d, color="#7f7f7f", label="doping d(x)")
    ax[0, 0].plot(x, u0, "--", color="#ff7f0e", label="net initial guess u0")
    ax[0, 0].plot(x, u_w, color="#1f77b4", label="converged FE u")
    ax[0, 0].set_title("representative case: net guess vs converged FE solution")
    ax[0, 0].set_xlabel("x"); ax[0, 0].legend()

    ax[0, 1].semilogy(range(len(res_c)), res_c, "-o", color="#d62728", label="cold start (u0=0)")
    ax[0, 1].semilogy(range(len(res_w)), res_w, "-o", color="#1f77b4", label="warm start (net)")
    ax[0, 1].set_title("Newton residual vs iteration")
    ax[0, 1].set_xlabel("Newton iteration"); ax[0, 1].set_ylabel("||R||"); ax[0, 1].legend()
    ax[0, 1].grid(True, which="both", alpha=0.3)

    xs = np.arange(len(cold_iters))
    ax[1, 0].plot(xs, cold_iters, "-o", ms=3, color="#d62728", label="cold start")
    ax[1, 0].plot(xs, warm_iters, "-o", ms=3, color="#1f77b4", label="warm start (adapting)")
    ax[1, 0].set_title("Newton iterations per streamed condition")
    ax[1, 0].set_xlabel("online condition"); ax[1, 0].set_ylabel("iterations to converge")
    ax[1, 0].legend()

    cold_e = np.mean(cold_iters[:half]); cold_l = np.mean(cold_iters[half:])
    warm_e = np.mean(warm_iters[:half]); warm_l = np.mean(warm_iters[half:])
    gx = np.arange(2)
    ax[1, 1].bar(gx - 0.2, [cold_e, cold_l], 0.4, color="#d62728", label="cold")
    ax[1, 1].bar(gx + 0.2, [warm_e, warm_l], 0.4, color="#1f77b4", label="warm")
    ax[1, 1].set_xticks(gx); ax[1, 1].set_xticklabels(["first half", "second half"])
    ax[1, 1].set_ylabel("avg Newton iterations")
    ax[1, 1].set_title("warm-start speedup grows as the operator adapts")
    for i, v in enumerate([cold_e, cold_l]):
        ax[1, 1].text(i - 0.2, v + 0.1, f"{v:.1f}", ha="center", fontsize=9)
    for i, v in enumerate([warm_e, warm_l]):
        ax[1, 1].text(i + 0.2, v + 0.1, f"{v:.1f}", ha="center", fontsize=9)
    ax[1, 1].legend()

    fig.suptitle("Neural warm-start for FE Newton on the nonlinear Poisson-Boltzmann "
                 "equation (FEM keeps accuracy; net cuts iterations)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main()
