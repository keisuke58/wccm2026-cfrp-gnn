"""cfrtp_process_surrogate.py — process -> residual-stress SURROGATE for the CFRTP theme
(Daikin/NEDO, research/MEMO_cfrp_residual_stress_collab.md), the ML-as-accelerator
layer sitting on top of the thermo-viscoelastic CFRTP FE
(cfrtp_viscoelastic_residual_stress.py). Same posture as ㉑: FE is the accuracy
authority and generates the data; the net just screens the process space fast for
process-window design / inverse design.

Task: the manufacturing PROCESS knobs
    (cooling rate, melt/process temperature, material crystallization propensity X_inf)
map to the RESIDUAL state (peak residual sigma_xx and attained crystallinity). We run
the viscoelastic FE across a Latin-hypercube-ish grid of process settings to build the
dataset, hold out a random subset, and train a small MLP surrogate. It then predicts an
unseen process setting's residual stress in microseconds (vs an FE solve), giving a
fast response surface for "which cooling schedule minimizes residual stress?".

Reported: held-out relative error of the surrogate for residual stress and crystallinity,
and a (cooling-rate x melt-temperature) response surface with FE check points.

Honest scope: the FE data itself is the reduced thermo-viscoelastic seed (not data-
calibrated), and the 3-parameter process space is smooth, so the surrogate's role is
fast screening / inverse design, not extrapolation; FE remains the authority. No new
physics here. Physics leads; ML is the subordinate accelerator.

Run:  python3 cfrtp_process_surrogate.py     (runs the FE sweep, then trains)
      python3 cfrtp_process_surrogate.py --help
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn

import cfrtp_residual_stress_fe as base
import cfrtp_viscoelastic_residual_stress as ve

SEED = 20260726


def fe_residual(rate, t_melt, x_inf, nx=24, nz=10, nstep=26):
    """One viscoelastic-FE evaluation: process setting -> (peak residual sigma_xx [MPa],
    attained crystallinity). Sets the base-module globals the FE reads."""
    base.T_MELT = t_melt; base.X_INF = x_inf
    _, _, sig, _, _, _, _, Xf, _, _ = ve.solve(nx, nz, nstep, rate=rate, viscoelastic=True)
    return float(np.max(np.abs(sig[:, 0])) / 1e6), float(Xf.max())


def build_dataset(nsamp, rng):
    lo = np.array([np.log10(0.5), 300.0, 0.40])         # log10(rate), T_melt, X_inf
    hi = np.array([np.log10(64.0), 360.0, 0.60])
    P = lo + (hi - lo) * rng.random((nsamp, 3))
    Y = np.zeros((nsamp, 2))
    for i, (lr, tm, xi) in enumerate(P):
        Y[i] = fe_residual(10 ** lr, tm, xi)
    return P, Y


class MLP(nn.Module):
    def __init__(self, nin=3, nout=2, width=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, nout))

    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nsamp", type=int, default=64, help="FE process samples")
    ap.add_argument("--ntest", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--out", type=str, default="cfrtp_process_surrogate.png")
    args = ap.parse_args()
    torch.manual_seed(SEED); np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    print(f"generating {args.nsamp} viscoelastic-FE process samples ...")
    P, Y = build_dataset(args.nsamp, rng)

    perm = rng.permutation(args.nsamp)
    te = np.sort(perm[:args.ntest]); tr = np.sort(perm[args.ntest:])
    p_mu, p_sd = P[tr].mean(0), P[tr].std(0) + 1e-9
    y_mu, y_sd = Y[tr].mean(0), Y[tr].std(0) + 1e-9
    Pn = (P - p_mu) / p_sd; Yn = (Y - y_mu) / y_sd
    Pt = torch.tensor(Pn, dtype=torch.float32); Yt = torch.tensor(Yn, dtype=torch.float32)
    tri = torch.tensor(tr)

    net = MLP()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    sch = torch.optim.lr_scheduler.StepLR(opt, 1200, 0.5)
    for ep in range(args.epochs):
        opt.zero_grad()
        loss = ((net(Pt[tri]) - Yt[tri]) ** 2).mean()
        loss.backward(); opt.step(); sch.step()
        if (ep + 1) % 1000 == 0:
            print(f"  epoch {ep+1} train MSE {loss.item():.4e}")

    net.eval()
    with torch.no_grad():
        pred = net(Pt).numpy() * y_sd + y_mu
    def rl2(a, b):
        return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))
    e_sig = rl2(pred[te, 0], Y[te, 0]); e_x = rl2(pred[te, 1], Y[te, 1])
    print(f"\nCFRTP process->residual surrogate ({len(tr)} train / {len(te)} held-out):")
    print(f"  held-out residual-stress rel-L2 {e_sig:.3f}; crystallinity rel-L2 {e_x:.3f}")

    # response surface over (cooling rate, melt temp) at mid crystallinity, from the net
    x_inf0 = 0.50
    rr = np.logspace(np.log10(0.5), np.log10(64), 40)
    tt = np.linspace(300, 360, 40)
    RR, TT = np.meshgrid(rr, tt)
    grid = np.stack([np.log10(RR).ravel(), TT.ravel(), np.full(RR.size, x_inf0)], 1)
    with torch.no_grad():
        Sg = (net(torch.tensor((grid - p_mu) / p_sd, dtype=torch.float32)).numpy()
              * y_sd + y_mu)[:, 0].reshape(RR.shape)
    # a few FE checks on the surface
    chk = [(1.0, 310, x_inf0), (8.0, 330, x_inf0), (40.0, 350, x_inf0)]
    chk_fe = [fe_residual(r, t, x) for r, t, x in chk]

    _plot(args.out, pred, Y, te, e_sig, e_x, rr, tt, RR, TT, Sg, chk, chk_fe)
    print(f"wrote {args.out}")


def _plot(out, pred, Y, te, e_sig, e_x, rr, tt, RR, TT, Sg, chk, chk_fe):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    lim = [Y[:, 0].min() * 0.9, Y[:, 0].max() * 1.05]
    ax[0, 0].plot(lim, lim, "k--", alpha=0.5)
    ax[0, 0].scatter(Y[te, 0], pred[te, 0], s=40, color="#d62728", alpha=0.75)
    ax[0, 0].set_xlabel("FE residual stress [MPa]"); ax[0, 0].set_ylabel("surrogate [MPa]")
    ax[0, 0].set_title(f"held-out residual stress: surrogate vs FE\nrel-L2 {e_sig:.3f}")
    ax[0, 0].grid(True, alpha=0.3)

    lim2 = [Y[:, 1].min() * 0.9, Y[:, 1].max() * 1.05]
    ax[0, 1].plot(lim2, lim2, "k--", alpha=0.5)
    ax[0, 1].scatter(Y[te, 1], pred[te, 1], s=40, color="#1f77b4", alpha=0.75)
    ax[0, 1].set_xlabel("FE crystallinity"); ax[0, 1].set_ylabel("surrogate crystallinity")
    ax[0, 1].set_title(f"held-out crystallinity: surrogate vs FE\nrel-L2 {e_x:.3f}")
    ax[0, 1].grid(True, alpha=0.3)

    cs = ax[1, 0].contourf(RR, TT, Sg, levels=18, cmap="viridis")
    fig.colorbar(cs, ax=ax[1, 0], label="residual stress [MPa]")
    for (r, t, x), v in zip(chk, chk_fe):
        ax[1, 0].plot(r, t, "o", ms=8, color="white", mec="k")
        ax[1, 0].annotate(f"FE {v[0]:.0f}", (r, t), textcoords="offset points",
                          xytext=(6, 4), fontsize=8, color="white")
    ax[1, 0].set_xscale("log")
    ax[1, 0].set_xlabel("cooling rate [°C/s]"); ax[1, 0].set_ylabel("melt temperature [°C]")
    ax[1, 0].set_title("surrogate response surface (X_inf=0.5)\n(white dots = FE checks)")

    # inverse-design read-off: residual stress vs cooling rate at 3 melt temps
    for t, c in [(305, "#2ca02c"), (330, "#b5651d"), (355, "#d62728")]:
        row = np.argmin(np.abs(tt - t))
        ax[1, 1].semilogx(rr, Sg[row], "-", color=c, label=f"T_melt={t}°C")
    ax[1, 1].set_xlabel("cooling rate [°C/s]"); ax[1, 1].set_ylabel("residual stress [MPa]")
    ax[1, 1].set_title("fast screening / inverse design:\nlower melt T & the right rate cut residual stress")
    ax[1, 1].legend(fontsize=8); ax[1, 1].grid(True, which="both", alpha=0.3)

    fig.suptitle("CFRTP process → residual-stress surrogate (ML = accelerator, FE = authority): "
                 "fast process-window screening / inverse design (Daikin theme)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main()
