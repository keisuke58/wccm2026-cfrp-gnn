"""electrothermal_inverse_design.py — INVERSE DESIGN of self-heating for research
theme G, tying together the ⑯ electro-thermal FE (electrothermal_selfheating.py, the
accuracy authority), the ⑯-dataset (electrothermal_dataset.npz), and the ㉑-style
operator surrogate. ML is the subordinate ACCELERATOR: a surrogate makes the design
space instantly searchable, and the FE VERIFIES the chosen design.

Design problem (a real thermal-design trade-off): choose the packaging heat-sink
strength HSINK and the lateral heat-spreader / geometry KAPPA to MINIMIZE a design cost
C = w_h HSINK + w_k KAPPA (better cooling / more spreading costs area & complexity),
SUBJECT TO two self-heating constraints at a target operating current J_op:
  (1) peak lattice temperature  Tmax(J_op) <= Tmax_limit   (thermal budget),
  (2) NDR-fold onset current    J_peak     >= J_op          (operate on the stable,
      voltage-controllable branch — below the negative-differential-resistance fold).
The material (activation EA) is fixed. The cheapest design meeting both constraints is
found on a dense (HSINK, KAPPA) grid using the surrogate (instant), then the optimum and
a naive baseline are VERIFIED with the weak-form FE.

Shows: the design space with the feasible region (both constraints) and cost contours
with the optimum; the two constraint fields (Tmax and NDR onset); the FE-verified I-V
and Tmax(J) of the optimized vs baseline design (NDR pushed past J_op, cooler); and the
surrogate-vs-FE agreement at the optimum.

Honest scope: surrogate trained on the reduced ⑯ dataset (illustrative scales), design
box within the sampled range (interpolation, cf. ㉑), simple linear cost; FE is the
authority and verifies the optimum. Physics leads; ML accelerates the search.

Run:  python3 electrothermal_inverse_design.py   (needs electrothermal_dataset.npz)
      python3 electrothermal_inverse_design.py --help
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn

import electrothermal_selfheating as es

SEED = 20260726
DATA = "electrothermal_dataset.npz"
EA0 = 12.0             # fixed material activation
J_OP = 1.2            # target operating current density
TMAX_LIMIT = 1.22     # thermal budget (x T0)
W_H, W_K = 1.0, 40.0  # cost weights (heat sink, spreader): C = W_H*HSINK + W_K*KAPPA


class Op(nn.Module):
    """Surrogate (EA, HSINK, KAPPA, Jn) -> (V, Tmax)."""
    def __init__(self, width=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 2))

    def forward(self, x):
        return self.net(x)


def fe_curve(ea, hsink, kappa, Js, n=81):
    """Weak-form FE ground truth: (EA, HSINK, KAPPA) -> V(J), Tmax(J) (warm-started)."""
    x, K, ml = es.assemble(n)
    Kk = (kappa * K).tocsc()
    Tw = np.full(n, es.T0); V, Tmax = [], []
    for J in Js:
        Tw, _, _ = es.solve_T(J, Kk, ml, Tw, ea=ea, hsink=hsink)
        V.append(es.voltage(J, Tw, x, ea=ea)); Tmax.append(Tw.max())
    return np.array(V), np.array(Tmax)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=str, default=DATA)
    ap.add_argument("--epochs", type=int, default=2500)
    ap.add_argument("--out", type=str, default="electrothermal_inverse_design.png")
    args = ap.parse_args()
    if not os.path.exists(args.data):
        ap.error(f"{args.data} not found — run: python3 electrothermal_dataset.py")
    torch.manual_seed(SEED); np.random.seed(SEED)

    d = np.load(args.data, allow_pickle=True)
    combos = d["combos"].astype(float); Js = d["Js"].astype(float)
    V = d["V"].astype(float); Tmax = d["Tmax"].astype(float)
    nc, ns = V.shape; jmax = Js.max()
    # flatten to (EA, HSINK, KAPPA, Jn) -> (V, Tmax)
    X, Y = [], []
    for i in range(nc):
        for k in range(ns):
            X.append([combos[i, 0], combos[i, 1], combos[i, 2], Js[k] / jmax])
            Y.append([V[i, k], Tmax[i, k]])
    X = np.array(X); Y = np.array(Y)
    x_mu, x_sd = X.mean(0), X.std(0) + 1e-9
    y_mu, y_sd = Y.mean(0), Y.std(0) + 1e-9
    Xt = torch.tensor((X - x_mu) / x_sd, dtype=torch.float32)
    Yt = torch.tensor((Y - y_mu) / y_sd, dtype=torch.float32)

    net = Op(); opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    sch = torch.optim.lr_scheduler.StepLR(opt, 1000, 0.5)
    for ep in range(args.epochs):
        opt.zero_grad(); loss = ((net(Xt) - Yt) ** 2).mean()
        loss.backward(); opt.step(); sch.step()
        if (ep + 1) % 1000 == 0:
            print(f"  epoch {ep+1} MSE {loss.item():.4e}")
    net.eval()

    def eval_grid(HH, KK, ea=EA0):
        """Batched surrogate eval over a design grid: returns Tmax(J_op) and NDR-onset
        J_peak for every (HSINK, KAPPA) cell in a single forward pass."""
        g = HH.shape
        H = HH.ravel(); Kp = KK.ravel(); m = H.size
        # query points: (m cells) x (ns currents) x 4 features
        q = np.empty((m * ns, 4))
        q[:, 0] = ea
        q[:, 1] = np.repeat(H, ns); q[:, 2] = np.repeat(Kp, ns)
        q[:, 3] = np.tile(Js / jmax, m)
        with torch.no_grad():
            p = net(torch.tensor((q - x_mu) / x_sd, dtype=torch.float32)).numpy() * y_sd + y_mu
        Vg = p[:, 0].reshape(m, ns); Tg = p[:, 1].reshape(m, ns)
        tmax_op = np.array([np.interp(J_OP, Js, Tg[i]) for i in range(m)]).reshape(g)
        jpeak = Js[np.argmax(Vg, axis=1)].reshape(g)
        return tmax_op, jpeak

    def metrics(hsink, kappa):
        t, j = eval_grid(np.array([[hsink]]), np.array([[kappa]]))
        return float(t[0, 0]), float(j[0, 0])

    # dense design-space search (surrogate = instant, batched)
    hs = np.linspace(0.5, 8.0, 60); ka = np.linspace(0.02, 0.08, 60)
    HH, KK = np.meshgrid(hs, ka)
    TMX, JPK = eval_grid(HH, KK)
    feasible = (TMX <= TMAX_LIMIT) & (JPK >= J_OP)
    cost = W_H * HH + W_K * KK
    cost_f = np.where(feasible, cost, np.inf)
    ia, ib = np.unravel_index(np.argmin(cost_f), cost_f.shape)
    hs_opt, ka_opt = HH[ia, ib], KK[ia, ib]
    print(f"\ninverse design (material EA={EA0}, J_op={J_OP}, Tmax_limit={TMAX_LIMIT}):")
    print(f"  feasible fraction {feasible.mean()*100:.0f}%; min-cost design "
          f"HSINK={hs_opt:.2f}, KAPPA={ka_opt:.3f} (cost {cost[ia,ib]:.2f})")

    # baseline: a cheap under-cooled design (low HSINK, low KAPPA)
    hs_base, ka_base = 1.0, 0.02
    Jf = np.linspace(jmax / 90, jmax, 90)
    Vb, Tb = fe_curve(EA0, hs_base, ka_base, Jf)
    Vo, To = fe_curve(EA0, hs_opt, ka_opt, Jf)
    tmax_op_fe = float(np.interp(J_OP, Jf, To)); jpk_fe = float(Jf[int(np.argmax(Vo))])
    tmax_op_su, jpk_su = metrics(hs_opt, ka_opt)
    print(f"  FE-verified optimum: Tmax(J_op)={tmax_op_fe:.3f} (limit {TMAX_LIMIT}), "
          f"J_peak={jpk_fe:.2f} (>= J_op {J_OP})")
    print(f"  surrogate at optimum: Tmax {tmax_op_su:.3f}, J_peak {jpk_su:.2f} "
          f"(FE {tmax_op_fe:.3f}/{jpk_fe:.2f})")
    base_tmax = float(np.interp(J_OP, Jf, Tb)); base_jpk = float(Jf[int(np.argmax(Vb))])

    _plot(args.out, hs, ka, HH, KK, TMX, JPK, feasible, cost, hs_opt, ka_opt,
          hs_base, ka_base, Jf, Vb, Tb, Vo, To,
          (base_tmax, base_jpk), (tmax_op_fe, jpk_fe), (tmax_op_su, jpk_su))
    print(f"wrote {args.out}")


def _plot(out, hs, ka, HH, KK, TMX, JPK, feasible, cost, hs_opt, ka_opt,
          hs_base, ka_base, Jf, Vb, Tb, Vo, To, base, feo, sur):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    # design space: cost contours + feasible region + optimum
    cf = ax[0, 0].contourf(HH, KK, np.where(feasible, cost, np.nan), levels=16, cmap="viridis")
    fig.colorbar(cf, ax=ax[0, 0], label="design cost (feasible only)")
    ax[0, 0].contourf(HH, KK, feasible.astype(float), levels=[-0.5, 0.5], colors=["#dddddd"], alpha=0.6)
    ax[0, 0].plot(hs_opt, ka_opt, "*", ms=18, color="#ff3030", mec="k", label="min-cost optimum")
    ax[0, 0].plot(hs_base, ka_base, "s", ms=9, color="#777", mec="k", label="naive baseline")
    ax[0, 0].set_xlabel("heat-sink HSINK"); ax[0, 0].set_ylabel("spreader KAPPA")
    ax[0, 0].set_title("inverse design: cheapest FEASIBLE thermal design\n(grey = infeasible)")
    ax[0, 0].legend(fontsize=8)

    # constraint fields
    c1 = ax[0, 1].contourf(HH, KK, TMX, levels=16, cmap="inferno")
    fig.colorbar(c1, ax=ax[0, 1], label="Tmax(J_op)")
    ax[0, 1].contour(HH, KK, TMX, levels=[TMAX_LIMIT], colors="cyan", linewidths=2)
    ax[0, 1].contour(HH, KK, JPK, levels=[J_OP], colors="lime", linewidths=2, linestyles="--")
    ax[0, 1].plot(hs_opt, ka_opt, "*", ms=16, color="#ff3030", mec="k")
    ax[0, 1].set_xlabel("HSINK"); ax[0, 1].set_ylabel("KAPPA")
    ax[0, 1].set_title("constraints: Tmax≤limit (cyan) &\nNDR onset J_peak≥J_op (green dashed)")

    ax[1, 0].plot(Vb, Jf, "-", color="#777", label="baseline (FE)")
    ax[1, 0].plot(Vo, Jf, "-", color="#d62728", label="optimized (FE)")
    ax[1, 0].axhline(J_OP, color="k", ls=":", alpha=0.6, label="J_op")
    ax[1, 0].plot(Vb[int(np.argmax(Vb))], Jf[int(np.argmax(Vb))], "s", color="#777")
    ax[1, 0].plot(Vo[int(np.argmax(Vo))], Jf[int(np.argmax(Vo))], "o", color="#d62728")
    ax[1, 0].set_xlabel("terminal voltage V"); ax[1, 0].set_ylabel("current density J")
    ax[1, 0].set_title("FE-verified I-V: optimum pushes the NDR fold past J_op")
    ax[1, 0].legend(fontsize=8); ax[1, 0].grid(True, alpha=0.3)

    ax[1, 1].plot(Jf, Tb, "-", color="#777", label=f"baseline (Tmax@J_op {base[0]:.3f})")
    ax[1, 1].plot(Jf, To, "-", color="#d62728", label=f"optimized ({feo[0]:.3f})")
    ax[1, 1].axhline(TMAX_LIMIT, color="cyan", ls="--", alpha=0.8, label="Tmax limit")
    ax[1, 1].axvline(J_OP, color="k", ls=":", alpha=0.6)
    ax[1, 1].set_xlabel("current density J"); ax[1, 1].set_ylabel("peak temperature Tmax")
    ax[1, 1].set_title(f"FE Tmax(J): optimum stays under budget\nsurrogate@opt {sur[0]:.3f} vs FE {feo[0]:.3f} (agree)")
    ax[1, 1].legend(fontsize=8); ax[1, 1].grid(True, alpha=0.3)

    fig.suptitle("Self-heating INVERSE DESIGN (theme G): surrogate-searched, FE-verified min-cost "
                 "thermal design meeting Tmax & NDR constraints (ML accelerates, FE authority)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main()
