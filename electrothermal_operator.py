"""electrothermal_operator.py — operator-learning surrogate for research theme G,
trained on the SELF-GENERATED design-space dataset from ⑯-dataset
(electrothermal_dataset.py -> electrothermal_dataset.npz). ML is the subordinate
ACCELERATOR here: the weak-form FE self-consistent solve (⑯) is the accuracy
authority and generated the data; this net just screens the design space fast.

Task (a DeepONet, same branch/trunk pattern as ⑪ gaa_operator_deeponet.py):
    branch : design params (EA, HSINK, KAPPA)          -> q latent
    trunk  : query current density J                    -> q latent
    output : <branch, trunk> -> (V, Tmax) at that (design, J)
So one forward pass predicts a whole self-heating I-V curve AND peak-temperature
curve for an UNSEEN (EA,HSINK,KAPPA) design — no FE continuation sweep needed. The
NDR-fold onset (argmax_J V) is read off the predicted curve, i.e. the regime label
comes for free.

Split: of the 80 design points, a random subset is held out entirely (unseen
designs). Param and target normalization use TRAIN designs only (no leakage of the
held-out designs into preprocessing). Reported: relative-L2 of the predicted I-V and
Tmax curves on the held-out designs, and predicted-vs-true NDR-onset current.

Honest scope: the design space is smooth and low-dimensional (3 params on an
8x5x2 grid), so held-out points are interpolations — the surrogate's job is fast
screening / inverse design, not extrapolation, and FE remains the authority. Scaled
units inherited from ⑯. This is the "surrogate trained on the ⑯ dataset" step; the
physics (⑯–⑳) is the substance.

Run:  python3 electrothermal_operator.py            (needs electrothermal_dataset.npz)
      python3 electrothermal_operator.py --help
If the dataset is missing:  python3 electrothermal_dataset.py
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn

SEED = 20260726
DATA = "electrothermal_dataset.npz"


class DeepONet2(nn.Module):
    """Branch (design params) x Trunk (query current) -> 2 outputs (V, Tmax)."""

    def __init__(self, n_param=3, n_feat=5, q=48, width=64, n_out=2):
        super().__init__()
        self.q = q; self.n_out = n_out
        self.branch = nn.Sequential(
            nn.Linear(n_param, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, n_out * q),
        )
        self.trunk = nn.Sequential(
            nn.Linear(n_feat, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, q),
        )
        self.b0 = nn.Parameter(torch.zeros(n_out))

    def forward(self, params, feats):
        b = self.branch(params).view(-1, self.n_out, self.q)   # (B, C, q)
        t = self.trunk(feats)                                  # (M, q)
        out = torch.einsum("bcq,mq->bmc", b, t) + self.b0      # (B, M, C)
        return out


def jfeats(Jn):
    """Trunk features for a (normalized) query current."""
    Jn = np.asarray(Jn, dtype=float)
    f = np.stack([Jn, Jn ** 2, np.sqrt(np.clip(Jn, 0, None)),
                  np.sin(np.pi * Jn), np.cos(np.pi * Jn)], axis=1)
    return torch.tensor(f, dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=str, default=DATA)
    ap.add_argument("--ntest", type=int, default=24, help="held-out (unseen) designs")
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--out", type=str, default="electrothermal_operator.png")
    args = ap.parse_args()

    if not os.path.exists(args.data):
        ap.error(f"{args.data} not found — generate it first: python3 electrothermal_dataset.py")
    torch.manual_seed(SEED); np.random.seed(SEED)

    d = np.load(args.data, allow_pickle=True)
    combos = d["combos"].astype(float)            # (ncombo, 3) EA,HSINK,KAPPA
    Js = d["Js"].astype(float)                    # (nstep,)
    V = d["V"].astype(float)                      # (ncombo, nstep)
    Tmax = d["Tmax"].astype(float)                # (ncombo, nstep)
    nc, ns = V.shape
    jmax = Js.max(); Jn = Js / jmax

    # train / test split over DESIGNS (whole I-V curves held out)
    perm = np.random.permutation(nc)
    test_idx = np.sort(perm[:args.ntest]); train_idx = np.sort(perm[args.ntest:])

    # normalization from TRAIN designs only (no leakage)
    p_mu = combos[train_idx].mean(0); p_sd = combos[train_idx].std(0) + 1e-9
    Pn = (combos - p_mu) / p_sd
    v_mu = V[train_idx].mean(); v_sd = V[train_idx].std() + 1e-9
    t_mu = Tmax[train_idx].mean(); t_sd = Tmax[train_idx].std() + 1e-9
    Vn = (V - v_mu) / v_sd; Tn = (Tmax - t_mu) / t_sd

    feats = jfeats(Jn)                                            # (ns, 5)
    P = torch.tensor(Pn, dtype=torch.float32)
    Y = torch.stack([torch.tensor(Vn, dtype=torch.float32),
                     torch.tensor(Tn, dtype=torch.float32)], dim=-1)   # (nc, ns, 2)
    tr = torch.tensor(train_idx)

    net = DeepONet2()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1200, gamma=0.5)
    for ep in range(args.epochs):
        opt.zero_grad()
        pred = net(P[tr], feats)                                  # (ntr, ns, 2)
        loss = ((pred - Y[tr]) ** 2).mean()
        loss.backward(); opt.step(); sched.step()
        if (ep + 1) % 600 == 0:
            print(f"  epoch {ep+1:4d}  train MSE {loss.item():.4e}")

    net.eval()
    with torch.no_grad():
        pred_all = net(P, feats).numpy()                         # (nc, ns, 2)
    V_pred = pred_all[..., 0] * v_sd + v_mu
    T_pred = pred_all[..., 1] * t_sd + t_mu

    def rel_l2(a, b):
        return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))

    v_err = np.array([rel_l2(V_pred[i], V[i]) for i in test_idx])
    t_err = np.array([rel_l2(T_pred[i], Tmax[i]) for i in test_idx])
    # NDR onset (argmax V) predicted vs true, on held-out designs
    ndr_true = Js[np.argmax(V[test_idx], axis=1)]
    ndr_pred = Js[np.argmax(V_pred[test_idx], axis=1)]
    ndr_mae = float(np.mean(np.abs(ndr_pred - ndr_true)))

    print(f"\nelectro-thermal operator surrogate (DeepONet on ⑯-dataset)")
    print(f"  designs: {len(train_idx)} train / {len(test_idx)} held-out (unseen), "
          f"{ns} currents each")
    print(f"  held-out I-V   relative-L2: mean {v_err.mean():.3f}  median {np.median(v_err):.3f}")
    print(f"  held-out Tmax  relative-L2: mean {t_err.mean():.3f}  median {np.median(t_err):.3f}")
    print(f"  NDR-onset current (argmax V) MAE on held-out: {ndr_mae:.3f} "
          f"(current grid step {Js[1]-Js[0]:.3f})")
    print(f"  -> 1 forward pass predicts a full I-V + Tmax curve for an unseen design "
          f"(no FE sweep); FE stays the accuracy authority")

    _plot(args.out, Js, V, V_pred, Tmax, T_pred, test_idx, v_err, t_err,
          ndr_true, ndr_pred)
    print(f"wrote {args.out}")


def _plot(out, Js, V, V_pred, Tmax, T_pred, test_idx, v_err, t_err, ndr_true, ndr_pred):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    # a few held-out I-V curves: true vs predicted
    order = test_idx[np.argsort(v_err)]
    show = order[np.linspace(0, len(order) - 1, 4).astype(int)]
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]
    for c, i in zip(colors, show):
        ax[0, 0].plot(V[i], Js, "-", color=c, alpha=0.9, label=f"design {i} (true)")
        ax[0, 0].plot(V_pred[i], Js, "--", color=c, alpha=0.9)
    ax[0, 0].set_xlabel("terminal voltage V"); ax[0, 0].set_ylabel("current density J")
    ax[0, 0].set_title("held-out (unseen) designs: I-V curve\nsolid = FE truth, dashed = 1-shot surrogate")
    ax[0, 0].legend(fontsize=7); ax[0, 0].grid(True, alpha=0.3)

    # Tmax curves true vs predicted
    for c, i in zip(colors, show):
        ax[0, 1].plot(Js, Tmax[i], "-", color=c, alpha=0.9)
        ax[0, 1].plot(Js, T_pred[i], "--", color=c, alpha=0.9)
    ax[0, 1].set_xlabel("current density J"); ax[0, 1].set_ylabel("peak temperature Tmax")
    ax[0, 1].set_title("held-out designs: peak temperature curve\nsolid = FE truth, dashed = surrogate")
    ax[0, 1].grid(True, alpha=0.3)

    # relative-L2 per held-out design
    xs = np.arange(len(test_idx))
    ax[1, 0].bar(xs - 0.2, np.sort(v_err), width=0.4, color="#1f77b4", label="I-V")
    ax[1, 0].bar(xs + 0.2, t_err[np.argsort(v_err)], width=0.4, color="#b5651d", label="Tmax")
    ax[1, 0].axhline(v_err.mean(), color="#1f77b4", ls="--", alpha=0.6)
    ax[1, 0].set_xlabel("held-out design (sorted)"); ax[1, 0].set_ylabel("relative L2 error")
    ax[1, 0].set_title(f"1-shot surrogate accuracy on unseen designs\n"
                       f"I-V mean {v_err.mean():.3f}, Tmax mean {t_err.mean():.3f}")
    ax[1, 0].legend(); ax[1, 0].grid(True, axis="y", alpha=0.3)

    # NDR-onset parity
    lim = [min(ndr_true.min(), ndr_pred.min()) - 0.1, max(ndr_true.max(), ndr_pred.max()) + 0.1]
    ax[1, 1].plot(lim, lim, "k--", alpha=0.5)
    ax[1, 1].scatter(ndr_true, ndr_pred, s=40, color="#d62728", alpha=0.7)
    ax[1, 1].set_xlim(lim); ax[1, 1].set_ylim(lim)
    ax[1, 1].set_xlabel("true NDR-onset current (argmax V)")
    ax[1, 1].set_ylabel("predicted NDR-onset current")
    ax[1, 1].set_title("NDR-fold onset: surrogate vs FE (held-out designs)")
    ax[1, 1].grid(True, alpha=0.3)

    fig.suptitle("Operator-learning surrogate on the self-generated electro-thermal "
                 "dataset (demo 16): 1-shot I-V / NDR prediction for unseen designs "
                 "(ML = accelerator, FE = authority)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main()
