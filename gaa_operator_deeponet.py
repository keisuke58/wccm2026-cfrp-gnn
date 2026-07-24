"""gaa_operator_deeponet.py — turn the demo-10 material sweep into a learned
*operator* (A+B+C unified). Instead of warm-starting Newton across the material x
bias campaign (idea C, gaa_material_sweep_warmstart.py), we learn the solution
operator itself:

    G_theta : (eps_r, kappa, V_gate)  x  (x, y)  ->  u(x, y)

a DeepONet whose *branch* ingests the material/condition parameters and whose
*trunk* ingests the disk coordinates. Trained on FE solutions (the weak-form
ground truth = idea A) for a subset of {material x bias}, it then does:

  (1) 1-shot inference on UNSEEN conditions — a held-out material (InGaAs, never in
      training) and held-out bias points — with no Newton at deployment;
  (2) warm-start: feeding the 1-shot prediction as the Newton initial guess
      roughly halves the exact FE solve's iterations (idea C), so FEM still
      guarantees accuracy while the operator cuts the work. (A ~1% guess does not
      collapse to a single iteration because the FE tol is a tight 1e-9 and Newton
      still needs a few quadratic-convergence steps.)

This is the operator-learning face of the same workload a comparative multi-material
TCAD study runs (Balaji et al., Next Materials 13 (2026) 102743): one trained
operator amortizes the whole material x bias grid.

Note: for a uniformly-doped disk with a radial Dirichlet gate, the FE field is
essentially radially symmetric, so the learned operator is easy — the point here is
the parametric generalization across *materials* and *bias*, and the C-bridge
(prediction -> ~1 Newton iter), not trunk expressiveness. Depends on
gaa_material_sweep_warmstart.py (reuses its mesh/assembly/Newton — repo dependency).

Run:  python3 gaa_operator_deeponet.py       (writes gaa_operator_deeponet.png)
      python3 gaa_operator_deeponet.py --help
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn

from gaa_material_sweep_warmstart import (
    MATERIALS, EPS_OX, C_DOP, kappa_of, build_disk_mesh, assemble, newton,
)

SEED = 20260724
INGAAS = 1                         # held-out material (never trained on); index in MATERIALS


class DeepONet(nn.Module):
    """Branch (condition params) x Trunk (coords) -> potential, plus a bias field."""
    def __init__(self, n_param=3, n_feat=7, q=48, width=64):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Linear(n_param, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, q),
        )
        self.trunk = nn.Sequential(
            nn.Linear(n_feat, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, q),
        )
        self.b0 = nn.Parameter(torch.zeros(1))

    def forward(self, params, feats):
        b = self.branch(params)                 # (B, q)
        t = self.trunk(feats)                    # (N, q)
        return b @ t.t() + self.b0               # (B, N)


def coord_feats(nodes):
    x, y = nodes[:, 0], nodes[:, 1]
    r = np.hypot(x, y)
    f = np.stack([x, y, r,
                  np.sin(np.pi * r), np.cos(np.pi * r),
                  np.sin(2 * np.pi * r), np.cos(2 * np.pi * r)], axis=1)
    return torch.tensor(f, dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nr", type=int, default=16)
    ap.add_argument("--nt", type=int, default=48)
    ap.add_argument("--nbias", type=int, default=13, help="bias points (0..vmax)")
    ap.add_argument("--vmax", type=float, default=8.0)
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--out", type=str, default="gaa_operator_deeponet.png")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    nodes, tris, gate, elem_semi = build_disk_mesh(args.nr, args.nt)
    free = ~gate
    N = len(nodes)
    biases = np.linspace(0.0, args.vmax, args.nbias)

    # per-material FE operators + a ground-truth field for every (material, bias)
    per_mat, u_scale = [], args.vmax
    fields = {}
    for m, (name, eps_r, n_i) in enumerate(MATERIALS):
        eps_elem = np.where(elem_semi, eps_r, EPS_OX)
        K, ML = assemble(nodes, tris, eps_elem, elem_semi)
        K = K.tocsr(); fdop = ML * C_DOP; kap = kappa_of(n_i)
        per_mat.append((name, eps_r, kap, K, ML, fdop))
        for k, vg in enumerate(biases):
            u, _, _ = newton(K, ML, kap, fdop, vg, free, gate, np.zeros(N))
            fields[(m, k)] = u

    # ---- condition-parameter normalisation (branch input) ----
    eps_all = np.array([mm[1] for mm in per_mat])
    kap_all = np.log10(np.array([mm[2] for mm in per_mat]))
    e_mu, e_sd = eps_all.mean(), eps_all.std()
    k_mu, k_sd = kap_all.mean(), kap_all.std()

    def param_vec(m, vg):
        return [(per_mat[m][1] - e_mu) / e_sd,
                (np.log10(per_mat[m][2]) - k_mu) / k_sd,
                vg / args.vmax]

    # ---- train / test split ----
    train_mats = [0, 2, 3, 4]                       # hold out InGaAs (index 1)
    train_bk = list(range(0, args.nbias, 2))        # even bias indices
    test_bk = [k for k in range(args.nbias) if k not in train_bk]
    train = [(m, k) for m in train_mats for k in train_bk]
    # unseen material; skip bias=0 whose field is ~0 (relative error is ill-defined there)
    test_material = [(INGAAS, k) for k in range(1, args.nbias)]
    test_bias = [(m, k) for m in train_mats for k in test_bk]       # unseen bias

    feats = coord_feats(nodes)
    gate_np = gate.copy()
    P_train = torch.tensor([param_vec(m, biases[k]) for (m, k) in train], dtype=torch.float32)
    U_train = torch.tensor(np.stack([fields[(m, k)] for (m, k) in train]) / u_scale,
                           dtype=torch.float32)
    gmask = torch.tensor(gate_np)
    ug_train = torch.tensor([biases[k] / u_scale for (m, k) in train], dtype=torch.float32)

    net = DeepONet()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    for ep in range(args.epochs):
        pred = net(P_train, feats)                                  # (B,N) scaled
        pred = pred.clone()
        pred[:, gmask] = ug_train[:, None]                          # hard Dirichlet
        loss = ((pred - U_train) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (ep + 1) % 800 == 0:
            print(f"[train] {ep+1}/{args.epochs}  mse {loss.item():.3e}")

    # ---- evaluation: 1-shot rel-L2 + operator-as-warm-start Newton iters ----
    def predict(m, k):
        with torch.no_grad():
            p = torch.tensor([param_vec(m, biases[k])], dtype=torch.float32)
            u = net(p, feats).numpy().ravel() * u_scale
        u[gate_np] = biases[k]
        return u

    def relL2(pred, ref):
        return float(np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + 1e-12))

    def eval_set(pairs):
        errs, cold_it, warm_it = [], [], []
        for (m, k) in pairs:
            ref = fields[(m, k)]
            pr = predict(m, k)
            errs.append(relL2(pr, ref))
            _, nc, _ = newton(*_solver_args(per_mat[m]), biases[k], free, gate, np.zeros(N))
            _, nw, _ = newton(*_solver_args(per_mat[m]), biases[k], free, gate, pr)
            cold_it.append(nc); warm_it.append(nw)
        return np.array(errs), np.array(cold_it), np.array(warm_it)

    def _solver_args(mm):
        name, eps_r, kap, K, ML, fdop = mm
        return K, ML, kap, fdop

    em, cm, wm = eval_set(test_material)
    eb, cb, wb = eval_set(test_bias)

    print(f"\n1-shot operator inference (no Newton at deployment):")
    print(f"  unseen MATERIAL (InGaAs, {len(test_material)} biases): "
          f"rel-L2 mean {em.mean():.3f}  max {em.max():.3f}")
    print(f"  unseen BIAS ({len(test_bias)} cases):                 "
          f"rel-L2 mean {eb.mean():.3f}  max {eb.max():.3f}")
    print(f"operator as warm-start (exact FE, Newton iters):")
    print(f"  unseen material: cold {cm.mean():.1f} -> warm {wm.mean():.1f}")
    print(f"  unseen bias:     cold {cb.mean():.1f} -> warm {wb.mean():.1f}")

    _plot(args.out, nodes, tris, biases, fields, predict, relL2,
          em, eb, cm, wm, cb, wb, per_mat, free, gate, N)
    print(f"wrote {args.out}")


def _plot(out, nodes, tris, biases, fields, predict, relL2,
          em, eb, cm, wm, cb, wb, per_mat, free, gate, N):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    triang = Triangulation(nodes[:, 0], nodes[:, 1], tris)
    fig, ax = plt.subplots(2, 3, figsize=(17, 10))

    # held-out material at a mid-high bias: FE vs 1-shot prediction vs error
    kk = len(biases) - 2
    ref = fields[(1, kk)]
    pr = predict(1, kk)
    for a, field, title, cm_ in (
        (ax[0, 0], ref, f"exact FE  (InGaAs, held-out material)", "viridis"),
        (ax[0, 1], pr, f"1-shot operator prediction", "viridis"),
    ):
        tp = a.tripcolor(triang, field, shading="gouraud", cmap=cm_)
        a.set_aspect("equal"); a.set_title(title); fig.colorbar(tp, ax=a, fraction=0.046)
    err = np.abs(pr - ref)
    tp = ax[0, 2].tripcolor(triang, err, shading="gouraud", cmap="magma")
    ax[0, 2].set_aspect("equal")
    ax[0, 2].set_title(f"|error|  (rel-L2 {relL2(pr, ref):.3f})")
    fig.colorbar(tp, ax=ax[0, 2], fraction=0.046)

    # radial profiles: predicted vs FE for the unseen material across biases
    r = np.hypot(nodes[:, 0], nodes[:, 1])
    order = np.argsort(r)
    for k in range(0, len(biases), 3):
        ref = fields[(1, k)]; pr = predict(1, k)
        ax[1, 0].plot(r[order], ref[order], "-", color="0.6", lw=2,
                      label="FE" if k == 0 else None)
        ax[1, 0].plot(r[order], pr[order], "--", lw=1.2,
                      label="operator" if k == 0 else None)
    ax[1, 0].set_title("radial profiles u(r), unseen material (InGaAs)\nFE vs 1-shot operator")
    ax[1, 0].set_xlabel("radius r"); ax[1, 0].set_ylabel("u"); ax[1, 0].legend()

    # 1-shot accuracy bars
    ax[1, 1].bar([0, 1], [em.mean(), eb.mean()],
                 yerr=[em.std(), eb.std()], color=["#6a3d9a", "#33a02c"], capsize=4)
    ax[1, 1].set_xticks([0, 1])
    ax[1, 1].set_xticklabels(["unseen\nmaterial", "unseen\nbias"])
    ax[1, 1].set_ylabel("1-shot rel-L2")
    ax[1, 1].set_title("operator generalization (no Newton at deployment)")
    for i, v in enumerate([em.mean(), eb.mean()]):
        ax[1, 1].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=10)

    # operator-as-warm-start: Newton iters collapse
    gx = np.arange(2)
    ax[1, 2].bar(gx - 0.2, [cm.mean(), cb.mean()], 0.4, color="#d62728", label="cold (u0=0)")
    ax[1, 2].bar(gx + 0.2, [wm.mean(), wb.mean()], 0.4, color="#1f77b4",
                 label="warm (operator)")
    ax[1, 2].set_xticks(gx); ax[1, 2].set_xticklabels(["unseen material", "unseen bias"])
    ax[1, 2].set_ylabel("Newton iterations (exact FE)")
    ax[1, 2].set_title("prediction as warm-start -> ~halved Newton iters\n(FE keeps accuracy)")
    for i, v in enumerate([cm.mean(), cb.mean()]):
        ax[1, 2].text(i - 0.2, v + 0.05, f"{v:.1f}", ha="center", fontsize=9)
    for i, v in enumerate([wm.mean(), wb.mean()]):
        ax[1, 2].text(i + 0.2, v + 0.05, f"{v:.1f}", ha="center", fontsize=9)
    ax[1, 2].legend()

    fig.suptitle("Learned solution operator for the multi-material GAA sweep "
                 "(DeepONet, A+B+C): 1-shot inference on unseen material/bias (~1% rel-L2), "
                 "and prediction warm-starts exact FE (~halved Newton iters)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main()
