"""cfrtp_deeponet_field.py -- B (field prediction): a DeepONet operator surrogate
that maps PROCESS -> through-thickness residual-stress FIELD sigma(z).

Physics-first, ML-subordinate (and on-brand for this branch, "deeponet-physics-
informed"). The accuracy AUTHORITY is the FE-verified 0D crystallization + VE law
(design/cfrtp_inverse_design.process), applied per depth with a conduction-lag
cooling profile: the surface cools at the imposed rate R, the core lags (thicker
plate -> more lag), so crystallinity and locked-in residual stress vary through the
thickness. At high R / thick plates the surface quenches (low alpha, low residual)
while the core fully crystallizes (higher residual) -- a real gradient.

The DeepONet learns the operator (R, L) -> sigma(zeta), zeta = z/L in [0,1]:
  branch net : process params [log10 R, L]  -> p-dim
  trunk  net : depth zeta                    -> p-dim
  output     : sum_k branch_k * trunk_k + b0
Trained on FE samples, tested on unseen (R, L); every prediction is checkable
against the physics (FE-verified). numpy + torch + matplotlib. Magnitudes
illustrative (uncalibrated).

    python3 design/cfrtp_deeponet_field.py
"""
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cfrtp_inverse_design import process     # FE authority (crystallization + VE)

torch.manual_seed(0); np.random.seed(0)
R_LO, R_HI = 100.0, 1500.0        # cooling rate [C/min]
L_LO, L_HI = 1.0, 8.0             # plate thickness [mm]
NZ = 21


def core_lag(L_mm):
    return min(0.85, 0.10 + 0.11 * L_mm)     # thicker -> core lags more


def field_fe(R, L_mm, zetas):
    """Through-thickness residual |sigma_11|(zeta) [MPa] from the per-depth physics."""
    lam = core_lag(L_mm)
    out = np.empty_like(zetas)
    for i, z in enumerate(zetas):
        r = R * (1.0 - lam * 4.0 * z * (1.0 - z))        # surface=R, core slower
        out[i], _ = process(max(r, 5.0))
    return out


def make_dataset(n, zetas):
    R = 10 ** np.random.uniform(np.log10(R_LO), np.log10(R_HI), n)
    L = np.random.uniform(L_LO, L_HI, n)
    Y = np.stack([field_fe(R[i], L[i], zetas) for i in range(n)])   # (n, NZ)
    Xb = np.stack([(np.log10(R) - np.log10(R_LO)) / (np.log10(R_HI) - np.log10(R_LO)),
                   (L - L_LO) / (L_HI - L_LO)], axis=1)             # (n, 2) normalized
    return Xb.astype(np.float32), Y.astype(np.float32), R, L


class MLP(nn.Module):
    def __init__(self, din, dout, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, h), nn.Tanh(), nn.Linear(h, h),
                                 nn.Tanh(), nn.Linear(h, dout))

    def forward(self, x):
        return self.net(x)


class DeepONet(nn.Module):
    def __init__(self, p=32):
        super().__init__()
        self.branch = MLP(2, p); self.trunk = MLP(1, p)
        self.b0 = nn.Parameter(torch.zeros(1))

    def forward(self, xb, zt):                # xb:(B,2)  zt:(NZ,1)
        b = self.branch(xb)                   # (B,p)
        t = self.trunk(zt)                    # (NZ,p)
        return b @ t.T + self.b0              # (B,NZ)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    zetas = np.linspace(0, 1, NZ)
    Xtr, Ytr, Rtr, Ltr = make_dataset(180, zetas)
    Xte, Yte, Rte, Lte = make_dataset(40, zetas)
    ymu, ysd = Ytr.mean(), Ytr.std() + 1e-9

    dev = "cpu"
    xb = torch.tensor(Xtr, device=dev); yb = torch.tensor((Ytr - ymu) / ysd, device=dev)
    zt = torch.tensor(zetas.reshape(-1, 1).astype(np.float32), device=dev)
    model = DeepONet(p=32).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    lossf = nn.MSELoss()
    for ep in range(4000):
        opt.zero_grad()
        loss = lossf(model(xb, zt), yb)
        loss.backward(); opt.step()
        if (ep + 1) % 1000 == 0:
            print("  epoch %4d  train MSE(std)=%.4e" % (ep + 1, loss.item()))

    # test
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(Xte, device=dev), zt).cpu().numpy() * ysd + ymu
    relL2 = np.linalg.norm(pred - Yte) / np.linalg.norm(Yte)
    per = np.linalg.norm(pred - Yte, axis=1) / (np.linalg.norm(Yte, axis=1) + 1e-9)
    print("DeepONet field prediction  test rel-L2 = %.3f  (median per-case %.3f, worst %.3f)"
          % (relL2, np.median(per), per.max()))
    print("  (operator: process (R,L) -> residual sigma(z) through-thickness; FE-authority)")

    # ---- figure ----
    TENS, HLC, INK, GRID = "#e23b48", "#7b5cff", "#141922", "#dfe4ea"
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GRID})
    fig, (axp, axh) = plt.subplots(1, 2, figsize=(11, 4.4), dpi=120)
    fig.suptitle("Field prediction: DeepONet operator  (R, L) → residual σ₁₁(z)  "
                 "[test rel-L2 = %.3f]" % relL2, fontweight="bold")
    # (a) a few test profiles: FE vs DeepONet
    order = np.argsort(Rte)
    pick = [order[2], order[len(order)//2], order[-3]]
    cols = plt.cm.viridis(np.linspace(0.1, 0.85, len(pick)))
    for c, k in zip(cols, pick):
        axp.plot(zetas, Yte[k], "o", color=c, ms=5)
        axp.plot(zetas, pred[k], "-", color=c, lw=2,
                 label="R=%.0f C/min, L=%.1fmm" % (Rte[k], Lte[k]))
    axp.set_xlabel("normalized depth ζ = z/L  (0,1 = surfaces)")
    axp.set_ylabel("residual |σ₁₁| [MPa]*")
    axp.set_title("(a) FE (○) vs DeepONet (—)", fontsize=10)
    axp.grid(True, color=GRID, lw=.7); axp.legend(fontsize=8)
    # (b) predicted field heatmap over (R, zeta) at fixed L=6mm
    Rs = np.logspace(np.log10(R_LO), np.log10(R_HI), 60); Lfix = 6.0
    Xg = np.stack([(np.log10(Rs)-np.log10(R_LO))/(np.log10(R_HI)-np.log10(R_LO)),
                   np.full_like(Rs, (Lfix-L_LO)/(L_HI-L_LO))], axis=1).astype(np.float32)
    with torch.no_grad():
        F = model(torch.tensor(Xg), zt).cpu().numpy() * ysd + ymu    # (60, NZ)
    im = axh.pcolormesh(zetas, Rs, F, shading="auto", cmap="magma")
    axh.set_yscale("log"); fig.colorbar(im, ax=axh, label="residual |σ₁₁| [MPa]*")
    axh.set_xlabel("normalized depth ζ"); axh.set_ylabel("cooling rate R [°C/min, log]")
    axh.set_title("(b) predicted field, L=6mm (surface quench, core locks in)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(here, "cfrtp_deeponet_field.png"); fig.savefig(out, dpi=140); plt.close(fig)
    print("wrote %s" % out)
    return dict(relL2=float(relL2))


if __name__ == "__main__":
    main()
