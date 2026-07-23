"""tsv_thermal_stress.py — TSV thermo-mechanical stress: FE thermoelasticity + a
learned stress-field surrogate + keep-out-zone (KOZ) extraction.

This is the repo-native semiconductor topic: through-silicon vias (TSVs) are the 3D-
integration / advanced-packaging layer, and their dominant simulation concern is
*thermo-mechanical stress*, not drift-diffusion. Cooling from processing temperature,
the CTE mismatch between the Cu via (alpha ~ 17 ppm/K) and the Si matrix (~2.6 ppm/K)
puts the surrounding silicon under stress; transistors placed in the high-stress ring
shift in performance, defining a keep-out zone. This is exactly a stress-field FEA
problem — the same machinery as the repo's CFRP stress x GNN line.

Method:
  * FE oracle: 2-D plane-strain linear thermoelasticity on a structured triangular
    mesh (constant-strain triangles). Cu vias in an Si matrix, uniform cooling dT.
    Assemble K u = f_thermal, solve, recover per-node von Mises stress.
  * Surrogate: a CNN maps the via-layout image (Cu indicator + coordinates) to the
    von Mises field, trained on FE solves across random via layouts and tested on
    held-out layouts. A learned stress operator = instant KOZ screening across many
    floorplans without re-meshing / re-solving.
  * KOZ: threshold the von Mises field; compare surrogate KOZ vs FE KOZ (IoU).

Concept demo (2-D, isotropic Si, linear thermoelastic, structured mesh): not a
device-accurate TCAD/packaging run. It shows a stress-field operator surrogate that
reproduces the FE keep-out zone.

Run:  python3 tsv_thermal_stress.py            (writes tsv_thermal_stress.png)
      python3 tsv_thermal_stress.py --help
"""
from __future__ import annotations

import argparse

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import torch.nn as nn

from pi_deeponet_fem_gaa import build_mesh          # reuse structured triangulation

SEED = 20260722
DT = -250.0                                          # cooling from processing (K)
# scaled-but-physical isotropic material constants (E in GPa, alpha in 1e-6/K)
E_CU, NU_CU, A_CU = 110.0, 0.34, 17.0e-6
E_SI, NU_SI, A_SI = 130.0, 0.28, 2.6e-6
KOZ_FRAC = 0.5                                        # KOZ threshold = frac * max FE stress


# ----------------------------------------------------------------------------
# Via layout -> per-element material
# ----------------------------------------------------------------------------
def rand_vias(rng):
    k = rng.integers(1, 5)
    cx = rng.uniform(0.28, 0.72, size=k)
    cy = rng.uniform(0.28, 0.72, size=k)
    r = rng.uniform(0.07, 0.13, size=k)
    return list(zip(cx, cy, r))


def cu_mask(points, vias):
    """Boolean: is each point inside any Cu via?"""
    m = np.zeros(len(points), bool)
    for cx, cy, r in vias:
        m |= (points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2 <= r ** 2
    return m


def d_matrix(E, nu):
    """Plane-strain elasticity matrix (3x3)."""
    c = E / ((1 + nu) * (1 - 2 * nu))
    return c * np.array([[1 - nu, nu, 0.0],
                         [nu, 1 - nu, 0.0],
                         [0.0, 0.0, (1 - 2 * nu) / 2]])


# ----------------------------------------------------------------------------
# FE thermoelastic solve (CST, plane strain) -> nodal von Mises
# ----------------------------------------------------------------------------
def fe_thermal(nodes, tris, on_bnd, vias):
    nN = len(nodes)
    ndof = 2 * nN
    elem_cu = cu_mask(nodes[tris].mean(axis=1), vias)          # material per element
    rows, cols, vals = [], [], []
    F = np.zeros(ndof)
    Bs, Ds = [], []
    for e, tri in enumerate(tris):
        (x1, y1), (x2, y2), (x3, y3) = nodes[tri]
        det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        A = 0.5 * abs(det)
        if A < 1e-14:
            Bs.append(None); Ds.append(None); continue
        b = np.array([y2 - y3, y3 - y1, y1 - y2]) / det
        c = np.array([x3 - x2, x1 - x3, x2 - x1]) / det
        B = np.zeros((3, 6))
        B[0, 0::2] = b; B[1, 1::2] = c; B[2, 0::2] = c; B[2, 1::2] = b
        if elem_cu[e]:
            D = d_matrix(E_CU, NU_CU); alpha = A_CU
        else:
            D = d_matrix(E_SI, NU_SI); alpha = A_SI
        Ke = A * (B.T @ D @ B)
        eps_th = alpha * DT * np.array([1.0, 1.0, 0.0])        # plane-strain thermal strain
        fe = A * (B.T @ (D @ eps_th))
        dofs = np.empty(6, dtype=np.int64)
        dofs[0::2] = 2 * tri; dofs[1::2] = 2 * tri + 1
        for a_ in range(6):
            F[dofs[a_]] += fe[a_]
            for b_ in range(6):
                rows.append(dofs[a_]); cols.append(dofs[b_]); vals.append(Ke[a_, b_])
        Bs.append((B, dofs)); Ds.append((D, eps_th))
    K = sp.csr_matrix((vals, (rows, cols)), shape=(ndof, ndof))

    # Dirichlet: clamp the outer boundary (u = 0) — via array in a constrained wafer
    fixed = np.zeros(ndof, bool)
    fixed[2 * np.where(on_bnd)[0]] = True
    fixed[2 * np.where(on_bnd)[0] + 1] = True
    free = ~fixed
    u = np.zeros(ndof)
    u[free] = spla.spsolve(K[free][:, free].tocsc(), F[free])

    # per-element stress -> von Mises, averaged to nodes
    vm_node = np.zeros(nN); cnt = np.zeros(nN)
    for e, tri in enumerate(tris):
        if Bs[e] is None:
            continue
        B, dofs = Bs[e]; D, eps_th = Ds[e]
        strain = B @ u[dofs]
        sig = D @ (strain - eps_th)                            # [sxx, syy, sxy]
        vm = np.sqrt(sig[0] ** 2 - sig[0] * sig[1] + sig[1] ** 2 + 3 * sig[2] ** 2)
        vm_node[tri] += vm; cnt[tri] += 1
    return vm_node / np.clip(cnt, 1, None)


# ----------------------------------------------------------------------------
# Surrogate: CNN  (Cu map, x, y) -> von Mises field
# ----------------------------------------------------------------------------
class StressCNN(nn.Module):
    def __init__(self, ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, ch, 3, padding=1), nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.SiLU(),
            nn.Conv2d(ch, 1, 3, padding=1),
        )

    def forward(self, x):
        return torch.relu(self.net(x).squeeze(1))              # stress >= 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=41, help="nodes per side")
    ap.add_argument("--n_train", type=int, default=48, help="training layouts")
    ap.add_argument("--n_test", type=int, default=12, help="held-out test layouts")
    ap.add_argument("--epochs", type=int, default=400, help="training epochs")
    ap.add_argument("--out", type=str, default="tsv_thermal_stress.png")
    args = ap.parse_args()
    if args.n_train < 1 or args.n_test < 1:
        ap.error("--n_train and --n_test must be >= 1")

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    n = args.n
    nodes, tris, on_bnd = build_mesh(n)
    coords = nodes.reshape(n, n, 2)
    xg = torch.tensor(coords[:, :, 0], dtype=torch.float32)
    yg = torch.tensor(coords[:, :, 1], dtype=torch.float32)

    def make_case(vias):
        vm = fe_thermal(nodes, tris, on_bnd, vias)
        cu = cu_mask(nodes, vias).reshape(n, n).astype(np.float32)
        inp = torch.stack([torch.tensor(cu), xg, yg], dim=0).unsqueeze(0)   # (1,3,n,n)
        return inp, torch.tensor(vm.reshape(n, n), dtype=torch.float32), vm

    print(f"[data] generating {args.n_train}+{args.n_test} FE thermoelastic solves ...")
    train = [make_case(rand_vias(rng)) for _ in range(args.n_train)]
    test = [make_case(rand_vias(rng)) for _ in range(args.n_test)]
    s_scale = float(torch.stack([t[1] for t in train]).abs().mean().clamp_min(1e-6))

    net = StressCNN()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    for ep in range(args.epochs):
        inp, tgt, _ = train[rng.integers(len(train))]
        pred = net(inp)
        loss = ((pred - tgt / s_scale) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 100 == 0:
            print(f"[train] {ep+1}/{args.epochs}  loss {loss.item():.3e}")

    # ---- evaluate on held-out layouts ----
    def koz(vm_flat, thr):
        return vm_flat > thr

    rels, ious = [], []
    for inp, tgt, vm in test:
        with torch.no_grad():
            pred = net(inp).numpy().reshape(-1) * s_scale
        rel = np.linalg.norm(pred - vm) / (np.linalg.norm(vm) + 1e-9)
        thr = KOZ_FRAC * vm.max()
        kf, kp = koz(vm, thr), koz(pred, thr)
        iou = (kf & kp).sum() / max((kf | kp).sum(), 1)
        rels.append(rel); ious.append(iou)
    print(f"\nheld-out: mean rel-L2 {np.mean(rels):.3f}   mean KOZ IoU {np.mean(ious):.3f}"
          f"   ({args.n_test} unseen layouts)")

    # representative held-out case for the figure
    inp, tgt, vm = test[0]
    with torch.no_grad():
        pred = net(inp).numpy().reshape(-1) * s_scale
    _plot(args.out, nodes, tris, inp[0, 0].numpy().reshape(-1), vm, pred, rels, ious)
    print(f"wrote {args.out}")


def _plot(out, nodes, tris, cu_flat, vm, pred, rels, ious):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    tri = Triangulation(nodes[:, 0], nodes[:, 1], tris)
    thr = KOZ_FRAC * vm.max()
    fig, ax = plt.subplots(2, 3, figsize=(16, 10))

    tp = ax[0, 0].tripcolor(tri, cu_flat, shading="gouraud", cmap="Greys")
    ax[0, 0].set_title("Cu via layout (held-out)"); ax[0, 0].set_aspect("equal")

    vmax = float(max(vm.max(), pred.max()))
    for a, f, t in ((ax[0, 1], vm, "von Mises (FE oracle) [GPa-scaled]"),
                    (ax[0, 2], pred, "von Mises (CNN surrogate)")):
        tp = a.tripcolor(tri, f, shading="gouraud", cmap="inferno", vmin=0, vmax=vmax)
        a.set_title(t); a.set_aspect("equal"); fig.colorbar(tp, ax=a, fraction=0.046)

    tp = ax[1, 0].tripcolor(tri, np.abs(pred - vm), shading="gouraud", cmap="viridis")
    ax[1, 0].set_title("|surrogate - FE|"); ax[1, 0].set_aspect("equal")
    fig.colorbar(tp, ax=ax[1, 0], fraction=0.046)

    # KOZ overlay: FE (filled) vs surrogate (contour)
    ax[1, 1].tripcolor(tri, (vm > thr).astype(float), shading="gouraud", cmap="Reds")
    ax[1, 1].tricontour(tri, (pred > thr).astype(float), levels=[0.5], colors="cyan",
                        linewidths=1.5)
    ax[1, 1].set_title(f"keep-out zone: FE (red) vs surrogate (cyan)\n(threshold {KOZ_FRAC:.0%} max)")
    ax[1, 1].set_aspect("equal")

    ax[1, 2].scatter(rels, ious, c="#1f77b4")
    ax[1, 2].set_xlabel("rel-L2 stress error"); ax[1, 2].set_ylabel("KOZ IoU")
    ax[1, 2].set_title(f"held-out layouts\nmean rel-L2 {np.mean(rels):.3f}, IoU {np.mean(ious):.3f}")
    ax[1, 2].grid(alpha=0.3)

    fig.suptitle("TSV thermo-mechanical stress: FE thermoelasticity + learned stress-field "
                 "surrogate + keep-out-zone (repo's CFRP stress x GNN line, on silicon)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main()
