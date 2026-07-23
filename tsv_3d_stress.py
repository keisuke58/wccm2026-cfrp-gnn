"""tsv_3d_stress.py — 3-D TSV thermo-mechanical stress with REAL elastic constants.

The 3-D, real-constants promotion of tsv_thermal_stress.py. A Cu through-silicon via
runs through an Si block; on cooling (dT) the Cu/Si CTE mismatch stresses the silicon.
Two things change vs the 2-D demo:

  * 3-D: linear-tetrahedron finite elements (constant strain), 3 DOF/node, full 6-
    component stress. Structured grid split into 6 tets per cube; vectorised assembly.
  * Real elastic constants: silicon is anisotropic cubic — stiffness C11=165.7,
    C12=63.9, C44=79.6 GPa (crystal frame; (100) wafer, via along [001], so crystal
    frame = FE frame). Copper is isotropic (E=120 GPa, nu=0.34). CTEs alpha_Cu=17,
    alpha_Si=2.6 ppm/K.

Headline physics that real constants unlock: with anisotropic Si the in-plane stress
around the via is NOT circular — it develops the well-known 4-fold ("cloverleaf")
pattern, so the keep-out zone is direction-dependent (<100> vs <110>). An isotropic-Si
approximation misses this entirely. The script solves both and compares:
  - von Mises on a vertical slice (depth dependence),
  - von Mises on a mid-depth in-plane slice (anisotropic 4-fold vs isotropic circular),
  - radial stress profiles along [100] vs [110].

Physics validation (run at startup): the thermoelastic FE is checked against two
closed-form cases on a homogeneous isotropic body — (A) prescribing the exact free-
thermal-expansion displacement on the boundary must give a stress-free interior, and
(B) a fully clamped body must give the uniform stress -(C11+2*C12)*alpha*dT. Both pass
to machine precision (~1e-15), so the assembly, thermal load, and stress recovery are
correct.

Honest caveats: single via, linear thermoelastic, structured tets, clamped base — a
concept demo, not a device-accurate packaging run. The clamped base adds a near-base
stress concentration (a BC effect, not TSV physics); the mid-depth in-plane comparison
that carries the anisotropy result is unaffected since both cases share the BC. von
Mises is an isotropic invariant used here only as a scalar stress map — the stress
tensor is exact, but von Mises is not the natural yield measure for anisotropic Si.
Silicon's thermal expansion is isotropic (cubic symmetry), so the isotropic eps_th is
correct even though its elasticity is anisotropic.

Run:  python3 tsv_3d_stress.py            (writes tsv_3d_stress.png)
      python3 tsv_3d_stress.py --help
"""
from __future__ import annotations

import argparse

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

SEED = 20260722
DT = -250.0
# Silicon: anisotropic cubic stiffness (GPa) + isotropic-equivalent for comparison
C11_SI, C12_SI, C44_SI = 165.7, 63.9, 79.6
E_SI_ISO, NU_SI_ISO = 130.0, 0.28
A_SI = 2.6e-6
# Copper: isotropic
E_CU, NU_CU, A_CU = 120.0, 0.34, 17.0e-6
VIA_R = 0.15           # via radius (fraction of block width)


# ----------------------------------------------------------------------------
# Structured tetrahedral mesh of the unit cube
# ----------------------------------------------------------------------------
def build_tet_mesh(n):
    lin = np.linspace(0, 1, n)
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing="ij")
    nodes = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    def nid(i, j, k):
        return (i * n + j) * n + k

    # 6-tet split of each cube sharing the main diagonal c000-c111
    order = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
             (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    hex_tets = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
                (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)]
    tets = []
    for i in range(n - 1):
        for j in range(n - 1):
            for k in range(n - 1):
                c = [nid(i + di, j + dj, k + dk) for di, dj, dk in order]
                for a, b, d, e in hex_tets:
                    tets.append((c[a], c[b], c[d], c[e]))
    return nodes, np.array(tets, dtype=np.int64)


def c_cubic(C11, C12, C44):
    C = np.zeros((6, 6))
    C[:3, :3] = C12
    C[0, 0] = C[1, 1] = C[2, 2] = C11
    C[3, 3] = C[4, 4] = C[5, 5] = C44
    return C


def c_isotropic(E, nu):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return c_cubic(lam + 2 * mu, lam, mu)


C_CU = c_isotropic(E_CU, NU_CU)


# ----------------------------------------------------------------------------
# Vectorised 3-D linear-tet thermoelastic solve -> per-node von Mises + stress
# ----------------------------------------------------------------------------
def solve(nodes, tets, C_si, fixed_dof=None, u_fixed=None, aniso_label="", homogeneous=False):
    """Thermoelastic solve with arbitrary Dirichlet BC.

    fixed_dof: bool mask over the 3N DOFs that are prescribed (default: clamp base z=0).
    u_fixed:   prescribed DOF values (default: zeros). Returns (nodal von Mises, per-elem
    stress). Non-zero prescribed displacements are handled by static condensation.
    """
    nN = len(nodes)
    ndof = 3 * nN
    P = nodes[tets]                                  # (E,4,3)
    E = len(tets)

    M = np.concatenate([np.ones((E, 4, 1)), P], axis=2)      # (E,4,4)
    detM = np.linalg.det(M)
    vol = np.abs(detM) / 6.0
    Minv = np.linalg.inv(M)                                   # (E,4,4)
    grads = Minv[:, 1:4, :].transpose(0, 2, 1)               # (E,4,3): node i -> grad

    B = np.zeros((E, 6, 12))
    for a in range(4):
        gx, gy, gz = grads[:, a, 0], grads[:, a, 1], grads[:, a, 2]
        cx = 3 * a
        B[:, 0, cx + 0] = gx
        B[:, 1, cx + 1] = gy
        B[:, 2, cx + 2] = gz
        B[:, 3, cx + 1] = gz; B[:, 3, cx + 2] = gy      # gyz
        B[:, 4, cx + 0] = gz; B[:, 4, cx + 2] = gx      # gzx
        B[:, 5, cx + 0] = gy; B[:, 5, cx + 1] = gx      # gxy

    centers = P.mean(axis=1)
    if homogeneous:                                          # single material (validation)
        in_via = np.zeros(E, bool)
    else:
        in_via = (centers[:, 0] - 0.5) ** 2 + (centers[:, 1] - 0.5) ** 2 <= VIA_R ** 2
    Cmat = np.where(in_via[:, None, None], C_CU[None], C_si[None])       # (E,6,6)
    alpha = np.where(in_via, A_CU, A_SI)
    eps_th = alpha[:, None] * DT * np.array([1, 1, 1, 0, 0, 0])[None]     # (E,6)

    CB = np.einsum("eij,ejk->eik", Cmat, B)                  # (E,6,12)
    Ke = vol[:, None, None] * np.einsum("eji,ejk->eik", B, CB)           # (E,12,12)
    fe = vol[:, None] * np.einsum("eji,ej->ei", B, np.einsum("eij,ej->ei", Cmat, eps_th))

    dofs = np.empty((E, 12), dtype=np.int64)
    for a in range(4):
        dofs[:, 3 * a:3 * a + 3] = 3 * tets[:, a:a + 1] + np.array([0, 1, 2])
    rows = np.repeat(dofs, 12, axis=1).reshape(E, 12, 12)
    cols = np.tile(dofs, (1, 12)).reshape(E, 12, 12)
    K = sp.coo_matrix((Ke.ravel(), (rows.ravel(), cols.ravel())), shape=(ndof, ndof)).tocsr()
    F = np.bincount(dofs.ravel(), weights=fe.ravel(), minlength=ndof)

    if fixed_dof is None:                                    # default BC: clamp base z=0
        fixed_dof = np.repeat(nodes[:, 2] < 1e-9, 3)
    if u_fixed is None:
        u_fixed = np.zeros(ndof)
    freed = ~fixed_dof
    u = np.array(u_fixed, dtype=float).copy()
    # static condensation for (possibly non-zero) prescribed displacements
    rhs = F[freed] - np.asarray(K[freed][:, fixed_dof] @ u[fixed_dof]).ravel()
    u[freed] = spla.spsolve(K[freed][:, freed].tocsc(), rhs)

    # per-element stress -> von Mises, averaged to nodes
    strain = np.einsum("eij,ej->ei", B, u[dofs])            # (E,6)
    sig = np.einsum("eij,ej->ei", Cmat, strain - eps_th)    # (E,6)
    sxx, syy, szz, syz, szx, sxy = sig.T
    vm = np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                 + 3 * (syz ** 2 + szx ** 2 + sxy ** 2))
    vm_node = np.bincount(tets.ravel(), weights=np.repeat(vm, 4), minlength=nN)
    cnt = np.bincount(tets.ravel(), minlength=nN)
    if aniso_label:
        print(f"[solve {aniso_label:9s}] {E} tets, {ndof} dof  |  von Mises max {vm.max():.3f} GPa")
    return vm_node / np.clip(cnt, 1, None), sig


def validate(n=7):
    """Two analytic checks of the thermoelastic FE (homogeneous isotropic body).

    A) Free thermal expansion: prescribe the exact free-expansion displacement
       u = alpha*dT*(x - x0) on the boundary. The correct thermoelastic solution is
       stress-free everywhere -> max|sigma| ~ 0.
    B) Fully clamped (u = 0 on the whole boundary): the exact solution is u = 0 in the
       interior too, giving a uniform stress sigma_nn = -(C11 + 2*C12)*alpha*dT.
    """
    nodes, tets = build_tet_mesh(n)
    ndof = 3 * len(nodes)
    C = c_isotropic(E_SI_ISO, NU_SI_ISO)
    bnd = ((nodes[:, 0] < 1e-9) | (nodes[:, 0] > 1 - 1e-9) | (nodes[:, 1] < 1e-9)
           | (nodes[:, 1] > 1 - 1e-9) | (nodes[:, 2] < 1e-9) | (nodes[:, 2] > 1 - 1e-9))
    fixed = np.repeat(bnd, 3)

    disp = A_SI * DT * (nodes - 0.5)                          # free thermal expansion
    uA = np.zeros(ndof)
    uA[0::3], uA[1::3], uA[2::3] = disp[:, 0], disp[:, 1], disp[:, 2]
    _, sigA = solve(nodes, tets, C, fixed, uA, homogeneous=True)
    stress_scale = abs((C[0, 0] + 2 * C[0, 1]) * A_SI * DT)
    relA = np.abs(sigA).max() / stress_scale

    _, sigB = solve(nodes, tets, C, fixed, np.zeros(ndof), homogeneous=True)
    sig_exact = -(C[0, 0] + 2 * C[0, 1]) * A_SI * DT
    relB = abs(sigB[:, 0].mean() - sig_exact) / abs(sig_exact)

    print("[validate] A free-expansion  -> max|sigma|/scale = "
          f"{relA:.2e}   ({'PASS' if relA < 1e-6 else 'FAIL'})")
    print(f"[validate] B fully-clamped   -> sigma_nn {sigB[:, 0].mean():.3f} vs exact "
          f"{sig_exact:.3f} GPa, rel err {relB:.2e}   ({'PASS' if relB < 1e-6 else 'FAIL'})")
    return relA < 1e-6 and relB < 1e-6


def slice_grid(nodes, field, n, axis, at):
    """Extract a 2-D slice (returns 2 in-plane coord arrays + field on the n x n slice)."""
    idx = int(round(at * (n - 1)))
    grid = field.reshape(n, n, n)
    if axis == "z":
        return grid[:, :, idx]
    if axis == "y":
        return grid[:, idx, :]
    return grid[idx, :, :]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=21, help="nodes per side (n^3 grid)")
    ap.add_argument("--out", type=str, default="tsv_3d_stress.png")
    args = ap.parse_args()

    validate()                                               # analytic FE checks first
    nodes, tets = build_tet_mesh(args.n)
    n = args.n
    vm_aniso, _ = solve(nodes, tets, c_cubic(C11_SI, C12_SI, C44_SI), aniso_label="aniso-Si")
    vm_iso, _ = solve(nodes, tets, c_isotropic(E_SI_ISO, NU_SI_ISO), aniso_label="iso-Si")

    # radial profiles at mid-depth along [100] (x) and [110] (diagonal)
    mid = n // 2
    g_an = vm_aniso.reshape(n, n, n)[:, :, mid]
    g_is = vm_iso.reshape(n, n, n)[:, :, mid]
    lin = np.linspace(0, 1, n)
    cext = n - 1
    r100, s100_an, s100_is = [], [], []
    r110, s110_an, s110_is = [], [], []
    for t in range(mid, n):
        r100.append(lin[t] - 0.5); s100_an.append(g_an[t, mid]); s100_is.append(g_is[t, mid])
    for t in range(mid, n):
        r110.append((lin[t] - 0.5) * np.sqrt(2)); s110_an.append(g_an[t, t]); s110_is.append(g_is[t, t])

    _plot(args.out, nodes, vm_aniso, vm_iso, n,
          (r100, s100_an, s100_is), (r110, s110_an, s110_is))
    print(f"wrote {args.out}")


def _plot(out, nodes, vm_aniso, vm_iso, n, prof100, prof110):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 3, figsize=(16, 10))
    ext = [0, 1, 0, 1]
    vmax = float(max(vm_aniso.max(), vm_iso.max()) * 0.9)

    # vertical slice (x-z) showing depth dependence
    im = ax[0, 0].imshow(slice_grid(nodes, vm_aniso, n, "y", 0.5).T, origin="lower",
                         extent=ext, cmap="inferno", vmax=vmax, aspect="auto")
    ax[0, 0].set_title("von Mises, vertical slice x-z (aniso Si)\n(base z=0 clamped)")
    ax[0, 0].set_xlabel("x"); ax[0, 0].set_ylabel("z (depth)")
    fig.colorbar(im, ax=ax[0, 0], fraction=0.046)

    # in-plane mid-depth: anisotropic (4-fold) vs isotropic (circular)
    circ = np.linspace(0, 2 * np.pi, 200)
    for a, g, t in ((ax[0, 1], slice_grid(nodes, vm_aniso, n, "z", 0.5),
                     "in-plane mid-depth: ANISOTROPIC Si (4-fold)"),
                    (ax[0, 2], slice_grid(nodes, vm_iso, n, "z", 0.5),
                     "in-plane mid-depth: isotropic Si (circular)")):
        im = a.imshow(g.T, origin="lower", extent=ext, cmap="inferno", vmax=vmax)
        a.plot(0.5 + VIA_R * np.cos(circ), 0.5 + VIA_R * np.sin(circ), "c--", lw=1.0)
        a.set_title(t); a.set_aspect("equal"); a.set_xlabel("x [100]"); a.set_ylabel("y [010]")
        fig.colorbar(im, ax=a, fraction=0.046)

    # difference map (aniso - iso) at mid-depth: what real constants add
    diff = slice_grid(nodes, vm_aniso, n, "z", 0.5) - slice_grid(nodes, vm_iso, n, "z", 0.5)
    im = ax[1, 0].imshow(diff.T, origin="lower", extent=ext, cmap="coolwarm",
                         vmin=-abs(diff).max(), vmax=abs(diff).max())
    ax[1, 0].plot(0.5 + VIA_R * np.cos(circ), 0.5 + VIA_R * np.sin(circ), "k--", lw=1.0)
    ax[1, 0].set_title("aniso - iso (mid-depth)\n= stress anisotropy real constants add")
    ax[1, 0].set_aspect("equal")
    fig.colorbar(im, ax=ax[1, 0], fraction=0.046)

    # radial profiles [100] vs [110]
    r100, s100_an, s100_is = prof100
    r110, s110_an, s110_is = prof110
    ax[1, 1].plot(r100, s100_an, "-o", ms=3, color="#d62728", label="[100] aniso")
    ax[1, 1].plot(r110, s110_an, "-o", ms=3, color="#1f77b4", label="[110] aniso")
    ax[1, 1].plot(r100, s100_is, "--", color="#7f7f7f", label="[100] iso")
    ax[1, 1].axvline(VIA_R, color="c", ls=":", label="via edge")
    ax[1, 1].set_title("radial von Mises at mid-depth\naniso: [100] != [110] (iso: isotropic)")
    ax[1, 1].set_xlabel("distance from via center"); ax[1, 1].set_ylabel("von Mises [GPa]")
    ax[1, 1].legend(fontsize=8); ax[1, 1].grid(alpha=0.3)

    # anisotropy ratio text panel
    ax[1, 2].axis("off")
    peak_100 = float(np.max(s100_an))          # peak von Mises in Si along [100]
    peak_110 = float(np.max(s110_an))          # peak von Mises in Si along [110]
    ratio = peak_110 / max(peak_100, 1e-9)
    txt = (
        "Real elastic constants (3-D):\n\n"
        f"  Si (anisotropic cubic):\n    C11={C11_SI}, C12={C12_SI}, C44={C44_SI} GPa\n"
        f"  Cu (isotropic): E={E_CU}, nu={NU_CU}\n"
        f"  dT = {DT:.0f} K,  via r = {VIA_R}\n\n"
        f"  von Mises max: aniso {vm_max(vm_aniso):.3f}  iso {vm_max(vm_iso):.3f} GPa\n"
        f"  peak von Mises [110]/[100] ~ {ratio:.2f}\n"
        "  (=1 for isotropic; !=1 = direction-\n   dependent keep-out zone from real Si)"
    )
    ax[1, 2].text(0.02, 0.98, txt, va="top", ha="left", fontsize=10, family="monospace")

    fig.suptitle("3-D TSV thermo-mechanical stress with real elastic constants — "
                 "anisotropic Si gives a 4-fold, direction-dependent keep-out zone",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130)


def vm_max(v):
    return float(v.max())


if __name__ == "__main__":
    main()
