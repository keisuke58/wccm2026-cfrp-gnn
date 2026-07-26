"""cfrp_cure_residual_stress.py — LIGHT seed for the CFRP residual-stress theme
(Daikin / NEDO joint research, see research/MEMO_cfrp_residual_stress_collab.md):
map the MANUFACTURING PROCESS (cure temperature drop + chemical cure shrinkage) to
the CURE-INDUCED RESIDUAL STRESS and warpage of a laminate — the "process -> residual
stress" quantity the collaboration is about.

Physics (classical lamination theory, CLT; a light first cut of the cure-hardening /
CHILE picture from the literature): a thermoset CFRP ply is orthotropic. Two process
eigenstrains build residual stress as the part solidifies and cools:
  * thermal:   alpha * dT   (dT = RT - cure temperature < 0),
  * chemical:  cure shrinkage of the epoxy (fibres inert; transverse a few tenths %).
Each ply's free strain is resisted by the neighbours at other angles, so residual
stresses lock in and an UNSYMMETRIC layup WARPS (mid-plane curvature). CLT assembles
the laminate ABD stiffness, forms the thermal+shrinkage force/moment resultants
(N*, M*), solves [A B; B D][eps0; kappa] = [N*; M*], and recovers per-ply stresses.

Validated: a SYMMETRIC layup has B = 0 -> zero curvature (no warp) but non-zero
interlaminar residual stress; an UNSYMMETRIC [0/90] warps. Both are reproduced.

What it shows:
  * warped shape of an unsymmetric [0/90] laminate from the process,
  * per-ply transverse residual stress sigma_2 (the matrix-cracking driver),
  * PROCESS -> RESIDUAL STRESS: max transverse residual stress vs cool-down dT and vs
    cure-shrinkage level (the input->output map the surrogate would learn),
  * warpage curvature vs process.

Honest scope: linear CLT with cure shrinkage as an equivalent eigenstrain (no explicit
cure-kinetics / degree-of-cure evolution, no viscoelastic stress relaxation, no tool-
part interaction) — the documented next steps toward a cure-hardening (CHILE) /
viscoelastic FE. Linear CLT also OVER-predicts the warpage of thin UNSYMMETRIC
laminates; geometric nonlinearity (Hyer) gives the true cylindrical shape and a smaller
magnitude — so the unsymmetric curvature here is qualitative (warps vs not), while the
residual-STRESS magnitudes (tens of MPa) are the physical takeaway. Illustrative
T300/epoxy-like properties. This is the physics core; a process -> residual-stress
surrogate (repo CFRP-GNN / DeepONet pipeline) sits on top as the subordinate
accelerator. No ML here.

Run:  python3 cfrp_cure_residual_stress.py     (writes cfrp_cure_residual_stress.png)
      python3 cfrp_cure_residual_stress.py --help
"""
from __future__ import annotations

import argparse

import numpy as np

SEED = 20260726
# T300/epoxy-like ply (GPa, 1/K)
E1, E2 = 135e9, 9e9
NU12 = 0.30
G12 = 5e9
A1, A2 = -0.3e-6, 28e-6          # thermal expansion [1/K] (fibre dir ~0, transverse large)
T_CURE, T_ROOM = 180.0, 25.0     # cure and room temperature [C]
TPLY = 0.15e-3                    # ply thickness [m]
BETA2 = -3e-3                    # transverse cure (chemical) shrinkage eigenstrain
BETA1 = -0.05e-3                 # longitudinal shrinkage (fibre-dominated, small)


def q_material():
    nu21 = NU12 * E2 / E1
    d = 1.0 - NU12 * nu21
    return np.array([[E1 / d, NU12 * E2 / d, 0],
                     [NU12 * E2 / d, E2 / d, 0],
                     [0, 0, G12]])


def transform_Q(Q, th):
    c, s = np.cos(th), np.sin(th)
    T = np.array([[c * c, s * s, 2 * s * c],
                  [s * s, c * c, -2 * s * c],
                  [-s * c, s * c, c * c - s * s]])
    R = np.diag([1, 1, 2])                        # Reuter matrix (engineering shear)
    return np.linalg.inv(T) @ Q @ R @ T @ np.linalg.inv(R)


def free_strain(th, dT):
    """Per-ply process free strain (thermal + cure shrinkage) in laminate axes."""
    c, s = np.cos(th), np.sin(th)
    a_mat = np.array([A1 * dT + BETA1, A2 * dT + BETA2, 0.0])   # material axes
    # rotate strain (material -> laminate): inverse of strain transform
    Te = np.array([[c * c, s * s, -2 * s * c],
                   [s * s, c * c, 2 * s * c],
                   [s * c, -s * c, c * c - s * s]])
    return Te @ a_mat


def laminate(layup, dT, tply=TPLY):
    """CLT: assemble ABD and thermal/shrinkage resultants, solve for mid-plane strain
    eps0 and curvature kappa, return per-ply material-axis stresses."""
    Qm = q_material()
    n = len(layup)
    z = (np.arange(n + 1) - n / 2.0) * tply        # ply interface heights
    A = np.zeros((3, 3)); B = np.zeros((3, 3)); D = np.zeros((3, 3))
    Nst = np.zeros(3); Mst = np.zeros(3)
    Qbars, efrees = [], []
    for k, deg in enumerate(layup):
        th = np.radians(deg)
        Qb = transform_Q(Qm, th); ef = free_strain(th, dT)
        Qbars.append(Qb); efrees.append(ef)
        dz = z[k + 1] - z[k]; dz2 = z[k + 1] ** 2 - z[k] ** 2; dz3 = z[k + 1] ** 3 - z[k] ** 3
        A += Qb * dz; B += Qb * dz2 / 2; D += Qb * dz3 / 3
        Nst += Qb @ ef * dz; Mst += Qb @ ef * dz2 / 2
    ABD = np.block([[A, B], [B, D]])
    sol = np.linalg.solve(ABD, np.concatenate([Nst, Mst]))
    eps0, kappa = sol[:3], sol[3:]
    # per-ply stresses at ply mid-height, back to material axes
    sig_mat = []
    for k, deg in enumerate(layup):
        th = np.radians(deg); zc = 0.5 * (z[k] + z[k + 1])
        eps = eps0 + zc * kappa - efrees[k]
        sig_lam = Qbars[k] @ eps
        c, s = np.cos(th), np.sin(th)
        Ts = np.array([[c * c, s * s, 2 * s * c],
                       [s * s, c * c, -2 * s * c],
                       [-s * c, s * c, c * c - s * s]])
        sig_mat.append(Ts @ sig_lam)               # [sigma1, sigma2, tau12]
    return eps0, kappa, np.array(sig_mat), z


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=str, default="cfrp_cure_residual_stress.png")
    args = ap.parse_args()
    global BETA2
    np.random.seed(SEED)
    dT = T_ROOM - T_CURE

    # symmetric vs unsymmetric validation
    _, k_sym, sig_sym, zsym = laminate([0, 90, 90, 0], dT)
    _, k_uns, sig_uns, zuns = laminate([0, 90], dT)
    warp_sym = float(np.max(np.abs(k_sym)))
    warp_uns = float(np.max(np.abs(k_uns)))
    print(f"CFRP cure residual stress (CLT), dT={dT:.0f} C, cure shrinkage beta2={BETA2*100:.2f}%")
    print(f"  symmetric [0/90]s : max curvature {warp_sym:.2e} 1/m (should be ~0)  "
          f"-> transverse sigma2 in 0-ply {sig_sym[0,1]/1e6:.1f} MPa, 90-ply {sig_sym[1,1]/1e6:.1f} MPa")
    print(f"  unsymmetric [0/90]: max curvature {warp_uns:.3f} 1/m (warps)  "
          f"-> transverse sigma2 0-ply {sig_uns[0,1]/1e6:.1f} MPa")

    # process -> residual stress sweeps
    dTs = np.linspace(-20, -180, 25)
    s2_dT = np.array([np.max(laminate([0, 90, 90, 0], d)[2][:, 1]) for d in dTs]) / 1e6
    betas = np.linspace(0.0, -6e-3, 25)
    s2_b = []
    b_save = BETA2
    for b in betas:
        BETA2 = b
        s2_b.append(np.max(laminate([0, 90, 90, 0], dT)[2][:, 1]) / 1e6)
    BETA2 = b_save
    s2_b = np.array(s2_b)
    kap_dT = np.array([np.max(np.abs(laminate([0, 90], d)[1])) for d in dTs])

    _plot(args.out, k_uns, sig_sym, sig_uns, zsym, dTs, s2_dT, betas, s2_b, kap_dT)
    print(f"wrote {args.out}")


def _plot(out, k_uns, sig_sym, sig_uns, zsym, dTs, s2_dT, betas, s2_b, kap_dT):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(13, 10))

    # warped shape of the unsymmetric [0/90] laminate
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    xx, yy = np.meshgrid(np.linspace(-0.05, 0.05, 30), np.linspace(-0.05, 0.05, 30))
    w = 0.5 * (k_uns[0] * xx ** 2 + k_uns[1] * yy ** 2)
    ax1.plot_surface(xx * 1e3, yy * 1e3, w * 1e3, cmap="coolwarm", alpha=0.9)
    ax1.set_xlabel("x [mm]"); ax1.set_ylabel("y [mm]"); ax1.set_zlabel("warp w [mm]")
    ax1.set_title(f"unsymmetric [0/90] warpage from cure+cooldown\n"
                  f"max curvature {np.max(np.abs(k_uns)):.2f} 1/m (saddle/cylindrical)")

    # per-ply transverse residual stress (symmetric laminate: no warp, real stress)
    ax2 = fig.add_subplot(2, 2, 2)
    plies = ["0°", "90°", "90°", "0°"]
    s2 = sig_sym[:, 1] / 1e6
    colors = ["#1f77b4" if p == "0°" else "#d62728" for p in plies]
    ax2.bar(range(len(plies)), s2, color=colors)
    ax2.axhline(0, color="k", lw=0.6)
    ax2.set_xticks(range(len(plies))); ax2.set_xticklabels(plies)
    ax2.set_ylabel("transverse residual stress σ₂ [MPa]")
    ax2.set_title("per-ply residual σ₂ in symmetric [0/90]s\n(90° plies: transverse tension → matrix-crack driver)")
    ax2.grid(True, axis="y", alpha=0.3)

    # process -> residual stress: vs cool-down and vs cure shrinkage
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(-dTs, s2_dT, "-o", ms=3, color="#b5651d", label="vs cool-down |ΔT| (β₂ fixed)")
    ax3.set_xlabel("cool-down |ΔT| [°C]"); ax3.set_ylabel("max transverse residual σ₂ [MPa]")
    ax3.set_title("PROCESS → RESIDUAL STRESS\n(input the surrogate would learn)")
    ax3b = ax3.twiny()
    ax3b.plot(-betas * 100, s2_b, "-s", ms=3, color="#2ca02c", label="vs cure shrinkage |β₂| (%)")
    ax3b.set_xlabel("cure shrinkage |β₂| [%]", color="#2ca02c")
    ax3.grid(True, alpha=0.3)
    lines = ax3.get_lines() + ax3b.get_lines()
    ax3.legend(lines, [ln.get_label() for ln in lines], fontsize=8, loc="upper left")

    # warpage vs process
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(-dTs, kap_dT, "-o", ms=3, color="#9467bd")
    ax4.set_xlabel("cool-down |ΔT| [°C]"); ax4.set_ylabel("warpage curvature [1/m]")
    ax4.set_title("warpage (process-induced deformation) vs cool-down\nunsymmetric [0/90]")
    ax4.grid(True, alpha=0.3)

    fig.suptitle("CFRP cure-induced residual stress & warpage (CLT): manufacturing process "
                 "(ΔT + cure shrinkage) → residual stress — Daikin/NEDO theme seed", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main()
