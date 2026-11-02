"""viz/denso_plate_features.py — through-thickness feature profiles of an injection-moulded
short-fibre plate, for building intuition on the Denso residual-strain theme
(research/DENSO_RA_RESIDUAL_STRAIN.md). Exports JSON for the interactive 3-D viewer and a
static figure.

What is computed, at NZ points through the thickness of a flat plate:
  1. fibre orientation tensor a_ij(z)  -- ILLUSTRATIVE skin / shell / core profile
     (flow along x). NOT from a flow simulation: no Folgar-Tucker solver exists in this
     repository. Shape chosen after the layer structure commonly reported for injection
     moulding: thin less-aligned skin, flow-aligned shell, transverse-leaning core.
  2. cooling and crystallinity  -- 1-D transient conduction between the two mould walls
     (explicit FD, Dirichlet wall temperatures), with the Nakamura law driven by the
     Hoffman-Lauritzen K(T) of validation/cfrtp_peek_validation.py (validated on carbon/
     PEEK: Tp band, alpha(T) shift, quench suppression). Crystallisation is integrated
     during the in-mould time T_MOULD_S only; after ejection the part is assumed to cool
     fast enough that crystallinity no longer changes.
  3. orientation-averaged expansion <alpha>(z) and crystallisation shrinkage <beta>(z) via
     <C>^-1 : <C : alpha> (fiber_orientation_cte.py, hybrid closure).
  4. residual strain and stress in the FREE plate after ejection, from classical
     lamination: eigenstrain e*(z) = <alpha> dT + <beta> X_c, with each layer stress-free
     at T_LOCK and cooled to room temperature; mid-plane strain and curvature from force
     and moment balance. Residual strain here means the ELASTIC (mechanical) strain
     e0 + z k - e*; definitions in measurement practice differ (see the Denso memo).

Two cases: SYMMETRIC mould temperatures (no warpage expected) and ASYMMETRIC ones (one wall
hotter), which crystallises one side more and bends the plate.

Validated:
  V1 force and moment balance: |int sigma dz| and |int sigma z dz| vanish relative to the
     stress magnitude (machine precision),
  V2 symmetric case: curvature vanishes (machine precision),
  V3 uniform orientation + uniform eigenstrain: residual stress vanishes (free contraction),
  V4 the fibre-sampling scales reproduce the target a_ij by Monte Carlo (the 3-D fibres
     drawn in the viewer have the stated second moment).

Honest scope: constants are ILLUSTRATIVE and PEEK-like (short carbon/PEEK elastic and
thermal values chosen for plausibility, not taken from a datasheet); the orientation
profile is assumed, not simulated; no latent heat, no pressure/packing history, no
viscoelastic relaxation (so stress magnitudes are overstated -- the repository's own
viscoelastic seed relaxes ~74%), crystallisation shrinkage counted in full after lock-in
(an upper bound), linear elastic lamination with orthotropic layers. The point is the SHAPE
of each profile and how the features connect, not the numbers.

Run:  python3 viz/denso_plate_features.py      (writes viz/denso_plate_features.json/.png)
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "validation"))
from fiber_orientation_cte import ud_stiffness, orient_avg_cte, orient_avg_stiffness  # noqa: E402
from cfrtp_peek_validation import K_hl, N_AVRAMI, HL  # noqa: E402

SEED = 20261102
NZ = 41
H = 3.0e-3                      # plate thickness [m]
L_PLATE = 0.10                  # plate length for the warpage sagitta [m]
DIFF = 1.8e-7                   # thermal diffusivity [m^2/s] (PEEK-like composite)
T_MELT = 380.0                  # melt temperature at the end of filling [C]
T_MOULD_S = 40.0                # time in the mould [s]
T_LOCK = 143.0                  # stress-free lock-in temperature ~ PEEK Tg [C]
T_ROOM = 25.0
X_INF = 0.30                    # absolute crystallinity at full relative crystallisation
# ILLUSTRATIVE short carbon/PEEK-like unidirectional (perfectly aligned) constants
UD = dict(EL=35.0e9, ET=7.0e9, nuLT=0.35, nuTT=0.45, GLT=2.8e9)
A_L, A_T = 5.0e-6, 55.0e-6      # CTE axial / transverse [1/K]
B_L, B_T = -0.002, -0.020       # crystallisation strain per unit X_c, axial / transverse
CASES = {"symmetric": (185.0, 185.0), "asymmetric": (170.0, 200.0)}   # (bottom, top) [C]
# Mould temperatures sit above Tg (as in real PEEK moulding): with a 100 C mould the
# validated kinetics quench the whole 2-3 mm plate amorphous (cooling far beyond the
# ~640 C/min suppression threshold found in validation/), leaving nothing to show.
IP = [0, 1, 5]                  # in-plane Voigt indices xx, yy, xy


def orientation_profile(z):
    """ILLUSTRATIVE skin / shell / core diagonal a_ij(z); flow along x, z in [-1, 1]."""
    s = np.abs(z)
    shell = 1.0 / (1.0 + np.exp(-(s - 0.35) / 0.06))   # 0 in the core -> 1 in the shell
    skin = 1.0 / (1.0 + np.exp(-(s - 0.93) / 0.02))    # 1 only in the outermost skin
    a33 = np.full_like(z, 0.06)
    a11 = 0.24 + (0.80 - 0.24) * shell - (0.80 - 0.55) * skin
    a22 = 1.0 - a11 - a33
    return a11, a22, a33


def fibre_scales(a_diag, n=60000, iters=40, rng=None):
    """Variances s of an angular central Gaussian whose second moment E[pp^T] = diag(a)."""
    g = rng.standard_normal((n, 3))
    s = np.array(a_diag, float)
    for _ in range(iters):
        p = g * np.sqrt(s)
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        est = (p ** 2).mean(axis=0)
        s = s * np.asarray(a_diag) / est
        s /= s.sum()
    p = g * np.sqrt(s); p /= np.linalg.norm(p, axis=1, keepdims=True)
    return s, (p ** 2).mean(axis=0)


def cool_and_crystallise(T_bot, T_top, nz=NZ, t_end=T_MOULD_S, a0=1e-3):
    dz = H / (nz - 1)
    dt = 0.4 * dz * dz / DIFF
    T = np.full(nz, T_MELT); T[0], T[-1] = T_bot, T_top
    a = np.full(nz, a0)
    t = 0.0
    n = N_AVRAMI
    while t < t_end:
        K = np.array([K_hl(Ti) for Ti in T])
        aa = np.clip(a, 1e-8, 1 - 1e-10)
        argl = np.maximum(-np.log(1 - aa), 1e-12)
        r = n * K * (1 - aa) * argl ** ((n - 1) / n)
        a = np.minimum(a + np.maximum(r * dt, 0.0), 1.0)
        Tn = T.copy()
        Tn[1:-1] = T[1:-1] + DIFF * dt / dz ** 2 * (T[2:] - 2 * T[1:-1] + T[:-2])
        T = Tn; t += dt
    return a, T


def reduced_Q(Cv):
    """Plane-stress (sigma_zz = 0) in-plane stiffness from a 6x6 Voigt stiffness."""
    Q = np.zeros((3, 3))
    for i, I in enumerate(IP):
        for j, J in enumerate(IP):
            Q[i, j] = Cv[I, J] - Cv[I, 2] * Cv[2, J] / Cv[2, 2]
    return Q


def lamination(zphys, Qs, estar):
    """Free plate: solve [A B; B D][e0; k] = [N*; M*]; return e0, k, mech strain, stress."""
    dz = zphys[1] - zphys[0]                  # trapezoid weights on the uniform grid
    w = np.full_like(zphys, dz); w[0] = w[-1] = dz / 2
    A = np.einsum("k,kij->ij", w, Qs)
    B = np.einsum("k,k,kij->ij", w, zphys, Qs)
    D = np.einsum("k,k,kij->ij", w, zphys ** 2, Qs)
    Ns = np.einsum("k,kij,kj->i", w, Qs, estar)
    Ms = np.einsum("k,k,kij,kj->i", w, zphys, Qs, estar)
    ABD = np.block([[A, B], [B, D]])
    sol = np.linalg.solve(ABD, np.concatenate([Ns, Ms]))
    e0, kap = sol[:3], sol[3:]
    em = e0[None, :] + zphys[:, None] * kap[None, :] - estar
    sig = np.einsum("kij,kj->ki", Qs, em)
    F = np.einsum("k,ki->i", w, sig); M = np.einsum("k,k,ki->i", w, zphys, sig)
    return e0, kap, em, sig, F, M, w


def layer_properties(a11, a22, a33, Cud):
    alpha, beta, Qs = [], [], []
    for x, y, zz in zip(a11, a22, a33):
        a = np.diag([x, y, zz])
        al, Cavg = orient_avg_cte(a, "hybrid", Cud, aL=A_L, aT=A_T)
        be, _ = orient_avg_cte(a, "hybrid", Cud, aL=B_L, aT=B_T)
        alpha.append(al); beta.append(be); Qs.append(reduced_Q(Cavg))
    return np.array(alpha), np.array(beta), np.array(Qs)


def run_case(name, T_bot, T_top, z, alpha, beta, Qs):
    arel, T_end = cool_and_crystallise(T_bot, T_top)
    Xc = X_INF * arel
    dT = T_ROOM - T_LOCK
    estar = np.stack([alpha[:, 0, 0] * dT + beta[:, 0, 0] * Xc,
                      alpha[:, 1, 1] * dT + beta[:, 1, 1] * Xc,
                      np.zeros_like(z)], axis=1)
    zphys = z * H / 2
    e0, kap, em, sig, F, M, w = lamination(zphys, Qs, estar)
    smag = np.einsum("k,k->", w, np.abs(sig[:, 0])) + 1e-30
    # Attribution. Lamination is linear in e*, so the thermal part (which carries the
    # orientation, through <alpha>) and the crystallisation part (<beta> X_c) superpose
    # EXACTLY: em = em_T + em_X, kappa = kappa_T + kappa_X.
    eT = estar.copy(); eT[:, 0] = alpha[:, 0, 0] * dT; eT[:, 1] = alpha[:, 1, 1] * dT
    eX = estar - eT
    _, kT, emT, *_ = lamination(zphys, Qs, eT)
    _, kX, emX, *_ = lamination(zphys, Qs, eX)
    superpose_err = float(np.max(np.abs(emT + emX - em)) / (np.max(np.abs(em)) + 1e-30))
    rng = lambda v: float(np.max(v) - np.min(v)) * 1e3
    return dict(
        attribution=dict(
            range_xx_thermal=rng(emT[:, 0]), range_xx_cryst=rng(emX[:, 0]),
            range_yy_thermal=rng(emT[:, 1]), range_yy_cryst=rng(emX[:, 1]),
            sag_y_thermal_mm=float(kT[1] * L_PLATE ** 2 / 8 * 1e3),
            sag_y_cryst_mm=float(kX[1] * L_PLATE ** 2 / 8 * 1e3),
            superpose_err=superpose_err),
        T_bottom=T_bot, T_top=T_top, T_end_max=float(T_end.max()),
        alpha_rel=arel.tolist(), Xc=Xc.tolist(),
        eps_star_xx=(estar[:, 0] * 1e3).tolist(), eps_star_yy=(estar[:, 1] * 1e3).tolist(),
        eps_res_xx=(em[:, 0] * 1e3).tolist(), eps_res_yy=(em[:, 1] * 1e3).tolist(),
        sigma_xx=(sig[:, 0] / 1e6).tolist(), sigma_yy=(sig[:, 1] / 1e6).tolist(),
        e0=(e0 * 1e3).tolist(), kappa=kap.tolist(),
        sag_x_mm=float(kap[0] * L_PLATE ** 2 / 8 * 1e3),
        sag_y_mm=float(kap[1] * L_PLATE ** 2 / 8 * 1e3),
        balance_F_rel=float(np.max(np.abs(F[:2])) * H / smag),
        balance_M_rel=float(np.max(np.abs(M[:2])) / (smag * H / 2)),
    )


def main():
    rng = np.random.default_rng(SEED)
    z = np.linspace(-1.0, 1.0, NZ)
    a11, a22, a33 = orientation_profile(z)
    Cud = ud_stiffness(**UD)
    alpha, beta, Qs = layer_properties(a11, a22, a33, Cud)

    scales, errs = [], []
    for x, y, zz in zip(a11, a22, a33):
        s, est = fibre_scales([x, y, zz], rng=rng)
        scales.append(s.tolist()); errs.append(np.max(np.abs(est - [x, y, zz])))
    print(f"[V4] fibre sampling reproduces a_ij: max |E[pp^T] - a| = {max(errs):.2e}")

    cases = {k: run_case(k, *v, z, alpha, beta, Qs) for k, v in CASES.items()}
    for k, c in cases.items():
        print(f"[V1] {k:10s}: force balance {c['balance_F_rel']:.1e}, moment balance "
              f"{c['balance_M_rel']:.1e} (relative; should be ~0)")
    ks = cases["symmetric"]["kappa"]
    print(f"[V2] symmetric case curvature: kx {ks[0]:.2e}, ky {ks[1]:.2e} 1/m (should be ~0)")

    # V3: uniform orientation + uniform eigenstrain -> no residual stress
    au = np.diag([0.5, 0.44, 0.06]); _, Cu = orient_avg_cte(au, "hybrid", Cud)
    Qu = np.repeat(reduced_Q(Cu)[None], NZ, axis=0)
    eu = np.tile([-3e-3, -5e-3, 0.0], (NZ, 1))
    _, _, emu, sigu, *_ = lamination(z * H / 2, Qu, eu)
    print(f"[V3] uniform plate, uniform eigenstrain: max |sigma| = {np.max(np.abs(sigu)):.2e} Pa "
          "(should be ~0)")

    for k, c in cases.items():
        at = c["attribution"]
        print(f"[attr] {k:10s}: residual-strain spread xx thermal {at['range_xx_thermal']:.2f} / "
              f"cryst {at['range_xx_cryst']:.2f}, yy thermal {at['range_yy_thermal']:.2f} / "
              f"cryst {at['range_yy_cryst']:.2f} x1e-3 | sag_y thermal {at['sag_y_thermal_mm']:+.3f} / "
              f"cryst {at['sag_y_cryst_mm']:+.3f} mm | superposition err {at['superpose_err']:.1e}")
    for k, c in cases.items():
        x = np.array(c["Xc"]); r = np.array(c["eps_res_xx"])
        print(f"{k:10s} (bottom {c['T_bottom']:.0f} C / top {c['T_top']:.0f} C): "
              f"X_c {x.min():.3f}..{x.max():.3f}, residual strain xx {r.min():+.2f}..{r.max():+.2f} "
              f"x1e-3, sag over {L_PLATE*1e3:.0f} mm: x {c['sag_x_mm']:+.3f} mm, "
              f"y {c['sag_y_mm']:+.3f} mm")

    out = dict(
        meta=dict(NZ=NZ, H_mm=H * 1e3, L_mm=L_PLATE * 1e3, T_melt=T_MELT, T_lock=T_LOCK,
                  T_room=T_ROOM, t_mould_s=T_MOULD_S, X_inf=X_INF, UD=UD,
                  alpha_L=A_L, alpha_T=A_T, beta_L=B_L, beta_T=B_T,
                  note="ILLUSTRATIVE constants; orientation profile assumed, not simulated"),
        z=z.tolist(), a11=a11.tolist(), a22=a22.tolist(), a33=a33.tolist(),
        fibre_scales=scales,
        alpha_xx=(alpha[:, 0, 0] * 1e6).tolist(), alpha_yy=(alpha[:, 1, 1] * 1e6).tolist(),
        cases=cases,
        validation=dict(fibre_moment_err=float(max(errs)),
                        sym_kappa=ks[:2], uniform_sigma_max=float(np.max(np.abs(sigu)))),
    )
    jpath = os.path.join(HERE, "denso_plate_features.json")
    with open(jpath, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"wrote {jpath}")
    _plot(os.path.join(HERE, "denso_plate_features.png"), out)


def _plot(path, d):
    import shutil
    sys.path.insert(0, os.path.join(ROOT, "slides", "figure_sources"))
    try:                                   # repo convention: thesis_style, graceful fallback
        from thesis_style import use
        use(width_frac=1.0, aspect=0.45)
        if shutil.which("latex") is None:  # no TeX: Times-compatible text + STIX math
            matplotlib.rcParams["text.usetex"] = False
            matplotlib.rcParams["font.family"] = "serif"
            matplotlib.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif",
                                                 "STIXGeneral", "DejaVu Serif"]
            matplotlib.rcParams["mathtext.fontset"] = "stix"
    except Exception:
        pass
    import matplotlib.pyplot as plt
    z = np.array(d["z"])
    fig, ax = plt.subplots(1, 4, figsize=(13, 4.2), sharey=True)
    ax[0].plot(d["a11"], z, "-", color="#1f4e79", lw=2, label=r"$a_{11}$ (flow, $x$)")
    ax[0].plot(d["a22"], z, "--", color="#7a3b2e", lw=2, label=r"$a_{22}$ (transverse, $y$)")
    ax[0].set_xlabel("orientation tensor component"); ax[0].set_ylabel(r"normalized thickness $z$")
    ax[0].set_title("(a) fibre orientation (assumed)")
    for k, ls in (("symmetric", "-"), ("asymmetric", "--")):
        c = d["cases"][k]
        ax[1].plot(c["Xc"], z, ls, color="#333333", lw=2,
                   label=f"{k}: {c['T_bottom']:.0f}/{c['T_top']:.0f} °C")
        ax[3].plot(c["eps_res_xx"], z, ls, color="#1f4e79", lw=2, label=rf"$\varepsilon_{{xx}}$, {k}")
        ax[3].plot(c["eps_res_yy"], z, ls, color="#7a3b2e", lw=1.4, label=rf"$\varepsilon_{{yy}}$, {k}")
    ax[1].set_xlabel(r"crystallinity $X_\mathrm{c}$"); ax[1].set_title("(b) crystallinity (HL kinetics)")
    ax[2].plot(d["alpha_xx"], z, "-", color="#1f4e79", lw=2, label=r"$\langle\alpha\rangle_{xx}$")
    ax[2].plot(d["alpha_yy"], z, "--", color="#7a3b2e", lw=2, label=r"$\langle\alpha\rangle_{yy}$")
    ax[2].set_xlabel(r"CTE [ppm/K]"); ax[2].set_title(r"(c) $\langle C\rangle^{-1}:\langle C:\alpha\rangle$")
    ax[3].axvline(0, color="#999999", lw=0.8)
    ax[3].set_xlabel(r"residual strain [$10^{-3}$]"); ax[3].set_title("(d) residual strain, free plate")
    for a in ax:
        a.grid(True, alpha=0.3); a.legend(fontsize=7, loc="lower right")
    fig.suptitle("Through-thickness features of an injection-moulded plate "
                 "(ILLUSTRATIVE constants; orientation assumed, not simulated)", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
