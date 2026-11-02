"""validation/denso_tsuda2026_compare.py — compare this repository's chain
(short fibre -> aligned Mori–Tanaka -> orientation averaging -> lamination) with the DENSO /
Polyplastics paper

  M. Tsuda, T. Shiramomo, M. Tanaka, E. Kokuryu, A. Mochizuki, "Residual stress analysis of
  short fiber reinforced plastic based on fiber orientation distribution measured using X-ray
  micro CT", J. Jpn. Soc. Compos. Mater. 52(3), 88–96 (2026), doi:10.6089/jscm.52.88.

Inputs taken from the paper (its Table 1 and the numbers printed in its text and figures):
  GF 30 wt% (18.2 vol%), fibre E = 72 GPa, nu = 0.24, alpha = 5.5 ppm/C; mean diameter 10.1 um,
  mean length 170.6 um (aspect ratio 16.9); PPS nu = 0.37 and E(T), instantaneous alpha(T) at
  20 / 80 / 100 / 180 C as identified by the authors' inverse analysis; stress-free T0 = 180 C,
  cooled to room temperature (20 C); plate 1.0 mm (tensile / TMA) and 3.08 mm (hole drilling).
All paper values are read from data/literature/tsuda2026_jscm52_88.json: exact values from its
table and text, and values READ BY EYE from its figures (orientation profiles, Figs. 7a/8a;
CTE and moduli of the 1.0 mm plate, Figs. 9/10), each tagged with its source and reading
uncertainty. The paper states its thermal strain as eps_th = alpha (T - T0) with Table 1 values
interpolated linearly; its closure approximation is not named.

What is compared:
  C1 in-plane CTE of the 1.0 mm plate in MD / TD at 20, 80, 100, 180 C: this chain vs the
     paper's measurement and its own simulation (Fig. 10, read by eye);
  C2 in-plane moduli of the 1.0 mm plate at 20 C (slope of Fig. 9, read by eye);
  C3 residual stress through the 3.08 mm plate after cooling 180 -> 20 C (free plate), for two
     constitutive readings of the paper's temperature tables (total thermoelastic with the
     room-temperature stiffness, as its eq. (2); incremental with C(T)), against the surface
     stress it reports at the centre of the drilled face (sigma_xx = -24.3, sigma_yy = +15.2 MPa).
     A lamination estimate of the same quantity, not a reproduction of its 3-D FE model.

Validated (printed): Eshelby tensor limits (sphere, long cylinder), Mori–Tanaka against the
Hashin–Shtrikman lower bound for spheres, f -> 0 / 1 limits, Levin's relation for alpha.

Honest scope: the PPS constants were identified BY THE AUTHORS to fit their own measurements
with their own micromechanics (Digimat-MF), so agreement in C1/C2 checks that this chain is
consistent with theirs, not an independent validation of either. Off-diagonal orientation
components (|a_ij| < 0.15 in the paper) are ignored; the hybrid closure is used (the paper's
closure is not known here). No crystallisation, no viscoelasticity, no hole-drilling simulation.

Run:  python3 validation/denso_tsuda2026_compare.py   (writes results/denso_tsuda2026/)
"""
from __future__ import annotations

import json
import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "viz"))
from fiber_orientation_cte import orient_avg_cte  # noqa: E402
from denso_plate_features import reduced_Q  # noqa: E402

OUT = os.path.join(ROOT, "results", "denso_tsuda2026")

# ---------------------------------------------------------------- paper inputs -----
# All paper values come from data/literature/tsuda2026_jscm52_88.json (exact table/text values
# and figure values read by eye, each block tagged with its source).
PAPER = json.load(open(os.path.join(ROOT, "data", "literature", "tsuda2026_jscm52_88.json"), encoding="utf-8"))
_m, _t = PAPER["material"], PAPER["table1_PPS_inverse"]
GF = dict(E=_m["fibre_E_GPa"] * 1e9, nu=_m["fibre_nu"], alpha=_m["fibre_CTE_ppm_per_C"] * 1e-6)
VF = _m["fibre_volume_fraction"]
ASPECT = _m["aspect_ratio_mean"]
PPS_T = np.array(_t["T_C"], float)
PPS_E = np.array(_t["E_GPa"]) * 1e9
PPS_A = np.array(_t["CTE_inst_ppm_per_C"]) * 1e-6     # instantaneous CTE
PPS_NU = _m["matrix_nu"]
T0, T_RT = 180.0, 20.0
_f7, _f8 = PAPER["fig7_orientation_1mm"], PAPER["fig8_orientation_3mm"]
PROF_1MM = dict(z=_f7["z_mm"], axx=_f7["a_xx"], ayy=_f7["a_yy"])          # mirrored about 0
PROF_3MM = dict(d=_f8["depth_mm"], axx=_f8["a_xx"], ayy=_f8["a_yy"])
H3 = PAPER["processing"]["measured_thickness_um"][1] * 1e-6
_f10, _f9 = PAPER["fig10_cte_1mm"], PAPER["fig9_stress_strain_secant"]
FIG10 = dict(T=_f10["sim_T_C"], md_exp=[float(np.interp(T, _f10["exp_T_C"], _f10["MD_exp"])) for T in _f10["sim_T_C"]],
             td_exp=[float(np.interp(T, _f10["exp_T_C"], _f10["TD_exp"])) for T in _f10["sim_T_C"]],
             md_sim=_f10["MD_sim"], td_sim=_f10["TD_sim"])
FIG9_E20 = dict(md_exp=_f9["MD_exp"][0], td_exp=_f9["TD_exp"][0], md_sim=_f9["MD_sim"][0], td_sim=_f9["TD_sim"][0])
_st = PAPER["results_text"]["surface_centre_residual_stress_MPa"]
PAPER_STRESS = dict(sxx_MPa=_st["sigma_xx"], syy_MPa=_st["sigma_yy"], note=_st["note"])

# ------------------------------------------------------------- micromechanics ------
SQ2 = np.sqrt(2.0)
MANDEL = np.diag([1, 1, 1, SQ2, SQ2, SQ2])     # order 11,22,33,23,13,12


def iso_C_mandel(E, nu):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu)); mu = E / (2 * (1 + nu))
    C = np.zeros((6, 6)); C[:3, :3] = lam; C[range(3), range(3)] += 2 * mu
    C[3, 3] = C[4, 4] = C[5, 5] = 2 * mu
    return C


def eshelby_prolate(s, nu):
    """Eshelby tensor (Mandel, axis 1 = long axis) of a prolate spheroid, aspect s > 1, in an
    isotropic matrix (Tandon & Weng 1984 closed forms)."""
    s2 = s * s; q = s2 - 1.0
    g = s / q ** 1.5 * (s * np.sqrt(q) - np.arccosh(s))
    c = 1.0 / (1.0 - nu)
    S1111 = 0.5 * c * (1 - 2 * nu + (3 * s2 - 1) / q - (1 - 2 * nu + 3 * s2 / q) * g)
    S2222 = 3 / 8 * c * s2 / q + 0.25 * c * (1 - 2 * nu - 9 / (4 * q)) * g
    S2233 = 0.25 * c * (s2 / (2 * q) - (1 - 2 * nu + 3 / (4 * q)) * g)
    S2211 = -0.5 * c * s2 / q + 0.25 * c * (3 * s2 / q - (1 - 2 * nu)) * g
    S1122 = -0.5 * c * (1 - 2 * nu + 1 / q) + 0.5 * c * (1 - 2 * nu + 3 / (2 * q)) * g
    S2323 = 0.25 * c * (s2 / (2 * q) + (1 - 2 * nu - 3 / (4 * q)) * g)
    S1212 = 0.25 * c * (1 - 2 * nu - (s2 + 1) / q - 0.5 * (1 - 2 * nu - 3 * (s2 + 1) / q) * g)
    S = np.zeros((6, 6))
    S[0, 0] = S1111; S[1, 1] = S[2, 2] = S2222
    S[1, 2] = S[2, 1] = S2233
    S[1, 0] = S[2, 0] = S2211
    S[0, 1] = S[0, 2] = S1122
    S[3, 3] = 2 * S2323; S[4, 4] = S[5, 5] = 2 * S1212       # Mandel: 2 S_ijij for shear
    return S


def mori_tanaka(Cm, Cf, S, f):
    I = np.eye(6)
    T = np.linalg.inv(I + S @ np.linalg.inv(Cm) @ (Cf - Cm))
    A = T @ np.linalg.inv((1 - f) * I + f * T)
    return Cm + f * (Cf - Cm) @ A


def levin_alpha(C, Cm, Cf, am, af):
    """Levin's exact relation for a two-phase composite with isotropic phase CTEs."""
    S, Sm, Sf = np.linalg.inv(C), np.linalg.inv(Cm), np.linalg.inv(Cf)
    da = (af - am) * np.array([1, 1, 1, 0, 0, 0.0])
    return am * np.array([1, 1, 1, 0, 0, 0.0]) + (S - Sm) @ np.linalg.solve(Sf - Sm, da)


def mandel_to_voigt(Cm6):
    """Mandel stiffness -> repo Voigt (engineering shear, no factors on C)."""
    D = np.diag([1, 1, 1, 1 / SQ2, 1 / SQ2, 1 / SQ2])
    return D @ Cm6 @ D


def ud(T):
    """Aligned GF/PPS at temperature T: repo-Voigt stiffness and (alpha_L, alpha_T)."""
    Em = float(np.interp(T, PPS_T, PPS_E)); am = float(np.interp(T, PPS_T, PPS_A))
    Cm = iso_C_mandel(Em, PPS_NU); Cf = iso_C_mandel(GF["E"], GF["nu"])
    S = eshelby_prolate(ASPECT, PPS_NU)
    C = mori_tanaka(Cm, Cf, S, VF)
    a = levin_alpha(C, Cm, Cf, am, GF["alpha"])
    return mandel_to_voigt(C), a[0], a[1]


def validate():
    nu = 0.3
    S = eshelby_prolate(1 + 1e-4, nu)
    sph = (7 - 5 * nu) / (15 * (1 - nu))
    S_long = eshelby_prolate(1e5, nu)
    cyl = dict(S2222=(5 - 4 * nu) / (8 * (1 - nu)), S2233=(4 * nu - 1) / (8 * (1 - nu)),
               S2211=nu / (2 * (1 - nu)), S1212=0.25, S2323=(3 - 4 * nu) / (8 * (1 - nu)))
    e_sph = abs(S[0, 0] - sph) / sph
    e_cyl = max(abs(S_long[0, 0]), abs(S_long[1, 1] - cyl["S2222"]), abs(S_long[1, 2] - cyl["S2233"]),
                abs(S_long[1, 0] - cyl["S2211"]), abs(S_long[5, 5] / 2 - cyl["S1212"]),
                abs(S_long[3, 3] / 2 - cyl["S2323"]))
    # MT with spheres = Hashin–Shtrikman lower bound (bulk modulus)
    Em, Ef, f = 3e9, 70e9, 0.3
    Km, Gm = Em / (3 * (1 - 2 * nu)), Em / (2 * (1 + nu)); Kf = Ef / (3 * (1 - 2 * nu))
    C = mori_tanaka(iso_C_mandel(Em, nu), iso_C_mandel(Ef, nu), eshelby_prolate(1 + 1e-6, nu), f)
    K_mt = (C[0, 0] + 2 * C[0, 1]) / 3
    K_hs = Km + f * (Kf - Km) / (1 + (1 - f) * (Kf - Km) / (Km + 4 / 3 * Gm))
    Cm, Cf = iso_C_mandel(Em, nu), iso_C_mandel(Ef, nu)
    S16 = eshelby_prolate(16.9, nu)
    e0 = np.max(np.abs(mori_tanaka(Cm, Cf, S16, 0.0) - Cm)) / Cm[0, 0]
    e1 = np.max(np.abs(mori_tanaka(Cm, Cf, S16, 1.0) - Cf)) / Cf[0, 0]
    print(f"[V] Eshelby: sphere limit {e_sph:.1e}, long-cylinder limit {e_cyl:.1e} | MT spheres vs "
          f"HS lower bound (K) {abs(K_mt - K_hs) / K_hs:.1e} | MT f=0 {e0:.1e}, f=1 {e1:.1e}")
    return dict(eshelby_sphere=e_sph, eshelby_cylinder=e_cyl, mt_vs_hs=abs(K_mt - K_hs) / K_hs,
                mt_f0=e0, mt_f1=e1)


# -------------------------------------------------------------- plates --------------
def layers(z_nodes, axx, ayy):
    azz = np.clip(1 - axx - ayy, 0.02, None)
    s = axx + ayy + azz
    return axx / s, ayy / s, azz / s


def layer_props(T, axx, ayy, azz):
    Cud, aL, aT = ud(T)
    Qs, al = [], []
    for x, y, w in zip(axx, ayy, azz):
        a, Cavg = orient_avg_cte(np.diag([x, y, w]), "hybrid", Cud, aL=aL, aT=aT)
        Qs.append(reduced_Q(Cavg)); al.append([a[0, 0], a[1, 1], 0.0])
    return np.array(Qs), np.array(al)


def abd(z, Qs):
    dz = np.gradient(z)
    A = np.einsum("k,kij->ij", dz, Qs); B = np.einsum("k,k,kij->ij", dz, z, Qs)
    D = np.einsum("k,k,kij->ij", dz, z * z, Qs)
    return A, B, D, dz


def plate_cte_modulus(z, axx, ayy, azz, T):
    """Free in-plane expansion per K and in-plane moduli of a plate (equivalent single layer)."""
    Qs, al = layer_props(T, axx, ayy, azz)
    A, B, D, dz = abd(z, Qs)
    ABD = np.block([[A, B], [B, D]])
    N = np.einsum("k,kij,kj->i", dz, Qs, al); M = np.einsum("k,k,kij,kj->i", dz, z, Qs, al)
    e0 = np.linalg.solve(ABD, np.concatenate([N, M]))[:3]
    H = z[-1] - z[0]
    a_inv = np.linalg.inv(ABD)[:3, :3]
    return e0[0], e0[1], 1 / (a_inv[0, 0] * H), 1 / (a_inv[1, 1] * H)


def residual_stress(z, axx, ayy, azz, mode, nT=161):
    """Free plate cooled T0 -> RT. mode 'total': sigma = Q(RT)(eps - int alpha dT);
    mode 'incremental': d sigma = Q(T)(d eps - alpha(T) dT)."""
    Ts = np.linspace(T0, T_RT, nT)
    if mode == "total":
        Qrt, _ = layer_props(T_RT, axx, ayy, azz)
        eth = np.zeros((len(z), 3))
        for T1, T2 in zip(Ts[:-1], Ts[1:]):
            _, a = layer_props(0.5 * (T1 + T2), axx, ayy, azz); eth += a * (T2 - T1)
        A, B, D, dz = abd(z, Qrt)
        N = np.einsum("k,kij,kj->i", dz, Qrt, eth); M = np.einsum("k,k,kij,kj->i", dz, z, Qrt, eth)
        sol = np.linalg.solve(np.block([[A, B], [B, D]]), np.concatenate([N, M]))
        return np.einsum("kij,kj->ki", Qrt, sol[None, :3] + z[:, None] * sol[None, 3:] - eth), sol[3:]
    sig = np.zeros((len(z), 3)); kap = np.zeros(3)
    for T1, T2 in zip(Ts[:-1], Ts[1:]):
        Qs, a = layer_props(0.5 * (T1 + T2), axx, ayy, azz)
        de = a * (T2 - T1)
        A, B, D, dz = abd(z, Qs)
        N = np.einsum("k,kij,kj->i", dz, Qs, de); M = np.einsum("k,k,kij,kj->i", dz, z, Qs, de)
        sol = np.linalg.solve(np.block([[A, B], [B, D]]), np.concatenate([N, M]))
        sig += np.einsum("kij,kj->ki", Qs, sol[None, :3] + z[:, None] * sol[None, 3:] - de)
        kap += sol[3:]
    return sig, kap


def main():
    os.makedirs(OUT, exist_ok=True)
    res = dict(validation=validate(), read_by_eye=dict(fig10=FIG10, fig9_E20=FIG9_E20,
                                                         profile_1mm=PROF_1MM, profile_3mm=PROF_3MM),
               paper_stress=PAPER_STRESS)
    Cud, aL, aT = ud(20.0)
    print(f"aligned GF/PPS at 20 C (Mori–Tanaka, s = {ASPECT:.1f}): E_L {1/np.linalg.inv(Cud)[0,0]/1e9:.2f} GPa, "
          f"E_T {1/np.linalg.inv(Cud)[1,1]/1e9:.2f} GPa, alpha_L {aL*1e6:.1f}, alpha_T {aT*1e6:.1f} ppm/C")
    # ---- C1/C2: 1 mm plate
    z1 = np.linspace(-0.54e-3, 0.54e-3, 109)
    zz = np.abs(z1) * 1e3
    axx = np.interp(-zz, PROF_1MM["z"], PROF_1MM["axx"]); ayy = np.interp(-zz, PROF_1MM["z"], PROF_1MM["ayy"])
    axx, ayy, azz = layers(z1, axx, ayy)
    rows = []
    for T in FIG10["T"]:
        amd, atd, Emd, Etd = plate_cte_modulus(z1, axx, ayy, azz, float(T))
        rows.append(dict(T=T, md=amd * 1e6, td=atd * 1e6, E_md=Emd / 1e9, E_td=Etd / 1e9))
    res["C1_cte_1mm"] = rows
    print("\n[C1] in-plane CTE of the 1.0 mm plate [ppm/C]   (paper values READ BY EYE from Fig. 10)")
    print("   T    this chain MD / TD    paper exp MD / TD    paper sim MD / TD")
    for r, me, te, ms, ts in zip(rows, FIG10["md_exp"], FIG10["td_exp"], FIG10["md_sim"], FIG10["td_sim"]):
        print(f" {r['T']:4.0f}     {r['md']:5.1f} / {r['td']:5.1f}         {me:4.0f} / {te:4.0f}          {ms:4.0f} / {ts:4.0f}")
    print(f"[C2] in-plane modulus at 20 C [GPa]: this chain MD {rows[0]['E_md']:.2f} / TD {rows[0]['E_td']:.2f}; "
          f"paper Fig. 9 secant (by eye) exp MD {FIG9_E20['md_exp']:.1f} / TD {FIG9_E20['td_exp']:.1f}, "
          f"sim MD {FIG9_E20['md_sim']:.1f} / TD {FIG9_E20['td_sim']:.1f}")
    # ---- C3: 3.08 mm plate
    z3 = np.linspace(-H3 / 2, H3 / 2, 155)
    d = (z3 + H3 / 2) * 1e3
    a3x = np.interp(d, PROF_3MM["d"], PROF_3MM["axx"]); a3y = np.interp(d, PROF_3MM["d"], PROF_3MM["ayy"])
    a3x, a3y, a3z = layers(z3, a3x, a3y)
    stress = {}
    for mode in ("total", "incremental"):
        sig, kap = residual_stress(z3, a3x, a3y, a3z, mode)
        stress[mode] = dict(sxx=(sig[:, 0] / 1e6).tolist(), syy=(sig[:, 1] / 1e6).tolist(), kappa=kap.tolist())
        print(f"[C3] {mode:11s}: sigma_xx {sig[:,0].min()/1e6:+6.1f} .. {sig[:,0].max()/1e6:+6.1f} MPa, "
              f"sigma_yy {sig[:,1].min()/1e6:+6.1f} .. {sig[:,1].max()/1e6:+6.1f} MPa, "
              f"surface sigma_xx {sig[0,0]/1e6:+.1f} / sigma_yy {sig[0,1]/1e6:+.1f} MPa")
    print(f"     paper (text, p. 93): surface sigma_xx {PAPER_STRESS['sxx_MPa']:+.1f} / sigma_yy "
          f"{PAPER_STRESS['syy_MPa']:+.1f} MPa — {PAPER_STRESS['note']}")
    res["C3_stress_3mm"] = dict(depth_mm=d.tolist(), axx=a3x.tolist(), ayy=a3y.tolist(), **stress)
    with open(os.path.join(OUT, "compare.json"), "w") as f:
        json.dump(res, f, indent=1)
    _plot(rows, d, a3x, a3y, stress)
    print(f"wrote {OUT}/compare.json and compare.png")


def _plot(rows, d, a3x, a3y, stress):
    sys.path.insert(0, os.path.join(ROOT, "slides", "figure_sources"))
    try:                                   # repo convention: thesis_style, graceful fallback
        from thesis_style import use
        use(width_frac=1.0, aspect=0.4)
        ok = shutil.which("latex") is not None
        if ok and matplotlib.rcParams["text.usetex"]:
            try:
                from matplotlib.texmanager import TexManager
                TexManager().get_text_width_height_descent("lp", 10)
            except Exception:
                ok = False
        if not ok:
            matplotlib.rcParams["text.usetex"] = False
            matplotlib.rcParams["font.family"] = "serif"
            matplotlib.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "STIXGeneral"]
            matplotlib.rcParams["mathtext.fontset"] = "stix"
    except Exception:
        pass
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
    T = [r["T"] for r in rows]
    ax[0].plot(T, [r["md"] for r in rows], "k-o", ms=4, label="this chain, MD")
    ax[0].plot(T, [r["td"] for r in rows], "k--s", ms=4, label="this chain, TD")
    ax[0].plot(FIG10["T"], FIG10["md_exp"], "o", mfc="none", mec="0.5", label="paper exp., MD (read)")
    ax[0].plot(FIG10["T"], FIG10["td_exp"], "s", mfc="none", mec="0.5", label="paper exp., TD (read)")
    ax[0].set_xlabel(r"temperature $T$ [$^\circ$C]"); ax[0].set_ylabel(r"in-plane CTE [ppm/$^\circ$C]")
    ax[0].set_title("(a) 1.0 mm plate", fontsize=8); ax[0].legend(frameon=False, fontsize=6.5)
    ax[1].plot(a3x, d, "k-", label=r"$a_{xx}$"); ax[1].plot(a3y, d, "k--", label=r"$a_{yy}$")
    ax[1].invert_yaxis(); ax[1].set_xlabel("orientation tensor component [$-$]")
    ax[1].set_ylabel("depth from surface [mm]"); ax[1].legend(frameon=False, fontsize=7)
    ax[1].set_title("(b) 3.08 mm plate, read from paper Fig. 8a", fontsize=8)
    for mode, ls in (("total", "-"), ("incremental", "--")):
        ax[2].plot(stress[mode]["sxx"], d, "k" + ls, lw=1.2, label=rf"$\sigma_{{xx}}$, {mode}")
        ax[2].plot(stress[mode]["syy"], d, ls, color="0.55", lw=1.2, label=rf"$\sigma_{{yy}}$, {mode}")
    ax[2].invert_yaxis(); ax[2].axvline(0, color="0.8", lw=0.6)
    ax[2].set_xlabel("residual stress [MPa]"); ax[2].set_title(r"(c) free plate, 180 $\to$ 20 $^\circ$C", fontsize=8)
    ax[2].legend(frameon=False, fontsize=6.5)
    fig.suptitle("Comparison with Tsuda et al., J. Jpn. Soc. Compos. Mater. 52 (2026) 88 — paper values read by eye",
                 fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "compare.png"), dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
