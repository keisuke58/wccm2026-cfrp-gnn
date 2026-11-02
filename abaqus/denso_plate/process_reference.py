"""abaqus/denso_plate/process_reference.py -- Python reference for the Denso PROCESS analysis
(sequential heat transfer -> crystallisation-coupled viscoelastic stress), mirroring the Abaqus
decks written by gen_denso_process.py and the UMAT denso_cryst_umat.f.

Process (all three steps are in both the Abaqus decks and this reference):
  1. MOULD  0-40 s   melt at 380 C between walls held at T_bottom / T_top; in-plane displacement
                     of the part blocked by the cavity (e0 = kappa = 0), thickness free;
  2. EJECT  0.01 s   the in-plane constraint is released (constraint force ramped to zero, as
                     Abaqus does for a removed boundary condition); walls replaced by air;
  3. AIR    1800 s   free cooling by convection (film coefficient H_FILM) on both faces.
Heat: 1-D linear finite elements through the thickness (40 elements, backward Euler, the same
fixed increments as the decks); in-plane uniform, edges adiabatic.
Stress: at 2 Gauss points per layer (as C3D8I through the thickness), a line-by-line vectorised
port of the UMAT update -- Nakamura/Hoffman-Lauritzen crystallisation (explicit in time, as in
the UMAT), generalized Maxwell with WLF x crystallinity shift, orthotropic thermal and
crystallisation eigenstrain -- with sigma_zz = 0 at each point and the plate equilibrium
(N, M) solved exactly per increment. The update is affine in the strain increment, so no
iteration is needed.

Validated (process_reference.py --check):
  P1 the port reproduces the compiled UMAT (gfortran driver, umat_driver.f) at a material point
     over a melt-to-room history (stress, alpha, Maxwell partial stresses);
  P2 elastic limit: with GINF = 1 and no crystallisation the free-plate result equals the
     lamination reference of gen_denso_plate.py for the same eigenstrain;
  P3 superposition: the residual state is linear in the eigenstrain, so total = thermal +
     crystallisation to round-off;
  P4 heat: the 1-D FE solution against the explicit finite differences of the ML script.
NOT validated: the decks in Abaqus itself (fifa), and every material constant below.

Honest scope: the Prony series, WLF constants, crystallinity shift and film coefficient are
ILLUSTRATIVE (the PEEK-like set of ../cfrtp_cryst_peek_hl.inp, with GINF lowered to 0.01 so
the melt carries almost no permanent stiffness); no latent heat, no packing pressure, no
pressure dependence of Tg, no mould deformation, no fibre-orientation change. The cavity
constraint in step 1 is an idealisation (full in-plane fixity, no friction law). The
Python lock-in model (stress-free at T_lock, all crystallisation shrinkage counted) is the
limit this model is compared with -- not a measurement.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (HERE, ROOT, os.path.join(ROOT, "validation"), os.path.join(ROOT, "viz")):
    sys.path.insert(0, p)
from fiber_orientation_cte import ud_stiffness, orient_avg_cte  # noqa: E402
from cfrtp_peek_validation import HL, N_AVRAMI, TABS  # noqa: E402
import denso_residual_strain_ml as ml  # noqa: E402

C = ml.SOURCE
NL = 40
H = C.H
RHO, CP = 1500.0, 1200.0                  # ILLUSTRATIVE; only their product matters here
K_COND = C.DIFF * RHO * CP                # = 0.324 W/(m K), so that k / (rho c) = DIFF
H_FILM = 10.0                             # W/(m^2 K), still air, ILLUSTRATIVE
STEPS = [("MOULD", 0.05, 40.0), ("EJECT", 0.01, 0.01), ("AIR", 5.0, 1800.0)]
# Viscoelastic set (ILLUSTRATIVE, after ../cfrtp_cryst_peek_hl.inp; GINF lowered, sum g = 1)
VE = dict(AGEL=0.0, G0=1.0, GINF=0.01, G=(0.50, 0.30, 0.19), TAU=(5.0, 50.0, 500.0),
          WLF_C1=17.4, WLF_C2=51.6, WLF_TREF=143.0, BX=2.0)
ALPHA0 = 1.0e-3                           # crystallinity seed, as the ML script and the decks
GP = np.array([-1.0, 1.0]) / np.sqrt(3.0)  # 2-point Gauss through each layer


# ------------------------------------------------------------------ material ------
def layer_props(orient_fn, c=C, part="total"):
    """34 UMAT constants per layer (layer mid-depth orientation); part zeroes A or B."""
    zm = np.linspace(-1, 1, NL + 1); zm = 0.5 * (zm[1:] + zm[:-1])
    a11, a22, a33 = orient_fn(zm)
    Cud = ud_stiffness(**c.UD)
    rows = []
    for x, y, w in zip(a11, a22, a33):
        a = np.diag([x, y, w])
        al, Cv = orient_avg_cte(a, "hybrid", Cud, aL=c.A_L, aT=c.A_T)
        be, _ = orient_avg_cte(a, "hybrid", Cud, aL=c.B_L, aT=c.B_T)
        S = np.linalg.inv(Cv)                  # repo Voigt 11,22,33,23,13,12 (engineering)
        E1, E2, E3 = 1 / S[0, 0], 1 / S[1, 1], 1 / S[2, 2]
        A = np.diag(al) * (part != "cryst")
        B = np.diag(be) * c.X_INF * (part != "thermal")   # per unit RELATIVE crystallinity
        p = [E1, E2, E3, -S[0, 1] * E1, -S[0, 2] * E1, -S[1, 2] * E2,
             1 / S[5, 5], 1 / S[4, 4], 1 / S[3, 3], A[0], A[1], A[2], B[0], N_AVRAMI,
             HL["K0"], HL["USTAR_R"], HL["KG"], HL["TM0"], HL["TINF"],
             VE["AGEL"], VE["G0"], VE["GINF"]]
        for g, t in zip(VE["G"], VE["TAU"]):
            p += [g, t]
        p += [VE["WLF_C1"], VE["WLF_C2"], VE["WLF_TREF"], VE["BX"], B[1], B[2]]
        rows.append(p)
    return np.array(rows), dict(zm=zm, a11=a11, a22=a22, a33=a33)


def stiffness_from_props(P):
    """Orthotropic stiffness in UMAT order 11,22,33,12,13,23, built exactly as the UMAT does."""
    n = P.shape[0]; SC = np.zeros((n, 6, 6))
    E1, E2, E3, n12, n13, n23, G12, G13, G23 = (P[:, i] for i in range(9))
    SC[:, 0, 0] = 1 / E1; SC[:, 1, 1] = 1 / E2; SC[:, 2, 2] = 1 / E3
    SC[:, 0, 1] = SC[:, 1, 0] = -n12 / E1
    SC[:, 0, 2] = SC[:, 2, 0] = -n13 / E1
    SC[:, 1, 2] = SC[:, 2, 1] = -n23 / E2
    SC[:, 3, 3] = 1 / G12; SC[:, 4, 4] = 1 / G13; SC[:, 5, 5] = 1 / G23
    return np.linalg.inv(SC)


def umat_coeffs(P, alpha, temp, dtemp, dtime):
    """Everything in the UMAT increment that does not depend on the strain increment."""
    n = P[:, 13]
    TKK = temp + 0.5 * dtemp + TABS
    TM0K, TINFK = P[:, 17] + TABS, P[:, 18] + TABS
    ok = (TKK < TM0K) & (TKK > TINFK)
    TKs = np.where(ok, TKK, 0.5 * (TM0K + TINFK))
    DTU = TM0K - TKs; FF = 2 * TKs / (TM0K + TKs)
    AEXP = np.maximum(-P[:, 15] / (TKs - TINFK) - P[:, 16] / (TKs * DTU * FF), -700.0)
    XK = np.where(ok, P[:, 14] * np.exp(AEXP), 0.0)
    a = np.clip(alpha, 1e-8, 1 - 1e-10)
    argl = np.maximum(-np.log(1 - a), 1e-12)
    dal = np.maximum(n * XK * (1 - a) * argl ** ((n - 1) / n) * dtime, 0.0)
    anew = np.minimum(a + dal, 1.0)
    X = np.clip((anew - P[:, 19]) / (1 - P[:, 19]), 0.0, 1.0)
    G = P[:, 20] + (1 - P[:, 20]) * X
    DTT = temp + 0.5 * dtemp - P[:, 30]
    DEN = np.maximum(P[:, 29] + DTT, 1.0)
    AT = 10.0 ** np.clip(-P[:, 28] * DTT / DEN, -30.0, 30.0)
    AX = 10.0 ** np.minimum(P[:, 31] * anew, 30.0)
    gk = P[:, [22, 24, 26]]; tauk = P[:, [23, 25, 27]]
    TAUS = np.maximum((AT * AX)[:, None] * tauk, 1e-20)
    XR = dtime / TAUS
    EXPK = np.exp(-XR)
    BETAK = np.where(XR > 1e-6, (1 - EXPK) / np.where(XR > 1e-6, XR, 1.0), 1 - 0.5 * XR)
    deig = np.zeros((P.shape[0], 6))
    deig[:, 0] = P[:, 9] * dtemp + P[:, 12] * dal
    deig[:, 1] = P[:, 10] * dtemp + P[:, 32] * dal
    deig[:, 2] = P[:, 11] * dtemp + P[:, 33] * dal
    GTAN = P[:, 21] + np.sum(gk * BETAK, axis=1)
    return dict(alpha=anew, G=G, AT=AT, AX=AX, EXPK=EXPK, BETAK=BETAK, gk=gk, deig=deig,
                GTAN=GTAN)


def umat_update(P, CC, stress, q, cf, dstran):
    """Stress and Maxwell partial stresses after the increment (UMAT lines, vectorised)."""
    DS = cf["G"][:, None] * np.einsum("nij,nj->ni", CC, dstran - cf["deig"])
    sumq = q.sum(axis=2)
    qn = cf["EXPK"][:, None, :] * q + (cf["gk"] * cf["BETAK"])[:, None, :] * DS[:, :, None]
    snew = stress - sumq + P[:, 21][:, None] * DS + qn.sum(axis=2)
    return snew, qn


# ------------------------------------------------------------------ heat ----------
def heat_1d(T_bot, T_top, lumped=True, steps=STEPS):
    """Nodal temperatures (NL+1) at the end of every increment, with times; backward Euler."""
    h = H / NL; nn = NL + 1
    K = np.zeros((nn, nn)); Cm = np.zeros((nn, nn))
    ke = K_COND / h * np.array([[1, -1], [-1, 1]])
    ce = RHO * CP * h * (np.array([[0.5, 0], [0, 0.5]]) if lumped else
                         np.array([[2, 1], [1, 2]]) / 6)
    for e in range(NL):
        K[e:e + 2, e:e + 2] += ke; Cm[e:e + 2, e:e + 2] += ce
    T = np.full(nn, C.T_MELT); t = 0.0
    times, hist = [0.0], [T.copy()]
    for name, dt, period in steps:
        A = Cm / dt + K; f0 = np.zeros(nn)
        if name != "MOULD":
            A = A.copy(); A[0, 0] += H_FILM; A[-1, -1] += H_FILM
            f0[0] = f0[-1] = H_FILM * C.T_ROOM
        for _ in range(int(round(period / dt))):
            b = Cm @ T / dt + f0
            if name == "MOULD":
                AA = A.copy(); bb = b.copy()
                for i, v in ((0, T_bot), (nn - 1, T_top)):
                    bb -= AA[:, i] * v; AA[i, :] = 0; AA[:, i] = 0; AA[i, i] = 1; bb[i] = v
                T = np.linalg.solve(AA, bb)
            else:
                T = np.linalg.solve(A, b)
            t += dt; times.append(t); hist.append(T.copy())
    return np.array(times), np.array(hist)


def step_of_increment(steps=STEPS):
    out = []
    for name, dt, period in steps:
        out += [(name, dt)] * int(round(period / dt))
    return out


# ------------------------------------------------------------------ plate ---------
def plate_history(P, times, Tn, steps=STEPS, record=False):
    """Through-thickness VE stress history of the plate; Tn: nodal temperatures per time."""
    h = H / NL
    zl = -H / 2 + h * (np.arange(NL) + 0.5)
    zp = (zl[:, None] + GP[None, :] * h / 2).ravel()          # 2 points per layer
    xi = (zp - (-H / 2 + h * np.repeat(np.arange(NL), 2))) / h  # local coordinate in layer
    kk = np.repeat(np.arange(NL), 2)
    Pp = np.repeat(P, 2, axis=0)
    CC = stiffness_from_props(Pp)
    w = np.full(zp.size, h / 2)
    Tp = lambda T: T[kk] * (1 - xi) + T[kk + 1] * xi
    npt = zp.size
    sig = np.zeros((npt, 6)); q = np.zeros((npt, 6, 3)); alpha = np.full(npt, ALPHA0)
    e0 = np.zeros(2); kap = np.zeros(2)
    incs = step_of_increment(steps)
    assert len(incs) == len(times) - 1
    rec = {}
    prev_step = None
    for n, (name, dt) in enumerate(incs):
        T0, T1 = Tp(Tn[n]), Tp(Tn[n + 1])
        cf = umat_coeffs(Pp, alpha, T0, T1 - T0, dt)
        K = cf["G"] * cf["GTAN"]
        R = np.einsum("nik,nk->ni", q, 1 - cf["EXPK"])
        D = cf["deig"]
        Qr = CC[:, :2, :2] - CC[:, :2, 2:3] * CC[:, 2:3, :2] / CC[:, 2:3, 2:3]
        c = CC[:, :2, 2] * (R[:, 2] / CC[:, 2, 2])[:, None] - R[:, :2]
        KQ = K[:, None, None] * Qr
        base = -np.einsum("nij,nj->ni", KQ, D[:, :2]) + c     # stress incr. at zero strain incr.
        if name == "MOULD":
            de0 = np.zeros(2); dk = np.zeros(2)
        else:
            if name == "EJECT" and prev_step != "EJECT":
                N0 = np.einsum("n,ni->i", w, sig[:, :2]); M0 = np.einsum("n,n,ni->i", w, zp, sig[:, :2])
                nsub = int(round(dict((s[0], s[2] / s[1]) for s in steps)["EJECT"]))
                dN, dM = -N0 / nsub, -M0 / nsub
            elif name != "EJECT":
                dN = dM = np.zeros(2)
            A = np.einsum("n,nij->ij", w, KQ); B = np.einsum("n,n,nij->ij", w, zp, KQ)
            Dm = np.einsum("n,n,nij->ij", w, zp ** 2, KQ)
            F0 = np.einsum("n,ni->i", w, base); F1 = np.einsum("n,n,ni->i", w, zp, base)
            sol = np.linalg.solve(np.block([[A, B], [B, Dm]]),
                                  np.concatenate([dN - F0, dM - F1]))
            de0, dk = sol[:2], sol[2:]
        dei = de0[None, :] + zp[:, None] * dk[None, :]
        de3 = D[:, 2] + (R[:, 2] / K - np.einsum("nj,nj->n", CC[:, 2, :2], dei - D[:, :2])) / CC[:, 2, 2]
        dstran = np.zeros((npt, 6)); dstran[:, :2] = dei; dstran[:, 2] = de3
        sig, q = umat_update(Pp, CC, sig, q, cf, dstran)
        alpha = cf["alpha"]; e0 = e0 + de0; kap = kap + dk
        prev_step = name
        if record and (n + 1 == len(incs) or incs[n + 1][0] != name):
            rec[name] = dict(sig=sig.copy(), alpha=alpha.copy(), e0=e0.copy(), kap=kap.copy())
    Qg = CC[:, :2, :2] - CC[:, :2, 2:3] * CC[:, 2:3, :2] / CC[:, 2:3, 2:3]   # glassy, g = 1
    eres = np.linalg.solve(Qg, sig[:, :2, None])[:, :, 0]
    lay = lambda v: v.reshape(NL, 2, *v.shape[1:]).mean(axis=1)            # layer = IP mean
    return dict(zp=zp, sig=sig, alpha=alpha, e0=e0, kappa=kap, eres=eres,
                sig_layer=lay(sig), alpha_layer=lay(alpha), eres_layer=lay(eres),
                sigzz_max=float(np.max(np.abs(sig[:, 2]))), steps=rec)


def run_case(orient_fn, T_bot, T_top, part="total", lumped=True):
    P, geo = layer_props(orient_fn, part=part)
    times, Tn = heat_1d(T_bot, T_top, lumped=lumped)
    out = plate_history(P, times, Tn, record=True)
    out.update(P=P, geo=geo, times=times, Tn=Tn)
    return out


# ------------------------------------------------------------------ checks --------
def check_p1(P, workdir):
    """Port vs compiled UMAT at one material point (prescribed strain + temperature)."""
    drv = os.path.join(HERE, "umat_driver.f"); umat = os.path.join(HERE, "denso_cryst_umat.f")
    exe = os.path.join(workdir, "umat_driver")
    os.makedirs(workdir, exist_ok=True)
    with open(os.path.join(workdir, "ABA_PARAM.INC"), "w") as f:
        f.write("      IMPLICIT REAL*8(A-H,O-Z)\n      PARAMETER (NPRECD=2)\n")
    subprocess.run(["gfortran", "-O0", "-ffixed-line-length-72", "-I", workdir, "-o", exe,
                    drv, umat], check=True)
    rng = np.random.default_rng(0)
    nstep = 400
    t = np.linspace(0, 400.0, nstep + 1)
    T = 25 + (380 - 25) * np.exp(-t / 60.0)                 # melt -> near room
    eps = np.cumsum(rng.normal(0, 2e-5, (nstep + 1, 6)), axis=0); eps[0] = 0
    p = P[:1]
    inp = os.path.join(workdir, "hist.txt")
    with open(inp, "w") as f:
        f.write(f"{nstep}\n" + " ".join(f"{v:.17e}" for v in p[0]) + "\n")
        for n in range(nstep):
            f.write(" ".join(f"{v:.17e}" for v in [t[n + 1] - t[n], T[n], T[n + 1] - T[n],
                                                      *(eps[n + 1] - eps[n])]) + "\n")
    outf = subprocess.run([exe, inp], check=True, capture_output=True, text=True).stdout
    F = np.array([[float(x) for x in l.split()] for l in outf.strip().splitlines()])
    CC = stiffness_from_props(p)
    sig = np.zeros((1, 6)); q = np.zeros((1, 6, 3)); a = np.full(1, ALPHA0); Pyr = []
    for n in range(nstep):
        cf = umat_coeffs(p, a, np.array([T[n]]), np.array([T[n + 1] - T[n]]), t[n + 1] - t[n])
        sig, q = umat_update(p, CC, sig, q, cf, (eps[n + 1] - eps[n])[None])
        a = cf["alpha"]
        Pyr.append(np.concatenate([sig[0], [a[0]], q[0].T.ravel()]))
    Pyr = np.array(Pyr)
    scale = np.max(np.abs(Pyr[:, :6]))
    err_s = np.max(np.abs(F[:, :6] - Pyr[:, :6])) / scale
    err_a = np.max(np.abs(F[:, 6] - Pyr[:, 6]))
    err_q = np.max(np.abs(F[:, 7:] - Pyr[:, 7:])) / scale
    return err_s, err_a, err_q, float(Pyr[-1, 6])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="run P1-P4")
    ap.add_argument("--summary", action="store_true",
                    help="VE process vs lock-in model for both cases, GINF sensitivity, figure")
    ap.add_argument("--out", default=os.path.join(HERE, "process_check"))
    a = ap.parse_args()
    from denso_plate_features import orientation_profile, CASES
    import gen_denso_plate as gp
    os.makedirs(a.out, exist_ok=True)
    P, _ = layer_props(orientation_profile)
    if a.summary:
        summary(a.out)
    if a.check:
        try:
            es, ea, eq, af = check_p1(P, os.path.join(a.out, "umat_build"))
            print(f"[P1] port vs compiled UMAT: stress {es:.1e} (rel), alpha {ea:.1e}, "
                  f"Maxwell q {eq:.1e} (rel); final alpha {af:.3f}")
        except (OSError, subprocess.CalledProcessError) as exc:
            print(f"[P1] skipped (needs gfortran): {exc}")
        # P2 elastic limit: GINF = 1, no crystallisation, uniform cool T_lock -> T_room in AIR
        op = (0.25, 0.80, 0.55, 0.35, 0.15, 0.06)                # core shifted: kappa != 0
        ofn = lambda z: ml.orientation(z, op)
        Pe, _ = layer_props(ofn); Pe[:, 21] = 1.0; Pe[:, [22, 24, 26]] = 0.0; Pe[:, 14] = 0.0
        steps = [("MOULD", 1.0, 1.0), ("EJECT", 1.0, 1.0), ("AIR", 1.0, 1.0)]
        Tn = np.array([np.full(NL + 1, v) for v in (C.T_LOCK, C.T_LOCK, C.T_LOCK, C.T_ROOM)])
        r = plate_history(Pe, np.arange(4.0), Tn, steps=steps)
        lay = gp.layer_data(ofn, np.zeros(NL + 1), C, part="thermal")
        ref = gp.lamination_pc(lay["zm"], lay["C"], lay["estar"], C.H)
        # IP-pair mean of the linear-in-z strain equals the layer-mid value
        e2 = max(np.max(np.abs(r["eres_layer"][:, 0] - ref["em"][:, 0])),
                 np.max(np.abs(r["eres_layer"][:, 1] - ref["em"][:, 1]))) / np.max(np.abs(ref["em"][:, :2]))
        print(f"[P2] elastic limit vs lamination reference: strain {e2:.1e} (rel), kappa_y "
              f"{r['kappa'][1]:.9e} vs {ref['kappa'][1]:.9e} 1/m, max |sigma_zz| {r['sigzz_max']:.1e} Pa")
        # P3 superposition and P4 heat on the asymmetric case
        Tb, Tt = CASES["asymmetric"]
        tot = run_case(orientation_profile, Tb, Tt, "total")
        th = run_case(orientation_profile, Tb, Tt, "thermal")
        cr = run_case(orientation_profile, Tb, Tt, "cryst")
        e3 = np.max(np.abs(th["sig"] + cr["sig"] - tot["sig"])) / np.max(np.abs(tot["sig"]))
        print(f"[P3] superposition total - (thermal + cryst): {e3:.1e} (rel)")
        Xc_fd = ml.crystallise(Tb, Tt, C)
        tl, Tl = heat_1d(Tb, Tt, lumped=True); tc, Tc = heat_1d(Tb, Tt, lumped=False)
        i40 = int(np.argmin(np.abs(tl - 40.0)))
        _, Tfd = gp_heat_fd(Tb, Tt)
        print(f"[P4] T at 40 s, FE lumped vs explicit FD: {np.max(np.abs(Tl[i40] - Tfd)):.2f} K; "
              f"lumped vs consistent: {np.max(np.abs(Tl[i40] - Tc[i40])):.2f} K; "
              f"T_max at end {Tl[-1].max():.2f} C")
        al_fd = Xc_fd / C.X_INF
        al_ve = tot["steps"]["MOULD"]["alpha"]
        al_nodes = np.interp(np.linspace(-1, 1, NL + 1) * H / 2, tot["zp"], al_ve)
        print(f"     alpha at 40 s, UMAT points vs FD nodes: max diff "
              f"{np.max(np.abs(al_nodes - al_fd)):.3f}; final alpha {tot['alpha'].min():.3f}.."
              f"{tot['alpha'].max():.3f}")


def summary(outdir):
    """Numbers and figure for the slides: VE process model vs the lock-in elastic model."""
    from denso_plate_features import orientation_profile, CASES
    import gen_denso_plate as gp
    L = gp.L_PLATE
    sag = lambda k: float(k * L ** 2 / 8 * 1e3)
    rng = lambda v: float((np.max(v) - np.min(v)) * 1e3)
    res = dict(constants=dict(VE=VE, H_FILM=H_FILM, STEPS=STEPS, note="ILLUSTRATIVE"), cases={})
    prof = {}
    for case, (Tb, Tt) in CASES.items():
        Xc = ml.crystallise(Tb, Tt, C)
        d = {}
        for part in ("total", "thermal", "cryst"):
            r = run_case(orientation_profile, Tb, Tt, part)
            lay = gp.layer_data(orientation_profile, Xc, C, part=part)
            ref = gp.lamination_pc(lay["zm"], lay["C"], lay["estar"], C.H)
            d[part] = dict(ve_range_xx=rng(r["eres_layer"][:, 0]), ve_range_yy=rng(r["eres_layer"][:, 1]),
                           ve_sag_y=sag(r["kappa"][1]), lock_range_xx=rng(ref["em"][:, 0]),
                           lock_range_yy=rng(ref["em"][:, 1]), lock_sag_y=sag(ref["kappa"][1]))
            if part == "total":
                prof[case] = dict(z=lay["zm"], ve=r["eres_layer"][:, 0] * 1e3, lock=ref["em"][:, 0] * 1e3,
                                  a_mould=r["steps"]["MOULD"]["alpha"].reshape(NL, 2).mean(1),
                                  a_end=r["alpha_layer"], a_lock=lay["Xc"] / C.X_INF,
                                  sxx=r["sig_layer"][:, 0] / 1e6)
                d["alpha_end_range"] = [float(r["alpha_layer"].min()), float(r["alpha_layer"].max())]
                d["sigma_xx_range_MPa"] = [float(r["sig_layer"][:, 0].min() / 1e6),
                                           float(r["sig_layer"][:, 0].max() / 1e6)]
        res["cases"][case] = d
    Tb, Tt = CASES["asymmetric"]; sweep = []
    for ginf in (1e-4, 1e-3, 1e-2, 1e-1):
        row = dict(GINF=ginf)
        for part in ("total", "thermal", "cryst"):
            P, _ = layer_props(orientation_profile, part=part)
            P[:, 21] = ginf; P[:, [22, 24, 26]] = np.array(VE["G"]) / sum(VE["G"]) * (1 - ginf)
            t, Tn = heat_1d(Tb, Tt); r = plate_history(P, t, Tn)
            row[part + "_range_xx"] = rng(r["eres_layer"][:, 0]); row[part + "_sag_y"] = sag(r["kappa"][1])
        sweep.append(row)
    res["ginf_sweep_asymmetric"] = sweep
    with open(os.path.join(outdir, "process_summary.json"), "w") as f:
        json.dump(res, f, indent=1)
    for case, d in res["cases"].items():
        for part in ("total", "thermal", "cryst"):
            x = d[part]
            print(f"{case:10s} {part:7s}: e_res,xx range VE {x['ve_range_xx']:.3f} / lock-in "
                  f"{x['lock_range_xx']:.3f} x1e-3 | sag_y VE {x['ve_sag_y']:+.3f} / lock-in {x['lock_sag_y']:+.3f} mm")
    for row in sweep:
        print(f"GINF {row['GINF']:.0e}: range total {row['total_range_xx']:.3f} cryst {row['cryst_range_xx']:.3f}"
              f" | sag total {row['total_sag_y']:+.3f} thermal {row['thermal_sag_y']:+.3f} cryst {row['cryst_sag_y']:+.3f}")
    _figure(prof, os.path.join(outdir, "denso_process_prediction.png"))


def _figure(prof, path):
    import shutil
    import matplotlib
    matplotlib.use("Agg")
    try:                                   # repo convention: thesis_style, graceful fallback
        sys.path.insert(0, os.path.join(ROOT, "slides", "figure_sources"))
        from thesis_style import use
        use(width_frac=1.0, aspect=0.45)
        if shutil.which("latex") is None:
            matplotlib.rcParams["text.usetex"] = False
            matplotlib.rcParams["font.family"] = "serif"
            matplotlib.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "STIXGeneral"]
            matplotlib.rcParams["mathtext.fontset"] = "stix"
    except Exception:
        pass
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.7), sharey=True)
    p = prof["asymmetric"]; z = p["z"]
    ax[0].plot(p["a_lock"], z, "k:", lw=1.2, label="end of mould (1-D FD, lock-in model)")
    ax[0].plot(p["a_mould"], z, "k--", lw=1.0, label="end of mould (process model)")
    ax[0].plot(p["a_end"], z, "k-", lw=1.4, label="after air cooling (process model)")
    ax[0].set_xlabel(r"relative crystallinity $\alpha$ [$-$]")
    ax[0].set_ylabel(r"normalised thickness coordinate $2z/H$")
    ax[0].set_title("(a) crystallinity, asymmetric mould", fontsize=8)
    for k, (case, ls) in enumerate((("symmetric", "-"), ("asymmetric", "-"))):
        q = prof[case]
        ax[1 + k].plot(q["lock"], z, color="0.55", ls="--", lw=1.2, label="lock-in elastic model")
        ax[1 + k].plot(q["ve"], z, "k-", lw=1.4, label="viscoelastic process model")
        ax[1 + k].set_xlabel(r"residual strain $\varepsilon_{\mathrm{res},xx}$ [$10^{-3}$]")
        ax[1 + k].set_title(f"({'bc'[k]}) {case} mould", fontsize=8)
    ax[0].legend(frameon=False, fontsize=6.5, loc="lower right")
    ax[1].legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.suptitle("Python reference of the Abaqus process decks (ILLUSTRATIVE constants; Abaqus not yet run)",
                 fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
    print("wrote", path)


def gp_heat_fd(Tb, Tt):
    """The explicit FD of the ML script, returning the end-of-mould temperatures."""
    c = C; dz = c.H / (ml.NZ - 1); dt = 0.4 * dz * dz / c.DIFF
    T = np.full(ml.NZ, c.T_MELT); T[0], T[-1] = Tb, Tt; t = 0.0
    while t < c.T_MOULD_S:
        T[1:-1] = T[1:-1] + c.DIFF * dt / dz ** 2 * (T[2:] - 2 * T[1:-1] + T[:-2]); t += dt
    return t, T


if __name__ == "__main__":
    main()
