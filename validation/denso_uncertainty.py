"""validation/denso_uncertainty.py -- how wide is the prediction?  Uncertainty of the stress-free temperature, the
residual stress and the strength of the GF/PPS plate, and how much the measured released strain narrows it.

Chain (the same physics as validation/denso_rescue_released.py and validation/denso_strength_residual.py):
  constants c, conditions theta = (T0_eff, Vf, s, w, m)
  -> eigenstrain e*(z) = int_{T0}^{20 C} alpha(T, z) dT   (Mori-Tanaka + orientation averaging)
  -> released strain of the hole drilling  y = G_xx ebar_xx + G_yy ebar_yy       (fifa §16 operator)      (1)
  -> residual stress of the free plate, sigma = Q(20 C) (eps - e*)
  -> strength, criterion A of denso_strength_residual.py (strain limit eps_c, set per sample so that the
     stress-free plate fails at the sampled datasheet strength in MD).
For speed, int alpha dT is tabulated once per sample on a 2 K grid (20..260 C) and interpolated in T0.

Checks and results (printed, written to results/denso_uncertainty/uncertainty.json):
  U0 the tabulated chain at the nominal constants reproduces T0_eff = 143 C (denso_rescue_released.py R4a) and the
     strengths 125 / 123 MPa (denso_strength_residual.py, A) -- consistency, not validation;
  U1 interval of T0_eff from the measurement at nominal constants: chi^2(T0) with sigma = the RMSE at the optimum
     (i.e. chi^2 / dof = 1; the reading errors are correlated, so the nominal 10 micro-strain would be too narrow),
     68 % interval where chi^2 - chi^2_min <= 1;
  U2 Monte Carlo over the constants (ranges below, uniform, ALL ASSUMED) in two cases:
       P  "prediction only": T0_eff = 180 C as assumed in the paper, no measurement;
       I  "with the measurement": T0_eff re-identified per sample from the measured released strain plus a fresh
          noise draw (10 micro-strain, the reading uncertainty);
     reported: median and 5-95 % band of T0_eff, the max residual tension, the strength MD / TD and its change by
     the residual stress;
  U3 which constant matters: Spearman rank correlation of each input with each output (case I);
  U4 the 3-D parts (Greif & Meyer, Moldflow data, not measurements): spread over the 196 shapes of the error after
     the 10-point correction and of the no-measurement (leave-one-out factor) prediction -- the band to expect for
     a new part.
Constant ranges (uniform, assumed, not from data): matrix modulus x 0.85-1.15, matrix CTE x 0.90-1.10, glass
modulus x 0.95-1.05, Vf 0.15-0.21, aspect ratio 12-22, core width w 0.75-1.25, orientation strength m 0.85-1.10
(the ranges of denso_pretrain_surrogate.py), datasheet strength 155 MPa x 0.90-1.10 (proxy grade 1130T6).
Honest scope: the operator G carries the paper's stiffness profile (a uniform modulus scale leaves it exact, a
change of the profile does not); one measured plate, read by eye; elastic chain; uniform independent ranges that
are guesses; criterion A only (criterion B failed, MEMO_denso_strength.md); the strength band is that of the MODEL,
not a measured scatter.
Run:  python3 validation/denso_uncertainty.py --n 300 --workers 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "denso_uncertainty")
sys.path.insert(0, HERE)
import denso_rescue_released as rr  # noqa: E402
import denso_strength_residual as sr  # noqa: E402
import denso_tsuda2026_compare as tc  # noqa: E402

BASE = dict(PPS_E=tc.PPS_E.copy(), PPS_A=tc.PPS_A.copy(), GF_E=tc.GF["E"])
NAMES = ["kE_matrix", "kA_matrix", "kE_glass", "Vf", "s", "w", "m", "k_datasheet"]
LO = np.array([0.85, 0.90, 0.95, 0.15, 12.0, 0.75, 0.85, 0.90])
HI = np.array([1.15, 1.10, 1.05, 0.21, 22.0, 1.25, 1.10, 1.10])
NOM = np.array([1.0, 1.0, 1.0, tc._m["fibre_volume_fraction"], tc._m["aspect_ratio_mean"], 1.0, 1.0, 1.0])
TG = np.arange(tc.T_RT, 260.0 + 1e-9, 2.0)
SIG_Y = 10.0                                                    # micro-strain, reading uncertainty
T_PAPER = 180.0


def setup(c):
    tc.PPS_E = BASE["PPS_E"] * c[0]; tc.PPS_A = BASE["PPS_A"] * c[1]; tc.GF["E"] = BASE["GF_E"] * c[2]
    tc.VF, tc.ASPECT = float(c[3]), float(c[4])


def plate(c):
    """layer orientations as in denso_rescue_released.eigenstrain (w, m applied to the paper's profile)."""
    zq = np.linspace(0.0, rr.ps.HMM, rr.NZ); mid = rr.ps.HMM / 2
    src = np.clip(mid + (zq - mid) / c[5], 0.0, rr.ps.HMM)
    axx = np.interp(src, rr.ps.DEP, rr.ps.AXX); ayy = np.interp(src, rr.ps.DEP, rr.ps.AYY)
    ax2 = np.clip(1 / 3 + c[6] * (axx - 1 / 3), 0.02, 0.96)
    ay2 = np.clip(ayy * (1 - ax2) / np.maximum(1 - axx, 1e-6), 0.02, 0.96)
    z = (zq - mid) * 1e-3
    return zq, z, tc.layers(z, ax2, ay2)


class Chain:
    """everything that depends on the constants only; T0 enters by interpolation of the tabulated integral."""

    def __init__(self, c):
        setup(c)
        self.c = c
        self.zq, self.z, (a1, a2, a3) = plate(c)
        cum = np.zeros((len(TG), rr.NZ, 3))
        for i in range(1, len(TG)):
            _, al = tc.layer_props(0.5 * (TG[i - 1] + TG[i]), a1, a2, a3); cum[i] = cum[i - 1] + al * (TG[i] - TG[i - 1])
        self.cum = cum                                         # int_{20}^{T} alpha dT
        self.Q, _ = tc.layer_props(tc.T_RT, a1, a2, a3)
        self.A, self.B, self.D, self.dz = tc.abd(self.z, self.Q)
        self.H = self.z[-1] - self.z[0]
        e1 = np.linalg.solve(self.A, np.array([1.0, 0, 0]))
        self.ec = float(np.max(sr.max_principal(sr.SIG_MD_DATASHEET * c[7] * self.H * e1[None, :].repeat(rr.NZ, 0))))
        self.band = np.minimum((self.zq // 0.2).astype(int), rr.NB - 1)

    def eth(self, T0):
        i = np.clip(np.searchsorted(TG, T0) - 1, 0, len(TG) - 2); f = (T0 - TG[i]) / (TG[i + 1] - TG[i])
        return -((1 - f) * self.cum[i] + f * self.cum[i + 1])  # int_{T0}^{20} alpha dT

    def released(self, T0):
        e = self.eth(T0)
        ex = np.array([e[self.band == b, 0].mean() for b in range(rr.NB)]); ey = np.array([e[self.band == b, 1].mean() for b in range(rr.NB)])
        return (rr.G["xx"] @ (ex / 1e-3) + rr.G["yy"] @ (ey / 1e-3)).reshape(-1)

    def identify(self, y):
        f = lambda T: float(np.sum((self.released(T) - y) ** 2))  # noqa: E731
        Ts = np.arange(80.0, 260.0 + 1e-9, 5.0); k = int(np.argmin([f(T) for T in Ts]))
        lo, hi = Ts[max(k - 1, 0)], Ts[min(k + 1, len(Ts) - 1)]
        g = (np.sqrt(5) - 1) / 2
        for _ in range(40):
            a, b = hi - g * (hi - lo), lo + g * (hi - lo)
            lo, hi = (lo, b) if f(a) < f(b) else (a, hi)
        T = 0.5 * (lo + hi)
        return T, float(np.sqrt(f(T) / len(y)))

    def strength(self, T0):
        e = self.eth(T0); Q, z, dz = self.Q, self.z, self.dz
        N = np.einsum("k,kij,kj->i", dz, Q, e); M = np.einsum("k,k,kij,kj->i", dz, z, Q, e)
        sol = np.linalg.solve(np.block([[self.A, self.B], [self.B, self.D]]), np.concatenate([N, M]))
        sres = np.einsum("kij,kj->ki", Q, sol[None, :3] + z[:, None] * sol[None, 3:] - e)
        e_res = np.linalg.solve(Q, sres[..., None])[..., 0]
        md, _ = sr.failure(Q, self.A, self.H, e_res, 0, self.ec); td, _ = sr.failure(Q, self.A, self.H, e_res, 1, self.ec)
        md0, _ = sr.failure(Q, self.A, self.H, 0 * e_res, 0, self.ec); td0, _ = sr.failure(Q, self.A, self.H, 0 * e_res, 1, self.ec)
        return dict(MD=md / 1e6, TD=td / 1e6, dMD=md / md0 - 1, dTD=td / td0 - 1,
                    smax=float(max(sres[:, 0].max(), sres[:, 1].max()) / 1e6))


def sample(args):
    c, seed = args
    ch = Chain(np.asarray(c)); rng = np.random.default_rng(seed)
    T_I, rm = ch.identify(rr.Y_MEAS + rng.normal(0, SIG_Y, rr.Y_MEAS.shape))
    return dict(c=list(map(float, c)), T0_I=T_I, rmse_I=rm, P=ch.strength(T_PAPER), I=ch.strength(T_I))


def q(v):
    v = np.asarray(v, float)
    return dict(median=float(np.median(v)), p05=float(np.percentile(v, 5)), p95=float(np.percentile(v, 95)))


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261108)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    out = {}
    # U0 consistency at nominal constants
    ch = Chain(NOM)
    T_nom, rm_nom = ch.identify(rr.Y_MEAS)
    s_nom = ch.strength(T_nom); s180 = ch.strength(T_PAPER)
    ref = json.load(open(os.path.join(ROOT, "results", "denso_strength", "strength.json")))["cases"]
    ref_I = [v for k, v in ref.items() if k.startswith("T0_eff")][0]
    out["U0"] = dict(T0=T_nom, rmse=rm_nom, T0_ref=sr.T0_ID,
                     MD=s_nom["MD"], TD=s_nom["TD"], MD_ref=ref_I["MD"]["strength_MPa"], TD_ref=ref_I["TD"]["strength_MPa"],
                     MD180=s180["MD"], MD180_ref=ref["T0 = 180 C (paper)"]["MD"]["strength_MPa"])
    print(f"[U0] nominal constants: T0_eff {T_nom:.1f} C (ref {sr.T0_ID:.1f}), RMSE {rm_nom:.1f} micro-strain; strength MD "
          f"{s_nom['MD']:.1f} / TD {s_nom['TD']:.1f} MPa (ref {ref_I['MD']['strength_MPa']:.1f} / {ref_I['TD']['strength_MPa']:.1f}); "
          f"at 180 C MD {s180['MD']:.1f} (ref {out['U0']['MD180_ref']:.1f})")
    # U1 chi^2 interval of T0 at nominal constants
    Ts = np.arange(100.0, 200.0 + 1e-9, 0.25)
    chi = np.array([np.sum((ch.released(T) - rr.Y_MEAS) ** 2) for T in Ts]) / rm_nom ** 2
    inside = Ts[chi - chi.min() <= 1.0]
    out["U1"] = dict(sigma=rm_nom, lo=float(inside.min()), hi=float(inside.max()))
    print(f"[U1] T0_eff from the measurement at nominal constants: {T_nom:.1f} C, 68 % interval {inside.min():.1f} - "
          f"{inside.max():.1f} C (sigma = RMSE at optimum {rm_nom:.1f} micro-strain)")
    # U2 Monte Carlo
    rng = np.random.default_rng(a.seed)
    C = LO + (HI - LO) * rng.random((a.n, len(NAMES)))
    t0 = time.time()
    with Pool(a.workers) as p:
        R = p.map(sample, [(tuple(c), a.seed + i) for i, c in enumerate(C)])
    print(f"[U2] {a.n} samples in {time.time() - t0:.0f} s")
    U2 = dict(T0_I=q([r["T0_I"] for r in R]), rmse_I=q([r["rmse_I"] for r in R]))
    for case in ("P", "I"):
        U2[case] = {k: q([r[case][k] for r in R]) for k in ("MD", "TD", "dMD", "dTD", "smax")}
    out["U2"] = U2
    print(f"     T0_eff identified: median {U2['T0_I']['median']:.0f} C, 5-95 % {U2['T0_I']['p05']:.0f} - {U2['T0_I']['p95']:.0f} C")
    for case, lab in (("P", "P prediction only (T0 = 180 C)"), ("I", "I with the measurement")):
        u = U2[case]
        print(f"     {lab:32s}: max residual tension {u['smax']['median']:.1f} [{u['smax']['p05']:.1f}, {u['smax']['p95']:.1f}] MPa; "
              f"strength MD {u['MD']['median']:.0f} [{u['MD']['p05']:.0f}, {u['MD']['p95']:.0f}], TD {u['TD']['median']:.0f} "
              f"[{u['TD']['p05']:.0f}, {u['TD']['p95']:.0f}] MPa; change MD {u['dMD']['median'] * 100:+.0f} "
              f"[{u['dMD']['p05'] * 100:+.0f}, {u['dMD']['p95'] * 100:+.0f}] %, TD {u['dTD']['median'] * 100:+.0f} "
              f"[{u['dTD']['p05'] * 100:+.0f}, {u['dTD']['p95'] * 100:+.0f}] %")
    wP = (U2["P"]["smax"]["p95"] - U2["P"]["smax"]["p05"]); wI = (U2["I"]["smax"]["p95"] - U2["I"]["smax"]["p05"])
    out["U2"]["width_ratio_smax_I_over_P"] = wI / wP
    print(f"     5-95 % width of the max residual tension: P {wP:.1f} MPa -> I {wI:.1f} MPa (ratio {wI / wP:.2f})")
    # U3 sensitivity
    U3 = {}
    for k_out, get in (("T0_I", lambda r: r["T0_I"]), ("smax_I", lambda r: r["I"]["smax"]), ("dMD_I", lambda r: r["I"]["dMD"]),
                       ("MD_I", lambda r: r["I"]["MD"]), ("smax_P", lambda r: r["P"]["smax"])):
        y = np.array([get(r) for r in R])
        U3[k_out] = {nm: spearman(C[:, j], y) for j, nm in enumerate(NAMES)}
        top = sorted(U3[k_out].items(), key=lambda kv: -abs(kv[1]))[:3]
        print(f"[U3] {k_out:7s}: " + ", ".join(f"{nm} {v:+.2f}" for nm, v in top))
    out["U3"] = U3
    # U4 3-D parts
    P3 = json.load(open(os.path.join(ROOT, "abaqus", "denso_plate", "abaqus_results", "greif3d_rescue", "per_geometry.json")))
    c3 = np.array([r["cos_prior_data"] for r in P3]); e0 = np.array([r["prior"] for r in P3]); afe = np.array([r["field_1f_a"][0] for r in P3])
    sq = np.sqrt(np.maximum(c3 ** 2 - 1 + e0 ** 2, 0.0))
    rq = np.where(np.abs(c3 / (c3 - sq) - afe) < np.abs(c3 / (c3 + sq) - afe), c3 - sq, c3 + sq)   # as greif3d_transfer_factor.py
    ast = c3 / rq; loo = np.array([np.median(np.delete(ast, i)) for i in range(len(P3))])
    eL = np.sqrt(np.maximum(loo ** 2 * rq ** 2 - 2 * loo * rq * c3 + 1, 0.0))
    out["U4"] = dict(n=len(P3), prior=q(e0), loo=q(eL), rescue10=q([r["rescue_10_2f"] for r in P3]), factor=q(ast))
    u = out["U4"]
    print(f"[U4] 3-D parts ({len(P3)} shapes, Moldflow data): relative warp error median [5-95 %]: prior {u['prior']['median'] * 100:.0f} "
          f"[{u['prior']['p05'] * 100:.0f}, {u['prior']['p95'] * 100:.0f}] %, no measurement {u['loo']['median'] * 100:.0f} "
          f"[{u['loo']['p05'] * 100:.0f}, {u['loo']['p95'] * 100:.0f}] %, 10 points {u['rescue10']['median'] * 100:.0f} "
          f"[{u['rescue10']['p05'] * 100:.0f}, {u['rescue10']['p95'] * 100:.0f}] %; factor {u['factor']['median']:.2f} "
          f"[{u['factor']['p05']:.2f}, {u['factor']['p95']:.2f}]")
    out["samples"] = R; out["names"] = NAMES; out["lo"] = LO.tolist(); out["hi"] = HI.tolist()
    json.dump(out, open(os.path.join(OUT, "uncertainty.json"), "w"), indent=1)
    print("wrote", os.path.join(OUT, "uncertainty.json"))


if __name__ == "__main__":
    main()
