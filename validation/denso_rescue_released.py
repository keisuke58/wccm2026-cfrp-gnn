"""validation/denso_rescue_released.py -- RESCUE in the measured quantity: moulding conditions -> eigenstrain
profile -> released strain of the hole-drilling gauges, through the fifa influence functions, and the
identification of the conditions from the paper's MEASURED released strain (Tsuda et al. 2026 Fig. 12,
read by eye).

Chain (all linear-elastic, the chain that reproduced the paper's surface stress):
  theta = (T0_eff, Vf, s, w, m)  (validation/denso_pretrain_surrogate.py: ranges, profile warp)
  -> integrated eigenstrain e*_xx(z), e*_yy(z) from T0_eff to 20 C (Mori-Tanaka + orientation averaging)
  -> 0.2 mm band averages -> released strain y = G_xx ebar_xx + G_yy ebar_yy  (16-band operator, fifa §16,
     check I2 0.8 % on the paper's profile)                                                          (1)
Then a surrogate MLP theta -> y (30 values: 3 gauges x 10 drilling steps) trained on N plates, and the
identification of theta from the measured y (noise 10 micro-strain, the reading uncertainty).
Checks (printed):
  R1 chain at the paper's conditions vs the Abaqus production run (expected small: same physics, coarser
     z-grid and temperature steps here);
  R2 surrogate error on held-out plates;
  R3 identification on held-out synthetic plates (T0_eff error), as in denso_pretrain_surrogate.py S2 but in
     the measured quantity;
  R4 identification from the paper's measurement: (a) T0_eff alone, others at the paper's values -- compare
     with the common factor 0.67 -> T0_eff 127 C of denso_tsuda2026_transfer.py; (b) all five conditions
     (multi-start; the spread shows what the measurement does not determine). Each identified theta is
     re-evaluated with the physics chain (1), not the surrogate.
Note on T0_eff: the common factor 0.67 of denso_tsuda2026_transfer.py was read as T0_eff = 20 + 0.67 x 160
= 127 C, i.e. assuming a constant CTE. With the temperature-dependent CTE of the paper's Table 1 (larger at
high temperature), the same reduction of the eigenstrain corresponds to a higher T0_eff (R4a).
Honest scope: the operator G carries the paper's stiffness profile (exact only for it); one measured plate,
read by eye; elastic chain (relaxation only through T0_eff); conditions ranges assumed.

Run:  python3 validation/denso_rescue_released.py --n 600 --workers 4   (writes results/denso_rescue/)
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
OUT = os.path.join(ROOT, "results", "denso_rescue")
sys.path.insert(0, HERE)
import denso_pretrain_surrogate as ps  # noqa: E402
import denso_tsuda2026_compare as tc  # noqa: E402

INF = os.path.join(ROOT, "abaqus", "denso_plate", "abaqus_results", "influence")
CH = ["GCH1", "GCH2", "GCH3"]
NB = 16
_op = json.load(open(os.path.join(INF, "influence_operator.json")))
G = {c: np.array(_op["G"][c])[:, 1:, :] for c in ("xx", "yy")}         # (3 gauges, 10 steps, 16 bands)
PAPER = json.load(open(os.path.join(ROOT, "data", "literature", "tsuda2026_jscm52_88.json"), encoding="utf-8"))
F12 = PAPER["fig12_released_strain"]
DEPTH = np.arange(1, 11) * 0.2
Y_MEAS = np.concatenate([np.interp(DEPTH, F12["depth_mm"], F12[f"Ch{g[-1]}_exp"]) for g in CH])
THETA_PAPER = (180.0, tc._m["fibre_volume_fraction"], tc._m["aspect_ratio_mean"], 1.0, 1.0)
NZ = 155


def eigenstrain(theta, nT=33):
    T0, vf, s, w, m = theta
    tc.T0, tc.VF, tc.ASPECT = float(T0), float(vf), float(s)
    zq = np.linspace(0.0, ps.HMM, NZ)
    mid = ps.HMM / 2
    src = np.clip(mid + (zq - mid) / w, 0.0, ps.HMM)
    axx = np.interp(src, ps.DEP, ps.AXX); ayy = np.interp(src, ps.DEP, ps.AYY)
    ax2 = np.clip(1 / 3 + m * (axx - 1 / 3), 0.02, 0.96)
    ay2 = np.clip(ayy * (1 - ax2) / np.maximum(1 - axx, 1e-6), 0.02, 0.96)
    z = (zq - mid) * 1e-3
    a1, a2, a3 = tc.layers(z, ax2, ay2)
    Ts = np.linspace(T0, tc.T_RT, nT); eth = np.zeros((NZ, 3))
    for T1, T2 in zip(Ts[:-1], Ts[1:]):
        _, al = tc.layer_props(0.5 * (T1 + T2), a1, a2, a3); eth += al * (T2 - T1)
    return zq, eth                                                     # depth from the top face [mm]


def released(theta):
    zq, eth = eigenstrain(theta)
    band = np.minimum((zq // 0.2).astype(int), NB - 1)
    ex = np.array([eth[band == b, 0].mean() for b in range(NB)]); ey = np.array([eth[band == b, 1].mean() for b in range(NB)])
    y = G["xx"] @ (ex / 1e-3) + G["yy"] @ (ey / 1e-3)                 # (3, 10)
    return y.reshape(-1)


def train(X, Y, torch, seed, epochs=3000):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(5, 64), torch.nn.SiLU(), torch.nn.Linear(64, 64), torch.nn.SiLU(),
                              torch.nn.Linear(64, Y.shape[1]))
    Xt = torch.tensor(X, dtype=torch.float32); Yt = torch.tensor(Y / 100.0, dtype=torch.float32)
    opt = torch.optim.Adam(net.parameters(), 3e-3); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        loss = ((net(Xt) - Yt) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return net


def norm(th):
    return (np.asarray(th) - ps.LO) / (ps.HI - ps.LO) * 2 - 1


def denorm(u):
    return (np.asarray(u) + 1) / 2 * (ps.HI - ps.LO) + ps.LO


def identify(net, torch, y, free, fixed_u, n_start=48, iters=600, seed=0, sy=10.0):
    """Least squares through the surrogate; 'free' = indices of conditions to fit, others fixed at fixed_u."""
    g = torch.Generator().manual_seed(seed)
    u0 = torch.tensor(np.tile(fixed_u, (n_start, 1)), dtype=torch.float32)
    raw = (torch.randn(n_start, len(free), generator=g)).requires_grad_(True)   # th = tanh(raw): inside the trained range
    opt = torch.optim.Adam([raw], 0.05); yt = torch.tensor(y / 100.0, dtype=torch.float32)
    for _ in range(iters):
        u = u0.clone(); u[:, free] = torch.tanh(raw)
        loss = (((net(u) - yt[None]) * 100.0 / sy) ** 2).sum(1)
        opt.zero_grad(); loss.sum().backward(); opt.step()
    u = u0.clone().detach(); u[:, free] = torch.tanh(raw.detach())
    L = loss.detach().numpy()
    return u.numpy(), L


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--n-test", type=int, default=40)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261104)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.workers)
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    # R1
    y0 = released(THETA_PAPER)
    full = json.load(open(os.path.join(ROOT, "abaqus", "denso_plate", "abaqus_results", "denso_hole_released.json")))
    yf = np.concatenate([np.array(full["released"][g][1:]) for g in CH])
    r1 = float(np.max(np.abs(y0 - yf)) / np.max(np.abs(yf)))
    print(f"[R1] chain at the paper's conditions vs Abaqus production run: max rel. difference {r1:.3f} "
          f"(Ch.3 peak {y0[20:].max():.0f} vs {yf[20:].max():.0f} micro-strain)")
    # data
    th_tr = ps.LO + (ps.HI - ps.LO) * rng.random((a.n, 5)); th_te = ps.LO + (ps.HI - ps.LO) * rng.random((a.n_test, 5))
    t0 = time.time()
    with Pool(a.workers) as p:
        Ytr = np.array(p.map(released, [tuple(t) for t in th_tr])); Yte = np.array(p.map(released, [tuple(t) for t in th_te]))
    print(f"generated {a.n} + {a.n_test} plates (released strain, 30 values each) in {time.time() - t0:.0f} s")
    net = train(norm(th_tr), Ytr, torch, a.seed)
    with torch.no_grad():
        pe = net(torch.tensor(norm(th_te), dtype=torch.float32)).numpy() * 100.0
    r2 = float(np.sqrt(np.mean((pe - Yte) ** 2)))
    print(f"[R2] surrogate theta -> released strain, held-out RMSE {r2:.2f} micro-strain "
          f"(range of the data {Ytr.min():.0f} .. {Ytr.max():.0f})")
    # R3 synthetic identification (all five free), noise 10 micro-strain
    eT = []
    for k in range(min(20, a.n_test)):
        y = Yte[k] + rng.normal(0, 10.0, 30)
        U, L = identify(net, torch, y, [0, 1, 2, 3, 4], np.zeros(5), seed=k)
        eT.append(denorm(U[int(np.argmin(L))])[0] - th_te[k, 0])
    eT = np.array(eT)
    print(f"[R3] synthetic plates, all five conditions from the 30 released strains (noise 10): "
          f"T0_eff error rms {np.sqrt(np.mean(eT ** 2)):.1f} C (mean {eT.mean():+.1f})")
    # R4 measured
    fixed = norm(THETA_PAPER)
    res = dict(R1=r1, R2=r2, R3_T0_rms=float(np.sqrt(np.mean(eT ** 2))), R4={})
    for tag, free in (("T0 only", [0]), ("all five", [0, 1, 2, 3, 4])):
        U, L = identify(net, torch, Y_MEAS, free, fixed, seed=7)
        order = np.argsort(L); best = denorm(U[order[0]])
        good = [denorm(U[i]) for i in order if L[i] <= L[order[0]] + 30.0 / 10.0]   # within ~ chi2 +3
        yb = released(tuple(best))
        rm = float(np.sqrt(np.mean((yb - Y_MEAS) ** 2))); rm0 = float(np.sqrt(np.mean((y0 - Y_MEAS) ** 2)))
        spread = np.ptp(np.array(good), axis=0) if len(good) > 1 else np.zeros(5)
        res["R4"][tag] = dict(theta=best.tolist(), rmse_physics_at_theta=rm, rmse_paper_conditions=rm0,
                              n_near_optimal=len(good), spread=spread.tolist())
        print(f"[R4 {tag}] identified T0_eff {best[0]:.0f} C, Vf {best[1]:.3f}, s {best[2]:.1f}, w {best[3]:.2f}, m {best[4]:.2f}; "
              f"physics RMSE vs measured {rm0:.1f} -> {rm:.1f} micro-strain; "
              f"near-optimal starts {len(good)}, spread T0 {spread[0]:.0f} C, Vf {spread[1]:.3f}, s {spread[2]:.1f}, w {spread[3]:.2f}, m {spread[4]:.2f}")
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    print(f"wrote {OUT}/results.json")


if __name__ == "__main__":
    main()
