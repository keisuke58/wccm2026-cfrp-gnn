"""validation/denso_pretrain_surrogate.py -- extending the calibration from one plate to many moulding
conditions: a pre-training set from the physics chain of denso_tsuda2026_compare.py (Mori-Tanaka ->
orientation averaging -> free-plate lamination, the chain that reproduced the paper's surface stress),
a surrogate trained on it, and identification of the conditions of an unseen plate from a few noisy
"measured" points.

Conditions theta (ranges are ASSUMED, around the paper's values):
  T0_eff   effective stress-free temperature   140 .. 200 C   (paper 180 C; calibration gave ~127 C,
                                                               so 127 C is OUTSIDE: also tested)
  Vf       fibre volume fraction                0.15 .. 0.21   (paper 0.182)
  s        fibre aspect ratio                   12 .. 22       (paper 16.8)
  w        core-width factor of the a_ij(z) profile (depth mapped about mid-plane) 0.75 .. 1.25
  m        scale of (a_xx - 1/3) in the profile 0.85 .. 1.10
Output: sigma_xx(z), sigma_yy(z) at 41 depths. Surrogate: point-wise MLP (z, theta) -> sigma.
Checks (printed):
  S1 surrogate error on held-out plates against the number of pre-training plates;
  S2 identification: for held-out plates, observe sigma_xx and sigma_yy at 5 depths with noise
     sd 1 MPa, find theta by least squares through the surrogate (multi-start), report the error of
     T0_eff and of the full predicted profile; repeat with the plate at T0_eff = 127 C (extrapolation).
Honest scope: synthetic "measurements" (stress at depths) stand in for the real released strains; the
map from stress to released strain (hole-drilling FE, linear) is not included. Elastic chain only.
NOT a result on real data.

Result (2026-11-03, --n 600, 40 held-out plates, CPU 4 cores, 1 min 46 s):
  S1 held-out RMSE 5.2 / 7.9 / 3.0 / 1.35 / 0.72 / 0.71 MPa for 25 / 50 / 100 / 200 / 400 / 600 plates
     (sigma range about -73 .. +43 MPa): levels off from about 400 plates.
  S2 the full profile of an unseen plate is predicted to 1.2 MPa from 5 depths (noise 1 MPa), but T0_eff
     itself is NOT identified (rms error 18 C over a 60 C range): the five conditions trade off against
     each other in the stress profile. The single plate at T0_eff = 127 C, outside the training range,
     gave 124 C and profile RMSE 1.4 MPa (one case only).

Run:  python3 validation/denso_pretrain_surrogate.py --n 600 --workers 4   (writes results/denso_pretrain/)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "denso_pretrain")
sys.path.insert(0, HERE)
import denso_tsuda2026_compare as tc  # noqa: E402

LO = np.array([140.0, 0.15, 12.0, 0.75, 0.85]); HI = np.array([200.0, 0.21, 22.0, 1.25, 1.10])
NAMES = ["T0_eff", "Vf", "s", "w", "m"]
NZ = 41
_P = json.load(open(os.path.join(ROOT, "results", "denso_tsuda2026", "compare.json")))["C3_stress_3mm"]
DEP = np.array(_P["depth_mm"]); AXX = np.array(_P["axx"]); AYY = np.array(_P["ayy"])
HMM = DEP[-1]


def profile(w, m):
    zq = np.linspace(0.0, HMM, NZ)
    mid = HMM / 2
    src = np.clip(mid + (zq - mid) / w, 0.0, HMM)          # core width factor w
    axx = np.interp(src, DEP, AXX); ayy = np.interp(src, DEP, AYY)
    ax2 = np.clip(1 / 3 + m * (axx - 1 / 3), 0.02, 0.96)    # MD level m, keep a_yy share of the rest
    ay2 = np.clip(ayy * (1 - ax2) / np.maximum(1 - axx, 1e-6), 0.02, 0.96)
    return zq, ax2, ay2, np.clip(1 - ax2 - ay2, 0.02, 1.0)


def plate(theta):
    T0, vf, s, w, m = theta
    tc.T0, tc.VF, tc.ASPECT = float(T0), float(vf), float(s)
    zq, ax, ay, az = profile(w, m)
    z = (zq - HMM / 2) * 1e-3
    sig, _ = tc.residual_stress(z, *tc.layers(z, ax, ay), mode="total", nT=33)
    return np.r_[sig[:, 0], sig[:, 1]] / 1e6


def generate(thetas, workers):
    with Pool(workers) as p:
        return np.array(p.map(plate, [tuple(t) for t in thetas]))


def make_xy(thetas, Y):
    z = np.linspace(-1, 1, NZ)
    tn = (thetas - LO) / (HI - LO) * 2 - 1
    X = np.concatenate([np.column_stack([np.full(NZ, 0.0) + z, np.repeat(t[None], NZ, 0)]) for t in tn])
    Yp = np.concatenate([np.column_stack([y[:NZ], y[NZ:]]) for y in Y])
    return X, Yp


def train(X, Y, torch, seed, epochs=600):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(6, 96), torch.nn.SiLU(), torch.nn.Linear(96, 96), torch.nn.SiLU(),
                              torch.nn.Linear(96, 96), torch.nn.SiLU(), torch.nn.Linear(96, 2))
    Xt = torch.tensor(X, dtype=torch.float32); Yt = torch.tensor(Y / 20.0, dtype=torch.float32)
    opt = torch.optim.Adam(net.parameters(), 3e-3); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = len(Xt)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 2048):
            b = perm[i:i + 2048]
            loss = ((net(Xt[b]) - Yt[b]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    return net


def predict(net, torch, theta_n):
    """theta_n: (k, 5) normalised; returns (k, NZ, 2) MPa."""
    z = torch.linspace(-1, 1, NZ)
    k = theta_n.shape[0]
    X = torch.cat([z.repeat(k)[:, None], theta_n.repeat_interleave(NZ, 0)], 1)
    return (net(X) * 20.0).reshape(k, NZ, 2)


def identify(net, torch, obs_idx, obs, n_start=32, iters=300, seed=0):
    g = torch.Generator().manual_seed(seed)
    th = (torch.rand(n_start, 5, generator=g) * 2 - 1).requires_grad_(True)
    opt = torch.optim.Adam([th], 0.05)
    ob = torch.tensor(obs, dtype=torch.float32)
    for _ in range(iters):
        pr = predict(net, torch, th)[:, obs_idx, :]
        loss = ((pr - ob[None]) ** 2).sum((1, 2)) + 1e-3 * (torch.relu(th.abs() - 1.3) ** 2).sum(1)
        opt.zero_grad(); loss.sum().backward(); opt.step()
    best = int(torch.argmin(loss.detach()))
    return th.detach()[best]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--n-test", type=int, default=40)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261103)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.workers)
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    th_tr = LO + (HI - LO) * rng.random((a.n, 5)); th_te = LO + (HI - LO) * rng.random((a.n_test, 5))
    th_127 = np.array([[127.0, 0.182, 16.8, 1.0, 1.0]])
    t0 = time.time()
    Ytr = generate(th_tr, a.workers); Yte = generate(th_te, a.workers); Y127 = generate(th_127, a.workers)
    print(f"generated {a.n} + {a.n_test} + 1 plates in {time.time() - t0:.0f} s "
          f"(sigma_xx range {Ytr[:, :NZ].min():.1f} .. {Ytr[:, :NZ].max():.1f} MPa)")
    paper = plate((180.0, tc._m["fibre_volume_fraction"], tc._m["aspect_ratio_mean"], 1.0, 1.0))
    print(f"[check] paper conditions, surface sigma_xx {paper[0]:+.1f} / {paper[NZ - 1]:+.1f} MPa (paper FE -24.3)")
    res = dict(ranges=dict(zip(NAMES, zip(LO.tolist(), HI.tolist()))), n_train=a.n, curve=[], ident={})
    Xte, Yte_p = make_xy(th_te, Yte)
    net = None
    for n in (25, 50, 100, 200, 400, a.n):
        if n > a.n:
            continue
        X, Y = make_xy(th_tr[:n], Ytr[:n])
        net = train(X, Y, torch, a.seed)
        with torch.no_grad():
            pe = net(torch.tensor(Xte, dtype=torch.float32)).numpy() * 20.0
        r = float(np.sqrt(np.mean((pe - Yte_p) ** 2)))
        res["curve"].append(dict(n=n, rmse_MPa=r)); print(f"[S1] {n:4d} plates: held-out RMSE {r:.2f} MPa")
    obs_idx = np.array([0, 5, 10, 20, 30])                     # depths 0 .. 2.3 mm from the top
    for tag, TH, YY in (("held-out", th_te, Yte), ("T0_eff 127 C (outside range)", th_127, Y127)):
        eT, ep = [], []
        for k in range(len(TH)):
            prof = np.column_stack([YY[k, :NZ], YY[k, NZ:]])
            obs = prof[obs_idx] + rng.normal(0, 1.0, (len(obs_idx), 2))
            thn = identify(net, torch, obs_idx, obs, seed=k)
            th = (thn.numpy() + 1) / 2 * (HI - LO) + LO
            with torch.no_grad():
                pp = predict(net, torch, thn[None])[0].numpy()
            eT.append(th[0] - TH[k, 0]); ep.append(float(np.sqrt(np.mean((pp - prof) ** 2))))
        eT = np.array(eT); ep = np.array(ep)
        res["ident"][tag] = dict(T0_err_mean=float(eT.mean()), T0_err_rms=float(np.sqrt(np.mean(eT ** 2))),
                                 profile_rmse_mean=float(ep.mean()), n=len(TH))
        print(f"[S2] {tag}: T0_eff error rms {np.sqrt(np.mean(eT**2)):.1f} C (mean {eT.mean():+.1f}); "
              f"full-profile RMSE {ep.mean():.2f} MPa from 5 depths x 2 components, noise 1 MPa")
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    print(f"wrote {OUT}/results.json")


if __name__ == "__main__":
    main()
