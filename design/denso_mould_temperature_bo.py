"""design/denso_mould_temperature_bo.py — stage 3 prototype for the DENSO theme: choose the two
mould-wall temperatures (T_bottom, T_top) that minimise the warpage of a plate, by Bayesian
optimisation (2-D Gaussian process + expected improvement) over the through-thickness physics
model of denso_residual_strain_ml.py (cooling + Hoffman–Lauritzen/Nakamura crystallisation +
orientation-averaged eigenstrain + free-plate lamination).

The "part" has an ASSUMED up/down-asymmetric orientation (core shifted by z_off = 0.15), so a
symmetric mould does not give a flat plate; asymmetric crystallisation can compensate. Design
space T_bottom, T_top in [165, 205] C (the range of the ML rehearsal). Objective: |sag| over a
100 mm span, sag = kappa_y L^2 / 8. Every point the optimiser proposes is EVALUATED by the
physics model (physics is the authority; in the project the physics call is replaced by the
transfer-learned predictor, which is not done here).

Validated (printed): the BO optimum against a brute-force 41 x 41 grid of the same physics.
Honest scope: illustrative PEEK-like constants (not the GF/PPS of Tsuda et al. 2026: PPS
crystallisation kinetics are not in the repository); orientation fixed and assumed, i.e. the
injection speed is NOT a design variable (it needs a flow simulation); flat plate, no
viscoelastic relaxation. Zero warpage is reached on a curve in (T_bottom, T_top), so the
optimum is not unique; among near-flat points the one with the smallest residual-strain range
is reported as an example tie-break.

Run:  python3 design/denso_mould_temperature_bo.py --workers 4   (writes results/denso_mould_bo/)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import denso_residual_strain_ml as ml  # noqa: E402

LO, HI = 165.0, 205.0
L_SPAN = 0.10
ORIENT = (0.25, 0.80, 0.55, 0.35, 0.15, 0.06)      # ASSUMED part: core shifted, asymmetric


def physics(T):
    Tb, Tt = T
    p = ml.plate(((Tb, Tt, ORIENT), ml.SOURCE))
    sag = float(p["kappa"][1] * L_SPAN ** 2 / 8 * 1e3)
    rng = float(max(np.ptp(p["exx"]), np.ptp(p["eyy"])))
    return sag, rng


class GP2:
    """GP with an isotropic RBF kernel on [0,1]^2; hyper-parameters by a marginal-likelihood grid."""
    def fit(self, X, y):
        self.X = np.asarray(X, float); self.mu = float(np.mean(y)); self.sd = float(np.std(y) + 1e-12)
        yn = (np.asarray(y, float) - self.mu) / self.sd; n = len(yn); best = None
        for ls in (0.08, 0.12, 0.2, 0.3, 0.5, 0.8):
            for sn2 in (1e-6, 1e-4, 1e-2):
                K = self._k(self.X, self.X, ls) + sn2 * np.eye(n)
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    continue
                a = np.linalg.solve(L.T, np.linalg.solve(L, yn))
                nll = 0.5 * yn @ a + np.log(np.diag(L)).sum()
                if best is None or nll < best[0]:
                    best = (nll, ls, sn2, L, a)
        _, self.ls, self.sn2, self.L, self.a = best
        return self

    def _k(self, A, B, ls):
        d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return np.exp(-0.5 * d2 / ls ** 2)

    def predict(self, Xs):
        Ks = self._k(np.asarray(Xs, float), self.X, self.ls)
        m = Ks @ self.a; v = np.linalg.solve(self.L, Ks.T)
        s = np.sqrt(np.clip(1.0 - (v * v).sum(0), 1e-12, None))
        return m * self.sd + self.mu, s * self.sd


def ei(mu, s, best):
    z = (best - mu) / s
    Phi = 0.5 * (1 + np.vectorize(math.erf)(z / math.sqrt(2)))
    phi = np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    return (best - mu) * Phi + s * phi


def to_T(u):
    return LO + (HI - LO) * np.asarray(u)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-init", type=int, default=6)
    ap.add_argument("--n-iter", type=int, default=18)
    ap.add_argument("--grid", type=int, default=41)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=ml.SEED)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "denso_mould_bo"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    # ---- reference: brute force on a grid (validation of the optimiser)
    t0 = time.time()
    g = np.linspace(LO, HI, a.grid)
    pts = [(tb, tt) for tt in g for tb in g]
    with Pool(a.workers) as pool:
        res = pool.map(physics, pts)
    sag = np.array([r[0] for r in res]).reshape(a.grid, a.grid)       # [i_top, i_bottom]
    rge = np.array([r[1] for r in res]).reshape(a.grid, a.grid)
    k = int(np.argmin(np.abs(sag))); gb, gt = pts[k]
    print(f"grid {a.grid}x{a.grid} ({len(pts)} physics calls, {time.time()-t0:.0f} s): min |sag| "
          f"{abs(sag.flat[k]):.4f} mm at T_bottom {gb:.1f} / T_top {gt:.1f} C; symmetric "
          f"185/185 C sag {physics((185.0, 185.0))[0]:+.3f} mm")
    # ---- Bayesian optimisation
    U = list(rng.uniform(0, 1, (a.n_init, 2)))
    Y = [abs(physics(tuple(to_T(u)))[0]) for u in U]
    cand = np.array([(x, y) for x in np.linspace(0, 1, 81) for y in np.linspace(0, 1, 81)])
    hist = [min(Y)]
    for it in range(a.n_iter):
        gp = GP2().fit(np.array(U), np.log(np.array(Y) + 1e-3))     # log for the |.| kink
        mu, s = gp.predict(cand)
        u = cand[int(np.argmax(ei(mu, s, np.log(min(Y) + 1e-3))))]
        U.append(u); Y.append(abs(physics(tuple(to_T(u)))[0])); hist.append(min(Y))
    U = np.array(U); Y = np.array(Y); T = to_T(U)
    kb = int(np.argmin(Y))
    flat = [i for i in range(len(Y)) if Y[i] < 0.05]
    rngs = {i: physics(tuple(T[i]))[1] for i in flat}
    kt = min(rngs, key=rngs.get) if rngs else kb
    print(f"BO ({a.n_init} initial + {a.n_iter} iterations = {len(Y)} physics calls): min |sag| "
          f"{Y[kb]:.4f} mm at T_bottom {T[kb,0]:.1f} / T_top {T[kb,1]:.1f} C")
    print(f"     near-flat points (|sag| < 0.05 mm): {len(flat)}; smallest residual-strain range among "
          f"them {rngs.get(kt, float('nan')):.3f} x1e-3 at {T[kt,0]:.1f} / {T[kt,1]:.1f} C")
    out = dict(orientation=ORIENT, bounds=[LO, HI], grid=dict(T=g.tolist(), sag_mm=sag.tolist(), range=rge.tolist(),
               best=[gb, gt], best_abs_sag=float(abs(sag.flat[k]))),
               bo=dict(T=T.tolist(), abs_sag=Y.tolist(), best=T[kb].tolist(), best_abs_sag=float(Y[kb]),
                       history=hist, tie_break=T[kt].tolist()),
               symmetric_185_sag=physics((185.0, 185.0))[0])
    json.dump(out, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    _plot(os.path.join(a.out, "denso_mould_bo.png"), g, sag, T, Y, kb, hist, a.n_init)
    print(f"wrote {a.out}/results.json and denso_mould_bo.png")


def _plot(path, g, sag, T, Y, kb, hist, n_init):
    sys.path.insert(0, os.path.join(ROOT, "slides", "figure_sources"))
    try:                                   # repo convention: thesis_style, graceful fallback
        from thesis_style import use
        use(width_frac=1.0, aspect=0.42)
    except Exception:
        pass
    matplotlib.rcParams.update({"text.usetex": False, "font.family": "serif",
                                "font.serif": ["Times New Roman", "Liberation Serif", "STIXGeneral"],
                                "mathtext.fontset": "stix"})
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8), gridspec_kw=dict(width_ratios=[1.25, 1]))
    cs = ax[0].contourf(g, g, sag, levels=21, cmap="Greys")
    ax[0].contour(g, g, sag, levels=[0.0], colors="k", linewidths=1.2)
    fig.colorbar(cs, ax=ax[0], label="sag over 100 mm [mm]")
    ax[0].plot(T[:n_init, 0], T[:n_init, 1], "o", mfc="white", mec="k", ms=5, label="initial")
    ax[0].plot(T[n_init:, 0], T[n_init:, 1], "^", mfc="white", mec="k", ms=5, label="BO proposals")
    ax[0].plot(T[kb, 0], T[kb, 1], "*", color="k", ms=12, label="BO best")
    ax[0].plot([185], [185], "s", mfc="none", mec="0.4", ms=7, label="symmetric 185/185")
    ax[0].set_xlabel(r"$T_{\mathrm{bottom}}$ [$^\circ$C]"); ax[0].set_ylabel(r"$T_{\mathrm{top}}$ [$^\circ$C]")
    ax[0].legend(frameon=False, fontsize=7, loc="lower right")
    ax[0].set_title("(a) warpage map (grid) and BO samples; solid line = flat", fontsize=8)
    ax[1].semilogy(range(len(hist)), np.maximum(hist, 1e-4), "k-o", ms=3)
    ax[1].set_xlabel("BO iteration"); ax[1].set_ylabel(r"best $|$sag$|$ so far [mm]")
    ax[1].set_title("(b) convergence", fontsize=8)
    fig.suptitle("Mould-temperature design, illustrative constants, assumed orientation", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
