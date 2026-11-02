"""denso_residual_strain_ml.py — ML prediction of residual strain from crystallinity and fibre
orientation, rehearsing the Denso plan (research/DENSO_RA_RESIDUAL_STRAIN.md): pre-train on
simulation, then calibrate on sparse measured points. Both domains here are SYNTHETIC: no
Denso data exists yet (pre-contract), so this tests the method, not the material.

Data (physics from viz/denso_plate_features.py, generalised to random plates):
  each plate draws mould temperatures and an orientation profile (skin / shell / core with
  random levels, core width and a core offset that breaks up/down symmetry), then gets
  crystallinity from 1-D cooling with the validated Hoffman-Lauritzen kinetics, orientation-
  averaged expansion and shrinkage via <C>^-1:<C:alpha>, and the free-plate residual strain
  e_res = e0 + z k - e* from force and moment balance, at NZ depths.
  SOURCE ("Ansys" role): the constants of viz/denso_plate_features.py, full profiles, no noise.
  TARGET ("experiment" role): systematically different constants (transverse CTE +15 %,
  transverse crystallisation shrinkage -15 %, X_inf 0.27, lock-in 150 C), observed only at
  M random depths per plate, with Gaussian measurement noise. Evaluated on noise-free full
  profiles of unseen target plates.

Experiment 1 — what must the input contain?
  e* is local, but e0 and k are set by the whole section's balance, so the residual strain at
  a point is NOT a function of the local crystallinity and orientation alone. Compared:
    local    [z, a11, a22, a33, X_c] at the point (the inputs as literally specified),
    context  local + six section moments (means and first moments in z of a11, a22, X_c).
Experiment 2 — transfer onto sparse, noisy measurements, versus number of measured plates:
    source-only (pre-trained context model, no calibration), target-only (trained from scratch
    on the sparse points), fine-tune (pre-trained, then trained on the sparse points).

Validated: the vectorised K(T) equals validation/cfrtp_peek_validation.K_hl to machine
precision; a symmetric plate has zero curvature; generated data are reproducible from SEED.

Honest scope: synthetic domains with a known, smooth discrepancy (constant shifts), which
is kinder than a real sim-to-real gap (orientation-model error, measurement definition,
spatially varying error); assumed orientation profiles; all limits of
viz/denso_plate_features.py apply (no latent heat, packing, viscoelasticity). Point-wise MLP;
a field model (DeepONet/GNN, as elsewhere in this repo) is the next step. GPU helps little
at this size: data generation is CPU-bound, so use --workers.

Run:  python3 denso_residual_strain_ml.py --quick            (CPU smoke test, minutes)
      python3 denso_residual_strain_ml.py --workers 16       (full run; see the guide)
Guide: research/GUIDE_copaam_ml_training.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict, replace
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "validation"))
sys.path.insert(0, os.path.join(HERE, "viz"))
from fiber_orientation_cte import ud_stiffness, orient_avg_cte  # noqa: E402
from cfrtp_peek_validation import K_hl, N_AVRAMI, HL, TABS  # noqa: E402
from denso_plate_features import reduced_Q, lamination  # noqa: E402

SEED = 20261102
NZ = 41


@dataclass(frozen=True)
class Consts:
    A_L: float = 5.0e-6
    A_T: float = 55.0e-6
    B_L: float = -0.002
    B_T: float = -0.020
    X_INF: float = 0.30
    T_LOCK: float = 143.0
    T_ROOM: float = 25.0
    H: float = 3.0e-3
    DIFF: float = 1.8e-7
    T_MELT: float = 380.0
    T_MOULD_S: float = 40.0
    UD: dict = field(default_factory=lambda: dict(EL=35.0e9, ET=7.0e9, nuLT=0.35,
                                                  nuTT=0.45, GLT=2.8e9))


SOURCE = Consts()
TARGET = replace(SOURCE, A_T=SOURCE.A_T * 1.15, B_T=SOURCE.B_T * 0.85, X_INF=0.27, T_LOCK=150.0)


# ------------------------------------------------------------------ physics -------
def K_hl_vec(T):
    """Vectorised copy of validation.cfrtp_peek_validation.K_hl (checked against it)."""
    TK = np.asarray(T, float) + TABS
    Tm0K, TinfK = HL["TM0"] + TABS, HL["TINF"] + TABS
    ok = (TK < Tm0K) & (TK > TinfK)
    TKs = np.where(ok, TK, 0.5 * (Tm0K + TinfK))
    dT = Tm0K - TKs; f = 2 * TKs / (Tm0K + TKs)
    a = -HL["USTAR_R"] / (TKs - TinfK) - HL["KG"] / (TKs * dT * f)
    return np.where(ok, HL["K0"] * np.exp(np.maximum(a, -700.0)), 0.0)


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def orientation(z, p):
    """Random skin / shell / core profile; p = (a11_core, a11_shell, a11_skin, w_core,
    z_off, a33). z_off shifts the core and breaks up/down symmetry."""
    a11c, a11s, a11k, wc, zoff, a33v = p
    shell = sig((np.abs(z - zoff) - wc) / 0.06)
    skin = sig((np.abs(z) - 0.93) / 0.02)
    a11 = (a11c + (a11s - a11c) * shell) * (1 - skin) + a11k * skin
    a33 = np.full_like(z, a33v)
    return a11, 1.0 - a11 - a33, a33


def crystallise(T_bot, T_top, c: Consts):
    dz = c.H / (NZ - 1); dt = 0.4 * dz * dz / c.DIFF
    T = np.full(NZ, c.T_MELT); T[0], T[-1] = T_bot, T_top
    a = np.full(NZ, 1e-3); n = N_AVRAMI; t = 0.0
    while t < c.T_MOULD_S:
        aa = np.clip(a, 1e-8, 1 - 1e-10)
        r = n * K_hl_vec(T) * (1 - aa) * np.maximum(-np.log(1 - aa), 1e-12) ** ((n - 1) / n)
        a = np.minimum(a + np.maximum(r * dt, 0.0), 1.0)
        T[1:-1] = T[1:-1] + c.DIFF * dt / dz ** 2 * (T[2:] - 2 * T[1:-1] + T[:-2])
        t += dt
    return c.X_INF * a


def plate(args):
    """One plate: returns inputs (a11, a22, a33, X_c) and residual strain (x1e-3) at NZ depths."""
    params, c = args
    T_bot, T_top, op = params
    z = np.linspace(-1.0, 1.0, NZ)
    a11, a22, a33 = orientation(z, op)
    Xc = crystallise(T_bot, T_top, c)
    Cud = ud_stiffness(**c.UD)
    dT = c.T_ROOM - c.T_LOCK
    Qs, es = [], []
    for x, y, w, X in zip(a11, a22, a33, Xc):
        a = np.diag([x, y, w])
        al, Cavg = orient_avg_cte(a, "hybrid", Cud, aL=c.A_L, aT=c.A_T)
        be, _ = orient_avg_cte(a, "hybrid", Cud, aL=c.B_L, aT=c.B_T)
        Qs.append(reduced_Q(Cavg))
        es.append([al[0, 0] * dT + be[0, 0] * X, al[1, 1] * dT + be[1, 1] * X, 0.0])
    _, kap, em, *_ = lamination(z * c.H / 2, np.array(Qs), np.array(es))
    return dict(z=z, a11=a11, a22=a22, a33=a33, Xc=Xc, exx=em[:, 0] * 1e3, eyy=em[:, 1] * 1e3,
                kappa=kap)


def sample_params(rng, n):
    out = []
    for _ in range(n):
        T_bot, T_top = rng.uniform(165, 205, 2)
        op = (rng.uniform(0.15, 0.40), rng.uniform(0.60, 0.85), rng.uniform(0.45, 0.65),
              rng.uniform(0.20, 0.50), rng.uniform(-0.20, 0.20), rng.uniform(0.03, 0.10))
        out.append((T_bot, T_top, op))
    return out


def generate(params, c, workers):
    jobs = [(p, c) for p in params]
    if workers > 1:
        with Pool(workers) as pool:
            return pool.map(plate, jobs, chunksize=max(1, len(jobs) // (4 * workers)))
    return [plate(j) for j in jobs]


# ----------------------------------------------------------------- features -------
def section_moments(p):
    z = p["z"]; w = np.full(NZ, 2.0 / (NZ - 1)); w[0] = w[-1] = 1.0 / (NZ - 1)
    m = lambda v: float(np.sum(w * v) / 2.0)          # thickness mean
    m1 = lambda v: float(np.sum(w * z * v) / 2.0)     # first moment
    return [m(p["a11"]), m(p["a22"]), m(p["Xc"]), m1(p["a11"]), m1(p["a22"]), m1(p["Xc"])]


def features(plates, context, idx=None):
    X, Y = [], []
    for k, p in enumerate(plates):
        sel = range(NZ) if idx is None else idx[k]
        ctx = section_moments(p) if context else []
        for i in sel:
            X.append([p["z"][i], p["a11"][i], p["a22"][i], p["a33"][i], p["Xc"][i]] + ctx)
            Y.append([p["exx"][i], p["eyy"][i]])
    return np.array(X, np.float32), np.array(Y, np.float32)


# --------------------------------------------------------------------- model -------
def make_model(nin, torch):
    nn = torch.nn
    return nn.Sequential(nn.Linear(nin, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(),
                         nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2))


class Scaler:
    def __init__(self, X, Y):
        self.mx, self.sx = X.mean(0), X.std(0) + 1e-8
        self.my, self.sy = Y.mean(0), Y.std(0) + 1e-8


def fit(model, X, Y, sc, torch, dev, epochs, lr, wd=0.0, batch=4096, seed=0):
    torch.manual_seed(seed)
    Xt = torch.tensor((X - sc.mx) / sc.sx, device=dev)
    Yt = torch.tensor((Y - sc.my) / sc.sy, device=dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = len(Xt); g = torch.Generator(device="cpu").manual_seed(seed)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g).to(dev)
        for s in range(0, n, batch):
            b = perm[s:s + batch]
            loss = torch.mean((model(Xt[b]) - Yt[b]) ** 2)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return model


def predict(model, X, sc, torch, dev):
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor((X - sc.mx) / sc.sx, device=dev)).cpu().numpy()
    return out * sc.sy + sc.my


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def r2(a, b):
    return float(1 - np.sum((a - b) ** 2) / np.sum((b - b.mean(0)) ** 2))


# ---------------------------------------------------------------------- main -------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="small CPU smoke test")
    ap.add_argument("--n-source", type=int, default=2000, help="simulated (source) plates")
    ap.add_argument("--n-target-test", type=int, default=200, help="unseen target plates for evaluation")
    ap.add_argument("--n-target", type=str, default="1,2,5,10,20,50", help="measured target plates")
    ap.add_argument("--points", type=int, default=5, help="measured depths per target plate")
    ap.add_argument("--noise", type=float, default=0.05, help="measurement noise sd [1e-3]")
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--ft-epochs", type=int, default=600)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--out", default="results/denso_ml")
    a = ap.parse_args()
    if a.quick:
        a.n_source, a.n_target_test, a.epochs, a.ft_epochs, a.repeats = 120, 40, 400, 300, 3
        a.n_target = "1,2,5,10"
    os.makedirs(a.out, exist_ok=True)

    import torch
    dev = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available())
                       else ("cpu" if a.device == "auto" else a.device))
    print(f"device {dev} | torch {torch.__version__} | workers {a.workers}")

    # validation of the physics port
    Ts = np.linspace(100, 420, 3201)
    ref = np.array([K_hl(t) for t in Ts]); vec = K_hl_vec(Ts)
    kerr = float(np.max(np.abs(vec - ref) / (np.abs(ref) + 1e-300)))
    print(f"[validate] vectorised K(T) vs K_hl: max relative difference {kerr:.1e}")
    sym = plate(((185.0, 185.0, (0.24, 0.80, 0.55, 0.35, 0.0, 0.06)), SOURCE))
    print(f"[validate] symmetric plate curvature: {np.max(np.abs(sym['kappa'])):.1e} 1/m (should be ~0)")

    rng = np.random.default_rng(SEED)
    t0 = time.time()
    src = generate(sample_params(rng, a.n_source), SOURCE, a.workers)
    tgt_test = generate(sample_params(rng, a.n_target_test), TARGET, a.workers)
    n_t_list = [int(v) for v in a.n_target.split(",")]
    tgt_pool = generate(sample_params(rng, max(n_t_list) * a.repeats), TARGET, a.workers)
    print(f"generated {len(src)} source + {len(tgt_test)} target-test + {len(tgt_pool)} "
          f"target-pool plates in {time.time()-t0:.0f} s")

    # ---------------- experiment 1: local vs context, in-domain (source) -----------
    ntr = int(0.8 * len(src)); tr, te = src[:ntr], src[ntr:]
    res = dict(args=vars(a), device=str(dev), K_err=kerr, sym_kappa=float(np.max(np.abs(sym["kappa"]))),
               source=asdict(SOURCE), target=asdict(TARGET))
    models, scalers = {}, {}
    for name, ctx in (("local", False), ("context", True)):
        Xtr, Ytr = features(tr, ctx); Xte, Yte = features(te, ctx)
        sc = Scaler(Xtr, Ytr); m = make_model(Xtr.shape[1], torch).to(dev)
        t1 = time.time(); fit(m, Xtr, Ytr, sc, torch, dev, a.epochs, 2e-3, seed=SEED)
        P = predict(m, Xte, sc, torch, dev)
        res[f"exp1_{name}"] = dict(rmse=rmse(P, Yte), r2=r2(P, Yte), train_s=time.time() - t1)
        models[name], scalers[name] = m, sc
        print(f"[exp1] {name:8s}: source test RMSE {rmse(P, Yte):.3f} x1e-3, R^2 {r2(P, Yte):.4f} "
              f"({time.time()-t1:.0f} s)")
    _, Yte = features(te, True)
    res["exp1_target_std"] = float(np.std(Yte))

    # ---------------- experiment 2: transfer onto sparse noisy target points --------
    import copy
    Xtt, Ytt = features(tgt_test, True)
    sc = scalers["context"]
    base_rmse = rmse(predict(models["context"], Xtt, sc, torch, dev), Ytt)
    print(f"[exp2] source-only on target: RMSE {base_rmse:.3f} x1e-3")
    curve = {m: {n: [] for n in n_t_list} for m in ("source-only", "target-only", "fine-tune")}
    for r in range(a.repeats):
        for n in n_t_list:
            plates = tgt_pool[r * max(n_t_list): r * max(n_t_list) + n]
            rr = np.random.default_rng(SEED + 1000 * r + n)
            idx = [rr.choice(NZ, size=a.points, replace=False) for _ in plates]
            Xs, Ys = features(plates, True, idx)
            Ys = Ys + rr.normal(0, a.noise, Ys.shape).astype(np.float32)
            curve["source-only"][n].append(base_rmse)
            m0 = make_model(Xs.shape[1], torch).to(dev)
            fit(m0, Xs, Ys, sc, torch, dev, a.ft_epochs, 2e-3, wd=1e-4, seed=SEED + r)
            curve["target-only"][n].append(rmse(predict(m0, Xtt, sc, torch, dev), Ytt))
            m1 = copy.deepcopy(models["context"])
            fit(m1, Xs, Ys, sc, torch, dev, a.ft_epochs, 3e-4, wd=1e-4, seed=SEED + r)
            curve["fine-tune"][n].append(rmse(predict(m1, Xtt, sc, torch, dev), Ytt))
        print(f"[exp2] repeat {r+1}/{a.repeats} done")
    summ = {m: {n: dict(mean=float(np.mean(v)), sd=float(np.std(v))) for n, v in d.items()}
            for m, d in curve.items()}
    res["exp2"] = summ
    print("[exp2] target RMSE [x1e-3], mean over repeats  (measured plates x %d points each)" % a.points)
    print("   n   " + "  ".join(f"{m:>12s}" for m in summ))
    for n in n_t_list:
        print(f"{n:4d}   " + "  ".join(f"{summ[m][n]['mean']:12.3f}" for m in summ))

    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    _plot(os.path.join(a.out, "denso_ml.png"), res, n_t_list, tgt_test[0], models, sc, torch, dev)
    print(f"wrote {a.out}/results.json and {a.out}/denso_ml.png")


def _plot(path, res, n_t_list, p, models, sc, torch, dev):
    import shutil
    sys.path.insert(0, os.path.join(HERE, "slides", "figure_sources"))
    try:                                   # repo convention: thesis_style, graceful fallback
        from thesis_style import use
        use(width_frac=1.0, aspect=0.45)
        if shutil.which("latex") is None:
            matplotlib.rcParams["text.usetex"] = False
            matplotlib.rcParams["font.family"] = "serif"
            matplotlib.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif",
                                                 "STIXGeneral", "DejaVu Serif"]
            matplotlib.rcParams["mathtext.fontset"] = "stix"
    except Exception:
        pass
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    names = ["local", "context"]
    ax[0].bar(names, [res[f"exp1_{n}"]["rmse"] for n in names], color=["#8a3e2c", "#1f4e79"])
    ax[0].set_ylabel(r"source test RMSE [$10^{-3}$]")
    ax[0].set_title("(a) input content (in-domain)")
    for i, n in enumerate(names):
        ax[0].text(i, res[f"exp1_{n}"]["rmse"], f"R$^2$={res[f'exp1_{n}']['r2']:.3f}",
                   ha="center", va="bottom", fontsize=9)
    sty = {"source-only": ("#777777", "s", ":"), "target-only": ("#8a3e2c", "o", "--"),
           "fine-tune": ("#1f4e79", "^", "-")}
    for m, (c, mk, ls) in sty.items():
        mu = [res["exp2"][m][n]["mean"] for n in n_t_list]
        sd = [res["exp2"][m][n]["sd"] for n in n_t_list]
        ax[1].errorbar(n_t_list, mu, yerr=sd, color=c, marker=mk, ls=ls, capsize=3, label=m)
    ax[1].set_xscale("log"); ax[1].set_xlabel("measured target plates")
    ax[1].set_ylabel(r"target RMSE [$10^{-3}$]"); ax[1].set_title("(b) transfer onto sparse points")
    ax[1].legend(fontsize=8); ax[1].grid(True, alpha=0.3, which="both")
    X, Y = features([p], True)
    P = predict(models["context"], X, sc, torch, dev)
    ax[2].plot(Y[:, 0], p["z"], "-", color="#15181d", lw=2, label=r"target truth $\varepsilon_{xx}$")
    ax[2].plot(P[:, 0], p["z"], "--", color="#777777", lw=1.6, label="source-only prediction")
    ax[2].set_xlabel(r"residual strain [$10^{-3}$]"); ax[2].set_ylabel(r"normalized thickness $z$")
    ax[2].set_title("(c) one unseen target plate"); ax[2].legend(fontsize=8); ax[2].grid(True, alpha=0.3)
    fig.suptitle("Residual-strain ML on SYNTHETIC source/target domains (illustrative constants)",
                 fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
