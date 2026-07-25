"""dd_breakdown_continuation.py — reverse-bias coupled drift-diffusion, where bias
continuation (warm-start, idea C) is *necessary*, not just faster.

The forward-bias Gummel demo (dd_full_1d.py) showed a robust map where warm-start saved
only ~10%. High reverse bias is the opposite regime: the coupled solve has a limited
convergence basin, so a cold start from equilibrium fails once the target bias is more
than a few V_t away, while ramping the bias in small steps and warm-starting each solve
from the previous converged state (continuation) reaches high reverse bias. That is where
warm-start is essential — and it is exactly the standard TCAD continuation practice.

Physics (1-D pn junction, scaled): Poisson + electron/hole continuity, Scharfetter-Gummel
edge currents, SRH recombination, and a field-dependent impact-ionization generation term
G = alpha(E)(|J_n|+|J_p|), alpha(E)=a0 exp(-b0/|E|), solved by the Gummel map.

Honest scope note: the explicit (lagged) impact-ionization term is kept moderate for
stability, so it adds only mild carrier multiplication (~2x by the end of the sweep) — it
does NOT produce a sharp avalanche runaway here. A true near-vertical avalanche needs an
implicit/damped generation term (and a fully-coupled Newton with proper variable scaling,
since the raw coupled DD Jacobian is famously ill-conditioned — the Poisson residual gets
"fixed" by moving minority carriers negative). Both are heavier builds; this demo makes
the continuation-is-necessary point cleanly with the robust Gummel map, and the numbers
below are honest about the mild multiplication.

Output: reverse-bias I-V traced by continuation; the maximum reverse bias a COLD solve
reaches before its basin gives out vs how far continuation reaches; and the Gummel update
histories (cold fails, warm converges) at a high reverse bias.

Run:  python3 dd_breakdown_continuation.py            (writes dd_breakdown_continuation.png)
      python3 dd_breakdown_continuation.py --help
"""
from __future__ import annotations

import argparse

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

LAMBDA2 = 2e-4
C0 = 30.0
XJ, WJ = 0.5, 0.03
TAU = 5.0
II_A0, II_B0 = 2.0, 12.0         # impact-ionization coefficients (moderate, stable multiplication)
GUMMEL_MAX = 60
GUMMEL_TOL = 1e-6


def bern(x):
    x = np.asarray(x, float)
    out = np.ones_like(x)
    b = np.abs(x) > 1e-10
    out[b] = x[b] / np.expm1(x[b])
    return out


def doping(x):
    return C0 * np.tanh((x - XJ) / WJ)


def equilibrium(x):
    C = doping(x)
    psi = np.arcsinh(C / 2.0)
    return psi, np.exp(psi), np.exp(-psi)


def laplacian(N, h):
    e = np.ones(N)
    return (sp.diags([e[1:], -2 * e, e[1:]], [-1, 0, 1], (N, N)) / h ** 2).tocsr()


def gummel(Vr, x, h, psi0, n0, p0):
    """Gummel map with impact ionization at reverse bias Vr. Returns (psi,n,p,hist,ok,M)."""
    N = len(x); C = doping(x); L = laplacian(N, h)
    interior = np.arange(1, N - 1)
    psi_eq, n_eq, p_eq = equilibrium(x)
    psi, n, p = psi0.copy(), n0.copy(), p0.copy()
    psi[0] = psi_eq[0] - Vr; psi[-1] = psi_eq[-1]
    n[0] = n_eq[0]; n[-1] = n_eq[-1]; p[0] = p_eq[0]; p[-1] = p_eq[-1]

    def gen_edge(psi, n, p):
        dpl = psi[1:] - psi[:-1]
        Jn = (bern(dpl) * n[1:] - bern(-dpl) * n[:-1]) / h
        Jp = (bern(-dpl) * p[1:] - bern(dpl) * p[:-1]) / h
        E = np.abs(dpl) / h
        a = II_A0 * np.exp(-II_B0 / (E + 1e-9))
        return a * (np.abs(Jn) + np.abs(Jp)), Jn, Jp

    def cont_matrix(psi, sign):
        dpl = psi[1:] - psi[:-1]; Bp, Bm = bern(dpl), bern(-dpl)
        A = sp.lil_matrix((N, N))
        for i in interior:
            if sign > 0:
                A[i, i + 1] += Bp[i] / h; A[i, i] += (-Bm[i] - Bp[i - 1]) / h
                A[i, i - 1] += Bm[i - 1] / h
            else:
                A[i, i + 1] += -Bm[i] / h; A[i, i] += (Bp[i] + Bm[i - 1]) / h
                A[i, i - 1] += -Bp[i - 1] / h
        return A.tocsr()

    hist = []
    ok = True
    for _ in range(GUMMEL_MAX):
        psi_k = psi.copy(); nk = n.copy(); pk = p.copy()
        for _ in range(40):                     # nonlinear Poisson Newton
            nn = nk * np.exp(psi - psi_k); pp = pk * np.exp(-(psi - psi_k))
            F = -LAMBDA2 * (L @ psi) - (pp - nn + C)
            J = (-LAMBDA2 * L + sp.diags(nn + pp)).tocsr()
            d = np.zeros(N)
            d[interior] = spla.spsolve(J[interior][:, interior].tocsc(), -F[interior])
            psi = psi + d
            if np.max(np.abs(d[interior])) < 1e-10:
                break
        n = nk * np.exp(psi - psi_k); p = pk * np.exp(-(psi - psi_k))
        n[0] = n_eq[0]; n[-1] = n_eq[-1]; p[0] = p_eq[0]; p[-1] = p_eq[-1]

        Gedge, _, _ = gen_edge(psi, n, p)
        Gnode = np.zeros(N); Gnode[interior] = 0.5 * (Gedge[interior - 1] + Gedge[interior])
        R = (n * p - 1.0) / (TAU * (n + p + 2.0))
        src = h * (R - Gnode)                    # net (recomb - generation)
        An = cont_matrix(psi, +1); Ap = cont_matrix(psi, -1)
        bn = src.copy(); bp = src.copy()
        bn[interior] -= An[interior][:, [0, -1]] @ np.array([n_eq[0], n_eq[-1]])
        bp[interior] -= Ap[interior][:, [0, -1]] @ np.array([p_eq[0], p_eq[-1]])
        n_new, p_new = n.copy(), p.copy()
        n_new[interior] = spla.spsolve(An[interior][:, interior].tocsc(), bn[interior])
        p_new[interior] = spla.spsolve(Ap[interior][:, interior].tocsc(), bp[interior])
        if not (np.all(np.isfinite(n_new)) and np.all(np.isfinite(p_new))):
            ok = False; break
        n_new = np.clip(n_new, 1e-30, 1e12); p_new = np.clip(p_new, 1e-30, 1e12)
        upd = np.max(np.abs(psi - psi_k))
        hist.append(upd)
        n, p = n_new, p_new
        if upd < GUMMEL_TOL:
            break
    else:
        ok = False
    if hist and hist[-1] > 1e-3:
        ok = False

    dpl = psi[1:] - psi[:-1]
    Jn = (bern(dpl) * n[1:] - bern(-dpl) * n[:-1]) / h
    Jp = (bern(-dpl) * p[1:] - bern(dpl) * p[:-1]) / h
    Jtot = float(np.mean(Jn + Jp))
    return psi, n, p, hist, ok, Jtot


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--N", type=int, default=81, help="grid nodes")
    ap.add_argument("--vmax", type=float, default=45.0, help="max reverse bias (Vt)")
    ap.add_argument("--nbias", type=int, default=46, help="bias points")
    ap.add_argument("--out", type=str, default="dd_breakdown_continuation.png")
    args = ap.parse_args()

    x = np.linspace(0, 1, args.N); h = x[1] - x[0]
    pe, ne, ppe = equilibrium(x)

    biases = np.linspace(0.0, args.vmax, args.nbias)

    # continuation: ramp reverse bias, warm-start from previous solution
    uw = (pe.copy(), ne.copy(), ppe.copy())
    IV, warm_reached = [], 0.0
    for Vr in biases:
        psi, n, p, hist, ok, J = gummel(Vr, x, h, *uw)
        if ok:
            uw = (psi, n, p); warm_reached = Vr; IV.append((Vr, abs(J)))
        else:
            break

    # cold: solve straight from equilibrium at each bias
    cold_ok, cold_reached = [], 0.0
    for Vr in biases:
        _, _, _, _, ok, _ = gummel(Vr, x, h, pe, ne, ppe)
        cold_ok.append(1.0 if ok else 0.0)
        if ok:
            cold_reached = Vr

    # residual histories near breakdown
    Vnear = 0.92 * warm_reached
    _, _, _, cold_h, _, _ = gummel(Vnear, x, h, pe, ne, ppe)
    ug = (pe.copy(), ne.copy(), ppe.copy())
    for Vr in biases[biases < Vnear]:
        ps, nn, pp, _, ok, _ = gummel(Vr, x, h, *ug)
        if ok:
            ug = (ps, nn, pp)
    _, _, _, warm_h, _, _ = gummel(Vnear, x, h, *ug)

    print(f"max reverse bias reached:  cold(from eq) {cold_reached:.1f} Vt   "
          f"continuation {warm_reached:.1f} Vt   (Vmax {args.vmax})")
    if len(IV) > 2:
        print(f"reverse |J| across sweep: {IV[1][1]:.2e} -> {IV[-1][1]:.2e}  "
              f"(mild impact-ionization multiplication ~x{IV[-1][1] / max(IV[1][1], 1e-30):.1f})")

    _plot(args.out, IV, biases, cold_ok, cold_reached, warm_reached, cold_h, warm_h, Vnear)
    print(f"wrote {args.out}")


def _plot(out, IV, biases, cold_ok, cold_reached, warm_reached, cold_h, warm_h, Vnear):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    IV = np.array(IV) if IV else np.zeros((1, 2))
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    ax[0, 0].semilogy(IV[:, 0], np.clip(IV[:, 1], 1e-30, None), "-o", ms=3, color="#1f77b4")
    ax[0, 0].set_title("reverse-bias I-V (leakage + mild impact-ionization\nmultiplication), traced by continuation")
    ax[0, 0].set_xlabel("reverse bias |V| / V_t"); ax[0, 0].set_ylabel("|J| (scaled, log)")
    ax[0, 0].grid(alpha=0.3, which="both")

    ax[0, 1].bar(["cold\n(from equilibrium)", "continuation\n(warm-start)"],
                 [cold_reached, warm_reached], color=["#d62728", "#1f77b4"])
    ax[0, 1].set_ylabel("max reverse bias converged / V_t")
    ax[0, 1].set_title("coupled-solve reach: cold leaves its basin early,\ncontinuation reaches high reverse bias")
    for i, v in enumerate([cold_reached, warm_reached]):
        ax[0, 1].text(i, v + 0.4, f"{v:.1f}", ha="center")

    ax[1, 0].semilogy(range(1, len(cold_h) + 1), np.clip(cold_h, 1e-12, None), "-o", ms=3,
                      color="#d62728", label="cold (from equilibrium)")
    ax[1, 0].semilogy(range(1, len(warm_h) + 1), np.clip(warm_h, 1e-12, None), "-o", ms=3,
                      color="#1f77b4", label="warm (continuation)")
    ax[1, 0].axhline(GUMMEL_TOL, color="gray", ls=":", label="tol")
    ax[1, 0].set_title(f"Gummel update norm at V={Vnear:.0f} Vt (near breakdown)\n"
                       "cold stalls/fails, warm converges")
    ax[1, 0].set_xlabel("Gummel iteration"); ax[1, 0].set_ylabel("max |psi update| (log)")
    ax[1, 0].legend(); ax[1, 0].grid(alpha=0.3, which="both")

    conv = np.array(cold_ok)
    ax[1, 1].plot(biases[:len(conv)], conv, "-o", ms=3, color="#d62728")
    ax[1, 1].axvline(cold_reached, color="#d62728", ls=":", alpha=0.6)
    ax[1, 1].set_title("cold solve convergence vs reverse bias\n(1 = converged, 0 = failed)")
    ax[1, 1].set_xlabel("reverse bias |V| / V_t"); ax[1, 1].set_ylabel("cold converged?")
    ax[1, 1].set_ylim(-0.1, 1.1)

    fig.suptitle("Reverse-bias coupled drift-diffusion — bias continuation (warm-start) is "
                 "necessary: a cold solve leaves its convergence basin", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main()
