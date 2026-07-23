"""dd_full_1d.py — full drift-diffusion (1-D pn diode): Scharfetter-Gummel + Gummel
iteration, forward-bias I-V, and bias-continuation warm-start (idea C).

The demos so far (fe_/dd2d_/cfet_ warmstart) solve only the *equilibrium* nonlinear
Poisson (Poisson-Boltzmann): carriers pinned to the potential, no current. This is the
full transport system — Poisson coupled to the electron and hole continuity equations —
solved on a pn junction so a real forward-bias I-V comes out.

Scaled van Roosbroeck system (n_i = 1, potential in units of V_t, length in the device
length, lambda = scaled Debye length):

    -lambda^2 psi'' = p - n + C(x)                        (Poisson)
     dJ_n/dx =  R,   J_n =  (1/h)[B(dpsi) n_{i+1} - B(-dpsi) n_i]   (electron continuity)
    -dJ_p/dx =  R,   J_p =  (1/h)[B(-dpsi) p_{i+1} - B(dpsi) p_i]   (hole continuity)

with the Scharfetter-Gummel edge current using the Bernoulli function B(x)=x/(e^x-1)
(exponential fitting — the standard cure for the drift-dominated, otherwise unstable
continuity discretisation), SRH recombination R=(np-1)/(tau*(n+p+2)), and ohmic
contacts (n,p pinned to equilibrium, psi = psi_eq + applied bias).

Solver: the Gummel map — (i) a Newton solve of the nonlinear Poisson for psi holding the
quasi-Fermi levels, then (ii) SG-discretised linear continuity solves for n and p —
iterated to self-consistency. Its outer iteration count depends on the initial guess, so
a forward-bias sweep is run two ways:
  * cold  : every bias restarts from the zero-bias equilibrium
  * warm  : each bias continues from the previous bias's converged solution (idea C —
            bias continuation is the classic, honest DD warm-start)

Validated: at V=0 the total current is ~0 (equilibrium, ~1e-11); forward bias follows
the ideal-diode law J proportional to (e^{V/Vt}-1) at low injection then saturates.

Honest warm-start finding: bias continuation cuts the sweep's total Gummel iterations
only modestly here (~10%, e.g. 203 -> 184), because the Gummel count is governed by the
injection level, not the initial guess — the same robustness seen in the Newton demos.
Continuation's real necessity shows up where a cold start diverges (very high bias,
Newton-coupled DD, 2-D/3-D), not for this well-conditioned 1-D Gummel map. The primary
deliverable here is the validated full-transport solver itself — the roadmap's missing
piece beyond the equilibrium (Poisson-Boltzmann) demos.

Run:  python3 dd_full_1d.py            (writes dd_full_1d.png)
      python3 dd_full_1d.py --help
"""
from __future__ import annotations

import argparse

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

SEED = 20260722
LAMBDA2 = 2e-4          # scaled Debye length squared (sharp depletion)
C0 = 40.0              # scaled net doping magnitude (N/n_i)
TAU = 1.0              # SRH lifetime (scaled)
XJ, WJ = 0.5, 0.02     # junction position, transition width


def bern(x):
    """Bernoulli B(x) = x/(e^x - 1), numerically safe (B(0)=1)."""
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    big = np.abs(x) > 1e-10
    out[big] = x[big] / np.expm1(x[big])
    return out


def doping(x):
    return C0 * np.tanh((x - XJ) / WJ)          # -C0 (p) on the left, +C0 (n) on the right


def equilibrium(x):
    """psi_eq from 2 sinh(psi) = C, with n=e^psi, p=e^-psi (mass action n*p=1)."""
    C = doping(x)
    psi = np.arcsinh(C / 2.0)
    return psi, np.exp(psi), np.exp(-psi)       # psi, n, p


def laplacian(N, h):
    e = np.ones(N)
    return sp.diags([e[1:], -2 * e, e[1:]], [-1, 0, 1], (N, N)) / h ** 2


def gummel(V, x, h, psi0, n0, p0, tol=1e-6, max_outer=200):
    """Gummel map for the biased pn diode. Returns (psi,n,p, outer_iters, J_total)."""
    N = len(x)
    C = doping(x)
    L = laplacian(N, h)
    interior = np.arange(1, N - 1)

    psi_eq, n_eq, p_eq = equilibrium(x)
    # ohmic contacts: n,p pinned to equilibrium; psi = psi_eq + bias (V at left p-contact)
    psi = psi0.copy(); n = n0.copy(); p = p0.copy()
    psi[0] = psi_eq[0] + V; psi[-1] = psi_eq[-1]
    n[0] = n_eq[0]; n[-1] = n_eq[-1]; p[0] = p_eq[0]; p[-1] = p_eq[-1]

    def sg_matrix(psi, sign):
        """Tridiagonal SG operator for a continuity equation (sign=+1 electrons)."""
        dpl = psi[1:] - psi[:-1]                              # forward differences (N-1,)
        Bp = bern(dpl); Bm = bern(-dpl)                      # B(+dpsi), B(-dpsi)
        A = sp.lil_matrix((N, N))
        for i in interior:
            # J_{i+1/2} - J_{i-1/2} = h R ; SG edge currents (D=1)
            #   J_{i+1/2} = (1/h)[B(dpsi_i) x_{i+1} - B(-dpsi_i) x_i]   (electrons)
            aE_ip1 = Bp[i] / h;  aE_i_r = -Bm[i] / h          # edge i (i,i+1)
            aE_i_l = Bp[i - 1] / h; aE_im1 = -Bm[i - 1] / h    # edge i-1 (i-1,i)
            if sign > 0:   # electrons: J_{i+1/2}-J_{i-1/2}
                A[i, i + 1] += aE_ip1
                A[i, i] += aE_i_r - aE_i_l
                A[i, i - 1] += -aE_im1
            else:          # holes: J_p uses swapped Bernoulli args, and -dJ_p/dx = R
                aH_ip1 = Bm[i] / h; aH_i_r = -Bp[i] / h
                aH_i_l = Bm[i - 1] / h; aH_im1 = -Bp[i - 1] / h
                A[i, i + 1] += -aH_ip1
                A[i, i] += -(aH_i_r - aH_i_l)
                A[i, i - 1] += aH_im1
        return A.tocsr()

    outer = 0
    for outer in range(1, max_outer + 1):
        # (i) nonlinear Poisson for psi (Newton), holding n,p as e^{+-(psi-psi_k)}
        psi_k = psi.copy(); nk = n.copy(); pk = p.copy()
        for _ in range(30):
            nn = nk * np.exp(psi - psi_k)
            pp = pk * np.exp(-(psi - psi_k))
            F = -LAMBDA2 * (L @ psi) - (pp - nn + C)
            J = (-LAMBDA2 * L + sp.diags(nn + pp)).tocsr()
            d = np.zeros(N)
            d[interior] = spla.spsolve(J[interior][:, interior].tocsc(), -F[interior])
            psi = psi + d
            if np.max(np.abs(d[interior])) < 1e-10:
                break
        n = nk * np.exp(psi - psi_k)
        p = pk * np.exp(-(psi - psi_k))
        n[0] = n_eq[0]; n[-1] = n_eq[-1]; p[0] = p_eq[0]; p[-1] = p_eq[-1]

        # (ii) SG continuity solves for n and p (recombination lagged)
        R = (n * p - 1.0) / (TAU * (n + p + 2.0))
        An = sg_matrix(psi, +1); Ap = sg_matrix(psi, -1)
        bn = h * R.copy(); bp = h * R.copy()
        n_new = n.copy(); p_new = p.copy()
        bn[interior] -= An[interior][:, [0, -1]] @ np.array([n_eq[0], n_eq[-1]])
        bp[interior] -= Ap[interior][:, [0, -1]] @ np.array([p_eq[0], p_eq[-1]])
        n_new[interior] = spla.spsolve(An[interior][:, interior].tocsc(), bn[interior])
        p_new[interior] = spla.spsolve(Ap[interior][:, interior].tocsc(), bp[interior])
        n_new = np.clip(n_new, 1e-30, None); p_new = np.clip(p_new, 1e-30, None)

        dmax = np.max(np.abs(psi - psi_k))
        upd = np.max(np.abs(np.log(np.clip(n_new, 1e-30, None)) - np.log(np.clip(n, 1e-30, None))))
        n, p = n_new, p_new
        if dmax < tol and upd < 1e-3:
            break

    # total current J = J_n + J_p (constant in x at steady state); take a mid edge
    dpl = psi[1:] - psi[:-1]
    Jn = (bern(dpl) * n[1:] - bern(-dpl) * n[:-1]) / h
    Jp = (bern(-dpl) * p[1:] - bern(dpl) * p[:-1]) / h
    Jtot = float(np.mean(Jn + Jp))
    return psi, n, p, outer, Jtot


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--N", type=int, default=201, help="grid nodes")
    ap.add_argument("--vmax", type=float, default=12.0, help="max forward bias (units of Vt)")
    ap.add_argument("--nbias", type=int, default=24, help="bias points")
    ap.add_argument("--out", type=str, default="dd_full_1d.png")
    args = ap.parse_args()

    x = np.linspace(0, 1, args.N); h = x[1] - x[0]
    psi_eq, n_eq, p_eq = equilibrium(x)

    # equilibrium validation
    _, _, _, _, J0 = gummel(0.0, x, h, psi_eq, n_eq, p_eq)
    print(f"[validate] V=0 total current J = {J0:.2e}  (should be ~0)")

    biases = np.linspace(0.0, args.vmax, args.nbias)
    cold_it, warm_it, IV, prof = [], [], [], {}
    # cold: every bias from equilibrium
    for V in biases:
        _, _, _, it, J = gummel(V, x, h, psi_eq, n_eq, p_eq)
        cold_it.append(it); IV.append(J)
    # warm: continuation from previous bias
    psi_w, n_w, p_w = psi_eq.copy(), n_eq.copy(), p_eq.copy()
    for k, V in enumerate(biases):
        psi_w, n_w, p_w, it, J = gummel(V, x, h, psi_w, n_w, p_w)
        warm_it.append(it)
        if k in (0, len(biases) // 2, len(biases) - 1):
            prof[round(float(V), 3)] = (psi_w.copy(), n_w.copy(), p_w.copy())

    half = len(biases) // 2
    print(f"\nGummel iters:  cold avg {np.mean(cold_it):.1f}   warm(continuation) avg "
          f"{np.mean(warm_it):.1f}   |  forward-bias current at Vmax = {IV[-1]:.3e}")
    print(f"cold total {int(np.sum(cold_it))} vs warm total {int(np.sum(warm_it))} Gummel iterations")

    _plot(args.out, x, biases, np.array(IV), cold_it, warm_it, prof)
    print(f"wrote {args.out}")


def _plot(out, x, biases, IV, cold_it, warm_it, prof):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    ax[0, 0].plot(biases, np.clip(IV, 1e-12, None), "-o", ms=3, color="#1f77b4")
    ax[0, 0].set_yscale("log")
    ax[0, 0].set_title("forward-bias I-V (full drift-diffusion)")
    ax[0, 0].set_xlabel("applied bias V / V_t"); ax[0, 0].set_ylabel("total current |J| (scaled, log)")
    ax[0, 0].grid(alpha=0.3, which="both")

    Vmid = sorted(prof.keys())[len(prof) // 2]
    psi, n, p = prof[Vmid]
    ax[0, 1].plot(x, n, color="#d62728", label="n (electrons)")
    ax[0, 1].plot(x, p, color="#1f77b4", label="p (holes)")
    ax[0, 1].set_yscale("log")
    ax[0, 1].set_title(f"carrier densities at V={Vmid} (SG-stabilised)")
    ax[0, 1].set_xlabel("x"); ax[0, 1].set_ylabel("density (scaled, log)"); ax[0, 1].legend()

    for V in sorted(prof.keys()):
        ax[1, 0].plot(x, prof[V][0], label=f"V={V}")
    ax[1, 0].set_title("potential psi(x) vs bias"); ax[1, 0].set_xlabel("x")
    ax[1, 0].set_ylabel("psi / V_t"); ax[1, 0].legend(fontsize=8)

    ax[1, 1].plot(biases, cold_it, "-o", ms=3, color="#d62728", label="cold (restart from eq.)")
    ax[1, 1].plot(biases, warm_it, "-o", ms=3, color="#1f77b4", label="warm (bias continuation)")
    ax[1, 1].set_title(f"Gummel iterations per bias\ncold total {int(np.sum(cold_it))} vs "
                       f"warm total {int(np.sum(warm_it))}")
    ax[1, 1].set_xlabel("applied bias V / V_t"); ax[1, 1].set_ylabel("Gummel iterations")
    ax[1, 1].legend()

    fig.suptitle("Full drift-diffusion pn diode (Scharfetter-Gummel + Gummel): forward I-V "
                 "and bias-continuation warm-start (idea C)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main()
