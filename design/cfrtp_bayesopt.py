"""cfrtp_bayesopt.py -- B-5 upgrade: Gaussian-process surrogate + constrained
Bayesian optimization for CFRTP process design, built to address the documented
real-application challenges (see research/ML_APPLICATION_CHALLENGES.md):

  * data scarcity   -> GP is data-efficient; BO minimizes the number of FE calls.
  * uncertainty     -> GP returns mean +/- 2 sigma; the acquisition uses it.
  * extrapolation   -> the design space is bounded, and the recommendation is
                       flagged if the GP predictive std there is large.
  * physics is authority -> every proposed point is EVALUATED by the physics model
                       (process(), the verified HL crystallization + VE law), and
                       the final optimum is FE-verified.

Problem: minimize residual |sigma_11| s.t. final crystallinity alpha >= Xmin, over
the cooling rate (log-spaced). numpy + matplotlib only (a compact GP is implemented
here; no sklearn). Magnitudes illustrative (uncalibrated).

    python3 design/cfrtp_bayesopt.py
"""
import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cfrtp_inverse_design import process, XMIN   # physics authority (shared)

R_LO, R_HI = 5.0, 3000.0        # cooling-rate bounds [C/min] (bounded design space)
_ERF = np.vectorize(math.erf)
def Phi(z):  # standard normal CDF
    return 0.5 * (1.0 + _ERF(z / math.sqrt(2.0)))
def phi(z):
    return np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)


class GP:
    """Minimal GP regression: RBF kernel + noise, y standardized. Lengthscale and
    signal/noise chosen by a small marginal-likelihood grid (robust, dependency-free)."""
    def __init__(self):
        self.ls = 0.2; self.sf2 = 1.0; self.sn2 = 1e-4

    def _K(self, A, B, ls, sf2):
        d = (A[:, None] - B[None, :]) / ls
        return sf2 * np.exp(-0.5 * d * d)

    def fit(self, X, y):
        self.X = np.asarray(X, float)
        self.ymu = float(np.mean(y)); self.ysd = float(np.std(y) + 1e-9)
        self.y = (np.asarray(y, float) - self.ymu) / self.ysd
        best = (np.inf, self.ls, self.sn2)
        n = len(self.X)
        for ls in (0.08, 0.15, 0.25, 0.4, 0.6):
            for sn2 in (1e-4, 1e-3, 1e-2):
                K = self._K(self.X, self.X, ls, 1.0) + sn2 * np.eye(n)
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    continue
                a = np.linalg.solve(L.T, np.linalg.solve(L, self.y))
                nll = 0.5 * self.y @ a + np.sum(np.log(np.diag(L))) + 0.5 * n * math.log(2 * math.pi)
                if nll < best[0]:
                    best = (nll, ls, sn2)
        _, self.ls, self.sn2 = best
        K = self._K(self.X, self.X, self.ls, 1.0) + self.sn2 * np.eye(n)
        self.L = np.linalg.cholesky(K)
        self.a = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y))
        return self

    def predict(self, Xs):
        Xs = np.asarray(Xs, float)
        Ks = self._K(Xs, self.X, self.ls, 1.0)
        mu = Ks @ self.a
        v = np.linalg.solve(self.L, Ks.T)
        var = 1.0 - np.sum(v * v, axis=0)
        var = np.clip(var, 1e-9, None)
        return mu * self.ysd + self.ymu, np.sqrt(var) * self.ysd


def u_of_rate(r):
    return (np.log10(r) - math.log10(R_LO)) / (math.log10(R_HI) - math.log10(R_LO))
def rate_of_u(u):
    return 10 ** (math.log10(R_LO) + u * (math.log10(R_HI) - math.log10(R_LO)))


def main(n_init=4, n_iter=10, seed=0):
    here = os.path.dirname(os.path.abspath(__file__))
    rng = np.random.default_rng(seed)
    # initial design (space-filling in u)
    U = list(np.linspace(0.0, 1.0, n_init))
    S = []; A = []
    for u in U:
        s, a = process(rate_of_u(u)); S.append(s); A.append(a)
    ufine = np.linspace(0, 1, 400)

    for it in range(n_iter):
        gs = GP().fit(U, S); ga = GP().fit(U, A)
        mus, ss = gs.predict(ufine)          # residual stress mean/std
        mua, sa = ga.predict(ufine)          # crystallinity mean/std
        pfeas = Phi((mua - XMIN) / sa)       # P(alpha >= Xmin)
        # best feasible observed so far
        feas_obs = [S[i] for i in range(len(S)) if A[i] >= XMIN]
        best = min(feas_obs) if feas_obs else max(S)
        z = (best - mus) / ss                # improvement for MINIMIZation
        EI = (best - mus) * Phi(z) + ss * phi(z)
        EI = np.clip(EI, 0, None)
        acq = EI * pfeas                     # constrained EI
        u_next = ufine[int(np.argmax(acq))]
        s, a = process(rate_of_u(u_next))    # physics evaluates the proposed point
        U.append(u_next); S.append(s); A.append(a)

    # final recommendation: best feasible sampled point, with GP uncertainty there
    gs = GP().fit(U, S); ga = GP().fit(U, A)
    feas_idx = [i for i in range(len(S)) if A[i] >= XMIN]
    kbest = min(feas_idx, key=lambda i: S[i])
    r_opt = rate_of_u(U[kbest]); sig_opt, alp_opt = S[kbest], A[kbest]
    _, ss_opt = gs.predict([U[kbest]]); std_here = float(ss_opt[0])
    # extrapolation guard: how big is GP std vs the observed spread?
    guard = std_here / (np.std(S) + 1e-9)
    sig_slow, alp_slow = process(R_LO)
    print("=== B-5 constrained Bayesian optimization (GP surrogate, FE-in-the-loop) ===")
    print("FE evaluations used: %d (%d init + %d BO)" % (len(S), n_init, n_iter))
    print("recommended cooling rate r* = %.0f C/min" % r_opt)
    print("  residual |sigma| = %.1f MPa*, alpha = %.2f (constraint >= %.2f: %s)" %
          (sig_opt, alp_opt, XMIN, "OK" if alp_opt >= XMIN - 0.02 else "CHECK"))
    print("  GP std at optimum = %.2f MPa (%.0f%% of data spread) -> %s" %
          (std_here, 100 * guard, "well-supported" if guard < 0.5 else "HIGH uncertainty (add data)"))
    print("  vs slow-safe %g C/min: |sigma|=%.1f -> residual cut %.0f%%" %
          (R_LO, sig_slow, 100 * (1 - sig_opt / sig_slow)))

    # ---- figure: GP mean +/- 2 sigma, feasibility, samples, optimum ----
    TENS, CRYST, INK, GRID = "#e23b48", "#12b3a0", "#141922", "#dfe4ea"
    mus, ss = gs.predict(ufine); mua, sa = ga.predict(ufine)
    rf = np.array([rate_of_u(u) for u in ufine])
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GRID})
    fig, ax = plt.subplots(figsize=(7.8, 4.8), dpi=120); ax.set_xscale("log")
    ax.fill_between(rf, mus - 2 * ss, mus + 2 * ss, color=TENS, alpha=0.15, label="GP ±2σ")
    ax.plot(rf, mus, color=TENS, lw=2.2, label="GP mean: residual |σ₁₁|")
    ax.plot([rate_of_u(u) for u in U], S, "o", color=TENS, ms=5, label="FE evaluations")
    ax.set_xlabel("cooling rate [°C/min, log]"); ax.set_ylabel("residual |σ₁₁| [MPa]*", color=TENS)
    ax.tick_params(axis="y", colors=TENS); ax.grid(True, which="both", color=GRID, lw=.7)
    ax2 = ax.twinx()
    ax2.plot(rf, mua, color=CRYST, lw=2.0, label="GP mean: α")
    ax2.axhline(XMIN, color=CRYST, ls="--", lw=1.2); ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("relative crystallinity α", color=CRYST); ax2.tick_params(axis="y", colors=CRYST)
    ax.axvline(r_opt, color=INK, ls=":", lw=1.4)
    ax.plot([r_opt], [sig_opt], "*", color="#fff", ms=18, mec=INK, mew=1.2, zorder=6)
    ax.set_title("B-5 constrained Bayesian optimization (GP + FE-in-the-loop)\n"
                 "%d FE evals → r*=%.0f °C/min, residual −%.0f%% (α≥%.2f), GP-uncertainty aware"
                 % (len(S), r_opt, 100 * (1 - sig_opt / sig_slow), XMIN), fontsize=9.5)
    fig.tight_layout()
    out = os.path.join(here, "cfrtp_bayesopt.png"); fig.savefig(out, dpi=140); plt.close(fig)
    print("wrote %s" % out)
    return dict(r_opt=r_opt, sig_opt=sig_opt, alp_opt=alp_opt, n_fe=len(S))


if __name__ == "__main__":
    main()
