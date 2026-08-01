"""gen_viz_anim.py -- animated GIFs of the CFRTP process physics for slides/thesis.

Same constitutive models as the Abaqus UMAT decks (Nakamura crystallization,
generalized-Maxwell + WLF viscoelasticity, mixed-mode cohesive front), rendered
as compact animations. Pure matplotlib + Pillow (no ffmpeg needed).

    python3 viz/gen_viz_anim.py            # writes viz/*.gif (+ final-frame PNGs)

Magnitudes are illustrative (uncalibrated); the shapes are the physics. The
interactive version lives as a claude.ai artifact (see viz/README.md)."""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# ---- palette (matches the interactive artifact) ----------------------------
HEAT, CRYST, COMP, TENS = "#ff7a1a", "#12b3a0", "#2f6fed", "#e23b48"
INK, MUTED, GRID, BG = "#141922", "#5a6675", "#dfe4ea", "#ffffff"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID,
    "font.size": 11,
})
NFR = 64


# ---- shared physics (ports of the verified mirrors) ------------------------
def _beta_exp(dt, tau):
    tau = max(tau, 1e-20); xr = dt / tau; e = math.exp(-xr)
    return ((1 - e) / xr if xr > 1e-6 else 1 - 0.5 * xr), e


def sim_process(Tc=290.0, W=45.0, Kmax=12.0, ginf=0.3, n=2.5, N=300):
    """Melt->cool: relative crystallinity (Nakamura) + constrained-ply residual
    stress with generalized-Maxwell/WLF+crystallinity-shift relaxation."""
    sgk = [0.4, 0.2, 0.1]; scale = (1 - ginf) / sum(sgk)
    GK = [g * scale for g in sgk]; TAU = [0.02, 0.1, 0.5]
    C, A, beta, g0, agel, BX = 1e10, 30e-6, -4e-3, 0.01, 0.1, 2.0
    wc1, wc2, wtref = 17.4, 51.6, 143.0
    Thot, Trt, thold = 380.0, 25.0, 0.15
    dt = 1.0 / N

    def Temp(t):
        return Thot if t <= thold else Thot - (Thot - Trt) * (t - thold) / (1 - thold)

    t = np.linspace(0, 1, N + 1)
    T = np.array([Temp(x) for x in t])
    al = np.zeros(N + 1); sg = np.zeros(N + 1)
    a = 1e-3; q = [0.0, 0.0, 0.0]; qinf = 0.0; prevT = Temp(0.0); al[0] = a
    for i in range(1, N + 1):
        Tn = T[i]; Tm = 0.5 * (prevT + Tn); dT = Tn - prevT
        aa = min(max(a, 1e-8), 1 - 1e-10); argl = max(-math.log(1 - aa), 1e-12)
        rate = n * Kmax * math.exp(max(-60.0, -((Tm - Tc) / W) ** 2)) * (1 - aa) * argl ** ((n - 1) / n)
        a = min(a + max(rate * dt, 0.0), 1.0); al[i] = a
        x = min(max((a - agel) / (1 - agel), 0.0), 1.0); g = g0 + (1 - g0) * x
        deig = A * dT + beta * (a - al[i - 1]); dsi = g * C * (0 - deig)
        dtt = Tm - wtref; den = wc2 + dtt; den = den if den >= 1 else 1.0
        pw = max(-30.0, min(30.0, -wc1 * dtt / den)); aT = 10 ** pw
        aX = 10 ** min(30.0, BX * a); ash = aT * aX
        qinf += ginf * dsi; s = qinf
        for k in range(3):
            b, e = _beta_exp(dt, ash * TAU[k]); q[k] = e * q[k] + GK[k] * b * dsi; s += q[k]
        sg[i] = s; prevT = Tn
    return t, T, al, sg / 1e6


def sim_cure(ginf=0.30, tauScale=1.0, N=300):
    sgk = [0.4, 0.2, 0.1]; scale = (1 - ginf) / sum(sgk); GK = [g * scale for g in sgk]
    TAU = [0.02 * tauScale, 0.1 * tauScale, 0.5 * tauScale]
    C, A = 1e10, 28e-6; wc1, wc2, wtref = 20.0, 80.0, 140.0
    amp = [(0, 25), (0.25, 180), (0.55, 180), (1.0, 25)]

    def Tof(t):
        for (a0, v0), (a1, v1) in zip(amp, amp[1:]):
            if t <= a1:
                return v0 + (v1 - v0) * (t - a0) / (a1 - a0)
        return 25.0
    dt = 1.0 / N; t = np.linspace(0, 1, N + 1)
    T = np.array([Tof(x) for x in t]); el = np.zeros(N + 1); ve = np.zeros(N + 1)
    prevT = Tof(0.0); qinf = 0.0; q = [0.0, 0.0, 0.0]; sigE = 0.0
    for i in range(1, N + 1):
        Tn = T[i]; Tm = 0.5 * (prevT + Tn); dT = Tn - prevT
        dsi = C * (-(A * dT)); sigE += dsi; el[i] = sigE
        dtt = Tm - wtref; den = wc2 + dtt; den = den if den >= 1 else 1.0
        pw = max(-30.0, min(30.0, -wc1 * dtt / den)); aT = 10 ** pw
        qinf += ginf * dsi; s = qinf
        for k in range(3):
            b, e = _beta_exp(dt, aT * TAU[k]); q[k] = e * q[k] + GK[k] * b * dsi; s += q[k]
        ve[i] = s; prevT = Tn
    return t, T, el / 1e6, ve / 1e6


def sdeg_field(load, theta_deg, nx=64, ny=24, a0=0.25):
    th = math.radians(theta_deg); GIc, GIIc, eta = 1.0, 3.0, 1.6
    B = math.sin(th) ** 2; Gc = GIc + (GIIc - GIc) * B ** eta
    drive = load * 2.4 / Gc
    F = np.zeros((ny, nx))
    for iy in range(ny):
        yc = (iy + 0.5) / ny; curve = (theta_deg / 90) * 0.10 * math.cos((yc - 0.5) * math.pi)
        for ix in range(nx):
            xc = ix / nx
            if xc < a0:
                F[iy, ix] = 1.0
            else:
                pos = (xc - a0) / (1 - a0)
                F[iy, ix] = min(max((min(max(drive + curve, 0), 1.4) - pos) / 0.12, 0.0), 1.0)
    return F, Gc


# ---- GIF 1: crystallization -> residual stress -----------------------------
def anim_cryst(path):
    t, T, al, sg = sim_process()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4.6), height_ratios=[1.15, 1], dpi=90)
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.11, hspace=0.35)
    fig.suptitle("CFRTP: melt → cool → residual stress", fontweight="bold", x=0.12, ha="left")
    axT = ax1; axA = ax1.twinx()
    axT.set_ylim(0, 400); axA.set_ylim(0, 1.05); axT.set_xlim(0, 1)
    axT.set_ylabel("T [°C]", color=HEAT); axA.set_ylabel("crystallinity α", color=CRYST)
    axT.grid(True, color=GRID, lw=.8)
    (lT,) = axT.plot([], [], color=HEAT, lw=2.4)
    (lA,) = axA.plot([], [], color=CRYST, lw=2.4)
    ptA = axA.scatter([], [], color=CRYST, zorder=5, s=28)
    ax2.set_xlim(0, 1); ax2.set_ylim(sg.min() * 1.15 - 2, sg.max() * 1.15 + 2)
    ax2.set_ylabel("residual σ₁₁ [MPa]*", color=TENS); ax2.set_xlabel("time (melt → cool)")
    ax2.grid(True, color=GRID, lw=.8); ax2.axhline(0, color=GRID, lw=1)
    (lS,) = ax2.plot([], [], color=TENS, lw=2.6)
    txt = ax2.text(0.03, 0.9, "", transform=ax2.transAxes, family="monospace",
                   fontsize=10, color=INK, va="top")

    def upd(f):
        k = int((f / (NFR - 1)) * (len(t) - 1))
        lT.set_data(t[:k + 1], T[:k + 1]); lA.set_data(t[:k + 1], al[:k + 1])
        lS.set_data(t[:k + 1], sg[:k + 1]); ptA.set_offsets([[t[k], al[k]]])
        txt.set_text("α = %.2f\nσ₁₁ = %+.1f MPa*" % (al[k], sg[k]))
        return lT, lA, lS, ptA, txt
    FuncAnimation(fig, upd, frames=NFR, blit=False).save(path, writer=PillowWriter(fps=18))
    upd(NFR - 1); fig.savefig(path.replace(".gif", ".png"), dpi=110); plt.close(fig)


# ---- GIF 2: elastic vs viscoelastic ----------------------------------------
def anim_visco(path):
    t, T, el, ve = sim_cure()
    fig, ax = plt.subplots(figsize=(8, 4.0), dpi=90)
    fig.subplots_adjust(left=0.11, right=0.88, top=0.88, bottom=0.14)
    fig.suptitle("Elastic vs. viscoelastic residual stress", fontweight="bold", x=0.11, ha="left")
    ax.set_xlim(0, 1); lo = min(el.min(), ve.min()) * 1.15 - 2; hi = max(el.max(), ve.max()) * 1.15 + 2
    ax.set_ylim(lo, hi); ax.set_xlabel("cure cycle: 25 → 180 → 25 °C")
    ax.set_ylabel("σ₁₁ [MPa]*"); ax.grid(True, color=GRID, lw=.8); ax.axhline(0, color=GRID, lw=1)
    axT = ax.twinx(); axT.set_ylim(0, 200); axT.set_ylabel("T [°C]", color=HEAT)
    (lTe,) = axT.plot([], [], color=HEAT, lw=1.5, ls=(0, (5, 4)))
    (lE,) = ax.plot([], [], color=COMP, lw=2.3, ls=(0, (6, 4)), label="elastic (no relaxation)")
    (lV,) = ax.plot([], [], color=TENS, lw=2.8, label="viscoelastic (relaxed)")
    ax.legend(loc="upper left", framealpha=.9, fontsize=9)
    txt = ax.text(0.97, 0.06, "", transform=ax.transAxes, family="monospace",
                  fontsize=10, color=INK, ha="right")

    def upd(f):
        k = int((f / (NFR - 1)) * (len(t) - 1))
        lTe.set_data(t[:k + 1], T[:k + 1]); lE.set_data(t[:k + 1], el[:k + 1]); lV.set_data(t[:k + 1], ve[:k + 1])
        ep = np.abs(el[:k + 1]).max() if k else 0.0
        pct = (1 - abs(ve[k]) / ep) * 100 if ep > 0 else 0.0
        txt.set_text("relaxed ≈ %2.0f%%" % max(pct, 0))
        return lTe, lE, lV, txt
    FuncAnimation(fig, upd, frames=NFR, blit=False).save(path, writer=PillowWriter(fps=18))
    upd(NFR - 1); fig.savefig(path.replace(".gif", ".png"), dpi=110); plt.close(fig)


# ---- GIF 3: delamination front ---------------------------------------------
def anim_delam(path, theta=30.0):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("sdeg", [CRYST, HEAT, TENS])
    fig, ax = plt.subplots(figsize=(8, 3.1), dpi=90)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.16)
    fig.suptitle("Mixed-mode delamination front (θ = %d°)" % theta, fontweight="bold", x=0.08, ha="left")
    F0, Gc = sdeg_field(0.0, theta)
    im = ax.imshow(F0, origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=1,
                   extent=[0, 1, 0, 1])
    ax.set_xlabel("length  (loaded tip → bonded)"); ax.set_ylabel("width")
    ax.set_yticks([])
    txt = ax.text(0.98, 0.06, "", transform=ax.transAxes, family="monospace",
                  fontsize=10, color="#ffffff", ha="right",
                  bbox=dict(boxstyle="round,pad=0.3", fc=INK, ec="none", alpha=.65))

    def upd(f):
        load = f / (NFR - 1)
        F, Gc = sdeg_field(load, theta)
        im.set_data(F)
        dmg = (F > 0.5).mean() * 100
        front = 0.0
        cols = (F > 0.5).any(axis=0)
        if cols.any():
            front = (np.where(cols)[0].max() + 1) / F.shape[1] * 100
        txt.set_text("opening %3.0f%%   front %3.0f%%   G_c/G_Ic %.2f" % (load * 100, front, Gc))
        return im, txt
    FuncAnimation(fig, upd, frames=NFR, blit=False).save(path, writer=PillowWriter(fps=18))
    upd(NFR - 1); fig.savefig(path.replace(".gif", ".png"), dpi=110); plt.close(fig)


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    anim_cryst(os.path.join(here, "cfrtp_crystallization_residual.gif"))
    anim_visco(os.path.join(here, "cfrtp_viscoelastic_relaxation.gif"))
    anim_delam(os.path.join(here, "cfrtp_delamination_front.gif"))
    print("wrote viz/*.gif (+ final-frame *.png)")
