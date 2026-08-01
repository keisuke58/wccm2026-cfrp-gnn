"""daikin_shapes_3d.py -- 3D figures of the Daikin CFRTP molded shapes.

From the Daikin fluoropolymer/carbon-fiber CFRTP development report (2021): the
demonstrated molded shapes are a deep-draw box (cubic), a hemisphere with a rib,
and a ring; the base product forms are UD [0/90] sheet and chopped sheet. This
renders those shapes in 3D for talks/thesis (matplotlib mplot3d, no extra deps).

    python3 design/daikin_shapes_3d.py     # writes design/daikin_shapes_3d.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CARBON = "#2b2f36"; EDGE = "#454b54"; ACC = "#12b3a0"


def ring(ax):
    """Washer / ring: annulus extruded in z."""
    Ro, Ri, h = 1.0, 0.62, 0.30
    th = np.linspace(0, 2*np.pi, 80); z = np.linspace(0, h, 2)
    T, Z = np.meshgrid(th, z)
    for R in (Ro, Ri):                                  # outer + inner walls
        ax.plot_surface(R*np.cos(T), R*np.sin(T), Z, color=CARBON, alpha=.95,
                        linewidth=0, antialiased=True)
    rr = np.linspace(Ri, Ro, 2)
    RR, TT = np.meshgrid(rr, th)
    for zc in (0, h):                                   # top + bottom annular caps
        ax.plot_surface(RR*np.cos(TT), RR*np.sin(TT), np.full_like(RR, zc),
                        color=CARBON, alpha=.98, linewidth=0)
    ax.set_title("Ring-shaped", color=ACC, fontsize=11)
    _fmt(ax, 1.1)


def box(ax):
    """Deep-draw box (open cup)."""
    a, h = 0.9, 0.9; t = a
    # four walls
    xs = np.linspace(-t, t, 2); zs = np.linspace(0, h, 2)
    X, Z = np.meshgrid(xs, zs)
    ax.plot_surface(X, np.full_like(X, -t), Z, color=CARBON, alpha=.9, linewidth=0)
    ax.plot_surface(X, np.full_like(X,  t), Z, color=CARBON, alpha=.6, linewidth=0)
    ax.plot_surface(np.full_like(X, -t), X, Z, color=CARBON, alpha=.8, linewidth=0)
    ax.plot_surface(np.full_like(X,  t), X, Z, color=CARBON, alpha=.7, linewidth=0)
    yy = np.linspace(-t, t, 2); Xb, Yb = np.meshgrid(xs, yy)   # bottom
    ax.plot_surface(Xb, Yb, np.zeros_like(Xb), color=CARBON, alpha=.98, linewidth=0)
    ax.set_title("Deep-draw box (cubic)", color=ACC, fontsize=11)
    _fmt(ax, 1.0)


def dome(ax):
    """Hemisphere with a base rib (flange)."""
    u = np.linspace(0, 2*np.pi, 60); v = np.linspace(0, np.pi/2, 30)
    U, V = np.meshgrid(u, v); r = 0.9
    ax.plot_surface(r*np.cos(U)*np.sin(V), r*np.sin(U)*np.sin(V), r*np.cos(V),
                    color=CARBON, alpha=.95, linewidth=0)
    # rib: a thin flange ring at the base
    Ro, Ri = 1.15, 0.9; th = np.linspace(0, 2*np.pi, 80)
    rr = np.linspace(Ri, Ro, 2); RR, TT = np.meshgrid(rr, th)
    ax.plot_surface(RR*np.cos(TT), RR*np.sin(TT), np.zeros_like(RR),
                    color=EDGE, alpha=.95, linewidth=0)
    z2 = np.linspace(0, 0.12, 2); T2, Z2 = np.meshgrid(th, z2)   # rib thickness
    ax.plot_surface(Ro*np.cos(T2), Ro*np.sin(T2), Z2, color=EDGE, alpha=.9, linewidth=0)
    ax.set_title("Hemisphere + rib", color=ACC, fontsize=11)
    _fmt(ax, 1.2)


def _fmt(ax, L):
    ax.set_box_aspect((1, 1, 0.7))
    ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_zlim(0, 1.3*L)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_facecolor("white"); ax.view_init(elev=22, azim=-60)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0.0)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    fig = plt.figure(figsize=(12, 4.4), dpi=130)
    fig.suptitle("Daikin fluoropolymer/CF CFRTP — demonstrated molded shapes (3D)",
                 fontweight="bold", color="#1f3864")
    for i, fn in enumerate((ring, box, dome)):
        ax = fig.add_subplot(1, 3, i+1, projection="3d"); fn(ax)
    fig.text(0.5, 0.02, "product forms: UD [0/90] sheet & chopped sheet · shapes per "
             "Daikin CFRTP development report (2021)", ha="center", fontsize=8.5, color="#5a6675")
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    out = os.path.join(here, "daikin_shapes_3d.png")
    fig.savefig(out, dpi=140, facecolor="white"); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
