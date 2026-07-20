"""GAT (Graph Attention Network) mechanism schematic.

A clean, publication-quality figure explaining how the Graph Attention layer
used in this repo (``GATModel`` / ``_make_conv`` in ``train.py``) turns a node
and its mesh neighbours into an updated feature vector.

The figure has three coupled panels that read left-to-right:

  (A) Local neighbourhood on the CFRP mesh graph.  A centre node ``i`` gathers
      messages from its 1-hop neighbours ``j in N(i)``.  Edge thickness encodes
      the learned attention weight ``alpha_ij`` (thicker = more important).

  (B) Attention-coefficient pipeline for a single edge (i, j):
        z = W h        (shared linear projection)
        e_ij = LeakyReLU( a^T [ W h_i || W h_j ] )   (GATv2: a^T LeakyReLU(...))
        alpha_ij = softmax_j( e_ij )                 (normalise over N(i))

  (C) Multi-head aggregation (K = 4 heads, as in ``_HEADS`` / ``heads=4``):
        h_i' = ||_{k=1}^{K}  sigma( sum_j alpha_ij^k  W^k h_j )
      i.e. each head attends independently and the K outputs are concatenated
      (``concat=True``), then fed to the next layer with a residual connection.

Style follows the other paper_figs plot scripts (rounded FancyBboxPatch boxes,
blue palette, optional thesis_style / usetex).

Run:  python3 paper_figs/plot_gat_mechanism.py
Out:  paper_figs/gat_mechanism.png  and  paper_figs/gat_mechanism.pdf
"""
import os
import sys

# Make TeX Live discoverable so thesis_style's usetex path works when present.
_TEXBIN = "/home/nishioka/texlive/2025/bin/x86_64-linux"
if os.path.isdir(_TEXBIN) and _TEXBIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _TEXBIN + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

# ── style: try thesis_style (usetex); fall back to a clean mathtext default ───
USETEX = False
try:
    sys.path.insert(
        0, os.path.join(os.path.dirname(__file__), "..", "slides", "figure_sources")
    )
    from thesis_style import use  # noqa: E402

    use(width_frac=1.0, aspect=0.42)
    USETEX = matplotlib.rcParams.get("text.usetex", False)
    if USETEX:
        from matplotlib.texmanager import TexManager

        try:
            TexManager().make_dvi(r"\textbf{x}", 9)
        except Exception:
            matplotlib.rcParams["text.usetex"] = False
            USETEX = False
except Exception:
    pass

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 11,
    "mathtext.fontset": "dejavusans",
    "savefig.dpi": 300,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# ── palette ───────────────────────────────────────────────────────────────
C_CENTER = "#1b4b91"   # centre node i (deep blue)
C_NEIGH = "#4c8fd1"    # neighbour nodes j (mid blue)
C_ACCENT = "#e8722c"   # attention / highlight (orange)
C_BOX = "#eaf1fb"      # soft box fill
C_BOX_EDGE = "#1b4b91"
C_HEAD = ["#1b4b91", "#3f78bd", "#6aa3d8", "#e8722c"]  # K=4 heads
C_GREY = "#5a5a5a"
C_LINE = "#9bb7d4"


def _box(ax, x, y, w, h, text, fc=C_BOX, ec=C_BOX_EDGE, fs=10.5, lw=1.4, tc="#12315e"):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.045",
        fc=fc, ec=ec, lw=lw, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc, zorder=4)


def _arrow(ax, p0, p1, color=C_GREY, lw=1.6, style="-|>", ms=9, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=ms,
        lw=lw, color=color, ls=ls,
        connectionstyle=f"arc3,rad={rad}", zorder=2,
        shrinkA=2, shrinkB=2))


# ── figure / panels ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13.4, 5.6))
gsA = fig.add_axes([0.010, 0.07, 0.30, 0.80]); gsA.axis("off")
gsB = fig.add_axes([0.335, 0.07, 0.34, 0.80]); gsB.axis("off")
gsC = fig.add_axes([0.700, 0.07, 0.295, 0.80]); gsC.axis("off")
for ax in (gsA, gsB, gsC):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

# panel titles
gsA.text(0.5, 1.02, r"(A) Mesh neighbourhood $\mathcal{N}(i)$",
         ha="center", va="bottom", fontsize=12.5, fontweight="bold", color="#12315e")
gsB.text(0.5, 1.02, r"(B) Attention coefficient  $\alpha_{ij}$",
         ha="center", va="bottom", fontsize=12.5, fontweight="bold", color="#12315e")
gsC.text(0.5, 1.02, r"(C) Multi-head update  $h_i'$",
         ha="center", va="bottom", fontsize=12.5, fontweight="bold", color="#12315e")

# ============================================================================
# Panel A — a patch of the CFRP FE mesh: unstructured grid of nodes, the
# centre node i and its 1-hop neighbours, edges weighted by attention.
# ============================================================================
from matplotlib.tri import Triangulation
from matplotlib.collections import LineCollection

rng = np.random.default_rng(7)
nx, ny = 7, 6
gx, gy = np.meshgrid(np.linspace(0.10, 0.90, nx), np.linspace(0.16, 0.86, ny))
# jitter interior nodes so the grid reads as an unstructured FE mesh
jit = 0.028
interior = (gx > 0.10 + 1e-3) & (gx < 0.90 - 1e-3) & (gy > 0.16 + 1e-3) & (gy < 0.86 - 1e-3)
gx = gx + jit * rng.standard_normal(gx.shape) * interior
gy = gy + jit * rng.standard_normal(gy.shape) * interior
px, py = gx.ravel(), gy.ravel()

# a smooth scalar field (evokes the DSPSS stress feature) for node colour
field = (np.exp(-((px - 0.5) ** 2 + (py - 0.52) ** 2) / 0.05)
         + 0.4 * np.exp(-((px - 0.72) ** 2 + (py - 0.30) ** 2) / 0.03))

tri = Triangulation(px, py)
# unique undirected edges from the triangulation
edges = set()
for a, b, c in tri.triangles:
    for u, v in ((a, b), (b, c), (c, a)):
        edges.add((min(u, v), max(u, v)))
edges = np.array(sorted(edges))

# faint mesh edges (background)
seg = np.stack([np.c_[px[edges[:, 0]], py[edges[:, 0]]],
                np.c_[px[edges[:, 1]], py[edges[:, 1]]]], axis=1)
gsA.add_collection(LineCollection(seg, colors="#c7d6ea", linewidths=0.8, zorder=1))

# faint field colouring of all nodes (context)
gsA.scatter(px, py, c=field, cmap="Blues", s=42, edgecolors="white",
            linewidths=0.7, zorder=2, vmin=field.min(), vmax=field.max() * 1.05)

# choose centre node i (closest to grid centre) and its 1-hop neighbours
ci = int(np.argmin((px - 0.5) ** 2 + (py - 0.52) ** 2))
nbr = sorted({v for (u, v) in edges if u == ci} | {u for (u, v) in edges if v == ci})
# illustrative normalised attention weights over the neighbours
araw = rng.uniform(0.4, 1.0, size=len(nbr)); araw[len(nbr) // 2] = 1.7  # one dominant
alpha = araw / araw.sum()

# attention-weighted edges from i to each neighbour (thickness + colour)
for a, jn in zip(alpha, nbr):
    lw = 1.2 + 10.0 * a
    col = C_ACCENT if a >= alpha.max() - 1e-9 else C_NEIGH
    gsA.plot([px[ci], px[jn]], [py[ci], py[jn]], color=col, lw=lw,
             solid_capstyle="round", zorder=3, alpha=0.95)

# neighbour nodes
for jj, (a, jn) in enumerate(zip(alpha, nbr)):
    gsA.add_patch(Circle((px[jn], py[jn]), 0.028, fc=C_NEIGH, ec="white",
                         lw=1.5, zorder=5))
    gsA.text(px[jn], py[jn], r"$j$", ha="center", va="center",
             color="white", fontsize=8.5, zorder=6)
# highlight the dominant-attention edge label
jmax = int(np.argmax(alpha)); jn = nbr[jmax]
mid = np.array([(px[ci] + px[jn]) / 2, (py[ci] + py[jn]) / 2])
gsA.text(mid[0] + 0.02, mid[1] + 0.03, r"$\alpha_{ij}$", ha="center", va="center",
         fontsize=10, color=C_ACCENT, zorder=7, fontweight="bold")

# centre node i
gsA.add_patch(Circle((px[ci], py[ci]), 0.040, fc=C_CENTER, ec="white",
                     lw=2.0, zorder=6))
gsA.text(px[ci], py[ci], r"$i$", ha="center", va="center",
         color="white", fontsize=12, fontweight="bold", zorder=7)

gsA.text(0.5, 0.045,
         "node = mesh vertex   feature $h=[x,y,z,\\mathrm{DSPSS}]$\n"
         "edge thickness $\\propto$ learned attention $\\alpha_{ij}$",
         ha="center", va="bottom", fontsize=8.8, color=C_GREY)

# ============================================================================
# Panel B — attention pipeline for a single edge (i, j)
# ============================================================================
xc = 0.5
# inputs
_box(gsB, 0.26, 0.90, 0.34, 0.11, r"$h_i$   (centre)", fc="#dbe6f7")
_box(gsB, 0.74, 0.90, 0.34, 0.11, r"$h_j$   (neighbour)", fc="#dbe6f7")

# shared linear projection W
_box(gsB, 0.26, 0.71, 0.34, 0.115, r"$W\,h_i$", fc=C_BOX)
_box(gsB, 0.74, 0.71, 0.34, 0.115, r"$W\,h_j$", fc=C_BOX)
_arrow(gsB, (0.26, 0.845), (0.26, 0.77))
_arrow(gsB, (0.74, 0.845), (0.74, 0.77))
gsB.text(0.5, 0.712, r"shared $W$", ha="center", va="center",
         fontsize=8.6, color=C_GREY, style="italic")

# concat + attention vector a  -> raw score e_ij
_box(gsB, 0.5, 0.505, 0.80, 0.13,
     r"$e_{ij}=\mathrm{LeakyReLU}\!\left(a^{\top}\,[\,W h_i \,\Vert\, W h_j\,]\right)$",
     fc="#fdece0", ec=C_ACCENT, tc="#8a3d12", fs=11)
_arrow(gsB, (0.26, 0.652), (0.42, 0.572), color=C_GREY, rad=0.15)
_arrow(gsB, (0.74, 0.652), (0.58, 0.572), color=C_GREY, rad=-0.15)

# softmax normalisation over N(i)
_box(gsB, 0.5, 0.285, 0.80, 0.13,
     r"$\alpha_{ij}=\dfrac{\exp(e_{ij})}{\sum_{k\in\mathcal{N}(i)}\exp(e_{ik})}$",
     fc="#eaf1fb", ec=C_BOX_EDGE, fs=11.5)
_arrow(gsB, (0.5, 0.44), (0.5, 0.352), color=C_ACCENT, lw=2.0)
gsB.text(0.845, 0.393, "softmax\nover $\\mathcal{N}(i)$", ha="center", va="center",
         fontsize=8.4, color=C_ACCENT, style="italic")

# output alpha
_box(gsB, 0.5, 0.085, 0.44, 0.11, r"$\alpha_{ij}\ \in[0,1]$",
     fc=C_ACCENT, ec=C_ACCENT, tc="white", fs=11.5, lw=1.4)
_arrow(gsB, (0.5, 0.22), (0.5, 0.142), color=C_ACCENT, lw=2.0)

# ============================================================================
# Panel C — multi-head aggregation
# ============================================================================
# K heads compute weighted sums in parallel
head_y = [0.80, 0.665, 0.53, 0.395]
for k, y in enumerate(head_y):
    _box(gsC, 0.5, y, 0.86, 0.10,
         rf"head $k{{=}}{k+1}$:  $\sigma\!\left(\sum_{{j}}\alpha_{{ij}}^{{{k+1}}}"
         rf"W^{{{k+1}}} h_j\right)$",
         fc="white", ec=C_HEAD[k], tc="#12315e", fs=9.6, lw=1.8)

gsC.text(0.5, 0.885, r"$K=4$ independent attention heads",
         ha="center", va="bottom", fontsize=9.2, color=C_GREY, style="italic")

# concat bar
_box(gsC, 0.5, 0.235, 0.86, 0.085,
     r"concatenate $K$ heads  (concat=True)",
     fc="#eaf1fb", ec=C_BOX_EDGE, fs=9.6)
for y in head_y:
    _arrow(gsC, (0.5, y - 0.05), (0.5, 0.278), color=C_LINE, lw=1.2, ms=7)

# output
_box(gsC, 0.5, 0.075, 0.72, 0.10,
     r"$h_i' = \Vert_{k=1}^{K}\, h_i^{(k)}$",
     fc=C_CENTER, ec=C_CENTER, tc="white", fs=12)
_arrow(gsC, (0.5, 0.192), (0.5, 0.128), color=C_CENTER, lw=2.0)

# ── connecting flow arrows between panels ────────────────────────────────────
# A -> B  and  B -> C  (drawn in figure coordinates)
figax = fig.add_axes([0, 0, 1, 1]); figax.axis("off")
figax.set_xlim(0, 1); figax.set_ylim(0, 1)
_arrow(figax, (0.305, 0.52), (0.345, 0.52), color=C_GREY, lw=2.2, ms=13)
_arrow(figax, (0.665, 0.30), (0.705, 0.55), color=C_ACCENT, lw=2.2, ms=13, rad=0.12)
figax.text(0.325, 0.55, "gather", ha="center", fontsize=8.2, color=C_GREY, style="italic")
figax.text(0.688, 0.40, r"$\alpha_{ij}$", ha="center", fontsize=9.5, color=C_ACCENT)

# suptitle / caption
fig.suptitle("Graph Attention (GAT) layer — message passing with learned edge weights",
             fontsize=14.5, fontweight="bold", color="#0f2647", y=0.985)
fig.text(0.5, 0.005,
         "As used in GATModel (train.py): GATConv/GATv2Conv, heads=4, concat=True, "
         "3 stacked layers with residual projection + BatchNorm.",
         ha="center", va="bottom", fontsize=8.6, color=C_GREY)

out_png = os.path.join(os.path.dirname(__file__), "gat_mechanism.png")
out_pdf = os.path.join(os.path.dirname(__file__), "gat_mechanism.pdf")
fig.savefig(out_png, bbox_inches="tight", facecolor="white")
fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
print("wrote", out_png)
print("wrote", out_pdf)
