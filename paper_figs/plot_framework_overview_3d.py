"""Framework Figure 1 — 3D-image variant (Composites-B / JSCES).

Same two-structure Stage 0-5 *decision* framework as plot_framework_overview.py,
but with the actual 3D structure renderings embedded as cropped thumbnails above
each structure box, so the two structures are visually concrete:
  - Interstage: cropped left panel "(a) Normalized DSPSS" of interstage_3d.png.
  - Fairing:    cropped rightmost "view 3" nose-cone of fairing_3d.png.

Run:  python3 paper_figs/plot_framework_overview_3d.py
"""
import os
import sys

import numpy as np

# Make TeX Live discoverable so thesis_style's usetex path works when present.
_TEXBIN = "/home/nishioka/texlive/2025/bin/x86_64-linux"
if os.path.isdir(_TEXBIN) and _TEXBIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _TEXBIN + os.pathsep + os.environ.get("PATH", "")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# ── style: try thesis_style (usetex); fall back to a clean mathtext default ───
USETEX = False
try:
    sys.path.insert(
        0, os.path.join(os.path.dirname(__file__), "..", "slides", "figure_sources")
    )
    from thesis_style import use  # noqa: E402

    figsize = use(width_frac=1.0, aspect=0.40)
    # widen for a 12 x ~6.6 full-width schematic regardless of thesis textwidth
    # (taller than the base figure to give the 3D thumbnails breathing room)
    figsize = (12.0, 6.6)
    USETEX = matplotlib.rcParams.get("text.usetex", False)
    # sanity: if latex is unavailable, usetex will crash at draw -> verify here
    if USETEX:
        from matplotlib.texmanager import TexManager

        try:
            TexManager().make_dvi(r"\textbf{x}", 9)
        except Exception:
            matplotlib.rcParams["text.usetex"] = False
            USETEX = False
except Exception:
    figsize = (12.0, 6.6)

if not USETEX:
    matplotlib.rcParams.update({
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 11,
        "mathtext.fontset": "dejavusans",
        "savefig.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def B(s):
    """Bold a string respecting the active text backend."""
    return r"\textbf{%s}" % s if USETEX else s


def pct(s):
    """Escape % for usetex."""
    return s.replace("%", r"\%") if USETEX else s


# ── palette ──────────────────────────────────────────────────────────────────
BLUE        = "#1565C0"   # interstage branch / shared core spine
BLUE_FILL   = "#E3F0FB"
AMBER       = "#E08A00"   # fairing branch
AMBER_FILL  = "#FCEFD6"
CORE_EDGE   = "#1565C0"
CORE_FILL   = "#EDF4FC"
GREY_EDGE   = "#9AA4AD"
GREY_FILL   = "#F4F5F6"
GREEN       = "#1B7A2E"
RED         = "#C62828"
INK         = "#2B2B2B"
BAND_GREY   = "#F0F1F3"
BAND_BLUE   = "#E8F1FB"

ROUND = "round,pad=0.02,rounding_size=0.06"


def box(ax, x, y, w, h, edge, fill, lw=1.3, z=3, shadow=True):
    if shadow:
        sh = FancyBboxPatch((x + 0.05, y - 0.06), w, h, boxstyle=ROUND,
                            linewidth=0, facecolor="#00000010", zorder=z - 1)
        ax.add_patch(sh)
    b = FancyBboxPatch((x, y), w, h, boxstyle=ROUND, linewidth=lw,
                       edgecolor=edge, facecolor=fill, zorder=z)
    ax.add_patch(b)
    return (x, y, w, h)


def arrow(ax, p0, p1, color=INK, lw=1.6, ms=14, z=5,
          style="-|>", cs=None):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, lw=lw,
                        color=color, zorder=z, shrinkA=0, shrinkB=0,
                        connectionstyle=cs, capstyle="round")
    ax.add_patch(a)
    return a


fig, ax = plt.subplots(figsize=figsize)
ax.set_xlim(0, 18.6)
ax.set_ylim(-0.4, 12.5)
ax.axis("off")

# ── background bands ─────────────────────────────────────────────────────────
# left band: structure-specific ; right band: shared decision core
# (taller than the base figure to host the 3D structure thumbnails up top)
BAND_TOP = 11.5
band = FancyBboxPatch((0.3, 0.7), 8.7, BAND_TOP - 0.7,
                      boxstyle="round,pad=0.02,rounding_size=0.25",
                      linewidth=0, facecolor=BAND_GREY, zorder=0)
ax.add_patch(band)
# right band hugs the core boxes + their result tags (no far-right whitespace)
band2 = FancyBboxPatch((9.5, 0.7), 8.8, BAND_TOP - 0.7,
                       boxstyle="round,pad=0.02,rounding_size=0.25",
                       linewidth=0, facecolor=BAND_BLUE, zorder=0)
ax.add_patch(band2)

ax.text(4.65, BAND_TOP + 0.22, B("STRUCTURE-SPECIFIC"), ha="center",
        va="center", fontsize=11, color="#6B7178", zorder=1,
        fontweight=("normal" if USETEX else "bold"))
ax.text(13.9, BAND_TOP + 0.22, B("SHARED DECISION CORE  (structure-agnostic)"),
        ha="center", va="center", fontsize=10.5, color="#3C5A86", zorder=1,
        fontweight=("normal" if USETEX else "bold"))

# ── LEFT: two structures -> two Stage-1 front-ends ───────────────────────────
SW, SH = 3.1, 1.55           # structure boxes
DW, DH = 3.7, 1.55           # detector boxes
struct_x = 0.9
det_x = 5.05

# interstage (blue, upper) ; fairing (amber, lower)
# rows spread apart vertically so each box has a clear strip of empty space
# directly above it to host its own 3D thumbnail (no thumbnail/box overlap)
sy_top, sy_bot = 6.55, 1.45
box(ax, struct_x, sy_top, SW, SH, BLUE, "#FFFFFF")
box(ax, struct_x, sy_bot, SW, SH, AMBER, "#FFFFFF")

ax.text(struct_x + SW / 2, sy_top + SH - 0.42, B("Interstage"),
        ha="center", va="center", fontsize=11.5, color=BLUE)
ax.text(struct_x + SW / 2, sy_top + 0.45, "perforated CFRP",
        ha="center", va="center", fontsize=9.5, color=INK, style="italic")
ax.text(struct_x + SW / 2, sy_bot + SH - 0.42, B("Fairing"),
        ha="center", va="center", fontsize=11.5, color=AMBER)
ax.text(struct_x + SW / 2, sy_bot + 0.45, "honeycomb skin-core",
        ha="center", va="center", fontsize=9.5, color=INK, style="italic")

# detectors
box(ax, det_x, sy_top, DW, DH, BLUE, BLUE_FILL)
box(ax, det_x, sy_bot, DW, DH, AMBER, AMBER_FILL)

ax.text(det_x + DW / 2, sy_top + SH - 0.40, B("surface-stress GNN"),
        ha="center", va="center", fontsize=10.5, color=BLUE)
ax.text(det_x + DW / 2, sy_top + 0.45, "DSPSS  ·  F1 0.79",
        ha="center", va="center", fontsize=9.5, color=GREEN)
ax.text(det_x + DW / 2, sy_bot + SH - 0.40, B("guided-wave GNN"),
        ha="center", va="center", fontsize=10.5, color=AMBER)
ax.text(det_x + DW / 2, sy_bot + 0.45, "Lamb  ·  F1 0.86",
        ha="center", va="center", fontsize=9.5, color=GREEN)

# structure -> detector arrows
arrow(ax, (struct_x + SW, sy_top + SH / 2), (det_x, sy_top + SH / 2),
      color=BLUE)
arrow(ax, (struct_x + SW, sy_bot + SH / 2), (det_x, sy_bot + SH / 2),
      color=AMBER)
ax.text(det_x + DW / 2, sy_top + SH + 0.30, "Stage 1  detection",
        ha="center", va="bottom", fontsize=9, color="#6B7178", style="italic")

# ── RIGHT: shared core spine (vertical stack of 5 boxes) ─────────────────────
CW = 5.9
core_x = 10.2
cx = core_x + CW / 2
# vertical positions (top -> bottom); lifted to centre the spine in the
# taller band used by this 3D variant
CORE_DY = 1.55
ys = [7.05 + CORE_DY, 5.55 + CORE_DY, 4.05 + CORE_DY, 2.55 + CORE_DY,
      1.05 + CORE_DY]
CHs = [1.15, 1.15, 1.30, 1.15, 1.15]

# Stage 0/2
box(ax, core_x, ys[0], CW, CHs[0], CORE_EDGE, CORE_FILL)
ax.text(cx, ys[0] + CHs[0] - 0.40, B("Stage 0/2"), ha="center", va="center",
        fontsize=10.5, color=BLUE)
ax.text(cx, ys[0] + 0.36, "anomaly screen + characterisation",
        ha="center", va="center", fontsize=9.5, color=INK)

# Stage 3 prognosis
box(ax, core_x, ys[1], CW, CHs[1], CORE_EDGE, CORE_FILL)
ax.text(cx, ys[1] + CHs[1] - 0.40, B("Stage 3  prognosis"), ha="center",
        va="center", fontsize=10.5, color=BLUE)
ax.text(cx, ys[1] + 0.36, "pluggable: phase-field / debond",
        ha="center", va="center", fontsize=9.5, color=INK, style="italic")

# Clearance (the eye-catcher: three coloured verdicts)
box(ax, core_x, ys[2], CW, CHs[2], CORE_EDGE, "#FBFCFE", lw=1.6)
ax.text(cx, ys[2] + CHs[2] - 0.38, B("Clearance"), ha="center", va="center",
        fontsize=10.5, color=INK)
# three verdicts on one baseline, coloured
vy = ys[2] + 0.42
ax.text(cx - 1.95, vy, B("OK"), ha="center", va="center", fontsize=11,
        color=GREEN)
ax.text(cx - 0.75, vy, "·", ha="center", va="center", fontsize=11,
        color="#999")
ax.text(cx + 0.05, vy, B("REPAIR"), ha="center", va="center", fontsize=11,
        color=AMBER)
ax.text(cx + 1.35, vy, "·", ha="center", va="center", fontsize=11,
        color="#999")
ax.text(cx + 2.15, vy, B("RETIRE"), ha="center", va="center", fontsize=11,
        color=RED)

# Stage 4 fleet learning
box(ax, core_x, ys[3], CW, CHs[3], CORE_EDGE, CORE_FILL)
ax.text(cx, ys[3] + CHs[3] / 2, B("Stage 4  fleet learning"), ha="center",
        va="center", fontsize=10.5, color=BLUE)

# Stage 5 design feedback
box(ax, core_x, ys[4], CW, CHs[4], CORE_EDGE, CORE_FILL)
ax.text(cx, ys[4] + CHs[4] / 2, B("Stage 5  design feedback"), ha="center",
        va="center", fontsize=10.5, color=BLUE)

# vertical spine arrows (top box bottom -> next box top)
for i in range(len(ys) - 1):
    y_from = ys[i]
    y_to = ys[i + 1] + CHs[i + 1]
    arrow(ax, (cx, y_from), (cx, y_to), color="#5A6B82", lw=1.7, ms=13)

# ── MERGE: two detector front-ends converge into Stage 0/2 ───────────────────
merge_target_top = (core_x, ys[0] + CHs[0] * 0.66)
merge_target_bot = (core_x, ys[0] + CHs[0] * 0.34)
arrow(ax, (det_x + DW, sy_top + SH / 2), merge_target_top, color=BLUE,
      lw=1.8, ms=14, cs="arc3,rad=-0.18")
arrow(ax, (det_x + DW, sy_bot + SH / 2), merge_target_bot, color=AMBER,
      lw=1.8, ms=14, cs="arc3,rad=0.22")
ax.text((det_x + DW + core_x) / 2 - 0.35, (sy_top + sy_bot) / 2 + 1.30,
        "merge", ha="center", va="center", fontsize=9, color="#6B7178",
        style="italic", rotation=0)

# ── DESIGN-FEEDBACK loop: Stage 5 -> back to structures (left) ───────────────
# Route as a low arc UNDER all boxes so it never crosses any box/text.
fb_start = (core_x + 0.9, ys[4])            # bottom edge of Stage 5
fb_end = (struct_x + SW * 0.5, sy_bot)      # bottom edge of Fairing box
arrow(ax, fb_start, fb_end, color="#7E57C2", lw=1.8, ms=14,
      cs="arc3,rad=-0.32", style="-|>")
ax.text(8.7, 0.72, B("design-feedback loop"),
        ha="center", va="center", fontsize=9.5, color="#7E57C2",
        style="italic")

# ── light result annotations (kept subtle) ───────────────────────────────────
# near Stage 0/2
ax.text(core_x + CW + 0.12, ys[0] + CHs[0] / 2, "AUROC 0.85", ha="left",
        va="center", fontsize=8.5, color=GREEN)
# near Clearance (to the right, mirroring the AUROC tag — avoids box overlap)
ax.text(core_x + CW + 0.20, ys[2] + CHs[2] / 2 + 0.20,
        pct("85% / 92.5% acc"), ha="left", va="center", fontsize=8.5,
        color=GREEN)
ax.text(core_x + CW + 0.20, ys[2] + CHs[2] / 2 - 0.20,
        pct("0% dangerous-miss"), ha="left", va="center", fontsize=8.5,
        color=GREEN)
# bottom-of-core calibration / sim-to-real tag (just under Stage 5 box)
ax.text(cx, ys[4] - 0.42, pct("ECE 0.13→0.03   ·   sim-to-real FPR 14→5%"),
        ha="center", va="center", fontsize=8.5, color="#3C5A86")

# ── 3D STRUCTURE THUMBNAILS ──────────────────────────────────────────────────
# Embed the real renderings, cropped in-code to the panel/view we want, as
# inset images directly ABOVE each structure box with a thin border.
_HERE = os.path.dirname(__file__)


def _add_thumb(crop, x_anchor, y_anchor, zoom, border):
    """Place a cropped image array centred-bottom at (x_anchor, y_anchor)."""
    oi = OffsetImage(crop, zoom=zoom, interpolation="hanning")
    ab = AnnotationBbox(
        oi, (x_anchor, y_anchor), xycoords="data",
        box_alignment=(0.5, 0.0), frameon=True, pad=0.15,
        bboxprops=dict(edgecolor=border, linewidth=1.1, facecolor="white"),
        zorder=6,
    )
    ax.add_artist(ab)


# interstage: LEFT panel "(a) Normalized DSPSS" (drop title + bottom axis label)
_inter = mpimg.imread(os.path.join(_HERE, "interstage_3d.png"))
_inter_crop = _inter[60:490, 55:430]
# fairing: rightmost "view 3" nose cone (exclude colourbar at far right)
_fair = mpimg.imread(os.path.join(_HERE, "fairing_3d.png"))
_fair_crop = _fair[40:600, 1130:1555]

thumb_cx = struct_x + SW / 2
# Each thumbnail sits centred directly ABOVE its OWN structure box, with a
# small caption tucked in the gap between the box top and the thumbnail.
# A clear vertical strip separates each (box top → caption → thumbnail) and the
# spread-apart rows guarantee the Fairing thumbnail never reaches the Interstage
# box above it.
CAP_DY = 0.30        # caption sits this far above its box top
THUMB_DY = 0.62      # thumbnail bottom sits this far above its box top

ax.text(thumb_cx, sy_top + SH + CAP_DY, "DSPSS field",
        ha="center", va="center", fontsize=8, color=BLUE, style="italic")
ax.text(thumb_cx, sy_bot + SH + CAP_DY, "guided-wave field",
        ha="center", va="center", fontsize=8, color=AMBER, style="italic")

_add_thumb(_inter_crop, thumb_cx, sy_top + SH + THUMB_DY, zoom=0.17, border=BLUE)
_add_thumb(_fair_crop, thumb_cx, sy_bot + SH + THUMB_DY, zoom=0.15, border=AMBER)

plt.tight_layout(pad=0.4)

out_pdf = os.path.join(os.path.dirname(__file__), "framework_overview_3d.pdf")
out_png = os.path.join(os.path.dirname(__file__), "framework_overview_3d.png")
fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print("saved →", os.path.abspath(out_pdf))
print("saved →", os.path.abspath(out_png))
