"""Composites-B figure 1: the Stage 0-3 SHM pipeline as one schematic, with the
key result number under each stage. Thesis style (lmodern usetex)."""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "slides", "figure_sources"))
from thesis_style import use
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=use(width_frac=1.0, aspect=0.40))
ax.set_xlim(0, 10); ax.set_ylim(0, 3.0); ax.axis("off")

blue   = "#1565C0"
green  = "#1B5E20"
stages = [
    ("Stage 0", "Screening",      r"Mahalanobis", r"AUROC 0.99"),
    ("Stage 1", "Classification", r"GNN",          r"F1 0.69"),
    ("Stage 2", "Characterisation", r"FMPE posterior", r"cov.\ 0.94"),
    ("Stage 3", "Prognosis",      r"phase-field+fatigue", r"P(survive $n$)"),
]
w, h, x0, gap = 2.05, 1.5, 0.25, 0.30
y0 = 1.0
centers = []
for i, (s, role, method, res) in enumerate(stages):
    x = x0 + i * (w + gap)
    box = FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
                         linewidth=1.1, edgecolor=blue, facecolor="#E3F0FB")
    ax.add_patch(box)
    cx = x + w / 2
    centers.append((x, cx, x + w))
    ax.text(cx, y0 + h - 0.27, r"\textbf{%s}" % s, ha="center", va="center",
            color=blue, fontsize=9)
    ax.text(cx, y0 + h - 0.60, role, ha="center", va="center", fontsize=8)
    ax.text(cx, y0 + 0.52, method, ha="center", va="center", fontsize=8,
            style="italic")
    ax.text(cx, y0 + 0.18, r"\textbf{%s}" % res, ha="center", va="center",
            color=green, fontsize=8)

# arrows + data contract labels
labels = [r"$|x_i-\mu_i|/\sigma_i$", r"defect class", r"$\theta=(c_x,c_y,\ell,s)$"]
for i in range(3):
    xa = centers[i][2]; xb = centers[i + 1][0]
    ax.add_patch(FancyArrowPatch((xa, y0 + h / 2), (xb, y0 + h / 2),
                 arrowstyle="-|>", mutation_scale=12, lw=1.2, color="#444"))
    ax.text((xa + xb) / 2, y0 + h / 2 + 0.18, labels[i], ha="center",
            va="bottom", fontsize=6.5, color="#444")

# top question banner
qs = ["where?", "what?", "how bad?", "fly again?"]
for (x, cx, xe), q in zip(centers, qs):
    ax.text(cx, y0 + h + 0.22, q, ha="center", va="center", fontsize=8.5,
            style="italic", color="#666")

# bottom decision output
ax.text(centers[-1][1], y0 - 0.32, r"$\Rightarrow$ OK / REPAIR / RETIRE",
        ha="center", va="center", fontsize=8.5, color=green)

plt.tight_layout()
out = "paper_figs/pipeline_overview.pdf"
plt.savefig(out, bbox_inches="tight")
plt.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
print("saved →", out)
