"""Publication figure: anomaly-detection zoo on fixed-geometry DSPSS fields.
Left: clean AUROC by method (2x2/4x4). Right: noise robustness (4x4).
Data: benchmark runs 2026-06-11 (anomaly_zoo.py, stfpm/gae/diffusion logs)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

methods = ["Mahalanobis", "PCA resid.", "PaDiM", "PatchCore", "STFPM",
           "Flow", "GAE", "Diffusion FM", "Harmonic (0-shot)"]
clean_2x2 = [0.9993, 0.9993, 0.9639, 0.9267, 0.7536, 0.5778, 0.6628, 0.5030, 0.4649]
clean_4x4 = [0.9949, 0.9949, 0.9435, 0.9176, 0.4958, 0.7067, 0.6998, 0.6545, 0.5903]

noise = [0.0, 0.1, 0.3, 0.5]
n4 = {  # 4x4 AUROC vs noise
    "Mahalanobis": [0.9949, 0.6614, 0.4737, 0.4364],
    "PaDiM":       [0.9435, 0.6395, 0.5363, 0.5171],
    "STFPM":       [0.4958, 0.5035, 0.5079, 0.5276],
    "Flow":        [0.7067, 0.5893, 0.5084, 0.4949],
    "GAE":         [0.6998, 0.5501, 0.5240, 0.5074],
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))

x = np.arange(len(methods))
ax1.barh(x + 0.2, clean_2x2, height=0.38, label="2x2 defect", color="#1f77b4")
ax1.barh(x - 0.2, clean_4x4, height=0.38, label="4x4 defect", color="#ff7f0e")
ax1.set_yticks(x); ax1.set_yticklabels(methods, fontsize=9)
ax1.axvline(0.5, color="gray", ls="--", lw=0.8)
ax1.set_xlabel("node-level AUROC (clean)")
ax1.set_xlim(0.3, 1.02); ax1.invert_yaxis()
ax1.legend(loc="lower right", fontsize=9)
ax1.set_title("(a) Simple node-space statistics beat deep methods", fontsize=10)

for name, v in n4.items():
    lw = 2.2 if name == "Mahalanobis" else 1.3
    ax2.plot(noise, v, "o-", label=name, lw=lw)
ax2.axhline(0.5, color="gray", ls="--", lw=0.8)
ax2.set_xlabel(r"additive noise $\sigma$ (units of field std)")
ax2.set_ylabel("AUROC (4x4)")
ax2.set_title("(b) All methods collapse by $\\sigma$=0.1:\ndefect amplitude < 0.1$\\sigma$ — per-measurement SNR limit", fontsize=10)
ax2.legend(fontsize=8)

plt.tight_layout()
out = "paper_figs/anomaly_zoo_benchmark.pdf"
plt.savefig(out, bbox_inches="tight")
plt.savefig(out.replace(".pdf", ".png"), dpi=160, bbox_inches="tight")
print("saved →", out)
