"""build_advisor_onepager.py -- one-page (16:9) advisor summary for the CFRTP work.

Composes a single sheet: current status (verified / in-progress / data-ask) plus the
two key result panels (PEEK method validation, B-5 inverse design). Japanese via
IPAGothic; pure matplotlib. Writes slides/cfrtp_advisor_onepager.png (+ .pdf).

    python3 slides/build_advisor_onepager.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import font_manager as fm

# Japanese-capable font
for name in ("IPAGothic", "IPAPGothic", "Noto Sans CJK JP", "VL Gothic"):
    if any(f.name == name for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False

NAVY, TEAL, RED, INK, MUTED = "#1f3864", "#12b3a0", "#e23b48", "#141922", "#5a6675"
here = os.path.dirname(os.path.abspath(__file__))
val_png = os.path.join(here, "..", "validation", "peek_crystallization_validation.png")
inv_png = os.path.join(here, "..", "design", "cfrtp_inverse_design.png")

fig = plt.figure(figsize=(13.33, 7.5), dpi=150)
fig.patch.set_facecolor("white")

# --- header ---
fig.text(0.035, 0.955, "CFRTP（ダイキン／NEDO）残留応力・界面・剥離 ― 現状サマリ",
         fontsize=21, fontweight="bold", color=NAVY, va="top")
fig.text(0.035, 0.905,
         "物理主役・ML従属：弱形式FE／Abaqus が精度の権威、ML はサロゲート・逆設計の加速層"
         "（数値はすべて例示・未校正）",
         fontsize=11.5, color=MUTED, va="top")
fig.add_artist(plt.Line2D([0.035, 0.965], [0.878, 0.878], color=NAVY, lw=2.2))

# --- three status columns ---
cols = [
    (0.035, "■ 検証済み（Abaqus 2024 実走）", TEAL, [
        "CHILE ／ 粘弾性(Prony＋WLF) ／ 結晶化",
        "  (Nakamura・Hoffman–Lauritzen) UMAT",
        "[0/90] 残留応力、混合モード剥離 2D/3D",
        "  (Benzeggagh–Kenane cohesive)",
        "手法検証：炭素/PEEK で結晶化ピーク",
        "  Tp ≈ 305℃ を文献値と一致 (HL)",
    ]),
    (0.365, "■ 実装・進行中", NAVY, [
        "逆設計 B-5：冷却速度で残留応力 −46%、",
        "  結晶化度 α≧0.6 を保つ最適 r*=445℃/min",
        "  (サロゲート探索 → FE 検証)",
        "可視化：インタラクティブ(GitHub Pages)",
        "  ＋ アニメGIF（スライド/論文用）",
        "ダイキン系(PFA)：接続済み（定数は仮）",
    ]),
    (0.695, "■ 先生・ダイキンへの依頼（データ）", RED, [
        "DSC 非等温（冷却速度→Tp, α(T)）",
        "DMA（緩和スペクトル）／ CTE(T)",
        "Tg/Tm/Tm0、ILSS、G_Ic/G_IIc",
        "残留応力・反りの実測",
        "→「例示」から「定量・検証済み」へ",
        "  ＝ 修論／WCCM の主結果に",
    ]),
]
for x, head, col, lines in cols:
    fig.text(x, 0.845, head, fontsize=13, fontweight="bold", color=col, va="top")
    y = 0.795
    for ln in lines:
        fig.text(x, y, ln, fontsize=10.5, color=INK, va="top")
        y -= 0.037

# --- two result panels ---
def panel(rect, png, caption_x, caption):
    ax = fig.add_axes(rect)
    try:
        ax.imshow(mpimg.imread(png))
    except Exception:
        ax.text(0.5, 0.5, "(figure missing)", ha="center", va="center")
    ax.axis("off")
    fig.text(caption_x, 0.075, caption, ha="center", va="center", fontsize=10, color=MUTED)

fig.text(0.035, 0.545, "検証と設計（クリックできる版は下記 Live）",
         fontsize=12, fontweight="bold", color=NAVY, va="top")
panel([0.035, 0.115, 0.455, 0.40], val_png, 0.2625,
      "手法検証：Hoffman–Lauritzen で Tp が文献バンド内（bell は外れ）")
panel([0.520, 0.115, 0.455, 0.40], inv_png, 0.7475,
      "逆設計 B-5：残留応力↓ × 結晶化度 制約 → 最適冷却速度")

# --- footer ---
fig.add_artist(plt.Line2D([0.035, 0.965], [0.045, 0.045], color="#d6dde5", lw=1))
fig.text(0.035, 0.025,
         "Live: keisuke58.github.io/wccm2026-cfrp-gnn   ・   physics-first / ML-subordinate"
         "   ・   Keio Muramatsu-lab",
         fontsize=9.5, color=MUTED, va="center")

out = os.path.join(here, "cfrtp_advisor_onepager.png")
fig.savefig(out, dpi=150, facecolor="white")
fig.savefig(out.replace(".png", ".pdf"), facecolor="white")
plt.close(fig)
print("wrote", out, "(+ .pdf)")
