#!/usr/bin/env python3
"""stage3_fatigue_link.py — 実データ疲労較正 → Stage-3 予後 の正式接続。

これまで Stage-3 のはく離成長 Paris 指数は代表値 m=3（illustrative）をハードコード
していた。本スクリプトは composite_fatigue_calibration.py が実 CFRP（4TU DCB モードI,
37試験体）から較正した**人口事後 (μ_m, σ_m)** を読み、フェアリング予後 P(grow) と
飛行可否を「代表 m=3」対「実較正 m 事後」で比較する＝life側を実破壊データに接続する。

正直な扱い: 単一 (C,m) は leave-one-specimen-out で外挿失敗（繊維ブリッジ）。よって
点推定 m でなく**人口事後を draw ごとにサンプル**して P(grow) に伝播する。Paris 係数 C
の絶対レートは依然 representative（較正したのは決定支配的な指数 m）。

出力: results/composite_fatigue_4tu/stage3_fatigue_link.{png,pdf,json}
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
                     "mathtext.fontset": "stix"})

import fairing_debond_prognosis as fdp

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "results", "composite_fatigue_4tu", "calibration.json")

def ensure_calibration():
    if not os.path.exists(ART):
        import composite_fatigue_calibration as cfc
        cfc.run_study(verbose=False)
    return json.load(open(ART))

def pgrow(cfg, a0, load, n=400, seed=0):
    rng = np.random.default_rng(seed)
    post = np.clip(a0 * (1 + 0.15 * rng.standard_normal(n)), 1e-3, None)
    return fdp.debond_growth_probability(post, load, cfg, n_draws=n, rng=rng)

def decide(p, ok=0.1, repair=0.5):
    return "OK" if p < ok else ("REPAIR" if p < repair else "RETIRE")

def main():
    cal = ensure_calibration()
    mu, sig = cal["mu_m"], cal["sigma_m"]
    LOAD = 0.10
    cases = [("small\n$a_0$=0.02", 0.02), ("medium\n$a_0$=0.06", 0.06),
             ("large\n$a_0$=0.11", 0.11)]
    configs = [("representative $m{=}3$", fdp.FairingConfig()),
               (f"calibrated $m{{=}}{mu:.0f}$ (point)",
                fdp.FairingConfig(paris_m=mu, paris_m_sigma=0.0)),
               (f"calibrated $m\\sim N({mu:.0f},{sig:.0f})$",
                fdp.FairingConfig.from_calibration(ART))]
    res = {}
    print("Stage-3 prognosis with REAL-data fatigue calibration")
    print("  4TU DCB mode-I: mu_m=%.1f  sigma_m=%.1f  | LOSO median R2=%.2f (frac>0=%.2f)"
          % (mu, sig, cal["loso_median_r2"], cal["loso_frac_pos"]))
    print("  P(grow) per case (load=%.2f):" % LOAD)
    grid = np.zeros((len(cases), len(configs)))
    for i, (cn, a0) in enumerate(cases):
        row = []
        for j, (lab, cfg) in enumerate(configs):
            p = pgrow(cfg, a0, LOAD)
            grid[i, j] = p; row.append("%s=%.2f(%s)" % (lab.split()[0], p, decide(p)))
        res[cn.replace("\n", " ")] = {configs[j][0]: float(grid[i, j]) for j in range(len(configs))}
        print("   %-14s %s" % (cn.replace("\n", " "), " | ".join(row)))

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    x = np.arange(len(cases)); w = 0.26
    cols = ["#9aa7b8", "#1565c0", "#0f8b8d"]
    for j, (lab, _) in enumerate(configs):
        ax.bar(x + (j - 1) * w, grid[:, j], w, label=lab, color=cols[j], edgecolor="k", lw=0.4)
    ax.axhline(0.1, ls="--", color="#2e7d32", lw=0.9); ax.text(2.35, 0.11, "OK<0.1", fontsize=7, color="#2e7d32")
    ax.axhline(0.5, ls="--", color="#c62828", lw=0.9); ax.text(2.3, 0.51, "RETIRE>0.5", fontsize=7, color="#c62828")
    ax.set_xticks(x); ax.set_xticklabels([c[0] for c in cases], fontsize=9)
    ax.set_ylabel("P(debond grows next flight)", fontsize=10); ax.set_ylim(0, 1.05)
    ax.set_title("Stage-3 fairing prognosis: representative vs REAL-data-calibrated Paris exponent\n"
                 "(4TU DCB mode-I, $m\\approx%.0f$ not 3; LOSO fails $\\Rightarrow$ propagate the population spread)"
                 % mu, fontsize=9)
    ax.legend(fontsize=8, loc="center left"); ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    d = os.path.dirname(ART)
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(d, "stage3_fatigue_link." + e), dpi=200, bbox_inches="tight")
    json.dump({"mu_m": mu, "sigma_m": sig, "loso_median_r2": cal["loso_median_r2"],
               "load": LOAD, "P_grow": res}, open(os.path.join(d, "stage3_fatigue_link.json"), "w"), indent=2)
    print("\n  INTERPRETATION: the calibrated steep exponent SHARPENS the safe/unsafe boundary —")
    print("  it suppresses sub-critical (medium) growth the representative m=3 over-calls, and")
    print("  drives super-critical (large) debonds to runaway. The life decision now consumes")
    print("  real fracture data (with bridging spread), not an illustrative exponent.")
    print("  saved results/composite_fatigue_4tu/stage3_fatigue_link.{png,pdf,json}")

if __name__ == "__main__":
    main()
