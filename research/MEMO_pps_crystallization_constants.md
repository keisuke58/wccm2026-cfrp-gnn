# メモ：PPS の結晶化の定数（文献調査）

> 作成：2026-11-02。対象は既報（津田ら 2026）の DURAFIDE PPS 1130A6（GF 30 wt%）。
> 下表は**文献の値をそのまま写したもの**であり、本銘柄・ガラス繊維入りでの値ではない（未校正）。

## 1. 見つかった値

| 定数 | 値 | 出典 | 備考 |
|---|---|---|---|
| 平衡融点 *T*<sub>m</sub><sup>0</sup> | 340 °C | Nohara et al. (2006) | 同論文内で別文献から引用 |
| ガラス転移温度 *T*<sub>g</sub> | 約 85 °C | Nohara et al. (2006) | |
| 融点（DSC） | 285 °C | Nohara et al. (2006) | |
| 完全結晶の融解熱 Δ*H*<sub>f</sub><sup>0</sup> | 76.5 J/g | Nohara et al. (2006)、Hay & Luck (2001) を引用 | 一次文献は未確認 |
| 最大結晶化度 | 約 50–56 % | Nohara et al. (2006) | 43.1/76.5 J/g から算出 |
| Avrami 指数 *n* | 2.5–3.0 | Nohara et al. (2006) | 等温 265–270 °C。PPS 単体で *n* = 4 とする報告もある（PPS/CNT の研究） |
| 等温結晶化が最も速い温度 | 268 °C | Nohara et al. (2006) | 試験範囲 265–270 °C 内での値 |

## 2. 見つかっていない値

- Hoffman–Lauritzen の核形成定数 *K*<sub>g</sub> と活性化エネルギー *U*\*、および *T*<sub>∞</sub>。
  非等温の Nakamura 式で *K*(*T*) を与えるのに必要である。
- 候補の一次文献：López & Wilkes, *Polymer* 29 (1988)（球晶成長速度の温度・分子量依存）。
  有料のため本文は未確認。
- 一部の文献は、PPS の等温結晶化では二次結晶化が起き、Avrami 式の当てはまりが悪いと報告している。

## 3. 注意点

- ガラス繊維は核剤として働き、結晶化を速めうる。樹脂単体の値をそのまま使うことはできない可能性がある。
- 銘柄（分子量・分岐・末端基）によって結晶化速度は変わる（López & Wilkes の一連の報告）。

## 4. 今後

1. 図書館経由で López & Wilkes (1988) を入手し、*K*<sub>g</sub> を確認する。
2. 最も確実なのは、本銘柄の DSC データ（等温・非等温）をデンソー様またはポリプラスチックス様から得ることである
   （`research/QUESTIONS_denso.md` §1 に記載済み）。

## 文献

- L. B. Nohara et al., "Study of crystallization behavior of poly(phenylene sulfide)", *Polímeros* 16(2), 2006. doi:10.1590/S0104-14282006000200009
- L. C. López, G. L. Wilkes, "Crystallization kinetics of poly(p-phenylene sulphide): effect of molecular weight", *Polymer* 29(1), 1988.（未読）
- L. C. López, G. L. Wilkes, J. F. Geibel, *Polymer* 30(1), 147–155, 1989.（未読）
