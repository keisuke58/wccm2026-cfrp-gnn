# メモ：PPS の結晶化の定数（文献調査）

> 作成：2026-11-02。対象は既報（津田ら 2026）の DURAFIDE PPS 1130A6（GF 30 wt%）。
> 下表は**文献の値をそのまま写したもの**であり、本銘柄・ガラス繊維入りでの値ではない（未校正）。

## 1. 見つかった値

| 定数 | 値 | 出典 | 備考 |
|---|---|---|---|
| 平衡融点 *T*<sub>m</sub><sup>0</sup> | 340 °C | Nohara et al. (2006) | 同論文が文献[10]から引用。自己核を消す加工温度としても用いている。López & Wilkes (1988) の Hoffman–Weeks 法では 304–312 °C（§1b） |
| ガラス転移温度 *T*<sub>g</sub> | 約 85 °C | Nohara et al. (2006) | |
| 融点（DSC） | 285 °C | Nohara et al. (2006) | |
| 完全結晶の融解熱 Δ*H*<sub>f</sub><sup>0</sup> | 76.5 J/g | Nohara et al. (2006)、Hay & Luck (2001) を引用 | 一次文献は未確認 |
| 最大結晶化度 | 約 50–56 % | Nohara et al. (2006) | 43.1/76.5 J/g から算出 |
| Avrami 指数 *n* | 2.5–3.0 | Nohara et al. (2006) | 等温 265–270 °C。PPS 単体で *n* = 4 とする報告もある（PPS/CNT の研究） |
| 半結晶化時間 *t*<sub>1/2</sub>（等温） | 4.1 / 12.6 / 26.8 min（265 / 268 / 270 °C） | Nohara et al. (2006) 表1 | 本文を確認。融解熱 43.1 / 41.0 / 14.5 J/g |
| 等温結晶化が最も速い温度 | 本文は 268 °C とするが、表1 では 265 °C が最短 | Nohara et al. (2006) | **論文内で食い違い**。表1 の数値を優先する |

## 1b. López & Wilkes (1988) から得た値（2026-11-03、本文を確認）

L. C. López, G. L. Wilkes, *Polymer* 29(1), 106–113, 1988（doi:10.1016/0032-3861(88)90207-8）。
線状 PPS、分子量 *M*<sub>w</sub> = 24 000、49 000、63 000 の 3 種。等温結晶化（DSC）と球晶成長速度（偏光顕微鏡）。

| 量 | 値 | 備考 |
|---|---|---|
| 平衡融点 *T*<sub>m</sub><sup>0</sup>（Hoffman–Weeks 法） | 304 / 308 / 312 °C | *M*<sub>w</sub> 24 000 / 49 000 / 63 000。Nohara et al. の 340 °C より約 30 °C 低い |
| Avrami 指数 *n*（*M*<sub>w</sub> 49 000、表2） | 2.6〜3.1（220〜255 °C） | 二重対数法と半結晶化時間法がよく一致。*M*<sub>w</sub> 24 000 では約 2.5 |
| 等温結晶化の温度範囲 | 220〜260 °C | 溶融から（320 °C で 4 min 保持後）。これより低温は DSC の冷却が間に合わない |
| 速度定数 *K*、半結晶化時間 *t*<sub>1/2</sub> | 図6 のみ（数値表なし） | 図からの読み取りが必要 |
| 成長速度の分子量依存 | 過冷却 70 °C で、*M*<sub>w</sub> 24 000 は 63 000 の約 4 倍速い | |

**Hoffman–Lauritzen の *K*<sub>g</sub>、*U*\* はこの論文にない。** 著者自身が「温度範囲が狭く、Hoffman らの領域理論による詳しい解析ができなかった」と書いている（p. 107）。
したがって *K*<sub>g</sub> は、図6 の *K*(*T*) を読み取って自分で当てはめるか、別の文献を探す必要がある。

## 1c. López, Wilkes & Geibel (1989) から得た値（2026-11-03、本文を確認）

L. C. López, G. L. Wilkes, J. F. Geibel, *Polymer* 30(1), 147–155, 1989（doi:10.1016/0032-3861(89)90395-9）。
分岐剤（トリクロロベンゼン）の量と、末端基の対イオン（H、Ca、Zn、Na）の影響。

| 量 | 値 | 備考 |
|---|---|---|
| 平衡融点 *T*<sub>m</sub><sup>0</sup>（Hoffman–Weeks 法） | 312 °C（線状）、304 °C（分岐剤 0.13 mol%）、301 °C（0.20 mol%） | 線状は Lovinger らの値より 3 °C 低いと記載 |
| Avrami 指数 *n* | ほぼ 3（末端 H のみ約 2.6） | |
| 全体の結晶化速度 *K* が最大になる温度 | 約 180 °C | ln *K*(*T*) を 3 次多項式で当てはめた結果。既報の値と一致と記載 |
| 分岐剤 0.20 mol% の影響 | 全体の結晶化速度 1/5、成長速度 1/2.2、核密度 1/3.5 | |
| 末端の対イオンの影響 | 全体の結晶化速度は H > Zn > Ca > Na（H は Na の 17 倍） | 同じ PPS でも銘柄・末端処理で 1 桁以上変わる |

**この論文にも *K*<sub>g</sub>、*U*\* はない。** ただし「速度が最大になる温度 約 180 °C」は、Hoffman–Lauritzen 式の形を決める手がかりになる（速度最大の温度は *K*<sub>g</sub>、*U*\*、*T*<sub>∞</sub> の組で決まる）。
また、銘柄や末端処理で速度が 1 桁以上変わるので、**文献値をそのまま 1130A6 に使うことはできない**。

## 2. 見つかっていない値

- Hoffman–Lauritzen の核形成定数 *K*<sub>g</sub> と活性化エネルギー *U*\*、および *T*<sub>∞</sub>。
  非等温の Nakamura 式で *K*(*T*) を与えるのに必要である。
- López & Wilkes (1988) にはないことを確認した（§1b）。
- 一部の文献は、PPS の等温結晶化では二次結晶化が起き、Avrami 式の当てはまりが悪いと報告している。

## 1d. Lovinger, Davis & Padden (1985)：Hoffman–Lauritzen の定数（2026-11-03、本文を確認）

A. J. Lovinger, D. D. Davis, F. J. Padden Jr., *Polymer* 26(11), 1595–1604, 1985（doi:10.1016/0032-3861(85)90270-8）。
球晶成長速度 *G* を 100〜280 °C（溶融から・急冷ガラスから）で測定し、二次核形成理論で解析。

    G = G0 exp[−U*/R(Tc − T∞)] exp[−Kg/(Tc ΔT f)],   ΔT = Tm0 − Tc

**表2（中分子量 PPS、*M*<sub>w</sub> ≈ 51 000、*T*<sub>m</sub><sup>0</sup> = 315 °C、*T*<sub>g</sub> = 92 °C）**

| 輸送項の選び方 | *U*\* | 領域 | 表の「*K*<sub>g</sub>」[10<sup>5</sup> K<sup>2</sup>] | *G*<sub>0</sub> [cm/s] | 相関係数 |
|---|---|---|---|---|---|
| *T*<sub>∞</sub> = *T*<sub>g</sub> − 30 K（**著者の推奨**） | 1400 cal/mol（5.86 kJ/mol） | II | 8.84 | 5.43 | 0.9994 |
| 同上 | 同上 | III | 18.35 | 3.86×10<sup>4</sup> | 0.9968 |
| *T*<sub>∞</sub> = *T*<sub>g</sub> − 51.6 K（WLF 型） | 2650 cal/mol | II | 9.39 | 1.61×10<sup>2</sup> | 0.9996 |
| 同上 | 同上 | III | 28.42 | 4.67×10<sup>9</sup> | 0.9976 |

- **注意（2026-11-03 に確認）**：表の「*K*<sub>g</sub>」の行は「傾き」の行の 2.303 倍で、log<sub>10</sub> 形の式の値とみられる。上の自然指数の式に表の「*K*<sub>g</sub>」を入れると、領域 II と III の式が転移温度 208 °C で約 10<sup>5</sup> 倍食い違う。「傾き」の行（II 3.841×10<sup>5</sup>、III 7.970×10<sup>5</sup> K<sup>2</sup>）を入れると 4 % 以内で一致し、本文の *G*<sub>0</sub> の比 7.1×10<sup>3</sup> とも合う。**計算には「傾き」の行を使う**（`validation/pps_crystallization_kinetics.py`）。これは本研究の読み方で、論文に明記はない。
- 領域 II → III の転移は 208 °C（推奨の組）。III/II の傾き比は 2.07（理論値 2）。
- 著者は WLF の値（*U*\* = 4120 cal/mol、*T*<sub>∞</sub> = *T*<sub>g</sub> − 51.6 K）では解析が成り立たないとし、*U*\* = 1400 cal/mol、*T*<sub>∞</sub> = *T*<sub>g</sub> − 30 K を採用した。
- 完全結晶の融解熱は約 80 J/g（19 cal/g、Brady の値からの外挿）。
- 低分子量 PPS（*M*<sub>w</sub> ≈ 15 000）は、どの定数でも一つの式に乗らなかった。

**検証（`validation/pps_crystallization_kinetics.py`）**：倍率 1 個を Nohara らの半結晶化時間 1 点（265 °C で 4.1 min）で合わせ、当てはめに使わないデータと比べた。
- 領域 II と III の式の、208 °C での一致：比 0.96。
- 速度が最大になる温度：176 °C（López et al. 1989 の約 180 °C）。
- Nohara らの残り 2 点の半結晶化時間：268 °C で 8.8 min（実測 12.6 min）、270 °C で 15.7 min（実測 26.8 min）。融点近くでは銘柄の違い（*T*<sub>m</sub><sup>0</sup>）に敏感で、1.4〜1.7 倍のずれは想定の範囲。

**本研究での使い方（案）**
- Nakamura 式の *K*(*T*) の温度依存に、この *U*\*、*T*<sub>∞</sub>、*K*<sub>g</sub> を使う。208 °C を境に領域 II（3.841×10<sup>5</sup> K<sup>2</sup>）と III（7.970×10<sup>5</sup> K<sup>2</sup>）を切り替える。
- ただし、これは**球晶の成長速度**の定数であり、全体の結晶化速度（核形成を含む）の温度依存とは同じではない。また、純 PPS（Phillips の試料）の値で、1130A6（GF 30 %）の値ではない。前指数因子は DSC で合わせる必要がある。

## 2b. *K*<sub>g</sub> が載っている可能性の高い文献（2026-11-03 調査、本文未入手）

1. （本文確認済み、§1d）A. J. Lovinger, D. D. Davis, F. J. Padden Jr., "Kinetic analysis of the crystallization of poly(p-phenylene sulphide)", *Polymer* 26(11), 1595–1604, 1985.
   球晶成長速度を 100〜280 °C の広い範囲で測り、二次核形成理論で解析。要旨によれば領域 I→II 転移 488 K、II→III 転移 468 K。**領域ごとの *K*<sub>g</sub> が本文にあるはず**（要旨には数値なし）。
   https://www.sciencedirect.com/science/article/abs/pii/0032386185902708
2. J. M. Kenny, A. Maffezzoli, "Crystallization kinetics of poly(phenylene sulfide) (PPS) and PPS/carbon fiber composites", *Polymer Engineering & Science* 31(8), 607–614, 1991（頁は未確認）。
   **繊維入りの結晶化速度式**の可能性が高い。
   https://4spepublications.onlinelibrary.wiley.com/doi/abs/10.1002/pen.760310812
3. Mehl, Rebenfeld, *J. Appl. Polym. Sci.* 57(2), 1995。PPS 複合材の結晶化と繊維表面の核密度（未確認）。
   https://onlinelibrary.wiley.com/doi/abs/10.1002/app.1995.070570207

## 3. 注意点

- ガラス繊維は核剤として働き、結晶化を速めうる。樹脂単体の値をそのまま使うことはできない可能性がある。
- 銘柄（分子量・分岐・末端基）によって結晶化速度は変わる（López & Wilkes の一連の報告）。

## 4. 今後

1. López & Wilkes (1988) 図6 の *K*(*T*) を読み取り、*K*<sub>g</sub> を当てはめる（溶融側 220〜260 °C のみで、外挿の幅は大きい）。
2. 最も確実なのは、本銘柄の DSC データ（等温・非等温）をデンソー様またはポリプラスチックス様から得ることである
   （`research/QUESTIONS_denso.md` §1 に記載済み）。

## 文献

- L. B. Nohara et al., "Study of crystallization behavior of poly(phenylene sulfide)", *Polímeros* 16(2), 104–110, 2006. doi:10.1590/S0104-14282006000200009（本文確認済み）
- L. C. López, G. L. Wilkes, "Crystallization kinetics of poly(p-phenylene sulphide): effect of molecular weight", *Polymer* 29(1), 106–113, 1988.（本文確認済み）
- L. C. López, G. L. Wilkes, J. F. Geibel, "Crystallization kinetics of poly(p-phenylene sulphide): the effect of branching agent content and endgroup counter-atom", *Polymer* 30(1), 147–155, 1989.（本文確認済み）

## PDF の扱い

PDF はリポジトリに入れていない（公開リポジトリで、論文には著作権がある）。入手方法：
- Nohara et al. (2006)：無料。`bash research/papers/fetch_papers.sh` で取得できる。
- López & Wilkes (1988)、López et al. (1989)、Lovinger et al. (1985)：Elsevier（有料）。大学のネットワークから ScienceDirect で取得し、`research/papers/` に置く（git の対象外）。
