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
- López & Wilkes (1988)、López et al. (1989)：Elsevier（有料）。大学のネットワークから ScienceDirect で取得し、`research/papers/` に置く（git の対象外）。
