# 文献メモ：先行研究と本研究の位置づけ（2026-11-04）

> Web 検索の要旨・抄録による整理。**本文は未確認**（★印は本文確認済み）。書誌は投稿前に一次資料で確かめること。

## 1. 射出成形品の反り・品質を機械学習で予測した研究

| 文献 | 内容（抄録による） | 本研究との違い |
|---|---|---|
| Composites Part A, 2026（S1359835X26001004）"Warpage prediction for fiber reinforced injection molding via geometric feature learning and differentiable FEM" | NN で繊維配向と初期ひずみを予測し、微分可能な FE で反りを損失に学習。反りの相対誤差 1 % 未満、従来計算の約 6 倍速 | **最も近い先行研究**。学習も評価も解析データ。実測による補正は扱っていない（抄録の範囲） |
| Applied Sciences 13, 12868, 2023（doi:10.3390/app132312868）"Prediction of Short-Shot Defects in Injection Molding by Transfer Learning" | CAE データで事前学習し、少数の実験で微調整。ショートショートの予測は向上、**反りの予測は弱いまま** | 転移学習の枠組みは同じ。対象がショートショートで、残留ひずみは扱わない |
| Int. J. Adv. Manuf. Technol., 2026（s00170-026-18480-0） | PA6/GF45 の解析で物理情報付き NN を学習し、3D スキャンの反りと比較 | 圧縮成形。実測は検証のみで補正に使わない |
| Springer 会議録, 2026（978-3-032-19524-1_42） | 成形条件 → 変位場の深層学習サロゲート（モデル縮約＋ML）、簡略化した自動車部品 | サロゲートのみ |
| Int. J. Adv. Manuf. Technol., 2016（s00170-016-9409-3） | Moldflow のデータで BP-NN と GA により短繊維複合材の反りを最小化 | 古典的な最適化。実測なし |

## 2. 穿孔法と繊維強化樹脂

| 文献 | 内容（抄録による） | 本研究との関係 |
|---|---|---|
| Shokrieh & Ghasemi, J. Compos. Mater., 2007（doi:10.1177/0021998306063803） | 直交異方性板・積層板の穿孔法の校正係数を FE で求める | 異方性材では FE による校正が要る、という本研究の前提 |
| J. Compos. Sci. 6, 138, 2022（doi:10.3390/jcs6050138） | CFRP の穿孔法。積層・板厚・欠陥を入れた FE 校正。気孔で応力が最大 46 % 変わる | 校正の不確かさの例 |
| Sensors 等, 2021（PMC8618666） | 増分穿孔法の校正。ASTM E837 の係数と FE による数値校正 | 規格の位置づけ |
| 射出成形品の残留応力（ResearchGate 261567088） | 穿孔法と数値解析がよく一致 | 射出成形品への適用例 |
| **空白**：短繊維強化の射出成形品に、配向を入れた FE 校正で穿孔法を当てた研究は見つからなかった | | **本研究（既報の模擬を FE で再現し補正に使う）は、この空白にあたる** |

## 3. 結晶性樹脂の残留応力の予測と検証

| 文献 | 内容 |
|---|---|
| Polym. Eng. Sci., 2002 | PS・HDPE の射出成形平板。熱弾性と熱粘弾性を比較し、層除去法で検証 |
| 2022（doi:10.1007/s00289-022-04276-1） | PP 平板。結晶化速度式を連成すると、反りの予測誤差が減る |
| Jansen（TU Delft） | エキシマレーザーによる層除去で残留応力分布を検証 |
| Residual Stresses in Short-Fibre Reinforced Injection-Moulded Thermoplastic Parts（Springer, 10.1023/A:1009814126576） | 層ごとの直交異方性・粘弾性モデル、層除去法で検証 |

## 4. 本研究の新しさ（案）

1. **実測で解析を補正し、その効果を当てはめに使わない点で評価した**（§1 の先行研究は、実測を検証にしか使わないか、反りでは効果が出ていない）。
2. **穿孔法の解放ひずみを、配向を入れた 3 次元 FE で再現し、補正に直接使った**（§2 の空白）。成分ごとの線形の重ね合わせ、影響関数で任意の分布に拡張できる。
3. **MD のみの計測で TD を予測できる条件**（共通の倍率＝実効無応力温度）を示した。
4. 弱点：試験片 1 枚・図の読み取り値。実データでの検証が要る（デンソー様のデータ）。

## 出典リンク

- https://www.sciencedirect.com/science/article/pii/S1359835X26001004
- https://doi.org/10.3390/app132312868
- https://link.springer.com/article/10.1007/s00170-026-18480-0
- https://link.springer.com/chapter/10.1007/978-3-032-19524-1_42
- https://link.springer.com/article/10.1007/s00170-016-9409-3
- https://doi.org/10.1177/0021998306063803
- https://doi.org/10.3390/jcs6050138
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8618666/
- https://www.researchgate.net/publication/261567088_Residual_stresses_in_injection_molded_products
- https://link.springer.com/article/10.1023/A:1009814126576
- https://springerlink.bibliotecabuap.elogim.com/content/pdf/10.1007/s00289-022-04276-1.pdf
