# 文献調査（軽）：CFRP 硬化誘起 残留応力（ダイキン/NEDO テーマ）

> `MEMO_cfrp_residual_stress_collab.md` の物理背景の軽い一次整理。種デモ
> [`cfrp_cure_residual_stress.py`](../cfrp_cure_residual_stress.py) の根拠。原著要確認。

## 1. 物理（合意されている枠組み）
- 硬化誘起残留応力＝**熱化学（cure kinetics/DoC）× 熱機械（熱膨張・硬化収縮・剛性発展）の連成**。
- **化学収縮（cure shrinkage）は樹脂で数％オーダー、炭素繊維は硬化温度で不活性**（収縮ほぼ無視）。
  → 残留応力は「樹脂の収縮＋冷却」を繊維が拘束して生じる（異方 CTE・角度違いプライ間の拘束）。
- 工程段階：**型内硬化 → 冷却 → 後硬化 → 最終冷却**。各段で応力が蓄積し、脱型後に**反り（cure deformation）**。

## 2. 構成則（軽→重）
- **CHILE**（cure-hardening instantaneously linear elastic）：硬化度に応じて弾性率が発展する簡便則。
- **粘弾性**：応力緩和を含む上位モデル（残留応力を CHILE より正確に）。
- 本デモは**さらに軽い CLT（古典積層理論）＋硬化収縮を等価固有ひずみ**で扱う一次近似
  （DoC 発展・緩和・工具拘束は次段）。

## 3. 我々の立ち位置（物理主役・ML 従属）
- 物理コア＝**硬化熱化学×（粘弾性）熱機械の連成FE**。本リポの **⑥⑦ 熱弾性（熱ひずみ αΔT・CTE ミスマッチ→応力）**を
  **硬化収縮＋剛性発展**へ拡張すれば地続き。
- **プロセス → 残留応力** のサロゲート（GNN/DeepONet, リポ CFRP-GNN と同一パイプライン）は**加速/予測の従属層**
  ＝㉑ と同枠。精度の権威は FE/実測。
- 正直な限界：線形 CLT は**非対称薄板の反りを過大評価**（幾何非線形＝Hyer で円筒形に）。残留応力の大きさ（数十 MPa）が本質。

## 4. 主要文献（要 原著確認）
- Cure-induced residual stress & distortion（確率的 熱化学＋粘弾性, 実験検証）:
  Mech. Adv. Mater. Struct. 29(19) — https://www.tandfonline.com/doi/full/10.1080/15376494.2021.1877376
- マルチスケール 硬化残留応力（マクロ熱化学×ミクロ RVE）:
  Front. Mech. Eng. — https://link.springer.com/article/10.1007/s11465-020-0590-6
- 構成則比較（硬化残留応力・変形の数値解析, CHILE/粘弾性）:
  PMC6416736 — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6416736/
- 硬化プロファイル → 残留応力分布・破壊靱性:
  J. Compos. Sci. 10(4) 206 — https://doi.org/10.3390/jcs10040206
- 硬化パラメータ → 残留応力・機械特性:
  Crystals 16(7) 446 — https://doi.org/10.3390/cryst16070446
- ミクロ 残留応力評価（fiber push-out ＋ FE）:
  PMC10305614 — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10305614/
