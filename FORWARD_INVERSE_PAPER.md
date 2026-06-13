# A Differentiable Graph-Network Surrogate for Phase-Field Fracture: Fast Forward Emulation and Gradient-Based Defect Inversion

作成 2026-06-13。順問題(⑥)＋逆問題(①)＋UQ(③)を1本に束ねた研究本命ドラフト。
**狙い**: 慶應修論 Phase-2 の主軸／USNCCM19 発表＋Composites B or CMAME 投稿。
**一筆書き**: 「phase-field 破壊は順問題が高価・逆問題はその多数回呼び出しが必要。**1つの微分可能GNN代理**が順・逆の両方を担う」。

---

## 0. 貢献（claims）

1. **Forward**: AT2 phase-field（Miehe staggered, MMS 2次精度検証済）の順写像 *seed欠陥場→破壊後き裂場* を、メッシュ非依存のメッセージパッシングGNNで代理。held-out **node-R²=0.94 / crack-IoU=0.89**（2D）、**0.90 / 0.62**（3D剥離前縁）。
2. **Mesh-agnostic は無償**: 同一問題で構造化グリッドCNN U-Net（R²0.94/IoU0.86）と**同等**＝メッシュ非依存の精度代償ほぼゼロ。3D不規則メッシュではU-Net不適＝GNNが唯一の選択。
3. **Inverse**: 凍結した微分可能エミュレータ越しに、観測き裂場から欠陥(cx,cy)を**勾配降下で復元**。真値はAT2実ソルバ。位置 **MAE≈0.05**（領域[0,1]）。**ノイズ頑健**（σ=0→0.2で誤差0.064→0.066とほぼ平坦）＋**多初期値スプレッド＝粗い逆推定UQプロキシ**（σで約3.4倍増だが非単調・較正済み事後ではない）。
4. **UQ は分散でなく conformal**: forward代理のMC-dropout(corr0.07)・deep ensemble(0.08)は**較正失敗**（誤差が系統的バイアス→分散UQ盲目）。**split-conformal残差較正が近似有効被覆**（0.76/0.87/0.91 @ 0.80/0.90/0.95）。
5. **正直な限界マップ**（§5）を主役級に：正規化2D・スナップ破壊・3D大前縁過小・逆問題の欠陥サイズ非識別。

## 1. Introduction

- phase-field破壊（AT2/Miehe）はき裂発生＋任意経路＋分岐を変分で統一＝研究最前線（CZM/VCCTは既存き裂/界面が要る）。代償は**順問題の計算コスト**（細メッシュ・staggered反復）。
- 欠陥同定（逆問題）は順問題を**多数回**呼ぶ＝二重に高価。従来はFMPE等のsampling-based逆問題（`fmpe_defect.py`）。
- 提案: **1つの微分可能なGNN代理**で、(a)順問題を高速化し、(b)その微分可能性で逆問題を勾配ベース化。順・逆を同じ代理で。

## 2. Forward emulator（順問題代理）

- データ: AT2 forward（`cfrp_phasefield_2d.simulate_growth`）を多欠陥(位置/サイズ/層)で実行→(seed場, 破壊後場)対。
- グラフ: 格子節点＋4近傍辺、節点特徴[seed損傷, 局所Gc, x, y, **seed重心ブロードキャスト(sx,sy)**]。3要素メッセージパッシング(自己/近傍平均/**グローバル文脈**)＋BCE＋8ラウンド。
- 結果: held-out **node-R²0.94/IoU0.89**（`phasefield_gnn_prognosis.py`）。U-Net比較で同等（`phasefield_emulator_compare.py`）。3D拡張（`phasefield_gnn_3d.py`）node-R²0.90。
- 図: `phasefield_gnn_prognosis.png`, `phasefield_emulator_compare.png`, `phasefield_gnn_3d.png`。

## 3. Inverse problem（逆問題）

- 微分可能ソフトseed s(θ), θ=(cx,cy,log r)。凍結エミュレータ E に通し、観測場 y との誤差 ‖E(s(θ))−y‖² を Adam で最小化→θ̂。
- 結果: 4欠陥で位置 **|Δcx|≈0.05, |Δcy(貫通方向)|≈0.04**（`inverse_defect_gnn.py`）。
- **頑健性＋UQ**（`inverse_rigor.py`, frontale決定的実行 threads=1）: 観測ノイズσ∈{0,0.05,0.10,0.20}で位置誤差は **0.064→0.066**（範囲0.057–0.066）＝**ほぼ平坦＝強いノイズ頑健性**（σ4倍でも誤差は実質不変）。多初期値スプレッド(逆UQ)は **0.024→0.083**（σ=0→0.2で約3.4倍）＝**ノイズに正しい向きで増加するが非単調**で位置誤差と同オーダー＝較正された事後ではなく**粗い不確かさプロキシ**（正直な限界）。注: 点推定はBLASスレッド数でrun-to-run変動するため、決定的な threads=1 ランを正典とし図と一致させた。
- 図: `inverse_defect_gnn.png`, `inverse_rigor.png`。
- FMPEとの差: 勾配ベースで高速・点推定＋スプレッドUQ。FMPEは完全事後だが高価。代理化で両者を橋渡し。

## 4. Uncertainty quantification

- Forward代理UQ: **分散UQ(dropout/ensemble)は較正失敗**（corr0.07-0.08, 局在なし）＝誤差が系統的バイアス。**split-conformal残差**が近似有効被覆（`conformal_emulator_uq.py`, 0.76/0.87/0.91）。
- Inverse UQ: 多初期値復元の散らばり＝安価な事後プロキシ（§3）。ノイズ範囲で約3.4倍増(0.024→0.083)＝向きは正しいが**σに単調追従しない**・誤差と同オーダー＝較正された被覆は主張しない（forward側 conformal に比べ弱い保証）。
- ⑤クリアランスconformal（`conformal_transfer.py`）と一本化＝end-to-endでUQが通る。

## 5. 限界（honest、主役級）

- 正規化2D・**スナップ破壊**（疲労の漸進Paris領域なし→残サイクル予後は基盤限界、`simulate_fatigue_flights`で確認）。
- 3D: 大きい剥離前縁を**過小予測**（vol-IoU0.62）。界面エッジ条件付MPは現データ規模で優位未実証（null, `phasefield_gnn_3d_edgecond.py`）。
- 逆問題: **欠陥サイズ/Gcは単一場から弱識別**（位置は強識別）。複数荷重/モダリティが要る。
- 物性・荷重は正規化。実材料(T700SC/LY556)・実荷重較正は将来。

## 6. Keio Phase-2 への接続

- [[project_keio_bridge]] 村松研（計算固体力学）。phase-field×微分可能GNN×逆問題＝計算固体力学とMLの結節。
- Phase-2拡張軸: 形態先鋭化(界面エッジ条件)・3D・実Gc・疲労Paris基盤・UQ伝播・逆設計（`PHASEFIELD_GNN_PLAN.md`）。
- 半導体ダイシングへの転用: 同じ手法（異方性エッジMP＋微分可能逆問題）がSiC/GaNへき開のチッピング制御に化ける（[[project_wafer_proc_sim]] `semiconductor_cleavage_anisotropy.py`）＝研究と DISCO/起業の共通核。

## 7. 図・コード一覧

| 役割 | 図 | コード |
|---|---|---|
| 順問題2D | phasefield_gnn_prognosis.png | phasefield_gnn_prognosis.py |
| GNN vs U-Net | phasefield_emulator_compare.png | phasefield_emulator_compare.py |
| 順問題3D | phasefield_gnn_3d.png | phasefield_gnn_3d.py |
| 逆問題 | inverse_defect_gnn.png | inverse_defect_gnn.py |
| 逆問題rigor | inverse_rigor.png | inverse_rigor.py |
| forward UQ | conformal_emulator_uq.png | conformal_emulator_uq.py / uq_emulator*.py |

関連: [[project_masterarbeit]] [[project_compositesB_paper]] [[project_conferences_2027]] [[reference_thesis_tex_links]]
