# Phase-field × GNN prognosis — Keio Phase-2 plan (idea #6)

作成 2026-06-13。慶應・村松研（計算固体力学）Phase 2（2027.04–2028.03 想定）の土台。
現在のIKM主題（連続体×ベイズ×空間PDE / biofilm）から、村松研の **phase-field / FEM / マルチスケール / ML** へ橋渡しする具体線。プロトタイプ＝`phasefield_gnn_prognosis.py`。

## 1. 何を作ったか（プロトタイプの到達点）

AT2 phase-field（`cfrp_phasefield_2d.py`、Miehe staggered、MMS 2次精度検証済）を**正解物理**に、
GNNが「**seed欠陥場 → 破壊後クラック場**」の順問題演算子を学習。

- データ: 正規化2D積層板、30欠陥（位置/サイズ/層をランダム）、固定過臨界荷重 L=0.16 でAT2を解き最終損傷場を取得。
- グラフ: 40×32格子=1280節点、4近傍辺、節点特徴 [seed損傷, 局所Gc, x, y]。torch_geometric非依存の3要素メッセージパッシング（自己・近傍平均・**グローバル平均**）。
- 学習: leave-defect-out（22学習/8評価）、節点回帰。グローバル文脈＋**seed重心ブロードキャスト特徴**＋BCE損失＋8ラウンドMP。
- 結果（held-out）: **node-R²=0.94、crack-IoU=0.89**（先鋭化前0.81/0.86から改善）。クラックの**貫通方向位置（seed層→破壊帯）と全体損傷場を高精度に汎化**。
- 正直な限界: 鋭い全幅帯の中心輝度は一部の欠陥で**やや局在**（IoU 0.89に残る差）。背景勾配は良好。形態のさらなる先鋭化（U-Net型/界面エッジ条件）はPhase 2へ。
- **疲労サイクル予後は基盤限界**: AT2正規化2Dでは亀裂が**ほぼbang-bang**（サブ臨界→破壊がフライト1-2で飽和、`simulate_fatigue_flights`で確認）＝漸進Paris領域が無く「残フライト数の場からの回帰」が退化。真の疲労余寿命予後にはPhase 2で**Paris領域を持つ基盤（3D/CZM/実Gc）**が必須（§3軸4）。

→ 「phase-field→GNNエミュレーションのループが回り、未知欠陥に高精度汎化する（node-R²0.94）」ことの実証。これが土台。

## 2. なぜ村松研と最相性か

- 村松研＝計算固体力学（FEM、phase-field、マルチスケール、量子アニーリング/ML）。本プロトは**phase-field破壊×GNN代理**そのもの。
- IKMからの前方接続: 連続体損傷（biofilm空間PDE / ベイズUQ）→ 固体破壊（phase-field）へ、**PDE＋ベイズ＋グラフ学習**という道具立てを保ったまま主題だけ移送できる（[[project_keio_bridge]]）。
- 既存資産が効く: CZM-GP代理(R²0.994)、FMPE逆問題、conformal/oracle分解UQ、MeshGraphNet検出は全てPhase 2の部品。

## 3. Phase 2 で伸ばす5軸（プロト→修論）

1. **形態の先鋭化**: ✅一部達成。seed重心ブロードキャスト＋BCE＋8ラウンドで held-out node-R²0.81→**0.94**。さらに `phasefield_emulator_compare.py` で**メッシュ非依存GNN(R²0.92/IoU0.89)が構造化グリッドCNN U-Net(0.94/0.86)と同等**＝GNN採用は精度を犠牲にしない（差≤0.03、IoUはGNN上）。残差は界面エッジ条件付きMPで詰める。
2. **3D化**: ✅プロト着手。`phasefield_gnn_3d.py`＝`phasefield_3d.py`の剥離前縁進展(12×16×16)をGNNがエミュレート（leave-defect-out）。**3D不規則メッシュではCNN U-Net不適＝GNNが唯一の選択**。
3. **実Gc・実荷重**: 正規化を捨て T700SC/LY556（DLR Readmeの実材料）＋実荷重スペクトルで較正。
4. **疲労サイクル予後**: Carrara型 fatigue degradation（`cfrp_phasefield_2d.fatigue_degradation` 既存）を時間軸に、GNNが**サイクル→損傷成長**を予測＝真の余寿命予後。
5. **UQ伝播**: FMPE欠陥事後分布→GNNエミュレータ→クラック場の事後分布→conformal被覆（[[idea #5]] `conformal_transfer.py`）でクリアランス判定に分布フリー保証。

## 4. 想定する修論の主張（Keio審査向け）

> 「phase-field破壊解の**構造横断・微分可能なグラフ代理**を構築し、欠陥→損傷場→余寿命を高速・UQ付きで予測。FEMの物理整合性とML代理の速度を両立し、実材料・3D・疲労へ汎化する。」

審査戦略: 村松研の計算固体力学に**物理（phase-field/MMS検証）で軸足**を置きつつ、IKMで培ったベイズ/グラフ学習を差別化要素に。overclaimせず、エミュレータの**検証（MMS・held-out・UQ較正）**を主役に据える（[[feedback_shm_depth_over_breadth]]）。

## 5. リンク・コード
- プロト: `phasefield_gnn_prognosis.py`（本リポ）、図 `cfrp_datasets/phasefield_gnn_prognosis.png`。
- 物理: `cfrp_phasefield_2d.py` / `phasefield_3d.py`（MMS検証 `tests/test_cfrp_phasefield.py`）。
- 部品: `crack_surrogate.py`(GP代理)、`fmpe_defect.py`(逆問題)、`conformal_transfer.py`(UQ保証)。
- 関連メモ: [[project_keio_bridge]] [[project_masterarbeit]] [[project_research_ideas]] [[reference_masterarbeit_authors]]
