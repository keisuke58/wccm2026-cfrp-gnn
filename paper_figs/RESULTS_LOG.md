# Composites B — results accumulation log (target submission ~2027)

このペーパーは1年かけて結果を貯める。新しい結果が出たらここに**日付・知見・図表・状態**を追記する。
図は `paper_figs/` に蓄積（再生成: `gen_compositesB_figs.py`、要 TeX Live for usetex）。

## 確定済みの主張（headline candidates）
1. **HybridMGN が in-distribution 新SOTA = 0.792**（表裏 cross-layer "world" edges）。> MeshGraphNet 0.761 > 公開GATベースライン 0.61。
2. **サイズOOD はアーキ非依存**: in-dist 0.55–0.79 に対し OOD-1×1 は全7アーキ 0.332–0.337（平坦）。アーキ改善は未知サイズに転移しない。
3. **検出は汎化・精密局在は崩壊**: OOD で detRec 0.64–0.96（GAT最良）だが defect-only F1 / exact ≈ 0。macro-F1 はクラス0で底上げ＝指標の落とし穴。
4. **正規化頑健**: DSPSSスケールを変えてもOOD不変＝0.33はスケール由来でない。

## 図（蓄積）
- `fig_arch_indist_vs_ood.png` — 主張2（アーキ比較, in-dist vs OOD）。
- `fig_detection_vs_localization.png` — 主張3（検出 vs 局在パネル, 3アーキ × in-dist/OOD）。

## アブレーション結果（2026-06-07 確定、GAT hidden16、多指標パネル）
| 構成 | macroF1 | defectF1 | exact | detRec | detFPR | AUPRC | AUC |
|---|---|---|---|---|---|---|---|
| **edge_geo（幾何エッジ特徴）** | **0.692** | **0.675** | 0.713 | 0.977 | 0.0003 | 0.988 | 0.9999 |
| baseline GAT（参考） | ~0.61 | – | – | – | – | – | – |
| logit-adjust + log_softmax | 0.065 | 0.043 | 0.217 | 1.000 | 0.707 | 0.944 | 0.938 |
| focal（誤killでtest未到達） | 0.555* | – | – | – | – | – | – | (*train best)

- **主張5候補: エッジ幾何特徴がGATを大きく押し上げる（0.61→0.69 macroF1, defectF1 0.675, detFPR 0.0003）**。メッセージ伝搬に相対位置/距離を入れる効果。
- **logit-adjust + log_softmax は有害**（macroF1 0.065、detFPR 0.71＝ほぼ全部欠陥判定で崩壊）。この不均衡設定には不適。

## 多軸OOD: 脆弱性は軸依存（2026-06-08）
サイズ以外の第2軸＝**ノイズ強度OOD**を測定（well-trained MeshGraphNet 0.7645、`ood_eval_noise.py`、clean test ↔ noise test、多指標パネル）:

| OOD軸 | in-dist | OOD | macroF1ギャップ | 挙動 |
|---|---|---|---|---|
| **サイズ**（2/4/8 → 未知1×1） | ~0.76 | 0.33 | −0.43 | 崩壊 |
| **ノイズ**（clean ↔ noise） | 0.762 | 0.761 | −0.001 | 頑健 |

ノイズtest詳細: macroF1 0.7612 / defectF1 0.8470 / detRec 0.9693 / detFPR 0.0006 / exact 0.9991（clean とほぼ同一）。

**ノイズ強度（非空虚性の担保）**: `_noise`データの実測ノイズ = **信号stdの12.0%**（noise_std 0.0706 / signal_std 0.5886、全ノードに付与、max|Δ|0.33）。
→ 12%という非自明な強度で性能ギャップ0.001＝robustは空虚でない。「**信号の約12%のノイズに対して頑健**」と定量明記できる（強度依存への対応）。

**主張7候補: OOD脆弱性は軸依存**。未知のサイズ/スケールには壊滅的だが、測定ノイズには完全頑健。
→ サイズOODの崩壊は「一般的な脆さ」ではなく**サイズ汎化に特有の失敗**であることを示す（主張2を鋭くする）。

## 逆方向OOD（train 1×1 → test 2/4/8）2026-06-07
- reverse macroF1 **0.462**（AUC 0.861） vs forward (2/4/8→1×1) 0.332。
- **主張6候補: OODは非対称＝最小サイズ（穴1個）で訓練する方が大サイズへ汎化する**（小さい局所応力場が大サイズの部分集合になるため）。サイズ汎化の実用指針。

## まだ足りない（1年で貯める結果）
- [ ] 複数シード ±std（rigor、勝ち構成=edge_geoだけでも）
- [x] アブレーション（edge_geo勝ち / logit-adj有害 / focalは要再走）← 2026-06-07確定
- [x] 逆方向OOD非対称性（reverse 0.462 > forward 0.332）← 2026-06-07確定
- [ ] focal再走（誤killで未完。CE/focal比較を埋める）
- [ ] 多軸OOD（未知の位置/層/穴形状/ノイズ強度）
- [ ] OODギャップを埋める非アーキ手段（サイズ認識特徴/augmentation/domain generalization）→ 0.33をどこまで上げられるか
- [ ] 欠陥単位評価（重心誤差[mm]/IoU）
- [ ] 強い非GNNベースライン（CNN等）で GNN の正当化
- [ ] 校正(ECE)/UQ・コスト表
- [ ] 公開データ＆コード（ベンチ化）

## ログ
- 2026-06-07: アーキ横断OOD完了（主張1-4確定）。図2枚作成。アブレ/逆実験 実行中。指標パネルを既定化(`ood_metrics.py`/`train.py full_metrics_panel`)。

## 定量目標（2026-06-07設定）
- in-dist: HybridMGN macro-F1 ≥0.80(現0.792) / defect-F1 ≥0.80(現0.781) / exact ≥0.85(現0.852✓)。
- **OODギャップ縮小**: 非アーキ手段で OOD defect-F1 ≥0.30(現≈0) or OOD detRec ≥0.85 @ FPR ≤0.10。
- rigor: 勝ち構成3シード±std / アブレ確定 / OOD ≥2軸(サイズ＋位置 or 層)。
