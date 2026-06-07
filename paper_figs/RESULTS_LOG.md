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

## まだ足りない（1年で貯める結果）
- [ ] 複数シード ±std（rigor、勝ち構成だけでも）
- [ ] アブレーション確定（loss: CE/focal/logit-adjust、NDF、balance、edge特徴、sampler）← 2026-06-07 実行中
- [ ] 逆方向OOD（train 1×1 → test 2/4/8）の非対称性 ← 実行中
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
