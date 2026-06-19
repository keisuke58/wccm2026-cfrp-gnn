# WCCM 進捗ミーティング ブリーフ — 2026-06-23 09:00 CET

> 自分用の進捗報告メモ。技術防御の詳細 Q&A は [`wccm_qa.md`](wccm_qa.md) に分離。
> 数字の正典は [`../RESULTS.md`](../RESULTS.md) / [`../OOD_RESULTS.md`](../OOD_RESULTS.md) / [`../paper_figs/RESULTS_LOG.md`](../paper_figs/RESULTS_LOG.md)。

## 30秒サマリ
構造非依存の判定コア × 構造固有のセンシング/予後で、**3つの実 CFRP 構造 × 3センシングモダリティ**を一気通貫でデモ。
登壇は **7/22 MS090E**。今日は (1) 結果と3本柱、(2) 前回以降の新規、(3) 正直な検証ステータス、(4) future 2軸、を共有して方針合意したい。

## 3本柱（実構造 × センシング）
| 構造 | センシング | headline 数字 | 実データ |
|---|---|---|---|
| 穿孔インターステージ | 表面応力 (DSPSS) GNN | 公開 macro-F1 **0.61** → 新 **HybridMGN 0.792**（in-dist SOTA）；検出 AUROC **0.999** | Kojima prosthetic TSA（sim2real proxy） |
| H3 サテライトフェアリング | guided-wave GNN | LGSTA node-F1 **0.86** > SAGE 0.79 | OGW long-term（実測） |
| SRB-3 モーターケース | AE | burst 予後 vs CZM 495-case FEM：pcr↓ ρ=**−0.994** | 4TU AE .pridb（実測, CC0） |

## 前回以降の新規（ここを強調）
1. **HybridMGN = 新 in-dist SOTA 0.792**（表裏 cross-ply "world" edges）。MeshGraphNet 0.761・公開 GAT 0.61 を上回る。ただし **0.80 には僅かに届かず**＝adjacent-layer 混同が上限（exact 19-class acc は既に 0.85）。
2. **Size-OOD（未知 1×1）の rigorous な反転結果**：**検出は汎化**（defect-node recall 最大 0.96）が、**細かい 19-class 局在は崩壊**（defect-only F1 ≈ 0）。全アーキで同一＝**データ/表現の問題でアーキの問題ではない**。macro-F1 には見えない（クラス0で底上げ）。
3. **Paris 則の実データ較正 DONE（初の positive な実データ予後較正）**：4TU DCB Mode-I 37 specimens に fit、within-specimen R²=0.98。**実 m = 16.7 [95%CI 15.0–18.3]** に対し代表値 m=3 は **約6倍 too shallow**＝要修正と判明。単一 (C,m) 仮定は LOSO で破綻（fibre-bridging R-curve）。
4. **NASA PCoE RUL = 正直な negative**：raw-strain proxy は dead-end（gauge debond）。RUL はこの経路では未検証と明言。

## 正直な検証ステータス（先に自分から認める）
- **[R] 実データで外的に意味あり** — 検出フロントエンド各種 ＋ Paris 較正のみ。
- **[S] 自己整合（≠ ground truth）** — oracle/decision-UQ/LOSO/conformal/system_baseline は全て同一 FEM/surrogate を predictor と "truth" の両方に使用。UQ が正しく伝播するかの内部チェック。
- **[U] 代表値/未較正** — Stage-3 予後物理（phase-field・debond ERR+Paris・burst・CZM pcr）と全コスト定数。trend-level、実破壊試験に未 fit。
- **コア gap**：実インターステージ/フェアリング/SRB-3 の実破壊、実 CFRP 破壊/疲労クーポンとの直接比較はまだゼロ。→ **depth > breadth**：軸を増やすより1スレッドを実データで端まで通す。

## Future 2軸（合意したい方向）
1. **ペイロード応用への展開** — フェアリング/実構造への転移（Payload2026 連携）。
2. **実験データ追加（sim2real）** — 実 IR 応力（TSA）測定への転移、Paris m の実較正を Stage-3 に反映。

## 今日 advisor に決めてほしいこと
- [ ] **7/22 デッキの語り** — 本命=公開0.61準拠で固く行くか、extended の 0.792 ＋ OOD まで前に出すか。
- [ ] **次の1スレッド優先順位** — (a) OOD 局在 gap を size-aware features で閉じる / (b) sim2real（実 TSA）/ (c) 別構造追加。depth>breadth 方針なら1つに絞る。
- [ ] **Paris m=16.7 の扱い** — Composites B 論文に今 fold するか（代表 m=3 の修正＋単一則批判が新規貢献になる）。
- [ ] 共著・データ利用の確認事項があれば（Kojima データは合意済み）。
