# results — 最新の結果図（2026-01-15 14:33:43 のラン）

差分正規化 zscore + GAT（既存ベスト設定）での評価図。`keisuke58/nishioka_cfrp_gnn` の
`Predict_truth/pred_vs_true_images_zscore_20260115_143343/` から取得した最新版。

| ファイル | 内容 |
|---|---|
| `confusion_matrix_20260115_143343.png` | 19クラス混同行列。層制約により層1=クラス1–9 / 層2=クラス10–18 のブロック構造 |
| `f1score_class_20260115_143343.png` | クラス別 F1 |
| `detailed_class_metrics_20260115_143343.png` | クラス別 Precision/Recall/F1 詳細 |
| `pred_vs_true_class_labels_20260115_143343.png` | 予測 vs 正解ラベル |

観察:
- クラス0（欠陥なし）が支配的。隣接クラス（特に 5/6 や深層 10–13）への吸収が残る
  → 論文の弱点「隣接層誤分類」。本リポの `--label_smoothing` / ordinal 拡張で対策予定。
- 左右領域の非対称も既知 → `--mirror_augment` で対策予定。

※ スライド（7月）にはこの最新図、または改良アブレーション後の更新図を使用。
