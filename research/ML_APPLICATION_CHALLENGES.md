# CFRTP への ML 実応用：実務課題と本リポの対応

工程・残留応力・結晶化への ML 適用でよく報告される実務課題を調べ、各々に対して本リポ
（物理主役・ML従属）でどう対処するかを対応づけた。数値はすべて例示（未校正）。

## 実務課題（文献ベース）

1. **データ不足**：FE も実験も高コストで学習データが少数。深層NNより**データ効率の高い**
   代理モデル（GP 等）が有利。injection molding では GPR が多くの品質指標で最良と報告。
2. **外挿に弱い**：GPR（＝BO の予測器）は**学習範囲外の予測に不適**。既存データは
   その条件外の最適化にほとんど使えない。→ 外挿は明示的に避ける/検知する必要。
3. **不確かさの定量が必須**：意思決定・獲得関数・「どこを信じるか」に不確かさが要る。
   決定論的NNは出さない。GP は分散を返す。
4. **汎化性の欠如**：単一材料・単純形状で学習したモデルは**他系へ移らない**（報告の
   代表的限界＝single material / simple geometry）。→ 物理を権威に、系ごとに再校正。
5. **製造ばらつき/分布シフト**：ボイド・工具拘束・冷却不均一など現場ノイズ。頑健化が必要
   （本リポの GNN 側は学習時オンラインノイズ注入で対処済み）。
6. **校正・モデル形式誤差**：構成則パラメータ（HL/Prony）が不確か。実測合わせ＋不確かさ。

## 本リポの対応

| 課題 | 対応（実装/方針） |
|---|---|
| ①データ不足 | **ガウス過程（GP）＋ベイズ最適化**でFE呼び出しを最小化（`design/cfrtp_bayesopt.py`）。少数点から逐次的に最適へ。 |
| ②外挿 | 設計空間を物理的に**有界化**し、GP 予測分散が大きい領域は**外挿ガードで却下/警告**。推薦点の不確かさを明示。 |
| ③不確かさ | GP の平均±2σを出力、獲得関数（制約付きEI×実行可能確率）に反映。 |
| ④汎化 | **物理（FE）が精度の権威**、ML は加速層。系（PEEK→ダイキンPFA）ごとに再学習・再校正前提。 |
| ⑤ばらつき | 学習時ノイズ注入（GNN 側 `--train_noise_std`）、将来は工程ばらつきを入力分布として伝播。 |
| ⑥校正 | ベイズ推定（MCMC/ブートストラップ）で HL/Prony を実測にフィット（`cfrtp_calibration.py` 路線、`validation/` の RMSE フック）。 |

## 実装方針（この課題調査を受けて）
- **スカラー代理（残留応力・α・Tp）＝ GP**。**逆設計＝制約付きベイズ最適化**（本命）。
- **場予測（応力分布）＝ DeepONet/FNO・GNN(GAT/MeshGraphNet)**（データが十分育ってから）。
- いずれも **提案点は必ず FE で検証**（surrogate-searched, FE-verified）。

## 出典
- Rapid evaluation of cure-induced residual stress (cGAN): <https://www.sciencedirect.com/science/article/abs/pii/S026382232301173X>
- Fast prediction of curing residual stress (deep learning): <https://doi.org/10.1002/pc.29501>
- ML for degree-of-cure uniformity (autoclave): <https://doi.org/10.3390/aerospace8050130>
- Experiment-driven GP surrogate + Bayesian optimization (injection molding): <https://www.mdpi.com/2073-4360/18/8/902>
- Physics-informed Bayesian optimization for extrapolation: <https://www.nature.com/articles/s41524-025-01522-8>
- Deep GP for Bayesian optimization (additive mfg.): <https://www.tandfonline.com/doi/abs/10.1080/24725854.2024.2312905>
- Extrapolative BO (GP + NN ensemble): <https://advanced.onlinelibrary.wiley.com/doi/10.1002/aisy.202100101>
