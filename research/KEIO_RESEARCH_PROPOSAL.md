# 研究計画書（慶應義塾大学・村松眞由研究室／2027年4月〜2028年3月）

**指導教員（予定）：** 村松 眞由 准教授（慶應義塾大学 理工学部 機械工学科）
**研究室：** Multiphysics Materials Computation Research Group（計算固体力学・マルチスケール／フェーズフィールド／量子アニーリング・機械学習）
**申請者：** 西岡 佳祐（Leibniz Universität Hannover 修士課程 → 慶應義塾大学 修士課程進学予定）
**作成日：** 2026-06-11（草稿）

> 注記：本草稿の数値・実装結果は、本人が実装・テスト・図化済みの研究資産（リポジトリ `wccm2026-cfrp-gnn`、`Payload_gnn`）に基づく予備的結果である。村松先生の論文・研究テーマは公開情報（researchmap・研究室公式サイト・各誌掲載論文）に基づいて参照した。確証が取れていない箇所は「[要確認]」と明示する。

---

## ① 研究題目

**「再使用型宇宙輸送機のための積層条件付き微分可能フェーズフィールド代理モデルとベイズ逆問題・設計最適化の統合フレームワーク」**

（英題案：*A layup-conditioned, differentiable phase-field surrogate framework for Bayesian inversion and design optimization of reusable launch-vehicle structures*）

副題的に：CFRP構造の損傷予後（prognosis）から設計フィードバックまでを、村松研究室の専門であるフェーズフィールド破壊・マルチスケール均質化・量子アニーリング最適化と接続して扱う。

---

## ② 背景・課題

### 再使用ロケット時代が変えるSHMの問い
SpaceX に代表される再使用型ロケットの実用化、および JAXA の再使用観測ロケット（CALLISTO 級）に向けた動きにより、構造ヘルスモニタリング（SHM）に求められる問いが変質している。従来の「この機体は今回の飛行に耐えるか（壊れないか）」から、**「次も飛べるか／あと何回飛べるか（再飛行クリアランス・余寿命）」**へと移行している。実運用では SpaceX が予防交換から状態基準保全（condition-based maintenance）へ転換し、20飛行認証→実績蓄積による40飛行への段階的延長、先頭機を低リスク便で先行させる「フリートリーダー」運用を行っている。これらは経験的に実施されているが、**ベイズ的・物理整合的に形式化された定量的枠組みは確立されていない。**

### CFRP構造に固有の難しさ
段間部・フェアリング等の主要構造に CFRP が用いられるが、CFRP は (1) 異方性（繊維方向に依存する破壊靱性）、(2) 層間剥離（delamination）という固有の損傷モード、(3) 積層構成（layup）に強く依存する力学応答、をもつ。したがって損傷の検出・特性評価・予後・設計改善のいずれの段階でも、繊維配向・界面・積層という固体力学的構造を陽に扱う必要がある。これはまさに**計算固体力学・フェーズフィールド破壊・マルチスケール均質化の問題**である。

### 未解決の核心
申請者はこれまでに、検出から設計フィードバックまでを貫く SHM フルスタック（後述 Stage 0–5）をプロトタイプとして実装・実証した。その過程で、**「速い・微分可能・物理整合な順モデルが、積層構成を入力として扱えない」**という本質的な障壁が明確になった（④で詳述）。これが解けないと、ベイズ逆問題も設計最適化も実用速度で回らない。本研究はこの障壁の解消を中核に据える。

---

## ③ 研究目的

本研究の目的は、再使用 CFRP 構造の損傷予後と設計最適化を、**積層条件付きの微分可能フェーズフィールド代理モデル**を中核として統合的に扱う計算フレームワークを構築することである。具体的には以下を達成する。

1. **積層条件付き中立オペレータ（layup-conditioned neural operator）** の構築：積層構成（繊維配向系列）を入力に取り込んだ Fourier Neural Operator / DeepONet を学習し、別積層を OOD（分布外）にしない、速い・微分可能な順写像を得る。
2. その微分可能順写像を用いて、損傷パラメータの**ベイズ逆問題**（申請者の現修士論文の中核手法である TMCMC × GPU を実用速度で適用）と、**設計最適化**（積層・界面靱化のロバスト最適化）を解く。
3. 村松研究室の専門であるフェーズフィールド破壊・マルチスケール均質化・量子アニーリング最適化と接続し、(a) 真の疲労フェーズフィールドへの拡張、(b) 単層 AT2 からマイクロ構造マルチスケールへの拡張、(c) 設計探索の QUBO/イジングマシン定式化、を発展課題として位置づける。

---

## ④ これまでの研究と予備的結果（実現可能性の根拠）

### (A) ベイズ推定・変分構造の素地（LUH 修士論文）
申請者は LUH（IKM, Philipp Junker 教授）にて、拡張ハミルトン原理に基づく多種バイオフィルム相互作用の熱力学整合モデルと、そのパラメータの**ベイズ推定（TMCMC: Transitional MCMC）を JAX により GPU 並列化**して同定する研究を行っている（修士論文：*GPU-accelerated Bayesian inference of multi-species biofilm interaction parameters via TMCMC*、2026年11月発表・12月提出予定）。ここで培った (1) 変分原理・熱力学整合のモデリング規律、(2) TMCMC/SMC による高次元事後分布推定、(3) GPU 並列化による大規模推論、は、本研究の**微分可能順モデル × ベイズ逆問題**にそのまま転用できる中核資産である。
> 補足：本テーマ（バイオフィルム）自体は固体力学とは別分野だが、変分構造とベイズ推定の方法論は共通であり、本提案では「手法の転用」として明示的に橋渡しする。

### (B) SHM フルスタック Stage 0–5 の実証（2026-06、CFRP×宇宙構造）
申請者は CFRP 段間構造を対象に、検出から設計改善までを貫く SHM スタックを実装・テスト（テスト総数 約80）・図化・コミット済みである（リポジトリ `wccm2026-cfrp-gnn`、論文ドラフト `composites_b_draft.tex`）。各段の予備的結果は以下の通り。

| Stage | 問い | 手法 | 主要結果（予備） |
|---|---|---|---|
| 0 検出 | 異常があるか | per-node マハラノビス距離 | ノードレベル AUROC ≈ 0.999（クリーン条件） |
| 1 分類 | 何の損傷か | 教師あり GNN（WCCM2026 資産） | ノイズ頑健（測定ノイズ下でも分類維持） |
| 2 特性評価 | どの程度深刻か | FMPE（flow-matching posterior） | 校正済み事後分布、被覆 0.87–0.94 |
| 3 予後・判断 | 次飛べるか | 異方性 AT2 フェーズフィールド + flight_clearance | 下記4面で完成 |
| 4 フリート学習 | 機体群で賢くなるか | 階層ベイズ（TMCMC 系譜） | フリートリーダー効果 2.75→1.25 回点検 |
| 5 設計進化 | 次世代機をどう作るか | フリートロバスト BO（積層・界面最適化） | E[残飛行] 1.9→6.0（+216%） |

**Stage 3 の「4面」**（本研究の出発点）：
- **正確（FD）**：保存型 flux 形式の多重 ply 異方性 AT2 フェーズフィールド。繊維30°でき裂23.3°偏向、界面バンド捕捉（73 vs 0 セル）等で物理的妥当性を確認。
- **微分可能（JAX & FNO）**：JAX 実装 AT2 の勾配を FD チェックで検証（rel ≈ 3e-7）。これにより微分物理ベースのベイズ逆問題が可能となり、FMPE が OOD で被覆 0.958→0.000 に崩壊するのに対し、微分物理事後（Laplace）は被覆 0.958 を維持＝**「償却推論は外挿不可、順物理は汎化する」を定量実証**。
- **高速（FNO 代理）**：演算子 G:(初期損傷 d0, 荷重)→(最終損傷場, P(grow)) を学習。**142–150倍速**（FD 1763ms→12.4ms）、場 rel-L2 3.5%、成長判定 acc 0.988、Brier 0.010（校正良）。FNO を微分可能順写像として用いた逆問題で、autograd vs FD 勾配一致（P(grow) 6.5e-4）、逆問題全体 534s→12.3s（43倍）。
- **正直（conformal）**：split-conformal による信用ゲート。分布内 95% 被覆を確認、OOD で被覆崩壊（δ0.06→80%、δ0.39+→0%）を定量化し、OOD の 33% を FD にフォールバックさせ被覆 61→88% に回復。**「速い代理は信用範囲内のみ、外れたら厳密 FD に戻す」**設計を明示的に実装。

**一気通貫**：`run_pipeline.py` により、検出→特性評価→予後→クリアランス→フリート更新がワンコマンドで全体 3.9–5.4 秒で走る。

### (C) これらが示す実現可能性
上記は、本研究で扱う要素技術（異方性フェーズフィールド、微分可能順モデル、中立オペレータ代理、ベイズ逆問題、ロバスト設計最適化）が**いずれも申請者の手元で動作実証済み**であることを意味する。本研究はゼロからの探索ではなく、**実証済みプロトタイプの明確な弱点を、村松研究室の専門性で正面から解く**ものである。

### (D) 村松研究室との既存の協働実績
申請者は既に村松准教授との共著で、CFRP 多孔構造の欠陥同定に FEM と GNN を用いた研究を *Frontiers in Materials*（Vol. 12, 2025, DOI: 10.3389/fmats.2025.1652484）に発表済みであり、さらに WCCM-ECCOMAS 2026（ミュンヘン、2026年7月、口頭発表 accepted）、COMPSAFE 2025（神戸、発表済み）でも共同発表している。本研究はこの協働の自然な発展である。

---

## ⑤ 慶應修士での研究計画（M1/M2 のマイルストーン）

本研究の中核課題は、Stage 0–5 実証の過程で「未解決」として明確化された以下である。

### 核心課題1：積層条件付き中立オペレータ（最重要・一本道）
Stage 5（設計フィードバック）で FNO 代理が使えなかった本質的理由は、**代理モデルが積層構成（layup）を入力に持たず、別積層がすべて OOD になる**ことであった（そのため Stage 5 は遅い FD 順モデルに依存している）。これを解く **layup-conditioned FNO / DeepONet** を構築する。積層は速い微分可能順モデルとなり、**TMCMC × GPU（現修士論文の中核手法）によるベイズ逆問題および設計最適化を実用速度で回す鍵**となる。これが現修論手法と村松研フェーズフィールドを橋渡しする一本道である。

### 核心課題2：真の疲労フェーズフィールド（Carrara 2020）
現 Stage 3 の多フライト損傷蓄積は i.i.d. 繰返し荷重ブロックの近似にとどまる。村松研究室のフェーズフィールド破壊の専門性を用い、**本物の疲労損傷蓄積（Carrara et al. 2020 の疲労フェーズフィールド）**へ拡張する。

### 核心課題3：マルチスケール／界面
現状は単層スケールの異方性 AT2 である。これを**マイクロ構造（繊維／マトリクス／界面）のマルチスケール・フェーズフィールド**へ拡張し、均質化を介してマクロ構造予後に接続する。村松研究室のマルチスケール固体力学（結晶塑性 × 均質化 FEM、二相材料のデータ駆動評価）のド真ん中である。

### 核心課題4：量子アニーリング／機械学習（発展）
Stage 5 の設計空間ベイズ最適化（BO）を、村松研究室の**量子アニーリング／イジングマシン最適化**に載せ替える発展を視野に入れる。

### 段階計画

| 期 | 主目標 | マイルストーン | 想定発表先 |
|---|---|---|---|
| **M1 前期**（2027 春〜夏） | 積層条件付きオペレータの構築 | (i) 積層を入力に取り込んだ FNO/DeepONet 学習、(ii) 別積層での汎化検証（OOD が解消されることの定量化）、(iii) 勾配品質の FD 検証 | 国内学会（計算工学講演会等） |
| **M1 後期**（2027 秋〜冬） | 微分可能逆問題の統合 | (iv) 積層オペレータ上で TMCMC×GPU ベイズ逆問題、(v) FMPE 比較で OOD ロバスト性を実証、(vi) 階層ベイズ（フリート学習）との結合 | 国際誌投稿準備（CMAME / Composites Part B 級） |
| **M2 前期**（2028 春〜夏） | 疲労・マルチスケール拡張 | (vii) Carrara 型疲労 PF への置換、(viii) 繊維／界面マルチスケール PF と均質化、(ix) 設計最適化の更新 | 国際誌投稿 |
| **M2 後期**（2028 秋〜冬） | 設計最適化・量子アニーリング・統合 | (x) フリートロバスト設計最適化の精緻化、(xi) 設計探索の QUBO 定式化（村松研手法）の予備検討、(xii) 修士論文統合・審査 | 修士論文・国際誌 |

> リスクと逃げ道：核心課題1（積層オペレータ）が M1 で十分な汎化を示さない場合でも、(a) 微分物理ベース逆問題（FD/JAX、実証済み）に退避して逆問題・疲労拡張を先行、(b) 合成データ生成は実証済み Stage 3 モデルで自給できるため外部データ依存が小さい。各核心課題は単独でも論文一本になる粒度であり、スコープ規律として「課題1が出荷されるまで課題3を始めない」を原則とする。

---

## ⑥ 村松研究室で行う必然性（先生の専門性との接続）

村松准教授は計算固体力学を基盤とし、FEM に他手法（フェーズフィールド・マルチスケール均質化・量子アニーリング・機械学習）を組み合わせて、金属・高分子・セラミックスの複雑現象を解明する研究を展開している（Multiphysics Materials Computation Research Group）。本研究の各核心課題は、以下の具体的な専門性と直接対応する。

- **フェーズフィールド破壊・相変態**：LSCF（SOFC 材料）の強弾性相変態をフェーズフィールド＋FEM で予測する研究（"Ferroelastic phase transformation of LSCF based on phase-field model"）、デンドライト形状（PLOS One 2025）、フェーズフィールド＋転位結晶塑性による動的再結晶など、**弾性エネルギーを組み込んだフェーズフィールドを FEM で解く**蓄積がある。本研究の核心課題2（疲労 PF）・課題3（マルチスケール／界面 PF）はこの中核と直結する。
- **量子アニーリング／イジングマシン最適化**：「フェーズフィールドモデルの QUBO 補正項の定式化」（*Int. J. Numer. Methods Eng.*, 2025, DOI: 10.1002/nme.70019）、「流路トポロジー最適化のイジングマシン定式化」（*Engineering with Computers*, 2026）、「量子アニーリングによるトラス構造最適化」（*Scientific Reports*, 2024）、「ジブロック共重合体相分離のイジングマシン PF」（*Scientific Reports*, 2022）など、**最小化問題を QUBO 化してイジングマシンで解く**第一線の業績がある。本研究の核心課題4（設計探索の QUBO 化）はこの専門性をそのまま借用する。
- **マルチスケール均質化・データ駆動**：二相鋼の結晶塑性 × 粒径依存（*Int. J. Solids Struct.*, 2025）、ポリカーボネートの均質化 FE × MD（secant 型マルチスケール構成則, 2026）、二相材料のデータ駆動力学評価（*Materials & Design*, 2025）など、**均質化と機械学習を結合**する蓄積がある。本研究の核心課題1（中立オペレータ代理）・課題3（マルチスケール均質化）はこの系譜に乗る。

すなわち本研究は、申請者が持ち込む **(I) ベイズ推定・微分可能順モデルの方法論**（LUH 修論の TMCMC×GPU、Stage 0–5 の微分可能 PF・中立オペレータ）と、村松研究室が持つ **(II) フェーズフィールド破壊・マルチスケール均質化・量子アニーリング最適化の専門性**とを統合する位置にあり、**いずれか一方だけでは成立しない**。村松研究室はこの統合を担える国内でも稀な環境である。

---

## ⑦ 期待される学術的・産業的貢献

### 学術的貢献
1. **積層条件付き微分可能オペレータ**：CFRP 破壊予後において、積層構成を陽に条件づけた中立オペレータ代理は文献に乏しく、これが速い微分可能順モデルとして TMCMC ベイズ逆問題・設計最適化を駆動できることを示す。
2. **償却推論の外挿限界の克服**：FMPE 等の償却（amortized）推論が OOD で破綻する一方、微分物理に基づく逆問題が汎化することを定量実証済み（被覆 0.958 維持 vs 0.000 崩壊）。本研究はこの知見を「物理を閉じた微分可能オペレータ」で実用速度と汎化性の両立へ昇華する。
3. **フェーズフィールド × ベイズ × 量子アニーリングの接続**：村松研のフェーズフィールド・QUBO 手法と、申請者のベイズ・微分可能順モデルを結合する新しい計算固体力学の方法論。CMAME / Composites Part B / JMPS 級を目標とする。

### 産業的貢献
- **JAXA 再使用機（CALLISTO 級）**：再飛行クリアランス・余寿命判定・設計フィードバックを一貫したベイズ物理フレームワークで提供。SpaceX が経験的に行う段階的認証・フリートリーダー運用の**形式化・ベイズ化**として位置づく。
- **起業構想（Physics-Informed AI Simulation Platform）への展開**：本スタックは「物理資産の運用知能」汎用スタックであり、再使用ロケットと半導体製造（ウェハ・ダイシングの異方性脆性破壊：同型の AT2 異方性 PF コード）に同一の枠組みを適用できる。研究の方法は公開（CFRP／ロケット）し、商用展開（半導体）に応用する戦略の学術的基盤となる。

---

## ⑧ 今後の課題（正直な限界の明示）

審査・再現性の観点から、現時点の限界を以下に明示し、本研究の射程内で順次解消する。

- **合成データ依存**：Stage 0–5 の予備結果は主に合成データに基づく。実験・実測（フルフィールド計測、フリート飛行履歴、実ウェハ NDT）はまだ無く、当面は実証済み Stage 3 モデルをデータ生成器として用いる。sim-to-real 検証は今後の課題。
- **粗格子・AT2 のℓ依存**：微分可能 PF は粗格子（例 28×20）で実装しており、AT2 の見かけ強度は正則化長さ ℓ に依存する（傾向把握には有効だが絶対値は要注意）。格子収束・MMS（製作解）による検証、ℓ 依存性の系統評価を行う。
- **疲労の近似**：現状の多フライト蓄積は i.i.d. 繰返し近似であり、Carrara 型疲労 PF への置換が必須（核心課題2）。
- **荷重モデルの簡略化**：現 Stage 3 の荷重は横方向（Mode-I peel）ひずみの proxy であり、α/β 閾値は暫定。荷重包絡の高度化が必要。
- **代理の信用範囲**：代理モデルは信用範囲内でのみ用い、OOD では厳密 FD にフォールバックする設計（conformal ゲートで実装済み）を堅持する。

> 審査戦略（examiner awareness）：村松研究室および慶應審査委員会は計算固体力学の専門家であるため、(1) 変分構造・熱力学整合（フェーズフィールドの自由エネルギー汎関数の整合）、(2) 数値検証（格子収束・MMS・保存則・FD/FEM/代理のトレードオフ）を既定で提示する。委員構成が定まり次第、村松先生の近著（上記）を踏まえた想定問答を別途整備する。なお発表では、先行研究としての自己引用方針（先行研究は共著者 Kojima 等を引用、自己引用は避ける）を踏襲する。

---

## ⑨ 参考文献（暫定・要確認）

**申請者の業績**
1. Nishioka, K., Kojima, Y., Saito, T., Kawakami, K., Washiya, M., Muramatsu, M., "Development of Defect Localization Method for Perforated Carbon-Fiber-Reinforced Plastic Specimens Using Finite Element Method and Graph Neural Network," *Frontiers in Materials*, Vol. 12, pp. 1–15, 2025. DOI: 10.3389/fmats.2025.1652484.
2. Nishioka, K. et al., "GPU-accelerated Bayesian inference of multi-species biofilm interaction parameters via TMCMC," Manuscript in Preparation, 2026.（LUH 修士論文・主論文）
3. Nishioka, K. et al., "Development of a Defect Estimation Method for CFRP Interstage Structures with Holes in Space Transportation Systems Using FEA and GNN," WCCM-ECCOMAS 2026, Munich.（口頭発表 accepted）

**村松研究室・関連手法**
4. Aoki, et al. (incl. Muramatsu), "Formulation of Correction Term in QUBO Form for Phase-Field Model," *International Journal for Numerical Methods in Engineering*, 2025. DOI: 10.1002/nme.70019.
5. (Muramatsu et al.), "An Ising machine formulation for design updates in topology optimization of flow channels," *Engineering with Computers*, 2026.
6. (Muramatsu et al.), "Development of optimization method for truss structure by quantum annealing," *Scientific Reports*, 2024.
7. (Muramatsu et al.), "A phase-field model by an Ising machine and its application to the phase-separation structure of a diblock polymer," *Scientific Reports*, 12, 2022.
8. Suzuki, Muramatsu, Reese, Prume, "Efficient evaluation of mechanical properties for two-phase materials using a direct data-driven approach," *Materials & Design*, 2025.
9. (Muramatsu et al.), "Dislocation-based crystal plasticity simulation on grain-size dependence of mechanical properties in dual-phase steels," *International Journal of Solids and Structures*, 2025.
10. (Muramatsu et al.), "Ferroelastic phase transformation of LSCF based on phase-field model," Keio Multiphysics Materials Computation Research Group.

**手法基盤**
11. Carrara, P., Ambati, M., Alessi, R., De Lorenzis, L., "A framework to model the fatigue behavior of brittle materials based on a variational phase-field approach," *Comput. Methods Appl. Mech. Eng.*, 361, 112731, 2020.
12. Ching, J., Chen, Y.-C., "Transitional Markov Chain Monte Carlo Method for Bayesian Model Updating, Model Class Selection, and Model Averaging," *J. Eng. Mech.*, 133(7), 816–832, 2007.（TMCMC）
13. Li, Z. et al., "Fourier Neural Operator for Parametric Partial Differential Equations," *ICLR* 2021.（FNO）
14. Lu, L. et al., "Learning nonlinear operators via DeepONet," *Nat. Mach. Intell.*, 3, 218–229, 2021.（DeepONet）
15. Junker, P., et al., 拡張ハミルトン原理に関する論文（LUH 修論の理論基盤）[要確認：正確な書誌].

> [要確認] 4–10 の著者順・巻号・DOI は村松先生 researchmap / 各誌で確定すること。11–14 は標準文献だが頁・DOI を最終稿で付す。
