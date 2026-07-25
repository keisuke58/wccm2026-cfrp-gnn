# 研究コンセプト: 半導体デバイスシミュレーションのための弱形式演算子学習＋オンライン適応

> **一行要旨**: 半導体デバイス解析の律速である**ポアソン/非線形ポアソン解**を、**弱形式(FEM)を
> 核とした演算子学習**で高速化する。強形式(有限差分/PINN)ベースの既存手法と**原理レベルで差別化**
> し、**FEM を精度の権威**に残したまま、**オンライン適応**で厳密解の起動と反復コストを削減する。
>
> 本ノートは散在する4デモ＋背景メモを1本に束ねた入口。各項目は該当コード/図/メモにリンク。

---

## 1. 問題設定と動機

現代デバイス(GAA/CFET)の TCAD は、ポアソン↔ドリフト拡散の自己整合ループ(Gummel/Newton)を
巨大メッシュ上で何度も回す。**最内で最も頻繁に呼ばれるのはポアソン解**であり、ここが律速。
→ これを機械学習サロゲートで高速化するのが分野のトレンド。
背景の基礎は [`BASICS_semiconductor_operator_learning.md`](./BASICS_semiconductor_operator_learning.md)。

**分野の定番(成熟→フロンティア)**: ①MLコンパクトモデル ②CNNグリッドPoissonサロゲート
③PINN連立(強形式) ④GNN/拡散(非構造) ⑤DeepONet/FNO演算子学習 ⑥ソルバ収束加速(ウォームスタート)
⑦MC輸送加速(=Otsuki & Mori) ⑧逆設計/オンライン学習。
→ **②〜④のほぼ全てが「強形式・格子/点ごと」。弱形式(FEM)演算子学習は空いている。**

---

## 2. 差別化の主張

出発点は Otsuki & Mori (WCCM-ECCOMAS 2026, STS415): **MC ループ内の有限差分(強形式)ポアソンを
PI-DeepONet で置換**（背景整理: [`PI_DEEPONET_MC_ONLINE_LEARNING.md`](./PI_DEEPONET_MC_ONLINE_LEARNING.md)）。
これをそのまま追わず、**FEM(弱形式・非構造メッシュ・a-posteriori 誤差評価)を核に据える**。
詳細設計: [`FEM_OPERATOR_LEARNING_GAA_IDEA.md`](./FEM_OPERATOR_LEARNING_GAA_IDEA.md)。

4つの構成要素:
- **A. 弱形式損失**: 物理損失を点ごとラプラシアン(強形式)でなく**組立済み FE 残差**(Galerkin)に。
  線形ポアソンなら `K(ε)φ−Mρ`、非線形ポアソン/Poisson–Boltzmann(③〜⑤⑩⑪)は φ 依存のキャリア項が入り
  残差は `K(ε)φ + κ·M·sinh(φ) − Mρ`(Dirichlet 境界で強制)。いずれも「組立済み FE 残差＝Galerkin」が核。
- **D. FE a-posteriori 誤差評価**: 残差指標が「メッシュ細分化」と「オンライン追学習トリガ」を統一。
- **B. GNN branch**: FE メッシュグラフを摂取し任意メッシュに対応(repo の mesh-agnostic GNN と地続き)。
- **C. FEM=精度の権威, ネット=加速器**: Newton の初期値だけを与えるウォームスタート(置換でなく加速)。

---

## 3. 実証（15デモ・正直な結果）

| # | デモ | 主張 | 実測結果 |
|---|---|---|---|
| ① | [`pi_deeponet_fem_gaa.py`](../pi_deeponet_fem_gaa.py) | A+D+B: 弱形式PI-DeepONet＋オンライン適応 | **トリガされた厳密FE解 24/60**(=デプロイ時コスト, 60%削減), トリガ率 **57%→23%**, rel-L2 **0.074** |
| ② | [`bench_weak_vs_strong.py`](../bench_weak_vs_strong.py) | 弱形式が界面で本質的に優れる(NN非介在) | 弱形式 **O(h²)収束**(0.13→0.0093) vs 強形式 **~0.55で頭打ち** |
| ③ | [`fe_newton_warmstart.py`](../fe_newton_warmstart.py) | C: 1D 非線形ポアソンのウォームスタート | Newton反復 cold **5.2** → warm **3.7**(適応で3.5) |
| ④ | [`dd2d_newton_warmstart.py`](../dd2d_newton_warmstart.py) | C: 2D GAAスライスへ昇格 | Newton反復 cold **5.2** → warm **3.1**(~40%削減) |
| ⑤ | [`cfet_stack_warmstart.py`](../cfet_stack_warmstart.py) | C: CFET 積層断面（多材料・n/p縦積み） | Newton反復 cold **4.8** → warm **3.0**(~38%削減) |
| ⑥ | [`tsv_thermal_stress.py`](../tsv_thermal_stress.py) | TSV 熱機械応力(2D): FE熱弾性＋応力場サロゲート＋KOZ（repo の CFRP応力×GNN と直結） | 未知レイアウトで rel-L2 **0.128**, keep-out zone **IoU 0.819** |
| ⑦ | [`tsv_3d_stress.py`](../tsv_3d_stress.py) | TSV 3D＋**実弾性定数**（Si異方性 C11/C12/C44, Cu等方）: 3Dテトラ熱弾性 | 異方性Siで応力が**4回対称(cloverleaf)**→**方向依存KOZ**（等方近似では出ない）。**解析解2検証を機械精度でPASS**（自由膨張→無応力, 完全拘束→−(C11+2C12)αΔT） |
| ⑧ | [`dd_full_1d.py`](../dd_full_1d.py) | **完全ドリフト拡散**（1D pnダイオード, Scharfetter–Gummel＋Gummel）＋バイアス継続ウォームスタート | **理想ダイオードI-V検証**（V=0でJ≈1e-11, 順バイアスで J∝(e^{V/Vt}−1)）。掃引の Gummel 反復 cold 203 → warm 184（~10%削減, 正直に控えめ＝反復は注入水準支配） |
| ⑨ | [`dd_breakdown_continuation.py`](../dd_breakdown_continuation.py) | 高逆バイアス coupled DD: **継続法が"必須"**（cold は基底を外れて発散） | cold（平衡から直接）は **4Vt** までしか収束せず発散、**継続法は 45Vt 到達**。案C が「速い」でなく「収束/発散を分ける」領域を実証（衝突イオン化は安定重視で mild ~x2.1, 真のavalanche/Full-Newton連立は要スケーリングの重い拡張として明記） |
| ⑩ | [`gaa_material_sweep_warmstart.py`](../gaa_material_sweep_warmstart.py) | C: **多材料 GAA スイープの償却**（円筒断面で Si/Ge/GeSn/InGaAs/MoS₂ ×バイアス掃引 = 実TCADワークロード, cf. Balaji+2026） | **45 自己整合FE解**の総 Newton 反復 cold **217 → warm 161（26%削減）**。バイアス継続＋**材料転写**（同一形状で ε/遮蔽のみ変更）で償却。円筒（曲面）断面ゆえ**弱形式FEの必然性**とも接続。材料依存電荷（小ギャップ Ge/GeSn ほど強遮蔽）も再現 |
| ⑪ | [`gaa_operator_deeponet.py`](../gaa_operator_deeponet.py) | **A+B+C 統合**: ⑩を**学習演算子化**（DeepONet: (ε,κ,V_g)×(x,y)→u）。1発推論＋warm-start | **未知材料**(InGaAs, 学習に不使用)へ 1-shot **rel-L2 0.015**、**未知バイアス** 0.013（デプロイ時 Newton 不要）。予測を warm-start にすると厳密FE Newton が cold **5.2 → 2.8**（約半減, tol=1e-9ゆえ1反復には潰れない）。正規化統計は学習材料のみで算出（未知材料の漏洩なし）。弱形式FEデータ(A)＋演算子学習(B)＋加速(C)を実ワークロードで一体化 |
| ⑫ | [`gaa_wfm_vth.py`](../gaa_wfm_vth.py) | **WFM→Vth 静電**: 仕事関数金属（Endura-3 相当）で閾値電圧を調整。GAA 断面の非線形ポアソン | 5 種 WFM（Φ_m 4.2–5.0 eV）で Q–V_g を FE 求解→Vth 抽出。**ΔVth = ΔΦ_WFM（slope 1.00 V/eV）**を GAA 断面上で再現、Vth 設計窓 0.04–0.84 V。製造の WFM 成膜工程を計算可能な Vth に写像（理想フラットバンド模型・散乱/トラップ無し, スケールは例示）。パイロットライン対応: [`SEMICON_PILOT_LINE_GAA_PROCESS.md`](./SEMICON_PILOT_LINE_GAA_PROCESS.md) |
| ⑬ | [`phasefield_fracture_warmstart.py`](../phasefield_fracture_warmstart.py) | **相場破壊(phase-field)の warm-start**（村松研＝破壊連成 への橋渡し, テーマAの種） | AT2 相場破壊 SENT（変位↔損傷の交互最小化, P1弱形式）＋荷重継続。**総 staggered 反復 cold 1046 → warm 532（49%削減）**。脆性伝播ステップは両者スパイク（staggered の既知の遅さ＝演算子学習加速の動機）、伝播後は warm ~3 vs cold ~30–70。荷重変位は線形→ピーク→脆性軟化を再現。テーマ提案: [`RESEARCH_THEMES_muramatsu.md`](./RESEARCH_THEMES_muramatsu.md) |
| ⑭ | [`tsv_interface_fracture.py`](../tsv_interface_fracture.py) | **TSV Cu/Si 界面剥離**（熱応力駆動 phase-field, テーマBの種＝⑥⑦×⑬融合） | Cu via/Si の CTE ミスマッチ＋弱界面で、熱負荷継続により**界面リング状の損傷＝剥離**を再現。貯蔵エネルギーが剥離開始で急落。総 staggered 反復 cold 231 → warm 187（19%削減, 開始ステップが律速）。Post-5G 後工程(3D実装)信頼性に直結。熱スケールは例示（CTE比 Cu:Si は物理的） |
| ⑮ | [`tsv_layout_gnn.py`](../tsv_layout_gnn.py) | **B の本丸**: レイアウト→剥離信頼性を **GNN で予測**（なぜGNN の定量裏付け, repo の CFRP×GNN 中核と同一枠） | 多 via レイアウトの**per-via 剥離リスク**を FE(線形熱弾性＋核形成基準)で正解化し、**距離をエッジ特徴に持つ MPNN**で予測。**test R² 0.60**。同容量の**構造無視 MLP（自ノード特徴のみ）は R² −1.95**＝relational gain **+2.55**。リスクは近傍相互作用支配ゆえ GNN が必須（線形近似・粗メッシュゆえ絶対値は控えめ, 種として） |

図: `pi_deeponet_fem_gaa.png`, `bench_weak_vs_strong.png`, `fe_newton_warmstart.png`, `dd2d_newton_warmstart.png`, `cfet_stack_warmstart.png`, `tsv_thermal_stress.png`, `tsv_3d_stress.png`, `dd_full_1d.png`, `dd_breakdown_continuation.png`, `gaa_material_sweep_warmstart.png`, `gaa_operator_deeponet.png`, `gaa_wfm_vth.png`, `phasefield_fracture_warmstart.png`, `tsv_interface_fracture.png`, `tsv_layout_gnn.png`。
> ①〜⑤⑩⑪⑫は半導体デバイス（電気物理; ①〜⑤⑩⑪⑫は平衡ポアソン、⑧⑨は非平衡・輸送の完全DD）、⑥⑦は
> 3D積層/パッケージ（後工程・熱機械）で、本リポジトリの CFRP 応力 FEA×GNN 中核に最も近い橋渡し。
> 依存関係（実 import 準拠）: ②④⑤⑥は `pi_deeponet_fem_gaa.py` の FE 資産（`build_mesh`/`assemble`/`eps_map`）を、
> ⑪⑫は `gaa_material_sweep_warmstart.py`（メッシュ/組立/Newton）を再利用。①③⑦⑧⑨⑩は単体で自己完結。
各デモは CPU で数分・seed 固定で再現可能。

**物語の流れ**: ①で「弱形式演算子学習＋オンライン適応」を提示 → ②で「なぜ弱形式か」を離散化
レベルで裏付け(新規性の一枚看板) → ③④で「FEMを権威に残す加速(案C)」を1D→2Dで実証。

---

## 4. 分野内での位置づけ（何が新しいか）

（以下、丸数字のうち §1 の**分類**は「定番⑥」等、§3 の**デモ**は「デモ②」等と明示して区別する。）

- **弱形式 × 演算子学習**: 定番②CNN格子/③PINN/④GNN がいずれも強形式・格子/点ごとなのに対し、
  **界面不連続 ε に頑健な弱形式**を核にした点（**デモ②**が定量的裏付け。⑩は円筒＝曲面境界にも適用）。
  真の非構造メッシュ＋AMR への一般化は §6 ロードマップの将来課題（本 PR では未実証）。
- **オンライン適応 × a-posteriori 誤差**: FE 誤差指標が細分化とオンライン追学習を統一（**デモ①**）。
- **ソルバ加速フレーム（定番⑥ウォームスタート）への接続**: 「ML で TCAD 収束を加速」の受けの良い
  王道に、弱形式演算子の新規性を接続（**デモ③④⑤**＝Newton warm-start、**⑩**＝材料/バイアス継続、
  **⑪**＝学習演算子 warm-start、案C）。**精度は FEM が保証**し置換しない、という安全な主張。

---

## 5. 正直な限界

- すべて**概念実証**。**1D の完全DD**（連立電流連続; Gummel＋Scharfetter–Gummel）は⑧⑨で実装済み。
  未実装は **2D/3D の連立半導体DD**（Full Newton・要変数スケーリング）、量子補正(Poisson–Schrödinger)、
  散乱。なお 3D は**熱弾性(⑦)は実装済み**であり、「3D 未実装」は半導体DDに限る（熱機械は該当せず）。
- ①の誤差指標は素の相対 Galerkin 残差(K が微分作用素ゆえ解誤差より過大)。energy-norm/CG 推定へ
  置換可能。
- 案C(③④): 減衰 Newton は sinh 単調性ゆえ条件が良く、warm 反復は下限(~3)付近に張り付くため
  **オンライン適応の追加ゲインは小さい**。本質価値は「良い初期値で反復(=高コストな疎行列解)を
  ~40%削減しつつ精度不変」。3D/連立DD/大バイアスほど削減幅は拡大する見込み。

---

## 6. ロードマップ（次の軸）

1. ~~**CFET 積層断面への拡張**~~ → **⑤で実装済み**（多材料スタック上の Newton ウォームスタート,
   cold 4.8→warm 3.0）。ロードマップ FinFET→GAA→Forksheet→**CFET**→2D-CFET→3Dモノリシック
   の CFET 段に対応。次は真の縦ゲート形状＋非構造メッシュ、2D-CFET(2D材料チャネル)。
   ~~**多材料チャネル比較の償却**~~ → **⑩で実装済み**（円筒 GAA 断面で Si/Ge/GeSn/InGaAs/MoS₂
   ×バイアス掃引 = 実TCADワークロード [Balaji+2026] を継続法で償却, 45解の総反復 217→161）。
   ~~**材料をパラメータ入力にした演算子学習で 1発推論化**~~ → **⑪で実装済み**（DeepONet で未知材料
   InGaAs へ 1-shot rel-L2 0.014、warm-start で Newton 半減）。次は非対称ドーピング/角度依存で
   trunk の表現力を要する場へ、および真の 2D 非構造メッシュでの GNN branch。
2. ~~**完全ドリフト拡散**: Scharfetter–Gummel の電流連続式を連立~~ → **⑧(順バイアス理想ダイオード)・
   ⑨(高逆バイアス継続法必須)で実装済み**。次は **真の avalanche**（陰的/減衰 G）と **2D の Full Newton
   連立**（要変数スケーリング; 生の連立DDヤコビアンは悪条件）——⑨で必要性は実証済み。
3. **真の非構造メッシュ + AMR**: η 駆動の h-細分化と GNN branch のメッシュ非依存性の実証。
4. ~~**TSV/3D積層の熱機械応力(FEA×GNN)**~~ → **⑥(2D)・⑦(3D＋実弾性定数)で実装済み**
   （⑥: CNN応力場サロゲート＋KOZ rel-L2 0.128/IoU 0.819。⑦: 3Dテトラ熱弾性、異方性Siで
   4回対称=方向依存KOZ）。次は 3D 応力場のサロゲート化・非構造メッシュ、および repo の CFRP
   欠陥localization GNN と同一パイプラインでの KOZ 分類。

---

## 7. 関連研究・一次情報

**演算子学習・弱形式**: Lu et al. DeepONet (*Nat. Mach. Intell.* 2021); Wang et al. PI-DeepONet
(*Sci. Adv.* 2021); Patel et al. VarMiON (弱形式/変分演算子)。
**隣接（PIC ポアソン）**: Kim et al. PaRO-DeepONet (arXiv:2504.19065, 粒子情報つき縮約演算子＝PIC 用で
弱形式FEM半導体研究ではない。関連手法として併記)。
**分野レビュー/定番**: *A Comprehensive Review of ML for Semiconductor Device Modeling* (2025);
DDNet (PINN連立); PCGD (arXiv:2606.29272, 物理ガイド拡散×TCAD); ML による TCAD 収束加速。
**元研究**: Otsuki & Mori, *Online Learning-Accelerated 3D MC for GAA*, WCCM-ECCOMAS 2026 STS415
（会議発表段階、ジャーナル/プレプリント索引は未確認）。
**適用先ワークロード(⑩の動機)**: S. Balaji, T.S. Balaji, P. Rathinakumar, S. Karthik,
*Comparative TCAD investigation of gate-all-around nanowire MOSFETs with emerging channel materials*,
**Next Materials 13 (2026) 102743** (doi:10.1016/j.nxmate.2026.102743, オープンアクセス)。Synopsys
Sentaurus で GAA ナノワイヤの Si/Ge/GeSn/InGaAs/MoS₂ を同一形状・同一ゲートスタックで比較（ML 非使用）。
＝「材料×バイアスの独立自己整合ソルブ多数」という、案C の継続法が償却する典型ワークロードの実例（⑩）。
**GAA/CFET デバイス(動機)**: 2D-Bi₂O₂Se GAA (*Nat. Mater.* 2025, s41563-025-02117-w);
2D トランジスタ ヒステリシス標準化 (*Nat. Commun.* 2025, s41467-025-65641-y);
Samsung 3D Stacked FET (VLSI 2026); CFET ロードマップ (IMEC 系)。
