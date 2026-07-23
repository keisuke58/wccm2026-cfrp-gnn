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
- **A. 弱形式損失**: 物理損失を点ごとラプラシアン(強形式)でなく**組立済み FE 残差 K(ε)φ−Mρ**(Galerkin)に。
- **D. FE a-posteriori 誤差評価**: 残差指標が「メッシュ細分化」と「オンライン追学習トリガ」を統一。
- **B. GNN branch**: FE メッシュグラフを摂取し任意メッシュに対応(repo の mesh-agnostic GNN と地続き)。
- **C. FEM=精度の権威, ネット=加速器**: Newton の初期値だけを与えるウォームスタート(置換でなく加速)。

---

## 3. 実証（5デモ・正直な結果）

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

図: `pi_deeponet_fem_gaa.png`, `bench_weak_vs_strong.png`, `fe_newton_warmstart.png`, `dd2d_newton_warmstart.png`, `cfet_stack_warmstart.png`, `tsv_thermal_stress.png`。
> ①〜⑤は半導体デバイス（前工程・電気物理）、⑥は 3D積層/パッケージ（後工程・熱機械）で、本リポジトリの
> CFRP 応力 FEA×GNN 中核に最も近い橋渡し。
全デモは自己完結・CPUで数分・seed固定で再現可能。

**物語の流れ**: ①で「弱形式演算子学習＋オンライン適応」を提示 → ②で「なぜ弱形式か」を離散化
レベルで裏付け(新規性の一枚看板) → ③④で「FEMを権威に残す加速(案C)」を1D→2Dで実証。

---

## 4. 分野内での位置づけ（何が新しいか）

- **弱形式 × 演算子学習**: ②〜④の定番が強形式/格子なのに対し、界面不連続 ε・非構造メッシュに
  頑健な**弱形式**を核にした点(②が定量的裏付け)。
- **オンライン適応 × a-posteriori 誤差**: FE 誤差指標が細分化とオンライン追学習を統一(①)。
- **加速フレーム(⑥)への接続**: 「ML で TCAD 収束を加速」の受けの良い王道に、弱形式演算子の
  新規性を接続(③④、案C)。**精度は FEM が保証**し置換しない、という安全な主張。

---

## 5. 正直な限界

- すべて**概念実証**: 3D・量子補正(Poisson–Schrödinger)・散乱・**連立電流連続(完全DD)**は未実装。
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
2. ~~**完全ドリフト拡散**: Scharfetter–Gummel の電流連続式を連立~~ → **⑧で実装済み**（1D pnダイオード,
   理想ダイオードI-V検証済み）。次は 2D DD・Full Newton 連立・降伏近傍（cold発散→継続法が必須になる領域）。
3. **真の非構造メッシュ + AMR**: η 駆動の h-細分化と GNN branch のメッシュ非依存性の実証。
4. ~~**TSV/3D積層の熱機械応力(FEA×GNN)**~~ → **⑥(2D)・⑦(3D＋実弾性定数)で実装済み**
   （⑥: CNN応力場サロゲート＋KOZ rel-L2 0.128/IoU 0.819。⑦: 3Dテトラ熱弾性、異方性Siで
   4回対称=方向依存KOZ）。次は 3D 応力場のサロゲート化・非構造メッシュ、および repo の CFRP
   欠陥localization GNN と同一パイプラインでの KOZ 分類。

---

## 7. 関連研究・一次情報

**演算子学習・弱形式**: Lu et al. DeepONet (*Nat. Mach. Intell.* 2021); Wang et al. PI-DeepONet
(*Sci. Adv.* 2021); Patel et al. VarMiON (弱形式演算子); Kim et al. PaRO-DeepONet (arXiv:2504.19065)。
**分野レビュー/定番**: *A Comprehensive Review of ML for Semiconductor Device Modeling* (2025);
DDNet (PINN連立); PCGD (arXiv:2606.29272, 物理ガイド拡散×TCAD); ML による TCAD 収束加速。
**元研究**: Otsuki & Mori, *Online Learning-Accelerated 3D MC for GAA*, WCCM-ECCOMAS 2026 STS415
（会議発表段階、ジャーナル/プレプリント索引は未確認）。
**GAA/CFET デバイス(動機)**: 2D-Bi₂O₂Se GAA (*Nat. Mater.* 2025, s41563-025-02117-w);
2D トランジスタ ヒステリシス標準化 (*Nat. Commun.* 2025, s41467-025-65641-y);
Samsung 3D Stacked FET (VLSI 2026); CFET ロードマップ (IMEC 系)。
