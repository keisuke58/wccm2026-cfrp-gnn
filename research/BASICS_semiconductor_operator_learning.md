# 基礎メモ: 半導体デバイスシミュレーション × 演算子学習

> 本リポジトリの FEM×演算子学習ライン（`pi_deeponet_fem_gaa.py`,
> `bench_weak_vs_strong.py`, `fe_newton_warmstart.py`）の土台となる概念整理。
> 「進む前の基本」を6段でまとめる。

---

## 1. デバイスの支配方程式（何を解いているか）

**(a) ポアソン方程式**（静電場）
$$-\nabla\cdot(\varepsilon \nabla\phi) = \rho,\qquad \rho = q(p - n + N_D^+ - N_A^-)$$
- φ: 電位、ε: 誘電率（Si≈11.7, SiO₂≈3.9）、ρ: 電荷密度。
- 電場 E = −∇φ がキャリアを動かす。

**(b) ドリフト拡散（電流連続）**
$$\nabla\cdot J_n = qR,\quad J_n = q\mu_n n E + qD_n\nabla n\quad(\text{正孔も同様})$$
- ドリフト（電場）＋拡散（濃度勾配）。

**自己整合ループ**: ρ→ポアソン解→E→キャリア移動→新ρ→… を収束まで反復
（Gummel / Newton）。**ポアソンを何度も解くのが律速** → サロゲート化の動機。

### 1b. ドリフト拡散(DD)モデルの全体像（標準3方程式）

実デバイス解析の標準は、上のポアソンに**電子・正孔の連続の式**を連立した3本立て:

- **① ポアソン**: $-\nabla\cdot(\varepsilon\nabla\phi)=q(p-n+N_D^+-N_A^-)$
- **② 電子 連続の式＋電流密度**:
  $$\nabla\cdot J_n = qR\ (\text{定常}),\qquad J_n = q\mu_n n\,E + qD_n\nabla n$$
- **③ 正孔 連続の式＋電流密度**（符号注意）:
  $$-\nabla\cdot J_p = qR\ (\text{定常}),\qquad J_p = q\mu_p p\,E - qD_p\nabla p$$

記号: $E=-\nabla\phi$（電場）, $\mu_{n,p}$（移動度）, $D_{n,p}$（拡散係数、アインシュタイン関係
$D=\mu k_BT/q$）, $R$（正味の生成・再結合率）。時間依存を残すなら
$\partial n/\partial t = \tfrac1q\nabla\cdot J_n - R$ の形。

**強い非線形性**: φを変えると $n,p$ が指数的に変わり($n\propto e^{\phi/V_t}$)、それが ρ を変えて φ に
返る。だから Gummel（分離反復）や Full Newton（全結合ヤコビアン）で反復する。

### 1c. 生成・再結合 R（補足方程式）

$R$ は定数でなく複数機構の和:
- **SRH**（Shockley–Read–Hall, 欠陥準位経由）:
  $R_{SRH}=\dfrac{np-n_i^2}{\tau_p(n+n_1)+\tau_n(p+p_1)}$
- **Auger**（高濃度で支配）: $R_{Aug}=(C_n n+C_p p)(np-n_i^2)$
- **衝突イオン化**（強電場, アバランシェ降伏）: 生成 $G=\alpha_n|J_n|/q+\alpha_p|J_p|/q$。

これらは $n,p,E$ の非線形関数 → 連立の非線形性をさらに強める。

### 1d. より高次のモデル（微細化対応）

- **エネルギー輸送 / 流体(hydrodynamic)モデル**: キャリア温度 $T_n,T_p$ のエネルギー平衡式を追加。
  ホットキャリア効果を高精度化（DD が仮定する局所平衡を超える）。
- **Poisson–Schrödinger 連立**: ゲート絶縁膜近傍の量子閉じ込め・トンネル電流（GAA/FinFET）を
  シュレーディンガー方程式と連立して解く。

> サロゲート化の狙いどころ: 上のどの層でも「**ポアソン（線形/非線形）解」が最内ループで最も
> 頻繁に呼ばれる**。ここを DeepONet / PI-DeepONet で高速化するのが現在のトレンド。本リポジトリは
> さらに **弱形式(FEM)** を核に据え、界面・非構造メッシュへの頑健性で差別化する。

---

## 2. 強形式 vs 弱形式（差別化の核心）

- **強形式**: 各点で $-\nabla\cdot(\varepsilon\nabla\phi)=\rho$。φに2階微分が必要。
  界面で ε が飛ぶと
  $\nabla\cdot(\varepsilon\nabla\phi)=\varepsilon\nabla^2\phi+\underbrace{\nabla\varepsilon\cdot\nabla\phi}_{\text{界面項・落としがち}}$
  の後者を扱えない。
- **弱形式**: 試験関数 v を掛けて部分積分
  $$\int \varepsilon\,\nabla\phi\cdot\nabla v = \int \rho\, v\quad(\forall v,\ v|_{\partial\Omega}=0)$$
  φは1階微分でよく、**界面フラックス連続 $[\varepsilon\,\partial\phi/\partial n]=0$ が自動的に成立**。

> 強形式＝点で成立、弱形式＝積分（平均）で成立。弱形式は要求が緩く界面・粗メッシュに頑健。
> `bench_weak_vs_strong.py` で実証: 強形式FDは界面フロアで頭打ち、弱形式FEMはO(h²)収束。

---

## 3. 離散化: FD と FEM

- **有限差分 FD**（強形式）: 微分を格子差分で近似。構造格子前提、実装簡単。
  元研究(Otsuki & Mori)のポアソンはこの系。素朴実装は界面項を落とす。
- **有限要素 FEM**（弱形式）: φを基底関数（P1）で展開し弱形式へ代入
  $$K\phi=b,\quad K_{ij}=\int\varepsilon\,\nabla\varphi_i\cdot\nabla\varphi_j,\quad b_i=\int\rho\,\varphi_i$$
  剛性 K を要素ごとに組立。任意の非構造メッシュ・界面OK（`assemble()`）。

---

## 4. サロゲート/演算子学習（普通のNN → PINN → DeepONet）

| | データ要求 | 新条件への汎化 | 物理 |
|---|---|---|---|
| 普通NNサロゲート | 多 | ○ | なし |
| PINN | 少 | ×（問題ごと再学習） | 強形式残差 |
| DeepONet | 中 | ◎（写像を学ぶ） | なし |
| **弱形式 PI-DeepONet** | 少 | ◎ | **弱形式残差** |

- **PINN**: NN が場 $\phi_\theta(x)$ を表現。損失＝PDE残差＋境界。メッシュフリー・少データだが1問題専用。
- **DeepONet**: 関数→関数の写像 $\rho\mapsto\phi$ を学習。
  $\phi(y)=\sum_k b_k(\rho)\,t_k(y)$（Branch=入力関数の符号化, Trunk=評価座標）。
  一度学習すれば**新しいρでも解き直さず前方推論**。
- **弱形式 PI**: 物理損失を点ごとラプラシアン（強形式）でなく **FE残差 $K\phi-M\rho$**（弱形式）に。

---

## 5. オンライン学習ループ（データ無し前提）

```
シミュ1ステップ → ネットでφ予測 → a-posteriori 誤差評価
  ├ 誤差小 → そのまま高速に進む
  └ 誤差大 → その場だけ厳密FE解 → その解で即追学習（リプレイ）
```
- **a-posteriori 誤差評価**: 解いた後に誤差を安く見積もる指標（FE残差等）。AMR細分化の判断にも使う古典FE道具。
- 適応が進むほど厳密解の起動が減る（デモ: 57%→23%）。

---

## 6. Newton / Gummel（非線形を解く反復）と「ウォームスタート」

非線形ポアソン（平衡, Poisson–Boltzmann）:
$$-\nabla\cdot(\varepsilon\nabla\phi) + \text{(非線形項 }\rho(\phi)) = 0$$
は φ について非線形 → **Newton 反復**で解く。各反復で線形系 $J\delta=-R$ を解く
（J=ヤコビアン, R=残差）。**収束の速さは初期値に強く依存**。

- **コールドスタート**: φ=0 から → 反復多い。
- **ウォームスタート（案C）**: 学習済み演算子ネットが初期値を与える → 反復少。
  **FEM が最終精度を保証**（ネットは加速のみ、置換でない）。オンラインで初期値モデルを更新。
- 産業評価が高い王道フレーム（「ML で TCAD 収束を安定化・高速化」）に、弱形式演算子の
  新規性を接続できる。`fe_newton_warmstart.py` で実証。

---

## コードとの対応マップ

| 概念 | ファイル / 関数 |
|---|---|
| ポアソン弱形式・FE組立 | `pi_deeponet_fem_gaa.py: assemble()` |
| 厳密FE解（オラクル） | `fe_solve()` |
| 弱形式物理損失 $K\phi-M\rho$ | `galerkin_residual()` |
| DeepONet（Branch=GNN, Trunk=座標） | `MeshGNNBranch`, `Trunk`, `GNNDeepONet` |
| a-posteriori誤差＋オンライン追学習 | `error_indicator()` ＋ online loop |
| 強形式FD vs 弱形式FEM 検証 | `bench_weak_vs_strong.py` |
| Newton ウォームスタート（非線形） | `fe_newton_warmstart.py` |

---

## 参考（分野の定番・現状 2026-07）

- Review: *A Comprehensive Review of ML Approaches for Semiconductor Device Modeling and Simulation* (2025).
- PINN 連立: *DDNet: Unified Physics-Informed DL for Semiconductor Device Modeling* (Poisson+DD).
- GNN/拡散: *PCGD: Physics-Guided Conditional Graph Diffusion for TCAD* (arXiv:2606.29272).
- 収束加速: ML による TCAD 収束安定化（ウォームスタート系）。
- 演算子学習: Lu et al. DeepONet (2021); Wang et al. PI-DeepONet (2021); PaRO-DeepONet (PIC, arXiv:2504.19065).
- 元研究: Otsuki & Mori, *Online Learning-Accelerated 3D MC for GAA*, WCCM-ECCOMAS 2026 STS415（会議発表段階）。
