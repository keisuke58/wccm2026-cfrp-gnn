# PI-DeepONet × 3D Monte Carlo のオンライン学習加速 — 技術ノート

> 対象研究: **Online Learning-Accelerated 3D Monte Carlo Simulation for Gate-All-Around Transistors**
> 著者: 大槻 祐介 (Yusuke Otsuki), 森 伸也 (Nobuya Mori, 大阪大学 大学院工学研究科)
> 発表: WCCM-ECCOMAS 2026 (ミュンヘン), セッション STS415, 2026-07-22
>
> 本ノートは上記発表の**公開情報からの技術的整理（背景）**。本リポジトリ (CFRP × GNN) の
> 直接の成果物ではなく、演算子学習 (operator learning) + 物理損失 + オンライン適応という
> 方法論の背景整理として置く。
>
> ⚠️ **差別化について**: 元研究をそのまま追うのは避け、FEM（弱形式・非構造メッシュ・
> a-posteriori 誤差評価）を中核に据えた**差別化アイデアと動作デモ**を別途用意した
> → [`FEM_OPERATOR_LEARNING_GAA_IDEA.md`](./FEM_OPERATOR_LEARNING_GAA_IDEA.md) /
> コード [`pi_deeponet_fem_gaa.py`](../pi_deeponet_fem_gaa.py)。本ノートはその**前提知識**。

---

## 1. DeepONet と PI-DeepONet の区別

| | **Vanilla DeepONet** | **PI-DeepONet** |
|---|---|---|
| 学習信号 | 入出力ペア（データ）の MSE のみ | データ MSE **＋ PDE 残差 ＋ 境界/初期条件残差** |
| データ要求 | 大量の教師データ（数値解・実験） | 少量（あるいはゼロ）で可 |
| 汎化 | 学習分布内に強い | 物理法則で拘束されるため外挿に強い |
| 計算 | 前方推論のみ | 学習時に自動微分で高階微分を評価 |
| 出典 | Lu et al., *Nat. Mach. Intell.* 3, 218 (2021) | Wang, Wang & Perdikaris, *Sci. Adv.* 7, eabi8605 (2021) |

**要点**: 「DeepONet = PINN」ではない。PINN の物理損失を DeepONet に融合したのが
PI-DeepONet。本研究で「少データ・物理整合・外挿頑健」を担うのはこの PI 版である。

### アーキテクチャ（共通）

```
u(x) を m 個の固定センサ点で離散化 ─▶ [ Branch Net ]  b ∈ R^p
                                                    │  ┐
                                                    │  ├─ 内積  φ(y) = Σ_k b_k t_k(y) (+ b0)
評価座標 y ────────────────────────▶ [ Trunk Net ]  t ∈ R^p  ┘
```

- **Branch**: 入力**関数**（電荷密度分布 ρ、バイアス電圧 V_gs/V_ds 等）を受ける。
- **Trunk**: 出力を評価したい**連続座標** (x,y,z) を受ける。
- 出力 = 両者の内積 → 任意座標でメッシュ非依存に φ を評価可能。

CNN 等の固定メッシュ画像モデルと違い、Trunk が連続座標を取るため
**メッシュを切り直さずに任意点で電場を評価**できるのが演算子学習の利点。

---

## 2. 物理的背景 — なぜ MC のボトルネックがポアソン方程式か

GAA (Gate-All-Around) など極微細デバイスでは、キャリアの熱的非平衡・散乱を捉えるため
**アンサンブル 3D Monte Carlo (MC)** が最高精度の輸送計算手段となる。MC の各時間ステップは
次のセルフコンシステント・ループを繰り返す:

1. 電子の自由飛行・散乱で位置を更新
2. 更新後の電荷分布 ρ から**ポアソン方程式**を解いて電位 φ を再計算
   ```
   -∇·(ε ∇φ) = ρ
   ```
3. E = -∇φ から次ステップの力を得る

**ステップ2 が律速**。メッシュ生成と大規模疎行列（連立一次方程式）の求解を毎ステップ
伴い、全体計算時間の大部分を占める。ここをサロゲート化するのが本研究の核心。

---

## 3. PI-DeepONet による代替と役割

ポアソンソルバを PI-DeepONet で置換する:

- **入力（Branch）**: 電荷密度 ρ(x,y,z)（＋バイアス条件）
- **入力（Trunk）**: 評価座標 (x,y,z)
- **出力**: 電位 φ（→ 微分して電場 E）
- **物理損失**: ポアソン残差 ‖-∇·(ε∇φ) - ρ‖² と境界条件残差を自動微分で評価し、
  Loss に加算 → 少データでも物理整合な解を出す。

演算子学習なので、**一度学習すれば ρ が変わるたびに行列を解き直さず前方推論だけ**で
φ を得られる。PIC / プラズマ分野の先行例（PaRO-DeepONet 等）と同じ発想。

---

## 4. オンライン学習による自動適応

事前に全動作条件の教師データを網羅するのは非現実的。本研究は
**MC を走らせながらリアルタイムに PI-DeepONet を育てる**:

```
[ MC 1ステップ実行 → 新しい ρ ]
          │
          ▼
  PI-DeepONet で φ を瞬時に予測
          │
          ├─ 残差指標が小さい ──▶ 予測をそのまま採用（高速パス）
          │
          └─ 未知領域で残差が大 ──▶ その ρ に限りポアソンを厳密求解（オラクル）
                                        │
                                        ▼
                               その解を教師にオンザフライ追学習
```

**利点**:
- 事前の大規模データ収集が不要（コールドスタート可）。
- 未知バイアス (V_gs, V_ds) やデバイス形状に対し、走行中に自律適応。
- 厳密ソルバの起動頻度を激減 → MC の精度を保ったまま総計算コストを大幅削減。

適応が進むほど「厳密求解トリガ率」が下がるのが定量的な成功指標になる
（デモ実装で再現している主要指標）。

---

## 5. 差別化した動作デモ（本リポジトリの実装）

元研究の「MC + 有限差分（強形式）」をそのまま模倣する代わりに、**FEM 弱形式 + FE 誤差評価 +
GNN branch** で差別化したデモを実装した。詳細と実測結果は
→ [`FEM_OPERATOR_LEARNING_GAA_IDEA.md`](./FEM_OPERATOR_LEARNING_GAA_IDEA.md)、
コードは [`pi_deeponet_fem_gaa.py`](../pi_deeponet_fem_gaa.py)。

要点だけ再掲:
- 物理損失は点ごとのラプラシアン（強形式）ではなく**組立済み FE 残差 K(ε)φ − Mρ**（弱形式）。
- 可変誘電率 ε(x)（GAA の Si コア/酸化膜リング）を要素剛性に反映。
- Branch は FE メッシュグラフ上の GNN（センサ節点読み出し）、Trunk は座標。
- FE 誤差指標が閾値超過のステップだけ厳密 FE 解＋オンライン追学習（リプレイ）。
- 既定実行で **厳密解 24/60 ステップ・トリガ率 57%→23%・rel-L2 0.074**。

```
python3 pi_deeponet_fem_gaa.py            # 学習＋オンラインデモ＋作図（pi_deeponet_fem_gaa.png）
python3 pi_deeponet_fem_gaa.py --help     # n, p, tol, steps, adapt_iters, loops ...
```

---

## 6. 出版状況（2026-07-22 時点の調査）

- WCCM-ECCOMAS 2026 の会議発表（STS415, 2026-07-22）として確認できる。
- **ジャーナル論文・arXiv プレプリントとして公開索引されたものは見つからなかった**
  （Google/学術検索、arXiv、著者所属の森研究室ページ等を確認。会議サイトの詳細
  プログラムは 403 で直接取得不可）。
- したがって現時点では **会議発表段階**。査読付き論文化は今後の可能性として扱うのが妥当。

参考（分野の先行研究・一次情報）:
- Lu et al., "Learning nonlinear operators…(DeepONet)", *Nat. Mach. Intell.* 3, 218 (2021).
- Wang, Wang, Perdikaris, "Learning the solution operator… (PI-DeepONet)", *Sci. Adv.* 7 (2021).
- Kim et al., "PaRO-DeepONet: particle-informed reduced-order DeepONet for Poisson solver in PIC", arXiv:2504.19065.
- 森研究室（大阪大学）: http://www.si.eei.eng.osaka-u.ac.jp/
