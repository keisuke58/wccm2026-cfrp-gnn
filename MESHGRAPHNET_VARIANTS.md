# PhysicsNeMo MeshGraphNet 系モデルの違いメモ

このメモは、次の import で出てくる PhysicsNeMo の MeshGraphNet 系モデルの違いと、CFRP メッシュ上の欠陥・応力場予測での使い分けをまとめたものです。

```python
from physicsnemo.models.meshgraphnet import (
    MeshGraphNet,
    HybridMeshGraphNet,
    MeshGraphKAN,
    BiStrideMeshGraphNet,
)
```

## まず結論

今回のような **固定メッシュ上の CFRP 欠陥・応力場予測** では、まずは **標準の `MeshGraphNet` を本命のまま使う** のがよいです。

理由は、`MeshGraphNet` [1] がメッシュ物理問題に合いやすい **Encode-Process-Decode** 構造を持ち、ノード特徴だけでなく **エッジ特徴も message passing の各ステップで更新する** ためです。

このリポジトリの `train.py` でも、`MeshGraphNetModel` は以下の構造になっています。

- node encoder
- edge encoder
- 複数個の `_GraphNetBlock`
- decoder
- processor block 内で edge と node を残差更新
- edge feature として座標差分 `[dx, dy, dz, dist]` を利用

そのため、GAT/PNA などよりも、メッシュ上の局所幾何と応力伝播を表現しやすい可能性があります。

## 4モデルの使い分け

| モデル | 位置づけ | 向いている場面 | 今回の優先度 |
|---|---|---|---:|
| `MeshGraphNet` | 標準版 | 固定メッシュ、FEM/CAE メッシュ、応力場、変位場、ノード分類 | 1 |
| `BiStrideMeshGraphNet` | マルチスケール版 | 長距離依存、広域応力伝播、高解像度メッシュ | 2 |
| `MeshGraphNet-Transformer` | Transformer global processor 版 | 高解像度メッシュ、長距離相互作用、標準 MGN の under-reaching が疑われる場合 | 2.5 |
| `MeshGraphKAN` | KAN を使った高表現力版 | 入力と出力の非線形関係が強い、標準 MGN が頭打ち | 3 |
| `HybridMeshGraphNet` | mesh edge + world edge 版 | 大変形、接触、衝突、現在位置で近い点同士の相互作用 | 4 |

## 1. MeshGraphNet

`MeshGraphNet` は標準の MeshGraphNet です。

基本構造は次の流れです。

```text
node/edge features
    ↓
Encoder
    ↓
Processor: message passing を複数回
    ↓
Decoder
    ↓
node ごとの予測
```

このリポジトリの実装では、`_GraphNetBlock` の中で次の順に更新しています。

```text
edge update: [edge, src node, dst node] から新しい edge 表現を作る
    ↓
edge aggregation: dst node ごとに edge を集約
    ↓
node update: [node, aggregated edge] から新しい node 表現を作る
```

重要なのは、**ノードだけでなくエッジ表現も毎ステップ更新する**ことです。

### 今回良かった理由の推測

CFRP のメッシュデータでは、各ノードの値だけでなく、隣接ノードとの位置関係・距離・方向が重要になりやすいです。

`MeshGraphNet` は edge feature として座標差分 `[dx, dy, dz, dist]` を使えるため、次のような情報を自然に学習できます。

- 隣接点との距離
- 方向依存性
- 局所的な応力集中
- メッシュ上の伝播
- 左右非対称性
- 欠陥周辺の局所パターン

したがって、今回 `meshgraphnet` の精度が良かったのは自然です。

## 2. HybridMeshGraphNet

`HybridMeshGraphNet` は、通常の mesh edge に加えて **world edge** を扱う版です。

| edge | 意味 |
|---|---|
| mesh edge | メッシュトポロジー上の接続 |
| world edge | 現在の空間位置で近いノード同士の接続 |

### 向いている問題

- 大変形
- 接触
- 衝突
- 破壊
- cloth / shell / impact 系
- メッシュ上では離れているが、実空間では近づく点同士が相互作用する問題

### 今回の判断

固定メッシュ上で、形状が大きく動いたり接触したりしない場合は、`HybridMeshGraphNet` の優先度は低いです。

world edge を作るコストと実装複雑性が増える一方で、今回のような固定メッシュ問題では追加効果が小さい可能性があります。

## 3. MeshGraphKAN

`MeshGraphKAN` は、標準の MeshGraphNet に KAN 系の表現力を入れた実験的な派生版です。

KAN は Kolmogorov-Arnold Network 系の考え方 [3] で、MLP よりも複雑な非線形関係を表現しやすい場合があります。PhysicsNeMo の `MeshGraphKAN` は Fourier KAN 系の node encoder 置換として説明されているため、Fourier KAN の関連文献 [4] も背景として引用候補になります。

### 向いている問題

- 入力特徴と出力の関係が強く非線形
- 標準 MGN がある程度良いが頭打ち
- hidden size や processor blocks を増やしても改善しにくい
- 計算コスト増を許容できる

### 今回の判断

試す価値はありますが、最初にやるべきではありません。

まずは標準 `MeshGraphNet` の以下を詰めるのが先です。

- `mgn_blocks`
- hidden size
- dropout
- edge dropout
- class weight / focal loss
- data split のリーク確認

その後で、まだ改善余地がありそうなら `MeshGraphKAN` を比較対象にします。

## 3.5. MeshGraphNet-Transformer (MGN-T)

`MeshGraphNet-Transformer` (MGN-T) は、標準 `MeshGraphNet` のメッシュ表現と幾何的 inductive bias を保ちながら、processor を Transformer 系の global processor に置き換える新しい候補です [6]。

MGN-T の狙いは、標準 MGN の弱点である **大規模・高解像度メッシュでの長距離情報伝播の遅さ** を改善することです。標準 MGN では、遠い節点の情報を取り込むには message-passing block を深くする必要があります。一方で MGN-T は physics-attention Transformer により全ノード状態を同時に更新し、深い message-passing stack や階層 coarsening に頼らず長距離相互作用を直接扱うことを目指します [6]。

### 向いている問題

- 高解像度メッシュ
- 長距離依存が強い固体力学問題
- impact dynamics
- self-contact
- plasticity
- boundary condition や geometry/topology がサンプルごとに変わる問題
- 標準 MGN で processor blocks を増やしても遠方影響を拾い切れない場合

### 今回の判断

これは **挑戦する価値があります**。特に、標準 `MeshGraphNet` が良い結果を出しているが、次のような傾向があるなら MGN-T は有力な次候補です。

- `mgn_blocks` を増やすと少し良くなるが、計算が重い
- 欠陥から離れた領域や境界条件の影響を外す
- 局所的な欠陥周辺は合うが、試験片全体の分布がズレる
- BiStride の multi-scale graph を作る実装コストを避けたい

ただし、2026年1月投稿・2026年2月改訂の比較的新しい arXiv 論文なので、WCCM で使う場合は **標準 MeshGraphNet [1] を主結果、MGN-T [6] を発展的比較または今後の挑戦** として位置づけるのが安全です。いきなり主張の中心にするより、まずは同じ data split・同じ入力特徴・同じ評価指標で ablation するのがよいです。

### 実装するときの注意

- 全ノード attention はメモリ消費が大きくなりやすい
- 14k nodes 規模では full attention が重い可能性がある
- attention を全ノードで行うのか、近傍・ブロック・低ランク近似にするのかを確認する
- edge attributes を Transformer processor 内でどう保持するかが重要
- 既存の `MeshGraphNetModel` とは processor 部分が大きく変わるため、まず小さい hidden size / 少ない層で smoke test する

### 優先順位

現時点では、標準 `MeshGraphNet` の次に試す候補として `BiStrideMeshGraphNet` と並ぶ位置づけです。

```text
MeshGraphNet
    ↓
BiStrideMeshGraphNet or MeshGraphNet-Transformer
    ↓
MeshGraphKAN
    ↓
HybridMeshGraphNet
```

判断の目安は次の通りです。

- multi-scale graph を作れるなら: `BiStrideMeshGraphNet`
- global attention 実装・GPU memory に余裕があるなら: `MeshGraphNet-Transformer`
- 非線形表現力を上げたいなら: `MeshGraphKAN`


## 4. BiStrideMeshGraphNet

`BiStrideMeshGraphNet` は、MeshGraphNet に **マルチスケール message passing** を入れた版です。PhysicsNeMo の実装は BSMS-GNN [2] に基づくと説明されています。

通常の `MeshGraphNet` は processor block の数だけ hop 方向に情報が伝わります。例えば 10 blocks なら、おおまかには 10-hop 程度の情報伝播になります。

一方で `BiStrideMeshGraphNet` は、粗いグラフ階層を使って、遠くの情報をより効率よく伝えます。

### 向いている問題

- 長距離依存が強い
- 広域の応力伝播が重要
- 境界条件の影響が遠くまで効く
- 高解像度メッシュで、通常 MGN を深くしすぎると重い
- 局所だけでは欠陥クラスを判定しにくい

### 今回の判断

標準 `MeshGraphNet` の次に試すなら、`BiStrideMeshGraphNet` が有力です。

CFRP の欠陥・応力分布で、局所特徴だけでなく積層全体や広域の荷重経路が効いているなら、BiStride は精度改善につながる可能性があります。

ただし、multi-scale edge や階層情報を用意する必要があるため、標準 `MeshGraphNet` より実装コストは高いです。

## おすすめの実験順

### Step 1: 標準 MeshGraphNet を詰める

まずは今の `--conv_type meshgraphnet` を本命にして、processor blocks を振ります。

```bash
--conv_type meshgraphnet --mgn_blocks 5
--conv_type meshgraphnet --mgn_blocks 10
--conv_type meshgraphnet --mgn_blocks 15
--conv_type meshgraphnet --mgn_blocks 20
```

目安:

- blocks が少ない: 局所しか見えない可能性
- blocks が多い: 長距離情報は入るが、過平滑化・過学習・計算コスト増の可能性
- まずは 10 / 15 を中心に比較

### Step 2: hidden size を振る

`hidden_channels` も比較します。

例:

```bash
--hidden_channels 8
--hidden_channels 16
--hidden_channels 32
```

ただし、大きくしすぎると過学習しやすくなります。

### Step 3: BiStrideMeshGraphNet を検討

標準 `MeshGraphNet` が良いが、まだ長距離依存を拾い切れていないと感じる場合は `BiStrideMeshGraphNet` を試します。

特に以下の傾向がある場合は有望です。

- 欠陥位置から離れた領域の予測が悪い
- 境界付近や積層全体の傾向を外す
- 局所的には合っているが、全体分布がズレる
- processor blocks を増やすと少し良くなるが、重くなる

### Step 3.5: MeshGraphNet-Transformer を検討

標準 `MeshGraphNet` で長距離依存が不足していそうな場合は、`BiStrideMeshGraphNet` と並行して `MeshGraphNet-Transformer` も検討します。

特に以下の場合は MGN-T が有望です。

- processor blocks を増やしても広域分布が改善しにくい
- mesh coarsening を準備するより Transformer processor を試したい
- boundary condition や geometry/topology の変化を大域的に扱いたい
- GPU memory と実装時間に余裕がある

最初は full-size 実験ではなく、小さい batch / 小さい hidden size / 少ない Transformer layers で memory と速度を確認します。


### Step 4: MeshGraphKAN を検討

標準 `MeshGraphNet` と BiStride の比較後に、非線形表現力を上げたい場合に `MeshGraphKAN` を試します。

### Step 5: HybridMeshGraphNet は最後

大変形・接触・衝突のような問題でなければ、`HybridMeshGraphNet` は後回しでよいです。

## 現時点の実務判断

今回の優先順位は次の通りです。

```text
MeshGraphNet
    ↓
BiStrideMeshGraphNet or MeshGraphNet-Transformer
    ↓
MeshGraphKAN
    ↓
HybridMeshGraphNet
```

特に、すでに `meshgraphnet` の精度が良かったなら、まずはそれを本命として ablation するのが一番堅いです。

## ほかに試す価値がありそうなモデル

`MeshGraphNet` が既に良い場合、次に探すべきモデルは **長距離依存をどう扱うか** と **メッシュ・形状の変化にどれだけ強いか** で選びます。今回の CFRP 固定メッシュ問題では、いきなり全部を試すより、標準 MGN を強い baseline にしてから、以下の候補を順に試すのが現実的です。

| 候補 | 何が良いか | 今回の優先度 | 注意点 |
|---|---|---:|---|
| `MeshGraphNet-Transformer` / MGN-T [6] | MGN の幾何 bias と Transformer の global modeling を両立 | 高 | 新しい論文なので、まず発展的比較として扱う |
| `Transolver` 系 [12] | physics-aware token / slice attention で大規模幾何の長距離相関を扱う | 高 | 既存 `train.py` への統合は大きめ。分類タスクへの適用設計が必要 |
| `GINO` [13] | irregular mesh / varying geometry に強い neural operator | 中〜高 | 連続場の operator learning 寄り。19クラス分類には head の調整が必要 |
| `X-MeshGraphNet` [5] | 大規模メッシュ向けの scalable multi-scale MGN | 中 | 実装・分割・階層化が重い |
| `Graph Neural Operator` / GNO [14] | 不規則点群・メッシュ上の operator learning | 中 | 標準 MGN より実験設計が変わる |
| Equivariant GNN | 座標変換に対する対称性を入れやすい | 低〜中 | CFRP は積層方向・異方性があり、単純な回転等価性が合わない可能性 |

### 1. Transolver 系

`Transolver` は、一般形状上の PDE を解くための Transformer 型 neural solver です [12]。通常の点ごとの full attention ではなく、物理状態に基づく learnable slice / physics-aware token を使い、複雑形状上の物理相関を効率よく扱うことを狙います。

今回の CFRP メッシュで試す価値がある理由は次の通りです。

- 標準 `MeshGraphNet` の message passing では遠方情報が届きにくい場合がある
- CFRP の応力・欠陥分布は、局所欠陥だけでなく境界条件や積層全体の影響を受ける可能性がある
- Transformer 系なので、global な依存関係を直接扱いやすい
- Transolver++ / Transolver-3 など、million-scale geometry を意識した発展版も出ている

ただし、現在の `train.py` にそのまま差し替えるのは簡単ではありません。まずは **node-wise 19クラス分類 head を付けた小規模 prototype** として試すのが安全です。

最初に確認すること:

- 入力を node coordinates + node features として渡せるか
- 出力を node-wise logits にできるか
- 14k nodes 規模で GPU memory が足りるか
- edge connectivity を使わない場合、MGN より局所幾何が弱くならないか
- edge / relative coordinate 情報を positional encoding として入れられるか

### 2. GINO: Geometry-Informed Neural Operator

`GINO` は、任意形状・不規則メッシュ上の PDE solution operator を学習するための neural operator です [13]。Graph Neural Operator と Fourier Neural Operator を組み合わせ、入力形状や点群から潜在格子へ写し、また任意点へ戻す構成を取ります。

今回の問題で有望なケース:

- サンプルごとに geometry や mesh が変わる
- 固定節点分類より、連続的な応力場・変位場の回帰に近い
- mesh resolution を変えても動く surrogate が欲しい
- 将来的に別形状・別境界条件へ generalize したい

一方で、現在のタスクが **固定メッシュ上の19クラス node classification** なら、標準 `MeshGraphNet` の方が素直です。GINO は次の段階、つまり「固定メッシュだけでなく、違うメッシュ・違う形状にも拡張したい」ときに強い候補になります。

### 3. X-MeshGraphNet

`X-MeshGraphNet` は、大規模物理シミュレーション向けの scalable multi-scale GNN です [5]。標準 MGN の良さを残しながら、メッシュ分割・マルチスケール化で大規模化に対応する方向です。

今回の問題での位置づけは、`BiStrideMeshGraphNet` よりさらに大規模・本格的な拡張候補です。

有望な場合:

- node 数が今後かなり増える
- 1枚の GPU で MGN が重くなる
- subdomain / partition ベースで学習したい
- 複数スケールの応力伝播を扱いたい

### 4. Graph Neural Operator / GNO

`Graph Neural Operator` は、不規則点群・メッシュ上で operator learning を行うためのモデルです [14]。MGN が「このグラフ上で message passing して予測する」寄りなのに対し、GNO は「入力関数から出力関数への写像を学ぶ」寄りです。

今回の固定メッシュ分類では最優先ではありませんが、次のような方向に研究を広げるなら候補になります。

- DSPSS などの入力場から応力場全体への operator を学ぶ
- メッシュ解像度に依存しない surrogate を作る
- 回帰タスクと分類タスクを multi-task にする

### 5. Equivariant GNN は慎重に

E(n)-equivariant GNN や SE(3)-equivariant GNN は、座標変換に対する対称性を組み込めるため、分子・粒子系では強い候補です。

ただし CFRP では、積層方向、繊維方向、上下層、境界条件が物理的に意味を持つため、単純な回転・反転 equivariance を入れると逆に合わない場合があります。使うなら、CFRP の異方性と層情報を壊さない設計が必要です。

### 今回の追加候補の実験優先順位

現実的には、次の順番が良いです。

```text
1. MeshGraphNet を本命として mgn_blocks / hidden size を詰める
2. MeshGraphNet-Transformer または Transolver 系で長距離依存を試す
3. BiStrideMeshGraphNet / X-MeshGraphNet で multi-scale 化を試す
4. GINO / GNO を、違うメッシュ・違う形状への generalization 用に試す
5. MeshGraphKAN は非線形表現力の追加比較として試す
```

短期的には **MGN-T と Transolver 系** が一番面白いです。どちらも「標準 MGN は局所 message passing が強いが、長距離伝播が遅い」という弱点を狙っているため、今回の `meshgraphnet` が良かった結果から自然に発展できます。


## WCCM で MeshGraphNet の結果を使う場合にどのくらい説明するか

WCCM などの計算力学系の発表で `MeshGraphNet` の結果を使う場合、説明量は **「GNN の新規提案なのか」**、それとも **「CFRP 解析への応用・比較実験なのか」** で変えます。

今回の研究目的が「新しい GNN アーキテクチャの提案」ではなく、CFRP メッシュ上の欠陥・応力場予測で `MeshGraphNet` が有効だったことを示すなら、`MeshGraphNet` の理論を深く説明しすぎる必要はありません。

ただし、査読者・聴衆が結果を信用できるように、**再現性・公平な比較・物理的妥当性** に関わる部分は必ず説明します。

### 最低限説明すべき内容

論文・予稿・スライドのどれでも、最低限以下は入れます。

1. **なぜ MeshGraphNet を使ったか**
   - FEM/CAE メッシュと相性が良い
   - node feature と edge feature を同時に扱える
   - message passing によりメッシュ上の局所相互作用を表現できる
   - edge update があるため、単純な GAT/GCN よりメッシュ幾何を使いやすい

2. **入力と出力**
   - node feature: 例 `x, y, z, DSPSS`
   - edge feature: 例 `[dx, dy, dz, dist]`
   - output: 各ノードの 19 クラス分類、または応力・欠陥ラベル

3. **グラフの作り方**
   - FEM メッシュの節点を node とする
   - メッシュ接続を edge とする
   - 固定メッシュなのか、サンプルごとに変わるメッシュなのか
   - edge は有向化しているか、双方向化しているか

4. **モデル構造の概要**
   - Encode-Process-Decode 構造
   - processor block 数、例 `mgn_blocks=10`
   - hidden size
   - dropout / edge dropout
   - edge と node を残差更新すること

5. **比較対象**
   - GAT, GATv2, TransformerConv, PNA, GINE など、実際に比較したモデル
   - 比較時に train/val/test split、入力特徴、損失関数、epoch 数を揃えたこと

6. **評価方法**
   - accuracy だけでなく macro-F1 / class-wise F1 / confusion matrix を示す
   - class imbalance がある場合は、少数クラスの recall / F1 を必ず示す
   - 欠陥検出では false positive / false negative の傾向も説明する

7. **データ分割とリーク対策**
   - 同一条件・同一試験片・同一グループが train と test にまたがっていないか
   - 正規化を train set の統計だけで行ったか
   - ミラー・回転・近い条件のデータがリークしていないか

8. **物理的な妥当性**
   - 予測が局所的に滑らかか
   - 欠陥周辺や応力集中部で予測が妥当か
   - 境界条件や積層方向と矛盾していないか
   - 誤分類が物理的に近いクラス間で起きているか

### 予稿・論文での書き方の目安

WCCM の短い予稿なら、`MeshGraphNet` の説明は **半ページ程度** で十分です。

目安は次の通りです。

| 場所 | 説明量 | 書く内容 |
|---|---:|---|
| Abstract | 1文 | MeshGraphNet を用いてメッシュ上のノード分類/応力場予測を行った、と書く |
| Method | 1/3〜1/2ページ | Encode-Process-Decode、node/edge feature、processor blocks、出力、損失関数 |
| Results | 主要部分 | GAT 等との比較、class-wise 指標、可視化、物理的妥当性 |
| Discussion | 数段落 | なぜ MGN が良かったか、限界、今後 BiStride/KAN を試す余地 |

重要なのは、**MeshGraphNet 自体の一般論より、今回の CFRP データでなぜ有効だったかを説明すること**です。

### 口頭発表スライドでの説明量

10〜15分発表なら、MeshGraphNet の説明は **1〜2枚** でよいです。

おすすめ構成:

1. **Method slide 1: Graph representation**
   - node = FEM node
   - edge = mesh connectivity
   - node feature と edge feature
   - output = node-wise class

2. **Method slide 2: MeshGraphNet architecture**
   - Encoder → Processor × M → Decoder の図
   - edge update と node update があること
   - `mgn_blocks` と hidden size

その後は、モデル説明よりも以下に時間を使います。

- どのベースラインより良かったか
- 少数クラスで改善したか
- 予測分布が物理的に自然か
- 失敗例はどこか

### 「MeshGraphNet がブラックボックスに見える」ことへの対策

WCCM では、単に「深層学習で精度が高い」だけだと弱いです。

次の説明を入れると、計算力学の聴衆に伝わりやすくなります。

- メッシュ接続をそのまま使うため、CNN のように格子化する必要がない
- edge feature に相対座標と距離を入れるため、幾何情報を明示的に使っている
- message passing は、隣接要素・隣接節点間の情報伝播として解釈できる
- processor block を増やすと、より遠い節点からの影響を取り込める
- 予測結果をメッシュ上に戻して可視化し、物理的な妥当性を確認している

### 結果を載せるときの注意

`MeshGraphNet` の結果を WCCM に載せるなら、次の点を必ず確認します。

- test set は完全に未使用か
- hyperparameter tuning に test set を使っていないか
- best epoch を validation で選んでいるか
- class imbalance に対して accuracy だけで議論していないか
- 同じデータ分割で他モデルと比較しているか
- seed を変えた複数回実験、または少なくとも seed を明記しているか
- 推論結果の可視化を載せて、物理的に変な予測がないか説明しているか

### 論文中の短い説明例

短く書くなら、例えば次のような説明で十分です。

```text
The CFRP specimen was represented as a graph whose nodes correspond to finite-element mesh nodes and whose edges follow the mesh connectivity. We used MeshGraphNet with an encode-process-decode architecture. Node features included spatial coordinates and the measured/derived scalar field, while edge features were computed from relative coordinates and Euclidean distance. The processor updated both edge and node embeddings through residual message-passing blocks, and the decoder produced node-wise class logits.
```

日本語なら次のように書けます。

```text
CFRP 試験片を FEM メッシュに基づくグラフとして表現し、節点をノード、メッシュ接続をエッジとした。モデルには Encode-Process-Decode 構造を持つ MeshGraphNet を用いた。ノード特徴量には座標および測定・派生スカラー量を用い、エッジ特徴量には隣接節点間の相対座標および距離を用いた。Processor ではエッジ表現とノード表現を残差的に更新し、Decoder により各節点のクラス logits を出力した。
```

### どこまで詳しく書かなくてよいか

今回の主張が「MeshGraphNet という新規モデルを提案した」ではないなら、以下は詳細に書きすぎなくてよいです。

- MeshGraphNet の全数式展開
- GNN の一般論
- KAN や BiStride の詳細理論
- PhysicsNeMo の内部実装の細かい API 説明

代わりに、**データ、分割、入力特徴、比較条件、評価指標、物理的解釈** を丁寧に書く方が WCCM では重要です。

## 引用・参考文献の入れ方

WCCM の原稿で `MeshGraphNet` の結果を使うなら、最低限 **MeshGraphNet 原論文 [1]** は引用します。`BiStrideMeshGraphNet` を実際に使う、または今後の候補として議論するなら **BSMS-GNN [2]** も引用します。`MeshGraphKAN` は標準 `MeshGraphNet` と比べた実験を載せる場合に、KAN 原論文 [3] と Fourier KAN 関連論文 [4] を引用候補にします。

### 必ず引用したいもの

1. **MeshGraphNet の原論文**
   - `MeshGraphNet` を使った主結果を載せるなら必須です。
   - メッシュをグラフとして扱い、node/edge feature を encode し、message passing processor で更新して decode する根拠になります。

2. **比較対象の GNN 論文**
   - GAT と比較したなら GAT [8]
   - PNA と比較したなら PNA [9]
   - GIN/GINE と比較したなら GIN [10]
   - GCN 系を baseline として説明するなら GCN [11]

3. **使った実装・ライブラリ**
   - PhysicsNeMo の実装を直接使う場合は、論文ではなく software / documentation として PhysicsNeMo docs [7] を引用または脚注に入れます。
   - このリポジトリ内の独自実装を使う場合は、原論文 [1] を引用し、実装差分を Method に短く書きます。

### 必要に応じて引用するもの

- **BiStrideMeshGraphNet を使う場合**: BSMS-GNN [2]
- **MeshGraphKAN を使う場合**: KAN [3] と Fourier KAN [4]
- **大規模メッシュ・分割・マルチスケール拡張を議論する場合**: X-MeshGraphNet [5]
- **world edge / contact / large deformation を説明する場合**: MeshGraphNet 原論文 [1] の world edge の説明、または PhysicsNeMo docs [7]

### WCCM 原稿での引用例

英語なら、Method で次のように書けます。

```text
We adopted MeshGraphNet (MGN) [1], an encode-process-decode graph neural network originally proposed for mesh-based physical simulation. The finite-element mesh was represented as a graph, where mesh nodes and mesh connectivity define graph nodes and edges, respectively. Edge attributes were computed from relative coordinates and Euclidean distance, and both node and edge embeddings were updated through residual message-passing blocks.
```

日本語メモなら、次のように整理できます。

```text
本研究では、メッシュベース物理シミュレーション向けに提案された MeshGraphNet [1] を用いた。FEM メッシュの節点をグラフノード、メッシュ接続をグラフエッジとして扱い、エッジ特徴量には隣接節点間の相対座標および距離を用いた。Processor ではノード表現とエッジ表現を message passing により残差更新し、Decoder により各節点のクラス logits を出力した。
```

### 参考文献リスト案

WCCM の参考文献には、まず [1] を入れます。比較モデルや追加実験の有無に応じて [2] 以降を追加します。

[1] T. Pfaff, M. Fortunato, A. Sanchez-Gonzalez, and P. W. Battaglia, “Learning Mesh-Based Simulation with Graph Networks,” ICLR, 2021. arXiv:2010.03409. <https://arxiv.org/abs/2010.03409>

[2] Y. Cao, M. Chai, M. Li, and C. Jiang, “Efficient Learning of Mesh-Based Physical Simulation with BSMS-GNN,” arXiv:2210.02573, 2022. <https://arxiv.org/abs/2210.02573>

[3] Z. Liu, Y. Wang, S. Vaidya, F. Ruehle, J. Halverson, M. Soljačić, T. Y. Hou, and M. Tegmark, “KAN: Kolmogorov-Arnold Networks,” arXiv:2404.19756, 2024. <https://arxiv.org/abs/2404.19756>

[4] A. Mehrabian, P. M. Adi, M. Heidari, and I. Hacihaliloglu, “Implicit Neural Representations with Fourier Kolmogorov-Arnold Networks,” arXiv:2409.09323, 2024. <https://arxiv.org/abs/2409.09323>

[5] M. A. Nabian, C. Liu, R. Ranade, and S. Choudhry, “X-MeshGraphNet: Scalable Multi-Scale Graph Neural Networks for Physics Simulation,” arXiv:2411.17164, 2024. <https://arxiv.org/abs/2411.17164>

[6] M. M. Iparraguirre, I. Alfaro, D. Gonzalez, and E. Cueto, “MeshGraphNet-Transformer: Scalable Mesh-based Learned Simulation for Solid Mechanics,” arXiv:2601.23177, 2026. DOI: 10.48550/arXiv.2601.23177. <https://arxiv.org/abs/2601.23177>

[7] NVIDIA, “MeshGraphNet: A Practical User Tutorial,” PhysicsNeMo documentation. <https://docs.nvidia.com/physicsnemo/latest/user-guide/model_architecture/meshgraphnet.html>

[8] P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, and Y. Bengio, “Graph Attention Networks,” ICLR, 2018. arXiv:1710.10903. <https://arxiv.org/abs/1710.10903>

[9] G. Corso, L. Cavalleri, D. Beaini, P. Liò, and P. Veličković, “Principal Neighbourhood Aggregation for Graph Nets,” NeurIPS, 2020. arXiv:2004.05718. <https://arxiv.org/abs/2004.05718>

[10] K. Xu, W. Hu, J. Leskovec, and S. Jegelka, “How Powerful are Graph Neural Networks?” ICLR, 2019. arXiv:1810.00826. <https://arxiv.org/abs/1810.00826>

[11] T. N. Kipf and M. Welling, “Semi-Supervised Classification with Graph Convolutional Networks,” ICLR, 2017. arXiv:1609.02907. <https://arxiv.org/abs/1609.02907>

[12] H. Wu, H. Luo, H. Wang, J. Wang, and M. Long, “Transolver: A Fast Transformer Solver for PDEs on General Geometries,” arXiv:2402.02366, 2024. <https://arxiv.org/abs/2402.02366>

[13] Z. Li, N. B. Kovachki, C. Choy, B. Li, J. Kossaifi, S. P. Otta, M. A. Nabian, M. Stadler, C. Hundt, K. Azizzadenesheli, and A. Anandkumar, “Geometry-Informed Neural Operator for Large-Scale 3D PDEs,” NeurIPS, 2023. arXiv:2309.00583. <https://arxiv.org/abs/2309.00583>

[14] Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart, and A. Anandkumar, “Neural Operator: Graph Neural Operator and Fourier Neural Operator,” arXiv:2003.03485, 2020. <https://arxiv.org/abs/2003.03485>

### BibTeX メモ

原稿作成時にそのまま `.bib` に移せるように、最低限の BibTeX 形式も置いておきます。会議名やページ番号は投稿フォーマットに合わせて調整してください。

```bibtex
@inproceedings{pfaff2021meshgraphnets,
  title     = {Learning Mesh-Based Simulation with Graph Networks},
  author    = {Pfaff, Tobias and Fortunato, Meire and Sanchez-Gonzalez, Alvaro and Battaglia, Peter W.},
  booktitle = {International Conference on Learning Representations},
  year      = {2021},
  url       = {https://arxiv.org/abs/2010.03409}
}

@article{cao2022bsmsgnn,
  title   = {Efficient Learning of Mesh-Based Physical Simulation with BSMS-GNN},
  author  = {Cao, Yadi and Chai, Menglei and Li, Minchen and Jiang, Chenfanfu},
  journal = {arXiv preprint arXiv:2210.02573},
  year    = {2022},
  url     = {https://arxiv.org/abs/2210.02573}
}

@article{liu2024kan,
  title   = {KAN: Kolmogorov-Arnold Networks},
  author  = {Liu, Ziming and Wang, Yixuan and Vaidya, Sachin and Ruehle, Fabian and Halverson, James and Solja{\v{c}}i{\'c}, Marin and Hou, Thomas Y. and Tegmark, Max},
  journal = {arXiv preprint arXiv:2404.19756},
  year    = {2024},
  url     = {https://arxiv.org/abs/2404.19756}
}

@article{mehrabian2024fkan,
  title   = {Implicit Neural Representations with Fourier Kolmogorov-Arnold Networks},
  author  = {Mehrabian, Ali and Adi, Parsa Mojarad and Heidari, Moein and Hacihaliloglu, Ilker},
  journal = {arXiv preprint arXiv:2409.09323},
  year    = {2024},
  url     = {https://arxiv.org/abs/2409.09323}
}

@article{nabian2024xmeshgraphnet,
  title   = {X-MeshGraphNet: Scalable Multi-Scale Graph Neural Networks for Physics Simulation},
  author  = {Nabian, Mohammad Amin and Liu, Chang and Ranade, Rishikesh and Choudhry, Sanjay},
  journal = {arXiv preprint arXiv:2411.17164},
  year    = {2024},
  url     = {https://arxiv.org/abs/2411.17164}
}


@article{iparraguirre2026mgnt,
  title   = {MeshGraphNet-Transformer: Scalable Mesh-based Learned Simulation for Solid Mechanics},
  author  = {Iparraguirre, Mikel M. and Alfaro, Iciar and Gonzalez, David and Cueto, Elias},
  journal = {arXiv preprint arXiv:2601.23177},
  year    = {2026},
  doi     = {10.48550/arXiv.2601.23177},
  url     = {https://arxiv.org/abs/2601.23177}
}

@inproceedings{velickovic2018gat,
  title     = {Graph Attention Networks},
  author    = {Veli{\v{c}}kovi{\'c}, Petar and Cucurull, Guillem and Casanova, Arantxa and Romero, Adriana and Li{\`o}, Pietro and Bengio, Yoshua},
  booktitle = {International Conference on Learning Representations},
  year      = {2018},
  url       = {https://arxiv.org/abs/1710.10903}
}

@inproceedings{corso2020pna,
  title     = {Principal Neighbourhood Aggregation for Graph Nets},
  author    = {Corso, Gabriele and Cavalleri, Luca and Beaini, Dominique and Li{\`o}, Pietro and Veli{\v{c}}kovi{\'c}, Petar},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2020},
  url       = {https://arxiv.org/abs/2004.05718}
}

@inproceedings{xu2019gin,
  title     = {How Powerful are Graph Neural Networks?},
  author    = {Xu, Keyulu and Hu, Weihua and Leskovec, Jure and Jegelka, Stefanie},
  booktitle = {International Conference on Learning Representations},
  year      = {2019},
  url       = {https://arxiv.org/abs/1810.00826}
}

@inproceedings{kipf2017gcn,
  title     = {Semi-Supervised Classification with Graph Convolutional Networks},
  author    = {Kipf, Thomas N. and Welling, Max},
  booktitle = {International Conference on Learning Representations},
  year      = {2017},
  url       = {https://arxiv.org/abs/1609.02907}
}

@article{wu2024transolver,
  title   = {Transolver: A Fast Transformer Solver for PDEs on General Geometries},
  author  = {Wu, Haixu and Luo, Huakun and Wang, Haowen and Wang, Jianmin and Long, Mingsheng},
  journal = {arXiv preprint arXiv:2402.02366},
  year    = {2024},
  url     = {https://arxiv.org/abs/2402.02366}
}

@inproceedings{li2023gino,
  title     = {Geometry-Informed Neural Operator for Large-Scale 3D PDEs},
  author    = {Li, Zongyi and Kovachki, Nikola Borislavov and Choy, Chris and Li, Boyi and Kossaifi, Jean and Otta, Shourya Prakash and Nabian, Mohammad Amin and Stadler, Maximilian and Hundt, Christian and Azizzadenesheli, Kamyar and Anandkumar, Anima},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2023},
  url       = {https://arxiv.org/abs/2309.00583}
}

@article{li2020neuraloperator,
  title   = {Neural Operator: Graph Neural Operator and Fourier Neural Operator},
  author  = {Li, Zongyi and Kovachki, Nikola and Azizzadenesheli, Kamyar and Liu, Burigede and Bhattacharya, Kaushik and Stuart, Andrew and Anandkumar, Anima},
  journal = {arXiv preprint arXiv:2003.03485},
  year    = {2020},
  url     = {https://arxiv.org/abs/2003.03485}
}
```

## 参考リンク

- NVIDIA PhysicsNeMo MeshGraphNet documentation: <https://docs.nvidia.com/physicsnemo/latest/user-guide/model_architecture/meshgraphnet.html>
- NVIDIA PhysicsNeMo API documentation: <https://docs.nvidia.com/physicsnemo/latest/api/physicsnemo.models.meshgraphnet.html>
