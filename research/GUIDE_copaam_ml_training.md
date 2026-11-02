# 指示書（copaam）：残留ひずみ予測モデルの学習

> **対象サーバ**：慶應 GPU サーバ copaam（PBS／qsub）。
> **対象スクリプト**：`denso_residual_strain_ml.py`、ジョブスクリプト `scripts/denso_ml.pbs`。
> **関連**：Abaqus による固体有限要素の検証は fifa サーバで行う（[`GUIDE_fifa_abaqus.md`](./GUIDE_fifa_abaqus.md)）。
>
> 本手順の目的は、デンソー様との共同研究（[`DENSO_RA_RESIDUAL_STRAIN.md`](./DENSO_RA_RESIDUAL_STRAIN.md)）で想定する
> 「数値解析で事前学習し、少数の実測点で合わせ込む」手順を、**合成データで**事前に試すことである。
> 実データは契約前のため未入手であり、ここで得られる数値は**手法の検証**であって、材料の予測ではない。

---

## 1. 学習の内容

### 1.1 データ
成形板を多数生成する。生成には、板厚方向の物理計算（`viz/denso_plate_features.py` を一般化したもの）を用いる。
各板では、金型温度と繊維配向の層構造（コアの配向・幅・上下のずれなど）を乱数で与え、板厚方向 41 点で次を計算する。

- 入力の材料
  - 結晶化度 *X*<sub>c</sub>：熱伝導と、検証済みの Hoffman–Lauritzen 結晶化速度式から求める。
  - 配向テンソル *a<sub>ij</sub>*：仮定した層構造から与える。
- 出力
  - 残留ひずみ *ε<sub>xx</sub>*, *ε<sub>yy</sub>*：配向平均した熱膨張と結晶化収縮を求め、自由な板のつり合いから計算する。

二つの「領域」を作る。

| 領域 | 役割 | 違い |
|---|---|---|
| source | 数値解析の役 | 基準の定数を使う。板厚方向の全点が得られ、ノイズはない |
| target | 実験の役 | 定数を系統的にずらす：直交方向の熱膨張 +15 %、直交方向の結晶化収縮 −15 %、*X*<sub>∞</sub> = 0.27、固定温度 150 °C。各板で **5 点だけ**を観測し、計測ノイズ（標準偏差 0.05×10<sup>−3</sup>）を加える |

### 1.2 実験

**実験1（入力に何が要るか）**

残留ひずみ *ε*<sub>res</sub> = *ε*<sub>0</sub> + *z κ* − *ε*\* のうち、*ε*\* はその点の量で決まる。
一方、*ε*<sub>0</sub> と *κ* は断面全体のつり合いで決まる。そこで次の2つを比較する。

- `local`：その点の [*z*, *a*<sub>11</sub>, *a*<sub>22</sub>, *a*<sub>33</sub>, *X*<sub>c</sub>] のみを入力とする（仕様どおりの入力）。
- `context`：上記に、断面の要約量6個を加える。要約量は *a*<sub>11</sub>, *a*<sub>22</sub>, *X*<sub>c</sub> それぞれの板厚平均と一次モーメントである。

**実験2（転移学習）**

測定する target 板の枚数を変え、次の3通りを比較する。評価には、未知の target 板 200 枚の全点を用いる。

- `source-only`：source で学習したモデルをそのまま使う。
- `target-only`：target の少数点だけで一から学習する。
- `fine-tune`：source で事前学習し、target の少数点で合わせ込む。

---

## 2. 準備（初回のみ）

```bash
# 1) リポジトリを取得（既にあれば git pull）
git clone https://github.com/keisuke58/wccm2026-cfrp-gnn.git
cd wccm2026-cfrp-gnn

# 2) GPU とドライバの確認。表示される "CUDA Version" を控える
nvidia-smi

# 3) 仮想環境を作る（場所は任意）
python3 -m venv $HOME/venvs/denso
source $HOME/venvs/denso/bin/activate
pip install --upgrade pip
pip install numpy scipy matplotlib pillow

# 4) PyTorch を入れる。cu の番号は nvidia-smi の CUDA Version 以下のものを選ぶ
#    （例：CUDA Version 12.4 なら cu121 または cu124）。対応表は https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 5) GPU が見えるか確認（True になればよい）
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

> ログインノードに GPU が無いクラスタでは、手順 5 が False になっても問題ない。
> その場合は計算ノードで確認する（§3.2 のジョブのログに表示される）。

---

## 3. 実行

### 3.1 動作確認（CPU 4 コアで約 30 秒。ログインノードでも可）
```bash
source $HOME/venvs/denso/bin/activate
python denso_residual_strain_ml.py --quick --workers 4 --out results/denso_ml_quick
```
冒頭の検証2行が次のようになることを確かめる。
```
[validate] vectorised K(T) vs K_hl: max relative difference 2.3e-16      ← 1e-12 以下なら正常
[validate] symmetric plate curvature: 1.1e-16 1/m (should be ~0)       ← 1e-10 以下なら正常
```

### 3.2 本番（qsub でジョブ投入）
`scripts/denso_ml.pbs` を使う。**投入前に次の2か所を確認する。**

1. **キュー名**
   - `qstat -Q` で GPU のキューを調べる。
   - スクリプトの `##PBS -q QUEUE_NAME` の先頭の `##` を `#` にし、キュー名を書く。
2. **資源指定の書式**
   - スクリプトは Torque 形式の `#PBS -l nodes=1:ppn=16:gpus=1` を使う（既存の `inverse_rigor.pbs` と同じ）。
   - qsub がこの行を拒否したら、copaam は PBS Pro である。その行を `##` でコメントにし、`##PBS -l select=1:ncpus=16:ngpus=1` を有効にする。

```bash
cd wccm2026-cfrp-gnn
qsub -v PY=$HOME/venvs/denso/bin/python scripts/denso_ml.pbs
qstat -u $USER                 # 状態確認
```
- ログはジョブ名 `denso_ml` の出力ファイル `denso_ml.o<ジョブ番号>` に出る。
- 結果は `results/denso_ml_<日時>/` に保存される。

### 3.3 ジョブを使わずに直接回す場合
```bash
nohup $HOME/venvs/denso/bin/python denso_residual_strain_ml.py --workers 16 \
      --out results/denso_ml_$(date +%Y%m%d_%H%M) > denso_ml.log 2>&1 &
```

### 3.4 主なオプション
| オプション | 既定値 | 意味 |
|---|---|---|
| `--n-source` | 2000 | source（数値解析役）の板の枚数 |
| `--n-target` | `1,2,5,10,20,50` | 測定する target 板の枚数（カンマ区切りで複数指定） |
| `--points` | 5 | target 板1枚あたりの測定点数 |
| `--noise` | 0.05 | 計測ノイズの標準偏差 [×10<sup>−3</sup>] |
| `--epochs` / `--ft-epochs` | 1500 / 600 | 事前学習／合わせ込みのエポック数 |
| `--repeats` | 10 | 実験2の繰り返し回数（測定板の選び方を変える） |
| `--workers` | 1 | データ生成の並列数（CPU コア数に合わせる） |
| `--device` | `auto` | `auto`（GPU があれば使う）、`cpu`、`cuda` |

---

## 4. 出力と読み方

`results/<出力名>/` に次の2つができる。
- `results.json`：全数値。実験1の RMSE と *R*²、実験2の手法 × 測定枚数ごとの RMSE の平均と標準偏差、使った定数を含む。
- `denso_ml.png`：(a) 実験1の比較、(b) 実験2の学習曲線、(c) 未知の target 板1枚の予測例。

### 4.1 参照値（本リポジトリの CPU 4 コア環境で実測）
- 乱数種が同じなので、生成されるデータは同じになる。
- 学習の数値は完全には一致しない。GPU と CPU では浮動小数点の演算順が異なるためである。値は近くなるはずだが、一致の度合いは未確認である。
- 表の RMSE の単位は ×10<sup>−3</sup> である。参考として、target の残留ひずみの標準偏差は 2.43×10<sup>−3</sup> である。

**表1** 実験1（source の検証用 20 % で評価）

| 入力 | `--quick` | 既定値（本番） |
|---|---|---|
| `local` | RMSE 0.827、*R*² 0.882 | RMSE 0.713、*R*² 0.914 |
| `context` | RMSE 0.137、*R*² 0.997 | RMSE 0.054、*R*² 0.9995 |

**表2** 実験2（既定値。未知の target 板 200 枚で評価、10 回の平均 ± 標準偏差）

| 測定 target 板の枚数 | 1 | 2 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|
| `source-only` | 0.294 | 0.294 | 0.294 | 0.294 | 0.294 | 0.294 |
| `target-only` | 1.585 ± 0.325 | 1.209 ± 0.210 | 0.765 ± 0.136 | 0.457 ± 0.081 | 0.264 ± 0.039 | 0.164 ± 0.010 |
| `fine-tune` | 0.456 ± 0.210 | 0.271 ± 0.100 | 0.163 ± 0.031 | 0.154 ± 0.014 | 0.108 ± 0.013 | 0.081 ± 0.004 |

**所要時間**（既定値、CPU 4 コア、`--workers 4`）：合計 約 9 分。内訳はデータ生成 100 s、実験1 が 156 s と 111 s、残りが実験2である。
`--quick` の実験2（測定 1 / 2 / 5 / 10 枚）は、source-only 0.320、target-only 1.754 / 1.302 / 0.795 / 0.483、
fine-tune 0.274 / 0.217 / 0.259 / 0.214 であった（繰り返し 3 回）。

### 4.2 結果の解釈

**示せたこと（この合成問題の範囲で）**
- **仕様どおりの入力（その点の結晶化度と配向のみ）では不十分である。** 断面の要約量を加えると、RMSE は 0.713 から 0.054 に下がった（約 1/13）。
  - 理由：残留ひずみは断面全体のつり合いで決まる。そのため入力には「その点」の情報だけでなく「その断面」の情報が要る。
  - デンソー様と入力の仕様を決める際の論点になる。
- **測定が 2 枚以上あれば、合わせ込みが最もよい。** 2 枚 × 5 点で source-only と同程度（0.271 対 0.294）、5 枚で約 0.55 倍（0.163）、50 枚で約 0.28 倍（0.081）になった。
  target-only は 20 枚でようやく source-only を下回り、どの枚数でも合わせ込みより悪い。

**示せていないこと・注意**
- **測定 1 枚では、合わせ込みは source-only より悪い**（0.456 ± 0.210 対 0.294）。5 点の測定値に過剰に合わせているとみられる。
  - `--quick`（繰り返し 3 回）では逆の結果だったが、本番設定で覆った。前版の本書が書いた「測定 1 枚で両者を上回った」は誤りであり、ここで訂正する。
  - 対策の候補：合わせ込みのエポックを減らす、事前学習の重みへの正則化（L2-SP）、最終層のみの更新。いずれも未実施である。
- いずれも、定数を一様にずらしただけの合成 target での結果である（§5）。

---

## 5. 注意

- **GPU の効果は小さい。** モデルは小さな MLP で、時間の大半はデータ生成（CPU）にかかる。速くしたいときは `--workers` を増やす。
  GPU が効いてくるのは、場を一括で出す大きなモデル（DeepONet や GNN）に進んでからである。
- **合成データの限界。** target は定数を一様にずらしただけである。実際の計算と実測の差は、これより複雑である。たとえば配向モデルの誤差、ひずみの定義の違い、場所ごとに異なる誤差がある。
  したがって、ここでの好結果は実データでの成功を保証しない。
- **配向は仮定である。** 層構造は乱数で与えたもので、流動解析の結果ではない。
- **source は Abaqus ではない。** 学習データは Python の積層理論で作っている。同じ板を Abaqus の固体要素で解いた結果との照合は fifa で行う（[`GUIDE_fifa_abaqus.md`](./GUIDE_fifa_abaqus.md)）。
  Abaqus の出力を学習データとして読み込む機能は、まだ実装していない。
- 物理モデルの限界（潜熱・保圧・粘弾性緩和なし）は `viz/denso_plate_features.py` の docstring を参照。

---

## 6. copaam 上の Claude Code に頼む場合の指示文

> リポジトリ wccm2026-cfrp-gnn の `research/GUIDE_copaam_ml_training.md` に従って、残留ひずみ予測の学習を実行してください。
> 1. §2 の手順で仮想環境と PyTorch（GPU 版）を用意し、`torch.cuda.is_available()` を確認する。
> 2. §3.1 の動作確認を実行し、検証2行が基準を満たすことと、結果が §4.1 の `--quick` の参照値に近いことを確認する。
> 3. `qstat -Q` でキュー名を調べ、`scripts/denso_ml.pbs` のキュー名と資源指定の書式を copaam に合わせる。そのうえで §3.2 の方法で投入する。
> 4. 終了後、`results/` に出た `results.json` と `denso_ml.png` を確認し、実験1と実験2の数値を §4.1 の表1・表2の形式にまとめて報告する。
>    参照値と大きく違う場合は、その旨と考えられる原因を書く。結果を良く見せるために設定を変えないこと。
