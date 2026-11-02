# 指示書（fifa）：Abaqus による残留ひずみモデルの照合

> **対象サーバ**：fifa（Abaqus のライセンスがあるサーバ）。
> **対象スクリプト**：`abaqus/denso_plate/` 以下。
> - `gen_denso_plate.py`：入力ファイルと参照解を書き出す。
> - `run_denso_plate.sh`：一括実行する。
> - `extract_denso_plate.py`：odb から抽出する。
> - `compare_denso_plate.py`：参照解と照合する。
>
> **関連**：学習（機械学習）は copaam サーバで行う（[`GUIDE_copaam_ml_training.md`](./GUIDE_copaam_ml_training.md)）。
>
> 本手順の目的：学習データの生成に使っている Python の積層理論モデルが、**同じ板を Abaqus の 3 次元固体要素で解いた結果と一致するか**を確かめること。
> あわせて「Abaqus で解く → 結果を抽出 → 学習データにする」という経路を用意する。
> 定数はすべて例示値（未校正）である。

---

## 1. 解く問題

### 1.1 モデル
- **形状**：100 mm × 100 mm × 3 mm の自由な平板である。対称性を使って 1/4（50 mm × 50 mm）を解く。
- **要素**：C3D8I（非適合モード要素。曲げを精度よく表す）。分割は面内 25 × 25、板厚方向 40 層とする。
- **材料**：層ごとに一つの材料を割り当てる。
  - 剛性は、配向テンソル *a<sub>ij</sub>* で配向平均した ⟨***C***⟩（ハイブリッド閉包）である。
  - *a<sub>ij</sub>* は対角なので、この剛性は全体座標で直交異方性になる。そのため `*ELASTIC, TYPE=ORTHOTROPIC` で表す。
- **固有ひずみ**：層 *k* の固有ひずみ（eigenstrain）を次式で与える。

  *ε*\*<sub>*k*</sub> = ⟨***α***⟩<sub>*k*</sub> Δ*T* + ⟨***β***⟩<sub>*k*</sub> *X*<sub>c,*k*</sub>,  Δ*T* = *T*<sub>room</sub> − *T*<sub>lock</sub>  (1)

  Abaqus には、これを等価な熱膨張係数 *α*<sub>eff,*k*</sub> = *ε*\*<sub>*k*</sub> / Δ*T* として与える。そのうえで、板全体を一様に *T*<sub>lock</sub> から *T*<sub>room</sub> へ冷やす。
  - 線形弾性なので、この置き換えで固有ひずみは厳密に再現される。
  - これは荷重を与えるための手段であり、結晶化収縮が熱的だと主張するものではない。
  - *X*<sub>c</sub> は Python 側の 1 次元冷却・結晶化計算から取る。**Abaqus の中では熱伝導も結晶化も解いていない。**
- **境界条件**：対称面は *x* = 0 で *u<sub>x</sub>* = 0、*y* = 0 で *u<sub>y</sub>* = 0 とする。中心の下面 1 点で *u<sub>z</sub>* = 0 とする。これ以外は自由である。

### 1.2 参照解（積層理論）
同じ区分一定の層について、自由な板のつり合いを厳密に積分して解く。

[***A*** ***B***; ***B*** ***D***] [***ε***<sub>0</sub>; ***κ***] = [***N***\*; ***M***\*]  (2)

残留ひずみ（弾性ひずみ）は次式で与えられる。

***ε***<sub>res</sub>(*z*) = ***ε***<sub>0</sub> + *z* ***κ*** − ***ε***\*(*z*)  (3)

これと Abaqus の比較量は次のとおりである。
- 中心列の要素の弾性ひずみ `EE11`, `EE22`（積分点 8 点の平均 = 要素中心の値）。
- 中央面の *u<sub>z</sub>* から求めた曲率。ここから板全体のそり量（矢高）*κ L*²/8 を得る。

---

## 2. 準備（初回のみ）

```bash
git clone https://github.com/keisuke58/wccm2026-cfrp-gnn.git   # 既にあれば: cd wccm2026-cfrp-gnn && git pull
cd wccm2026-cfrp-gnn

# Abaqus のコマンド名と版を確認（abaqus 以外の名前なら、以降 ABAQUS=<名前> で指定）
which abaqus && abaqus information=release

# 入力ファイル生成用の Python（numpy と matplotlib が要る。Abaqus 付属の Python は使わない）
python3 -c "import numpy, matplotlib; print('ok')" \
  || python3 -m pip install --user numpy matplotlib
```
- ユーザサブルーチン（UMAT）は使わないので、Fortran コンパイラは不要である。

---

## 3. 実行

### 3.1 一括実行（推奨）
```bash
cd wccm2026-cfrp-gnn
CPUS=4 bash abaqus/denso_plate/run_denso_plate.sh
```
処理は次の4段階である。

1. `gen_denso_plate.py` が `abaqus/denso_plate/jobs/` に次の2ケースを書き出す。
   - `denso_symmetric`：金型 185 / 185 °C。
   - `denso_asymmetric`：金型 170 / 200 °C。

   書き出すファイルは、ケースごとに入力ファイル（約 3 MB）と参照解 `*_ref.json` である。
2. 各ケースを `abaqus job=... interactive double` で解く。
3. `abaqus python extract_denso_plate.py` で `*_abaqus.json` を書き出す。
4. `compare_denso_plate.py` で照合し、`*_compare.png` を描く。不合格があれば終了コード 1 を返す。

| 環境変数 | 既定値 | 意味 |
|---|---|---|
| `ABAQUS` | `abaqus` | Abaqus のコマンド名（例：`abq2024`） |
| `CPUS` | 4 | Abaqus の並列数 |
| `PY` | `python3` | 入力ファイル生成と照合に使う Python |

### 3.2 熱と結晶化の寄与を分けて解く
式 (1) は線形なので、熱の項と結晶化の項は重ね合わせられる。
```bash
bash abaqus/denso_plate/run_denso_plate.sh --part thermal   # 熱の項だけ
bash abaqus/denso_plate/run_denso_plate.sh --part cryst     # 結晶化の項だけ
```
total = thermal + cryst が成り立つことは、`*_abaqus.json` を足して確かめられる。

### 3.3 手で1ケースずつ回す場合
```bash
python3 abaqus/denso_plate/gen_denso_plate.py
cd abaqus/denso_plate/jobs
abaqus job=denso_asymmetric cpus=4 interactive double
abaqus python ../extract_denso_plate.py denso_asymmetric
python3 ../compare_denso_plate.py denso_asymmetric --fig
```

### 3.4 ジョブスケジューラ
fifa に qsub などのスケジューラがある場合は、サーバの規定に従ってジョブにする。中身は §3.1 の1行でよい。

---

## 4. 判定と参照値

### 4.1 合格基準
次の2つを満たせば合格とする（`compare_denso_plate.py` が `PASS` を出す）。
- 残留ひずみの誤差：max |FE − 参照| / max |参照| < 5×10<sup>−3</sup>
- そり量の差：max(5×10<sup>−3</sup> |参照|, 0.001 mm) 未満

### 4.2 参照値（CalculiX による事前確認と fifa の Abaqus）
クラウドの環境には Abaqus が無いので、まず同じ入力ファイルを CalculiX 2.21 で解いて確かめた（表1）。
- 変換は `abaqus/denso_plate/to_calculix.sh` で行う。変えたのはキーワードの綴りと出力要求だけである。
- その後、fifa の Abaqus 2024 で解いた（表2）。

**表2** fifa の Abaqus 2024 による照合（2026-11-03、`CPUS=4`、同じ 1/4 板 25 × 25 × 40 要素）

| ケース | ひずみ誤差 | そり量 *x* [mm]（参照） | そり量 *y* [mm]（参照） | 判定 |
|---|---|---|---|---|
| `denso_symmetric` | 5.5×10<sup>−4</sup> | −0.0000（−0.0000） | −0.0000（−0.0000） | PASS |
| `denso_asymmetric` | 6.9×10<sup>−4</sup> | −0.0562（−0.0563） | −1.3619（−1.3618） | PASS |
| `denso_symmetric_thermal` | 7.4×10<sup>−4</sup> | −0.0000（−0.0000） | −0.0000（+0.0000） | PASS |
| `denso_asymmetric_thermal` | 7.4×10<sup>−4</sup> | −0.0000（−0.0000） | −0.0000（+0.0000） | PASS |
| `denso_symmetric_cryst` | 4.7×10<sup>−4</sup> | −0.0000（−0.0000） | −0.0000（+0.0000） | PASS |
| `denso_asymmetric_cryst` | 5.0×10<sup>−4</sup> | −0.0562（−0.0563） | −1.3619（−1.3618） | PASS |

- total と thermal + cryst の差は、`EE11`, `EE22`, 中央面の *u<sub>z</sub>* のどれも 1.3×10<sup>−10</sup> 以下で、重ね合わせが成り立つ。
- 最初の実行は入力の段階で止まった（`***ERROR: Anisotropic material properties without a local orientation system`）。
  Abaqus は直交異方性の材料に局所座標系を求めるが、CalculiX は求めないためである。
  そこで `gen_denso_plate.py` で全体座標と同じ向きの `*ORIENTATION` を定義し、各層の `*SOLID SECTION` から参照した（`eeb8a71`）。
  局所座標が全体座標と一致するので、剛性・荷重・出力の向きは変わらない。

**表1** CalculiX による照合（例示定数、1/4 板 25 × 25 × 40 要素、4 コアで 30〜55 s）

| ケース | ひずみ誤差 | そり量 *x* [mm]（参照） | そり量 *y* [mm]（参照） | 判定 |
|---|---|---|---|---|
| `denso_symmetric` | 5.5×10<sup>−4</sup> | −0.0000（−0.0000） | −0.0000（−0.0000） | PASS |
| `denso_asymmetric` | 6.9×10<sup>−4</sup> | −0.0562（−0.0563） | −1.3619（−1.3618） | PASS |

- 図：`abaqus/denso_plate/calculix_check/denso_asymmetric_calculix.png`
- Abaqus でも表1と同じ値になった（表2、誤差 10<sup>−3</sup> 未満）。
- 合格しない場合は、パラメータを合わせにいかずに、出力（`*.dat`, `*.msg`, 照合の数値）をそのまま報告する。

### 4.3 生成器自身の検証（`gen_denso_plate.py` が毎回表示）
- **G1**：層剛性の直交異方性からのずれ。0 であること。
- **G2**：参照解の力・モーメントのつり合い。10<sup>−16</sup> 程度であること。
- **G3**：学習に使っている 41 点の台形則モデルとの差。ひずみ幅の 2〜4 % である。
  - これは層の中点で評価するか節点で評価するかの離散化の差であり、誤りではない。
  - 表層の急な配向変化の部分で差が出る。

---

## 5. 次の段階（未実施）

1. **学習データの Abaqus 化。** `gen_denso_plate.py --sample N --seed S` で、学習に使うのと同じ乱数で N 枚の入力ファイルを作れる。
   これを §3.3 の要領で解き、抽出する。
   - copaam の学習スクリプトが Abaqus の結果を読み込む部分は、まだ書いていない。
   - 現在のモデルでは、Abaqus の結果は積層理論と一致する（§4）。したがって学習データの中身はほぼ変わらない。
     この段階の意味は、経路を整えることにある。
2. **物理を Abaqus 側に移す。** 次の逐次解析に置き換えると、Abaqus が初めて新しい物理を足す。
   - 熱伝導解析で、板厚方向の温度履歴を求める。
   - 結晶化を連成した粘弾性 UMAT（`abaqus/cfrtp_cryst_umat_hl.f`）で、残留応力を求める。
   - 加える物理は、粘弾性緩和、端部、実形状である。
3. デンソー様の Ansys 解析と同じ量（ひずみの定義、評価位置）に合わせる。

---

## 6. fifa 上の Claude Code に頼む場合の指示文

> リポジトリ wccm2026-cfrp-gnn の `research/GUIDE_fifa_abaqus.md` に従って、Abaqus による照合を実行してください。
> 1. §2 の手順で Abaqus のコマンド名と版、python3 の numpy と matplotlib を確認する。
> 2. §3.1 の一括実行を行い、`compare_denso_plate.py` の出力を §4.2 の表1の形式にまとめる。`*_compare.png` も確認する。
> 3. 余裕があれば §3.2（`--part thermal` と `--part cryst`）も実行し、total = thermal + cryst が成り立つかを確かめる。
> 4. 不合格や Abaqus のエラーが出た場合は、`.msg` と `.dat` の該当箇所を添えて報告する。合格させるために設定や許容値を変えないこと。
