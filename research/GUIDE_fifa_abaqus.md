# 指示書（fifa）：Abaqus による残留ひずみ解析

> **対象サーバ**：fifa（Abaqus のライセンスがあるサーバ）。
> **対象**：`abaqus/denso_plate/` 以下。
> **関連**：学習（機械学習）は copaam で行う（[`GUIDE_copaam_ml_training.md`](./GUIDE_copaam_ml_training.md)）。
> 定数はすべて例示値（未校正）である。

## 0. 何を、どの順で行うか

| 順 | 内容 | 実行するもの | 必要なもの | 目安 |
|---|---|---|---|---|
| 1 | **弾性照合**（第 I 部）：学習データ生成に使う積層理論を、3 次元固体要素で確かめる | `run_denso_plate.sh` | Abaqus | **実施済み**（2026-11-03、6 ケースすべて PASS、§4 表2） |
| 2 | **穿孔法の模擬**（第 III 部）：デンソー様の既報（津田ら 2026）と同じ入力で穴あけを計算し、**既報の実験値（図12）と比べる** | `run_hole_drilling.sh` | Abaqus | **最優先**。1 ジョブ（約 13 万要素 × 11 ステップ） |
| 3 | **成形プロセス解析**（第 II 部）：熱伝導 → 結晶化を連成した粘弾性応力の逐次解析 | `run_denso_process.sh` | Abaqus ＋ UMAT 用 Fortran コンパイラ | 第 III 部の後。2 ケース、所要時間は未計測 |

第 I 部は物理を足さない確認である。**Abaqus が新しい物理（粘弾性緩和、脱型前後の拘束の変化、空冷中の結晶化）を加えるのは第 II 部**である。

## 1. 準備（初回のみ）

```bash
git clone https://github.com/keisuke58/wccm2026-cfrp-gnn.git   # 既にあれば: cd wccm2026-cfrp-gnn && git pull
cd wccm2026-cfrp-gnn

# Abaqus のコマンド名と版（abaqus 以外の名前なら、以降 ABAQUS=<名前> を前につける）
which abaqus && abaqus information=release

# UMAT のコンパイル確認（第 II 部のみ）。"PASS" が出ればよい
abaqus verify -user_std

# 入力ファイル生成用の Python（numpy と matplotlib。Abaqus 付属の Python は使わない）
python3 -c "import numpy, matplotlib; print('ok')" || python3 -m pip install --user numpy matplotlib
```

---

# 第 I 部 弾性照合

## 2. 解く問題

- **形状**：100 mm × 100 mm × 3 mm の自由な平板である。対称性を使って 1/4（50 mm × 50 mm）を解く。
- **要素**：C3D8I（非適合モード要素）。分割は面内 25 × 25、板厚方向 40 層とする。
- **材料**：層ごとに、配向テンソル *a<sub>ij</sub>* で配向平均した剛性 ⟨***C***⟩（ハイブリッド閉包）を与える。*a<sub>ij</sub>* が対角なので直交異方性になり、`*ELASTIC, TYPE=ORTHOTROPIC` で表せる。
- **固有ひずみ**：層 *k* について次式で与える。

  *ε*\*<sub>*k*</sub> = ⟨***α***⟩<sub>*k*</sub> Δ*T* + ⟨***β***⟩<sub>*k*</sub> *X*<sub>c,*k*</sub>,  Δ*T* = *T*<sub>room</sub> − *T*<sub>lock</sub>  (1)

  これを等価な熱膨張係数 *α*<sub>eff,*k*</sub> = *ε*\*<sub>*k*</sub> / Δ*T* として与え、板全体を一様に冷やす。線形弾性なので、固有ひずみは厳密に再現される。
  *X*<sub>c</sub> は Python 側の 1 次元計算から取る。
- **参照解**：同じ区分一定の層について、つり合い式 (2) を厳密に積分して求める。残留ひずみは式 (3) で与える。

  [***A*** ***B***; ***B*** ***D***] [***ε***<sub>0</sub>; ***κ***] = [***N***\*; ***M***\*]  (2)

  ***ε***<sub>res</sub>(*z*) = ***ε***<sub>0</sub> + *z* ***κ*** − ***ε***\*(*z*)  (3)

## 3. 実行
```bash
CPUS=4 bash abaqus/denso_plate/run_denso_plate.sh                 # 対称・非対称の2ケース
bash abaqus/denso_plate/run_denso_plate.sh --part thermal          # 熱の項だけ（任意）
bash abaqus/denso_plate/run_denso_plate.sh --part cryst            # 結晶化の項だけ（任意）
```
スクリプトは次の4段階を順に行う。
1. 入力ファイルと参照解を `abaqus/denso_plate/jobs/` に書き出す。
2. Abaqus で解く。
3. odb から抽出する（`abaqus python`）。
4. 参照解と照合し、`*_compare.png` を描く。

## 4. 判定と参照値

**合格基準**：次の2つを満たすこと。
- ひずみの誤差：max |FE − 参照| / max |参照| < 5×10<sup>−3</sup>
- そり量の差：max(5×10<sup>−3</sup> |参照|, 0.001 mm) 未満

**表1** CalculiX 2.21 による事前確認。同じ入力ファイルを `to_calculix.py` で変換した（キーワードの綴りのみ変更）。

| ケース | ひずみ誤差 | そり量 *y* [mm]（参照） | 判定 |
|---|---|---|---|
| `denso_symmetric` | 5.5×10<sup>−4</sup> | −0.0000（−0.0000） | PASS |
| `denso_asymmetric` | 6.9×10<sup>−4</sup> | −1.3619（−1.3618） | PASS |

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
  そこで全体座標と同じ向きの `*ORIENTATION` を定義し、各層の `*SOLID SECTION` から参照した（`eeb8a71`）。
  局所座標が全体座標と一致するので、剛性・荷重・出力の向きは変わらない。第 II 部の応力ジョブにも同じ定義を入れてある。
- 図：`abaqus/denso_plate/calculix_check/denso_asymmetric_calculix.png`

**示せたこと**：学習データの生成に使う積層理論は、同じ層構成の 3 次元固体要素（Abaqus）と、ひずみで 10<sup>−3</sup> 未満、そり量で 10<sup>−4</sup> mm の桁まで一致する。

---

# 第 II 部 成形プロセス解析（熱伝導 → 粘弾性応力の逐次解析）

## 5. 解く問題

同じ 1/4 板を、2 つのジョブで解く。節点・要素の番号は両ジョブで同一である。

**熱伝導ジョブ** `denso_heat_<case>`（DC3D8）
- 初期温度：380 °C。
- 次の3ステップを、固定増分で解く。

| ステップ | 時間 | 増分 | 熱境界条件 | 力学境界条件（応力ジョブ） |
|---|---|---|---|---|
| MOULD | 40 s | 0.05 s | 上下面を金型温度に固定（170 / 200 °C、対称ケースは 185 / 185 °C） | 外周面の面内変位を拘束（キャビティによる拘束）、厚さ方向は自由 |
| EJECT | 0.01 s | 0.01 s | 固定を外し、上下面に熱伝達（*h* = 10 W/(m² K)、25 °C） | 外周の拘束を外す（反力は増分内で 0 に戻る） |
| AIR | 1800 s | 5 s | 熱伝達のまま空冷 | 自由な板 |

**応力ジョブ** `denso_proc_<case>`（C3D8I ＋ UMAT `denso_cryst_umat.f`）
- 温度は熱伝導ジョブの odb から、ステップごとに読む（`*TEMPERATURE, FILE=...`）。
- UMAT の構成は次のとおりである。
  - 結晶化：Nakamura 則と Hoffman–Lauritzen の *K*(*T*) による。
  - 粘弾性：一般化 Maxwell 模型（Prony 級数 3 項 ＋ 平衡項）とし、緩和時間を温度（WLF）と結晶化度の両方でずらす。
  - 固有ひずみ：熱膨張と結晶化収縮を直交異方性で与える。
- 増分の式は次のとおりである。

  Δ*ε*\*<sub>*i*</sub> = *A<sub>i</sub>* Δ*T* + *B<sub>i</sub>* Δ*α*,  *B<sub>i</sub>* = ⟨*β*⟩<sub>*ii*</sub> *X*<sub>∞</sub>  (4)

  *τ*<sub>*k*</sub>\* = *a<sub>T</sub>*(*T*) *a<sub>X</sub>*(*α*) *τ<sub>k</sub>*,  log<sub>10</sub> *a<sub>T</sub>* = −*C*<sub>1</sub>(*T* − *T*<sub>ref</sub>) / (*C*<sub>2</sub> + *T* − *T*<sub>ref</sub>),  *a<sub>X</sub>* = 10<sup>*B*<sub>X</sub> *α*</sup>  (5)

- 定数（例示値）は次のとおりである。
  - Prony 級数：*g*<sub>∞</sub> = 0.01、*g<sub>k</sub>* = 0.50 / 0.30 / 0.19、*τ<sub>k</sub>* = 5 / 50 / 500 s。
  - WLF：*C*<sub>1</sub> = 17.4、*C*<sub>2</sub> = 51.6、*T*<sub>ref</sub> = 143 °C。
  - 結晶化度による移動：*B*<sub>X</sub> = 2。
- UMAT は、Abaqus 2024 で動作を確認済みの `abaqus/cfrtp_cryst_umat_hl.f` からの派生である。変更点は 1 箇所だけで、結晶化収縮を 3 方向別々に与えるようにした（親は一方向材用で、直交 2 方向に同じ値を与える）。

## 6. 実行

```bash
cd wccm2026-cfrp-gnn
CPUS=4 bash abaqus/denso_plate/run_denso_process.sh               # 対称・非対称の2ケース
bash abaqus/denso_plate/run_denso_process.sh asymmetric            # 1ケースだけ
```
各ケースで、次を順に行う。
1. 熱伝導ジョブを解く。
2. 応力ジョブを解く（`user=denso_cryst_umat.f`）。
3. 抽出する（`extract_denso_process.py`）。

最後に照合（`compare_denso_process.py`）を行う。

**時間の目安**：熱伝導は 1161 増分、応力も 1161 増分である（各増分は線形なので 1 回の反復で収束するはず）。
- 参考として CalculiX では、熱伝導の全メッシュが 1 増分あたり約 7 s だった。Abaqus での時間は未計測である。
- 長い場合は、スケジューラ（あれば）でジョブにする。中身は上の1行でよい。

**よくあるエラーと対処**

| 症状 | 対処 |
|---|---|
| `*TEMPERATURE` でファイルが見つからない | `FILE=denso_heat_<case>.odb` の `.odb` を外す。熱伝導ジョブの odb と同じディレクトリで実行しているかも確かめる |
| UMAT のコンパイルに失敗する | §1 の `abaqus verify -user_std` を確かめる。Fortran 環境の設定は管理者に確認する |
| 増分数の上限で止まる | 入力ファイルの `*STEP, INC=` を増やす。固定増分なので、必要数は 800 / 1 / 360 である |

## 7. 判定と参照値

`compare_denso_process.py` は、熱と応力を分けて照合する。

- **(H) 熱**：中心列の温度履歴を、1 次元有限要素の参照解と比べる。合格は、1 s 以降の最大差が 0.5 K 未満であることとする。
  - 金型に接した直後の数増分は、容量行列の積分法で差が出るので、合格判定から除く（最大差は表示される）。
- **(M) 応力**：Python の参照解を、**Abaqus 自身が出した温度で計算し直して**比べる。比べる量と合格基準は次のとおりである。
  - 応力：max |*σ*| の 2 % 未満。
  - 結晶化度：差 0.01 未満。
  - そり量：差が max(2 %, 0.01 mm) 未満。

**表3** 事前確認（Abaqus なし）

| 確認 | 結果 |
|---|---|
| UMAT（Fortran）と Python 移植の一致（gfortran で単点を駆動） | 応力 1.8×10<sup>−15</sup>、結晶化度 1.1×10<sup>−16</sup> |
| 弾性極限（*g*<sub>∞</sub> = 1、結晶化なし）と積層理論の一致 | ひずみ 1.8×10<sup>−15</sup>、曲率は 10 桁一致 |
| 熱と結晶化の重ね合わせ | 1.7×10<sup>−15</sup> |
| 熱伝導の入力ファイル（CalculiX、面内 1 × 1 に縮約、全 3 ステップ） | 1 s 以降の最大差 0.31 K、終了時の最高温度 25.24 °C（参照解も 25.24 °C） |
| 応力の入力ファイル | **未確認**（標準の CalculiX は UMAT を扱えない）。fifa での実行が最初の確認になる |

**表4** Python 参照解による予測（Abaqus での確認前、例示定数）

| ケース | 残留ひずみ *ε*<sub>res,*xx*</sub> の幅 [×10<sup>−3</sup>]（ロックイン模型） | そり量 *y* [mm]（ロックイン模型） |
|---|---|---|
| 対称 185 / 185 °C | 4.82（6.54） | 0.000（0.000） |
| 非対称 170 / 200 °C | 4.83（6.66） | −1.652（−1.362） |

Abaqus の結果が表4から大きく外れた場合は、パラメータを合わせにいかずに、`*.msg`、`*.dat` と照合の数値をそのまま報告すること。

## 8. Claude Code に頼む場合の指示文（fifa 上で渡す）

> リポジトリ wccm2026-cfrp-gnn の `research/GUIDE_fifa_abaqus.md` に従って、Abaqus の解析を実行してください。
> 1. §1 の準備を行う。Abaqus のコマンド名と版、`abaqus verify -user_std` の結果、python3 の numpy と matplotlib を確認する。
> 2. 第 I 部は実施済みなので飛ばしてよい（§4 表2）。入力ファイルの生成器を変えた場合だけ、§3 を回して表2と同じ結果になることを確かめる。
> 3. 第 II 部（§6）を実行し、`compare_denso_process.py` の (H) と (M) の出力、各ジョブの所要時間、`denso_proc_*_compare.png` を報告する。
>    あわせて、残留ひずみの幅とそり量を §7 の表4と並べる。
> 4. エラーや不合格が出た場合は、§6 の表で対処できるものは対処し、それ以外は `.msg` と `.dat` の該当箇所を添えて報告する。
>    合格させるために、材料定数・増分・許容値を変えないこと。
> 5. `jobs/` は git の管理外である。結果のファイル（`jobs/` 以下の `*_abaqus.json` と `*_compare.png`）を `abaqus/denso_plate/abaqus_results/` にコピーし、
>    ブランチを切ってコミットし、プッシュする。odb と inp は容量が大きいのでコミットしない。

---

# 第 III 部 穿孔法の模擬（既報の実験値との比較）

## 9. 解く問題

既報（津田・白桃・田中・國立・望月、日本複合材料学会誌 52(3) 88–96, 2026）の穿孔法を、**既報と同じ入力だけで**計算する。
既報の値は `data/literature/tsuda2026_jscm52_88.json` から読む。

**既報どおりの設定**
- **試験片**：20 × 20 × 3.08 mm、板厚方向は 20 µm × 154 層とする。各層の配向は既報 図8(a) の読み取り値で、深さは穴をあける面から測る。
- **材料**：Mori–Tanaka と配向平均で求める（`validation/denso_tsuda2026_compare.py`）。値は既報 表1 の PPS と GF のものを使う。
  - 剛性は 20 °C の値とする。熱ひずみは、瞬間線膨張係数を *T*<sub>0</sub> = 180 °C から 20 °C まで積分したものとする。
  - これは、既報の表面応力 −24.3 MPa を −24.4 MPa で再現した読み方である。
- **穴あけ**：中心に φ1.6 mm の穴を、0.2 mm ずつ 10 ステップで深さ 2.0 mm まであける（`*MODEL CHANGE, REMOVE`）。
  - 既報は、同じ要素の剛性を UMAT で 0 にしている。

**本リポジトリで置いた仮定**
- **1/4 モデル**：対称面 *x* = 0 と *y* = 0 で切る。配向の非対角成分（既報で 0.15 以下）を無視すると、各層は直交異方性になるので、1/4 モデルで厳密に解ける。
- **閉包近似**：ハイブリッドとした。既報には閉包近似の種類が書かれていない。
- **ゲージの寸法**：既報 図6 の注記（1.1 × 2.0、φ7.0）と、東京測器の FRAS-2 のカタログの値（ゲージ中心の直径 7.0 mm、格子 2.0 × 1.1 mm）による。
  - 半径方向に長さ 2.0 mm、幅 1.1 mm の格子を、*r* = 2.5〜4.5 mm に置いた。
  - カタログの値は検索結果の抜粋で確認したもので、公式サイトは確認時に保守中だった。
  - Ch.3 は MD（*x*）、Ch.1 は TD（*y*）、Ch.2 は 45° 方向である。
- **解放ひずみ**：各ゲージの範囲にある表面の層の要素について、穴あけ後のひずみから冷却後のひずみを引き、面積重み付きで平均する。

## 10. 実行

```bash
cd wccm2026-cfrp-gnn
CPUS=8 bash abaqus/denso_plate/run_hole_drilling.sh          # 本番（154 層、約 13 万要素）
bash abaqus/denso_plate/run_hole_drilling.sh coarse          # 確認用（77 層、約 3.4 万要素）
```
処理は次の4段階である。
1. 入力ファイル `jobs/denso_hole.inp` と `*_meta.json` を作る。
2. Abaqus で解く（11 ステップ）。
3. `extract_hole_drilling.py` で、各ステップのゲージ要素のひずみを抽出する。
4. `compare_hole_drilling.py` で、既報 図12 と比べる。

出力は `*_released.json` と `*_released.png` である。

## 11. 判定と参照値

**比べる量**：Ch.1〜3 の解放ひずみと深さの関係を、既報の**実測値**と既報の**解析値**（どちらも図12 の読み取り）と比べる。

**合否の基準は置かない。**この計算は、合わせるためのものではなく、比べるためのものだからである。報告するのは次の3点である。
- ピーク値の比。本計算／実測、本計算／既報の解析。
- ピークの深さ。既報では 0.8 mm（スキン層の厚さ）である。
- 符号が一致するか。

**既報の結果**：解析は実測より約 40 % 大きい（既報 図12）。本計算の表面応力は既報の解析とほぼ同じなので、同程度になると予想する。

**表5** 事前確認（CalculiX 2.21、確認用の粗いメッシュ）
確認用の粗いメッシュ（77 層、448 要素／層）で計算した。図：`results/denso_tsuda2026/hole_drilling_coarse_calculix.png`。

| ゲージ | 本計算のピーク [µε]（深さ） | 既報の実測 | 既報の解析 | 本計算／実測 | 本計算／既報の解析 |
|---|---|---|---|---|---|
| Ch.1（TD） | −270（1.0 mm） | −170 | −225 | 1.59 | 1.20 |
| Ch.3（MD） | +307（0.8 mm） | +195 | +272 | 1.57 | 1.13 |

- 符号はすべてのゲージで実測と一致した。ピークの深さも 0.8〜1.0 mm で、実測と一致した。
- 大きさは実測の約 1.6 倍、既報の解析の約 1.1〜1.2 倍である。既報の解析は実測の約 1.3〜1.4 倍なので、本計算は既報の解析と同じ向きにさらに少し大きい。
- 粗いメッシュではゲージ内の要素が少ない（Ch.1・Ch.3 で 7 要素）。本番のメッシュで値が変わりうる。
**表6** 本番の結果（fifa、Abaqus 2024、2026-11-03、コミット 5989ec5）

| ゲージ | 本番メッシュ [µε]（深さ） | 粗いメッシュ（Abaqus） | 粗いメッシュ（CalculiX、表5） | 実測 | 既報の解析 | 本番／実測 | 本番／既報の解析 |
|---|---|---|---|---|---|---|---|
| Ch.1（TD） | −261（1.0 mm） | −272 | −270 | −170 | −225 | 1.54 | 1.16 |
| Ch.3（MD） | +299（0.8 mm） | +309 | +307 | +195 | +272 | 1.54 | 1.10 |

- 同じ粗いメッシュで、Abaqus と CalculiX の差は 1 % 以内であった。
- 本番のメッシュにすると、ピークは約 3 % 小さくなった。実測より約 1.5 倍大きいという結論は変わらない。
- 図：`abaqus/denso_plate/abaqus_results/denso_hole_released.png`。

- ゲージの位置を初め *r* = 1.5〜3.5 mm と読み違えたときは、約 4〜5 倍になった。解放ひずみは穴からの距離で急に小さくなるので、ゲージの位置が結果を強く左右する。

## 12. Claude Code に頼む場合の指示文（第 III 部）

> リポジトリ wccm2026-cfrp-gnn の `research/GUIDE_fifa_abaqus.md` 第 III 部に従って、穿孔法の模擬を実行してください。
> 1. `git pull` の後、§10 の確認用 `coarse` を先に回し、エラーなく 11 ステップ終わることを確かめる。そのうえで本番を回す。
> 2. `compare_hole_drilling.py` の出力表、Ch.1 と Ch.3 のピークの比、`denso_hole_released.png`、所要時間を報告する。
> 3. 合わせるために材料定数・配向・ゲージ寸法・メッシュを変えないこと。エラーは `.msg` と `.dat` の該当箇所を添えて報告する。
> 4. 結果（`*_abaqus.json`、`*_released.json`、`*_released.png`）を `abaqus/denso_plate/abaqus_results/` にコピーし、ブランチを切ってコミットし、プッシュする（`.odb` と `.inp` は入れない）。

## 13. 追加の計算（2026-11-03）：転移学習用の成分別の本番計算

**目的**：第 III 部の本番に加えて、固有ひずみの成分を 1 つだけ残した計算（xx、yy、zz の 3 本）を本番メッシュで回す。
解放ひずみは固有ひずみに線形なので、3 本の和は本番の結果と一致するはずである。
この 3 本を使い、既報の実測（図 12）への補正（`validation/denso_tsuda2026_transfer.py`）を、粗いメッシュから本番メッシュへ置き換える。

**順序**：第 II 部の非対称ケース → 第 III 部の本番 → 本節。本番の結果が先に要る（重ね合わせの確認に使う）。

```bash
git pull
CPUS=8 bash abaqus/denso_plate/run_hole_drilling.sh basis     # denso_hole_{xx,yy,zz}、本番と同じメッシュで 3 ジョブ
```

**判定**：
- 3 本とも 11 ステップ終わること。
- 重ね合わせの誤差：次を実行し、1 行目の `[check] superposition` が 10<sup>−4</sup> 未満であること。

```bash
cp abaqus/denso_plate/jobs/denso_hole*_released.json abaqus/denso_plate/abaqus_results/
python3 validation/denso_tsuda2026_transfer.py --basis-dir abaqus/denso_plate/abaqus_results --prefix denso_hole \
    --full abaqus/denso_plate/abaqus_results/denso_hole_released.json
```

参考（粗いメッシュ、CalculiX、例示）：重ね合わせ誤差 7.6×10<sup>−7</sup>、補正の共通倍率 0.65（*T*<sub>0,eff</sub> ≈ 124 °C）。
本番メッシュで値が変わっても、合わせにいかずにそのまま報告すること。

**報告と保存**：上のスクリプトの全出力、`results/denso_tsuda2026/transfer_hole_drilling.{json,png}`、各ジョブの所要時間。
`denso_hole_{xx,yy,zz}_{abaqus,released}.json` を `abaqus/denso_plate/abaqus_results/` に入れ、第 III 部と同じブランチにコミット・プッシュする（`.odb` と `.inp` は入れない）。

**Claude Code に渡す指示文**：

> `research/GUIDE_fifa_abaqus.md` §13 に従い、第 III 部の本番が終わった後に `run_hole_drilling.sh basis` を回してください。
> 重ね合わせの誤差と、`denso_tsuda2026_transfer.py` の全出力を報告してください。定数・メッシュは変えないこと。

## 14. 追加の計算（2026-11-03）：ゲージ位置の感度

**目的**：本番の解放ひずみは実測の約 1.54 倍であった（表6）。解放ひずみは穴からの距離で急に小さくなるので、
ゲージの位置のわずかな違いで値が変わる。ゲージ（FRAS-2）の位置はカタログの抜粋から読んだ値であり、
貼付位置のばらつきもありうる。ゲージを径方向に ±0.2 mm ずらした計算で、1.54 倍のうち位置で説明できる量を見積もる。

```bash
git pull
CPUS=8 bash abaqus/denso_plate/run_hole_drilling.sh gauge     # denso_hole_gm02（内側へ 0.2 mm）、denso_hole_gp02（外側へ 0.2 mm）
```

メッシュ・材料・配向は本番と同じで、ゲージに含める要素だけが変わる（本番メッシュでの要素数は Ch.1／Ch.3 で 13・12・12 個、−0.2・0・+0.2 mm）。

**判定**：2 本とも 11 ステップ終わること。結果の数値に合格の基準はない（感度を見る計算である）。

参考（粗いメッシュ、CalculiX、ここでの事前確認）：内側へ 0.2 mm で Ch.3 +359（実測の 1.84 倍）、Ch.1 −329（1.93 倍）。
外側へ 0.2 mm では、粗いメッシュではゲージに入る要素が変わらず、ずらす前と同じ値（1.57・1.59 倍）になった。
外側の感度は、要素の細かい本番メッシュで見る必要がある。

**表7** 本番の結果（fifa、Abaqus 2024、2026-11-03、コミット eb9fb7c）

| ゲージの位置 | Ch.3（MD）ピーク [µε] | 実測比 | Ch.1（TD）ピーク [µε] | 実測比 |
|---|---|---|---|---|
| 内側へ 0.2 mm | +409 | 2.10 | −379 | 2.23 |
| ずらさない（表6） | +299 | 1.54 | −261 | 1.54 |
| 外側へ 0.2 mm | +299 | 1.54 | −261 | 1.54 |

- 内側へ 0.2 mm ずらすと、解放ひずみは約 4 割大きくなった。
- 外側へ 0.2 mm では、本番メッシュでもゲージに入る要素が変わらず、値は同じであった。
  ゲージは要素の中心で選んでおり、ゲージ外縁付近の要素が径方向に 0.2 mm より大きいためである。
  **外側の感度はこの方法では求められていない**。求めるには、要素とゲージの重なり面積で重み付けするか、ゲージ付近のメッシュを細かくする必要がある。

**報告**：2 本それぞれの `compare_hole_drilling.py` の出力（ピークと実測比）、所要時間。
`denso_hole_g{m02,p02}_{abaqus,released}.json` と `_released.png` を `abaqus/denso_plate/abaqus_results/` に入れ、
ブランチを切ってコミット・プッシュする（`.odb` と `.inp` は入れない）。

**Claude Code に渡す指示文**：

> `research/GUIDE_fifa_abaqus.md` §14 に従い、`run_hole_drilling.sh gauge` を回してください。
> 2 本のピークと実測比を表にして報告してください。定数・メッシュは変えないこと。

## 15. 追加の計算（2026-11-03）：重なり面積によるゲージ位置の感度

**目的**：§14 では、ゲージに含める要素を中心位置で選んだため、外側へのずれを解像できなかった（表7）。
本節では、ゲージ付近の要素（ゲージを 1.0 mm 広げた範囲、本番メッシュで 258 要素）のひずみも出力し、
要素とゲージの**重なり面積で重み付けして平均**する（`abaqus/denso_plate/gauge_overlap.py`）。
1 本の計算で、−0.4〜+0.4 mm の任意のずれを評価できる。

```bash
git pull
CPUS=8 bash abaqus/denso_plate/run_hole_drilling.sh overlap   # denso_hole_zone（本番メッシュ、1 ジョブ）
```

**判定**：
- 11 ステップ終わること。
- `compare_hole_drilling.py` の出力が表6（本番）と同じであること（メッシュと材料は同じで、出力する要素が増えただけ）。
- `gauge_overlap.py` の面積の確認欄が、すべてのずれで 1.1, 1.1, 2.2 mm² であること（REJECTED が出ないこと）。

**報告**：`gauge_overlap.py` の出力表、所要時間。`denso_hole_zone_{abaqus,released,overlap}.json` を
`abaqus/denso_plate/abaqus_results/` に入れ、ブランチを切ってコミット・プッシュする（`.odb` と `.inp` は入れない）。

**Claude Code に渡す指示文**：

> `research/GUIDE_fifa_abaqus.md` §15 に従い、`run_hole_drilling.sh overlap` を回してください。
> `gauge_overlap.py` の出力表をそのまま報告してください。定数・メッシュは変えないこと。

**表8** 本番の結果（fifa、Abaqus 2024、2026-11-03、コミット 0cedc22）。ゲージで平均した解放ひずみのピーク [µε] と実測比

| ずれ [mm] | Ch.3（MD） | 実測比 | Ch.1（TD） | 実測比 |
|---|---|---|---|---|
| −0.4 | 453 | 2.32 | 437 | 2.57 |
| −0.2 | 378 | 1.94 | 350 | 2.06 |
| 0 | 322 | 1.65 | 287 | 1.69 |
| +0.2 | 275 | 1.41 | 236 | 1.39 |
| +0.4 | 231 | 1.19 | 191 | 1.12 |

- 判定は 3 つとも満たした（11 ステップ完了、`compare_hole_drilling.py` が表6と一致、面積の確認はすべてのずれで 1.1・1.1・2.2 mm²）。
- ずれ 0 の値（1.65・1.69 倍）は、要素の中心で選んだ表6（1.54 倍）より大きい。重なり面積による平均のほうがゲージの形に忠実なので、こちらを基準とする。
- 外側へ 0.1 mm ずれるごとに、実測比は約 0.11〜0.14 下がる。外側へ 0.4 mm ずれていても 1.1〜1.2 倍であり、**ゲージ位置のずれだけでは実測との差を説明できない**。
- 同じ方法を粗いメッシュの CalculiX で試したところ、冷却ステップの結果がスレッド数（2 と 4）で 1〜2 % 変わった。Abaqus の本番の結果には影響しない。

## 16. 追加の計算（2026-11-03）：穿孔法の影響関数（深さ方向の層ごと）

**目的**：既報以外の配向・成形条件でも、予測した固有ひずみの分布から解放ひずみ（実際に測る量）を計算できるようにする。
深さを 0.2 mm ごとの 16 層に分け、1 つの層だけに単位の固有ひずみ（1×10<sup>−3</sup>、xx または yy）を与えた計算を行う。
解放ひずみは固有ひずみに線形なので、任意の分布の解放ひずみは、層ごとの応答の重ね合わせで求まる（`abaqus/denso_plate/influence_check.py` の式(1)）。
剛性は既報の配向のままなので、配向が大きく違う場合は近似になる。

```bash
git pull
CPUS=8 bash abaqus/denso_plate/run_hole_drilling.sh influence    # 34 ジョブ（xx・yy × 16 層 ＋ 全深さ 2 本）、本番メッシュ
```

- §15 の 1 ジョブが約 3 分だったので、全体で約 2 時間の見込み。途中で止まっても、終わったジョブは飛ばして再開する。
- ディスク節約のため、各ジョブの `.odb` は解放ひずみを取り出した後に削除する（スクリプト内）。
- 最後に `influence_check.py` が自動で走る。

**判定**：
- 34 ジョブとも 11 ステップ終わること。
- I1（線形性）：全深さに同じ単位の膨張を与えた計算は、自由な板の一様な膨張なので応力が生じず、解放ひずみは厳密に 0 になる。16 層の和もこれに打ち消し合うこと（差が最大の層の応答の 10<sup>−4</sup> 未満）。
- **結果（2026-11-04、fifa、コミット f222cea）**：I1 は xx 2.3×10<sup>−7</sup>、yy 1.2×10<sup>−7</sup>（合格。全深さの計算の解放ひずみは 10<sup>−10</sup> µε）。I2 は 8.3×10<sup>−3</sup>（0.2 mm の層平均による誤差 0.8 %）。最初の版の照合スクリプトは 0 で割って 3×10<sup>4</sup> と表示していた（照合側の誤り、計算は正しい）。
- I2（層平均の誤差）：既報の固有ひずみを層ごとに平均して重ね合わせた値と、本番の計算（表6）との差。合格の基準はなく、数値をそのまま報告する（数 % なら層の分け方で十分）。

**報告と保存**：`influence_check.py` の出力全文、所要時間。`jobs/influence_operator.json` と、各ジョブの `*_released.json` を
`abaqus/denso_plate/abaqus_results/influence/` に入れ、ブランチを切ってコミット・プッシュする（`.odb` と `.inp` は入れない）。

**Claude Code に渡す指示文**：

> `research/GUIDE_fifa_abaqus.md` §16 に従い、`run_hole_drilling.sh influence` を qsub で回してください（約 2 時間）。
> `influence_check.py` の出力（I1、I2）をそのまま報告してください。定数・メッシュは変えないこと。

## 17. 追加の作業（2026-11-03）：オープンソースの射出成形解析 openInjMoldSim で繊維配向を計算する

**目的**：既報の試験片（80×80 mm、厚さ 3.08 mm、フィルムゲート、樹脂 320 °C、金型 150 °C）の充填を 3 次元で計算し、
**ゲートから 50 mm の位置**の繊維配向の板厚方向分布を、既報 図8a（`data/literature/tsuda2026_jscm52_88.json` の `fig8_orientation_3mm`）と比べる。
本リポジトリの 1 次元モデル（`validation/denso_flow_orientation.py`）は、芯の層の厚さが 0.92 mm 以下にしかならず、既報の 1.60 mm を再現できなかった。

**ソルバ**：openInjMoldSim（OpenFOAM の compressibleInterFoam を改造した充填・保圧・冷却のソルバ。繊維配向は openInjMoldSimF、Folgar–Tucker 型）。
OpenFOAM の一部ではない。**元の GitHub リポジトリ、ライセンス、対応する OpenFOAM の版をまず確認すること**（本リポジトリ側では gitee のミラーしか見ていない）。

**段階**（各段階の結果を報告してから次に進む）：
1. **導入の可否**：fifa に必要な版の OpenFOAM を入れられるか（管理者権限、ネットワーク、ディスク）。openInjMoldSim をビルドできるか。できなければ、その理由を報告して止める。
2. **付属の例題**：openInjMoldSim の例題（繊維配向を含むもの）が動くか。所要時間。
3. **既報の板**：
   - 形状：板 80 × 80 × 3.08 mm、板の 1 辺の全幅をフィルムゲートとする（ゲート部の厚さは既報に記載がないので、板厚と同じとし、仮定と明記）。
   - 板厚方向に 30 要素以上。
   - 条件：樹脂温度 320 °C、金型温度 150 °C（上下同じ）。充填時間は既報に記載がないので 1 s と仮定する。
   - 材料：PPS/GF30 の粘度（Cross–WLF など）と PVT は、openInjMoldSim の例題の値か文献値を使い、**出典を書く**。見つからなければ例示値として明記する。繊維のアスペクト比 16.8、*C*<sub>i</sub> は例題の値。
   - 出力：ゲートから 50 mm、幅の中央の位置での *a*<sub>xx</sub>、*a*<sub>yy</sub>、*a*<sub>zz</sub> の板厚方向分布（CSV、深さは上面から）。
4. **比較**：既報 図8a と重ねた図と、芯の層（*a*<sub>yy</sub> > *a*<sub>xx</sub>）の厚さ（既報 1.60 mm）。

**守ること**：既報に合わせるために *C*<sub>i</sub> や材料定数を調整しないこと。合わなければそのまま報告する。仮定した値はすべて一覧にする。

**報告と保存**：各段階の結果、所要時間、CSV と比較の図。`abaqus/denso_plate/abaqus_results/flow/` に入れ、ブランチを切ってコミット・プッシュする（OpenFOAM のケースの大きな時刻ディレクトリは入れない）。

**Claude Code に渡す指示文**：

> `research/GUIDE_fifa_abaqus.md` §17 に従い、openInjMoldSim の導入を試してください。段階 1 の結果（導入できるか、できないならその理由）をまず報告してください。定数を合わせにいかないこと。

## 18. 追加の作業（2026-11-04）：3 次元部品の公開データの中身の確認（RESCUE の 3 次元版の準備）

**目的**：本研究の補正（RESCUE）を 3 次元の部品で試すため、Greif & Meyer の公開データ（Moldflow の計算結果）の中身を確かめる。計画は `research/PLAN_rescue_3d.md`。**この段階では計算はしない。**

**データ**：Greif & Meyer, "Injection Molding Simulations with Artificially Enhanced Warpage", Zenodo 17831586、CC BY 4.0、196 形状、`data.zip` 2.03 GB。
PP/GF30（Celanese Factor PP GF 30）、Moldflow 2023、保圧なし。各節点に反りの変位と繊維配向テンソル、各要素に金型内の残留応力と弾性定数がある。

```bash
mkdir -p ~/data/greif2025_warp && cd ~/data/greif2025_warp
curl -L -o data.zip "https://zenodo.org/api/records/17831586/files/data.zip/content"
unzip -l data.zip | head -30; unzip -q data.zip
python3 -m pip install --user meshio        # VTU の読み込み
```

**確かめて報告すること**（5 形状程度で十分）：
1. ファイルの構成（VTU、LOG、YML の数と名前の付け方）。
2. 1 つの VTU について：要素の種類（四面体など）と数、節点の数、座標の単位（mm か m か）、形状の大きさ。
3. 節点と要素のフィールドの名前の一覧。特に、反り（`Deflection,_all_effects`）、繊維配向テンソル（`Fiber_orientation_tensor_on_nodes_(3D)`）、弾性定数（`Tensile_modulus_...`、`Poisson's_ratio_...`、`Shear_modulus_...`）、金型内の残留応力（`In-cavity_residual_stress_...`）の成分の並びと単位。
4. 反りの大きさ（最大値）の分布を、196 形状について一覧にする（形状の大きさで割った値も）。
5. 板厚方向の要素の数（薄い形状で 1〜2 要素しかないか）。

**守ること**：データを書き換えないこと。結果は `abaqus/denso_plate/abaqus_results/greif3d/` に、報告（Markdown）と小さな一覧（CSV）だけを入れ、ブランチを切ってコミット・プッシュする。VTU や zip はリポジトリに入れない。

**Claude Code に渡す指示文**：

> `research/GUIDE_fifa_abaqus.md` §18 に従い、Zenodo 17831586 のデータ（2 GB）を落として、中身（ファイル構成、フィールド名と単位、要素の種類、反りの大きさの分布、板厚方向の要素数）を報告してください。計算はまだしないこと。

## 19. 追加の計算（2026-11-04）：3 次元部品での RESCUE（簡単な物理モデル＋少数点の補正）

**目的**：§18 で確認した Greif & Meyer のデータ（196 形状）で、平板と同じ RESCUE を 3 次元の部品で試す。
データの繊維配向から本研究の簡単な物理モデルで反りを計算し（事前の予測）、データの Moldflow の反りの**数点だけ**を「計測」とみなして補正する。残りの全節点の反りがどこまで当たるかを調べる。計画は `research/PLAN_rescue_3d.md`。**GPU は使わない**（CalculiX、CPU）。

**スクリプト**：`validation/greif3d_rescue.py`（docstring に式と正直な限界）。
- 事前の予測：配向テンソル → k-means で 32 群 → 群ごとに Mori–Tanaka＋配向平均（混合近似）で剛性と熱膨張係数 → CalculiX の線形弾性 FE（データの四面体に中間節点を足した C3D10、3-2-1 で保持、一様な温度変化 −100 K）。熱膨張係数を等方部分と偏差部分に分けた 2 ケースを解く（重ね合わせ）。
- 補正：等方・偏差の 2 倍率（または共通の 1 倍率）と剛体運動 6 自由度を、K 個の表面節点（5／10／20、無作為に 10 回）の 3 成分で当てはめる（L2-SP、事前値 1、*s*<sub>a</sub> = 0.5、計測の不確かさ 0.01 mm は例示）。
- 評価：全節点で、予測とデータの差から最適な剛体運動を除いた rms を、データの反り（剛体運動を除く）の rms で割った値（相対誤差）。
- 材料定数は **PP/GF30 の一般値**（PP 1.5 GPa・0.40・100 ppm/K、GF 72 GPa・0.22・5 ppm/K、*V*<sub>f</sub> 0.131、*s* 20 は仮定）。例示値であり、校正していない。

**ここでの自己チェック（2026-11-04、合成の板、`results/greif3d_rescue/selftest.json`）**：
- G1 パッチテスト（等方・一様な膨張）：6.8×10⁻¹²。
- G2 合成データ（0.7 *U*<sub>iso</sub> + 1.3 *U*<sub>dev</sub> + 剛体運動 + 雑音）：10 節点の 2 倍率で 0.704／1.298 を回復、相対誤差 0.37 → 0.021。
- G3 C3D4 は曲げで固くなる（反りが C3D10 の 0.42 倍）。**C3D10 を既定とする**。G3b：C3D10 は厚さ方向 4 → 8 要素で 1.7 % の変化。

**準備**：
```bash
cd ~/wccm2026-cfrp-gnn && git pull
which ccx || conda install -y -c conda-forge calculix     # CalculiX が無ければ（conda 環境に）
python3 -m pip install --user meshio                      # §18 で入れた venv でもよい
python3 validation/greif3d_rescue.py --selftest --out /tmp/g3d_selftest   # G1〜G3b がここと同じ値になること
```

**実行**（データは §18 の `~/data/greif2025_warp/`）：
1. 小さい 20 形状（四面体の数で昇順）。うち最初の 3 形状で群の数の確認（G5、K = 64）も行う。
   ```bash
   python3 validation/greif3d_rescue.py --data ~/data/greif2025_warp --smallest 3 --k-check 64 --threads 4 --out ~/g3d_k
   python3 validation/greif3d_rescue.py --data ~/data/greif2025_warp --smallest 20 --threads 4 --out ~/g3d_20
   ```
2. 1 形状あたりの時間が 10 分以下なら、全 196 形状（`--smallest 196`）。それより長ければ、60 形状で止めて報告する。
   メモリが足りずに落ちる形状は飛ばしてよい（`--ids` で個別に指定できる）。落ちた形状の id と四面体の数を報告に書く。

**報告すること**：
- 自己チェック G1〜G3b の値（ここと一致するか）。
- G5：K = 32 と 64 の差（3 形状）。G4：事前の予測の主方向の弾性率と、データの `Tensile_modulus_in_first_principal_direction` の比（中央値）。
- `summary.json` の中央値と四分位：prior、cos_prior_data、field_1f／field_2f（場全体で当てはめたときの限界）、rescue_{5,10,20}_{1f,2f}。
- 補正がよく効く形状と効かない形状の例を 2〜3 個ずつ（id、大きさ、反りの最大値、cos）。

**判定の目安**（失敗したら、定数を合わせにいかずに失敗として報告する）：
- G1 が 10⁻⁸ を超える、または G2 で倍率を回復できない → 実行環境の問題。そこで止めて報告する。
- cos_prior_data が負や 0 付近の形状が多い → 事前の予測の形が合っていない（物理モデルの限界）。それ自体が結果なので、そのまま報告する。

**守ること**：データを書き換えないこと。結果は `abaqus/denso_plate/abaqus_results/greif3d_rescue/` に `per_geometry.csv`、`per_geometry.json`、`summary.json` と報告（Markdown）だけを入れ、ブランチを切ってコミット・プッシュする。`work/` と VTU はリポジトリに入れない。

**Claude Code に渡す指示文**：

> `research/GUIDE_fifa_abaqus.md` §19 に従い、`validation/greif3d_rescue.py` で自己チェックをしてから、Greif & Meyer の形状（まず小さい 20 個、時間が許せば全部）で RESCUE の 3 次元版を計算し、結果を報告してください。GPU は使いません。定数は合わせにいかないこと。
