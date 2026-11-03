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
