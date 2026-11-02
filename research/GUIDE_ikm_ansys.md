# 指示書（IKM）：Ansys による残留ひずみモデルの照合

> **対象サーバ**：IKM の Ansys が使える計算機。
> **対象**：`ansys/denso_plate/` 以下。照合スクリプトは Abaqus 版と共通（`abaqus/denso_plate/compare_denso_plate.py`）。
> **関連**：同じ問題を fifa の Abaqus 2024 で解いた結果がある（[`GUIDE_fifa_abaqus.md`](./GUIDE_fifa_abaqus.md) 表2、6 ケースすべて合格）。
> 定数はすべて例示値（未校正）である。

## 0. 何をするか

学習データの生成に使っている積層理論を、**Ansys の 3 次元固体要素**で確かめる。Abaqus で行った第 I 部と同じ問題である。
デンソー様は Ansys を使うので、Ansys で同じ答えが出ることを先に押さえておく。

| 順 | 内容 | 実行するもの | 状態 |
|---|---|---|---|
| 1 | 弾性照合（Abaqus 第 I 部と同じ問題） | `run_denso_plate_ansys.sh` | **これから**。Ansys では未実行 |
| 2 | 成形プロセス解析（熱伝導 → 粘弾性） | — | **未作成**。Abaqus の UMAT を Ansys の USERMAT に移す必要がある |

## 1. 解く問題

Abaqus 版と、節点・要素の番号まで同じである。

- **形状**：100 mm × 100 mm × 3 mm の自由な平板。対称性を使って 1/4 を解く。
- **要素**：SOLID185、`KEYOPT(2) = 3`（簡略化した拡張ひずみ。Abaqus の C3D8I に相当）。分割は面内 25 × 25、板厚方向 40 層。
- **材料**：層ごとに直交異方性の材料を与える（`MP,EX … GXZ`）。値は、配向平均した剛性 ⟨***C***⟩ から求めた。
  - ポアソン比は主ポアソン比（`PRXY`, `PRYZ`, `PRXZ`）で与える。
  - 材料の軸は全体座標と同じなので、要素座標系（`ESYS`）は使わない。
- **固有ひずみ**：等価な熱膨張係数 `ALPX/ALPY/ALPZ` = *ε*\* / Δ*T* として与える。`TREF` = 143 °C から、板全体を一様に 25 °C にする（`BFUNIF`）。
- **出力**：次の値を CSV に書き出す（`*VWRITE`）。
  - 中心列の要素の弾性ひずみ `EPEL X, Y`。
  - 中央面の節点の `UZ`。

## 2. 準備（初回のみ）

```bash
git clone https://github.com/keisuke58/wccm2026-cfrp-gnn.git   # 既にあれば: cd wccm2026-cfrp-gnn && git pull
cd wccm2026-cfrp-gnn

# Ansys（MAPDL）のコマンド名を調べる。版によって名前が違う（例：mapdl, ansys241, ansys242）
which mapdl ansys241 ansys242 ansys251 2>/dev/null

# 入力ファイル生成用の Python（numpy と matplotlib）
python3 -c "import numpy, matplotlib; print('ok')" || python3 -m pip install --user numpy matplotlib
```

## 3. 実行

```bash
ANSYS=<§2で見つけたコマンド名> NP=4 bash ansys/denso_plate/run_denso_plate_ansys.sh
```
処理は次の3段階である。
1. `ansys/denso_plate/jobs/` に入力ファイル（`*.dat`）と参照解（`*_ref.json`）を書き出す。
2. Ansys をバッチで実行する（`-b -i <job>.dat -o <job>.out`）。
3. 参照解と照合し、`*_compare.png` を描く。

熱の項・結晶化の項だけを解く場合は、`--part thermal` または `--part cryst` を付ける。

| 環境変数 | 既定値 | 意味 |
|---|---|---|
| `ANSYS` | `mapdl` | Ansys のコマンド名 |
| `NP` | 4 | 並列数（ライセンスの範囲で） |
| `PY` | `python3` | 生成と照合に使う Python |

手で 1 ケースずつ回す場合：
```bash
python3 ansys/denso_plate/gen_denso_plate_ansys.py
cd ansys/denso_plate/jobs
mapdl -b -np 4 -j denso_asymmetric -i denso_asymmetric.dat -o denso_asymmetric.out
python3 ../../../abaqus/denso_plate/compare_denso_plate.py --ansys --fig denso_asymmetric
```

## 4. 判定と参照値

**合格基準**（Abaqus 版と同じ）：次の2つを満たすこと。
- ひずみ誤差：max |FE − 参照| / max |参照| < 5×10<sup>−3</sup>
- そり量の差：max(5×10<sup>−3</sup> |参照|, 0.001 mm) 未満

**表1** 参照値（同じ問題の他のソルバでの結果）

| ケース | 参照解のそり量 *y* [mm] | Abaqus 2024（fifa） | CalculiX 2.21 |
|---|---|---|---|
| `denso_symmetric` | −0.0000 | ひずみ誤差 5.5×10<sup>−4</sup>、PASS | 5.5×10<sup>−4</sup>、PASS |
| `denso_asymmetric` | −1.3618 | ひずみ誤差 6.9×10<sup>−4</sup>、そり −1.3619 mm、PASS | 6.9×10<sup>−4</sup>、−1.3619 mm、PASS |

Ansys でも同程度（誤差 10<sup>−3</sup> 未満）になるはずだが、**未確認である**。要素の定式化が違うので、数値が完全に一致する必要はない。

**事前確認（Ansys なし）**
- 材料定数 `MP` から層の剛性を組み直すと、⟨***C***⟩ と 4.1×10<sup>−16</sup> で一致した。
- 照合の経路（CSV の読み込みから判定まで）は、参照解から作った CSV で 1.7×10<sup>−16</sup>、PASS を確かめた。

## 5. よくあるエラーと対処

| 症状 | 対処 |
|---|---|
| `mapdl: command not found` | §2 でコマンド名を調べ、`ANSYS=` で指定する |
| ライセンスが取れない | `NP` を減らす。混む時間を避ける。利用規定は管理者に確認する |
| CSV ができない | `<job>.out` の末尾で `*VWRITE` や `*GET` のエラーを確かめる。ジョブは `jobs/` の中で実行すること（CSV はそこに書かれる） |
| 照合が FAIL | パラメータを合わせにいかずに、`.out` と照合の数値をそのまま報告する |

## 6. 次の段階（未作成）

成形プロセス解析（熱伝導 → 結晶化を連成した粘弾性応力）の Ansys 版には、Abaqus の UMAT（`abaqus/denso_plate/denso_cryst_umat.f`）を、Ansys の USERMAT（`usermat.F`）に移す必要がある。
主な違いは次の2点である。
- 引数の並び。
- せん断成分の順番（Abaqus は 12, 13, 23、Ansys は xy, yz, xz）。

熱伝導の部分は、Ansys の標準機能（過渡熱解析 → 温度を構造解析に読み込む）で作れる。

## 7. IKM 上の Claude Code に頼む場合の指示文

> リポジトリ wccm2026-cfrp-gnn の `research/GUIDE_ikm_ansys.md` に従って、Ansys による照合を実行してください。
> 1. §2 の準備を行い、Ansys（MAPDL）のコマンド名と版、python3 の numpy と matplotlib を確認する。
> 2. §3 の一括実行を行い、`compare_denso_plate.py` の出力を §4 の表1と並べた表にまとめる。`*_compare.png` も確認する。
> 3. 余裕があれば `--part thermal` と `--part cryst` も実行する。
> 4. エラーが出た場合は、§5 の表で対処できるものは対処し、それ以外は `.out` の該当箇所を添えて報告する。合格させるために、材料定数・メッシュ・許容値を変えないこと。
> 5. `jobs/` は git の管理外である。結果のファイル（`*_ansys_*.csv`、`*_compare.png`、照合の出力）を `ansys/denso_plate/ansys_results/` にコピーし、ブランチを切ってコミットし、プッシュする。`.dat`、`.rst` などの大きいファイルはコミットしない。
