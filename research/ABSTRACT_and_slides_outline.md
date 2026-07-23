# アブストラクト＋スライド構成: 弱形式FEM演算子学習＋オンライン適応

> 本リポジトリの半導体デバイスシミュレーション向け研究ラインを、発表（会議/ゼミ）で使える形に
> まとめた草稿。実体は5デモ＋コンセプトノート（`CONCEPT_weakform_operator_learning_semiconductor.md`）。

---

## 1. アブストラクト（日本語, ~250字）

半導体デバイス解析（TCAD）では、ポアソン方程式とドリフト拡散方程式の自己整合ループを
巨大メッシュ上で反復するため、最内で頻繁に呼ばれるポアソン解が計算律速となる。本研究は、
この解法を**弱形式（Galerkin 有限要素）を核とした演算子学習**で加速する枠組みを提案する。
既存の強形式（有限差分・PINN）ベース手法と異なり、物理損失を組立済み FE 残差 K(ε)φ−Mρ と
することで、材料界面のフラックス連続を内在的に満たし、可変誘電率・非構造メッシュに頑健化する。
さらに、FE の a-posteriori 誤差評価を「メッシュ細分化」と「オンライン追学習トリガ」に統一し、
FEM を精度の権威に残したまま Newton 反復（＝高コストな疎行列解）を削減する。GAA/CFET 断面の
非線形ポアソンで、弱形式が界面誤差を解消し（強形式は頭打ち）、学習初期値により Newton 反復を
約40%削減することを、自己完結な概念実証で示す。

---

## 2. Abstract (English, ~180 words)

Device-scale TCAD is bottlenecked by the Poisson solve inside the self-consistent
Poisson / drift-diffusion loop, which is re-solved thousands of times on large meshes.
We propose accelerating this solve with **weak-form (Galerkin finite-element) operator
learning**. Unlike strong-form (finite-difference / PINN) surrogates, the physics loss
is the assembled FE residual K(eps)phi - M rho, so the interface flux-continuity
condition is built into the discretisation — making the surrogate robust to variable
permittivity and unstructured meshes. An FE a-posteriori error estimator unifies mesh
refinement and the online-learning trigger, keeping the finite-element solver as the
authority on accuracy while a learned operator only (i) reduces exact-solve frequency
via online adaptation and (ii) warm-starts Newton to cut iterations. On GAA and CFET
device cross-sections (nonlinear Poisson / Poisson-Boltzmann) we show, in self-contained
concept demos, that the weak form removes the interface-error floor that the strong form
stalls at, and that a learned Newton warm-start cuts iteration count ~40% at unchanged
(exact FE) accuracy, with the exact-solve trigger rate falling as the operator adapts.

---

## 3. スライド構成（12枚）

1. **タイトル**: Weak-form FEM operator learning + online adaptation for semiconductor
   device simulation. 著者/所属/日付。
2. **背景**: デバイス方程式（ポアソン＋ドリフト拡散）、自己整合ループ、ポアソン解が律速。
   図: ループ模式図。（出典: BASICS メモ §1）
3. **分野の現状（定番マップ）**: ①コンパクトモデル ②CNN Poisson ③PINN連立 ④GNN/拡散
   ⑤DeepONet/FNO ⑥収束加速 ⑦MC加速 ⑧逆設計。→ 「②〜④は強形式/格子が主流」。
4. **課題と着眼**: 強形式は材料界面（可変ε）でフラックス連続を破る。→ **弱形式(FEM)を核に**。
5. **提案(A+D+B+C)**: A 弱形式損失 / D a-posteriori誤差トリガ / B GNN branch / C ウォームスタート。
   図: Branch(GNN)-Trunk(座標)-内積 の DeepONet 模式。
6. **実証① 弱形式PI-DeepONet＋オンライン**: トリガされた厳密FE解 24/60(=デプロイ時コスト,60%減)、トリガ率 57%→23%、
   rel-L2 0.074。図: `pi_deeponet_fem_gaa.png`。
7. **実証② 弱形式 vs 強形式ベンチ（新規性の一枚看板）**: 弱形式 O(h²)収束 vs 強形式 ~0.55頭打ち、
   誤差マップ。図: `bench_weak_vs_strong.png`。
8. **実証③④ Newton ウォームスタート（1D→2D GAA）**: cold 5.2→warm 3.1(~40%減)、残差曲線。
   図: `fe_newton_warmstart.png` / `dd2d_newton_warmstart.png`。
9. **実証⑤ CFET 積層断面**: n/p縦積み＋誘電体、cold 4.8→warm 3.0。ロードマップ
   FinFET→GAA→Forksheet→**CFET**→2D-CFET→3Dモノリシック の位置づけ。図: `cfet_stack_warmstart.png`。
10. **正直な限界**: 概念実証（3D・量子・散乱・完全DD未実装）、誤差指標は素残差（tolは較正値）、
    ウォームスタートは減衰Newtonの下限付近で適応ゲイン小（本質は反復~40%減で精度不変）。
11. **ロードマップ**: 真の縦ゲート＋非構造メッシュ / 完全DD(Scharfetter–Gummel) / AMR /
    2D-CFET / TSV熱機械応力(FEA×GNN, repo親和)。
12. **まとめ**: 「弱形式を核にすることで界面に頑健、FEMを精度の権威に残しつつオンライン適応と
    ウォームスタートで加速」。1行テイクアウェイ。

---

## 4. 想定 Q&A（3点）

- **Q: なぜ強形式でなく弱形式?** A: 界面フラックス連続 [ε∂φ/∂n]=0 が組立に内在。②で定量実証
  （強形式は界面フロアで頭打ち）。
- **Q: サロゲートで精度は落ちない?** A: 案Cは置換でなく加速。FEM が最終精度を保証し、ネットは
  Newton 初期値のみ。全解が厳密FE解。
- **Q: 元研究(Otsuki & Mori, MC+FD)との差は?** A: 離散化の核が強形式FD→弱形式FEM。応用でなく
  方法レベルで別物（②が裏付け）。

---

## 5. 出典・図の対応

| スライド | 図/メモ |
|---|---|
| 2,3 | `BASICS_semiconductor_operator_learning.md` |
| 4,5,10,11 | `FEM_OPERATOR_LEARNING_GAA_IDEA.md`, `CONCEPT_...md` |
| 6 | `pi_deeponet_fem_gaa.png` |
| 7 | `bench_weak_vs_strong.png` |
| 8 | `fe_newton_warmstart.png`, `dd2d_newton_warmstart.png` |
| 9 | `cfet_stack_warmstart.png` |

> 注: 数値はいずれも既定実行の実測（seed固定）。pi_deeponet は GNN scatter による軽微な
> run-to-run 変動あり（committed 図の値を採用）。実証①の「24/60」は**トリガされた厳密FE解**
> （＝デプロイ時コスト）であり、各ステップの rel-L2 検証用 FE 解は診断目的でデプロイ時には
> 走らない別枠（非計上）。
