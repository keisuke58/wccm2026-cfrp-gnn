# 文献調査：テーマG 電気–熱連成（デバイス自己発熱, GAA/CFET）

> テーマG（GAA/CFET の自己発熱＝電気–熱連成と信頼性、そのシミュレーション/加速）の文献レビュー。
> 物理を主役に、既存手法・フロンティア・研究の隙間を整理（ML は加速の道具として位置づけ）。
> ⚠️ 本メモは Web 検索スニペット段階の一次整理。各文献の主張は原著で要確認（DOI/URL 併記）。

---

## 1. スコープ
先端トランジスタ（GAA ナノシート, CFET）の**自己発熱効果 (Self-Heating Effect, SHE)** と、その
**電気–熱自己整合シミュレーション**（非等温ドリフト拡散＝Poisson＋電流連続＋熱伝導の連成）、および
サロゲート/加速の現状。本リポジトリの半導体電気(⑧⑨)＋熱(⑥⑦)資産の連成先。

## 2. 自己発熱の物理と影響（GAA/ナノシート）
- SHE でチャネル格子温度が上昇 → **ION・ION/IOFF の劣化**。3nm 級で顕在化。
  [Analysis of DC SHE in stacked nanosheet GAA (RG 327517475)](https://www.researchgate.net/publication/327517475)
- **High-k（低熱伝導）ゲート酸化膜**と、リーク抑制の**底部誘電体(bottom dielectric)**が
  基板への主放熱経路を断ち、SHE を悪化。
  [SHE with bottom dielectric, 5nm stacked NS (ScienceDirect S2773064623000336)](https://www.sciencedirect.com/science/article/pii/S2773064623000336)
- **ゲート/ドレイン電圧依存**の SHE を 3D TCAD で解析。
  [Gate/drain-bias thermal, Microelectronics J. (ACM 10.1016/j.mejo.2023.105970)](https://dl.acm.org/doi/abs/10.1016/j.mejo.2023.105970)
- **幾何依存**（ナノワイヤ vs ナノシート）、S/D コンタクト工学での放熱改善。
  [Geometry SHE NW/NS TCAD (RG 342218351)](https://www.researchgate.net/publication/342218351) /
  [Electro-thermal boosting w/ engineered S/D (RG 353690075)](https://www.researchgate.net/publication/353690075)
- **解析的多段熱抵抗モデル**（Reff, 界面熱接触抵抗）。
  [Analytical multistage thermal resistance for NSFET SHE (ScienceDirect S1879239124002030)](https://www.sciencedirect.com/science/article/abs/pii/S1879239124002030)
- SHE と信頼性/放射線（single-event transient）の相互作用も。
  [SHE × SET in triple-layer nanosheets, Electronics 2026 (10.3390/electronics15010085)](https://doi.org/10.3390/electronics15010085)

## 3. CFET / 3D積層の自己発熱（フロンティア）
- **縦積み CFET は熱的に孤立** → SHE が横型 CMOS の**約2倍**、nFET↔pFET の**device 内熱クロストーク**。
  [Self-Heating & Thermal Network Model for CFET (IEEE 9633122)](https://ieeexplore.ieee.org/document/9633122/)
- 緩和策: **Buried Power Rail (BPR) / Buried Thermal Rail (BTR)**、電力供給構造の最適化で Rth 低減。
  [Buried Thermal Rail for CFET (PMC10536949)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10536949/) /
  [Thermal perf. via power delivery structures, J. Appl. Phys. 2025 (138/14/145704)](https://pubs.aip.org/aip/jap/article/138/14/145704/3367073)
- **マルチ段 CFET の自己発熱＋寄生**（2026 プレプリント＝最前線）。
  [Self-Heating and Parasitic Effects in Multi-Tier CFET (arXiv:2603.21910)](https://arxiv.org/abs/2603.21910)
- SHE は**欠陥生成・エージングを加速**し、デバイス/回路双方で劣化。

## 4. 連成解法（物理・数値）
- 標準モデル＝**非等温ドリフト拡散（van Roosbroeck ＋ 熱伝導）**、熱源は**ジュール熱＋再結合熱**、
  **温度依存移動度**。自己整合。
  [Self-consistent heating in nanoscale devices (academia 28607092)](https://www.academia.edu/28607092/)
- **非等温 Scharfetter–Gummel** 離散化（Fermi–Dirac 対応, 熱力学的整合を保存）。
  [Non-isothermal SG for degenerate semiconductors (arXiv:2002.10133)](https://arxiv.org/pdf/2002.10133) /
  [Generalized SG schemes for electro-thermal (arXiv:1911.00377)](https://arxiv.org/pdf/1911.00377)
- 自己発熱による **S 字 I–V／負性微分抵抗(NDR)／多重解**（同一電圧に複数の温度分布解）。
  ＝**継続法/分岐追跡が本質**になる領域。
  [Electrothermal DD, S-shaped I-V (RG 341716957)](https://www.researchgate.net/publication/341716957)
- ナノスケールでは Monte-Carlo × エネルギーバランス（フォノン浴）連成も。

## 5. ML / サロゲート（＝既に多数, ML は新規性でない）
- **ANN で GAAFET の幾何依存 電気熱 co-simulation** を高速化（2026）。
  [ANN-assisted GAAFET electrothermal co-sim, J. Comput. Electron. 2026 (10.1007/s10825-026-02598-1)](https://link.springer.com/article/10.1007/s10825-026-02598-1)
- FinFET ニューラルサロゲートで **~10^5 倍高速, R²0.99**、能動学習で TCAD 問い合わせ削減、
  PINN で外挿、**物理ガイド拡散(PCGD)** など TCAD×ML は活況。
  [PCGD (arXiv:2606.29272)](https://arxiv.org/pdf/2606.29272) /
  [PINN out-of-range TCAD (arXiv:2408.07921)](https://arxiv.org/pdf/2408.07921)
- → **ML による電気熱サロゲートは既に存在**。よって本テーマの新規性を ML に置くのは弱い。

## 6. 研究の隙間（gap）と我々の立ち位置（物理主役）
- SHE の物理（非等温 DD）と ML サロゲートは既に確立/活況。**空いているのは以下**:
  1. **弱形式 FE による非等温 DD 連成**：多くの SHE 研究は商用 box-method TCAD。**弱形式 FEM**で
     電気(⑧⑨)＋熱(⑥⑦)を連成し、界面不連続（High-k/底部誘電体の低熱伝導）に頑健化する実装は差別化余地。
  2. **NDR/多重解・S字領域の継続法**：自己発熱の多重解・分岐は**継続法が決定的**（我々の⑨と同型の物語）。
     「速い」でなく「収束/正しい枝の選択」を分ける領域。
  3. **CFET 熱クロストーク × 熱機械応力**：電気→熱→**熱応力(⑥⑦)→（将来）界面剥離**の
     **電気–熱–力の連鎖**。村松研の連成×本リポの半導体を統合、非破壊の multiphysics として立つ。
  4. **ML は従属**：動作点/レイアウト掃引の加速・サロゲートに限定（既存 ANN 電気熱と同枠, 精度は FEM が権威）。
- ＝**主役は「弱形式 FE 非等温 DD 連成＋継続法」という物理・数値**。ML は加速層。

## 7. 主要文献（要 原著確認）
- SHE/GAA: ScienceDirect S1879239124002030; RG 327517475; ScienceDirect S2773064623000336; ACM mejo.2023.105970; RG 342218351/353690075; Electronics 10.3390/electronics15010085。
- CFET/3D: IEEE 9633122; PMC10536949; JAP 138/14/145704; arXiv:2603.21910。
- 連成数値: arXiv:2002.10133; arXiv:1911.00377; RG 341716957; academia 28607092。
- ML×TCAD: Springer 10.1007/s10825-026-02598-1; arXiv:2606.29272; arXiv:2408.07921。
