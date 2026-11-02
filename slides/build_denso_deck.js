const pptxgen = require("pptxgenjs");
const path = require("path");

const OUT = process.argv[2] || "denso_residual_strain_draft.pptx";
const FIG = process.argv[3];

// Restrained academic palette: near-black text, one accent, grey rules.
const THEME = {
  name: "Academic",
  headFontFace: "Meiryo",
  bodyFontFace: "Meiryo",
  colors: {
    dk1: "1A1A1A", lt1: "FFFFFF", dk2: "4A4A4A", lt2: "F4F4F4",
    accent1: "8C2F28", accent2: "4A4A4A", accent3: "7A7A7A",
    accent4: "2E4A62", accent5: "9A9A9A", accent6: "C4C4C4",
    hlink: "2E4A62", folHlink: "7A7A7A",
  },
};
const INK = "1A1A1A", MUTED = "5A5A5A", RULE = "C9C9C9", ACC = "8C2F28";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.author = "Keisuke Nishioka";
pres.title = "繊維強化樹脂の残留ひずみ予測（仮）";
const C = pres.SchemeColor;

pres.defineSlideMaster({
  title: "TITLE", background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 1.0, y: 2.4, w: 11.3, h: 1.4,
      fontSize: 32, bold: true, color: C.text1, valign: "bottom", align: "left" }, text: " " } },
    { placeholder: { options: { name: "body", type: "body", x: 1.0, y: 4.0, w: 11.3, h: 1.6,
      fontSize: 15, color: MUTED, valign: "top", align: "left" }, text: " " } },
  ],
});

pres.defineSlideMaster({
  title: "CONTENT", background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 0.4, w: 11.9, h: 0.6,
      fontSize: 24, bold: true, color: C.text1, valign: "middle", align: "left" }, text: " " } },
    { line: { x: 0.7, y: 1.02, w: 11.9, h: 0, line: { color: RULE, width: 0.75 } } },
    { placeholder: { options: { name: "body", type: "body", x: 0.7, y: 1.25, w: 11.9, h: 5.4,
      fontSize: 14, color: C.text1, valign: "top", align: "left" }, text: " " } },
    { text: { text: "繊維強化樹脂の残留ひずみ予測（デンソー共同研究）／仮", options: { x: 0.7,
      y: 6.95, w: 8, h: 0.3, fontSize: 9, color: "9A9A9A", isTextBox: true, margin: 0 } } },
    { placeholder: { options: { name: "slideNumber", type: "sldNum", x: 12.1, y: 6.95, w: 0.5,
      h: 0.3, fontSize: 9, color: "9A9A9A", align: "right" } } },
  ],
});

// helpers -------------------------------------------------------------------
const sub = (t) => ({ text: t, options: { subscript: true } });
const eq = (s, x, y, w, text, no) => {
  s.addText(text, { x, y, w: w - 0.8, h: 0.52, fontSize: 15, color: INK, italic: true,
    align: "center", valign: "middle", isTextBox: true, margin: 0 });
  if (no) s.addText(no, { x: x + w - 0.75, y, w: 0.7, h: 0.52, fontSize: 13, color: MUTED,
    align: "right", valign: "middle", isTextBox: true, margin: 0 });
};
const caption = (s, x, y, w, t) => s.addText(t, { x, y, w, h: 0.5, fontSize: 10,
  color: MUTED, isTextBox: true, margin: 0, lineSpacing: 13 });
const rule = (s, y, x = 0.7, w = 11.9) => s.addShape(pres.ShapeType.line,
  { x, y, w, h: 0, line: { color: RULE, width: 0.75 }, objectName: `rule_${y}` });

// ===================================================== title ================
pres.addSection({ title: "表紙" });
let s = pres.addSlide({ masterName: "TITLE", sectionTitle: "表紙" });
s.addText("繊維強化樹脂の射出成形における\n残留ひずみの機械学習予測", { placeholder: "title",
  lineSpacing: 40 });
s.addText("デンソー共同研究（RA 2件目）／検討段階の資料\n背景・手法・今後の計画",
  { placeholder: "body", lineSpacing: 22 });
rule(s, 2.3, 1.0, 11.3);
s.addText("2026年11月  Keisuke Nishioka（慶應義塾大学 村松研究室）",
  { x: 1.0, y: 5.6, w: 8, h: 0.35, fontSize: 12, color: MUTED, isTextBox: true, margin: 0 });
s.addText("注：本資料は契約前の検討段階のものであり、材料系・数値はいずれも未確定である。",
  { x: 1.0, y: 6.1, w: 11.3, h: 0.35, fontSize: 11, color: MUTED, isTextBox: true, margin: 0 });
s.addNotes("契約前の検討資料。背景・手法・計画のみを示す。");

// ================================================ 1. 背景 ===================
pres.addSection({ title: "1. 背景" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "1. 背景" });
s.addText("1. 背景 — 成形後の残留ひずみが寸法精度と強度を支配する", { placeholder: "title" });
s.addText([
  { text: "繊維強化熱可塑性樹脂を金型内で成形すると、冷却過程で樹脂が結晶化しながら収縮する。収縮は繊維と金型に拘束されるため、成形品には残留ひずみが残る。", options: { breakLine: true } },
  { text: "残留ひずみは反り等の寸法誤差を生じ、また初期応力として強度低下の要因となる。その大きさは成形条件に強く依存するため、条件出しを試作に依存すると時間と費用を要する。", options: { breakLine: true } },
  { text: "一方、実測は局所的かつ破壊的になりやすく、部品全体の分布を得ることは難しい。予測モデルが得られれば、設計段階での条件選定が可能となる。" },
], { x: 0.7, y: 1.3, w: 11.9, h: 1.9, fontSize: 14, color: INK, lineSpacing: 23, paraSpaceAfter: 9 });

rule(s, 3.35);
s.addText("残留ひずみの発生機構", { x: 0.7, y: 3.5, w: 6, h: 0.35, fontSize: 14, bold: true,
  color: ACC, isTextBox: true, margin: 0 });
const chain = ["充填（流動）", "冷却・結晶化", "拘束された収縮", "残留ひずみ", "反り・強度低下"];
chain.forEach((t, i) => {
  const x = 0.7 + i * 2.42;
  s.addText(t, { x, y: 3.95, w: 2.0, h: 0.5, fontSize: 12.5, color: INK, align: "center",
    valign: "middle", isTextBox: true, margin: 0,
    line: { color: RULE, width: 0.75 } });
  if (i < chain.length - 1) s.addText("→", { x: x + 2.0, y: 3.95, w: 0.42, h: 0.5,
    fontSize: 13, color: MUTED, align: "center", valign: "middle", isTextBox: true, margin: 0 });
});

s.addText("本研究の目的", { x: 0.7, y: 4.8, w: 6, h: 0.35, fontSize: 14, bold: true,
  color: ACC, isTextBox: true, margin: 0 });
s.addText("結晶化度および繊維配向を入力として、成形後の残留ひずみを予測する機械学習モデルを構築する。",
  { x: 0.7, y: 5.2, w: 11.9, h: 0.45, fontSize: 14.5, bold: true, color: INK,
    isTextBox: true, margin: 0 });
s.addNotes("背景と目的。予測が必要な理由は、実測が局所的で全体分布が得られないこと。");

// ============================================= 2. 課題 ======================
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "1. 背景" });
s.addText("2. 予測を困難にする要因", { placeholder: "title" });
const issues = [
  ["2.1", "結晶化", "半結晶性樹脂では冷却速度により結晶化度が変化し、結晶化収縮量が変わる。したがって温度履歴が残留ひずみに直接寄与する。"],
  ["2.2", "流動誘起の繊維配向", "繊維配向は充填時の流動場により決まり、位置ごとに異なる。配向は熱膨張および剛性の異方性を通じて残留ひずみに寄与する。"],
  ["2.3", "スキン–コア構造", "射出成形品では板厚方向に配向が大きく変化する。表層は流動方向に強配向、内層は異なる配向を示すため、板厚方向の分布を無視できない。"],
  ["2.4", "計測の制約", "実験による残留ひずみの計測は点計測となる場合が多く、場としての分布は直接得られない。"],
];
issues.forEach(([no, head, body], i) => {
  const y = 1.30 + i * 1.27;
  s.addText(no, { x: 0.7, y, w: 0.65, h: 0.35, fontSize: 13, bold: true, color: ACC,
    isTextBox: true, margin: 0 });
  s.addText(head, { x: 1.35, y, w: 4.2, h: 0.35, fontSize: 14.5, bold: true, color: INK,
    isTextBox: true, margin: 0 });
  s.addText(body, { x: 1.35, y: y + 0.4, w: 11.2, h: 0.75, fontSize: 13, color: INK,
    isTextBox: true, margin: 0, lineSpacing: 20 });
  if (i < issues.length - 1) rule(s, y + 1.17, 1.35, 11.2);
});
s.addText("結晶化と配向が同時に作用し、かつ配向が空間分布を持つ点が、本問題の中心的な困難である。",
  { x: 0.7, y: 6.42, w: 11.9, h: 0.35, fontSize: 13, bold: true, color: ACC, isTextBox: true, margin: 0 });
s.addNotes("4つの要因。2.3 と 2.4 がデータ設計に直結する。");

// ============================================= 3. 手法 ======================
pres.addSection({ title: "2. 手法" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "2. 手法" });
s.addText("3. 手法 — 問題設定とデータ構成", { placeholder: "title" });

s.addText("3.1  問題設定", { x: 0.7, y: 1.3, w: 6, h: 0.35, fontSize: 14, bold: true,
  color: ACC, isTextBox: true, margin: 0 });
eq(s, 0.7, 1.72, 11.9, [{ text: "ε" }, sub("res"), { text: "  =  f ( X" }, sub("c"),
  { text: " ,  a" }, sub("ij"), { text: " ;  θ )" }], "(1)");
s.addText([{ text: "ここで ε" }, sub("res"), { text: " は残留ひずみ、X" }, sub("c"),
  { text: " は結晶化度、a" }, sub("ij"), { text: " は繊維配向テンソル、θ は学習パラメータである。" }],
  { x: 0.7, y: 2.3, w: 11.9, h: 0.35, fontSize: 12.5, color: MUTED, isTextBox: true, margin: 0 });

rule(s, 2.78);
s.addText("3.2  データ", { x: 0.7, y: 2.92, w: 6, h: 0.35, fontSize: 14, bold: true,
  color: ACC, isTextBox: true, margin: 0 });

const rows = [
  ["", "Ansys による数値解析", "デンソーによる実験計測"],
  ["形式", "場（要素・節点ごとの分布）", "点（離散的な計測位置）"],
  ["数量", "条件を振って生成可能", "少数に限られる"],
  ["位置づけ", "事前学習の対象", "較正の対象"],
];
s.addTable(rows.map((r, ri) => r.map((cell, ci) => ({
  text: cell,
  options: { bold: ri === 0 || ci === 0, fontSize: 12.5, color: INK,
    fill: ri === 0 ? "F4F4F4" : "FFFFFF", valign: "middle" },
}))), { x: 0.7, y: 3.35, w: 11.9, colW: [1.9, 5.0, 5.0], rowH: 0.42,
  border: { type: "solid", color: RULE, pt: 0.75 }, margin: 0.08 });

s.addText("3.3  学習方法：転移学習", { x: 0.7, y: 5.35, w: 6, h: 0.35, fontSize: 14, bold: true,
  color: ACC, isTextBox: true, margin: 0 });
s.addText("数値解析データで事前学習したモデルを、実験データにより較正する。すなわち場で学習し、点で合わせ込む構成とする。計算と実測の差を定量化した先行例（小島, TSA 全場計測）が研究室内に存在する。",
  { x: 0.7, y: 5.75, w: 11.9, h: 0.75, fontSize: 13, color: INK, isTextBox: true, margin: 0,
    lineSpacing: 20 });
s.addNotes("式(1)で問題設定を明示。データは粒度が異なる2系統。");

// ============================================= 4. 配向の表現 ================
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "2. 手法" });
s.addText("4. 繊維配向の表現 — 配向テンソルと入力特徴量の選択", { placeholder: "title" });

s.addText("4.1  配向テンソル", { x: 0.7, y: 1.28, w: 6, h: 0.3, fontSize: 13.5, bold: true,
  color: ACC, isTextBox: true, margin: 0 });
s.addText("繊維の向きを単位ベクトル p、その分布を ψ(p) とすると、Advani–Tucker の配向テンソルは",
  { x: 0.7, y: 1.62, w: 7.0, h: 0.35, fontSize: 12.5, color: INK, isTextBox: true, margin: 0 });
eq(s, 0.7, 1.98, 7.0, [{ text: "a" }, sub("ij"), { text: " = ∫ p" }, sub("i"),
  { text: " p" }, sub("j"), { text: " ψ(p) dp ,    a" }, sub("ii"), { text: " = 1" }], "(2)");
s.addText("で定義される。対称かつトレースが 1 であるため、独立成分は 5 である。固有値は配向の強さ、固有ベクトルは主配向方向を表す。",
  { x: 0.7, y: 2.48, w: 7.0, h: 0.6, fontSize: 12.5, color: INK, isTextBox: true, margin: 0,
    lineSpacing: 19 });

s.addText("4.2  入力特徴量の候補", { x: 0.7, y: 3.15, w: 6, h: 0.3, fontSize: 13.5, bold: true,
  color: ACC, isTextBox: true, margin: 0 });
const cands = [
  ["(a) 成分表示", "aij の独立成分をそのまま用いる。情報の損失がない。座標系に依存するため、全要素で共通の基底をとる必要がある。"],
  ["(b) 配向平均した熱膨張係数 α", "配向は熱膨張と剛性を通じてのみ残留ひずみに寄与する。物理的経路を経た特徴量であり、(a) との比較対象とする。"],
  ["(c) 固有値と主方向", "固有値が縮退する近傍で主方向が不連続に変化するため、本用途では採用しない。"],
];
cands.forEach(([h, b], i) => {
  const y = 3.5 + i * 0.95;
  s.addText(h, { x: 0.7, y, w: 7.0, h: 0.3, fontSize: 13, bold: true, color: INK,
    isTextBox: true, margin: 0 });
  s.addText(b, { x: 0.7, y: y + 0.3, w: 7.0, h: 0.6, fontSize: 11.5, color: MUTED,
    isTextBox: true, margin: 0, lineSpacing: 17 });
});

if (FIG) s.addImage({ path: FIG, x: 8.0, y: 1.3, w: 4.6, h: 2.5 });
caption(s, 8.0, 3.88, 4.6,
  "図1  配向平均した熱膨張係数（式(3)）と、直接平均（破線）の比較。\n定数は例示値であり校正していない。");

s.addText("4.3  配向平均の定式化", { x: 8.0, y: 4.72, w: 5, h: 0.3, fontSize: 13.5, bold: true,
  color: ACC, isTextBox: true, margin: 0 });
eq(s, 8.0, 5.07, 4.6, "〈α〉 = 〈C〉⁻¹ : 〈C : α〉", "(3)");
s.addText("α は 2 階テンソルであるが、配向の異なる領域が相互に拘束するため直接平均は成立しない。熱応力 C:α を平均し、平均剛性の逆を作用させる。",
  { x: 8.0, y: 5.6, w: 4.6, h: 0.9, fontSize: 11.5, color: INK, isTextBox: true, margin: 0,
    lineSpacing: 17 });
s.addNotes("式(2)(3)を明示。(b)は実装済みで、直接平均との差は部分配向で数倍に達する。");

// ============================================= 5. 計画 ======================
pres.addSection({ title: "3. 計画" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "3. 計画" });
s.addText("5. 研究計画", { placeholder: "title" });

const plan = [
  ["第 1 段階", "残留ひずみの予測", "入力表現の決定、数値解析データによる事前学習、実験データによる較正。", "本年度の中心"],
  ["第 2 段階", "変形および強度の評価", "第 1 段階で得た残留ひずみ場を用い、反りおよび強度低下を評価する。", "第 1 段階の後"],
  ["第 3 段階", "成形条件の設計", "金型温度および充填速度を設計変数とする逆問題として扱う。", "将来の課題"],
];
s.addTable([["段階", "内容", "概要", "時期"].map(t => ({ text: t,
  options: { bold: true, fontSize: 12.5, color: INK, fill: "F4F4F4", valign: "middle" } })),
  ...plan.map(r => r.map((cell, ci) => ({ text: cell,
    options: { fontSize: 12.5, color: INK, bold: ci === 1, valign: "middle" } })))],
  { x: 0.7, y: 1.35, w: 11.9, colW: [1.6, 3.0, 5.3, 2.0], rowH: [0.42, 0.72, 0.72, 0.72],
    border: { type: "solid", color: RULE, pt: 0.75 }, margin: 0.08 });

rule(s, 4.1);
s.addText("5.1  現状", { x: 0.7, y: 4.25, w: 6, h: 0.3, fontSize: 13.5, bold: true,
  color: ACC, isTextBox: true, margin: 0 });
s.addText("採択は決定しているが、交付申請および契約が未了であり、予算の執行および雇用手続きを開始できない。開始は 11 月中の見込みである。詳細資料は雇用手続きの際に共有される予定である。",
  { x: 0.7, y: 4.6, w: 11.9, h: 0.6, fontSize: 13, color: INK, isTextBox: true, margin: 0,
    lineSpacing: 20 });

s.addText("5.2  実施体制と到達目標", { x: 0.7, y: 5.3, w: 6, h: 0.3, fontSize: 13.5, bold: true,
  color: ACC, isTextBox: true, margin: 0 });
s.addText([
  { text: "打合せは月 1 回程度を想定する（先方の状況により隔週）。", options: { bullet: true, breakLine: true } },
  { text: "本年度の到達目標は研究会での発表とする。", options: { bullet: true, breakLine: true } },
  { text: "開始までの期間は、配向の入力表現の検討および転移学習の枠組みの整備に充てる。", options: { bullet: true } },
], { x: 0.7, y: 5.65, w: 11.9, h: 1.0, fontSize: 13, color: INK, paraSpaceAfter: 5 });
s.addNotes("計画表と現状。第1段階に絞る。");

// ============================================= 6. 要確認 ====================
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "3. 計画" });
s.addText("6. 開始時に確定すべき事項", { placeholder: "title" });
s.addText("以下はいずれも先方との確認を要する。第 1 段階の設計はこれらに依存する。",
  { x: 0.7, y: 1.3, w: 11.9, h: 0.3, fontSize: 12.5, color: MUTED, isTextBox: true, margin: 0 });

const confirm = [
  ["材料系", "樹脂のグレード、繊維種（炭素／ガラス）、繊維長、体積含有率"],
  ["成形法", "射出成形か圧縮成形か。充填速度の定義が異なる"],
  ["配向データ", "配向モデルと閉包近似、板厚方向の分割数、受け渡し形式および基底"],
  ["計測位置", "計測点が表層か内層か。スキン–コアのいずれを代表するか"],
  ["残留ひずみ", "計測法、および予測対象とする成分（テンソルか等価量か）"],
  ["データ規模", "解析条件数および実験点数。転移学習の構成がこれにより決まる"],
];
s.addTable([["項目", "確認内容"].map(t => ({ text: t,
  options: { bold: true, fontSize: 12.5, color: INK, fill: "F4F4F4", valign: "middle" } })),
  ...confirm.map(r => r.map((cell, ci) => ({ text: cell,
    options: { fontSize: 12.5, color: INK, bold: ci === 0, valign: "middle" } })))],
  { x: 0.7, y: 1.75, w: 11.9, colW: [2.6, 9.3], rowH: 0.52,
    border: { type: "solid", color: RULE, pt: 0.75 }, margin: 0.08 });

s.addText("特に、数値解析により得られる配向場は配向モデル・閉包近似・同定済みパラメータに依存する。したがって転移学習は、離散化誤差に加えて配向モデルの誤差を併せて補正することになる点に留意する。",
  { x: 0.7, y: 5.3, w: 11.9, h: 0.6, fontSize: 12.5, color: INK, isTextBox: true, margin: 0,
    lineSpacing: 19 });
s.addNotes("確認事項。配向データの形式と計測位置が設計を左右する。");

// ============================================= 7. 参考文献 ==================
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "3. 計画" });
s.addText("参考文献", { placeholder: "title" });
const refs = [
  "[1] S. G. Advani and C. L. Tucker III, “The use of tensors to describe and predict fiber orientation in short fiber composites,” Journal of Rheology, 31(8), 751–784, 1987.",
  "[2] F. Folgar and C. L. Tucker III, “Orientation behavior of fibers in concentrated suspensions,” Journal of Reinforced Plastics and Composites, 3(2), 98–119, 1984.",
  "[3] J. Wang, J. F. O’Gara and C. L. Tucker III, “An objective model for slow orientation kinetics in concentrated fiber suspensions,” Journal of Rheology, 52(5), 1179–1200, 2008.",
  "[4] J. H. Phelps and C. L. Tucker III, “An anisotropic rotary diffusion model for fiber orientation in short- and long-fiber thermoplastics,” Journal of Non-Newtonian Fluid Mechanics, 156(3), 165–176, 2009.",
  "[5] J. S. Cintra and C. L. Tucker III, “Orthotropic closure approximations for flow-induced fiber orientation,” Journal of Rheology, 39(6), 1095–1122, 1995.",
  "[6] D. H. Chung and T. H. Kwon, “Invariant-based optimal fitting closure approximation for the numerical prediction of flow-induced fiber orientation,” Journal of Rheology, 46(1), 169–194, 2002.",
];
s.addText(refs.map((r, i) => ({ text: r,
  options: { breakLine: i < refs.length - 1 } })),
  { x: 0.7, y: 1.35, w: 11.9, h: 4.4, fontSize: 11.5, color: INK, lineSpacing: 17,
    paraSpaceAfter: 7 });
rule(s, 6.05);
s.addText("注：書誌情報は一次文献で未照合の箇所を含む。引用前に確認すること。",
  { x: 0.7, y: 6.2, w: 11.9, h: 0.3, fontSize: 11, color: MUTED, isTextBox: true, margin: 0 });
s.addNotes("配向テンソル・配向発展・閉包近似の基本文献。書誌は未照合。");

pres.writeFile({ fileName: OUT }).then(async () => {
  const { applyTheme } = require(path.join(process.env.PPTX_SKILL, "scripts/apply_theme.js"));
  await applyTheme(OUT, THEME);
  console.log("wrote " + OUT);
});
