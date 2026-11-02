// build_denso_deck.js — draft deck for the joint research with DENSO CORPORATION
// (background / method / plan), in the academic register required by CLAUDE.md.
//
// Typography (CLAUDE.md): Japanese in Mincho (游明朝), Latin / digits / mathematics in
// Times New Roman. PowerPoint chooses the font per character from the theme: `latin` for
// Latin script, `ea` for East-Asian script. pptxgenjs writes the theme with `ea` EMPTY and
// the Japanese script font set to 游ゴシック, so Japanese would come out in Gothic. The theme
// is therefore rewritten after generation (patchThemeFonts below).
// Mathematics: variables italic, vectors/tensors bold italic, label subscripts (res, c),
// operators, brackets and the differential d upright; minus is U+2212.
//
// Run:  NODE_PATH="$(npm root -g)" PPTX_SKILL=<pptx skill dir> \
//       node slides/build_denso_deck.js slides/denso_residual_strain_draft.pptx \
//            slides/figures_paper/fig1_cte_orientation.png
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const OUT = process.argv[2] || "denso_residual_strain_draft.pptx";
const FIG = process.argv[3];

const LATIN = "Times New Roman";
const EA = "游明朝";
const BRACKET = "Cambria Math";          // U+27E8/27E9 are absent from Times New Roman

const THEME = {
  name: "Academic Mincho Times",
  headFontFace: LATIN, bodyFontFace: LATIN,
  colors: {
    dk1: "111111", lt1: "FFFFFF", dk2: "444444", lt2: "F3F3F3",
    accent1: "333333", accent2: "555555", accent3: "777777",
    accent4: "999999", accent5: "BBBBBB", accent6: "DDDDDD",
    hlink: "1F3A5F", folHlink: "555555",
  },
};
const INK = "111111", MUTED = "555555", RULE = "BFBFBF";
const FOOT = "繊維強化樹脂の残留ひずみ予測（株式会社デンソー様との共同研究）／検討段階";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.theme = { headFontFace: LATIN, bodyFontFace: LATIN };
pres.author = "西岡 佳祐";
pres.title = "繊維強化樹脂の射出成形における残留ひずみの機械学習予測（検討段階）";
const C = pres.SchemeColor;

pres.defineSlideMaster({
  title: "TITLE", background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.9, y: 1.7, w: 11.5, h: 1.5,
      fontSize: 30, bold: true, color: C.text1, valign: "bottom", align: "center" }, text: " " } },
    { placeholder: { options: { name: "body", type: "body", x: 0.9, y: 3.4, w: 11.5, h: 0.9,
      fontSize: 16, color: C.text1, valign: "top", align: "center" }, text: " " } },
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
    { text: { text: FOOT, options: { x: 0.7, y: 6.95, w: 9, h: 0.3, fontSize: 9,
      color: "8A8A8A", isTextBox: true, margin: 0 } } },
    { placeholder: { options: { name: "slideNumber", type: "sldNum", x: 12.1, y: 6.95, w: 0.5,
      h: 0.3, fontSize: 9, color: "8A8A8A", align: "right" } } },
  ],
});

// ---------------------------------------------------------- math run helpers ----
const it = (t) => ({ text: t, options: { italic: true } });                 // variable
const bi = (t) => ({ text: t, options: { italic: true, bold: true } });     // vector/tensor
const ro = (t) => ({ text: t, options: { italic: false } });                // upright
const si = (t) => ({ text: t, options: { italic: true, subscript: true } }); // index sub
const sr = (t) => ({ text: t, options: { italic: false, subscript: true } });// label sub
const sup = (t) => ({ text: t, options: { italic: false, superscript: true } });
const lb = () => ({ text: "⟨", options: { italic: false, fontFace: BRACKET } });
const rb = () => ({ text: "⟩", options: { italic: false, fontFace: BRACKET } });
const tx = (t, o = {}) => ({ text: t, options: o });                         // prose

// layout helpers -------------------------------------------------------------
const eq = (s, x, y, w, runs, no) => {
  s.addText(runs, { x, y, w: w - 0.8, h: 0.55, fontSize: 17, color: INK, align: "center",
    valign: "middle", isTextBox: true, margin: 0 });
  if (no) s.addText(no, { x: x + w - 0.75, y, w: 0.7, h: 0.55, fontSize: 14, color: INK,
    align: "right", valign: "middle", isTextBox: true, margin: 0 });
};
const head = (s, x, y, w, t) => s.addText(t, { x, y, w, h: 0.34, fontSize: 14.5, bold: true,
  color: INK, isTextBox: true, margin: 0 });
const para = (s, x, y, w, h, runs, size = 13.5) => s.addText(runs, { x, y, w, h, fontSize: size,
  color: INK, isTextBox: true, margin: 0, lineSpacing: Math.round(size * 1.55) });
const rule = (s, y, x = 0.7, w = 11.9) => s.addShape(pres.ShapeType.line,
  { x, y, w, h: 0, line: { color: RULE, width: 0.75 }, objectName: `rule_${y}` });
const tableOpts = (x, y, w, colW, rowH) => ({ x, y, w, colW, rowH,
  border: { type: "solid", color: RULE, pt: 0.75 }, margin: 0.08 });
const cell = (t, o = {}) => ({ text: t, options: Object.assign({ fontSize: 13, color: INK,
  valign: "middle" }, o) });

// ===================================================== title ================
pres.addSection({ title: "表紙" });
let s = pres.addSlide({ masterName: "TITLE", sectionTitle: "表紙" });
s.addText("繊維強化樹脂の射出成形における\n残留ひずみの機械学習予測", { placeholder: "title",
  lineSpacing: 40 });
s.addText("株式会社デンソー様との共同研究 — 背景・手法・研究計画（検討段階）",
  { placeholder: "body" });
rule(s, 4.45, 4.15, 5.0);
s.addText([tx("西岡 佳祐", { breakLine: true }), tx("慶應義塾大学 修士課程", { breakLine: true }),
  tx("2026年11月")],
  { x: 0.9, y: 4.65, w: 11.5, h: 1.2, fontSize: 15, color: INK, align: "center",
    isTextBox: true, margin: 0, lineSpacing: 24 });
s.addText("注：本資料は研究開始前の検討段階のものであり、材料系および数値はいずれも未確定である。",
  { x: 0.9, y: 6.35, w: 11.5, h: 0.35, fontSize: 11, color: MUTED, align: "center",
    isTextBox: true, margin: 0 });
s.addNotes("開始前の検討資料。背景・手法・研究計画のみを示す。");

// ================================================ 1. 背景 ===================
pres.addSection({ title: "1. 背景" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "1. 背景" });
s.addText("1. 背景 — 成形後の残留ひずみが寸法精度と強度を支配する", { placeholder: "title" });
para(s, 0.7, 1.3, 11.9, 1.95, [
  tx("繊維強化熱可塑性樹脂を金型内で成形すると、冷却過程で樹脂が結晶化しながら収縮する。この収縮は繊維と金型により拘束されるため、成形品には残留ひずみが残る。", { breakLine: true }),
  tx("残留ひずみは反り等の寸法誤差を生じ、また初期応力として強度低下の要因となる。その大きさは成形条件に強く依存するため、条件出しを試作に依存すると多大な時間と費用を要する。", { breakLine: true }),
  tx("一方、残留ひずみの計測は局所的かつ破壊的となりやすく、部品全体の分布を得ることは難しい。予測手法が得られれば、設計段階での条件選定が可能となる。"),
], 14);
rule(s, 3.4);
head(s, 0.7, 3.55, 6, "残留ひずみの発生過程");
const chain = ["充填（流動）", "冷却・結晶化", "拘束された収縮", "残留ひずみ", "反り・強度低下"];
chain.forEach((t, i) => {
  const x = 0.7 + i * 2.42;
  s.addText(t, { x, y: 4.0, w: 2.0, h: 0.5, fontSize: 13, color: INK, align: "center",
    valign: "middle", isTextBox: true, margin: 0, line: { color: RULE, width: 0.75 } });
  if (i < chain.length - 1) s.addText("→", { x: x + 2.0, y: 4.0, w: 0.42, h: 0.5,
    fontSize: 14, color: MUTED, align: "center", valign: "middle", isTextBox: true, margin: 0 });
});
head(s, 0.7, 4.85, 6, "本研究の目的");
para(s, 0.7, 5.25, 11.9, 0.45, [tx("結晶化度および繊維配向を入力として、成形後の残留ひずみを予測する機械学習モデルを構築する。", { bold: true })], 14.5);
s.addNotes("背景と目的。予測が必要な理由は、計測が局所的で全体分布が得られないことにある。");

// ============================================= 2. 要因 ======================
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "1. 背景" });
s.addText("2. 予測を困難にする要因", { placeholder: "title" });
const issues = [
  ["2.1", "結晶化", [tx("半結晶性樹脂では冷却速度により結晶化度が変化し、結晶化収縮量が変わる。したがって温度履歴が残留ひずみに直接寄与する。")]],
  ["2.2", "流動誘起の繊維配向", [tx("繊維配向は充填時の流動場により決まり、位置ごとに異なる[2]。配向は熱膨張および剛性の異方性を通じて残留ひずみに寄与する。")]],
  ["2.3", "スキン–コア構造", [tx("射出成形品では板厚方向に配向が大きく変化する。表層は流動方向に強く配向し、内層は異なる配向を示すため、板厚方向の分布を無視できない。")]],
  ["2.4", "計測の制約", [tx("残留ひずみの実験計測は点計測となる場合が多く、場としての分布は直接得られない。")]],
];
issues.forEach(([no, h, body], i) => {
  const y = 1.30 + i * 1.27;
  s.addText(no, { x: 0.7, y, w: 0.65, h: 0.34, fontSize: 14, bold: true, color: INK,
    isTextBox: true, margin: 0 });
  head(s, 1.35, y, 5, h);
  para(s, 1.35, y + 0.4, 11.2, 0.75, body, 13.5);
  if (i < issues.length - 1) rule(s, y + 1.17, 1.35, 11.2);
});
para(s, 0.7, 6.42, 11.9, 0.35, [tx("結晶化と配向が同時に作用し、かつ配向が空間分布を持つ点が、本問題の中心的な困難である。", { bold: true })], 13.5);
s.addNotes("4つの要因。2.3と2.4がデータの設計に直結する。");

// ============================================= 3. 手法 ======================
pres.addSection({ title: "2. 手法" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "2. 手法" });
s.addText("3. 手法 — 問題設定とデータ構成", { placeholder: "title" });
head(s, 0.7, 1.3, 6, "3.1　問題設定");
eq(s, 0.7, 1.72, 11.9, [it("ε"), sr("res"), ro(" = "), it("f"), ro("("), it("X"), sr("c"),
  ro(", "), it("a"), si("ij"), ro("; "), bi("θ"), ro(")")], "(1)");
para(s, 0.7, 2.32, 11.9, 0.38, [tx("ここで"), it("ε"), sr("res"), tx("は残留ひずみ、"), it("X"),
  sr("c"), tx("は結晶化度、"), it("a"), si("ij"), tx("は繊維配向テンソル[1]、"), bi("θ"),
  tx("は学習パラメータである。")], 13);
rule(s, 2.82);
head(s, 0.7, 2.95, 6, "3.2　データ");
const rows = [["", "Ansysによる数値解析", "デンソー様による実験計測"],
  ["形式", "場（要素・節点ごとの分布）", "点（離散的な計測位置）"],
  ["数量", "条件を変えて生成可能", "少数に限られる"],
  ["役割", "事前学習", "較正"]];
s.addTable(rows.map((r, ri) => r.map((c, ci) => cell(c, { bold: ri === 0 || ci === 0,
  fill: ri === 0 ? "F3F3F3" : "FFFFFF" }))), tableOpts(0.7, 3.38, 11.9, [1.9, 5.0, 5.0], 0.42));
head(s, 0.7, 5.35, 6, "3.3　学習方法：転移学習");
para(s, 0.7, 5.75, 11.9, 0.8, [tx("数値解析データにより事前学習したモデルを、実験データにより較正する。すなわち、場として得られる解析結果で学習し、点として得られる計測値で合わせ込む構成とする。計算と実測の差を定量化した先行例（TSA全場計測）が研究室内にある。")], 13.5);
s.addNotes("式(1)で問題を定義する。データは粒度の異なる2系統である。");

// ============================================= 4. 配向の表現 ================
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "2. 手法" });
s.addText("4. 繊維配向の表現 — 配向テンソルと入力特徴量の選択", { placeholder: "title" });
head(s, 0.7, 1.28, 6, "4.1　配向テンソル");
para(s, 0.7, 1.64, 7.0, 0.38, [tx("繊維の向きを単位ベクトル"), bi("p"), tx("、その分布を"), it("ψ"),
  ro("("), bi("p"), ro(")"), tx("とすると、配向テンソルは[1]")], 13);
eq(s, 0.7, 2.02, 7.0, [it("a"), si("ij"), ro(" = ∫ "), it("p"), si("i"), ro(" "), it("p"),
  si("j"), ro(" "), it("ψ"), ro("("), bi("p"), ro(") d"), bi("p"), ro(",      "), it("a"),
  si("ii"), ro(" = 1")], "(2)");
para(s, 0.7, 2.58, 7.0, 0.62, [tx("で定義される。対称かつトレースが1であるため独立成分は5である。固有値は配向の強さを、固有ベクトルは主配向方向を表す。")], 12.5);
head(s, 0.7, 3.25, 6, "4.2　入力特徴量の候補");
const cands = [
  ["(a)　成分表示", [it("a"), si("ij"), tx("の独立成分をそのまま用いる。情報の損失はないが座標系に依存するため、全要素で共通の基底をとる必要がある。")]],
  ["(b)　配向平均した熱膨張係数", [tx("配向は熱膨張と剛性を通じてのみ残留ひずみに寄与する。この物理的経路を経た特徴量であり、(a)との比較対象とする。")]],
  ["(c)　固有値と主方向", [tx("固有値が縮退する近傍で主方向が不連続に変化するため、本用途には採用しない。")]],
];
cands.forEach(([h, b], i) => {
  const y = 3.62 + i * 0.95;
  s.addText(h, { x: 0.7, y, w: 7.0, h: 0.3, fontSize: 13, bold: true, color: INK,
    isTextBox: true, margin: 0 });
  para(s, 0.7, y + 0.31, 7.0, 0.6, b, 12);
});
if (FIG) s.addImage({ path: FIG, x: 8.0, y: 1.3, w: 4.6, h: 2.56 });
para(s, 8.0, 3.92, 4.6, 0.62, [tx("図1　式(3)による配向平均熱膨張係数（実線）と、拘束を無視した直接平均（破線）。定数は例示値であり、校正していない。")], 10.5);
head(s, 8.0, 4.68, 4.6, "4.3　配向平均");
eq(s, 8.0, 5.05, 4.6, [lb(), bi("α"), rb(), ro(" = "), lb(), bi("C"), rb(), sup("−1"),
  ro(" : "), lb(), bi("C"), ro(" : "), bi("α"), rb()], "(3)");
para(s, 8.0, 5.62, 4.6, 0.95, [bi("α"), tx("は2階テンソルであるが、配向の異なる領域が相互に拘束するため直接平均は成立しない。熱応力"), bi("C"), ro(" : "), bi("α"), tx("を平均し、平均剛性の逆を作用させる[1]。")], 12);
s.addNotes("式(2)(3)を示す。(b)は実装済みであり、直接平均との差は部分配向で数倍に達する。");

// ============================================= 5. 計画 ======================
pres.addSection({ title: "3. 計画" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "3. 計画" });
s.addText("5. 研究計画", { placeholder: "title" });
const plan = [
  ["第1段階", "残留ひずみの予測", "入力表現の決定、数値解析データによる事前学習、実験データによる較正。", "本年度の中心"],
  ["第2段階", "変形および強度の評価", "第1段階で得た残留ひずみ場を用い、反りおよび強度低下を評価する。", "第1段階の後"],
  ["第3段階", "成形条件の設計", "金型温度および射出速度を設計変数とする逆問題として扱う。", "今後の課題"],
];
s.addTable([["段階", "内容", "概要", "時期"].map(t => cell(t, { bold: true, fill: "F3F3F3" })),
  ...plan.map(r => r.map((c, ci) => cell(c, { bold: ci === 1 })))],
  tableOpts(0.7, 1.35, 11.9, [1.6, 3.0, 5.3, 2.0], [0.42, 0.72, 0.72, 0.72]));
rule(s, 4.1);
head(s, 0.7, 4.25, 6, "5.1　現状");
para(s, 0.7, 4.62, 11.9, 0.55, [tx("採択は決定しており、現在、交付申請および契約の手続き中である。研究の開始は11月中を見込んでいる。")], 13.5);
head(s, 0.7, 5.3, 6, "5.2　進め方と到達目標");
s.addText([
  { text: "打合せは月1回程度を想定し、状況に応じて隔週とする。", options: { bullet: true, breakLine: true } },
  { text: "本年度は研究会での発表を到達目標とする。", options: { bullet: true, breakLine: true } },
  { text: "開始までの期間は、繊維配向の入力表現の検討と転移学習の枠組みの整備に充てる。", options: { bullet: true } },
], { x: 0.7, y: 5.67, w: 11.9, h: 1.0, fontSize: 13.5, color: INK, paraSpaceAfter: 5 });
s.addNotes("計画表と現状。本年度は第1段階に絞る。");

// ============================================= 6. 確認事項 ==================
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "3. 計画" });
s.addText("6. 開始時に確認させていただきたい事項", { placeholder: "title" });
para(s, 0.7, 1.3, 11.9, 0.32, [tx("以下の事項は第1段階の設計を左右するため、開始時にデンソー様と確認させていただきたい。")], 13);
const confirm = [
  ["材料系", "樹脂のグレード、繊維種（炭素／ガラス）、繊維長、体積含有率"],
  ["成形法", "射出成形か圧縮成形か（射出速度の定義が異なるため）"],
  ["配向データ", "配向モデルと閉包近似、板厚方向の分割数、受け渡し形式および座標基底"],
  ["計測位置", "計測点が表層か内層か（スキン–コアのいずれを代表するか）"],
  ["残留ひずみ", "計測法、および予測対象とする成分（テンソル成分か等価量か）"],
  ["データ規模", "解析条件数および実験点数（転移学習の構成がこれにより決まる）"],
];
s.addTable([["項目", "確認内容"].map(t => cell(t, { bold: true, fill: "F3F3F3" })),
  ...confirm.map(r => r.map((c, ci) => cell(c, { bold: ci === 0 })))],
  tableOpts(0.7, 1.75, 11.9, [2.6, 9.3], 0.52));
para(s, 0.7, 5.65, 11.9, 0.65, [tx("なお、数値解析により得られる配向場は、配向モデル・閉包近似・同定済みパラメータに依存する[2–6]。したがって転移学習は、離散化誤差に加えて配向モデルの誤差も併せて補正することになる。")], 13);
s.addNotes("確認事項。特に配向データの形式と計測位置が設計を左右する。");

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
s.addText(refs.map((r, i) => ({ text: r, options: { breakLine: i < refs.length - 1 } })),
  { x: 0.7, y: 1.35, w: 11.9, h: 4.4, fontSize: 12, color: INK, lineSpacing: 18, paraSpaceAfter: 7 });
rule(s, 6.05);
s.addText("注：書誌情報は一次文献で未照合の箇所を含む。引用前に確認すること。",
  { x: 0.7, y: 6.2, w: 11.9, h: 0.3, fontSize: 11, color: MUTED, isTextBox: true, margin: 0 });
s.addNotes("配向テンソル・配向発展・閉包近似の基本文献。書誌は未照合。");

// ----------------------------------------------- write, theme, Mincho patch ----
function loadJSZip() {
  return require(require.resolve("jszip", { paths: [require.resolve("pptxgenjs")] }));
}
async function patchThemeFonts(file) {
  const zip = await loadJSZip().loadAsync(fs.readFileSync(file));
  const parts = Object.keys(zip.files).filter(n => /^ppt\/theme\/theme\d+\.xml$/.test(n));
  for (const n of parts) {
    let x = await zip.file(n).async("string");
    x = x.replace(/(<a:(?:major|minor)Font>\s*<a:latin typeface=")[^"]*(")/g, `$1${LATIN}$2`)
         .replace(/(<a:(?:major|minor)Font>(?:(?!<\/a:(?:major|minor)Font>).)*?<a:ea typeface=")[^"]*(")/gs, `$1${EA}$2`)
         .replace(/(<a:font script="Jpan" typeface=")[^"]*(")/g, `$1${EA}$2`);
    zip.file(n, x);
  }
  fs.writeFileSync(file, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
  return parts;
}

pres.writeFile({ fileName: OUT }).then(async () => {
  const { applyTheme } = require(path.join(process.env.PPTX_SKILL, "scripts/apply_theme.js"));
  await applyTheme(OUT, THEME);
  const parts = await patchThemeFonts(OUT);
  console.log("wrote " + OUT + " (theme fonts patched in " + parts.join(", ") + ")");
});
