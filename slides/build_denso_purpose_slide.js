// build_denso_purpose_slide.js — one slide: the purpose of the DENSO residual-strain study as a
// four-step flow with an evidence panel under each step, built on the DENSO / Polyplastics paper
// (Tsuda et al., J. Jpn. Soc. Compos. Mater. 52(3) 88–96, 2026). Academic register (CLAUDE.md):
// white background, one accent colour (only for what this study adds), Mincho / Times through the
// theme patch, numbered equation, every panel captioned with its source.
// Panels: python3 slides/figure_sources/denso_purpose_panels.py
// Run:    NODE_PATH="$(npm root -g)" PPTX_SKILL=/mnt/skills/public/pptx \
//         node slides/build_denso_purpose_slide.js slides/denso_purpose_flow.pptx
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const OUT = process.argv[2] || "denso_purpose_flow.pptx";
const FIG = (n) => path.join(__dirname, "figures_paper", n);
const LATIN = "Times New Roman", EA = "游明朝";
const INK = "111111", MUTED = "5A5A5A", RULE = "BFBFBF", ACCENT = "1F3A5F";
const THEME = { name: "Academic Mincho Times", headFontFace: LATIN, bodyFontFace: LATIN,
  colors: { dk1: "111111", lt1: "FFFFFF", dk2: "444444", lt2: "F3F3F3", accent1: "333333",
    accent2: "555555", accent3: "777777", accent4: "999999", accent5: "BBBBBB", accent6: "DDDDDD",
    hlink: "1F3A5F", folHlink: "555555" } };

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.theme = { headFontFace: LATIN, bodyFontFace: LATIN };
pres.author = "西岡 佳祐";
pres.title = "本研究の目的";

const it = (t, o = {}) => ({ text: t, options: Object.assign({ italic: true }, o) });
const bi = (t, o = {}) => ({ text: t, options: Object.assign({ italic: true, bold: true }, o) });
const ro = (t, o = {}) => ({ text: t, options: Object.assign({ italic: false }, o) });
const si = (t, o = {}) => ({ text: t, options: Object.assign({ italic: true, subscript: true }, o) });
const sr = (t, o = {}) => ({ text: t, options: Object.assign({ italic: false, subscript: true }, o) });
const tx = (t, o = {}) => ({ text: t, options: o });
const br = (t, o = {}) => ({ text: t, options: Object.assign({ breakLine: true }, o) });
const line = (s, x, y, w, color = RULE, width = 0.75) =>
  s.addShape(pres.ShapeType.line, { x, y, w, h: 0, line: { color, width } });

const s = pres.addSlide();
s.background = { color: "FFFFFF" };

// ---------------------------------------------------------------- title + purpose ----
s.addText([tx("本研究の目的", { bold: true, fontSize: 25 }),
  tx("　結晶化度と繊維配向からの残留ひずみ予測", { fontSize: 15, color: MUTED })],
  { x: 0.6, y: 0.28, w: 10.0, h: 0.55, color: INK, isTextBox: true, margin: 0, valign: "bottom" });
s.addText([tx("太字・紺は本研究で加える要素", { color: ACCENT })],
  { x: 9.6, y: 0.46, w: 3.14, h: 0.3, fontSize: 10, align: "right", isTextBox: true, margin: 0 });
line(s, 0.6, 0.9, 12.14);
s.addText([tx("目的：", { bold: true }),
  tx("射出成形した短繊維強化樹脂について、結晶化度と繊維配向から成形後の残留ひずみを予測するモデルを、数値解析による事前学習と少数の実測点による転移学習で構築し、計算と実測の定量差を補正する。")],
  { x: 0.6, y: 0.98, w: 12.14, h: 0.6, fontSize: 13.5, color: INK, isTextBox: true, margin: 0, lineSpacing: 19, valign: "top" });

// ---------------------------------------------------------------- four-step flow -----
const X0 = 0.6, CW = 2.86, CG = 0.233;
const cx = (i) => X0 + i * (CW + CG);
const steps = [
  { no: "1", head: "入力", added: true,
    desc: [it("X"), sr("c"), ro("("), it("z"), ro(")"), tx(" 結晶化度、"), it("a"), si("ij"), ro("("), it("z"), ro(")"), tx(" 繊維配向（X線CT または流動解析）、成形条件 "), bi("s")],
    fig: "purpose_p1_orientation.png",
    cap: [tx("繊維配向の板厚方向分布（GF30/PPS、3.08 mm 板）。既報 図8(a) の読み取り値。")] },
  { no: "2", head: "数値解析", added: false,
    desc: [tx("Ansys で残留ひずみの場を多数の条件で計算し、事前学習のデータとする")],
    fig: "purpose_p2_stress.png",
    cap: [tx("既報の入力による残留応力（本リポジトリの計算、全量形式）。× は既報の表面値 −24.3／+15.2 MPa。")] },
  { no: "3", head: "計算と実測の差", added: false,
    desc: [tx("傾向は合うが、解析は実測より大きい（既報 1.4 倍、本計算 1.5 倍）")],
    fig: "purpose_p4_hole_drilling.png",
    cap: [tx("穿孔法の解放ひずみと深さ。実測と既報の解析は既報 図12 の読み取り値、本計算は既報と同じ入力（Abaqus、本番メッシュ）。")] },
  { no: "4", head: "転移学習による補正", added: true,
    desc: [tx("数値解析で事前学習した "), it("f"), si("θ"), tx(" を、デンソー様の少数の点計測で合わせ込む")],
    fig: "purpose_p3_transfer.png",
    cap: [tx("測定枚数と予測誤差。合成データによる予備検討（定数は例示値、実データでは未実施）。")] },
];
const yH = 1.72, yD = 2.12, yF = 2.66, FA = 896 / 594, yC = yF + CW / FA + 0.04;
steps.forEach((st, i) => {
  const col = st.added ? ACCENT : INK;
  s.addText([tx(st.no + "　", { bold: true, color: col }), tx(st.head, { bold: true, color: col })],
    { x: cx(i), y: yH, w: CW, h: 0.34, fontSize: 15, isTextBox: true, margin: 0, valign: "middle" });
  line(s, cx(i), yH + 0.37, CW, st.added ? ACCENT : "8C8C8C", st.added ? 1.75 : 0.75);
  s.addText(st.desc, { x: cx(i), y: yD, w: CW, h: 0.52, fontSize: 11, color: INK, isTextBox: true,
    margin: 0, lineSpacing: 14.5, valign: "top" });
  s.addImage({ path: FIG(st.fig), x: cx(i), y: yF, w: CW, h: CW / FA });
  s.addText([tx(`図${i + 1}　`, { bold: true }), ...st.cap], { x: cx(i), y: yC, w: CW, h: 0.5,
    fontSize: 9, color: MUTED, isTextBox: true, margin: 0, lineSpacing: 12, valign: "top" });
  if (i < steps.length - 1) s.addShape(pres.ShapeType.line, { x: cx(i) + CW + 0.02, y: yH + 0.17,
    w: CG - 0.1, h: 0, line: { color: INK, width: 1.25, endArrowType: "triangle" } });
});

// ---------------------------------------------------------------- equation + additions
const yB = 5.25;
line(s, 0.6, yB - 0.08, 12.14);
s.addText([it("ε"), sr("res"), ro("("), it("z"), ro(")  =  "), it("f"), si("θ"), ro("( "),
  it("X"), sr("c"), ro("("), it("z"), ro("), "), it("a"), si("ij"), ro("("), it("z"), ro("), "),
  bi("m", { color: ACCENT }), ro(" ; "), bi("s"), ro(" )")],
  { x: 0.6, y: yB + 0.05, w: 5.3, h: 0.55, fontSize: 19, color: INK, align: "center", valign: "middle", isTextBox: true, margin: 0 });
s.addText("(1)", { x: 5.9, y: yB + 0.05, w: 0.55, h: 0.55, fontSize: 14, color: INK, align: "right", valign: "middle", isTextBox: true, margin: 0 });
s.addText([it("f"), si("θ"), tx("：予測モデル（"), it("θ"), tx(" は学習パラメータ）、"),
  bi("m", { color: ACCENT }), tx("：断面の要約量（板厚平均と一次モーメント）、"), bi("s"), tx("：成形条件（金型温度など）。"),
  tx("残留ひずみは断面全体のつり合いで決まるので、点の量に加えて "), bi("m", { color: ACCENT }), tx(" を入力とする。")],
  { x: 0.6, y: yB + 0.68, w: 5.9, h: 0.9, fontSize: 10.5, color: INK, isTextBox: true, margin: 0, lineSpacing: 14, valign: "top" });

s.addShape(pres.ShapeType.line, { x: 6.78, y: yB, w: 0, h: 1.52, line: { color: RULE, width: 0.75 } });
s.addText("既報に対して本研究で加える要素", { x: 7.0, y: yB, w: 5.74, h: 0.3, fontSize: 12.5, bold: true, color: ACCENT, isTextBox: true, margin: 0 });
s.addText([
  tx("(i)　結晶化度の入力：", { bold: true, color: ACCENT }), br("既報は金型 150 °C・アニールで影響小と判断。金型温度の低い条件まで対象を広げる。"),
  tx("(ii)　断面の情報 ", { bold: true, color: ACCENT }), bi("m", { color: ACCENT }), tx("：", { bold: true, color: ACCENT }),
  br("点の量だけでは不足（予備検討で決定係数 0.914 → 0.9995、合成データ）。"),
  tx("(iii)　転移学習：", { bold: true, color: ACCENT }), tx("計算と実測の定量差（既報で約 40 %、同じ入力の本計算で約 50 %）を、少数の点計測で補正する。"),
], { x: 7.0, y: yB + 0.33, w: 5.74, h: 1.2, fontSize: 10.5, color: INK, isTextBox: true, margin: 0, lineSpacing: 14, paraSpaceAfter: 2, valign: "top" });

// ---------------------------------------------------------------- footer -------------
s.addText("既報：津田・白桃・田中・國立・望月、日本複合材料学会誌 52(3), 88–96 (2026)。既報の図の値は読み取り値（data/literature/tsuda2026_jscm52_88.json）。手順 4 は計画であり、実データでは未実施。以後、段階② 変形・強度、段階③ 成形条件の設計へ展開する。",
  { x: 0.6, y: 6.9, w: 12.14, h: 0.4, fontSize: 9, color: MUTED, isTextBox: true, margin: 0, lineSpacing: 12 });
s.addNotes("左から：1 入力（配向の実測例）、2 数値解析（既報の入力での残留応力）、3 計算と実測の差（既報で約 40 %）、4 転移学習による補正（合成データでの予備検討）。紺は本研究で加える要素。");

// ----------------------------------------------- write, theme, Mincho patch ----
function loadJSZip() { return require(require.resolve("jszip", { paths: [require.resolve("pptxgenjs")] })); }
async function patchThemeFonts(file) {
  const zip = await loadJSZip().loadAsync(fs.readFileSync(file));
  const parts = Object.keys(zip.files).filter((n) => /^ppt\/theme\/theme\d+\.xml$/.test(n));
  for (const n of parts) {
    let x = await zip.file(n).async("string");
    x = x.replace(/(<a:(?:major|minor)Font>\s*<a:latin typeface=")[^"]*(")/g, `$1${LATIN}$2`)
      .replace(/(<a:(?:major|minor)Font>(?:(?!<\/a:(?:major|minor)Font>).)*?<a:ea typeface=")[^"]*(")/gs, `$1${EA}$2`)
      .replace(/(<a:font script="Jpan" typeface=")[^"]*(")/g, `$1${EA}$2`);
    zip.file(n, x);
  }
  fs.writeFileSync(file, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
}
pres.writeFile({ fileName: OUT }).then(async () => {
  const { applyTheme } = require(path.join(process.env.PPTX_SKILL, "scripts/apply_theme.js"));
  await applyTheme(OUT, THEME);
  await patchThemeFonts(OUT);
  console.log("wrote " + OUT);
});
