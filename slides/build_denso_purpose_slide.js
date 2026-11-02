// build_denso_purpose_slide.js — one slide: the purpose of the DENSO residual-strain study as an
// algorithm flow, set against the procedure of the DENSO / Polyplastics paper it builds on
// (Tsuda et al., J. Jpn. Soc. Compos. Mater. 52(3) 88–96, 2026; values in
// data/literature/tsuda2026_jscm52_88.json). Academic register (CLAUDE.md): white background,
// one accent colour (used only for the boxes this study adds), Mincho / Times via the theme patch.
//
// Run:  NODE_PATH="$(npm root -g)" PPTX_SKILL=/mnt/skills/public/pptx \
//       node slides/build_denso_purpose_slide.js slides/denso_purpose_flow.pptx
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const OUT = process.argv[2] || "denso_purpose_flow.pptx";
const LATIN = "Times New Roman", EA = "游明朝", BRACKET = "Cambria Math";
const INK = "111111", MUTED = "555555", RULE = "8C8C8C", ACCENT = "1F3A5F";
const THEME = { name: "Academic Mincho Times", headFontFace: LATIN, bodyFontFace: LATIN,
  colors: { dk1: "111111", lt1: "FFFFFF", dk2: "444444", lt2: "F3F3F3", accent1: "333333",
    accent2: "555555", accent3: "777777", accent4: "999999", accent5: "BBBBBB", accent6: "DDDDDD",
    hlink: "1F3A5F", folHlink: "555555" } };

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.theme = { headFontFace: LATIN, bodyFontFace: LATIN };
pres.author = "西岡 佳祐";
pres.title = "本研究の目的";

const it = (t) => ({ text: t, options: { italic: true } });
const bi = (t) => ({ text: t, options: { italic: true, bold: true } });
const ro = (t, o = {}) => ({ text: t, options: Object.assign({ italic: false }, o) });
const si = (t) => ({ text: t, options: { italic: true, subscript: true } });
const sr = (t) => ({ text: t, options: { italic: false, subscript: true } });
const sup = (t) => ({ text: t, options: { italic: false, superscript: true } });
const lb = () => ({ text: "⟨", options: { italic: false, fontFace: BRACKET } });
const rb = () => ({ text: "⟩", options: { italic: false, fontFace: BRACKET } });
const tx = (t, o = {}) => ({ text: t, options: o });
const br = (t, o = {}) => ({ text: t, options: Object.assign({ breakLine: true }, o) });

const s = pres.addSlide();
s.background = { color: "FFFFFF" };
s.addText("本研究の目的 — 既報の解析手順と、本研究で加える要素", { x: 0.6, y: 0.32, w: 12.1, h: 0.55,
  fontSize: 23, bold: true, color: INK, isTextBox: true, margin: 0 });
s.addShape(pres.ShapeType.line, { x: 0.6, y: 0.92, w: 12.1, h: 0, line: { color: "BFBFBF", width: 0.75 } });

// ---------------------------------------------------------------- flow helpers ----
const BW = 2.2, GAP = 0.27, X0 = 0.6;
const bx = (i) => X0 + i * (BW + GAP);
function box(i, y, h, head, body, accent = false) {
  s.addShape(pres.ShapeType.rect, { x: bx(i), y, w: BW, h,
    line: { color: accent ? ACCENT : RULE, width: accent ? 2.0 : 0.75 }, fill: { color: "FFFFFF" } });
  s.addText([br(head, { bold: true, fontSize: 12.5, color: accent ? ACCENT : INK }), ...body],
    { x: bx(i) + 0.08, y: y + 0.05, w: BW - 0.16, h: h - 0.1, fontSize: 11, color: INK,
      isTextBox: true, margin: 0, valign: "top", lineSpacing: 15 });
}
function arrowH(i, y) {
  s.addShape(pres.ShapeType.line, { x: bx(i) + BW + 0.03, y, w: GAP - 0.06, h: 0,
    line: { color: INK, width: 1.0, endArrowType: "triangle" } });
}
function arrowV(x, y1, y2, dashed = false) {
  s.addShape(pres.ShapeType.line, { x, y: y1, w: 0, h: y2 - y1,
    line: { color: ACCENT, width: 1.0, endArrowType: "triangle", dashType: dashed ? "dash" : "solid" } });
}

// ------------------------------------------------------------- (a) the paper ------
s.addText([tx("(a)　既報の手順", { bold: true }), tx("　津田ら（デンソー・ポリプラスチックス），日本複合材料学会誌 52(3), 88–96, 2026", { color: MUTED, fontSize: 11 })],
  { x: 0.6, y: 1.05, w: 12.1, h: 0.32, fontSize: 13.5, color: INK, isTextBox: true, margin: 0 });
const yA = 1.42, hA = 1.12;
box(0, yA, hA, "配向の実測", [tx("X線マイクロCT で板厚方向の "), it("a"), si("ij"), ro("("), it("z"), ro(")")]);
box(1, yA, hA, "平均場均質化", [tx("Mori–Tanaka（Digimat-MF）で層ごとの "), lb(), bi("C"), rb(), ro(", "), lb(), bi("α"), rb()]);
box(2, yA, hA, "多層の有限要素解析", [tx("Abaqus、20 µm × 154 層、熱弾性、"), it("T"), sr("0"), tx(" = 180 °C から一様冷却")]);
box(3, yA, hA, "残留応力", [tx("表面中心で "), it("σ"), si("xx"), tx(" = −24.3 MPa、"), it("σ"), si("yy"), tx(" = +15.2 MPa")]);
box(4, yA, hA, "穿孔法による検証", [tx("φ1.6 mm、深さ 2.0 mm まで 0.2 mm ごとに解放ひずみを計測")]);
for (let i = 0; i < 4; i++) arrowH(i, yA + hA / 2);
s.addText([tx("結果：", { bold: true }), tx("符号と深さ方向の傾向は一致（定性的に一致）。定量的には、解析の解放ひずみが約 40 % 大きい。"),
  tx("著者らの挙げる原因：", { bold: true }), tx("応力 0 とする温度 "), it("T"), sr("0"),
  tx(" の仮定（冷却過程の温度変化と応力緩和）、および結晶化度の分布（金型 150 °C・アニールでは小さいと判断）。")],
  { x: 0.6, y: 2.62, w: 12.1, h: 0.55, fontSize: 11.5, color: INK, isTextBox: true, margin: 0, lineSpacing: 16, valign: "top" });

// ------------------------------------------------------------- (b) this study -----
s.addText([tx("(b)　本研究の手順（計画）", { bold: true }), tx("　太枠は既報に対して本研究で加える要素", { color: MUTED, fontSize: 11 })],
  { x: 0.6, y: 3.25, w: 12.1, h: 0.32, fontSize: 13.5, color: INK, isTextBox: true, margin: 0 });
const yB = 4.05, hB = 1.25;
box(0, yB, hB, "入力", [it("X"), sr("c"), ro("("), it("z"), ro(")"), tx("：結晶化度"), ro("", { breakLine: true }),
  it("a"), si("ij"), ro("("), it("z"), ro(")"), tx("：繊維配向"), ro("", { breakLine: true }), tx("成形条件（金型温度など）")], true);
box(1, yB, hB, "数値解析", [tx("Ansys により条件を変えて残留ひずみの場を計算する（多数・ノイズなし）")]);
box(2, yB, hB, "事前学習", [tx("場のデータで予測モデル "), it("f"), si("θ"), tx(" を学習する")], true);
box(3, yB, hB, "転移学習（較正）", [tx("デンソー様の点計測（少数、穿孔法を想定）で "), it("f"), si("θ"), tx(" を合わせ込む")], true);
box(4, yB, hB, "予測", [tx("成形後の残留ひずみ "), it("ε"), sr("res"), ro("", { breakLine: true }),
  tx("→ 変形・強度（段階②）、成形条件の設計（段階③）へ")]);
for (let i = 0; i < 4; i++) arrowH(i, yB + hB / 2);
// links from the paper's open issues to the added elements
arrowV(bx(0) + BW / 2, 3.6, yB - 0.02, true);
arrowV(bx(3) + BW / 2, 3.6, yB - 0.02, true);
s.addText("既報の課題：結晶化度の分布", { x: bx(0) + BW / 2 + 0.08, y: 3.66, w: 2.5, h: 0.28, fontSize: 10, color: ACCENT, isTextBox: true, margin: 0 });
s.addText("既報の課題：計算と実測の定量差（約 40 %）", { x: bx(3) + BW / 2 + 0.08, y: 3.66, w: 3.2, h: 0.28, fontSize: 10, color: ACCENT, isTextBox: true, margin: 0 });

// ------------------------------------------------------------- purpose + eq. ------
s.addShape(pres.ShapeType.line, { x: 0.6, y: 5.42, w: 12.1, h: 0, line: { color: "BFBFBF", width: 0.75 } });
s.addText([tx("目的：", { bold: true }),
  tx("結晶化度と繊維配向から射出成形後の残留ひずみを予測するモデルを、数値解析による事前学習と少数の実測点による転移学習により構築し、計算と実測の定量差を補正する。")],
  { x: 0.6, y: 5.52, w: 12.1, h: 0.6, fontSize: 13.5, color: INK, isTextBox: true, margin: 0, lineSpacing: 20, valign: "top" });
s.addText([it("ε"), sr("res"), ro("("), it("z"), ro(") = "), it("f"), si("θ"), ro("("), it("X"), sr("c"), ro("("), it("z"), ro("), "),
  it("a"), si("ij"), ro("("), it("z"), ro("), "), bi("m"), ro("; "), bi("s"), ro(")")],
  { x: 0.6, y: 6.16, w: 5.6, h: 0.45, fontSize: 17, color: INK, align: "center", valign: "middle", isTextBox: true, margin: 0 });
s.addText("(1)", { x: 6.0, y: 6.16, w: 0.6, h: 0.45, fontSize: 14, color: INK, align: "right", valign: "middle", isTextBox: true, margin: 0 });
s.addText([bi("m"), tx("：断面の要約量（板厚平均・一次モーメント）。残留ひずみは断面全体のつり合いで決まるため、点の量だけでは不足する（予備検討、合成データ）。"),
  bi("s"), tx("：成形条件。")],
  { x: 6.8, y: 6.12, w: 5.9, h: 0.6, fontSize: 10.5, color: INK, isTextBox: true, margin: 0, lineSpacing: 14, valign: "top" });
s.addText("注：(b) は計画であり、実データでは未実施。予備検討は合成データと例示値（未校正）による。既報の数値は同論文の本文・表による。",
  { x: 0.6, y: 6.9, w: 12.1, h: 0.3, fontSize: 9.5, color: MUTED, isTextBox: true, margin: 0 });
s.addNotes("上段は既報（津田ら 2026）の手順と、著者らが挙げた課題。下段は本研究の計画。課題から本研究で加える要素へ点線で結んだ。");

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
