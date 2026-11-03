// build_denso_overview.js — the whole flow of the DENSO residual-strain study on one slide, minimal text:
// prediction chain as thumbnails and the three stages. Figure: python3 slides/figure_sources/denso_overview_flow.py
// Typography as build_denso_kenkyukai_deck.js. Panels: python3 slides/figure_sources/denso_onepage_panels.py
//
// Run:  NODE_PATH="$(npm root -g)" PPTX_SKILL=/mnt/skills/public/pptx node slides/build_denso_overview.js slides/denso_overview.pptx
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");
const OUT = process.argv[2] || "denso_overview.pptx";
const ROOT = path.resolve(__dirname, "..");
const F = (p) => path.join(ROOT, p);

const LATIN = "Times New Roman";
const EA = "游明朝";
const BRACKET = "Cambria Math";          // U+27E8/27E9 are absent from Times New Roman
const THEME = {
  name: "Academic Mincho Times", headFontFace: LATIN, bodyFontFace: LATIN,
  colors: { dk1: "111111", lt1: "FFFFFF", dk2: "444444", lt2: "F3F3F3", accent1: "333333",
    accent2: "555555", accent3: "777777", accent4: "999999", accent5: "BBBBBB",
    accent6: "DDDDDD", hlink: "1F3A5F", folHlink: "555555" },
};
const INK = "111111", MUTED = "555555", RULE = "BFBFBF";
const FOOT = "平板は既報の実測（試験片 1 枚、図の読み取り値）。3 次元部品は公開の Moldflow 解析データ（142 形状、途中経過）で実測ではない。材料定数は一般値。";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.theme = { headFontFace: LATIN, bodyFontFace: LATIN };
pres.author = "西岡 佳祐";
pres.title = "残留ひずみ・反り予測の全体の流れ";
const C = pres.SchemeColor;

pres.defineSlideMaster({
  title: "TITLE", background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.9, y: 1.5, w: 11.5, h: 1.7,
      fontSize: 30, bold: true, color: C.text1, valign: "bottom", align: "center" }, text: " " } },
    { placeholder: { options: { name: "body", type: "body", x: 0.9, y: 3.35, w: 11.5, h: 0.9,
      fontSize: 16, color: C.text1, valign: "top", align: "center" }, text: " " } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT", background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 0.4, w: 11.9, h: 0.6,
      fontSize: 24, bold: true, color: C.text1, valign: "middle", align: "left" }, text: " " } },
    { line: { x: 0.7, y: 1.02, w: 11.9, h: 0, line: { color: RULE, width: 0.75 } } },
    { text: { text: FOOT, options: { x: 0.7, y: 6.95, w: 9.5, h: 0.3, fontSize: 9,
      color: "8A8A8A", isTextBox: true, margin: 0 } } },
    { placeholder: { options: { name: "slideNumber", type: "sldNum", x: 12.1, y: 6.95, w: 0.5,
      h: 0.3, fontSize: 9, color: "8A8A8A", align: "right" } } },
  ],
});

// ---------------------------------------------------------- math run helpers ----
const it = (t) => ({ text: t, options: { italic: true } });                  // variable
const bi = (t) => ({ text: t, options: { italic: true, bold: true } });      // vector/tensor
const ro = (t) => ({ text: t, options: { italic: false } });                 // upright
const si = (t) => ({ text: t, options: { italic: true, subscript: true } }); // index sub
const sr = (t) => ({ text: t, options: { italic: false, subscript: true } }); // label sub
const sup = (t) => ({ text: t, options: { italic: false, superscript: true } });
const supi = (t) => ({ text: t, options: { italic: true, superscript: true } });
const lb = () => ({ text: "⟨", options: { italic: false, fontFace: BRACKET } });
const rb = () => ({ text: "⟩", options: { italic: false, fontFace: BRACKET } });
const tx = (t, o = {}) => ({ text: t, options: o });
const br = (t, o = {}) => ({ text: t, options: Object.assign({ breakLine: true }, o) });

const eq = (s, x, y, w, runs, no, size = 17) => {
  s.addText(runs, { x, y, w: w - 0.8, h: 0.5, fontSize: size, color: INK, align: "center",
    valign: "middle", isTextBox: true, margin: 0 });
  if (no) s.addText(no, { x: x + w - 0.75, y, w: 0.7, h: 0.5, fontSize: 14, color: INK,
    align: "right", valign: "middle", isTextBox: true, margin: 0 });
};
const head = (s, x, y, w, t) => s.addText(t, { x, y, w, h: 0.34, fontSize: 14.5, bold: true,
  color: INK, isTextBox: true, margin: 0 });
const para = (s, x, y, w, h, runs, size = 13) => s.addText(runs, { x, y, w, h, fontSize: size,
  color: INK, isTextBox: true, margin: 0, lineSpacing: Math.round(size * 1.5), valign: "top" });
const cap = (s, x, y, w, h, runs) => para(s, x, y, w, h, runs, 10.5);
const rule = (s, y, x = 0.7, w = 11.9) => s.addShape(pres.ShapeType.line,
  { x, y, w, h: 0, line: { color: RULE, width: 0.75 } });
const tableOpts = (x, y, w, colW, rowH) => ({ x, y, w, colW, rowH,
  border: { type: "solid", color: RULE, pt: 0.75 }, margin: 0.06 });
const cell = (t, o = {}) => ({ text: t, options: Object.assign({ fontSize: 12, color: INK,
  valign: "middle", align: "center" }, o) });
const hdr = (arr) => arr.map((t) => cell(t, { bold: true, fill: "F3F3F3" }));

const img = (s, p, x, y, w, aspect) => s.addImage({ path: p, x, y, w, h: w / aspect });

let s = pres.addSlide({ masterName: "CONTENT" });
s.addText("短繊維強化樹脂の射出成形品：残留ひずみ・反り予測の流れ", { placeholder: "title", fontSize: 24 });
img(s, F("slides/figures_paper/overview_flow.png"), 0.55, 1.05, 12.0, 2.1667);
s.addText([tx("次：", { bold: true }), tx("デンソー様の実データ（流動解析・実部品の計測）で、入口の繊維の向きと補正を確かめる")],
  { x: 0.7, y: 6.6, w: 11.9, h: 0.3, fontSize: 12.5, color: "1F3A5F", isTextBox: true, margin: 0 });

// ----------------------------------------------- write, theme, Mincho patch ----
function loadJSZip() {
  return require(require.resolve("jszip", { paths: [require.resolve("pptxgenjs")] }));
}
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
  return parts;
}

pres.writeFile({ fileName: OUT }).then(async () => {
  const { applyTheme } = require(path.join(process.env.PPTX_SKILL, "scripts/apply_theme.js"));
  await applyTheme(OUT, THEME);
  const parts = await patchThemeFonts(OUT);
  console.log("wrote " + OUT + " (theme fonts patched in " + parts.join(", ") + ")");
});
