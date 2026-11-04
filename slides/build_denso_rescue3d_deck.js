// build_denso_rescue3d_deck.js — short talk: RESCUE on 3-D parts (Greif & Meyer geometries, fifa §19,
// interim 46 geometries). Academic register and typography as
// build_denso_analysis_deck.js (Mincho / Times, theme patched after writing). Every number comes from
// a file in the repository: abaqus/denso_plate/abaqus_results/greif3d_rescue/, results/greif3d_rescue/.
// Figures: python3 slides/figure_sources/denso_rescue3d_figs.py
//
// Run:  NODE_PATH="$(npm root -g)" PPTX_SKILL=/mnt/skills/public/pptx //       node slides/build_denso_rescue3d_deck.js slides/denso_rescue3d.pptx
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const OUT = process.argv[2] || "denso_rescue3d.pptx";
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
const FOOT = "繊維強化樹脂の残留ひずみ予測（株式会社デンソー様との共同研究）／3 次元部品での検討・途中経過（142 形状）・データは公開の解析結果";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.theme = { headFontFace: LATIN, bodyFontFace: LATIN };
pres.author = "西岡 佳祐";
pres.title = "3 次元部品における少数点計測による反り予測の補正";
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

// ---------------------------------------------------------------- numbers ----
const RES = JSON.parse(fs.readFileSync(F("abaqus/denso_plate/abaqus_results/greif3d_rescue/summary.json")));
const TF = JSON.parse(fs.readFileSync(F("abaqus/denso_plate/abaqus_results/greif3d_rescue/transfer_factor.json")));
const ST = JSON.parse(fs.readFileSync(F("results/greif3d_rescue/selftest.json")));
const NG = RES.n;
const pc = (v) => `${Math.round(v * 100)} %`;
const md = (k) => RES.summary[k].median;
const iq = (k) => `${Math.round(RES.summary[k].q25 * 100)}〜${Math.round(RES.summary[k].q75 * 100)} %`;

// ===================================================== title ================
let s = pres.addSlide({ masterName: "TITLE" });
s.addText("3 次元部品における少数点の計測による反り予測の補正", { placeholder: "title", fontSize: 28 });
s.addText("— 平板で示した補正法の 3 次元部品への適用（途中経過）—", { placeholder: "body" });
rule(s, 4.4, 4.15, 5.0);
s.addText([br("西岡 佳祐"), tx("2026年11月")],
  { x: 0.9, y: 4.6, w: 11.5, h: 0.9, fontSize: 15, color: INK, align: "center", isTextBox: true, margin: 0, lineSpacing: 24 });
s.addText(`注：反りの「データ」は公開の Moldflow 解析結果であり、実測ではない。${NG} 形状の途中経過である。`,
  { x: 0.9, y: 6.3, w: 11.5, h: 0.4, fontSize: 11, color: MUTED, align: "center", isTextBox: true, margin: 0 });

// ===================================================== 1. 背景と問い =========
s = pres.addSlide({ masterName: "CONTENT" });
s.addText("1. 背景と問い", { placeholder: "title" });
para(s, 0.7, 1.25, 11.9, 1.5, [
  br("これまでに、平板（既報[1]の試験片 1 枚）で、物理モデルの予測を穿孔法の少数の実測点で補正できることを示した。"),
  tx("しかし、多数の 3 次元部品を扱う先行研究[2]と比べると、平板 1 枚では手法の一般性を示せない。そこで、公開されている 3 次元部品の解析データで、同じ手順を試した。"),
], 13.5);
head(s, 0.7, 2.95, 11.9, "問い");
para(s, 0.7, 3.35, 11.9, 1.2, [
  br("(1) 3 次元の部品でも、表面の数点の反りだけで、部品全体の反りの予測を直せるか。"),
  tx("(2) 補正量（倍率）は部品の形によらず同じか。同じなら、新しい部品は 1 点も測らずに予測できる。"),
], 13.5);
img(s, F("slides/figures_paper/rescue3d_gallery.png"), 1.4, 4.45, 10.5, 4.8);

// ===================================================== 2. データ =============
s = pres.addSlide({ masterName: "CONTENT" });
s.addText("2. 用いたデータ", { placeholder: "title" });
para(s, 0.7, 1.2, 11.9, 0.5, [tx("Greif と Meyer[2,3] が公開した射出成形の解析結果を用いた。中身は fifa で確認した（ファイル構成・単位・成分の並び）。")], 13.5);
s.addTable([hdr(["項目", "内容"]),
  ...[["出典", "Greif & Meyer, Zenodo 17831586（CC BY 4.0）。保圧を外して反りを大きくした解析"],
    ["材料・解析", "PP/GF30（Celanese Factor PP GF 30）、Autodesk Moldflow 2023"],
    ["形状", "196 形状、4 節点四面体 2.5 万〜36.5 万要素、大きさ 17〜354 mm"],
    ["使う量", "各節点の繊維配向テンソル（入力）、反りの変位（「計測」の代わり）"],
    ["反りの大きさ", "最大値 0.08〜5.3 mm（中央値 0.53 mm）"],
    ["注意", "解析結果であり実測ではない（先行研究[2]と同じ限界）。材料は PP で本研究の PPS と異なる。配向の非対角成分の並びは推定"]].map((r) => r.map((c, i) => cell(c, { align: "left", bold: i === 0 })))],
  tableOpts(0.7, 1.8, 11.9, [2.4, 9.5], 0.44));
s.addTable([hdr(["", "平板（これまで）", "3 次元部品（今回）"]),
  ...[["対象", "PPS/GF30 平板 1 枚", `PP/GF30 の部品 ${NG} 形状（196 形状のうち）`],
    ["「計測」", "穿孔法の解放ひずみ（実測）", "Moldflow の反り（解析）の数点"],
    ["補正するもの", "無応力温度（倍率 1 個）", "等方・異方の縮みの倍率 2 個"]].map((r) => r.map((c, i) => cell(c, { align: i === 0 ? "center" : "left", bold: i === 0 })))],
  tableOpts(0.7, 5.3, 11.9, [2.2, 4.6, 5.1], 0.38));

// ===================================================== 3. 手法 ===============
s = pres.addSlide({ masterName: "CONTENT" });
s.addText("3. 手法", { placeholder: "title" });
img(s, F("slides/figures_paper/rescue3d_flow.png"), 0.9, 1.15, 11.5, 4.318);
eq(s, 0.7, 3.95, 11.9, [bi("u"), sr("prior"), tx(" = "), bi("U"), sr("iso"), tx(" + "), bi("U"), sr("dev")], "(1)");
eq(s, 0.7, 4.5, 11.9, [bi("u"), tx(" = "), it("a"), sr("iso"), tx(" "), bi("U"), sr("iso"), tx(" + "), it("a"), sr("dev"), tx(" "), bi("U"), sr("dev"), tx(" + "), bi("t"), tx(" + "), bi("ω"), tx(" × ("), bi("X"), tx(" − "), bi("X"), sr("c"), tx(")")], "(2)");
para(s, 0.7, 5.1, 11.9, 1.8, [
  tx("ここで "), bi("U"), sr("iso"), tx("、"), bi("U"), sr("dev"), tx(" は、配向平均した熱膨張係数 "), lb(), bi("α"), rb(), tx(" の等方部分 tr"), lb(), bi("α"), rb(), tx("/3 と残りの異方部分による反り（FE、一様な温度変化 −100 K は例示）。補正では倍率 "), it("a"), sr("iso"), tx("、"), it("a"), sr("dev"), tx(" と剛体運動 "), bi("t"), tx("、"), bi("ω"), tx(" を、表面の "), it("K"), tx(" 点の反りに当てはめる（事前値 1 からの L2-SP[4]、計測の不確かさ 0.01 mm は例示）。"),
  br(""),
  tx("評価は全節点の相対誤差：予測とデータの差から最適な剛体運動を除いた rms を、データの反りの rms で割った値。"),
], 12.5);

// ===================================================== 4. 手法の確認 =========
s = pres.addSlide({ masterName: "CONTENT" });
s.addText("4. 手法の確認", { placeholder: "title" });
s.addTable([hdr(["確認", "内容", "結果"]),
  ...[["G1", "等方・一様な膨張（答えが分かっている）", `${ST.G1_patch.toExponential(1)}（機械精度）`],
    ["G2", "合成データ（倍率 0.7 と 1.3 を仕込む）を 10 点で補正", `${ST.G2.rescue_10_2f_a_median.map((v) => v.toFixed(3)).join("、")} を回復、誤差 ${pc(ST.G2.prior)} → ${pc(ST.G2.rescue_10_2f)}`],
    ["G3", "1 次四面体（C3D4）と 2 次四面体（C3D10）", `C3D4 は曲げで固く、反りが小さく出る → C3D10 を使う`],
    ["G3b", "C3D10 でメッシュを 2 倍に細分", `反りの変化 ${(ST.G3b_c3d10_refined_change * 100).toFixed(1)} %`],
    ["G4", "主方向の弾性率：本モデル ÷ データ", `${md("G4_E1_ratio_median").toFixed(2)}（材料定数が一般値のため低い）`],
    ["G5", "配向の群の数 32 → 64（3 形状）", "予測の変化 2〜11 %"]].map((r) => r.map((c, i) => cell(c, { align: i === 0 ? "center" : "left", bold: i === 0 })))],
  tableOpts(0.7, 1.3, 11.9, [1.0, 5.6, 5.3], 0.52));
cap(s, 0.7, 5.1, 11.9, 0.5, [tx("G1〜G3b はここで合成の板により確認、G4・G5 は fifa の計算による（`results/greif3d_rescue/selftest.json`、`summary.json`）。")]);
para(s, 0.7, 5.6, 11.9, 1.2, [tx("FE と補正の手順は正しく動く。一方、材料定数は一般値であり、弾性率はデータの約 0.57 倍である。この差も含めて、補正の倍率がどこまで吸収できるかを次に見る。")], 13);

// ===================================================== 5. 結果 ===============
s = pres.addSlide({ masterName: "CONTENT" });
s.addText("5. 結果：数点の計測で部品全体の反りが直る", { placeholder: "title" });
img(s, F("slides/figures_paper/rescue3d_errors.png"), 0.7, 1.15, 11.9, 3.077);
cap(s, 0.7, 5.05, 11.9, 0.4, [tx(`図　${NG} 形状の反りの相対誤差。(a) 補正のしかた別、(b) 最適な倍率と部品の大きさ、(c) 補正前の形の一致と補正後の誤差。`)]);
para(s, 0.7, 5.5, 11.9, 1.4, [
  br(`補正前の予測は、形はデータとよく合う（cos の中央値 ${md("cos_prior_data").toFixed(2)}）が、大きさが約半分で、誤差は ${pc(md("prior"))} である。`),
  tx(`表面の 10 点で補正すると誤差は ${pc(md("rescue_10_2f"))}（四分位 ${iq("rescue_10_2f")}）に下がり、全点を使った補正の限界 ${pc(md("field_2f"))} に近い。形の一致が低い部品ほど、補正後も誤差が残る。`),
], 13);

// ===================================================== 6. 部品の上で見る ======
s = pres.addSlide({ masterName: "CONTENT" });
s.addText("6. 結果：部品の上で見る", { placeholder: "title" });
img(s, F("slides/figures_paper/rescue3d_example.png"), 0.95, 1.1, 11.4, 2.222);
cap(s, 0.7, 6.3, 11.9, 0.55, [tx("図　上：補正がよく効く部品。補正前の予測は反りの形が合い、大きさが約半分。10 点（赤）で倍率を決めると全体が直る。下：効きにくい部品。反りの形そのものが合わず、倍率では直らない（補正前の予測と FE はここで再計算し、fifa の値と一致を確認）。")]);

// ===================================================== 7. 測定なし ============
s = pres.addSlide({ masterName: "CONTENT" });
s.addText("7. 結果：新しい部品は測らずに予測できるか", { placeholder: "title" });
para(s, 0.7, 1.25, 11.9, 1.1, [tx(`最適な倍率は部品によらずほぼ同じ（中央値 ${TF.a_star_median.toFixed(2)}、四分位 ${TF.a_star_q[0].toFixed(2)}〜${TF.a_star_q[1].toFixed(2)}）。そこで、1 つの部品を除いた残りの部品の倍率の中央値を、除いた部品に測定なしで用いた（一つ抜き）。`)], 13.5);
img(s, F("slides/figures_paper/rescue3d_transfer.png"), 0.7, 2.2, 11.9, 3.75);
cap(s, 0.7, 5.4, 11.9, 0.35, [tx(`図　左：一つ抜きの考え方。右：誤差の中央値。測定なしで悪化したのは ${NG} 形状中 ${TF.n_loo_worse_than_prior} 形状。`)]);
para(s, 0.7, 5.8, 11.9, 1.1, [tx("解釈：倍率は部品の形ではなく、材料と成形の性質（ここでは一般値の定数と例示の温度変化の誤差、結晶化収縮の欠落）を表すとみられる。ただし、全形状が同じ材料・同じ解析設定であり、材料や成形条件が変わったときに倍率が使えるかは確かめていない。")], 13);
s.addNotes("一つ抜きの誤差は、保存した prior 誤差と cos から倍率の 2 次式として厳密に求めた（validation/greif3d_transfer_factor.py、T1 で FE の後処理と 2e-14 で一致）。");

// ===================================================== 7. まとめ =============
s = pres.addSlide({ masterName: "CONTENT" });
s.addText("8. まとめと今後", { placeholder: "title" });
head(s, 0.7, 1.25, 11.9, "示せたこと");
s.addText([
  { text: `3 次元の部品 ${NG} 形状で、簡単な物理モデルの反りは形がデータとよく合い、大きさのずれは倍率で直せた。表面の 10 点で誤差は ${pc(md("prior"))} から ${pc(md("rescue_10_2f"))} に下がった。`, options: { bullet: true, breakLine: true } },
  { text: `倍率は部品によらずほぼ同じで、ほかの部品の倍率を使えば、測定なしでも誤差は ${pc(TF["LOO factor, no measurement"].median)} になった。`, options: { bullet: true } },
], { x: 0.7, y: 1.6, w: 11.9, h: 1.4, fontSize: 13, color: INK, paraSpaceAfter: 4, valign: "top" });
head(s, 0.7, 3.15, 11.9, "示せていないこと");
s.addText([
  { text: "データは解析結果であり、実測ではない。材料は PP で、本研究の PPS ではない。", options: { bullet: true, breakLine: true } },
  { text: "全形状が同じ材料・同じ解析設定。材料・成形条件が変わったときに倍率を使い回せるかは未確認。", options: { bullet: true, breakLine: true } },
  { text: `途中経過（${NG} 形状）。形の一致が低い部品（cos < 0.9）では補正後も誤差が 30〜60 % 残る。`, options: { bullet: true } },
], { x: 0.7, y: 3.5, w: 11.9, h: 1.5, fontSize: 13, color: INK, paraSpaceAfter: 4, valign: "top" });
head(s, 0.7, 5.15, 11.9, "今後");
s.addText([
  { text: "100 形状の最終結果。形の一致が低い部品の原因（配向の読み方、厚さ方向の要素数）を調べる。", options: { bullet: true, breakLine: true } },
  { text: "デンソー様の実部品の実測で、同じ手順を確かめる。", options: { bullet: true } },
], { x: 0.7, y: 5.5, w: 11.9, h: 1.0, fontSize: 13, color: INK, paraSpaceAfter: 4, valign: "top" });

// ===================================================== 参考文献 =============
s = pres.addSlide({ masterName: "CONTENT" });
s.addText("参考文献", { placeholder: "title" });
const refs = [
  "[1] 津田ほか, 日本複合材料学会誌, 52(3), 88–96, 2026.",
  "[2] N. Greif and N. Meyer, Composites Part A, 204, 109653, 2026.",
  "[3] N. Greif and N. Meyer, “Injection Molding Simulations with Artificially Enhanced Warpage,” Zenodo, record 17831586 (CC BY 4.0).",
  "[4] X. Li, Y. Grandvalet and F. Davoine, “Explicit inductive bias for transfer learning with convolutional networks,” Proc. ICML, 2018.",
  "[5] T. Mori and K. Tanaka, Acta Metallurgica, 21(5), 571–574, 1973.",
  "[6] S. G. Advani and C. L. Tucker III, Journal of Rheology, 31(8), 751–784, 1987.",
  "[7] G. Dhondt, CalculiX CrunchiX User's Manual, version 2.21.",
];
s.addText(refs.map((r, i) => ({ text: r, options: { breakLine: i < refs.length - 1 } })),
  { x: 0.7, y: 1.3, w: 11.9, h: 4.5, fontSize: 12, color: INK, lineSpacing: 18, paraSpaceAfter: 6, valign: "top" });
rule(s, 6.15);
s.addText("注：[2] の題目・著者名の表記、[1] の著者名の全員は、発表前に一次文献で確認すること。",
  { x: 0.7, y: 6.25, w: 11.9, h: 0.5, fontSize: 11, color: MUTED, isTextBox: true, margin: 0 });

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
