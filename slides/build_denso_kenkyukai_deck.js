// build_denso_kenkyukai_deck.js — kenkyukai talk for the DENSO residual-strain theme (outline:
// slides/OUTLINE_denso_kenkyukai.md). Academic register and typography as
// build_denso_analysis_deck.js (Mincho / Times, theme patched after writing). Every number comes from
// a file in the repository: results/denso_tsuda2026/, abaqus/denso_plate/abaqus_results/.
// Figure for slide 6: python3 slides/figure_sources/denso_kenkyukai_panels.py
//
// Run:  NODE_PATH="$(npm root -g)" PPTX_SKILL=/mnt/skills/public/pptx //       node slides/build_denso_kenkyukai_deck.js slides/denso_kenkyukai.pptx
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const OUT = process.argv[2] || "denso_kenkyukai.pptx";
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
const FOOT = "繊維強化樹脂の残留ひずみ予測（株式会社デンソー様との共同研究）／研究会発表・実測は既報の図からの読み取り値";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.theme = { headFontFace: LATIN, bodyFontFace: LATIN };
pres.author = "西岡 佳祐";
pres.title = "穿孔法の点計測による物理モデルの補正を用いた短繊維強化 PPS 射出成形品の残留ひずみ予測";
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

// ===================================================== title ================
let s = pres.addSlide({ masterName: "TITLE" });
s.addText("穿孔法の点計測による物理モデルの補正を用いた短繊維強化 PPS 射出成形品の残留ひずみ予測", { placeholder: "title", fontSize: 26 });
s.addText("— 数値解析による物理モデルと少数の実測点による補正 —", { placeholder: "body" });
rule(s, 4.4, 4.15, 5.0);
s.addText([br("西岡 佳祐"), br("慶應義塾大学 修士課程"), tx("2026年11月")],
  { x: 0.9, y: 4.6, w: 11.5, h: 1.2, fontSize: 15, color: INK, align: "center",
    isTextBox: true, margin: 0, lineSpacing: 24 });
s.addText("注：実測値は既報[1]の図から目で読み取った値である。デンソー様の実データによる検証は行っていない。",
  { x: 0.9, y: 6.3, w: 11.5, h: 0.4, fontSize: 11, color: MUTED, align: "center", isTextBox: true, margin: 0 });

// ===================================================== 1. 背景 ===============
s = pres.addSlide({ masterName: "CONTENT" });
s.addText("1. 背景", { placeholder: "title" });
para(s, 0.7, 1.25, 11.9, 1.9, [
  br("短繊維強化樹脂の射出成形品では、繊維が流動方向（MD）に配向し、MD と直角方向（TD）とで熱膨張・収縮が異なる。結晶性樹脂ではさらに結晶化収縮が加わる。この異方性と板厚方向の分布が残留ひずみを生み、反りや寸法誤差の原因となる。"),
  tx("津田ら[1]は、GF 30 wt% の PPS 平板について、配向（X 線 CT）と物性から残留応力を有限要素解析で求め、穿孔法の実測と比べた。"),
], 13.5);
s.addTable([hdr(["項目", "既報[1]の値・結果"]),
  ...[["材料", "DURAFIDE PPS 1130A6（GF 30 wt%、18.2 vol%、繊維アスペクト比 16.8）"],
    ["試験片", "20 × 20 × 3.08 mm（成形後に 180 °C × 4 h アニール）"],
    ["解析の仮定", "無応力温度 T₀ = 180 °C、弾性"],
    ["表面の残留応力（解析）", "σxx = −24.3 MPa、σyy = +15.2 MPa"],
    ["穿孔法との比較", "傾向は一致、解析は実測の約 1.3〜1.4 倍。T₀ ≈ 120 °C とすると合う"]].map((r) => r.map((c, i) => cell(c, { align: "left", bold: i === 0 })))],
  tableOpts(0.7, 3.35, 11.9, [3.2, 8.7], 0.46));
para(s, 0.7, 6.25, 11.9, 0.5, [tx("課題：解析と実測の定量的な差を、実部品ごとの多数の計測なしに補正する方法が要る。")], 13.5);
s.addNotes("data/literature/tsuda2026_jscm52_88.json。表の数値は既報の本文・表から。");

// ===================================================== 2. 目的 ===============
s = pres.addSlide({ masterName: "CONTENT" });
s.addText("2. 目的", { placeholder: "title" });
para(s, 0.7, 1.25, 11.9, 0.9, [tx("結晶化度と繊維配向から成形後の残留ひずみを予測するモデルを、数値解析による事前学習と、少数の実測点による補正（転移学習）とで構築する。")], 14);
eq(s, 0.7, 2.2, 11.9, [it("ε"), sr("res"), ro("("), it("z"), ro(") = "), it("f"), si("θ"), ro("("), it("X"), sr("c"), ro("("), it("z"), ro("), "), it("a"), si("ij"), ro("("), it("z"), ro("), "), bi("m"), ro(" ; "), it("s"), ro(")")], "(1)", 18);
para(s, 0.7, 2.8, 11.9, 0.7, [it("X"), sr("c"), tx("：結晶化度、"), it("a"), si("ij"), tx("：配向テンソル、"), bi("m"), tx("：断面の要約量、"), it("s"), tx("：成形条件、"), it("θ"), tx("：モデルのパラメータ。精度の基準は物理モデルと有限要素解析に置き、機械学習はその加速層とする。")], 12.5);
s.addTable([hdr(["段階", "予測・設計の対象", "本発表での扱い"]),
  ...[["①", "残留ひずみ", "既報の穿孔法データで、物理モデルの再現と補正を検討（4・5 節）"],
    ["②", "変形（反り）・強度", "成形プロセス解析で反りまで計算できることを確認（Abaqus）"],
    ["③", "成形条件の設計", "例示定数でベイズ最適化を試作（本発表では扱わない）"]].map((r) => r.map((c, i) => cell(c, { align: i === 0 ? "center" : "left" })))],
  tableOpts(0.7, 3.75, 11.9, [0.9, 3.0, 8.0], 0.5));
para(s, 0.7, 5.95, 11.9, 0.8, [tx("本発表は段階①の前段として、(a) 既報と同じ入力で解析を再現できるか、(b) 少数の実測点で解析を補正できるか、の 2 点を示す。")], 13);

// ===================================================== 3.1 手法 ==============
s = pres.addSlide({ masterName: "CONTENT" });
s.addText("3. 手法 (1) — 配向平均と積層理論", { placeholder: "title" });
para(s, 0.7, 1.2, 11.9, 0.7, [tx("繊維 1 本まわりの剛性と熱膨張は Mori–Tanaka 法[2]（Eshelby テンソルは回転楕円体[3]）と Levin の関係[4]で求め、配向テンソル "), it("a"), si("ij"), tx(" で配向平均する[5]。")], 13);
eq(s, 0.7, 1.95, 11.9, [lb(), bi("α"), rb(), ro(" = "), lb(), bi("C"), rb(), sup("−1"), ro(" : "), lb(), bi("C"), ro(" : "), bi("α"), rb()], "(2)", 17);
para(s, 0.7, 2.55, 11.9, 0.4, [tx("層ごとの固有ひずみ（無応力温度 "), it("T"), sr("0"), tx(" から室温まで積分）と、自由な板のつり合いは")], 13);
eq(s, 0.7, 3.0, 11.9, [bi("ε"), sup("*"), ro("("), it("z"), ro(") = ∫"), sr("  "), lb(), bi("α"), rb(), ro("("), it("T"), ro(", "), it("z"), ro(") d"), it("T"), ro("   ("), it("T"), sr("0"), ro(" → 20 °C)")], "(3)", 17);
eq(s, 0.7, 3.5, 11.9, [ro("["), bi("A"), ro(" "), bi("B"), ro("; "), bi("B"), ro(" "), bi("D"), ro("] ["), bi("ε"), sr("0"), ro("; "), bi("κ"), ro("] = ["), bi("N"), sup("*"), ro("; "), bi("M"), sup("*"), ro("]")], "(4)", 17);
eq(s, 0.7, 4.0, 11.9, [bi("σ"), ro("("), it("z"), ro(") = "), bi("Q"), ro("("), it("z"), ro(") ("), bi("ε"), sr("0"), ro(" + "), it("z"), ro(" "), bi("κ"), ro(" − "), bi("ε"), sup("*"), ro("("), it("z"), ro("))")], "(5)", 17);
rule(s, 4.7);
para(s, 0.7, 4.85, 11.9, 1.9, [
  br("入力は既報[1]の値に揃えた：PPS の物性（既報 表1）、GF の定数、配向の板厚方向分布（既報 図8a、読み取り値）、T₀ = 180 °C。"),
  br("既報は閉包近似を明記していないので、混合近似を用いた（仮定）。"),
  tx("式(4)(5)は固体有限要素（Abaqus 2024）と、ひずみで 10⁻³ 未満の差で一致することを確認済みである。"),
], 12.5);
s.addNotes("validation/denso_tsuda2026_compare.py。research/GUIDE_fifa_abaqus.md 第 I 部。");

// ===================================================== 3.2 穿孔法 ============
s = pres.addSlide({ masterName: "CONTENT" });
s.addText("3. 手法 (2) — 穿孔法の模擬と補正", { placeholder: "title" });
para(s, 0.7, 1.2, 6.0, 2.3, [
  br("1/4 対称の 3 次元モデル（C3D8I、板厚方向 154 層 × 20 µm、約 13 万要素）に式(3)の固有ひずみを与え、冷却後、φ1.6 mm の穴を 0.2 mm ずつ 10 段階で除去した（要素除去）。"),
  br("ゲージは既報と同じ FRAS-2（グリッド 2.0 × 1.1 mm、r = 2.5〜4.5 mm）。Ch.3 を MD、Ch.1 を TD、Ch.2 を 45° に置き、グリッド内の表層要素で平均した。"),
  tx("解放ひずみは固有ひずみについて線形なので、成分ごとの応答 "), bi("h"), si("c"), tx(" の和に厳密に分解できる。"),
], 12.5);
eq(s, 0.7, 3.55, 6.0, [bi("y"), ro(" = "), it("a"), sr("xx"), ro(" "), bi("h"), sr("xx"), ro(" + "), it("a"), sr("yy"), ro(" "), bi("h"), sr("yy"), ro(" + "), it("a"), sr("zz"), ro(" "), bi("h"), sr("zz")], "(6)", 16);
para(s, 6.9, 1.2, 5.7, 1.2, [tx("補正量 "), bi("a"), tx(" は、実測 "), bi("y"), sr("m"), tx(" への当てはめで、事前値（解析のまま "), bi("a"), ro(" = "), bi("1"), tx("）からの離れに罰則を課して決める（L2-SP[6]）。")], 12.5);
eq(s, 6.9, 2.45, 5.7, [ro("min"), sr("  "), it("‖"), bi("H a"), ro(" − "), bi("y"), sr("m"), it("‖"), sup("2"), ro("/"), it("s"), si("y"), sup("2"), ro(" + "), it("‖"), bi("a"), ro(" − "), bi("1"), it("‖"), sup("2"), ro("/"), it("s"), si("a"), sup("2")], "(7)", 16);
para(s, 6.9, 3.05, 5.7, 1.4, [it("s"), si("y"), tx(" = 10 µε（図の読み取りの不確かさ）、"), it("s"), si("a"), tx(" = 0.5（各成分 ±50 % の事前幅）。共通の倍率 "), it("a"), tx(" とすると、"), it("T"), sr("0"), tx(" を 1 つ補正すること "), it("T"), sr("0,eff"), ro(" = 20 + "), it("a"), ro("(180 − 20)"), tx(" に相当する。")], 12.5);
rule(s, 5.1);
para(s, 0.7, 5.25, 11.9, 1.5, [
  br("検証：成分ごとの 3 本の計算の和は、全成分の計算と相対 2.5×10⁻⁷ で一致した（Abaqus、本番メッシュ）。"),
  tx("補正の評価は、当てはめに使わなかった実測点の誤差（RMSE）で行う。"),
], 12.5);
s.addNotes("abaqus/denso_plate/gen_hole_drilling.py、validation/denso_tsuda2026_transfer.py。");

// ===================================================== 4.1 再現 表面応力 =====
s = pres.addSlide({ masterName: "CONTENT" });
s.addText("4. 既報の再現 (1) — 熱膨張と表面の残留応力", { placeholder: "title" });
img(s, F("results/denso_tsuda2026/compare.png"), 0.7, 1.15, 11.9, 3.043);
cap(s, 0.7, 5.1, 11.9, 0.5, [tx("図1　(a) 1 mm 板の面内熱膨張係数、(b) 3.08 mm 板の配向（既報 図8a の読み取り値）、(c) 自由な板の残留応力の板厚方向分布（実線を採用）。既報の値は図からの読み取り値。")]);
s.addTable([hdr(["", "本計算", "既報[1]"]),
  ...[["表面応力 σxx [MPa]", "−24.4", "−24.3（解析）"], ["表面応力 σyy [MPa]", "+19.6", "+15.2（解析）"],
    ["1 mm 板の弾性率 MD / TD [GPa]", "9.6 / 6.6", "9.1 / 6.8（実測）"]].map((r) => r.map((c, i) => cell(c, { align: i === 0 ? "left" : "center", fontSize: 11.5 })))],
  tableOpts(0.7, 5.65, 7.2, [3.4, 1.6, 2.2], 0.3));
para(s, 8.1, 5.65, 4.5, 1.2, [tx("σxx は一致した。σyy は約 30 % 大きい。高温での MD の熱膨張係数が実測より小さく、その原因は特定できていない。")], 11.5);

// ===================================================== 4.2 再現 穿孔法 ======
s = pres.addSlide({ masterName: "CONTENT" });
s.addText("4. 既報の再現 (2) — 穿孔法の解放ひずみ", { placeholder: "title" });
img(s, F("abaqus/denso_plate/abaqus_results/denso_hole_released.png"), 0.7, 1.15, 5.6, 1.312);
cap(s, 0.7, 5.5, 5.6, 0.9, [tx("図2　解放ひずみと穿孔深さ。実測と既報の解析は既報 図12 の読み取り値、本計算は Abaqus 2024（本番メッシュ）。")]);
s.addTable([hdr(["", "Ch.3（MD）", "Ch.1（TD）"]),
  ...[["実測（読み取り）", "195", "170"], ["既報の解析", "272（1.39 倍）", "225（1.32 倍）"],
    ["本計算（要素中心で選択）", "299（1.54 倍）", "261（1.54 倍）"], ["本計算（重なり面積で平均）", "322（1.65 倍）", "287（1.69 倍）"]].map((r) => r.map((c, i) => cell(c, { align: i === 0 ? "left" : "center", fontSize: 11.5 })))],
  tableOpts(6.6, 1.2, 6.0, [2.6, 1.7, 1.7], 0.4));
cap(s, 6.6, 3.25, 6.0, 0.5, [tx("表1　解放ひずみのピーク [µε]。括弧内は実測に対する比。")]);
head(s, 6.6, 3.85, 6.0, "示せたこと");
para(s, 6.6, 4.2, 6.0, 1.0, [tx("符号、ピークの深さ（MD 0.8 mm、TD 1.0 mm）、波形は実測と一致した。大きさは既報の解析と同じ向きに大きい。")], 12.5);
head(s, 6.6, 5.25, 6.0, "示せていないこと");
para(s, 6.6, 5.6, 6.0, 1.2, [tx("大きさの差（1.5〜1.7 倍）の原因は特定していない（6 節）。メッシュを細かくしてもピークは約 3 % しか変わらなかった。")], 12.5);

// ===================================================== 5. 補正 =============
s = pres.addSlide({ masterName: "CONTENT" });
s.addText("5. 少数の実測点による補正", { placeholder: "title" });
img(s, F("results/denso_tsuda2026/transfer_hole_drilling.png"), 0.7, 1.15, 7.4, 2.517);
cap(s, 0.7, 4.15, 7.4, 0.5, [tx("図3　(a) 浅い側で補正し深い側を予測、(b) 当てはめに使わなかった点の誤差。Abaqus 本番メッシュ。")]);
s.addTable([hdr(["検証", "補正前", "補正後", "既報の解析"]),
  ...[["H1：深さ ≤ 1.0 mm で補正 → 1.2〜2.0 mm を予測", "44", "22", "25"],
    ["H2：MD のみで 3 成分を補正 → TD・45° を予測", "46", "173", "29"],
    ["H3：MD のみで共通の倍率を補正 → TD・45° を予測", "46", "11", "29"]].map((r) => r.map((c, i) => cell(c, { align: i === 0 ? "left" : "center", fontSize: 11.5, bold: i === 0 && r[0].startsWith("H3") })))],
  tableOpts(0.7, 4.7, 7.4, [4.4, 1.0, 1.0, 1.0], 0.38));
cap(s, 0.7, 6.3, 7.4, 0.4, [tx("表2　当てはめに使わなかった点の RMSE [µε]。")]);
head(s, 8.4, 1.2, 4.2, "示せたこと");
para(s, 8.4, 1.55, 4.2, 2.5, [tx("浅い側の実測で補正すると、深い側の誤差は半分になり、既報の解析と同程度になった。共通の倍率 0.67 は "), it("T"), sr("0,eff"), tx(" ≈ 142〜143 °C に相当する（温度依存の熱膨張係数で解放ひずみから同定。熱膨張係数一定の換算では 127 °C、既報の推定は約 120 °C）。MD のみの実測でも、共通の倍率を仮定すれば TD を予測できた。")], 12.5);
head(s, 8.4, 4.1, 4.2, "示せていないこと");
para(s, 8.4, 4.45, 4.2, 2.4, [tx("MD のみで成分ごとに補正すると、TD の補正量が決まらず失敗した（H2）。共通の倍率の仮定が他の部品・条件で成り立つかは未確認である。試験片 1 枚、読み取り値による結果である。")], 12.5);

// ===================================================== 6. 差の要因 ==========
s = pres.addSlide({ masterName: "CONTENT" });
s.addText("6. 解析と実測の差の要因の検討", { placeholder: "title" });
img(s, F("slides/figures_paper/kk_gauge_shift.png"), 0.7, 1.15, 5.3, 1.41);
cap(s, 0.7, 4.95, 5.6, 0.6, [tx("図4　ゲージを径方向にずらしたときの解放ひずみのピーク／実測。要素とグリッドの重なり面積で平均（Abaqus 本番メッシュ）。")]);
img(s, F("results/denso_tsuda2026/anneal_relaxation.png"), 6.8, 1.15, 5.3, 1.397);
cap(s, 6.8, 4.95, 5.8, 0.6, [tx("図5　アニール後の冷却中の応力緩和による倍率（Maxwell 要素 1 個、WLF。"), it("T"), sr("g"), tx(" = 85 °C[7]、WLF 定数は汎用値[8]、冷却速度は仮定）。")]);
para(s, 0.7, 5.6, 5.6, 1.3, [tx("ゲージ位置：外側へ 0.1 mm ずれるごとに比は約 0.11〜0.14 下がる。0.4 mm ずれても 1.1〜1.2 倍であり、位置のずれだけでは差を説明できない。")], 12);
para(s, 6.8, 5.6, 5.8, 1.3, [tx("応力緩和：倍率 0.67 には "), it("τ"), sr("g"), tx(" ≈ 10⁹〜10¹⁰ s が要る。非晶性高分子で通常想定する値では緩和しすぎる（0.41〜0.43）。寄与の大きさは PPS の緩和データなしには決められない。")], 12);

// ===================================================== 7. まとめ ===========
s = pres.addSlide({ masterName: "CONTENT" });
s.addText("7. まとめと今後", { placeholder: "title" });
head(s, 0.7, 1.25, 11.9, "示せたこと");
s.addText([
  { text: "既報と同じ入力で、表面の残留応力（σxx −24.4 対 −24.3 MPa）と、穿孔法の解放ひずみの傾向を再現した。", options: { bullet: true, breakLine: true } },
  { text: "解析は実測より約 1.5〜1.7 倍大きく、既報の解析（1.3〜1.4 倍）と同じ向きである。ゲージ位置のずれだけでは説明できない。", options: { bullet: true, breakLine: true } },
  { text: "共通の倍率による補正で、当てはめに使わない実測点の誤差が半分以下になった。倍率は T₀,eff ≈ 142〜143 °C に相当する（熱膨張係数一定の換算では 127 °C、既報の推定は約 120 °C）。", options: { bullet: true } },
], { x: 0.7, y: 1.6, w: 11.9, h: 1.6, fontSize: 13, color: INK, paraSpaceAfter: 4, valign: "top" });
head(s, 0.7, 3.35, 11.9, "示せていないこと");
s.addText([
  { text: "実測は既報の図からの読み取り値で、試験片 1 枚・成形条件 1 つである。デンソー様の実データでは検証していない。", options: { bullet: true, breakLine: true } },
  { text: "補正は 1〜3 個のパラメータであり、多数の条件で学習するニューラルネットワークの転移学習はまだ実データで試していない。", options: { bullet: true, breakLine: true } },
  { text: "差の原因（応力緩和、材料定数、閉包近似など）は特定していない。PPS の結晶化と粘弾性の定数が未入手である。", options: { bullet: true } },
], { x: 0.7, y: 3.7, w: 11.9, h: 1.6, fontSize: 13, color: INK, paraSpaceAfter: 4, valign: "top" });
head(s, 0.7, 5.4, 11.9, "今後");
s.addText([
  { text: "デンソー様の数値解析・計測データで、事前学習と補正を行う（MD と、一部の条件で TD の計測をお願いしたい）。", options: { bullet: true, breakLine: true } },
  { text: "DSC・DMA のデータを得て結晶化と応力緩和を入れ、段階②（反り）に進む。", options: { bullet: true } },
], { x: 0.7, y: 5.75, w: 11.9, h: 1.0, fontSize: 13, color: INK, paraSpaceAfter: 4, valign: "top" });

// ===================================================== 参考文献 ==============
s = pres.addSlide({ masterName: "CONTENT" });
s.addText("参考文献", { placeholder: "title" });
const refs = [
  "[1] 津田ほか, 日本複合材料学会誌, 52(3), 88–96, 2026.",
  "[2] T. Mori and K. Tanaka, “Average stress in matrix and average elastic energy of materials with misfitting inclusions,” Acta Metallurgica, 21(5), 571–574, 1973.",
  "[3] G. P. Tandon and G. J. Weng, “The effect of aspect ratio of inclusions on the elastic properties of unidirectionally aligned composites,” Polymer Composites, 5(4), 327–333, 1984.",
  "[4] V. M. Levin, “On the coefficients of thermal expansion of heterogeneous materials,” Mechanics of Solids, 2, 58–61, 1967.",
  "[5] S. G. Advani and C. L. Tucker III, “The use of tensors to describe and predict fiber orientation in short fiber composites,” Journal of Rheology, 31(8), 751–784, 1987.",
  "[6] X. Li, Y. Grandvalet and F. Davoine, “Explicit inductive bias for transfer learning with convolutional networks,” Proc. 35th International Conference on Machine Learning (ICML), 2018.",
  "[7] L. B. Nohara et al., “Study of crystallization behavior of poly(phenylene sulfide),” Polímeros, 16(2), 2006.",
  "[8] M. L. Williams, R. F. Landel and J. D. Ferry, “The temperature dependence of relaxation mechanisms in amorphous polymers and other glass-forming liquids,” Journal of the American Chemical Society, 77(14), 3701–3707, 1955.",
];
s.addText(refs.map((r, i) => ({ text: r, options: { breakLine: i < refs.length - 1 } })),
  { x: 0.7, y: 1.3, w: 11.9, h: 4.7, fontSize: 11.5, color: INK, lineSpacing: 17, paraSpaceAfter: 6, valign: "top" });
rule(s, 6.15);
s.addText("注：書誌情報は一次文献で未照合の箇所を含む（特に [1] の著者名の全員、[4]）。発表前に確認すること。",
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
