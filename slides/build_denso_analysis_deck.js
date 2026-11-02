// build_denso_analysis_deck.js — preliminary analyses for the DENSO residual-strain theme
// (through-thickness model, Abaqus check, ML rehearsal, process analysis), academic register.
// Typography as slides/build_denso_deck.js (CLAUDE.md): Japanese in Mincho (游明朝), Latin,
// digits and mathematics in Times New Roman; the pptxgenjs theme is patched after writing.
// Every number on the slides comes from a file in the repository (paths in the notes).
//
// Run:  NODE_PATH="$(npm root -g)" PPTX_SKILL=/mnt/skills/public/pptx \
//       node slides/build_denso_analysis_deck.js slides/denso_analysis_prelim.pptx
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const OUT = process.argv[2] || "denso_analysis_prelim.pptx";
const ROOT = path.resolve(__dirname, "..");
const F = (p) => path.join(ROOT, p);
const FIG = {
  plate: F("viz/denso_plate_features.png"),
  fe: F("abaqus/denso_plate/calculix_check/denso_asymmetric_calculix.png"),
  ml: F("results/denso_ml_copaam_gpu/denso_ml.png"),
  proc: F("abaqus/denso_plate/process_results/denso_process_prediction.png"),
};

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
const FOOT = "繊維強化樹脂の残留ひずみ予測（株式会社デンソー様との共同研究）／予備解析・数値は例示";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.theme = { headFontFace: LATIN, bodyFontFace: LATIN };
pres.author = "西岡 佳祐";
pres.title = "繊維強化樹脂の残留ひずみ予測に向けた予備解析";
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
s.addText("繊維強化樹脂の残留ひずみ予測に向けた予備解析", { placeholder: "title" });
s.addText("板厚方向モデル、固体有限要素による照合、機械学習、成形プロセス解析", { placeholder: "body" });
rule(s, 4.4, 4.15, 5.0);
s.addText([br("西岡 佳祐"), br("慶應義塾大学 修士課程"), tx("2026年11月")],
  { x: 0.9, y: 4.6, w: 11.5, h: 1.2, fontSize: 15, color: INK, align: "center",
    isTextBox: true, margin: 0, lineSpacing: 24 });
s.addText("注：材料定数はすべて例示値（未校正）であり、配向分布は仮定である。示す数値は手法の検証であって、材料の予測ではない。",
  { x: 0.9, y: 6.3, w: 11.5, h: 0.4, fontSize: 11, color: MUTED, align: "center",
    isTextBox: true, margin: 0 });

// ===================================================== 0. 構成 ===============
s = pres.addSlide({ masterName: "CONTENT" });
s.addText("本資料の構成", { placeholder: "title" });
para(s, 0.7, 1.25, 11.9, 0.7, [tx("研究開始前に、入力（結晶化度・繊維配向）から出力（残留ひずみ）までを一貫して計算できる系を作り、各段階を独立に検証した。精度の基準は物理モデルと有限要素解析に置き、機械学習はその加速層として位置づける。")], 13.5);
s.addTable([hdr(["節", "解析", "目的", "計算環境", "状態"]),
  ...[
    ["1", "板厚方向モデル", "入力と出力の関係を物理的に定める", "Python", "検証済み"],
    ["2", "固体有限要素による照合", "1 の積層理論を 3 次元要素で確かめる", "Abaqus 2024（fifa）", "全 6 ケース合格"],
    ["3", "機械学習の予備実験", "入力の要件と転移学習の効果を調べる", "PyTorch（copaam, GPU）", "完了"],
    ["4", "成形プロセス解析", "粘弾性緩和を含めて残留ひずみとそりを求める", "Python 参照解／Abaqus", "Abaqus 実行待ち"],
  ].map((r) => r.map((c, i) => cell(c, { align: i < 1 ? "center" : "left", bold: i === 1 })))],
  tableOpts(0.7, 2.15, 11.9, [0.7, 2.7, 4.3, 2.4, 1.8], 0.5));
para(s, 0.7, 4.85, 11.9, 1.6, [
  br("対象は 100 mm × 100 mm × 3 mm の平板とし、板厚方向の分布を主に扱う。"),
  br("材料は短繊維強化 PEEK を想定した例示値である。配向分布はスキン–シェル–コア構造を仮定して与えたもので、流動解析の結果ではない。"),
  tx("金型温度は、上下対称（185 / 185 °C）と非対称（170 / 200 °C）の 2 条件とした。"),
], 13);
s.addNotes("4 つの解析の位置づけ。数値は例示値であることを最初に断る。");

// ===================================================== 1. 板厚方向モデル ======
s = pres.addSlide({ masterName: "CONTENT" });
s.addText("1. 板厚方向モデル — 結晶化・配向平均・積層理論", { placeholder: "title" });
para(s, 0.7, 1.2, 5.9, 0.35, [tx("結晶化は Nakamura 則[2] に Hoffman–Lauritzen の速度定数 "), it("K"), ro("("), it("T"), ro(")"), tx("[3] を与えて解く。")], 12.5);
eq(s, 0.7, 1.55, 5.9, [ro("d"), it("α"), ro("/d"), it("t"), ro(" = "), it("n K"), ro("("), it("T"),
  ro(")(1 − "), it("α"), ro(")[−ln(1 − "), it("α"), ro(")]"), sup("("), supi("n"), sup(" − 1)/"), supi("n")], "(1)", 16);
para(s, 0.7, 2.08, 5.9, 0.35, [tx("熱膨張と結晶化収縮は、配向テンソル "), it("a"), si("ij"), tx(" で配向平均する[1]。")], 12.5);
eq(s, 0.7, 2.43, 5.9, [lb(), bi("α"), rb(), ro(" = "), lb(), bi("C"), rb(), sup("−1"), ro(" : "),
  lb(), bi("C"), ro(" : "), bi("α"), rb()], "(2)", 16);
para(s, 6.8, 1.2, 5.8, 0.35, [tx("層ごとの固有ひずみと、自由な板のつり合いは")], 12.5);
eq(s, 6.8, 1.55, 5.8, [bi("ε"), sup("*"), ro(" = "), lb(), bi("α"), rb(), ro(" Δ"), it("T"), ro(" + "),
  lb(), bi("β"), rb(), ro(" "), it("X"), sr("c")], "(3)", 16);
eq(s, 6.8, 2.0, 5.8, [ro("["), bi("A"), ro(" "), bi("B"), ro("; "), bi("B"), ro(" "), bi("D"), ro("] ["),
  bi("ε"), sr("0"), ro("; "), bi("κ"), ro("] = ["), bi("N"), sup("*"), ro("; "), bi("M"), sup("*"), ro("]")], "(4)", 16);
eq(s, 6.8, 2.45, 5.8, [bi("ε"), sr("res"), ro("("), it("z"), ro(") = "), bi("ε"), sr("0"), ro(" + "),
  it("z"), ro(" "), bi("κ"), ro(" − "), bi("ε"), sup("*"), ro("("), it("z"), ro(")")], "(5)", 16);
img(s, FIG.plate, 1.25, 2.98, 10.8, 3.101);
s.addText([tx("図1　板厚方向の分布（左から配向 "), it("a"), si("ij"), tx("、結晶化度 "), it("X"), sr("c"),
  tx("、配向平均熱膨張係数、残留ひずみ）。定数は例示値、配向は仮定。残留ひずみ "), it("ε"), sr("res"),
  tx(" は弾性ひずみとして定義した。")],
  { x: 0.7, y: 6.52, w: 11.9, h: 0.35, fontSize: 10.5, color: INK, isTextBox: true, margin: 0 });
s.addNotes("viz/denso_plate_features.py。式(2)は直接平均ではなく応力平均である。式(5)の ε0 と κ は断面全体で決まる。");

// ===================================================== 2. 固体有限要素 ========
s = pres.addSlide({ masterName: "CONTENT" });
s.addText("2. 固体有限要素による照合 — Abaqus 2024", { placeholder: "title" });
para(s, 0.7, 1.2, 6.9, 1.6, [
  br("1/4 板（対称面 2 面）を C3D8I 要素（面内 25 × 25、板厚方向 40 層）で解いた。各層に式(2)の配向平均剛性を直交異方性として与え、式(3)の固有ひずみを等価な熱膨張として与えた。"),
  tx("参照解は、同じ区分一定の層について式(4)を厳密に積分したものである。比較量は中心列の弾性ひずみと、中央面の変位から求めたそり量である。"),
], 13);
s.addTable([hdr(["ケース", "ひずみ誤差", "そり量 [mm]（参照）", "判定"]),
  ...[
    ["対称 185 / 185 °C", "5.5×10⁻⁴", "−0.0000（−0.0000）", "合格"],
    ["非対称 170 / 200 °C", "6.9×10⁻⁴", "−1.3619（−1.3618）", "合格"],
    ["熱・結晶化の項のみ（4 ケース）", "≤ 7.4×10⁻⁴", "参照と一致", "合格"],
  ].map((r) => r.map((c, i) => cell(c, { align: i === 0 ? "left" : "center" })))],
  tableOpts(0.7, 3.0, 6.9, [2.9, 1.3, 1.9, 0.8], 0.44));
cap(s, 0.7, 4.85, 6.9, 0.55, [tx("表1　Abaqus 2024（fifa）と積層理論の比較。ひずみ誤差は max|FE − 参照| / max|参照|、そり量は 100 mm あたり。合格基準は 5×10⁻³。")]);
para(s, 0.7, 5.45, 6.9, 1.4, [
  br("示せたこと：学習データの生成に用いる積層理論は、3 次元固体要素とひずみで 10⁻³ 未満、そり量で 10⁻⁴ mm の桁まで一致する。熱の項と結晶化の項の重ね合わせも 1.3×10⁻¹⁰ で成り立つ。", { bold: false }),
  tx("この段階では物理は追加していない（同じ線形弾性模型）。"),
], 12.5);
img(s, FIG.fe, 8.0, 1.2, 4.3, 1.215);
cap(s, 8.0, 4.8, 4.6, 0.9, [tx("図2　非対称ケースの残留ひずみの板厚方向分布。線は積層理論、記号は固体要素（CalculiX 2.21 による事前確認。Abaqus も表1のとおり同値）。定数は例示値。")]);
s.addNotes("research/GUIDE_fifa_abaqus.md 表2。最初の実行では Abaqus が直交異方性材料に局所座標系を要求したため、全体座標と同じ *ORIENTATION を追加した。");

// ===================================================== 3. ML 実験1 ============
s = pres.addSlide({ masterName: "CONTENT" });
s.addText("3. 機械学習の予備実験 (1) — 入力に何が要るか", { placeholder: "title" });
para(s, 0.7, 1.2, 11.9, 0.7, [tx("式(5)において、固有ひずみ "), bi("ε"), sup("*"), tx(" はその点の量で決まるが、"), bi("ε"), sr("0"),
  tx(" と "), bi("κ"), tx(" は断面全体のつり合いで決まる。そこで入力を次の 2 通りとし、物理モデルで生成した 2000 枚の板（各 41 点）で学習した。")], 13);
s.addTable([hdr(["入力", "内容", "RMSE [10⁻³]", "決定係数 R²"]),
  [cell("local"), cell([tx("その点の "), it("z"), ro(", "), it("a"), sr("11"), ro(", "), it("a"), sr("22"), ro(", "), it("a"), sr("33"), ro(", "), it("X"), sr("c"), tx("（仕様どおりの入力）")], { align: "left" }), cell("0.713"), cell("0.914")],
  [cell("context"), cell([tx("上記 ＋ 断面の要約量 6 個（"), it("a"), sr("11"), ro(", "), it("a"), sr("22"), ro(", "), it("X"), sr("c"), tx(" の板厚平均と一次モーメント）")], { align: "left" }), cell("0.054"), cell("0.9995")],
], tableOpts(0.7, 2.05, 11.9, [1.3, 6.8, 1.9, 1.9], 0.5));
cap(s, 0.7, 3.62, 11.9, 0.35, [tx("表2　入力による予測誤差（検証用 20 % で評価、多層パーセプトロン 3 層 × 64）。残留ひずみの標準偏差は 2.43×10⁻³。copaam（GPU）と CPU で小数第 3 位まで一致。")]);
rule(s, 4.15);
head(s, 0.7, 4.3, 11.9, "示せたこと");
para(s, 0.7, 4.68, 11.9, 0.8, [tx("その点の結晶化度と配向だけでは残留ひずみは決まらない。断面の情報を加えると誤差は約 1/13 になった。したがって入力の仕様には「その点」に加えて「その断面」の情報が要る。これはデンソー様と入力の形式を決める際の論点である。")], 13);
head(s, 0.7, 5.6, 11.9, "示せていないこと");
para(s, 0.7, 5.98, 11.9, 0.8, [tx("断面の要約量 6 個が最適な表現であるかは調べていない。実形状では面内方向の非局所性も加わる。")], 13);
s.addNotes("results/denso_ml_copaam_gpu/results.json の exp1。local の RMSE は copaam で 0.712。");

// ===================================================== 3. ML 実験2 ============
s = pres.addSlide({ masterName: "CONTENT" });
s.addText("3. 機械学習の予備実験 (2) — 少数の実測点への転移学習", { placeholder: "title" });
para(s, 0.7, 1.2, 11.9, 0.7, [tx("「実験」の役の target 領域では定数を系統的にずらし（直交方向の熱膨張 +15 %、結晶化収縮 −15 %、"),
  it("X"), sr("∞"), tx(" = 0.27、固定温度 150 °C）、各板で 5 点だけを観測し、計測ノイズを加えた。未知の target 板 200 枚の全点で評価した。")], 13);
const T2 = [["source-only", "0.294", "0.294", "0.294", "0.294", "0.294", "0.294"],
  ["target-only", "1.585", "1.209", "0.765", "0.457", "0.264", "0.164"],
  ["fine-tune", "0.456", "0.271", "0.163", "0.154", "0.108", "0.081"]];
s.addTable([hdr(["測定した板の枚数", "1", "2", "5", "10", "20", "50"]),
  ...T2.map((r) => r.map((c, i) => cell(c, { align: i === 0 ? "left" : "center", bold: r[0] === "fine-tune" })))],
  tableOpts(0.7, 1.95, 6.2, [1.7, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75], 0.4));
cap(s, 0.7, 3.6, 6.2, 0.55, [tx("表3　target での RMSE [10⁻³]（10 回の平均）。source-only は数値解析のみ、target-only は実測のみ、fine-tune は事前学習後の合わせ込み。")]);
head(s, 0.7, 4.3, 6.2, "示せたこと");
para(s, 0.7, 4.66, 6.2, 0.95, [tx("測定が 2 枚以上あれば合わせ込みが最もよい。5 枚で source-only の約 0.55 倍、50 枚で約 0.28 倍になった。")], 12.5);
head(s, 0.7, 5.55, 6.2, "示せていないこと");
para(s, 0.7, 5.91, 6.2, 0.95, [tx("測定 1 枚では合わせ込みが source-only より悪い（0.456 対 0.294）。5 点への過剰適合とみられ、事前学習の重みへの正則化[6] 等は未検討である。")], 12.5);
img(s, FIG.ml, 7.1, 1.95, 5.5, 3.096);
cap(s, 7.1, 3.8, 5.5, 0.75, [tx("図3　(a) 入力の比較、(b) 測定枚数に対する誤差、(c) 未知の板の予測例。合成データ、定数は例示値。")]);
s.addNotes("target は定数を一様にずらしただけで、実際の計算と実測の差より素直である。好結果は実データでの成功を保証しない。");

// ===================================================== 4. プロセス解析 モデル ==
s = pres.addSlide({ masterName: "CONTENT" });
s.addText("4. 成形プロセス解析 (1) — 熱伝導と粘弾性応力の逐次解析", { placeholder: "title" });
para(s, 0.7, 1.2, 11.9, 0.65, [tx("1〜3 の模型は、固定温度 "), it("T"), sr("lock"),
  tx(" で応力が凍結し、結晶化収縮はすべて残る（ロックイン模型、上限）と仮定していた。ここではこの仮定を外し、熱伝導の結果を応力解析に渡す逐次解析とした。")], 13);
s.addTable([hdr(["工程", "時間", "熱の境界条件", "力学の境界条件"]),
  ...[["金型内", "40 s", "上下面を金型温度に固定", "面内変位を拘束（キャビティ）"],
    ["脱型", "0.01 s", "熱伝達 h = 10 W/(m² K) に切替", "拘束を解除"],
    ["空冷", "1800 s", "熱伝達のまま 25 °C へ", "自由な板"]].map((r) => r.map((c, i) => cell(c, { align: i === 0 ? "center" : "left" })))],
  tableOpts(0.7, 1.95, 6.0, [1.0, 0.8, 2.2, 2.0], 0.42));
cap(s, 0.7, 3.7, 6.0, 0.3, [tx("表4　工程と境界条件。初期温度 380 °C、固定増分。")]);
para(s, 6.9, 1.95, 5.7, 0.35, [tx("応力は一般化 Maxwell 模型とし、緩和時間を温度[4] と結晶化度でずらす。")], 12.5);
eq(s, 6.9, 2.32, 5.7, [bi("σ"), ro("("), it("t"), ro(") = ∫ "), it("g"), ro("("), it("ξ"), ro("("), it("t"), ro(") − "), it("ξ"), ro("("), it("s"), ro(")) "), bi("C"), ro(" : ("), ro("d"), bi("ε"), ro(" − d"), bi("ε"), sup("*"), ro(")")], "(6)", 15);
eq(s, 6.9, 2.82, 5.7, [it("g"), ro("("), it("ξ"), ro(") = "), it("g"), sr("∞"), ro(" + Σ "), it("g"), si("k"), ro(" exp(−"), it("ξ"), ro("/"), it("τ"), si("k"), ro("),   d"), it("ξ"), ro(" = d"), it("t"), ro(" / ("), it("a"), si("T"), ro(" "), it("a"), si("X"), ro(")")], "(7)", 15);
eq(s, 6.9, 3.32, 5.7, [ro("log"), sr("10"), ro(" "), it("a"), si("T"), ro(" = −"), it("C"), sr("1"), ro("("), it("T"), ro(" − "), it("T"), sr("ref"), ro(")/("), it("C"), sr("2"), ro(" + "), it("T"), ro(" − "), it("T"), sr("ref"), ro("),   "), it("a"), si("X"), ro(" = 10"), supi("B"), { text: "X", options: { italic: false, superscript: true } }, supi("α")], "(8)", 15);
s.addText([tx("固有ひずみの増分は d"), it("ε"), supi("*"), si("i"), ro(" = "), it("A"), si("i"), ro(" d"), it("T"), ro(" + "), it("B"), si("i"), ro(" d"), it("α"),
  tx("（"), it("i"), tx(" = 1, 2, 3）とした。既存の UMAT（一方向材用、Abaqus 2024 で動作確認済み）から、結晶化収縮を 3 方向別々に与えるように 1 箇所だけ変更した。定数は例示値（"),
  it("g"), sr("∞"), tx(" = 0.01、"), it("τ"), si("k"), tx(" = 5, 50, 500 s、"), it("C"), sr("1"), tx(" = 17.4、"), it("C"), sr("2"), tx(" = 51.6、"), it("T"), sr("ref"), tx(" = 143 °C）である。")],
  { x: 0.7, y: 4.15, w: 11.9, h: 0.95, fontSize: 12.5, color: INK, isTextBox: true, margin: 0, lineSpacing: 19, valign: "top" });
s.addTable([hdr(["確認項目（Abaqus なし）", "結果"]),
  ...[["UMAT（Fortran）と Python 移植の一致（単点、溶融から室温）", "応力 1.8×10⁻¹⁵、結晶化度 1.1×10⁻¹⁶"],
    ["弾性極限（g∞ = 1、結晶化なし）と積層理論の一致", "ひずみ 1.8×10⁻¹⁵、曲率 10 桁一致"],
    ["熱の項と結晶化の項の重ね合わせ", "1.7×10⁻¹⁵"],
    ["熱伝導の入力ファイル（CalculiX、全 3 工程）", "1 s 以降の最大差 0.31 K"]].map((r) => r.map((c, i) => cell(c, { align: "left" })))],
  tableOpts(0.7, 5.1, 11.9, [7.0, 4.9], 0.34));
s.addNotes("abaqus/denso_plate/denso_cryst_umat.f, process_reference.py --check。応力の入力ファイルは標準の CalculiX では解けないため、Abaqus での実行が最初の確認になる。");

// ===================================================== 4. プロセス解析 結果 ===
s = pres.addSlide({ masterName: "CONTENT" });
s.addText("4. 成形プロセス解析 (2) — 結果（Python 参照解、Abaqus 実行前）", { placeholder: "title" });
img(s, FIG.proc, 1.35, 1.12, 10.6, 2.833);
cap(s, 0.7, 4.9, 11.9, 0.35, [tx("図4　(a) 非対称ケースの相対結晶化度、(b)(c) 残留ひずみ "), it("ε"), sr("res,"), si("xx"),
  tx(" の板厚方向分布（実線：粘弾性プロセス模型、破線：ロックイン模型）。定数は例示値。")]);
s.addTable([hdr(["", "残留ひずみの幅 [10⁻³]：粘弾性（ロックイン）", "そり量 [mm]：粘弾性（ロックイン）"]),
  ...[["熱の項", "4.81（4.50）", "−0.19（0.00）"], ["結晶化の項", "0.02（2.23）", "−1.46（−1.36）"],
    ["合計", "4.83（6.66）", "−1.65（−1.36）"]].map((r) => r.map((c, i) => cell(c, { align: i === 0 ? "left" : "center", fontSize: 11.5 })))],
  tableOpts(0.7, 5.32, 8.0, [1.6, 3.2, 3.2], 0.3));
para(s, 8.9, 5.32, 3.7, 1.1, [tx("表5　非対称ケースの寄与の分解。模型が固有ひずみについて線形なので、分解は厳密である。")], 10.5);
s.addNotes("abaqus/denso_plate/process_results/process_summary.json。対称ケースは合計 4.82（6.54）、そり 0。");

// ===================================================== 4. 解釈 ===============
s = pres.addSlide({ masterName: "CONTENT" });
s.addText("4. 成形プロセス解析 (3) — 結晶化はひずみを残さず、そりを残す", { placeholder: "title" });
para(s, 0.7, 1.2, 11.9, 1.25, [
  br("結晶化は大部分が金型内、すなわちガラス転移温度より高く緩和の速い温度域で進む。このため結晶化収縮による応力はほぼ緩和し、残留ひずみへの寄与は 2.23×10⁻³ から 0.02×10⁻³ に下がった（表5）。"),
  tx("一方、そり量は残った。緩和しない平衡項（"), it("g"), sr("∞"), tx("）が金型内で拘束された収縮を記憶し、脱型時に解放するためである。この解釈を平衡項の大きさを変えて確かめた。"),
], 13);
s.addTable([hdr(["平衡項 g∞", "10⁻⁴", "10⁻³", "10⁻²（基準）", "10⁻¹"]),
  ...[["残留ひずみの幅（合計）[10⁻³]", "4.72", "4.73", "4.83", "5.83"],
    ["そり量：結晶化の項 [mm]", "−1.461", "−1.460", "−1.459", "−1.459"],
    ["そり量：熱の項 [mm]", "−0.423", "−0.302", "−0.193", "−0.093"]].map((r) => r.map((c, i) => cell(c, { align: i === 0 ? "left" : "center" })))],
  tableOpts(0.7, 2.6, 11.9, [4.3, 1.9, 1.9, 1.9, 1.9], 0.42));
cap(s, 0.7, 4.32, 11.9, 0.35, [tx("表6　平衡項の大きさに対する感度（非対称ケース、Prony 係数の和を 1 に保って配分）。")]);
head(s, 0.7, 4.8, 11.9, "示せたこと（この模型の範囲で）");
para(s, 0.7, 5.15, 11.9, 0.55, [tx("結晶化に由来するそりは平衡項の大きさに依存しない（10⁻⁴〜10⁻¹ で −1.46 mm）。熱に由来するそりは、ロックイン模型では 0 であるが、粘弾性模型では非対称な冷却履歴により生じる。")], 12.5);
head(s, 0.7, 5.75, 11.9, "示せていないこと");
para(s, 0.7, 6.1, 11.9, 0.7, [tx("平衡項が長時間で流動するか（溶融域、結晶の網目の緩和）はこの模型に含まれず、結晶化由来のそりの有無はそれに依存する。保圧・潜熱・金型変形も含めていない。Abaqus での確認は未了である。")], 12.5);
s.addNotes("process_reference.py --summary の ginf_sweep_asymmetric。結晶化由来のそりは平衡項の値ではなく、平衡項が緩和しないという模型の構造で決まる。");

// ===================================================== 5. まとめ ==============
s = pres.addSlide({ masterName: "CONTENT" });
s.addText("5. まとめと今後", { placeholder: "title" });
head(s, 0.7, 1.25, 11.9, "示せたこと");
s.addText([
  { text: "学習データの生成に用いる積層理論は、Abaqus の 3 次元固体要素と 10⁻³ 未満で一致した（2 節）。", options: { bullet: true, breakLine: true } },
  { text: "残留ひずみの予測には、その点の結晶化度と配向に加えて断面の情報が要る（R² 0.914 → 0.9995、3 節）。", options: { bullet: true, breakLine: true } },
  { text: "合成データでは、測定 2 枚以上で転移学習（合わせ込み）が最もよい（3 節）。", options: { bullet: true, breakLine: true } },
  { text: "粘弾性緩和を入れると、結晶化収縮は残留ひずみをほとんど残さないが、そりは残す（Python 参照解、4 節）。", options: { bullet: true } },
], { x: 0.7, y: 1.6, w: 11.9, h: 1.75, fontSize: 13, color: INK, paraSpaceAfter: 4, valign: "top" });
head(s, 0.7, 3.5, 11.9, "示せていないこと");
s.addText([
  { text: "材料定数・配向分布はすべて例示値・仮定であり、実材料との比較は行っていない。", options: { bullet: true, breakLine: true } },
  { text: "測定 1 枚では転移学習が悪化する。target は定数を一様にずらした合成データにすぎない。", options: { bullet: true, breakLine: true } },
  { text: "プロセス解析の応力は Abaqus で未実行であり、平衡項の長時間挙動に結論が依存する。", options: { bullet: true } },
], { x: 0.7, y: 3.85, w: 11.9, h: 1.3, fontSize: 13, color: INK, paraSpaceAfter: 4, valign: "top" });
head(s, 0.7, 5.3, 11.9, "今後");
s.addText([
  { text: "fifa でプロセス解析を実行し、Python 参照解と照合する。", options: { bullet: true, breakLine: true } },
  { text: "測定が少ない場合の合わせ込みを改善する（事前学習の重みへの正則化[6]、最終層のみの更新）。", options: { bullet: true, breakLine: true } },
  { text: "デンソー様と、入力の形式（断面の情報の受け渡し）、残留ひずみの定義、計測位置を確認する。", options: { bullet: true } },
], { x: 0.7, y: 5.65, w: 11.9, h: 1.2, fontSize: 13, color: INK, paraSpaceAfter: 4, valign: "top" });

// ===================================================== 参考文献 ==============
s = pres.addSlide({ masterName: "CONTENT" });
s.addText("参考文献", { placeholder: "title" });
const refs = [
  "[1] S. G. Advani and C. L. Tucker III, “The use of tensors to describe and predict fiber orientation in short fiber composites,” Journal of Rheology, 31(8), 751–784, 1987.",
  "[2] K. Nakamura, T. Watanabe, K. Katayama and T. Amano, “Some aspects of nonisothermal crystallization of polymers. I. Relationship between crystallization temperature, crystallinity, and cooling conditions,” Journal of Applied Polymer Science, 16(5), 1077–1091, 1972.",
  "[3] J. D. Hoffman and J. I. Lauritzen Jr., “Crystallization of bulk polymers with chain folding: theory of growth of lamellar spherulites,” Journal of Research of the National Bureau of Standards A, 65A(4), 297–336, 1961.",
  "[4] M. L. Williams, R. F. Landel and J. D. Ferry, “The temperature dependence of relaxation mechanisms in amorphous polymers and other glass-forming liquids,” Journal of the American Chemical Society, 77(14), 3701–3707, 1955.",
  "[5] L. M. Parlevliet, H. E. N. Bersee and A. Beukers, “Residual stresses in thermoplastic composites—A study of the literature—Part I: Formation of residual stresses,” Composites Part A, 37(11), 1847–1857, 2006.",
  "[6] X. Li, Y. Grandvalet and F. Davoine, “Explicit inductive bias for transfer learning with convolutional networks,” Proceedings of the 35th International Conference on Machine Learning (ICML), 2018.",
];
s.addText(refs.map((r, i) => ({ text: r, options: { breakLine: i < refs.length - 1 } })),
  { x: 0.7, y: 1.3, w: 11.9, h: 4.6, fontSize: 11.5, color: INK, lineSpacing: 17, paraSpaceAfter: 6, valign: "top" });
rule(s, 6.1);
s.addText("注：書誌情報は一次文献で未照合の箇所を含む。引用前に確認すること。熱可塑性複合材の残留応力の形成過程は [5] に概観がある。",
  { x: 0.7, y: 6.22, w: 11.9, h: 0.5, fontSize: 11, color: MUTED, isTextBox: true, margin: 0 });

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
