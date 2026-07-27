// build_cfrtp_deck.js -- simple, academic 10-slide deck on the Daikin/NEDO CFRTP work.
// Run:  node build_cfrtp_deck.js   (needs pptxgenjs; writes ../cfrtp_daikin.pptx)
const path = require("path");
const PptxGenJS = require("pptxgenjs");
const p = new PptxGenJS();
p.defineLayout({ name: "W", width: 13.333, height: 7.5 });
p.layout = "W";

const REPO = path.resolve(__dirname, "..", "..");   // repo root (figures live here)
const NAVY = "1F3864", INK = "222222", GRAY = "5A5A5A", RULE = "C9C9C9", ACC = "2E5A88";
const F = "Calibri";
const fig = (n) => path.join(REPO, n);

// ---- header/footer helpers (restrained academic style) ----
function head(s, title, kicker) {
  s.background = { color: "FFFFFF" };
  if (kicker) s.addText(kicker, { x: 0.6, y: 0.34, w: 12, h: 0.3, fontFace: F, fontSize: 12, color: ACC, charSpacing: 1 });
  s.addText(title, { x: 0.6, y: 0.6, w: 12.1, h: 0.7, fontFace: F, fontSize: 24, bold: true, color: NAVY });
  s.addShape(p.ShapeType.line, { x: 0.6, y: 1.38, w: 12.13, h: 0, line: { color: RULE, width: 1 } });
}
function foot(s, n) {
  s.addShape(p.ShapeType.line, { x: 0.6, y: 7.02, w: 12.13, h: 0, line: { color: RULE, width: 0.75 } });
  s.addText("CFRTP (fluoropolymer/carbon) — Daikin/NEDO joint research", { x: 0.6, y: 7.05, w: 9, h: 0.3, fontFace: F, fontSize: 9, color: GRAY });
  s.addText(String(n), { x: 12.2, y: 7.05, w: 0.5, h: 0.3, fontFace: F, fontSize: 9, color: GRAY, align: "right" });
}
function bullets(s, items, o = {}) {
  s.addText(items.map(t => ({ text: t.t !== undefined ? t.t : t, options: { bullet: t.b === false ? false : { code: "2013" }, indentLevel: t.lvl || 0, bold: !!t.bold, color: t.c || INK, fontSize: t.fs || 14, paraSpaceAfter: 6 } })),
    Object.assign({ x: 0.6, y: 1.7, w: 6.1, h: 4.9, fontFace: F, valign: "top", lineSpacingMultiple: 1.05 }, o));
}
function figure(s, name, o = {}) {
  s.addImage(Object.assign({ path: fig(name), x: 7.0, y: 1.6, w: 5.9, h: 4.35, sizing: { type: "contain", w: 5.9, h: 4.35 } }, o));
}
function caption(s, txt, o = {}) {
  s.addText(txt, Object.assign({ x: 7.0, y: 6.0, w: 5.9, h: 0.5, fontFace: F, fontSize: 10, italic: true, color: GRAY, align: "center" }, o));
}

// ===== 1. Title =====
{
  const s = p.addSlide(); s.background = { color: "FFFFFF" };
  s.addShape(p.ShapeType.rect, { x: 0, y: 0, w: 13.333, h: 0.28, fill: { color: NAVY } });
  s.addText("フッ素樹脂／炭素繊維 熱可塑複合材（CFRTP）の計算力学", { x: 0.8, y: 2.2, w: 11.7, h: 1.0, fontFace: F, fontSize: 30, bold: true, color: NAVY });
  s.addText("残留応力・界面接着・剥離の物理ベース有限要素解析と，従属的な機械学習", { x: 0.8, y: 3.25, w: 11.7, h: 0.6, fontFace: F, fontSize: 17, color: INK });
  s.addText("Computational mechanics of CFRTP: cure/solidification residual stress, cohesive-zone interlaminar delamination, and impregnation — with ML as a subordinate accelerator", { x: 0.8, y: 3.95, w: 11.7, h: 0.8, fontFace: F, fontSize: 12.5, italic: true, color: GRAY });
  s.addShape(p.ShapeType.line, { x: 0.8, y: 4.95, w: 6.5, h: 0, line: { color: RULE, width: 1 } });
  s.addText("ダイキン工業 × NEDO 共同研究（種検討）／慶應義塾大学 村松研究室", { x: 0.8, y: 5.1, w: 11.7, h: 0.4, fontFace: F, fontSize: 13, color: INK });
  s.addText("物理（弱形式FE）を精度の権威に，ML は設計加速に限定", { x: 0.8, y: 5.55, w: 11.7, h: 0.4, fontFace: F, fontSize: 12, color: ACC });
}

// ===== 2. Background & objective =====
{
  const s = p.addSlide(); head(s, "背景と目的", "1. Introduction");
  bullets(s, [
    { t: "材料：ダイキンの CFRTP はフッ素樹脂マトリクスの炭素繊維強化“熱可塑”複合材（航空機構造材料）。", bold: true },
    { t: "熱硬化エポキシではない → 残留応力の構成則は 溶融→結晶化→冷却（cure でなく）。", lvl: 1 },
    { t: "特長：耐薬品性・難燃・耐熱・撥水撥油・摺動性。炭素繊維“開繊”技術で含浸性を確保。", lvl: 1 },
    { t: "製造上の3課題（本検討のターゲット）：", bold: true },
    { t: "① 界面接着（低表面エネルギー）→ 層間せん断強度 ILSS が律速", lvl: 1 },
    { t: "② 高温成形 → 冷却ΔT・結晶化収縮・CTEミスマッチ → 残留応力・反り", lvl: 1 },
    { t: "③ 高粘度溶融の含浸 → ボイド／半結晶ゆえ冷却速度依存", lvl: 1 },
    { t: "目的：3課題を物理ベース FE で定量化し，ML を従属の設計加速層として付す。", bold: true, c: ACC },
  ]);
  s.addText([
    { text: "本研究の姿勢\n", options: { bold: true, fontSize: 14, color: NAVY } },
    { text: "・精度の権威：弱形式 FE ／ 実測\n", options: { fontSize: 12.5, color: INK } },
    { text: "・ML の役割：プロセス→特性のサロゲート，逆設計，高速スクリーニング\n", options: { fontSize: 12.5, color: INK } },
    { text: "・出口：Abaqus 本番実装（UMAT / cohesive）への橋渡し", options: { fontSize: 12.5, color: INK } },
  ], { x: 7.0, y: 1.9, w: 5.9, h: 3.4, fontFace: F, valign: "top", lineSpacingMultiple: 1.15, fill: { color: "F4F6FA" }, line: { color: RULE, width: 1 }, margin: 10 });
  foot(s, 2);
}

// ===== 3. Residual stress (thermoplastic) =====
{
  const s = p.addSlide(); head(s, "課題① 残留応力：溶融→結晶化→冷却の熱粘弾性", "2. Residual stress");
  bullets(s, [
    { t: "構成則：固化/結晶化度 α の発展（cure kinetics 類似）", },
    { t: "CHILE：固化に伴い剛性 g(α) が発達 → ゲル/固化後に応力ロック", lvl: 1 },
    { t: "固有ひずみ：熱 α_CTE·ΔT ＋ 結晶化収縮 β·Δα（増分）", lvl: 1 },
    { t: "熱粘弾性緩和 τ(T)：高温で速く緩和，冷却で凍結", },
    { t: "弾性のみは過大評価 → 緩和で実測級へ", lvl: 1 },
    { t: "結果：弾性 500 → 粘弾性 131 MPa（74% 緩和，凍結 ~120℃）", bold: true, c: ACC },
    { t: "速い冷却ほど残留応力が増大（緩和時間不足；実験的に既知の CFRTP 挙動）", lvl: 1 },
    { t: "検証：単一プライ自由収縮 → 残留応力 ≈ 0", },
  ]);
  figure(s, "cfrtp_viscoelastic_residual_stress.png");
  caption(s, "図 応力緩和が弾性の過大評価を実測級へ引き下げ，冷却速度感度が発現");
  foot(s, 3);
}

// ===== 4. Interface adhesion (ILSS) + mixed mode =====
{
  const s = p.addSlide(); head(s, "課題② 界面接着：凝集域と混合モード（B–K）", "3. Interface / delamination");
  bullets(s, [
    { t: "界面を cohesive-zone 界面要素で表現（双線形 traction–separation）", },
    { t: "弱い界面（フッ素樹脂，低表面エネルギー）", },
    { t: "ILSS 13.8 MPa vs 表面処理 38.1 MPa（2.8 倍）", lvl: 1, bold: true, c: ACC },
    { t: "せん断ラグ応力集中ゆえ ILSS < τmax（過程帯）", lvl: 1 },
    { t: "残留応力 × 弱界面の連成：残留せん断が容量を食う", },
    { t: "見かけ ILSS 13.8 → 3.9 MPa", lvl: 1 },
    { t: "混合モード：Camanho–Davila ＋ Benzeggagh–Kenane", },
    { t: "消散エネルギー = Gc(B) を機械精度で検証（エネルギー整合）", lvl: 1 },
  ]);
  figure(s, "cfrtp_ilss_interface.png");
  caption(s, "図 弱い界面 vs 表面処理の ILSS，および残留せん断による余裕の低下");
  foot(s, 4);
}

// ===== 5. Delamination front propagation (2D FE) =====
{
  const s = p.addSlide(); head(s, "混合モード剥離の前縁進展（2D 有限要素）", "3. Interface / delamination");
  bullets(s, [
    { t: "二層梁 ＋ 界面 cohesive（B–K）を弱形式 FE で解く", },
    { t: "初期き裂 ＋ 先端に角度 θ の変位（開口＋せん断＝混合モード）", lvl: 1 },
    { t: "secant（損傷陽解法）反復，不可逆損傷で前縁が進展", lvl: 1 },
    { t: "結果（θ = 25°）：", bold: true },
    { t: "剥離前縁 a0 = 5 → 19.8 mm 進展", lvl: 1, c: ACC },
    { t: "荷重–変位：上昇 → 伝播ピーク（10.3 kN/m）→ 軟化", lvl: 1 },
    { t: "界面損傷場と cohesive 過程帯を可視化", lvl: 1 },
    { t: "Python 種を Abaqus 組み込み cohesive へ移行可能（後述）", },
  ]);
  figure(s, "cfrtp_delamination_2d_fe.png");
  caption(s, "図 変形二層・界面損傷（前縁），荷重–変位，前縁位置，過程帯");
  foot(s, 5);
}

// ===== 6. Impregnation & voids =====
{
  const s = p.addSlide(); head(s, "課題③ 含浸とボイド：Darcy 流と開繊", "4. Impregnation / voids");
  bullets(s, [
    { t: "透過率：Gebart 横流れ K(Vf)", },
    { t: "1D Darcy 含浸：t_imp = μ(1−Vf) h² / (2 K ΔP)", },
    { t: "高粘度フッ素樹脂溶融 → 含浸が遅くボイド", lvl: 1 },
    { t: "開繊（fibre spreading）：h → h/s ⇒ t_imp → t_imp/s²", bold: true },
    { t: "未開繊 tow (100 µm)：t_imp 55 s > 40 s → ボイド 15.9%", lvl: 1 },
    { t: "開繊 ×2 → 0.8%（μ=6000 では 41% → <1%）", lvl: 1, c: ACC },
    { t: "検証：Darcy 前縁が t_imp で中心到達（機械精度）", },
    { t: "→ ダイキンの含浸課題と“開繊”解決策に直結", },
  ]);
  figure(s, "cfrtp_impregnation_void.png");
  caption(s, "図 含浸前縁，含浸時間，ボイド工程窓，開繊による激減");
  foot(s, 6);
}

// ===== 7. Calibration & identifiability =====
{
  const s = p.addSlide(); head(s, "モデル校正と識別性（ツイン実験）", "5. Calibration");
  bullets(s, [
    { t: "実測データ差し替えを前提とした校正フレームワーク", },
    { t: "現状はツイン実験：既知 θ → 疑似実測（＋ノイズ）→ 逆推定で θ 回復", lvl: 1 },
    { t: "識別性：残留応力のみでは freeze-in TREF と結晶化 X_inf が縮退（尾根）", bold: true },
    { t: "2 観測量（残留応力 ＋ 結晶化度）で尾根を解消", lvl: 1, c: ACC },
    { t: "回復：TREF 118.6（真 118），X_inf 0.486（真 0.50）", lvl: 1 },
    { t: "実測（穴あけ / XRD / 曲率）を差し替えれば本番校正", },
    { t: "限界：粗グリッド最小二乗，要ベイズ/ブートストラップ不確かさ", },
  ]);
  figure(s, "cfrtp_calibration.png");
  caption(s, "図 2 観測量で単一最小の誤差曲面 → パラメータを識別");
  foot(s, 7);
}

// ===== 8. Surrogate & inverse design =====
{
  const s = p.addSlide(); head(s, "サロゲートと逆設計（ML は従属）", "6. Surrogate / design");
  bullets(s, [
    { t: "FE データでプロセス → 残留応力の小型 MLP を学習", },
    { t: "入力：冷却速度・溶融温度・結晶化傾向 X_inf", lvl: 1 },
    { t: "未知プロセスへ 1 発推論：残留応力 rel-L2 0.001", lvl: 1, c: ACC },
    { t: "応答曲面（冷却速度 × 溶融温度 → 残留応力）＋ FE 検証点", },
    { t: "逆設計：低溶融温度＋適切な冷却で残留応力を低減", lvl: 1 },
    { t: "位置づけ：ML は探索を加速，精度の権威は FE（置換しない）", bold: true },
    { t: "同枠：半導体側の自己発熱サロゲート／逆設計と同一パイプライン", },
  ]);
  figure(s, "cfrtp_process_surrogate.png");
  caption(s, "図 予測 vs FE（対角一致），応答曲面と逆設計の読み取り");
  foot(s, 8);
}

// ===== 9. Abaqus implementation =====
{
  const s = p.addSlide(); head(s, "本番実装：Abaqus（UMAT / cohesive）", "7. Implementation");
  bullets(s, [
    { t: "cfrtp_cure_umat.f：直交異方性 CHILE UMAT", bold: true },
    { t: "硬化/結晶化カイネティクス（α を STATEV）＋収縮＋熱固有ひずみ，増分応力更新", lvl: 1 },
    { t: "cfrtp_cure_residual.inp：3D [0/90] 残留応力", bold: true },
    { t: "硬化サイクル温度場駆動，105 節点 / 48 C3D8，3-2-1 拘束", lvl: 1 },
    { t: "cfrtp_delamination_mixedmode.inp：2D 二層 ＋ 組み込み cohesive＋B–K", bold: true },
    { t: "366 節点 / 240 CPE4 ＋ 45 COH2D4，初期き裂＋混合モード変位", lvl: 1 },
    { t: "gen_inp.py（再生成）／ run_all.sh ＋ postprocess.py（投入＋odb後処理）", },
    { t: "検証手順：1 要素自由収縮 → 残留応力 ≈ 0（未実行・未コンパイル，各自サーバーで検証）", c: GRAY },
  ], { w: 12.1 });
  s.addText("Python 種（物理）  →  Abaqus 本番（UMAT/cohesive）  →  ML サロゲートは両ソルバの上に被覆", { x: 0.6, y: 6.35, w: 12.1, h: 0.4, fontFace: F, fontSize: 12.5, italic: true, color: ACC, align: "center" });
  foot(s, 9);
}

// ===== 10. Summary & outlook =====
{
  const s = p.addSlide(); head(s, "まとめと今後", "8. Summary");
  bullets(s, [
    { t: "残留応力：熱可塑（溶融→結晶化→冷却）＋CHILE＋粘弾性で実測級（500→131 MPa）", },
    { t: "界面：cohesive＋B–K で ILSS と混合モード剥離前縁を定量（自由収縮・エネルギー整合を検証）", },
    { t: "含浸：Darcy＋Gebart で開繊効果（t_imp∝1/s²，ボイド 16→<1%）", },
    { t: "校正・サロゲート・逆設計：物理主役・ML 従属で一貫", },
    { t: "Abaqus 実装（UMAT/.inp）で本番へ橋渡し", },
    { t: "今後：", bold: true },
    { t: "3D 混合モード剥離前縁，Prony 粘弾性・工具拘束，開繊/含浸→力学特性連成", lvl: 1 },
    { t: "実測データでの本番校正，逆設計（低残留応力・アンチスティクション成形）", lvl: 1 },
  ], { w: 12.1 });
  s.addText("結言：フッ素樹脂 CFRTP の作りにくさ（界面・残留応力・含浸）を，物理 FE で正しく捉え，ML で速く設計に回す。", { x: 0.6, y: 6.3, w: 12.1, h: 0.5, fontFace: F, fontSize: 13, bold: true, color: NAVY, align: "center" });
  foot(s, 10);
}

p.writeFile({ fileName: path.resolve(__dirname, "..", "cfrtp_daikin.pptx") }).then(f => console.log("wrote", f));
