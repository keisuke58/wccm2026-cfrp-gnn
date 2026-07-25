// WCCM deck: weak-form FEM operator learning + online adaptation for semiconductor sim
const pptxgen = require("pptxgenjs");
const path = require("path");

const REPO = "/home/user/wccm2026-cfrp-gnn";
const OUT = path.join(REPO, "slides", "weakform_operator_learning_semiconductor.pptx");

// ---- palette (semiconductor: deep navy / teal / amber) ----
const NAVY = "0B1F3A";      // dark bg
const NAVY2 = "13355E";     // panel on dark
const INK = "1A2B45";       // dark text on light
const BLUE = "0E5A8A";      // primary
const TEAL = "1C9AA8";      // secondary
const AMBER = "E9A13B";     // accent
const RED = "C0413B";       // "cold/strong-form" contrast
const MUTED = "6B7A8F";
const PANEL = "EEF3F8";     // light panel tint
const ICE = "CADCFC";
const WHITE = "FFFFFF";

const HEAD = "Cambria";
const BODY = "Calibri";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";           // 13.3 x 7.5
pres.defineSlideMaster({ title: "L", background: { color: WHITE } });
const W = 13.33, H = 7.5;
const FIG = (n) => path.join(REPO, n);

// helpers -------------------------------------------------------------
function pageNum(s, n) {
  s.addText(String(n), { x: W - 0.7, y: H - 0.45, w: 0.5, h: 0.3,
    fontFace: BODY, fontSize: 10, color: MUTED, align: "right" });
}
function kicker(s, txt, color) {
  s.addText(txt.toUpperCase(), { x: 0.7, y: 0.5, w: 11, h: 0.3, fontFace: BODY,
    fontSize: 13, color: color || TEAL, bold: true, charSpacing: 2, margin: 0 });
}
function title(s, txt, color) {
  s.addText(txt, { x: 0.7, y: 0.8, w: 11.9, h: 0.9, fontFace: HEAD,
    fontSize: 30, color: color || INK, bold: true, margin: 0 });
}
function numBadge(s, x, y, n, color) {
  s.addShape(pres.ShapeType.ellipse, { x, y, w: 0.5, h: 0.5, fill: { color: color || TEAL } });
  s.addText(String(n), { x, y, w: 0.5, h: 0.5, align: "center", valign: "middle",
    fontFace: HEAD, fontSize: 18, bold: true, color: WHITE, margin: 0 });
}
function figBox(s, file, x, y, w, h) {
  // white card behind figure + contained image
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.06,
    fill: { color: WHITE }, line: { color: "D6E0EA", width: 1 },
    shadow: { type: "outer", color: "9AA9B8", blur: 6, offset: 2, angle: 90, opacity: 0.35 } });
  s.addImage({ path: file, x: x + 0.12, y: y + 0.12, w: w - 0.24, h: h - 0.24,
    sizing: { type: "contain", w: w - 0.24, h: h - 0.24 } });
}
function bullets(s, items, x, y, w, h, fs) {
  s.addText(items.map((t, i) => ({ text: t,
      options: { bullet: { code: "2022", indent: 14 }, color: INK, breakLine: true,
        paraSpaceAfter: 8, fontSize: fs || 15 } })),
    { x, y, w, h, fontFace: BODY, valign: "top", margin: 0 });
}
function statCallout(s, x, y, w, big, label, color) {
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h: 1.15, rectRadius: 0.08,
    fill: { color: PANEL }, line: { type: "none" } });
  s.addText(big, { x: x + 0.1, y: y + 0.08, w: w - 0.2, h: 0.6, align: "center",
    fontFace: HEAD, fontSize: 26, bold: true, color: color || BLUE, margin: 0 });
  s.addText(label, { x: x + 0.1, y: y + 0.66, w: w - 0.2, h: 0.42, align: "center",
    fontFace: BODY, fontSize: 11, color: MUTED, margin: 0 });
}

// ============================================================ 1. TITLE
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  // subtle motif: concentric ring (cylindrical GAA nod), no stripes
  s.addShape(pres.ShapeType.ellipse, { x: 9.7, y: -1.6, w: 5.2, h: 5.2,
    fill: { type: "solid", color: NAVY2 }, line: { color: TEAL, width: 1 } });
  s.addShape(pres.ShapeType.ellipse, { x: 10.55, y: -0.75, w: 3.5, h: 3.5,
    fill: { color: NAVY }, line: { color: AMBER, width: 1 } });
  s.addText("WCCM–ECCOMAS 2026  ·  CONCEPT STUDY", { x: 0.9, y: 1.5, w: 10, h: 0.4,
    fontFace: BODY, fontSize: 15, color: TEAL, bold: true, charSpacing: 2, margin: 0 });
  s.addText("Weak-form FEM Operator Learning\n+ Online Adaptation", { x: 0.9, y: 2.05, w: 11.3, h: 1.9,
    fontFace: HEAD, fontSize: 44, bold: true, color: WHITE, lineSpacingMultiple: 1.0, margin: 0 });
  s.addText("for semiconductor device simulation (GAA / CFET / TSV)", { x: 0.9, y: 3.95, w: 11, h: 0.5,
    fontFace: BODY, fontSize: 20, color: ICE, margin: 0 });
  s.addText([
    { text: "FEM as the authority on accuracy · networks & continuation as accelerators", options: { color: ICE, fontSize: 14, breakLine: true } },
    { text: "12 self-contained, seed-fixed concept demos", options: { color: AMBER, fontSize: 14, bold: true } },
  ], { x: 0.9, y: 4.75, w: 11, h: 0.8, fontFace: BODY, margin: 0 });
  s.addText("〈氏名・所属〉   ·   github: keisuke58/wccm2026-cfrp-gnn", { x: 0.9, y: 6.7, w: 11.5, h: 0.4,
    fontFace: BODY, fontSize: 12, color: MUTED, margin: 0 });
}

// ============================================================ 2. MOTIVATION
{
  const s = pres.addSlide();
  kicker(s, "Motivation");
  title(s, "The Poisson solve is the TCAD bottleneck");
  bullets(s, [
    "Device TCAD self-consistently couples Poisson ↔ drift-diffusion (Gummel / Newton).",
    "The innermost, most-repeated step is the Poisson (nonlinear Poisson) solve.",
    "On GAA/CFET meshes this is re-solved thousands of times per bias & per design.",
    "Goal: accelerate it with machine learning — without giving up FE accuracy.",
  ], 0.7, 1.9, 6.2, 3.6, 16);

  // simple self-consistent loop diagram (boxes + arrows), right side
  const bx = 7.5, by = 2.1, bw = 4.7, bh = 0.95;
  const box = (yy, txt, col) => {
    s.addShape(pres.ShapeType.roundRect, { x: bx, y: yy, w: bw, h: bh, rectRadius: 0.08,
      fill: { color: col }, line: { type: "none" } });
    s.addText(txt, { x: bx, y: yy, w: bw, h: bh, align: "center", valign: "middle",
      fontFace: BODY, fontSize: 14, bold: true, color: WHITE, margin: 0 });
  };
  box(by, "Poisson  −∇·(ε∇φ) = ρ(φ)", BLUE);
  box(by + 1.5, "Drift-diffusion  (n, p currents)", TEAL);
  s.addText("bottleneck", { x: bx + bw + 0.05, y: by + 0.2, w: 1.1, h: 0.5, fontFace: BODY,
    fontSize: 12, italic: true, color: AMBER, bold: true, margin: 0 });
  // arrows
  s.addShape(pres.ShapeType.line, { x: bx + 1.2, y: by + bh, y2: by + 1.5, x2: bx + 1.2,
    line: { color: MUTED, width: 2, endArrowType: "triangle" } });
  s.addShape(pres.ShapeType.line, { x: bx + bw - 1.2, y: by + 1.5, y2: by + bh, x2: bx + bw - 1.2,
    line: { color: MUTED, width: 2, endArrowType: "triangle" } });
  s.addText("self-consistent loop  (repeat to convergence)", { x: bx, y: by + 2.7, w: bw, h: 0.4,
    align: "center", fontFace: BODY, fontSize: 12, color: MUTED, italic: true, margin: 0 });
  pageNum(s, 2);
}

// ============================================================ 3. LANDSCAPE
{
  const s = pres.addSlide();
  kicker(s, "Where the field is");
  title(s, "ML for device simulation — and the open lane");
  const items = [
    ["Compact / SPICE models", "mature"],
    ["CNN grid-Poisson surrogates", "strong-form"],
    ["PINN coupled systems", "strong-form"],
    ["GNN / diffusion on meshes", "strong-form"],
    ["DeepONet / FNO operators", "operator"],
    ["Solver convergence acceleration", "warm-start"],
    ["MC transport (Otsuki & Mori)", "MC + FD"],
    ["Inverse design / online learning", "frontier"],
  ];
  const x0 = 0.7, y0 = 1.9, cw = 5.9, ch = 0.68, gap = 0.14;
  items.forEach((it, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = x0 + col * (cw + 0.3), y = y0 + row * (ch + gap);
    s.addShape(pres.ShapeType.roundRect, { x, y, w: cw, h: ch, rectRadius: 0.06,
      fill: { color: PANEL }, line: { type: "none" } });
    numBadge(s, x + 0.08, y + 0.06, i + 1, i >= 1 && i <= 3 ? RED : TEAL);
    s.addText(it[0], { x: x + 0.7, y, w: cw - 2.15, h: ch, valign: "middle", fontFace: BODY,
      fontSize: 13.5, bold: true, color: INK, margin: 0 });
    s.addText(it[1], { x: x + cw - 1.65, y, w: 1.55, h: ch, valign: "middle", align: "right",
      fontFace: BODY, fontSize: 11, italic: true, color: MUTED, margin: 0 });
  });
  s.addText("Surrogates ②–④ are overwhelmingly strong-form / grid-based → the weak-form (FEM) operator lane is open.",
    { x: 0.7, y: 6.5, w: 11.9, h: 0.5, fontFace: BODY, fontSize: 15, bold: true, color: BLUE, margin: 0 });
  pageNum(s, 3);
}

// ============================================================ 4. IDEA
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  kicker(s, "The idea", AMBER);
  title(s, "Put the weak form (FEM) at the core", WHITE);
  bullets(s, [], 0, 0, 0.1, 0.1); // noop
  s.addText([
    { text: "Strong form (finite differences / PINN) enforces the PDE pointwise — and ", options: { color: ICE, fontSize: 17, breakLine: false } },
    { text: "breaks flux continuity", options: { color: AMBER, fontSize: 17, bold: true, breakLine: false } },
    { text: " [ε ∂φ/∂n] = 0 at material interfaces (variable ε).", options: { color: ICE, fontSize: 17, breakLine: true } },
  ], { x: 0.7, y: 2.0, w: 11.9, h: 1.0, fontFace: BODY, valign: "top", margin: 0 });
  s.addText([
    { text: "Weak / Galerkin form ", options: { color: WHITE, fontSize: 17, bold: true, breakLine: false } },
    { text: "builds interface flux-continuity into the assembled residual — naturally robust to variable ε and unstructured meshes.", options: { color: ICE, fontSize: 17, breakLine: true } },
  ], { x: 0.7, y: 3.1, w: 11.9, h: 1.0, fontFace: BODY, valign: "top", margin: 0 });
  // residual chips
  s.addShape(pres.ShapeType.roundRect, { x: 0.7, y: 4.4, w: 5.6, h: 1.7, rectRadius: 0.1, fill: { color: NAVY2 }, line: { color: RED, width: 1 } });
  s.addText("strong form (pointwise)", { x: 0.9, y: 4.55, w: 5.2, h: 0.4, fontFace: BODY, fontSize: 13, color: RED, bold: true, margin: 0 });
  s.addText("residual = Δφ + ρ/ε   at nodes", { x: 0.9, y: 5.0, w: 5.2, h: 0.5, fontFace: BODY, fontSize: 16, color: WHITE, margin: 0 });
  s.addText("interface flux continuity: not guaranteed", { x: 0.9, y: 5.55, w: 5.2, h: 0.4, fontFace: BODY, fontSize: 12, italic: true, color: ICE, margin: 0 });

  s.addShape(pres.ShapeType.roundRect, { x: 6.9, y: 4.4, w: 5.7, h: 1.7, rectRadius: 0.1, fill: { color: NAVY2 }, line: { color: TEAL, width: 1 } });
  s.addText("weak form (Galerkin FE)", { x: 7.1, y: 4.55, w: 5.3, h: 0.4, fontFace: BODY, fontSize: 13, color: TEAL, bold: true, margin: 0 });
  s.addText("residual = K(ε)φ + κ·M·sinh(φ) − Mρ", { x: 7.1, y: 5.0, w: 5.3, h: 0.5, fontFace: BODY, fontSize: 16, color: WHITE, margin: 0 });
  s.addText("interface flux continuity: built in", { x: 7.1, y: 5.55, w: 5.3, h: 0.4, fontFace: BODY, fontSize: 12, italic: true, color: ICE, margin: 0 });
  pageNum(s, 4);
}

// ============================================================ 5. PROPOSAL A+D+B+C
{
  const s = pres.addSlide();
  kicker(s, "Proposal");
  title(s, "Four components: A · D · B · C");
  const cards = [
    ["A", "Weak-form loss / data", "Physics loss = assembled FE residual K(ε)φ−Mρ (Galerkin), not a pointwise Laplacian.", BLUE],
    ["D", "FE a-posteriori error", "One residual estimator unifies mesh refinement and the online-learning trigger.", TEAL],
    ["B", "Operator learning branch", "GNN / DeepONet branch ingests mesh & conditions (mesh-agnostic, repo-native).", BLUE],
    ["C", "Net = accelerator", "FEM stays the accuracy authority; the net only warm-starts Newton / continuation.", TEAL],
  ];
  const x0 = 0.7, y0 = 1.95, cw = 5.9, chh = 2.0, gx = 0.3, gy = 0.3;
  cards.forEach((c, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = x0 + col * (cw + gx), y = y0 + row * (chh + gy);
    s.addShape(pres.ShapeType.roundRect, { x, y, w: cw, h: chh, rectRadius: 0.08,
      fill: { color: PANEL }, line: { type: "none" },
      shadow: { type: "outer", color: "AEBECC", blur: 5, offset: 2, angle: 90, opacity: 0.3 } });
    s.addShape(pres.ShapeType.ellipse, { x: x + 0.3, y: y + 0.35, w: 0.85, h: 0.85, fill: { color: c[3] } });
    s.addText(c[0], { x: x + 0.3, y: y + 0.35, w: 0.85, h: 0.85, align: "center", valign: "middle",
      fontFace: HEAD, fontSize: 30, bold: true, color: WHITE, margin: 0 });
    s.addText(c[1], { x: x + 1.4, y: y + 0.3, w: cw - 1.6, h: 0.6, fontFace: HEAD, fontSize: 18,
      bold: true, color: INK, margin: 0, valign: "middle" });
    s.addText(c[2], { x: x + 1.4, y: y + 0.95, w: cw - 1.65, h: 0.95, fontFace: BODY, fontSize: 13.5,
      color: INK, margin: 0, valign: "top" });
  });
  pageNum(s, 5);
}

// ============================================================ 6. DIFFERENTIATION
{
  const s = pres.addSlide();
  kicker(s, "Differentiation");
  title(s, "Method-level, not application-level");
  // two columns
  const colY = 2.0, colH = 3.6, cwid = 5.7;
  const mkCol = (x, head, headColor, rows) => {
    s.addShape(pres.ShapeType.roundRect, { x, y: colY, w: cwid, h: colH, rectRadius: 0.08,
      fill: { color: PANEL }, line: { type: "none" } });
    s.addText(head, { x: x + 0.25, y: colY + 0.2, w: cwid - 0.5, h: 0.55, fontFace: HEAD,
      fontSize: 18, bold: true, color: headColor, margin: 0 });
    s.addText(rows.map((r, i) => ({ text: r, options: { bullet: { code: "2022", indent: 12 },
      color: INK, fontSize: 14, breakLine: true, paraSpaceAfter: 8 } })),
      { x: x + 0.25, y: colY + 0.85, w: cwid - 0.5, h: colH - 1.05, fontFace: BODY, valign: "top", margin: 0 });
  };
  mkCol(0.7, "Otsuki & Mori (baseline)", RED, [
    "3D Monte-Carlo transport for GAA",
    "Poisson inside the MC loop replaced by a PI-DeepONet",
    "Discretization core: strong-form finite differences",
    "Online learning accelerates the MC run",
  ]);
  mkCol(6.9, "This work", TEAL, [
    "Weak-form (Galerkin FE) at the operator core",
    "Interface-robust residual K(ε)φ−Mρ as the physics loss",
    "FE a-posteriori estimator drives refinement + online trigger",
    "FEM keeps accuracy; net / continuation only accelerate",
  ]);
  s.addText("Same receptive frame (\"ML accelerates TCAD\"); the discretization core differs — demo ② quantifies it.",
    { x: 0.7, y: 5.8, w: 11.9, h: 0.5, fontFace: BODY, fontSize: 15, bold: true, color: BLUE, margin: 0 });
  pageNum(s, 6);
}

// ============================================================ demo slide helper
function demoSlide(opts) {
  const s = pres.addSlide();
  kicker(s, opts.kicker);
  title(s, opts.title);
  // figure on the right (or full), bullets/stats left
  const fx = opts.figX ?? 6.35, fy = 1.9, fw = opts.figW ?? 6.3, fh = opts.figH ?? 4.0;
  figBox(s, opts.fig, fx, fy, fw, fh);
  if (opts.caption) s.addText(opts.caption, { x: fx, y: fy + fh + 0.05, w: fw, h: 0.35,
    align: "center", fontFace: BODY, fontSize: 10, italic: true, color: MUTED, margin: 0 });
  if (opts.badges) {
    let bx = 0.7;
    opts.badges.forEach((b) => { numBadge(s, bx, 1.95, b, TEAL); bx += 0.62; });
  }
  bullets(s, opts.bullets, 0.7, opts.badges ? 2.65 : 2.0, 5.4, 2.6, 14.5);
  if (opts.stats) {
    let sy = 5.35;
    const sw = (5.4 - (opts.stats.length - 1) * 0.25) / opts.stats.length;
    opts.stats.forEach((st, i) => statCallout(s, 0.7 + i * (sw + 0.25), sy, sw, st[0], st[1], st[2] || BLUE));
  }
  pageNum(s, opts.page);
  return s;
}

// ============================================================ 7. demo ②  (marquee)
demoSlide({
  kicker: "Demo ② — the one-panel novelty",
  title: "Weak form vs strong form (no NN involved)",
  fig: FIG("bench_weak_vs_strong.png"), figX: 6.6, figW: 6.1, figH: 4.4,
  badges: [2],
  bullets: [
    "Pure discretization test at a material interface (variable ε).",
    "Weak form converges at O(h²); strong form stalls at an interface-error floor.",
    "This is the quantitative backbone for \"why weak form\".",
  ],
  stats: [["O(h²)", "weak: 0.13 → 0.0093", TEAL], ["~0.55", "strong: floor", RED]],
  caption: "bench_weak_vs_strong.py — FE vs FD, refinement + interface study",
  page: 7,
});

// ============================================================ 8. demo ①
demoSlide({
  kicker: "Demo ① — operator learning + online adaptation",
  title: "Weak-form PI-DeepONet with an FE error trigger",
  fig: FIG("pi_deeponet_fem_gaa.png"), figX: 6.6, figW: 6.1, figH: 4.4,
  badges: [1],
  bullets: [
    "GNN-branch DeepONet trained on the Galerkin residual (idea A+B).",
    "FE a-posteriori estimator triggers an exact solve only when needed (idea D).",
    "Replay adaptation lowers the trigger rate as deployment proceeds.",
  ],
  stats: [["24 / 60", "triggered exact solves", BLUE], ["57%→23%", "trigger rate", TEAL]],
  caption: "pi_deeponet_fem_gaa.py — rel-L2 0.074 at unchanged FE accuracy",
  page: 8,
});

// ============================================================ 9. demos ③④⑤
demoSlide({
  kicker: "Demos ③ ④ ⑤ — idea C, Newton warm-start",
  title: "Learned warm-start: 1D → 2D GAA → CFET stack",
  fig: FIG("cfet_stack_warmstart.png"), figX: 6.6, figW: 6.1, figH: 4.4,
  badges: [3, 4, 5],
  bullets: [
    "A learned initial guess seeds damped Newton; every solve stays exact FE.",
    "1D nonlinear Poisson → 2D GAA slice → multi-material CFET cross-section.",
    "FEM guarantees accuracy; the net only cuts iterations.",
  ],
  stats: [["~40%", "fewer Newton iters", TEAL], ["4.8→3.0", "CFET cold→warm", BLUE]],
  caption: "cfet_stack_warmstart.py (n/p stack) — cf. fe_/dd2d_ warm-start demos",
  page: 9,
});

// ============================================================ 10. demos ⑧⑨
demoSlide({
  kicker: "Demos ⑧ ⑨ — full drift-diffusion",
  title: "Coupled DD: ideal diode & continuation at high bias",
  fig: FIG("dd_breakdown_continuation.png"), figX: 6.6, figW: 6.1, figH: 4.4,
  badges: [8, 9],
  bullets: [
    "1D pn junction, Scharfetter–Gummel + Gummel (⑧): ideal-diode I–V validated.",
    "High reverse bias (⑨): cold start diverges at 4 V_t; continuation reaches 45 V_t.",
    "Here idea C is not \"faster\" — it decides convergence vs divergence.",
  ],
  stats: [["4 V_t → 45 V_t", "cold diverges → continues", AMBER], ["J∝e^{V/Vt}−1", "diode validated", BLUE]],
  caption: "dd_breakdown_continuation.py — impact ionization kept mild (~×2.1)",
  page: 10,
});

// ============================================================ 11. demos ⑥⑦
demoSlide({
  kicker: "Demos ⑥ ⑦ — TSV thermo-mechanical stress",
  title: "3D-stack packaging: FE thermoelasticity + KOZ",
  fig: FIG("tsv_3d_stress.png"), figX: 6.6, figW: 6.1, figH: 4.4,
  badges: [6, 7],
  bullets: [
    "2D (⑥): CNN stress-field surrogate + keep-out-zone from FE thermoelasticity.",
    "3D (⑦): real elastic constants — anisotropic Si (C11/C12/C44) vs isotropic Cu.",
    "Anisotropy gives 4-fold (cloverleaf) stress → direction-dependent KOZ.",
  ],
  stats: [["IoU 0.819", "⑥ keep-out zone", TEAL], ["2 analytic PASS", "⑦ machine precision", BLUE]],
  caption: "tsv_3d_stress.py — bridges to the repo's CFRP stress × GNN core",
  page: 11,
});

// ============================================================ 12. demos ⑩⑪
demoSlide({
  kicker: "Demos ⑩ ⑪ — real multi-material TCAD workload",
  title: "Material sweep amortized, then made a learned operator",
  fig: FIG("gaa_operator_deeponet.png"), figX: 6.6, figW: 6.1, figH: 4.4,
  badges: [10, 11],
  bullets: [
    "Mirrors Balaji+2026 (Sentaurus): Si/Ge/GeSn/InGaAs/MoS₂ × bias, cylindrical GAA.",
    "⑩ continuation amortizes 45 self-consistent FE solves (217→161 Newton iters).",
    "⑪ DeepONet: 1-shot on an unseen material, then warm-starts FE (A+B+C unified).",
  ],
  stats: [["rel-L2 0.015", "⑪ unseen material", BLUE], ["5.2→2.8", "⑪ warm-start iters", TEAL]],
  caption: "gaa_operator_deeponet.py — normalization uses train materials only (no leak)",
  page: 12,
});

// ============================================================ 13. demo ⑫ (WFM Vth)
demoSlide({
  kicker: "Demo ⑫ — WFM tunes the threshold voltage",
  title: "Work-function metal → Vth on the GAA cross-section",
  fig: FIG("gaa_wfm_vth.png"), figX: 6.6, figW: 6.1, figH: 4.4,
  badges: [12],
  bullets: [
    "Maps the pilot-line WFM step (Applied Endura-3 class) to a computed threshold.",
    "5 WFM (Φ_m 4.2–5.0 eV): FE Q–V_g solves on the disk → extract Vth per metal.",
    "Recovers the ideal flat-band law ΔVth = ΔΦ_WFM on the real cross-section.",
  ],
  stats: [["1.00 V/eV", "ΔVth / ΔΦ_WFM (slope)", TEAL], ["0.04→0.84 V", "Vth design window", BLUE]],
  caption: "gaa_wfm_vth.py — ideal flat-band model (no traps/poly-depletion), illustrative scale",
  page: 13,
});

// ============================================================ 14. MANUFACTURING <-> SIM
{
  const s = pres.addSlide();
  kicker(s, "From fab to simulation");
  title(s, "Pilot-line process ↔ what we simulate");
  s.addText("AIST advanced-semiconductor pilot line (GAA nanosheet)   →   this work's demos",
    { x: 0.7, y: 1.55, w: 11.9, h: 0.3, fontFace: BODY, fontSize: 12, italic: true, color: MUTED, margin: 0 });
  const rows = [
    ["Si nanosheet channel — SiGe release (Certas)", "GAA nonlinear Poisson  ①③④⑩⑪⑫"],
    ["High-k + WFM gate (Eagle-XP4 / Endura-3)", "variable-ε interface · WFM→Vth  ②⑫"],
    ["SiGe → Ge / GeSn channels (EpiPrime)", "multi-material sweep  ⑩⑪ (Balaji+2026)"],
    ["CFET / 3D stack / TSV (next node)", "CFET slice ⑤ · TSV thermo-mech ⑥⑦"],
    ["material × bias TCAD sweep (Sentaurus)", "continuation ⑩ · learned operator ⑪"],
  ];
  const y0 = 2.05, rh = 0.86, gap = 0.14, lw = 5.1, rw = 5.6, rxc = 6.3;
  rows.forEach((r, i) => {
    const y = y0 + i * (rh + gap);
    s.addShape(pres.ShapeType.roundRect, { x: 0.7, y, w: lw, h: rh, rectRadius: 0.07,
      fill: { color: PANEL }, line: { type: "none" } });
    s.addText(r[0], { x: 0.9, y, w: lw - 0.4, h: rh, valign: "middle", fontFace: BODY,
      fontSize: 13, bold: true, color: INK, margin: 0 });
    s.addShape(pres.ShapeType.line, { x: 0.7 + lw + 0.05, y: y + rh / 2, x2: rxc - 0.05, y2: y + rh / 2,
      line: { color: TEAL, width: 2.5, endArrowType: "triangle" } });
    s.addShape(pres.ShapeType.roundRect, { x: rxc, y, w: rw, h: rh, rectRadius: 0.07,
      fill: { color: "E7F0F7" }, line: { color: TEAL, width: 1 } });
    s.addText(r[1], { x: rxc + 0.2, y, w: rw - 0.4, h: rh, valign: "middle", fontFace: BODY,
      fontSize: 13, color: BLUE, margin: 0 });
  });
  pageNum(s, 16);
}

// ============================================================ 15. LIMITATIONS
{
  const s = pres.addSlide();
  kicker(s, "Honest limitations", AMBER);
  title(s, "What is — and is not — demonstrated");
  bullets(s, [
    "All 11 are concept demos (CPU, minutes, seed-fixed).",
    "①'s error indicator is the raw relative Galerkin residual; the tolerance is calibrated, not a solution-error bound.",
    "Warm-start on robust monotone solves gives modest gains (~10–40%); the decisive case is ⑨ (converge vs diverge).",
    "1D full DD (⑧⑨) is implemented; coupled 2D/3D semiconductor DD (Full Newton, needs variable scaling) is future work.",
    "\"3D unimplemented\" refers to semiconductor DD only — 3D thermoelasticity (⑦) is implemented.",
    "⑩⑪ screening κ is an illustrative non-dimensional scale (ordering & trends are physical); ⑪'s disk is near radially symmetric.",
  ], 0.7, 1.95, 11.9, 4.6, 15.5);
  pageNum(s, 17);
}

// ============================================================ 14. ROADMAP
{
  const s = pres.addSlide();
  kicker(s, "Roadmap");
  title(s, "Next axes");
  const items = [
    ["True vertical-gate geometry + unstructured mesh", TEAL],
    ["2D-CFET (2D-material channels)", BLUE],
    ["True avalanche (implicit G) & 2D Full-Newton coupled DD (variable scaling)", TEAL],
    ["Learned operator on asymmetric doping (trunk expressiveness) + GNN branch", BLUE],
    ["η-driven h-refinement (AMR) with mesh-agnostic branch", TEAL],
    ["3D stress-field surrogate; KOZ classification in the CFRP-defect GNN pipeline", BLUE],
  ];
  const x0 = 0.7, y0 = 1.95, cw = 11.9, ch = 0.68, gap = 0.14;
  items.forEach((it, i) => {
    const y = y0 + i * (ch + gap);
    s.addShape(pres.ShapeType.roundRect, { x: x0, y, w: cw, h: ch, rectRadius: 0.06,
      fill: { color: PANEL }, line: { type: "none" } });
    s.addShape(pres.ShapeType.ellipse, { x: x0 + 0.2, y: y + 0.19, w: 0.3, h: 0.3, fill: { color: it[1] } });
    s.addText(it[0], { x: x0 + 0.7, y, w: cw - 1.0, h: ch, valign: "middle", fontFace: BODY,
      fontSize: 15, color: INK, margin: 0 });
  });
  pageNum(s, 16);
}

// ============================================================ 15. SUMMARY
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addShape(pres.ShapeType.ellipse, { x: -1.5, y: 4.4, w: 5.0, h: 5.0,
    fill: { color: NAVY2 }, line: { color: TEAL, width: 1 } });
  kicker(s, "Takeaway", AMBER);
  s.addText("Weak form at the core makes the surrogate\ninterface-robust — FEM stays the authority,\nlearning & continuation only accelerate.", { x: 0.9, y: 1.9, w: 11.5, h: 2.4,
    fontFace: HEAD, fontSize: 30, bold: true, color: WHITE, lineSpacingMultiple: 1.05, margin: 0 });
  const chips = [
    ["②", "weak form: O(h²) vs strong-form floor"],
    ["①", "operator learning + FE-triggered online adaptation"],
    ["③–⑤,⑩", "warm-start / continuation: ~40% fewer iters"],
    ["⑨", "continuation decides converge vs diverge"],
    ["⑪", "learned operator: 1-shot on unseen material"],
    ["⑥⑦", "TSV thermo-mechanics → bridges to CFRP × GNN"],
  ];
  const x0 = 0.9, y0 = 4.5, cw = 5.75, ch = 0.72, gx = 0.35, gy = 0.16;
  chips.forEach((c, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = x0 + col * (cw + gx), y = y0 + row * (ch + gy);
    s.addShape(pres.ShapeType.roundRect, { x, y, w: cw, h: ch, rectRadius: 0.08,
      fill: { color: NAVY2 }, line: { type: "none" } });
    s.addShape(pres.ShapeType.ellipse, { x: x + 0.12, y: y + 0.13, w: 0.46, h: 0.46, fill: { color: AMBER } });
    s.addText(c[0], { x: x + 0.12, y: y + 0.13, w: 0.46, h: 0.46, align: "center", valign: "middle",
      fontFace: HEAD, fontSize: 13, bold: true, color: NAVY, margin: 0 });
    s.addText(c[1], { x: x + 0.72, y, w: cw - 0.85, h: ch, valign: "middle", fontFace: BODY,
      fontSize: 12.5, color: ICE, margin: 0 });
  });
  pageNum(s, 17);
}

pres.writeFile({ fileName: OUT }).then((f) => console.log("wrote", f));
