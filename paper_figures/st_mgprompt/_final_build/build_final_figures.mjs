import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const OUT_DIR = "D:/PaperProject/GyxPaper2/paper_figures/st_mgprompt";
const BUILD_DIR = path.join(OUT_DIR, "_final_build");
const FINAL_PPTX = path.join(OUT_DIR, "ST-MGPrompt_Final_Figures.pptx");
const W = 1280;
const H = 720;
const FONT = "Times New Roman";
const PT = 96 / 72;

const C = {
  ink: "#17232C",
  muted: "#536774",
  line: "#78909C",
  lightLine: "#B8C7CE",
  panelA: "#F4F8FB",
  panelB: "#F0FAF8",
  panelC: "#F7F3FC",
  panelD: "#FFF8F0",
  fine: "#E69A54",
  fineLight: "#FCEAD8",
  coarse: "#4F9A96",
  coarseLight: "#DDF2EF",
  prompt: "#8367B2",
  promptLight: "#ECE5F7",
  shared: "#6F8795",
  sharedLight: "#E5EDF2",
  rose: "#B76F68",
  roseLight: "#F7E9E6",
  white: "#FFFFFF",
  gray: "#9AA7AE",
  grayLight: "#EEF2F4",
  darkGray: "#5F6A70",
  green: "#4A8B6F",
};

const noLine = { style: "solid", fill: "none", width: 0 };

function textStyle(pt, color = C.ink, bold = false, align = "center", valign = "middle") {
  return {
    fontSize: pt * PT,
    typeface: FONT,
    color,
    bold,
    alignment: align,
    verticalAlignment: valign,
    autoFit: "shrinkText",
    wrap: "square",
    lineSpacing: 0.95,
    insets: { left: 6, right: 6, top: 4, bottom: 4 },
  };
}

function addText(slide, name, text, x, y, w, h, pt = 12, opts = {}) {
  const s = slide.shapes.add({
    geometry: "textbox",
    name,
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: noLine,
  });
  s.text = text;
  s.text.style = textStyle(
    pt,
    opts.color || C.ink,
    opts.bold || false,
    opts.align || "center",
    opts.valign || "middle",
  );
  if (opts.italic) s.text.italic = true;
  return s;
}

function addBox(slide, name, text, x, y, w, h, fill = C.white, stroke = C.lightLine, pt = 12, opts = {}) {
  const geometry = opts.geometry || "roundRect";
  const config = {
    geometry,
    name,
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: { style: opts.dashed ? "dashed" : "solid", fill: stroke, width: opts.lineWidth || 1.25 },
    shadow: opts.shadow || "shadow-none",
  };
  if (["rect", "textbox", "roundRect"].includes(geometry)) config.borderRadius = opts.radius || "rounded-lg";
  const s = slide.shapes.add(config);
  if (text) {
    s.text = text;
    s.text.style = textStyle(pt, opts.color || C.ink, opts.bold || false, opts.align || "center", opts.valign || "middle");
  }
  return s;
}

function addPanel(slide, name, title, x, y, w, h, fill, stroke) {
  addBox(slide, `${name}__BACKGROUND`, "", x, y, w, h, fill, stroke, 12, {
    radius: "rounded-xl",
    lineWidth: 1.35,
    shadow: "2px 3px 8px #23343D/10",
  });
  addText(slide, `${name}__TITLE`, title, x + 4, y + 5, w - 8, 56, 18, { bold: true });
}

function addHLine(slide, name, x1, y, x2, color = C.lightLine, width = 1.2, dashed = false) {
  return slide.shapes.add({
    geometry: "line",
    name,
    position: { left: x1, top: y, width: x2 - x1, height: 0.1 },
    fill: "none",
    line: { style: dashed ? "dashed" : "solid", fill: color, width },
  });
}

function addVLine(slide, name, x, y1, y2, color = C.lightLine, width = 1.2, dashed = false) {
  return slide.shapes.add({
    geometry: "line",
    name,
    position: { left: x, top: y1, width: 0.1, height: y2 - y1 },
    fill: "none",
    line: { style: dashed ? "dashed" : "solid", fill: color, width },
  });
}

function connect(slide, name, from, to, options = {}) {
  const fp = from.position;
  const tp = to.position;
  const point = (p, side) => {
    if (side === "left") return { x: p.left, y: p.top + p.height / 2 };
    if (side === "right") return { x: p.left + p.width, y: p.top + p.height / 2 };
    if (side === "top") return { x: p.left + p.width / 2, y: p.top };
    return { x: p.left + p.width / 2, y: p.top + p.height };
  };
  const a = point(fp, options.fromSide || (fp.left < tp.left ? "right" : "bottom"));
  const b = point(tp, options.toSide || (fp.left < tp.left ? "left" : "top"));
  const color = options.color || C.line;
  const width = options.width || 2.55;
  const lineStyle = options.dashed ? "dashed" : "solid";
  const drawSegment = (segName, p1, p2) => {
    const l = slide.shapes.add({
      geometry: "line",
      name: segName,
      position: {
        left: Math.min(p1.x, p2.x), top: Math.min(p1.y, p2.y),
        width: Math.max(0.1, Math.abs(p2.x - p1.x)), height: Math.max(0.1, Math.abs(p2.y - p1.y)),
        horizontalFlip: p2.x < p1.x,
        verticalFlip: p2.y < p1.y,
      },
      fill: "none",
      line: { style: lineStyle, fill: color, width },
    });
    return l;
  };
  if ((options.kind || "straight").startsWith("elbow")) {
    if (options.fromSide === "right" || options.fromSide === "left") {
      const midX = (a.x + b.x) / 2;
      drawSegment(`${name}__1`, a, { x: midX, y: a.y });
      drawSegment(`${name}__2`, { x: midX, y: a.y }, { x: midX, y: b.y });
      drawSegment(`${name}__3`, { x: midX, y: b.y }, b);
    } else {
      const midY = (a.y + b.y) / 2;
      drawSegment(`${name}__1`, a, { x: a.x, y: midY });
      drawSegment(`${name}__2`, { x: a.x, y: midY }, { x: b.x, y: midY });
      drawSegment(`${name}__3`, { x: b.x, y: midY }, b);
    }
  } else {
    drawSegment(`${name}__LINE`, a, b);
  }
  if (!options.noHead) {
    const angle = Math.atan2(b.y - a.y, b.x - a.x) * 180 / Math.PI;
    slide.shapes.add({
      geometry: "triangle",
      name: `${name}__HEAD`,
      position: { left: b.x - 5, top: b.y - 5, width: 10, height: 10, rotation: angle + 90 },
      fill: color,
      line: { style: "solid", fill: color, width: 0.5 },
    });
  }
  return null;
}

function addCircle(slide, name, text, cx, cy, r, fill, stroke, pt = 12, opts = {}) {
  const circle = addBox(slide, name, text, cx - r, cy - r, r * 2, r * 2, fill, stroke, pt, {
    geometry: "ellipse",
    lineWidth: opts.lineWidth || 1.4,
    bold: opts.bold || false,
    color: opts.color || C.ink,
  });
  if (text) circle.text.insets = { left: 1, right: 1, top: 1, bottom: 1 };
  return circle;
}

function addDiamond(slide, name, text, cx, cy, size, fill, stroke, pt = 12) {
  return addBox(slide, name, text, cx - size / 2, cy - size / 2, size, size, fill, stroke, pt, {
    geometry: "diamond",
    lineWidth: 1.5,
  });
}

function addTokens(slide, prefix, x, y, count, tokenW, tokenH, fill, stroke, labels = null, gap = 4) {
  const out = [];
  for (let i = 0; i < count; i++) {
    out.push(addBox(slide, `${prefix}__TOKEN_${i + 1}`, labels ? labels[i] : "", x + i * (tokenW + gap), y, tokenW, tokenH, fill, stroke, labels ? 10.5 : 10, {
      radius: "rounded-sm",
      lineWidth: 1,
    }));
  }
  return out;
}

function addTensorStack(slide, prefix, x, y, w, h, fill, stroke) {
  for (let i = 2; i >= 0; i--) {
    addBox(slide, `${prefix}__SHEET_${i}`, "", x + i * 6, y - i * 5, w, h, fill, stroke, 10, {
      geometry: "rect",
      lineWidth: 1,
      radius: 4,
    });
  }
  for (let i = 1; i < 4; i++) addVLine(slide, `${prefix}__GRID_V_${i}`, x + (w * i) / 4, y, y + h, stroke, 0.65);
  for (let i = 1; i < 3; i++) addHLine(slide, `${prefix}__GRID_H_${i}`, x, y + (h * i) / 3, x + w, stroke, 0.65);
}

function addWaveform(slide, prefix, x, y, w, h, color, smooth = 0, width = 2.2) {
  const n = 18;
  let prev = null;
  for (let i = 0; i < n; i++) {
    const xx = x + (i / (n - 1)) * w;
    const base = Math.sin(i * (smooth ? 0.5 : 1.6)) + 0.45 * Math.sin(i * (smooth ? 0.19 : 2.7));
    const yy = y + h / 2 - base * h * (smooth ? 0.24 : 0.34);
    if (prev) {
      const left = Math.min(prev.x, xx);
      const top = Math.min(prev.y, yy);
      slide.shapes.add({
        geometry: "line",
        name: `${prefix}__SEG_${i}`,
        position: {
          left,
          top,
          width: Math.abs(xx - prev.x),
          height: Math.max(0.1, Math.abs(yy - prev.y)),
          verticalFlip: yy < prev.y,
        },
        fill: "none",
        line: { style: "solid", fill: color, width },
      });
    }
    prev = { x: xx, y: yy };
  }
}

function addLayerStack(slide, prefix, x, y, count, w, h, fill, stroke, label) {
  for (let i = count - 1; i >= 0; i--) {
    addBox(slide, `${prefix}__LAYER_${i + 1}`, "", x + i * 5, y - i * 4, w, h, fill, stroke, 10, {
      geometry: "rect",
      radius: 3,
      lineWidth: 1,
    });
  }
  if (label) addText(slide, `${prefix}__LABEL`, label, x - 12, y + h + 4, w + 32, 32, 10.5, { color: C.muted });
}

function graphEdges(n, weighted = false) {
  const base = [[0,1],[0,2],[1,2],[1,3],[2,3],[2,4],[3,5],[4,5],[1,5]];
  return base.slice(0, n).map((e, i) => ({ e, width: weighted ? [1, 3.1, 1.8, 4, 2.4, 1.3, 3.6, 2.0, 1.1][i] : 1.55 }));
}

function addGraph(slide, prefix, x, y, w, h, color, weighted = false, mutedUnsupported = false) {
  const pts = [
    [0.08,0.54],[0.32,0.18],[0.43,0.62],[0.67,0.22],[0.76,0.70],[0.94,0.46],
  ].map(([px,py]) => [x + px*w, y + py*h]);
  for (const {e:[a,b], width} of graphEdges(9, weighted)) {
    const [x1,y1] = pts[a]; const [x2,y2] = pts[b];
    slide.shapes.add({
      geometry: "line",
      name: `${prefix}__EDGE_${a}_${b}`,
      position: {
        left: Math.min(x1,x2), top: Math.min(y1,y2), width: Math.abs(x2-x1), height: Math.max(0.1,Math.abs(y2-y1)),
        verticalFlip: y2 < y1,
      },
      fill: "none",
      line: { style: mutedUnsupported && a === 1 && b === 5 ? "dashed" : "solid", fill: mutedUnsupported && a === 1 && b === 5 ? C.gray : color, width },
    });
  }
  for (let i = 0; i < pts.length; i++) addCircle(slide, `${prefix}__NODE_${i + 1}`, "", pts[i][0], pts[i][1], 5.2, C.white, color, 10, { lineWidth: 1.5 });
  return pts;
}

function addSigma(slide, name, cx, cy, color = C.shared) {
  return addCircle(slide, name, "Σ", cx, cy, 14, C.white, color, 14, { lineWidth: 1.5 });
}

function addPlus(slide, name, cx, cy, color = C.shared) {
  return addCircle(slide, name, "+", cx, cy, 11, C.white, color, 12, { lineWidth: 1.4 });
}

function addTitle(slide, text) {
  addText(slide, "FIGURE_TITLE", text, 18, 8, 1244, 42, 28, { bold: true, align: "left" });
}

function figure1(pres) {
  const s = pres.slides.add();
  s.background.fill = C.white;
  addTitle(s, "ST-MGPrompt Overall Framework");
  addPanel(s, "FIG1_PANEL_A", "(A) Feature Input &\nTemporal Decomposition", 18, 58, 280, 646, C.panelA, "#9AB7C6");
  addPanel(s, "FIG1_PANEL_B", "(B) Dual-Semantic Graph-\nTemporal Encoding", 310, 58, 290, 646, C.panelB, "#86B9B4");
  addPanel(s, "FIG1_PANEL_C", "(C) Prompt-Guided Cross-\nGranularity Fusion", 612, 58, 292, 646, C.panelC, "#A893C5");
  addPanel(s, "FIG1_PANEL_D", "(D) Multi-Horizon Forecasting &\nDifficulty-Aware Training", 916, 58, 346, 646, C.panelD, "#D8A777");

  // Panel A — objects first, then connectors.
  addTensorStack(s, "FIG1_PANEL_A__INPUT", 32, 127, 68, 58, C.sharedLight, C.shared);
  const inputLabel = addText(s, "FIG1_PANEL_A__INPUT_LABEL", "Historical\nWind-Farm Input", 20, 185, 100, 31, 12);
  addText(s, "FIG1_PANEL_A__INPUT_META", "134 turbines\n16 variables\n144 steps / 24 h", 20, 216, 100, 42, 10.5, { color: C.muted });
  const proj = addBox(s, "FIG1_PANEL_A__PROJECTION", "16 → 64", 125, 142, 60, 34, C.sharedLight, C.shared, 14, { geometry: "chevron", lineWidth: 1.3 });
  addText(s, "FIG1_PANEL_A__PROJECTION_LABEL", "Shared Input Projection", 108, 183, 94, 42, 14);
  const decomp = addBox(s, "FIG1_PANEL_A__DECOMP", "Fine\n+\nCoarse", 211, 127, 72, 64, C.white, C.shared, 12, { lineWidth: 1.35 });
  addText(s, "FIG1_PANEL_A__DECOMP_LABEL", "Same-Time-Axis Fixed Dual-Granularity Decomposition", 190, 196, 104, 58, 14);
  addHLine(s, "FIG1_PANEL_A__TIMELINE", 35, 280, 276, C.shared, 1.25);
  for (let i = 0; i < 13; i++) addVLine(s, `FIG1_PANEL_A__TICK_${i}`, 35 + i * 20, 276, 284, C.shared, 0.8);
  addText(s, "FIG1_PANEL_A__SAME_TIME", "Same-Time-Axis · 144 timestamps", 50, 260, 220, 20, 10.5, { color: C.shared });

  addText(s, "FIG1_PANEL_A__FINE_LABEL", "Fine", 31, 297, 44, 26, 12, { color: C.fine });
  addWaveform(s, "FIG1_PANEL_A__FINE_WAVE", 77, 300, 185, 39, C.fine, 0, 2.15);
  addText(s, "FIG1_PANEL_A__FINE_METHOD", "Causal Depthwise Separable TCN", 38, 345, 222, 30, 14);
  const finePlus = addPlus(s, "FIG1_PANEL_A__FINE_PLUS", 247, 397, C.fine);
  addText(s, "FIG1_PANEL_A__FINE_RESIDUAL", "Scale-Learned Residual Connection", 38, 378, 190, 39, 14);
  addBox(s, "FIG1_PANEL_A__FINE_OUT", "Fine representation", 58, 423, 183, 30, C.fineLight, C.fine, 12, { lineWidth: 1.1 });

  addText(s, "FIG1_PANEL_A__COARSE_LABEL", "Coarse", 29, 475, 54, 26, 12, { color: C.coarse });
  addWaveform(s, "FIG1_PANEL_A__COARSE_W1", 86, 470, 176, 25, "#6EB1AC", 1, 1.5);
  addWaveform(s, "FIG1_PANEL_A__COARSE_W2", 86, 491, 176, 25, C.coarse, 2, 1.8);
  addWaveform(s, "FIG1_PANEL_A__COARSE_W3", 86, 512, 176, 25, "#2F7C79", 3, 2.0);
  addText(s, "FIG1_PANEL_A__COARSE_METHOD", "Multi-Window Causal Rolling Mean", 36, 542, 224, 31, 14);
  addText(s, "FIG1_PANEL_A__COARSE_WINDOWS", "w = 6 / 18 / 36", 55, 572, 186, 23, 10.5, { color: C.muted });
  addPlus(s, "FIG1_PANEL_A__COARSE_PLUS", 247, 616, C.coarse);
  addText(s, "FIG1_PANEL_A__COARSE_RESIDUAL", "Scale-Learned Residual Connection", 38, 597, 190, 39, 14);
  addBox(s, "FIG1_PANEL_A__COARSE_OUT", "Coarse representation", 58, 645, 183, 30, C.coarseLight, C.coarse, 12, { lineWidth: 1.1 });
  addText(s, "FIG1_PANEL_A__TENSOR_SHAPE", "B × 144 × 134 × 16", 66, 677, 165, 19, 10, { color: C.muted });
  connect(s, "FIG1_PANEL_A__FLOW_1", inputLabel, proj, { fromSide: "right", toSide: "left", color: C.shared });
  connect(s, "FIG1_PANEL_A__FLOW_2", proj, decomp, { fromSide: "right", toSide: "left", color: C.shared });

  // Panel B.
  addText(s, "FIG1_PANEL_B__MICRO_LANE", "Fine / Micro", 322, 122, 88, 26, 12, { color: C.fine, align: "left" });
  const microGraph = addBox(s, "FIG1_PANEL_B__MICRO_GRAPH_ANCHOR", "", 325, 151, 62, 78, "none", "none", 10, { geometry: "rect", lineWidth: 0 });
  addGraph(s, "FIG1_PANEL_B__MICRO_GRAPH", 329, 156, 53, 60, C.fine, false);
  addText(s, "FIG1_PANEL_B__MICRO_GRAPH_LABEL", "Micro Local Graph", 319, 229, 76, 27, 12);
  addText(s, "FIG1_PANEL_B__MICRO_GRAPH_NOTE", "1st-order ΔWspd\n+ distance", 318, 254, 79, 38, 10.5, { color: C.muted });
  const microAdapt = addBox(s, "FIG1_PANEL_B__MICRO_ADAPT", "Prior ×\nlearned weights", 405, 170, 68, 48, C.fineLight, C.fine, 11, { lineWidth: 1.1 });
  const microDiff = addBox(s, "FIG1_PANEL_B__MICRO_DIFF", "A → 1-hop → 2-hop\nAᵀ → 1-hop → 2-hop", 482, 161, 86, 66, C.white, C.fine, 10.5, { lineWidth: 1.1 });
  addText(s, "FIG1_PANEL_B__MICRO_DIFF_LABEL", "Bidirectional 2nd-Order Graph Diffusion", 466, 229, 118, 49, 14);
  addLayerStack(s, "FIG1_PANEL_B__MICRO_TCN", 540, 284, 2, 37, 28, C.fineLight, C.fine, null);
  addText(s, "FIG1_PANEL_B__MICRO_TCN_PARAM", "k = 3 · d = {1,2}", 493, 315, 102, 22, 10, { color: C.muted });
  addText(s, "FIG1_PANEL_B__MICRO_TCN_LABEL", "Asymmetric Causal TCN", 506, 335, 88, 28, 14);
  connect(s, "FIG1_PANEL_B__MICRO_FLOW_1", microGraph, microAdapt, { fromSide: "right", toSide: "left", color: C.fine });
  connect(s, "FIG1_PANEL_B__MICRO_FLOW_2", microAdapt, microDiff, { fromSide: "right", toSide: "left", color: C.fine });

  addBox(s, "FIG1_PANEL_B__ADAPTIVE_SHARED", "Prior-Constrained Adaptive Edge Re-weighting", 335, 365, 240, 45, C.white, C.prompt, 14, { lineWidth: 1.35 });

  addText(s, "FIG1_PANEL_B__MACRO_LANE", "Coarse / Macro", 322, 420, 102, 26, 12, { color: C.coarse, align: "left" });
  const macroGraph = addBox(s, "FIG1_PANEL_B__MACRO_GRAPH_ANCHOR", "", 325, 462, 62, 78, "none", "none", 10, { geometry: "rect", lineWidth: 0 });
  addGraph(s, "FIG1_PANEL_B__MACRO_GRAPH", 329, 467, 53, 60, C.coarse, false);
  addText(s, "FIG1_PANEL_B__MACRO_GRAPH_LABEL", "Macro Trend Graph", 317, 539, 82, 27, 12);
  addText(s, "FIG1_PANEL_B__MACRO_GRAPH_NOTE", "Patv trend\n+ distance", 319, 565, 78, 38, 10.5, { color: C.muted });
  const macroAdapt = addBox(s, "FIG1_PANEL_B__MACRO_ADAPT", "Prior ×\nlearned weights", 405, 481, 68, 48, C.coarseLight, C.coarse, 11, { lineWidth: 1.1 });
  const macroDiff = addBox(s, "FIG1_PANEL_B__MACRO_DIFF", "A → 1-hop → 2-hop\nAᵀ → 1-hop → 2-hop", 482, 472, 86, 66, C.white, C.coarse, 10.5, { lineWidth: 1.1 });
  addText(s, "FIG1_PANEL_B__MACRO_DIFF_LABEL", "Bidirectional 2nd-Order Graph Diffusion", 466, 540, 118, 49, 14);
  addLayerStack(s, "FIG1_PANEL_B__MACRO_TCN", 526, 585, 4, 37, 28, C.coarseLight, C.coarse, null);
  addText(s, "FIG1_PANEL_B__MACRO_TCN_PARAM", "k = 5 · d = {1,2,4,8}", 476, 616, 119, 30, 10, { color: C.muted });
  addText(s, "FIG1_PANEL_B__MACRO_TCN_LABEL", "Asymmetric Causal TCN", 502, 648, 92, 34, 14);
  connect(s, "FIG1_PANEL_B__MACRO_FLOW_1", macroGraph, macroAdapt, { fromSide: "right", toSide: "left", color: C.coarse });
  connect(s, "FIG1_PANEL_B__MACRO_FLOW_2", macroAdapt, macroDiff, { fromSide: "right", toSide: "left", color: C.coarse });

  // Panel C.
  addText(s, "FIG1_PANEL_C__COARSE_SOURCE_LABEL", "Coarse representation", 636, 122, 151, 27, 12, { color: C.coarse });
  const coarseStrip = addTokens(s, "FIG1_PANEL_C__COARSE_SOURCE", 646, 145, 7, 17, 24, C.coarseLight, C.coarse, null, 3);
  const promptLabel = addText(s, "FIG1_PANEL_C__PROMPT_METHOD", "Macro-Trend Prompt", 665, 188, 184, 30, 14);
  const prompts = addTokens(s, "FIG1_PANEL_C__PROMPTS", 681, 225, 4, 37, 30, C.promptLight, C.prompt, ["P1","P2","P3","P4"], 5);
  addText(s, "FIG1_PANEL_C__PROMPT_NOTE", "4 prompt tokens\n/ turbine", 620, 254, 88, 35, 10.5, { color: C.muted });
  const fusionTitle = addText(s, "FIG1_PANEL_C__FUSION_TITLE", "Bidirectional Cross-Fusion", 643, 297, 230, 31, 14);
  const attn1 = addCircle(s, "FIG1_PANEL_C__ATTN_FINE", "Attn", 692, 373, 23, C.white, C.prompt, 12);
  const gate = addDiamond(s, "FIG1_PANEL_C__GATE", "G", 759, 373, 48, C.promptLight, C.prompt, 12);
  const attn2 = addCircle(s, "FIG1_PANEL_C__ATTN_COARSE", "Attn", 828, 373, 23, C.white, C.prompt, 12);
  addText(s, "FIG1_PANEL_C__DIRECTION1", "Macro-Trend Prompt → Fine", 624, 418, 134, 40, 12, { color: C.prompt });
  addText(s, "FIG1_PANEL_C__DIRECTION2", "Recent Fine → Coarse", 760, 418, 128, 40, 12, { color: C.prompt });
  addText(s, "FIG1_PANEL_C__RECENT", "24 recent steps", 652, 506, 96, 24, 10.5, { color: C.fine });
  const updatedFine = addBox(s, "FIG1_PANEL_C__UPDATED_FINE", "Updated Fine", 638, 548, 112, 34, C.fineLight, C.fine, 12, { lineWidth: 1.2 });
  const updatedCoarse = addBox(s, "FIG1_PANEL_C__UPDATED_COARSE", "Updated Coarse", 770, 548, 112, 34, C.coarseLight, C.coarse, 12, { lineWidth: 1.2 });
  addTokens(s, "FIG1_PANEL_C__UPDATED_FINE_STRIP", 644, 592, 6, 14, 22, C.fineLight, C.fine, null, 3);
  addTokens(s, "FIG1_PANEL_C__UPDATED_COARSE_STRIP", 776, 592, 6, 14, 22, C.coarseLight, C.coarse, null, 3);
  addText(s, "FIG1_PANEL_C__SEPARATE_NOTE", "Fine and Coarse remain separate", 660, 633, 174, 26, 10.5, { color: C.muted });
  connect(s, "FIG1_PANEL_C__PROMPT_FLOW", coarseStrip[3], promptLabel, { fromSide: "bottom", toSide: "top", color: C.coarse });
  connect(s, "FIG1_PANEL_C__PROMPT_TO_FUSION", prompts[1], fusionTitle, { fromSide: "bottom", toSide: "top", color: C.prompt });
  connect(s, "FIG1_PANEL_C__FUSION_TO_ATTN", fusionTitle, attn1, { fromSide: "bottom", toSide: "top", color: C.prompt });
  connect(s, "FIG1_PANEL_C__ATTN_TO_GATE_1", attn1, gate, { fromSide: "right", toSide: "left", color: C.prompt });
  connect(s, "FIG1_PANEL_C__ATTN_TO_GATE_2", attn2, gate, { fromSide: "left", toSide: "right", color: C.prompt });
  connect(s, "FIG1_PANEL_C__GATE_TO_FINE", gate, updatedFine, { fromSide: "bottom", toSide: "top", color: C.fine, kind: "elbow" });
  connect(s, "FIG1_PANEL_C__GATE_TO_COARSE", gate, updatedCoarse, { fromSide: "bottom", toSide: "top", color: C.coarse, kind: "elbow" });

  // Panel D.
  addText(s, "FIG1_PANEL_D__INFERENCE_LABEL", "Inference / Forecasting", 938, 122, 172, 28, 12, { color: C.shared, align: "left" });
  const dual = addBox(s, "FIG1_PANEL_D__DUAL_UPDATED", "Updated Fine + Updated Coarse", 958, 145, 260, 34, C.white, C.shared, 12);
  const lastState = addBox(s, "FIG1_PANEL_D__LAST_STATE", "Last-State Context", 993, 194, 190, 32, C.sharedLight, C.shared, 12);
  const stPrompt = addBox(s, "FIG1_PANEL_D__ST_PROMPT", "Node-Horizon ST Prompt", 976, 242, 224, 34, C.promptLight, C.prompt, 14);
  const decoder = addBox(s, "FIG1_PANEL_D__DECODER", "Last-State Direct Multi-Output Decoding", 954, 291, 268, 38, C.white, C.shared, 14);
  addText(s, "FIG1_PANEL_D__FORECAST_LABEL", "10-Step Forecast", 1034, 342, 120, 26, 12);
  const future = addTokens(s, "FIG1_PANEL_D__FORECAST", 950, 371, 10, 24, 27, C.white, C.shared, ["1","2","3","4","5","6","7","8","9","10"], 3);
  addHLine(s, "FIG1_PANEL_D__H3", 950, 407, 1028, C.fine, 2.1);
  addText(s, "FIG1_PANEL_D__H3_LABEL", "H3", 973, 409, 34, 19, 10, { color: C.fine });
  addHLine(s, "FIG1_PANEL_D__H6", 950, 432, 1106, C.prompt, 2.1);
  addText(s, "FIG1_PANEL_D__H6_LABEL", "H6", 1010, 434, 34, 19, 10, { color: C.prompt });
  addHLine(s, "FIG1_PANEL_D__H10", 950, 457, 1210, C.coarse, 2.1);
  addText(s, "FIG1_PANEL_D__H10_LABEL", "H10", 1060, 459, 40, 19, 10, { color: C.coarse });
  connect(s, "FIG1_PANEL_D__FORECAST_FLOW_1", dual, lastState, { fromSide: "bottom", toSide: "top", color: C.shared });
  connect(s, "FIG1_PANEL_D__FORECAST_FLOW_2", lastState, stPrompt, { fromSide: "bottom", toSide: "top", color: C.prompt });
  connect(s, "FIG1_PANEL_D__FORECAST_FLOW_3", stPrompt, decoder, { fromSide: "bottom", toSide: "top", color: C.shared });

  addHLine(s, "FIG1_PANEL_D__DIVIDER", 932, 490, 1246, "#DAB99A", 1.1);
  addText(s, "FIG1_PANEL_D__TRAINING_LABEL", "Training Objective", 938, 495, 150, 28, 12, { color: C.rose, align: "left" });
  addText(s, "FIG1_PANEL_D__LOSS_INPUTS", "Prediction  ·  Target  ·  Mask", 966, 523, 247, 23, 10.5, { color: C.muted });
  const smooth = addBox(s, "FIG1_PANEL_D__SMOOTH_L1", "Masked Smooth L1 Loss", 985, 549, 208, 32, C.roseLight, C.rose, 14);
  const hBranch = addBox(s, "FIG1_PANEL_D__H_BRANCH", "Horizon Difficulty-Rate EMA\nWeighting + Clipping\n& Normalization", 936, 583, 150, 55, C.white, C.rose, 10.5, { lineWidth: 1.05 });
  const sBranch = addBox(s, "FIG1_PANEL_D__S_BRANCH", "Dynamic Site EMA Weighting\nWeight Clipping\n& Normalization", 1094, 583, 150, 55, C.white, C.rose, 10.5, { lineWidth: 1.05 });
  const mergeLoss = addPlus(s, "FIG1_PANEL_D__LOSS_MERGE", 1090, 645, C.rose);
  const msmg = addBox(s, "FIG1_PANEL_D__MSMG", "", 970, 653, 240, 44, C.roseLight, C.rose, 12, { lineWidth: 1.25 });
  addText(s, "FIG1_PANEL_D__MSMG_TITLE", "MS-MG-DWU", 978, 653, 224, 17, 13);
  addText(s, "FIG1_PANEL_D__MSMG_SUBTITLE", "Difficulty-Rate Dynamic Weighting Loss", 978, 670, 224, 25, 10);
  connect(s, "FIG1_PANEL_D__LOSS_SPLIT_H", smooth, hBranch, { fromSide: "bottom", toSide: "top", color: C.rose, dashed: true, kind: "elbow" });
  connect(s, "FIG1_PANEL_D__LOSS_SPLIT_S", smooth, sBranch, { fromSide: "bottom", toSide: "top", color: C.rose, dashed: true, kind: "elbow" });
  connect(s, "FIG1_PANEL_D__LOSS_MERGE_H", hBranch, mergeLoss, { fromSide: "bottom", toSide: "left", color: C.rose, dashed: true, noHead: true });
  connect(s, "FIG1_PANEL_D__LOSS_MERGE_S", sBranch, mergeLoss, { fromSide: "bottom", toSide: "right", color: C.rose, dashed: true, noHead: true });
  connect(s, "FIG1_PANEL_D__LOSS_TO_MSMG", mergeLoss, msmg, { fromSide: "bottom", toSide: "top", color: C.rose, dashed: true });

  // Main inter-panel flow.
  connect(s, "FIG1_MAIN_FLOW_A_B", decomp, microGraph, { fromSide: "right", toSide: "left", color: C.line, kind: "elbow" });
  connect(s, "FIG1_MAIN_FLOW_B_C", microDiff, promptLabel, { fromSide: "right", toSide: "left", color: C.line, kind: "elbow" });
  connect(s, "FIG1_MAIN_FLOW_C_D", updatedCoarse, dual, { fromSide: "right", toSide: "left", color: C.line, kind: "elbow" });

  s.speakerNotes.textFrame.setText("[Sources]\n- D:/PaperProject/GyxPaper2/custom_models/src/st_mgprompt/model.py\n- D:/PaperProject/GyxPaper2/custom_models/src/st_mgprompt/volatility_patching.py\n- D:/PaperProject/GyxPaper2/custom_models/src/st_mgprompt/temporal_layers.py\n- D:/PaperProject/GyxPaper2/custom_models/src/st_mgprompt/cross_fusion.py\n- D:/PaperProject/GyxPaper2/custom_models/src/st_mgprompt/decoder.py\n- D:/PaperProject/GyxPaper2/custom_models/src/st_mgprompt/losses.py\n- D:/PaperProject/GyxPaper2/custom_models/src/st_mgprompt/experiment_protocol.py");
  return s;
}

function figure2(pres) {
  const s = pres.slides.add();
  s.background.fill = C.white;
  addTitle(s, "Dual Semantic Spatial Graph Construction");
  addPanel(s, "FIG2_DISTANCE", "(A) Gaussian Kernel\nDistance Graph", 18, 58, 254, 646, C.panelA, "#9AB7C6");
  addPanel(s, "FIG2_PRIOR", "(B) Dual Semantic Prior Graph Construction", 284, 58, 456, 646, C.panelB, "#86B9B4");
  addPanel(s, "FIG2_ADAPTIVE", "(C) Prior-Constrained Adaptive Edge Re-weighting", 752, 58, 510, 646, C.panelC, "#A893C5");

  // A.
  addText(s, "FIG2_DISTANCE__COORD_LABEL", "Turbine Coordinates", 58, 120, 170, 28, 14);
  const coords = [[70,185],[105,161],[126,205],[163,173],[197,216],[220,183]];
  for (let i = 0; i < coords.length; i++) {
    addCircle(s, `FIG2_DISTANCE__COORD_${i + 1}`, "", coords[i][0], coords[i][1], 6, C.white, C.shared, 10);
  }
  addText(s, "FIG2_DISTANCE__XY", "(x, y) positions", 72, 235, 150, 24, 10.5, { color: C.muted });
  const kernel = addBox(s, "FIG2_DISTANCE__KERNEL", "Gaussian Kernel Distance Graph", 45, 286, 200, 40, C.sharedLight, C.shared, 14);
  const topk = addBox(s, "FIG2_DISTANCE__TOPK", "Top-K Sparsification & Row Normalization", 38, 357, 214, 48, C.white, C.shared, 14);
  addText(s, "FIG2_DISTANCE__K", "K = 5", 99, 408, 92, 22, 10.5, { color: C.muted });
  const distGraphAnchor = addBox(s, "FIG2_DISTANCE__GRAPH_ANCHOR", "", 58, 456, 174, 142, "none", "none", 10, { geometry: "rect", lineWidth: 0 });
  addGraph(s, "FIG2_DISTANCE__GRAPH", 71, 468, 144, 112, C.shared, true);
  addText(s, "FIG2_DISTANCE__GRAPH_LABEL", "Distance Graph", 72, 608, 146, 29, 14);
  connect(s, "FIG2_DISTANCE__FLOW_1", kernel, topk, { fromSide: "bottom", toSide: "top", color: C.shared });
  connect(s, "FIG2_DISTANCE__FLOW_2", topk, distGraphAnchor, { fromSide: "bottom", toSide: "top", color: C.shared });

  // B — macro lane.
  addText(s, "FIG2_PRIOR__MACRO_TAG", "Macro prior", 300, 112, 102, 26, 12, { color: C.coarse, align: "left" });
  const macroInput = addBox(s, "FIG2_PRIOR__MACRO_INPUT", "Train-only Patv_raw", 296, 145, 130, 34, C.coarseLight, C.coarse, 12);
  addWaveform(s, "FIG2_PRIOR__MACRO_TREND", 310, 194, 100, 35, C.coarse, 2, 2.0);
  addText(s, "FIG2_PRIOR__MACRO_TREND_LABEL", "Multi-Window / Trend extraction", 294, 230, 134, 37, 10.5, { color: C.muted });
  const pearson = addBox(s, "FIG2_PRIOR__PEARSON", "Trend Pearson Correlation Graph", 437, 145, 132, 54, C.white, C.coarse, 14);
  const macroTopk = addBox(s, "FIG2_PRIOR__MACRO_TOPK", "Top-K Sparsification &\nRow Normalization", 437, 213, 132, 49, C.white, C.coarse, 12);
  addText(s, "FIG2_PRIOR__MACRO_K", "K = 10", 472, 262, 65, 20, 10.5, { color: C.muted });
  const trendGraph = addBox(s, "FIG2_PRIOR__TREND_GRAPH_ANCHOR", "", 585, 145, 68, 78, "none", "none", 10, { geometry: "rect", lineWidth: 0 });
  addGraph(s, "FIG2_PRIOR__TREND_GRAPH", 589, 150, 58, 59, C.coarse, false);
  addText(s, "FIG2_PRIOR__TREND_GRAPH_LABEL", "Trend Graph\n0.7", 580, 222, 80, 38, 10.5, { color: C.coarse });
  const macroSigma = addSigma(s, "FIG2_PRIOR__MACRO_SIGMA", 677, 205, C.coarse);
  addText(s, "FIG2_PRIOR__MACRO_DIST", "Distance Graph 0.3", 625, 242, 103, 25, 10.5, { color: C.shared });
  const macroPrior = addBox(s, "FIG2_PRIOR__MACRO_PRIOR", "Macro Prior Graph", 590, 300, 128, 36, C.coarseLight, C.coarse, 14);
  addText(s, "FIG2_PRIOR__MACRO_FUSION", "Prior Graph Fusion", 560, 272, 122, 25, 12);
  connect(s, "FIG2_PRIOR__MACRO_FLOW_1", macroInput, pearson, { fromSide: "right", toSide: "left", color: C.coarse });
  connect(s, "FIG2_PRIOR__MACRO_FLOW_2", pearson, macroTopk, { fromSide: "bottom", toSide: "top", color: C.coarse });
  connect(s, "FIG2_PRIOR__MACRO_FLOW_3", macroTopk, trendGraph, { fromSide: "right", toSide: "left", color: C.coarse });
  connect(s, "FIG2_PRIOR__MACRO_FLOW_4", trendGraph, macroSigma, { fromSide: "right", toSide: "left", color: C.coarse });
  connect(s, "FIG2_PRIOR__MACRO_FLOW_5", macroSigma, macroPrior, { fromSide: "right", toSide: "right", color: C.coarse, kind: "elbow" });
  addHLine(s, "FIG2_PRIOR__DIVIDER", 298, 357, 726, "#A6C9C5", 1.0);

  // B — micro lane.
  addText(s, "FIG2_PRIOR__MICRO_TAG", "Micro prior", 300, 371, 102, 26, 12, { color: C.fine, align: "left" });
  const microInput = addBox(s, "FIG2_PRIOR__MICRO_INPUT", "Train-only Wspd", 302, 405, 118, 34, C.fineLight, C.fine, 12);
  addWaveform(s, "FIG2_PRIOR__MICRO_DIFF_WAVE", 310, 452, 100, 35, C.fine, 0, 2.0);
  addText(s, "FIG2_PRIOR__MICRO_DIFF_LABEL", "1st-Order Difference", 302, 491, 118, 27, 12, { color: C.muted });
  const cosine = addBox(s, "FIG2_PRIOR__COSINE", "Cosine Similarity on 1st-Order Difference", 437, 405, 132, 58, C.white, C.fine, 14);
  const microTopk = addBox(s, "FIG2_PRIOR__MICRO_TOPK", "Top-K Sparsification &\nRow Normalization", 437, 478, 132, 49, C.white, C.fine, 12);
  addText(s, "FIG2_PRIOR__MICRO_K", "K = 5", 472, 527, 65, 20, 10.5, { color: C.muted });
  const localGraph = addBox(s, "FIG2_PRIOR__LOCAL_GRAPH_ANCHOR", "", 585, 405, 68, 78, "none", "none", 10, { geometry: "rect", lineWidth: 0 });
  addGraph(s, "FIG2_PRIOR__LOCAL_GRAPH", 589, 410, 58, 59, C.fine, false);
  addText(s, "FIG2_PRIOR__LOCAL_GRAPH_LABEL", "Local-Change\nGraph 0.3", 579, 481, 82, 40, 10.5, { color: C.fine });
  const microSigma = addSigma(s, "FIG2_PRIOR__MICRO_SIGMA", 677, 465, C.fine);
  addText(s, "FIG2_PRIOR__MICRO_DIST", "Distance Graph 0.7", 625, 507, 103, 25, 10.5, { color: C.shared });
  const microPrior = addBox(s, "FIG2_PRIOR__MICRO_PRIOR", "Micro Prior Graph", 590, 565, 128, 36, C.fineLight, C.fine, 14);
  addText(s, "FIG2_PRIOR__MICRO_FUSION", "Prior Graph Fusion", 560, 537, 122, 25, 12);
  connect(s, "FIG2_PRIOR__MICRO_FLOW_1", microInput, cosine, { fromSide: "right", toSide: "left", color: C.fine });
  connect(s, "FIG2_PRIOR__MICRO_FLOW_2", cosine, microTopk, { fromSide: "bottom", toSide: "top", color: C.fine });
  connect(s, "FIG2_PRIOR__MICRO_FLOW_3", microTopk, localGraph, { fromSide: "right", toSide: "left", color: C.fine });
  connect(s, "FIG2_PRIOR__MICRO_FLOW_4", localGraph, microSigma, { fromSide: "right", toSide: "left", color: C.fine });
  connect(s, "FIG2_PRIOR__MICRO_FLOW_5", microSigma, microPrior, { fromSide: "right", toSide: "right", color: C.fine, kind: "elbow" });

  // C — four-step reader test.
  addText(s, "FIG2_ADAPTIVE__SUBTITLE", "Learn edge importance without creating unsupported connections", 780, 112, 455, 30, 12, { color: C.muted });
  const stepXs = [770, 891, 1012, 1133];
  const stepTitles = [
    "Prior Graph Defines\nAllowed Edges",
    "Learn Adaptive\nEdge Importance",
    "Mask Unsupported\nEdges",
    "Re-weight &\nRow Normalize",
  ];
  for (let i = 0; i < 4; i++) {
    addCircle(s, `FIG2_ADAPTIVE__STEP_${i + 1}_NUM`, String(i + 1), stepXs[i] + 52, 177, 18, i === 1 ? C.promptLight : C.white, C.prompt, 14);
    addText(s, `FIG2_ADAPTIVE__STEP_${i + 1}_TITLE`, stepTitles[i], stepXs[i], 202, 105, 47, 12);
  }
  const priorAnchor = addBox(s, "FIG2_ADAPTIVE__STEP1_ANCHOR", "", 783, 267, 90, 118, "none", "none", 10, { geometry: "rect", lineWidth: 0 });
  addGraph(s, "FIG2_ADAPTIVE__STEP1_GRAPH", 790, 276, 76, 75, C.shared, false);
  addText(s, "FIG2_ADAPTIVE__STEP1_ALLOWED", "Allowed connections", 781, 360, 94, 28, 10.5, { color: C.shared });

  addText(s, "FIG2_ADAPTIVE__STEP2_E_LABEL", "Independent node embeddings", 891, 262, 105, 34, 10.5, { color: C.muted });
  addText(s, "FIG2_ADAPTIVE__STEP2_EM_LABEL", "Macro: E1ᵐ / E2ᵐ", 891, 298, 105, 22, 10, { color: C.coarse });
  addTokens(s, "FIG2_ADAPTIVE__STEP2_EM", 918, 322, 2, 20, 36, C.coarseLight, C.coarse, null, 8);
  addText(s, "FIG2_ADAPTIVE__STEP2_EU_LABEL", "Micro: E1μ / E2μ", 891, 362, 105, 22, 10, { color: C.fine });
  addTokens(s, "FIG2_ADAPTIVE__STEP2_EU", 918, 386, 2, 20, 36, C.fineLight, C.fine, null, 8);
  addText(s, "FIG2_ADAPTIVE__STEP2_DIM", "dₑ = 10", 913, 425, 70, 22, 10.5, { color: C.muted });
  const score = addBox(s, "FIG2_ADAPTIVE__STEP2_SCORE", "Adaptive\nedge scores", 911, 451, 72, 45, C.promptLight, C.prompt, 12);

  addBox(s, "FIG2_ADAPTIVE__STEP3_SCORE", "Adaptive scores", 1018, 278, 92, 32, C.promptLight, C.prompt, 11);
  addBox(s, "FIG2_ADAPTIVE__STEP3_MASK", "Prior support mask", 1018, 326, 92, 32, C.sharedLight, C.shared, 11);
  addText(s, "FIG2_ADAPTIVE__STEP3_OVERLAY", "×", 1047, 358, 34, 30, 18, { color: C.prompt });
  const maskedAnchor = addBox(s, "FIG2_ADAPTIVE__STEP3_ANCHOR", "", 1021, 393, 86, 89, "none", "none", 10, { geometry: "rect", lineWidth: 0 });
  addGraph(s, "FIG2_ADAPTIVE__STEP3_GRAPH", 1025, 398, 76, 67, C.prompt, false, true);
  addText(s, "FIG2_ADAPTIVE__STEP3_REMOVED", "Unsupported edges removed", 1017, 470, 96, 35, 10.5, { color: C.rose });
  addText(s, "FIG2_ADAPTIVE__STEP3_X", "×", 1084, 412, 24, 24, 16, { color: C.rose });

  const effectiveAnchor = addBox(s, "FIG2_ADAPTIVE__STEP4_ANCHOR", "", 1141, 275, 94, 142, "none", "none", 10, { geometry: "rect", lineWidth: 0 });
  addGraph(s, "FIG2_ADAPTIVE__STEP4_GRAPH", 1148, 284, 80, 82, C.prompt, true);
  addText(s, "FIG2_ADAPTIVE__STEP4_ROW", "Row Normalization", 1141, 373, 94, 27, 10.5, { color: C.muted });
  addText(s, "FIG2_ADAPTIVE__STEP4_EFFECTIVE", "Effective Graph", 1141, 408, 94, 30, 12);
  addText(s, "FIG2_ADAPTIVE__STEP4_SAME", "Same connectivity\nnew edge strengths", 1141, 443, 94, 43, 10.5, { color: C.muted });
  connect(s, "FIG2_ADAPTIVE__FLOW_1", priorAnchor, score, { fromSide: "right", toSide: "left", color: C.prompt, kind: "elbow" });
  connect(s, "FIG2_ADAPTIVE__FLOW_2", score, maskedAnchor, { fromSide: "right", toSide: "left", color: C.prompt, kind: "elbow" });
  connect(s, "FIG2_ADAPTIVE__FLOW_3", maskedAnchor, effectiveAnchor, { fromSide: "right", toSide: "left", color: C.prompt, kind: "elbow" });
  addBox(s, "FIG2_ADAPTIVE__EXPLAIN", "Prior fixes connectivity; learning only adjusts the strength of supported edges.", 785, 605, 444, 42, C.white, C.prompt, 11, { lineWidth: 1.15 });
  addText(s, "FIG2_ADAPTIVE__BRANCH_NOTE", "Macro and Micro use independent adaptive embeddings", 835, 655, 345, 28, 10.5, { color: C.muted });

  s.speakerNotes.textFrame.setText("[Sources]\n- D:/PaperProject/GyxPaper2/custom_models/src/st_mgprompt/graph_prior.py\n- D:/PaperProject/GyxPaper2/custom_models/src/st_mgprompt/graph_layers.py\n- D:/PaperProject/GyxPaper2/custom_models/src/st_mgprompt/config.py");
  return s;
}

function figure3(pres) {
  const s = pres.slides.add();
  s.background.fill = "#FCFAFE";
  addTitle(s, "Macro-Trend Prompt and Bidirectional Cross-Fusion");
  addBox(s, "FIG3_CANVAS", "", 18, 58, 1244, 646, "#F9F6FC", "#C8BADB", 12, { radius: "rounded-xl", lineWidth: 1.2, shadow: "2px 3px 8px #23343D/08" });
  addText(s, "FIG3_PROMPT__BAND_TITLE", "Macro-Trend Prompt Generation", 42, 72, 500, 34, 18, { bold: true, align: "left" });
  addText(s, "FIG3_CROSSFUSION__BAND_TITLE", "Bidirectional Cross-Fusion", 42, 236, 420, 34, 18, { bold: true, align: "left" });
  addText(s, "FIG3_OUTPUT__BAND_TITLE", "Updated Dual Representations", 42, 544, 420, 34, 18, { bold: true, align: "left" });
  addHLine(s, "FIG3_PROMPT__DIVIDER", 40, 224, 1240, "#D7CCE7", 1.0);
  addHLine(s, "FIG3_CROSSFUSION__DIVIDER", 40, 532, 1240, "#D7CCE7", 1.0);

  // Top band.
  addText(s, "FIG3_PROMPT__COARSE_HISTORY_LABEL", "Coarse History", 75, 116, 150, 26, 14, { color: C.coarse });
  const coarseTop = addTokens(s, "FIG3_PROMPT__COARSE_HISTORY", 70, 151, 10, 24, 31, C.coarseLight, C.coarse, null, 4);
  const pool = addCircle(s, "FIG3_PROMPT__POOL", "Pool", 430, 166, 32, C.white, C.coarse, 12);
  addText(s, "FIG3_PROMPT__POOL_LABEL", "Temporal Weighted Pooling", 340, 197, 180, 24, 12);
  const macroLabel = addText(s, "FIG3_PROMPT__MACRO_LABEL", "Macro-Trend Prompt", 570, 113, 185, 29, 14);
  const promptTop = addTokens(s, "FIG3_PROMPT__TOKENS", 578, 151, 4, 39, 32, C.promptLight, C.prompt, ["P1","P2","P3","P4"], 7);
  addText(s, "FIG3_PROMPT__TOKEN_NOTE", "4 tokens / turbine", 760, 202, 120, 20, 10.5, { color: C.muted });
  addText(s, "FIG3_PROMPT__NODE_SPECIFIC", "Node-Specific Macro-Trend Prompt", 850, 134, 294, 38, 14);
  const nodePrompt = addTokens(s, "FIG3_PROMPT__NODE_PROMPT", 892, 179, 6, 27, 25, C.promptLight, C.prompt, null, 5);
  connect(s, "FIG3_PROMPT__FLOW_1", coarseTop[9], pool, { fromSide: "right", toSide: "left", color: C.coarse });
  connect(s, "FIG3_PROMPT__FLOW_2", pool, macroLabel, { fromSide: "right", toSide: "left", color: C.prompt });
  connect(s, "FIG3_PROMPT__FLOW_3", promptTop[3], nodePrompt[0], { fromSide: "right", toSide: "left", color: C.prompt });

  // Middle band: source sequences and two distinct directions.
  addText(s, "FIG3_CROSSFUSION__FINE_LABEL", "Fine History", 68, 279, 145, 26, 14, { color: C.fine });
  const fineSeq = addTokens(s, "FIG3_CROSSFUSION__FINE_HISTORY", 65, 314, 12, 21, 29, C.fineLight, C.fine, null, 3);
  addHLine(s, "FIG3_CROSSFUSION__RECENT_BRACKET", 256, 351, 350, C.fine, 2.0);
  addVLine(s, "FIG3_CROSSFUSION__RECENT_BRACKET_L", 256, 345, 357, C.fine, 1.5);
  addVLine(s, "FIG3_CROSSFUSION__RECENT_BRACKET_R", 350, 345, 357, C.fine, 1.5);
  addText(s, "FIG3_CROSSFUSION__RECENT_FINE", "Recent Fine", 257, 358, 92, 24, 12, { color: C.fine });
  addText(s, "FIG3_CROSSFUSION__RECENT_NOTE", "24 steps ≈ 4 h", 257, 381, 92, 21, 10.5, { color: C.muted });

  addText(s, "FIG3_CROSSFUSION__COARSE_LABEL", "Coarse History", 1040, 279, 160, 26, 14, { color: C.coarse });
  const coarseSeq = addTokens(s, "FIG3_CROSSFUSION__COARSE_HISTORY", 905, 314, 12, 21, 29, C.coarseLight, C.coarse, null, 3);

  addText(s, "FIG3_CROSSFUSION__DIR1", "Macro-Trend Prompt → Fine", 270, 270, 220, 31, 14, { color: C.prompt });
  const attn1 = addCircle(s, "FIG3_CROSSFUSION__ATTN1", "Attn", 516, 329, 28, C.white, C.prompt, 12);
  addText(s, "FIG3_CROSSFUSION__ATTN1_QKV", "Q: Fine\nK,V: Macro Prompt", 350, 398, 150, 38, 10.5, { color: C.muted });
  addText(s, "FIG3_CROSSFUSION__DIR1_OUT", "Trend-guided Fine update", 350, 442, 188, 29, 12, { color: C.fine });

  addText(s, "FIG3_CROSSFUSION__DIR2", "Recent Fine → Coarse", 701, 270, 220, 31, 14, { color: C.prompt });
  const attn2 = addCircle(s, "FIG3_CROSSFUSION__ATTN2", "Attn", 806, 405, 28, C.white, C.prompt, 12);
  addText(s, "FIG3_CROSSFUSION__ATTN2_QKV", "Q: Coarse\nK,V: Recent Fine", 835, 421, 135, 43, 10.5, { color: C.muted, align: "left" });
  addText(s, "FIG3_CROSSFUSION__DIR2_OUT", "Coarse update", 760, 482, 122, 25, 12, { color: C.coarse });

  const gate = addDiamond(s, "FIG3_CROSSFUSION__GATE", "G", 660, 474, 62, C.promptLight, C.prompt, 14);
  addText(s, "FIG3_CROSSFUSION__GATE_LABEL", "Fusion Gate", 606, 434, 108, 25, 14);
  addText(s, "FIG3_CROSSFUSION__G_LABEL", "G", 568, 508, 28, 21, 10.5, { color: C.fine });
  addText(s, "FIG3_CROSSFUSION__ONE_MINUS_G", "1−G", 720, 508, 42, 21, 10.5, { color: C.coarse });
  connect(s, "FIG3_CROSSFUSION__FINE_Q", fineSeq[11], attn1, { fromSide: "right", toSide: "left", color: C.fine });
  connect(s, "FIG3_CROSSFUSION__PROMPT_KV", promptTop[0], attn1, { fromSide: "bottom", toSide: "top", color: C.prompt, kind: "elbow" });
  connect(s, "FIG3_CROSSFUSION__COARSE_Q", coarseSeq[0], attn2, { fromSide: "left", toSide: "right", color: C.coarse });
  connect(s, "FIG3_CROSSFUSION__RECENT_KV", fineSeq[11], attn2, { fromSide: "bottom", toSide: "left", color: C.fine, kind: "elbow" });
  connect(s, "FIG3_CROSSFUSION__ATTN1_GATE", attn1, gate, { fromSide: "bottom", toSide: "left", color: C.prompt });
  connect(s, "FIG3_CROSSFUSION__ATTN2_GATE", attn2, gate, { fromSide: "bottom", toSide: "right", color: C.prompt });

  // Bottom band.
  const outFine = addBox(s, "FIG3_OUTPUT__FINE_LABEL", "Updated Fine Representation", 180, 588, 275, 38, C.fineLight, C.fine, 14);
  addTokens(s, "FIG3_OUTPUT__FINE_STRIP", 208, 638, 9, 23, 28, C.fineLight, C.fine, null, 4);
  const outCoarse = addBox(s, "FIG3_OUTPUT__COARSE_LABEL", "Updated Coarse Representation", 760, 588, 290, 38, C.coarseLight, C.coarse, 14);
  addTokens(s, "FIG3_OUTPUT__COARSE_STRIP", 797, 638, 9, 23, 28, C.coarseLight, C.coarse, null, 4);
  addText(s, "FIG3_OUTPUT__SEPARATE", "Representations remain separate after fusion", 470, 633, 280, 28, 11, { color: C.muted });
  const decoderExit = addBox(s, "FIG3_OUTPUT__DECODER_EXIT", "To Last-State Direct\nMulti-Output Decoder", 1084, 606, 148, 52, C.white, C.shared, 12, { dashed: true, lineWidth: 1.05 });
  connect(s, "FIG3_OUTPUT__GATE_FINE", gate, outFine, { fromSide: "bottom", toSide: "top", color: C.fine });
  connect(s, "FIG3_OUTPUT__GATE_COARSE", gate, outCoarse, { fromSide: "bottom", toSide: "top", color: C.coarse });
  connect(s, "FIG3_OUTPUT__TO_DECODER", outCoarse, decoderExit, { fromSide: "right", toSide: "left", color: C.shared });

  s.speakerNotes.textFrame.setText("[Sources]\n- D:/PaperProject/GyxPaper2/custom_models/src/st_mgprompt/coupling_block.py\n- D:/PaperProject/GyxPaper2/custom_models/src/st_mgprompt/cross_fusion.py\n- D:/PaperProject/GyxPaper2/custom_models/src/st_mgprompt/prompt_alignment.py\n- D:/PaperProject/GyxPaper2/custom_models/src/st_mgprompt/decoder.py");
  return s;
}

async function writeBlob(file, blob) {
  await fs.writeFile(file, new Uint8Array(await blob.arrayBuffer()));
}

async function main() {
  await fs.mkdir(BUILD_DIR, { recursive: true });
  const p = Presentation.create({ slideSize: { width: W, height: H } });
  p.theme.colorScheme = {
    name: "ST-MGPrompt Scientific",
    themeColors: {
      accent1: C.coarse, accent2: C.fine, accent3: C.prompt, accent4: C.rose,
      accent5: C.shared, accent6: C.green, bg1: C.white, bg2: C.grayLight,
      tx1: C.ink, tx2: C.muted, dk1: "#000000", dk2: C.ink,
      lt1: C.white, lt2: C.grayLight, hlink: C.prompt, folHlink: C.rose,
    },
  };
  p.view.hideGridlines();
  p.view.hideGuides();
  figure1(p);
  figure2(p);
  figure3(p);

  for (let i = 0; i < p.slides.items.length; i++) {
    const slide = p.slides.items[i];
    await writeBlob(path.join(BUILD_DIR, `draft_slide_${i + 1}.png`), await p.export({ slide, format: "png", scale: 2 }));
    await fs.writeFile(path.join(BUILD_DIR, `draft_slide_${i + 1}.layout.json`), await (await slide.export({ format: "layout" })).text());
  }
  await writeBlob(path.join(BUILD_DIR, "draft_montage.webp"), await p.export({ format: "webp", montage: { columns: 1, slideWidth: 1280, gap: 24, padding: 24, background: "#FFFFFF" } }));
  const snapshot = await p.inspect({ kind: "slide,textbox,shape,notes", maxChars: 12000 });
  await fs.writeFile(path.join(BUILD_DIR, "inspect.txt"), snapshot.ndjson, "utf8");
  const pptx = await PresentationFile.exportPptx(p);
  await pptx.save(FINAL_PPTX);
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
