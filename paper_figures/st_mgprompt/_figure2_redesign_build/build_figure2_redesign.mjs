import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const SOURCE = "D:/PaperProject/GyxPaper2/paper_figures/st_mgprompt/ST-MGPrompt_Fig2_Fig3_Compact_Final.pptx";
const OUT = "D:/PaperProject/GyxPaper2/paper_figures/st_mgprompt/ST-MGPrompt_Figure2_Redesigned_Final.pptx";
const QA_DIR = "D:/PaperProject/GyxPaper2/paper_figures/st_mgprompt/_figure2_redesign_build/final_qa";
const W = 864;
const H = 480;
const FONT = "Times New Roman";

const C = {
  ink: "#17232C",
  secondary: "#5D7280",
  line: "#6F8795",
  distance: "#66879A",
  sharedBg: "#F3F7F9",
  teal: "#3E8F8A",
  tealLight: "#E6F3F1",
  tealBg: "#F0F8F6",
  orange: "#D9863D",
  orangeLight: "#FFF0DF",
  purple: "#7858A6",
  purpleLight: "#F1ECF8",
  purpleBg: "#F8F5FC",
  unsupported: "#B7BEC4",
  red: "#B86660",
  white: "#FFFFFF",
};

const PRIOR_EDGES = [[0, 1], [0, 5], [1, 2], [1, 5], [2, 5], [3, 4], [4, 5]];
const SCORE_EDGES = [...PRIOR_EDGES, [2, 3], [2, 4]];

async function saveBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

function noLine() {
  return { style: "solid", fill: "none", width: 0 };
}

function addShape(slide, name, geometry, x, y, w, h, fill = "none", line = noLine(), radius = undefined) {
  return slide.shapes.add({
    geometry,
    name,
    position: { left: x, top: y, width: w, height: h },
    fill,
    line,
    ...(radius !== undefined && ["rect", "textbox", "roundRect"].includes(geometry) ? { borderRadius: radius } : {}),
  });
}

function setText(shape, text, opts = {}) {
  shape.text = text;
  shape.text.style = {
    typeface: FONT,
    fontSize: (opts.size ?? 11) * 4 / 3,
    bold: opts.bold ?? false,
    italic: opts.italic ?? false,
    color: opts.color ?? C.ink,
    alignment: opts.align ?? "center",
    verticalAlignment: opts.valign ?? "middle",
    autoFit: opts.autoFit ?? "shrinkText",
    wrap: opts.wrap ?? "square",
    insets: opts.insets ?? { left: 1.5, right: 1.5, top: 0.5, bottom: 0.5 },
  };
  return shape;
}

function addText(slide, name, text, x, y, w, h, opts = {}) {
  return setText(addShape(slide, name, "textbox", x, y, w, h, "none", noLine()), text, opts);
}

function addLine(slide, name, x1, y1, x2, y2, color, width = 1.2, dashed = false) {
  const left = Math.min(x1, x2);
  const top = Math.min(y1, y2);
  return slide.shapes.add({
    geometry: "line",
    name,
    position: {
      left,
      top,
      width: Math.max(Math.abs(x2 - x1), 0.1),
      height: Math.max(Math.abs(y2 - y1), 0.1),
      horizontalFlip: x2 < x1,
      verticalFlip: y2 < y1,
    },
    fill: "none",
    line: { style: dashed ? "dashed" : "solid", fill: color, width },
  });
}

function connect(slide, from, to, opts = {}) {
  const connector = slide.shapes.connect(from, to, {
    kind: opts.kind ?? "straight",
    fromSide: opts.fromSide,
    toSide: opts.toSide,
    line: { style: opts.dashed ? "dashed" : "solid", fill: opts.color ?? C.line, width: opts.width ?? 1.6 },
    head: { type: "none" },
    tail: { type: opts.arrow === false ? "none" : "triangle", width: "sm", length: "sm" },
  });
  if (opts.name) connector.name = opts.name;
  connector.bringToFront();
  return connector;
}

function addNode(slide, name, x, y, color, r = 3.2, fill = C.white) {
  return addShape(slide, name, "ellipse", x - r, y - r, 2 * r, 2 * r, fill, { style: "solid", fill: color, width: 1.05 });
}

function graphPoints(x, y, w, h) {
  return [[0.10, 0.52], [0.28, 0.18], [0.53, 0.25], [0.82, 0.50], [0.63, 0.82], [0.33, 0.72]]
    .map(([px, py]) => [x + px * w, y + py * h]);
}

function drawGraph(slide, prefix, x, y, w, h, color, opts = {}) {
  const anchor = addShape(slide, prefix, "rect", x, y, w, h, "none", noLine());
  const pts = graphPoints(x, y, w, h);
  const edges = opts.edges ?? PRIOR_EDGES;
  const widths = opts.widths ?? edges.map((_, i) => 1 + (i % 3) * 0.35);
  const unsupported = opts.unsupported ?? [];
  for (let i = 0; i < edges.length; i += 1) {
    const [a, b] = edges[i];
    const isUnsupported = unsupported.includes(i);
    addLine(slide, `${prefix}__EDGE_${i + 1}`, pts[a][0], pts[a][1], pts[b][0], pts[b][1], isUnsupported ? C.unsupported : color, widths[i] ?? 1.2, isUnsupported);
  }
  for (let i = 0; i < pts.length; i += 1) {
    addNode(slide, `${prefix}__NODE_${i + 1}`, pts[i][0], pts[i][1], color, opts.nodeRadius ?? 2.8);
  }
  if (opts.crossUnsupported) {
    for (const i of unsupported) {
      const [a, b] = edges[i];
      const mx = (pts[a][0] + pts[b][0]) / 2;
      const my = (pts[a][1] + pts[b][1]) / 2;
      addLine(slide, `${prefix}__CROSS_${i}_A`, mx - 4, my - 4, mx + 4, my + 4, C.red, 1.25);
      addLine(slide, `${prefix}__CROSS_${i}_B`, mx - 4, my + 4, mx + 4, my - 4, C.red, 1.25);
    }
  }
  return anchor;
}

function drawDenseGraph(slide, prefix, x, y, w, h, color) {
  const denseEdges = [[0,1],[0,2],[0,3],[0,4],[0,5],[1,2],[1,3],[1,4],[1,5],[2,3],[2,4],[2,5],[3,4],[3,5],[4,5]];
  return drawGraph(slide, prefix, x, y, w, h, color, {
    edges: denseEdges,
    widths: denseEdges.map((_, i) => 0.55 + (i % 4) * 0.18),
    nodeRadius: 2.35,
  });
}

function drawRibbon(slide, prefix, x, y, w, h, color, fill, vals) {
  const anchor = addShape(slide, prefix, "roundRect", x, y, w, h, fill, { style: "solid", fill: color, width: 0.9 }, 8);
  const px0 = x + 5;
  const py0 = y + 4;
  const iw = w - 10;
  const ih = h - 8;
  for (let i = 0; i < vals.length - 1; i += 1) {
    addLine(slide, `${prefix}__WAVE_${i + 1}`, px0 + i * iw / (vals.length - 1), py0 + vals[i] * ih, px0 + (i + 1) * iw / (vals.length - 1), py0 + vals[i + 1] * ih, color, 1.35);
  }
  return anchor;
}

function drawTrend(slide, prefix, x, y, w, h, color) {
  const anchor = addShape(slide, prefix, "rect", x, y, w, h, "none", noLine());
  const rough = [0.62,0.28,0.68,0.38,0.57,0.24,0.52,0.33,0.40];
  const smooth = [0.60,0.54,0.49,0.44,0.39,0.35,0.32,0.30,0.28];
  for (const [vals, yy, width] of [[rough, y + 2, 0.75], [smooth, y + h * 0.43, 1.8]]) {
    for (let i = 0; i < vals.length - 1; i += 1) {
      addLine(slide, `${prefix}__${width > 1 ? "TREND" : "RAW"}_${i + 1}`, x + i * w / 8, yy + vals[i] * h * 0.45, x + (i + 1) * w / 8, yy + vals[i + 1] * h * 0.45, width > 1 ? color : C.unsupported, width);
    }
  }
  return anchor;
}

function drawDifference(slide, prefix, x, y, w, h, color) {
  const anchor = addShape(slide, prefix, "rect", x, y, w, h, "none", noLine());
  const vals = [0.52,0.17,0.82,0.31,0.68,0.20,0.77,0.37,0.58];
  addLine(slide, `${prefix}__ZERO`, x, y + h / 2, x + w, y + h / 2, C.unsupported, 0.65, true);
  for (let i = 0; i < vals.length - 1; i += 1) {
    addLine(slide, `${prefix}__DELTA_${i + 1}`, x + i * w / 8, y + vals[i] * h, x + (i + 1) * w / 8, y + vals[i + 1] * h, color, 1.45);
  }
  return anchor;
}

function drawPairedCurves(slide, prefix, x, y, w, h, color, symbol) {
  const anchor = addShape(slide, prefix, "rect", x, y, w, h, "none", noLine());
  const a = [0.62,0.42,0.26,0.31,0.47,0.61,0.54,0.35,0.25];
  const b = [0.72,0.53,0.37,0.40,0.54,0.69,0.63,0.46,0.34];
  for (const [vals, yy, shade] of [[a, y, color], [b, y + 3, C.line]]) {
    for (let i = 0; i < vals.length - 1; i += 1) {
      addLine(slide, `${prefix}__CURVE_${shade === color ? "A" : "B"}_${i + 1}`, x + i * w / 8, yy + vals[i] * (h - 5), x + (i + 1) * w / 8, yy + vals[i + 1] * (h - 5), shade, 1.25);
    }
  }
  addText(slide, `${prefix}__SYMBOL`, symbol, x + w * 0.20, y + h * 0.23, w * 0.60, h * 0.48, { size: 12.5, color, wrap: "none" });
  return anchor;
}

function drawEmbeddingStrips(slide, prefix, x, y, w, h) {
  const anchor = addShape(slide, prefix, "rect", x, y, w, h, "none", noLine());
  const groups = [{ x: x + 2, color: C.teal, fill: C.tealLight }, { x: x + w / 2 + 1, color: C.orange, fill: C.orangeLight }];
  for (let g = 0; g < groups.length; g += 1) {
    for (let b = 0; b < 2; b += 1) {
      const bx = groups[g].x + b * 7.5;
      addShape(slide, `${prefix}__G${g + 1}_BAR_${b + 1}`, "roundRect", bx, y + 3, 6, h - 6, groups[g].fill, { style: "solid", fill: groups[g].color, width: 0.75 }, 2);
      for (let s = 1; s <= 5; s += 1) {
        addLine(slide, `${prefix}__G${g + 1}_STRIPE_${b + 1}_${s}`, bx + 0.8, y + 3 + s * (h - 6) / 6, bx + 5.2, y + 3 + s * (h - 6) / 6, groups[g].color, 0.55);
      }
    }
  }
  return anchor;
}

function drawSmallLegendStrip(slide, prefix, x, y, color, fill) {
  const anchor = addShape(slide, prefix, "rect", x, y, 29, 13, "none", noLine());
  for (let b = 0; b < 3; b += 1) {
    addShape(slide, `${prefix}__BAR_${b + 1}`, "roundRect", x + b * 9.5, y, 7, 13, fill, { style: "solid", fill: color, width: 0.7 }, 2);
  }
  return anchor;
}

function buildFigure2(slide) {
  slide.background.fill = C.white;
  addText(slide, "FIGURE_TITLE", "Dual Semantic Spatial Graph Construction", 14, 5, 836, 35, { size: 25, bold: true, align: "left" });

  const sharedBg = addShape(slide, "FIG2_SHARED_DISTANCE__REGION", "roundRect", 10, 47, 196, 421, C.sharedBg, noLine(), 22);
  const dualBg = addShape(slide, "FIG2_DUAL_STREAMS__REGION", "roundRect", 191, 67, 342, 340, C.tealBg, noLine(), 58);
  const adaptiveBg = addShape(slide, "FIG2_ADAPTIVE__REGION", "roundRect", 527, 47, 327, 421, C.purpleBg, noLine(), 26);
  sharedBg.sendToBack();
  dualBg.sendToBack();
  adaptiveBg.sendToBack();

  addText(slide, "FIG2_SHARED__HEADING", "Shared Spatial Prior", 20, 54, 174, 23, { size: 15.5 });
  addText(slide, "FIG2_DUAL__HEADING", "Dual Semantic Priors", 215, 54, 291, 23, { size: 15.5 });
  addText(slide, "FIG2_ADAPTIVE__HEADING", "Prior-Constrained Edge Re-weighting", 540, 54, 302, 23, { size: 15 });

  const coordAnchor = addShape(slide, "FIG2_SHARED_DISTANCE__COORDS", "rect", 25, 88, 65, 55, "none", noLine());
  const coordPts = [[33,126],[47,102],[61,132],[76,110],[86,137],[53,119]];
  coordPts.forEach(([x, y], i) => addNode(slide, `FIG2_SHARED_DISTANCE__COORD_${i + 1}`, x, y, C.distance, 3));
  addText(slide, "FIG2_SHARED_DISTANCE__XY", "(x, y)", 25, 143, 65, 20, { size: 10.5, color: C.secondary, autoFit: "none" });

  const gaussianAnchor = addShape(slide, "FIG2_SHARED_DISTANCE__GAUSSIAN", "rect", 112, 88, 76, 55, "none", noLine());
  const gp = [[119,126],[137,100],[156,130],[178,104],[181,138]];
  gp.forEach(([x, y], i) => addNode(slide, `FIG2_SHARED_DISTANCE__G_NODE_${i + 1}`, x, y, C.distance, 3));
  addLine(slide, "FIG2_SHARED_DISTANCE__G_NEAR", 119,126,137,100,C.distance,2.6);
  addLine(slide, "FIG2_SHARED_DISTANCE__G_MID1", 137,100,156,130,C.distance,1.8);
  addLine(slide, "FIG2_SHARED_DISTANCE__G_MID2", 156,130,178,104,C.distance,1.25);
  addLine(slide, "FIG2_SHARED_DISTANCE__G_FAR", 178,104,181,138,C.distance,0.65);
  connect(slide, coordAnchor, gaussianAnchor, { name: "FIG2_SHARED_DISTANCE__COORD_TO_GAUSS", fromSide: "right", toSide: "left", color: C.distance, width: 1.55 });
  addText(slide, "FIG2_SHARED_DISTANCE__METHOD", "Gaussian Kernel", 92, 149, 102, 20, { size: 12.5 });

  const denseDistance = drawDenseGraph(slide, "FIG2_SHARED_DISTANCE__DENSE", 27, 211, 58, 44, C.distance);
  const distanceGraph = drawGraph(slide, "FIG2_SHARED_DISTANCE", 125, 207, 67, 51, C.distance, {
    widths: [2.1, 1.0, 1.75, 2.4, 1.15, 1.9, 1.25],
    nodeRadius: 2.8,
  });
  connect(slide, denseDistance, distanceGraph, { name: "FIG2_SHARED_DISTANCE__SPARSIFY", fromSide: "right", toSide: "left", color: C.distance, width: 1.6 });
  addText(slide, "FIG2_SHARED_DISTANCE__K", "K=5", 88, 204, 36, 19, { size: 10.2, color: C.secondary, autoFit: "none" });
  addText(slide, "FIG2_SHARED_DISTANCE__ROW_NORM", "RN", 80, 259, 101, 20, { size: 11, color: C.secondary, autoFit: "none" });
  addText(slide, "FIG2_SHARED_DISTANCE__LABEL", "Distance\nGraph", 116, 280, 84, 36, { size: 12.5, autoFit: "none" });

  addText(slide, "FIG2_MACRO_STREAM__LANE", "Macro", 280, 81, 91, 18, { size: 13, color: C.teal, align: "left" });
  const macroInput = drawRibbon(slide, "FIG2_MACRO_STREAM__INPUT", 212, 111, 52, 39, C.teal, C.tealLight, [0.67,0.54,0.40,0.31,0.28,0.35,0.48,0.57,0.50]);
  addText(slide, "FIG2_MACRO_STREAM__INPUT_LABEL", "Train-only\nPatv_raw", 204, 78, 72, 32, { size: 11, autoFit: "none" });
  const macroTrend = drawTrend(slide, "FIG2_MACRO_STREAM__TREND", 279, 109, 40, 42, C.teal);
  const macroPearson = drawPairedCurves(slide, "FIG2_MACRO_STREAM__PEARSON", 337, 108, 49, 43, C.teal, "ρ");
  addText(slide, "FIG2_MACRO_STREAM__PEARSON_LABEL", "Trend Pearson\nCorrelation\nGraph", 296, 153, 100, 54, { size: 12.5, autoFit: "none" });
  const macroDense = drawDenseGraph(slide, "FIG2_MACRO_STREAM__DENSE", 405, 111, 25, 37, C.teal);
  const macroGraph = drawGraph(slide, "FIG2_MACRO_STREAM__GRAPH", 439, 108, 31, 42, C.teal, { nodeRadius: 2.0, widths: [1.2,0.7,1.1,1.35,0.8,1.3,0.9] });
  connect(slide, macroDense, macroGraph, { name: "FIG2_MACRO_STREAM__TOPK", fromSide: "right", toSide: "left", color: C.teal, width: 1.5 });
  addText(slide, "FIG2_MACRO_STREAM__K", "K=10", 409, 90, 48, 19, { size: 10.2, color: C.secondary, autoFit: "none" });
  addText(slide, "FIG2_MACRO_STREAM__GRAPH_LABEL", "Trend\nGraph", 398, 153, 69, 36, { size: 12.5, autoFit: "none" });
  addText(slide, "FIG2_MACRO_STREAM__RN", "RN", 365, 211, 90, 20, { size: 11, color: C.secondary, autoFit: "none" });

  const macroFusionAnchor = addShape(slide, "FIG2_MACRO_FUSION", "rect", 476, 91, 80, 100, "none", noLine());
  const macroSigma = addShape(slide, "FIG2_MACRO_FUSION__SIGMA", "ellipse", 478, 115, 28, 28, C.white, { style: "solid", fill: C.teal, width: 1.15 });
  setText(macroSigma, "Σ", { size: 13, color: C.teal });
  const macroPrior = drawGraph(slide, "FIG2_MACRO_FUSION__PRIOR", 516, 108, 37, 44, C.teal, { nodeRadius: 2.25, widths: [1.7,0.9,1.4,1.9,1.1,1.5,1.2] });
  connect(slide, macroInput, macroTrend, { name: "FIG2_MACRO_STREAM__A1", fromSide: "right", toSide: "left", color: C.teal, width: 1.6 });
  connect(slide, macroTrend, macroPearson, { name: "FIG2_MACRO_STREAM__A2", fromSide: "right", toSide: "left", color: C.teal, width: 1.6 });
  connect(slide, macroPearson, macroDense, { name: "FIG2_MACRO_STREAM__A3", fromSide: "right", toSide: "left", color: C.teal, width: 1.6 });
  connect(slide, macroGraph, macroSigma, { name: "FIG2_MACRO_FUSION__SEMANTIC", fromSide: "right", toSide: "left", color: C.teal, width: 1.65 });
  connect(slide, macroSigma, macroPrior, { name: "FIG2_MACRO_FUSION__OUTPUT", fromSide: "right", toSide: "left", color: C.teal, width: 1.65 });
  addText(slide, "FIG2_MACRO_FUSION__W_SEMANTIC", "0.7", 459, 96, 30, 18, { size: 10.2, color: C.teal, autoFit: "none" });
  addText(slide, "FIG2_MACRO_FUSION__PRIOR_LABEL", "Macro\nPrior", 504, 154, 62, 38, { size: 12.5, autoFit: "none" });

  addText(slide, "FIG2_MICRO_STREAM__LANE", "Micro", 280, 276, 91, 18, { size: 13, color: C.orange, align: "left" });
  const microInput = drawRibbon(slide, "FIG2_MICRO_STREAM__INPUT", 212, 305, 52, 39, C.orange, C.orangeLight, [0.52,0.20,0.74,0.33,0.68,0.24,0.79,0.42,0.57]);
  addText(slide, "FIG2_MICRO_STREAM__INPUT_LABEL", "Train-only\nWspd", 204, 273, 72, 32, { size: 11, autoFit: "none" });
  const microDiff = drawDifference(slide, "FIG2_MICRO_STREAM__DIFF", 279, 304, 40, 40, C.orange);
  const microCos = drawPairedCurves(slide, "FIG2_MICRO_STREAM__COSINE", 337, 302, 49, 43, C.orange, "cos");
  addText(slide, "FIG2_MICRO_STREAM__COSINE_LABEL", "Cosine\nSimilarity on\n1st-Order\nDifference", 270, 347, 122, 68, { size: 12.5, autoFit: "none" });
  const microDense = drawDenseGraph(slide, "FIG2_MICRO_STREAM__DENSE", 405, 305, 25, 37, C.orange);
  const microGraph = drawGraph(slide, "FIG2_MICRO_STREAM__GRAPH", 439, 302, 31, 42, C.orange, { nodeRadius: 2.0, widths: [1.1,0.8,1.3,0.9,1.2,1.4,0.75] });
  connect(slide, microDense, microGraph, { name: "FIG2_MICRO_STREAM__TOPK", fromSide: "right", toSide: "left", color: C.orange, width: 1.5 });
  addText(slide, "FIG2_MICRO_STREAM__K", "K=5", 409, 284, 48, 19, { size: 10.2, color: C.secondary, autoFit: "none" });
  addText(slide, "FIG2_MICRO_STREAM__GRAPH_LABEL", "Local-\nChange\nGraph", 393, 347, 74, 54, { size: 12.5, autoFit: "none" });
  addText(slide, "FIG2_MICRO_STREAM__RN", "RN", 397, 403, 80, 20, { size: 11, color: C.secondary, autoFit: "none" });

  const microFusionAnchor = addShape(slide, "FIG2_MICRO_FUSION", "rect", 476, 285, 80, 100, "none", noLine());
  const microSigma = addShape(slide, "FIG2_MICRO_FUSION__SIGMA", "ellipse", 478, 309, 28, 28, C.white, { style: "solid", fill: C.orange, width: 1.15 });
  setText(microSigma, "Σ", { size: 13, color: C.orange });
  const microPrior = drawGraph(slide, "FIG2_MICRO_FUSION__PRIOR", 516, 302, 37, 44, C.orange, { nodeRadius: 2.25, widths: [1.1,0.8,1.55,1.35,0.95,1.7,1.25] });
  connect(slide, microInput, microDiff, { name: "FIG2_MICRO_STREAM__A1", fromSide: "right", toSide: "left", color: C.orange, width: 1.6 });
  connect(slide, microDiff, microCos, { name: "FIG2_MICRO_STREAM__A2", fromSide: "right", toSide: "left", color: C.orange, width: 1.6 });
  connect(slide, microCos, microDense, { name: "FIG2_MICRO_STREAM__A3", fromSide: "right", toSide: "left", color: C.orange, width: 1.6 });
  connect(slide, microGraph, microSigma, { name: "FIG2_MICRO_FUSION__SEMANTIC", fromSide: "right", toSide: "left", color: C.orange, width: 1.65 });
  connect(slide, microSigma, microPrior, { name: "FIG2_MICRO_FUSION__OUTPUT", fromSide: "right", toSide: "left", color: C.orange, width: 1.65 });
  addText(slide, "FIG2_MICRO_FUSION__W_SEMANTIC", "0.3", 459, 290, 30, 18, { size: 10.2, color: C.orange, autoFit: "none" });
  addText(slide, "FIG2_MICRO_FUSION__PRIOR_LABEL", "Micro\nPrior", 504, 348, 62, 38, { size: 12.5, autoFit: "none" });

  const branchJunction = addShape(slide, "FIG2_SHARED_DISTANCE__JUNCTION", "ellipse", 461, 240, 8, 8, C.distance, noLine());
  connect(slide, distanceGraph, branchJunction, { name: "FIG2_SHARED_DISTANCE__BACKBONE", fromSide: "right", toSide: "left", color: C.distance, width: 1.35, arrow: false });
  connect(slide, branchJunction, macroSigma, { name: "FIG2_SHARED_DISTANCE__TO_MACRO", kind: "elbow", fromSide: "top", toSide: "bottom", color: C.distance, width: 1.3 });
  connect(slide, branchJunction, microSigma, { name: "FIG2_SHARED_DISTANCE__TO_MICRO", kind: "elbow", fromSide: "bottom", toSide: "top", color: C.distance, width: 1.3 });
  addText(slide, "FIG2_SHARED_DISTANCE__W_MACRO", "0.3", 456, 196, 25, 19, { size: 10.2, color: C.distance, autoFit: "none" });
  addText(slide, "FIG2_SHARED_DISTANCE__W_MICRO", "0.7", 451, 265, 25, 19, { size: 10.2, color: C.distance, autoFit: "none" });

  const adaptivePrior = drawGraph(slide, "FIG2_ADAPTIVE_PRIOR", 548, 220, 48, 52, C.distance, { edges: PRIOR_EDGES, widths: PRIOR_EDGES.map(() => 1.15), nodeRadius: 2.6 });
  connect(slide, macroPrior, adaptivePrior, { name: "FIG2_ADAPTIVE_PRIOR__FROM_MACRO", kind: "curved", fromSide: "right", toSide: "left", color: C.teal, width: 1.55 });
  connect(slide, microPrior, adaptivePrior, { name: "FIG2_ADAPTIVE_PRIOR__FROM_MICRO", kind: "curved", fromSide: "right", toSide: "left", color: C.orange, width: 1.55 });
  addText(slide, "FIG2_ADAPTIVE_PRIOR__LABEL", "Prior", 535, 276, 68, 20, { size: 12.5 });

  const adaptiveEmbed = drawEmbeddingStrips(slide, "FIG2_ADAPTIVE_EMBED", 610, 220, 36, 52);
  addText(slide, "FIG2_ADAPTIVE_EMBED__DE", "d_e=10", 597, 198, 60, 20, { size: 10.2, color: C.secondary, autoFit: "none" });
  addText(slide, "FIG2_ADAPTIVE_EMBED__LABEL", "Learnable\nNode\nEmbeddings", 580, 326, 96, 54, { size: 12.5, autoFit: "none" });

  const adaptiveScores = drawGraph(slide, "FIG2_ADAPTIVE_SCORES", 659, 220, 48, 52, C.purple, {
    edges: SCORE_EDGES,
    unsupported: [7, 8],
    widths: [1.1,1.7,1.2,1.9,1.4,1.65,1.0,0.85,0.85],
    nodeRadius: 2.6,
  });
  addText(slide, "FIG2_ADAPTIVE_SCORES__LABEL", "Edge\nImportance", 635, 276, 96, 38, { size: 12.5, autoFit: "none" });

  const supportStencil = drawGraph(slide, "FIG2_ADAPTIVE_MASK__STENCIL", 720, 166, 43, 36, C.distance, { edges: PRIOR_EDGES, widths: PRIOR_EDGES.map(() => 0.8), nodeRadius: 2.1 });
  addText(slide, "FIG2_ADAPTIVE_MASK__STENCIL_LABEL", "Support\nMask", 704, 125, 74, 38, { size: 12.5, autoFit: "none" });
  const adaptiveMask = drawGraph(slide, "FIG2_ADAPTIVE_MASK", 718, 220, 46, 52, C.purple, {
    edges: SCORE_EDGES,
    unsupported: [7, 8],
    crossUnsupported: true,
    widths: [1.1,1.7,1.2,1.9,1.4,1.65,1.0,0.7,0.7],
    nodeRadius: 2.6,
  });
  connect(slide, supportStencil, adaptiveMask, { name: "FIG2_ADAPTIVE_MASK__APPLY", fromSide: "bottom", toSide: "top", color: C.distance, width: 1.25 });
  addText(slide, "FIG2_ADAPTIVE_MASK__LABEL", "Remove\nUnsupported\nEdges", 695, 326, 96, 54, { size: 12.5, autoFit: "none" });

  addText(slide, "FIG2_ADAPTIVE_MASK__ROW_NORM_LABEL", "Row\nNormalization", 740, 180, 114, 38, { size: 12.5, autoFit: "none" });

  const effectiveGraph = drawGraph(slide, "FIG2_EFFECTIVE_GRAPH", 807, 216, 45, 58, C.purple, {
    edges: PRIOR_EDGES,
    widths: [0.75,2.15,1.15,2.45,1.45,1.9,0.95],
    nodeRadius: 2.65,
  });
  addText(slide, "FIG2_EFFECTIVE_GRAPH__LABEL", "Effective\nGraph", 790, 278, 72, 34, { size: 12.5 });
  connect(slide, adaptivePrior, adaptiveEmbed, { name: "FIG2_ADAPTIVE__A1", fromSide: "right", toSide: "left", color: C.purple, width: 1.65 });
  connect(slide, adaptiveEmbed, adaptiveScores, { name: "FIG2_ADAPTIVE__A2", fromSide: "right", toSide: "left", color: C.purple, width: 1.65 });
  connect(slide, adaptiveScores, adaptiveMask, { name: "FIG2_ADAPTIVE__A3", fromSide: "right", toSide: "left", color: C.purple, width: 1.65 });
  connect(slide, adaptiveMask, effectiveGraph, { name: "FIG2_ADAPTIVE__A4_ROW_NORM", fromSide: "right", toSide: "left", color: C.purple, width: 1.65 });

  drawSmallLegendStrip(slide, "FIG2_ADAPTIVE_EMBED__MACRO_LEGEND", 565, 383, C.teal, C.tealLight);
  drawSmallLegendStrip(slide, "FIG2_ADAPTIVE_EMBED__MICRO_LEGEND", 565, 407, C.orange, C.orangeLight);
  addText(slide, "FIG2_ADAPTIVE_EMBED__MACRO_TEXT", "E_macro", 600, 380, 90, 19, { size: 11.5, color: C.teal, align: "left", wrap: "none" });
  addText(slide, "FIG2_ADAPTIVE_EMBED__MICRO_TEXT", "E_micro", 600, 404, 90, 19, { size: 11.5, color: C.orange, align: "left", wrap: "none" });
  addText(slide, "FIG2_ADAPTIVE_EMBED__INDEPENDENT", "independent", 735, 392, 106, 19, { size: 11, color: C.secondary, align: "left" });
  addText(slide, "FIG2_ADAPTIVE__BOTTOM_NOTE", "Prior topology is fixed; learning only\nadjusts supported edge strengths.", 538, 425, 305, 38, { size: 11, autoFit: "none" });
}

await fs.mkdir(QA_DIR, { recursive: true });
const presentation = await PresentationFile.importPptx(await FileBlob.load(SOURCE));
if (presentation.slides.items.length !== 2) throw new Error(`Expected 2 source slides, got ${presentation.slides.items.length}`);
presentation.slides.items[1].delete();
const slide = presentation.slides.items[0];
slide.setViewportSize(W, H);
slide.shapes.deleteAll();
slide.speakerNotes?.setText?.("");
buildFigure2(slide);

const frameMap = {
  outputSlides: [{
    outputSlide: 1,
    sourceSlide: 1,
    narrativeRole: "Figure 2 technical mechanism",
    reuseMode: "duplicate-slide",
    editTargets: [
      { action: "rewrite-and-reposition", sourceElementId: "FIGURE_TITLE" },
      { action: "add", newPrimitiveAllowed: true, mustNotOverlapInherited: true, reason: "User explicitly requested a full Figure 2 redesign while preserving the source master/theme and frozen scientific logic.", zone: { left: 0, top: 0, width: W, height: H } },
    ],
  }],
  omittedSourceSlides: [{ sourceSlide: 2, reason: "Figure 3 is frozen and explicitly omitted from the one-slide deliverable without editing." }],
};
await fs.writeFile(path.join(QA_DIR, "template-frame-map.json"), `${JSON.stringify(frameMap, null, 2)}\n`, "utf8");
await fs.writeFile(path.join(QA_DIR, "template-audit.txt"), "Source deck: 2 slides, 960 × 537.67 px. Output duplicates source slide 1 as the only deliverable slide, retains its master/theme, changes the viewport to 864 × 480 px (9.0 × 5.0 in at 96 dpi), and replaces slide-local Figure 2 objects under the user's explicit redesign instruction. Source slide 2 (Figure 3) is omitted unchanged.\n", "utf8");
await fs.writeFile(path.join(QA_DIR, "deviation-log.txt"), "Output slide 1: viewport changed from 960 × 537.67 px to 864 × 480 px. The former three boxed A/B/C panels were replaced by a shared-spatial region, paired semantic ribbons, and a horizontal prior-constrained graph-transformation strip. Scientific mechanisms, weights, K values, row normalization, d_e, and topology constraint are unchanged.\n", "utf8");
await fs.writeFile(path.join(QA_DIR, "source-notes.txt"), "No external sources or assets used. Scientific terminology and parameter values are inherited from the user-provided ST-MGPrompt deck and task specification.\n", "utf8");

await saveBlob(path.join(QA_DIR, "slide-1.png"), await presentation.export({ slide, format: "png", scale: 2 }));
await saveBlob(path.join(QA_DIR, "slide-1.layout.json"), await slide.export({ format: "layout" }));
await saveBlob(path.join(QA_DIR, "montage.webp"), await presentation.export({ format: "webp", montage: true, scale: 1 }));
const inspected = await presentation.inspect({ kind: "deck,slide,textbox,shape,layout", include: "id,slide,name,text,textPreview,textChars,textLines,bbox,bboxUnit", maxChars: 250000 });
await fs.writeFile(path.join(QA_DIR, "final-inspect.ndjson"), inspected.ndjson ?? "", "utf8");
const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save(OUT);
console.log(OUT);
