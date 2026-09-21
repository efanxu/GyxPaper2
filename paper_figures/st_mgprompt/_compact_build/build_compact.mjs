import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const SOURCE = "D:/PaperProject/GyxPaper2/paper_figures/st_mgprompt/ST-MGPrompt_Final_Figures.pptx";
const OUT = "D:/PaperProject/GyxPaper2/paper_figures/st_mgprompt/ST-MGPrompt_Fig2_Fig3_Compact_Final.pptx";
const QA_DIR = "D:/PaperProject/GyxPaper2/paper_figures/st_mgprompt/_compact_build/final_qa";
const W = 960;
const H = 537.6;
const FONT = "Times New Roman";

const C = {
  ink: "#17232C",
  secondary: "#5D7280",
  line: "#6F8795",
  blueBg: "#F4F8FA",
  blueLine: "#9AB8C8",
  teal: "#4F9A96",
  tealLight: "#E9F5F3",
  tealPale: "#F2FAF8",
  orange: "#E69A54",
  orangeLight: "#FFF1E3",
  purple: "#8367B2",
  purpleLight: "#F3EEF9",
  purplePale: "#FBF9FE",
  lavenderLine: "#B7A3D4",
  grayFill: "#EDF2F4",
  grayLine: "#91A3AD",
  unsupported: "#B8BEC5",
  red: "#B76F68",
  white: "#FFFFFF",
};

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
    fontSizePt: opts.size ?? 11,
    bold: opts.bold ?? false,
    italic: opts.italic ?? false,
    color: opts.color ?? C.ink,
    alignment: opts.align ?? "center",
    verticalAlignment: opts.valign ?? "middle",
    autoFit: "shrinkText",
    wrap: opts.wrap ?? "square",
    insets: opts.insets ?? { left: 2, right: 2, top: 1, bottom: 1 },
  };
  return shape;
}

function addText(slide, name, text, x, y, w, h, opts = {}) {
  return setText(addShape(slide, name, "textbox", x, y, w, h, "none", noLine()), text, opts);
}

function addBox(slide, name, text, x, y, w, h, opts = {}) {
  const s = addShape(
    slide,
    name,
    opts.geometry ?? "roundRect",
    x,
    y,
    w,
    h,
    opts.fill ?? C.white,
    { style: opts.dash ? "dashed" : "solid", fill: opts.line ?? C.line, width: opts.lineWidth ?? 1.1 },
    opts.radius ?? 7,
  );
  return setText(s, text, opts);
}

function addLine(slide, name, x1, y1, x2, y2, color, width = 1.2, dashed = false) {
  const left = Math.min(x1, x2);
  const top = Math.min(y1, y2);
  const lineWidth = Math.max(Math.abs(x2 - x1), 0.1);
  const lineHeight = Math.max(Math.abs(y2 - y1), 0.1);
  return slide.shapes.add({
    geometry: "line",
    name,
    position: {
      left,
      top,
      width: lineWidth,
      height: lineHeight,
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
    line: { style: opts.dashed ? "dashed" : "solid", fill: opts.color ?? C.line, width: opts.width ?? 1.5 },
    head: { type: "none" },
    tail: { type: opts.head ?? "triangle", width: "sm", length: "sm" },
  });
  connector.bringToFront();
  return connector;
}

function addNode(slide, name, x, y, color, r = 4.2) {
  return addShape(slide, name, "ellipse", x - r, y - r, r * 2, r * 2, C.white, { style: "solid", fill: color, width: 1.25 });
}

function drawGraph(slide, prefix, x, y, w, h, color, opts = {}) {
  const anchor = addShape(slide, prefix, "rect", x, y, w, h, "none", noLine());
  const pts = [
    [0.10, 0.52], [0.28, 0.18], [0.53, 0.25], [0.82, 0.50], [0.63, 0.82], [0.33, 0.72],
  ].map(([px, py]) => [x + px * w, y + py * h]);
  const edges = opts.edges ?? [[0,1],[0,5],[1,2],[1,5],[2,3],[2,5],[3,4],[4,5],[2,4]];
  const widths = opts.widths ?? edges.map((_, i) => 1 + (i % 4) * 0.42);
  for (let i = 0; i < edges.length; i += 1) {
    const [a,b] = edges[i];
    const unsupported = (opts.unsupported ?? []).includes(i);
    addLine(slide, `${prefix}__EDGE_${i + 1}`, pts[a][0], pts[a][1], pts[b][0], pts[b][1], unsupported ? C.unsupported : color, widths[i] ?? 1.2, unsupported);
  }
  for (let i = 0; i < pts.length; i += 1) addNode(slide, `${prefix}__NODE_${i + 1}`, pts[i][0], pts[i][1], color, opts.nodeRadius ?? 3.8);
  if (opts.crossUnsupported) {
    for (const i of opts.unsupported ?? []) {
      const [a,b] = edges[i];
      const mx = (pts[a][0] + pts[b][0]) / 2;
      const my = (pts[a][1] + pts[b][1]) / 2;
      addText(slide, `${prefix}__CROSS_${i}`, "×", mx - 7, my - 8, 14, 16, { size: 13, color: C.red, bold: false });
    }
  }
  return anchor;
}

function drawWaveRibbon(slide, prefix, x, y, w, h, color, fill, kind = "fine", recent = false) {
  const anchor = addShape(slide, prefix, "roundRect", x, y, w, h, fill, { style: "solid", fill: color, width: 1.05 }, 8);
  const fine = [0.56,0.24,0.72,0.38,0.66,0.18,0.80,0.42,0.64,0.31,0.75,0.47,0.20,0.60,0.36,0.70,0.28,0.53];
  const coarse = [0.63,0.54,0.42,0.31,0.28,0.36,0.50,0.59,0.56,0.44,0.32,0.27,0.34,0.48,0.58,0.52];
  const vals = kind === "fine" ? fine : coarse;
  const px0 = x + 8;
  const py0 = y + 4;
  const iw = w - 16;
  const ih = h - 8;
  for (let i = 0; i < vals.length - 1; i += 1) {
    const x1 = px0 + (i / (vals.length - 1)) * iw;
    const x2 = px0 + ((i + 1) / (vals.length - 1)) * iw;
    const y1 = py0 + vals[i] * ih;
    const y2 = py0 + vals[i + 1] * ih;
    addLine(slide, `${prefix}__WAVE_${i + 1}`, x1, y1, x2, y2, color, kind === "fine" ? 1.7 : 2.0, false);
  }
  if (recent) {
    const rx = x + w * 0.73;
    addShape(slide, `${prefix}__RECENT_SHADE`, "roundRect", rx, y + 2, w * 0.25, h - 4, "#F9D8B7", { style: "solid", fill: C.orange, width: 0.9 }, 5);
    for (let i = Math.floor((vals.length - 1) * 0.73); i < vals.length - 1; i += 1) {
      const x1 = px0 + (i / (vals.length - 1)) * iw;
      const x2 = px0 + ((i + 1) / (vals.length - 1)) * iw;
      const y1 = py0 + vals[i] * ih;
      const y2 = py0 + vals[i + 1] * ih;
      addLine(slide, `${prefix}__RECENT_WAVE_${i + 1}`, x1, y1, x2, y2, C.orange, 2.0, false);
    }
  }
  return anchor;
}

function drawEmbeddingBars(slide, prefix, x, y, color, fill) {
  const anchor = addShape(slide, prefix, "rect", x, y, 52, 66, "none", noLine());
  const groupColors = [color, C.orange];
  const groupFills = [fill, C.orangeLight];
  for (let g = 0; g < 2; g += 1) {
    for (let b = 0; b < 2; b += 1) {
      const bx = x + 2 + g * 26 + b * 10;
      addShape(slide, `${prefix}__G${g + 1}_BAR_${b + 1}`, "roundRect", bx, y + 6, 8, 44, groupFills[g], { style: "solid", fill: groupColors[g], width: 0.9 }, 2);
      for (let s = 1; s <= 4; s += 1) {
        addLine(slide, `${prefix}__G${g + 1}_STRIPE_${b + 1}_${s}`, bx + 1, y + 6 + s * 8, bx + 7, y + 6 + s * 8, groupColors[g], 0.65);
      }
    }
  }
  addText(slide, `${prefix}__MACRO_TAG`, "M", x + 5, y + 51, 16, 14, { size: 10, color });
  addText(slide, `${prefix}__MICRO_TAG`, "μ", x + 31, y + 51, 16, 14, { size: 10, color: C.orange });
  return anchor;
}

function drawUpdatedRibbon(slide, prefix, x, y, w, h, color, fill, kind, label, state) {
  const anchor = addShape(slide, prefix, "roundRect", x, y, w, h, fill, { style: "solid", fill: color, width: 1.05 }, 8);
  addText(slide, `${prefix}__LABEL`, label, x + 8, y + 3, 122, 17, { size: 11.5, color, align: "left" });
  addText(slide, `${prefix}__STATE`, state, x + w - 43, y + 3, 34, 17, { size: 11, color, bold: true, align: "right" });
  const fine = [0.56,0.24,0.72,0.38,0.66,0.18,0.80,0.42,0.64,0.31,0.75,0.47,0.20,0.60,0.36,0.70,0.28,0.53];
  const coarse = [0.63,0.54,0.42,0.31,0.28,0.36,0.50,0.59,0.56,0.44,0.32,0.27,0.34,0.48,0.58,0.52];
  const vals = kind === "fine" ? fine : coarse;
  const px0 = x + 10;
  const py0 = y + 22;
  const iw = w - 20;
  const ih = h - 28;
  for (let i = 0; i < vals.length - 1; i += 1) {
    const x1 = px0 + (i / (vals.length - 1)) * iw;
    const x2 = px0 + ((i + 1) / (vals.length - 1)) * iw;
    const y1 = py0 + vals[i] * ih;
    const y2 = py0 + vals[i + 1] * ih;
    addLine(slide, `${prefix}__FEATURE_${i + 1}`, x1, y1, x2, y2, color, kind === "fine" ? 1.6 : 1.9);
  }
  return anchor;
}

function addPromptCapsules(slide, prefix, x, y, w, h) {
  const anchor = addShape(slide, prefix, "rect", x, y, w, h, "none", noLine());
  const gap = 8;
  const cw = (w - 3 * gap) / 4;
  for (let i = 0; i < 4; i += 1) {
    addBox(slide, `${prefix}__P${i + 1}`, `P${i + 1}`, x + i * (cw + gap), y, cw, h, {
      fill: C.purpleLight, line: C.purple, size: 11.5, radius: 11,
    });
  }
  return anchor;
}

function addPanel(slide, name, x, y, w, h, fill, line) {
  const s = addShape(slide, name, "roundRect", x, y, w, h, fill, { style: "solid", fill: line, width: 1.2 }, 10);
  s.sendToBack();
  return s;
}

function buildFigure2(slide) {
  slide.background.fill = C.white;
  addText(slide, "FIGURE_TITLE", "Dual Semantic Spatial Graph Construction", 14, 6, 932, 38, { size: 25, bold: true, align: "left" });

  addPanel(slide, "FIG2_PANEL_A_DISTANCE", 12, 51, 195, 474, C.blueBg, C.blueLine);
  addPanel(slide, "FIG2_PANEL_B", 215, 51, 350, 474, C.tealPale, "#90C2BD");
  addPanel(slide, "FIG2_PANEL_C", 573, 51, 375, 474, C.purplePale, C.lavenderLine);

  addText(slide, "FIG2_PANEL_A__TITLE", "(A) Gaussian Kernel\nDistance Graph", 18, 56, 183, 42, { size: 16, bold: true });
  const coords = addShape(slide, "FIG2_PANEL_A_DISTANCE__COORDINATES", "rect", 44, 106, 130, 49, "none", noLine());
  addText(slide, "FIG2_PANEL_A_DISTANCE__COORD_LABEL", "Turbine Coordinates", 30, 99, 158, 18, { size: 11.5 });
  [[58,133],[83,117],[108,142],[131,119],[158,137],[139,151]].forEach(([x,y],i) => addNode(slide, `FIG2_PANEL_A_DISTANCE__COORD_${i+1}`, x, y, C.line, 3.7));
  const gauss = addShape(slide, "FIG2_PANEL_A_DISTANCE__GAUSSIAN", "rect", 37, 178, 145, 55, "none", noLine());
  addText(slide, "FIG2_PANEL_A_DISTANCE__GAUSSIAN_LABEL", "Gaussian Kernel", 116, 157, 75, 20, { size: 11.5, align: "right" });
  const gp = [[55,213],[91,189],[125,214],[163,188]];
  gp.forEach(([x,y],i)=>addNode(slide,`FIG2_PANEL_A_DISTANCE__G_NODE_${i+1}`,x,y,C.line,3.6));
  addLine(slide,"FIG2_PANEL_A_DISTANCE__G_NEAR",55,213,91,189,C.line,2.6);
  addLine(slide,"FIG2_PANEL_A_DISTANCE__G_MID",91,189,125,214,C.line,1.7);
  addLine(slide,"FIG2_PANEL_A_DISTANCE__G_FAR",125,214,163,188,C.line,0.8);
  const topk = addBox(slide, "FIG2_PANEL_A_DISTANCE__TOPK", "Top-K Sparsification\n& Row Normalization", 35, 265, 148, 52, { fill: C.white, line: C.line, size: 11.5, radius: 7 });
  addText(slide,"FIG2_PANEL_A_DISTANCE__K", "K = 5", 132, 325, 48, 18, { size: 10.5, color: C.secondary, align: "right" });
  const dist = drawGraph(slide, "FIG2_PANEL_A_DISTANCE__GRAPH", 50, 374, 120, 72, C.line, { widths:[2.3,1.0,1.8,2.6,1.2,1.8,2.4,0.9,1.5] });
  addText(slide, "FIG2_PANEL_A_DISTANCE__DIST_LABEL", "Distance Graph", 48, 452, 124, 24, { size: 12.5 });
  connect(slide, coords, gauss, { fromSide:"bottom", toSide:"top", color:C.line });
  connect(slide, gauss, topk, { fromSide:"bottom", toSide:"top", color:C.line });
  connect(slide, topk, dist, { fromSide:"bottom", toSide:"top", color:C.line });

  addText(slide, "FIG2_PANEL_B__TITLE", "(B) Dual Semantic Prior Graph Construction", 222, 56, 336, 27, { size: 16, bold: true });
  addText(slide, "FIG2_PANEL_B_MACRO__LANE", "Macro prior", 224, 87, 80, 19, { size: 12.5, color: C.teal, align: "left" });
  addText(slide, "FIG2_PANEL_B_MICRO__LANE", "Micro prior", 224, 287, 80, 19, { size: 12.5, color: C.orange, align: "left" });
  addLine(slide,"FIG2_PANEL_B__DIVIDER",228,278,552,278,"#B9D7D3",0.8);

  const macroInput = drawWaveRibbon(slide,"FIG2_PANEL_B_MACRO",229,119,62,35,C.teal,C.tealLight,"coarse");
  addText(slide,"FIG2_PANEL_B_MACRO__INPUT","Patv_raw",229,107,62,16,{size:10.5});
  const macroExtract = addBox(slide,"FIG2_PANEL_B_MACRO__EXTRACT","Trend\nextraction",298,112,57,49,{fill:C.white,line:C.teal,size:10,wrap:"none",radius:6});
  const macroGraph = drawGraph(slide,"FIG2_PANEL_B_MACRO__GRAPH",362,112,54,50,C.teal,{nodeRadius:2.7,widths:[1.3,0.8,1.2,1.5,0.9,1.0,1.4,0.7,1.1]});
  addText(slide,"FIG2_PANEL_B_MACRO__GRAPH_LABEL","Pearson graph",356,164,66,16,{size:10.5,color:C.teal});
  addText(slide,"FIG2_PANEL_B_MACRO__K","K = 10",358,181,50,16,{size:10,color:C.secondary});
  const macroSigma = addBox(slide,"FIG2_PANEL_B_MACRO__SIGMA","Σ",458,119,32,32,{geometry:"ellipse",fill:C.white,line:C.teal,size:13,radius:16});
  addText(slide,"FIG2_PANEL_B_MACRO__FUSION_LABEL","Prior Graph\nFusion",452,153,45,28,{size:10});
  const macroPrior = drawGraph(slide,"FIG2_PANEL_B_MACRO__PRIOR",506,111,48,45,C.teal,{nodeRadius:2.5,widths:[1.4,1.0,1.2,1.5,1.1,1.4,1.6,0.9,1.2]});
  addText(slide,"FIG2_PANEL_B_MACRO__PRIOR_LABEL","Macro Prior\nGraph",498,159,64,31,{size:11.5});
  connect(slide,macroInput,macroExtract,{fromSide:"right",toSide:"left",color:C.teal});
  connect(slide,macroExtract,macroGraph,{fromSide:"right",toSide:"left",color:C.teal});
  connect(slide,macroGraph,macroSigma,{fromSide:"right",toSide:"left",color:C.teal});
  connect(slide,macroSigma,macroPrior,{fromSide:"right",toSide:"left",color:C.teal});
  addText(slide,"FIG2_PANEL_B_MACRO__W07","0.7",423,106,30,15,{size:10,color:C.teal});

  const microInput = drawWaveRibbon(slide,"FIG2_PANEL_B_MICRO",229,323,62,35,C.orange,C.orangeLight,"fine");
  addText(slide,"FIG2_PANEL_B_MICRO__INPUT","Wspd",229,307,62,16,{size:10.5});
  const microExtract = addBox(slide,"FIG2_PANEL_B_MICRO__EXTRACT","1st\nDifference",298,316,57,49,{fill:C.white,line:C.orange,size:10,wrap:"none",radius:6});
  const microGraph = drawGraph(slide,"FIG2_PANEL_B_MICRO__GRAPH",362,316,54,50,C.orange,{nodeRadius:2.7,widths:[1.2,0.9,1.4,1.0,0.8,1.3,1.5,0.9,1.1]});
  addText(slide,"FIG2_PANEL_B_MICRO__GRAPH_LABEL","Cosine graph",356,368,66,16,{size:10.5,color:C.orange});
  addText(slide,"FIG2_PANEL_B_MICRO__K","K = 5",380,385,46,16,{size:10,color:C.secondary});
  const microSigma = addBox(slide,"FIG2_PANEL_B_MICRO__SIGMA","Σ",458,323,32,32,{geometry:"ellipse",fill:C.white,line:C.orange,size:13,radius:16});
  addText(slide,"FIG2_PANEL_B_MICRO__FUSION_LABEL","Prior Graph\nFusion",452,357,45,28,{size:10});
  const microPrior = drawGraph(slide,"FIG2_PANEL_B_MICRO__PRIOR",506,315,48,45,C.orange,{nodeRadius:2.5,widths:[1.1,0.8,1.4,1.3,0.9,1.2,1.5,1.0,1.3]});
  addText(slide,"FIG2_PANEL_B_MICRO__PRIOR_LABEL","Micro Prior\nGraph",498,363,64,31,{size:11.5});
  connect(slide,microInput,microExtract,{fromSide:"right",toSide:"left",color:C.orange});
  connect(slide,microExtract,microGraph,{fromSide:"right",toSide:"left",color:C.orange});
  connect(slide,microGraph,microSigma,{fromSide:"right",toSide:"left",color:C.orange});
  connect(slide,microSigma,microPrior,{fromSide:"right",toSide:"left",color:C.orange});
  addText(slide,"FIG2_PANEL_B_MICRO__W03","0.3",423,310,30,15,{size:10,color:C.orange});

  const sharedDistance = drawGraph(slide,"FIG2_PANEL_B__SHARED_DISTANCE",376,222,54,43,C.line,{nodeRadius:2.5,widths:[1.3,0.8,1.2,1.5,1.0,1.1,1.5,0.8,1.2]});
  addText(slide,"FIG2_PANEL_B__SHARED_DISTANCE_LABEL","Distance Graph",358,203,90,18,{size:10.5,color:C.secondary});
  connect(slide,sharedDistance,macroSigma,{kind:"elbow",fromSide:"right",toSide:"left",color:C.line,width:1.2});
  connect(slide,sharedDistance,microSigma,{kind:"elbow",fromSide:"right",toSide:"left",color:C.line,width:1.2});
  addText(slide,"FIG2_PANEL_B__DIST_W03","0.3",414,184,28,15,{size:10,color:C.secondary});
  addText(slide,"FIG2_PANEL_B__DIST_W07","0.7",414,271,28,15,{size:10,color:C.secondary});
  addText(slide,"FIG2_PANEL_B__SUMMARY_MACRO","Patv trend + distance → Macro Prior",233,439,318,19,{size:10.5,color:C.teal});
  addText(slide,"FIG2_PANEL_B__SUMMARY_MICRO","ΔWspd + distance → Micro Prior",233,464,318,19,{size:10.5,color:C.orange});

  addText(slide,"FIG2_PANEL_C__TITLE","(C) Prior-Constrained Adaptive Edge Re-weighting",580,56,361,27,{size:15.5,bold:true});
  addText(slide,"FIG2_PANEL_C__SUBTITLE","Learn edge importance without creating unsupported connections",590,84,341,18,{size:10.5,color:C.secondary});
  addText(slide,"FIG2_PANEL_C_PRIOR__STEP","①",596,111,36,26,{size:16,color:C.purple});
  addText(slide,"FIG2_PANEL_C_PRIOR__STEP_TITLE","Prior defines\nallowed edges",581,136,66,37,{size:10.5});
  addText(slide,"FIG2_PANEL_C_EMBED__STEP","②",665,111,36,26,{size:16,color:C.purple});
  addText(slide,"FIG2_PANEL_C_EMBED__STEP_TITLE","Learn adaptive\nedge importance",649,136,69,37,{size:10.5});
  addText(slide,"FIG2_PANEL_C_MASK__STEP","③",766,111,36,26,{size:16,color:C.purple});
  addText(slide,"FIG2_PANEL_C_MASK__STEP_TITLE","Mask unsupported\nedges",746,136,77,37,{size:10.5});
  addText(slide,"FIG2_PANEL_C_EFFECTIVE__STEP","④",884,111,36,26,{size:16,color:C.purple});
  addText(slide,"FIG2_PANEL_C_EFFECTIVE__STEP_TITLE","Re-weight &\nrow normalize",867,136,71,37,{size:10.5});

  const prior = drawGraph(slide,"FIG2_PANEL_C_PRIOR",586,226,58,58,C.line,{nodeRadius:3.1,widths:[1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1]});
  addText(slide,"FIG2_PANEL_C_PRIOR__LABEL","Allowed edges",582,289,66,20,{size:10.5,color:C.secondary});
  const embed = drawEmbeddingBars(slide,"FIG2_PANEL_C_EMBED",655,218,C.teal,C.tealLight);
  addText(slide,"FIG2_PANEL_C_EMBED__DE","d_e = 10",652,303,52,18,{size:10,color:C.secondary});
  const adaptive = drawGraph(slide,"FIG2_PANEL_C_MASK__ADAPTIVE",713,220,58,58,C.purple,{nodeRadius:3.0,unsupported:[2,6],widths:[1.1,1.8,1.0,1.5,1.2,1.7,1.0,1.4,1.8]});
  addText(slide,"FIG2_PANEL_C_MASK__ADAPTIVE_LABEL","Adaptive scores",707,283,70,18,{size:10.5});
  const mask = addBox(slide,"FIG2_PANEL_C_MASK","Prior\nMask",780,224,44,48,{fill:C.grayFill,line:C.line,size:10.5,radius:8});
  addText(slide,"FIG2_PANEL_C_MASK__X","×",795,273,15,20,{size:14,color:C.red});
  const supported = drawGraph(slide,"FIG2_PANEL_C_MASK__SUPPORTED",831,220,58,58,C.purple,{nodeRadius:3.0,edges:[[0,1],[0,5],[1,2],[1,5],[2,5],[3,4],[4,5]],widths:[1.1,1.8,1.5,1.2,1.7,1.4,1.8]});
  addText(slide,"FIG2_PANEL_C_MASK__SUPPORTED_LABEL","Supported scores",816,283,72,18,{size:10.5});
  const effective = drawGraph(slide,"FIG2_PANEL_C_EFFECTIVE",897,218,46,62,C.purple,{nodeRadius:2.9,edges:[[0,1],[0,5],[1,2],[1,5],[2,5],[3,4],[4,5]],widths:[0.8,2.2,1.2,2.5,1.5,2.0,1.0]});
  addText(slide,"FIG2_PANEL_C_EFFECTIVE__LABEL","Effective\nGraph",892,285,56,31,{size:10.5});
  connect(slide,prior,embed,{fromSide:"right",toSide:"left",color:C.purple,width:1.4});
  connect(slide,embed,adaptive,{fromSide:"right",toSide:"left",color:C.purple,width:1.4});
  connect(slide,adaptive,mask,{fromSide:"right",toSide:"left",color:C.purple,width:1.4});
  connect(slide,mask,supported,{fromSide:"right",toSide:"left",color:C.purple,width:1.4});
  connect(slide,supported,effective,{fromSide:"right",toSide:"left",color:C.purple,width:1.4});
  addText(slide,"FIG2_PANEL_C__ROW_NORM","Row\nnormalize",886,324,58,31,{size:10.5,color:C.secondary});
  addText(slide,"FIG2_PANEL_C__TOPOLOGY","Same topology · learned strengths",615,385,289,20,{size:11.5,color:C.purple});
  addText(slide,"FIG2_PANEL_C__NOTE1","Prior connectivity is fixed; only supported edge strengths are learned.",589,456,343,20,{size:10.5});
  addText(slide,"FIG2_PANEL_C__NOTE2","Macro and Micro use independent embeddings.",619,481,283,18,{size:10,color:C.secondary});
}

function buildFigure3(slide) {
  slide.background.fill = C.purplePale;
  addText(slide,"FIGURE_TITLE","Macro-Trend Prompt and Bidirectional Cross-Fusion",14,6,932,38,{size:25,bold:true,align:"left"});
  addText(slide,"FIG3_PROMPT__HEADING","Macro-Trend Prompt Generation",27,51,360,26,{size:16,bold:true,align:"left"});
  addText(slide,"FIG3_ATTN_MACRO_FINE__HEADING","Bidirectional Cross-Fusion",27,181,330,26,{size:16,bold:true,align:"left"});
  addText(slide,"FIG3_UPDATED_FINE__HEADING","Updated Dual Representations",27,414,290,25,{size:16,bold:true,align:"left"});
  addLine(slide,"FIG3__DIVIDER_TOP",27,176,933,176,"#D8CCE9",0.9);
  addLine(slide,"FIG3__DIVIDER_BOTTOM",27,409,933,409,"#D8CCE9",0.9);

  const coarseTop = drawWaveRibbon(slide,"FIG3_PROMPT__COARSE_RIBBON",63,101,208,42,C.teal,C.tealLight,"coarse");
  addText(slide,"FIG3_PROMPT__COARSE_LABEL","Coarse History",70,79,194,20,{size:12.5,color:C.teal});
  const pool = addBox(slide,"FIG3_PROMPT__POOL","Pool",311,94,62,58,{geometry:"ellipse",fill:C.white,line:C.teal,size:10.5,wrap:"none",radius:29});
  addText(slide,"FIG3_PROMPT__POOL_LABEL","Temporal Weighted Pooling",274,153,136,19,{size:10.5});
  const prompt = addPromptCapsules(slide,"FIG3_PROMPT",430,95,190,43);
  addText(slide,"FIG3_PROMPT__LABEL","Macro-Trend Prompt",430,73,190,20,{size:13,color:C.purple});
  addText(slide,"FIG3_PROMPT__META","4 node-specific tokens / turbine",430,142,190,18,{size:10.5,color:C.secondary});
  connect(slide,coarseTop,pool,{fromSide:"right",toSide:"left",color:C.teal,width:1.6});
  connect(slide,pool,prompt,{fromSide:"right",toSide:"left",color:C.purple,width:1.6});

  const fineHistory = drawWaveRibbon(slide,"FIG3_FINE_HISTORY",81,241,230,48,C.orange,C.orangeLight,"fine",true);
  addText(slide,"FIG3_FINE_HISTORY__LABEL","Fine History",93,216,120,20,{size:12.5,color:C.orange,align:"left"});
  addText(slide,"FIG3_FINE_HISTORY__RECENT_LABEL","Recent Fine",238,216,82,18,{size:10.5,color:C.orange});
  addText(slide,"FIG3_FINE_HISTORY__RECENT_META","24 steps ≈ 4 h",228,291,94,18,{size:10,color:C.secondary});
  const coarseHistory = drawWaveRibbon(slide,"FIG3_COARSE_HISTORY",649,241,230,48,C.teal,C.tealLight,"coarse");
  addText(slide,"FIG3_COARSE_HISTORY__LABEL","Coarse History",714,216,150,20,{size:12.5,color:C.teal});

  const attnFine = addBox(slide,"FIG3_ATTN_MACRO_FINE","Attn",366,286,62,62,{geometry:"ellipse",fill:C.white,line:C.purple,lineWidth:1.4,size:10.5,wrap:"none",radius:31});
  const attnCoarse = addBox(slide,"FIG3_ATTN_FINE_COARSE","Attn",548,286,62,62,{geometry:"ellipse",fill:C.white,line:C.purple,lineWidth:1.4,size:10.5,wrap:"none",radius:31});
  addText(slide,"FIG3_ATTN_MACRO_FINE__Q","Q",344,295,20,17,{size:10.5,color:C.orange,bold:true});
  addText(slide,"FIG3_ATTN_MACRO_FINE__KV","K,V",396,268,34,17,{size:10.5,color:C.purple,bold:true});
  addText(slide,"FIG3_ATTN_FINE_COARSE__Q","Q",611,295,20,17,{size:10.5,color:C.teal,bold:true});
  addText(slide,"FIG3_ATTN_FINE_COARSE__KV","K,V",518,268,34,17,{size:10.5,color:C.orange,bold:true});
  addText(slide,"FIG3_ATTN_MACRO_FINE__LEGEND","Q: Fine · K,V: Macro",284,349,150,16,{size:10,color:C.secondary});
  addText(slide,"FIG3_ATTN_FINE_COARSE__LEGEND","Q: Coarse · K,V: Recent",542,349,166,16,{size:10,color:C.secondary});

  connect(slide,fineHistory,attnFine,{fromSide:"right",toSide:"left",color:C.orange,width:1.7});
  connect(slide,prompt,attnFine,{kind:"elbow",fromSide:"bottom",toSide:"top",color:C.purple,width:1.7});
  connect(slide,coarseHistory,attnCoarse,{fromSide:"left",toSide:"right",color:C.teal,width:1.7});
  connect(slide,fineHistory,attnCoarse,{kind:"curved",fromSide:"right",toSide:"left",color:C.orange,width:1.45});

  const gate = addBox(slide,"FIG3_GATE","G",458,367,60,42,{geometry:"diamond",fill:C.purpleLight,line:C.purple,lineWidth:1.4,size:12.5,wrap:"none",radius:0});
  connect(slide,attnFine,gate,{fromSide:"bottom",toSide:"top",color:C.purple,width:1.6});
  connect(slide,attnCoarse,gate,{fromSide:"bottom",toSide:"top",color:C.purple,width:1.6});

  const updatedFine = drawUpdatedRibbon(slide,"FIG3_UPDATED_FINE",176,452,240,57,C.orange,C.orangeLight,"fine","Updated Fine","H'f");
  const updatedCoarse = drawUpdatedRibbon(slide,"FIG3_UPDATED_COARSE",544,452,240,57,C.teal,C.tealLight,"coarse","Updated Coarse","H'c");
  connect(slide,gate,updatedFine,{fromSide:"bottom",toSide:"top",color:C.orange,width:1.7});
  connect(slide,gate,updatedCoarse,{fromSide:"bottom",toSide:"top",color:C.teal,width:1.7});
  addText(slide,"FIG3_GATE__G_LABEL","G",404,434,24,16,{size:10.5,color:C.orange});
  addText(slide,"FIG3_GATE__ONE_G_LABEL","1-G",532,434,34,16,{size:10.5,color:C.teal});

  const decoder = addBox(slide,"FIG3_UPDATED_COARSE__DECODER","Direct Multi-Output\nDecoder",812,458,136,48,{fill:C.white,line:C.line,size:10.5,radius:7});
  connect(slide,updatedCoarse,decoder,{fromSide:"right",toSide:"left",color:C.line,width:1.5});
  addText(slide,"FIG3__SEPARATE_NOTE","Fine and coarse representations remain separate.",311,515,338,18,{size:10.5,color:C.secondary});
}

await fs.mkdir(QA_DIR, { recursive: true });
const presentation = await PresentationFile.importPptx(await FileBlob.load(SOURCE));
const sourceSlides = [...presentation.slides.items];
sourceSlides[0].delete();
const slides = presentation.slides.items;
if (slides.length !== 2) throw new Error(`Expected 2 slides after removing Figure 1, got ${slides.length}`);
for (const slide of slides) {
  slide.setViewportSize(W, H);
  slide.shapes.deleteAll();
  slide.speakerNotes?.setText?.("");
}
buildFigure2(slides[0]);
buildFigure3(slides[1]);

const frameMap = {
  outputSlides: [
    { outputSlide: 1, sourceSlide: 2, narrativeRole: "technical mechanism content", reuseMode: "duplicate-slide", editTargets: [{ action: "rewrite-and-reposition", sourceElementId: "sh/ny1czapg" }, { action: "add", newPrimitiveAllowed: true, mustNotOverlapInherited: true, reason: "User explicitly requested a full compact re-layout on a smaller canvas after replacing inherited slide-local objects.", zone: { left: 0, top: 0, width: W, height: H } }] },
    { outputSlide: 2, sourceSlide: 3, narrativeRole: "technical mechanism content", reuseMode: "duplicate-slide", editTargets: [{ action: "rewrite-and-reposition", sourceElementId: "sh/el0jux8b" }, { action: "add", newPrimitiveAllowed: true, mustNotOverlapInherited: true, reason: "User explicitly requested waveform/trend ribbons and a compact cross-fusion re-layout after replacing inherited slide-local objects.", zone: { left: 0, top: 0, width: W, height: H } }] },
  ],
  omittedSourceSlides: [{ sourceSlide: 1, reason: "Figure 1 is frozen and explicitly excluded from the final two-slide deck." }],
};
await fs.writeFile(path.join(QA_DIR, "template-frame-map.json"), `${JSON.stringify(frameMap, null, 2)}\n`, "utf8");
await fs.writeFile(path.join(QA_DIR, "template-audit.txt"), "Source deck: 3 slides, 1280 × 720 px. Output maps source slide 2 to output slide 1 and source slide 3 to output slide 2. The source master/theme and scientific palette are preserved. Figure 1 remains unchanged in the source and is omitted from the output. All slide-local objects on mapped slides are replaced because the user explicitly requested a smaller 10.0 × 5.6 in canvas and full connector/layout reconstruction.\n", "utf8");
await fs.writeFile(path.join(QA_DIR, "deviation-log.txt"), "Slides 1–2: viewport changed from 1280 × 720 px to 960 × 537.6 px. Inherited slide-local diagram objects were replaced with compact editable shapes; source master/theme retained. Figure 3 history token boxes replaced by waveform/trend ribbons.\n", "utf8");
await fs.writeFile(path.join(QA_DIR, "source-notes.txt"), "No external sources or assets used. Scientific content and terminology are inherited from the user-provided ST-MGPrompt source deck and task specification.\n", "utf8");

for (let i = 0; i < slides.length; i += 1) {
  await saveBlob(path.join(QA_DIR, `slide-${i + 1}.png`), await presentation.export({ slide: slides[i], format: "png", scale: 2 }));
  await saveBlob(path.join(QA_DIR, `slide-${i + 1}.layout.json`), await slides[i].export({ format: "layout" }));
}
await saveBlob(path.join(QA_DIR, "montage.webp"), await presentation.export({ format: "webp", montage: true, scale: 1 }));
const inspected = await presentation.inspect({ kind: "deck,slide,textbox,shape,layout", include: "id,slide,name,text,textPreview,bbox,bboxUnit", maxChars: 200000 });
await fs.writeFile(path.join(QA_DIR, "final-inspect.ndjson"), inspected.ndjson ?? "", "utf8");
const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save(OUT);
console.log(OUT);
