import fs from "node:fs/promises";
import path from "node:path";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const sharp = require("C:/Users/12811/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/sharp");
const { PDFDocument } = require("C:/Users/12811/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/pdf-lib");

const args = process.argv.slice(2);
function argument(name) {
  const index = args.indexOf(name);
  if (index < 0 || index + 1 >= args.length) throw new Error(`Missing ${name}`);
  return args[index + 1];
}

function svgDimensions(svg) {
  const match = svg.match(/<svg[^>]*width="([0-9.]+)pt"[^>]*height="([0-9.]+)pt"/s);
  if (!match) throw new Error("Could not read SVG width/height in points");
  return { widthPt: Number(match[1]), heightPt: Number(match[2]) };
}

async function renderOne(svgPath, referencePngPath, outputStem) {
  const svg = await fs.readFile(svgPath);
  const { widthPt, heightPt } = svgDimensions(svg.toString("utf8"));
  const reference = await sharp(referencePngPath, { limitInputPixels: false }).metadata();
  if (!reference.width || !reference.height) throw new Error(`Missing reference PNG size: ${referencePngPath}`);

  console.log(`RENDER_START ${svgPath}`);
  // Render above the requested raster size and then downsample.  This keeps
  // the modified SVG typography and embedded point layer crisp at 600 dpi.
  const rendered = await sharp(svg, { density: 300, limitInputPixels: false })
    .png()
    .toBuffer();
  const png = await sharp(rendered, { limitInputPixels: false })
    .resize({ width: reference.width, height: reference.height, fit: "fill", kernel: "lanczos3" })
    .withMetadata({ density: 600 })
    .png()
    .toBuffer();
  const pngPath = `${outputStem}.png`;
  const pdfPath = `${outputStem}.pdf`;
  await fs.writeFile(pngPath, png);

  const pdf = await PDFDocument.create();
  const page = pdf.addPage([widthPt, heightPt]);
  const embedded = await pdf.embedPng(png);
  page.drawImage(embedded, { x: 0, y: 0, width: widthPt, height: heightPt });
  await fs.writeFile(pdfPath, await pdf.save());
  console.log(`RENDER_DONE ${pngPath}`);
  console.log(`PDF_DONE ${pdfPath}`);
}

const svgPaths = args.filter((value) => value.endsWith("_final2.svg"));
if (svgPaths.length !== 2) throw new Error("Pass exactly two *_final2.svg paths");
for (const svgPath of svgPaths) {
  const outputStem = svgPath.slice(0, -4);
  const referencePngPath = `${outputStem.replace(/_final2$/, "_final")}.png`;
  await renderOne(svgPath, referencePngPath, outputStem);
}
