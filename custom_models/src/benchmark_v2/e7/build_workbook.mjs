import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const [inputPath, outputPath] = process.argv.slice(2);
if (!inputPath || !outputPath) throw new Error("Usage: build_workbook.mjs <input.json> <output.xlsx>");
const payload = JSON.parse(await fs.readFile(inputPath, "utf8"));
const workbook = Workbook.create();

function columnName(index) {
  let result = "";
  while (index > 0) {
    const remainder = (index - 1) % 26;
    result = String.fromCharCode(65 + remainder) + result;
    index = Math.floor((index - 1) / 26);
  }
  return result;
}

for (const [name, rawRows] of Object.entries(payload.sheets)) {
  const rows = rawRows.length ? rawRows : [["status"], ["NOT_AVAILABLE"]];
  const width = Math.max(...rows.map((row) => row.length), 1);
  const normalized = rows.map((row) => [...row, ...Array(width - row.length).fill(null)]);
  const sheet = workbook.worksheets.add(name);
  sheet.showGridLines = false;
  const end = `${columnName(width)}${normalized.length}`;
  const used = sheet.getRange(`A1:${end}`);
  used.values = normalized;
  const header = sheet.getRange(`A1:${columnName(width)}1`);
  header.format = {
    fill: "#1F4E78",
    font: { bold: true, color: "#FFFFFF" },
    wrapText: true,
    borders: { preset: "outside", style: "thin", color: "#9EADBA" },
  };
  used.format.autofitColumns();
  used.format.autofitRows();
  sheet.freezePanes.freezeRows(1);
}

for (const name of Object.keys(payload.sheets)) {
  const check = await workbook.inspect({ kind: "table", range: `${name}!A1:Z25`, tableMaxRows: 25, tableMaxCols: 26, maxChars: 5000 });
  if (!check.ndjson) throw new Error(`Empty inspection for ${name}`);
  const preview = await workbook.render({ sheetName: name, autoCrop: "all", scale: 1, format: "png" });
  if ((await preview.arrayBuffer()).byteLength === 0) throw new Error(`Empty render for ${name}`);
}
const errors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: { useRegex: true, maxResults: 300 }, summary: "final formula error scan" });
if (errors.ndjson.includes("#REF!") || errors.ndjson.includes("#DIV/0!") || errors.ndjson.includes("#VALUE!")) throw new Error("Formula error found");
await fs.mkdir(path.dirname(outputPath), { recursive: true });
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
