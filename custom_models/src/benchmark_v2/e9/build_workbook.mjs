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

function scalar(value) {
  if (value === null || value === undefined) return null;
  if (typeof value === "object") return JSON.stringify(value);
  return value;
}

function objectRows(rows) {
  if (!rows || rows.length === 0) return [["status"], ["NOT_AVAILABLE"]];
  const keys = [...new Set(rows.flatMap((row) => Object.keys(row)))];
  return [keys, ...rows.map((row) => keys.map((key) => scalar(row[key])))];
}

const audit = payload.audit || {};
const tables = payload.tables || {};
const evidence = payload.evidence || [];
const sheets = {
  README: [
    ["E9 MS-MG-DWU Transfer Analysis", "Value"],
    ["E5", "Cancelled; not read or restored"],
    ["Control", "original26 Original Masked-MSE formal results (read-only; not retrained)"],
    ["New formal runs", "Six MS-MG-DWU transfers only"],
    ["Frozen models", "LightTS, TiDE, PatchTST, iTransformer, DCRNN, MTGNN"],
    ["Metric direction", "Score/MAE/RMSE lower is better; R2 higher is better"],
    ["Improvement sign", "Positive means MS-MG-DWU is better for Score/MAE/RMSE"],
    ["R2", "Absolute R2 gain only"],
    ["Interpretation", "Engineering portability does not imply every model improves"],
    ["Reporting", "All unfavorable results are retained; all H3/H6/H10 are reported"],
    ["Portable subset", "Not used; if used it could not be called full MS-MG-DWU"],
    ["A0/A8", "Optional related internal reference; excluded from the core six-model summary"],
  ],
  Portability_Audit: objectRows(audit.portability?.dependency_closure || []),
  Loss_Input_Contract: objectRows(audit.loss_contract?.fields || []),
  Evidence_Readiness: objectRows(payload.readiness?.pairs || []),
  Original_Control_Audit: objectRows(evidence.filter((row) => row.evidence_role === "ORIGINAL26_CONTROL")),
  Pairing_Audit: objectRows(audit.pairing?.pairs || []),
  Config_Diff: objectRows((audit.config_diff?.pairs || []).map((row) => ({
    model_id: row.model_id, control_run_id: row.control_run_id, transfer_run_id: row.transfer_run_id,
    loss_only_diff_valid: row.loss_only_diff_valid, pair_status: row.pair_status,
    unexpected_differences: row.unexpected_differences,
  }))),
  Main_Transfer_Results: objectRows(tables["e9_model_pair_comparison.csv"] || []),
  H3_Comparison: objectRows((tables["e9_transfer_metrics.csv"] || []).filter((row) => row.horizon === 3)),
  H6_Comparison: objectRows((tables["e9_transfer_metrics.csv"] || []).filter((row) => row.horizon === 6)),
  H10_Comparison: objectRows((tables["e9_transfer_metrics.csv"] || []).filter((row) => row.horizon === 10)),
  Family_Summary: objectRows(tables["e9_family_summary.csv"] || []),
  Training_Dynamics: objectRows(tables["e9_training_dynamics.csv"] || []),
  Loss_Components: objectRows(tables["e9_loss_component_statistics.csv"] || []),
  Efficiency_Overhead: objectRows(tables["e9_efficiency_overhead.csv"] || []),
  Optional_A0_A8_Reference: [["status", "role"], ["NOT_INCLUDED", "RELATED_REFERENCE only; excluded from core E9"]],
  Preflight_Inventory: objectRows(audit.preflight?.inventory || []),
  Failures: objectRows(tables["e9_failure_inventory.csv"] || []),
  Artifact_Manifest: objectRows(audit.artifact_manifest?.artifacts || []),
};

for (const [name, rows] of Object.entries(sheets)) {
  const width = Math.max(...rows.map((row) => row.length), 1);
  const normalized = rows.map((row) => [...row, ...Array(width - row.length).fill(null)]);
  const sheet = workbook.worksheets.add(name);
  sheet.showGridLines = false;
  const endColumn = columnName(width);
  const used = sheet.getRange(`A1:${endColumn}${normalized.length}`);
  used.values = normalized;
  sheet.getRange(`A1:${endColumn}1`).format = {
    fill: "#174A5B", font: { bold: true, color: "#FFFFFF" }, wrapText: true,
    borders: { preset: "outside", style: "thin", color: "#9FB4BD" },
  };
  used.format.autofitColumns();
  used.format.autofitRows();
  if (width > 1) sheet.getRange(`B1:${endColumn}${normalized.length}`).format.columnWidth = 18;
  sheet.getRange(`A1:${endColumn}${normalized.length}`).format.wrapText = true;
  sheet.freezePanes.freezeRows(1);
}

for (const name of Object.keys(sheets)) {
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
