import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const nodeExecutable = "C:/Users/12811/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe";
const originalPath = "C:/Users/12811/Desktop/实验结果/original26_filtered20.xlsx";
const transferPath = "C:/Users/12811/Desktop/实验结果/MSMG_DWU_TRANSFER_ANALYSIS.xlsx";
const desktopDir = "C:/Users/12811/Desktop";
const previewDir = "C:/Users/12811/AppData/Local/Temp/codex_spreadsheet_20260906";

const modelOrder = [
  { id: "lightts", name: "LightTS" },
  { id: "tide", name: "TiDE" },
  { id: "patchtst", name: "PatchTST" },
  { id: "itransformer", name: "iTransformer" },
  { id: "dcrnn", name: "DCRNN" },
  { id: "mtgnn", name: "MTGNN" },
];
const horizons = [3, 6, 10];
const metricKeys = ["Score", "MAE", "RMSE", "R2"];
const metricColumns = { Score: "F", MAE: "G", RMSE: "H", R2: "I" };
const resultMetricColumns = {
  3: { Score: "B", MAE: "C", RMSE: "D", R2: "E" },
  6: { Score: "F", MAE: "G", RMSE: "H", R2: "I" },
  10: { Score: "J", MAE: "K", RMSE: "L", R2: "M" },
};

function headerIndex(headerRow) {
  return Object.fromEntries(headerRow.map((value, index) => [value, index]));
}

function sourceRowLocation(sheetName, excelRow, lastColumn) {
  return `${sheetName}!A${excelRow}:${lastColumn}${excelRow}`;
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

async function importWorkbook(path) {
  return SpreadsheetFile.importXlsx(await FileBlob.load(path));
}

const originalWorkbook = await importWorkbook(originalPath);
const transferWorkbook = await importWorkbook(transferPath);

const originalSheet = originalWorkbook.worksheets.getItem("原始指标");
const originalValues = originalSheet.getUsedRange().values;
const originalHeader = headerIndex(originalValues[0]);
const originalRows = originalValues
  .map((row, index) => ({ row, excelRow: index + 1 }))
  .filter(({ row }) => modelOrder.some((model) => model.id === row[originalHeader.model_id]));

const mainTransferSheet = transferWorkbook.worksheets.getItem("Main_Transfer_Results");
const mainTransferValues = mainTransferSheet.getUsedRange().values;
const mainTransferHeader = headerIndex(mainTransferValues[3]);
const mainTransferRows = new Map(
  mainTransferValues.slice(4).map((row, index) => [row[mainTransferHeader.model_id], { row, excelRow: index + 5 }]),
);

const comparisons = {};
for (const horizon of horizons) {
  const sheetName = `H${horizon}_Comparison`;
  const sheet = transferWorkbook.worksheets.getItem(sheetName);
  const values = sheet.getUsedRange().values;
  const header = headerIndex(values[3]);
  comparisons[horizon] = new Map(
    values.slice(4).map((row, index) => [row[header.model_id], { row, excelRow: index + 5, header }]),
  );
}

const originalByModel = new Map();
for (const model of modelOrder) {
  const rows = originalRows.filter(({ row }) => row[originalHeader.model_id] === model.id);
  assert(rows.length === 3, `${model.id} should have exactly 3 original26 rows, found ${rows.length}`);
  const byHorizon = new Map(rows.map(({ row, excelRow }) => [row[originalHeader.Horizon], { row, excelRow }]));
  assert(horizons.every((horizon) => byHorizon.has(horizon)), `${model.id} is missing an original26 horizon`);
  originalByModel.set(model.id, byHorizon);
}

for (const model of modelOrder) {
  assert(mainTransferRows.has(model.id), `${model.id} is missing an E9 transfer run`);
  for (const horizon of horizons) {
    assert(comparisons[horizon].has(model.id), `${model.id} is missing an E9 H${horizon} comparison row`);
  }
}

const runData = [];
const sourceRows = [];
const sourceRowMap = new Map();

function addSourceRow(record) {
  const sourceRowNumber = sourceRows.length + 5;
  sourceRows.push(record);
  sourceRowMap.set(`${record.model_id}|${record.variant}|${record.Horizon}`, sourceRowNumber);
}

for (const model of modelOrder) {
  const originalRun = originalByModel.get(model.id);
  const originalFirst = originalRun.get(3).row;
  const transferMain = mainTransferRows.get(model.id).row;
  const transferName = transferMain[mainTransferHeader.transfer_run_id];
  const originalRunId = originalFirst[originalHeader.模型];

  const originalRecord = {
    model_id: model.id,
    model_name: model.name,
    variant: "Original Masked-MSE",
    run_id: originalRunId,
    run_status: originalFirst[originalHeader.run_status],
    protocol_status: originalFirst[originalHeader.protocol_status],
    loss_only_pair_status: "N/A",
    formal_evidence_ready: "PASS",
    pairing_status: "NOT_VERIFIED",
    source_workbook: originalPath,
    source_sheet: "原始指标",
  };
  const transferRecord = {
    model_id: model.id,
    model_name: model.name,
    variant: "E9 MS-MG-DWU",
    run_id: transferName,
    run_status: transferMain[mainTransferHeader.transfer_status],
    protocol_status: "PASS",
    loss_only_pair_status: transferMain[mainTransferHeader.loss_only_pair_status],
    formal_evidence_ready: transferMain[mainTransferHeader.formal_evidence_ready],
    pairing_status: transferMain[mainTransferHeader.pairing_status],
    source_workbook: transferPath,
    source_sheet: "H3/H6/H10_Comparison",
  };

  for (const horizon of horizons) {
    const original = originalRun.get(horizon);
    const originalRow = original.row;
    const originalData = {
      ...originalRecord,
      Horizon: horizon,
      Score: originalRow[originalHeader.Score],
      MAE: originalRow[originalHeader.MAE],
      RMSE: originalRow[originalHeader.RMSE],
      R2: originalRow[originalHeader.R2],
      valid_target_ratio: originalRow[originalHeader.valid_target_ratio],
      source_location: sourceRowLocation("原始指标", original.excelRow, "P"),
    };
    addSourceRow(originalData);

    const comparison = comparisons[horizon].get(model.id);
    const comparisonRow = comparison.row;
    const transferData = {
      ...transferRecord,
      Horizon: horizon,
      Score: comparisonRow[comparison.header.transfer_Score],
      MAE: comparisonRow[comparison.header.transfer_MAE],
      RMSE: comparisonRow[comparison.header.transfer_RMSE],
      R2: comparisonRow[comparison.header.transfer_R2],
      valid_target_ratio: comparisonRow[comparison.header.valid_target_ratio_transfer],
      source_location: sourceRowLocation(`H${horizon}_Comparison`, comparison.excelRow, "U"),
    };
    addSourceRow(transferData);
  }

  runData.push({ model, original: originalRecord, transfer: transferRecord });
}

assert(sourceRows.length === 36, `Expected 36 source rows, found ${sourceRows.length}`);

const workbook = Workbook.create();
const resultSheet = workbook.worksheets.add("E9结果");
const comparisonSheet = workbook.worksheets.add("对比分析");
const sourceSheet = workbook.worksheets.add("源数据");
const notesSheet = workbook.worksheets.add("说明");

for (const sheet of [resultSheet, comparisonSheet, sourceSheet, notesSheet]) {
  sheet.showGridLines = false;
}

const dark = "#1F2937";
const medium = "#4B5563";
const lightBorder = "#D1D5DB";
const headerFill = "#EAF0F6";
const groupFill = "#D9E6F2";
const e9Fill = "#F6FAFE";

// Main presentation sheet: a compact 12-row table styled after the supplied reference.
resultSheet.getRange("A1:M1").merge();
resultSheet.getRange("A2:M2").merge();
resultSheet.getRange("A4:A5").merge();
resultSheet.getRange("B4:E4").merge();
resultSheet.getRange("F4:I4").merge();
resultSheet.getRange("J4:M4").merge();
resultSheet.getRange("A1").values = [["表·E9·MS-MG-DWU迁移实验结果"]];
resultSheet.getRange("A2").values = [["6 个 Original Masked-MSE references 与 6 个 E9 MS-MG-DWU formal runs；Score / MAE / RMSE 越低越好，R² 越高越好。"]];
resultSheet.getRange("A4").values = [["模型/变体"]];
resultSheet.getRange("B4").values = [["H3"]];
resultSheet.getRange("F4").values = [["H6"]];
resultSheet.getRange("J4").values = [["H10"]];
resultSheet.getRange("B5:M5").values = [["Score", "MAE", "RMSE", "R²", "Score", "MAE", "RMSE", "R²", "Score", "MAE", "RMSE", "R²"]];

const resultLabels = [];
const resultFormulaRows = [];
for (const item of runData) {
  resultLabels.push([`${item.model.name} · Original Masked-MSE`]);
  resultLabels.push([`${item.model.name} · E9 MS-MG-DWU`]);
  const originalFormulaRow = [];
  const transferFormulaRow = [];
  for (const horizon of horizons) {
    const originalRowBase = sourceRowMap.get(`${item.model.id}|Original Masked-MSE|${horizon}`);
    const transferRowBase = sourceRowMap.get(`${item.model.id}|E9 MS-MG-DWU|${horizon}`);
    for (const metric of metricKeys) {
      const sourceColumn = metricColumns[metric];
      originalFormulaRow.push(`='源数据'!${sourceColumn}${originalRowBase}`);
      transferFormulaRow.push(`='源数据'!${sourceColumn}${transferRowBase}`);
    }
  }
  resultFormulaRows.push(originalFormulaRow, transferFormulaRow);
}
resultSheet.getRange("A6:A17").values = resultLabels;
resultSheet.getRange("B6:M17").formulas = resultFormulaRows;
resultSheet.getRange("A19:M19").merge();
resultSheet.getRange("A19").values = [["加粗表示同一模型的 Original 与 E9 配对中该指标更优；E9 配对的初始状态字节级一致性沿用源审计结论：NOT_VERIFIED。"]];

resultSheet.getRange("A1:M1").format = { font: { bold: true, color: dark }, horizontalAlignment: "center", verticalAlignment: "center" };
resultSheet.getRange("A2:M2").format = { font: { color: medium }, horizontalAlignment: "center", verticalAlignment: "center", wrapText: true };
resultSheet.getRange("A4:M5").format = { fill: headerFill, font: { bold: true, color: dark }, horizontalAlignment: "center", verticalAlignment: "center", wrapText: true, borders: { preset: "all", style: "thin", color: lightBorder } };
resultSheet.getRange("B4:E4").format.fill = groupFill;
resultSheet.getRange("F4:I4").format.fill = groupFill;
resultSheet.getRange("J4:M4").format.fill = groupFill;
resultSheet.getRange("A6:A17").format = { font: { color: dark }, horizontalAlignment: "left", verticalAlignment: "center" };
resultSheet.getRange("B6:M17").format = { font: { color: dark }, horizontalAlignment: "right", verticalAlignment: "center" };
resultSheet.getRange("A7:M7").format.fill = e9Fill;
resultSheet.getRange("A9:M9").format.fill = e9Fill;
resultSheet.getRange("A11:M11").format.fill = e9Fill;
resultSheet.getRange("A13:M13").format.fill = e9Fill;
resultSheet.getRange("A15:M15").format.fill = e9Fill;
resultSheet.getRange("A17:M17").format.fill = e9Fill;
resultSheet.getRange("A6:M17").format.borders = { insideHorizontal: { style: "thin", color: lightBorder }, insideVertical: { style: "thin", color: "#E5E7EB" }, left: { style: "medium", color: medium }, right: { style: "medium", color: medium }, top: { style: "medium", color: medium }, bottom: { style: "medium", color: medium } };
for (const row of [7, 9, 11, 13, 15, 17]) {
  resultSheet.getRange(`A${row}:M${row}`).format.borders = { bottom: { style: "medium", color: medium } };
}
resultSheet.getRange("A4:M5").format.borders = { top: { style: "medium", color: medium }, bottom: { style: "thin", color: lightBorder }, left: { style: "medium", color: medium }, right: { style: "medium", color: medium }, insideHorizontal: { style: "thin", color: lightBorder }, insideVertical: { style: "thin", color: lightBorder } };
resultSheet.getRange("A19:M19").format = { font: { color: medium }, horizontalAlignment: "left", verticalAlignment: "center", wrapText: true };
resultSheet.getRange("A1:M1").format.rowHeight = 28;
resultSheet.getRange("A2:M2").format.rowHeight = 34;
resultSheet.getRange("A4:M5").format.rowHeight = 22;
resultSheet.getRange("A6:M17").format.rowHeight = 21;
resultSheet.getRange("A19:M19").format.rowHeight = 32;
resultSheet.getRange("A1:A19").format.columnWidth = 31;
resultSheet.getRange("B1:M19").format.columnWidth = 11.5;
resultSheet.getRange("B6:M17").setNumberFormat("0.000");

// Bold the better member of each Original/E9 pair for every metric, matching the reference emphasis.
for (let i = 0; i < runData.length; i += 1) {
  const item = runData[i];
  const originalRow = 6 + i * 2;
  const transferRow = originalRow + 1;
  const sourceOriginal = sourceRows.filter((row) => row.model_id === item.model.id && row.variant === "Original Masked-MSE");
  const sourceTransfer = sourceRows.filter((row) => row.model_id === item.model.id && row.variant === "E9 MS-MG-DWU");
  for (const horizon of horizons) {
    const originalMetricRow = sourceOriginal.find((row) => row.Horizon === horizon);
    const transferMetricRow = sourceTransfer.find((row) => row.Horizon === horizon);
    for (const metric of metricKeys) {
      const lowerIsBetter = metric !== "R2";
      const originalValue = originalMetricRow[metric];
      const transferValue = transferMetricRow[metric];
      const transferWins = lowerIsBetter ? transferValue < originalValue : transferValue > originalValue;
      const winnerRow = transferWins ? transferRow : originalRow;
      const col = resultMetricColumns[horizon][metric];
      resultSheet.getRange(`${col}${winnerRow}`).format.font = { bold: true, color: dark };
    }
  }
}
resultSheet.freezePanes.freezeRows(5);
resultSheet.freezePanes.freezeColumns(1);

// Formula-backed paired comparison sheet.
comparisonSheet.getRange("A1:I1").merge();
comparisonSheet.getRange("A2:I2").merge();
comparisonSheet.getRange("A1").values = [["E9 MS-MG-DWU 与 Original Masked-MSE 配对对比"]];
comparisonSheet.getRange("A2").values = [["Δ = E9 − Original；Score 改进 = (Original − E9) / Original；R² 增益 = E9 − Original。"]];
comparisonSheet.getRange("A4:I4").values = [["模型", "H3 Score Δ", "H3 Score 改进", "H6 Score Δ", "H6 Score 改进", "H10 Score Δ", "H10 Score 改进", "H10 R² 增益", "H10 Score 更优?"]];
const comparisonFormulas = [];
for (let i = 0; i < runData.length; i += 1) {
  const resultOriginalRow = 6 + i * 2;
  const resultTransferRow = resultOriginalRow + 1;
  const outputRow = 5 + i;
  comparisonFormulas.push([
    `='E9结果'!A${resultTransferRow}`,
    `='E9结果'!B${resultTransferRow}-'E9结果'!B${resultOriginalRow}`,
    `=('E9结果'!B${resultOriginalRow}-'E9结果'!B${resultTransferRow})/'E9结果'!B${resultOriginalRow}`,
    `='E9结果'!F${resultTransferRow}-'E9结果'!F${resultOriginalRow}`,
    `=('E9结果'!F${resultOriginalRow}-'E9结果'!F${resultTransferRow})/'E9结果'!F${resultOriginalRow}`,
    `='E9结果'!J${resultTransferRow}-'E9结果'!J${resultOriginalRow}`,
    `=('E9结果'!J${resultOriginalRow}-'E9结果'!J${resultTransferRow})/'E9结果'!J${resultOriginalRow}`,
    `='E9结果'!M${resultTransferRow}-'E9结果'!M${resultOriginalRow}`,
    `=IF(G${outputRow}>0,"是","否")`,
  ]);
}
comparisonSheet.getRange("A5:I10").formulas = comparisonFormulas;
comparisonSheet.getRange("A1:I1").format = { font: { bold: true, color: dark }, horizontalAlignment: "center", verticalAlignment: "center" };
comparisonSheet.getRange("A2:I2").format = { font: { color: medium }, horizontalAlignment: "center", verticalAlignment: "center", wrapText: true };
comparisonSheet.getRange("A4:I4").format = { fill: headerFill, font: { bold: true, color: dark }, horizontalAlignment: "center", verticalAlignment: "center", wrapText: true, borders: { preset: "all", style: "thin", color: lightBorder } };
comparisonSheet.getRange("A5:A10").format = { font: { color: dark }, horizontalAlignment: "left", verticalAlignment: "center" };
comparisonSheet.getRange("B5:H10").format = { font: { color: dark }, horizontalAlignment: "right", verticalAlignment: "center" };
comparisonSheet.getRange("I5:I10").format = { font: { color: dark }, horizontalAlignment: "center", verticalAlignment: "center" };
comparisonSheet.getRange("A4:I10").format.borders = { insideHorizontal: { style: "thin", color: lightBorder }, insideVertical: { style: "thin", color: "#E5E7EB" }, left: { style: "medium", color: medium }, right: { style: "medium", color: medium }, top: { style: "medium", color: medium }, bottom: { style: "medium", color: medium } };
comparisonSheet.getRange("A1:I1").format.rowHeight = 28;
comparisonSheet.getRange("A2:I2").format.rowHeight = 30;
comparisonSheet.getRange("A4:I4").format.rowHeight = 34;
comparisonSheet.getRange("A5:I10").format.rowHeight = 21;
comparisonSheet.getRange("A1:A10").format.columnWidth = 26;
comparisonSheet.getRange("B1:H10").format.columnWidth = 14;
comparisonSheet.getRange("I1:I10").format.columnWidth = 16;
comparisonSheet.getRange("B5:B10").setNumberFormat("0.000");
comparisonSheet.getRange("D5:D10").setNumberFormat("0.000");
comparisonSheet.getRange("F5:F10").setNumberFormat("0.000");
comparisonSheet.getRange("H5:H10").setNumberFormat("0.000");
comparisonSheet.getRange("C5:C10").setNumberFormat("0.0%");
comparisonSheet.getRange("E5:E10").setNumberFormat("0.0%");
comparisonSheet.getRange("G5:G10").setNumberFormat("0.0%");
comparisonSheet.freezePanes.freezeRows(4);

// Flat source/processed-data sheet for traceability and reproducibility.
sourceSheet.getRange("A1:R1").merge();
sourceSheet.getRange("A2:R2").merge();
sourceSheet.getRange("A1").values = [["E9 源数据（36 条：12 个 run × 3 个 horizon）"]];
sourceSheet.getRange("A2").values = [["原始指标直接来自两个用户提供的工作簿；E9结果页与对比分析页通过公式引用本页数据。"]];
const sourceHeaders = ["model_id", "模型", "变体", "run_id", "Horizon", "Score", "MAE", "RMSE", "R²", "valid_target_ratio", "run_status", "protocol_status", "loss_only_pair_status", "formal_evidence_ready", "pairing_status", "来源工作簿", "来源工作表", "来源位置"];
sourceSheet.getRange("A4:R4").values = [sourceHeaders];
sourceSheet.getRange("A5:R40").values = sourceRows.map((row) => [
  row.model_id,
  row.model_name,
  row.variant,
  row.run_id,
  row.Horizon,
  row.Score,
  row.MAE,
  row.RMSE,
  row.R2,
  row.valid_target_ratio,
  row.run_status,
  row.protocol_status,
  row.loss_only_pair_status,
  row.formal_evidence_ready,
  row.pairing_status,
  row.source_workbook,
  row.source_sheet,
  row.source_location,
]);
sourceSheet.getRange("A1:R1").format = { font: { bold: true, color: dark }, horizontalAlignment: "center", verticalAlignment: "center" };
sourceSheet.getRange("A2:R2").format = { font: { color: medium }, horizontalAlignment: "center", verticalAlignment: "center", wrapText: true };
sourceSheet.getRange("A4:R4").format = { fill: headerFill, font: { bold: true, color: dark }, horizontalAlignment: "center", verticalAlignment: "center", wrapText: true, borders: { preset: "all", style: "thin", color: lightBorder } };
sourceSheet.getRange("A5:R40").format = { font: { color: dark }, verticalAlignment: "center", wrapText: false };
sourceSheet.getRange("A4:R40").format.borders = { insideHorizontal: { style: "thin", color: lightBorder }, insideVertical: { style: "thin", color: "#E5E7EB" }, left: { style: "medium", color: medium }, right: { style: "medium", color: medium }, top: { style: "medium", color: medium }, bottom: { style: "medium", color: medium } };
sourceSheet.getRange("E5:E40").setNumberFormat("0");
sourceSheet.getRange("F5:I40").setNumberFormat("0.000");
sourceSheet.getRange("J5:J40").setNumberFormat("0.0%");
sourceSheet.getRange("A1:R1").format.rowHeight = 28;
sourceSheet.getRange("A2:R2").format.rowHeight = 30;
sourceSheet.getRange("A4:R4").format.rowHeight = 34;
sourceSheet.getRange("A5:R40").format.rowHeight = 20;
sourceSheet.getRange("A1:A40").format.columnWidth = 13;
sourceSheet.getRange("B1:C40").format.columnWidth = 20;
sourceSheet.getRange("D1:D40").format.columnWidth = 44;
sourceSheet.getRange("E1:E40").format.columnWidth = 9;
sourceSheet.getRange("F1:I40").format.columnWidth = 12;
sourceSheet.getRange("J1:J40").format.columnWidth = 16;
sourceSheet.getRange("K1:O40").format.columnWidth = 18;
sourceSheet.getRange("P1:P40").format.columnWidth = 46;
sourceSheet.getRange("Q1:Q40").format.columnWidth = 24;
sourceSheet.getRange("R1:R40").format.columnWidth = 28;
sourceSheet.freezePanes.freezeRows(4);

// Notes and provenance sheet.
notesSheet.getRange("A1:F1").merge();
notesSheet.getRange("A1").values = [["E9 结果文件说明"]];
notesSheet.getRange("A3:B3").values = [["项目", "内容"]];
notesSheet.getRange("A4:B12").values = [
  ["实验对象", "LightTS、TiDE、PatchTST、iTransformer、DCRNN、MTGNN"],
  ["数据构成", "6 个 original26 Original Masked-MSE references + 6 个 E9 MS-MG-DWU formal runs"],
  ["展示指标", "H3 / H6 / H10；每个 horizon 展示 Score、MAE、RMSE、R²"],
  ["指标方向", "Score / MAE / RMSE 越低越好；R² 越高越好"],
  ["加粗规则", "同一模型的 Original 与 E9 配对中，该指标更优的单元格加粗"],
  ["原始工作簿", originalPath],
  ["E9 工作簿", transferPath],
  ["审计状态", "6/6 controls、6/6 transfers、6/6 loss-only pairings；formal evidence ready"],
  ["限制说明", "源审计未验证每对 run 的初始权重字节级一致性；本文件不扩大该结论"],
];
notesSheet.getRange("A1:F1").format = { font: { bold: true, color: dark }, horizontalAlignment: "center", verticalAlignment: "center" };
notesSheet.getRange("A3:B3").format = { fill: headerFill, font: { bold: true, color: dark }, horizontalAlignment: "center", verticalAlignment: "center", borders: { preset: "all", style: "thin", color: lightBorder } };
notesSheet.getRange("A4:A12").format = { font: { bold: true, color: dark }, horizontalAlignment: "left", verticalAlignment: "center", wrapText: true };
notesSheet.getRange("B4:B12").format = { font: { color: dark }, horizontalAlignment: "left", verticalAlignment: "center", wrapText: true };
notesSheet.getRange("A3:B12").format.borders = { insideHorizontal: { style: "thin", color: lightBorder }, insideVertical: { style: "thin", color: lightBorder }, left: { style: "medium", color: medium }, right: { style: "medium", color: medium }, top: { style: "medium", color: medium }, bottom: { style: "medium", color: medium } };
notesSheet.getRange("A1:F1").format.rowHeight = 28;
notesSheet.getRange("A3:B3").format.rowHeight = 24;
notesSheet.getRange("A4:B12").format.rowHeight = 32;
notesSheet.getRange("A1:A12").format.columnWidth = 18;
notesSheet.getRange("B1:B12").format.columnWidth = 95;
notesSheet.freezePanes.freezeRows(3);

await fs.mkdir(previewDir, { recursive: true });
for (const [sheetName, fileName] of [["E9结果", "E9_result_preview.png"], ["对比分析", "E9_comparison_preview.png"], ["源数据", "E9_source_preview.png"], ["说明", "E9_notes_preview.png"]]) {
  const preview = await workbook.render({ sheetName, autoCrop: "all", scale: 2, format: "png" });
  await fs.writeFile(`${previewDir}/${fileName}`, new Uint8Array(await preview.arrayBuffer()));
}

const formulaErrors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: { useRegex: true, maxResults: 300 }, summary: "formula error scan" });
console.log("FORMULA_ERRORS");
console.log(formulaErrors.ndjson);
const resultInspect = await workbook.inspect({ kind: "table", sheetId: "E9结果", range: "A1:M19", include: "values,formulas", tableMaxRows: 20, tableMaxCols: 13, maxChars: 12000 });
console.log("RESULT_INSPECT");
console.log(resultInspect.ndjson);
const comparisonInspect = await workbook.inspect({ kind: "table", sheetId: "对比分析", range: "A1:I10", include: "values,formulas", tableMaxRows: 12, tableMaxCols: 10, maxChars: 8000 });
console.log("COMPARISON_INSPECT");
console.log(comparisonInspect.ndjson);

let outputPath = `${desktopDir}/E9_MS-MG-DWU_结果_6模型.xlsx`;
try {
  await fs.access(outputPath);
  outputPath = `${desktopDir}/E9_MS-MG-DWU_结果_6模型_20260906.xlsx`;
} catch {
  // The requested output name is available.
}
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(`OUTPUT_PATH=${outputPath}`);
console.log(`PREVIEW_DIR=${previewDir}`);
