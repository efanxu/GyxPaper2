from __future__ import annotations

import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from xml.etree import ElementTree as ET

from .constants import EXTERNAL_MODEL_IDS, HORIZONS, METRICS
from .io_utils import finite_number

MAIN_SHEET = "原始指标"
DETAIL_SHEET = "模型明细"
REQUIRED_MAIN_COLUMNS = {
    "实验组", "模型", "model_id", "Horizon", "MAE", "RMSE", "R2", "Score",
    "metric_status", "run_status", "protocol_status", "source_file",
}
REQUIRED_DETAIL_COLUMNS = {
    "实验组", "模型", "model_id", "运行状态", "协议检查",
    "H3 MAE", "H3 RMSE", "H3 R²", "H3 Score",
    "H6 MAE", "H6 RMSE", "H6 R²", "H6 Score",
    "H10 MAE", "H10 RMSE", "H10 R²", "H10 Score",
}
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL_NS = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}


@dataclass(frozen=True)
class Original26Model:
    model_id: str
    run_id: str
    group_id: str
    run_status: str
    protocol_status: str
    metrics: dict[int, dict[str, float]]
    source_file: str
    parameter_count: int | None


def discover_original26(project_root: str | Path) -> tuple[Path | None, list[str]]:
    matches: list[Path] = []
    for root, dirs, files in os.walk(Path(project_root), onerror=lambda _: None):
        dirs[:] = [name for name in dirs if not name.startswith(".") and name not in {"results_smoke", "smoke"}]
        for name in files:
            if name.lower().startswith("original26") and name.lower().endswith(".xlsx"):
                matches.append(Path(root) / name)
    matches.sort()
    if len(matches) == 1:
        return matches[0].resolve(), []
    if not matches:
        return None, ["MISSING_ORIGINAL26_XLSX"]
    return None, ["AMBIGUOUS_ORIGINAL26_XLSX", *map(str, matches)]


def _column_index(reference: str) -> int:
    letters = re.match(r"[A-Z]+", reference)
    if not letters:
        raise ValueError(f"Invalid cell reference: {reference}")
    value = 0
    for char in letters.group(0):
        value = value * 26 + ord(char) - 64
    return value - 1


class _XlsxReader:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.archive = zipfile.ZipFile(self.path)
        self.shared = self._shared_strings()
        self.sheets = self._sheet_paths()

    def close(self) -> None:
        self.archive.close()

    def _shared_strings(self) -> list[str]:
        if "xl/sharedStrings.xml" not in self.archive.namelist():
            return []
        root = ET.fromstring(self.archive.read("xl/sharedStrings.xml"))
        return ["".join(node.text or "" for node in item.iterfind(".//m:t", NS)) for item in root]

    def _sheet_paths(self) -> dict[str, str]:
        workbook = ET.fromstring(self.archive.read("xl/workbook.xml"))
        rels = ET.fromstring(self.archive.read("xl/_rels/workbook.xml.rels"))
        targets = {item.attrib["Id"]: item.attrib["Target"] for item in rels.findall("r:Relationship", REL_NS)}
        relation_key = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
        result = {}
        for sheet in workbook.findall(".//m:sheet", NS):
            target = targets[sheet.attrib[relation_key]].lstrip("/")
            if not target.startswith("xl/"):
                target = str(PurePosixPath("xl") / target)
            result[sheet.attrib["name"]] = target
        return result

    def rows(self, sheet_name: str) -> list[list[Any]]:
        if sheet_name not in self.sheets:
            raise ValueError(f"Missing sheet: {sheet_name}")
        root = ET.fromstring(self.archive.read(self.sheets[sheet_name]))
        output: list[list[Any]] = []
        for row in root.findall(".//m:sheetData/m:row", NS):
            values: list[Any] = []
            for cell in row.findall("m:c", NS):
                index = _column_index(cell.attrib["r"])
                while len(values) <= index:
                    values.append(None)
                cell_type = cell.attrib.get("t")
                if cell_type == "inlineStr":
                    value = "".join(node.text or "" for node in cell.findall(".//m:t", NS))
                else:
                    node = cell.find("m:v", NS)
                    raw = None if node is None else node.text
                    if raw is None:
                        value = None
                    elif cell_type == "s":
                        value = self.shared[int(raw)]
                    elif cell_type in {"str", "e"}:
                        value = raw
                    elif cell_type == "b":
                        value = raw == "1"
                    else:
                        number = float(raw)
                        value = int(number) if number.is_integer() else number
                values[index] = value
            output.append(values)
        return output


def _header_table(rows: list[list[Any]], required: set[str]) -> tuple[list[str], list[dict[str, Any]]]:
    for index, row in enumerate(rows):
        headers = [str(item).strip() if item is not None else "" for item in row]
        if required.issubset(set(headers)):
            table = []
            for source in rows[index + 1:]:
                if not any(item is not None for item in source):
                    continue
                padded = source + [None] * (len(headers) - len(source))
                table.append(dict(zip(headers, padded)))
            return headers, table
    raise ValueError(f"No header row contains required columns: {sorted(required)}")


def load_original26(path: str | Path) -> tuple[dict[str, Original26Model], dict[str, Any]]:
    reader = _XlsxReader(path)
    try:
        main_headers, rows = _header_table(reader.rows(MAIN_SHEET), REQUIRED_MAIN_COLUMNS)
        detail_headers, details = _header_table(reader.rows(DETAIL_SHEET), REQUIRED_DETAIL_COLUMNS)
    finally:
        reader.close()
    selected = [row for row in rows if str(row.get("model_id", "")).lower() in EXTERNAL_MODEL_IDS]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in selected:
        grouped.setdefault(str(row["model_id"]).lower(), []).append(row)
    errors: list[str] = []
    if set(grouped) != set(EXTERNAL_MODEL_IDS):
        errors.append("ORIGINAL26_MODEL_WHITELIST_INCOMPLETE")
    models: dict[str, Original26Model] = {}
    for model_id in EXTERNAL_MODEL_IDS:
        model_rows = grouped.get(model_id, [])
        horizons = [int(row["Horizon"]) for row in model_rows]
        if sorted(horizons) != list(HORIZONS) or len(set(horizons)) != len(horizons):
            errors.append(f"{model_id}:INVALID_HORIZONS")
            continue
        run_ids = {str(row["模型"]) for row in model_rows}
        groups = {str(row["实验组"]) for row in model_rows}
        sources = {str(row["source_file"]) for row in model_rows}
        if len(run_ids) != 1 or len(groups) != 1 or len(sources) != 1:
            errors.append(f"{model_id}:NON_UNIQUE_IDENTITY")
            continue
        if {str(row["run_status"]).upper() for row in model_rows} != {"COMPLETED"}:
            errors.append(f"{model_id}:FORMAL_RUN_STATUS_INVALID")
        if {str(row["protocol_status"]).upper() for row in model_rows} != {"PASS"}:
            errors.append(f"{model_id}:PROTOCOL_STATUS_INVALID")
        if {str(row["metric_status"]).upper() for row in model_rows} != {"OK"}:
            errors.append(f"{model_id}:METRIC_STATUS_INVALID")
        metrics = {int(row["Horizon"]): {metric: finite_number(row[metric]) for metric in METRICS} for row in model_rows}
        parameters = {row.get("parameter_count") for row in model_rows}
        parameter_count = next(iter(parameters)) if len(parameters) == 1 else None
        models[model_id] = Original26Model(
            model_id=model_id, run_id=next(iter(run_ids)), group_id=next(iter(groups)),
            run_status="COMPLETED", protocol_status="PASS", metrics=metrics,
            source_file=next(iter(sources)), parameter_count=None if parameter_count is None else int(parameter_count),
        )
    detail_ids = [str(row.get("model_id", "")).lower() for row in details if row.get("model_id")]
    for model_id in EXTERNAL_MODEL_IDS:
        if detail_ids.count(model_id) != 1:
            errors.append(f"{model_id}:MODEL_DETAIL_NOT_UNIQUE")
            continue
        detail = next(row for row in details if str(row.get("model_id", "")).lower() == model_id)
        if str(detail.get("运行状态", "")).upper() != "COMPLETED" or str(detail.get("协议检查", "")).upper() != "PASS":
            errors.append(f"{model_id}:MODEL_DETAIL_STATUS_INVALID")
        if model_id in models:
            for horizon in HORIZONS:
                for metric, label in (("MAE", "MAE"), ("RMSE", "RMSE"), ("R2", "R²"), ("Score", "Score")):
                    if finite_number(detail[f"H{horizon} {label}"]) != models[model_id].metrics[horizon][metric]:
                        errors.append(f"{model_id}:MODEL_DETAIL_METRIC_MISMATCH:H{horizon}:{metric}")
    if errors:
        raise ValueError(";".join(errors))
    return models, {
        "path": str(Path(path).resolve()), "sheets": [MAIN_SHEET, DETAIL_SHEET],
        "main_columns": main_headers, "detail_columns": detail_headers,
        "selected_model_ids": list(models), "selected_row_count": len(selected), "schema_valid": True,
    }
