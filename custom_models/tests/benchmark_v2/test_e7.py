from __future__ import annotations

import json
import math
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "custom_models" / "src"
if str(SOURCE_ROOT) in sys.path:
    sys.path.remove(str(SOURCE_ROOT))
sys.path.insert(0, str(SOURCE_ROOT))
loaded = sys.modules.get("benchmark_v2")
if loaded is not None and Path(getattr(loaded, "__file__", "")).resolve() == Path(__file__).parent / "__init__.py":
    del sys.modules["benchmark_v2"]

from benchmark_v2.e7.aggregate import (
    IncompleteEvidenceError,
    aggregate,
    external_context_rows,
    internal_ablation_rows,
)
from benchmark_v2.e7.capability_matrix import build_capability_matrix
from benchmark_v2.e7.config_audit import audit_internal_config
from benchmark_v2.e7.constants import (
    EXCLUDED_MODEL_IDS,
    EXTERNAL_MODEL_IDS,
    WORKBOOK_SHEETS,
)
from benchmark_v2.e7.evidence import _metrics_equal, build_missing_workbook_manifest
from benchmark_v2.e7.graph_extractors import side_effect_audit
from benchmark_v2.e7.graph_statistics import matched_edge_overlap, row_top_k, validate_matrix
from benchmark_v2.e7.io_utils import finite_number, forbidden_source
from benchmark_v2.e7.original26_loader import load_original26
from benchmark_v2.e7.readiness import build_readiness
from benchmark_v2.e7.xlsx_writer import write_workbook


RUN_IDS = {
    "gcn": "GCN_native_agcn_h64_l2_bs4_seed2026",
    "stgcn": "STGCN_native_cheb3_kt3_b2_bs4_seed2026",
    "dcrnn": "DCRNN_native_dualrw_k2_h64_l2_ar_bs4_seed2026",
    "mtgnn": "MTGNN_native_adaptive_k20_gdep2_l3_bs4_seed2026",
    "stid": "STID_native_node32_tid32_diw32_mlp3_bs4_seed2026",
    "tsmixer": "TSMixer_node_shared_el2_d32_bs4_seed2026",
}


def _cell(reference: str, value) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return f'<c r="{reference}"><v>{value}</v></c>'
    return f'<c r="{reference}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'


def _column(index: int) -> str:
    value = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        value = chr(65 + remainder) + value
    return value


def _sheet(rows: list[list]) -> str:
    rendered = []
    for row_index, row in enumerate(rows, 1):
        cells = "".join(_cell(f"{_column(col)}{row_index}", value) for col, value in enumerate(row, 1))
        rendered.append(f'<row r="{row_index}">{cells}</row>')
    return '<?xml version="1.0" encoding="UTF-8"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>' + "".join(rendered) + "</sheetData></worksheet>"


def _write_original26_fixture(path: Path, *, bad_value=None) -> None:
    main_header = ["实验组", "模型", "model_id", "Horizon", "MAE", "RMSE", "R2", "Score", "metric_status", "run_status", "protocol_status", "source_file", "parameter_count"]
    detail_header = ["实验组", "模型", "model_id", "运行状态", "协议检查", "H3 MAE", "H3 RMSE", "H3 R²", "H3 Score", "H6 MAE", "H6 RMSE", "H6 R²", "H6 Score", "H10 MAE", "H10 RMSE", "H10 R²", "H10 Score"]
    main = [main_header]
    detail = [detail_header]
    for model_index, model_id in enumerate(EXTERNAL_MODEL_IDS, 1):
        values = {}
        for horizon in (3, 6, 10):
            metrics = {"MAE": 70 + model_index + horizon, "RMSE": 100 + model_index + horizon, "R2": 0.9 - model_index / 100, "Score": 7 + model_index + horizon / 10}
            if bad_value is not None and model_id == "gcn" and horizon == 3:
                metrics["Score"] = bad_value
            values[horizon] = metrics
            main.append(["fixture", RUN_IDS[model_id], model_id, horizon, metrics["MAE"], metrics["RMSE"], metrics["R2"], metrics["Score"], "OK", "COMPLETED", "PASS", f"fixture/{RUN_IDS[model_id]}/metrics.csv", 1000 + model_index])
        detail.append(["fixture", RUN_IDS[model_id], model_id, "COMPLETED", "PASS", values[3]["MAE"], values[3]["RMSE"], values[3]["R2"], values[3]["Score"], values[6]["MAE"], values[6]["RMSE"], values[6]["R2"], values[6]["Score"], values[10]["MAE"], values[10]["RMSE"], values[10]["R2"], values[10]["Score"]])
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>')
        archive.writestr("xl/workbook.xml", '<?xml version="1.0"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="原始指标" sheetId="1" r:id="rId1"/><sheet name="模型明细" sheetId="2" r:id="rId2"/></sheets></workbook>')
        archive.writestr("xl/_rels/workbook.xml.rels", '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Target="worksheets/sheet2.xml"/></Relationships>')
        archive.writestr("xl/worksheets/sheet1.xml", _sheet(main))
        archive.writestr("xl/worksheets/sheet2.xml", _sheet(detail))


def _metrics(value: float) -> dict[int, dict[str, float]]:
    return {horizon: {"Score": value + horizon, "MAE": value + horizon + 1, "RMSE": value + horizon + 2, "R2": 0.9 - value / 100} for horizon in (3, 6, 10)}


def _complete_evidence() -> list[dict]:
    rows = [{"evidence_id": "E7_INTERNAL_A0", "model_id": "st_mgprompt", "display_name": "ST-MGPrompt A0", "variant_id": "A0", "metrics": _metrics(1), "resource_stats": {}}]
    for index, variant in enumerate(("A1", "A2", "A3"), 2):
        rows.append({"evidence_id": f"E7_INTERNAL_{variant}", "model_id": "st_mgprompt", "display_name": variant, "variant_id": variant, "metrics": _metrics(index), "resource_stats": {}})
    for index, model_id in enumerate(EXTERNAL_MODEL_IDS, 5):
        rows.append({"evidence_id": f"E7_EXTERNAL_{model_id.upper()}", "model_id": model_id, "display_name": model_id.upper(), "variant_id": None, "workbook_metrics": _metrics(index), "resource_stats": {}})
    return rows


def test_original26_schema_fixture(tmp_path: Path):
    fixture = tmp_path / "original26_fixture.xlsx"
    _write_original26_fixture(fixture)
    models, audit = load_original26(fixture)
    assert tuple(models) == EXTERNAL_MODEL_IDS
    assert audit["schema_valid"] and audit["selected_row_count"] == 18


def test_original26_nonfinite_is_never_zero(tmp_path: Path):
    fixture = tmp_path / "original26_nan.xlsx"
    _write_original26_fixture(fixture, bad_value="NaN")
    with pytest.raises(ValueError, match="non-finite"):
        load_original26(fixture)
    with pytest.raises(ValueError):
        finite_number(None)


def test_frozen_model_whitelist_and_excluded_rejection():
    assert EXTERNAL_MODEL_IDS == ("gcn", "stgcn", "dcrnn", "mtgnn", "stid", "tsmixer")
    assert set(EXCLUDED_MODEL_IDS) == {"persistence", "gru", "dlinear", "crossformer", "graph_wavenet", "agcrn", "segrnn", "msgnet"}
    assert set(EXTERNAL_MODEL_IDS).isdisjoint(EXCLUDED_MODEL_IDS)
    assert forbidden_source("x/results_smoke/gcn") == "results_smoke"
    assert forbidden_source("x/batch32/gcn") == "batch32"
    assert forbidden_source("x/e5/common_loss/gcn") in {"e5", "common_loss"}


def test_internal_config_diff_exact_targets():
    audit = audit_internal_config()
    assert audit["status"] == "PASS"
    assert [row["allowed_differences"] for row in audit["comparisons"]] == [["use_graph_in_temporal_encoder"], ["use_adaptive_graph"], ["graph_operator"]]
    assert not any(row["CONFIG_DIFF_INVALID"] for row in audit["comparisons"])


def test_readiness_missing_evidence_and_require_complete(tmp_path: Path):
    manifest = build_missing_workbook_manifest()
    readiness = build_readiness(manifest, grouped_status="BLOCKED_DEFINITION_MISSING", diagnostics_status="NOT_RUN", node_status="NOT_RUN")
    assert readiness["CORE_E7_READY"] != "10/10"
    with pytest.raises(IncompleteEvidenceError):
        aggregate(manifest, readiness, output_root=tmp_path, require_complete=True)
    assert not (tmp_path / "GRAPH_MECHANISM_ANALYSIS.xlsx").exists()


def test_metric_directions_and_all_horizons():
    evidence = _complete_evidence()
    internal = internal_ablation_rows(evidence)
    assert {row["horizon"] for row in internal} == {3, 6, 10}
    assert all(row["Score_degradation_absolute"] > 0 for row in internal)
    assert all(row["r2_drop"] > 0 for row in internal)
    external = external_context_rows(evidence)
    assert all("Score_rank" in row and "R2_rank" in row for row in external)
    assert next(row for row in external if row["variant_id"] == "A0" and row["horizon"] == 3)["Score_rank"] == 1


def test_original26_artifact_mismatch_is_detected():
    left, right = _metrics(1), _metrics(1)
    right[10]["Score"] += 0.01
    assert not _metrics_equal(left, right)


def test_graph_shape_node_order_and_matched_budget():
    matrix = np.zeros((134, 134))
    for row in range(134):
        for offset in range(1, 6):
            matrix[row, (row + offset) % 134] = 6 - offset
    assert validate_matrix(matrix).shape == (134, 134)
    top4 = row_top_k(matrix, 4)
    assert np.all(np.count_nonzero(top4, axis=1) == 4)
    overlap = matched_edge_overlap(matrix, matrix, k=4)
    assert overlap["prior_edges"] == overlap["learned_edges"] == 134 * 4
    assert overlap["jaccard"] == overlap["precision"] == overlap["recall"] == 1.0
    with pytest.raises(ValueError):
        validate_matrix(np.zeros((133, 133)))


def test_graph_free_controls_are_not_applicable():
    rows = {row["model_id"]: row for row in build_capability_matrix()}
    for model_id in ("stid", "tsmixer"):
        assert rows[model_id]["uses_fixed_graph"] == "NOT_APPLICABLE"
        assert rows[model_id]["effective_graph_extraction_supported"] == "NOT_APPLICABLE"


def test_diagnostic_side_effect_audit(tmp_path: Path):
    model = torch.nn.Linear(2, 1)
    checkpoint = tmp_path / "best_checkpoint.pt"
    torch.save(model.state_dict(), checkpoint)
    result = side_effect_audit(model, lambda: model.weight.detach().clone(), checkpoint=checkpoint, input_shape=(1, 2))
    assert result["status"] == "PASS"
    assert result["parameters_and_buffers_unchanged"]
    assert result["checkpoint_file_unchanged"] and result["mode_restored"]


def test_excel_schema_with_artifact_tool(tmp_path: Path):
    target = tmp_path / "GRAPH_MECHANISM_ANALYSIS.xlsx"
    write_workbook(target, {name: [["status"], ["fixture"]] for name in WORKBOOK_SHEETS})
    with zipfile.ZipFile(target) as archive:
        root = ET.fromstring(archive.read("xl/workbook.xml"))
    namespace = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    names = tuple(sheet.attrib["name"] for sheet in root.findall(".//m:sheet", namespace))
    assert names == WORKBOOK_SHEETS
