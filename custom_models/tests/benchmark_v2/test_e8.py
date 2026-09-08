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

from benchmark_v2.e8.aggregate import (
    IncompleteEvidenceError, aggregate, direction_decomposition_rows, external_context_rows, internal_ablation_rows,
)
from benchmark_v2.e8.config_audit import audit_internal_config
from benchmark_v2.e8.constants import EXCLUDED_MODEL_IDS, EXTERNAL_MODEL_IDS, INTERNAL_ROOT, WORKBOOK_SHEETS
from benchmark_v2.e8.cross_fusion_analysis import NOT_APPLICABLE, attention_summary
from benchmark_v2.e8.diagnostic_hooks import DiagnosticCapture, DiagnosticMode, side_effect_audit
from benchmark_v2.e8.evidence import build_missing_workbook_manifest, metrics_equal
from benchmark_v2.e8.io_utils import finite_number, forbidden_source
from benchmark_v2.e8.original26_loader import discover_original26, load_original26
from benchmark_v2.e8.paired_window_analysis import align_window_identity, paired_signal_analysis
from benchmark_v2.e8.readiness import build_readiness
from benchmark_v2.e8.representation_analysis import deterministic_align, linear_cka, pooled_cosine
from benchmark_v2.e8.xlsx_writer import write_workbook

RUN_IDS = {
    "patchtst": "PatchTST_node_shared_p16_s8_bs4_seed2026",
    "itransformer": "iTransformer_node_shared_d512_bs4_seed2026",
    "timexer": "TimeXer_node_shared_ms_d512_bs4_seed2026",
    "multipatchformer": "MultiPatchFormer_node_shared_p8_16_24_32_d256_bs4_seed2026",
    "timemixer": "TimeMixer_node_shared_ci1_ds3_w2_avg_d16_bs4_seed2026",
    "timefilter": "TimeFilter_node_shared_p16_d512_el2_tp0p5_bs4_seed2026",
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
    main, detail = [main_header], [detail_header]
    for model_index, model_id in enumerate(EXTERNAL_MODEL_IDS, 1):
        values = {}
        for horizon in (3, 6, 10):
            metrics = {"MAE": 70 + model_index + horizon, "RMSE": 100 + model_index + horizon, "R2": 0.9 - model_index / 100, "Score": 7 + model_index + horizon / 10}
            if bad_value is not None and model_id == "patchtst" and horizon == 3:
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
    return {h: {"Score": value + h, "MAE": value + h + 1, "RMSE": value + h + 2, "R2": 0.95 - value / 100} for h in (3, 6, 10)}


def _complete_evidence() -> list[dict]:
    rows = [{"evidence_id": "E8_INTERNAL_A0", "model_id": "st_mgprompt", "display_name": "ST-MGPrompt A0", "variant_id": "A0", "metrics": _metrics(1), "resource_stats": {}}]
    for index, variant in enumerate(("A4", "A5", "A6", "A7"), 2):
        rows.append({"evidence_id": f"E8_INTERNAL_{variant}", "model_id": "st_mgprompt", "display_name": variant, "variant_id": variant, "metrics": _metrics(index), "resource_stats": {}})
    for index, model_id in enumerate(EXTERNAL_MODEL_IDS, 7):
        rows.append({"evidence_id": f"E8_EXTERNAL_{model_id.upper()}", "model_id": model_id, "display_name": model_id, "variant_id": None, "workbook_metrics": _metrics(index), "resource_stats": {}})
    return rows


def test_original26_schema_fixture_and_nan_rejection(tmp_path: Path):
    fixture = tmp_path / "original26_fixture.xlsx"
    _write_original26_fixture(fixture)
    models, audit = load_original26(fixture)
    assert tuple(models) == EXTERNAL_MODEL_IDS and audit["selected_row_count"] == 18
    bad = tmp_path / "original26_bad.xlsx"
    _write_original26_fixture(bad, bad_value="NaN")
    with pytest.raises(ValueError, match="non-finite"):
        load_original26(bad)
    with pytest.raises(ValueError):
        finite_number(None)


def test_original26_missing_discovery_fail_closed(tmp_path: Path):
    path, errors = discover_original26(tmp_path)
    assert path is None and errors == ["MISSING_ORIGINAL26_XLSX"]


def test_fixed_whitelist_excluded_and_e5_rejection():
    assert EXTERNAL_MODEL_IDS == ("patchtst", "itransformer", "timexer", "multipatchformer", "timemixer", "timefilter")
    assert set(EXCLUDED_MODEL_IDS) == {"persistence", "gru", "dlinear", "crossformer", "graph_wavenet", "agcrn", "segrnn", "msgnet"}
    assert set(EXTERNAL_MODEL_IDS).isdisjoint(EXCLUDED_MODEL_IDS)
    for path in ("x/results_smoke/patchtst", "x/batch32/patchtst", "x/e5/common_loss/patchtst"):
        assert forbidden_source(path) is not None


def test_config_diff_and_redesigned_semantics():
    audit = audit_internal_config()
    assert audit["status"] == "PASS"
    assert [row["allowed_differences"] for row in audit["comparisons"]] == [["macro_prompt_pooling"], ["cross_fusion_recent_len"], ["fusion_mode"], ["st_prompt_use_node_identity"]]
    assert not audit["direction_decomposition"]["DIRECTION_DECOMPOSITION_VALID"]
    assert "both attention directions" in audit["source_semantics"]["A6_change"]


def test_config_audit_rejects_extra_difference(tmp_path: Path):
    manifest = json.loads((INTERNAL_ROOT / "experiment_manifest.json").read_text(encoding="utf-8"))
    (tmp_path / "experiment_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for variant in ("A0", "A4", "A5", "A6", "A7"):
        target = tmp_path / variant
        target.mkdir()
        config = json.loads((INTERNAL_ROOT / variant / "effective_config.json").read_text(encoding="utf-8"))
        if variant == "A6":
            config["hidden_dim"] = 65
        (target / "effective_config.json").write_text(json.dumps(config), encoding="utf-8")
        (target / "effective_config_diff.json").write_text("{}", encoding="utf-8")
    audit = audit_internal_config(tmp_path)
    assert audit["status"] == "FAIL"
    assert next(row for row in audit["comparisons"] if row["variant_id"] == "A6")["CONFIG_DIFF_INVALID"]


def test_readiness_and_require_complete_fail_closed(tmp_path: Path):
    manifest = build_missing_workbook_manifest()
    readiness = build_readiness(manifest, representation_status="BLOCKED", cross_status="BLOCKED", macro_status="BLOCKED", st_status="BLOCKED", paired_status="BLOCKED", grouped_status="BLOCKED_DEFINITION_MISSING")
    assert readiness["CORE_E8_READY"] != "11/11"
    with pytest.raises(IncompleteEvidenceError):
        aggregate(manifest, readiness, audit_internal_config(), output_root=tmp_path, require_complete=True)
    assert not (tmp_path / "PROMPT_CROSS_FUSION_ANALYSIS.xlsx").exists()


def test_metric_directions_rank_and_direction_rows():
    evidence = _complete_evidence()
    audit = audit_internal_config()
    internal = internal_ablation_rows(evidence, audit)
    assert {row["horizon"] for row in internal} == {3, 6, 10}
    assert all(row["Score_degradation_absolute"] > 0 and row["r2_drop"] > 0 for row in internal)
    directions = direction_decomposition_rows(evidence, audit)
    assert directions == []
    external = external_context_rows(evidence)
    assert all("Score_rank" in row and "R2_rank" in row for row in external)


def test_original26_artifact_mismatch_detection():
    left, right = _metrics(1), _metrics(1)
    right[10]["Score"] += 0.01
    assert not metrics_equal(left, right)


def test_diagnostic_hook_shape_and_side_effect(tmp_path: Path):
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.block = torch.nn.Linear(3, 2)
            self.diagnostics_level = "none"
        def forward(self, x):
            return self.block(x)
    model, value = Toy(), torch.ones(4, 3)
    with DiagnosticCapture(model, {"representation": ("block", "output", None)}) as capture:
        model(value)
    assert capture.captured["representation"].shape == (4, 2)
    with DiagnosticMode(model):
        assert model.diagnostics_level == "full"
    assert model.diagnostics_level == "none"
    checkpoint = tmp_path / "best_checkpoint.pt"
    torch.save(model.state_dict(), checkpoint)
    def predict(enabled):
        if enabled:
            with DiagnosticCapture(model, {"representation": ("block", "output", None)}):
                return model(value)
        return model(value)
    result = side_effect_audit(model, predict, checkpoint=checkpoint)
    assert result["status"] == "PASS" and result["parameters_unchanged"] and result["buffers_unchanged"]


def test_alignment_cosine_and_cka_numerics():
    rng = np.random.default_rng(2026)
    left = rng.normal(size=(8, 12, 5))
    right = left.copy()
    assert np.allclose(pooled_cosine(left, right), 1.0)
    assert linear_cka(left, right) == pytest.approx(1.0)
    a, b, audit = deterministic_align(left, rng.normal(size=(8, 6, 5)))
    assert a.shape == b.shape == (8, 6, 5) and audit["method"] == "deterministic_equal_bins_mean"


def test_attention_not_applicable_no_fabrication():
    result = attention_summary(None)
    assert result["status"] == NOT_APPLICABLE and result["attention_entropy"] == NOT_APPLICABLE


def test_window_identity_mask_and_paired_analysis():
    ids = ["w1", "w2", "w3", "w4"]
    left = np.arange(8).reshape(4, 2)
    mask = np.array([[1, 1], [1, 0], [0, 1], [1, 1]], dtype=bool)
    a, b, aligned_mask = align_window_identity(ids, ids, left, left + 1, mask, mask.copy())
    assert a.shape == b.shape and np.array_equal(mask, aligned_mask)
    assert paired_signal_analysis(np.arange(8), np.arange(8))["spearman"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="WINDOW_IDENTITY_MISMATCH"):
        align_window_identity(ids, list(reversed(ids)), left, left)
    with pytest.raises(ValueError, match="MASK_IDENTITY_MISMATCH"):
        align_window_identity(ids, ids, left, left, mask, ~mask)


def test_excel_schema_with_artifact_tool(tmp_path: Path):
    target = tmp_path / "PROMPT_CROSS_FUSION_ANALYSIS.xlsx"
    write_workbook(target, {name: [["status"], ["fixture"]] for name in WORKBOOK_SHEETS})
    with zipfile.ZipFile(target) as archive:
        root = ET.fromstring(archive.read("xl/workbook.xml"))
    namespace = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    names = tuple(sheet.attrib["name"] for sheet in root.findall(".//m:sheet", namespace))
    assert names == WORKBOOK_SHEETS
