from __future__ import annotations

import json
import math
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape

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

from benchmark_v2.contracts import LossInputBundle
from benchmark_v2.checkpointing import CheckpointManager
from benchmark_v2.e9.aggregate import IncompleteE9EvidenceError, aggregate_complete, build_tables, metric_comparison
from benchmark_v2.e9.config_diff import build_transfer_config, compare_configs
from benchmark_v2.e9.constants import EXCLUDED_MODEL_IDS, MODEL_IDS, WORKBOOK_SHEETS
from benchmark_v2.e9.loss_contract import loss_input_contract, loss_state_manifest
from benchmark_v2.e9.original26_resolver import control_identity_errors, discover_original26, load_workbook_controls
from benchmark_v2.e9.portability_audit import classify_portability, portability_audit
from benchmark_v2.e9.evidence import build_evidence_manifest, build_transfer_readiness
from benchmark_v2.e9.readiness import build_readiness
from benchmark_v2.e9.variants import CONTROL_RUN_IDS, canonical_transfer_run_id, variant_manifest
from benchmark_v2.e9.xlsx_writer import write_workbook
from benchmark_v2.losses import build_msmg_dwu_loss


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


def _write_original26_fixture(path: Path, *, missing_model: str | None = None, bad_status: bool = False) -> None:
    main_header = ["实验组", "模型", "model_id", "Horizon", "MAE", "RMSE", "R2", "Score", "metric_status", "run_status", "protocol_status", "source_file"]
    detail_header = ["实验组", "模型", "model_id", "运行状态", "协议检查", "H3 MAE", "H3 RMSE", "H3 R²", "H3 Score", "H6 MAE", "H6 RMSE", "H6 R²", "H6 Score", "H10 MAE", "H10 RMSE", "H10 R²", "H10 Score"]
    main, detail = [main_header], [detail_header]
    for model_index, model_id in enumerate(MODEL_IDS, 1):
        if model_id == missing_model:
            continue
        values = {}
        for horizon in (3, 6, 10):
            metrics = {"MAE": 70 + model_index + horizon, "RMSE": 100 + model_index + horizon, "R2": 0.9 - model_index / 100, "Score": 7 + model_index + horizon / 10}
            values[horizon] = metrics
            status = "FAILED" if bad_status and model_id == "lightts" else "COMPLETED"
            main.append(["fixture", CONTROL_RUN_IDS[model_id], model_id, horizon, metrics["MAE"], metrics["RMSE"], metrics["R2"], metrics["Score"], "OK", status, "PASS", f"fixture/{CONTROL_RUN_IDS[model_id]}/metrics.csv"])
        detail.append(["fixture", CONTROL_RUN_IDS[model_id], model_id, "COMPLETED", "PASS", values[3]["MAE"], values[3]["RMSE"], values[3]["R2"], values[3]["Score"], values[6]["MAE"], values[6]["RMSE"], values[6]["R2"], values[6]["Score"], values[10]["MAE"], values[10]["RMSE"], values[10]["R2"], values[10]["Score"]])
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>')
        archive.writestr("xl/workbook.xml", '<?xml version="1.0"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="原始指标" sheetId="1" r:id="rId1"/><sheet name="模型明细" sheetId="2" r:id="rId2"/></sheets></workbook>')
        archive.writestr("xl/_rels/workbook.xml.rels", '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Target="worksheets/sheet2.xml"/></Relationships>')
        archive.writestr("xl/worksheets/sheet1.xml", _sheet(main))
        archive.writestr("xl/worksheets/sheet2.xml", _sheet(detail))


def _bundle(mask: torch.Tensor) -> LossInputBundle:
    prediction = torch.randn(*mask.shape, requires_grad=True)
    target = torch.randn(*mask.shape)
    return LossInputBundle(prediction, target, mask, tuple(range(mask.shape[1])), tuple(range(1, mask.shape[2] + 1)), "normalized_target_space", "sdwpf_benchmark_v2", "train")


def _evidence(complete: bool = True):
    rows = []
    for index, model_id in enumerate(MODEL_IDS):
        metrics_control = {h: {"Score": 10 + index + h / 10, "MAE": 100 + h, "RMSE": 150 + h, "R2": 0.7} for h in (3, 6, 10)}
        metrics_transfer = {h: {"Score": 9 + index + h / 10, "MAE": 90 + h, "RMSE": 140 + h, "R2": 0.72} for h in (3, 6, 10)}
        rows.append({"evidence_role": "ORIGINAL26_CONTROL", "model_id": model_id, "run_id": CONTROL_RUN_IDS[model_id], "metrics": metrics_control, "ready": True})
        ready = complete or model_id != MODEL_IDS[-1]
        rows.append({"evidence_role": "E9_TRANSFER", "model_id": model_id, "run_id": canonical_transfer_run_id(model_id), "metrics": metrics_transfer if ready else None, "ready": ready, "loss_only_diff": {"loss_only_diff_valid": ready}})
    return {"evidence": rows}


def test_scope_exact_six_and_exclusions():
    assert MODEL_IDS == ("lightts", "tide", "patchtst", "itransformer", "dcrnn", "mtgnn")
    assert set(MODEL_IDS).isdisjoint(EXCLUDED_MODEL_IDS)
    manifest = variant_manifest()
    assert [row["model_id"] for row in manifest["variants"]] == list(MODEL_IDS)
    assert all("e5" not in json.dumps(row).lower() for row in manifest["variants"])


def test_portability_and_contract():
    audit = portability_audit()
    assert audit["classification"] == "PORTABLE_WITH_TRAIN_STATE"
    assert audit["FULL_MS_MG_DWU_TRANSFER_READY"] and not audit["portable_subset_required"]
    required = {row["dependency"]: row["required"] for row in audit["dependency_closure"]}
    assert required["prediction"] and required["training-only fitted statistics"]
    assert not required["Fine representation"] and not required["hidden state"] and not required["Macro Prompt"]
    contract = loss_input_contract()
    assert contract["validation_status"] == "PASS"
    assert "hidden_states" in contract["forbidden_fields"]


def test_portability_classifier_fail_closed_and_subset_naming():
    assert classify_portability(requires_train_state=False, model_internal_dependencies=[])["classification"] == "FULLY_PORTABLE"
    assert classify_portability(requires_train_state=True, model_internal_dependencies=[])["classification"] == "PORTABLE_WITH_TRAIN_STATE"
    blocked = classify_portability(requires_train_state=False, model_internal_dependencies=["hidden_state"])
    assert blocked["classification"] == "NOT_PORTABLE" and not blocked["e9_b_allowed"]
    with pytest.raises(ValueError, match="new explicit loss name"):
        classify_portability(
            requires_train_state=False, model_internal_dependencies=["Fine"],
            semantics_preserved_without_internal=True, portable_subset_name="msmg_dwu",
        )
    subset = classify_portability(
        requires_train_state=False, model_internal_dependencies=["Fine"],
        semantics_preserved_without_internal=True, portable_subset_name="portable_msmg_dwu_v1",
    )
    assert subset["classification"] == "MODEL_INTERNAL_DEPENDENT"
    assert not subset["FULL_MS_MG_DWU_TRANSFER_READY"] and subset["PORTABLE_SUBSET_TRANSFER_READY"]


def test_loss_finite_backward_masks_and_frozen_eval_state():
    loss_fn = build_msmg_dwu_loss(eval_horizons=[3, 6, 10], num_nodes=4, granularity_weight_mode="difficulty_rate")
    partial = torch.ones(2, 4, 10, dtype=torch.bool)
    partial[:, 0, 0] = False
    bundle = _bundle(partial)
    value = loss_fn(bundle)
    assert value is not None and torch.isfinite(value)
    value.backward()
    assert torch.isfinite(bundle.prediction.grad).all()
    before = {key: tensor.clone() for key, tensor in loss_fn.state_dict().items()}
    loss_fn.eval()
    eval_bundle = _bundle(partial)
    assert loss_fn(eval_bundle) is not None
    after = loss_fn.state_dict()
    assert all(torch.equal(before[key], after[key]) for key in before)
    assert loss_fn(_bundle(torch.zeros(2, 4, 10, dtype=torch.bool))) is None


def test_leakage_manifest():
    state = loss_state_manifest()
    assert state["fit_scope"] == "training batches only"
    assert not state["uses_validation"] and not state["uses_test"] and not state["uses_test_target"]
    assert not state["updates_in_validation"] and not state["updates_in_test"]


def test_original26_resolver_fixture_missing_ambiguous_and_status(tmp_path: Path):
    fixture = tmp_path / "original26_fixture.xlsx"
    _write_original26_fixture(fixture)
    controls, audit = load_workbook_controls(fixture)
    assert tuple(controls) == MODEL_IDS and audit["selected_row_count"] == 18
    empty = tmp_path / "empty"
    empty.mkdir()
    assert discover_original26(empty)[1] == ["MISSING_ORIGINAL26_XLSX"]
    (empty / "original26_a.xlsx").write_bytes(fixture.read_bytes())
    (empty / "original26_b.xlsx").write_bytes(fixture.read_bytes())
    assert discover_original26(empty)[1][0] == "AMBIGUOUS_ORIGINAL26_XLSX"
    missing = tmp_path / "missing.xlsx"
    _write_original26_fixture(missing, missing_model="mtgnn")
    with pytest.raises(ValueError, match="INVALID_HORIZONS"):
        load_workbook_controls(missing)
    bad = tmp_path / "bad.xlsx"
    _write_original26_fixture(bad, bad_status=True)
    with pytest.raises(ValueError, match="RUN_STATUS_INVALID"):
        load_workbook_controls(bad)


def test_control_identity_rejects_wrong_loss_and_display_name_cannot_bypass_model_id():
    effective = {
        "model_id": "display-name-only", "loss": "masked_score_aligned_hybrid",
        "training_batch_profile_id": "uniform_train_batch4_v1",
        "train_batch_size": 4, "val_batch_size": 4, "test_batch_size": 4,
    }
    errors = control_identity_errors(
        "lightts", effective=effective, resolved={"model_id": "lightts", "loss": "masked_mse"},
        status={"status": "COMPLETED", "run_mode": "formal"}, protocol={"status": "PASS"},
    )
    assert "INVALID_ORIGINAL_CONTROL_LOSS" in errors
    assert "MODEL_ID_ARTIFACT_MISMATCH" in errors


def _base_config():
    return {
        "model_id": "lightts", "loss": "masked_mse", "run_id": "control", "output_root": "control",
        "train_batch_size": 4, "val_batch_size": 4, "test_batch_size": 4, "optimizer": "Adam",
        "learning_rate": 0.001, "epochs": 20, "patience": 6, "hidden_dim": 64,
        "ordered_input_features": ["Wspd", "Patv_clean_for_input"], "dropout": 0.1,
        "training_batch_profile_id": "uniform_train_batch4_v1", "formal_training": True,
    }


def test_e5_named_source_is_forbidden_from_evidence(tmp_path: Path):
    controls = [
        {
            "model_id": model_id, "run_id": CONTROL_RUN_IDS[model_id], "validation_errors": [],
            "effective_config": {**_base_config(), "model_id": model_id}, "metrics": {}, "ready": True,
        }
        for model_id in MODEL_IDS
    ]
    root = tmp_path / "common_loss_architecture_seed2026"
    evidence = build_evidence_manifest({"controls": controls}, root)
    transfers = [row for row in evidence["evidence"] if row["evidence_role"] == "E9_TRANSFER"]
    assert transfers and all("FORBIDDEN_TRANSFER_SOURCE" in row["validation_errors"] for row in transfers)


@pytest.mark.parametrize("field,value", [("train_batch_size", 2), ("hidden_dim", 32), ("optimizer", "SGD"), ("learning_rate", 0.01), ("epochs", 30), ("patience", 9), ("ordered_input_features", ["Patv_clean_for_input"])])
def test_config_diff_rejects_non_loss_changes(field, value):
    control = _base_config()
    transfer = build_transfer_config(control, "lightts", "transfer")
    transfer[field] = value
    assert not compare_configs(control, transfer)["loss_only_diff_valid"]


def test_config_diff_only_loss_provenance_passes_and_graph_fails():
    control = _base_config()
    control["formal_worker_pid"] = 12345
    transfer = build_transfer_config(control, "lightts", "transfer")
    assert "formal_worker_pid" not in transfer
    assert compare_configs(control, transfer)["loss_only_diff_valid"]
    graph = {**control, "model_id": "dcrnn", "graph_id": "g1", "ordered_node_ids": [1, 2]}
    graph_transfer = build_transfer_config(graph, "dcrnn", "transfer")
    graph_transfer["graph_id"] = "g2"
    assert not compare_configs(graph, graph_transfer, graph_model=True)["loss_only_diff_valid"]


def test_six_variants_construct_same_loss_profile():
    for model_id in MODEL_IDS:
        config = build_transfer_config({**_base_config(), "model_id": model_id}, model_id, "transfer")
        assert config["loss"] == "msmg_dwu_loss"
        assert config["loss_profile"]["granularity_weight_mode"] == "difficulty_rate"


def test_readiness_fail_closed_and_smoke_not_evidence(tmp_path: Path):
    portability = {"e9_b_allowed": True, "FULL_MS_MG_DWU_TRANSFER_READY": True, "PORTABLE_SUBSET_TRANSFER_READY": False}
    incomplete = _evidence(complete=False)
    readiness = build_readiness(portability, incomplete)
    assert readiness["ORIGINAL_MASKED_MSE_REFERENCE_READY"] == "6/6"
    assert readiness["MSMG_DWU_TRANSFER_READY"] == "5/6"
    assert readiness["CORE_E9_READY"] == "11/12" and not readiness["CORE_E9_READY_BOOL"]
    with pytest.raises(IncompleteE9EvidenceError):
        aggregate_complete(incomplete, readiness, output_root=tmp_path, require_complete=True)
    assert not (tmp_path / "MSMG_DWU_TRANSFER_ANALYSIS.xlsx").exists()


def test_transfer_only_readiness_is_separate_and_fail_closed(tmp_path: Path):
    status = build_transfer_readiness(tmp_path)
    assert status["MSMG_DWU_TRANSFER_READY"] == "0/6"
    assert not status["MSMG_DWU_TRANSFER_READY_BOOL"]
    assert status["requires_original26_for_core_e9"]


def test_checkpoint_roundtrip_restores_loss_training_state(tmp_path: Path):
    model = torch.nn.Linear(2, 2)
    loss_fn = build_msmg_dwu_loss(eval_horizons=[3, 6, 10], num_nodes=4, granularity_weight_mode="difficulty_rate")
    value = loss_fn(_bundle(torch.ones(2, 4, 10, dtype=torch.bool)))
    assert value is not None
    expected = {key: tensor.clone() for key, tensor in loss_fn.state_dict().items()}
    manager = CheckpointManager(tmp_path, protocol_id="fixture", model_id="lightts", resolved_config={}, effective_config={})
    checkpoint = manager.save("state.pt", epoch=1, global_step=1, monitor_value=1.0, model=model, loss_fn=loss_fn)
    restored_model = torch.nn.Linear(2, 2)
    restored_loss = build_msmg_dwu_loss(eval_horizons=[3, 6, 10], num_nodes=4, granularity_weight_mode="difficulty_rate")
    manager.load(checkpoint, restored_model, loss_fn=restored_loss)
    assert all(torch.equal(expected[key], restored_loss.state_dict()[key]) for key in expected)


def test_metric_directions_and_all_horizons():
    delta, improvement = metric_comparison(10.0, 8.0, "Score")
    assert delta == 2.0 and improvement == 20.0
    r2_delta, r2_gain = metric_comparison(0.7, 0.75, "R2")
    assert r2_delta == pytest.approx(0.05) and r2_gain == pytest.approx(0.05)
    tables = build_tables(_evidence())
    assert {row["horizon"] for row in tables["e9_transfer_metrics.csv"]} == {3, 6, 10}
    assert all(row["improvement_pct_Score"] > 0 for row in tables["e9_transfer_metrics.csv"])


def test_excel_schema_with_artifact_tool(tmp_path: Path):
    target = tmp_path / "NOT_FORMAL_E9_FIXTURE.xlsx"
    payload = {"readiness": {"pairs": []}, "tables": {}, "evidence": [], "audit": {}}
    write_workbook(target, payload)
    with zipfile.ZipFile(target) as archive:
        root = ET.fromstring(archive.read("xl/workbook.xml"))
    namespace = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    names = tuple(sheet.attrib["name"] for sheet in root.findall(".//m:sheet", namespace))
    assert names == WORKBOOK_SHEETS


def test_linux_scripts_scope_paths_and_shutdown_order():
    docs = PROJECT_ROOT / "custom_models" / "docs" / "benchmark_v2" / "E9"
    names = (
        "E9_PRECHECK_LINUX.sh", "E9_PREFLIGHT_ALL_6_LINUX.sh", "E9_RUN_ALL_6_LINUX.sh",
        "E9_RUN_ALL_6_LINUX_AUTOSHUTDOWN.sh", "E9_READINESS_AND_AGGREGATE_LINUX.sh",
    )
    contents = {name: (docs / name).read_text(encoding="utf-8") for name in names}
    for content in contents.values():
        assert "/root/autodl-tmp/GyxPaper2" in content
        assert "D:\\" not in content and "D:/" not in content
    run_all = contents["E9_RUN_ALL_6_LINUX.sh"]
    assert "E9_VARIANT_MANIFEST.json" in run_all and "train-one" in run_all and "transfer-readiness" in run_all
    assert "evaluate-only" not in run_all and "STMGPrompt" not in run_all
    auto = contents["E9_RUN_ALL_6_LINUX_AUTOSHUTDOWN.sh"]
    assert auto.index("E9_RUN_ALL_6_LINUX.sh") < auto.index("run_code=$?") < auto.index("sync") < auto.index("/usr/bin/shutdown -h now")
