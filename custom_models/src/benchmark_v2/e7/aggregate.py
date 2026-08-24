from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .capability_matrix import build_capability_matrix
from .constants import ANALYSIS_ROOT, HORIZONS, LOWER_IS_BETTER, METRICS, WORKBOOK_SHEETS
from .io_utils import write_csv, write_json
from .xlsx_writer import write_workbook


class IncompleteEvidenceError(RuntimeError):
    pass


def internal_ablation_rows(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_variant = {row["variant_id"]: row for row in evidence if row.get("variant_id")}
    a0 = by_variant["A0"]["metrics"]
    rows = []
    for variant in ("A1", "A2", "A3"):
        for horizon in HORIZONS:
            row = {"variant": variant, "horizon": horizon}
            for metric in METRICS:
                baseline, ablation = a0[horizon][metric], by_variant[variant]["metrics"][horizon][metric]
                row[f"A0_{metric}"] = baseline
                row[f"ablation_{metric}"] = ablation
                if metric == "R2":
                    row["r2_drop"] = baseline - ablation
                else:
                    row[f"{metric}_degradation_absolute"] = ablation - baseline
                    row[f"{metric}_degradation_percent"] = (ablation - baseline) / baseline * 100.0
            rows.append(row)
    return rows


def external_context_rows(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    a0 = next(row for row in evidence if row.get("variant_id") == "A0")
    external = [row for row in evidence if row["evidence_id"].startswith("E7_EXTERNAL_")]
    rows = []
    for model in [a0, *external]:
        metrics = model.get("metrics") or model.get("workbook_metrics")
        for horizon in HORIZONS:
            row = {"model_id": model["model_id"], "display_name": model["display_name"], "variant_id": model.get("variant_id"), "horizon": horizon, **metrics[horizon]}
            row.update(model.get("resource_stats") or {})
            if model is not a0:
                for metric in LOWER_IS_BETTER:
                    row[f"st_mgprompt_{metric}_improvement_percent"] = (metrics[horizon][metric] - a0["metrics"][horizon][metric]) / metrics[horizon][metric] * 100.0
                row["R2_gain_A0_minus_baseline"] = a0["metrics"][horizon]["R2"] - metrics[horizon]["R2"]
            rows.append(row)
    for horizon in HORIZONS:
        horizon_rows = [row for row in rows if row["horizon"] == horizon]
        for metric in METRICS:
            ordered = sorted(horizon_rows, key=lambda row: row[metric], reverse=metric == "R2")
            for rank, row in enumerate(ordered, 1):
                row[f"{metric}_rank"] = rank
    mean_rank = {}
    for row in rows:
        mean_rank.setdefault(row["display_name"], []).append(row["Score_rank"])
    for row in rows:
        row["Score_mean_rank_H3_H6_H10"] = sum(mean_rank[row["display_name"]]) / 3.0
    return rows


def _table(rows: list[dict[str, Any]]) -> list[list]:
    if not rows:
        return [["status"], ["NOT_AVAILABLE"]]
    fields = list(rows[0])
    def scalar(value):
        return json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value
    return [fields, *[[scalar(row.get(field)) for field in fields] for row in rows]]


def _read_csv_rows(path: Path, status: str) -> list[dict[str, Any]]:
    if not path.is_file():
        return [{"status": status, "reason": f"{path.name} not available"}]
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json_rows(path: Path, status: str) -> list[dict[str, Any]]:
    if not path.is_file():
        return [{"status": status, "reason": f"{path.name} not available"}]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    return [{"section": key, "value": value} for key, value in payload.items()]


def aggregate(evidence_manifest: dict[str, Any], readiness: dict[str, Any], *, output_root: str | Path = ANALYSIS_ROOT, require_complete: bool = False) -> dict[str, Any]:
    root = Path(output_root)
    if not readiness["core_ready"]:
        if require_complete:
            raise IncompleteEvidenceError(f"CORE_E7_READY={readiness['CORE_E7_READY']}; refusing formal aggregate")
        preview = {"status": "BLOCKED_INCOMPLETE_EVIDENCE", "CORE_E7_READY": readiness["CORE_E7_READY"], "formal_outputs_created": False}
        write_json(root / "e7_aggregate_preview.json", preview)
        return preview
    evidence = evidence_manifest["evidence"]
    internal = internal_ablation_rows(evidence)
    external = external_context_rows(evidence)
    capability = build_capability_matrix()
    write_csv(root / "e7_internal_ablation.csv", internal)
    write_csv(root / "e7_original26_external_comparison.csv", external)
    write_csv(root / "E7_GRAPH_CAPABILITY_MATRIX.csv", capability)
    graph_stats = _read_csv_rows(root / "e7_graph_structure_statistics.csv", readiness["GRAPH_DIAGNOSTICS_READY"])
    overlaps = _read_csv_rows(root / "e7_prior_learned_overlap.csv", readiness["GRAPH_DIAGNOSTICS_READY"])
    distance = _read_csv_rows(root / "e7_distance_weight_analysis.csv", readiness["GRAPH_DIAGNOSTICS_READY"])
    node = _read_csv_rows(root / "e7_node_level_analysis.csv", readiness["NODE_ANALYSIS_READY"])
    grouped = _read_csv_rows(root / "e7_grouped_analysis.csv", readiness["GROUPED_ANALYSIS_READY"])
    if not (root / "e7_node_level_analysis.csv").is_file():
        write_csv(root / "e7_node_level_analysis.csv", node)
    if not (root / "e7_grouped_analysis.csv").is_file():
        write_csv(root / "e7_grouped_analysis.csv", grouped)
    readme = [
        ["E7 Graph Mechanism Analysis"],
        ["Internal Ablation is the primary causal evidence for graph components."],
        ["Original26 External Context is native-training-system end-to-end competitiveness/mechanism context, not a unified-loss structure-controlled comparison."],
        ["E5 is cancelled; no common-loss table exists; A8 is not E7 evidence."],
        ["Score/MAE/RMSE lower is better; R2 higher is better."],
        ["Missing values are never replaced with zero; all formal numbers are loaded automatically."],
    ]
    sheet_data = {
        "README": readme,
        "Evidence_Readiness": _table([readiness]),
        "Protocol_Audit": _table(_read_json_rows(root / "E7_PROTOCOL_AUDIT.json", "NOT_AVAILABLE")),
        "Internal_Ablation": _table(internal),
        "Original26_External_Context": _table(external),
        "Graph_Capability_Matrix": _table(capability),
        "Graph_Structure_Stats": _table(graph_stats),
        "Prior_Learned_Overlap": _table(overlaps),
        "Distance_Weight_Analysis": _table(distance),
        "Node_Level_Analysis": _table(node),
        "Conditional_Groups": _table(grouped),
        "Diagnostic_SideEffect_Audit": _table(_read_json_rows(root / "E7_DIAGNOSTIC_SIDE_EFFECT_AUDIT.json", readiness["GRAPH_DIAGNOSTICS_READY"])),
        "Artifact_Manifest": _table(_read_json_rows(root / "E7_ARTIFACT_MANIFEST.json", "NOT_AVAILABLE")),
    }
    assert tuple(sheet_data) == WORKBOOK_SHEETS
    workbook = write_workbook(root / "GRAPH_MECHANISM_ANALYSIS.xlsx", sheet_data)
    report = root / "GRAPH_MECHANISM_ANALYSIS.md"
    report.write_text("# E7 Graph Mechanism Analysis\n\nInternal ablation is the primary causal evidence. Original26 external comparisons provide native-training-system context only. E5 is cancelled and no common-loss structure-controlled table exists.\n", encoding="utf-8")
    return {"status": "COMPLETE", "workbook": str(workbook), "report": str(report)}
