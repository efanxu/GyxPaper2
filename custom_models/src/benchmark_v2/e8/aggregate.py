from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .capability_matrix import build_capability_matrix
from .constants import ANALYSIS_ROOT, HORIZONS, LOWER_IS_BETTER, METRICS, WORKBOOK_SHEETS
from .io_utils import write_csv, write_json
from .report import write_artifact_manifest, write_markdown_report
from .xlsx_writer import write_workbook


class IncompleteEvidenceError(RuntimeError):
    pass


def internal_ablation_rows(evidence: list[dict[str, Any]], config_audit: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    by_variant = {row["variant_id"]: row for row in evidence if row.get("variant_id")}
    a0 = by_variant["A0"]["metrics"]
    valid = {row["variant_id"]: row["config_diff_valid"] for row in (config_audit or {}).get("comparisons", [])}
    rows = []
    for variant in ("A4", "A5", "A6", "A7"):
        for horizon in HORIZONS:
            row = {"variant": variant, "horizon": horizon, "CONFIG_DIFF_INVALID": not valid.get(variant, True)}
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


def direction_decomposition_rows(evidence: list[dict[str, Any]], config_audit: dict[str, Any]) -> list[dict[str, Any]]:
    if not config_audit["direction_decomposition"]["DIRECTION_DECOMPOSITION_VALID"]:
        return []
    by_variant = {row["variant_id"]: row for row in evidence if row.get("variant_id")}
    comparisons = (
        ("A0_vs_A6", "A0", "A6", "Macro/Coarse-to-Fine cross interaction contribution"),
    )
    rows = []
    for comparison, reference, ablation, meaning in comparisons:
        for horizon in HORIZONS:
            row = {"comparison": comparison, "meaning": meaning, "horizon": horizon, "DIRECTION_DECOMPOSITION_VALID": True}
            for metric in METRICS:
                left, right = by_variant[reference]["metrics"][horizon][metric], by_variant[ablation]["metrics"][horizon][metric]
                row[f"reference_{metric}"] = left
                row[f"ablation_{metric}"] = right
                if metric == "R2":
                    row["r2_drop"] = left - right
                else:
                    row[f"{metric}_degradation_absolute"] = right - left
                    row[f"{metric}_degradation_percent"] = (right - left) / left * 100.0
            rows.append(row)
    return rows


def external_context_rows(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    a0 = next(row for row in evidence if row.get("variant_id") == "A0")
    external = [row for row in evidence if row["evidence_id"].startswith("E8_EXTERNAL_")]
    rows = []
    for model in [a0, *external]:
        metrics = model.get("metrics") or model.get("workbook_metrics")
        for horizon in HORIZONS:
            row = {"model_id": model["model_id"], "display_name": model["display_name"], "variant_id": model.get("variant_id"), "horizon": horizon, **metrics[horizon]}
            row.update(model.get("resource_stats") or {})
            if model is not a0:
                for metric in LOWER_IS_BETTER:
                    row[f"A0_{metric}_improvement_percent"] = (metrics[horizon][metric] - a0["metrics"][horizon][metric]) / metrics[horizon][metric] * 100.0
                row["R2_A0_minus_baseline"] = a0["metrics"][horizon]["R2"] - metrics[horizon]["R2"]
            rows.append(row)
    for horizon in HORIZONS:
        horizon_rows = [row for row in rows if row["horizon"] == horizon]
        for metric in METRICS:
            ordered = sorted(horizon_rows, key=lambda row: row[metric], reverse=metric == "R2")
            for rank, row in enumerate(ordered, 1):
                row[f"{metric}_rank"] = rank
    ranks: dict[str, list[int]] = {}
    for row in rows:
        ranks.setdefault(row["display_name"], []).append(row["Score_rank"])
    for row in rows:
        row["Score_mean_rank_H3_H6_H10"] = sum(ranks[row["display_name"]]) / 3.0
    return rows


def _read_csv_rows(path: Path, status: str) -> list[dict[str, Any]]:
    if not path.is_file():
        return [{"status": status, "reason": f"{path.name} not available"}]
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json_rows(path: Path, status: str) -> list[dict[str, Any]]:
    if not path.is_file():
        return [{"status": status, "reason": f"{path.name} not available"}]
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else [{"section": key, "value": value} for key, value in payload.items()]


def _table(rows: list[dict[str, Any]]) -> list[list]:
    if not rows:
        return [["status"], ["NOT_APPLICABLE"]]
    fields = list(rows[0])
    def scalar(value):
        return json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value
    return [fields, *[[scalar(row.get(field)) for field in fields] for row in rows]]


def aggregate(evidence_manifest: dict[str, Any], readiness: dict[str, Any], config_audit: dict[str, Any], *, output_root: str | Path = ANALYSIS_ROOT, require_complete: bool = False) -> dict[str, Any]:
    root = Path(output_root)
    if not readiness["core_ready"]:
        if require_complete:
            raise IncompleteEvidenceError(f"CORE_E8_READY={readiness['CORE_E8_READY']}; refusing formal aggregate")
        preview = {"status": "BLOCKED_INCOMPLETE_EVIDENCE", "CORE_E8_READY": readiness["CORE_E8_READY"], "formal_outputs_created": False}
        write_json(root / "e8_aggregate_preview.json", preview)
        return preview
    evidence = evidence_manifest["evidence"]
    internal = internal_ablation_rows(evidence, config_audit)
    directions = direction_decomposition_rows(evidence, config_audit)
    external = external_context_rows(evidence)
    capability = build_capability_matrix()
    outputs = {
        "e8_internal_ablation.csv": internal, "e8_cross_direction_decomposition.csv": directions,
        "e8_original26_external_comparison.csv": external, "e8_mechanism_capability_matrix.csv": capability,
    }
    for name, rows in outputs.items():
        write_csv(root / name, rows)
    diagnostic_files = {
        "e8_representation_similarity.csv": readiness["REPRESENTATION_DIAGNOSTICS_READY"],
        "e8_representation_shift.csv": readiness["REPRESENTATION_DIAGNOSTICS_READY"],
        "e8_cross_fusion_statistics.csv": readiness["CROSS_FUSION_DIAGNOSTICS_READY"],
        "e8_macro_prompt_statistics.csv": readiness["MACRO_PROMPT_DIAGNOSTICS_READY"],
        "e8_st_prompt_horizon_statistics.csv": readiness["ST_PROMPT_DIAGNOSTICS_READY"],
        "e8_paired_window_analysis.csv": readiness["PAIRED_WINDOW_ANALYSIS_READY"],
        "e8_grouped_analysis.csv": readiness["GROUPED_ANALYSIS_READY"],
    }
    diagnostic_rows = {name: _read_csv_rows(root / name, status) for name, status in diagnostic_files.items()}
    for name, rows in diagnostic_rows.items():
        if not (root / name).is_file():
            write_csv(root / name, rows)
    artifact_manifest = write_artifact_manifest(root)
    readme = [
        ["E8 Prompt and Cross-Fusion Analysis"],
        ["Internal Ablation is the primary causal evidence for components."],
        ["Original26 External Context is native-training-system architecture/end-to-end context, not unified-loss structure control."],
        ["E5 is cancelled; no common-loss external table exists; A8 is not E8 evidence."],
        ["Score/MAE/RMSE lower is better; R2 higher is better."],
        ["Correlation diagnostics are not causal proof. Missing values are not zero-filled. All formal numbers are loaded automatically."],
    ]
    sheet_data = {
        "README": readme, "Evidence_Readiness": _table([readiness]),
        "Protocol_Audit": _table(_read_json_rows(root / "E8_PROTOCOL_AUDIT.json", "NOT_AVAILABLE")),
        "Internal_Ablation": _table(internal), "Cross_Direction_Decomposition": _table(directions),
        "Original26_External_Context": _table(external), "Mechanism_Capability_Matrix": _table(capability),
        "Representation_Similarity": _table(diagnostic_rows["e8_representation_similarity.csv"]),
        "Representation_Shift": _table(diagnostic_rows["e8_representation_shift.csv"]),
        "Cross_Fusion_Statistics": _table(diagnostic_rows["e8_cross_fusion_statistics.csv"]),
        "Macro_Prompt_Statistics": _table(diagnostic_rows["e8_macro_prompt_statistics.csv"]),
        "ST_Prompt_Horizon": _table(diagnostic_rows["e8_st_prompt_horizon_statistics.csv"]),
        "Paired_Window_Analysis": _table(diagnostic_rows["e8_paired_window_analysis.csv"]),
        "Conditional_Groups": _table(diagnostic_rows["e8_grouped_analysis.csv"]),
        "Diagnostic_SideEffect_Audit": _table(_read_json_rows(root / "E8_DIAGNOSTIC_SIDE_EFFECT_AUDIT.json", "NOT_RUN")),
        "Artifact_Manifest": _table(_read_json_rows(artifact_manifest, "NOT_AVAILABLE")),
    }
    assert tuple(sheet_data) == WORKBOOK_SHEETS
    workbook = write_workbook(root / "PROMPT_CROSS_FUSION_ANALYSIS.xlsx", sheet_data)
    report = write_markdown_report(root / "PROMPT_CROSS_FUSION_ANALYSIS.md", readiness, internal, external)
    write_artifact_manifest(root)
    return {"status": "COMPLETE", "workbook": str(workbook), "report": str(report)}
