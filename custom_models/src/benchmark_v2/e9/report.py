from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .constants import PROJECT_ROOT, TRANSFER_ROOT
from .io_utils import file_identity, write_json
from .xlsx_writer import write_workbook


def analysis_markdown(payload: dict[str, Any]) -> str:
    summary = payload["tables"]["transfer_summary"][0]
    lines = [
        "# MS-MG-DWU Transfer Analysis", "",
        "E5 is cancelled and was not read. Controls are read-only original26 Original Masked-MSE formal results; only six E9 MS-MG-DWU runs are newly trained.", "",
        "Score/MAE/RMSE are lower-is-better and positive improvement means MS-MG-DWU is better. R2 is higher-is-better and is reported as absolute gain.", "",
        "Engineering portability does not imply every model improves; all unfavorable results are retained.", "",
        "## Cross-model summary", "",
        f"- H10 Score improved: {summary['H10_Score_improved_models']}/6",
        f"- H10 Score degraded: {summary['H10_Score_degraded_models']}/6",
        f"- All H3/H6/H10 Score improved: {summary['all_three_horizons_improved_models']}/6",
        f"- At least two horizons improved: {summary['at_least_two_horizons_improved_models']}/6",
        f"- Mean H10 Score improvement: {summary['mean_H10_Score_improvement_pct']:.4f}%",
        f"- Median H10 Score improvement: {summary['median_H10_Score_improvement_pct']:.4f}%",
        "", "## Pair results", "",
        "| Model | Family | H3 Score improvement | H6 | H10 |", "|---|---|---:|---:|---:|",
    ]
    for row in payload["tables"]["e9_model_pair_comparison.csv"]:
        lines.append(
            f"| {row['model_id']} | {row['architecture_family']} | {row['H3_Score_improvement_pct']:.4f}% | "
            f"{row['H6_Score_improvement_pct']:.4f}% | {row['H10_Score_improvement_pct']:.4f}% |"
        )
    lines.extend(["", "A0/A8, when available, are optional related references and are excluded from this six-model summary.", ""])
    return "\n".join(lines)


def write_artifact_manifest(root: str | Path = TRANSFER_ROOT) -> Path:
    base = Path(root)
    names = [
        "MSMG_DWU_TRANSFER_ANALYSIS.xlsx", "MSMG_DWU_TRANSFER_ANALYSIS.md",
        "e9_transfer_metrics.csv", "e9_model_pair_comparison.csv", "e9_family_summary.csv",
        "e9_training_dynamics.csv", "e9_loss_component_statistics.csv", "e9_efficiency_overhead.csv",
        "e9_failure_inventory.csv",
    ]
    payload = {
        "schema_version": "e9_artifact_manifest_v1",
        "artifacts": [file_identity(base / name) or {"path": str((base / name).resolve()), "missing": True} for name in names],
        "e5_consumed": False,
    }
    return write_json(base / "E9_ARTIFACT_MANIFEST.json", payload)


def build_formal_report(output_root: str | Path = TRANSFER_ROOT) -> dict[str, Any]:
    root = Path(output_root)
    payload_path = root / "e9_report_payload.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if not payload.get("formal") or not payload.get("readiness", {}).get("CORE_E9_READY_BOOL"):
        raise RuntimeError("Formal report creation is blocked until CORE_E9_READY=12/12.")
    (root / "MSMG_DWU_TRANSFER_ANALYSIS.md").write_text(analysis_markdown(payload), encoding="utf-8")
    write_workbook(root / "MSMG_DWU_TRANSFER_ANALYSIS.xlsx", payload)
    manifest = write_artifact_manifest(root)
    return {"status": "PASS", "xlsx": str(root / "MSMG_DWU_TRANSFER_ANALYSIS.xlsx"), "markdown": str(root / "MSMG_DWU_TRANSFER_ANALYSIS.md"), "manifest": str(manifest)}
