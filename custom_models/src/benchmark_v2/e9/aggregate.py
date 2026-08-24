from __future__ import annotations

import csv
import json
import math
import subprocess
from pathlib import Path
from statistics import median
from typing import Any

from .constants import ARCHITECTURE_FAMILIES, HORIZONS, METRICS, MODEL_IDS, TRANSFER_ROOT
from .io_utils import read_json, write_csv, write_json


class IncompleteE9EvidenceError(RuntimeError):
    pass


def metric_comparison(control: float, transfer: float, metric: str) -> tuple[float, float]:
    if metric == "R2":
        return transfer - control, transfer - control
    delta = control - transfer
    improvement = delta / control * 100.0 if control != 0 else float("nan")
    return delta, improvement


def build_tables(evidence_manifest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    controls = {row["model_id"]: row for row in evidence_manifest["evidence"] if row["evidence_role"] == "ORIGINAL26_CONTROL"}
    transfers = {row["model_id"]: row for row in evidence_manifest["evidence"] if row["evidence_role"] == "E9_TRANSFER"}
    long_rows, pair_rows = [], []
    for model_id in MODEL_IDS:
        control, transfer = controls[model_id], transfers[model_id]
        for horizon in HORIZONS:
            row = {
                "model_id": model_id, "architecture_family": ARCHITECTURE_FAMILIES[model_id], "horizon": horizon,
                "control_run_id": control["run_id"], "transfer_run_id": transfer["run_id"],
            }
            for metric in METRICS:
                c = float(control["metrics"][horizon][metric])
                t = float(transfer["metrics"][horizon][metric])
                delta, improvement = metric_comparison(c, t, metric)
                row[f"control_{metric}"] = c
                row[f"transfer_{metric}"] = t
                row[f"absolute_delta_{metric}"] = delta
                row[f"improvement_pct_{metric}" if metric != "R2" else "R2_gain"] = improvement
            long_rows.append(row)
        h10 = next(row for row in long_rows if row["model_id"] == model_id and row["horizon"] == 10)
        scores = [next(row for row in long_rows if row["model_id"] == model_id and row["horizon"] == h)["improvement_pct_Score"] for h in HORIZONS]
        pair_rows.append({
            "model_id": model_id, "architecture_family": ARCHITECTURE_FAMILIES[model_id],
            "control_run_id": control["run_id"], "transfer_run_id": transfer["run_id"],
            "H3_Score_improvement_pct": scores[0], "H6_Score_improvement_pct": scores[1],
            "H10_Score_improvement_pct": scores[2], "H10_Score_delta": h10["absolute_delta_Score"],
            "all_three_score_improved": all(value > 0 for value in scores),
            "at_least_two_score_improved": sum(value > 0 for value in scores) >= 2,
        })
    family_rows = []
    for model_id in MODEL_IDS:
        rows = [row for row in long_rows if row["model_id"] == model_id]
        family_rows.append({
            "architecture_family": ARCHITECTURE_FAMILIES[model_id], "model_id": model_id,
            "mean_score_improvement_pct": sum(row["improvement_pct_Score"] for row in rows) / len(rows),
            "H10_score_improvement_pct": next(row["improvement_pct_Score"] for row in rows if row["horizon"] == 10),
            "positive_horizons": sum(row["improvement_pct_Score"] > 0 for row in rows),
        })
    h10_values = [row["H10_Score_improvement_pct"] for row in pair_rows]
    transfer_summary = [{
        "H10_Score_improved_models": sum(value > 0 for value in h10_values),
        "H10_Score_degraded_models": sum(value < 0 for value in h10_values),
        "all_three_horizons_improved_models": sum(row["all_three_score_improved"] for row in pair_rows),
        "at_least_two_horizons_improved_models": sum(row["at_least_two_score_improved"] for row in pair_rows),
        "mean_H10_Score_improvement_pct": sum(h10_values) / len(h10_values),
        "median_H10_Score_improvement_pct": median(h10_values),
        "max_H10_improvement_pct": max(h10_values), "max_H10_degradation_pct": min(h10_values),
        "denominator": 6,
    }]
    return {
        "e9_transfer_metrics.csv": long_rows,
        "e9_model_pair_comparison.csv": pair_rows,
        "e9_family_summary.csv": family_rows,
        "transfer_summary": transfer_summary,
    }


def _training_tables(evidence_manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dynamics, components = [], []
    for evidence in evidence_manifest["evidence"]:
        if evidence["evidence_role"] != "E9_TRANSFER":
            continue
        path = Path(evidence["source_root"]) / "train_log.csv"
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                base = {"model_id": evidence["model_id"], "run_id": evidence["run_id"], **row}
                dynamics.append(base)
                component = {key: value for key, value in base.items() if key in {"model_id", "run_id", "epoch"} or any(token in key for token in ("granularity_", "site_", "difficulty_", "relative_training_rate", "weighted_contribution", "total_loss"))}
                components.append(component)
    return dynamics, components


def aggregate_complete(evidence_manifest: dict[str, Any], readiness: dict[str, Any], *, output_root: str | Path = TRANSFER_ROOT, require_complete: bool = True) -> dict[str, Any]:
    root = Path(output_root)
    if not readiness.get("CORE_E9_READY_BOOL") or readiness.get("LOSS_ONLY_PAIRING_READY") != "6/6":
        raise IncompleteE9EvidenceError(
            f"CORE_E9_READY={readiness.get('CORE_E9_READY')}; LOSS_ONLY_PAIRING_READY={readiness.get('LOSS_ONLY_PAIRING_READY')}"
        )
    tables = build_tables(evidence_manifest)
    dynamics, components = _training_tables(evidence_manifest)
    tables["e9_training_dynamics.csv"] = dynamics
    tables["e9_loss_component_statistics.csv"] = components
    tables["e9_efficiency_overhead.csv"] = []
    tables["e9_failure_inventory.csv"] = []
    for name, rows in tables.items():
        if name.endswith(".csv"):
            write_csv(root / name, rows)
    payload = {
        "schema_version": "e9_report_payload_v1", "formal": True,
        "readiness": readiness, "tables": tables,
        "evidence": evidence_manifest["evidence"],
    }
    write_json(root / "e9_report_payload.json", payload)
    return {"status": "PASS", "output_root": str(root), "tables": {key: len(value) for key, value in tables.items()}}
