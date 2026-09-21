from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import resolve_project_path
from .empirical_protocol import EMPIRICAL_FAMILIES, EMPIRICAL_PROTOCOL_ID, FROZEN_PROTOCOL_FIELDS, dry_run_report
from .experiment_protocol import CANONICAL_ID, canonical_config, canonical_directory, config_diff


HORIZONS = (3, 6, 10)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_t0_reference(t_root: Path) -> dict[str, Any]:
    source = canonical_directory(resolve_project_path("."))
    source_config_path = source / "effective_config.json"
    source_summary = _json(source / "model_summary.json")
    source_config = _json(source_config_path)
    semantic_audit = config_diff(source_config, "A0", "component_ablation") if source_config else {"passed": False}
    payload = {
        "protocol_id": EMPIRICAL_PROTOCOL_ID,
        "variant_id": "T0",
        "canonical_run_id": CANONICAL_ID,
        "reference_only": True,
        "paired_reference": "T0",
        "protocol_profile": "STMG_FORMAL_V2",
        "source_scope": "internal_mechanism",
        "source_run_dir": str(source.resolve()),
        "source_config": str(source_config_path.resolve()),
        "source_model_summary": str((source / "model_summary.json").resolve()),
        "source_checkpoint": str((source / "best_checkpoint.pt").resolve()),
        "source_semantic_audit": semantic_audit,
        "model_summary": {
            "total_parameters": source_summary.get("total_parameters"),
            "trainable_parameters": source_summary.get("trainable_parameters"),
            "input_shape": [32, 144, 134, 16],
            "output_shape": [32, 10, 134],
            "graph_operator": source_config.get("graph_operator"),
            "loss_function": source_config.get("loss_function"),
            "decoder_context_mode": source_config.get("decoder_context_mode"),
        },
        "checkpoint_copied": False,
        "metrics_copied": False,
        "created_at": _utc_now(),
    }
    _write_json(t_root / "T0" / "reference.json", payload)
    return payload


def write_step3_reports(results_root: str | Path) -> dict[str, Any]:
    root = resolve_project_path(results_root)
    t_root = root / "T"
    t_root.mkdir(parents=True, exist_ok=True)
    reference = write_t0_reference(t_root)
    variants = list(EMPIRICAL_FAMILIES["T"].values())
    manifest_rows = []
    diff_rows = []
    protocol_rows = []
    alignment_rows = []
    for variant in variants:
        report = dry_run_report(variant)
        manifest_rows.append(
            {
                **variant.to_dict(),
                "protocol_profile": "STMG_FORMAL_V2",
                "source_scope": "internal_mechanism",
                "paired_reference": "T0",
                "formal_result_status": "reference-only" if variant.variant_id == "T0" else "not_started",
            }
        )
        diff_rows.append(
            {
                "variant_id": variant.variant_id,
                "implementation_status": variant.implementation_status,
                "expected_diff_fields": json.dumps(report.get("expected_diff_fields", []), ensure_ascii=False),
                "actual_diff_fields": json.dumps(report.get("actual_diff_fields", []), ensure_ascii=False),
                "frozen_protocol_changes": json.dumps(report.get("frozen_protocol_changes", []), ensure_ascii=False),
                "passed": bool(report.get("passed")),
                "execution_ready": bool(report.get("execution_ready")),
            }
        )
        protocol_rows.append(
            {
                "variant_id": variant.variant_id,
                "unique_diff_passed": bool(report.get("passed")),
                "frozen_protocol_changes": report.get("frozen_protocol_changes", []),
                "protocol_profile": "STMG_FORMAL_V2",
                "source_scope": "internal_mechanism",
            }
        )
        config = canonical_config()
        for key, value in variant.config_overrides.items():
            setattr(config, key, value)
        alignment_rows.append(
            {
                "variant_id": variant.variant_id,
                "coarse_alignment_mode": config.coarse_alignment_mode,
                "coarse_windows": json.dumps(config.coarse_windows),
                "upsample_rule": config.coarse_upsample_rule if variant.variant_id == "T4" else "not_applicable",
                "causal_access_required": variant.variant_id == "T4",
                "same_history_axis_output": config.coarse_alignment_mode == "same_axis",
                "trace_path": str((t_root / "smoke" / "T4" / "coarse_alignment_trace.csv").resolve()) if variant.variant_id == "T4" else "",
            }
        )

    _write_json(
        t_root / "T_VARIANT_MANIFEST.json",
        {
            "protocol_id": EMPIRICAL_PROTOCOL_ID,
            "protocol_profile": "STMG_FORMAL_V2",
            "source_scope": "internal_mechanism",
            "canonical_reference": reference,
            "variants": manifest_rows,
            "formal_training_started": False,
            "updated_at": _utc_now(),
        },
    )
    _write_json(
        t_root / "T_PROTOCOL_AUDIT.json",
        {
            "protocol_id": EMPIRICAL_PROTOCOL_ID,
            "passed": all(row["unique_diff_passed"] for row in protocol_rows),
            "frozen_fields": list(FROZEN_PROTOCOL_FIELDS),
            "canonical_values": {key: canonical_config().to_dict().get(key) for key in FROZEN_PROTOCOL_FIELDS},
            "variant_audits": protocol_rows,
            "canonical_source_semantic_audit": reference["source_semantic_audit"],
            "updated_at": _utc_now(),
        },
    )
    _write_csv(
        t_root / "T_UNIQUE_DIFF_MATRIX.csv",
        diff_rows,
        [
            "variant_id",
            "implementation_status",
            "expected_diff_fields",
            "actual_diff_fields",
            "frozen_protocol_changes",
            "passed",
            "execution_ready",
        ],
    )
    _write_csv(
        t_root / "T_ALIGNMENT_SUMMARY.csv",
        alignment_rows,
        [
            "variant_id",
            "coarse_alignment_mode",
            "coarse_windows",
            "upsample_rule",
            "causal_access_required",
            "same_history_axis_output",
            "trace_path",
        ],
    )

    performance_rows = []
    difficulty_rows = []
    failures = []
    canonical = canonical_directory(resolve_project_path("."))
    canonical_summary = _json(canonical / "model_summary.json")
    canonical_parameters = canonical_summary.get("total_parameters")
    for variant in variants:
        full_shape_summary = _json(t_root / "full_shape" / variant.variant_id / "model_summary.json")
        parameter_count = (
            canonical_parameters if variant.variant_id == "T0" else full_shape_summary.get("total_parameters")
        )
        parameter_delta = (
            parameter_count - canonical_parameters
            if isinstance(parameter_count, int) and isinstance(canonical_parameters, int)
            else None
        )
        for horizon in HORIZONS:
            source = canonical / f"metrics_eval_h{horizon}.json" if variant.variant_id == "T0" else None
            metric = _json(source) if source else {}
            performance_rows.append(
                {
                    "variant_id": variant.variant_id,
                    "horizon": horizon,
                    "run_status": "reference-only" if variant.variant_id == "T0" else "not_started",
                    "Score": metric.get("Score"),
                    "MAE": metric.get("MAE"),
                    "RMSE": metric.get("RMSE"),
                    "R2": metric.get("R2"),
                    "valid_target_count": metric.get("valid_target_count"),
                    "parameter_count": parameter_count,
                    "parameter_delta_vs_t0": parameter_delta,
                    "metrics_source": str(source.resolve()) if source else "",
                }
            )
            for group in ("Normal", "High-volatility", "Ramp-up", "Ramp-down", "Shared difficult Top10%"):
                difficulty_rows.append(
                    {
                        "variant_id": variant.variant_id,
                        "group": group,
                        "horizon": horizon,
                        "run_status": "not_available_until_formal_predictions",
                        "Score": "",
                        "MAE": "",
                        "RMSE": "",
                        "R2": "",
                        "valid_target_count": "",
                    }
                )
        if variant.variant_id != "T0":
            failure_path = t_root / "seed_2026" / variant.variant_id / "failure_report.json"
            failure_payload = {
                "protocol_id": EMPIRICAL_PROTOCOL_ID,
                "variant_id": variant.variant_id,
                "status": "not_started",
                "reasons": ["formal_long_training_requires_explicit_user_approval"],
                "retry_from": "dry-run",
                "contaminated_checkpoint_reused": False,
                "updated_at": _utc_now(),
            }
            _write_json(failure_path, failure_payload)
            failures.append(
                {
                    "variant_id": variant.variant_id,
                    "run_status": "not_started",
                    "reason": failure_payload["reasons"][0],
                    "failure_report": str(failure_path.resolve()),
                }
            )
    _write_csv(
        t_root / "T_PERFORMANCE_SUMMARY.csv",
        performance_rows,
        [
            "variant_id",
            "horizon",
            "run_status",
            "Score",
            "MAE",
            "RMSE",
            "R2",
            "valid_target_count",
            "parameter_count",
            "parameter_delta_vs_t0",
            "metrics_source",
        ],
    )
    _write_csv(
        t_root / "T_DIFFICULTY_GROUPS.csv",
        difficulty_rows,
        ["variant_id", "group", "horizon", "run_status", "Score", "MAE", "RMSE", "R2", "valid_target_count"],
    )
    _write_csv(
        t_root / "T_FAILURES.csv",
        failures,
        ["variant_id", "run_status", "reason", "failure_report"],
    )
    return {
        "t_root": str(t_root.resolve()),
        "variant_count": len(variants),
        "formal_training_started": False,
    }


__all__ = ["write_step3_reports", "write_t0_reference"]
