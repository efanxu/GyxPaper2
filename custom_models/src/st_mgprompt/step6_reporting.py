from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import resolve_project_path
from .empirical_protocol import (
    EMPIRICAL_FAMILIES,
    EMPIRICAL_PROTOCOL_ID,
    FROZEN_PROTOCOL_FIELDS,
    apply_empirical_variant,
    dry_run_report,
)
from .experiment_protocol import CANONICAL_ID, canonical_directory


HORIZONS = (3, 6, 10)
SCENES = ("synthetic_preflight",)
SAMPLE_SELECTION_RULE = "deterministic_seed_2026_full_shape_preflight"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


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


def _find_artifact(root: Path, variant_id: str, filename: str) -> Path | None:
    candidates = [path for path in root.rglob(filename) if variant_id.lower() in path.as_posix().lower()]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda path: (
            path.parent == root / f"{variant_id.lower()}_empirical_v1",
            "smoke" not in path.as_posix().lower(),
            path.stat().st_mtime,
        ),
        reverse=True,
    )[0]


def _source_dir(f_root: Path, variant_id: str) -> Path | None:
    reference_path = _find_artifact(f_root, variant_id, "reference.json")
    reference = _json(reference_path)
    source = reference.get("source_run_dir")
    return Path(source) if source else None


def _metric_value(payload: dict[str, Any], key: str) -> Any:
    return payload.get(key, payload.get(key.lower(), ""))


def _diagnostic_value(payload: dict[str, Any], key: str) -> Any:
    value = payload.get(key)
    return "NOT_APPLICABLE" if value is None else value


def _diagnostic_common(variant_id: str, status: str, reason: str = "") -> dict[str, Any]:
    return {
        "run_id": f"{variant_id.lower()}_empirical_v1",
        "variant_id": variant_id,
        "seed": 2026,
        "horizon": "NOT_APPLICABLE",
        "scene": SCENES[0],
        "sample_selection_rule": SAMPLE_SELECTION_RULE,
        "status": status,
        "not_applicable_reason": reason,
    }


def _reference_diagnostics(source: Path, variant_id: str) -> tuple[dict[str, Any], str]:
    checkpoint_path = source / "best_checkpoint.pt"
    if not checkpoint_path.is_file():
        return {}, "reference_checkpoint_missing"
    try:
        import numpy as np
        import torch

        from .registry import build_model

        config = apply_empirical_variant(None, variant_id, "F")
        config.device = "cpu"
        config.diagnostics_level = "standard"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else None
        if not isinstance(state, dict):
            return {}, "reference_checkpoint_has_no_model_state_dict"
        macro = state["A_macro_prior"].detach().cpu().numpy().astype(np.float32)
        micro = state["A_micro_prior"].detach().cpu().numpy().astype(np.float32)
        graph_data = {
            "A_macro_trend": macro,
            "A_micro_local": micro,
            "metadata": {
                "graph_uses_train_only_statistics": True,
                "fit_split": "canonical_train_only_checkpoint",
                "graph_identity_reference": "G0/CANONICAL",
            },
        }
        model = build_model(config, input_dim=len(config.feature_cols), graph_data=graph_data)
        model.load_state_dict(state, strict=True)
        model.eval()
        generator = torch.Generator(device="cpu").manual_seed(2026)
        x = torch.randn(
            1,
            config.lookback,
            config.num_nodes,
            len(config.feature_cols),
            generator=generator,
        )
        with torch.inference_mode():
            first = model(x)
            second = model(x)
        keys = (
            "pre_fusion_cosine_similarity",
            "post_fusion_cosine_similarity",
            "fine_representation_shift",
            "coarse_representation_shift",
            "fusion_gate_mean",
            "fusion_gate_std",
            "fusion_gate_q05",
            "fusion_gate_q25",
            "fusion_gate_q50",
            "fusion_gate_q75",
            "fusion_gate_q95",
            "fusion_gate_saturation_ratio",
            "macro_attn_entropy",
            "fine_attn_entropy",
            "macro_prompt_attn_entropy",
            "macro_prompt_pairwise_cosine_mean",
            "macro_prompt_pairwise_cosine_max",
            "macro_to_fine_interaction_increment_norm",
            "fine_to_coarse_interaction_increment_norm",
            "macro_to_fine_query_source",
            "macro_to_fine_key_source",
            "macro_to_fine_value_source",
            "macro_to_fine_query_history_range",
            "fine_to_coarse_query_source",
            "fine_to_coarse_key_source",
            "fine_to_coarse_value_source",
            "fine_to_coarse_history_range",
            "macro_to_fine_attention_shape",
            "fine_to_coarse_attention_shape",
            "macro_to_fine_attention_row_sum_max_error",
            "fine_to_coarse_attention_row_sum_max_error",
        )
        diagnostics = {key: first["aux"].get(key) for key in keys}
        diagnostics.update(
            {
                "diagnostic_prediction_regression_equal": bool(
                    torch.equal(first["pred"], second["pred"])
                ),
                "diagnostic_source_checkpoint": str(checkpoint_path.resolve()),
                "diagnostic_mode": "inference_only_no_prediction_export",
            }
        )
        return diagnostics, ""
    except (KeyError, RuntimeError, ValueError) as exc:
        return {}, f"reference_inference_diagnostic_failed:{type(exc).__name__}:{exc}"


def write_step6_reports(results_root: str | Path) -> dict[str, Any]:
    root = resolve_project_path(results_root)
    f_root = root / "F"
    f_root.mkdir(parents=True, exist_ok=True)
    variants = list(EMPIRICAL_FAMILIES["F"].values())

    manifest_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    similarity_rows: list[dict[str, Any]] = []
    shift_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    entropy_rows: list[dict[str, Any]] = []
    performance_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    smoke_ready = True
    full_shape_ready = True
    formal_training_started = False

    for variant in variants:
        report = dry_run_report(variant)
        summary_path = _find_artifact(f_root, variant.variant_id, "model_summary.json")
        summary = _json(summary_path)
        diagnostics = summary.get("fusion_diagnostics") or {}
        structure = summary.get("fusion_structure") or {}
        reference_path = _find_artifact(f_root, variant.variant_id, "reference.json")
        reference = _json(reference_path)
        reference_diagnostic_reason = ""
        if variant.reference_only and not diagnostics:
            source_for_diagnostics = _source_dir(f_root, variant.variant_id)
            if source_for_diagnostics is not None:
                diagnostics, reference_diagnostic_reason = _reference_diagnostics(
                    source_for_diagnostics, variant.variant_id
                )
        smoke_path = f_root / f"{variant.variant_id.lower()}_smoke_v1"
        full_shape_path = f_root / f"{variant.variant_id.lower()}_empirical_v1"
        if variant.reference_only:
            smoke_ready = smoke_ready and bool(reference.get("source_semantic_audit", {}).get("passed"))
            full_shape_ready = full_shape_ready and bool(reference.get("source_semantic_audit", {}).get("passed"))
        else:
            smoke_ready = smoke_ready and any(smoke_path.rglob("model_summary.json"))
            full_shape_ready = full_shape_ready and (full_shape_path / "model_summary.json").is_file()

        manifest_rows.append(
            {
                **variant.to_dict(),
                "protocol_profile": "STMG_FORMAL_V2",
                "source_scope": "internal_fusion",
                "formal_result_status": "reference-only" if variant.reference_only else "preflight-only",
                "model_summary_path": str(summary_path.resolve()) if summary_path else "",
                "reference_path": str(reference_path.resolve()) if reference_path else "",
                "fusion_structure": structure,
            }
        )
        audit_rows.append(
            {
                "variant_id": variant.variant_id,
                "passed": bool(report.get("passed")),
                "execution_ready": bool(report.get("execution_ready")),
                "expected_diff_fields": report.get("expected_diff_fields", []),
                "actual_diff_fields": report.get("actual_diff_fields", []),
                "frozen_protocol_changes": report.get("frozen_protocol_changes", []),
                "reference_semantic_audit": reference.get("source_semantic_audit") if variant.reference_only else None,
                "reference_inference_diagnostics": bool(diagnostics) if variant.reference_only else None,
                "diagnostic_prediction_regression_equal": (
                    diagnostics.get("diagnostic_prediction_regression_equal")
                    if variant.reference_only
                    else None
                ),
            }
        )

        diagnostic_available = bool(diagnostics)
        diagnostic_reason = "" if diagnostic_available else (
            reference_diagnostic_reason or "reference_only_source_has_no_step6_inference_diagnostics"
            if variant.reference_only
            else "full_shape_diagnostics_missing"
        )
        common = _diagnostic_common(
            variant.variant_id,
            "available" if diagnostic_available else "NOT_APPLICABLE",
            diagnostic_reason,
        )
        similarity_rows.append(
            {
                **common,
                "pre_fusion_cosine_similarity": _diagnostic_value(diagnostics, "pre_fusion_cosine_similarity"),
                "post_fusion_cosine_similarity": _diagnostic_value(diagnostics, "post_fusion_cosine_similarity"),
                "macro_prompt_pairwise_cosine_mean": _diagnostic_value(diagnostics, "macro_prompt_pairwise_cosine_mean"),
                "macro_prompt_pairwise_cosine_max": _diagnostic_value(diagnostics, "macro_prompt_pairwise_cosine_max"),
                "diagnostic_prediction_regression_equal": _diagnostic_value(
                    diagnostics, "diagnostic_prediction_regression_equal"
                ),
            }
        )
        shift_rows.append(
            {
                **common,
                "fine_representation_shift": _diagnostic_value(diagnostics, "fine_representation_shift"),
                "coarse_representation_shift": _diagnostic_value(diagnostics, "coarse_representation_shift"),
                "macro_to_fine_interaction_increment_norm": _diagnostic_value(
                    diagnostics, "macro_to_fine_interaction_increment_norm"
                ),
                "fine_to_coarse_interaction_increment_norm": _diagnostic_value(
                    diagnostics, "fine_to_coarse_interaction_increment_norm"
                ),
            }
        )
        direction_rows.append(
            {
                **common,
                "status": (
                    "available"
                    if diagnostic_available and variant.variant_id in {"F4", "F5", "F6", "F7", "F8"}
                    else "NOT_APPLICABLE"
                ),
                "not_applicable_reason": (
                    ""
                    if diagnostic_available and variant.variant_id in {"F4", "F5", "F6", "F7", "F8"}
                    else diagnostic_reason or "fusion_operator_has_no_directional_cross_attention"
                ),
                "macro_to_fine_enabled": not bool(variant.config_overrides.get("disable_macro_to_fine_cross", False))
                and variant.variant_id not in {"F0", "F1", "F2", "F3"},
                "fine_to_coarse_enabled": not bool(variant.config_overrides.get("disable_reverse_cross", False))
                and variant.variant_id not in {"F0", "F1", "F2", "F3"},
                "macro_to_fine_query_source": _diagnostic_value(diagnostics, "macro_to_fine_query_source"),
                "macro_to_fine_key_source": _diagnostic_value(diagnostics, "macro_to_fine_key_source"),
                "macro_to_fine_value_source": _diagnostic_value(diagnostics, "macro_to_fine_value_source"),
                "macro_to_fine_history_range": _diagnostic_value(diagnostics, "macro_to_fine_query_history_range"),
                "fine_to_coarse_query_source": _diagnostic_value(diagnostics, "fine_to_coarse_query_source"),
                "fine_to_coarse_key_source": _diagnostic_value(diagnostics, "fine_to_coarse_key_source"),
                "fine_to_coarse_value_source": _diagnostic_value(diagnostics, "fine_to_coarse_value_source"),
                "fine_to_coarse_history_range": _diagnostic_value(diagnostics, "fine_to_coarse_history_range"),
                "macro_to_fine_attention_shape": json.dumps(
                    _diagnostic_value(diagnostics, "macro_to_fine_attention_shape"), ensure_ascii=False
                ),
                "fine_to_coarse_attention_shape": json.dumps(
                    _diagnostic_value(diagnostics, "fine_to_coarse_attention_shape"), ensure_ascii=False
                ),
                "future_horizon_used_as_input": False,
            }
        )
        gate_applicable = variant.variant_id in {"F3", "F4", "F5", "F6", "F7", "F8"} and diagnostic_available
        gate_rows.append(
            {
                **common,
                "status": "available" if gate_applicable else "NOT_APPLICABLE",
                "not_applicable_reason": "" if gate_applicable else (
                    diagnostic_reason or "fusion_operator_has_no_auditable_gate"
                ),
                "gate_level": diagnostics.get("gate_level") or (
                    "node_time_channel"
                    if variant.variant_id in {"F4", "F5", "F6", "F7", "F8"}
                    else "NOT_APPLICABLE"
                ),
                "gate_mean": _diagnostic_value(diagnostics, "fusion_gate_mean"),
                "gate_std": _diagnostic_value(diagnostics, "fusion_gate_std"),
                "gate_q05": _diagnostic_value(diagnostics, "fusion_gate_q05"),
                "gate_q25": _diagnostic_value(diagnostics, "fusion_gate_q25"),
                "gate_q50": _diagnostic_value(diagnostics, "fusion_gate_q50"),
                "gate_q75": _diagnostic_value(diagnostics, "fusion_gate_q75"),
                "gate_q95": _diagnostic_value(diagnostics, "fusion_gate_q95"),
                "gate_saturation_ratio": _diagnostic_value(diagnostics, "fusion_gate_saturation_ratio"),
            }
        )
        attention_applicable = variant.variant_id in {"F4", "F5", "F6", "F7", "F8"} and diagnostic_available
        entropy_rows.append(
            {
                **common,
                "status": "available" if attention_applicable else "NOT_APPLICABLE",
                "not_applicable_reason": "" if attention_applicable else (
                    diagnostic_reason or "fusion_operator_has_no_cross_attention"
                ),
                "macro_to_fine_attention_entropy": _diagnostic_value(diagnostics, "macro_attn_entropy"),
                "fine_to_coarse_attention_entropy": _diagnostic_value(diagnostics, "fine_attn_entropy"),
                "macro_prompt_pooling_entropy": _diagnostic_value(diagnostics, "macro_prompt_attn_entropy"),
                "macro_to_fine_row_sum_max_error": _diagnostic_value(
                    diagnostics, "macro_to_fine_attention_row_sum_max_error"
                ),
                "fine_to_coarse_row_sum_max_error": _diagnostic_value(
                    diagnostics, "fine_to_coarse_attention_row_sum_max_error"
                ),
                "causal_interpretation": "historical_representation_association_only",
            }
        )

        source = _source_dir(f_root, variant.variant_id) if variant.reference_only else None
        for horizon in HORIZONS:
            metric_path = source / f"metrics_eval_h{horizon}.json" if source else None
            metric = _json(metric_path)
            if metric and not variant.reference_only:
                formal_training_started = True
            performance_rows.append(
                {
                    "variant_id": variant.variant_id,
                    "seed": 2026,
                    "horizon": horizon,
                    "status": "reference-only" if metric and variant.reference_only else "not_started",
                    "Score": _metric_value(metric, "Score"),
                    "MAE": _metric_value(metric, "MAE"),
                    "RMSE": _metric_value(metric, "RMSE"),
                    "R2": _metric_value(metric, "R2"),
                    "parameter_count": summary.get("total_parameters", ""),
                    "metrics_source": str(metric_path.resolve()) if metric_path and metric_path.is_file() else "",
                }
            )

        if not variant.reference_only:
            failure_path = (
                f_root
                / f"{variant.variant_id.lower()}_empirical_v1"
                / "seed_2026"
                / "STMGPrompt_ComponentAblation"
                / "failure_report.json"
            )
            _write_json(
                failure_path,
                {
                    "protocol_id": EMPIRICAL_PROTOCOL_ID,
                    "variant_id": variant.variant_id,
                    "status": "not_started",
                    "reasons": ["formal_long_training_requires_separate_user_approval"],
                    "metrics_zero_filled": False,
                    "checkpoint_copied": False,
                    "updated_at": _utc_now(),
                },
            )
            failure_rows.append(
                {
                    "variant_id": variant.variant_id,
                    "run_status": "preflight-only",
                    "reason": "formal_long_training_requires_separate_user_approval",
                    "failure_report": str(failure_path.resolve()),
                }
            )

    sensitivity_rows: list[dict[str, Any]] = []
    for value in (6, 12, 24, 36, 48):
        sensitivity_rows.append(
            {
                "sensitivity_parent": "F7",
                "parameter": "cross_fusion_recent_len",
                "value": value,
                "seed": 2026,
                "selection_split": "validation_only",
                "selection_metric": "official_Score_H10",
                "status": "reference_available" if value in {6, 24} else "not_started",
                "reference": "A5" if value == 6 else ("Canonical" if value == 24 else ""),
                "selected": "NOT_APPLICABLE",
                "reason": "formal_sensitivity_training_requires_separate_user_approval",
            }
        )
    for value in (1, 2, 4, 8):
        sensitivity_rows.append(
            {
                "sensitivity_parent": "F7",
                "parameter": "macro_prompt_len",
                "value": value,
                "seed": 2026,
                "selection_split": "validation_only",
                "selection_metric": "official_Score_H10",
                "status": "reference_available" if value == 4 else "not_started",
                "reference": "Canonical" if value == 4 else "",
                "selected": "NOT_APPLICABLE",
                "reason": "formal_sensitivity_training_requires_separate_user_approval",
            }
        )

    direction_semantics = """# F DIRECTION SEMANTICS

## Frozen mapping

The names below are frozen from the executed query key value flow, not inferred from the word reverse.

| Direction | Query | Key and value | Updated branch | Disable field |
| --- | --- | --- | --- | --- |
| Macro to Fine | Fine history `[B*N,L,D]` | Macro Prompt `[B*N,P,D]` | Fine | `disable_macro_to_fine_cross` |
| Fine to Coarse | Coarse history `[B*N,L,D]` | Recent Fine history `[B*N,R,D]` | Coarse | `disable_reverse_cross` |

F4 sets `disable_reverse_cross=true`, so only Macro to Fine remains. F5 sets
`disable_macro_to_fine_cross=true`, so only Fine to Coarse remains. The Fine memory range is
`[L-min(cross_fusion_recent_len,L), L)`. Macro to Fine uses historical representations only and never
reads a future target or prediction horizon. These attention weights describe internal historical
representation association and are not causal horizon attributions.
"""
    (f_root / "F_DIRECTION_SEMANTICS.md").write_text(direction_semantics, encoding="utf-8")
    _write_json(
        f_root / "F_VARIANT_MANIFEST.json",
        {
            "protocol_id": EMPIRICAL_PROTOCOL_ID,
            "canonical_run_id": CANONICAL_ID,
            "protocol_profile": "STMG_FORMAL_V2",
            "source_scope": "internal_fusion",
            "variants": manifest_rows,
            "formal_training_started": formal_training_started,
            "updated_at": _utc_now(),
        },
    )
    _write_json(
        f_root / "F_PROTOCOL_AUDIT.json",
        {
            "protocol_id": EMPIRICAL_PROTOCOL_ID,
            "passed": all(row["passed"] and not row["frozen_protocol_changes"] for row in audit_rows),
            "frozen_fields": list(FROZEN_PROTOCOL_FIELDS),
            "variant_audits": audit_rows,
            "direction_semantics_frozen": True,
            "future_target_access_prohibited": True,
            "horizon_causal_attention_interpretation_prohibited": True,
            "updated_at": _utc_now(),
        },
    )
    _write_csv(f_root / "F_REPRESENTATION_SIMILARITY.csv", similarity_rows, list(similarity_rows[0]))
    _write_csv(f_root / "F_REPRESENTATION_SHIFT.csv", shift_rows, list(shift_rows[0]))
    _write_csv(f_root / "F_DIRECTION_TRACE.csv", direction_rows, list(direction_rows[0]))
    _write_csv(f_root / "F_GATE_STATISTICS.csv", gate_rows, list(gate_rows[0]))
    _write_csv(f_root / "F_ATTENTION_ENTROPY.csv", entropy_rows, list(entropy_rows[0]))
    _write_csv(f_root / "F_SENSITIVITY_SUMMARY.csv", sensitivity_rows, list(sensitivity_rows[0]))
    _write_csv(f_root / "F_PERFORMANCE_SUMMARY.csv", performance_rows, list(performance_rows[0]))
    _write_csv(f_root / "F_FAILURES.csv", failure_rows, ["variant_id", "run_status", "reason", "failure_report"])

    handoff = f"""# HANDOFF STEP6

## 状态

F0-F8 已注册到 `EMPIRICAL_ANALYSIS_V1`。F4/F5 的方向已按实际 query key value 流冻结：
F4 仅保留 Macro to Fine，F5 仅保留 Fine to Coarse。F1/F2/F3 不实例化 Macro Prompt 或
Cross-Attention；F3 的 gate 为节点 时间 通道级，范围为 `[0,1]`，初始 bias 为 0，默认不
stop-gradient。

- dry-run 唯一差异审计：已实现。
- 普通 smoke：{'已完成' if smoke_ready else '尚未全部完成'}。
- `[32, 10, 134]` 正式形状前向/反向预检：{'已完成' if full_shape_ready else '尚未全部完成'}。
- F6/F7/F8：只读引用 A7 Canonical A4，不复制 checkpoint 或预测数组。
- 正式长训练与敏感性筛选：{'已启动' if formal_training_started else '未启动；需要用户单独批准'}。

缺失的正式指标和诊断均保留为空或 `NOT_APPLICABLE`，没有补零。Cross-Fusion 权重仅解释为
历史表示层关联，不解释为未来 horizon 因果注意力。

## PowerShell

```powershell
Set-Location -LiteralPath 'D:\\PaperProject\\GyxPaper2'
$env:PYTHONPATH = 'D:\\PaperProject\\GyxPaper2\\custom_models\\src'
python -m compileall custom_models/src/st_mgprompt
python -m pytest custom_models/tests/empirical_analysis/test_step6_fusion.py -q
python -m st_mgprompt.run_empirical --family F --variants F0 F1 F2 F3 F4 F5 F6 F7 F8 --dry-run
python -m st_mgprompt.run_empirical --family F --variants F0 F1 F2 F3 F4 F5 F6 F7 F8 --smoke
python -m st_mgprompt.run_empirical --family F --variants F0 F1 F2 F3 F4 F5 F6 F7 F8 --full-shape
```

经用户批准后才运行正式单 seed 长训练和验证集敏感性筛选。
"""
    (f_root / "HANDOFF_STEP6.md").write_text(handoff, encoding="utf-8")
    return {
        "f_root": str(f_root.resolve()),
        "variant_count": len(variants),
        "smoke_ready": smoke_ready,
        "full_shape_ready": full_shape_ready,
        "formal_training_started": formal_training_started,
    }


__all__ = ["write_step6_reports"]
