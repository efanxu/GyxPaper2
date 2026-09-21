from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import STMGPromptConfig, resolve_project_path
from .empirical_protocol import EMPIRICAL_FAMILIES, EMPIRICAL_PROTOCOL_ID, FROZEN_PROTOCOL_FIELDS, dry_run_report
from .experiment_protocol import CANONICAL_ID, canonical_config, canonical_directory


HORIZONS = (3, 6, 10)
SCENARIOS = ("spatial_consistent", "localized_disturbance", "high_volatility")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
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


def _contract(config: STMGPromptConfig) -> dict[str, Any]:
    if config.graph_operator != "bidirectional_diffusion":
        return {
            "graph_operator": config.graph_operator,
            "operator_direction": "none",
            "diffusion_order_micro": 0,
            "diffusion_order_macro": 0,
            "diffusion_state_count": 0,
            "diffusion_state_count_micro": 0,
            "diffusion_state_count_macro": 0,
            "diffusion_stream_count_micro": 1,
            "diffusion_stream_count_macro": 1,
            "state_concat_order": ["input"],
            "projection_mode": "not_applicable",
            "projection_input_dim_micro": config.hidden_dim,
            "projection_input_dim_macro": config.hidden_dim,
            "projection_output_dim": config.hidden_dim,
            "forward_normalization": "simple_local_graph_control",
            "reverse_normalization": "not_applicable",
            "transpose_topk_recomputed": False,
        }
    directions = ("forward", "reverse") if config.diffusion_direction == "bidirectional" else (config.diffusion_direction,)
    micro_count = len(directions) * config.diffusion_order_micro
    macro_count = len(directions) * config.diffusion_order_macro
    concat_order = ["input"]
    for direction in directions:
        concat_order.extend(f"{direction}_hop_{hop}" for hop in range(1, config.diffusion_order_micro + 1))
    return {
        "graph_operator": config.graph_operator,
        "operator_direction": config.diffusion_direction,
        "diffusion_order_micro": config.diffusion_order_micro,
        "diffusion_order_macro": config.diffusion_order_macro,
        "diffusion_state_count": micro_count,
        "diffusion_state_count_micro": micro_count,
        "diffusion_state_count_macro": macro_count,
        "diffusion_stream_count_micro": 1 + micro_count,
        "diffusion_stream_count_macro": 1 + macro_count,
        "state_concat_order": concat_order,
        "projection_mode": config.diffusion_projection_mode,
        "projection_input_dim_micro": config.hidden_dim * (1 + micro_count),
        "projection_input_dim_macro": config.hidden_dim * (1 + macro_count),
        "projection_output_dim": config.hidden_dim,
        "forward_normalization": "row_normalize(A)",
        "reverse_normalization": "row_normalize(A_transpose)",
        "transpose_topk_recomputed": False,
    }


def _find_artifact(d_root: Path, variant_id: str, filename: str) -> Path | None:
    candidates = []
    for path in d_root.rglob(filename):
        config = _json(path.parent / "effective_config.json") or _json(path.parent / "resolved_config.json")
        if str(config.get("variant", "")).upper() == variant_id or variant_id.lower() in path.as_posix().lower():
            candidates.append(path)
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda path: (
            "full_shape" in path.as_posix().lower() or path.parent == d_root / f"{variant_id.lower()}_empirical_v1",
            "smoke" not in path.as_posix().lower(),
            path.stat().st_mtime,
        ),
        reverse=True,
    )[0]


def _formal_metric_path(d_root: Path, variant_id: str, horizon: int) -> Path | None:
    candidates = [
        path
        for path in d_root.rglob(f"metrics_eval_h{horizon}.json")
        if variant_id.lower() in path.as_posix().lower()
        and "smoke" not in path.as_posix().lower()
        and any(part.startswith("seed_") for part in path.parts)
    ]
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)[0] if candidates else None


def _metric_value(metric: dict[str, Any], name: str) -> Any:
    return metric.get(name, metric.get(name.lower(), ""))


def write_step5_reports(results_root: str | Path) -> dict[str, Any]:
    root = resolve_project_path(results_root)
    d_root = root / "D"
    d_root.mkdir(parents=True, exist_ok=True)
    variants = list(EMPIRICAL_FAMILIES["D"].values())
    canonical = canonical_config()
    canonical_root = canonical_directory(resolve_project_path("."))

    manifest_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    operator_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    performance_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    efficiency_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    smoke_ready = True
    full_shape_ready = True
    formal_training_started = False
    for variant in variants:
        config = canonical_config()
        config.component_ablation = None
        config.variant = variant.variant_id
        config.definition = variant.display_name
        for key, value in variant.config_overrides.items():
            setattr(config, key, value)
        config.validate()
        contract = _contract(config)
        report = dry_run_report(variant)
        summary_path = _find_artifact(d_root, variant.variant_id, "model_summary.json")
        summary = _json(summary_path) if summary_path else {}
        graph_path = _find_artifact(d_root, variant.variant_id, "graph_identity.json")
        reference_path = _find_artifact(d_root, variant.variant_id, "reference.json")
        smoke_path = d_root / f"{variant.variant_id.lower()}_smoke_v1"
        full_shape_path = d_root / f"{variant.variant_id.lower()}_empirical_v1"
        if variant.reference_only:
            smoke_ready = smoke_ready and reference_path is not None
            full_shape_ready = full_shape_ready and reference_path is not None
        else:
            smoke_ready = smoke_ready and any(smoke_path.rglob("model_summary.json"))
            full_shape_ready = full_shape_ready and (full_shape_path / "model_summary.json").is_file()

        manifest_rows.append(
            {
                **variant.to_dict(),
                "protocol_profile": "STMG_FORMAL_V2",
                "source_scope": "internal_diffusion",
                "paired_reference": "D4/T0",
                "graph_identity_reference": "G0/CANONICAL",
                "formal_result_status": "reference-only" if variant.reference_only else "preflight-only",
                "model_summary_path": str(summary_path.resolve()) if summary_path else "",
                "graph_identity_path": str(graph_path.resolve()) if graph_path else "",
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
            }
        )
        operator_rows.append(
            {
                "variant_id": variant.variant_id,
                "graph_operator": contract["graph_operator"],
                "operator_direction": contract["operator_direction"],
                "diffusion_order_micro": contract["diffusion_order_micro"],
                "diffusion_order_macro": contract["diffusion_order_macro"],
                "forward_normalization": contract["forward_normalization"],
                "reverse_normalization": contract["reverse_normalization"],
                "transpose_topk_recomputed": contract["transpose_topk_recomputed"],
                "graph_identity_reference": "G0/CANONICAL",
                "state_concat_order": json.dumps(contract["state_concat_order"], ensure_ascii=False),
                "trace_source": str((summary_path or reference_path).resolve()) if (summary_path or reference_path) else "",
            }
        )
        state_rows.append(
            {
                "variant_id": variant.variant_id,
                "diffusion_state_count": contract["diffusion_state_count"],
                "diffusion_state_count_micro": contract["diffusion_state_count_micro"],
                "diffusion_state_count_macro": contract["diffusion_state_count_macro"],
                "diffusion_stream_count_micro": contract["diffusion_stream_count_micro"],
                "diffusion_stream_count_macro": contract["diffusion_stream_count_macro"],
                "projection_mode": contract["projection_mode"],
                "projection_input_dim_micro": contract["projection_input_dim_micro"],
                "projection_input_dim_macro": contract["projection_input_dim_macro"],
                "projection_output_dim": contract["projection_output_dim"],
                "parameter_count": summary.get("total_parameters", ""),
                "peak_memory_mb": summary.get("peak_memory_mb", ""),
            }
        )

        for horizon in HORIZONS:
            metric_path = canonical_root / f"metrics_eval_h{horizon}.json" if variant.reference_only else _formal_metric_path(d_root, variant.variant_id, horizon)
            metric = _json(metric_path) if metric_path and metric_path.is_file() else {}
            if metric_path and not variant.reference_only:
                formal_training_started = True
            performance_rows.append(
                {
                    "variant_id": variant.variant_id,
                    "horizon": horizon,
                    "status": "reference-only" if variant.reference_only and metric else ("completed" if metric else "not_started"),
                    "Score": _metric_value(metric, "Score"),
                    "MAE": _metric_value(metric, "MAE"),
                    "RMSE": _metric_value(metric, "RMSE"),
                    "R2": _metric_value(metric, "R2"),
                    "metrics_source": str(metric_path.resolve()) if metric_path and metric_path.is_file() else "",
                }
            )
            for scenario in SCENARIOS:
                scenario_rows.append(
                    {
                        "variant_id": variant.variant_id,
                        "horizon": horizon,
                        "scenario": scenario,
                        "status": "not_available_until_formal_predictions",
                        "Score": "",
                        "MAE": "",
                        "RMSE": "",
                        "R2": "",
                    }
                )

        efficiency_path = _find_artifact(d_root, variant.variant_id, "efficiency.json")
        efficiency = _json(efficiency_path) if efficiency_path else {}
        if variant.reference_only:
            canonical_summary = _json(canonical_root / "model_summary.json")
            efficiency = {
                "parameter_count": canonical_summary.get("total_parameters"),
                "trainable_parameter_count": canonical_summary.get("trainable_parameters"),
                "model_file_size_bytes": (canonical_root / "best_checkpoint.pt").stat().st_size,
                "total_train_seconds": canonical_summary.get("train_time_sec"),
                "best_epoch": canonical_summary.get("best_epoch"),
                "inference_latency_ms": (canonical_summary.get("inference_efficiency") or {}).get("inference_time_ms_per_window"),
                "throughput": (canonical_summary.get("inference_efficiency") or {}).get("inference_windows_per_sec"),
                "peak_memory_mb": canonical_summary.get("peak_memory_mb"),
                "measurement_scope": "canonical_reference",
            }
        efficiency_rows.append(
            {
                "variant_id": variant.variant_id,
                "status": efficiency.get("measurement_scope", "not_available"),
                "projection_mode": contract["projection_mode"],
                "diffusion_state_count": contract["diffusion_state_count"],
                "parameter_count": efficiency.get("parameter_count", summary.get("total_parameters", "")),
                "trainable_parameter_count": efficiency.get("trainable_parameter_count", summary.get("trainable_parameters", "")),
                "model_file_size_bytes": efficiency.get("model_file_size_bytes", ""),
                "total_train_seconds": efficiency.get("total_train_seconds", ""),
                "best_epoch": efficiency.get("best_epoch", ""),
                "inference_latency_ms": efficiency.get("inference_latency_ms", ""),
                "throughput": efficiency.get("throughput", ""),
                "peak_memory_mb": efficiency.get("peak_memory_mb", summary.get("peak_memory_mb", "")),
                "source": str(efficiency_path.resolve()) if efficiency_path else (str(canonical_root.resolve()) if variant.reference_only else ""),
            }
        )

        if not variant.reference_only and not all(_formal_metric_path(d_root, variant.variant_id, h) for h in HORIZONS):
            failure_path = (
                d_root
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

    _write_json(
        d_root / "D_VARIANT_MANIFEST.json",
        {
            "protocol_id": EMPIRICAL_PROTOCOL_ID,
            "canonical_run_id": CANONICAL_ID,
            "paired_reference": "D4/T0",
            "graph_identity_reference": "G0/CANONICAL",
            "variants": manifest_rows,
            "formal_training_started": formal_training_started,
            "updated_at": _utc_now(),
        },
    )
    _write_json(
        d_root / "D_PROTOCOL_AUDIT.json",
        {
            "protocol_id": EMPIRICAL_PROTOCOL_ID,
            "passed": all(row["passed"] and not row["frozen_protocol_changes"] for row in audit_rows),
            "frozen_fields": list(FROZEN_PROTOCOL_FIELDS),
            "variant_audits": audit_rows,
            "graph_reconstruction_prohibited": True,
            "physical_direction_interpretation_prohibited": True,
            "future_target_access_prohibited": True,
            "updated_at": _utc_now(),
        },
    )
    _write_csv(d_root / "D_OPERATOR_TRACE.csv", operator_rows, list(operator_rows[0]))
    _write_csv(d_root / "D_STATE_COUNT_SUMMARY.csv", state_rows, list(state_rows[0]))
    _write_csv(d_root / "D_PERFORMANCE_SUMMARY.csv", performance_rows, list(performance_rows[0]))
    _write_csv(d_root / "D_SCENARIO_METRICS.csv", scenario_rows, list(scenario_rows[0]))
    _write_csv(d_root / "D_EFFICIENCY_SUMMARY.csv", efficiency_rows, list(efficiency_rows[0]))
    _write_csv(d_root / "D_FAILURES.csv", failure_rows, ["variant_id", "run_status", "reason", "failure_report"])

    handoff = f"""# HANDOFF STEP5

## 状态

D0-D5 已注册到 `EMPIRICAL_ANALYSIS_V1`。D0 使用 simple/local graph control 且扩散状态数为 0；D1、D2 分别使用 `A` 与 `A^T` 的一阶状态；D3、D4、D5 分别使用双向一阶、二阶和三阶递推。所有变体固定引用 `G0/CANONICAL` 图身份，转置方向只做 `row_normalize(A_transpose)`，不重新 TopK。

- dry-run 唯一差异审计：已实现。
- 普通 smoke：{'已完成' if smoke_ready else '尚未全部完成'}。
- `[32, 10, 134]` 正式形状前向/反向预检：{'已完成' if full_shape_ready else '尚未全部完成'}。
- 正式长训练：{'已启动' if formal_training_started else '未启动；需要用户单独批准'}。

## 产物

- `D_VARIANT_MANIFEST.json`
- `D_PROTOCOL_AUDIT.json`
- `D_OPERATOR_TRACE.csv`
- `D_STATE_COUNT_SUMMARY.csv`
- `D_PERFORMANCE_SUMMARY.csv`
- `D_SCENARIO_METRICS.csv`
- `D_EFFICIENCY_SUMMARY.csv`
- `D_FAILURES.csv`

未运行的正式指标保持为空并写入 failure report；没有补零、复制 checkpoint 或把矩阵方向解释为物理风向。

## PowerShell

```powershell
Set-Location -LiteralPath 'D:\\PaperProject\\GyxPaper2'
$env:PYTHONPATH = 'D:\\PaperProject\\GyxPaper2\\custom_models\\src'
python -m compileall custom_models/src/st_mgprompt
python -m pytest custom_models/tests/empirical_analysis/test_step5_diffusion.py -q
python -m st_mgprompt.run_empirical --family D --variants D0 D1 D2 D3 D4 D5 --dry-run
python -m st_mgprompt.run_empirical --family D --variants D0 D1 D2 D3 D4 D5 --smoke
python -m st_mgprompt.run_empirical --family D --variants D0 D1 D2 D3 D4 D5 --full-shape
```

经用户批准后才可运行正式长训练：

```powershell
python -m st_mgprompt.run_empirical --family D --variants D0 D1 D2 D3 D5 `
  --seeds 2026 --profile STMG_FORMAL_V2 --source-scope internal_diffusion `
  --run-full --skip-completed
```
"""
    (d_root / "HANDOFF_STEP5.md").write_text(handoff, encoding="utf-8")
    return {
        "d_root": str(d_root.resolve()),
        "variant_count": len(variants),
        "smoke_ready": smoke_ready,
        "full_shape_ready": full_shape_ready,
        "formal_training_started": formal_training_started,
    }


__all__ = ["write_step5_reports"]
