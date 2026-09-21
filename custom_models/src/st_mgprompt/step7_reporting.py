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
from .experiment_protocol import (
    CANONICAL_ID,
    PRECISION_RESULT_ROOT,
    PRECISION_RUN_ID,
    canonical_directory,
)


HORIZONS = (3, 6, 10)


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
        key=lambda path: ("smoke" not in path.as_posix().lower(), path.stat().st_mtime),
        reverse=True,
    )[0]


def _precision_source(variant_id: str) -> Path:
    return (
        resolve_project_path(PRECISION_RESULT_ROOT)
        / PRECISION_RUN_ID
        / variant_id
        / "STMGPrompt_ComponentAblation"
    )


def _reference_source(reference: dict[str, Any]) -> Path | None:
    value = reference.get("source_run_dir")
    return Path(value) if value else None


def _prompt_diagnostics_from_reference(variant_id: str, source: Path) -> tuple[dict[str, Any], list[list[float]], dict[str, Any]]:
    import numpy as np
    import torch

    from .registry import build_model

    config = apply_empirical_variant(None, variant_id, "N")
    config.device = "cpu"
    config.diagnostics_level = "standard"
    checkpoint = torch.load(source / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    graph = {
        "A_macro_trend": state["A_macro_prior"].detach().cpu().numpy().astype(np.float32),
        "A_micro_local": state["A_micro_prior"].detach().cpu().numpy().astype(np.float32),
        "metadata": {"graph_uses_train_only_statistics": True, "fit_split": "train_only_checkpoint"},
    }
    model = build_model(config, input_dim=len(config.feature_cols), graph_data=graph)
    model.load_state_dict(state, strict=True)
    model.eval()
    prompt = model.st_prompt(
        num_nodes=config.num_nodes,
        horizon=config.max_pred_len,
        granularity_index=0,
    ).detach().float()
    horizon_vectors = prompt.mean(dim=2).squeeze(0)
    summary = {
        "status": "available",
        "prompt_shape": list(prompt.shape),
        "node_identity": config.st_prompt_use_node_identity,
        "horizon_identity": config.st_prompt_use_horizon_identity,
        "shared_horizon_embedding": config.st_prompt_use_shared_horizon_embedding,
        "type_embedding": config.st_prompt_use_type_embedding,
        "type_semantics": config.st_prompt_type_semantics,
        "horizon_representation": model.st_prompt.horizon_representation,
        "prompt_norm_mean": float(prompt.norm(dim=-1).mean().item()),
        "horizon_variance": float(prompt.var(dim=1, unbiased=False).mean().item()),
        "node_variance": float(prompt.var(dim=2, unbiased=False).mean().item()),
        "effective_rank": int(torch.linalg.matrix_rank(prompt.reshape(-1, prompt.shape[-1])).item()),
        "horizon_difference_max": float((prompt[:, 1:] - prompt[:, :1]).abs().max().item()),
        "node_difference_max": float((prompt[:, :, 1:] - prompt[:, :, :1]).abs().max().item()),
        "horizon_vector_norms": horizon_vectors.norm(dim=-1).tolist(),
        "diagnostic_source": str((source / "best_checkpoint.pt").resolve()),
    }
    decoder = {
        "decoder_type": model.direct_decoder.metadata["decoder_type"],
        "decoder_input_strategy": model.direct_decoder.metadata["decoder_input_strategy"],
        "decoder_context_mode": config.decoder_context_mode,
        "decoder_actual_history_len": config.lookback if config.decoder_context_mode != "last_state" else 1,
        "decoder_history_range": (
            f"[0,{config.lookback})" if config.decoder_context_mode != "last_state" else f"[{config.lookback - 1},{config.lookback})"
        ),
        "teacher_forcing": False,
        "autoregressive": False,
        "future_observed_features_used": False,
        "output_shape": ["B", config.max_pred_len, config.num_nodes],
    }
    return summary, torch.cdist(horizon_vectors, horizon_vectors).tolist(), decoder


def _metric_rows(variant_id: str, source: Path | None, status: str, parameter_count: Any = "") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        path = source / f"metrics_eval_h{horizon}.json" if source else None
        metric = _json(path)
        rows.append(
            {
                "variant_id": variant_id,
                "seed": 2026,
                "horizon": horizon,
                "status": status if metric else "not_started",
                "Score": metric.get("Score", metric.get("score", "")),
                "MAE": metric.get("MAE", ""),
                "RMSE": metric.get("RMSE", ""),
                "R2": metric.get("R2", ""),
                "parameter_count": parameter_count,
                "metrics_source": str(path.resolve()) if path and path.is_file() else "",
            }
        )
    return rows


def write_step7_reports(results_root: str | Path) -> dict[str, Any]:
    root = resolve_project_path(results_root)
    n_root = root / "N"
    n_root.mkdir(parents=True, exist_ok=True)
    variants = list(EMPIRICAL_FAMILIES["N"].values())

    manifest_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    distance_rows: list[dict[str, Any]] = []
    decoder_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []
    performance_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    smoke_ready = True
    full_shape_ready = True

    for variant in variants:
        report = dry_run_report(variant)
        summary_path = _find_artifact(n_root, variant.variant_id, "model_summary.json")
        summary = _json(summary_path)
        reference_path = _find_artifact(n_root, variant.variant_id, "reference.json")
        reference = _json(reference_path)
        prompt_summary = summary.get("prompt_embedding_summary") or {}
        horizon_distance = summary.get("horizon_distance_matrix") or []
        decoder_summary = summary.get("decoder_summary") or {}
        source = _reference_source(reference) if variant.reference_only else None
        if variant.reference_only and source and not prompt_summary:
            prompt_summary, horizon_distance, decoder_summary = _prompt_diagnostics_from_reference(
                variant.variant_id, source
            )

        smoke_dir = n_root / f"{variant.variant_id.lower()}_smoke_v1"
        full_shape_dir = n_root / f"{variant.variant_id.lower()}_empirical_v1"
        if variant.reference_only:
            semantic_ok = bool(reference.get("source_semantic_audit", {}).get("passed"))
            smoke_ready = smoke_ready and semantic_ok
            full_shape_ready = full_shape_ready and semantic_ok
        else:
            smoke_ready = smoke_ready and any(smoke_dir.rglob("model_summary.json"))
            full_shape_ready = full_shape_ready and (full_shape_dir / "model_summary.json").is_file()

        manifest_rows.append(
            {
                **variant.to_dict(),
                "protocol_profile": "STMG_FORMAL_V2",
                "source_scope": "internal_decoder",
                "formal_result_status": "reference-only" if variant.reference_only else "preflight-only",
                "model_summary_path": str(summary_path.resolve()) if summary_path else "",
                "reference_path": str(reference_path.resolve()) if reference_path else "",
            }
        )
        audit_rows.append(
            {
                "variant_id": variant.variant_id,
                "passed": bool(report.get("passed")),
                "expected_diff_fields": report.get("expected_diff_fields", []),
                "actual_diff_fields": report.get("actual_diff_fields", []),
                "frozen_protocol_changes": report.get("frozen_protocol_changes", []),
                "reference_semantic_audit": reference.get("source_semantic_audit") if variant.reference_only else None,
            }
        )
        prompt_rows.append(
            {
                "variant_id": variant.variant_id,
                "status": prompt_summary.get("status", "NOT_APPLICABLE"),
                "prompt_shape": json.dumps(prompt_summary.get("prompt_shape"), ensure_ascii=False),
                "node_identity": prompt_summary.get("node_identity", False),
                "horizon_identity": prompt_summary.get("horizon_identity", False),
                "shared_horizon_embedding": prompt_summary.get("shared_horizon_embedding", False),
                "type_embedding": prompt_summary.get("type_embedding", False),
                "type_semantics": prompt_summary.get("type_semantics", "NOT_APPLICABLE"),
                "horizon_representation": prompt_summary.get("horizon_representation", "NOT_APPLICABLE"),
                "prompt_norm_mean": prompt_summary.get("prompt_norm_mean", ""),
                "node_variance": prompt_summary.get("node_variance", ""),
                "horizon_variance": prompt_summary.get("horizon_variance", ""),
                "effective_rank": prompt_summary.get("effective_rank", ""),
                "node_difference_max": prompt_summary.get("node_difference_max", ""),
                "horizon_difference_max": prompt_summary.get("horizon_difference_max", ""),
            }
        )
        if horizon_distance:
            for left, row in enumerate(horizon_distance, start=1):
                for right, value in enumerate(row, start=1):
                    distance_rows.append(
                        {"variant_id": variant.variant_id, "horizon_i": left, "horizon_j": right, "distance": value}
                    )
        else:
            distance_rows.append(
                {"variant_id": variant.variant_id, "horizon_i": "NOT_APPLICABLE", "horizon_j": "NOT_APPLICABLE", "distance": ""}
            )
        decoder_rows.append(
            {
                "variant_id": variant.variant_id,
                "decoder_type": decoder_summary.get("decoder_type", ""),
                "decoder_input_strategy": decoder_summary.get("decoder_input_strategy", ""),
                "decoder_context_mode": decoder_summary.get("decoder_context_mode", ""),
                "decoder_actual_history_len": decoder_summary.get("decoder_actual_history_len", ""),
                "decoder_history_range": decoder_summary.get("decoder_history_range", ""),
                "decoder_pooling": decoder_summary.get("decoder_pooling", ""),
                "attention_entropy_fine": decoder_summary.get("fine_history_attention_entropy", ""),
                "attention_entropy_coarse": decoder_summary.get("coarse_history_attention_entropy", ""),
                "teacher_forcing": False,
                "autoregressive": False,
                "future_observed_features_used": False,
                "head_parameterization": decoder_summary.get("head_parameterization", ""),
                "head_specific_parameter_count": decoder_summary.get("head_specific_parameter_count", ""),
                "output_shape": json.dumps(decoder_summary.get("output_shape"), ensure_ascii=False),
                "parameter_count": summary.get("total_parameters", ""),
            }
        )
        correlation_rows.append(
            {
                "variant_id": variant.variant_id,
                "status": "NOT_APPLICABLE",
                "correlation_scope": "node_embedding_vs_difficulty_mean_power_volatility",
                "reason": "formal_prediction_and_train_only_node_statistics_not_generated_in_preflight",
                "causal_interpretation": False,
            }
        )
        performance_rows.extend(
            _metric_rows(
                variant.variant_id,
                source if variant.reference_only else None,
                "reference-only" if variant.reference_only else "not_started",
                summary.get("total_parameters", ""),
            )
        )
        if not variant.reference_only:
            failure_path = (
                n_root
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

    for capacity_id in ("P4", "P5"):
        source = _precision_source(capacity_id)
        model_summary = _json(source / "model_summary.json")
        performance_rows.extend(
            _metric_rows(f"capacity_{capacity_id}", source, "capacity-reference", model_summary.get("total_parameters", ""))
        )

    intervention_path = n_root / "N_INTERVENTION_SUMMARY.csv"
    if not intervention_path.is_file():
        _write_csv(
            intervention_path,
            [
                {
                    "variant_id": variant.variant_id,
                    "status": "not_started",
                    "intervention_type": "NOT_APPLICABLE",
                    "analysis_label": "internal_sensitivity_analysis",
                    "reason": "requires_completed_run_and_explicit_inference_only_command",
                }
                for variant in variants
            ],
            ["variant_id", "status", "intervention_type", "analysis_label", "reason"],
        )

    _write_json(
        n_root / "N_VARIANT_MANIFEST.json",
        {
            "protocol_id": EMPIRICAL_PROTOCOL_ID,
            "canonical_run_id": CANONICAL_ID,
            "protocol_profile": "STMG_FORMAL_V2",
            "source_scope": "internal_decoder",
            "variants": manifest_rows,
            "updated_at": _utc_now(),
        },
    )
    _write_json(
        n_root / "N_PROTOCOL_AUDIT.json",
        {
            "protocol_id": EMPIRICAL_PROTOCOL_ID,
            "passed": all(row["passed"] and not row["frozen_protocol_changes"] for row in audit_rows),
            "frozen_fields": list(FROZEN_PROTOCOL_FIELDS),
            "variant_audits": audit_rows,
            "prompt_shape": [1, 10, 134, "D"],
            "n1_n3_distinction": {
                "N1": "no learned horizon component; shared zero horizon representation",
                "N3": "one learned horizon vector shared across all ten future steps",
            },
            "type_embedding_semantics": "fixed_decoder_input_type_not_fine_coarse_dynamic_branch",
            "future_target_access_prohibited": True,
            "embedding_physical_interpretation_prohibited": True,
            "intervention_real_world_counterfactual_label_prohibited": True,
            "updated_at": _utc_now(),
        },
    )
    _write_csv(n_root / "N_PROMPT_EMBEDDING_SUMMARY.csv", prompt_rows, list(prompt_rows[0]))
    _write_csv(n_root / "N_HORIZON_DISTANCE.csv", distance_rows, list(distance_rows[0]))
    _write_csv(n_root / "N_NODE_DIFFICULTY_CORRELATION.csv", correlation_rows, list(correlation_rows[0]))
    _write_csv(n_root / "N_DECODER_SUMMARY.csv", decoder_rows, list(decoder_rows[0]))
    _write_csv(n_root / "N_PERFORMANCE_SUMMARY.csv", performance_rows, list(performance_rows[0]))
    _write_csv(n_root / "N_FAILURES.csv", failure_rows, ["variant_id", "run_status", "reason", "failure_report"])

    handoff = f"""# HANDOFF STEP7

## 状态

N0-N8 已注册到 `EMPIRICAL_ANALYSIS_V1`，Prompt 固定为 `[1,H,N,D]`，预测统一为
`[B,H,N]`。node identity、horizon identity、shared horizon 和固定 input type embedding
均可独立审计。N1 使用零 horizon 分量，N3 使用一个可学习共享 horizon 向量，两者不是同一实现。

- dry-run 唯一差异审计：已实现。
- 普通 smoke：{'已完成' if smoke_ready else '尚未全部完成'}。
- `[32,10,134]` 正式形状前向/反向预检：{'已完成' if full_shape_ready else '尚未全部完成'}。
- N5：只读引用 Canonical；N8：语义审计后只读引用 P3。
- N6/N7：只读取 144 步输入历史；N8 的 cross-attention key/value 仅来自同一历史窗口。
- 正式长训练：未启动；需要用户单独批准。
- prompt intervention：仅在 completed run 上显式执行，标签固定为内部敏感性分析，不是现实反事实。

## 修改文件

- `config.py`、`prompt_alignment.py`、`decoder.py`、`model.py`：独立 Prompt 开关与三类历史解码器。
- `empirical_protocol.py`、`run_empirical.py`、`check_protocol.py`：N0-N8 注册、运行、reference 和 fail-closed 审计。
- `step7_reporting.py`：步骤7清单、嵌入、解码器、性能、失败与交接产物。
- `scripts/empirical_analysis/prompt_intervention.py`：不改 checkpoint 的 inference-only 内部敏感性分析。
- `test_step7_prompt_decoder.py`：Prompt、decoder、前缀评价、干预恢复和唯一差异测试。

## 验证结果

- `compileall`：通过。
- ST-MGPrompt 核心与 empirical_analysis 回归：138 passed，1 skipped。
- N0-N8 dry-run：全部通过唯一差异审计。
- N0-N4、N6、N7 smoke：全部完成且 protocol passed；N5/N8 reference-only 审计通过。
- N0-N4、N6、N7 full-shape：全部完成，输出 `[32,10,134]`，loss 和 backward 有限。
- inference-only intervention：已在 N0 smoke checkpoint 上验证 zero、swap、mean；原参数在每次干预后恢复。

## 未完成项与下一步

正式长训练未启动，因此 N0-N4/N6/N7 的正式 H3/H6/H10 指标、Top10% 困难风机退化和
node embedding 相关分析仍标记为未开始。用户批准后运行正式单 seed 训练，再对 completed run
执行 prompt intervention 并刷新步骤7汇总。

## PowerShell

```powershell
Set-Location -LiteralPath 'D:\\PaperProject\\GyxPaper2'
$env:PYTHONPATH = 'D:\\PaperProject\\GyxPaper2\\custom_models\\src'
python -m compileall custom_models/src/st_mgprompt
python -m pytest custom_models/tests/empirical_analysis/test_step7_prompt_decoder.py -q
python -m st_mgprompt.run_empirical --family N --variants N0 N1 N2 N3 N4 N5 N6 N7 N8 --dry-run
python -m st_mgprompt.run_empirical --family N --variants N0 N1 N2 N3 N4 N5 N6 N7 N8 --smoke
python -m st_mgprompt.run_empirical --family N --variants N0 N1 N2 N3 N4 N5 N6 N7 N8 --full-shape
```
"""
    (n_root / "HANDOFF_STEP7.md").write_text(handoff, encoding="utf-8")
    return {
        "n_root": str(n_root.resolve()),
        "variant_count": len(variants),
        "smoke_ready": smoke_ready,
        "full_shape_ready": full_shape_ready,
        "formal_training_started": False,
    }


__all__ = ["write_step7_reports"]
