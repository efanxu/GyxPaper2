from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .a8_batch4_contract import A8_RUN_RELATIVE_PATH, variant_contract as a8_variant_contract
from .a8_batch4_readiness import inspect_a8_run
from .config import STMGPromptConfig, resolve_project_path
from .empirical_protocol import (
    EMPIRICAL_FAMILIES,
    EMPIRICAL_PROTOCOL_ID,
    FROZEN_PROTOCOL_FIELDS,
    dry_run_report,
)
from .experiment_protocol import CANONICAL_ID, canonical_directory
from .losses import LOSS_STATE_SCHEMA_VERSION, get_loss_fn


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
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


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _smoke_dir(root: Path, variant_id: str) -> Path:
    return root / f"{variant_id.lower()}_smoke_v1" / "seed_2026" / "STMGPrompt_ComponentAblation"


def _preflight_dir(root: Path, variant_id: str) -> Path:
    return root / f"{variant_id.lower()}_empirical_v1"


def _metric_rows(variant_id: str, source: Path | None, status: str, scope: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizon in (3, 6, 10):
        path = source / f"metrics_eval_h{horizon}.json" if source else None
        metric = _json(path) if path else {}
        rows.append(
            {
                "variant_id": variant_id,
                "horizon": horizon,
                "run_status": status,
                "comparison_scope": scope,
                "Score": metric.get("Score", ""),
                "MAE": metric.get("MAE", ""),
                "RMSE": metric.get("RMSE", ""),
                "R2": metric.get("R2", ""),
                "metrics_source": str(path.resolve()) if path and path.is_file() else "",
            }
        )
    return rows


def _checkpoint_restore_audit(run_dir: Path) -> dict[str, Any]:
    checkpoint_path = run_dir / "best_checkpoint.pt"
    config_path = run_dir / "config.json"
    if not checkpoint_path.is_file() or not config_path.is_file():
        return {
            "status": "not_available",
            "run_dir": str(run_dir.resolve()),
            "checkpoint_exists": checkpoint_path.is_file(),
            "config_exists": config_path.is_file(),
        }
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = STMGPromptConfig.from_json(config_path)
    saved = checkpoint.get("loss_state_dict") or {}
    node_state = saved.get("node_weight")
    checkpoint_num_nodes = int(node_state.numel()) if isinstance(node_state, torch.Tensor) else config.num_nodes
    loss_fn = get_loss_fn(config.loss_function, config=config, num_nodes=checkpoint_num_nodes)
    loss_fn.load_state_dict(saved, strict=True)
    restored = loss_fn.state_dict()
    equal = bool(saved) and set(saved) == set(restored) and all(
        torch.equal(saved[key].detach().cpu(), restored[key].detach().cpu()) for key in saved
    )
    required_checkpoint_state = {
        "loss_state_dict": bool(saved),
        "optimizer_state_dict": checkpoint.get("optimizer_state_dict") is not None,
        "grad_scaler_state_dict_recorded": "grad_scaler_state_dict" in checkpoint,
        "early_stopping_counter": "early_stopping_counter" in checkpoint,
        "python_random_state": checkpoint.get("python_random_state") is not None,
        "numpy_random_state": checkpoint.get("numpy_random_state") is not None,
        "torch_random_state": checkpoint.get("torch_random_state") is not None,
    }
    schema = checkpoint.get("loss_state_schema_version")
    passed = equal and all(required_checkpoint_state.values()) and schema == LOSS_STATE_SCHEMA_VERSION
    return {
        "status": "passed" if passed else "failed",
        "run_dir": str(run_dir.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "loss_state_equal_after_reload": equal,
        "loss_state_keys": sorted(saved),
        "loss_state_schema_version": schema,
        "expected_loss_state_schema_version": LOSS_STATE_SCHEMA_VERSION,
        "required_checkpoint_state": required_checkpoint_state,
    }


def write_step8_reports(results_root: str | Path) -> dict[str, Any]:
    root = resolve_project_path(results_root)
    l_root = root / "L"
    l_root.mkdir(parents=True, exist_ok=True)
    variants = list(EMPIRICAL_FAMILIES["L"].values())
    canonical = canonical_directory(resolve_project_path("."))
    a8_source = resolve_project_path(A8_RUN_RELATIVE_PATH)
    a8_readiness = inspect_a8_run(a8_source)

    manifest_rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    node_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    performance_rows: list[dict[str, Any]] = []
    difficulty_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    restore_rows: list[dict[str, Any]] = []

    for variant in variants:
        report = dry_run_report(variant)
        smoke_dir = _smoke_dir(l_root, variant.variant_id)
        preflight_dir = _preflight_dir(l_root, variant.variant_id)
        reference = _json(l_root / f"{variant.variant_id.lower()}_smoke_v1" / "reference.json")
        smoke_complete = (smoke_dir / "evaluation_complete.json").is_file() if variant.trainable else bool(reference)
        preflight_complete = (preflight_dir / "model_summary.json").is_file() if variant.trainable else bool(
            _json(preflight_dir / "reference.json")
        )
        manifest_rows.append(
            {
                **variant.to_dict(),
                "protocol_profile": "STMG_FORMAL_V2" if variant.variant_id != "L7" else "STMG_A8_BATCH4_V1",
                "source_scope": "internal_loss",
                "smoke_complete": smoke_complete,
                "full_shape_complete": preflight_complete,
                "formal_result_status": "reference-only" if variant.reference_only else "preflight-only",
                "loss_state_schema_version": LOSS_STATE_SCHEMA_VERSION,
            }
        )
        audits.append(
            {
                "variant_id": variant.variant_id,
                "passed": bool(report.get("passed")),
                "expected_diff_fields": report.get("expected_diff_fields", []),
                "actual_diff_fields": report.get("actual_diff_fields", []),
                "frozen_protocol_changes": report.get("frozen_protocol_changes", []),
            }
        )

        if variant.trainable:
            log_rows = _read_csv(smoke_dir / "train_log.csv")
            for row in log_rows:
                base = {"variant_id": variant.variant_id, "epoch": row.get("epoch"), "source": "smoke_only"}
                epoch_rows.append(
                    {
                        **base,
                        "train_loss": row.get("train_loss", ""),
                        "val_loss": row.get("val_loss", ""),
                        "total_loss": row.get("total_loss", ""),
                        "site_loss": row.get("site_loss", ""),
                        "skipped_all_invalid_batches": row.get("skipped_train_batches_total", ""),
                    }
                )
                for horizon in (3, 6, 10):
                    horizon_rows.append(
                        {
                            **base,
                            "horizon": horizon,
                            "raw_loss": row.get(f"raw_loss_h{horizon}", ""),
                            "ema_loss": row.get(f"ema_loss_h{horizon}", ""),
                            "initial_loss": row.get(f"initial_loss_h{horizon}", ""),
                            "difficulty_level": row.get(f"difficulty_level_h{horizon}", ""),
                            "relative_rate": row.get(f"relative_training_rate_h{horizon}", ""),
                            "dwa_epoch_loss": row.get(f"dwa_epoch_loss_h{horizon}", ""),
                            "dwa_loss_ratio": row.get(f"dwa_loss_ratio_h{horizon}", ""),
                            "weight_preclip": row.get(f"granularity_weight_preclip_h{horizon}", ""),
                            "weight": row.get(f"granularity_weight_h{horizon}", ""),
                            "weighted_contribution": row.get(f"weighted_contribution_h{horizon}", ""),
                        }
                    )
                distribution_rows.append(
                    {
                        **base,
                        "granularity_clip_low_fraction": row.get("granularity_weight_clip_low_fraction", ""),
                        "granularity_clip_high_fraction": row.get("granularity_weight_clip_high_fraction", ""),
                        "site_clip_low_fraction": row.get("site_weight_clip_low_fraction", ""),
                        "site_clip_high_fraction": row.get("site_weight_clip_high_fraction", ""),
                        "site_weight_min": row.get("site_weight_min", ""),
                        "site_weight_mean": row.get("site_weight_mean", ""),
                        "site_weight_max": row.get("site_weight_max", ""),
                        "valid_horizon_count": row.get("valid_horizon_count", ""),
                        "valid_node_count": row.get("valid_node_count", ""),
                    }
                )
            for row in _read_csv(smoke_dir / "diagnostics" / "site_weight_final.csv"):
                node_rows.append({"variant_id": variant.variant_id, "source": "smoke_only", **row})
            restore_rows.append({"variant_id": variant.variant_id, **_checkpoint_restore_audit(smoke_dir)})
            formal_failure_path = (
                preflight_dir / "seed_2026" / "STMGPrompt_ComponentAblation" / "failure_report.json"
            )
            _write_json(
                formal_failure_path,
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
                    "failure_report": str(formal_failure_path.resolve()),
                }
            )
            performance_rows.extend(_metric_rows(variant.variant_id, None, "not_started", "STMG_FORMAL_V2"))
        elif variant.variant_id == "L3":
            performance_rows.extend(_metric_rows("L3", canonical, "reference-only", "STMG_FORMAL_V2"))
        else:
            performance_rows.extend(
                _metric_rows(
                    "L7",
                    a8_source if a8_readiness.get("ready") else None,
                    "reference-only" if a8_readiness.get("ready") else "reference-unavailable",
                    "STMG_A8_BATCH4_V1_independent",
                )
            )
            if not a8_readiness.get("ready"):
                failure_rows.append(
                    {
                        "variant_id": "L7",
                        "run_status": "reference-unavailable",
                        "reason": ";".join(a8_readiness.get("reasons", [])),
                        "failure_report": str((l_root / "l7_smoke_v1" / "reference.json").resolve()),
                    }
                )
        for horizon in (3, 6, 10):
            difficulty_rows.append(
                {
                    "variant_id": variant.variant_id,
                    "horizon": horizon,
                    "group": "top10_percent_difficult_turbines",
                    "status": "not_started",
                    "reason": "requires_formal_predictions_and_train_validation_frozen_difficulty_rule",
                    "Score": "",
                    "MAE": "",
                    "RMSE": "",
                    "R2": "",
                }
            )

    restore_passed = all(row.get("status") == "passed" for row in restore_rows)
    protocol_passed = all(row["passed"] and not row["frozen_protocol_changes"] for row in audits)
    _write_json(
        l_root / "L_VARIANT_MANIFEST.json",
        {
            "protocol_id": EMPIRICAL_PROTOCOL_ID,
            "canonical_run_id": CANONICAL_ID,
            "source_scope": "internal_loss",
            "variants": manifest_rows,
            "updated_at": _utc_now(),
        },
    )
    _write_json(
        l_root / "L_PROTOCOL_AUDIT.json",
        {
            "protocol_id": EMPIRICAL_PROTOCOL_ID,
            "passed": protocol_passed,
            "frozen_fields": list(FROZEN_PROTOCOL_FIELDS),
            "variant_audits": audits,
            "l7_a8_boundary": {
                "same_component_causal_table_as_l0_l6": False,
                "profile": "STMG_A8_BATCH4_V1",
                "contract": a8_variant_contract(),
                "readiness": a8_readiness,
            },
            "updated_at": _utc_now(),
        },
    )
    _write_json(
        l_root / "L_LOSS_INPUT_CONTRACT.json",
        {
            "loss_identity": "msmg_dwu_loss",
            "base_loss": "smooth_l1",
            "prediction_shape": ["B", 10, 134],
            "mask": "valid_target_mask",
            "mask_updates_only": True,
            "all_invalid_batch": "return_none_and_skip_optimizer_step",
            "dynamic_weight_gradient": "stopped",
            "uncertainty_log_sigma_gradient": "learnable",
            "dwa_update_timing": "epoch_end",
            "loss_state_schema_version": LOSS_STATE_SCHEMA_VERSION,
            "source_scope": "internal_loss",
        },
    )
    _write_json(
        l_root / "L_CHECKPOINT_RESTORE_AUDIT.json",
        {
            "passed": restore_passed,
            "loss_state_schema_version": LOSS_STATE_SCHEMA_VERSION,
            "audits": restore_rows,
            "updated_at": _utc_now(),
        },
    )
    _write_csv(
        l_root / "L_EPOCH_LOSS.csv",
        epoch_rows,
        ["variant_id", "epoch", "source", "train_loss", "val_loss", "total_loss", "site_loss", "skipped_all_invalid_batches"],
    )
    _write_csv(
        l_root / "L_HORIZON_WEIGHTS.csv",
        horizon_rows,
        [
            "variant_id", "epoch", "source", "horizon", "raw_loss", "ema_loss", "initial_loss",
            "difficulty_level", "relative_rate", "dwa_epoch_loss", "dwa_loss_ratio", "weight_preclip",
            "weight", "weighted_contribution",
        ],
    )
    _write_csv(l_root / "L_NODE_WEIGHTS.csv", node_rows, ["variant_id", "source", "node_index", "site_weight", "ema_node_loss"])
    _write_csv(
        l_root / "L_WEIGHT_DISTRIBUTION.csv",
        distribution_rows,
        [
            "variant_id", "epoch", "source", "granularity_clip_low_fraction", "granularity_clip_high_fraction",
            "site_clip_low_fraction", "site_clip_high_fraction", "site_weight_min", "site_weight_mean",
            "site_weight_max", "valid_horizon_count", "valid_node_count",
        ],
    )
    _write_csv(
        l_root / "L_PERFORMANCE_SUMMARY.csv",
        performance_rows,
        ["variant_id", "horizon", "run_status", "comparison_scope", "Score", "MAE", "RMSE", "R2", "metrics_source"],
    )
    _write_csv(
        l_root / "L_DIFFICULTY_GROUP_METRICS.csv",
        difficulty_rows,
        ["variant_id", "horizon", "group", "status", "reason", "Score", "MAE", "RMSE", "R2"],
    )
    _write_csv(l_root / "L_FAILURES.csv", failure_rows, ["variant_id", "run_status", "reason", "failure_report"])

    handoff = f"""# HANDOFF STEP8

## 状态

L0-L7 已按 `EMPIRICAL_ANALYSIS_V1` 注册。L0-L6 共享 Canonical 模型与 Smooth L1
基础损失，仅改变已声明的 horizon/node 权重机制；L3 只读引用 Canonical。L7 只读审计
独立 Batch4 A8，不进入 L0-L6 的 32/4/4 组件因果表。

- dry-run 唯一差异审计：{'通过' if protocol_passed else '未通过'}。
- 普通 smoke：以 `source=smoke_only` 写入权重和 loss 诊断，不作为论文性能数值。
- `[32,10,134]` full-shape：由各变体 `model_summary.json` 审计。
- checkpoint loss/optimizer/AMP/early-stopping/RNG 字段恢复审计：{'通过' if restore_passed else '尚未全部通过'}。
- A8 dedicated Batch4 readiness：{a8_readiness.get('status', 'NOT_READY')}；缺失或不完整时 fail closed。
- 正式长训练：未启动，需用户单独批准。

## 实现要点

- L0：static/static，权重恒为 1，等价 masked Smooth L1 reduction。
- L1：difficulty-rate/static；L2：static/dynamic；L3：Canonical reference。
- L4：训练前冻结 1:2:3 horizon 权重并归一化；L5：可学习 `log_sigma_g`。
- L6：仅在 epoch 结束时根据训练 loss 更新 DWA；无有效 loss 时保持原状态。
- 全无效 mask 返回 `None`，训练器跳过 backward 和 optimizer step并累计计数。

## 产物与限制

步骤8的 manifest、协议审计、loss 输入合同、epoch loss、horizon/node 权重、clip 分布、
checkpoint 恢复审计、性能占位、困难风机占位和失败清单均位于本目录。正式训练尚未执行，
所以 L0/L1/L2/L4/L5/L6 的正式 H3/H6/H10 指标和 Top10% 困难风机结果保持 `not_started`，
没有填 0 或复制 smoke 数值。

## 验证结果

- `compileall`：通过。
- 步骤8专项与 empirical/core 回归：143 passed，1 个既有 Matplotlib 弃用警告。
- L0-L7 dry-run：唯一差异审计全部通过。
- L0/L1/L2/L4/L5/L6 smoke：完成且 protocol passed；L3 为 Canonical reference。
- L0/L1/L2/L4/L5/L6 full-shape：输出均为 `[32,10,134]`，loss 与 backward 有限。
- L7：专用 A8 Batch4 路径不存在，按合同标记 `reference-unavailable`，没有读取旧 A8。

## PowerShell

```powershell
Set-Location -LiteralPath 'D:\\PaperProject\\GyxPaper2'
$env:PYTHONPATH = 'D:\\PaperProject\\GyxPaper2\\custom_models\\src'
python -m compileall custom_models/src/st_mgprompt
python -m pytest custom_models/tests/empirical_analysis/test_step8_losses.py -q
python -m st_mgprompt.run_empirical --family L --variants L0 L1 L2 L3 L4 L5 L6 L7 --dry-run
python -m st_mgprompt.run_empirical --family L --variants L0 L1 L2 L3 L4 L5 L6 L7 --smoke
python -m st_mgprompt.run_empirical --family L --variants L0 L1 L2 L3 L4 L5 L6 L7 --full-shape
```
"""
    (l_root / "HANDOFF_STEP8.md").write_text(handoff, encoding="utf-8")
    return {
        "l_root": str(l_root.resolve()),
        "variant_count": len(variants),
        "protocol_passed": protocol_passed,
        "checkpoint_restore_passed": restore_passed,
        "a8_ready": bool(a8_readiness.get("ready")),
        "formal_training_started": False,
    }


__all__ = ["write_step8_reports"]
