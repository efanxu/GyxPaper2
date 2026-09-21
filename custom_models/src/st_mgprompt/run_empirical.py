from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from st_mgprompt.config import STMGPromptConfig, resolve_project_path
from st_mgprompt.empirical_protocol import (
    EMPIRICAL_FAMILIES,
    EMPIRICAL_PROTOCOL_ID,
    EMPIRICAL_RESULT_ROOT,
    EmpiricalVariant,
    apply_empirical_variant,
    assert_empirical_expected_diff,
    dry_run_report,
    get_empirical_variant,
    write_empirical_matrix,
)
from st_mgprompt.experiment_protocol import (
    CANONICAL_ID,
    COMPONENT_RESULT_ROOT,
    COMPONENT_RUN_ID,
    canonical_directory,
    config_diff,
)
from st_mgprompt.graph_prior import reconstruct_variant_graphs_from_fused, summarize_graph_identity
from st_mgprompt.losses import get_loss_fn
from st_mgprompt.registry import build_model
from st_mgprompt.run_st_mgprompt import _autocast_context, _run_once, resolve_device, update_run_status
from st_mgprompt.step3_reporting import write_step3_reports
from st_mgprompt.step4_reporting import write_step4_reports
from st_mgprompt.step5_reporting import write_step5_reports
from st_mgprompt.summarize_empirical import summarize_empirical


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _effective_source_scope(args: argparse.Namespace, variant: EmpiricalVariant) -> str:
    if variant.family == "D" and args.source_scope == "internal_mechanism":
        return "internal_diffusion"
    return str(args.source_scope)


def _diffusion_config_contract(config: STMGPromptConfig) -> dict[str, Any]:
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
    concat_order = ["input"]
    for direction in directions:
        concat_order.extend(
            f"{direction}_hop_{hop}"
            for hop in range(1, config.diffusion_order_micro + 1)
        )
    micro_count = len(directions) * config.diffusion_order_micro
    macro_count = len(directions) * config.diffusion_order_macro
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


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _merge_statuses(path: Path, new_statuses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    previous: list[dict[str, Any]] = []
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and isinstance(payload.get("statuses"), list):
                previous = [row for row in payload["statuses"] if isinstance(row, dict)]
        except (OSError, json.JSONDecodeError):
            previous = []

    def identity(row: dict[str, Any]) -> tuple[Any, ...]:
        return (
            row.get("variant_id"),
            row.get("seed"),
            row.get("run_dir") or row.get("run_root"),
        )

    merged = {identity(row): row for row in previous}
    for row in new_statuses:
        merged[identity(row)] = row
    return sorted(
        merged.values(),
        key=lambda row: (
            str(row.get("family", "")),
            str(row.get("variant_id", "")),
            int(row.get("seed") or -1),
            str(row.get("run_dir") or row.get("run_root") or ""),
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run empirical_analysis_v1 ST-MGPrompt variants.")
    parser.add_argument("--family", nargs="+", choices=tuple(EMPIRICAL_FAMILIES), default=list(EMPIRICAL_FAMILIES))
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full-shape", action="store_true")
    parser.add_argument("--run-full", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", default=EMPIRICAL_RESULT_ROOT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--profile", default="STMG_FORMAL_V2")
    parser.add_argument("--source-scope", default="internal_mechanism")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def _selected_variants(families: Iterable[str], names: list[str] | None) -> list[EmpiricalVariant]:
    family_ids = [str(item).upper() for item in families]
    if names is None:
        return [variant for family in family_ids for variant in EMPIRICAL_FAMILIES[family].values()]
    selected: list[EmpiricalVariant] = []
    for name in names:
        variant = get_empirical_variant(name)
        if variant.family not in family_ids:
            raise ValueError(f"{variant.variant_id} does not belong to selected families {family_ids}.")
        selected.append(variant)
    return selected


def _variant_run_id(
    base_run_id: str | None,
    variant: EmpiricalVariant,
    selection_size: int,
    *,
    smoke: bool,
) -> str:
    if base_run_id is None:
        suffix = "smoke_v1" if smoke else "empirical_v1"
        return f"{variant.variant_id.lower()}_{suffix}"
    if selection_size == 1:
        return base_run_id
    return f"{base_run_id}_{variant.variant_id.lower()}"


def _reference_source(variant: EmpiricalVariant) -> Path:
    target = variant.paired_reference
    if target in {None, CANONICAL_ID, "T0", "G0", "F7", "L3", "R0"}:
        return canonical_directory(resolve_project_path("."))
    if target in {"A4", "A7", "A8"}:
        return (
            resolve_project_path(COMPONENT_RESULT_ROOT)
            / COMPONENT_RUN_ID
            / target
            / "STMGPrompt_ComponentAblation"
        )
    return resolve_project_path(EMPIRICAL_RESULT_ROOT) / variant.family / str(target)


def _write_reference(
    variant: EmpiricalVariant,
    run_root: Path,
    *,
    protocol_profile: str = "STMG_FORMAL_V2",
    source_scope: str = "internal_mechanism",
) -> dict[str, Any]:
    source = _reference_source(variant)
    formal_variant = variant.paired_reference if variant.paired_reference in {"A4", "A7", "A8"} else "A0"
    source_config_path = source / "effective_config.json"
    if not source_config_path.is_file():
        source_config_path = source / "active_config.json"
    if not source_config_path.is_file():
        source_config_path = source / "config.json"
    source_audit: dict[str, Any] = {
        "formal_variant": formal_variant,
        "config_path": str(source_config_path.resolve()),
        "passed": False,
        "reason": "source config is missing",
    }
    if source_config_path.is_file():
        source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
        source_audit = config_diff(source_config, formal_variant, "component_ablation")
        source_audit["config_path"] = str(source_config_path.resolve())
    payload = {
        "protocol_id": EMPIRICAL_PROTOCOL_ID,
        "variant_id": variant.variant_id,
        "display_name": variant.display_name,
        "family": variant.family,
        "reference_only": True,
        "paired_reference": variant.paired_reference,
        "protocol_profile": protocol_profile,
        "source_scope": source_scope,
        "source_run_dir": str(source.resolve()),
        "source_exists": source.is_dir(),
        "source_semantic_audit": source_audit,
        "created_at": _utc_now(),
        "note": "This is a reference; no checkpoint or metrics were copied.",
    }
    if variant.family == "D":
        reference_config = apply_empirical_variant(None, variant.variant_id, variant.family)
        payload.update(
            {
                "paired_reference": "D4/T0",
                "graph_identity_reference": "G0/CANONICAL",
                **_diffusion_config_contract(reference_config),
            }
        )
    _atomic_json(run_root / "reference.json", payload)
    if not source.is_dir() or not source_audit.get("passed"):
        raise RuntimeError(
            f"Reference source for {variant.variant_id} is missing or fails its formal semantic audit: {source}"
        )
    return payload


def _smoke_config(config: STMGPromptConfig) -> STMGPromptConfig:
    config = deepcopy(config)
    config.smoke = True
    config.smoke_use_synthetic = True
    config.lookback = min(config.lookback, 12)
    config.batch_size = min(config.batch_size, 4)
    config.eval_batch_size = min(config.eval_batch_size, 4)
    config.train_batch_size = None
    config.val_batch_size = None
    config.test_batch_size = None
    config.hidden_dim = min(config.hidden_dim, 16)
    config.epochs = 1
    config.patience = 1
    config.train_sample_stride = 2
    config.val_sample_stride = 1
    config.test_sample_stride = 1
    config.smoke_num_time_steps = max(config.smoke_num_time_steps, 240)
    config.amp_enabled = False
    config.device = "cpu"
    config.diagnostics_level = "minimal"
    config.prediction_accumulation = "streaming"
    return config


def _completed(run_dir: Path) -> bool:
    required = [
        "active_config.json",
        "config.json",
        "run_status.json",
        "train_log.csv",
        "best_checkpoint.pt",
        "metrics.csv",
        "metrics_eval_h3.json",
        "metrics_eval_h6.json",
        "metrics_eval_h10.json",
        "model_summary.json",
        "evaluation_complete.json",
    ]
    return all((run_dir / name).is_file() for name in required)


def _runner_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        resume=bool(args.resume),
        resume_from=None,
        evaluate_only=bool(args.evaluate_only),
        checkpoint="best" if args.evaluate_only else None,
        allow_warm_start_only=False,
        train_only=False,
        skip_test=False,
    )


def _write_identity_contract(
    run_dir: Path,
    formal_config: STMGPromptConfig,
    effective_config: STMGPromptConfig,
    variant: EmpiricalVariant,
    audit: dict[str, Any],
    mode: str,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    diffusion_contract = _diffusion_config_contract(effective_config)
    _atomic_json(run_dir / "resolved_config.json", formal_config.to_dict())
    _atomic_json(run_dir / "effective_config.json", effective_config.to_dict())
    _atomic_json(
        run_dir / "protocol_check.json",
        {
            "protocol_id": EMPIRICAL_PROTOCOL_ID,
            "protocol_profile": formal_config.protocol_profile,
            "source_scope": formal_config.source_scope,
            "mode": mode,
            "unique_difference_audit": audit,
            "formal_batch": {
                "train": formal_config.train_batch_size or formal_config.batch_size,
                "val": formal_config.val_batch_size or formal_config.eval_batch_size,
                "test": formal_config.test_batch_size or formal_config.eval_batch_size,
            },
            "effective_batch": {
                "train": effective_config.train_batch_size or effective_config.batch_size,
                "val": effective_config.val_batch_size or effective_config.eval_batch_size,
                "test": effective_config.test_batch_size or effective_config.eval_batch_size,
            },
            "mask": formal_config.target_mask_col,
            "prediction_start_index_rule": "[t-lookback,t)->[t,t+max_pred_len)",
            "graph_identity_reference": "G0/CANONICAL" if variant.family == "D" else None,
            **(diffusion_contract if variant.family == "D" else {}),
            "passed": bool(audit.get("passed")),
        },
    )
    _atomic_json(
        run_dir / "source_scope.json",
        {
            "protocol_id": EMPIRICAL_PROTOCOL_ID,
            "protocol_profile": formal_config.protocol_profile,
            "source_scope": formal_config.source_scope,
            "variant_id": variant.variant_id,
            "paired_reference": (
                "T0" if variant.family == "T" else ("D4/T0" if variant.family == "D" else variant.paired_reference)
            ),
            "graph_identity_reference": "G0/CANONICAL" if variant.family == "D" else None,
            "reference_only": variant.reference_only,
            "formal_training": mode == "full",
            "smoke_or_preflight": mode in {"smoke", "full_shape"},
            "checkpoint_copied": False,
            "metrics_copied": False,
            **(diffusion_contract if variant.family == "D" else {}),
        },
    )


def _augment_completed_run_artifacts(
    run_dir: Path,
    config: STMGPromptConfig,
    variant: EmpiricalVariant,
    audit: dict[str, Any],
) -> None:
    diffusion_contract = _diffusion_config_contract(config) if variant.family == "D" else {}
    protocol = {}
    protocol_path = run_dir / "protocol_check.json"
    if protocol_path.is_file():
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol.update(
        {
            "protocol_id": EMPIRICAL_PROTOCOL_ID,
            "protocol_profile": config.protocol_profile,
            "source_scope": config.source_scope,
            "empirical_unique_difference_audit": audit,
            "effective_batch": {
                "train": config.train_batch_size or config.batch_size,
                "val": config.val_batch_size or config.eval_batch_size,
                "test": config.test_batch_size or config.eval_batch_size,
            },
            "mask": config.target_mask_col,
            "prediction_start_index_rule": "[t-lookback,t)->[t,t+max_pred_len)",
            "graph_identity_reference": "G0/CANONICAL" if variant.family == "D" else None,
            **diffusion_contract,
        }
    )
    _atomic_json(protocol_path, protocol)
    prediction_path = run_dir / "prediction_metadata.json"
    if prediction_path.is_file():
        prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
        prediction.update(
            {
                "protocol_profile": config.protocol_profile,
                "source_scope": config.source_scope,
                "target_mask": config.target_mask_col,
                "prediction_start_index_rule": "[t-lookback,t)->[t,t+max_pred_len)",
                "node_count": config.num_nodes,
                "horizon": config.max_pred_len,
                "graph_identity_reference": "G0/CANONICAL" if variant.family == "D" else None,
                **diffusion_contract,
                "effective_batch": {
                    "train": config.train_batch_size or config.batch_size,
                    "val": config.val_batch_size or config.eval_batch_size,
                    "test": config.test_batch_size or config.eval_batch_size,
                },
            }
        )
        _atomic_json(prediction_path, prediction)
    summary_path = run_dir / "model_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.update(
            {
                "protocol_profile": config.protocol_profile,
                "source_scope": config.source_scope,
                "variant_id": variant.variant_id,
                "output_shape": [config.batch_size, config.max_pred_len, config.num_nodes]
                if not config.smoke
                else [min(config.batch_size, 4), config.max_pred_len, min(config.num_nodes, 8)],
                **diffusion_contract,
            }
        )
        _atomic_json(summary_path, summary)


def _write_graph_audit_artifacts(run_dir: Path, config: STMGPromptConfig) -> None:
    if not str(config.variant or "").upper().startswith("G"):
        return
    graph_dir = config.resolve_path(config.output_root) / str(config.run_id) / "_graph_artifacts" / config.graph_tag
    metadata_path = graph_dir / "metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        identity = metadata.get("graph_identity")
        if isinstance(identity, dict):
            identity = {**identity, "artifact_directory": str(graph_dir.resolve())}
            _atomic_json(run_dir / "graph_identity.json", identity)
    summary_path = run_dir / "model_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        branch = summary.get("branch_diagnostics", {})
        trace = branch.get("branch_graph_assignment_trace") if isinstance(branch, dict) else None
        if isinstance(trace, dict):
            _atomic_json(
                run_dir / "branch_graph_assignment_trace.json",
                {
                    "variant_id": config.variant,
                    "layers": [
                        {
                            "layer": layer,
                            "fine_branch": trace.get("fine_graph_id"),
                            "coarse_branch": trace.get("coarse_graph_id"),
                            "fine_builder": trace.get("fine_builder"),
                            "coarse_builder": trace.get("coarse_builder"),
                        }
                        for layer in range(int(config.num_coupling_layers))
                    ],
                    **trace,
                },
            )


def _write_diffusion_audit_artifacts(
    run_dir: Path,
    config: STMGPromptConfig,
    *,
    checkpoint_path: Path | None = None,
) -> None:
    if not str(config.variant or "").upper().startswith("D"):
        return
    contract = _diffusion_config_contract(config)
    _atomic_json(
        run_dir / "operator_trace.json",
        {
            "protocol_id": EMPIRICAL_PROTOCOL_ID,
            "variant_id": config.variant,
            "protocol_profile": config.protocol_profile,
            "source_scope": config.source_scope,
            "paired_reference": "D4/T0",
            "graph_identity_reference": "G0/CANONICAL",
            **contract,
        },
    )
    selected_checkpoint = checkpoint_path
    if selected_checkpoint is None:
        for name in ("best_checkpoint.pt", "last_checkpoint.pt"):
            candidate = run_dir / name
            if candidate.is_file():
                selected_checkpoint = candidate
                break
    graph_identity: dict[str, Any] = {
        "protocol_id": EMPIRICAL_PROTOCOL_ID,
        "variant_id": config.variant,
        "graph_identity_reference": "G0/CANONICAL",
        "graph_operator": config.graph_operator,
        "operator_direction": contract["operator_direction"],
        "graph_reconstructed": False,
        "transpose_topk_recomputed": False,
    }
    if selected_checkpoint is not None and selected_checkpoint.is_file():
        import torch

        payload = torch.load(selected_checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("model_state_dict") if isinstance(payload, dict) else None
        if isinstance(state, dict) and {"A_macro_prior", "A_micro_prior"}.issubset(state):
            macro = state["A_macro_prior"].detach().cpu().numpy().astype(np.float32)
            micro = state["A_micro_prior"].detach().cpu().numpy().astype(np.float32)
            graph_identity.update(
                summarize_graph_identity(
                    macro,
                    micro,
                    config,
                    fit_split=(
                        "synthetic_smoke_preflight"
                        if config.smoke
                        else "canonical_train_only_checkpoint"
                    ),
                )
            )
            graph_identity.update(
                {
                    "checkpoint_path": str(selected_checkpoint.resolve()),
                    "macro_buffer": "A_macro_prior",
                    "micro_buffer": "A_micro_prior",
                    "macro_shape": list(macro.shape),
                    "micro_shape": list(micro.shape),
                    "formal_graph_identity_match": not config.smoke and macro.shape == (134, 134),
                }
            )
    _atomic_json(run_dir / "graph_identity.json", graph_identity)
    _atomic_json(
        run_dir / "node_order_manifest.json",
        {
            "protocol_id": EMPIRICAL_PROTOCOL_ID,
            "variant_id": config.variant,
            "graph_identity_reference": "G0/CANONICAL",
            "node_count": graph_identity.get("node_count"),
            "node_order": graph_identity.get("node_order", []),
            "node_order_source": graph_identity.get("node_order_source"),
            "fit_split": graph_identity.get("fit_split"),
        },
    )

    summary_path = run_dir / "model_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    summary.update(contract)
    summary.update(
        {
            "variant_id": config.variant,
            "graph_identity_reference": "G0/CANONICAL",
            "paired_reference": "D4/T0",
        }
    )
    _atomic_json(summary_path, summary)

    prediction = {}
    prediction_path = run_dir / "prediction_metadata.json"
    if prediction_path.is_file():
        prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    environment_path = run_dir / "environment.json"
    efficiency = {
        "protocol_id": EMPIRICAL_PROTOCOL_ID,
        "variant_id": config.variant,
        "projection_mode": contract["projection_mode"],
        "diffusion_state_count": contract["diffusion_state_count"],
        "projection_input_dim_micro": contract["projection_input_dim_micro"],
        "projection_input_dim_macro": contract["projection_input_dim_macro"],
        "projection_output_dim": contract["projection_output_dim"],
        "parameter_count": summary.get("total_parameters"),
        "trainable_parameter_count": summary.get("trainable_parameters"),
        "model_file_size_bytes": (
            selected_checkpoint.stat().st_size
            if selected_checkpoint is not None and selected_checkpoint.is_file()
            else None
        ),
        "total_train_seconds": summary.get("train_time_sec"),
        "best_epoch": summary.get("best_epoch"),
        "inference_latency_ms": prediction.get("inference_time_ms_per_window"),
        "throughput": prediction.get("inference_windows_per_sec"),
        "peak_memory_mb": summary.get("peak_memory_mb"),
        "effective_batch": {
            "train": config.train_batch_size or config.batch_size,
            "val": config.val_batch_size or config.eval_batch_size,
            "test": config.test_batch_size or config.eval_batch_size,
        },
        "hardware_software_environment": str(environment_path.resolve()) if environment_path.is_file() else None,
        "measurement_scope": "smoke_preflight" if config.smoke else "full_shape_or_formal",
    }
    _atomic_json(run_dir / "efficiency.json", efficiency)


def _write_alignment_trace(
    path: Path,
    *,
    lookback: int,
    windows: list[int],
    sample_count: int,
    node_count: int,
    full_trace: bool,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample",
        "node",
        "scale_window",
        "output_position",
        "source_interval_start",
        "source_interval_end",
        "causal_cutoff",
        "access_upper_bound",
        "upsample_rule",
        "tail_rule",
    ]
    rows_written = 0
    max_access_le_cutoff = True
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        samples = range(sample_count) if full_trace else range(1)
        nodes = range(node_count) if full_trace else range(1)
        for sample in samples:
            for node in nodes:
                for window in windows:
                    for position in range(lookback):
                        anchor = (position // window) * window
                        start = max(0, anchor - window + 1)
                        max_access_le_cutoff &= anchor <= position
                        writer.writerow(
                            {
                                "sample": sample,
                                "node": node,
                                "scale_window": window,
                                "output_position": position,
                                "source_interval_start": start,
                                "source_interval_end": anchor,
                                "causal_cutoff": position,
                                "access_upper_bound": anchor,
                                "upsample_rule": "causal_repeat",
                                "tail_rule": "repeat_latest_causal_anchor",
                            }
                        )
                        rows_written += 1
    return {
        "path": str(path.resolve()),
        "rows": rows_written,
        "full_trace": full_trace,
        "max_access_le_cutoff": max_access_le_cutoff,
    }


def _set_synthetic_robust_statistics(model, num_nodes: int) -> None:
    vadsp = getattr(model, "vadsp", None)
    if vadsp is None or not getattr(vadsp, "requires_robust_statistics", False):
        return
    source_count = len(vadsp.source_indices)
    shape = (num_nodes, source_count) if vadsp.volatility_mode == "per_node" else (1, source_count)
    import torch

    zeros = torch.zeros(shape)
    ones = torch.ones(shape)
    vadsp.set_robust_statistics(
        {
            "q25": zeros,
            "median": zeros,
            "q75": ones,
            "center": zeros,
            "scale": ones,
            "delta_abs_iqr": ones,
            "delta_log_median": zeros,
            "delta_log_iqr": ones,
            "fit_split": "synthetic_full_shape_preflight",
            "fit_source": "deterministic_preflight_fixture",
            "feature_space": "model_input_scaled",
            "exact": True,
            "device": "cpu",
            "mode": vadsp.volatility_mode,
            "source_cols": list(vadsp.volatility_source_cols),
            "num_time_points": 145,
            "num_time_deltas": 144,
            "shape": {"center": list(shape), "scale": list(shape)},
            "depends_on_window_stride": False,
        }
    )


def _run_full_shape(
    args: argparse.Namespace,
    variant: EmpiricalVariant,
    run_dir: Path,
) -> dict[str, Any]:
    if variant.reference_only:
        reference = _write_reference(
            variant,
            run_dir,
            protocol_profile=args.profile,
            source_scope=_effective_source_scope(args, variant),
        )
        return {
            "variant_id": variant.variant_id,
            "family": variant.family,
            "status": "REFERENCE_ONLY",
            "run_dir": str(run_dir.resolve()),
            "source_exists": reference["source_exists"],
        }
    formal = apply_empirical_variant(None, variant.variant_id, variant.family)
    formal.protocol_profile = args.profile
    formal.device = args.device
    audit = assert_empirical_expected_diff(formal, variant.variant_id, variant.family)
    formal.source_scope = _effective_source_scope(args, variant)
    effective = deepcopy(formal)
    effective.diagnostics_level = "minimal"
    effective.prediction_accumulation = "streaming"
    _write_identity_contract(run_dir, formal, effective, variant, audit, "full_shape")
    started_at = _utc_now()
    try:
        import torch

        device = resolve_device(args.device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        canonical_checkpoint = canonical_directory(resolve_project_path(".")) / "best_checkpoint.pt"
        if not canonical_checkpoint.is_file():
            raise FileNotFoundError(f"Canonical checkpoint is missing: {canonical_checkpoint}")
        checkpoint = torch.load(canonical_checkpoint, map_location="cpu", weights_only=False)
        state = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else None
        if not isinstance(state, dict):
            raise ValueError("Canonical checkpoint does not contain model_state_dict")
        graph_keys = ("A_macro_prior", "A_micro_prior")
        if any(key not in state for key in graph_keys):
            raise ValueError(f"Canonical checkpoint is missing graph buffers: {graph_keys}")
        macro = state[graph_keys[0]].detach().cpu().numpy().astype(np.float32)
        micro = state[graph_keys[1]].detach().cpu().numpy().astype(np.float32)
        if macro.shape != (effective.num_nodes, effective.num_nodes) or micro.shape != macro.shape:
            raise ValueError(f"Canonical graph shape mismatch: macro={macro.shape}, micro={micro.shape}")
        transform_metadata: dict[str, Any] = {
            "macro_source_components": ["distance", "train_statistics"],
            "micro_source_components": ["distance", "train_statistics"],
            "reconstructed_from_fused_checkpoint": False,
            "graph_rewire_seed": None,
        }
        if effective.graph_prior_component != "fused" or effective.graph_rewire_mode != "none":
            setattr(effective, "_graph_num_nodes", effective.num_nodes)
            setattr(effective, "_graph_turbine_ids", list(range(1, effective.num_nodes + 1)))
            macro, micro, transform_metadata = reconstruct_variant_graphs_from_fused(
                macro,
                micro,
                effective.resolve_path(effective.location_path),
                effective,
            )
        graph = {
            "A_macro_trend": macro,
            "A_micro_local": micro,
            "metadata": {
                **checkpoint.get("graph_metadata", {"graph_uses_train_only_statistics": True}),
                **transform_metadata,
                "branch_graph_assignment": effective.branch_graph_assignment,
                "graph_prior_component": effective.graph_prior_component,
                "adaptive_support_mode": effective.adaptive_support_mode,
                "graph_rewire_mode": effective.graph_rewire_mode,
            },
        }
        graph_identity = summarize_graph_identity(
            macro,
            micro,
            effective,
            source_components=transform_metadata["macro_source_components"],
            fit_split="canonical_train_only_checkpoint_reconstruction",
        )
        _atomic_json(
            run_dir / "graph_identity.json",
            {
                **graph_identity,
                "graph_tag": effective.graph_tag,
                "source": "canonical_checkpoint_buffers",
                "checkpoint_path": str(canonical_checkpoint.resolve()),
                "macro_buffer": graph_keys[0],
                "micro_buffer": graph_keys[1],
                "macro_shape": list(macro.shape),
                "micro_shape": list(micro.shape),
                "graph_operator": effective.graph_operator,
                "transform_metadata": transform_metadata,
            },
        )
        model = build_model(effective, input_dim=len(effective.feature_cols), graph_data=graph).to(device)
        _set_synthetic_robust_statistics(model, effective.num_nodes)
        model.train()
        x = torch.randn(
            effective.train_batch_size or effective.batch_size,
            effective.lookback,
            effective.num_nodes,
            len(effective.feature_cols),
            device=device,
        )
        target = torch.randn(
            x.shape[0], effective.max_pred_len, effective.num_nodes, device=device
        )
        mask = torch.ones_like(target)
        with _autocast_context(device, bool(effective.amp_enabled)):
            output = model(x)
        loss_fn = get_loss_fn(effective.loss_function, config=effective, num_nodes=effective.num_nodes)
        if hasattr(loss_fn, "to"):
            loss_fn = loss_fn.to(device)
        with torch.autocast(device_type=device.type, enabled=False):
            loss = loss_fn(output["pred"].float(), target.float(), mask.float())
        if loss is None or not torch.isfinite(loss).all():
            raise RuntimeError("Full-shape loss is missing or non-finite.")
        loss.backward()
        finite_gradient = any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        if not finite_gradient:
            raise RuntimeError("Full-shape backward produced no finite gradient.")
        total = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        first_block = model.coupling_blocks[0]
        fine_temporal = first_block.fine_micro_graph_temporal_encoder.block.causal_tcn
        coarse_temporal = first_block.coarse_macro_graph_temporal_encoder.block.causal_tcn
        shared = fine_temporal is coarse_temporal
        summary = {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "shared_parameter_count": sum(p.numel() for p in fine_temporal.parameters()) if shared else 0,
            "branch_output_dimensions": {"fine": effective.hidden_dim, "coarse": effective.hidden_dim},
            "input_shape": list(x.shape),
            "output_shape": list(output["pred"].shape),
            "loss": float(loss.detach().cpu()),
            "loss_finite": True,
            "gradient_finite": True,
            "device": str(device),
            "amp_enabled": bool(effective.amp_enabled),
            "peak_memory_mb": (
                float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else None
            ),
            "temporal_branch_transform": effective.temporal_branch_transform,
            "coarse_alignment_mode": effective.coarse_alignment_mode,
            "cross_granularity_interaction": effective.cross_granularity_interaction,
            "direct_concat_mlp": {
                "layers": effective.direct_concat_mlp_layers,
                "hidden_dim": effective.direct_concat_mlp_hidden_dim,
                "activation": effective.direct_concat_mlp_activation,
                "dropout": effective.dropout,
                "output_dim": effective.hidden_dim,
                "macro_prompt_instantiated": first_block.macro_prompt_encoder is not None,
                "cross_fusion_instantiated": first_block.symmetric_cross_fusion is not None,
            },
            "branch_graph_assignment_trace": output["aux"].get("branch_graph_assignment_trace"),
            "graph_prior_component": effective.graph_prior_component,
            "adaptive_support_mode": effective.adaptive_support_mode,
            "graph_rewire_mode": effective.graph_rewire_mode,
            **(_diffusion_config_contract(effective) if variant.family == "D" else {}),
        }
        _atomic_json(run_dir / "model_summary.json", summary)
        _write_diffusion_audit_artifacts(
            run_dir,
            effective,
            checkpoint_path=canonical_checkpoint,
        )
        trace = output["aux"].get("branch_graph_assignment_trace")
        if isinstance(trace, dict):
            _atomic_json(
                run_dir / "branch_graph_assignment_trace.json",
                {
                    "variant_id": variant.variant_id,
                    "layers": [
                        {
                            "layer": layer,
                            "fine_branch": trace.get("fine_graph_id"),
                            "coarse_branch": trace.get("coarse_graph_id"),
                            "fine_builder": trace.get("fine_builder"),
                            "coarse_builder": trace.get("coarse_builder"),
                        }
                        for layer in range(int(effective.num_coupling_layers))
                    ],
                    **trace,
                },
            )
        trace_summary = None
        if variant.variant_id == "T4":
            trace_summary = _write_alignment_trace(
                run_dir / "coarse_alignment_trace.csv",
                lookback=effective.lookback,
                windows=effective.coarse_windows,
                sample_count=int(x.shape[0]),
                node_count=effective.num_nodes,
                full_trace=False,
            )
            _atomic_json(run_dir / "coarse_alignment_summary.json", trace_summary)
        _atomic_json(run_dir / "failure_report.json", {"status": "not_applicable", "reasons": []})
        _atomic_json(
            run_dir / "run_status.json",
            {
                "status": "full_shape_completed",
                "variant_id": variant.variant_id,
                "started_at": started_at,
                "finished_at": _utc_now(),
                "trace_summary": trace_summary,
            },
        )
        result = {
            "variant_id": variant.variant_id,
            "family": variant.family,
            "status": "FULL_SHAPE_COMPLETED",
            "run_dir": str(run_dir.resolve()),
            "output_shape": summary["output_shape"],
            "peak_memory_mb": summary["peak_memory_mb"],
        }
        del model, x, target, mask, output, loss, loss_fn
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return result
    except Exception as exc:
        failure = {
            "protocol_id": EMPIRICAL_PROTOCOL_ID,
            "variant_id": variant.variant_id,
            "status": "failed",
            "started_at": started_at,
            "finished_at": _utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _atomic_json(run_dir / "failure_report.json", failure)
        _atomic_json(run_dir / "run_status.json", failure)
        if not args.continue_on_error:
            raise
        return {**failure, "run_dir": str(run_dir.resolve())}


def _run_seed(
    args: argparse.Namespace,
    variant: EmpiricalVariant,
    run_root: Path,
    seed: int,
    *,
    direct_layout: bool = False,
) -> dict[str, Any]:
    formal_config = apply_empirical_variant(None, variant.variant_id, variant.family)
    formal_config.protocol_profile = args.profile
    audit = assert_empirical_expected_diff(formal_config, variant.variant_id, variant.family)
    formal_config.source_scope = _effective_source_scope(args, variant)
    config = deepcopy(formal_config)
    config.seed = int(seed)
    config.device = args.device
    config.output_root = str(run_root.parent.resolve())
    config.run_id = run_root.name
    if args.smoke:
        config = _smoke_config(config)
    run_dir = run_root / config.model_name if direct_layout else run_root / f"seed_{seed}" / config.model_name
    if args.skip_completed and _completed(run_dir):
        return {
            "variant_id": variant.variant_id,
            "seed": seed,
            "status": "completed",
            "decision": "SKIP_COMPLETED",
            "run_dir": str(run_dir.resolve()),
        }
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_identity_contract(
        run_dir,
        formal_config,
        config,
        variant,
        audit,
        "smoke" if args.smoke else ("evaluate_only" if args.evaluate_only else "full"),
    )
    _atomic_json(run_dir / "empirical_config_diff.json", audit)
    _atomic_json(
        run_dir / "empirical_identity.json",
        {
            "protocol_id": EMPIRICAL_PROTOCOL_ID,
            "variant": variant.to_dict(),
            "seed": seed,
            "mode": "smoke" if args.smoke else ("evaluate_only" if args.evaluate_only else "full"),
        },
    )
    started_at = _utc_now()
    try:
        result = _run_once(_runner_args(args), config, run_dir)
    except Exception as exc:
        failure = {
            "protocol_id": EMPIRICAL_PROTOCOL_ID,
            "variant_id": variant.variant_id,
            "seed": seed,
            "status": "failed",
            "started_at": started_at,
            "finished_at": _utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _atomic_json(run_dir / "failure.json", failure)
        _atomic_json(run_dir / "failure_report.json", failure)
        update_run_status(run_dir, "PROCESS_EXCEPTION", failure)
        if not args.continue_on_error:
            raise
        return {**failure, "run_dir": str(run_dir.resolve())}
    _augment_completed_run_artifacts(run_dir, config, variant, audit)
    _write_graph_audit_artifacts(run_dir, config)
    _write_diffusion_audit_artifacts(run_dir, config)
    trace_summary = None
    if variant.variant_id == "T4":
        trace_summary = _write_alignment_trace(
            run_root / "coarse_alignment_trace.csv",
            lookback=config.lookback,
            windows=config.coarse_windows,
            sample_count=int(config.batch_size),
            node_count=min(config.num_nodes, 8) if config.smoke and config.smoke_use_synthetic else config.num_nodes,
            full_trace=bool(config.smoke),
        )
        _atomic_json(run_root / "coarse_alignment_summary.json", trace_summary)
    _atomic_json(run_dir / "failure_report.json", {"status": "not_applicable", "reasons": []})
    return {
        "variant_id": variant.variant_id,
        "seed": seed,
        "status": "smoke_completed" if args.smoke else "completed",
        "decision": "EVALUATE_ONLY" if args.evaluate_only else ("RESUME" if args.resume else "TRAIN"),
        "run_dir": result["run_dir"],
        "protocol_passed": bool(result["protocol_passed"]),
        "started_at": started_at,
        "finished_at": _utc_now(),
        "trace_summary": trace_summary,
    }


def run_empirical(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    execution_modes = sum(
        bool(value) for value in (args.dry_run, args.smoke, args.full_shape, args.run_full, args.evaluate_only)
    )
    if execution_modes > 1:
        raise ValueError("Choose only one of --dry-run, --smoke, --full-shape, --run-full, or --evaluate-only.")
    if execution_modes == 0:
        args.dry_run = True
    if args.resume and not (args.run_full or args.smoke):
        raise ValueError("--resume requires --run-full or --smoke.")
    variants = _selected_variants(args.family, args.variants)
    if not variants:
        raise ValueError("No empirical variants selected.")
    output_root = resolve_project_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    write_empirical_matrix(output_root / "variant_config_matrix.csv")
    statuses: list[dict[str, Any]] = []
    for variant in variants:
        report = dry_run_report(variant)
        if variant.family == "T":
            if args.dry_run:
                run_root = output_root / "T" / "dry_run" / variant.variant_id
            elif args.smoke:
                run_root = output_root / "T" / "smoke" / variant.variant_id
            elif args.full_shape:
                run_root = output_root / "T" / "full_shape" / variant.variant_id
            else:
                run_root = output_root / "T" / f"seed_{args.seeds[0]}" / variant.variant_id
        else:
            run_id = _variant_run_id(args.run_id, variant, len(variants), smoke=bool(args.smoke))
            run_root = output_root / variant.family / run_id
        run_root.mkdir(parents=True, exist_ok=True)
        _atomic_json(run_root / "dry_run_report.json", report)
        if args.dry_run and variant.family == "T":
            legacy_root = output_root / "T" / _variant_run_id(None, variant, len(variants), smoke=False)
            _atomic_json(legacy_root / "dry_run_report.json", report)
        if args.dry_run:
            statuses.append({**report, "status": "DRY_RUN", "run_root": str(run_root.resolve())})
            print(json.dumps(statuses[-1], ensure_ascii=False), flush=True)
            continue
        if variant.implementation_status == "pending":
            status = {
                **report,
                "status": "PENDING_IMPLEMENTATION",
                "run_root": str(run_root.resolve()),
            }
            statuses.append(status)
            print(json.dumps(status, ensure_ascii=False), flush=True)
            continue
        if args.full_shape:
            status = _run_full_shape(args, variant, run_root)
            statuses.append(status)
            print(json.dumps(status, ensure_ascii=False), flush=True)
            continue
        if variant.reference_only:
            reference = _write_reference(
                variant,
                run_root,
                protocol_profile=args.profile,
                source_scope=_effective_source_scope(args, variant),
            )
            status = {
                "variant_id": variant.variant_id,
                "family": variant.family,
                "status": "REFERENCE_ONLY",
                "run_root": str(run_root.resolve()),
                "source_exists": reference["source_exists"],
            }
            statuses.append(status)
            print(json.dumps(status, ensure_ascii=False), flush=True)
            continue
        for seed in args.seeds:
            seed_run_root = (
                output_root / "T" / f"seed_{int(seed)}" / variant.variant_id
                if variant.family == "T" and not args.smoke
                else run_root
            )
            status = _run_seed(
                args,
                variant,
                seed_run_root,
                int(seed),
                direct_layout=variant.family == "T",
            )
            statuses.append(status)
            print(json.dumps(status, ensure_ascii=False), flush=True)
    status_path = output_root / "experiment_status.json"
    merged_statuses = _merge_statuses(status_path, statuses)
    _atomic_json(
        status_path,
        {
            "protocol_id": EMPIRICAL_PROTOCOL_ID,
            "updated_at": _utc_now(),
            "mode": "dry_run"
            if args.dry_run
            else (
                "smoke"
                if args.smoke
                else ("full_shape" if args.full_shape else ("evaluate_only" if args.evaluate_only else "full"))
            ),
            "statuses": merged_statuses,
        },
    )
    step3 = (
        write_step3_reports(output_root)
        if any(variant.family == "T" for variant in variants) and not args.dry_run
        else None
    )
    step4 = (
        write_step4_reports(output_root)
        if any(variant.family == "G" for variant in variants) and not args.dry_run
        else None
    )
    step5 = (
        write_step5_reports(output_root)
        if any(variant.family == "D" for variant in variants) and not args.dry_run
        else None
    )
    summary = summarize_empirical(output_root)
    return {
        "output_root": str(output_root.resolve()),
        "statuses": statuses,
        "step3": step3,
        "step4": step4,
        "step5": step5,
        "summary": summary,
    }


def main() -> None:
    result = run_empirical()
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
