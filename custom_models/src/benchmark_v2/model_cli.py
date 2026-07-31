from __future__ import annotations

import os
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .artifacts import (
    atomic_write_csv,
    atomic_write_json,
    safe_run_dir,
    validate_run,
    write_status,
)
from .contracts import BenchmarkBatch
from .data import SDWPFDataProvider, StandardScaler
from .engine import Evaluator, Trainer
from .errors import ModelUnavailableError
from .experiments.e5_common_loss.config_diff import (
    optimizer_group_signature,
    state_dict_hash,
)
from .experiments.e5_common_loss.contracts import SMOKE_OUTPUT_ROOT_RELATIVE
from .experiments.e5_common_loss.loss_profile import (
    CLI_PROFILE_ID as E5_PROFILE_ID,
    DEFAULT_PROFILE_ID,
)
from .experiments.e5_common_loss.runner import (
    apply_experiment_profile,
    normalize_profile,
    runtime_loss,
)
from .experiments.e5_common_loss.scope27_contract import (
    E5_SCOPE27_ID,
    PREFLIGHT_POLICY_ID,
    is_scope27_train_request,
    validate_scope27_request,
)
from .losses import get_loss
from .metrics import evaluate_horizons
from .model_runtime import ModelRuntime, build_model_runtime
from .models.graph_models.adaptive_common import (
    canonical_tensor_hash,
    learned_graph_summary,
)
from .protocol import check_protocol, load_protocol
from .registry import load_registry
from .runtime import (
    FORMAL_ROOT,
    PROJECT_ROOT,
    SMOKE_ROOT,
    ProviderBatchIterable,
    environment_snapshot,
)
from .seeds import seed_everything
from .training_profiles import (
    apply_training_profile,
    batch_identity,
    load_training_profile,
    resolved_batch_sizes,
)


E1_A_SMOKE_ROOT = SMOKE_ROOT / "e1_a"
E1_B_SMOKE_ROOT = SMOKE_ROOT / "e1_b"
E2_A_SMOKE_ROOT = SMOKE_ROOT / "e2_a"
E2_B_SMOKE_ROOT = SMOKE_ROOT / "e2_b"
E2_C_SMOKE_ROOT = SMOKE_ROOT / "e2_c"
E2_D_SMOKE_ROOT = SMOKE_ROOT / "e2_d"
E3_B_SMOKE_ROOT = SMOKE_ROOT / "e3_b"
E3_C_SMOKE_ROOT = SMOKE_ROOT / "e3_c"
E1_B_MODELS = {"dlinear", "lightts", "tide", "segrnn"}
E2_A_MODELS = {"transformer", "patchtst", "itransformer", "timexer"}
E2_B_MODELS = {"timesnet", "micn", "wpmixer", "multipatchformer"}
E2_C_MODELS = {"timemixer", "tsmixer", "frets"}
E2_D_MODELS = {"crossformer", "msgnet", "timefilter"}
E3_B_MODELS = {"gcn", "stgcn", "dcrnn"}
E3_C_MODELS = {"graph_wavenet", "mtgnn", "agcrn", "stid"}
NATIVE_NODE_MODELS = E3_B_MODELS | E3_C_MODELS
IDENTITY_METADATA_KEYS = (
    "graph_id",
    "graph_context_id",
    "graph_protocol_hash",
    "node_order_hash",
    "graph_bundle_hash",
    "location_source_hash",
    "selected_k",
    "graph_support_names",
    "graph_support_hashes",
    "graph_runtime_dtype",
    "uses_physical_support",
    "physical_support_names",
    "physical_support_hashes",
    "adaptive_graph_policy",
    "node_identity_policy",
    "temporal_identity_policy",
    "adaptive_initialization_policy",
)
BATCH_IDENTITY_KEYS = (
    "base_benchmark_protocol_hash",
    "training_batch_profile_id",
    "training_batch_profile_hash",
    "train_batch_size",
    "val_batch_size",
    "test_batch_size",
    "gradient_accumulation_steps",
    "effective_train_batch_size",
)
DEFAULT_INPUT_PATH = PROJECT_ROOT / "dataset" / "sdwpf_model_input_base.parquet"
DEFAULT_TARGET_PATH = PROJECT_ROOT / "dataset" / "sdwpf_eval_target.parquet"


def _parameter_summary(runtime: ModelRuntime) -> dict[str, Any]:
    config = runtime.effective_config
    native_node_model = runtime.model_id in NATIVE_NODE_MODELS
    summary = {
        "model_id": runtime.model_id,
        "implementation_class": type(runtime.model).__name__,
        "parameter_count": runtime.parameter_count,
        "trainable_parameter_count": runtime.trainable_parameter_count,
        "initialization_policy": config.get("initialization_policy", "not_applicable"),
        "optimizer": config.get("optimizer"),
        "learning_rate": config.get("learning_rate"),
        "weight_decay": config.get("weight_decay"),
        "scheduler": config.get("scheduler"),
        "gradient_clip": config.get("gradient_clip"),
        "node_semantics": (
            config.get("node_semantics", "node_shared")
        ),
        "cross_node_interaction": bool(
            config.get("cross_node_interaction", False)
        ),
        "uses_graph": bool(config.get("uses_graph", False)),
        "uses_node_embedding": bool(config.get("uses_node_embedding", False)),
        "uses_future_covariates": False,
    }
    if native_node_model:
        summary.update(
            {
                key: config[key]
                for key in IDENTITY_METADATA_KEYS
                if key in config
            }
        )
    return summary


def _write_common(
    run_dir: Path,
    runtime: ModelRuntime,
    protocol,
    *,
    profile: str,
    run_mode: str,
    data_signature: dict[str, Any],
    formal_training: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    effective = {
        **runtime.effective_config,
        "model_id": runtime.model_id,
        "run_mode": run_mode,
        "artifact_profile": profile,
        "formal_registry_entry": True,
        "formal_training": bool(formal_training),
    }
    resolved = dict(effective)
    atomic_write_json(run_dir / "resolved_config.json", resolved)
    atomic_write_json(run_dir / "effective_config.json", effective)
    protocol_check = check_protocol(protocol, mode=run_mode)
    graph_identity = {
        key: effective[key]
        for key in (*IDENTITY_METADATA_KEYS, *BATCH_IDENTITY_KEYS)
        if key in effective
    }
    protocol_check.update(graph_identity)
    atomic_write_json(run_dir / "protocol_check.json", protocol_check)
    atomic_write_json(run_dir / "environment.json", environment_snapshot())
    atomic_write_json(run_dir / "data_signature.json", data_signature)
    model_summary = {
        **_parameter_summary(runtime),
        "formal_registry_entry": True,
        "run_mode": run_mode,
        "artifact_profile": profile,
        "effective_model_config": runtime.effective_config,
    }
    atomic_write_json(run_dir / "model_summary.json", model_summary)
    atomic_write_json(
        run_dir / "artifact_manifest.json",
        {
            "schema_version": "artifact_schema_v1",
            "artifact_profile": profile,
            "protocol_hash": protocol.protocol_hash,
            "run_mode": run_mode,
            "formal_training": bool(formal_training),
            "model_id": runtime.model_id,
            **graph_identity,
        },
    )
    return resolved, effective


def _write_metrics(
    run_dir: Path,
    metrics: list[dict[str, Any]],
    *,
    runtime: ModelRuntime,
    protocol,
    prediction_shape: list[int],
    source_checkpoint: str | None,
) -> None:
    def json_safe(value):
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        return value

    by_horizon = {int(row["horizon"]): row for row in metrics}
    for horizon in (3, 6, 10):
        atomic_write_json(
            run_dir / f"metrics_eval_h{horizon}.json",
            json_safe(by_horizon[horizon]),
        )
    atomic_write_csv(
        run_dir / "metrics.csv",
        [
            "horizon",
            "MAE",
            "RMSE",
            "R2",
            "Score",
            "score",
            "valid_target_count",
            "total_target_count",
            "valid_target_ratio",
            "status",
        ],
        metrics,
    )
    graph_identity = {
        key: runtime.effective_config[key]
        for key in (*IDENTITY_METADATA_KEYS, *BATCH_IDENTITY_KEYS)
        if key in runtime.effective_config
    }
    atomic_write_json(
        run_dir / "prediction_metadata.json",
        {
            "run_mode": runtime.effective_config.get("run_mode"),
            "model_id": runtime.model_id,
            "formal_registry_entry": True,
            "formal_training": bool(
                runtime.effective_config.get("optimizer")
                and runtime.effective_config.get("run_mode") == "formal"
            ),
            "protocol_hash": protocol.protocol_hash,
            "prediction_shape": prediction_shape,
            "target_column": "Patv_raw",
            "output_space_before_inverse": "normalized_target_space",
            "output_space_after_inverse": "physical_kW",
            "source_checkpoint": source_checkpoint,
            **graph_identity,
        },
    )


def _move_batch(batch: BenchmarkBatch, device) -> BenchmarkBatch:
    move = lambda value: value.to(device) if hasattr(value, "to") else value
    return BenchmarkBatch(
        x=move(batch.x),
        target=move(batch.target),
        mask=move(batch.mask),
        sample_ids=batch.sample_ids,
        window_end_indices=batch.window_end_indices,
        node_ids=batch.node_ids,
        x_mark=move(batch.x_mark) if batch.x_mark is not None else None,
        y_mark=move(batch.y_mark) if batch.y_mark is not None else None,
        split=batch.split,
        metadata=batch.metadata,
        target_raw_or_inverse_transform=move(batch.target_raw_or_inverse_transform)
        if batch.target_raw_or_inverse_transform is not None
        else None,
    )


def _evaluate(
    runtime: ModelRuntime,
    batches: Iterable[BenchmarkBatch],
    protocol,
    *,
    device: str,
    diagnostic_loss_fn=None,
) -> tuple[list[dict[str, Any]], list[int], float | None]:
    import torch

    target_device = torch.device(device)
    runtime.model.to(target_device).eval()
    predictions, targets, masks = [], [], []
    normalized_predictions, normalized_targets = [], []
    with torch.no_grad():
        for batch in batches:
            moved = _move_batch(batch, target_device)
            output = runtime.adapter(
                runtime.model,
                moved,
                expected_horizon=int(protocol["max_pred_len"]),
                expected_features=int(protocol["feature_count"]),
            )
            predictions.append(runtime.inverse_target(output.prediction).detach().cpu())
            if diagnostic_loss_fn is not None:
                normalized_predictions.append(output.prediction.detach().cpu())
                normalized_targets.append(moved.target.detach().cpu())
            target = (
                moved.target_raw_or_inverse_transform
                if moved.target_raw_or_inverse_transform is not None
                else runtime.inverse_target(moved.target)
            )
            targets.append(target.detach().cpu())
            masks.append(moved.mask.detach().cpu())
    if not predictions:
        raise RuntimeError("Evaluation loader produced no batches.")
    pred = torch.cat(predictions)
    target = torch.cat(targets)
    mask = torch.cat(masks)
    metrics = evaluate_horizons(
        pred,
        target,
        mask,
        protocol["eval_horizons"],
        num_nodes=int(pred.shape[1]),
        physical_clip=(
            float(protocol["physical_power_min_kw"]),
            float(protocol["physical_power_max_kw"]),
        ),
    )
    diagnostic = None
    if diagnostic_loss_fn is not None:
        diagnostic_tensor = diagnostic_loss_fn(
            torch.cat(normalized_predictions),
            torch.cat(normalized_targets),
            mask,
        )
        diagnostic = (
            None
            if diagnostic_tensor is None
            else float(diagnostic_tensor.detach().cpu())
        )
    return metrics, list(pred.shape), diagnostic


def _synthetic_scalers() -> tuple[StandardScaler, StandardScaler]:
    input_scaler = StandardScaler(
        mean=np.linspace(10.0, 160.0, 16, dtype=np.float64),
        std=np.linspace(1.0, 16.0, 16, dtype=np.float64),
    )
    target_scaler = StandardScaler(
        mean=np.asarray(500.0, dtype=np.float64),
        std=np.asarray(125.0, dtype=np.float64),
    )
    return input_scaler, target_scaler


def _synthetic_batch(
    *,
    batch_size: int,
    time_steps: int,
    nodes: int,
    seed: int = 2026,
    canonical_graph_nodes: bool = False,
) -> BenchmarkBatch:
    import torch

    generator = torch.Generator().manual_seed(seed)
    input_scaler, target_scaler = _synthetic_scalers()
    physical_power = torch.rand(
        batch_size, time_steps, nodes, generator=generator
    ) * 1000.0
    x = torch.randn(
        batch_size, time_steps, nodes, 16, generator=generator
    )
    power_index = 15
    x[..., power_index] = (
        physical_power - float(input_scaler.mean[power_index])
    ) / float(input_scaler.std[power_index])
    target_raw = torch.rand(batch_size, nodes, 10, generator=generator) * 1000.0
    target = (target_raw - float(target_scaler.mean)) / float(target_scaler.std)
    mask = torch.ones(batch_size, nodes, 10, dtype=torch.bool)
    mask[:, 0, 0] = False
    return BenchmarkBatch(
        x=x,
        target=target,
        mask=mask,
        target_raw_or_inverse_transform=target_raw,
        sample_ids=[f"e1-a-synthetic-{index}" for index in range(batch_size)],
        window_end_indices=list(range(batch_size)),
        node_ids=(
            list(range(1, nodes + 1))
            if canonical_graph_nodes
            else list(range(nodes))
        ),
        split="train",
        metadata={
            "contains_future_target": False,
            "target_model_space": "standard_zscore",
            "target_raw_space": "kW",
            "history_end_timestamp": [
                "2021-01-04 12:00:00" for _ in range(batch_size)
            ],
            "history_end_time_of_day_id": [72 for _ in range(batch_size)],
            "history_end_day_of_week_id": [0 for _ in range(batch_size)],
            "timezone_policy": "SOURCE_NAIVE_UNCHANGED",
        },
    )


def _model_identity_diagnostic(runtime: ModelRuntime) -> dict[str, Any] | None:
    model = runtime.model
    if runtime.model_id == "graph_wavenet":
        adjacency = model.adaptive_adjacency()
        return learned_graph_summary(
            adjacency,
            physical_supports={
                "P_forward": model.P_forward,
                "P_reverse": model.P_reverse,
            },
        )
    if runtime.model_id == "mtgnn":
        adjacency = model.learned_adjacency()
        return learned_graph_summary(adjacency, top_k=20)
    if runtime.model_id == "agcrn":
        adjacency = model.adaptive_adjacency()
        return learned_graph_summary(adjacency)
    if runtime.model_id == "stid":
        embedding = model.node_embedding.detach()
        normalized = embedding / embedding.norm(dim=1, keepdim=True).clamp_min(
            1e-12
        )
        cosine = normalized @ normalized.T
        return {
            "learned_graph": None,
            "identity_type": "node_embedding_not_adjacency",
            "node_embedding_shape": list(embedding.shape),
            "node_embedding_hash": canonical_tensor_hash(embedding),
            "cosine_similarity_diagnostic_only": {
                "shape": list(cosine.shape),
                "min": float(cosine.min().cpu()),
                "max": float(cosine.max().cpu()),
                "mean": float(cosine.mean().cpu()),
            },
        }
    return None


def _diagnostic_hash(payload: dict[str, Any] | None) -> str | None:
    if payload is None:
        return None
    return str(
        payload.get("canonical_content_hash")
        or payload.get("node_embedding_hash")
    )


def _synthetic_signature(batch: BenchmarkBatch) -> dict[str, Any]:
    return {
        "dataset_id": "SYNTHETIC_E1_A_SMOKE",
        "shape": list(batch.x.shape),
        "feature_names": load_protocol()["ordered_input_features"],
        "feature_order_hash": load_protocol()["feature_order_hash"],
        "node_count": batch.node_count,
        "node_order_hash": "synthetic",
        "timestamp_count": int(batch.x.shape[1]),
        "normalization_fit_scope": "synthetic_explicit_no_fit",
    }


def _write_non_trainable_state(
    run_dir: Path, runtime: ModelRuntime, protocol
) -> None:
    assert runtime.scaler_context is not None
    e5 = runtime.effective_config.get("experiment_profile_id") is not None
    payload = {
            "model_id": runtime.model_id,
            "protocol_hash": protocol.protocol_hash,
            "feature_order_hash": protocol["feature_order_hash"],
            "baseline_rule": runtime.effective_config["baseline_rule"],
            "ma_window": runtime.effective_config.get("ma_window"),
            "optimizer": None,
            "scheduler": None,
            "epochs": None,
            "early_stopping": None,
            "scaler_context": runtime.scaler_context.to_dict(),
        }
    if e5:
        payload.update(
            {
                "training_mode": "EVALUATE_ONLY",
                "training_loss": "NOT_APPLICABLE",
                "trained_with_common_loss": False,
                "common_loss_evaluation_applied": True,
                "diagnostic_loss": "masked_score_aligned_hybrid",
                "best_checkpoint": None,
                "best_epoch": None,
                "comparison_role": "NON_TRAINABLE_REFERENCE",
            }
        )
    atomic_write_json(run_dir / "baseline_state.json", payload)


def _run_non_trainable(
    runtime: ModelRuntime,
    batches: Iterable[BenchmarkBatch],
    protocol,
    *,
    run_dir: Path,
    run_mode: str,
    data_signature: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_common(
        run_dir,
        runtime,
        protocol,
        profile="NON_TRAINABLE",
        run_mode=run_mode,
        data_signature=data_signature,
        formal_training=False,
    )
    _write_non_trainable_state(run_dir, runtime, protocol)
    write_status(
        run_dir,
        status="RUNNING",
        run_mode=run_mode,
        artifact_profile="NON_TRAINABLE",
        formal_training=False,
    )
    e5 = runtime.effective_config.get("experiment_profile_id") is not None
    metrics, prediction_shape, diagnostic_loss = _evaluate(
        runtime,
        batches,
        protocol,
        device=device,
        diagnostic_loss_fn=runtime_loss(runtime) if e5 else None,
    )
    if e5:
        atomic_write_json(
            run_dir / "common_loss_diagnostic.json",
            {
                "loss_id": "masked_score_aligned_hybrid",
                "loss_space": "normalized Patv_raw",
                "value": diagnostic_loss,
                "training_loss": "NOT_APPLICABLE",
                "trained_with_common_loss": False,
                "common_loss_evaluation_applied": True,
            },
        )
    _write_metrics(
        run_dir,
        metrics,
        runtime=runtime,
        protocol=protocol,
        prediction_shape=prediction_shape,
        source_checkpoint=None,
    )
    write_status(
        run_dir,
        status="COMPLETED",
        run_mode=run_mode,
        artifact_profile="NON_TRAINABLE",
        formal_training=False,
        exit_code=0,
    )
    validation = validate_run(
        run_dir, expected_protocol_hash=protocol.protocol_hash
    )
    result = {
        "status": "PASS",
        "model_id": runtime.model_id,
        "run_mode": run_mode,
        "formal_training": False,
        "artifact_profile": "NON_TRAINABLE",
        "run_dir": str(run_dir),
        "prediction_shape": prediction_shape,
        "parameter_count": runtime.parameter_count,
        "trainable_parameter_count": runtime.trainable_parameter_count,
        "metrics": metrics,
        "artifact_validation": validation,
    }
    if e5:
        result.update(
            {
                "diagnostic_loss": diagnostic_loss,
                "training_loss": "NOT_APPLICABLE",
                "trained_with_common_loss": False,
                "common_loss_evaluation_applied": True,
            }
        )
    return result


def _run_trainable_smoke(
    runtime: ModelRuntime,
    train_loader,
    val_loader,
    eval_loader,
    protocol,
    *,
    run_dir: Path,
    data_signature: dict[str, Any],
    device: str,
    initialization_parity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=False)
    runtime.effective_config.update(
        {
            "run_mode": "smoke",
            "artifact_profile": "SMOKE",
            "formal_training": False,
            "epochs": 1,
        }
    )
    resolved, effective = _write_common(
        run_dir,
        runtime,
        protocol,
        profile="SMOKE",
        run_mode="smoke",
        data_signature=data_signature,
        formal_training=False,
    )
    trainer = Trainer(
        model=runtime.model,
        adapter=runtime.adapter,
        loss_fn=runtime_loss(runtime),
        protocol=protocol,
        run_dir=run_dir,
        model_id=runtime.model_id,
        resolved_config=resolved,
        effective_config=effective,
        device=device,
        inverse_target=runtime.inverse_target,
    )
    history = trainer.fit(train_loader, val_loader)
    pre_reload_diagnostic = _model_identity_diagnostic(runtime)
    metrics = Evaluator(trainer).evaluate(eval_loader)
    post_reload_diagnostic = _model_identity_diagnostic(runtime)
    if post_reload_diagnostic is not None:
        post_reload_diagnostic["checkpoint_reload_hash_before"] = (
            _diagnostic_hash(pre_reload_diagnostic)
        )
        post_reload_diagnostic["checkpoint_reload_hash_after"] = (
            _diagnostic_hash(post_reload_diagnostic)
        )
        post_reload_diagnostic["checkpoint_reload_hash_matches"] = (
            _diagnostic_hash(pre_reload_diagnostic)
            == _diagnostic_hash(post_reload_diagnostic)
        )
        atomic_write_json(
            run_dir / "learned_graph_summary.json",
            post_reload_diagnostic,
        )
    first_batch = next(iter(eval_loader))
    prediction_shape = [
        first_batch.batch_size,
        first_batch.node_count,
        int(protocol["max_pred_len"]),
    ]
    _write_metrics(
        run_dir,
        metrics,
        runtime=runtime,
        protocol=protocol,
        prediction_shape=prediction_shape,
        source_checkpoint="best_checkpoint.pt",
    )
    write_status(
        run_dir,
        status="COMPLETED",
        run_mode="smoke",
        artifact_profile="SMOKE",
        formal_training=False,
        exit_code=0,
    )
    validation = validate_run(
        run_dir, expected_protocol_hash=protocol.protocol_hash
    )
    if initialization_parity is not None:
        atomic_write_json(
            run_dir / "initialization_parity.json", initialization_parity
        )
    return {
        "status": "PASS",
        "model_id": runtime.model_id,
        "run_mode": "smoke",
        "formal_training": False,
        "artifact_profile": "SMOKE",
        "run_dir": str(run_dir),
        "prediction_shape": prediction_shape,
        "parameter_count": runtime.parameter_count,
        "trainable_parameter_count": runtime.trainable_parameter_count,
        "loss": history[-1]["train_loss"],
        "validation_score_h10": history[-1]["val_score_h10"],
        "backward_completed": True,
        "optimizer_step_completed": True,
        "strict_reload_completed": True,
        "metrics": metrics,
        "artifact_validation": validation,
        "identity_diagnostic": post_reload_diagnostic,
        "initialization_parity": initialization_parity,
    }


def _default_smoke_root(model_id: str) -> Path:
    if model_id in E3_C_MODELS:
        return E3_C_SMOKE_ROOT
    if model_id in E3_B_MODELS:
        return E3_B_SMOKE_ROOT
    if model_id in E2_D_MODELS:
        return E2_D_SMOKE_ROOT
    if model_id in E2_C_MODELS:
        return E2_C_SMOKE_ROOT
    if model_id in E2_B_MODELS:
        return E2_B_SMOKE_ROOT
    if model_id in E2_A_MODELS:
        return E2_A_SMOKE_ROOT
    return E1_B_SMOKE_ROOT if model_id in E1_B_MODELS else E1_A_SMOKE_ROOT


def model_smoke(
    model_id: str,
    *,
    root: Path | None = None,
    experiment_profile: str | None = None,
    training_profile: str | None = None,
) -> dict[str, Any]:
    protocol = load_protocol()
    selected_profile = normalize_profile(experiment_profile)
    seed_everything(int(protocol["default_seed"]))
    graph_model = model_id in NATIVE_NODE_MODELS
    profile = load_training_profile(training_profile)
    batch = _synthetic_batch(
        batch_size=profile.train_batch_size if profile is not None else 2,
        time_steps=144,
        nodes=134 if graph_model else 4,
        canonical_graph_nodes=graph_model,
    )
    input_scaler, target_scaler = _synthetic_scalers()
    runtime = build_model_runtime(
        model_id,
        protocol,
        run_mode="smoke",
        input_scaler=input_scaler,
        target_scaler=target_scaler,
    )
    initialization_parity = None
    if (
        (selected_profile == E5_PROFILE_ID or profile is not None)
        and model_id not in {"persistence", "moving_average"}
    ):
        import torch

        base_runtime = runtime
        base_state_hash = state_dict_hash(base_runtime.model)
        base_parameter_count = base_runtime.parameter_count
        base_trainable_count = base_runtime.trainable_parameter_count
        base_optimizer = torch.optim.Adam(
            base_runtime.model.parameters(),
            lr=float(base_runtime.effective_config.get("learning_rate", 1e-3)),
            weight_decay=float(base_runtime.effective_config.get("weight_decay", 0.0)),
        )
        with torch.no_grad():
            base_prediction = base_runtime.adapter(
                base_runtime.model,
                batch,
                expected_horizon=int(protocol["max_pred_len"]),
                expected_features=int(protocol["feature_count"]),
            ).prediction.detach().clone()
        base_optimizer_signature = optimizer_group_signature(base_optimizer)
        seed_everything(int(protocol["default_seed"]))
        runtime = build_model_runtime(
            model_id,
            protocol,
            run_mode="smoke",
            input_scaler=input_scaler,
            target_scaler=target_scaler,
        )
        apply_experiment_profile(runtime, selected_profile)
        apply_training_profile(runtime, training_profile)
        e5_state_hash = state_dict_hash(runtime.model)
        with torch.no_grad():
            e5_prediction = runtime.adapter(
                runtime.model,
                batch,
                expected_horizon=int(protocol["max_pred_len"]),
                expected_features=int(protocol["feature_count"]),
            ).prediction.detach()
        e5_optimizer = torch.optim.Adam(
            runtime.model.parameters(),
            lr=float(runtime.effective_config.get("learning_rate", 1e-3)),
            weight_decay=float(runtime.effective_config.get("weight_decay", 0.0)),
        )
        initialization_parity = {
            "status": "PASS",
            "state_dict_keys_equal": list(base_runtime.model.state_dict())
            == list(runtime.model.state_dict()),
            "base_state_dict_hash": base_state_hash,
            "e5_state_dict_hash": e5_state_hash,
            "parameter_count_equal": base_parameter_count == runtime.parameter_count,
            "trainable_parameter_count_equal": (
                base_trainable_count == runtime.trainable_parameter_count
            ),
            "forward_equal": bool(
                torch.allclose(base_prediction, e5_prediction, atol=1e-7, rtol=1e-6)
            ),
            "optimizer_param_groups_equal": (
                base_optimizer_signature
                == optimizer_group_signature(e5_optimizer)
            ),
        }
        if not all(
            value
            for key, value in initialization_parity.items()
            if key.endswith("_equal")
        ) or base_state_hash != initialization_parity["e5_state_dict_hash"]:
            initialization_parity["status"] = "FAIL"
            raise RuntimeError(
                f"E5 model initialization parity failed for {model_id}: "
                f"{initialization_parity}"
            )
    else:
        apply_experiment_profile(runtime, selected_profile)
        apply_training_profile(runtime, training_profile)
    data_signature = _synthetic_signature(batch)
    if graph_model:
        data_signature.update(
            {
                key: runtime.effective_config[key]
                for key in (
                    "graph_id",
                    "graph_protocol_hash",
                    "node_order_hash",
                    "graph_bundle_hash",
                    "location_source_hash",
                )
            }
        )
    root = root or (
        PROJECT_ROOT / SMOKE_OUTPUT_ROOT_RELATIVE
        if selected_profile == E5_PROFILE_ID
        else _default_smoke_root(model_id)
    )
    if (
        model_id in E2_C_MODELS
        or model_id in E2_D_MODELS
        or model_id in E3_B_MODELS
        or model_id in E3_C_MODELS
    ):
        root = root / "ordinary"
    run_dir = safe_run_dir(root, model_id)
    if model_id in {"persistence", "moving_average"}:
        return _run_non_trainable(
            runtime,
            [batch],
            protocol,
            run_dir=run_dir,
            run_mode="smoke",
            data_signature=data_signature,
            device="cpu",
        )
    return _run_trainable_smoke(
        runtime,
        [batch],
        [batch],
        [batch],
        protocol,
        run_dir=run_dir,
        data_signature=data_signature,
        device="cuda" if environment_snapshot()["cuda_available"] else "cpu",
        initialization_parity=initialization_parity,
    )


def full_shape_model_smoke(
    model_id: str,
    *,
    root: Path | None = None,
    experiment_profile: str | None = None,
    training_profile: str | None = None,
) -> dict[str, Any]:
    import torch

    protocol = load_protocol()
    selected_profile = normalize_profile(experiment_profile)
    root = root or (
        PROJECT_ROOT / SMOKE_OUTPUT_ROOT_RELATIVE
        if selected_profile == E5_PROFILE_ID
        else _default_smoke_root(model_id)
    )
    seed_everything(int(protocol["default_seed"]))
    profile = load_training_profile(training_profile)
    requested = {
        "B": profile.train_batch_size if profile is not None else 32,
        "T": 144,
        "N": 134,
        "C": 16,
        "H": 10,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = {
        "model_id": model_id,
        "run_mode": "smoke",
        "formal_registry_entry": True,
        "formal_training": False,
        "requested_shape": requested,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "amp_enabled": bool(protocol["amp_enabled"]),
    }
    if selected_profile == E5_PROFILE_ID:
        result["experiment_profile"] = selected_profile
    result.update(batch_identity(training_profile))
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        batch = _synthetic_batch(
            batch_size=requested["B"],
            time_steps=requested["T"],
            nodes=requested["N"],
            canonical_graph_nodes=model_id in NATIVE_NODE_MODELS,
        )
        input_scaler, target_scaler = _synthetic_scalers()
        runtime = build_model_runtime(
            model_id,
            protocol,
            run_mode="smoke",
            input_scaler=input_scaler,
            target_scaler=target_scaler,
        )
        apply_experiment_profile(runtime, selected_profile)
        apply_training_profile(runtime, training_profile)
        if selected_profile == E5_PROFILE_ID:
            result["loss_profile_hash"] = runtime.effective_config["loss"][
                "profile_hash"
            ]
        moved = _move_batch(batch, device)
        runtime.model.to(device)
        backward_completed = False
        forward_completed = False
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=bool(protocol["amp_enabled"]) and device.type == "cuda",
        ):
            output = runtime.adapter(
                runtime.model,
                moved,
                expected_horizon=10,
                expected_features=16,
            )
            forward_completed = True
            if runtime.trainable_parameter_count > 0:
                loss = runtime_loss(runtime)(
                    output.prediction, moved.target, moved.mask
                )
            else:
                loss = None
        if runtime.trainable_parameter_count > 0:
            if loss is None:
                raise RuntimeError("Full-shape GRU smoke has no valid targets.")
            loss.backward()
            backward_completed = True
            parity = None
        else:
            physical_prediction = runtime.inverse_target(output.prediction)
            power_index = runtime.scaler_context.power_feature_index
            physical_history = runtime.scaler_context.input_to_physical(
                moved.x[..., power_index]
            )
            expected = (
                physical_history[:, -1, :]
                if model_id == "persistence"
                else physical_history[:, -144:, :].mean(dim=1)
            ).unsqueeze(-1).expand(-1, -1, 10)
            parity = bool(
                torch.allclose(
                    physical_prediction.float(),
                    expected.float(),
                    atol=1e-3,
                    rtol=1e-5,
                )
            )
            if not parity:
                raise RuntimeError("Full-shape inverse-transform parity failed.")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        result.update(
            {
                "status": "PASS",
                "output_shape": list(output.prediction.shape),
                "loss": None if loss is None else float(loss.detach().cpu()),
                "elapsed_seconds": time.perf_counter() - start,
                "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated())
                if torch.cuda.is_available()
                else None,
                "peak_allocated_memory": int(torch.cuda.max_memory_allocated())
                if torch.cuda.is_available()
                else None,
                "peak_reserved_memory": int(torch.cuda.max_memory_reserved())
                if torch.cuda.is_available()
                else None,
                "gpu_name": torch.cuda.get_device_name()
                if torch.cuda.is_available()
                else None,
                "gpu_total_memory": int(
                    torch.cuda.get_device_properties(device).total_memory
                )
                if torch.cuda.is_available()
                else None,
                "parameter_count": runtime.parameter_count,
                "trainable_parameter_count": runtime.trainable_parameter_count,
                "backward_completed": backward_completed,
                "forward_completed": forward_completed,
                "inverse_transform_parity": parity,
                "identity_diagnostic": _model_identity_diagnostic(runtime),
                **(
                    dict(runtime.model.graph_identity)
                    if model_id in NATIVE_NODE_MODELS
                    else {}
                ),
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": (
                    "FAIL_OOM"
                    if "out of memory" in str(exc).casefold()
                    else "FAIL_NON_OOM"
                ),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback_tail": traceback.format_exc().splitlines()[-20:],
                "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated())
                if torch.cuda.is_available()
                else None,
                "peak_allocated_memory": int(torch.cuda.max_memory_allocated())
                if torch.cuda.is_available()
                else None,
                "peak_reserved_memory": int(torch.cuda.max_memory_reserved())
                if torch.cuda.is_available()
                else None,
                "gpu_name": torch.cuda.get_device_name()
                if torch.cuda.is_available()
                else None,
                "gpu_total_memory": int(
                    torch.cuda.get_device_properties(device).total_memory
                )
                if torch.cuda.is_available()
                else None,
                "parameter_count": runtime.parameter_count
                if "runtime" in locals()
                else None,
                "trainable_parameter_count": runtime.trainable_parameter_count
                if "runtime" in locals()
                else None,
                "backward_completed": False,
                "forward_completed": (
                    forward_completed if "forward_completed" in locals() else False
                ),
                **(
                    dict(runtime.model.graph_identity)
                    if model_id in NATIVE_NODE_MODELS and "runtime" in locals()
                    else {}
                ),
            }
        )
    path = root / "full_shape" / f"{model_id}.json"
    atomic_write_json(path, result)
    return result


def real_data_model_smoke(
    model_id: str,
    *,
    input_path: str | Path = DEFAULT_INPUT_PATH,
    target_path: str | Path = DEFAULT_TARGET_PATH,
    root: Path | None = None,
    attempt_tag: str = "",
    experiment_profile: str | None = None,
    training_profile: str | None = None,
) -> dict[str, Any]:
    protocol = load_protocol()
    selected_profile = normalize_profile(experiment_profile)
    root = root or (
        PROJECT_ROOT / SMOKE_OUTPUT_ROOT_RELATIVE
        if selected_profile == E5_PROFILE_ID
        else _default_smoke_root(model_id)
    )
    seed_everything(int(protocol["default_seed"]))
    provider = SDWPFDataProvider.from_files(
        input_path, target_path, protocol=protocol
    )
    runtime = build_model_runtime(
        model_id,
        protocol,
        run_mode="smoke",
        input_scaler=provider.scalers["input"],
        target_scaler=provider.scalers["target"],
    )
    apply_experiment_profile(runtime, selected_profile)
    apply_training_profile(runtime, training_profile)
    run_name = model_id if not attempt_tag else f"{model_id}_{attempt_tag}"
    run_dir = safe_run_dir(root, "real_data", run_name)
    if model_id in {"persistence", "moving_average"}:
        return _run_non_trainable(
            runtime,
            ProviderBatchIterable(provider, "test", 4, limit_batches=1),
            protocol,
            run_dir=run_dir,
            run_mode="smoke",
            data_signature=provider.signature(),
            device="cpu",
        )
    device = "cuda" if environment_snapshot()["cuda_available"] else "cpu"
    sizes = resolved_batch_sizes(protocol, training_profile)
    train_batch_size = sizes["train"] if training_profile else 2
    train_batch_limit = 2
    if model_id in {"msgnet", "timefilter"}:
        import torch

        # Limited real-data smoke is a semantic/data-path check, not the exact
        # hardware gate.  These two exact CUDA shapes are routed separately by
        # hardware preflight, so keep this bounded check executable on low VRAM.
        device = "cpu"
        train_batch_size = sizes["train"] if training_profile else 1
        train_batch_limit = 2 if training_profile else 1
        torch.set_num_threads(1)
        runtime.effective_config["limited_real_smoke_cpu_threads"] = 1
    if model_id in NATIVE_NODE_MODELS:
        train_batch_size = sizes["train"] if training_profile else 2
        train_batch_limit = 2
    if training_profile:
        val_batch_size = sizes["val"]
        test_batch_size = sizes["test"]
    elif (
        selected_profile == E5_PROFILE_ID
        and model_id not in {"msgnet", "timefilter"}
    ):
        # The exact B=32 hardware contract is exercised separately above.
        # This bounded real-data path preserves model/config identity while
        # limiting only the number and size of diagnostic smoke batches.
        train_batch_size = 2
        train_batch_limit = 2
        val_batch_size = 1
        test_batch_size = 1
    else:
        val_batch_size = 1
        test_batch_size = 1
    return _run_trainable_smoke(
        runtime,
        ProviderBatchIterable(
            provider, "train", train_batch_size, limit_batches=train_batch_limit
        ),
        ProviderBatchIterable(provider, "val", val_batch_size, limit_batches=1),
        ProviderBatchIterable(provider, "test", test_batch_size, limit_batches=1),
        protocol,
        run_dir=run_dir,
        data_signature=provider.signature(),
        device=device,
    )


def formal_evaluate_only(
    model_id: str,
    *,
    input_path: str | Path,
    target_path: str | Path,
    output_root: str | Path,
    run_id: str,
    device: str,
    experiment_profile: str | None = None,
    training_profile: str | None = None,
    formal_scope_id: str | None = None,
) -> dict[str, Any]:
    entry = load_registry().get(model_id)
    if model_id not in {"persistence", "moving_average"}:
        raise ModelUnavailableError(
            f"{entry.display_name} is not a deterministic evaluate-only E1-A baseline. "
            "No formal run was started."
        )
    protocol = load_protocol()
    provider = SDWPFDataProvider.from_files(
        input_path, target_path, protocol=protocol
    )
    runtime = build_model_runtime(
        model_id,
        protocol,
        run_mode="formal",
        input_scaler=provider.scalers["input"],
        target_scaler=provider.scalers["target"],
    )
    selected_profile = normalize_profile(experiment_profile)
    validate_scope27_request(
        model_id=model_id,
        formal_scope_id=formal_scope_id,
        experiment_profile=selected_profile,
        training_profile=training_profile,
        trainable=False,
    )
    apply_experiment_profile(
        runtime,
        selected_profile,
        run_id=run_id,
        output_root=output_root,
        provenance=(
            {"active_scope_id": E5_SCOPE27_ID}
            if formal_scope_id == E5_SCOPE27_ID
            else None
        ),
    )
    apply_training_profile(runtime, training_profile)
    sizes = resolved_batch_sizes(protocol, training_profile)
    run_dir = safe_run_dir(output_root, run_id)
    return _run_non_trainable(
        runtime,
        ProviderBatchIterable(
            provider, "test", sizes["test"]
        ),
        protocol,
        run_dir=run_dir,
        run_mode="formal",
        data_signature=provider.signature(),
        device=device,
    )


def formal_train(
    model_id: str,
    *,
    input_path: str | Path,
    target_path: str | Path,
    output_root: str | Path,
    run_id: str,
    device: str,
    preflight_root: str | Path | None = None,
    experiment_profile: str | None = None,
    training_profile: str | None = None,
    formal_scope_id: str | None = None,
) -> dict[str, Any]:
    entry = load_registry().get(model_id)
    if model_id in {"persistence", "moving_average"}:
        raise ModelUnavailableError(
            f"{entry.display_name} is a deterministic non-trainable baseline. "
            "Use evaluate-only for a formal run. No training was started."
        )
    if not entry.supports_train:
        raise ModelUnavailableError(
            f"{entry.display_name} is unavailable for training; "
            f"planned_stage={entry.planned_stage}. No formal run was started."
        )
    selected_profile = normalize_profile(experiment_profile)
    validate_scope27_request(
        model_id=model_id,
        formal_scope_id=formal_scope_id,
        experiment_profile=selected_profile,
        training_profile=training_profile,
        trainable=True,
    )
    scope27_request = is_scope27_train_request(
        model_id=model_id,
        formal_scope_id=formal_scope_id,
        experiment_profile=selected_profile,
        training_profile=training_profile,
    )
    if not scope27_request and (
        training_profile is not None
        or selected_profile == E5_PROFILE_ID
        or bool(entry.values.get("formal_hardware_preflight_required", False))
    ):
        from .hardware_preflight import read_matching_pass

        if (
            read_matching_pass(
                model_id,
                root=preflight_root,
                experiment_profile=selected_profile,
                training_profile=training_profile,
            )
            is None
        ):
            raise ModelUnavailableError(
                f"{entry.display_name} requires an exact matching PASS hardware "
                "preflight before formal training. No formal run was started."
            )
    protocol = load_protocol()
    seed_everything(int(protocol["default_seed"]))
    provider = SDWPFDataProvider.from_files(
        input_path, target_path, protocol=protocol
    )
    runtime = build_model_runtime(
        model_id,
        protocol,
        run_mode="formal",
        target_scaler=provider.scalers["target"],
    )
    from .hardware_preflight import preflight_identity

    apply_experiment_profile(
        runtime,
        selected_profile,
        run_id=run_id,
        output_root=output_root,
        preflight_identity=(
            {
                "status": "NOT_REQUIRED_FOR_CURRENT_FORMAL_SCOPE",
                "policy_id": PREFLIGHT_POLICY_ID,
                "active_scope_id": E5_SCOPE27_ID,
            }
            if scope27_request
            else preflight_identity(
                model_id,
                experiment_profile=selected_profile,
                training_profile=training_profile,
            )
        ),
        provenance=(
            {"active_scope_id": E5_SCOPE27_ID}
            if scope27_request
            else None
        ),
    )
    apply_training_profile(runtime, training_profile)
    sizes = resolved_batch_sizes(protocol, training_profile)
    run_dir = safe_run_dir(output_root, run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    runtime.effective_config.update(
        {
            "run_mode": "formal",
            "artifact_profile": "TRAIN",
            "formal_training": True,
            "formal_worker_pid": os.getpid(),
        }
    )
    resolved, effective = _write_common(
        run_dir,
        runtime,
        protocol,
        profile="TRAIN",
        run_mode="formal",
        data_signature=provider.signature(),
        formal_training=True,
    )
    trainer = Trainer(
        model=runtime.model,
        adapter=runtime.adapter,
        loss_fn=runtime_loss(runtime),
        protocol=protocol,
        run_dir=run_dir,
        model_id=model_id,
        resolved_config=resolved,
        effective_config=effective,
        device=device,
        inverse_target=runtime.inverse_target,
    )
    history = trainer.fit(
        ProviderBatchIterable(
            provider, "train", sizes["train"]
        ),
        ProviderBatchIterable(provider, "val", sizes["val"]),
    )
    metrics = Evaluator(trainer).evaluate(
        ProviderBatchIterable(provider, "test", sizes["test"])
    )
    _write_metrics(
        run_dir,
        metrics,
        runtime=runtime,
        protocol=protocol,
        prediction_shape=[
            len(provider.starts["test"]),
            len(provider.node_ids),
            int(protocol["max_pred_len"]),
        ],
        source_checkpoint="best_checkpoint.pt",
    )
    write_status(
        run_dir,
        status="COMPLETED",
        run_mode="formal",
        artifact_profile="TRAIN",
        formal_training=True,
        exit_code=0,
    )
    return {
        "status": "PASS",
        "model_id": model_id,
        "run_mode": "formal",
        "formal_training": True,
        "run_dir": str(run_dir),
        "epochs_completed": len(history),
        "metrics": metrics,
        "artifact_validation": validate_run(
            run_dir, expected_protocol_hash=protocol.protocol_hash
        ),
    }
