from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .artifacts import atomic_write_json
from .checkpointing import CheckpointManager
from .contracts import BenchmarkBatch
from .data import SDWPFDataProvider
from .errors import ContractError
from .losses import get_loss
from .model_runtime import build_model_runtime
from .nonfinite import (
    NonfiniteDiagnostics,
    rng_state_summary,
    tensor_summary,
)
from .protocol import load_protocol
from .runtime import ProviderBatchIterable
from .training_profiles import apply_training_profile, resolved_batch_sizes
from .transformer_diagnostics import TransformerHookCapture


def _capture_rng_state():
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
        "cuda": [item.clone() for item in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else None,
    }


def _restore_rng_state(state) -> None:
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _prediction_from(value: Any):
    return value.prediction if hasattr(value, "prediction") else value


def compare_amp_fp32(
    model,
    batch: BenchmarkBatch,
    forward: Callable[[], Any],
    *,
    device,
    amp_enabled: bool = True,
    hook_factory: Callable[[Any], TransformerHookCapture] | None = None,
) -> dict[str, Any]:
    """Compare one batch without backward or optimizer.step.

    The same callable and batch object are used twice.  The complete RNG state
    is restored immediately before each forward so train-mode dropout has the
    same random condition; only autocast is changed.
    """

    import torch

    was_training = bool(model.training)
    model.train(was_training)
    state = _capture_rng_state()

    def run(enabled: bool) -> tuple[dict[str, Any], dict[str, Any] | None]:
        _restore_rng_state(state)
        events: list[dict[str, Any]] = []
        capture = hook_factory(events.append) if hook_factory is not None else None
        if capture is None:
            context = None
        else:
            context = capture
        try:
            if context is None:
                with torch.no_grad():
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.float16,
                        enabled=bool(enabled and device.type == "cuda"),
                    ):
                        prediction = _prediction_from(forward())
            else:
                with context:
                    with torch.no_grad():
                        with torch.autocast(
                            device_type=device.type,
                            dtype=torch.float16,
                            enabled=bool(enabled and device.type == "cuda"),
                        ):
                            prediction = _prediction_from(forward())
            summary = tensor_summary(prediction)
            return summary, events[0] if events else None
        finally:
            _restore_rng_state(state)

    amp_summary, amp_hook = run(bool(amp_enabled))
    fp32_summary, fp32_hook = run(False)
    model.train(was_training)
    amp_bad = any(
        amp_summary.get(key, 0)
        for key in ("nan_count", "posinf_count", "neginf_count")
    )
    fp32_bad = any(
        fp32_summary.get(key, 0)
        for key in ("nan_count", "posinf_count", "neginf_count")
    )
    if amp_bad and not fp32_bad:
        classification = "AMP_ONLY_FORWARD_OVERFLOW"
    elif amp_bad and fp32_bad:
        classification = "FORWARD_NONFINITE_IN_BOTH_PRECISIONS"
    elif not amp_bad and not fp32_bad:
        classification = "FAILURE_NOT_REPRODUCED"
    else:
        classification = "UNRESOLVED"
    return {
        "classification": classification,
        "same_batch_callable": True,
        "same_rng_state": True,
        "model_training": was_training,
        "backward_executed": False,
        "optimizer_step_executed": False,
        "amp_prediction": amp_summary,
        "fp32_prediction": fp32_summary,
        "amp_first_nonfinite_module": amp_hook,
        "fp32_first_nonfinite_module": fp32_hook,
        "rng_state_after": rng_state_summary(),
    }


def _move_batch(batch: BenchmarkBatch, device) -> BenchmarkBatch:
    move = lambda value: value.to(device) if hasattr(value, "to") else value
    return BenchmarkBatch(
        x=move(batch.x),
        target=move(batch.target),
        mask=move(batch.mask),
        target_raw_or_inverse_transform=(
            move(batch.target_raw_or_inverse_transform)
            if batch.target_raw_or_inverse_transform is not None
            else None
        ),
        sample_ids=batch.sample_ids,
        window_end_indices=batch.window_end_indices,
        node_ids=batch.node_ids,
        x_mark=move(batch.x_mark) if batch.x_mark is not None else None,
        y_mark=move(batch.y_mark) if batch.y_mark is not None else None,
        split=batch.split,
        metadata=batch.metadata,
    )


def _forward_without_validation(runtime, batch: BenchmarkBatch):
    inputs = runtime.adapter.prepare_model_inputs(
        batch, expected_horizon=10, expected_features=16
    )
    raw = runtime.adapter.forward_model(
        runtime.model,
        inputs,
        expected_horizon=10,
        expected_features=16,
    )
    return runtime.adapter.normalize_output(
        raw, batch, expected_horizon=10, expected_features=16
    )


def run_transformer_nonfinite_diagnostic(
    *,
    model_id: str,
    run_dir: str | Path,
    checkpoint: str | Path,
    output_root: str | Path,
    input_path: str | Path,
    target_path: str | Path,
    device: str,
    training_profile: str | None = None,
    exact_update_replay: bool = False,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Replay one Transformer run and write only bounded diagnostic metadata.

    ``exact_update_replay`` is an explicit forensic mode used when the source
    checkpoint predates diagnostics and the failure occurs after later updates.
    The final AMP/FP32 comparison itself always runs through
    :func:`compare_amp_fp32`, which performs no backward or optimizer step.
    """

    if model_id != "transformer":
        raise ValueError("nonfinite diagnostic currently supports model_id=transformer only")
    run_dir = Path(run_dir).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "transformer_nonfinite_diagnostic.json"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic output: {output_path}")

    protocol = load_protocol()
    provider = SDWPFDataProvider.from_files(input_path, target_path, protocol=protocol)
    resolved = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    effective = json.loads((run_dir / "effective_config.json").read_text(encoding="utf-8"))
    runtime = build_model_runtime(
        model_id,
        protocol,
        run_mode="formal",
        target_scaler=provider.scalers["target"],
    )
    apply_training_profile(runtime, training_profile)
    import torch

    replay_amp_enabled = bool(effective.get("amp_enabled", False))
    # A forensic replay must preserve the historical precision identity even
    # when the current source contains a model-specific fix.
    runtime.effective_config["amp_enabled"] = replay_amp_enabled

    target_device = torch.device(device)
    runtime.model.to(target_device)
    optimizer = torch.optim.Adam(
        runtime.model.parameters(),
        lr=float(effective.get("learning_rate", 1e-3)),
        weight_decay=float(effective.get("weight_decay", 0.0)),
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=bool(effective.get("amp_enabled", False)) and target_device.type == "cuda"
    )
    manager = CheckpointManager(
        run_dir,
        protocol_hash=protocol.protocol_hash,
        model_id=model_id,
        resolved_config=resolved,
        effective_config=effective,
    )
    payload = manager.load(checkpoint, runtime.model, optimizer, amp_scaler=scaler)
    seed_state = payload.get("seed_state")
    if not isinstance(seed_state, dict):
        raise ContractError("checkpoint has no complete seed_state for deterministic replay")
    _restore_rng_state(seed_state)
    runtime.model.train()
    loss_fn = get_loss("masked_mse")
    sizes = resolved_batch_sizes(protocol, training_profile)
    starts = list(provider.starts["train"])
    steps = int(payload.get("global_step", 0))
    skipped = 0
    failure: dict[str, Any] | None = None
    comparison: dict[str, Any] | None = None
    start_epoch = int(payload.get("epoch", 0)) + 1
    epoch = start_epoch
    for batch_index, offset in enumerate(range(0, len(starts), sizes["train"])):
        if max_batches is not None and batch_index >= max_batches:
            break
        provider.starts["train"] = starts[offset : offset + sizes["train"]]
        source = provider.batches("train", sizes["train"])[0]
        batch = _move_batch(source, target_device)
        if exact_update_replay:
            optimizer.zero_grad(set_to_none=True)
            try:
                with torch.autocast(
                    device_type=target_device.type,
                    dtype=torch.float16,
                    enabled=replay_amp_enabled,
                ):
                    output = runtime.adapter(
                        runtime.model,
                        batch,
                        expected_horizon=10,
                        expected_features=16,
                    )
                    loss = loss_fn(output.prediction, batch.target, batch.mask)
            except Exception as exc:
                failure = {
                    "epoch": epoch,
                    "batch_index": batch_index,
                    "global_step": steps,
                    "window_end_indices": list(source.window_end_indices),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
                break
            if loss is None:
                skipped += 1
                continue
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            steps += 1
        else:
            try:
                with torch.no_grad():
                    output = runtime.adapter(
                        runtime.model,
                        batch,
                        expected_horizon=10,
                        expected_features=16,
                    )
            except Exception as exc:
                failure = {
                    "epoch": epoch,
                    "batch_index": batch_index,
                    "global_step": steps,
                    "window_end_indices": list(source.window_end_indices),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
                break
            if not torch.isfinite(output.prediction).all():
                failure = {
                    "epoch": epoch,
                    "batch_index": batch_index,
                    "global_step": steps,
                    "window_end_indices": list(source.window_end_indices),
                    "error_type": "ContractError",
                    "error_message": "prediction contains NaN/Inf",
                }
                break
    provider.starts["train"] = starts

    if failure is not None:
        failure_offset = int(failure["batch_index"]) * sizes["train"]
        provider.starts["train"] = starts[
            failure_offset : failure_offset + sizes["train"]
        ]
        failure_source = provider.batches("train", sizes["train"])[0]
        failure_batch = _move_batch(failure_source, target_device)
        diagnostic = NonfiniteDiagnostics(
            {
                "model_id": model_id,
                "run_id": run_dir.name,
                "run_mode": "formal",
                "artifact_profile": "DIAGNOSTIC",
                "phase": "replay",
                "epoch": failure["epoch"],
                "batch_index": failure["batch_index"],
                "global_step": failure["global_step"],
                "resolved_config_hash": manager.resolved_hash,
                "effective_config_hash": manager.effective_hash,
                "checkpoint": {"path": str(Path(checkpoint).resolve())},
            }
        )
        diagnostic.record_batch(failure_batch)
        try:
            model_inputs = runtime.adapter.prepare_model_inputs(
                failure_batch, expected_horizon=10, expected_features=16
            )
            diagnostic.record_model_inputs(model_inputs)
            with TransformerHookCapture(runtime.model, diagnostic.record_module):
                comparison = compare_amp_fp32(
                    runtime.model,
                    failure_batch,
                    lambda: _forward_without_validation(runtime, failure_batch),
                    device=target_device,
                    amp_enabled=True,
                )
            if comparison and comparison.get("amp_prediction"):
                diagnostic.record_prediction_summary(
                    comparison["amp_prediction"], autocast_enabled=True
                )
        except Exception as exc:
            comparison = {
                "classification": "UNRESOLVED",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "backward_executed": False,
                "optimizer_step_executed": False,
            }
        diagnostic.classification = comparison.get("classification") if comparison else "UNRESOLVED"
        diagnostic.record_exception(ContractError(failure["error_message"]))

    diagnostic_payload = None
    if failure is not None:
        diagnostic_payload = diagnostic.payload()
    result = {
        "schema_version": "benchmark_v2_transformer_nonfinite_replay_v1",
        "model_id": model_id,
        "run_dir": str(run_dir),
        "checkpoint": str(Path(checkpoint).resolve()),
        "training_profile": training_profile,
        "exact_update_replay": bool(exact_update_replay),
        "backward_executed_during_comparison": False,
        "optimizer_step_executed_during_comparison": False,
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_global_step": payload.get("global_step"),
        "failure": failure,
        "comparison": comparison,
        "diagnostic": diagnostic_payload,
        "skipped_all_masked_batches": skipped,
        "rng_state_summary": rng_state_summary(),
        "status": "PASS" if comparison and comparison.get("classification") else "UNRESOLVED",
    }
    atomic_write_json(output_path, result)
    return {**result, "output_path": str(output_path)}
