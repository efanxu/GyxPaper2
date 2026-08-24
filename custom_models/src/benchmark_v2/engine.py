from __future__ import annotations

import traceback
import time
from pathlib import Path
from typing import Any, Callable

from .adapters import adapter_forward
from .artifacts import atomic_write_csv, write_status
from .checkpointing import CheckpointManager
from .contracts import BenchmarkBatch, BenchmarkAdapter, call_loss
from .errors import ContractError
from .metrics import evaluate_horizons
from .nonfinite import (
    NonfiniteDiagnostics,
    diagnostics_enabled,
    rng_state_summary,
    summarize_parameter_state,
    write_nonfinite_artifact,
)
from .transformer_diagnostics import TransformerHookCapture
from .seeds import seed_everything


def _move_batch(batch: BenchmarkBatch, device) -> BenchmarkBatch:
    move = lambda x: x.to(device) if hasattr(x, "to") else x
    return BenchmarkBatch(x=move(batch.x), target=move(batch.target), mask=move(batch.mask), target_raw_or_inverse_transform=move(batch.target_raw_or_inverse_transform) if batch.target_raw_or_inverse_transform is not None else None, sample_ids=batch.sample_ids, window_end_indices=batch.window_end_indices, node_ids=batch.node_ids, x_mark=move(batch.x_mark) if batch.x_mark is not None else None, y_mark=move(batch.y_mark) if batch.y_mark is not None else None, split=batch.split, metadata=batch.metadata)


class Trainer:
    """Training boundary intentionally accepts only train_loader and val_loader."""

    def __init__(self, *, model, adapter: BenchmarkAdapter, loss_fn: Callable, protocol: Any, run_dir: str | Path, model_id: str, resolved_config: dict[str, Any], effective_config: dict[str, Any], device: str = "cpu", inverse_target: Callable[[Any], Any] | None = None):
        import torch
        self.model, self.adapter, self.loss_fn = model, adapter, loss_fn
        self.protocol = protocol
        self.run_dir = Path(run_dir)
        self.device = torch.device(device)
        self.model.to(self.device)
        if hasattr(self.loss_fn, "to"):
            self.loss_fn.to(self.device)
        self.inverse_target = inverse_target or (lambda x: x)
        self.resolved_config, self.effective_config = resolved_config, effective_config
        optimizer_name = str(effective_config.get("optimizer", "Adam"))
        if optimizer_name != "Adam":
            raise ValueError(f"benchmark_v2 Trainer currently supports optimizer=Adam, got {optimizer_name}")
        optimizer_parameters = list(self.model.parameters())
        if hasattr(self.loss_fn, "parameters"):
            optimizer_parameters.extend(
                parameter for parameter in self.loss_fn.parameters() if parameter.requires_grad
            )
        self.optimizer = torch.optim.Adam(
            optimizer_parameters,
            lr=float(effective_config.get("learning_rate", 1e-3)),
            weight_decay=float(effective_config.get("weight_decay", 0.0)),
        )
        self.amp_enabled = bool(effective_config.get("amp_enabled", False)) and self.device.type == "cuda"
        self.amp_scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.ckpt = CheckpointManager(self.run_dir, protocol_id=str(protocol.get("protocol_id", protocol.get("protocol_version", "benchmark_v2"))), model_id=model_id, resolved_config=resolved_config, effective_config=effective_config)
        self.model_id = model_id
        self.nonfinite_diagnostics_enabled = diagnostics_enabled()
        self._last_checkpoint: dict[str, Any] | None = None
        self._last_optimizer_update: dict[str, Any] | None = None

    def _diagnostic_context(
        self,
        *,
        phase: str,
        epoch: int,
        batch_index: int,
        global_step: int,
        batch,
        loader,
    ) -> dict[str, Any]:
        import torch

        checkpoint = dict(self._last_checkpoint or {})
        return {
            "model_id": self.model_id,
            "run_id": self.run_dir.name,
            "run_mode": self.effective_config.get("run_mode"),
            "artifact_profile": self.effective_config.get("artifact_profile"),
            "protocol_id": self.protocol.get("protocol_id", self.protocol.get("protocol_version")),
            "batch_profile_id": self.effective_config.get("training_batch_profile_id"),
            "phase": phase,
            "epoch": int(epoch),
            "epoch_is_zero_based": False,
            "batch_index": int(batch_index),
            "global_step": int(global_step),
            "optimizer_step_count": int(global_step),
            "gradient_accumulation_step": 1,
            "dataloader": {
                "class": type(loader).__name__,
                "split": getattr(loader, "split", None),
                "batch_size": getattr(loader, "batch_size", None),
                "shuffle": getattr(loader, "shuffle", None),
                "sampler_class": type(getattr(loader, "sampler", None)).__name__
                if getattr(loader, "sampler", None) is not None
                else None,
            },
            "checkpoint": checkpoint,
            "precision": {
                "amp_enabled": bool(self.amp_enabled),
                "autocast_enabled": bool(self.amp_enabled),
                "autocast_dtype": "float16" if self.amp_enabled else "float32",
                "model_parameter_dtype": str(next(self.model.parameters()).dtype),
                "input_dtype": str(getattr(batch.x, "dtype", None)),
                "cuda_device": str(self.device) if self.device.type == "cuda" else None,
                "torch_version": str(torch.__version__),
                "cuda_version": None if torch.version.cuda is None else str(torch.version.cuda),
                "grad_scaler_scale": float(self.amp_scaler.get_scale()),
            },
            "random_state": rng_state_summary(),
        }

    def _new_nonfinite_diagnostic(
        self, *, phase: str, epoch: int, batch_index: int, global_step: int, batch, loader
    ) -> NonfiniteDiagnostics | None:
        if not self.nonfinite_diagnostics_enabled:
            return None
        return NonfiniteDiagnostics(
            self._diagnostic_context(
                phase=phase,
                epoch=epoch,
                batch_index=batch_index,
                global_step=global_step,
                batch=batch,
                loader=loader,
            )
        )

    def _write_diagnostic_failure(
        self, diagnostic: NonfiniteDiagnostics | None, exc: BaseException
    ) -> Path | None:
        if diagnostic is None:
            return None
        diagnostic.record_exception(exc)
        if diagnostic.parameters is None:
            diagnostic.parameters = summarize_parameter_state(
                self.model, self.optimizer
            )
        if diagnostic.optimizer_update is None and self._last_optimizer_update is not None:
            diagnostic.optimizer_update = dict(self._last_optimizer_update)
        path = write_nonfinite_artifact(self.run_dir, diagnostic)
        if path is not None:
            diagnostic.context["artifact_path"] = str(path)
        return path

    @staticmethod
    def _found_inf(optimizer, scaler) -> float:
        states = getattr(scaler, "_per_optimizer_states", {}).get(id(optimizer), {})
        return sum(
            float(value.detach().cpu())
            for value in states.get("found_inf_per_device", {}).values()
        )

    @staticmethod
    def _gradient_norm(parameters) -> float:
        import torch

        squared = torch.zeros((), dtype=torch.float64)
        for parameter in parameters:
            if parameter.grad is not None:
                squared += parameter.grad.detach().double().square().sum().cpu()
        return float(torch.sqrt(squared))

    @staticmethod
    def _parameter_abs_max(summary: dict[str, Any]) -> float | None:
        values = [
            item.get("abs_max")
            for item in summary.get("largest_abs_parameters", [])
            if item.get("abs_max") is not None
        ]
        return max(values) if values else None

    def _diagnostic_optimizer_update(
        self,
        *,
        diagnostic: NonfiniteDiagnostics,
        epoch: int,
        batch_index: int,
        global_step: int,
        before_scale: float,
        found_inf: float,
        after_scale: float,
        before_parameters: dict[str, Any],
    ) -> None:
        after_parameters = summarize_parameter_state(self.model, self.optimizer)
        before_abs_max = self._parameter_abs_max(before_parameters)
        after_abs_max = self._parameter_abs_max(after_parameters)
        abnormal_growth = bool(
            after_abs_max is not None
            and before_abs_max is not None
            and after_abs_max > max(1_000_000.0, before_abs_max * 10.0)
        )
        update = {
            "epoch": int(epoch),
            "batch_index": int(batch_index),
            "global_step": int(global_step),
            "scaler_scale_before": float(before_scale),
            "scaler_scale_after": after_scale,
            "found_inf": float(found_inf),
            "step_skipped": bool(found_inf > 0.0),
            "parameter_abs_max_before": before_abs_max,
            "parameter_abs_max_after": after_abs_max,
            "parameter_abs_growth_abnormal": abnormal_growth,
        }
        self._last_optimizer_update = dict(update)
        diagnostic.record_optimizer_update(
            update, self.model, self.optimizer, before_parameters=before_parameters
        )
        if diagnostic.classification is None and abnormal_growth:
            diagnostic.classification = "PARAMETER_ABS_MAX_ABNORMAL"
        if diagnostic.classification is not None:
            error = ContractError(
                "nonfinite diagnostic detected " + str(diagnostic.classification)
            )
            self._write_diagnostic_failure(diagnostic, error)
            raise error

    def fit(self, train_loader, val_loader) -> list[dict[str, Any]]:
        import torch
        seed_everything(int(self.effective_config.get("seed", 2026)))
        self.run_dir.mkdir(parents=True, exist_ok=True)
        write_status(self.run_dir, status="RUNNING", run_mode=self.effective_config.get("run_mode", "formal"), artifact_profile=self.effective_config.get("artifact_profile", "TRAIN"))
        epochs = int(self.effective_config.get("epochs", 20))
        patience = int(self.effective_config.get("patience", 6))
        min_delta = float(self.effective_config.get("min_delta", 0.01))
        best = float("inf")
        bad = 0
        history: list[dict[str, Any]] = []
        global_step = 0
        try:
            for epoch in range(1, epochs + 1):
                epoch_started = time.perf_counter()
                self.model.train()
                if hasattr(self.loss_fn, "train"):
                    self.loss_fn.train()
                train_values = []
                loss_detail_values: dict[str, list[float]] = {}
                gradient_norms: list[float] = []
                valid_target_count = 0
                all_masked_batch_count = 0
                for batch_index, batch in enumerate(train_loader):
                    batch = _move_batch(batch, self.device)
                    valid_target_count += int(batch.mask.detach().sum().item())
                    diagnostic = self._new_nonfinite_diagnostic(
                        phase="fit",
                        epoch=epoch,
                        batch_index=batch_index,
                        global_step=global_step,
                        batch=batch,
                        loader=train_loader,
                    )
                    hook_capture = (
                        TransformerHookCapture(
                            self.model, diagnostic.record_module
                        )
                        if diagnostic is not None and self.model_id == "transformer"
                        else None
                    )
                    try:
                        if hook_capture is None:
                            hook_context = None
                        else:
                            hook_context = hook_capture
                        if hook_context is None:
                            with torch.autocast(
                                device_type=self.device.type,
                                dtype=torch.float16,
                                enabled=self.amp_enabled,
                            ):
                                output = adapter_forward(
                                    self.adapter,
                                    self.model,
                                    batch,
                                    expected_horizon=int(self.protocol["max_pred_len"]),
                                    expected_features=int(self.protocol["feature_count"]),
                                    _nonfinite_diagnostics=diagnostic,
                                )
                                loss = call_loss(self.loss_fn, output.prediction, batch, self.protocol)
                        else:
                            with hook_context:
                                with torch.autocast(
                                    device_type=self.device.type,
                                    dtype=torch.float16,
                                    enabled=self.amp_enabled,
                                ):
                                    output = adapter_forward(
                                        self.adapter,
                                        self.model,
                                        batch,
                                        expected_horizon=int(self.protocol["max_pred_len"]),
                                        expected_features=int(self.protocol["feature_count"]),
                                        _nonfinite_diagnostics=diagnostic,
                                    )
                                    loss = call_loss(self.loss_fn, output.prediction, batch, self.protocol)
                    except Exception as exc:
                        self._write_diagnostic_failure(diagnostic, exc)
                        raise
                    if loss is None:
                        all_masked_batch_count += 1
                        continue
                    self.optimizer.zero_grad(set_to_none=True)
                    if diagnostic is None:
                        self.amp_scaler.scale(loss).backward()
                        self.amp_scaler.unscale_(self.optimizer)
                        gradient_norms.append(self._gradient_norm(self.optimizer.param_groups[0]["params"]))
                        self.amp_scaler.step(self.optimizer)
                        self.amp_scaler.update()
                    else:
                        if not bool(torch.isfinite(loss).all().item()):
                            error = ContractError("loss contains NaN/Inf")
                            self._write_diagnostic_failure(diagnostic, error)
                            raise error
                        self.amp_scaler.scale(loss).backward()
                        self.amp_scaler.unscale_(self.optimizer)
                        gradient_norms.append(self._gradient_norm(self.optimizer.param_groups[0]["params"]))
                        before_parameters = summarize_parameter_state(
                            self.model, self.optimizer
                        )
                        before_scale = float(self.amp_scaler.get_scale())
                        self.amp_scaler.step(self.optimizer)
                        found_inf = self._found_inf(self.optimizer, self.amp_scaler)
                        self.amp_scaler.update()
                        after_scale = float(self.amp_scaler.get_scale())
                        self._diagnostic_optimizer_update(
                            diagnostic=diagnostic,
                            epoch=epoch,
                            batch_index=batch_index,
                            global_step=global_step + 1,
                            before_scale=before_scale,
                            found_inf=found_inf,
                            after_scale=after_scale,
                            before_parameters=before_parameters,
                        )
                    train_values.append(float(loss.detach().cpu()))
                    for key, value in dict(getattr(self.loss_fn, "last_details", {})).items():
                        if isinstance(value, (int, float)) and value == value:
                            loss_detail_values.setdefault(str(key), []).append(float(value))
                    global_step += 1
                val_score, val_loss = self._validate(val_loader)
                row = {
                    "epoch": epoch, "train_loss": sum(train_values) / max(len(train_values), 1),
                    "val_loss": val_loss, "val_score_h10": val_score, "is_best": False,
                    "valid_target_count": valid_target_count,
                    "all_masked_batch_count": all_masked_batch_count,
                    "gradient_norm": sum(gradient_norms) / max(len(gradient_norms), 1),
                    "epoch_time_seconds": time.perf_counter() - epoch_started,
                    "nan_inf_event_count": 0,
                }
                row.update({
                    key: sum(values) / len(values)
                    for key, values in loss_detail_values.items()
                    if values
                })
                is_best = val_score == val_score and val_score < best - min_delta
                if is_best:
                    best, bad, row["is_best"] = val_score, 0, True
                else:
                    bad += 1
                history.append(row)
                self.ckpt.save("last_checkpoint.pt", epoch=epoch, global_step=global_step, monitor_value=val_score, model=self.model, loss_fn=self.loss_fn, optimizer=self.optimizer, amp_scaler=self.amp_scaler)
                self._last_checkpoint = {
                    "path": str(self.run_dir / "last_checkpoint.pt"),
                    "epoch": epoch,
                    "global_step": global_step,
                }
                if is_best:
                    self.ckpt.save("best_checkpoint.pt", epoch=epoch, global_step=global_step, monitor_value=val_score, model=self.model, loss_fn=self.loss_fn, optimizer=self.optimizer, amp_scaler=self.amp_scaler)
                if bad >= patience:
                    write_status(self.run_dir, status="EARLY_STOPPED", run_mode=self.effective_config.get("run_mode", "formal"), artifact_profile=self.effective_config.get("artifact_profile", "TRAIN"))
                    break
            fields = list(dict.fromkeys(
                ["epoch", "train_loss", "val_loss", "val_score_h10", "is_best"]
                + [key for row in history for key in row]
            ))
            atomic_write_csv(self.run_dir / "train_log.csv", fields, history)
            write_status(self.run_dir, status="TRAINING_COMPLETED", run_mode=self.effective_config.get("run_mode", "formal"), artifact_profile=self.effective_config.get("artifact_profile", "TRAIN"))
            return history
        except Exception as exc:
            write_status(self.run_dir, status="FAILED", run_mode=self.effective_config.get("run_mode", "formal"), artifact_profile=self.effective_config.get("artifact_profile", "FAILED"), exit_code=1, failure_stage="fit", error_type=type(exc).__name__, error_message=str(exc), traceback_tail=traceback.format_exc().splitlines()[-20:])
            raise

    def _validate(self, val_loader) -> tuple[float, float]:
        import torch
        self.model.eval()
        if hasattr(self.loss_fn, "eval"):
            self.loss_fn.eval()
        scores, losses = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = _move_batch(batch, self.device)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.amp_enabled,
                ):
                    output = adapter_forward(self.adapter, self.model, batch, expected_horizon=int(self.protocol["max_pred_len"]), expected_features=int(self.protocol["feature_count"]))
                    loss = call_loss(self.loss_fn, output.prediction, batch, self.protocol)
                if loss is not None:
                    losses.append(float(loss.detach().cpu()))
                target_raw = batch.target_raw_or_inverse_transform if batch.target_raw_or_inverse_transform is not None else batch.target
                pred_raw = self.inverse_target(output.prediction)
                row = evaluate_horizons(pred_raw, target_raw, batch.mask, (10,), num_nodes=batch.node_count, physical_clip=(float(self.protocol["physical_power_min_kw"]), float(self.protocol["physical_power_max_kw"]))) [0]
                if row["Score"] == row["Score"]:
                    scores.append(float(row["Score"]))
        return (sum(scores) / max(len(scores), 1) if scores else float("nan"), sum(losses) / max(len(losses), 1) if losses else float("nan"))


class Evaluator:
    def __init__(self, trainer: Trainer):
        self.trainer = trainer

    def evaluate(self, test_loader) -> list[dict[str, Any]]:
        import torch
        best_path = self.trainer.run_dir / "best_checkpoint.pt"
        self.trainer.ckpt.load(best_path, self.trainer.model, loss_fn=self.trainer.loss_fn)
        self.trainer.model.eval()
        predictions, targets, masks = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                moved = _move_batch(batch, self.trainer.device)
                output = adapter_forward(self.trainer.adapter, self.trainer.model, moved, expected_horizon=int(self.trainer.protocol["max_pred_len"]), expected_features=int(self.trainer.protocol["feature_count"]))
                predictions.append(self.trainer.inverse_target(output.prediction).detach().cpu())
                targets.append((moved.target_raw_or_inverse_transform if moved.target_raw_or_inverse_transform is not None else moved.target).detach().cpu())
                masks.append(moved.mask.detach().cpu())
        pred, target, mask = torch.cat(predictions), torch.cat(targets), torch.cat(masks)
        return evaluate_horizons(pred, target, mask, self.trainer.protocol["eval_horizons"], num_nodes=int(pred.shape[1]), physical_clip=(float(self.trainer.protocol["physical_power_min_kw"]), float(self.trainer.protocol["physical_power_max_kw"])))
