from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .artifacts import atomic_write_json


DIAGNOSTICS_ENV = "BENCHMARK_V2_NONFINITE_DIAGNOSTICS"
DIAGNOSTIC_SCHEMA = "benchmark_v2_nonfinite_diagnostic_v1"


def diagnostics_enabled() -> bool:
    return os.environ.get(DIAGNOSTICS_ENV, "") == "1"


def _finite_mask(value):
    import torch

    if value.is_floating_point() or value.is_complex():
        return torch.isfinite(value)
    return torch.ones_like(value, dtype=torch.bool)


def _finite_numeric_values(value, finite):
    import torch

    if not value.numel():
        return torch.empty(0, dtype=torch.float32, device=value.device)
    selected = value.detach()[finite]
    if selected.numel() == 0:
        return torch.empty(0, dtype=torch.float32, device=value.device)
    return selected.abs().float() if value.is_complex() else selected.float()


def tensor_summary(value: Any) -> dict[str, Any]:
    """Return bounded scalar statistics for one tensor; never stores tensor data."""

    import torch

    if not isinstance(value, torch.Tensor):
        return {"type": type(value).__name__, "tensor": False}
    detached = value.detach()
    finite = _finite_mask(detached)
    finite_values = _finite_numeric_values(detached, finite)
    nan_count = 0
    posinf_count = 0
    neginf_count = 0
    if detached.is_floating_point() or detached.is_complex():
        nan_count = int(torch.isnan(detached).sum().item())
        posinf_count = int(torch.isposinf(detached).sum().item())
        neginf_count = int(torch.isneginf(detached).sum().item())
    summary: dict[str, Any] = {
        "shape": [int(item) for item in detached.shape],
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "numel": int(detached.numel()),
        "nan_count": nan_count,
        "posinf_count": posinf_count,
        "neginf_count": neginf_count,
        "finite_count": int(finite.sum().item()),
        "finite_ratio": (
            float(finite.sum().item() / detached.numel())
            if detached.numel()
            else 1.0
        ),
    }
    if finite_values.numel():
        summary.update(
            {
                "min": float(finite_values.min().cpu().item()),
                "max": float(finite_values.max().cpu().item()),
                "mean": float(finite_values.mean().cpu().item()),
                "abs_max": float(finite_values.abs().max().cpu().item()),
            }
        )
    else:
        summary.update({"min": None, "max": None, "mean": None, "abs_max": None})
    if not bool(finite.all().item()):
        first = torch.nonzero(~finite, as_tuple=False)[0]
        summary["first_nonfinite_index"] = [int(item) for item in first.cpu().tolist()]
    return summary


def iter_tensors(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    import torch

    if isinstance(value, torch.Tensor):
        yield prefix or "tensor", value
    elif isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from iter_tensors(item, child)
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]" if prefix else f"[{index}]"
            yield from iter_tensors(item, child)


def nested_tensor_summaries(value: Any, prefix: str) -> dict[str, dict[str, Any]]:
    return {name: tensor_summary(item) for name, item in iter_tensors(value, prefix)}


def _summary_is_nonfinite(summary: dict[str, Any]) -> bool:
    return bool(summary.get("nan_count", 0) or summary.get("posinf_count", 0) or summary.get("neginf_count", 0))


def _first_nonfinite(mapping: dict[str, dict[str, Any]]) -> str | None:
    for name, summary in mapping.items():
        if _summary_is_nonfinite(summary):
            return name
    return None


def rng_state_summary() -> dict[str, Any]:
    import torch

    return {
        "python_state_available": random.getstate() is not None,
        "torch_cpu_state_available": torch.get_rng_state() is not None,
        "cuda_state_count": len(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else 0,
    }


def _top_abs_items(items: list[tuple[float | None, str, bool]], limit: int = 8) -> list[dict[str, Any]]:
    def sort_key(item):
        value = item[0]
        return float("-inf") if value is None else value

    return [
        {"name": name, "abs_max": abs_max, "finite": finite}
        for abs_max, name, finite in sorted(items, key=sort_key, reverse=True)[:limit]
    ]


def summarize_parameter_state(model, optimizer=None) -> dict[str, Any]:
    import torch

    parameter_items: list[tuple[float | None, str, bool]] = []
    first_parameter = None
    for name, parameter in model.named_parameters():
        summary = tensor_summary(parameter)
        finite = not _summary_is_nonfinite(summary)
        if not finite and first_parameter is None:
            first_parameter = name
        parameter_items.append((summary.get("abs_max"), name, finite))

    gradient_items: list[tuple[float | None, str, bool]] = []
    first_gradient = None
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        summary = tensor_summary(parameter.grad)
        finite = not _summary_is_nonfinite(summary)
        if not finite and first_gradient is None:
            first_gradient = name
        gradient_items.append((summary.get("abs_max"), name, finite))

    optimizer_items: list[tuple[float | None, str, bool]] = []
    first_optimizer = None
    if optimizer is not None:
        names = {id(parameter): name for name, parameter in model.named_parameters()}
        for parameter, state in optimizer.state.items():
            parameter_name = names.get(id(parameter), f"parameter_{id(parameter)}")
            for key, value in state.items():
                if not isinstance(value, torch.Tensor):
                    continue
                summary = tensor_summary(value)
                finite = not _summary_is_nonfinite(summary)
                path = f"optimizer.state[{parameter_name}].{key}"
                if not finite and first_optimizer is None:
                    first_optimizer = path
                optimizer_items.append((summary.get("abs_max"), path, finite))

    return {
        "parameters_all_finite": first_parameter is None,
        "first_nonfinite_parameter": first_parameter,
        "largest_abs_parameters": _top_abs_items(parameter_items),
        "gradients_all_finite": first_gradient is None,
        "first_nonfinite_gradient": first_gradient,
        "largest_abs_gradients": _top_abs_items(gradient_items),
        "optimizer_tensors_all_finite": first_optimizer is None,
        "first_nonfinite_optimizer_tensor": first_optimizer,
        "largest_abs_optimizer_tensors": _top_abs_items(optimizer_items),
    }


@dataclass
class NonfiniteDiagnostics:
    context: dict[str, Any]
    inputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    model_inputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    prediction: dict[str, Any] | None = None
    first_module: dict[str, Any] | None = None
    optimizer_update: dict[str, Any] | None = None
    parameters: dict[str, Any] | None = None
    random_state: dict[str, Any] = field(default_factory=rng_state_summary)
    exception: dict[str, Any] | None = None
    classification: str | None = None

    def record_batch(self, batch) -> None:
        fields = {
            "x": batch.x,
            "target": batch.target,
            "mask": batch.mask,
            "encoder_time_marks": batch.x_mark,
            "decoder_time_marks": batch.y_mark,
        }
        for name, value in fields.items():
            if value is not None:
                self.inputs[name] = tensor_summary(value)
        if (first := _first_nonfinite(self.inputs)) is not None:
            self.classification = "INPUT_NONFINITE"
            self.context.setdefault("input_first_nonfinite", first)

    def record_model_inputs(self, value: Any) -> None:
        self.model_inputs.update(nested_tensor_summaries(value, "adapter_input"))
        if self.classification is None and (first := _first_nonfinite(self.model_inputs)) is not None:
            self.classification = "INPUT_NONFINITE"
            self.context.setdefault("model_input_first_nonfinite", first)

    def record_prediction(self, value: Any) -> None:
        self.prediction = tensor_summary(value)
        self._record_prediction_precision(self.prediction)

    def record_prediction_summary(
        self,
        summary: dict[str, Any],
        *,
        autocast_enabled: bool | None = None,
    ) -> None:
        """Record bounded prediction statistics produced outside the adapter."""

        self.prediction = dict(summary)
        self._record_prediction_precision(
            self.prediction, autocast_enabled=autocast_enabled
        )

    def _record_prediction_precision(
        self,
        summary: dict[str, Any],
        *,
        autocast_enabled: bool | None = None,
    ) -> None:
        precision = self.context.setdefault("precision", {})
        precision["prediction_dtype"] = summary.get("dtype")
        if autocast_enabled is not None:
            precision["autocast_enabled"] = bool(autocast_enabled)
        else:
            try:
                import torch

                precision["autocast_enabled"] = bool(
                    torch.is_autocast_enabled("cuda")
                )
            except (ImportError, TypeError):
                pass
        if self.classification is None and _summary_is_nonfinite(summary):
            self.classification = "FORWARD_NONFINITE"

    def record_module(self, event: dict[str, Any]) -> None:
        if self.first_module is None:
            self.first_module = event

    def record_optimizer_update(
        self,
        update: dict[str, Any],
        model,
        optimizer,
        before_parameters: dict[str, Any] | None = None,
    ) -> None:
        self.optimizer_update = dict(update)
        self.parameters = summarize_parameter_state(model, optimizer)
        if before_parameters is not None:
            self.parameters["before_update"] = before_parameters
        if self.classification is None:
            if before_parameters is not None and not before_parameters["gradients_all_finite"]:
                self.classification = "NONFINITE_GRADIENT"
            elif not self.parameters["parameters_all_finite"]:
                self.classification = "PARAMETER_NONFINITE_AFTER_UPDATE"
            elif not self.parameters["optimizer_tensors_all_finite"]:
                self.classification = "OPTIMIZER_STATE_NONFINITE"

    def record_exception(self, exc: BaseException) -> None:
        self.exception = {"type": type(exc).__name__, "message": str(exc)}

    @property
    def has_nonfinite(self) -> bool:
        return bool(
            self.classification is not None
            or _first_nonfinite(self.inputs)
            or _first_nonfinite(self.model_inputs)
            or (self.prediction is not None and _summary_is_nonfinite(self.prediction))
            or (
                self.first_module is not None
                and any(
                    _summary_is_nonfinite(item)
                    for item in self.first_module.get("outputs", [])
                )
            )
            or (
                self.parameters is not None
                and (
                    not self.parameters["parameters_all_finite"]
                    or not self.parameters["gradients_all_finite"]
                    or not self.parameters["optimizer_tensors_all_finite"]
                )
            )
        )

    def payload(self) -> dict[str, Any]:
        classification = self.classification
        if classification is None and self.has_nonfinite:
            classification = "UNRESOLVED"
        return {
            "schema_version": DIAGNOSTIC_SCHEMA,
            "classification": classification,
            "run_identity": dict(self.context),
            "inputs": dict(self.inputs),
            "adapter_inputs": dict(self.model_inputs),
            "prediction": self.prediction,
            "first_nonfinite_module": self.first_module,
            "optimizer_update": self.optimizer_update,
            "parameters": self.parameters,
            "random_state": dict(self.random_state),
            "exception": self.exception,
        }


def write_nonfinite_artifact(run_dir: str | Path, diagnostic: NonfiniteDiagnostics) -> Path | None:
    if not diagnostic.has_nonfinite:
        return None
    context = diagnostic.context
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    model_id = str(context.get("model_id", "model")).replace("/", "_")
    epoch = int(context.get("epoch", -1))
    batch_index = int(context.get("batch_index", -1))
    global_step = int(context.get("global_step", -1))
    path = (
        Path(run_dir)
        / "nonfinite_diagnostics"
        / f"{model_id}_epoch{epoch}_batch{batch_index}_step{global_step}_{stamp}.json"
    )
    diagnostic.context["artifact_path"] = str(path)
    atomic_write_json(path, diagnostic.payload())
    return path
