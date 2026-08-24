from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Callable

import torch

from .cross_fusion_analysis import implementation_audit
from .io_utils import file_identity, write_json


def _tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        return next((item for item in value if torch.is_tensor(item)), None)
    if isinstance(value, dict):
        return next((item for item in value.values() if torch.is_tensor(item)), None)
    return None


class DiagnosticCapture(AbstractContextManager):
    """Forward hooks that are disabled unless this context is entered."""

    def __init__(self, model: torch.nn.Module, specs: dict[str, tuple[str, str, int | None]]):
        self.model, self.specs = model, specs
        self.handles: list[Any] = []
        self.captured: dict[str, torch.Tensor] = {}

    def __enter__(self):
        modules = dict(self.model.named_modules())
        for semantic, (module_path, location, index) in self.specs.items():
            if module_path not in modules:
                raise KeyError(f"Diagnostic module not found: {module_path}")
            module = modules[module_path]
            if location == "input":
                def pre_hook(_module, inputs, kwargs, *, name=semantic, position=index):
                    if len(inputs) > (position or 0):
                        value = inputs[position or 0]
                    else:
                        keyword = "z_fine" if "fine" in name else "z_coarse" if "coarse" in name else None
                        value = kwargs.get(keyword) if keyword else None
                    tensor = _tensor(value)
                    if tensor is not None:
                        self.captured[name] = tensor.detach().cpu()
                self.handles.append(module.register_forward_pre_hook(pre_hook, with_kwargs=True))
            elif location == "output":
                def hook(_module, _inputs, output, *, name=semantic, position=index):
                    value = output[position] if position is not None and isinstance(output, (tuple, list)) else output
                    tensor = _tensor(value)
                    if tensor is not None:
                        self.captured[name] = tensor.detach().cpu()
                self.handles.append(module.register_forward_hook(hook))
            else:
                raise ValueError("Hook location must be input or output.")
        return self

    def __exit__(self, exc_type, exc, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False


class DiagnosticMode(AbstractContextManager):
    """Temporarily request real attention diagnostics and restore every flag."""

    def __init__(self, model: torch.nn.Module, level: str = "full"):
        if level not in {"standard", "full"}:
            raise ValueError("DiagnosticMode level must be standard or full.")
        self.model, self.level = model, level
        self.previous: list[tuple[Any, str, Any]] = []

    def __enter__(self):
        objects = [self.model, *self.model.modules()]
        config = getattr(self.model, "config", None)
        if config is not None:
            objects.append(config)
        seen = set()
        for obj in objects:
            if id(obj) in seen or not hasattr(obj, "diagnostics_level"):
                continue
            seen.add(id(obj))
            self.previous.append((obj, "diagnostics_level", getattr(obj, "diagnostics_level")))
            setattr(obj, "diagnostics_level", self.level)
        return self

    def __exit__(self, exc_type, exc, traceback):
        for obj, name, value in reversed(self.previous):
            setattr(obj, name, value)
        self.previous.clear()
        return False


def default_a0_hook_specs(layer: int = 0) -> dict[str, tuple[str, str, int | None]]:
    fusion = f"coupling_blocks.{layer}.symmetric_cross_fusion"
    coupling = f"coupling_blocks.{layer}"
    decoder = "direct_decoder"
    return {
        "fine_pre_fusion": (fusion, "input", 0), "coarse_pre_fusion": (fusion, "input", 1),
        "fine_after_cross": (fusion, "output", 0), "coarse_after_cross": (fusion, "output", 1),
        "coarse_to_fine_attention_weights": (f"{fusion}.macro_to_fine", "output", 1),
        "fine_to_coarse_attention_weights": (f"{fusion}.fine_to_coarse", "output", 1),
        "fusion_gate": (f"{fusion}.gate", "output", None),
        "macro_prompt": (f"{coupling}.macro_prompt_encoder", "output", 0),
        "st_prompt": ("st_prompt", "output", None),
        "decoder_fine_input": (decoder, "input", 0), "decoder_coarse_input": (decoder, "input", 1),
        "prediction_head_input": (f"{decoder}.predict_head", "input", 0),
        "prediction_head_output": (f"{decoder}.predict_head", "output", None),
    }


def side_effect_audit(
    model: torch.nn.Module, predict: Callable[[bool], torch.Tensor], *, checkpoint: str | Path | None = None,
    atol: float = 1e-6, rtol: float = 1e-6,
) -> dict[str, Any]:
    original_mode = model.training
    params_before = {name: value.detach().cpu().clone() for name, value in model.named_parameters()}
    buffers_before = {name: value.detach().cpu().clone() for name, value in model.named_buffers()}
    rng_cpu = torch.random.get_rng_state().clone()
    rng_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    checkpoint_before = file_identity(checkpoint)
    try:
        model.eval()
        torch.random.set_rng_state(rng_cpu)
        if rng_cuda is not None:
            torch.cuda.set_rng_state_all(rng_cuda)
        with torch.inference_mode():
            prediction_off = predict(False).detach().cpu()
        torch.random.set_rng_state(rng_cpu)
        if rng_cuda is not None:
            torch.cuda.set_rng_state_all(rng_cuda)
        with torch.inference_mode():
            prediction_on = predict(True).detach().cpu()
    finally:
        torch.random.set_rng_state(rng_cpu)
        if rng_cuda is not None:
            torch.cuda.set_rng_state_all(rng_cuda)
        model.train(original_mode)
    parameters_unchanged = all(torch.equal(params_before[name], value.detach().cpu()) for name, value in model.named_parameters())
    buffers_unchanged = all(torch.equal(buffers_before[name], value.detach().cpu()) for name, value in model.named_buffers())
    prediction_equal = prediction_off.shape == prediction_on.shape and torch.allclose(prediction_off, prediction_on, atol=atol, rtol=rtol)
    checkpoint_after = file_identity(checkpoint)
    checkpoint_unchanged = checkpoint_before == checkpoint_after
    passed = prediction_equal and parameters_unchanged and buffers_unchanged and checkpoint_unchanged and model.training == original_mode
    return {
        "status": "PASS" if passed else "FAIL", "prediction_equal_within_tolerance": prediction_equal,
        "prediction_shape": list(prediction_off.shape), "atol": atol, "rtol": rtol,
        "parameters_unchanged": parameters_unchanged, "buffers_unchanged": buffers_unchanged,
        "checkpoint_file_unchanged": checkpoint_unchanged, "checkpoint_identity_before": checkpoint_before,
        "checkpoint_identity_after": checkpoint_after, "rng_isolated_and_restored": True,
        "eval_mode_used": True, "mode_restored": model.training == original_mode,
    }


def static_mechanism_audit() -> dict[str, Any]:
    from st_mgprompt.cross_fusion import SymmetricCrossFusion
    module = SymmetricCrossFusion(hidden_dim=64, num_heads=4, fusion_mode="cross")
    result = implementation_audit(module)
    result.update({
        "fine_coarse_actual_shape": "[B,144,134,64] under current formal input shape",
        "fine_coarse_time_alignment": "identity; both branches preserve L=144",
        "macro_prompt_physical_trend_output": False,
        "st_prompt_horizon_conditioned": True,
        "diagnostic_hooks_default_enabled": False,
    })
    return result


def write_blocked_diagnostic_audit(output_root: str | Path, reason: str) -> dict[str, Any]:
    root = Path(output_root)
    diagnostic_root = root / "mechanism_diagnostics"
    diagnostic_root.mkdir(parents=True, exist_ok=True)
    static = static_mechanism_audit()
    payload = {
        "schema_version": "e8_diagnostic_side_effect_audit_v1", "status": "BLOCKED_NOT_RUN",
        "reason": reason, "formal_checkpoint_evaluate_only_executed": False,
        "source_artifacts_modified": False, "static_mechanism_audit": static,
        "side_effect_test_requirement": "Formal diagnostics remain ineligible until all five internal current-batch4 evidence slots pass.",
    }
    write_json(root / "E8_DIAGNOSTIC_SIDE_EFFECT_AUDIT.json", payload)
    write_json(diagnostic_root / "diagnostics_manifest.json", {
        "status": "BLOCKED_PROTOCOL_MISMATCH", "exports": [], "post_fusion_single_tensor": "NOT_APPLICABLE",
        "attention_and_gate": "EXTRACTABLE_FROM_REAL_MODULE_WHEN_READY", "source_run_writeback": False,
    })
    return payload
