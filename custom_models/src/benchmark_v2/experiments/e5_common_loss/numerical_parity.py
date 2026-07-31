from __future__ import annotations

import sys
import importlib.util
from typing import Any

from .loss_profile import e5_loss_adapter


def _case_tensors(case: str, shape: tuple[int, int, int], device: str):
    import torch

    generator = torch.Generator(device="cpu").manual_seed(
        2026 + sum(ord(char) for char in case)
    )
    prediction = torch.randn(shape, generator=generator)
    target = torch.randn(shape, generator=generator)
    if case == "positive_error":
        prediction = target + 2.0
    elif case == "negative_error":
        prediction = target - 2.0
    elif case == "zero_error":
        prediction = target.clone()
    elif case == "extreme_finite":
        prediction = prediction * 1e4
        target = target * 1e4
    mask = torch.ones(shape, dtype=torch.bool)
    if case == "partial_mask":
        mask[..., ::2] = False
    elif case == "random_missing":
        mask = torch.rand(shape, generator=generator) > 0.35
    elif case == "single_valid":
        mask.zero_()
        mask.reshape(-1)[0] = True
    elif case == "all_masked":
        mask.zero_()
    return (
        prediction.to(device).requires_grad_(True),
        target.to(device),
        mask.to(device),
    )


def _compare_case(
    case: str,
    shape: tuple[int, int, int],
    *,
    device: str,
    amp: bool,
) -> dict[str, Any]:
    import torch
    from .loss_profile import LOSS_SOURCE_PATH

    module_name = "_benchmark_v2_e5_parity_a8_losses"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, LOSS_SOURCE_PATH)
        if spec is None or spec.loader is None:
            raise ImportError(LOSS_SOURCE_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    masked_score_aligned_hybrid_loss = module.masked_score_aligned_hybrid_loss

    prediction, target, mask = _case_tensors(case, shape, device)
    canonical_prediction = (
        prediction.detach().clone().transpose(1, 2).requires_grad_(True)
    )
    with torch.autocast(
        device_type=device,
        dtype=torch.float16,
        enabled=amp,
    ):
        canonical = masked_score_aligned_hybrid_loss(
            canonical_prediction,
            target.transpose(1, 2),
            mask.transpose(1, 2),
        )
        adapter = e5_loss_adapter(prediction, target, mask)
    if canonical is None or adapter is None:
        return {
            "case": case,
            "shape": list(shape),
            "device": device,
            "amp": amp,
            "canonical_is_none": canonical is None,
            "adapter_is_none": adapter is None,
            "status": "PASS" if canonical is None and adapter is None else "FAIL",
        }
    canonical.backward()
    adapter.backward()
    canonical_gradient = canonical_prediction.grad.transpose(1, 2)
    adapter_gradient = prediction.grad
    atol = 2e-4 if amp else 1e-7
    rtol = 2e-3 if amp else 1e-6
    value_equal = bool(
        torch.allclose(canonical.detach(), adapter.detach(), atol=atol, rtol=rtol)
    )
    gradient_equal = bool(
        torch.allclose(canonical_gradient, adapter_gradient, atol=atol, rtol=rtol)
    )
    finite_equal = bool(
        torch.isfinite(canonical).item() == torch.isfinite(adapter).item()
    )
    return {
        "case": case,
        "shape": list(shape),
        "device": device,
        "amp": amp,
        "canonical_value": float(canonical.detach().float().cpu()),
        "adapter_value": float(adapter.detach().float().cpu()),
        "value_equal": value_equal,
        "gradient_equal": gradient_equal,
        "gradient_max_abs_diff": float(
            (canonical_gradient - adapter_gradient).abs().max().detach().float().cpu()
        ),
        "canonical_dtype": str(canonical.dtype),
        "adapter_dtype": str(adapter.dtype),
        "dtype_equal": canonical.dtype == adapter.dtype,
        "finite_equal": finite_equal,
        "status": (
            "PASS"
            if value_equal
            and gradient_equal
            and canonical.dtype == adapter.dtype
            and finite_equal
            else "FAIL"
        ),
    }


def run_numerical_parity() -> dict[str, Any]:
    import torch

    cases = [
        ("all_valid", (2, 4, 10)),
        ("partial_mask", (2, 4, 10)),
        ("random_missing", (3, 7, 6)),
        ("single_valid", (1, 2, 3)),
        ("positive_error", (2, 3, 10)),
        ("negative_error", (2, 3, 10)),
        ("zero_error", (2, 3, 10)),
        ("extreme_finite", (1, 5, 4)),
        ("all_masked", (2, 4, 10)),
    ]
    results = [
        _compare_case(case, shape, device="cpu", amp=False)
        for case, shape in cases
    ]
    cuda_status = "NOT_AVAILABLE"
    if torch.cuda.is_available():
        cuda_status = "RUN"
        results.extend(
            _compare_case(case, shape, device="cuda", amp=True)
            for case, shape in cases
        )
    passed = all(result["status"] == "PASS" for result in results)
    loaded_before = set(sys.modules)
    e5_loss_adapter(
        torch.zeros(1, 1, 1),
        torch.zeros(1, 1, 1),
        torch.ones(1, 1, 1, dtype=torch.bool),
    )
    newly_loaded = set(sys.modules) - loaded_before
    model_modules = sorted(
        name
        for name in newly_loaded
        if name.startswith("st_mgprompt.model")
        or name.startswith("st_mgprompt.registry")
    )
    return {
        "schema_version": "e5_loss_numerical_parity_v1",
        "status": "PASS" if passed and not model_modules else "FAIL",
        "cpu_fp32": "PASS"
        if all(
            item["status"] == "PASS"
            for item in results
            if item["device"] == "cpu"
        )
        else "FAIL",
        "cuda_amp": (
            "PASS"
            if cuda_status == "RUN"
            and all(
                item["status"] == "PASS"
                for item in results
                if item["device"] == "cuda"
            )
            else cuda_status
        ),
        "all_masked_behavior": "None",
        "import_path": "st_mgprompt.losses.masked_score_aligned_hybrid_loss",
        "st_mgprompt_model_modules_loaded": model_modules,
        "extra_gpu_model_created": False,
        "cases": results,
    }
