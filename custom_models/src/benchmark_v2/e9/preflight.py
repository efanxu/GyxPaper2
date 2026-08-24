from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any

from benchmark_v2.adapters import adapter_forward
from benchmark_v2.contracts import call_loss
from benchmark_v2.losses import build_msmg_dwu_loss
from benchmark_v2.model_cli import _synthetic_batch, _synthetic_scalers
from benchmark_v2.model_runtime import build_model_runtime
from benchmark_v2.precision import apply_model_precision_policy
from benchmark_v2.protocol import load_protocol
from benchmark_v2.seeds import seed_everything
from benchmark_v2.training_profiles import apply_training_profile

from .constants import LOSS_PROFILE, MODEL_IDS, PREFLIGHT_ROOT, TRAINING_PROFILE_ID
from .io_utils import write_json
from .variants import canonical_transfer_run_id


def _finite_gradients(model) -> bool:
    import torch
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    return bool(gradients) and all(bool(torch.isfinite(value).all().item()) for value in gradients)


def preflight_model(model_id: str, *, device: str = "cuda") -> dict[str, Any]:
    import torch

    protocol = load_protocol()
    seed_everything(int(protocol["default_seed"]))
    input_scaler, target_scaler = _synthetic_scalers()
    started = time.perf_counter()
    result = {
        "model_id": model_id, "run_id": canonical_transfer_run_id(model_id),
        "requested_shape": {"B": 4, "T": 144, "N": 134, "C": 16, "H": 10},
        "device": device, "loss_id": "msmg_dwu_loss", "formal_training_started": False,
    }
    try:
        target_device = torch.device(device)
        if target_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable for the required E9 full-shape preflight.")
        runtime = build_model_runtime(
            model_id, protocol, run_mode="formal", input_scaler=input_scaler, target_scaler=target_scaler,
        )
        apply_training_profile(runtime, TRAINING_PROFILE_ID)
        apply_model_precision_policy(runtime)
        runtime.model.to(target_device).train()
        loss_fn = build_msmg_dwu_loss(**LOSS_PROFILE).to(target_device).train()
        batch = _synthetic_batch(
            batch_size=4, time_steps=144, nodes=134, canonical_graph_nodes=model_id in {"dcrnn", "mtgnn"},
        )
        for name in ("x", "target", "mask", "target_raw_or_inverse_transform"):
            setattr(batch, name, getattr(batch, name).to(target_device))
        amp = bool(runtime.effective_config.get("amp_enabled", False)) and target_device.type == "cuda"
        with torch.autocast(device_type=target_device.type, dtype=torch.float16, enabled=amp):
            output = adapter_forward(runtime.adapter, runtime.model, batch, expected_horizon=10, expected_features=16)
            loss = call_loss(loss_fn, output.prediction, batch, protocol)
        if loss is None or not bool(torch.isfinite(loss).all().item()):
            raise FloatingPointError("MS-MG-DWU returned no finite loss.")
        loss.backward()
        if not _finite_gradients(runtime.model):
            raise FloatingPointError("Model gradients are missing or non-finite.")
        partial_details = dict(loss_fn.last_details)
        all_masked = _synthetic_batch(
            batch_size=4, time_steps=144, nodes=134, canonical_graph_nodes=model_id in {"dcrnn", "mtgnn"},
        )
        all_masked.mask.zero_()
        for name in ("x", "target", "mask", "target_raw_or_inverse_transform"):
            setattr(all_masked, name, getattr(all_masked, name).to(target_device))
        with torch.no_grad():
            all_output = adapter_forward(runtime.adapter, runtime.model, all_masked, expected_horizon=10, expected_features=16)
            all_loss = call_loss(loss_fn, all_output.prediction, all_masked, protocol)
        if all_loss is not None:
            raise RuntimeError("All-masked batch must return None.")
        graph_ok = True
        if model_id in {"dcrnn", "mtgnn"}:
            graph_ok = list(runtime.effective_config.get("ordered_node_ids", [])) == list(range(1, 135))
            if not graph_ok:
                raise RuntimeError("Graph model node order is not the frozen 1..134 order.")
        result.update({
            "status": "PASS", "output_shape": list(output.prediction.shape),
            "loss_finite": True, "loss_components_finite": all(
                not isinstance(value, float) or value == value for value in partial_details.values()
            ),
            "backward_completed": True, "gradient_finite": True,
            "all_masked_behavior": "PASS", "partial_mask_behavior": "PASS",
            "amp_enabled": amp, "amp_behavior": "PASS", "graph_protocol": "PASS" if graph_ok else "FAIL",
        })
    except torch.cuda.OutOfMemoryError as exc:
        result.update({"status": "FAIL_OOM", "error_type": type(exc).__name__, "error_message": str(exc)})
    except Exception as exc:
        result.update({
            "status": "FAIL_NON_OOM", "error_type": type(exc).__name__, "error_message": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-20:],
        })
    result["elapsed_seconds"] = time.perf_counter() - started
    return result


def preflight_all(*, device: str = "cuda", output_root: str | Path = PREFLIGHT_ROOT) -> dict[str, Any]:
    inventory = [preflight_model(model_id, device=device) for model_id in MODEL_IDS]
    payload = {
        "schema_version": "e9_preflight_inventory_v1", "inventory": inventory,
        "passed": sum(row["status"] == "PASS" for row in inventory),
        "expected": 6, "all_passed": all(row["status"] == "PASS" for row in inventory),
        "formal_training_started": False,
    }
    write_json(Path(output_root) / "E9_PREFLIGHT_INVENTORY.json", payload)
    return payload
