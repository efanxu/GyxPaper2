from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .artifacts import atomic_write_json
from .configs import (
    resolve_dlinear_config,
    resolve_gru_config,
    resolve_itransformer_config,
    resolve_lightts_config,
    resolve_patchtst_config,
    resolve_segrnn_config,
    resolve_tide_config,
    resolve_timexer_config,
    resolve_transformer_config,
    resolve_timesnet_config,
    resolve_micn_config,
    resolve_wpmixer_config,
    resolve_multipatchformer_config,
    resolve_timemixer_config,
    resolve_tsmixer_config,
    resolve_frets_config,
    resolve_crossformer_config,
    resolve_msgnet_config,
    resolve_timefilter_config,
    resolve_gcn_config,
    resolve_stgcn_config,
    resolve_dcrnn_config,
    resolve_graph_wavenet_config,
    resolve_mtgnn_config,
    resolve_agcrn_config,
    resolve_stid_config,
)
from .contracts import BenchmarkBatch
from .errors import ModelUnavailableError
from .experiments.e5_common_loss.loss_profile import (
    CLI_PROFILE_ID as E5_PROFILE_ID,
    DEFAULT_PROFILE_ID,
    get_profile_metadata,
    loss_for_profile,
)
from .experiments.e5_common_loss.runner import normalize_profile
from .experiments.e5_common_loss.scope27_contract import (
    E5_SCOPE27_ID,
    is_scope27_train_request,
    validate_scope27_request,
)
from .losses import get_loss
from .model_runtime import build_model_runtime
from .protocol import load_protocol
from .registry import load_registry
from .graph import load_graph_bundle
from .models.graph_models.common import MODEL_SUPPORTS, graph_identity_dict
from .models.graph_models.adaptive_common import e3_c_graph_identity
from .runtime import PROJECT_ROOT, SMOKE_ROOT
from .seeds import seed_everything
from .training_profiles import (
    apply_training_profile,
    batch_identity,
    load_training_profile,
)


PREFLIGHT_SHAPE = {"B": 32, "T": 144, "N": 134, "C": 16, "H": 10}
PREFLIGHT_ROOT = SMOKE_ROOT / "hardware_preflight"
RUNNER_PATH = PROJECT_ROOT / "custom_models/src/benchmark_v2/run_benchmark.py"
_RESOLVERS = {
    "gru": resolve_gru_config,
    "dlinear": resolve_dlinear_config,
    "lightts": resolve_lightts_config,
    "tide": resolve_tide_config,
    "segrnn": resolve_segrnn_config,
    "transformer": resolve_transformer_config,
    "patchtst": resolve_patchtst_config,
    "itransformer": resolve_itransformer_config,
    "timexer": resolve_timexer_config,
    "timesnet": resolve_timesnet_config,
    "micn": resolve_micn_config,
    "wpmixer": resolve_wpmixer_config,
    "multipatchformer": resolve_multipatchformer_config,
    "timemixer": resolve_timemixer_config,
    "tsmixer": resolve_tsmixer_config,
    "frets": resolve_frets_config,
    "crossformer": resolve_crossformer_config,
    "msgnet": resolve_msgnet_config,
    "timefilter": resolve_timefilter_config,
    "gcn": resolve_gcn_config,
    "stgcn": resolve_stgcn_config,
    "dcrnn": resolve_dcrnn_config,
    "graph_wavenet": resolve_graph_wavenet_config,
    "mtgnn": resolve_mtgnn_config,
    "agcrn": resolve_agcrn_config,
    "stid": resolve_stid_config,
}
E3_C_MODELS = {"graph_wavenet", "mtgnn", "agcrn", "stid"}
NATIVE_NODE_MODELS = set(MODEL_SUPPORTS) | E3_C_MODELS


def _safe_machine_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_.-")
    return token or "UNKNOWN"


def _nvidia_smi_value(query: str) -> str | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    value = completed.stdout.splitlines()[0].strip() if completed.stdout else ""
    return value or None


@lru_cache(maxsize=1)
def machine_identity() -> dict[str, Any]:
    """Return the immutable machine binding used by exact preflight artifacts."""
    import torch

    hostname = socket.gethostname()
    operating_system = f"{platform.system()} {platform.release()}".strip()
    gpu_name: str | None = None
    gpu_uuid: str | None = None
    gpu_total_memory: int | None = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu_name = str(properties.name)
        gpu_total_memory = int(properties.total_memory)
        raw_uuid = getattr(properties, "uuid", None)
        if raw_uuid is not None:
            gpu_uuid = str(raw_uuid)
    gpu_uuid = gpu_uuid or _nvidia_smi_value("uuid")
    driver_version = _nvidia_smi_value("driver_version")
    override = os.environ.get("UNIFORM_BATCH4_MACHINE_ID", "").strip()
    if override:
        machine_id = _safe_machine_token(override)
    else:
        gpu_token = gpu_uuid or gpu_name or "NO_GPU"
        machine_id = _safe_machine_token(
            f"{hostname}_{platform.system()}_{gpu_token}"
        )
    return {
        "machine_id": machine_id,
        "hostname": hostname,
        "operating_system": operating_system,
        "python_version": platform.python_version(),
        "pytorch_version": str(torch.__version__),
        "cuda_version": (
            None if torch.version.cuda is None else str(torch.version.cuda)
        ),
        "gpu_name": gpu_name,
        "gpu_uuid": gpu_uuid,
        "gpu_total_memory": gpu_total_memory,
        "driver_version": driver_version,
    }


def stable_hash(payload: Mapping[str, Any]) -> str:
    material = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def preflight_shape(training_profile: str | None = None) -> dict[str, int]:
    profile = load_training_profile(training_profile)
    shape = dict(PREFLIGHT_SHAPE)
    if profile is not None:
        shape["B"] = profile.train_batch_size
    return shape


def _loss_identity(selected_profile: str) -> dict[str, Any]:
    if selected_profile == E5_PROFILE_ID:
        profile = get_profile_metadata(selected_profile)
        return {
            "loss_id": profile["loss_id"],
            "loss_source_hash": profile["loss_source_hash"],
            "loss_profile_hash": profile["loss_profile_hash"],
        }
    loss_source = PROJECT_ROOT / "custom_models/src/benchmark_v2/losses.py"
    loss_source_hash = hashlib.sha256(loss_source.read_bytes()).hexdigest()
    return {
        "loss_id": "masked_mse",
        "loss_source_hash": loss_source_hash,
        "loss_profile_hash": stable_hash(
            {"loss_id": "masked_mse", "source_hash": loss_source_hash}
        ),
    }


def preflight_identity(
    model_id: str,
    *,
    experiment_profile: str | None = None,
    training_profile: str | None = None,
) -> dict[str, Any]:
    protocol = load_protocol()
    entry = load_registry().get(model_id)
    if entry.canonical_id not in _RESOLVERS:
        raise ModelUnavailableError(
            f"{entry.display_name} has no trainable hardware preflight configuration."
        )
    config = _RESOLVERS[entry.canonical_id](protocol, run_mode="formal")
    source_hash = entry.values.get("source_sha256")
    if not source_hash:
        source_path = PROJECT_ROOT / str(entry.source_path)
        source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    identity = {
        **machine_identity(),
        "model_id": entry.canonical_id,
        "model_config_hash": stable_hash(config),
        "protocol_hash": protocol.protocol_hash,
        "source_hash": source_hash,
        **preflight_shape(training_profile),
        "amp": bool(protocol["amp_enabled"]),
    }
    if entry.canonical_id in MODEL_SUPPORTS:
        identity["benchmark_protocol_hash"] = protocol.protocol_hash
        identity.update(
            graph_identity_dict(
                load_graph_bundle(), MODEL_SUPPORTS[entry.canonical_id]
            )
        )
    elif entry.canonical_id in E3_C_MODELS:
        identity["benchmark_protocol_hash"] = protocol.protocol_hash
        identity["source_closure_hash"] = source_hash
        identity.update(
            e3_c_graph_identity(load_graph_bundle(), entry.canonical_id)
        )
    selected_profile = normalize_profile(experiment_profile)
    if selected_profile == E5_PROFILE_ID:
        profile = get_profile_metadata(selected_profile)
        identity.update(
            {
                "base_model_config_hash": identity["model_config_hash"],
                "model_source_closure_hash": identity.get(
                    "source_closure_hash", identity["source_hash"]
                ),
                "benchmark_protocol_hash": protocol.protocol_hash,
                "experiment_profile_id": profile["profile_id"],
                "e5_common_loss_protocol_hash": profile[
                    "e5_common_loss_protocol_hash"
                ],
                "loss_id": profile["loss_id"],
                "loss_source_hash": profile["loss_source_hash"],
                "loss_profile_hash": profile["loss_profile_hash"],
                "seed": int(protocol["default_seed"]),
            }
        )
    if training_profile is not None:
        identity.update(
            {
                "base_benchmark_protocol_hash": protocol.protocol_hash,
                "model_source_closure_hash": identity.get(
                    "source_closure_hash", identity["source_hash"]
                ),
                "graph_protocol_hash": (
                    "f8224287f2a41be0e67e808445ce9a43b9973e7b075d1e1b4768396d6f98e1ef"
                ),
                "graph_bundle_hash": (
                    "388a87dcba4fa0b897ab9b40b546d844433a26c799d50f4a009d832c6e335f0e"
                ),
                "node_order_hash": (
                    "1bcfaa0afafb0a6535b9d24129bb2ba260cfd1782b9b485d960e6cfe4af90c35"
                ),
                "timestamp_identity_policy": "SOURCE_NAIVE_UNCHANGED",
                "seed": int(protocol["default_seed"]),
                **_loss_identity(selected_profile),
                **batch_identity(training_profile),
            }
        )
    return identity


def preflight_artifact_path(
    model_id: str,
    *,
    root: str | Path | None = None,
    experiment_profile: str | None = None,
    training_profile: str | None = None,
) -> Path:
    identity = preflight_identity(
        model_id,
        experiment_profile=experiment_profile,
        training_profile=training_profile,
    )
    key = stable_hash(identity)[:20]
    return Path(root or PREFLIGHT_ROOT) / identity["model_id"] / key / "hardware_preflight.json"


def read_matching_pass(
    model_id: str,
    *,
    root: str | Path | None = None,
    experiment_profile: str | None = None,
    training_profile: str | None = None,
) -> dict[str, Any] | None:
    path = preflight_artifact_path(
        model_id,
        root=root,
        experiment_profile=experiment_profile,
        training_profile=training_profile,
    )
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = preflight_identity(
        model_id,
        experiment_profile=experiment_profile,
        training_profile=training_profile,
    )
    if any(payload.get(key) != value for key, value in expected.items()):
        return None
    if (
        payload.get("status") != "PASS"
        or payload.get("forward_completed") is not True
        or payload.get("backward_completed") is not True
    ):
        return None
    return payload


def _synthetic_batch(
    model_id: str, *, training_profile: str | None = None
) -> BenchmarkBatch:
    import torch

    generator = torch.Generator().manual_seed(2026)
    shape = preflight_shape(training_profile)
    return BenchmarkBatch(
        x=torch.randn(
            shape["B"], shape["T"], shape["N"], shape["C"], generator=generator
        ),
        target=torch.randn(
            shape["B"], shape["N"], shape["H"], generator=generator
        ),
        mask=torch.ones(shape["B"], shape["N"], shape["H"], dtype=torch.bool),
        sample_ids=list(range(shape["B"])),
        window_end_indices=list(range(shape["B"])),
        node_ids=(
            list(range(1, shape["N"] + 1))
            if model_id in NATIVE_NODE_MODELS
            else list(range(shape["N"]))
        ),
        metadata={
            "contains_future_target": False,
            "hardware_preflight": True,
            "history_end_timestamp": [
                "2021-01-04 12:00:00" for _ in range(shape["B"])
            ],
            "history_end_time_of_day_id": [72 for _ in range(shape["B"])],
            "history_end_day_of_week_id": [0 for _ in range(shape["B"])],
            "timezone_policy": "SOURCE_NAIVE_UNCHANGED",
        },
    )


def run_preflight_worker(
    model_id: str,
    *,
    root: str | Path | None = None,
    experiment_profile: str | None = None,
    training_profile: str | None = None,
) -> dict[str, Any]:
    import torch

    protocol = load_protocol()
    selected_profile = normalize_profile(experiment_profile)
    identity = preflight_identity(
        model_id,
        experiment_profile=selected_profile,
        training_profile=training_profile,
    )
    path = preflight_artifact_path(
        model_id,
        root=root,
        experiment_profile=selected_profile,
        training_profile=training_profile,
    )
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if all(existing.get(key) == value for key, value in identity.items()):
            return existing
        raise RuntimeError(
            f"Refusing to overwrite non-matching preflight artifact: {path}"
        )
    result: dict[str, Any] = {
        **identity,
        "preflight_pid": os.getpid(),
        "forward_completed": False,
        "backward_completed": False,
        "prediction_shape": None,
        "loss": None,
        "gpu_name": None,
        "gpu_total_memory": None,
        "peak_allocated_memory": None,
        "peak_reserved_memory": None,
        "status": "FAIL_NON_OOM",
        "error_type": None,
        "error_message": None,
        "traceback": None,
        "failure_stage": "construction",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for exact formal hardware preflight.")
        device = torch.device("cuda")
        properties = torch.cuda.get_device_properties(device)
        result["gpu_name"] = properties.name
        result["gpu_total_memory"] = int(properties.total_memory)
        torch.cuda.reset_peak_memory_stats(device)
        seed_everything(int(protocol["default_seed"]))
        batch = _synthetic_batch(
            model_id, training_profile=training_profile
        )
        runtime = build_model_runtime(model_id, protocol, run_mode="formal")
        apply_training_profile(runtime, training_profile)
        runtime.model.to(device)
        result["failure_stage"] = "forward"
        moved = BenchmarkBatch(
            x=batch.x.to(device),
            target=batch.target.to(device),
            mask=batch.mask.to(device),
            sample_ids=batch.sample_ids,
            window_end_indices=batch.window_end_indices,
            node_ids=batch.node_ids,
            split=batch.split,
            metadata=batch.metadata,
        )
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=bool(protocol["amp_enabled"]),
        ):
            output = runtime.adapter(
                runtime.model,
                moved,
                expected_horizon=identity["H"],
                expected_features=identity["C"],
            )
            result["forward_completed"] = True
            result["prediction_shape"] = list(output.prediction.shape)
            if result["prediction_shape"] != [
                identity["B"],
                identity["N"],
                identity["H"],
            ]:
                raise RuntimeError(
                    f"Prediction shape mismatch: {result['prediction_shape']}"
                )
            loss = loss_for_profile(selected_profile)(
                output.prediction, moved.target, moved.mask
            )
        if not torch.isfinite(loss):
            raise RuntimeError("Preflight loss is not finite.")
        result["failure_stage"] = "backward"
        loss.backward()
        finite_gradients = all(
            torch.isfinite(parameter.grad).all().item()
            for parameter in runtime.model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        )
        if not finite_gradients:
            raise RuntimeError("Preflight gradients are not finite.")
        result["backward_completed"] = True
        result["finite_gradients"] = True
        torch.cuda.synchronize(device)
        result["loss"] = float(loss.detach().cpu())
        result["status"] = "PASS"
        result["failure_stage"] = None
    except Exception as exc:
        oom_type = getattr(torch, "OutOfMemoryError", None)
        is_oom = (
            (oom_type is not None and isinstance(exc, oom_type))
            or "out of memory" in str(exc).casefold()
        )
        result["status"] = "FAIL_OOM" if is_oom else "FAIL_NON_OOM"
        result["error_type"] = type(exc).__name__
        result["error_message"] = str(exc)
        import traceback

        result["traceback"] = traceback.format_exc()
    finally:
        if torch.cuda.is_available():
            result["peak_allocated_memory"] = int(torch.cuda.max_memory_allocated())
            result["peak_reserved_memory"] = int(torch.cuda.max_memory_reserved())
        atomic_write_json(path, result)
        if result["status"] == "FAIL_NON_OOM":
            atomic_write_json(
                path.with_name("engineering_block.json"),
                {
                    "model_id": identity["model_id"],
                    "runtime_status": "BLOCKED_NON_OOM",
                    "source": str(path),
                    "error_type": result["error_type"],
                    "error_message": result["error_message"],
                    "created_at": result["created_at"],
                },
            )
    return result


def _run_child(
    args: Sequence[str],
    *,
    runner: Callable[..., Any] = subprocess.run,
) -> int:
    completed = runner(
        [str(Path(sys.executable).resolve()), str(RUNNER_PATH), *args],
        check=False,
    )
    return int(completed.returncode)


def launch_preflight(
    model_id: str,
    *,
    root: str | Path | None = None,
    experiment_profile: str | None = None,
    training_profile: str | None = None,
    runner: Callable[..., Any] = subprocess.run,
) -> int:
    args = ["_hardware-preflight-worker", "--model", model_id]
    selected_profile = normalize_profile(experiment_profile)
    if selected_profile != DEFAULT_PROFILE_ID:
        args.extend(["--experiment-profile", selected_profile])
    if training_profile is not None:
        args.extend(["--training-profile", training_profile])
    if root is not None:
        args.extend(["--preflight-root", str(Path(root))])
    return _run_child(args, runner=runner)


def launch_formal_train(
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
    runner: Callable[..., Any] = subprocess.run,
) -> int:
    entry = load_registry().get(model_id)
    status = str(entry.runtime_status)
    if bool(entry.supports_non_trainable):
        raise ModelUnavailableError(
            f"{entry.display_name} is a deterministic non-trainable baseline. "
            "Use evaluate-only for a formal run. No training was started."
        )
    true_blocked = (
        status.startswith("BLOCKED_")
        or status in {"NAME_CONFLICT", "MISSING", "UNAVAILABLE_PENDING_STAGE"}
    )
    if true_blocked or not entry.supports_train:
        raise ModelUnavailableError(
            f"{entry.display_name} is unavailable for training; "
            f"runtime_status={status}; planned_stage={entry.planned_stage}. "
            "No formal run was started."
        )
    selected_profile = normalize_profile(experiment_profile)
    validate_scope27_request(
        model_id=entry.canonical_id,
        formal_scope_id=formal_scope_id,
        experiment_profile=selected_profile,
        training_profile=training_profile,
        trainable=True,
    )
    scope27_request = is_scope27_train_request(
        model_id=entry.canonical_id,
        formal_scope_id=formal_scope_id,
        experiment_profile=selected_profile,
        training_profile=training_profile,
    )
    needs_preflight = not scope27_request and (
        training_profile is not None
        or selected_profile == E5_PROFILE_ID
        or bool(entry.values.get("formal_hardware_preflight_required", False))
    )
    if needs_preflight:
        if (
            read_matching_pass(
                entry.canonical_id,
                root=preflight_root,
                experiment_profile=selected_profile,
                training_profile=training_profile,
            )
            is None
        ):
            code = launch_preflight(
                entry.canonical_id,
                root=preflight_root,
                experiment_profile=selected_profile,
                training_profile=training_profile,
                runner=runner,
            )
            if code != 0:
                return code
        if (
            read_matching_pass(
                entry.canonical_id,
                root=preflight_root,
                experiment_profile=selected_profile,
                training_profile=training_profile,
            )
            is None
        ):
            return 3
    worker_args = [
        "_formal-train-worker",
        "--model", entry.canonical_id,
        "--input-path", str(input_path),
        "--target-path", str(target_path),
        "--output-root", str(output_root),
        "--run-id", run_id,
        "--device", device,
    ]
    if selected_profile != DEFAULT_PROFILE_ID:
        worker_args.extend(["--experiment-profile", selected_profile])
    if training_profile is not None:
        worker_args.extend(["--training-profile", training_profile])
    if formal_scope_id == E5_SCOPE27_ID:
        worker_args.extend(["--formal-scope-id", formal_scope_id])
    if preflight_root is not None:
        worker_args.extend(["--preflight-root", str(Path(preflight_root))])
    return _run_child(worker_args, runner=runner)


def launch_model_suite(
    requests: Sequence[Mapping[str, Any]],
    *,
    launcher: Callable[..., int] = launch_formal_train,
) -> list[dict[str, Any]]:
    """Run one model launcher at a time; the next starts only after exit."""
    results: list[dict[str, Any]] = []
    for request in requests:
        model_id = str(request["model_id"])
        code = int(launcher(**dict(request)))
        results.append({"model_id": model_id, "exit_code": code})
        if code != 0:
            break
    return results
