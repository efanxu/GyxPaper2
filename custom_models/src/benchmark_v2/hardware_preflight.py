from __future__ import annotations

import json
import os
import platform
import re
import socket
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .artifacts import atomic_write_json
from .contracts import BenchmarkBatch
from .errors import ModelUnavailableError
from .experiments.e5_common_loss.loss_profile import CLI_PROFILE_ID, DEFAULT_PROFILE_ID, loss_for_profile
from .experiments.e5_common_loss.runner import apply_experiment_profile, normalize_profile
from .experiments.e5_common_loss.scope27_contract import CURRENT_SCOPE26_ID, E5_SCOPE27_ID, is_current_scope26_train_request, is_scope27_train_request, validate_scope27_request
from .model_runtime import build_model_runtime
from .precision import apply_model_precision_policy, expected_model_precision_identity
from .protocol import load_protocol
from .registry import load_registry
from .runtime import PROJECT_ROOT, SMOKE_ROOT
from .seeds import seed_everything
from .training_profiles import apply_training_profile, load_training_profile


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PREFLIGHT_SHAPE = {"B": 32, "T": 144, "N": 134, "C": 16, "H": 10}
PREFLIGHT_ROOT = SMOKE_ROOT / "hardware_preflight"
RUNNER_PATH = PROJECT_ROOT / "custom_models/src/benchmark_v2/run_benchmark.py"


def machine_identity() -> dict[str, Any]:
    import torch
    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = str(torch.cuda.get_device_properties(0).name)
    return {
        "hostname": socket.gethostname(),
        "operating_system": f"{platform.system()} {platform.release()}".strip(),
        "python_version": platform.python_version(),
        "pytorch_version": str(torch.__version__),
        "cuda_version": None if torch.version.cuda is None else str(torch.version.cuda),
        "gpu_name": gpu_name,
    }


def preflight_shape(training_profile: str | None = None, formal_scope_id: str | None = None) -> dict[str, int]:
    shape = dict(PREFLIGHT_SHAPE)
    if formal_scope_id in {CURRENT_SCOPE26_ID, E5_SCOPE27_ID} and training_profile is None:
        training_profile = "uniform_train_batch4_v1"
    profile = load_training_profile(training_profile)
    if profile is not None:
        shape["B"] = profile.train_batch_size
    if formal_scope_id in {CURRENT_SCOPE26_ID, E5_SCOPE27_ID} and shape["B"] != 4:
        raise ValueError("Formal Original/E5 preflight requires Batch4.")
    return shape


def preflight_identity(
    model_id: str, *, experiment_profile: str | None = None,
    training_profile: str | None = None, formal_scope_id: str | None = None,
    source_revision: str | None = None, manifest: Mapping[str, Any] | None = None,
    run_map: Mapping[str, Any] | None = None, freeze: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    del source_revision, run_map, freeze
    protocol = load_protocol()
    entry = load_registry().get(model_id)
    shape = preflight_shape(training_profile, formal_scope_id)
    precision = expected_model_precision_identity(entry.canonical_id, training_profile)
    run_id = None
    if manifest:
        for row in manifest.get("entries", []):
            if row.get("model_id") == entry.canonical_id:
                run_id = row.get("run_id") or row.get("e5_run_id")
                break
    return {
        **machine_identity(),
        "scope_id": formal_scope_id,
        "model_id": entry.canonical_id,
        "run_id": run_id,
        "batch_size": shape["B"],
        "lookback": shape["T"],
        "node_count": shape["N"],
        "feature_count": shape["C"],
        "horizon": shape["H"],
        "precision": precision.get("precision_policy"),
        "amp_enabled": bool(precision.get("amp_enabled", protocol["amp_enabled"])),
        "loss_id": "masked_score_aligned_hybrid" if normalize_profile(experiment_profile) != DEFAULT_PROFILE_ID else "masked_mse",
        "training_profile_id": training_profile,
    }


def _model_root(model_id: str, root: str | Path | None) -> Path:
    return Path(root or PREFLIGHT_ROOT).resolve() / str(model_id)


def preflight_artifact_path(model_id: str, *, root: str | Path | None = None, **_: Any) -> Path:
    return _model_root(model_id, root) / "latest.json"


def new_preflight_attempt_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex


def _attempt_result_path(model_id: str, root: str | Path | None, attempt_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", attempt_id):
        raise ValueError("Invalid preflight attempt id.")
    return _model_root(model_id, root) / "attempts" / attempt_id / "result.json"


def read_matching_pass(
    model_id: str, *, root: str | Path | None = None,
    experiment_profile: str | None = None, training_profile: str | None = None,
    formal_scope_id: str | None = None, source_revision: str | None = None,
    attempt_id: str | None = None,
) -> dict[str, Any] | None:
    del source_revision
    path = _attempt_result_path(model_id, root, attempt_id) if attempt_id else preflight_artifact_path(model_id, root=root)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    expected = preflight_identity(
        model_id, experiment_profile=experiment_profile,
        training_profile=training_profile, formal_scope_id=formal_scope_id,
    )
    required = {
        "status": "PASS", "model_id": expected["model_id"],
        "scope_id": expected["scope_id"], "batch_size": expected["batch_size"],
        "precision": expected["precision"], "forward_pass": True,
        "backward_pass": True, "finite": True,
    }
    if any(payload.get(key) != value for key, value in required.items()):
        return None
    if payload.get("output_shape") != [expected["batch_size"], expected["node_count"], expected["horizon"]]:
        return None
    return {**payload, "artifact_path": str(path), "preflight_attempt_id": payload.get("attempt_id")}


def build_preflight_child_contract(model_id: str, *, exit_code: int, root: str | Path | None = None, attempt_id: str | None = None, **kwargs: Any) -> dict[str, Any]:
    kwargs.pop("log_path", None)
    kwargs.pop("expected_identity", None)
    payload = read_matching_pass(model_id, root=root, attempt_id=attempt_id, **kwargs)
    return {
        "model_id": model_id, "child_exit_code": int(exit_code),
        "artifact_written": payload is not None,
        "artifact_status": None if payload is None else payload.get("status"),
        "matched_explicit_metadata": payload is not None,
        "attempt_id": attempt_id,
    }


def _synthetic_batch(model_id: str, *, training_profile: str | None = None, formal_scope_id: str | None = None) -> BenchmarkBatch:
    import torch
    shape = preflight_shape(training_profile, formal_scope_id)
    generator = torch.Generator().manual_seed(2026)
    return BenchmarkBatch(
        x=torch.randn(shape["B"], shape["T"], shape["N"], shape["C"], generator=generator),
        target=torch.randn(shape["B"], shape["N"], shape["H"], generator=generator),
        mask=torch.ones(shape["B"], shape["N"], shape["H"], dtype=torch.bool),
        sample_ids=list(range(shape["B"])), window_end_indices=list(range(shape["B"])),
        node_ids=list(range(1, shape["N"] + 1)),
        metadata={"contains_future_target": False, "hardware_preflight": True,
                  "history_end_timestamp": ["2021-01-04 12:00:00"] * shape["B"],
                  "history_end_time_of_day_id": [72] * shape["B"],
                  "history_end_day_of_week_id": [0] * shape["B"]},
    )


def run_preflight_worker(
    model_id: str, *, root: str | Path | None = None,
    experiment_profile: str | None = None, training_profile: str | None = None,
    formal_scope_id: str | None = None, source_revision: str | None = None,
    manifest: Mapping[str, Any] | None = None, run_map: Mapping[str, Any] | None = None,
    freeze: Mapping[str, Any] | None = None, attempt_id: str | None = None,
) -> dict[str, Any]:
    del source_revision, run_map, freeze
    import torch
    selected = attempt_id or new_preflight_attempt_id()
    identity = preflight_identity(model_id, experiment_profile=experiment_profile, training_profile=training_profile, formal_scope_id=formal_scope_id, manifest=manifest)
    path = _attempt_result_path(model_id, root, selected)
    result = {
        "status": "FAIL", **identity, "attempt_id": selected,
        "device": "cuda", "forward_pass": False, "backward_pass": False,
        "finite": False, "output_shape": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for formal hardware preflight.")
        protocol = load_protocol()
        batch = _synthetic_batch(model_id, training_profile=training_profile, formal_scope_id=formal_scope_id)
        runtime = build_model_runtime(model_id, protocol, run_mode="formal")
        apply_experiment_profile(runtime, normalize_profile(experiment_profile))
        apply_training_profile(runtime, training_profile)
        apply_model_precision_policy(runtime)
        device = torch.device("cuda")
        runtime.model.to(device)
        moved = BenchmarkBatch(x=batch.x.to(device), target=batch.target.to(device), mask=batch.mask.to(device), sample_ids=batch.sample_ids, window_end_indices=batch.window_end_indices, node_ids=batch.node_ids, metadata=batch.metadata)
        output = runtime.adapter(runtime.model, moved, expected_horizon=identity["horizon"], expected_features=identity["feature_count"])
        result["forward_pass"] = True
        result["output_shape"] = list(output.prediction.shape)
        finite_output = bool(torch.isfinite(output.prediction).all().item())
        loss = loss_for_profile(normalize_profile(experiment_profile))(output.prediction, moved.target, moved.mask)
        finite_loss = bool(torch.isfinite(loss).item())
        loss.backward()
        finite_gradients = all(torch.isfinite(parameter.grad).all().item() for parameter in runtime.model.parameters() if parameter.grad is not None)
        result["backward_pass"] = True
        result["finite"] = finite_output and finite_loss and finite_gradients
        expected_shape = [identity["batch_size"], identity["node_count"], identity["horizon"]]
        if result["output_shape"] != expected_shape or not result["finite"]:
            raise RuntimeError("Preflight output validation failed.")
        result["status"] = "PASS"
    except Exception as exc:
        result.update({"error_type": type(exc).__name__, "error_message": str(exc)})
    atomic_write_json(path, result)
    atomic_write_json(preflight_artifact_path(model_id, root=root), result)
    result["artifact_path"] = str(path)
    return result


def _run_child(args: Sequence[str], *, runner: Callable[..., Any] = subprocess.run) -> int:
    command = [str(Path(sys.executable).resolve()), str(RUNNER_PATH), *args]
    if runner is subprocess.run:
        source_root = PROJECT_ROOT / "custom_models" / "src"
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(PROJECT_ROOT), str(source_root), os.environ.get("PYTHONPATH", "")])}
        completed = runner(command, check=False, cwd=str(PROJECT_ROOT), env=env)
    else:
        completed = runner(command, check=False)
    return int(completed.returncode)


def launch_preflight(model_id: str, *, root: str | Path | None = None, experiment_profile: str | None = None, training_profile: str | None = None, formal_scope_id: str | None = None, source_revision: str | None = None, attempt_id: str | None = None, runner: Callable[..., Any] = subprocess.run) -> int:
    args = ["_hardware-preflight-worker", "--model", model_id]
    for flag, value in (("--experiment-profile", experiment_profile), ("--training-profile", training_profile), ("--formal-scope-id", formal_scope_id), ("--source-revision", source_revision), ("--preflight-root", root), ("--preflight-attempt-id", attempt_id)):
        if value is not None:
            args.extend([flag, str(value)])
    return _run_child(args, runner=runner)


def launch_formal_train(
    model_id: str, *, input_path: str | Path, target_path: str | Path,
    output_root: str | Path, run_id: str, device: str,
    preflight_root: str | Path | None = None, experiment_profile: str | None = None,
    training_profile: str | None = None, formal_scope_id: str | None = None,
    source_revision: str | None = None, preflight_attempt_id: str | None = None,
    runner: Callable[..., Any] = subprocess.run,
) -> int:
    entry = load_registry().get(model_id)
    if bool(entry.supports_non_trainable) or not entry.supports_train:
        raise ModelUnavailableError(f"{entry.display_name} is unavailable for training.")
    selected_profile = normalize_profile(experiment_profile)
    validate_scope27_request(model_id=entry.canonical_id, formal_scope_id=formal_scope_id, experiment_profile=selected_profile, training_profile=training_profile, trainable=True)
    exact_scope = is_scope27_train_request(model_id=entry.canonical_id, formal_scope_id=formal_scope_id, experiment_profile=selected_profile, training_profile=training_profile) or is_current_scope26_train_request(model_id=entry.canonical_id, formal_scope_id=formal_scope_id, experiment_profile=selected_profile, training_profile=training_profile)
    existing_pass = read_matching_pass(entry.canonical_id, root=preflight_root, experiment_profile=selected_profile, training_profile=training_profile, formal_scope_id=formal_scope_id, attempt_id=preflight_attempt_id)
    if exact_scope and existing_pass is None:
        return 3
    needs_preflight = not exact_scope and (
        training_profile is not None
        or selected_profile == CLI_PROFILE_ID
        or bool(entry.values.get("formal_hardware_preflight_required", False))
    )
    if needs_preflight and existing_pass is None:
        code = launch_preflight(
            entry.canonical_id,
            root=preflight_root,
            experiment_profile=selected_profile,
            training_profile=training_profile,
            attempt_id=preflight_attempt_id,
            runner=runner,
        )
        if code != 0:
            return code
        existing_pass = read_matching_pass(
            entry.canonical_id,
            root=preflight_root,
            experiment_profile=selected_profile,
            training_profile=training_profile,
            attempt_id=preflight_attempt_id,
        )
        if existing_pass is None:
            return 3
    args = ["_formal-train-worker", "--model", entry.canonical_id, "--input-path", str(input_path), "--target-path", str(target_path), "--output-root", str(output_root), "--run-id", run_id, "--device", device]
    for flag, value in (("--experiment-profile", experiment_profile), ("--training-profile", training_profile), ("--formal-scope-id", formal_scope_id), ("--source-revision", source_revision), ("--preflight-root", preflight_root), ("--preflight-attempt-id", preflight_attempt_id)):
        if value is not None:
            args.extend([flag, str(value)])
    return _run_child(args, runner=runner)


def launch_model_suite(requests: Sequence[Mapping[str, Any]], *, launcher: Callable[..., int] = launch_formal_train) -> list[dict[str, Any]]:
    results = []
    for request in requests:
        model_id = str(request["model_id"])
        try:
            results.append({"model_id": model_id, "exit_code": int(launcher(**dict(request)))})
        except Exception as exc:
            results.append({"model_id": model_id, "exit_code": 1, "error_type": type(exc).__name__, "error_message": str(exc)})
    return results
