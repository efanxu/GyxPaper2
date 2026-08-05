from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .artifacts import atomic_write_json
from .contracts import BenchmarkBatch
from .errors import ModelUnavailableError
from .experiments.e5_common_loss.loss_profile import (
    CLI_PROFILE_ID as E5_PROFILE_ID,
    DEFAULT_PROFILE_ID,
    get_profile_metadata,
    loss_for_profile,
)
from .experiments.e5_common_loss.runner import (
    apply_experiment_profile,
    normalize_profile,
)
from .experiments.e5_common_loss.scope27_contract import (
    CURRENT_SCOPE26_ID,
    E5_SCOPE27_ID,
    is_current_scope26_train_request,
    is_scope27_train_request,
    validate_scope27_request,
)
from .experiments.e5_common_loss.active_scope import load_active_scope_pointer
from .losses import get_loss
from .model_runtime import build_model_runtime
from .model_factories.registry import get_model_factory
from .model_source_identity import canonical_model_source_identity
from .precision import (
    apply_model_precision_policy,
    expected_model_precision_identity,
)
from .protocol import load_protocol
from .registry import load_registry
from .graph import load_graph_bundle
from .graph.source_doctor import checkout_identity as graph_checkout_identity
from .models.graph_models.common import MODEL_SUPPORTS, graph_identity_dict
from .models.graph_models.adaptive_common import e3_c_graph_identity
from .runtime import PROJECT_ROOT, SMOKE_ROOT
from .seeds import seed_everything
from .training_profiles import (
    apply_training_profile,
    batch_identity,
    load_training_profile,
)


# Worker processes are launched from a file path, so Python does not
# necessarily add the repository root to sys.path.  Formal Original/E5
# identities import their gate modules from ``scripts``; make that package
# location deterministic in every platform and launcher entrypoint.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


PREFLIGHT_SHAPE = {"B": 32, "T": 144, "N": 134, "C": 16, "H": 10}
PREFLIGHT_ROOT = SMOKE_ROOT / "hardware_preflight"
RUNNER_PATH = PROJECT_ROOT / "custom_models/src/benchmark_v2/run_benchmark.py"
E3_C_MODELS = {"graph_wavenet", "mtgnn", "agcrn", "stid"}
NATIVE_NODE_MODELS = set(MODEL_SUPPORTS) | E3_C_MODELS
PREFLIGHT_LATEST_SCHEMA = "hardware_preflight_latest_v1"
PREFLIGHT_ATTEMPT_SCHEMA = "hardware_preflight_attempt_v1"


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


def preflight_shape(
    training_profile: str | None = None,
    formal_scope_id: str | None = None,
) -> dict[str, int]:
    profile = load_training_profile(training_profile)
    shape = dict(PREFLIGHT_SHAPE)
    if profile is not None:
        shape["B"] = profile.train_batch_size
    if formal_scope_id == CURRENT_SCOPE26_ID:
        if training_profile != "uniform_train_batch4_v1":
            raise ValueError(
                "Current Original scope26 exact preflight requires "
                "uniform_train_batch4_v1."
            )
        shape["B"] = 4
    if formal_scope_id == E5_SCOPE27_ID:
        if training_profile != "uniform_train_batch4_v1":
            raise ValueError(
                "Current E5 scope27 exact preflight requires "
                "uniform_train_batch4_v1."
            )
        shape["B"] = 4
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
    formal_scope_id: str | None = None,
    source_revision: str | None = None,
    manifest: Mapping[str, Any] | None = None,
    run_map: Mapping[str, Any] | None = None,
    freeze: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    protocol = load_protocol()
    entry = load_registry().get(model_id)
    factory = get_model_factory(entry.canonical_id)
    config = factory.resolve_config(protocol, run_mode="formal")
    source_identity = canonical_model_source_identity(entry.canonical_id)
    source_hash = source_identity["canonical_combined_hash"]
    identity = {
        **machine_identity(),
        "model_id": entry.canonical_id,
        "model_config_hash": stable_hash(config),
        "protocol_hash": protocol.protocol_hash,
        "source_hash": source_hash,
        "source_identity_schema_version": source_identity[
            "source_identity_schema_version"
        ],
        "source_closure_files": source_identity["source_closure_files"],
        **preflight_shape(training_profile, formal_scope_id),
        "amp": bool(protocol["amp_enabled"]),
    }
    graph_bundle = None
    if entry.canonical_id in MODEL_SUPPORTS:
        graph_bundle = load_graph_bundle()
        identity["benchmark_protocol_hash"] = protocol.protocol_hash
        identity.update(
            graph_identity_dict(
                graph_bundle, MODEL_SUPPORTS[entry.canonical_id]
            )
        )
    elif entry.canonical_id in E3_C_MODELS:
        graph_bundle = load_graph_bundle()
        identity["benchmark_protocol_hash"] = protocol.protocol_hash
        identity["source_closure_hash"] = source_hash
        identity.update(
            e3_c_graph_identity(graph_bundle, entry.canonical_id)
        )
    selected_profile = normalize_profile(experiment_profile)
    if selected_profile == E5_PROFILE_ID:
        profile = get_profile_metadata(selected_profile)
        identity.update(
            {
                "base_model_config_hash": identity["model_config_hash"],
                "model_source_closure_hash": source_hash,
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
        if graph_bundle is None:
            graph_bundle = load_graph_bundle()
        identity.update(
            {
                "base_benchmark_protocol_hash": protocol.protocol_hash,
                "model_source_closure_hash": source_hash,
                "graph_protocol_hash": graph_bundle.graph_protocol_hash,
                "graph_bundle_hash": graph_bundle.graph_bundle_hash,
                "node_order_hash": graph_bundle.node_order_hash,
                "location_source_hash": graph_bundle.location_source_hash,
                "checkout_identity": graph_checkout_identity(PROJECT_ROOT),
                "timestamp_identity_policy": "SOURCE_NAIVE_UNCHANGED",
                "seed": int(protocol["default_seed"]),
                **_loss_identity(selected_profile),
                **batch_identity(training_profile),
            }
        )
    identity.update(
        expected_model_precision_identity(
            entry.canonical_id, training_profile
        )
    )
    if formal_scope_id == CURRENT_SCOPE26_ID:
        from .original_scope26 import (
            current_manifest_hash,
            current_model_config_hash,
            load_current_scope_manifest,
            resolve_source_revision,
        )

        active = dict(manifest or load_current_scope_manifest())
        selected_map = dict(run_map or json.loads(
            (PROJECT_ROOT / "custom_models/docs/benchmark_v2/BATCH4/ORIGINAL_SCOPE26_RUN_ID_MAP.json")
            .read_text(encoding="utf-8")
        ))
        revision = resolve_source_revision(
            project_root=PROJECT_ROOT,
            explicit_source_revision=source_revision,
            manifest=active,
        )
        if freeze is None:
            from scripts.original_batch4_scope26_gate import compute_original_freeze

            freeze = compute_original_freeze(
                active,
                project_root=PROJECT_ROOT,
                run_map=selected_map,
                source_revision=source_revision,
            )
        precision = expected_model_precision_identity(
            entry.canonical_id, training_profile
        )
        identity.update(
            {
                "scope_id": formal_scope_id,
                "git_commit": revision["git_commit"],
                "source_revision_type": revision["source_revision_type"],
                "manifest_hash": stable_hash(active),
                "run_map_hash": stable_hash(selected_map),
                "freeze_hash": freeze["freeze_hash"],
                "freeze_schema_version": freeze.get("schema_version"),
                "source_closure_hash": source_hash,
                "model_config_hash": current_model_config_hash(entry.canonical_id),
                "precision_identity": precision,
                "precision_identity_hash": stable_hash(precision),
                "protocol_hash": protocol.protocol_hash,
                "training_profile_id": training_profile,
                "training_profile_hash": (
                    load_training_profile(training_profile).profile_hash
                    if load_training_profile(training_profile) is not None
                    else None
                ),
                "batch4_profile_id": training_profile,
                "batch4_profile_hash": (
                    load_training_profile(training_profile).profile_hash
                    if load_training_profile(training_profile) is not None
                    else None
                ),
                "dataset_identity_hash": stable_hash(active["dataset_identity"]),
                "graph_identity_hash": stable_hash(active["graph_identity"]),
                "exact_shape": dict(preflight_shape(training_profile, formal_scope_id)),
                "precision_identity_name": precision.get("precision_policy"),
                "effective_amp_enabled": precision.get("amp_enabled"),
            }
        )
    if formal_scope_id == E5_SCOPE27_ID:
        from scripts.e5_batch4_scope27_gate import (
            DEFAULT_MANIFEST,
            _load_run_map,
            compute_e5_freeze,
            load_manifest,
        )

        active = dict(manifest or load_manifest(DEFAULT_MANIFEST))
        selected_map = dict(run_map or _load_run_map())
        if freeze is None:
            freeze = compute_e5_freeze(active, run_map=selected_map)
        revision = {
            "git_commit": str(source_revision or freeze["git_commit"]),
            "source_revision_type": "git",
        }
        precision = expected_model_precision_identity(
            entry.canonical_id, training_profile
        )
        entry_manifest = next(
            row for row in active["entries"] if row["model_id"] == entry.canonical_id
        )
        identity.update(
            {
                "scope_id": formal_scope_id,
                "active_scope_id": formal_scope_id,
                "git_commit": revision["git_commit"],
                "source_revision_type": revision["source_revision_type"],
                "active_pointer_hash": stable_hash(load_active_scope_pointer()),
                "manifest_hash": stable_hash(active),
                "run_map_hash": stable_hash(selected_map),
                "freeze_hash": freeze["freeze_hash"],
                "freeze_schema_version": freeze.get("schema_version"),
                "model_config_hash": entry_manifest["base_model_config_hash"],
                "base_model_config_hash": entry_manifest["base_model_config_hash"],
                "source_hash": entry_manifest["base_model_source_hash"],
                "source_closure_hash": entry_manifest["base_model_source_hash"],
                "model_source_closure_hash": entry_manifest[
                    "base_model_source_hash"
                ],
                "precision_identity": precision,
                "precision_identity_hash": stable_hash(precision),
                "protocol_hash": active["benchmark_protocol_hash"],
                "training_profile_id": training_profile,
                "training_profile_hash": active["training_profile_hash"],
                "batch4_profile_id": training_profile,
                "batch4_profile_hash": active["training_profile_hash"],
                "dataset_identity_hash": stable_hash(active["dataset_identity"]),
                "graph_identity_hash": stable_hash(active["graph_identity"]),
                "loss_identity_hash": stable_hash(active["loss_identity"]),
                "loss_id": active["loss_identity"]["loss_id"],
                "loss_source_hash": active["loss_identity"]["loss_source_hash"],
                "loss_profile_hash": active["loss_identity"]["loss_profile_hash"],
                "e5_common_loss_protocol_hash": active["loss_identity"][
                    "e5_common_loss_protocol_hash"
                ],
                "experiment_profile_id": active["experiment_profile_id"],
                "training_profile_id": active["training_profile_id"],
                "exact_shape": dict(preflight_shape(training_profile, formal_scope_id)),
                "precision_identity_name": precision.get("precision_policy"),
                "effective_amp_enabled": precision.get("amp_enabled"),
            }
        )
    return identity


def _preflight_model_root(model_id: str, root: str | Path | None) -> Path:
    return Path(root or PREFLIGHT_ROOT).resolve() / str(model_id)


def _new_attempt_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{timestamp}-{uuid.uuid4().hex}"


def new_preflight_attempt_id() -> str:
    """Create an attempt id that is safe to pass through every launcher."""

    return _new_attempt_id()


def _validated_attempt_id(value: str) -> str:
    token = str(value)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", token):
        raise ValueError(f"Invalid preflight attempt id: {value!r}")
    return token


def _attempt_result_path(model_id: str, root: str | Path | None, attempt_id: str) -> Path:
    return (
        _preflight_model_root(model_id, root)
        / "attempts"
        / _validated_attempt_id(attempt_id)
        / "result.json"
    )


def _attempt_request_path(model_id: str, root: str | Path | None, attempt_id: str) -> Path:
    return _attempt_result_path(model_id, root, attempt_id).with_name("request.json")


def _latest_path(model_id: str, root: str | Path | None) -> Path:
    return _preflight_model_root(model_id, root) / "latest.json"


def _artifact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preflight_artifact_path(
    model_id: str,
    *,
    root: str | Path | None = None,
    experiment_profile: str | None = None,
    training_profile: str | None = None,
    formal_scope_id: str | None = None,
    source_revision: str | None = None,
    manifest: Mapping[str, Any] | None = None,
    run_map: Mapping[str, Any] | None = None,
    freeze: Mapping[str, Any] | None = None,
) -> Path:
    """Return the attempt-aware ``latest.json`` compatibility path.

    New workers write immutable ``attempts/<id>/result.json`` records and an
    atomic latest pointer.  Keeping this function as the latest path lets
    existing diagnostic/unit-test fixtures create an explicit latest PASS
    without writing into an attempt selected by production code.
    """

    del experiment_profile, training_profile, formal_scope_id
    del source_revision, manifest, run_map, freeze
    return _latest_path(model_id, root)


def _load_latest_result(
    model_id: str,
    *,
    root: str | Path | None,
) -> tuple[dict[str, Any] | None, Path | None, dict[str, Any] | None]:
    latest_path = _latest_path(model_id, root)
    if not latest_path.is_file():
        return None, None, None
    try:
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None, latest_path, None
    if not isinstance(latest, dict):
        return None, latest_path, None
    if latest.get("schema_version") != PREFLIGHT_LATEST_SCHEMA:
        return latest, latest_path, {"result_sha256": _artifact_sha256(latest_path)}
    attempt_id = latest.get("attempt_id")
    if not isinstance(attempt_id, str):
        return None, latest_path, latest
    result_path = _attempt_result_path(model_id, root, attempt_id)
    if not result_path.is_file():
        return None, result_path, latest
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None, result_path, latest
    if not isinstance(payload, dict):
        return None, result_path, latest
    expected_sha = latest.get("result_sha256")
    if expected_sha and expected_sha != _artifact_sha256(result_path):
        return None, result_path, latest
    return payload, result_path, latest


def _identity_mismatch_fields(
    expected: Mapping[str, Any], actual: Mapping[str, Any]
) -> list[str]:
    return sorted(
        key for key, value in expected.items() if actual.get(key) != value
    )


def read_preflight_attempt(
    model_id: str,
    *,
    root: str | Path | None,
    attempt_id: str,
) -> tuple[dict[str, Any] | None, Path | None]:
    path = _attempt_result_path(model_id, root, attempt_id)
    if not path.is_file():
        return None, path
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None, path
    return (payload if isinstance(payload, dict) else None), path


def read_matching_pass(
    model_id: str,
    *,
    root: str | Path | None = None,
    experiment_profile: str | None = None,
    training_profile: str | None = None,
    formal_scope_id: str | None = None,
    source_revision: str | None = None,
    manifest: Mapping[str, Any] | None = None,
    run_map: Mapping[str, Any] | None = None,
    freeze: Mapping[str, Any] | None = None,
    attempt_id: str | None = None,
    artifact_sha256: str | None = None,
) -> dict[str, Any] | None:
    if attempt_id is not None:
        payload, path = read_preflight_attempt(
            model_id,
            root=root,
            attempt_id=attempt_id,
        )
        pointer = None
        latest_payload, _, latest_pointer = _load_latest_result(model_id, root=root)
        if latest_pointer and latest_pointer.get("attempt_id") == attempt_id:
            pointer = latest_pointer
        del latest_payload
    else:
        payload, path, pointer = _load_latest_result(model_id, root=root)
    if payload is None or path is None:
        return None
    if attempt_id is not None and payload.get("attempt_id") not in {None, attempt_id}:
        return None
    if artifact_sha256 is not None:
        actual_artifact_sha = (
            pointer.get("result_sha256")
            if pointer and pointer.get("schema_version") == PREFLIGHT_LATEST_SCHEMA
            else _artifact_sha256(path)
        )
        if actual_artifact_sha != artifact_sha256:
            return None
    expected = preflight_identity(
        model_id,
        experiment_profile=experiment_profile,
        training_profile=training_profile,
        formal_scope_id=formal_scope_id,
        source_revision=source_revision,
        manifest=manifest,
        run_map=run_map,
        freeze=freeze,
    )
    if _identity_mismatch_fields(expected, payload):
        return None
    if (
        payload.get("status") != "PASS"
        or payload.get("forward_completed") is not True
        or payload.get("backward_completed") is not True
    ):
        return None
    if formal_scope_id in {CURRENT_SCOPE26_ID, E5_SCOPE27_ID}:
        required_flags = (
            "finite_prediction",
            "finite_loss",
            "finite_gradients",
        )
        if any(payload.get(key) is not True for key in required_flags):
            return None
        if payload.get("prediction_shape") != [
            expected["B"],
            expected["N"],
            expected["H"],
        ]:
            return None
        for key in ("peak_allocated_memory", "peak_reserved_memory"):
            value = payload.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value < 0
            ):
                return None
    result = dict(payload)
    if pointer and pointer.get("schema_version") == PREFLIGHT_LATEST_SCHEMA:
        result["preflight_attempt_id"] = pointer.get("attempt_id")
        result["preflight_artifact_sha256"] = pointer.get("result_sha256")
    return result


def _preflight_log_lines(log_path: str | Path | None) -> list[str]:
    if log_path is None:
        return []
    path = Path(log_path)
    if not path.is_file():
        return []
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []


def _preflight_exception(lines: list[str]) -> tuple[str | None, str | None]:
    pattern = re.compile(
        r"(?P<type>[A-Za-z_][\w.]*(?:Error|Exception|ContractError))(?::\s*(?P<message>.*))?"
    )
    for line in reversed(lines):
        match = pattern.search(line)
        if match:
            return (
                match.group("type").split(".")[-1],
                (match.group("message") or "").strip() or None,
            )
    return None, None


def build_preflight_child_contract(
    model_id: str,
    *,
    exit_code: int,
    log_path: str | Path | None,
    root: str | Path | None,
    attempt_id: str | None = None,
    expected_identity: Mapping[str, Any] | None = None,
    attempt_payload: Mapping[str, Any] | None = None,
    artifact_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the shared, failure-first child result contract for both gates."""

    payload = dict(attempt_payload or {})
    path = Path(artifact_path) if artifact_path is not None else None
    if attempt_payload is None and attempt_id is not None:
        loaded, loaded_path = read_preflight_attempt(
            model_id,
            root=root,
            attempt_id=attempt_id,
        )
        payload = dict(loaded or {})
        path = loaded_path
    lines = _preflight_log_lines(log_path)
    traceback_text = payload.get("traceback")
    if isinstance(traceback_text, str) and traceback_text:
        traceback_tail = traceback_text.splitlines()[-80:]
    else:
        traceback_tail = lines[-80:]
    exception_type = payload.get("error_type")
    exception_message = payload.get("error_message")
    parsed_type, parsed_message = _preflight_exception(traceback_tail)
    exception_type = str(exception_type or parsed_type) if (exception_type or parsed_type) else None
    exception_message = str(exception_message or parsed_message) if (exception_message or parsed_message) else None
    expected = dict(expected_identity or {})
    actual = {key: payload.get(key) for key in expected}
    artifact_written = bool(path is not None and path.is_file())
    artifact_status = payload.get("status") if artifact_written else "MISSING"
    identity_material_present = bool(
        expected
        and any(
            key in payload
            for key in expected
            if key not in {"model_id", "attempt_id"}
        )
    )
    identity_mismatch_fields = (
        _identity_mismatch_fields(expected, payload)
        if expected and artifact_written and identity_material_present
        else []
    )
    text = "\n".join(traceback_tail).casefold()
    raw_text = f"{exception_type or ''} {exception_message or ''} {text}".casefold()
    if identity_mismatch_fields:
        if any("location_source" in field for field in identity_mismatch_fields):
            category = "LOCATION_SOURCE_HASH_MISMATCH"
        elif any("graph_protocol" in field for field in identity_mismatch_fields):
            category = "GRAPH_PROTOCOL_HASH_MISMATCH"
        elif any("graph_bundle" in field for field in identity_mismatch_fields):
            category = "GRAPH_BUNDLE_HASH_MISMATCH"
        elif any("matrix" in field for field in identity_mismatch_fields):
            category = "MATRIX_HASH_MISMATCH"
        elif any("node_order" in field for field in identity_mismatch_fields):
            category = "NODE_ORDER_HASH_MISMATCH"
        else:
            category = "PREFLIGHT_ARTIFACT_IDENTITY_MISMATCH"
    elif "out of memory" in raw_text or "outofmemory" in raw_text:
        category = "OOM"
    elif "cuda" in raw_text:
        category = "CUDA_FAILURE"
    elif re.search(r"nan|inf|nonfinite", raw_text):
        category = "NONFINITE"
    elif not artifact_written and exception_type:
        category = "CHILD_EXCEPTION"
    elif artifact_status not in {None, "PASS"}:
        category = "CHILD_EXCEPTION"
    elif not artifact_written:
        category = "CHILD_FAILED_BEFORE_ARTIFACT"
    else:
        category = None
    return {
        "error_type": category,
        "error_message": exception_message or (f"child exit code {exit_code}" if int(exit_code) != 0 else None),
        "exception_type": exception_type,
        "failure_stage": payload.get("failure_stage") or ("child_process" if not artifact_written else None),
        "traceback_tail": traceback_tail,
        "child_log_path": str(log_path) if log_path is not None else None,
        "child_artifact_path": str(path) if path is not None and artifact_written else None,
        "child_exit_code": int(exit_code),
        "artifact_written": artifact_written,
        "artifact_status": artifact_status,
        "identity_mismatch_fields": identity_mismatch_fields,
        "expected_identity": expected,
        "actual_identity": actual,
        "matched_exact_identity": bool(
            artifact_written
            and artifact_status == "PASS"
            and not identity_mismatch_fields
        ),
        "attempt_id": payload.get("attempt_id") or attempt_id,
    }


def _synthetic_batch(
    model_id: str,
    *,
    training_profile: str | None = None,
    formal_scope_id: str | None = None,
) -> BenchmarkBatch:
    import torch

    generator = torch.Generator().manual_seed(2026)
    shape = preflight_shape(training_profile, formal_scope_id)
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


def _publish_latest(
    model_id: str,
    *,
    root: str | Path | None,
    attempt_id: str,
    result: Mapping[str, Any],
    result_path: Path,
) -> dict[str, Any]:
    current, _, current_pointer = _load_latest_result(model_id, root=root)
    preserve_current_pass = bool(
        current
        and current.get("status") == "PASS"
        and current.get("identity_hash") == result.get("identity_hash")
    )
    pointer = {
        "schema_version": PREFLIGHT_LATEST_SCHEMA,
        "model_id": model_id,
        "attempt_id": attempt_id,
        "result_path": str(result_path.relative_to(_preflight_model_root(model_id, root))).replace("\\", "/"),
        "result_sha256": _artifact_sha256(result_path),
        "identity_hash": result.get("identity_hash"),
        "status": result.get("status"),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    latest_path = _latest_path(model_id, root)
    if preserve_current_pass and latest_path.is_file():
        preserved = dict(current_pointer or {})
        preserved.setdefault("attempt_id", current.get("attempt_id"))
        if not preserved.get("result_sha256") and isinstance(
            preserved.get("attempt_id"), str
        ):
            preserved["result_sha256"] = _artifact_sha256(
                _attempt_result_path(
                    model_id, root, str(preserved["attempt_id"])
                )
            )
        preserved["preserved"] = True
        return preserved
    atomic_write_json(latest_path, pointer)
    return pointer


def run_preflight_worker(
    model_id: str,
    *,
    root: str | Path | None = None,
    experiment_profile: str | None = None,
    training_profile: str | None = None,
    formal_scope_id: str | None = None,
    source_revision: str | None = None,
    manifest: Mapping[str, Any] | None = None,
    run_map: Mapping[str, Any] | None = None,
    freeze: Mapping[str, Any] | None = None,
    attempt_id: str | None = None,
) -> dict[str, Any]:
    selected_profile = normalize_profile(experiment_profile)
    selected_attempt = _validated_attempt_id(attempt_id or _new_attempt_id())
    attempt_root = _preflight_model_root(model_id, root) / "attempts" / selected_attempt
    result_path = attempt_root / "result.json"
    atomic_write_json(
        attempt_root / "request.json",
        {
            "schema_version": PREFLIGHT_ATTEMPT_SCHEMA,
            "model_id": model_id,
            "attempt_id": selected_attempt,
            "experiment_profile": selected_profile,
            "training_profile": training_profile,
            "formal_scope_id": formal_scope_id,
            "source_revision": source_revision,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    torch = None
    protocol = None
    identity: dict[str, Any] | None = None
    result: dict[str, Any] = {
        "model_id": model_id,
        "attempt_id": selected_attempt,
        "preflight_pid": os.getpid(),
        "forward_completed": False,
        "backward_completed": False,
        "finite_prediction": False,
        "finite_loss": False,
        "finite_gradients": False,
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
        "failure_stage": "identity",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        import torch as torch_module

        torch = torch_module
        protocol = load_protocol()
        identity = preflight_identity(
            model_id,
            experiment_profile=selected_profile,
            training_profile=training_profile,
            formal_scope_id=formal_scope_id,
            source_revision=source_revision,
            manifest=manifest,
            run_map=run_map,
            freeze=freeze,
        )
        result.update(identity)
        result["attempt_id"] = selected_attempt
        result["identity_hash"] = stable_hash(identity)
        result["failure_stage"] = "construction"
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for exact formal hardware preflight.")
        device = torch.device("cuda")
        properties = torch.cuda.get_device_properties(device)
        result["gpu_name"] = properties.name
        result["gpu_total_memory"] = int(properties.total_memory)
        torch.cuda.reset_peak_memory_stats(device)
        seed_everything(int(protocol["default_seed"]))
        batch = _synthetic_batch(
            model_id,
            training_profile=training_profile,
            formal_scope_id=formal_scope_id,
        )
        runtime = build_model_runtime(model_id, protocol, run_mode="formal")
        apply_experiment_profile(runtime, selected_profile)
        apply_training_profile(runtime, training_profile)
        apply_model_precision_policy(runtime)
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
            enabled=bool(
                runtime.effective_config.get(
                    "amp_enabled", protocol["amp_enabled"]
                )
            ),
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
            if not torch.isfinite(output.prediction).all().item():
                raise RuntimeError("Preflight prediction is not finite.")
            result["finite_prediction"] = True
            loss = loss_for_profile(selected_profile)(
                output.prediction, moved.target, moved.mask
            )
        if not torch.isfinite(loss):
            raise RuntimeError("Preflight loss is not finite.")
        result["finite_loss"] = True
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
        is_oom = bool(
            (oom_type is not None and isinstance(exc, oom_type))
            or "out of memory" in str(exc).casefold()
        )
        result["status"] = "FAIL_OOM" if is_oom else "FAIL_NON_OOM"
        result["error_type"] = type(exc).__name__
        result["error_message"] = str(exc)
        import traceback

        result["traceback"] = traceback.format_exc()
    finally:
        if torch is not None:
            try:
                if torch.cuda.is_available():
                    result["peak_allocated_memory"] = int(torch.cuda.max_memory_allocated())
                    result["peak_reserved_memory"] = int(torch.cuda.max_memory_reserved())
            except Exception:
                pass
        atomic_write_json(result_path, result)
        pointer = _publish_latest(
            model_id,
            root=root,
            attempt_id=selected_attempt,
            result=result,
            result_path=result_path,
        )
        if result["status"] == "FAIL_NON_OOM":
            atomic_write_json(
                attempt_root / "engineering_block.json",
                {
                    "model_id": model_id,
                    "attempt_id": selected_attempt,
                    "runtime_status": "BLOCKED_NON_OOM",
                    "source": str(result_path),
                    "error_type": result["error_type"],
                    "error_message": result["error_message"],
                    "created_at": result["created_at"],
                },
            )
    result["artifact_path"] = str(result_path)
    result["artifact_sha256"] = _artifact_sha256(result_path)
    result["latest_pointer"] = pointer
    return result


def _run_child(
    args: Sequence[str],
    *,
    runner: Callable[..., Any] = subprocess.run,
) -> int:
    command = [str(Path(sys.executable).resolve()), str(RUNNER_PATH), *args]
    if runner is subprocess.run:
        source_root = PROJECT_ROOT / "custom_models" / "src"
        old_python_path = os.environ.get("PYTHONPATH", "")
        python_path = os.pathsep.join(
            value
            for value in (str(PROJECT_ROOT), str(source_root), old_python_path)
            if value
        )
        completed = runner(
            command,
            check=False,
            cwd=str(PROJECT_ROOT),
            env={**os.environ, "PYTHONPATH": python_path},
        )
    else:
        completed = runner(command, check=False)
    return int(completed.returncode)


def launch_preflight(
    model_id: str,
    *,
    root: str | Path | None = None,
    experiment_profile: str | None = None,
    training_profile: str | None = None,
    formal_scope_id: str | None = None,
    source_revision: str | None = None,
    attempt_id: str | None = None,
    runner: Callable[..., Any] = subprocess.run,
) -> int:
    args = ["_hardware-preflight-worker", "--model", model_id]
    selected_profile = normalize_profile(experiment_profile)
    if selected_profile != DEFAULT_PROFILE_ID:
        args.extend(["--experiment-profile", selected_profile])
    if training_profile is not None:
        args.extend(["--training-profile", training_profile])
    if formal_scope_id is not None:
        args.extend(["--formal-scope-id", formal_scope_id])
    if source_revision is not None:
        args.extend(["--source-revision", source_revision])
    if root is not None:
        args.extend(["--preflight-root", str(Path(root))])
    if attempt_id is not None:
        args.extend(["--preflight-attempt-id", _validated_attempt_id(attempt_id)])
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
    source_revision: str | None = None,
    preflight_attempt_id: str | None = None,
    preflight_artifact_sha256: str | None = None,
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
    current_scope26_request = is_current_scope26_train_request(
        model_id=entry.canonical_id,
        formal_scope_id=formal_scope_id,
        experiment_profile=selected_profile,
        training_profile=training_profile,
    )
    scope_request = scope27_request or current_scope26_request
    preflight_pass: dict[str, Any] | None = None
    if scope27_request or current_scope26_request:
        exact_scope_id = (
            E5_SCOPE27_ID if scope27_request else CURRENT_SCOPE26_ID
        )
        preflight_pass = read_matching_pass(
            entry.canonical_id,
            root=preflight_root,
            experiment_profile=selected_profile,
            training_profile=training_profile,
            formal_scope_id=exact_scope_id,
            source_revision=source_revision,
            attempt_id=preflight_attempt_id,
            artifact_sha256=preflight_artifact_sha256,
        )
        if preflight_pass is None:
            # Current Original formal runs are fail-closed.  The explicit
            # ``preflight`` command is the only operation that may create the
            # GPU artifact; a train request never creates one implicitly.
            return 3
    needs_preflight = not scope_request and (
        training_profile is not None
        or selected_profile == E5_PROFILE_ID
        or bool(entry.values.get("formal_hardware_preflight_required", False))
    )
    if needs_preflight:
        preflight_pass = read_matching_pass(
            entry.canonical_id,
            root=preflight_root,
            experiment_profile=selected_profile,
            training_profile=training_profile,
            attempt_id=preflight_attempt_id,
            artifact_sha256=preflight_artifact_sha256,
        )
        if preflight_pass is None:
            code = launch_preflight(
                entry.canonical_id,
                root=preflight_root,
                experiment_profile=selected_profile,
                training_profile=training_profile,
                runner=runner,
            )
            if code != 0:
                return code
            preflight_pass = read_matching_pass(
                entry.canonical_id,
                root=preflight_root,
                experiment_profile=selected_profile,
                training_profile=training_profile,
            )
        if preflight_pass is None:
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
    if formal_scope_id in (E5_SCOPE27_ID, CURRENT_SCOPE26_ID):
        worker_args.extend(["--formal-scope-id", formal_scope_id])
    if source_revision is not None:
        worker_args.extend(["--source-revision", source_revision])
    if preflight_root is not None:
        worker_args.extend(["--preflight-root", str(Path(preflight_root))])
    if preflight_pass is not None:
        if preflight_pass.get("preflight_attempt_id"):
            worker_args.extend(
                ["--preflight-attempt-id", str(preflight_pass["preflight_attempt_id"])]
            )
        if preflight_pass.get("preflight_artifact_sha256"):
            worker_args.extend(
                [
                    "--preflight-artifact-sha256",
                    str(preflight_pass["preflight_artifact_sha256"]),
                ]
            )
    return _run_child(worker_args, runner=runner)


def launch_model_suite(
    requests: Sequence[Mapping[str, Any]],
    *,
    launcher: Callable[..., int] = launch_formal_train,
) -> list[dict[str, Any]]:
    """Run isolated model launchers and continue after individual failures."""
    results: list[dict[str, Any]] = []
    for request in requests:
        model_id = str(request["model_id"])
        try:
            code = int(launcher(**dict(request)))
            result = {"model_id": model_id, "exit_code": code}
        except Exception as exc:  # Keep the suite moving; caller gets failure evidence.
            result = {
                "model_id": model_id,
                "exit_code": 1,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        results.append(result)
    return results
