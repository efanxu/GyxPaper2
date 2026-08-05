"""Independent formal gate for the ST-MGPrompt A8 Batch4 prerequisite.

The gate deliberately does not import the E5 manifest to define A8.  A8 is a
first-class prerequisite artifact: it is trained once, audited by this file,
and then consumed read-only by E5.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "custom_models" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from benchmark_v2.training_profiles import load_training_profile  # noqa: E402
from st_mgprompt.a8_batch4_contract import (  # noqa: E402
    A8ContractError,
    A8_DEFINITION,
    A8_MODEL_ID,
    A8_OUTPUT_ROOT,
    A8_REFERENCE_ID,
    A8_REFERENCE_RELATIVE_PATH,
    A8_RUN_ID,
    A8_RUN_RELATIVE_PATH,
    A8_SCOPE_ID,
    A8_VARIANT,
    DATASET_ID,
    DATA_SIGNATURE_SCHEMA_VERSION,
    DATA_SPLIT_RATIOS,
    DATA_STRIDES,
    EXPECTED_CONFIG,
    FEATURE_ORDER,
    FEATURE_ORDER_HASH,
    INPUT_PATV_COL,
    INPUT_RELATIVE_PATH,
    LOSS_ID,
    LOSS_PROTOCOL,
    PRECISION_POLICY,
    TARGET_COL,
    TARGET_MASK_COL,
    TARGET_RELATIVE_PATH,
    TRAINING_PROFILE_ID,
    TRAINING_ROLE,
    canonical_hash,
    graph_identity,
    loss_identity,
    precision_identity,
    sha256_file,
    source_closure,
    variant_contract_hash,
)


RESULT_ROOT = PROJECT_ROOT / A8_OUTPUT_ROOT
RUN_DIR = RESULT_ROOT / A8_RUN_ID / "STMGPrompt_ComponentAblation"
AUDIT_ROOT = PROJECT_ROOT / "custom_models/logs/uniform_bs4/audit/st_mgprompt_a8_batch4"
PREFLIGHT_ROOT = PROJECT_ROOT / "custom_models/logs/uniform_bs4/preflight/st_mgprompt_a8_batch4"
LOCK_PATH = PROJECT_ROOT / "custom_models/logs/uniform_bs4/st_mgprompt_a8_batch4.lock"
QUARANTINE_DIRECTORY = ".st_mgprompt_a8_batch4_quarantine"
ARCHIVE_DIRECTORY = ".st_mgprompt_a8_batch4_archived_attempts"
INTENT_DIRECTORY = ".intents"
PREFLIGHT_RESULT_PATH = PREFLIGHT_ROOT / "a8_batch4_preflight.json"
HISTORICAL_A8_ROOT = (
    PROJECT_ROOT
    / "custom_models/results/st_mgprompt_component_ablation"
    / "component_ablation_fixed_dual_seed2026/A8/STMGPrompt_ComponentAblation"
)
REQUIRED_HORIZONS = (3, 6, 10)
REQUIRED_METRICS = ("MAE", "RMSE", "R2", "Score")
LOCK_SCHEMA_VERSION = "st_mgprompt_a8_batch4_lock_v1"
RECEIPT_SCHEMA_VERSION = "st_mgprompt_a8_batch4_execution_receipt_v1"
_SELF_PROCESS_START_FALLBACK = time.monotonic()


class A8GateError(RuntimeError):
    """Raised for a fail-closed A8 gate decision."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> tuple[Any | None, str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, "MISSING_FILE"
    except (OSError, UnicodeError, ValueError) as exc:
        return None, f"INVALID_JSON:{type(exc).__name__}:{exc}"
    return value, None


def _write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _file_hash(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _process_start_time(pid: int) -> float | None:
    if pid <= 0:
        return None
    try:
        import psutil  # type: ignore

        return float(psutil.Process(pid).create_time())
    except ImportError:
        # A monotonic value is stable for this process and is never a wall
        # clock timestamp masquerading as process creation time.
        return _SELF_PROCESS_START_FALLBACK if pid == os.getpid() else None
    except Exception as exc:
        if type(exc).__name__ in {"NoSuchProcess", "AccessDenied", "ZombieProcess"}:
            return _SELF_PROCESS_START_FALLBACK if pid == os.getpid() else None
        raise


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return result.returncode == 0 and str(pid) in result.stdout
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _worker_evidence(run_dir: Path) -> tuple[bool, str]:
    """Return active/blocking state; unreadable evidence is active by default."""

    status, status_error = _read_json(run_dir / "run_status.json")
    effective, effective_error = _read_json(run_dir / "effective_config.json")
    if status_error or effective_error:
        return True, "WORKER_EVIDENCE_MISSING_OR_UNREADABLE"
    if not isinstance(status, dict) or not isinstance(effective, dict):
        return True, "WORKER_EVIDENCE_NOT_OBJECTS"
    status_name = str(status.get("status") or status.get("current_stage") or "").upper()
    if status_name in {"RUNNING", "STARTING", "ACTIVE", "TRAINING", "EVALUATING", "PREFLIGHT_RUNNING"}:
        return True, f"WORKER_STATUS_{status_name}"
    if status_name not in {
        "FAILED",
        "COMPLETED",
        "PROCESS_FINISHED",
        "INCOMPLETE",
        "ARCHIVED",
        "NOT_STARTED",
    }:
        return True, "WORKER_STATUS_AMBIGUOUS"
    for payload in (status, effective):
        for key in ("pid", "process_id", "worker_pid", "formal_worker_pid"):
            if key not in payload or payload[key] in (None, ""):
                continue
            try:
                pid = int(payload[key])
            except (TypeError, ValueError):
                return True, "WORKER_PID_MALFORMED"
            if _pid_alive(pid):
                return True, f"WORKER_PID_ACTIVE:{pid}"
    return False, "WORKER_EVIDENCE_DEFINITELY_INACTIVE"


def _active_worker(run_dir: Path) -> bool:
    return _worker_evidence(run_dir)[0]


def lock_status(path: Path = LOCK_PATH) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "ABSENT", "path": str(path)}
    payload, error = _read_json(path)
    if error or not isinstance(payload, dict):
        return {
            "status": "MALFORMED",
            "path": str(path),
            "error": error or "LOCK_NOT_OBJECT",
        }
    required = {
        "schema_version",
        "scope_id",
        "hostname",
        "pid",
        "process_start_time",
        "lock_owner",
        "git_commit",
        "created_at",
    }
    if payload.get("schema_version") != LOCK_SCHEMA_VERSION or not required.issubset(payload):
        return {
            "status": "MALFORMED",
            "path": str(path),
            "missing": sorted(required - set(payload)),
            **payload,
        }
    try:
        pid = int(payload["pid"])
        recorded_start = float(payload["process_start_time"])
    except (TypeError, ValueError):
        return {"status": "MALFORMED", "path": str(path), **payload}
    if str(payload["hostname"]) != socket.gethostname():
        state = "UNKNOWN_REMOTE"
    else:
        current_start = _process_start_time(pid)
        if current_start is None:
            state = "UNKNOWN_PROCESS_START"
        elif not _pid_alive(pid):
            state = "STALE"
        elif abs(current_start - recorded_start) <= 1e-3:
            state = "ACTIVE"
        else:
            state = "STALE"
    return {"status": state, "path": str(path), **payload}


def _lock_payload() -> dict[str, Any]:
    start = _process_start_time(os.getpid())
    if start is None:
        raise A8GateError("Cannot establish a stable current process start identity.")
    owner = f"{socket.gethostname()}:{os.getpid()}:{start:.6f}"
    return {
        "schema_version": LOCK_SCHEMA_VERSION,
        "scope_id": A8_SCOPE_ID,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "process_start_time": start,
        "lock_owner": owner,
        "git_commit": _git_commit(),
        "created_at": utc_now(),
    }


def _acquire_lock(path: Path = LOCK_PATH) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _lock_payload()
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    except FileExistsError as exc:
        raise A8GateError(f"A8_LOCK_EXISTS:{lock_status(path).get('status', 'UNKNOWN')}") from exc
    return payload


def _release_lock(path: Path = LOCK_PATH, owner: Mapping[str, Any] | None = None) -> None:
    if not path.is_file():
        return
    state = lock_status(path)
    payload = owner or _lock_payload()
    if state.get("status") != "ACTIVE":
        return
    for key in ("pid", "process_start_time", "scope_id", "lock_owner"):
        if state.get(key) != payload.get(key):
            return
    path.unlink()


def clear_stale_lock(path: Path = LOCK_PATH) -> dict[str, Any]:
    state = lock_status(path)
    if state["status"] == "ABSENT":
        return state
    if state["status"] != "STALE":
        raise A8GateError(f"Only a confirmed STALE A8 lock may be cleared: {state['status']}")
    path.unlink()
    return {"status": "CLEARED_STALE", "path": str(path)}


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise A8GateError("Cannot resolve current Git commit for A8 identity.")
    return result.stdout.strip()


def validate_contract(project_root: str | Path | None = None) -> dict[str, Any]:
    errors: list[str] = []
    try:
        profile = load_training_profile(TRAINING_PROFILE_ID)
        if profile is None:
            errors.append("TRAINING_PROFILE_MISSING")
        else:
            expected_profile = {
                "train_batch_size": 4,
                "val_batch_size": 4,
                "test_batch_size": 4,
                "gradient_accumulation_steps": 1,
                "effective_train_batch_size": 4,
            }
            profile_identity = profile.identity()
            errors.extend(
                f"PROFILE_{key}_MISMATCH"
                for key, expected in expected_profile.items()
                if profile_identity.get(key) != expected
            )
        source = source_closure(project_root)
        graph = graph_identity(project_root)
        loss = loss_identity(project_root)
    except (A8ContractError, OSError, ValueError, KeyError) as exc:
        source = {}
        graph = {}
        loss = {}
        errors.append(f"CONTRACT_UNPROVABLE:{type(exc).__name__}:{exc}")
    if A8_VARIANT != EXPECTED_CONFIG["component_ablation"]:
        errors.append("A8_VARIANT_CONSTANT_MISMATCH")
    return {
        "schema_version": "st_mgprompt_a8_batch4_contract_validation_v1",
        "status": "PASS" if not errors else "BLOCKED_A8_CONTRACT",
        "scope_id": A8_SCOPE_ID,
        "model_id": A8_MODEL_ID,
        "variant": A8_VARIANT,
        "definition": A8_DEFINITION,
        "training_profile_id": TRAINING_PROFILE_ID,
        "loss_identity": loss,
        "precision_identity": precision_identity(),
        "variant_contract_hash": variant_contract_hash(),
        "source_closure_hash": source.get("canonical_combined_hash"),
        "source_closure": source,
        "graph_identity": graph,
        "graph_identity_hash": canonical_hash(graph) if graph else None,
        "errors": errors,
    }


def _expected_run_dir(
    output_root: str | Path | None = None,
    project_root: str | Path | None = None,
) -> Path:
    repository_root = Path(project_root or PROJECT_ROOT).resolve()
    canonical_root = (repository_root / A8_OUTPUT_ROOT).resolve()
    root = Path(output_root or canonical_root).resolve()
    if project_root is None and root != RESULT_ROOT.resolve():
        raise A8GateError("Formal A8 output root must be the current Batch4 A8 root.")
    if not _path_within(root, repository_root):
        raise A8GateError("A8 output root must remain inside the repository root.")
    return root / A8_RUN_ID / "STMGPrompt_ComponentAblation"


def _metric_validation(run_dir: Path) -> tuple[dict[int, dict[str, Any]], list[str], str | None, str | None]:
    metrics: dict[int, dict[str, Any]] = {}
    reasons: list[str] = []
    for horizon in REQUIRED_HORIZONS:
        path = run_dir / f"metrics_eval_h{horizon}.json"
        payload, error = _read_json(path)
        if error:
            reasons.append(f"{error}:{path.name}")
            continue
        if not isinstance(payload, dict):
            reasons.append(f"NOT_OBJECT:{path.name}")
            continue
        metrics[horizon] = payload
        if payload.get("horizon") != horizon:
            reasons.append(f"HORIZON_MISMATCH:{path.name}")
        for key in REQUIRED_METRICS:
            if not _finite(payload.get(key)):
                reasons.append(f"NONFINITE_OR_MISSING:{path.name}:{key}")
        count = payload.get("valid_target_count")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            reasons.append(f"INVALID_VALID_TARGET_COUNT:{path.name}")
    csv_path = run_dir / "metrics.csv"
    csv_values: dict[int, dict[str, str]] = {}
    if not csv_path.is_file():
        reasons.append("MISSING_FILE:metrics.csv")
    else:
        try:
            with csv_path.open(encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    try:
                        csv_values[int(row.get("horizon", ""))] = row
                    except (TypeError, ValueError):
                        reasons.append("INVALID_CSV_HORIZON")
        except (OSError, csv.Error) as exc:
            reasons.append(f"INVALID_CSV:{type(exc).__name__}")
    for horizon, payload in metrics.items():
        row = csv_values.get(horizon)
        if row is None:
            reasons.append(f"JSON_CSV_MISSING_H{horizon}")
            continue
        for key in REQUIRED_METRICS:
            try:
                value = float(row.get(key, ""))
            except (TypeError, ValueError):
                value = float("nan")
            if not math.isfinite(value) or not math.isclose(
                value, float(payload[key]), rel_tol=1e-9, abs_tol=1e-9
            ):
                reasons.append(f"JSON_CSV_MISMATCH_H{horizon}_{key}")
        try:
            count = int(row.get("valid_target_count", ""))
        except (TypeError, ValueError):
            count = None
        if count != payload.get("valid_target_count"):
            reasons.append(f"JSON_CSV_MISMATCH_H{horizon}_valid_target_count")
    bundle_records = [
        {"path": "metrics.csv", "sha256": _file_hash(csv_path)},
        *[
            {
                "path": f"metrics_eval_h{horizon}.json",
                "sha256": _file_hash(run_dir / f"metrics_eval_h{horizon}.json"),
            }
            for horizon in REQUIRED_HORIZONS
        ],
    ]
    bundle_hash = (
        canonical_hash(bundle_records)
        if all(record["sha256"] is not None for record in bundle_records)
        else None
    )
    csv_hash = _file_hash(csv_path)
    return metrics, reasons, bundle_hash, csv_hash


def _config_reasons(run_dir: Path) -> list[str]:
    payload, error = _read_json(run_dir / "effective_config.json")
    if error or not isinstance(payload, dict):
        return [f"EFFECTIVE_CONFIG_{error or 'NOT_OBJECT'}"]
    reasons = []
    for key, expected in EXPECTED_CONFIG.items():
        if payload.get(key) != expected:
            reasons.append(f"A8_CONFIG_MISMATCH:{key}")
    if payload.get("training_batch_profile_id") != TRAINING_PROFILE_ID:
        reasons.append("A8_TRAINING_PROFILE_MISMATCH")
    profile = load_training_profile(TRAINING_PROFILE_ID)
    if profile is None or payload.get("training_batch_profile_hash") != profile.profile_hash:
        reasons.append("A8_TRAINING_PROFILE_HASH_MISMATCH")
    if payload.get("run_id") not in {None, A8_RUN_ID}:
        reasons.append("A8_RUN_ID_MISMATCH")
    if payload.get("run_mode") == "smoke" or payload.get("smoke") is True:
        reasons.append("SMOKE_A8_NOT_FORMAL")
    return reasons


def _data_signature_reasons(
    run_dir: Path,
    *,
    project_root: str | Path | None = None,
) -> tuple[list[str], dict[str, Any] | None, str | None]:
    """Validate the real data bundle identity and return its canonical hash."""

    path = run_dir / "data_signature.json"
    signature, error = _read_json(path)
    if error or not isinstance(signature, dict):
        return [f"A8_DATA_SIGNATURE_{error or 'NOT_OBJECT'}"], None, None
    reasons: list[str] = []
    repository_root = Path(project_root or PROJECT_ROOT).resolve()

    expected_simple = {
        "schema_version": DATA_SIGNATURE_SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "run_id": A8_RUN_ID,
        "node_count": 134,
        "feature_order": list(FEATURE_ORDER),
        "feature_names": list(FEATURE_ORDER),
        "feature_order_hash": FEATURE_ORDER_HASH,
        "target_col": TARGET_COL,
        "input_patv_col": INPUT_PATV_COL,
        "target_mask_col": TARGET_MASK_COL,
        "split_ratios": list(DATA_SPLIT_RATIOS),
        "lookback": 144,
        "max_pred_len": 10,
        "eval_horizons": [3, 6, 10],
        "stride": dict(DATA_STRIDES),
        "strides": dict(DATA_STRIDES),
        "seed": 2026,
        "training_profile_id": TRAINING_PROFILE_ID,
    }
    for key, expected in expected_simple.items():
        if signature.get(key) != expected:
            reasons.append(f"A8_DATA_SIGNATURE_MISMATCH:{key}")
    profile = load_training_profile(TRAINING_PROFILE_ID)
    if profile is None or signature.get("training_profile_hash") != profile.profile_hash:
        reasons.append("A8_DATA_SIGNATURE_PROFILE_HASH_MISMATCH")

    resolved_paths: dict[str, Path] = {}
    for field, expected_relative in (
        ("input_relative_path", INPUT_RELATIVE_PATH),
        ("target_relative_path", TARGET_RELATIVE_PATH),
    ):
        value = signature.get(field)
        normalized = str(value).replace("\\", "/") if isinstance(value, str) else ""
        candidate = Path(normalized)
        safe = bool(
            normalized
            and not candidate.is_absolute()
            and not normalized.startswith("/")
            and not normalized.startswith("\\")
            and ":" not in normalized.split("/", 1)[0]
            and ".." not in candidate.parts
        )
        if not safe or normalized != expected_relative:
            reasons.append(f"A8_DATA_SIGNATURE_PATH_MISMATCH:{field}")
            continue
        resolved = (repository_root / candidate).resolve()
        if not _path_within(resolved, repository_root):
            reasons.append(f"A8_DATA_SIGNATURE_PATH_ESCAPE:{field}")
            continue
        resolved_paths[field] = resolved
        alias = "input_path" if field.startswith("input") else "target_path"
        if signature.get(alias) != expected_relative:
            reasons.append(f"A8_DATA_SIGNATURE_PATH_ALIAS_MISMATCH:{alias}")
    for field, path_key in (("input_sha256", "input_relative_path"), ("target_sha256", "target_relative_path")):
        path_value = resolved_paths.get(path_key)
        actual = _file_hash(path_value) if path_value is not None else None
        if not isinstance(signature.get(field), str) or signature.get(field) != actual:
            reasons.append(f"A8_DATA_SIGNATURE_HASH_MISMATCH:{field}")

    identity_hash = canonical_hash(signature)
    return reasons, signature, identity_hash


def _receipt_reasons(
    run_dir: Path,
    metrics_bundle_hash: str | None,
    metrics_csv_hash: str | None,
    *,
    project_root: str | Path | None = None,
) -> tuple[list[str], dict[str, Any] | None]:
    receipt, error = _read_json(run_dir / "a8_batch4_execution_receipt.json")
    if error or not isinstance(receipt, dict):
        return [f"A8_RECEIPT_{error or 'NOT_OBJECT'}"], None
    reasons: list[str] = []
    expected_simple = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "scope_id": A8_SCOPE_ID,
        "training_role": TRAINING_ROLE,
        "model_id": A8_MODEL_ID,
        "variant": A8_VARIANT,
        "definition": A8_DEFINITION,
        "run_id": A8_RUN_ID,
        "output_root": A8_OUTPUT_ROOT,
        "training_profile_id": TRAINING_PROFILE_ID,
        "variant_contract_hash": variant_contract_hash(),
        "trained_for_e5_scope27": True,
        "consumed_read_only_by_e5": True,
        "checkpoint_copied": False,
        "metrics_copied": False,
        "warm_started_from_historical_a8": False,
    }
    for key, expected in expected_simple.items():
        if receipt.get(key) != expected:
            reasons.append(f"A8_RECEIPT_MISMATCH:{key}")
    receipt_status = receipt.get("status")
    if receipt_status not in {"COMPLETED", "FAILED"}:
        reasons.append("A8_RECEIPT_MISMATCH:status")
    elif receipt_status == "FAILED":
        # A failed attempt is an identity-matched incomplete artifact.  It is
        # eligible for the explicit safe archive path, but can never satisfy
        # readiness.  Do not classify it as an identity mismatch because a
        # retry would otherwise be blocked by the no-overwrite receipt rule.
        reasons.append("A8_RECEIPT_INCOMPLETE")
    elif receipt.get("exit_code") != 0:
        reasons.append("A8_RECEIPT_MISMATCH:exit_code")
    required = (
        "git_commit",
        "source_closure_hash",
        "effective_config_hash",
        "training_profile_hash",
        "data_signature_path",
        "data_signature_hash",
        "dataset_identity_hash",
        "macro_graph_identity_hash",
        "micro_graph_identity_hash",
        "loss_identity_hash",
        "protocol_hash",
        "precision_identity_hash",
        "best_checkpoint_sha256",
        "last_checkpoint_sha256",
        "metrics_bundle_hash",
        "metrics_csv_hash",
        "train_log_hash",
        "command",
        "started_at",
        "finished_at",
    )
    reasons.extend(f"A8_RECEIPT_MISSING:{key}" for key in required if not receipt.get(key))
    data_reasons, data_signature, data_identity_hash = _data_signature_reasons(
        run_dir,
        project_root=project_root,
    )
    reasons.extend(data_reasons)
    if receipt.get("data_signature_path") != "data_signature.json":
        reasons.append("A8_DATA_SIGNATURE_PATH_RECEIPT_MISMATCH")
    if data_identity_hash is None or receipt.get("data_signature_hash") != data_identity_hash:
        reasons.append("A8_DATA_SIGNATURE_HASH_RECEIPT_MISMATCH")
    if data_identity_hash is None or receipt.get("dataset_identity_hash") != data_identity_hash:
        reasons.append("A8_DATASET_IDENTITY_HASH_MISMATCH")
    if receipt.get("source_closure_hash") != source_closure(project_root)["canonical_combined_hash"]:
        reasons.append("A8_SOURCE_CLOSURE_HASH_MISMATCH")
    try:
        current_commit = _git_commit()
    except A8GateError:
        current_commit = None
    if current_commit is not None and receipt.get("git_commit") != current_commit:
        reasons.append("A8_GIT_COMMIT_MISMATCH")
    if receipt.get("effective_config_hash") != _file_hash(run_dir / "effective_config.json"):
        reasons.append("A8_EFFECTIVE_CONFIG_HASH_MISMATCH")
    profile = load_training_profile(TRAINING_PROFILE_ID)
    if profile is None or receipt.get("training_profile_hash") != profile.profile_hash:
        reasons.append("A8_RECEIPT_PROFILE_HASH_MISMATCH")
    graph = graph_identity(project_root)
    expected_macro_hash = canonical_hash(
        {key: graph[key] for key in ("macro_graph_source", "macro_graph_hash", "graph_generation_protocol_hash")}
    )
    expected_micro_hash = canonical_hash(
        {key: graph[key] for key in ("micro_graph_source", "micro_graph_hash", "graph_generation_protocol_hash")}
    )
    if receipt.get("macro_graph_identity_hash") != expected_macro_hash:
        reasons.append("A8_MACRO_GRAPH_IDENTITY_MISMATCH")
    if receipt.get("micro_graph_identity_hash") != expected_micro_hash:
        reasons.append("A8_MICRO_GRAPH_IDENTITY_MISMATCH")
    if receipt.get("loss_identity_hash") != loss_identity(project_root)["loss_identity_hash"]:
        reasons.append("A8_LOSS_IDENTITY_MISMATCH")
    if receipt.get("precision_identity_hash") != canonical_hash(precision_identity()):
        reasons.append("A8_PRECISION_IDENTITY_MISMATCH")
    if receipt.get("best_checkpoint_sha256") != _file_hash(run_dir / "best_checkpoint.pt"):
        reasons.append("A8_BEST_CHECKPOINT_HASH_MISMATCH")
    if receipt.get("last_checkpoint_sha256") != _file_hash(run_dir / "last_checkpoint.pt"):
        reasons.append("A8_LAST_CHECKPOINT_HASH_MISMATCH")
    if receipt.get("metrics_bundle_hash") != metrics_bundle_hash:
        reasons.append("A8_METRICS_BUNDLE_HASH_MISMATCH")
    if receipt.get("metrics_csv_hash") != metrics_csv_hash:
        reasons.append("A8_METRICS_CSV_HASH_MISMATCH")
    if receipt.get("train_log_hash") != _file_hash(run_dir / "train_log.csv"):
        reasons.append("A8_TRAIN_LOG_HASH_MISMATCH")
    for key in ("macro_graph_identity", "micro_graph_identity"):
        if receipt.get(key) != graph:
            reasons.append(f"A8_{key.upper()}_MISMATCH")
    return reasons, receipt


def inspect_a8_artifact(
    output_root: str | Path | None = None,
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Inspect only the current A8 path; historical/smoke paths never pass."""

    repository_root = Path(project_root or PROJECT_ROOT).resolve()
    run_dir = _expected_run_dir(output_root, project_root=project_root)
    historical_root = (
        repository_root
        / "custom_models/results/st_mgprompt_component_ablation"
        / "component_ablation_fixed_dual_seed2026/A8/STMGPrompt_ComponentAblation"
    ).resolve()
    reasons: list[str] = []
    found = run_dir.exists()
    if not found:
        current_root = run_dir.parent.parent
        candidates = [path for path in _candidate_dirs(current_root) if path != run_dir]
        if candidates:
            for candidate in candidates:
                candidate_config, candidate_error = _read_json(
                    candidate / "effective_config.json"
                )
                if candidate_error or not isinstance(candidate_config, dict):
                    reasons.append("A8_NONCANONICAL_CANDIDATE_UNREADABLE")
                    continue
                if candidate_config.get("smoke") is True or candidate_config.get("run_mode") == "smoke":
                    reasons.append("SMOKE_A8_NOT_FORMAL")
                else:
                    reasons.append("A8_NONCANONICAL_RUN_ID_OR_PATH")
        if historical_root.is_dir():
            reasons.append("HISTORICAL_BATCH32_A8_NOT_CURRENT")
        return {
            "scope_id": A8_SCOPE_ID,
            "model_id": A8_MODEL_ID,
            "variant": A8_VARIANT,
            "run_id": A8_RUN_ID,
            "run_dir": str(run_dir),
            "found": bool(candidates),
            "ready": False,
            "action": (
                "BLOCK_EXISTING_IDENTITY_MISMATCH"
                if candidates
                else "RUN_MISSING"
            ),
            "reasons": reasons or ["A8_BATCH4_FORMAL_DIRECTORY_MISSING"],
            "training_role": TRAINING_ROLE,
            "candidate_dirs": [str(path) for path in candidates],
        }
    if run_dir.is_symlink() or not run_dir.is_dir():
        reasons.append("A8_RUN_DIRECTORY_UNSAFE")
    reasons.extend(_config_reasons(run_dir))
    metrics, metric_reasons, bundle_hash, csv_hash = _metric_validation(run_dir)
    reasons.extend(metric_reasons)
    for filename in (
        "best_checkpoint.pt",
        "last_checkpoint.pt",
        "train_complete.json",
        "evaluation_complete.json",
        "protocol_check.json",
        "run_status.json",
        "data_signature.json",
    ):
        path = run_dir / filename
        if not path.is_file() or (path.suffix == ".pt" and path.stat().st_size <= 0):
            reasons.append(f"MISSING_OR_EMPTY:{filename}")
    status, status_error = _read_json(run_dir / "run_status.json")
    train, train_error = _read_json(run_dir / "train_complete.json")
    evaluation, evaluation_error = _read_json(run_dir / "evaluation_complete.json")
    protocol, protocol_error = _read_json(run_dir / "protocol_check.json")
    if status_error or not isinstance(status, dict) or status.get("status") != "COMPLETED" or status.get("exit_code") != 0:
        reasons.append("A8_RUN_STATUS_NOT_COMPLETED")
    if train_error or not isinstance(train, dict) or train.get("status") != "completed":
        reasons.append("A8_TRAINING_NOT_COMPLETE")
    if evaluation_error or not isinstance(evaluation, dict) or evaluation.get("status") != "completed":
        reasons.append("A8_EVALUATION_NOT_COMPLETE")
    if protocol_error or not isinstance(protocol, dict) or protocol.get("passed") is not True:
        reasons.append("A8_PROTOCOL_NOT_PASS")
    worker_active, worker_reason = _worker_evidence(run_dir)
    if worker_active:
        reasons.append(worker_reason)
    receipt_reasons, receipt = _receipt_reasons(
        run_dir,
        bundle_hash,
        csv_hash,
        project_root=project_root,
    )
    reasons.extend(receipt_reasons)
    if any(reason == "SMOKE_A8_NOT_FORMAL" for reason in reasons):
        action = "BLOCK_EXISTING_IDENTITY_MISMATCH"
    elif any("MISMATCH" in reason or "HISTORICAL" in reason for reason in reasons):
        action = "BLOCK_EXISTING_IDENTITY_MISMATCH"
    elif reasons:
        action = "ARCHIVE_INCOMPLETE_THEN_RUN" if not worker_active else "BLOCK_EXISTING_IDENTITY_MISMATCH"
    else:
        action = "SKIP_COMPLETED_IDENTITY_MATCH"
    return {
        "scope_id": A8_SCOPE_ID,
        "model_id": A8_MODEL_ID,
        "variant": A8_VARIANT,
        "definition": A8_DEFINITION,
        "run_id": A8_RUN_ID,
        "run_dir": str(run_dir),
        "found": True,
        "ready": not reasons,
        "action": action,
        "reasons": reasons,
        "training_role": TRAINING_ROLE,
        "metrics": metrics,
        "metrics_bundle_hash": bundle_hash,
        "metrics_csv_hash": csv_hash,
        "receipt": receipt,
        "worker_evidence": worker_reason,
        "best_checkpoint_sha256": _file_hash(run_dir / "best_checkpoint.pt"),
        "last_checkpoint_sha256": _file_hash(run_dir / "last_checkpoint.pt"),
        "effective_config_sha256": _file_hash(run_dir / "effective_config.json"),
    }


def build_readiness(
    output_root: str | Path | None = None,
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    contract = validate_contract(project_root)
    artifact = inspect_a8_artifact(output_root, project_root=project_root)
    status = "READY_A8_BATCH4_PREREQUISITE" if contract["status"] == "PASS" and artifact["ready"] else (
        "BLOCKED_A8_BATCH4_PREREQUISITE" if "HISTORICAL_BATCH32_A8_NOT_CURRENT" in artifact["reasons"] else "NOT_READY_A8_BATCH4"
    )
    return {
        "schema_version": "st_mgprompt_a8_batch4_readiness_v1",
        "status": status,
        "scope_id": A8_SCOPE_ID,
        "model_id": A8_MODEL_ID,
        "variant": A8_VARIANT,
        "definition": A8_DEFINITION,
        "training_role": TRAINING_ROLE,
        "run_id": A8_RUN_ID,
        "output_root": A8_OUTPUT_ROOT,
        "run_dir": artifact["run_dir"],
        "formal_training_executed": bool(artifact.get("ready")),
        "contract": contract,
        "artifact": artifact,
        "trained_for_e5_scope27": bool(artifact.get("ready")),
        "consumed_read_only_by_e5": bool(artifact.get("ready")),
        "generated_at": utc_now(),
    }


def build_freeze_plan() -> dict[str, Any]:
    contract = validate_contract()
    artifact = inspect_a8_artifact()
    formal_freeze = None
    if artifact.get("ready"):
        formal_freeze = canonical_hash(
            {
                "scope_id": A8_SCOPE_ID,
                "source_closure_hash": contract.get("source_closure_hash"),
                "effective_config_hash": artifact.get("effective_config_sha256"),
                "receipt_hash": _file_hash(RUN_DIR / "a8_batch4_execution_receipt.json"),
                "graph_identity_hash": contract.get("graph_identity_hash"),
            }
        )
    return {
        "schema_version": "st_mgprompt_a8_batch4_freeze_plan_v1",
        "scope_id": A8_SCOPE_ID,
        "formal_freeze_available": formal_freeze is not None,
        "formal_freeze_hash": formal_freeze,
        "contract": contract,
        "artifact": artifact,
        "generated_at": utc_now(),
    }


def build_preflight_plan() -> dict[str, Any]:
    return {
        "schema_version": "st_mgprompt_a8_batch4_exact_preflight_plan_v1",
        "scope_id": A8_SCOPE_ID,
        "model_id": A8_MODEL_ID,
        "variant": A8_VARIANT,
        "training_role": TRAINING_ROLE,
        "execution_performed": False,
        "gpu_preflight_performed": False,
        "exact_shape": {"B": 4, "T": 144, "N": 134, "C": 16, "H": 10},
        "loss_id": LOSS_ID,
        "loss_protocol": LOSS_PROTOCOL,
        "precision_identity": precision_identity(),
        "same_graph_resources": True,
        "checks": [
            "model_construction",
            "forward",
            "prediction_shape",
            "finite_prediction",
            "finite_loss",
            "backward",
            "finite_gradients",
            "peak_allocated_memory",
            "peak_reserved_memory",
            "strict_checkpoint_reload",
        ],
    }


def _preflight_command(source_revision: str | None = None) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "custom_models/src/st_mgprompt/run_st_mgprompt.py"),
        "--preflight-full-shape",
        "--component-ablation",
        A8_VARIANT,
        "--training-profile",
        TRAINING_PROFILE_ID,
        "--run-id",
        A8_RUN_ID,
        "--output-root",
        str(PREFLIGHT_ROOT),
        "--model-input-path",
        str(PROJECT_ROOT / "dataset/sdwpf_model_input_base.parquet"),
        "--eval-target-path",
        str(PROJECT_ROOT / "dataset/sdwpf_eval_target.parquet"),
        "--device",
        "cuda",
        "--no-amp",
        "--windows-safe-mode",
    ]
    if source_revision:
        command.extend(["--source-revision", source_revision])
    return command


def read_matching_preflight_pass(path: Path = PREFLIGHT_RESULT_PATH) -> dict[str, Any] | None:
    payload, error = _read_json(path)
    if error or not isinstance(payload, dict):
        return None
    contract = validate_contract()
    if payload.get("status") != "PASS":
        return None
    if payload.get("scope_id") != A8_SCOPE_ID or payload.get("model_id") != A8_MODEL_ID:
        return None
    if payload.get("exact_shape") != {"B": 4, "T": 144, "N": 134, "C": 16, "H": 10}:
        return None
    if payload.get("source_closure_hash") != contract.get("source_closure_hash"):
        return None
    if payload.get("variant_contract_hash") != variant_contract_hash():
        return None
    if payload.get("graph_identity_hash") != contract.get("graph_identity_hash"):
        return None
    profile = load_training_profile(TRAINING_PROFILE_ID)
    expected_protocol_hash = (
        profile.identity().get("base_benchmark_protocol_hash") if profile else None
    )
    if payload.get("loss_identity_hash") != loss_identity()["loss_identity_hash"]:
        return None
    if payload.get("precision_identity_hash") != canonical_hash(precision_identity()):
        return None
    if payload.get("protocol_hash") != expected_protocol_hash:
        return None
    if payload.get("git_commit") != _git_commit():
        return None
    if any(payload.get(key) is not True for key in ("forward_completed", "backward_completed", "finite_prediction", "finite_loss", "finite_gradients", "strict_checkpoint_reload")):
        return None
    return payload


def run_preflight(
    *,
    report_path: Path | None = None,
    child_log_root: Path | None = None,
    source_revision: str | None = None,
    runner=subprocess.run,
) -> tuple[int, dict[str, Any]]:
    contract = validate_contract()
    artifact = inspect_a8_artifact()
    if contract["status"] != "PASS":
        payload = {"status": "BLOCKED_A8_CONTRACT", "contract": contract, "child_launch_count": 0}
        if report_path:
            _write_json(report_path, payload)
        return 74, payload
    if artifact.get("action") == "BLOCK_EXISTING_IDENTITY_MISMATCH":
        payload = {
            "status": "BLOCK_EXISTING_IDENTITY_MISMATCH",
            "artifact": artifact,
            "child_launch_count": 0,
            "gpu_preflight_performed": False,
        }
        if report_path:
            _write_json(report_path, payload)
        return 74, payload
    child_root = Path(child_log_root or PREFLIGHT_ROOT / "child").resolve()
    child_root.mkdir(parents=True, exist_ok=True)
    log_path = child_root / "a8_preflight.log"
    started_at = utc_now()
    exit_code = 74
    error_type = None
    error_message = None
    trace_tail: list[str] = []
    selected_revision = source_revision or _git_commit()
    try:
        with log_path.open("w", encoding="utf-8", newline="") as handle:
            completed = runner(
                _preflight_command(selected_revision),
                cwd=str(PROJECT_ROOT),
                env={**os.environ, "PYTHONPATH": str(SOURCE_ROOT)},
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        exit_code = int(completed.returncode)
    except Exception as exc:
        error_type = type(exc).__name__
        error_message = str(exc)
        trace_tail = traceback.format_exc().splitlines()[-40:]
    preflight_run_dir = PREFLIGHT_ROOT / A8_RUN_ID / "STMGPrompt_ComponentAblation"
    child_report, child_error = _read_json(preflight_run_dir / "full_shape_smoke_report.json")
    passed = (
        exit_code == 0
        and child_error is None
        and isinstance(child_report, dict)
        and child_report.get("status") == "passed"
        and child_report.get("strict_reload_completed") is True
        and child_report.get("batch_shape", {}).get("train_x", [None])[0] == 4
    )
    contract = validate_contract()
    payload = {
        "schema_version": "st_mgprompt_a8_batch4_preflight_result_v1",
        "status": "PASS" if passed else "FAILED",
        "scope_id": A8_SCOPE_ID,
        "model_id": A8_MODEL_ID,
        "variant": A8_VARIANT,
        "exact_shape": {"B": 4, "T": 144, "N": 134, "C": 16, "H": 10},
        "loss_id": LOSS_ID,
        "precision_identity": precision_identity(),
        "source_closure_hash": contract.get("source_closure_hash"),
        "variant_contract_hash": variant_contract_hash(),
        "graph_identity_hash": contract.get("graph_identity_hash"),
        "loss_identity_hash": loss_identity()["loss_identity_hash"],
        "precision_identity_hash": canonical_hash(precision_identity()),
        "protocol_hash": (
            load_training_profile(TRAINING_PROFILE_ID).identity().get(
                "base_benchmark_protocol_hash"
            )
            if load_training_profile(TRAINING_PROFILE_ID)
            else None
        ),
        "git_commit": selected_revision,
        "forward_completed": bool(child_report and child_report.get("gradient_checks") is not None),
        "backward_completed": bool(child_report and child_report.get("gradient_checks") is not None),
        "finite_prediction": bool(child_report and child_report.get("loss_finite") is True),
        "finite_loss": bool(child_report and child_report.get("loss_finite") is True),
        "finite_gradients": bool(child_report and all((child_report.get("gradient_checks") or {}).values())),
        "strict_checkpoint_reload": bool(child_report and child_report.get("strict_reload_completed") is True),
        "peak_allocated_memory": (child_report or {}).get("train_backward_memory_mib", {}).get("peak_allocated"),
        "peak_reserved_memory": (child_report or {}).get("train_backward_memory_mib", {}).get("peak_reserved"),
        "exit_code": exit_code,
        "child_log": str(log_path),
        "child_report": str(preflight_run_dir / "full_shape_smoke_report.json"),
        "error_type": error_type,
        "error_message": error_message,
        "traceback_tail": trace_tail,
        "started_at": started_at,
        "finished_at": utc_now(),
        "gpu_preflight_performed": True,
    }
    PREFLIGHT_RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _write_json(PREFLIGHT_RESULT_PATH, payload)
    if report_path:
        _write_json(report_path, payload)
    return (0 if passed else 1), payload


def _candidate_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    result: list[Path] = []
    for path in root.rglob("effective_config.json"):
        if any(part in {ARCHIVE_DIRECTORY, QUARANTINE_DIRECTORY, INTENT_DIRECTORY} for part in path.parts):
            continue
        if path.is_symlink():
            continue
        payload, error = _read_json(path)
        if not error and isinstance(payload, dict) and (
            payload.get("component_ablation") == A8_VARIANT
            or payload.get("variant") == A8_VARIANT
            or payload.get("model_id") == A8_MODEL_ID
        ):
            result.append(path.parent)
    return sorted(set(result), key=lambda value: value.as_posix())


def _archive_target(root: Path, kind: str) -> Path:
    if kind not in {ARCHIVE_DIRECTORY, QUARANTINE_DIRECTORY}:
        raise A8GateError(f"Unsupported A8 archive kind: {kind}")
    parent = (root / kind / A8_RUN_ID).resolve()
    target = (parent / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}_{os.getpid()}_{uuid.uuid4().hex}").resolve()
    if not _path_within(parent, root) or not _path_within(target, root):
        raise A8GateError("A8 archive target escapes the current output root.")
    if target.exists() or parent.is_symlink():
        raise A8GateError("A8 archive target collision or symlink refused.")
    return target


def archive_or_quarantine(*, kind: str = ARCHIVE_DIRECTORY, apply: bool = False, reason: str = "explicit A8 action") -> dict[str, Any]:
    root = RESULT_ROOT.resolve()
    source = RUN_DIR.resolve()
    if not source.is_dir() or source.is_symlink() or not _path_within(source, root):
        raise A8GateError(f"A8 canonical source is missing or unsafe: {source}")
    active, worker_reason = _worker_evidence(source)
    if active:
        raise A8GateError(f"A8 active-worker evidence blocks archive/quarantine: {worker_reason}")
    inspected = inspect_a8_artifact()
    mismatch = any("MISMATCH" in reason or "HISTORICAL" in reason for reason in inspected.get("reasons", []))
    if kind == QUARANTINE_DIRECTORY and not mismatch:
        raise A8GateError("A8 quarantine requires explicit identity-mismatch evidence.")
    if kind == ARCHIVE_DIRECTORY and mismatch:
        raise A8GateError("A8 identity mismatch cannot be auto-archived.")
    target = _archive_target(root, kind)
    receipt = {
        "schema_version": f"st_mgprompt_a8_batch4_{'archive' if kind == ARCHIVE_DIRECTORY else 'quarantine'}_receipt_v1",
        "scope_id": A8_SCOPE_ID,
        "model_id": A8_MODEL_ID,
        "run_id": A8_RUN_ID,
        "original_path": str(source),
        "archive_path": str(target),
        "reason": reason,
        "reasons": inspected.get("reasons", []),
        "created_at": utc_now(),
    }
    if not apply:
        return {
            "status": "READY_TO_APPLY",
            "action": "QUARANTINE_EXISTING" if kind == QUARANTINE_DIRECTORY else "ARCHIVE_INCOMPLETE_THEN_RUN",
            "source": str(source),
            "target": str(target),
            "receipt": receipt,
        }
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    intent_dir = (root / kind / INTENT_DIRECTORY).resolve()
    intent_dir.mkdir(parents=True, exist_ok=True)
    intent_path = intent_dir / f"{kind}_{A8_RUN_ID}_{uuid.uuid4().hex}.json"
    _write_json(intent_path, {"status": "PENDING", "source": str(source), "target": str(target), "receipt": receipt})
    try:
        os.replace(source, target)
    except Exception as exc:
        _write_json(intent_path, {"status": "FAILED", "source": str(source), "target": str(target), "error": str(exc), "receipt": receipt})
        raise
    receipt_path = target / ("quarantine_receipt.json" if kind == QUARANTINE_DIRECTORY else "archive_receipt.json")
    _write_json(receipt_path, receipt)
    _write_json(intent_path, {"status": "COMPLETED", "source": str(source), "target": str(target), "receipt": str(receipt_path)})
    return {"status": "QUARANTINED" if kind == QUARANTINE_DIRECTORY else "ARCHIVED", "source": str(source), "target": str(target), "receipt": str(receipt_path), "intent": str(intent_path)}


def build_plan() -> dict[str, Any]:
    artifact = inspect_a8_artifact()
    return {
        "schema_version": "st_mgprompt_a8_batch4_resume_plan_v1",
        "scope_id": A8_SCOPE_ID,
        "output_root": str(RESULT_ROOT),
        "counts": {"total": 1, "ready": int(artifact.get("ready", False)), "blocked": int(artifact.get("action") == "BLOCK_EXISTING_IDENTITY_MISMATCH"), "run": int(artifact.get("action") in {"RUN_MISSING", "ARCHIVE_INCOMPLETE_THEN_RUN"})},
        "entries": [artifact],
        "generated_at": utc_now(),
    }


def build_inventory() -> dict[str, Any]:
    return {
        "schema_version": "st_mgprompt_a8_batch4_inventory_v1",
        "scope_id": A8_SCOPE_ID,
        "current_run_dir": str(RUN_DIR),
        "current_exists": RUN_DIR.is_dir(),
        "historical_batch32_root": str(HISTORICAL_A8_ROOT),
        "historical_exists": HISTORICAL_A8_ROOT.is_dir(),
        "candidates": [str(path) for path in _candidate_dirs(RESULT_ROOT)],
        "generated_at": utc_now(),
    }


def _formal_run_command(input_path: str | None, target_path: str | None, source_revision: str | None) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "custom_models/src/st_mgprompt/run_st_mgprompt.py"),
        "--component-ablation",
        A8_VARIANT,
        "--training-profile",
        TRAINING_PROFILE_ID,
        "--run-id",
        A8_RUN_ID,
        "--output-root",
        str(RESULT_ROOT),
        "--model-name",
        "STMGPrompt_ComponentAblation",
        "--device",
        "cuda",
        "--no-amp",
        "--windows-safe-mode",
    ]
    if input_path:
        command.extend(["--model-input-path", str(input_path)])
    if target_path:
        command.extend(["--eval-target-path", str(target_path)])
    if source_revision:
        command.extend(["--source-revision", source_revision])
    return command


def run(
    *,
    input_path: str | None = None,
    target_path: str | None = None,
    log_root: Path | None = None,
    source_revision: str | None = None,
) -> tuple[int, dict[str, Any]]:
    contract = validate_contract()
    if contract["status"] != "PASS":
        return 74, {"status": "BLOCKED_A8_CONTRACT", "contract": contract}
    if read_matching_preflight_pass() is None:
        return 74, {"status": "PREFLIGHT_MISSING_OR_MISMATCH", "scope_id": A8_SCOPE_ID, "gpu_child_started": False}
    plan = build_plan()
    row = plan["entries"][0]
    if row["action"] == "BLOCK_EXISTING_IDENTITY_MISMATCH":
        return 74, {"status": "BLOCK_EXISTING_IDENTITY_MISMATCH", "plan": plan}
    lock_owner = _acquire_lock()
    logs = Path(log_root or AUDIT_ROOT / "formal").resolve()
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / "a8_batch4_formal.log"
    command = _formal_run_command(input_path, target_path, source_revision or _git_commit())
    try:
        if row["action"] == "ARCHIVE_INCOMPLETE_THEN_RUN":
            archive_or_quarantine(kind=ARCHIVE_DIRECTORY, apply=True, reason="identity-matched incomplete A8 attempt")
        with log_path.open("w", encoding="utf-8", newline="") as handle:
            completed = subprocess.run(
                command,
                cwd=str(PROJECT_ROOT),
                env={**os.environ, "PYTHONPATH": str(SOURCE_ROOT)},
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        child_exit_code = int(completed.returncode)
        readiness = build_readiness()
        report: dict[str, Any] = {
            "status": "FAILED" if child_exit_code != 0 else "COMPLETED_BUT_NOT_READY",
            "scope_id": A8_SCOPE_ID,
            "command": command,
            "log_path": str(log_path),
            "readiness": readiness,
            "readiness_status": readiness.get("status"),
            "reference_written": False,
            "child_exit_code": child_exit_code,
            "gate_exit_code": child_exit_code,
            # Keep the historical field for consumers while making the two
            # exit domains explicit above.
            "exit_code": child_exit_code,
        }
        if child_exit_code != 0:
            return child_exit_code, report
        if readiness.get("status") != "READY_A8_BATCH4_PREREQUISITE":
            report["gate_exit_code"] = 74
            report["exit_code"] = 74
            return 74, report
        reference_path = PROJECT_ROOT / A8_REFERENCE_RELATIVE_PATH
        try:
            reference = write_reference(reference_path)
            actual, reference_error = _read_json(reference_path)
            if (
                reference.get("status") != "VALID"
                or reference_error
                or not isinstance(actual, dict)
                or actual != reference
            ):
                raise A8GateError(
                    f"A8 reference was not written and verified: {reference_error or reference.get('status')}"
                )
            report["reference_written"] = True
        except Exception as exc:
            report.update(
                {
                    "status": "REFERENCE_WRITE_FAILED",
                    "reference_error": f"{type(exc).__name__}:{exc}",
                    "gate_exit_code": 74,
                    "exit_code": 74,
                }
            )
            return 74, report
        report["status"] = "COMPLETED"
        report["gate_exit_code"] = 0
        report["exit_code"] = 0
        return 0, report
    finally:
        _release_lock(LOCK_PATH, lock_owner)


def build_reference_payload() -> dict[str, Any]:
    artifact = inspect_a8_artifact()
    if not artifact.get("ready"):
        return {
            "status": "BLOCKED_A8_BATCH4_PREREQUISITE",
            "reference_id": A8_REFERENCE_ID,
            "training_role": TRAINING_ROLE,
            "reasons": artifact.get("reasons", []),
        }
    receipt = artifact["receipt"] or {}
    return {
        "schema_version": "st_mgprompt_a8_batch4_reference_v1",
        "reference_type": "FORMAL_A8_BATCH4_PREREQUISITE",
        "reference_id": A8_REFERENCE_ID,
        "status": "VALID",
        "training_role": TRAINING_ROLE,
        "trained_for_e5_scope27": True,
        "consumed_read_only_by_e5": True,
        "model_id": A8_MODEL_ID,
        "variant": A8_VARIANT,
        "definition": A8_DEFINITION,
        "run_id": A8_RUN_ID,
        "source_relative_path": A8_RUN_RELATIVE_PATH,
        "source_checkpoint_path": f"{A8_RUN_RELATIVE_PATH}/best_checkpoint.pt",
        "source_checkpoint_sha256": artifact.get("best_checkpoint_sha256"),
        "source_metrics_sha256": artifact.get("metrics_bundle_hash"),
        "source_config_sha256": artifact.get("effective_config_sha256"),
        "source_protocol_hash": receipt.get("protocol_hash"),
        "source_receipt_path": f"{A8_RUN_RELATIVE_PATH}/a8_batch4_execution_receipt.json",
        "source_receipt_sha256": _file_hash(RUN_DIR / "a8_batch4_execution_receipt.json"),
        "source_loss_id": LOSS_ID,
        "training_profile_id": TRAINING_PROFILE_ID,
        "training_profile_hash": receipt.get("training_profile_hash"),
        "checkpoint_copied": False,
        "metrics_copied": False,
        "warm_started_from_historical_a8": False,
    }


def write_reference(path: Path | None = None) -> dict[str, Any]:
    target = Path(path or PROJECT_ROOT / A8_REFERENCE_RELATIVE_PATH)
    payload = build_reference_payload()
    if payload.get("status") == "VALID":
        _write_json(target, payload)
        actual, error = _read_json(target)
        if error or actual != payload:
            raise A8GateError(
                f"A8 reference write verification failed: {error or 'PAYLOAD_MISMATCH'}"
            )
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="st_mgprompt_a8_batch4_gate")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate-contract")
    sub.add_parser("freeze-plan")
    sub.add_parser("preflight-plan")
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--report-path", default=str(AUDIT_ROOT / "a8_batch4_preflight_summary.json"))
    preflight.add_argument("--child-log-root", default=str(AUDIT_ROOT / "preflight"))
    preflight.add_argument("--source-revision")
    for name in ("dry-run", "static-audit"):
        sub.add_parser(name)
    inventory = sub.add_parser("inventory")
    inventory.add_argument("--report-path", default=str(AUDIT_ROOT / "a8_batch4_inventory.json"))
    sub.add_parser("lock-status")
    clear = sub.add_parser("clear-stale-lock")
    clear.add_argument("--lock-path", default=str(LOCK_PATH))
    quarantine = sub.add_parser("quarantine-existing")
    quarantine.add_argument("--model", default=A8_MODEL_ID)
    quarantine.add_argument("--apply", action="store_true")
    readiness = sub.add_parser("readiness")
    readiness.add_argument("--report-path", default=str(AUDIT_ROOT / "a8_batch4_readiness.json"))
    readiness.add_argument("--reference-path", default=str(PROJECT_ROOT / A8_REFERENCE_RELATIVE_PATH))
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--input-path")
    run_parser.add_argument("--target-path")
    run_parser.add_argument("--log-root", default=str(AUDIT_ROOT / "formal"))
    run_parser.add_argument("--source-revision")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate-contract":
            result = validate_contract()
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["status"] == "PASS" else 74
        if args.command == "freeze-plan":
            print(json.dumps(build_freeze_plan(), ensure_ascii=False, indent=2))
            return 0
        if args.command == "preflight-plan":
            print(json.dumps(build_preflight_plan(), ensure_ascii=False, indent=2))
            return 0
        if args.command in {"dry-run", "static-audit"}:
            result = build_plan()
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 74 if result["counts"]["blocked"] else 0
        if args.command == "inventory":
            result = build_inventory()
            _write_json(Path(args.report_path), result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "lock-status":
            result = lock_status()
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["status"] == "ABSENT" else 74
        if args.command == "clear-stale-lock":
            result = clear_stale_lock(Path(args.lock_path).resolve())
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "quarantine-existing":
            if args.model != A8_MODEL_ID:
                raise A8GateError("A8 quarantine accepts only st_mgprompt_a8.")
            result = archive_or_quarantine(kind=QUARANTINE_DIRECTORY, apply=args.apply, reason="explicit quarantine-existing")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "preflight":
            code, result = run_preflight(
                report_path=Path(args.report_path).resolve(),
                child_log_root=Path(args.child_log_root).resolve(),
                source_revision=args.source_revision,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return code
        if args.command == "readiness":
            result = build_readiness()
            _write_json(Path(args.report_path).resolve(), result)
            if result["status"] == "READY_A8_BATCH4_PREREQUISITE":
                write_reference(Path(args.reference_path).resolve())
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["status"] == "READY_A8_BATCH4_PREREQUISITE" else 4
        if args.command == "run":
            code, result = run(
                input_path=args.input_path,
                target_path=args.target_path,
                log_root=Path(args.log_root).resolve(),
                source_revision=args.source_revision,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return code
    except (A8GateError, A8ContractError, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 74
    return 74


if __name__ == "__main__":
    raise SystemExit(main())
