"""Gate, inventory, resume planner, and fail-safe runner for Original scope26.

The active manifest is the only source of current Original entries.  This
module never imports a legacy model list to form the denominator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "custom_models/src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from benchmark_v2.original_scope26 import (  # noqa: E402
    CURRENT_MANIFEST_PATH,
    CURRENT_SCOPE26_ID,
    OriginalScope26Error,
    current_manifest_hash,
    current_model_config_hash,
    load_current_scope_manifest,
    resolve_source_revision,
)
from benchmark_v2.precision import expected_model_precision_identity  # noqa: E402
from benchmark_v2.registry import load_registry  # noqa: E402


CURRENT_MANIFEST = CURRENT_MANIFEST_PATH
CURRENT_RUN_MAP = (
    PROJECT_ROOT
    / "custom_models/docs/benchmark_v2/BATCH4/ORIGINAL_SCOPE26_RUN_ID_MAP.json"
)
CURRENT_EXCLUSIONS = (
    PROJECT_ROOT
    / "custom_models/docs/benchmark_v2/BATCH4/CURRENT_SCOPE_EXCLUSIONS.json"
)
RESULT_ROOT = PROJECT_ROOT / "custom_models/results/benchmark_v2_uniform_bs4"
AUDIT_ROOT = PROJECT_ROOT / "custom_models/logs/uniform_bs4/audit"
LOCK_PATH = PROJECT_ROOT / "custom_models/logs/uniform_bs4/original_scope26.lock"
HORIZONS = (3, 6, 10)
REQUIRED_METRICS = ("MAE", "RMSE", "R2", "Score")
EXCLUDED_IDS = {"segrnn", "msgnet"}
TRAINABLE_TYPE = "TRAINABLE"
EVALUATE_ONLY_TYPE = "EVALUATE_ONLY"


class Scope26GateError(ValueError):
    """Raised when current Original identity or readiness is unsafe."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> tuple[Any | None, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except FileNotFoundError:
        return None, "MISSING_FILE"
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"INVALID_JSON:{type(exc).__name__}:{exc}"


def _canonical_json_hash(payload: Any) -> str:
    material = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_text_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    return hashlib.sha256(text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")).hexdigest()


def _resolve_repo_file(relative_path: str, project_root: Path = PROJECT_ROOT) -> Path:
    root = project_root.resolve()
    candidate = (root / Path(relative_path)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise Scope26GateError(f"Source path escapes project root: {relative_path}") from exc
    if not candidate.is_file():
        raise Scope26GateError(f"Source closure file is missing: {relative_path}")
    return candidate


def _source_identity_from_manifest(
    entry: Mapping[str, Any], project_root: Path = PROJECT_ROOT
) -> dict[str, Any]:
    source = entry.get("source_identity")
    if not isinstance(source, Mapping):
        raise Scope26GateError(f"Missing source identity: {entry.get('model_id')}")
    paths = source.get("source_closure_paths")
    if not isinstance(paths, list) or not paths:
        raise Scope26GateError(f"Empty source closure: {entry.get('model_id')}")
    records = []
    for path in paths:
        resolved = _resolve_repo_file(str(path), project_root)
        records.append({"path": str(path).replace("\\", "/"), "sha256": _canonical_text_sha256(resolved)})
    records.sort(key=lambda row: row["path"])
    material = json.dumps(
        {"schema_version": source.get("source_identity_schema_version"), "files": records},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "source_identity_schema_version": source.get("source_identity_schema_version"),
        "source_closure_files": records,
        "canonical_combined_hash": hashlib.sha256(material).hexdigest(),
    }


def _expected_entry(manifest: Mapping[str, Any], model_id: str) -> Mapping[str, Any]:
    for entry in manifest.get("entries", []):
        if entry.get("model_id") == model_id:
            return entry
    raise Scope26GateError(f"Unknown current scope26 model: {model_id}")


def _load_run_map(path: Path = CURRENT_RUN_MAP) -> dict[str, Any]:
    payload, error = _read_json(path)
    if error or not isinstance(payload, dict):
        raise Scope26GateError(f"Current Original run map unavailable: {error or path}")
    return payload


def validate_manifest(
    manifest: Mapping[str, Any] | None = None,
    *,
    project_root: Path = PROJECT_ROOT,
    run_map: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail closed on scope, identity, precision, and run-map mismatches."""

    active = dict(manifest or load_current_scope_manifest())
    errors: list[str] = []
    if active.get("scope_id") != CURRENT_SCOPE26_ID:
        errors.append("SCOPE_ID_MISMATCH")
    if active.get("status") != "ACTIVE":
        errors.append("MANIFEST_NOT_ACTIVE")
    if active.get("benchmark_family") != "Original":
        errors.append("BENCHMARK_FAMILY_MISMATCH")
    if active.get("training_profile_id") != "uniform_train_batch4_v1":
        errors.append("TRAINING_PROFILE_ID_MISMATCH")
    if active.get("training_profile_hash") != "f58bbc161dfba0f00774879fdf2f78ec7c59a7faff9ef28a1733a2ed6835fe66":
        errors.append("TRAINING_PROFILE_HASH_MISMATCH")
    if active.get("benchmark_protocol_hash") != "0140d8774e2cc189a8bd99f1fc9c8a120b565265bf9a7c729ed1d47a0b1c069b":
        errors.append("BENCHMARK_PROTOCOL_HASH_MISMATCH")
    if active.get("output_root") != "custom_models/results/benchmark_v2_uniform_bs4":
        errors.append("OUTPUT_ROOT_MISMATCH")
    batch = active.get("batch_identity") or {}
    expected_batch = {
        "train_batch_size": 4,
        "val_batch_size": 4,
        "test_batch_size": 4,
        "gradient_accumulation_steps": 1,
        "effective_train_batch_size": 4,
    }
    if {key: batch.get(key) for key in expected_batch} != expected_batch:
        errors.append("BATCH4_IDENTITY_MISMATCH")
    if active.get("seed") != 2026 or active.get("lookback") != 144 or active.get("prediction_horizon") != 10:
        errors.append("FORECAST_IDENTITY_MISMATCH")
    if tuple(active.get("evaluation_horizons", ())) != HORIZONS:
        errors.append("EVALUATION_HORIZONS_MISMATCH")
    declared_counts = active.get("counts") or {}
    if declared_counts.get("total") != 26:
        errors.append("DECLARED_TOTAL_COUNT_MISMATCH")
    if declared_counts.get("trainable") != 24:
        errors.append("DECLARED_TRAINABLE_COUNT_MISMATCH")
    if declared_counts.get("evaluate_only") != 2:
        errors.append("DECLARED_EVALUATE_ONLY_COUNT_MISMATCH")
    if declared_counts.get("excluded") != 2:
        errors.append("DECLARED_EXCLUDED_COUNT_MISMATCH")

    entries = list(active.get("entries") or [])
    model_ids = [entry.get("model_id") for entry in entries]
    run_ids = [entry.get("run_id") for entry in entries]
    if len(entries) != 26:
        errors.append(f"ENTRY_COUNT_{len(entries)}_EXPECTED_26")
    if len(set(model_ids)) != len(model_ids):
        errors.append("DUPLICATE_MODEL_ID")
    if len(set(run_ids)) != len(run_ids):
        errors.append("DUPLICATE_RUN_ID")
    if [entry.get("ordinal") for entry in entries] != list(range(1, 27)):
        errors.append("ORDINALS_NOT_1_TO_26")
    if set(model_ids) & EXCLUDED_IDS:
        errors.append("EXCLUDED_MODEL_IN_ACTIVE_ENTRIES")
    trainable = [entry for entry in entries if entry.get("entry_type") == TRAINABLE_TYPE]
    evaluate_only = [entry for entry in entries if entry.get("entry_type") == EVALUATE_ONLY_TYPE]
    if len(trainable) != 24:
        errors.append(f"TRAINABLE_COUNT_{len(trainable)}_EXPECTED_24")
    if len(evaluate_only) != 2:
        errors.append(f"EVALUATE_ONLY_COUNT_{len(evaluate_only)}_EXPECTED_2")
    if {entry.get("model_id") for entry in evaluate_only} != {"persistence", "moving_average"}:
        errors.append("EVALUATE_ONLY_SET_MISMATCH")

    exclusion_ids = {item.get("model_id") for item in active.get("exclusions", [])}
    if exclusion_ids != EXCLUDED_IDS:
        errors.append("ACTIVE_EXCLUSION_SET_MISMATCH")
    for item in active.get("exclusions", []):
        if item.get("status") != "EXCLUDED_FROM_CURRENT_FORMAL_SCOPE":
            errors.append(f"EXCLUSION_STATUS_MISMATCH:{item.get('model_id')}")
        if item.get("reason") != "RESOURCE_REQUIREMENT_EXCEEDS_AVAILABLE_FORMAL_HARDWARE":
            errors.append(f"EXCLUSION_REASON_MISMATCH:{item.get('model_id')}")

    registry_ids = {item.canonical_id for item in load_registry().list()}
    for entry in entries:
        model_id = entry.get("model_id")
        if model_id not in registry_ids:
            errors.append(f"UNKNOWN_REGISTRY_MODEL:{model_id}")
        root = str(entry.get("output_root", "")).replace("\\", "/")
        if not root.startswith("custom_models/results/benchmark_v2_uniform_bs4/"):
            errors.append(f"INVALID_ENTRY_OUTPUT_ROOT:{model_id}")
        if entry.get("formal_training") != (entry.get("entry_type") == TRAINABLE_TYPE):
            errors.append(f"FORMAL_TRAINING_FLAG_MISMATCH:{model_id}")
        source = entry.get("source_identity") or {}
        if not isinstance(source.get("canonical_combined_hash"), str) or len(source["canonical_combined_hash"]) != 64:
            errors.append(f"SOURCE_HASH_MISSING:{model_id}")
        config = entry.get("model_config_identity") or {}
        if not isinstance(config.get("config_hash"), str) or len(config["config_hash"]) != 64:
            errors.append(f"CONFIG_HASH_MISSING:{model_id}")
        precision = entry.get("precision_identity") or {}
        if entry.get("entry_type") == EVALUATE_ONLY_TYPE:
            expected_precision = {
                "amp_enabled": False,
                "precision_policy": "cpu_baseline",
            }
        else:
            expected_precision = expected_model_precision_identity(
                str(model_id), "uniform_train_batch4_v1"
            )
        for key, value in expected_precision.items():
            if precision.get(key) != value:
                errors.append(f"PRECISION_IDENTITY_MISMATCH:{model_id}:{key}")
        try:
            actual_source = _source_identity_from_manifest(entry, project_root)
            if actual_source["canonical_combined_hash"] != source.get("canonical_combined_hash"):
                errors.append(f"SOURCE_CLOSURE_IDENTITY_MISMATCH:{model_id}")
            actual_config = current_model_config_hash(str(model_id))
            if actual_config != config.get("config_hash"):
                errors.append(f"MODEL_CONFIG_IDENTITY_MISMATCH:{model_id}")
        except (OSError, ValueError, KeyError, Scope26GateError) as exc:
            errors.append(f"MODEL_IDENTITY_UNAVAILABLE:{model_id}:{type(exc).__name__}")

    selected_map = dict(run_map or _load_run_map())
    if selected_map.get("scope_id") != CURRENT_SCOPE26_ID:
        errors.append("RUN_MAP_SCOPE_ID_MISMATCH")
    map_entries = list(selected_map.get("entries") or [])
    if len(map_entries) != 26:
        errors.append(f"RUN_MAP_ENTRY_COUNT_{len(map_entries)}_EXPECTED_26")
    manifest_triplets = {
        (entry.get("model_id"), entry.get("run_id"), entry.get("output_root"))
        for entry in entries
    }
    map_triplets = {
        (entry.get("model_id"), entry.get("run_id"), entry.get("output_root"))
        for entry in map_entries
    }
    if manifest_triplets != map_triplets:
        errors.append("RUN_MAP_MANIFEST_MISMATCH")

    return {
        "status": "PASS" if not errors else "BLOCKED_GLOBAL_IDENTITY",
        "scope_id": active.get("scope_id"),
        "counts": {
            "total": len(entries),
            "trainable": len(trainable),
            "evaluate_only": len(evaluate_only),
            "excluded": len(exclusion_ids),
        },
        "entries": entries,
        "errors": errors,
        "manifest_hash": _canonical_json_hash(active),
        "run_map_hash": _canonical_json_hash(selected_map),
    }


def _actual_source_closure_hashes(
    manifest: Mapping[str, Any], *, project_root: Path = PROJECT_ROOT
) -> dict[str, dict[str, Any]]:
    result = {}
    for entry in manifest.get("entries", []):
        result[str(entry["model_id"])] = _source_identity_from_manifest(
            entry, project_root
        )
    return result


def compute_original_freeze(
    manifest: Mapping[str, Any] | None = None,
    *,
    project_root: Path = PROJECT_ROOT,
    run_map: Mapping[str, Any] | None = None,
    source_revision: str | None = None,
) -> dict[str, Any]:
    """Compute an Original-only freeze; E5 and excluded source are absent."""

    active = dict(manifest or load_current_scope_manifest())
    selected_map = dict(run_map or _load_run_map())
    source_closures = _actual_source_closure_hashes(active, project_root=project_root)
    profile_path = project_root / "custom_models/src/benchmark_v2/training_profiles/uniform_train_batch4_v1.json"
    protocol_path = project_root / "custom_models/src/benchmark_v2/protocol/benchmark_protocol_v1.json"
    loss_path = project_root / "custom_models/src/benchmark_v2/losses.py"
    gate_path = project_root / "scripts/original_batch4_scope26_gate.py"
    launcher_path = project_root / "custom_models/docs/benchmark_v2/BATCH4/ORIGINAL_RUN_ALL_26_BATCH4_WINDOWS.ps1"
    revision = resolve_source_revision(
        project_root=project_root,
        explicit_source_revision=source_revision,
        manifest=active,
    )
    material = {
        "schema_version": "original_batch4_scope26_freeze_v1",
        "scope_id": active.get("scope_id"),
        "manifest_hash": _canonical_json_hash(active),
        "run_map_hash": _canonical_json_hash(selected_map),
        "entries": [
            {
                "ordinal": entry.get("ordinal"),
                "model_id": entry.get("model_id"),
                "entry_type": entry.get("entry_type"),
                "run_id": entry.get("run_id"),
                "output_root": entry.get("output_root"),
                "model_config_hash": (entry.get("model_config_identity") or {}).get("config_hash"),
                "source_closure_hash": source_closures[str(entry["model_id"])]["canonical_combined_hash"],
                "precision_identity": entry.get("precision_identity"),
            }
            for entry in active.get("entries", [])
        ],
        "batch4_profile": {
            "manifest_identity": active.get("batch_identity"),
            "profile_file_sha256": _sha256_file(profile_path),
            "profile_payload": json.loads(profile_path.read_text(encoding="utf-8")),
        },
        "benchmark_protocol": {
            "manifest_hash": active.get("benchmark_protocol_hash"),
            "protocol_file_sha256": _sha256_file(protocol_path),
            "dataset_identity": active.get("dataset_identity"),
            "lookback": active.get("lookback"),
            "prediction_horizon": active.get("prediction_horizon"),
            "evaluation_horizons": active.get("evaluation_horizons"),
        },
        "graph_identity": active.get("graph_identity"),
        "seed": active.get("seed"),
        "original_loss_identity": {
            **(active.get("loss_identity") or {}),
            "source_file_sha256": _canonical_text_sha256(loss_path),
        },
        "active_gate_revision": _canonical_text_sha256(gate_path),
        "active_launcher_revision": _canonical_text_sha256(launcher_path),
        "source_revision": revision,
    }
    return {
        "schema_version": "original_batch4_scope26_freeze_v1",
        "scope_id": CURRENT_SCOPE26_ID,
        "git_commit": revision["git_commit"],
        "source_revision_type": revision["source_revision_type"],
        "source_closure_hash": _canonical_json_hash(source_closures),
        "manifest_hash": material["manifest_hash"],
        "freeze_material": material,
        "freeze_hash": _canonical_json_hash(material),
    }


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _metric_payloads(run_dir: Path) -> tuple[dict[int, dict[str, Any]], list[str]]:
    metrics: dict[int, dict[str, Any]] = {}
    reasons: list[str] = []
    for horizon in HORIZONS:
        path = run_dir / f"metrics_eval_h{horizon}.json"
        payload, error = _read_json(path)
        if error:
            reasons.append(f"METRICS_H{horizon}_{error}")
            continue
        if not isinstance(payload, dict):
            reasons.append(f"METRICS_H{horizon}_NOT_OBJECT")
            continue
        if payload.get("horizon") != horizon:
            reasons.append(f"METRICS_H{horizon}_HORIZON_MISMATCH")
        for name in REQUIRED_METRICS:
            if name not in payload:
                reasons.append(f"METRICS_H{horizon}_{name}_MISSING")
            elif not _finite_number(payload[name]):
                reasons.append(f"METRICS_H{horizon}_{name}_NONFINITE_OR_BOOL")
        count = payload.get("valid_target_count")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            reasons.append(f"METRICS_H{horizon}_VALID_TARGET_COUNT_INVALID")
        metrics[horizon] = payload

    csv_path = run_dir / "metrics.csv"
    if not csv_path.is_file():
        reasons.append("METRICS_CSV_MISSING")
    else:
        try:
            with csv_path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
        except (OSError, UnicodeError, csv.Error) as exc:
            rows = []
            reasons.append(f"METRICS_CSV_INVALID:{type(exc).__name__}")
        by_horizon: dict[int, dict[str, str]] = {}
        for row in rows:
            try:
                h = int(row.get("horizon", ""))
            except (TypeError, ValueError):
                continue
            if h in HORIZONS:
                by_horizon[h] = row
        for horizon in HORIZONS:
            row = by_horizon.get(horizon)
            if row is None:
                reasons.append(f"METRICS_CSV_H{horizon}_MISSING")
                continue
            for name in REQUIRED_METRICS:
                raw = row.get(name)
                try:
                    parsed = float(raw)
                except (TypeError, ValueError):
                    parsed = None
                if not _finite_number(parsed):
                    reasons.append(f"METRICS_CSV_H{horizon}_{name}_INVALID")
                elif horizon in metrics:
                    json_value = metrics[horizon].get(name)
                    if _finite_number(json_value) and not math.isclose(
                        parsed, float(json_value), rel_tol=1e-9, abs_tol=1e-9
                    ):
                        reasons.append(f"METRICS_JSON_CSV_MISMATCH_H{horizon}_{name}")
            raw_count = row.get("valid_target_count")
            try:
                csv_count = int(raw_count)
            except (TypeError, ValueError):
                csv_count = None
            if horizon in metrics and csv_count != metrics[horizon].get("valid_target_count"):
                reasons.append(f"METRICS_JSON_CSV_MISMATCH_H{horizon}_VALID_TARGET_COUNT")
    return metrics, reasons


def _expected_precision(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    return entry.get("precision_identity") or {}


def _path_value_matches(actual: Any, expected_relative: str, output_root: Path) -> bool:
    if actual == expected_relative:
        return True
    if actual is None:
        return False
    try:
        return Path(str(actual)).resolve() == output_root.resolve()
    except (OSError, ValueError):
        return False


def inspect_run(
    manifest: Mapping[str, Any],
    entry: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    """Inspect one exact canonical directory without modifying it."""

    run_id = str(entry["run_id"])
    run_dir = (Path(output_root) / run_id).resolve()
    result: dict[str, Any] = {
        "model_id": entry.get("model_id"),
        "run_id": run_id,
        "run_dir": str(run_dir),
        "entry_type": entry.get("entry_type"),
        "found": run_dir.is_dir(),
        "ready": False,
        "reasons": [],
        "metrics": {},
        "metrics_complete": False,
        "metrics_finite": False,
        "execution_receipt_status": "MISSING",
        "checkpoint_sha256": None,
        "checkpoint_present": False,
        "checkpoint_nonempty": False,
    }
    if not run_dir.is_dir():
        result["reasons"] = ["RUN_DIRECTORY_MISSING"]
        return result

    status, status_error = _read_json(run_dir / "run_status.json")
    effective, effective_error = _read_json(run_dir / "effective_config.json")
    resolved, resolved_error = _read_json(run_dir / "resolved_config.json")
    artifact, artifact_error = _read_json(run_dir / "artifact_manifest.json")
    protocol, protocol_error = _read_json(run_dir / "protocol_check.json")
    data_signature, data_error = _read_json(run_dir / "data_signature.json")
    prediction, prediction_error = _read_json(run_dir / "prediction_metadata.json")
    for name, error in (
        ("run_status", status_error),
        ("effective_config", effective_error),
        ("resolved_config", resolved_error),
        ("artifact_manifest", artifact_error),
        ("protocol_check", protocol_error),
        ("data_signature", data_error),
        ("prediction_metadata", prediction_error),
    ):
        if error:
            result["reasons"].append(f"{name.upper()}_{error}")
    status = status if isinstance(status, dict) else {}
    effective = effective if isinstance(effective, dict) else {}
    resolved = resolved if isinstance(resolved, dict) else {}
    artifact = artifact if isinstance(artifact, dict) else {}
    protocol = protocol if isinstance(protocol, dict) else {}
    data_signature = data_signature if isinstance(data_signature, dict) else {}
    prediction = prediction if isinstance(prediction, dict) else {}
    result.update(
        {
            "status": status.get("status"),
            "exit_code": status.get("exit_code"),
            "run_mode": status.get("run_mode") or effective.get("run_mode"),
            "artifact_profile": status.get("artifact_profile") or effective.get("artifact_profile"),
            "formal_training": status.get("formal_training", effective.get("formal_training")),
            "precision_policy": effective.get("precision_policy"),
            "amp_enabled": effective.get("amp_enabled"),
            "train_batch_size": effective.get("train_batch_size"),
            "val_batch_size": effective.get("val_batch_size"),
            "test_batch_size": effective.get("test_batch_size"),
            "effective_train_batch_size": effective.get("effective_train_batch_size"),
            "protocol_hash": protocol.get("protocol_hash") or effective.get("protocol_hash"),
            "training_profile_id": effective.get("training_batch_profile_id"),
            "training_profile_hash": effective.get("training_batch_profile_hash"),
            "source_identity": (effective.get("provenance") or {}).get("base_model_source_closure_hash") or effective.get("source_closure_hash"),
            "model_config_hash": (effective.get("provenance") or {}).get("base_model_config_hash") or effective.get("model_config_hash"),
            "data_signature": data_signature,
            "effective_config": effective,
            "resolved_config": resolved,
        }
    )
    expected_formal = entry.get("entry_type") == TRAINABLE_TYPE
    if effective.get("model_id") != entry.get("model_id"):
        result["reasons"].append("MODEL_ID_MISMATCH")
    if result["run_mode"] != "formal":
        result["reasons"].append("RUN_MODE_NOT_FORMAL")
    if result["artifact_profile"] != entry.get("artifact_profile"):
        result["reasons"].append("ARTIFACT_PROFILE_MISMATCH")
    if result["formal_training"] is not expected_formal:
        result["reasons"].append("FORMAL_TRAINING_FLAG_MISMATCH")
    if status.get("status") != "COMPLETED":
        result["reasons"].append(f"STATUS_NOT_COMPLETED:{status.get('status')}")
    if status.get("exit_code") != 0:
        result["reasons"].append(f"EXIT_CODE_NOT_ZERO:{status.get('exit_code')}")
    if effective.get("run_id") not in (None, entry.get("run_id")):
        result["reasons"].append("RUN_ID_IDENTITY_MISMATCH")
    if not _path_value_matches(effective.get("output_root"), str(entry.get("output_root")), Path(output_root)):
        result["reasons"].append("OUTPUT_ROOT_IDENTITY_MISMATCH")
    expected_batch = {"train_batch_size": 4, "val_batch_size": 4, "test_batch_size": 4, "effective_train_batch_size": 4}
    for key, value in expected_batch.items():
        if effective.get(key) != value:
            result["reasons"].append(f"{key.upper()}_MISMATCH")
    if effective.get("gradient_accumulation_steps") != 1:
        result["reasons"].append("GRADIENT_ACCUMULATION_MISMATCH")
    if effective.get("training_batch_profile_id") != "uniform_train_batch4_v1":
        result["reasons"].append("TRAINING_PROFILE_ID_MISMATCH")
    if effective.get("training_batch_profile_hash") != "f58bbc161dfba0f00774879fdf2f78ec7c59a7faff9ef28a1733a2ed6835fe66":
        result["reasons"].append("TRAINING_PROFILE_HASH_MISMATCH")
    if result["protocol_hash"] != manifest.get("benchmark_protocol_hash"):
        result["reasons"].append("PROTOCOL_HASH_MISMATCH")
    if effective.get("seed") != 2026:
        result["reasons"].append("SEED_MISMATCH")
    if effective.get("seq_len") not in (None, 144):
        result["reasons"].append("LOOKBACK_MISMATCH")
    if effective.get("pred_len") not in (None, 10):
        result["reasons"].append("PREDICTION_HORIZON_MISMATCH")
    expected_precision = _expected_precision(entry)
    for key in ("amp_enabled", "precision_policy", "precision_resolution"):
        if key in expected_precision and effective.get(key) != expected_precision.get(key):
            result["reasons"].append(f"PRECISION_IDENTITY_MISMATCH:{key}")
        if key in expected_precision and artifact.get(key) not in (None, expected_precision.get(key)):
            result["reasons"].append(f"ARTIFACT_PRECISION_IDENTITY_MISMATCH:{key}")
    if data_signature.get("dataset_id") != (manifest.get("dataset_identity") or {}).get("dataset_id"):
        result["reasons"].append("DATASET_ID_MISMATCH")
    if data_signature.get("feature_order_hash") != (manifest.get("dataset_identity") or {}).get("feature_order_hash"):
        result["reasons"].append("FEATURE_ORDER_HASH_MISMATCH")
    if data_signature.get("node_count") != (manifest.get("dataset_identity") or {}).get("expected_node_count"):
        result["reasons"].append("NODE_COUNT_MISMATCH")
    provenance = effective.get("provenance") or {}
    expected_source = (entry.get("source_identity") or {}).get("canonical_combined_hash")
    expected_config = (entry.get("model_config_identity") or {}).get("config_hash")
    if provenance.get("active_scope_id") != CURRENT_SCOPE26_ID and effective.get("active_scope_id") != CURRENT_SCOPE26_ID:
        result["reasons"].append("ACTIVE_SCOPE_ID_MISMATCH")
    if result["source_identity"] != expected_source:
        result["reasons"].append("SOURCE_CLOSURE_IDENTITY_MISMATCH")
    if result["model_config_hash"] != expected_config:
        result["reasons"].append("MODEL_CONFIG_IDENTITY_MISMATCH")
    if protocol.get("base_benchmark_protocol_hash") not in (None, manifest.get("benchmark_protocol_hash")):
        result["reasons"].append("PROTOCOL_BASE_HASH_MISMATCH")

    metrics, metric_reasons = _metric_payloads(run_dir)
    result["metrics"] = metrics
    result["metrics_complete"] = not any("MISSING" in reason for reason in metric_reasons)
    result["metrics_finite"] = not any("NONFINITE" in reason or "INVALID" in reason for reason in metric_reasons)
    result["metrics_csv_consistent"] = not any("CSV" in reason or "JSON_CSV" in reason for reason in metric_reasons)
    result["reasons"].extend(metric_reasons)

    if expected_formal:
        for filename in ("best_checkpoint.pt", "last_checkpoint.pt", "train_log.csv"):
            if not (run_dir / filename).is_file():
                result["reasons"].append(f"{filename.upper()}_MISSING")
        checkpoint = run_dir / "best_checkpoint.pt"
        result["checkpoint_present"] = checkpoint.is_file()
        result["checkpoint_nonempty"] = checkpoint.is_file() and checkpoint.stat().st_size > 0
        if not result["checkpoint_nonempty"]:
            result["reasons"].append("BEST_CHECKPOINT_MISSING_OR_EMPTY")
        elif checkpoint.is_file():
            result["checkpoint_sha256"] = _sha256_file(checkpoint)
    else:
        baseline, baseline_error = _read_json(run_dir / "baseline_state.json")
        if baseline_error:
            result["reasons"].append(f"BASELINE_STATE_{baseline_error}")
        elif not isinstance(baseline, dict) or baseline.get("model_id") != entry.get("model_id"):
            result["reasons"].append("BASELINE_STATE_IDENTITY_MISMATCH")
        if prediction.get("source_checkpoint") is not None:
            result["reasons"].append("EVALUATE_ONLY_SOURCE_CHECKPOINT_NOT_NULL")

    receipt, receipt_error = _read_json(run_dir / "execution_receipt.json")
    if receipt_error:
        result["reasons"].append("EXECUTION_RECEIPT_MISSING")
    elif not isinstance(receipt, dict):
        result["reasons"].append("EXECUTION_RECEIPT_INVALID")
    else:
        result["execution_receipt_status"] = receipt.get("status")
        if receipt.get("status") != "SUCCESS":
            result["reasons"].append("EXECUTION_RECEIPT_NOT_SUCCESS")
        if receipt.get("model_id") != entry.get("model_id") or receipt.get("run_id") != entry.get("run_id"):
            result["reasons"].append("EXECUTION_RECEIPT_IDENTITY_MISMATCH")
        if receipt.get("exit_code") != 0:
            result["reasons"].append("EXECUTION_RECEIPT_EXIT_CODE_MISMATCH")
        result["git_commit"] = receipt.get("git_commit")
        result["source_revision_type"] = receipt.get("source_revision_type")
        result["source_closure_hash"] = receipt.get("source_closure_hash")
        result["manifest_hash"] = receipt.get("manifest_hash")
        if not receipt.get("git_commit") or not receipt.get("source_revision_type"):
            result["reasons"].append("EXECUTION_RECEIPT_SOURCE_REVISION_MISSING")
        if receipt.get("source_closure_hash") != expected_source:
            result["reasons"].append("EXECUTION_RECEIPT_SOURCE_CLOSURE_MISMATCH")

    result["ready"] = not result["reasons"]
    return result


def _find_model_directories(root: Path, model_id: str) -> list[Path]:
    found: list[Path] = []
    if not root.is_dir():
        return found
    for path in root.rglob("effective_config.json"):
        payload, error = _read_json(path)
        if not error and isinstance(payload, dict) and payload.get("model_id") == model_id:
            found.append(path.parent)
    return found


def _plan_action(
    manifest: Mapping[str, Any],
    entry: Mapping[str, Any],
    output_root: Path,
) -> tuple[str, dict[str, Any]]:
    inspected = inspect_run(manifest, entry, output_root)
    if inspected["ready"]:
        return "SKIP_COMPLETED_IDENTITY_MATCH", inspected
    if not inspected["found"]:
        other = [
            path
            for path in _find_model_directories(output_root, str(entry["model_id"]))
            if path.name != entry["run_id"]
        ]
        if other:
            inspected["found_noncanonical"] = [str(path) for path in other]
            inspected["reasons"].append("RUN_ID_RENAMED_OR_NONCANONICAL")
            return "BLOCK_EXISTING_IDENTITY_MISMATCH", inspected
        return "RUN_MISSING", inspected
    if any(
        reason.endswith("MISMATCH")
        or "IDENTITY" in reason
        or "PRECISION" in reason
        or "PROFILE" in reason
        or "DATASET" in reason
        for reason in inspected["reasons"]
    ):
        return "BLOCK_EXISTING_IDENTITY_MISMATCH", inspected
    return "ARCHIVE_INCOMPLETE_THEN_RUN", inspected


def _entry_output_root(
    manifest: Mapping[str, Any], entry: Mapping[str, Any], overall_root: Path
) -> Path:
    """Resolve an entry's group root from the active overall result root."""

    active_root = Path(str(manifest["output_root"]).replace("\\", "/"))
    entry_root = Path(str(entry["output_root"]).replace("\\", "/"))
    try:
        relative = entry_root.relative_to(active_root)
    except ValueError as exc:
        raise Scope26GateError(
            f"Entry output root is outside active result root: {entry['model_id']}"
        ) from exc
    return (Path(overall_root) / relative).resolve()


def build_plan(
    manifest: Mapping[str, Any] | None = None,
    *,
    output_root: Path | None = None,
) -> dict[str, Any]:
    active = dict(manifest or load_current_scope_manifest())
    validation = validate_manifest(active)
    if validation["errors"]:
        raise Scope26GateError(";".join(validation["errors"]))
    root = Path(output_root or PROJECT_ROOT / active["output_root"]).resolve()
    rows = []
    for entry in active["entries"]:
        action, inspected = _plan_action(
            active, entry, _entry_output_root(active, entry, root)
        )
        rows.append(
            {
                "ordinal": entry["ordinal"],
                "model_id": entry["model_id"],
                "run_id": entry["run_id"],
                "output_root": entry["output_root"],
                "entry_type": entry["entry_type"],
                "device": entry["device"],
                "action": action,
                "run_dir": inspected["run_dir"],
                "reasons": inspected.get("reasons", []),
                "ready": inspected.get("ready", False),
            }
        )
    return {
        "schema_version": "original_batch4_scope26_plan_v1",
        "scope_id": CURRENT_SCOPE26_ID,
        "generated_at": _utc_now(),
        "output_root": str(root),
        "counts": {
            "total": len(rows),
            "trainable": sum(row["entry_type"] == TRAINABLE_TYPE for row in rows),
            "evaluate_only": sum(row["entry_type"] == EVALUATE_ONLY_TYPE for row in rows),
            "ready": sum(row["ready"] for row in rows),
        },
        "entries": rows,
    }


def build_preflight_plan(
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe the current preflight denominator without executing GPU work."""

    active = dict(manifest or load_current_scope_manifest())
    validation = validate_manifest(active)
    if validation["errors"]:
        raise Scope26GateError(";".join(validation["errors"]))
    entries = [
        {
            "ordinal": entry["ordinal"],
            "model_id": entry["model_id"],
            "run_id": entry["run_id"],
            "entry_type": entry["entry_type"],
            "device": entry["device"],
            "preflight_action": (
                "TRAINABLE_EXACT_PREFLIGHT_EXPECTED"
                if entry["entry_type"] == TRAINABLE_TYPE
                else "EVALUATE_ONLY_CPU_PRECHECK"
            ),
            "precision_identity": entry["precision_identity"],
            "training_profile_id": active["training_profile_id"],
            "training_profile_hash": active["training_profile_hash"],
        }
        for entry in active["entries"]
    ]
    return {
        "schema_version": "original_scope26_preflight_plan_v1",
        "scope_id": CURRENT_SCOPE26_ID,
        "execution_performed": False,
        "gpu_preflight_performed": False,
        "counts": {
            "trainable_expected": sum(row["entry_type"] == TRAINABLE_TYPE for row in entries),
            "evaluate_only_expected": sum(row["entry_type"] == EVALUATE_ONLY_TYPE for row in entries),
            "total_current_scope": len(entries),
            "excluded_historical": 2,
        },
        "entries": entries,
    }


def _inventory_candidate_dirs(root: Path) -> set[Path]:
    result: set[Path] = set()
    if not root.is_dir():
        return result
    for path in root.rglob("run_status.json"):
        result.add(path.parent)
    for path in root.rglob("effective_config.json"):
        result.add(path.parent)
    return result


def _infer_model_id(run_dir: Path, root: Path, manifest: Mapping[str, Any]) -> str | None:
    for name in ("effective_config.json", "resolved_config.json", "model_summary.json"):
        payload, error = _read_json(run_dir / name)
        if not error and isinstance(payload, dict) and payload.get("model_id"):
            return str(payload["model_id"])
    for entry in manifest.get("entries", []):
        if run_dir.name == entry.get("run_id"):
            return str(entry["model_id"])
    for model_id in EXCLUDED_IDS:
        if model_id.casefold() in run_dir.name.casefold():
            return model_id
    return None


def _classify_inventory_dir(
    run_dir: Path, root: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    model_id = _infer_model_id(run_dir, root, manifest)
    entry = None
    if model_id:
        try:
            entry = _expected_entry(manifest, model_id)
        except Scope26GateError:
            entry = None
    relative = str(run_dir.resolve().relative_to(root.resolve())).replace("\\", "/")
    if model_id in EXCLUDED_IDS:
        category = "EXCLUDED_MODEL_HISTORICAL"
        reasons = ["EXCLUDED_FROM_CURRENT_FORMAL_SCOPE", "RESOURCE_REQUIREMENT_EXCEEDS_AVAILABLE_FORMAL_HARDWARE"]
    elif entry is None:
        category = "UNKNOWN_REQUIRES_MANUAL_REVIEW"
        reasons = ["MODEL_NOT_IN_CURRENT_SCOPE_OR_MODEL_ID_UNRESOLVED"]
    elif run_dir.name != entry.get("run_id"):
        category = "ORPHAN_NONCANONICAL_RUN_ID"
        reasons = ["DIRECTORY_NAME_IS_NOT_ACTIVE_CANONICAL_RUN_ID"]
    else:
        inspected = inspect_run(manifest, entry, run_dir.parent)
        reasons = list(inspected.get("reasons", []))
        if inspected.get("ready"):
            category = "CURRENT_SCOPE26_REUSABLE"
        elif inspected.get("status") == "FAILED" or inspected.get("exit_code") not in (None, 0):
            category = "CURRENT_SCOPE26_FAILED"
        elif any("MISMATCH" in reason or "IDENTITY" in reason for reason in reasons):
            category = "CURRENT_SCOPE26_IDENTITY_MISMATCH"
        else:
            category = "CURRENT_SCOPE26_INCOMPLETE"
    row: dict[str, Any] = {
        "model_id": model_id,
        "run_id": run_dir.name,
        "relative_path": relative,
        "classification": category,
        "status": None,
        "exit_code": None,
        "run_mode": None,
        "artifact_profile": None,
        "formal_training": None,
        "protocol_hash": None,
        "training_profile_id": None,
        "training_profile_hash": None,
        "train_batch_size": None,
        "val_batch_size": None,
        "test_batch_size": None,
        "effective_train_batch_size": None,
        "precision_policy": None,
        "amp_enabled": None,
        "source_identity_hash": None,
        "model_config_hash": None,
        "checkpoint_present": False,
        "checkpoint_nonempty": False,
        "checkpoint_sha256": None,
        "metrics_completeness": False,
        "metrics_finite": False,
        "metrics_csv_consistent": False,
        "execution_receipt_status": "MISSING",
        "current_scope_membership": bool(entry),
        "canonical_run_id": entry.get("run_id") if entry else None,
        "adopted": False,
        "rejection_reasons": reasons,
    }
    generic_status, _ = _read_json(run_dir / "run_status.json")
    generic_effective, _ = _read_json(run_dir / "effective_config.json")
    generic_protocol, _ = _read_json(run_dir / "protocol_check.json")
    generic_status = generic_status if isinstance(generic_status, dict) else {}
    generic_effective = generic_effective if isinstance(generic_effective, dict) else {}
    generic_protocol = generic_protocol if isinstance(generic_protocol, dict) else {}
    row.update(
        {
            "status": generic_status.get("status"),
            "exit_code": generic_status.get("exit_code"),
            "run_mode": generic_status.get("run_mode") or generic_effective.get("run_mode"),
            "artifact_profile": generic_status.get("artifact_profile") or generic_effective.get("artifact_profile"),
            "formal_training": generic_status.get("formal_training", generic_effective.get("formal_training")),
            "protocol_hash": generic_protocol.get("protocol_hash") or generic_effective.get("protocol_hash"),
            "training_profile_id": generic_effective.get("training_batch_profile_id"),
            "training_profile_hash": generic_effective.get("training_batch_profile_hash"),
            "train_batch_size": generic_effective.get("train_batch_size"),
            "val_batch_size": generic_effective.get("val_batch_size"),
            "test_batch_size": generic_effective.get("test_batch_size"),
            "effective_train_batch_size": generic_effective.get("effective_train_batch_size"),
            "precision_policy": generic_effective.get("precision_policy"),
            "amp_enabled": generic_effective.get("amp_enabled"),
            "source_identity_hash": (generic_effective.get("provenance") or {}).get("base_model_source_closure_hash") or generic_effective.get("source_closure_hash"),
            "model_config_hash": (generic_effective.get("provenance") or {}).get("base_model_config_hash") or generic_effective.get("model_config_hash"),
        }
    )
    generic_checkpoint = run_dir / "best_checkpoint.pt"
    row["checkpoint_present"] = generic_checkpoint.is_file()
    row["checkpoint_nonempty"] = generic_checkpoint.is_file() and generic_checkpoint.stat().st_size > 0
    if row["checkpoint_nonempty"]:
        row["checkpoint_sha256"] = _sha256_file(generic_checkpoint)
    generic_metrics, generic_metric_reasons = _metric_payloads(run_dir)
    row["metrics_completeness"] = not any("MISSING" in reason for reason in generic_metric_reasons)
    row["metrics_finite"] = not any("NONFINITE" in reason or "INVALID" in reason for reason in generic_metric_reasons)
    row["metrics_csv_consistent"] = not any("CSV" in reason or "JSON_CSV" in reason for reason in generic_metric_reasons)
    for horizon, metrics in generic_metrics.items():
        for metric in REQUIRED_METRICS:
            row[f"H{horizon}_{metric}"] = metrics.get(metric)
        row[f"H{horizon}_valid_target_count"] = metrics.get("valid_target_count")
    generic_receipt, generic_receipt_error = _read_json(run_dir / "execution_receipt.json")
    if not generic_receipt_error and isinstance(generic_receipt, dict):
        row["execution_receipt_status"] = generic_receipt.get("status")
    if entry and run_dir.name == entry.get("run_id"):
        inspected = inspect_run(manifest, entry, run_dir.parent)
        for key in (
            "status", "exit_code", "run_mode", "artifact_profile", "formal_training",
            "protocol_hash", "training_profile_id", "training_profile_hash",
            "train_batch_size", "val_batch_size", "test_batch_size",
            "effective_train_batch_size", "precision_policy", "amp_enabled",
            "source_identity", "model_config_hash", "checkpoint_present",
            "checkpoint_nonempty", "checkpoint_sha256", "metrics_complete",
            "metrics_finite", "metrics_csv_consistent", "execution_receipt_status",
        ):
            target = {
                "source_identity": "source_identity_hash",
                "metrics_complete": "metrics_completeness",
            }.get(key, key)
            if key in inspected:
                row[target] = inspected[key]
        for horizon, metrics in inspected.get("metrics", {}).items():
            for metric in REQUIRED_METRICS:
                row[f"H{horizon}_{metric}"] = metrics.get(metric)
            row[f"H{horizon}_valid_target_count"] = metrics.get("valid_target_count")
    else:
        for horizon in HORIZONS:
            for metric in REQUIRED_METRICS:
                row[f"H{horizon}_{metric}"] = None
            row[f"H{horizon}_valid_target_count"] = None
    return row


def build_result_inventory(
    manifest: Mapping[str, Any] | None = None,
    *,
    root: Path = RESULT_ROOT,
) -> dict[str, Any]:
    active = dict(manifest or load_current_scope_manifest())
    validation = validate_manifest(active)
    if validation["errors"]:
        raise Scope26GateError(";".join(validation["errors"]))
    rows = [
        _classify_inventory_dir(path, root, active)
        for path in sorted(_inventory_candidate_dirs(root), key=lambda value: str(value))
    ]
    counts = {category: sum(row["classification"] == category for row in rows) for category in (
        "CURRENT_SCOPE26_REUSABLE", "CURRENT_SCOPE26_INCOMPLETE", "CURRENT_SCOPE26_FAILED",
        "CURRENT_SCOPE26_IDENTITY_MISMATCH", "ORPHAN_NONCANONICAL_RUN_ID",
        "EXCLUDED_MODEL_HISTORICAL", "LEGACY_SCOPE27_OR_28", "SMOKE_NOT_FORMAL",
        "UNKNOWN_REQUIRES_MANUAL_REVIEW",
    )}
    by_current = {
        entry["model_id"]: next(
            (row for row in rows if row.get("model_id") == entry["model_id"] and row.get("run_id") == entry["run_id"]),
            {"classification": "RUN_MISSING"},
        )
        for entry in active["entries"]
    }
    return {
        "schema_version": "original_scope26_result_inventory_v1",
        "scope_id": CURRENT_SCOPE26_ID,
        "generated_at": _utc_now(),
        "root": str(root.resolve()),
        "scan_read_only": True,
        "counts": counts,
        "current_scope26_models": by_current,
        "runs": rows,
    }


def write_inventory(
    inventory: Mapping[str, Any],
    *,
    json_path: Path = AUDIT_ROOT / "original_scope26_result_inventory.json",
    csv_path: Path = AUDIT_ROOT / "original_scope26_result_inventory.csv",
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(inventory, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    rows = list(inventory.get("runs", []))
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _command_for_entry(entry: Mapping[str, Any], *, input_path: str | None, target_path: str | None) -> list[str]:
    command = "evaluate-only" if entry["entry_type"] == EVALUATE_ONLY_TYPE else "train"
    args = [
        sys.executable,
        "-m",
        "benchmark_v2.cli",
        command,
        "--model",
        str(entry["model_id"]),
        "--output-root",
        str((PROJECT_ROOT / entry["output_root"]).resolve()),
        "--run-id",
        str(entry["run_id"]),
        "--device",
        str(entry["device"]),
        "--training-profile",
        "uniform_train_batch4_v1",
        "--formal-scope-id",
        CURRENT_SCOPE26_ID,
    ]
    if input_path:
        args.extend(["--input-path", input_path])
    if target_path:
        args.extend(["--target-path", target_path])
    return args


def _write_execution_receipt(
    run_dir: Path,
    entry: Mapping[str, Any],
    *,
    command: Sequence[str],
    exit_code: int,
    started_at: str,
    finished_at: str,
    revision: Mapping[str, str],
    manifest_hash: str,
) -> None:
    path = run_dir / "execution_receipt.json"
    if path.exists():
        return
    checkpoint = run_dir / "best_checkpoint.pt"
    payload = {
        "schema_version": "original_scope26_execution_receipt_v1",
        "status": "SUCCESS" if exit_code == 0 else "FAILED",
        "scope_id": CURRENT_SCOPE26_ID,
        "model_id": entry["model_id"],
        "run_id": entry["run_id"],
        "output_root": entry["output_root"],
        "command": [str(value) for value in command],
        "started_at": started_at,
        "finished_at": finished_at,
        "exit_code": int(exit_code),
        "git_commit": revision["git_commit"],
        "source_revision_type": revision["source_revision_type"],
        "source_closure_hash": entry["source_identity"]["canonical_combined_hash"],
        "manifest_hash": manifest_hash,
        "checkpoint_sha256": _sha256_file(checkpoint) if checkpoint.is_file() else None,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _acquire_lock(path: Path = LOCK_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise Scope26GateError(f"CURRENT_SCOPE26_LOCK_EXISTS:{path}") from exc
    handle.write(json.dumps({"scope_id": CURRENT_SCOPE26_ID, "pid": os.getpid(), "started_at": _utc_now()}))
    handle.close()


def _release_lock(path: Path = LOCK_PATH) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def run_suite(
    manifest: Mapping[str, Any] | None = None,
    *,
    input_path: str | None = None,
    target_path: str | None = None,
    log_root: Path | None = None,
    source_revision: str | None = None,
) -> tuple[int, dict[str, Any]]:
    """Run a precomputed plan, preserving failures and continuing afterward."""

    active = dict(manifest or load_current_scope_manifest())
    validation = validate_manifest(active)
    if validation["errors"]:
        return 74, {"status": "BLOCKED_GLOBAL_IDENTITY", "errors": validation["errors"]}
    plan = build_plan(active)
    revision = resolve_source_revision(
        explicit_source_revision=source_revision, manifest=active
    )
    _acquire_lock()
    logs = Path(log_root or PROJECT_ROOT / "custom_models/logs/uniform_bs4/formal/original_scope26")
    logs.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    try:
        for row in plan["entries"]:
            entry = _expected_entry(active, row["model_id"])
            action = row["action"]
            if action == "SKIP_COMPLETED_IDENTITY_MATCH":
                results.append({"model_id": row["model_id"], "action": action, "exit_code": 0, "readiness": "READY"})
                continue
            if action != "RUN_MISSING":
                results.append({"model_id": row["model_id"], "action": action, "exit_code": 74, "readiness": "BLOCKED", "reasons": row["reasons"]})
                failures.append(results[-1])
                continue
            command = _command_for_entry(entry, input_path=input_path, target_path=target_path)
            log_path = logs / f"{entry['ordinal']:02d}_{entry['model_id']}.log"
            started_at = _utc_now()
            exit_code = 1
            error_type = None
            error_message = None
            traceback_tail = None
            try:
                with log_path.open("w", encoding="utf-8", newline="") as handle:
                    completed = subprocess.run(
                        command,
                        cwd=str(PROJECT_ROOT),
                        env={**os.environ, "PYTHONPATH": str(SOURCE_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")},
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                exit_code = int(completed.returncode)
            except Exception as exc:  # pragma: no cover - defensive launcher boundary
                error_type = type(exc).__name__
                error_message = str(exc)
                traceback_tail = traceback.format_exc().splitlines()[-20:]
            finished_at = _utc_now()
            run_dir = PROJECT_ROOT / entry["output_root"] / entry["run_id"]
            if exit_code == 0 and run_dir.is_dir():
                _write_execution_receipt(
                    run_dir,
                    entry,
                    command=command,
                    exit_code=exit_code,
                    started_at=started_at,
                    finished_at=finished_at,
                    revision=revision,
                    manifest_hash=validation["manifest_hash"],
                )
            evidence = {
                "model_id": entry["model_id"],
                "run_id": entry["run_id"],
                "action": action,
                "start": started_at,
                "end": finished_at,
                "exit_code": exit_code,
                "per_model_log": str(log_path),
                "run_directory": str(run_dir),
                "failure_artifact": None,
                "traceback_tail": traceback_tail,
                "error_type": error_type,
                "error_message": error_message,
                "oom": bool(error_message and "out of memory" in error_message.casefold()),
                "readiness": "READY" if exit_code == 0 else "FAILED",
            }
            if exit_code != 0:
                failure_path = logs / f"{entry['ordinal']:02d}_{entry['model_id']}.failure.json"
                evidence["failure_artifact"] = str(failure_path)
                failure_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
                failures.append(evidence)
                print(
                    f"failed_model={entry['model_id']} error_type={error_type or 'ChildProcessError'} "
                    f"error_message={error_message or f'child exit code {exit_code}'} log_path={log_path}",
                    file=sys.stderr,
                )
            results.append(evidence)
        readiness = build_readiness(active)
        if failures:
            final_status = "COMPLETED_WITH_FAILURES"
            code = 1
        elif readiness["status"] != "COMPLETED_READY_26_OF_26":
            final_status = "NOT_READY"
            code = 4
        else:
            final_status = "COMPLETED_READY_26_OF_26"
            code = 0
        report = {
            "schema_version": "original_scope26_run_status_v1",
            "scope_id": CURRENT_SCOPE26_ID,
            "status": final_status,
            "exit_code": code,
            "plan": plan,
            "results": results,
            "failures": failures,
            "readiness": readiness,
        }
        (logs / "final_status.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return code, report
    finally:
        _release_lock()


def build_readiness(
    manifest: Mapping[str, Any] | None = None,
    *,
    output_root: Path | None = None,
) -> dict[str, Any]:
    active = dict(manifest or load_current_scope_manifest())
    validation = validate_manifest(active)
    if validation["errors"]:
        raise Scope26GateError(";".join(validation["errors"]))
    root = Path(output_root or PROJECT_ROOT / active["output_root"]).resolve()
    entries = []
    for entry in active["entries"]:
        action, inspected = _plan_action(
            active, entry, _entry_output_root(active, entry, root)
        )
        entries.append(
            {
                "model_id": entry["model_id"],
                "run_id": entry["run_id"],
                "entry_type": entry["entry_type"],
                "action": action,
                "ready": inspected.get("ready", False),
                "reasons": inspected.get("reasons", []),
                "run_dir": inspected.get("run_dir"),
                "checkpoint_sha256": inspected.get("checkpoint_sha256"),
                "execution_receipt_status": inspected.get("execution_receipt_status"),
                "metrics_complete": inspected.get("metrics_complete"),
                "metrics_finite": inspected.get("metrics_finite"),
            }
        )
    ready = sum(row["ready"] for row in entries)
    blocked = any(row["action"] == "BLOCK_EXISTING_IDENTITY_MISMATCH" for row in entries)
    if ready == 26:
        status = "COMPLETED_READY_26_OF_26"
    elif blocked:
        status = "BLOCKED_GLOBAL_IDENTITY"
    else:
        status = "NOT_READY"
    return {
        "schema_version": "original_scope26_readiness_v1",
        "scope_id": CURRENT_SCOPE26_ID,
        "status": status,
        "require_complete": True,
        "counts": {"active_total": {"ready": ready, "expected": 26}, "trainable": {"ready": sum(row["ready"] for row in entries if row["entry_type"] == TRAINABLE_TYPE), "expected": 24}, "evaluate_only": {"ready": sum(row["ready"] for row in entries if row["entry_type"] == EVALUATE_ONLY_TYPE), "expected": 2}},
        "excluded_historical": 2,
        "output_root": str(root),
        "generated_at": _utc_now(),
        "entries": entries,
    }


def write_readiness(report: Mapping[str, Any], report_path: Path, evidence_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    evidence = {
        "schema_version": "original_scope26_evidence_manifest_v1",
        "scope_id": CURRENT_SCOPE26_ID,
        "status": report["status"],
        "require_complete": True,
        "entries": report["entries"],
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def aggregate(
    manifest: Mapping[str, Any] | None = None,
    *,
    output_root: Path | None = None,
    require_complete: bool,
) -> tuple[int, dict[str, Any]]:
    if not require_complete:
        return 2, {"status": "AGGREGATE_SKIPPED_NOT_COMPLETE", "reason": "--require-complete is mandatory."}
    active = dict(manifest or load_current_scope_manifest())
    readiness = build_readiness(active, output_root=output_root)
    if readiness["status"] != "COMPLETED_READY_26_OF_26":
        return 4, {"status": "AGGREGATE_SKIPPED_NOT_COMPLETE", "readiness": readiness}
    root = Path(output_root or PROJECT_ROOT / active["output_root"]).resolve()
    rows: list[dict[str, Any]] = []
    for entry in active["entries"]:
        inspected = inspect_run(
            active, entry, _entry_output_root(active, entry, root)
        )
        row = {"model_id": entry["model_id"], "model": entry["display_name"], "run_id": entry["run_id"], "training_mode": entry["entry_type"]}
        for horizon in HORIZONS:
            for metric in REQUIRED_METRICS:
                row[f"H{horizon}_{metric}"] = inspected["metrics"][horizon][metric]
        rows.append(row)
    path = root / "original_scope26_metrics.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return 0, {"status": "PASS", "scope_id": CURRENT_SCOPE26_ID, "row_count": len(rows), "path": str(path)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="original_batch4_scope26_gate")
    parser.add_argument("--manifest", default=str(CURRENT_MANIFEST))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate-manifest")
    sub.add_parser("preflight-plan")
    sub.add_parser("freeze")
    dry = sub.add_parser("dry-run")
    dry.add_argument("--output-root", default=str(RESULT_ROOT))
    inv = sub.add_parser("inventory")
    inv.add_argument("--root", default=str(RESULT_ROOT))
    inv.add_argument("--json-path", default=str(AUDIT_ROOT / "original_scope26_result_inventory.json"))
    inv.add_argument("--csv-path", default=str(AUDIT_ROOT / "original_scope26_result_inventory.csv"))
    ready = sub.add_parser("readiness")
    ready.add_argument("--output-root", default=str(RESULT_ROOT))
    ready.add_argument("--report-path", default=str(AUDIT_ROOT / "original_scope26_readiness.json"))
    ready.add_argument("--evidence-path", default=str(AUDIT_ROOT / "original_scope26_evidence_manifest.json"))
    aggregate_parser = sub.add_parser("aggregate")
    aggregate_parser.add_argument("--output-root", default=str(RESULT_ROOT))
    aggregate_parser.add_argument("--require-complete", action="store_true")
    run = sub.add_parser("run")
    run.add_argument("--input-path")
    run.add_argument("--target-path")
    run.add_argument("--source-revision")
    run.add_argument("--log-root", default=str(PROJECT_ROOT / "custom_models/logs/uniform_bs4/formal/original_scope26"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = load_current_scope_manifest(args.manifest)
        validation = validate_manifest(manifest)
        if args.command == "validate-manifest":
            print(json.dumps(validation, ensure_ascii=False, indent=2))
            return 0 if not validation["errors"] else 74
        if validation["errors"]:
            print(json.dumps({"status": "BLOCKED_GLOBAL_IDENTITY", "errors": validation["errors"]}, ensure_ascii=False, indent=2), file=sys.stderr)
            return 74
        if args.command == "freeze":
            print(json.dumps(compute_original_freeze(manifest), ensure_ascii=False, indent=2))
            return 0
        if args.command == "preflight-plan":
            print(json.dumps(build_preflight_plan(manifest), ensure_ascii=False, indent=2))
            return 0
        if args.command == "dry-run":
            plan = build_plan(manifest, output_root=Path(args.output_root))
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0
        if args.command == "inventory":
            inventory = build_result_inventory(manifest, root=Path(args.root))
            write_inventory(inventory, json_path=Path(args.json_path), csv_path=Path(args.csv_path))
            print(json.dumps(inventory, ensure_ascii=False, indent=2))
            return 0
        if args.command == "readiness":
            report = build_readiness(manifest, output_root=Path(args.output_root))
            write_readiness(report, Path(args.report_path), Path(args.evidence_path))
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["status"] == "COMPLETED_READY_26_OF_26" else 4
        if args.command == "aggregate":
            code, report = aggregate(manifest, output_root=Path(args.output_root), require_complete=args.require_complete)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return code
        if args.command == "run":
            code, report = run_suite(manifest, input_path=args.input_path, target_path=args.target_path, log_root=Path(args.log_root), source_revision=args.source_revision)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return code
    except (OSError, ValueError, KeyError, OriginalScope26Error, Scope26GateError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 74
    return 74


if __name__ == "__main__":
    raise SystemExit(main())
