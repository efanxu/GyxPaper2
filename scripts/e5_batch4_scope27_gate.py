from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "custom_models" / "src"
sys.path.insert(0, str(SRC_ROOT))

from benchmark_v2.artifacts import atomic_write_json, validate_run  # noqa: E402
from benchmark_v2.experiments.e5_common_loss.active_scope import (  # noqa: E402
    ActiveScopeError,
    active_scope_path,
    load_active_scope_pointer,
)
from benchmark_v2.experiments.e5_common_loss.a8_reference import (  # noqa: E402
    validate_a8_reference,
)
from benchmark_v2.experiments.e5_common_loss.loss_profile import (  # noqa: E402
    CLI_PROFILE_ID,
    get_profile_metadata,
)
from benchmark_v2.experiments.e5_common_loss.runner import (  # noqa: E402
    canonical_base_model_config_hash,
)
from benchmark_v2.experiments.e5_common_loss.scope27_contract import (  # noqa: E402
    E5_SCOPE27_EVALUATE_ONLY_MODELS,
    E5_SCOPE27_EXCLUDED_MODELS,
    E5_SCOPE27_ID,
    E5_SCOPE27_REFERENCE_MODELS,
    E5_SCOPE27_TRAINABLE_MODELS,
    TRAINING_PROFILE_ID,
)
from benchmark_v2.protocol import load_protocol  # noqa: E402
from benchmark_v2.registry import load_registry  # noqa: E402
from benchmark_v2.model_source_identity import (  # noqa: E402
    MODEL_SOURCE_IDENTITY_SCHEMA_VERSION,
    ModelSourceIdentityError,
    canonical_model_source_identity,
)
from benchmark_v2.precision import expected_model_precision_identity  # noqa: E402
from benchmark_v2.training_profiles import load_training_profile  # noqa: E402


DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "custom_models"
    / "docs"
    / "benchmark_v2"
    / "E5"
    / "E5_SCOPE27_VARIANT_MANIFEST.json"
)
EXPECTED_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "custom_models"
    / "results"
    / "benchmark_v2_uniform_bs4"
    / "common_loss_architecture_seed2026"
)
LEGACY_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "custom_models"
    / "results"
    / "benchmark_v2"
    / "common_loss_architecture_seed2026"
)
AUDIT_ROOT = PROJECT_ROOT / "custom_models/logs/uniform_bs4/audit/e5_scope27"
LOCK_PATH = PROJECT_ROOT / "custom_models/logs/uniform_bs4/e5_scope27.lock"
ARCHIVE_DIRECTORY = ".e5_scope27_archived_attempts"
QUARANTINE_DIRECTORY = ".e5_scope27_quarantine"
INTENT_DIRECTORY = ".intents"
EXECUTION_RECEIPT_SCHEMA_VERSION = "e5_scope27_execution_receipt_v1"
LOCK_SCHEMA_VERSION = "e5_scope27_lock_v2"
REQUIRED_METRICS = ("Score", "MAE", "RMSE", "R2")
HORIZONS = (3, 6, 10)
SOURCE_IDENTITY_SCHEMA_VERSION = MODEL_SOURCE_IDENTITY_SCHEMA_VERSION
TRAINABLE_ENTRY_TYPE = "TRAIN_COMMON_LOSS"
EVALUATE_ONLY_ENTRY_TYPE = "EVALUATE_ONLY_COMMON_LOSS_DIAGNOSTIC"
REFERENCE_ENTRY_TYPE = "REFERENCE_ONLY_FORMAL_A8"
PREFLIGHT_SHAPE = {"B": 4, "T": 144, "N": 134, "C": 16, "H": 10}


class ScopeGateError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def combined_hash(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _output_root_matches(value: Any, expected_relative: str) -> bool:
    if value == expected_relative:
        return True
    if value is None:
        return False
    try:
        return Path(str(value)).resolve() == (PROJECT_ROOT / expected_relative).resolve()
    except (OSError, ValueError):
        return False


def _load_run_map() -> dict[str, Any]:
    pointer = load_active_scope_pointer()
    payload = load_json(active_scope_path(pointer, "run_id_map"))
    if not isinstance(payload, dict):
        raise ScopeGateError("Active E5 run map must be a JSON object.")
    if payload.get("scope_id") != E5_SCOPE27_ID:
        raise ScopeGateError("Active E5 run map scope mismatch.")
    if payload.get("output_root") != (
        "custom_models/results/benchmark_v2_uniform_bs4/common_loss_architecture_seed2026"
    ):
        raise ScopeGateError("Active E5 run map output root mismatch.")
    entries = payload.get("entries")
    if not isinstance(entries, list) or len(entries) != 27:
        raise ScopeGateError("Active E5 run map must contain exactly 27 entries.")
    return payload


def _source_revision() -> dict[str, str]:
    completed = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        raise ScopeGateError("Cannot resolve the current Git source revision.")
    return {"git_commit": completed.stdout.strip(), "source_revision_type": "git"}


def compute_e5_freeze(
    manifest: Mapping[str, Any] | None = None,
    *,
    run_map: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute the E5-only deterministic freeze material."""

    active = dict(manifest or load_manifest(DEFAULT_MANIFEST))
    validate_manifest(active)
    selected_map = dict(run_map or _load_run_map())
    pointer = load_active_scope_pointer()
    source_closures = {
        str(entry["model_id"]): canonical_model_source_identity(
            str(entry["model_id"])
        )["canonical_combined_hash"]
        for entry in active["entries"]
        if entry["entry_type"] != REFERENCE_ENTRY_TYPE
    }
    a8 = validate_a8_reference(training_profile=TRAINING_PROFILE_ID)
    reference_identity = {
        key: a8.get(key)
        for key in (
            "reference_id",
            "source_relative_path",
            "source_checkpoint_sha256",
            "source_metrics_sha256",
            "source_config_sha256",
            "source_protocol_evidence_sha256",
            "source_protocol_hash",
            "source_loss_id",
            "retrained_in_e5",
            "checkpoint_copied",
            "metrics_copied",
            "status",
        )
    }
    material = {
        "schema_version": "e5_scope27_freeze_v1",
        "scope_id": E5_SCOPE27_ID,
        "active_pointer_hash": canonical_hash(pointer),
        "manifest_hash": canonical_hash(active),
        "run_map_hash": canonical_hash(selected_map),
        "readiness_policy_hash": sha256_file(active_scope_path(pointer, "readiness_policy")),
        "gate_hash": sha256_file(active_scope_path(pointer, "gate_source")),
        "linux_launcher_hash": sha256_file(active_scope_path(pointer, "linux_launcher")),
        "autoshutdown_launcher_hash": sha256_file(
            active_scope_path(pointer, "linux_autoshutdown_launcher")
        ),
        "source_revision": _source_revision(),
        "training_profile_id": active["training_profile_id"],
        "training_profile_hash": active["training_profile_hash"],
        "benchmark_protocol_hash": active["benchmark_protocol_hash"],
        "loss_identity": active.get("loss_identity"),
        "dataset_identity": active.get("dataset_identity"),
        "graph_identity": active.get("graph_identity"),
        "model_config_hashes": {
            entry["model_id"]: entry.get("base_model_config_hash")
            for entry in active["entries"]
            if entry["entry_type"] != REFERENCE_ENTRY_TYPE
        },
        "source_closure_hashes": source_closures,
        "precision_identities": {
            entry["model_id"]: entry.get("precision_identity")
            for entry in active["entries"]
            if entry["entry_type"] != REFERENCE_ENTRY_TYPE
        },
        "a8_reference_identity": reference_identity,
        "current_batch4_output_root": active["output_root"],
        "forbidden_inputs": [
            "Batch32",
            "scope29",
            "SegRNN",
            "MSGNet",
            "Original result directories",
            "Transformer retry2",
        ],
    }
    return {
        "schema_version": material["schema_version"],
        "scope_id": E5_SCOPE27_ID,
        "git_commit": material["source_revision"]["git_commit"],
        "source_revision_type": material["source_revision"]["source_revision_type"],
        "manifest_hash": material["manifest_hash"],
        "run_map_hash": material["run_map_hash"],
        "freeze_material": material,
        "freeze_hash": canonical_hash(material),
    }


def load_manifest(path: str | Path) -> dict[str, Any]:
    pointer = load_active_scope_pointer()
    manifest_path = Path(path).resolve()
    expected_manifest = active_scope_path(pointer, "manifest").resolve()
    if manifest_path != expected_manifest:
        raise ScopeGateError(
            "The active scope pointer does not authorize this manifest; "
            "legacy/superseded manifests are not current scope27 inputs."
        )
    payload = load_json(manifest_path)
    if not isinstance(payload, dict):
        raise ScopeGateError("Manifest root must be a JSON object.")
    validate_manifest(payload)
    return payload


def _entry_sets(manifest: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    entries = list(manifest["entries"])
    return {
        "trainable": tuple(
            row["model_id"]
            for row in entries
            if row["entry_type"] == "TRAIN_COMMON_LOSS"
        ),
        "evaluate_only": tuple(
            row["model_id"]
            for row in entries
            if row["entry_type"] == "EVALUATE_ONLY_COMMON_LOSS_DIAGNOSTIC"
        ),
        "reference": tuple(
            row["model_id"]
            for row in entries
            if row["entry_type"] == "REFERENCE_ONLY_FORMAL_A8"
        ),
    }


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    run_map: Mapping[str, Any] | None = None,
) -> None:
    if manifest.get("scope_id") != E5_SCOPE27_ID:
        raise ScopeGateError("Active E5 scope identity mismatch.")
    if manifest.get("training_profile_id") != TRAINING_PROFILE_ID:
        raise ScopeGateError("Training profile identity mismatch.")
    if manifest.get("cli_profile_id") != CLI_PROFILE_ID:
        raise ScopeGateError("E5 CLI profile identity mismatch.")
    if manifest.get("output_root") != (
        "custom_models/results/benchmark_v2_uniform_bs4/common_loss_architecture_seed2026"
    ):
        raise ScopeGateError("Formal output root mismatch.")
    if manifest.get("legacy_output_root") != (
        "custom_models/results/benchmark_v2/common_loss_architecture_seed2026"
    ):
        raise ScopeGateError("Legacy E5 output root declaration is missing.")
    if manifest.get("legacy_output_policy") != (
        "LEGACY_OR_HISTORICAL_READ_ONLY_NOT_CURRENT_BATCH4_OUTPUT"
    ):
        raise ScopeGateError("Legacy E5 output root policy is not read-only.")
    if manifest.get("source_identity_schema_version") != SOURCE_IDENTITY_SCHEMA_VERSION:
        raise ScopeGateError("Model source identity schema mismatch.")
    entries = list(manifest.get("entries", []))
    counts = manifest.get("counts", {})
    expected_counts = {
        "trainable": 24,
        "evaluate_only": 2,
        "a8_reference": 1,
        "total_evidence": 27,
    }
    if counts != expected_counts or len(entries) != 27:
        raise ScopeGateError(f"E5 scope27 count mismatch: {counts}")
    entry_sets = _entry_sets(manifest)
    expected_sets = {
        "trainable": E5_SCOPE27_TRAINABLE_MODELS,
        "evaluate_only": E5_SCOPE27_EVALUATE_ONLY_MODELS,
        "reference": E5_SCOPE27_REFERENCE_MODELS,
    }
    if entry_sets != expected_sets:
        raise ScopeGateError(
            f"E5 scope27 entry order/content mismatch: {entry_sets}"
        )
    model_ids = [row["model_id"] for row in entries]
    run_ids = [row["e5_run_id"] for row in entries]
    if len(set(model_ids)) != 27 or len(set(run_ids)) != 27:
        raise ScopeGateError("Model ids and run ids must be unique.")
    if set(model_ids) & set(E5_SCOPE27_EXCLUDED_MODELS):
        raise ScopeGateError("Excluded model is present in the active manifest.")
    if [row.get("ordinal") for row in entries] != list(range(1, 28)):
        raise ScopeGateError("Manifest ordinals must be exactly 1..27.")
    if entries[-1].get("e5_run_id") != "STMGPrompt_A8_loss_msa_hybrid_bs4_seed2026_reference":
        raise ScopeGateError("The active E5 reference id is not the Batch4 A8 id.")
    if manifest.get("a8_reference_identity", {}).get("reference_id") != entries[-1].get("e5_run_id"):
        raise ScopeGateError("A8 reference identity does not match the active entry.")
    for entry in entries[:-1]:
        if entry.get("output_root") != manifest.get("output_root"):
            raise ScopeGateError(
                f"E5 entry output root mismatch: {entry.get('model_id')}"
            )

    protocol = load_protocol()
    if manifest.get("benchmark_protocol_hash") != protocol.protocol_hash:
        raise ScopeGateError("Benchmark Protocol identity mismatch.")
    profile = load_training_profile(TRAINING_PROFILE_ID)
    if profile is None:
        raise ScopeGateError("Frozen batch4 training profile is unavailable.")
    identity = profile.identity()
    for key, expected in (
        ("training_profile_hash", profile.profile_hash),
        ("train_batch_size", 4),
        ("val_batch_size", 4),
        ("test_batch_size", 4),
    ):
        actual = (
            manifest.get("batch_identity", {}).get(key)
            if key.endswith("_batch_size")
            else manifest.get(key)
        )
        if actual != expected:
            raise ScopeGateError(
                f"Frozen batch4 identity mismatch for {key}: {actual!r}"
            )
    if any(identity[key] != 4 for key in (
        "train_batch_size",
        "val_batch_size",
        "test_batch_size",
    )):
        raise ScopeGateError("The active training profile is not batch=4.")

    loss = get_profile_metadata(CLI_PROFILE_ID)
    expected_loss = manifest.get("loss_identity", {})
    for key in (
        "loss_id",
        "loss_source_hash",
        "loss_profile_hash",
        "e5_common_loss_protocol_hash",
    ):
        if expected_loss.get(key) != loss[key]:
            raise ScopeGateError(f"E5 loss identity mismatch for {key}.")

    registry = load_registry()
    for entry in entries:
        if entry["model_id"] in E5_SCOPE27_REFERENCE_MODELS:
            continue
        try:
            source_identity = canonical_model_source_identity(entry["model_id"])
        except (ModelSourceIdentityError, KeyError, OSError) as exc:
            raise ScopeGateError(
                f"Model source closure cannot be validated: {entry['model_id']}: {exc}"
            ) from exc
        if (
            entry.get("base_model_source_hash")
            != source_identity["canonical_combined_hash"]
            or entry.get("base_model_source_closure_hash")
            != source_identity["canonical_combined_hash"]
        ):
            raise ScopeGateError(
                f"Model source identity mismatch: {entry['model_id']}"
            )
        stored_identity = entry.get("source_identity")
        if not isinstance(stored_identity, Mapping):
            raise ScopeGateError(
                f"Model source closure manifest is missing: {entry['model_id']}"
            )
        if stored_identity.get("source_identity_schema_version") != SOURCE_IDENTITY_SCHEMA_VERSION:
            raise ScopeGateError(
                f"Model source closure schema mismatch: {entry['model_id']}"
            )
        if stored_identity.get("canonical_combined_hash") != source_identity[
            "canonical_combined_hash"
        ] or stored_identity.get("source_closure_files") != source_identity[
            "source_closure_files"
        ]:
            raise ScopeGateError(
                f"Model source closure records mismatch: {entry['model_id']}"
            )
        if (
            canonical_base_model_config_hash(entry["model_id"])
            != entry["base_model_config_hash"]
        ):
            raise ScopeGateError(
                f"Model config identity mismatch: {entry['model_id']}"
            )
        if entry.get("precision_identity") != expected_model_precision_identity(
            entry["model_id"], TRAINING_PROFILE_ID
        ):
            raise ScopeGateError(
                f"Precision identity mismatch: {entry['model_id']}"
            )
    selected_map = dict(run_map or _load_run_map())
    map_entries = list(selected_map.get("entries") or [])
    manifest_triplets = {
        (
            row.get("model_id"),
            row.get("e5_run_id"),
            row.get("entry_type"),
            row.get("output_root"),
        )
        for row in entries
    }
    map_triplets = {
        (
            row.get("model_id"),
            row.get("e5_run_id"),
            row.get("entry_type"),
            row.get("output_root"),
        )
        for row in map_entries
    }
    if manifest_triplets != map_triplets:
        raise ScopeGateError("E5 run-map and manifest entries differ.")


def _entry(manifest: Mapping[str, Any], model_id: str) -> Mapping[str, Any]:
    for row in manifest["entries"]:
        if row["model_id"] == model_id:
            return row
    raise ScopeGateError(f"Model is not in current E5 scope: {model_id}")


def _metric_payloads(paths: list[Path]) -> tuple[dict[int, dict[str, Any]], list[str]]:
    reasons: list[str] = []
    payloads: dict[int, dict[str, Any]] = {}
    for horizon, path in zip(HORIZONS, paths):
        if not path.is_file():
            reasons.append(f"MISSING:{path.name}")
            continue
        payload = load_json(path)
        if not isinstance(payload, dict):
            reasons.append(f"NOT_OBJECT:{path.name}")
            continue
        if payload.get("horizon") != horizon:
            reasons.append(f"HORIZON_MISMATCH:{path.name}")
        for key in REQUIRED_METRICS:
            value = payload.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                reasons.append(f"NON_FINITE_OR_MISSING:{path.name}:{key}")
        valid_target_count = payload.get("valid_target_count")
        if (
            isinstance(valid_target_count, bool)
            or not isinstance(valid_target_count, int)
            or valid_target_count <= 0
        ):
            reasons.append(
                f"INVALID_VALID_TARGET_COUNT:{path.name}:valid_target_count"
            )
        payloads[horizon] = payload
    return payloads, reasons


def _metrics_bundle_hash(run_dir: Path) -> str | None:
    paths = [
        *(run_dir / f"metrics_eval_h{horizon}.json" for horizon in HORIZONS),
        run_dir / "metrics.csv",
    ]
    if not all(path.is_file() for path in paths):
        return None
    records = [
        {"path": path.name, "sha256": sha256_file(path)}
        for path in sorted(paths, key=lambda item: item.name)
    ]
    return hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _metrics_csv_reasons(run_dir: Path, payloads: Mapping[int, Mapping[str, Any]]) -> list[str]:
    path = run_dir / "metrics.csv"
    if not path.is_file():
        return ["METRICS_CSV_MISSING"]
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
    except (OSError, UnicodeError, csv.Error) as exc:
        return [f"METRICS_CSV_INVALID:{type(exc).__name__}"]
    by_horizon: dict[int, Mapping[str, str]] = {}
    for row in rows:
        try:
            horizon = int(row.get("horizon", ""))
        except (TypeError, ValueError):
            continue
        by_horizon[horizon] = row
    reasons: list[str] = []
    for horizon in HORIZONS:
        row = by_horizon.get(horizon)
        payload = payloads.get(horizon)
        if row is None or payload is None:
            reasons.append(f"METRICS_CSV_H{horizon}_MISSING")
            continue
        for name in (*REQUIRED_METRICS, "valid_target_count"):
            raw = row.get(name)
            if name == "valid_target_count":
                try:
                    parsed: Any = int(raw)
                except (TypeError, ValueError):
                    parsed = None
                if parsed != payload.get(name):
                    reasons.append(f"METRICS_JSON_CSV_MISMATCH_H{horizon}_{name}")
                continue
            try:
                parsed = float(raw)
            except (TypeError, ValueError):
                parsed = None
            expected = payload.get(name)
            if (
                parsed is None
                or not math.isfinite(parsed)
                or not isinstance(expected, (int, float))
                or isinstance(expected, bool)
                or not math.isclose(parsed, float(expected), rel_tol=1e-9, abs_tol=1e-9)
            ):
                reasons.append(f"METRICS_JSON_CSV_MISMATCH_H{horizon}_{name}")
    return reasons


def _base_run_result(
    entry: Mapping[str, Any], output_root: Path
) -> tuple[dict[str, Any], Path]:
    run_dir = output_root / entry["e5_run_id"]
    return (
        {
            "entry_id": entry["entry_id"],
            "model_id": entry["model_id"],
            "entry_type": entry["entry_type"],
            "run_id": entry["e5_run_id"],
            "run_dir": str(run_dir),
            "found": run_dir.is_dir(),
            "ready": False,
            "reasons": [],
        },
        run_dir,
    )


def _load_common_run(
    manifest: Mapping[str, Any],
    entry: Mapping[str, Any],
    output_root: Path,
    *,
    expected_artifact_profile: str,
    expected_formal_training: bool,
) -> tuple[dict[str, Any], Path, dict[str, Any] | None]:
    result, run_dir = _base_run_result(entry, output_root)
    if not run_dir.is_dir():
        result["reasons"].append("RUN_NOT_FOUND")
        return result, run_dir, None
    try:
        status = load_json(run_dir / "run_status.json")
        effective = load_json(run_dir / "effective_config.json")
        data_signature = load_json(run_dir / "data_signature.json")
        protocol_check = load_json(run_dir / "protocol_check.json")
        artifact_manifest = load_json(run_dir / "artifact_manifest.json")
        validate_run(
            run_dir,
            expected_protocol_hash=manifest["benchmark_protocol_hash"],
            expected_training_batch_profile_id=TRAINING_PROFILE_ID,
            expected_training_batch_profile_hash=manifest[
                "training_profile_hash"
            ],
        )
    except Exception as exc:
        result["reasons"].append(
            f"ARTIFACT_INTEGRITY:{type(exc).__name__}:{exc}"
        )
        return result, run_dir, None

    provenance = effective.get("provenance", {})
    stored_identity = entry.get("source_identity", {})
    expected_precision = entry.get("precision_identity", {})
    run_map = _load_run_map()
    current_freeze = compute_e5_freeze(manifest, run_map=run_map)
    pointer = load_active_scope_pointer()
    pointer_hash = canonical_hash(pointer)
    checks = (
        (status.get("status") == "COMPLETED", "NOT_COMPLETED"),
        (status.get("exit_code") == 0, "EXIT_CODE_NOT_ZERO"),
        (status.get("run_mode") == "formal", "NOT_FORMAL"),
        (
            status.get("artifact_profile") == expected_artifact_profile,
            "ARTIFACT_PROFILE_MISMATCH",
        ),
        (
            status.get("formal_training") is expected_formal_training,
            "FORMAL_TRAINING_FLAG_MISMATCH",
        ),
        (effective.get("model_id") == entry["model_id"], "MODEL_ID_MISMATCH"),
        (effective.get("run_id") == entry["e5_run_id"], "RUN_ID_MISMATCH"),
        (_output_root_matches(effective.get("output_root"), manifest["output_root"]), "OUTPUT_ROOT_MISMATCH"),
        (
            effective.get("artifact_profile") == expected_artifact_profile,
            "EFFECTIVE_ARTIFACT_PROFILE_MISMATCH",
        ),
        (
            effective.get("formal_training") is expected_formal_training,
            "EFFECTIVE_FORMAL_TRAINING_FLAG_MISMATCH",
        ),
        (
            effective.get("experiment_profile_id")
            == manifest["experiment_profile_id"],
            "E5_PROFILE_MISMATCH",
        ),
        (effective.get("training_batch_profile_id") == TRAINING_PROFILE_ID, "TRAINING_PROFILE_ID_MISMATCH"),
        (effective.get("training_batch_profile_hash") == manifest["training_profile_hash"], "TRAINING_PROFILE_HASH_MISMATCH"),
        (effective.get("train_batch_size") == 4, "TRAIN_BATCH_SIZE_MISMATCH"),
        (effective.get("val_batch_size") == 4, "VAL_BATCH_SIZE_MISMATCH"),
        (effective.get("test_batch_size") == 4, "TEST_BATCH_SIZE_MISMATCH"),
        (effective.get("effective_train_batch_size") == 4, "EFFECTIVE_BATCH_SIZE_MISMATCH"),
        (effective.get("gradient_accumulation_steps") == 1, "GRADIENT_ACCUMULATION_MISMATCH"),
        (effective.get("seq_len") == 144, "LOOKBACK_MISMATCH"),
        (effective.get("pred_len") == 10, "PREDICTION_HORIZON_MISMATCH"),
        (
            provenance.get("active_scope_id") == E5_SCOPE27_ID,
            "ACTIVE_SCOPE_ID_MISMATCH",
        ),
        (
            provenance.get("base_model_config_hash")
            == entry["base_model_config_hash"],
            "BASE_MODEL_CONFIG_HASH_MISMATCH",
        ),
        (
            provenance.get("base_model_source_closure_hash")
            == entry["base_model_source_hash"],
            "BASE_MODEL_SOURCE_HASH_MISMATCH",
        ),
        (
            provenance.get("base_model_source_identity_schema_version")
            == SOURCE_IDENTITY_SCHEMA_VERSION,
            "BASE_MODEL_SOURCE_SCHEMA_MISMATCH",
        ),
        (
            provenance.get("base_model_source_closure_files")
            == stored_identity.get("source_closure_files"),
            "BASE_MODEL_SOURCE_CLOSURE_MISMATCH",
        ),
        (
            effective.get("loss", {}).get("id")
            == manifest["loss_identity"]["loss_id"],
            "LOSS_ID_MISMATCH",
        ),
        (
            effective.get("loss", {}).get("source_hash")
            == manifest["loss_identity"]["loss_source_hash"],
            "LOSS_SOURCE_HASH_MISMATCH",
        ),
        (
            effective.get("loss", {}).get("profile_hash")
            == manifest["loss_identity"]["loss_profile_hash"],
            "LOSS_PROFILE_HASH_MISMATCH",
        ),
        (
            provenance.get("e5_common_loss_protocol_hash")
            == manifest["loss_identity"]["e5_common_loss_protocol_hash"],
            "E5_PROTOCOL_HASH_MISMATCH",
        ),
        (
            protocol_check.get("protocol_hash")
            == manifest["benchmark_protocol_hash"],
            "BENCHMARK_PROTOCOL_HASH_MISMATCH",
        ),
        (artifact_manifest.get("model_id") == entry["model_id"], "ARTIFACT_MODEL_ID_MISMATCH"),
        (
            artifact_manifest.get("artifact_profile") == expected_artifact_profile,
            "ARTIFACT_MANIFEST_PROFILE_MISMATCH",
        ),
        (data_signature.get("dataset_id") == "SDWPF", "DATASET_ID_MISMATCH"),
        (data_signature.get("node_count") == 134, "DATASET_NODE_COUNT_MISMATCH"),
        (
            data_signature.get("feature_order_hash")
            == manifest["dataset_identity"]["feature_order_hash"],
            "DATASET_FEATURE_ORDER_MISMATCH",
        ),
        (
            all(
                effective.get(key) == value
                for key, value in expected_precision.items()
            ),
            "PRECISION_IDENTITY_MISMATCH",
        ),
        (effective.get("active_scope_id") == E5_SCOPE27_ID, "E5_SCOPE_ID_MISMATCH"),
        (effective.get("active_pointer_hash") == pointer_hash, "ACTIVE_POINTER_HASH_MISMATCH"),
        (effective.get("manifest_hash") == canonical_hash(manifest), "MANIFEST_HASH_MISMATCH"),
        (effective.get("run_map_hash") == canonical_hash(run_map), "RUN_MAP_HASH_MISMATCH"),
        (effective.get("freeze_hash") == current_freeze["freeze_hash"], "FREEZE_HASH_MISMATCH"),
    )
    result["reasons"].extend(reason for passed, reason in checks if not passed)
    return result, run_dir, {
        "status": status,
        "effective": effective,
        "data_signature": data_signature,
        "protocol_check": protocol_check,
        "artifact_manifest": artifact_manifest,
    }


def _execution_receipt_reasons(
    manifest: Mapping[str, Any],
    entry: Mapping[str, Any],
    run_dir: Path,
    *,
    checkpoint_sha256: str | None,
    metrics_sha256: str | None,
) -> tuple[list[str], dict[str, Any] | None]:
    path = run_dir / "execution_receipt.json"
    if not path.is_file():
        return ["EXECUTION_RECEIPT_MISSING"], None
    try:
        receipt = load_json(path)
    except Exception as exc:
        return [f"EXECUTION_RECEIPT_INVALID:{type(exc).__name__}"], None
    if not isinstance(receipt, dict):
        return ["EXECUTION_RECEIPT_NOT_OBJECT"], None
    pointer = load_active_scope_pointer()
    run_map = _load_run_map()
    freeze = compute_e5_freeze(manifest, run_map=run_map)
    revision = _source_revision()
    precision = entry.get("precision_identity") or {}
    expected = {
        "schema_version": EXECUTION_RECEIPT_SCHEMA_VERSION,
        "status": "SUCCESS",
        "scope_id": E5_SCOPE27_ID,
        "entry_id": entry["entry_id"],
        "model_id": entry["model_id"],
        "run_id": entry["e5_run_id"],
        "output_root": manifest["output_root"],
        "entry_type": entry["entry_type"],
        "git_commit": revision["git_commit"],
        "source_revision_type": revision["source_revision_type"],
        "active_pointer_hash": canonical_hash(pointer),
        "manifest_hash": canonical_hash(manifest),
        "run_map_hash": canonical_hash(run_map),
        "freeze_hash": freeze["freeze_hash"],
        "source_closure_hash": entry.get("base_model_source_hash"),
        "model_config_hash": entry.get("base_model_config_hash"),
        "precision_identity_hash": canonical_hash(precision),
        "loss_identity_hash": canonical_hash(manifest["loss_identity"]),
        "protocol_hash": manifest["benchmark_protocol_hash"],
        "training_profile_id": TRAINING_PROFILE_ID,
        "training_profile_hash": manifest["training_profile_hash"],
        "dataset_identity_hash": canonical_hash(manifest.get("dataset_identity")),
        "graph_identity_hash": canonical_hash(manifest.get("graph_identity")),
        "checkpoint_sha256": checkpoint_sha256,
        "metrics_bundle_hash": metrics_sha256,
        "exit_code": 0,
    }
    required = {
        *expected,
        "command",
        "started_at",
        "finished_at",
    }
    reasons = [
        f"EXECUTION_RECEIPT_FIELD_MISSING:{field}"
        for field in sorted(required)
        if field not in receipt
    ]
    for key, value in expected.items():
        actual = receipt.get(key)
        if key == "output_root":
            passed = _output_root_matches(actual, str(value))
        else:
            passed = actual == value
        if not passed and key in receipt:
            reasons.append(f"EXECUTION_RECEIPT_{key.upper()}_MISMATCH")
    if not isinstance(receipt.get("command"), list) or not receipt.get("command"):
        reasons.append("EXECUTION_RECEIPT_COMMAND_INVALID")
    for key in ("started_at", "finished_at"):
        if not isinstance(receipt.get(key), str) or not receipt.get(key):
            reasons.append(f"EXECUTION_RECEIPT_{key.upper()}_INVALID")
    return reasons, receipt


def inspect_trainable_run(
    manifest: Mapping[str, Any],
    entry: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    result, run_dir, common = _load_common_run(
        manifest,
        entry,
        output_root,
        expected_artifact_profile="TRAIN",
        expected_formal_training=True,
    )
    if common is None:
        return result
    metrics_paths = [
        run_dir / f"metrics_eval_h{horizon}.json" for horizon in HORIZONS
    ]
    metric_payloads, metric_reasons = _metric_payloads(metrics_paths)
    metric_reasons.extend(_metrics_csv_reasons(run_dir, metric_payloads))
    result["reasons"].extend(metric_reasons)
    checkpoint = run_dir / "best_checkpoint.pt"
    for filename in ("best_checkpoint.pt", "last_checkpoint.pt", "train_log.csv"):
        if not (run_dir / filename).is_file():
            result["reasons"].append(f"{filename.upper()}_MISSING")
    if checkpoint.is_file() and checkpoint.stat().st_size <= 0:
        result["reasons"].append("BEST_CHECKPOINT_EMPTY")
    checkpoint_sha256 = sha256_file(checkpoint) if checkpoint.is_file() and checkpoint.stat().st_size > 0 else None
    metrics_sha256 = _metrics_bundle_hash(run_dir)
    if metrics_sha256 is None:
        result["reasons"].append("METRICS_BUNDLE_HASH_UNAVAILABLE")
    receipt_reasons, receipt = _execution_receipt_reasons(
        manifest,
        entry,
        run_dir,
        checkpoint_sha256=checkpoint_sha256,
        metrics_sha256=metrics_sha256,
    )
    result["reasons"].extend(receipt_reasons)
    if not result["reasons"]:
        result.update(
            {
                "ready": True,
                "checkpoint_sha256": checkpoint_sha256,
                "metrics_sha256": metrics_sha256,
                "run_status_sha256": sha256_file(run_dir / "run_status.json"),
                "effective_config_sha256": sha256_file(
                    run_dir / "effective_config.json"
                ),
                "artifact_manifest_sha256": sha256_file(
                    run_dir / "artifact_manifest.json"
                ),
                "execution_receipt_status": receipt.get("status") if receipt else None,
            }
        )
    return result


def inspect_evaluate_only_run(
    manifest: Mapping[str, Any],
    entry: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    result, run_dir, common = _load_common_run(
        manifest,
        entry,
        output_root,
        expected_artifact_profile="NON_TRAINABLE",
        expected_formal_training=False,
    )
    if common is None:
        return result
    baseline_path = run_dir / "baseline_state.json"
    diagnostic_path = run_dir / "common_loss_diagnostic.json"
    prediction_path = run_dir / "prediction_metadata.json"
    try:
        baseline = load_json(baseline_path)
        diagnostic = load_json(diagnostic_path)
        prediction = load_json(prediction_path)
    except Exception as exc:
        result["reasons"].append(
            f"EVALUATE_ONLY_EVIDENCE:{type(exc).__name__}:{exc}"
        )
        return result
    baseline_checks = (
        (
            baseline.get("schema_version") == "e5_non_trainable_baseline_v1",
            "BASELINE_SCHEMA_MISMATCH",
        ),
        (baseline.get("model_id") == entry["model_id"], "BASELINE_MODEL_ID_MISMATCH"),
        (baseline.get("training_mode") == "EVALUATE_ONLY", "BASELINE_MODE_MISMATCH"),
        (baseline.get("training_loss") == "NOT_APPLICABLE", "BASELINE_TRAINING_LOSS_MISMATCH"),
        (baseline.get("trained_with_common_loss") is False, "BASELINE_TRAINED_FLAG_MISMATCH"),
        (baseline.get("common_loss_evaluation_applied") is True, "BASELINE_DIAGNOSTIC_FLAG_MISMATCH"),
        (baseline.get("best_checkpoint") is None, "BASELINE_CHECKPOINT_NOT_NULL"),
        (baseline.get("best_epoch") is None, "BASELINE_BEST_EPOCH_NOT_NULL"),
    )
    diagnostic_checks = (
        (
            diagnostic.get("schema_version") == "e5_common_loss_diagnostic_v1",
            "DIAGNOSTIC_SCHEMA_MISMATCH",
        ),
        (
            diagnostic.get("model_id") == entry["model_id"],
            "DIAGNOSTIC_MODEL_ID_MISMATCH",
        ),
        (diagnostic.get("loss_id") == "masked_score_aligned_hybrid", "DIAGNOSTIC_LOSS_ID_MISMATCH"),
        (
            diagnostic.get("loss_space") == "normalized Patv_raw",
            "DIAGNOSTIC_LOSS_SPACE_MISMATCH",
        ),
        (diagnostic.get("training_loss") == "NOT_APPLICABLE", "DIAGNOSTIC_TRAINING_LOSS_MISMATCH"),
        (diagnostic.get("trained_with_common_loss") is False, "DIAGNOSTIC_TRAINED_FLAG_MISMATCH"),
        (diagnostic.get("common_loss_evaluation_applied") is True, "DIAGNOSTIC_APPLIED_FLAG_MISMATCH"),
        (
            isinstance(diagnostic.get("value"), (int, float))
            and not isinstance(diagnostic.get("value"), bool)
            and math.isfinite(float(diagnostic["value"])),
            "DIAGNOSTIC_VALUE_NON_FINITE",
        ),
    )
    result["reasons"].extend(
        reason
        for passed, reason in (*baseline_checks, *diagnostic_checks)
        if not passed
    )
    if prediction.get("source_checkpoint") is not None:
        result["reasons"].append("EVALUATE_ONLY_SOURCE_CHECKPOINT_NOT_NULL")
    metrics_paths = [
        run_dir / f"metrics_eval_h{horizon}.json" for horizon in HORIZONS
    ]
    metric_payloads, metric_reasons = _metric_payloads(metrics_paths)
    metric_reasons.extend(_metrics_csv_reasons(run_dir, metric_payloads))
    result["reasons"].extend(metric_reasons)
    metrics_sha256 = _metrics_bundle_hash(run_dir)
    if metrics_sha256 is None:
        result["reasons"].append("METRICS_BUNDLE_HASH_UNAVAILABLE")
    receipt_reasons, receipt = _execution_receipt_reasons(
        manifest,
        entry,
        run_dir,
        checkpoint_sha256=None,
        metrics_sha256=metrics_sha256,
    )
    result["reasons"].extend(receipt_reasons)
    if not result["reasons"]:
        result.update(
            {
                "ready": True,
                "checkpoint_sha256": None,
                "baseline_state_sha256": sha256_file(baseline_path),
                "common_loss_diagnostic_sha256": sha256_file(diagnostic_path),
                "metrics_sha256": metrics_sha256,
                "run_status_sha256": sha256_file(run_dir / "run_status.json"),
                "effective_config_sha256": sha256_file(
                    run_dir / "effective_config.json"
                ),
                "artifact_manifest_sha256": sha256_file(
                    run_dir / "artifact_manifest.json"
                ),
                "execution_receipt_status": receipt.get("status") if receipt else None,
            }
        )
    return result


def inspect_run(
    manifest: Mapping[str, Any],
    entry: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    if entry["entry_type"] == "TRAIN_COMMON_LOSS":
        return inspect_trainable_run(manifest, entry, output_root)
    if entry["entry_type"] == "EVALUATE_ONLY_COMMON_LOSS_DIAGNOSTIC":
        return inspect_evaluate_only_run(manifest, entry, output_root)
    raise ScopeGateError(f"Unsupported run entry type: {entry['entry_type']}")


def inspect_a8(
    manifest: Mapping[str, Any], entry: Mapping[str, Any]
) -> dict[str, Any]:
    expected = manifest["a8_reference_identity"]
    result = validate_a8_reference(training_profile=TRAINING_PROFILE_ID)
    reasons: list[str] = []
    checks = (
        (result.get("status") == "VALID", "A8_REFERENCE_INVALID"),
        (result.get("reference_id") == entry["e5_run_id"], "A8_ID_MISMATCH"),
        (
            result.get("source_checkpoint_sha256")
            == expected["checkpoint_sha256"],
            "A8_CHECKPOINT_HASH_MISMATCH",
        ),
        (
            result.get("source_metrics_sha256") == expected["metrics_sha256"],
            "A8_METRICS_HASH_MISMATCH",
        ),
        (
            result.get("source_config_sha256") == expected["config_sha256"],
            "A8_CONFIG_HASH_MISMATCH",
        ),
        (
            result.get("source_protocol_evidence_sha256")
            == expected["protocol_evidence_sha256"],
            "A8_PROTOCOL_EVIDENCE_HASH_MISMATCH",
        ),
        (
            result.get("source_protocol_hash")
            == manifest["benchmark_protocol_hash"],
            "A8_PROTOCOL_HASH_MISMATCH",
        ),
        (
            result.get("source_loss_id")
            == manifest["loss_identity"]["loss_id"],
            "A8_LOSS_ID_MISMATCH",
        ),
        (result.get("retrained_in_e5") is False, "A8_MUST_BE_READ_ONLY"),
        (result.get("checkpoint_copied") is False, "A8_CHECKPOINT_COPIED"),
        (result.get("metrics_copied") is False, "A8_METRICS_COPIED"),
        (result.get("trained_with_common_loss") is True, "A8_LOSS_EVIDENCE_MISSING"),
        (result.get("train_batch_size") == 4, "A8_TRAIN_BATCH_MISMATCH"),
        (result.get("val_batch_size") == 4, "A8_VAL_BATCH_MISMATCH"),
        (result.get("test_batch_size") == 4, "A8_TEST_BATCH_MISMATCH"),
        (result.get("lookback") == 144, "A8_LOOKBACK_MISMATCH"),
        (result.get("max_pred_len") == 10, "A8_HORIZON_MISMATCH"),
    )
    reasons.extend(reason for passed, reason in checks if not passed)
    return {
        "entry_id": entry["entry_id"],
        "model_id": entry["model_id"],
        "entry_type": entry["entry_type"],
        "run_id": entry["e5_run_id"],
        "run_dir": result.get("source_absolute_or_resolved_path"),
        "found": not bool(result.get("missing")),
        "ready": not reasons,
        "reasons": reasons,
        "checkpoint_sha256": result.get("source_checkpoint_sha256"),
        "metrics_sha256": result.get("source_metrics_sha256"),
        "effective_config_sha256": result.get("source_config_sha256"),
        "protocol_evidence_sha256": result.get(
            "source_protocol_evidence_sha256"
        ),
        "reference": result,
    }


def _is_identity_reason(reason: str) -> bool:
    if "METRICS" in reason:
        return False
    tokens = (
        "IDENTITY",
        "SOURCE_",
        "MODEL_CONFIG",
        "PRECISION",
        "PROFILE",
        "DATASET",
        "GRAPH",
        "ACTIVE_",
        "MANIFEST",
        "RUN_MAP",
        "FREEZE",
        "OUTPUT_ROOT",
        "LOSS_",
        "PROTOCOL",
        "TRAINING_",
    )
    return any(token in reason for token in tokens)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            completed = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return completed.returncode == 0 and re.search(
            rf"\b{pid}\b", completed.stdout
        ) is not None
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _process_start_time(pid: int) -> float | None:
    if pid <= 0:
        return None
    try:
        import psutil  # type: ignore

        return float(psutil.Process(pid).create_time())
    except ImportError:
        return None
    except Exception as exc:  # psutil exception classes vary by platform.
        if type(exc).__name__ not in {"NoSuchProcess", "AccessDenied", "ZombieProcess"}:
            raise
        return None


def lock_status(path: Path = LOCK_PATH) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "ABSENT", "path": str(path)}
    try:
        payload = load_json(path)
    except Exception as exc:
        return {
            "status": "MALFORMED",
            "path": str(path),
            "error": f"{type(exc).__name__}:{exc}",
        }
    if not isinstance(payload, dict):
        return {"status": "MALFORMED", "path": str(path), "error": "LOCK_NOT_OBJECT"}
    required = {
        "schema_version",
        "scope_id",
        "hostname",
        "pid",
        "process_start_time",
        "git_commit",
        "manifest_hash",
        "run_map_hash",
        "freeze_hash",
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
        if current_start is None or not _pid_alive(pid):
            state = "STALE"
        elif abs(current_start - recorded_start) <= 1e-3:
            state = "ACTIVE"
        else:
            state = "STALE"
    return {"status": state, "path": str(path), **payload}


def clear_stale_lock(path: Path = LOCK_PATH) -> dict[str, Any]:
    state = lock_status(path)
    if state["status"] != "STALE":
        raise ScopeGateError(
            f"Only a confirmed STALE E5 lock may be cleared; got {state['status']}."
        )
    path.unlink()
    return {"status": "CLEARED_STALE", "path": str(path)}


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _recursive_file_manifest(root: Path) -> list[dict[str, str]]:
    if not root.is_dir() or root.is_symlink():
        raise ScopeGateError(f"Unsafe or missing E5 archive source: {root}")
    records: list[dict[str, str]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file():
            continue
        if path.is_symlink():
            raise ScopeGateError(f"E5 archive source contains a symlink: {path}")
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
            }
        )
    return records


def _active_worker(run_dir: Path) -> bool:
    try:
        status = load_json(run_dir / "run_status.json")
        effective = load_json(run_dir / "effective_config.json")
    except Exception:
        return False
    if isinstance(status, dict) and status.get("status") == "RUNNING":
        return True
    candidates = []
    for payload in (status, effective):
        if isinstance(payload, dict):
            candidates.append(payload.get("formal_worker_pid"))
            candidates.append(payload.get("pid"))
    return any(isinstance(pid, int) and _pid_alive(pid) for pid in candidates)


def _canonical_identity_evidence(
    manifest: Mapping[str, Any], entry: Mapping[str, Any], run_dir: Path
) -> bool:
    try:
        effective = load_json(run_dir / "effective_config.json")
        artifact = load_json(run_dir / "artifact_manifest.json")
    except Exception:
        return False
    if not isinstance(effective, dict) or not isinstance(artifact, dict):
        return False
    provenance = effective.get("provenance") or {}
    source_hash = entry.get("base_model_source_hash")
    config_hash = entry.get("base_model_config_hash")
    return all(
        (
            effective.get("model_id") == entry["model_id"],
            effective.get("run_id") == entry["e5_run_id"],
            _output_root_matches(effective.get("output_root"), manifest["output_root"]),
            effective.get("active_scope_id") == E5_SCOPE27_ID,
            provenance.get("active_scope_id") == E5_SCOPE27_ID,
            effective.get("experiment_profile_id") == manifest["experiment_profile_id"],
            effective.get("training_batch_profile_id") == TRAINING_PROFILE_ID,
            effective.get("training_batch_profile_hash") == manifest["training_profile_hash"],
            effective.get("source_closure_hash") == source_hash
            or provenance.get("base_model_source_closure_hash") == source_hash,
            effective.get("model_config_hash") == config_hash
            or provenance.get("base_model_config_hash") == config_hash,
            effective.get("loss", {}).get("id") == manifest["loss_identity"]["loss_id"],
            artifact.get("model_id") == entry["model_id"],
            artifact.get("run_id") == entry["e5_run_id"],
        )
    )


def _find_model_directories(root: Path, model_id: str) -> list[Path]:
    if not root.is_dir():
        return []
    found: list[Path] = []
    for path in root.rglob("effective_config.json"):
        if any(part in {ARCHIVE_DIRECTORY, QUARANTINE_DIRECTORY, INTENT_DIRECTORY} for part in path.parts):
            continue
        try:
            payload = load_json(path)
        except Exception:
            continue
        if isinstance(payload, dict) and payload.get("model_id") == model_id:
            found.append(path.parent)
    return found


def _archive_target(allowed_root: Path, run_id: str, kind: str, *, apply: bool) -> Path:
    if kind not in {ARCHIVE_DIRECTORY, QUARANTINE_DIRECTORY}:
        raise ScopeGateError(f"Unsupported E5 archive kind: {kind}")
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ScopeGateError("E5 run id must be a safe single path component.")
    parent = (allowed_root / kind / run_id).resolve()
    target = (parent / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}_{os.getpid()}_{uuid.uuid4().hex}").resolve()
    if not _path_within(parent, allowed_root) or not _path_within(target, allowed_root):
        raise ScopeGateError("E5 archive target escapes the allowed root.")
    if apply:
        parent.mkdir(parents=True, exist_ok=True)
        if parent.is_symlink() or target.exists():
            raise ScopeGateError("E5 archive target collision or symlink refused.")
    return target


def archive_or_quarantine(
    manifest: Mapping[str, Any],
    entry: Mapping[str, Any],
    output_root: Path,
    *,
    kind: str = ARCHIVE_DIRECTORY,
    apply: bool = False,
    reason: str = "identity-matched incomplete canonical attempt",
) -> dict[str, Any]:
    allowed = Path(output_root).resolve()
    source = (allowed / str(entry["e5_run_id"])).resolve()
    if not _path_within(source, allowed) or source.name != entry["e5_run_id"]:
        raise ScopeGateError("E5 canonical source escapes the current output root.")
    if source.is_symlink() or not source.is_dir():
        raise ScopeGateError(f"E5 canonical source is missing or unsafe: {source}")
    if _active_worker(source):
        raise ScopeGateError("Active E5 worker evidence blocks archival action.")
    inspected = inspect_run(manifest, entry, allowed)
    identity_reasons = [reason for reason in inspected.get("reasons", []) if _is_identity_reason(reason)]
    if kind == ARCHIVE_DIRECTORY:
        if identity_reasons or not _canonical_identity_evidence(manifest, entry, source):
            raise ScopeGateError("Identity-mismatch/incomplete E5 source cannot be auto-archived.")
        classification = "CURRENT_E5_FAILED" if inspected.get("reasons") and any(
            token in reason for reason in inspected["reasons"] for token in ("NOT_COMPLETED", "EXIT_CODE_NOT_ZERO")
        ) else "CURRENT_E5_INCOMPLETE"
        receipt_name = "archive_receipt.json"
    else:
        if not identity_reasons:
            raise ScopeGateError("E5 quarantine requires explicit identity-mismatch evidence.")
        classification = "CURRENT_E5_IDENTITY_MISMATCH"
        receipt_name = "quarantine_receipt.json"
    target = _archive_target(allowed, entry["e5_run_id"], kind, apply=apply)
    receipt = {
        "schema_version": f"e5_scope27_{'archive' if kind == ARCHIVE_DIRECTORY else 'quarantine'}_receipt_v1",
        "scope_id": E5_SCOPE27_ID,
        "model_id": entry["model_id"],
        "run_id": entry["e5_run_id"],
        "original_path": str(source),
        "archive_path": str(target),
        "archive_reason": reason,
        "original_classification": classification,
        "original_rejection_reasons": list(inspected.get("reasons", [])),
        "manifest_hash": canonical_hash(manifest),
        "run_map_hash": canonical_hash(_load_run_map()),
        "freeze_hash": compute_e5_freeze(manifest)["freeze_hash"],
        "attempt_id": uuid.uuid4().hex,
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
    intent_root = (allowed / kind / INTENT_DIRECTORY).resolve()
    if not _path_within(intent_root, allowed):
        raise ScopeGateError("E5 intent path escapes current output root.")
    intent_root.mkdir(parents=True, exist_ok=True)
    if intent_root.is_symlink():
        raise ScopeGateError("E5 intent directory cannot be a symlink.")
    intent_path = intent_root / f"{kind}_{entry['e5_run_id']}_{receipt['attempt_id']}.json"
    intent = {
        "schema_version": "e5_scope27_archive_intent_v1",
        "status": "PENDING",
        "operation": "ARCHIVE" if kind == ARCHIVE_DIRECTORY else "QUARANTINE",
        "source": str(source),
        "target": str(target),
        "receipt_name": receipt_name,
        "receipt": receipt,
        "started_at": utc_now(),
        "file_manifest": _recursive_file_manifest(source),
    }
    atomic_write_json(intent_path, intent)
    try:
        os.replace(str(source), str(target))
    except Exception as exc:
        intent.update({"status": "FAILED", "error_type": type(exc).__name__, "error_message": str(exc), "finished_at": utc_now()})
        atomic_write_json(intent_path, intent)
        raise
    try:
        final_receipt = target / receipt_name
        if final_receipt.exists():
            raise ScopeGateError("E5 archival receipt collision refused.")
        atomic_write_json(final_receipt, receipt)
    except Exception as exc:
        intent.update({"status": "FAILED_AFTER_MOVE", "error_type": type(exc).__name__, "error_message": str(exc), "finished_at": utc_now()})
        atomic_write_json(intent_path, intent)
        raise
    intent.update({"status": "COMPLETED", "finished_at": utc_now()})
    atomic_write_json(intent_path, intent)
    return {
        "status": "QUARANTINED" if kind == QUARANTINE_DIRECTORY else "ARCHIVED",
        "action": "QUARANTINE_EXISTING" if kind == QUARANTINE_DIRECTORY else "ARCHIVE_INCOMPLETE_THEN_RUN",
        "source": str(source),
        "target": str(target),
        "receipt": str(target / receipt_name),
        "intent": str(intent_path),
    }


def build_preflight_plan(manifest: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for entry in manifest["entries"]:
        trainable = entry["entry_type"] == TRAINABLE_ENTRY_TYPE
        rows.append(
            {
                "ordinal": entry["ordinal"],
                "model_id": entry["model_id"],
                "run_id": entry["e5_run_id"],
                "entry_type": entry["entry_type"],
                "preflight_action": "EXACT_GPU_PREFLIGHT" if trainable else "SKIP_NON_TRAINABLE_SCOPE_ENTRY",
                "requires_exact_pass": trainable,
                "gpu_preflight_started": False,
                "exact_shape": PREFLIGHT_SHAPE if trainable else None,
                "batch_profile_id": TRAINING_PROFILE_ID if trainable else None,
                "precision_identity": entry.get("precision_identity") if trainable else None,
                "loss_identity": manifest.get("loss_identity") if trainable else None,
                "scope_id": E5_SCOPE27_ID,
            }
        )
    return {
        "schema_version": "e5_scope27_exact_preflight_plan_v1",
        "scope_id": E5_SCOPE27_ID,
        "execution_performed": False,
        "gpu_preflight_performed": False,
        "counts": {
            "trainable_expected": 24,
            "evaluate_only_skipped": 2,
            "a8_reference_skipped": 1,
            "excluded_historical_skipped": 2,
            "total_evidence": 27,
        },
        "entries": rows,
    }


def _plan_action(
    manifest: Mapping[str, Any], entry: Mapping[str, Any], output_root: Path
) -> tuple[str, dict[str, Any]]:
    if entry["entry_type"] == REFERENCE_ENTRY_TYPE:
        inspected = inspect_a8(manifest, entry)
        return (
            "SKIP_COMPLETED_IDENTITY_MATCH" if inspected["ready"] else "BLOCK_A8_BATCH4_REFERENCE",
            inspected,
        )
    inspected = inspect_run(manifest, entry, output_root)
    if inspected["ready"]:
        return "SKIP_COMPLETED_IDENTITY_MATCH", inspected
    if not inspected["found"]:
        renamed = [
            path
            for path in _find_model_directories(output_root, str(entry["model_id"]))
            if path.name != entry["e5_run_id"]
        ]
        if renamed:
            inspected["found_noncanonical"] = [str(path) for path in renamed]
            inspected.setdefault("reasons", []).append("RUN_ID_RENAMED_OR_NONCANONICAL")
            return "BLOCK_EXISTING_IDENTITY_MISMATCH", inspected
        return "RUN_MISSING", inspected
    if any(_is_identity_reason(reason) for reason in inspected.get("reasons", [])):
        return "BLOCK_EXISTING_IDENTITY_MISMATCH", inspected
    if _canonical_identity_evidence(manifest, entry, Path(inspected["run_dir"])) and not _active_worker(Path(inspected["run_dir"])):
        return "ARCHIVE_INCOMPLETE_THEN_RUN", inspected
    inspected.setdefault("reasons", []).append("ARCHIVE_NOT_SAFE_OR_ACTIVE_WORKER")
    return "BLOCK_EXISTING_IDENTITY_MISMATCH", inspected


def build_plan(manifest: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    if output_root.resolve() != EXPECTED_OUTPUT_ROOT.resolve():
        raise ScopeGateError("Formal E5 output root must be the current uniform_bs4 root.")
    rows: list[dict[str, Any]] = []
    for entry in manifest["entries"]:
        action, inspected = _plan_action(manifest, entry, output_root)
        rows.append(
            {
                "ordinal": entry["ordinal"],
                "model_id": entry["model_id"],
                "run_id": entry["e5_run_id"],
                "entry_id": entry["entry_id"],
                "entry_type": entry["entry_type"],
                "output_root": entry.get("output_root") or manifest["output_root"],
                "device": entry.get("device") or (
                    "cpu" if entry["command"] == "evaluate-only" else "cuda"
                ),
                "action": action,
                "run_dir": inspected.get("run_dir"),
                "found": inspected.get("found", False),
                "ready": inspected.get("ready", False),
                "reasons": inspected.get("reasons", []),
                "found_noncanonical": inspected.get("found_noncanonical", []),
            }
        )
    counts = {
        "total": len(rows),
        "trainable": sum(row["entry_type"] == TRAINABLE_ENTRY_TYPE for row in rows),
        "evaluate_only": sum(row["entry_type"] == EVALUATE_ONLY_ENTRY_TYPE for row in rows),
        "a8_reference": sum(row["entry_type"] == REFERENCE_ENTRY_TYPE for row in rows),
        "ready": sum(row["ready"] for row in rows),
        "run": sum(row["action"] in {"RUN_MISSING", "ARCHIVE_INCOMPLETE_THEN_RUN"} for row in rows),
        "blocked": sum(row["action"].startswith("BLOCK") for row in rows),
    }
    return {
        "schema_version": "e5_scope27_resume_plan_v1",
        "scope_id": E5_SCOPE27_ID,
        "output_root": str(output_root.resolve()),
        "generated_at": utc_now(),
        "counts": counts,
        "entries": rows,
    }


def _command_for_entry(
    entry: Mapping[str, Any],
    *,
    input_path: str | None,
    target_path: str | None,
    preflight_root: Path | None,
    source_revision: str,
) -> list[str]:
    command = "evaluate-only" if entry["entry_type"] == EVALUATE_ONLY_ENTRY_TYPE else "train"
    args = [
        sys.executable,
        "-m",
        "benchmark_v2.cli",
        command,
        "--model",
        str(entry["model_id"]),
        "--output-root",
        str(EXPECTED_OUTPUT_ROOT.resolve()),
        "--run-id",
        str(entry["e5_run_id"]),
        "--device",
        "cpu" if command == "evaluate-only" else "cuda",
        "--experiment-profile",
        CLI_PROFILE_ID,
        "--training-profile",
        TRAINING_PROFILE_ID,
        "--formal-scope-id",
        E5_SCOPE27_ID,
        "--source-revision",
        source_revision,
    ]
    if input_path:
        args.extend(["--input-path", input_path])
    if target_path:
        args.extend(["--target-path", target_path])
    if preflight_root is not None and command == "train":
        args.extend(["--preflight-root", str(preflight_root)])
    return args


def _write_execution_receipt(
    run_dir: Path,
    entry: Mapping[str, Any],
    *,
    command: list[str],
    started_at: str,
    finished_at: str,
    exit_code: int,
    manifest: Mapping[str, Any],
) -> Path:
    path = run_dir / "execution_receipt.json"
    if path.exists():
        raise ScopeGateError(f"Refusing to overwrite an existing E5 execution receipt: {path}")
    checkpoint = run_dir / "best_checkpoint.pt"
    receipt = {
        "schema_version": EXECUTION_RECEIPT_SCHEMA_VERSION,
        "status": "SUCCESS",
        "scope_id": E5_SCOPE27_ID,
        "entry_id": entry["entry_id"],
        "model_id": entry["model_id"],
        "run_id": entry["e5_run_id"],
        "output_root": manifest["output_root"],
        "entry_type": entry["entry_type"],
        **_source_revision(),
        "active_pointer_hash": canonical_hash(load_active_scope_pointer()),
        "manifest_hash": canonical_hash(manifest),
        "run_map_hash": canonical_hash(_load_run_map()),
        "freeze_hash": compute_e5_freeze(manifest)["freeze_hash"],
        "source_closure_hash": entry.get("base_model_source_hash"),
        "model_config_hash": entry.get("base_model_config_hash"),
        "precision_identity_hash": canonical_hash(entry.get("precision_identity") or {}),
        "loss_identity_hash": canonical_hash(manifest["loss_identity"]),
        "protocol_hash": manifest["benchmark_protocol_hash"],
        "training_profile_id": TRAINING_PROFILE_ID,
        "training_profile_hash": manifest["training_profile_hash"],
        "dataset_identity_hash": canonical_hash(manifest.get("dataset_identity")),
        "graph_identity_hash": canonical_hash(manifest.get("graph_identity")),
        "checkpoint_sha256": sha256_file(checkpoint) if entry["entry_type"] == TRAINABLE_ENTRY_TYPE and checkpoint.is_file() else None,
        "metrics_bundle_hash": _metrics_bundle_hash(run_dir),
        "command": [str(value) for value in command],
        "started_at": started_at,
        "finished_at": finished_at,
        "exit_code": int(exit_code),
    }
    if receipt["metrics_bundle_hash"] is None:
        raise ScopeGateError("Cannot issue an E5 receipt without a complete metrics bundle.")
    if entry["entry_type"] == TRAINABLE_ENTRY_TYPE and receipt["checkpoint_sha256"] is None:
        raise ScopeGateError("Cannot issue a trainable E5 receipt without a non-empty checkpoint.")
    atomic_write_json(path, receipt)
    return path


def _failure_diagnostics(
    log_path: Path,
    *,
    exit_code: int,
    error_type: str | None = None,
    error_message: str | None = None,
    traceback_tail: list[str] | None = None,
) -> dict[str, Any]:
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        lines = []
    tail = lines[-80:]
    if traceback_tail:
        tail = [*tail, *traceback_tail][-80:]
    joined = "\n".join(tail)
    match = re.search(r"(?P<type>[A-Za-z_][\w.]*(?:Error|Exception|ContractError))(?::\s*(?P<message>.*))?", joined)
    error_type = error_type or (match.group("type").split(".")[-1] if match else "ChildProcessError")
    error_message = error_message or ((match.group("message") or "").strip() if match else f"child exit code {exit_code}")
    return {
        "error_type": error_type,
        "error_message": error_message,
        "traceback_tail": tail,
        "oom": bool(re.search(r"cuda.*out of memory|out of memory|outofmemory", joined, re.I)),
        "nonfinite": bool(re.search(r"nan|inf|nonfinite", joined, re.I)),
        "preflight_failure": bool(re.search(r"preflight|PREFLIGHT_MISSING_OR_MISMATCH", joined, re.I)),
        "identity_failure": bool(re.search(r"identity|mismatch|source/config", joined, re.I)),
        "collision": bool(re.search(r"FileExistsError|collision|already exists", joined, re.I)),
        "archive_failure": bool(re.search(r"archive|quarantine|os\.replace|atomic move", joined, re.I)),
        "lock_failure": bool(re.search(r"lock|active worker|UNKNOWN_REMOTE|STALE", joined, re.I)),
    }


def _write_failure_artifact(log_root: Path, evidence: dict[str, Any], ordinal: int, model_id: str) -> Path:
    path = log_root / f"{ordinal:02d}_{model_id}.failure.json"
    evidence["failure_artifact"] = str(path)
    atomic_write_json(path, evidence)
    return path


def _acquire_lock(manifest: Mapping[str, Any], freeze: Mapping[str, Any]) -> dict[str, Any]:
    state = lock_status()
    if state["status"] != "ABSENT":
        raise ScopeGateError(f"E5 lock blocks execution: {state['status']}")
    payload = {
        "schema_version": LOCK_SCHEMA_VERSION,
        "scope_id": E5_SCOPE27_ID,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "process_start_time": time.time(),
        "git_commit": _source_revision()["git_commit"],
        "manifest_hash": canonical_hash(manifest),
        "run_map_hash": canonical_hash(_load_run_map()),
        "freeze_hash": freeze["freeze_hash"],
        "created_at": utc_now(),
    }
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(LOCK_PATH, payload)
    return payload


def run_scope(
    manifest: Mapping[str, Any],
    *,
    input_path: str | None,
    target_path: str | None,
    log_root: Path,
    preflight_root: Path,
    source_revision: str,
) -> tuple[int, dict[str, Any]]:
    freeze = compute_e5_freeze(manifest)
    plan = build_plan(manifest, EXPECTED_OUTPUT_ROOT)
    if any(row["action"] == "BLOCK_A8_BATCH4_REFERENCE" for row in plan["entries"]):
        return 74, {"status": "BLOCKED_A8_BATCH4_REFERENCE", "plan": plan}
    if any(row["action"] == "BLOCK_EXISTING_IDENTITY_MISMATCH" for row in plan["entries"]):
        return 74, {"status": "BLOCKED_EXISTING_ARTIFACTS", "plan": plan}
    log_root.mkdir(parents=True, exist_ok=True)
    lock_payload = _acquire_lock(manifest, freeze)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    try:
        for entry, row in zip(manifest["entries"], plan["entries"]):
            if row["action"] == "SKIP_COMPLETED_IDENTITY_MATCH":
                results.append({"model_id": entry["model_id"], "action": row["action"], "exit_code": 0, "readiness": "READY"})
                continue
            if row["action"] == "ARCHIVE_INCOMPLETE_THEN_RUN":
                try:
                    archive_or_quarantine(manifest, entry, EXPECTED_OUTPUT_ROOT, apply=True)
                except Exception as exc:
                    evidence = {
                        "failed_model": entry["model_id"],
                        "model_id": entry["model_id"],
                        "run_id": entry["e5_run_id"],
                        "action": "ARCHIVE_INCOMPLETE_THEN_RUN",
                        "command": [],
                        "exit_code": 74,
                        "started_at": utc_now(),
                        "finished_at": utc_now(),
                        "per_model_log": None,
                        "run_directory": str(EXPECTED_OUTPUT_ROOT / entry["e5_run_id"]),
                        "failure_artifact": None,
                        **_failure_diagnostics(Path(), exit_code=74, error_type=type(exc).__name__, error_message=str(exc)),
                        "readiness": "BLOCKED",
                    }
                    _write_failure_artifact(log_root, evidence, entry["ordinal"], entry["model_id"])
                    failures.append(evidence)
                    results.append(evidence)
                    continue
            command = _command_for_entry(
                entry,
                input_path=input_path,
                target_path=target_path,
                preflight_root=preflight_root,
                source_revision=source_revision,
            )
            log_path = log_root / f"{entry['ordinal']:02d}_{entry['model_id']}.log"
            started_at = utc_now()
            exit_code = 1
            error_type = None
            error_message = None
            traceback_tail = None
            try:
                if entry["entry_type"] == TRAINABLE_ENTRY_TYPE:
                    from benchmark_v2.hardware_preflight import read_matching_pass

                    if read_matching_pass(
                        entry["model_id"],
                        root=preflight_root,
                        experiment_profile=CLI_PROFILE_ID,
                        training_profile=TRAINING_PROFILE_ID,
                        formal_scope_id=E5_SCOPE27_ID,
                        source_revision=source_revision,
                    ) is None:
                        raise ScopeGateError("PREFLIGHT_MISSING_OR_MISMATCH")
                with log_path.open("w", encoding="utf-8", newline="") as handle:
                    completed = subprocess.run(
                        command,
                        cwd=str(PROJECT_ROOT),
                        env={**os.environ, "PYTHONPATH": str(SRC_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")},
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                exit_code = int(completed.returncode)
            except Exception as exc:
                error_type = type(exc).__name__
                error_message = str(exc)
                traceback_tail = traceback.format_exc().splitlines()[-20:]
            finished_at = utc_now()
            run_dir = EXPECTED_OUTPUT_ROOT / entry["e5_run_id"]
            receipt_path = None
            if exit_code == 0 and run_dir.is_dir():
                try:
                    receipt_path = _write_execution_receipt(
                        run_dir,
                        entry,
                        command=command,
                        started_at=started_at,
                        finished_at=finished_at,
                        exit_code=exit_code,
                        manifest=manifest,
                    )
                except Exception as exc:
                    exit_code = 74
                    error_type = type(exc).__name__
                    error_message = str(exc)
                    traceback_tail = traceback.format_exc().splitlines()[-20:]
            diagnostics = _failure_diagnostics(
                log_path,
                exit_code=exit_code,
                error_type=error_type,
                error_message=error_message,
                traceback_tail=traceback_tail,
            )
            evidence = {
                "failed_model": entry["model_id"] if exit_code != 0 else None,
                "model_id": entry["model_id"],
                "run_id": entry["e5_run_id"],
                "action": "RUN_MISSING" if row["action"] == "RUN_MISSING" else "ARCHIVE_INCOMPLETE_THEN_RUN",
                "command": command,
                "exit_code": exit_code,
                "started_at": started_at,
                "finished_at": finished_at,
                "per_model_log": str(log_path),
                "run_directory": str(run_dir),
                "failure_artifact": None,
                "execution_receipt": str(receipt_path) if receipt_path else None,
                **diagnostics,
                "readiness": "READY" if exit_code == 0 else "FAILED",
            }
            if exit_code != 0:
                _write_failure_artifact(log_root, evidence, entry["ordinal"], entry["model_id"])
                failures.append(evidence)
            results.append(evidence)
        readiness = build_readiness(manifest, EXPECTED_OUTPUT_ROOT)
        status = "COMPLETED_WITH_FAILURES" if failures else readiness["status"]
        code = 1 if failures else 0 if status == "COMPLETED_READY_27_OF_27" else 4
        report = {
            "schema_version": "e5_scope27_run_status_v1",
            "scope_id": E5_SCOPE27_ID,
            "status": status,
            "exit_code": code,
            "lock": lock_payload,
            "plan": plan,
            "results": results,
            "failures": failures,
            "readiness": readiness,
        }
        atomic_write_json(log_root / "e5_scope27_run_status.json", report)
        return code, report
    finally:
        try:
            if LOCK_PATH.is_file() and load_json(LOCK_PATH).get("pid") == os.getpid():
                LOCK_PATH.unlink()
        except (OSError, ValueError, KeyError):
            pass


def inventory(manifest: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    plan = build_plan(manifest, output_root)
    runs = []
    if output_root.is_dir():
        for child in sorted(output_root.iterdir(), key=lambda item: item.name):
            if child.name in {ARCHIVE_DIRECTORY, QUARANTINE_DIRECTORY} or not child.is_dir():
                continue
            run_status = None
            try:
                payload = load_json(child / "run_status.json")
                run_status = payload if isinstance(payload, dict) else None
            except Exception:
                pass
            runs.append({"run_id": child.name, "path": str(child), "status": run_status})
    return {
        "schema_version": "e5_scope27_inventory_v1",
        "scope_id": E5_SCOPE27_ID,
        "root": str(output_root.resolve()),
        "scan_read_only": True,
        "plan_counts": plan["counts"],
        "runs": runs,
    }


def build_readiness(
    manifest: Mapping[str, Any], output_root: Path
) -> dict[str, Any]:
    expected = EXPECTED_OUTPUT_ROOT.resolve()
    actual = output_root.resolve()
    if actual != expected:
        raise ScopeGateError(
            f"Output root mismatch: actual={actual}, expected={expected}"
        )
    rows = []
    for entry in manifest["entries"]:
        if entry["entry_type"] == "REFERENCE_ONLY_FORMAL_A8":
            rows.append(inspect_a8(manifest, entry))
        else:
            rows.append(inspect_run(manifest, entry, actual))
    by_type = {
        "trainable": [
            row for row in rows if row["entry_type"] == "TRAIN_COMMON_LOSS"
        ],
        "evaluate_only": [
            row
            for row in rows
            if row["entry_type"]
            == "EVALUATE_ONLY_COMMON_LOSS_DIAGNOSTIC"
        ],
        "a8_reference": [
            row
            for row in rows
            if row["entry_type"] == "REFERENCE_ONLY_FORMAL_A8"
        ],
    }
    counts = {
        key: {
            "ready": sum(row["ready"] for row in selected),
            "expected": len(selected),
        }
        for key, selected in by_type.items()
    }
    counts["total_evidence"] = {
        "ready": sum(row["ready"] for row in rows),
        "expected": 27,
    }
    a8_rows = by_type["a8_reference"]
    a8_ready = bool(a8_rows and a8_rows[0]["ready"])
    identity_blocked = any(
        not row["ready"]
        and any(_is_identity_reason(reason) for reason in row.get("reasons", []))
        for row in rows
        if row["entry_type"] != REFERENCE_ENTRY_TYPE
    )
    found_incomplete = any(
        row.get("found") and not row.get("ready")
        for row in rows
        if row["entry_type"] != REFERENCE_ENTRY_TYPE
    )
    if not a8_ready:
        status = "BLOCKED_A8_BATCH4_REFERENCE"
    elif identity_blocked:
        status = "BLOCKED_EXISTING_ARTIFACTS"
    elif counts["total_evidence"]["ready"] == 27:
        status = "COMPLETED_READY_27_OF_27"
    elif found_incomplete:
        status = "COMPLETED_WITH_FAILURES"
    else:
        status = "NOT_READY_MISSING_RESULTS"
    return {
        "schema_version": "e5_scope27_readiness_v1",
        "scope_id": E5_SCOPE27_ID,
        "status": status,
        "require_complete": True,
        "counts": counts,
        "output_root": str(actual),
        "generated_at": utc_now(),
        "entries": rows,
    }


def write_readiness_outputs(
    readiness: Mapping[str, Any],
    report_path: Path,
    evidence_path: Path,
) -> None:
    atomic_write_json(report_path, readiness)
    evidence = {
        "schema_version": "e5_scope27_evidence_manifest_v1",
        "scope_id": E5_SCOPE27_ID,
        "status": readiness["status"],
        "require_complete": True,
        "generated_at": readiness["generated_at"],
        "entries": [
            {
                key: row.get(key)
                for key in (
                    "entry_id",
                    "model_id",
                    "entry_type",
                    "run_id",
                    "run_dir",
                    "ready",
                    "checkpoint_sha256",
                    "baseline_state_sha256",
                    "common_loss_diagnostic_sha256",
                    "metrics_sha256",
                    "run_status_sha256",
                    "effective_config_sha256",
                    "artifact_manifest_sha256",
                    "protocol_evidence_sha256",
                )
            }
            for row in readiness["entries"]
        ],
    }
    atomic_write_json(evidence_path, evidence)


def _aggregate_rows(
    manifest: Mapping[str, Any], readiness: Mapping[str, Any]
) -> list[dict[str, Any]]:
    entries = {row["model_id"]: row for row in manifest["entries"]}
    result: list[dict[str, Any]] = []
    for evidence in readiness["entries"]:
        entry = entries[evidence["model_id"]]
        if entry["entry_type"] == "REFERENCE_ONLY_FORMAL_A8":
            reference = evidence["reference"]
            metrics_paths = [
                Path(path) for path in reference["source_metrics_paths"]
            ]
            effective = load_json(Path(reference["source_config_path"]))
            training_status = "REFERENCE_ONLY"
        else:
            run_dir = Path(evidence["run_dir"])
            metrics_paths = [
                run_dir / f"metrics_eval_h{horizon}.json"
                for horizon in HORIZONS
            ]
            effective = load_json(run_dir / "effective_config.json")
            training_status = load_json(run_dir / "run_status.json")["status"]
        metrics, reasons = _metric_payloads(metrics_paths)
        if reasons:
            raise ScopeGateError(
                f"Refusing non-finite/incomplete aggregate for "
                f"{entry['model_id']}: {reasons}"
            )
        row: dict[str, Any] = {
            "model": entry["display_name"],
            "model_id": entry["model_id"],
            "entry_type": entry["entry_type"],
            "training_mode": (
                "REFERENCE_ONLY"
                if entry["entry_type"] == "REFERENCE_ONLY_FORMAL_A8"
                else ("EVALUATE_ONLY" if entry["command"] == "evaluate-only" else "TRAIN")
            ),
            "loss_id": entry["loss_id"],
            "training_status": training_status,
            "checkpoint": (
                "best_checkpoint.pt"
                if entry["entry_type"] == "TRAIN_COMMON_LOSS"
                else None
            ),
            "best_epoch": effective.get("best_epoch"),
            "parameter_count": effective.get("parameter_count"),
            "trainable_parameter_count": effective.get(
                "trainable_parameter_count"
            ),
            "active_scope_id": E5_SCOPE27_ID,
            "training_batch_profile_id": (
                None
                if entry["entry_type"] == "REFERENCE_ONLY_FORMAL_A8"
                else TRAINING_PROFILE_ID
            ),
        }
        for horizon in HORIZONS:
            for metric in REQUIRED_METRICS:
                row[f"H{horizon}_{metric}"] = metrics[horizon][metric]
        row["average_Score"] = sum(
            float(metrics[horizon]["Score"]) for horizon in HORIZONS
        ) / len(HORIZONS)
        result.append(row)
    return result


def aggregate(
    manifest: Mapping[str, Any],
    output_root: Path,
    *,
    require_complete: bool,
    report_path: Path,
    evidence_path: Path,
) -> dict[str, Any]:
    if not require_complete:
        raise ScopeGateError("Final E5 aggregation requires --require-complete.")
    readiness = build_readiness(manifest, output_root)
    write_readiness_outputs(readiness, report_path, evidence_path)
    if readiness["status"] != "COMPLETED_READY_27_OF_27":
        ready = readiness["counts"]["total_evidence"]["ready"]
        raise ScopeGateError(f"E5_SCOPE27_NOT_READY:{ready}/27")
    rows = _aggregate_rows(manifest, readiness)
    if len(rows) != 27:
        raise ScopeGateError("Aggregate row count must be exactly 27.")

    fieldnames = list(rows[0])
    csv_path = output_root / "common_loss_architecture_scope27_seed2026.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    md_path = output_root / "COMMON_LOSS_ARCHITECTURE_SCOPE27_SEED2026.md"
    lines = [
        "# E5 common-loss architecture: batch4 scope27 seed2026",
        "",
        "| model | mode | checkpoint | H3 Score | H6 Score | H10 Score | average Score |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['training_mode']} | "
            f"{row['checkpoint'] or 'null'} | "
            f"{row['H3_Score']:.6f} | {row['H6_Score']:.6f} | "
            f"{row['H10_Score']:.6f} | {row['average_Score']:.6f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    audit_path = output_root / "common_loss_architecture_scope27_audit.json"
    atomic_write_json(
        audit_path,
        {
            "status": "PASS",
            "scope_id": E5_SCOPE27_ID,
            "require_complete": True,
            "row_count": 27,
            "readiness": readiness["counts"],
            "generated_at": utc_now(),
        },
    )
    xlsx_path = output_root / "COMMON_LOSS_ARCHITECTURE_SCOPE27_SEED2026.xlsx"
    try:
        from openpyxl import Workbook
    except ImportError as exc:
        raise ScopeGateError(
            "openpyxl is required for the complete E5 aggregate."
        ) from exc
    workbook = Workbook()
    trained = workbook.active
    trained.title = "Trainable structures"
    references = workbook.create_sheet("Evaluate-only and A8")
    all_rows = workbook.create_sheet("All 27")
    selections = (
        (
            trained,
            [row for row in rows if row["entry_type"] == "TRAIN_COMMON_LOSS"],
        ),
        (
            references,
            [row for row in rows if row["entry_type"] != "TRAIN_COMMON_LOSS"],
        ),
        (all_rows, rows),
    )
    for sheet, selected in selections:
        sheet.append(fieldnames)
        for row in selected:
            sheet.append([row.get(field) for field in fieldnames])
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
    workbook.save(xlsx_path)
    return {
        "status": "PASS",
        "scope_id": E5_SCOPE27_ID,
        "row_count": 27,
        "files": [
            str(xlsx_path),
            str(md_path),
            str(csv_path),
            str(audit_path),
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="e5_batch4_scope27_gate")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate-manifest")
    subparsers.add_parser("list-runnable")
    subparsers.add_parser("freeze")
    plan = subparsers.add_parser("plan-runs")
    plan.add_argument("--output-root", default=str(EXPECTED_OUTPUT_ROOT))
    dry_run = subparsers.add_parser("dry-run")
    dry_run.add_argument("--output-root", default=str(EXPECTED_OUTPUT_ROOT))
    inventory_parser = subparsers.add_parser("inventory")
    inventory_parser.add_argument("--output-root", default=str(EXPECTED_OUTPUT_ROOT))
    preflight_plan = subparsers.add_parser("preflight-plan")
    inspect = subparsers.add_parser("inspect-entry")
    inspect.add_argument("--model-id", required=True)
    inspect.add_argument("--output-root", default=str(EXPECTED_OUTPUT_ROOT))
    readiness = subparsers.add_parser("readiness")
    readiness.add_argument("--output-root", default=str(EXPECTED_OUTPUT_ROOT))
    readiness.add_argument(
        "--report-path", default=str(AUDIT_ROOT / "e5_scope27_readiness.json")
    )
    readiness.add_argument(
        "--evidence-path", default=str(AUDIT_ROOT / "e5_scope27_evidence.json")
    )
    lock_status_parser = subparsers.add_parser("lock-status")
    lock_status_parser.add_argument("--lock-path", default=str(LOCK_PATH))
    clear_lock = subparsers.add_parser("clear-stale-lock")
    clear_lock.add_argument("--lock-path", default=str(LOCK_PATH))
    quarantine = subparsers.add_parser("quarantine-existing")
    quarantine.add_argument("--model-id", required=True)
    quarantine.add_argument("--output-root", default=str(EXPECTED_OUTPUT_ROOT))
    quarantine.add_argument("--apply", action="store_true")
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--preflight-root", default=str(AUDIT_ROOT / "preflight"))
    preflight.add_argument("--source-revision")
    run = subparsers.add_parser("run")
    run.add_argument("--input-path")
    run.add_argument("--target-path")
    run.add_argument("--log-root", default=str(AUDIT_ROOT / "runs"))
    run.add_argument("--preflight-root", default=str(AUDIT_ROOT / "preflight"))
    run.add_argument("--source-revision")
    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument(
        "--output-root", default=str(EXPECTED_OUTPUT_ROOT)
    )
    aggregate_parser.add_argument(
        "--report-path", default=str(AUDIT_ROOT / "e5_scope27_readiness.json")
    )
    aggregate_parser.add_argument(
        "--evidence-path", default=str(AUDIT_ROOT / "e5_scope27_evidence.json")
    )
    aggregate_parser.add_argument("--require-complete", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        if args.command == "validate-manifest":
            print(
                json.dumps(
                    {
                        "status": "PASS",
                        "scope_id": E5_SCOPE27_ID,
                        "counts": manifest["counts"],
                    },
                    ensure_ascii=False,
                )
            )
            return 0
        if args.command == "freeze":
            print(json.dumps(compute_e5_freeze(manifest), ensure_ascii=False, indent=2))
            return 0
        if args.command == "list-runnable":
            for entry in manifest["entries"]:
                if entry["entry_type"] == "REFERENCE_ONLY_FORMAL_A8":
                    continue
                device = "cpu" if entry["command"] == "evaluate-only" else "cuda"
                print(
                    "\t".join(
                        (
                            entry["model_id"],
                            entry["command"],
                            entry["e5_run_id"],
                            device,
                        )
                    )
                )
            return 0
        if args.command in {"plan-runs", "dry-run"}:
            output_root = Path(args.output_root).resolve()
            plan_payload = build_plan(manifest, output_root)
            print(json.dumps(plan_payload, ensure_ascii=False, indent=2))
            return 0
        if args.command == "inventory":
            payload = inventory(manifest, Path(args.output_root).resolve())
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0
        if args.command == "preflight-plan":
            print(json.dumps(build_preflight_plan(manifest), ensure_ascii=False, indent=2))
            return 0
        if args.command == "lock-status":
            payload = lock_status(Path(args.lock_path).resolve())
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0 if payload["status"] == "ABSENT" else 74
        if args.command == "clear-stale-lock":
            payload = clear_stale_lock(Path(args.lock_path).resolve())
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0
        if args.command == "quarantine-existing":
            entry = _entry(manifest, args.model_id)
            if entry["entry_type"] == REFERENCE_ENTRY_TYPE:
                raise ScopeGateError("The A8 reference cannot be quarantined by the E5 run gate.")
            result = archive_or_quarantine(
                manifest,
                entry,
                Path(args.output_root).resolve(),
                kind=QUARANTINE_DIRECTORY,
                apply=args.apply,
                reason="explicit quarantine-existing --apply",
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "preflight":
            source_revision = args.source_revision or _source_revision()["git_commit"]
            preflight_root = Path(args.preflight_root).resolve()
            results = []
            from benchmark_v2.hardware_preflight import launch_preflight, read_matching_pass

            for entry in manifest["entries"]:
                if entry["entry_type"] != TRAINABLE_ENTRY_TYPE:
                    results.append({"model_id": entry["model_id"], "status": "SKIPPED_NON_TRAINABLE"})
                    continue
                code = launch_preflight(
                    entry["model_id"],
                    root=preflight_root,
                    experiment_profile=CLI_PROFILE_ID,
                    training_profile=TRAINING_PROFILE_ID,
                    formal_scope_id=E5_SCOPE27_ID,
                    source_revision=source_revision,
                )
                matched = read_matching_pass(
                    entry["model_id"],
                    root=preflight_root,
                    experiment_profile=CLI_PROFILE_ID,
                    training_profile=TRAINING_PROFILE_ID,
                    formal_scope_id=E5_SCOPE27_ID,
                    source_revision=source_revision,
                )
                results.append({"model_id": entry["model_id"], "exit_code": code, "status": "PASS" if code == 0 and matched else "FAIL"})
            passed = sum(row["status"] == "PASS" for row in results)
            payload = {"scope_id": E5_SCOPE27_ID, "status": "PASS" if passed == 24 else "FAIL", "counts": {"pass": passed, "expected": 24}, "entries": results}
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0 if passed == 24 else 3
        if args.command == "inspect-entry":
            entry = _entry(manifest, args.model_id)
            if entry["entry_type"] == "REFERENCE_ONLY_FORMAL_A8":
                result = inspect_a8(manifest, entry)
            else:
                result = inspect_run(
                    manifest, entry, Path(args.output_root).resolve()
                )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            if result["ready"]:
                return 0
            if not result["found"]:
                return 10
            return 20
        if args.command == "readiness":
            report = build_readiness(
                manifest, Path(args.output_root).resolve()
            )
            write_readiness_outputs(
                report,
                Path(args.report_path).resolve(),
                Path(args.evidence_path).resolve(),
            )
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["status"] == "COMPLETED_READY_27_OF_27" else 4
        if args.command == "run":
            source_revision = args.source_revision or _source_revision()["git_commit"]
            code, report = run_scope(
                manifest,
                input_path=args.input_path,
                target_path=args.target_path,
                log_root=Path(args.log_root).resolve(),
                preflight_root=Path(args.preflight_root).resolve(),
                source_revision=source_revision,
            )
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return code
        if args.command == "aggregate":
            result = aggregate(
                manifest,
                Path(args.output_root).resolve(),
                require_complete=args.require_complete,
                report_path=Path(args.report_path).resolve(),
                evidence_path=Path(args.evidence_path).resolve(),
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
    except (OSError, ValueError, KeyError, ScopeGateError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
