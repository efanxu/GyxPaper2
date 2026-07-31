from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "custom_models" / "src"
sys.path.insert(0, str(SRC_ROOT))

from benchmark_v2.experiments.e5_common_loss.contracts import (  # noqa: E402
    NONTRAINABLE_MODELS,
    TRAINABLE_MODELS,
)
from benchmark_v2.artifacts import PROFILES, validate_run  # noqa: E402
from benchmark_v2.experiments.e5_common_loss.loss_profile import (  # noqa: E402
    CLI_PROFILE_ID as E5_PROFILE_ID,
)
from benchmark_v2.hardware_preflight import (  # noqa: E402
    machine_identity,
    preflight_artifact_path,
    preflight_identity,
    stable_hash,
)
from benchmark_v2.protocol import load_protocol  # noqa: E402
from benchmark_v2.registry import load_registry  # noqa: E402
from benchmark_v2.training_profiles import (  # noqa: E402
    UNIFORM_BATCH4_PROFILE_ID,
    load_training_profile,
)
from st_mgprompt.config import STMGPromptConfig  # noqa: E402
from st_mgprompt.experiment_protocol import (  # noqa: E402
    apply_variant,
    canonical_config,
)


DOC_ROOT = (
    PROJECT_ROOT
    / "custom_models"
    / "docs"
    / "benchmark_v2"
    / "UNIFORM_BATCH4_PROTOCOL"
)
PREFLIGHT_ROOT = (
    PROJECT_ROOT
    / "custom_models"
    / "results_smoke"
    / "benchmark_v2_uniform_bs4"
    / "hardware_preflight"
)
ST_PREFLIGHT_ROOT = (
    PROJECT_ROOT
    / "custom_models"
    / "results_smoke"
    / "st_mgprompt_uniform_bs4"
    / "machine_preflight"
)
RUN_MAP_PATH = DOC_ROOT / "UNIFORM_BATCH4_RUN_ID_MAP.json"
PROFILE_ID = UNIFORM_BATCH4_PROFILE_ID
PROFILE_HASH = (
    "f58bbc161dfba0f00774879fdf2f78ec7c59a7faff9ef28a1733a2ed6835fe66"
)
PROTOCOL_HASH_PATH = DOC_ROOT / "UNIFORM_BATCH4_PROTOCOL_HASH.json"
SHAPE = {"B": 4, "T": 144, "N": 134, "C": 16, "H": 10}
DATASET_PATHS = (
    "dataset/sdwpf_model_input_base.parquet",
    "dataset/sdwpf_eval_target.parquet",
    "dataset/sdwpf_raw_aligned.parquet",
    "dataset/sdwpf_turb_location_elevation.csv",
)
MACHINE_FIELDS = (
    "machine_id",
    "hostname",
    "operating_system",
    "python_version",
    "pytorch_version",
    "cuda_version",
    "gpu_name",
    "gpu_uuid",
    "gpu_total_memory",
    "driver_version",
)
PASS_STATUSES = {"PASS", "passed"}
COMPLETED_STATUSES = {
    "COMPLETED",
    "PROCESS_FINISHED",
    "EVALUATION_FINISHED",
    "completed",
}
ORIGINAL_SKIP_MSGNET_SUITE = "original_skip_msgnet"
ORIGINAL_SKIP_MSGNET_MODEL = "msgnet"
ORIGINAL_SKIP_MSGNET_EXPECTED = 27
ORIGINAL_SKIP_MSGNET_SUITE_STATUS = "PARTIAL_ORIGINAL_SUITE"
ORIGINAL_SKIP_MSGNET_REASON = "BLOCKED_ON_THIS_MACHINE_OOM"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _run_capture(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def git_identity() -> dict[str, Any]:
    commit_override = os.environ.get("UNIFORM_BATCH4_GIT_COMMIT", "").strip()
    commit = _run_capture(["git", "rev-parse", "HEAD"])
    if commit.returncode != 0:
        return {
            "git_commit": commit_override or None,
            "git_metadata_available": False,
            "dirty": None,
            "dirty_files": [],
            "warning": (
                "Git metadata is unavailable in this frozen snapshot. "
                "Set UNIFORM_BATCH4_GIT_COMMIT on both machines when the "
                "deployment archive does not include .git."
            ),
        }
    status = _run_capture(["git", "status", "--porcelain=v1", "--untracked-files=all"])
    dirty_files = [
        line[3:]
        for line in status.stdout.splitlines()
        if len(line) >= 4 and line[3:].strip()
    ]
    return {
        "git_commit": commit_override or commit.stdout.strip(),
        "git_metadata_available": True,
        "dirty": bool(dirty_files),
        "dirty_files": dirty_files,
        "warning": None,
    }


def _protected_paths() -> list[str]:
    result: set[str] = set()
    for path in DOC_ROOT.glob("protected_*_before.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for row in payload.get("files", []):
            relative = row.get("path")
            if relative:
                result.add(str(relative).replace("\\", "/"))
    return sorted(result)


def _hash_inventory(paths: Iterable[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for relative in paths:
        path = PROJECT_ROOT / relative
        result[relative] = {
            "exists": path.is_file(),
            "sha256": sha256_file(path) if path.is_file() else None,
        }
    return result


def _without_machine(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in identity.items() if key not in MACHINE_FIELDS}


def current_freeze_identity() -> dict[str, Any]:
    profile = load_training_profile(PROFILE_ID)
    protocol = load_protocol()
    protocol_hash = json.loads(
        PROTOCOL_HASH_PATH.read_text(encoding="utf-8")
    )["protocol_hash"]
    protected_files = _hash_inventory(_protected_paths())
    dataset_hashes = _hash_inventory(DATASET_PATHS)
    original_models = {
        model: _without_machine(
            preflight_identity(model, training_profile=PROFILE_ID)
        )
        for model in TRAINABLE_MODELS
    }
    e5_models = {
        model: _without_machine(
            preflight_identity(
                model,
                experiment_profile=E5_PROFILE_ID,
                training_profile=PROFILE_ID,
            )
        )
        for model in TRAINABLE_MODELS
    }
    run_map_hash = sha256_file(RUN_MAP_PATH)
    payload = {
        **git_identity(),
        "base_benchmark_protocol_hash": protocol.protocol_hash,
        "uniform_batch4_protocol_hash": protocol_hash,
        "training_batch_profile_id": PROFILE_ID,
        "training_batch_profile_hash": profile.profile_hash,
        "graph_protocol_hash": original_models[TRAINABLE_MODELS[0]][
            "graph_protocol_hash"
        ],
        "graph_bundle_hash": original_models[TRAINABLE_MODELS[0]][
            "graph_bundle_hash"
        ],
        "node_order_hash": original_models[TRAINABLE_MODELS[0]][
            "node_order_hash"
        ],
        "dataset_hashes": dataset_hashes,
        "run_id_map_hash": run_map_hash,
        "original_model_identities_hash": stable_hash(original_models),
        "e5_model_identities_hash": stable_hash(e5_models),
        "protected_files_hash": stable_hash(protected_files),
        "seed": int(protocol["default_seed"]),
        "amp": bool(protocol["amp_enabled"]),
        **SHAPE,
    }
    comparable = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "git_metadata_available",
            "dirty",
            "dirty_files",
            "warning",
        }
    }
    payload["frozen_identity_hash"] = stable_hash(comparable)
    return payload


def freeze_identities_match(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> tuple[bool, list[str]]:
    required = (
        "git_commit",
        "base_benchmark_protocol_hash",
        "uniform_batch4_protocol_hash",
        "training_batch_profile_hash",
        "graph_protocol_hash",
        "graph_bundle_hash",
        "node_order_hash",
        "dataset_hashes",
        "run_id_map_hash",
        "original_model_identities_hash",
        "e5_model_identities_hash",
        "protected_files_hash",
        "seed",
        "amp",
        "B",
        "T",
        "N",
        "C",
        "H",
    )
    mismatches = [key for key in required if left.get(key) != right.get(key)]
    return not mismatches, mismatches


def expected_preflight_kind(suite: str) -> str:
    return {
        "original": "ORIGINAL_LOSS_ALL26",
        "e5": "E5_COMMON_LOSS_ALL26",
        "full": "FULL_EXACT",
        "a8": "A8_EXACT",
    }[suite]


def preflight_kind_matches(suite: str, artifact_kind: str) -> bool:
    return artifact_kind == expected_preflight_kind(suite)


def e5_core_allowed(
    *,
    e5_preflight_pass: bool,
    base28_complete: bool,
    a8_complete: bool,
) -> bool:
    del base28_complete, a8_complete
    return e5_preflight_pass


def parallel_execution_allowed(
    left_freeze: Mapping[str, Any],
    right_freeze: Mapping[str, Any],
    left_output_roots: Iterable[str],
    right_output_roots: Iterable[str],
) -> tuple[bool, list[str]]:
    matched, mismatches = freeze_identities_match(left_freeze, right_freeze)
    conflicts = sorted(set(left_output_roots) & set(right_output_roots))
    reasons = [f"freeze_identity.{key}" for key in mismatches]
    reasons.extend(f"output_root_conflict:{root}" for root in conflicts)
    return matched and not conflicts, reasons


def machine_identities_match(
    expected: Mapping[str, Any], current: Mapping[str, Any]
) -> tuple[bool, list[str]]:
    mismatches = [
        key for key in MACHINE_FIELDS if expected.get(key) != current.get(key)
    ]
    return not mismatches, mismatches


def machine_manifest_paths(identity: Mapping[str, Any] | None = None) -> tuple[Path, Path]:
    current = dict(identity or machine_identity())
    stem = f"UNIFORM_BATCH4_MACHINE_PREFLIGHT_{current['machine_id']}"
    return DOC_ROOT / f"{stem}.json", DOC_ROOT / f"{stem}.md"


def _new_machine_manifest() -> dict[str, Any]:
    return {
        "schema_version": "uniform_batch4_machine_preflight_v2",
        "LOCAL_DEVELOPMENT_MACHINE_STATUS": "BLOCKED_MACHINE_OOM",
        "LOCAL_DEVELOPMENT_MACHINE_ID": "GTX1060_WINDOWS",
        "TARGET_MACHINE_STATUS": "PENDING_TARGET_MACHINE_PREFLIGHT",
        "GLOBAL_ENGINEERING_STATUS": "READY_FOR_TARGET_MACHINE_VALIDATION",
        "machine_identity": machine_identity(),
        "freeze_identity": current_freeze_identity(),
        "suites": {},
        "updated_at": utc_now(),
    }


def load_machine_manifest() -> dict[str, Any]:
    json_path, _ = machine_manifest_paths()
    if not json_path.is_file():
        return _new_machine_manifest()
    return json.loads(json_path.read_text(encoding="utf-8"))


def _machine_status_from_suites(suites: Mapping[str, Any]) -> str:
    statuses = [str(value.get("status")) for value in suites.values()]
    if any(status == "BLOCKED_ON_THIS_MACHINE_NON_OOM" for status in statuses):
        return "BLOCKED_ON_THIS_MACHINE_NON_OOM"
    if any(status == "BLOCKED_ON_THIS_MACHINE_OOM" for status in statuses):
        return "BLOCKED_ON_THIS_MACHINE_OOM"
    if statuses and all(status == "READY_FOR_FORMAL_RUN" for status in statuses):
        return "READY_FOR_FORMAL_RUN"
    return "PENDING_TARGET_MACHINE_PREFLIGHT"


def _write_machine_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    identity = payload["machine_identity"]
    lines = [
        f"# Uniform Batch4 machine preflight: {identity['machine_id']}",
        "",
        f"- TARGET_MACHINE_STATUS: `{payload['TARGET_MACHINE_STATUS']}`",
        "- GLOBAL_ENGINEERING_STATUS: `READY_FOR_TARGET_MACHINE_VALIDATION`",
        "- LOCAL_DEVELOPMENT_MACHINE_ID: `GTX1060_WINDOWS`",
        "- LOCAL_DEVELOPMENT_MACHINE_STATUS: `BLOCKED_MACHINE_OOM`",
        f"- Hostname: `{identity['hostname']}`",
        f"- OS: `{identity['operating_system']}`",
        f"- GPU: `{identity.get('gpu_name')}`",
        f"- GPU UUID: `{identity.get('gpu_uuid')}`",
        f"- Profile: `{PROFILE_ID}` / `{PROFILE_HASH}`",
        "",
        "## Suite results",
        "",
        "| suite | status | PASS | FAIL_OOM | FAIL_NON_OOM | NOT_RUN |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for suite, result in sorted(payload.get("suites", {}).items()):
        counts = result.get("counts", {})
        lines.append(
            f"| {suite} | {result.get('status')} | {counts.get('PASS', 0)} | "
            f"{counts.get('FAIL_OOM', 0)} | {counts.get('FAIL_NON_OOM', 0)} | "
            f"{counts.get('NOT_RUN', 0)} |"
        )
    lines.extend(
        [
            "",
            "Historical GTX 1060 FAIL_OOM evidence is retained and is not a "
            "gate for this machine. Only PASS artifacts produced by this exact "
            "machine and frozen identity can release a formal suite.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_machine_manifest(payload: Mapping[str, Any]) -> None:
    json_path, md_path = machine_manifest_paths(payload["machine_identity"])
    write_json(json_path, payload)
    _write_machine_markdown(md_path, payload)


def _validate_manifest_binding(payload: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    current_machine = machine_identity()
    machine_match, machine_mismatches = machine_identities_match(
        payload.get("machine_identity", {}), current_machine
    )
    if not machine_match:
        blockers.extend(
            f"machine_identity.{key}: mismatch" for key in machine_mismatches
        )
    current_freeze = current_freeze_identity()
    if not current_freeze.get("git_commit"):
        blockers.append(
            "freeze_identity.git_commit: unavailable; set "
            "UNIFORM_BATCH4_GIT_COMMIT to the frozen source commit"
        )
    matched, mismatches = freeze_identities_match(
        payload.get("freeze_identity", {}), current_freeze
    )
    blockers.extend(f"freeze_identity.{key}: mismatch" for key in mismatches)
    if current_freeze["training_batch_profile_hash"] != PROFILE_HASH:
        blockers.append("training_batch_profile_hash: unexpected frozen value")
    return blockers


def _artifact_row(
    model: str,
    *,
    experiment_profile: str | None,
    exit_code: int | None,
    log_path: Path | None,
) -> dict[str, Any]:
    path = preflight_artifact_path(
        model,
        root=PREFLIGHT_ROOT,
        experiment_profile=experiment_profile,
        training_profile=PROFILE_ID,
    )
    if not path.is_file():
        return {
            "model_id": model,
            "status": "NOT_RUN",
            "exit_code": exit_code,
            "artifact_path": str(path),
            "log_path": None if log_path is None else str(log_path),
            "blocked_reason": "MISSING_CURRENT_MACHINE_ARTIFACT",
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = preflight_identity(
        model,
        experiment_profile=experiment_profile,
        training_profile=PROFILE_ID,
    )
    mismatches = [
        key for key, value in expected.items() if payload.get(key) != value
    ]
    status = str(payload.get("status", "FAIL_NON_OOM"))
    if mismatches:
        status = "FAIL_NON_OOM"
    return {
        **payload,
        "status": status,
        "exit_code": exit_code,
        "artifact_path": str(path),
        "artifact_sha256": sha256_file(path),
        "log_path": None if log_path is None else str(log_path),
        "identity_mismatches": mismatches,
        "blocked_reason": (
            None if not mismatches else f"IDENTITY_MISMATCH: {', '.join(mismatches)}"
        ),
    }


def status_from_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[str, dict[str, int]]:
    counts = {
        status: sum(str(row.get("status")) == status for row in rows)
        for status in ("PASS", "FAIL_OOM", "FAIL_NON_OOM", "NOT_RUN")
    }
    if counts["FAIL_NON_OOM"]:
        status = "BLOCKED_ON_THIS_MACHINE_NON_OOM"
    elif counts["FAIL_OOM"]:
        status = "BLOCKED_ON_THIS_MACHINE_OOM"
    elif counts["NOT_RUN"]:
        status = "PENDING_TARGET_MACHINE_PREFLIGHT"
    else:
        status = "READY_FOR_FORMAL_RUN"
    return status, counts


def _evaluate_only_prechecks(suite: str) -> list[dict[str, Any]]:
    identity = machine_identity()
    registry = load_registry()
    rows = []
    for model in NONTRAINABLE_MODELS:
        entry = registry.get(model)
        source = PROJECT_ROOT / str(entry.source_path)
        missing_data = [
            relative
            for relative in DATASET_PATHS[:2]
            if not (PROJECT_ROOT / relative).is_file()
        ]
        passed = (
            bool(entry.supports_non_trainable)
            and bool(entry.supports_evaluate)
            and source.is_file()
            and not missing_data
        )
        row = {
            **identity,
            "suite": suite,
            "model_id": model,
            "mode": "EVALUATE_ONLY_PRECHECK",
            "status": "PASS" if passed else "FAIL_NON_OOM",
            "source_hash": sha256_file(source) if source.is_file() else None,
            "missing_data": missing_data,
            "loss_id": "NOT_APPLICABLE",
            "created_at": utc_now(),
        }
        target = (
            PREFLIGHT_ROOT
            / "evaluate_only"
            / identity["machine_id"]
            / suite
            / model
            / "evaluate_only_precheck.json"
        )
        if not target.is_file():
            write_json(target, row)
        else:
            row = json.loads(target.read_text(encoding="utf-8"))
        row["artifact_path"] = str(target)
        row["artifact_sha256"] = sha256_file(target)
        rows.append(row)
    return rows


def _unique_log_path(suite: str, model: str) -> Path:
    root = (
        PROJECT_ROOT
        / "custom_models"
        / "logs"
        / "uniform_bs4"
        / "preflight"
        / machine_identity()["machine_id"]
        / suite
    )
    root.mkdir(parents=True, exist_ok=True)
    base = root / f"{model}.log"
    if not base.exists():
        return base
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return root / f"{model}_{stamp}_{os.getpid()}.log"


def run_inventory_preflight(suite: str, python_executable: str) -> int:
    if suite not in {"original", "e5"}:
        raise ValueError(f"Unknown inventory suite: {suite}")
    experiment_profile = E5_PROFILE_ID if suite == "e5" else None
    existing_manifest = load_machine_manifest()
    previous_rows = {
        row.get("model_id"): row
        for row in existing_manifest.get("suites", {})
        .get(suite, {})
        .get("models", [])
    }
    exits: dict[str, tuple[int | None, Path | None]] = {}
    for model in TRAINABLE_MODELS:
        artifact = preflight_artifact_path(
            model,
            root=PREFLIGHT_ROOT,
            experiment_profile=experiment_profile,
            training_profile=PROFILE_ID,
        )
        if artifact.is_file():
            previous = previous_rows.get(model, {})
            previous_log = previous.get("log_path")
            exits[model] = (
                previous.get("exit_code"),
                None if not previous_log else Path(previous_log),
            )
            print(f"[{suite}] {model}: preserving existing artifact {artifact}")
            continue
        log_path = _unique_log_path(suite, model)
        command = [
            python_executable,
            str(PROJECT_ROOT / "custom_models/src/benchmark_v2/run_benchmark.py"),
            "hardware-preflight",
            "--model",
            model,
            "--training-profile",
            PROFILE_ID,
            "--preflight-root",
            str(PREFLIGHT_ROOT),
        ]
        if experiment_profile:
            command.extend(["--experiment-profile", experiment_profile])
        print(f"[{suite}] {model}: starting isolated exact preflight")
        with log_path.open("x", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                check=False,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        exits[model] = (int(completed.returncode), log_path)
        print(f"[{suite}] {model}: exit={completed.returncode}; log={log_path}")
    rows = [
        _artifact_row(
            model,
            experiment_profile=experiment_profile,
            exit_code=exits[model][0],
            log_path=exits[model][1],
        )
        for model in TRAINABLE_MODELS
    ]
    eval_rows = _evaluate_only_prechecks(suite)
    status, counts = status_from_rows([*rows, *eval_rows])
    suite_result = {
        "suite": suite,
        "preflight_kind": (
            "E5_COMMON_LOSS_ALL26"
            if suite == "e5"
            else "ORIGINAL_LOSS_ALL26"
        ),
        "loss_id": (
            "masked_score_aligned_hybrid" if suite == "e5" else "masked_mse"
        ),
        "status": status,
        "models": rows,
        "evaluate_only_prechecks": eval_rows,
        "counts": counts,
        "completed_at": utc_now(),
    }
    manifest = existing_manifest
    manifest["machine_identity"] = machine_identity()
    manifest["freeze_identity"] = current_freeze_identity()
    manifest.setdefault("suites", {})[suite] = suite_result
    manifest["TARGET_MACHINE_STATUS"] = _machine_status_from_suites(
        manifest["suites"]
    )
    manifest["updated_at"] = utc_now()
    save_machine_manifest(manifest)
    return 0 if status == "READY_FOR_FORMAL_RUN" else 3


def _st_source_closure_hash() -> str:
    rows = {
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256_file(path)
        for path in sorted(
            (PROJECT_ROOT / "custom_models/src/st_mgprompt").rglob("*.py")
        )
    }
    return stable_hash(rows)


def st_preflight_identity(suite: str) -> dict[str, Any]:
    if suite not in {"full", "a8"}:
        raise ValueError(suite)
    cfg = (
        canonical_config(STMGPromptConfig())
        if suite == "full"
        else apply_variant(STMGPromptConfig(), "A8", family="component_ablation")
    )
    profile = load_training_profile(PROFILE_ID)
    cfg.batch_size = profile.train_batch_size
    cfg.eval_batch_size = profile.val_batch_size
    cfg.train_batch_size = profile.train_batch_size
    cfg.val_batch_size = profile.val_batch_size
    cfg.test_batch_size = profile.test_batch_size
    cfg.amp_enabled = True
    config = cfg.to_dict()
    for key in ("run_id", "output_root", "device"):
        config.pop(key, None)
    freeze = current_freeze_identity()
    return {
        **machine_identity(),
        "object_id": "st_mgprompt_full" if suite == "full" else "st_mgprompt_a8",
        "preflight_kind": "FULL_EXACT" if suite == "full" else "A8_EXACT",
        "model_config_hash": stable_hash(config),
        "model_source_closure_hash": _st_source_closure_hash(),
        "base_benchmark_protocol_hash": freeze["base_benchmark_protocol_hash"],
        "uniform_batch4_protocol_hash": freeze["uniform_batch4_protocol_hash"],
        "training_batch_profile_id": PROFILE_ID,
        "training_batch_profile_hash": PROFILE_HASH,
        "loss_id": (
            "msmg_dwu_loss"
            if suite == "full"
            else "masked_score_aligned_hybrid"
        ),
        "graph_protocol_hash": freeze["graph_protocol_hash"],
        "graph_bundle_hash": freeze["graph_bundle_hash"],
        "node_order_hash": freeze["node_order_hash"],
        "seed": 2026,
        "amp": True,
        **SHAPE,
    }


def st_preflight_artifact_path(suite: str) -> Path:
    identity = st_preflight_identity(suite)
    key = stable_hash(identity)[:20]
    return (
        ST_PREFLIGHT_ROOT
        / identity["machine_id"]
        / suite
        / key
        / "hardware_preflight.json"
    )


def run_st_preflight(suite: str, python_executable: str) -> int:
    identity = st_preflight_identity(suite)
    artifact = st_preflight_artifact_path(suite)
    if artifact.is_file():
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        print(f"[{suite}] preserving existing artifact {artifact}")
        suite_status, counts = status_from_rows([payload])
        manifest = load_machine_manifest()
        manifest["machine_identity"] = machine_identity()
        manifest["freeze_identity"] = current_freeze_identity()
        manifest.setdefault("suites", {})[suite] = {
            "suite": suite,
            "preflight_kind": identity["preflight_kind"],
            "loss_id": identity["loss_id"],
            "status": suite_status,
            "models": [payload],
            "evaluate_only_prechecks": [],
            "counts": counts,
            "completed_at": payload.get("completed_at", utc_now()),
        }
        manifest["TARGET_MACHINE_STATUS"] = _machine_status_from_suites(
            manifest["suites"]
        )
        manifest["updated_at"] = utc_now()
        save_machine_manifest(manifest)
        return 0 if payload.get("status") == "PASS" else 3
    run_stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_id = f"preflight_{suite}_{identity['machine_id']}_{run_stamp}_{os.getpid()}"
    output_root = ST_PREFLIGHT_ROOT / identity["machine_id"] / suite / "runs"
    log_path = _unique_log_path(suite, f"st_mgprompt_{suite}")
    command = [
        python_executable,
        str(PROJECT_ROOT / "custom_models/src/st_mgprompt/run_st_mgprompt.py"),
        "--preflight-full-shape",
        "--training-profile",
        PROFILE_ID,
        "--run-id",
        run_id,
        "--output-root",
        str(output_root),
        "--amp",
    ]
    if suite == "full":
        command.append("--canonical-full")
    else:
        command.extend(["--component-ablation", "A8"])
    if platform.system() == "Windows":
        command.append("--windows-safe-mode")
    started = utc_now()
    with log_path.open("x", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    reports = list((output_root / run_id).rglob("full_shape_smoke_report.json"))
    report = (
        json.loads(reports[0].read_text(encoding="utf-8"))
        if len(reports) == 1
        else None
    )
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    if completed.returncode == 0 and report and report.get("status") in PASS_STATUSES:
        status = "PASS"
    elif "out of memory" in log_text.casefold():
        status = "FAIL_OOM"
    else:
        status = "FAIL_NON_OOM"
    payload = {
        **identity,
        "status": status,
        "forward_completed": bool(report),
        "backward_completed": bool(report),
        "strict_reload_completed": bool(
            report and report.get("strict_reload_completed")
        ),
        "exit_code": int(completed.returncode),
        "run_id": run_id,
        "run_root": str(output_root / run_id),
        "source_report": None if not reports else str(reports[0]),
        "log_path": str(log_path),
        "traceback_preserved_in_log": True,
        "started_at": started,
        "completed_at": utc_now(),
    }
    write_json(artifact, payload)
    suite_status, counts = status_from_rows([payload])
    suite_result = {
        "suite": suite,
        "preflight_kind": identity["preflight_kind"],
        "loss_id": identity["loss_id"],
        "status": suite_status,
        "models": [payload],
        "evaluate_only_prechecks": [],
        "counts": counts,
        "completed_at": utc_now(),
    }
    manifest = load_machine_manifest()
    manifest["machine_identity"] = machine_identity()
    manifest["freeze_identity"] = current_freeze_identity()
    manifest.setdefault("suites", {})[suite] = suite_result
    manifest["TARGET_MACHINE_STATUS"] = _machine_status_from_suites(
        manifest["suites"]
    )
    manifest["updated_at"] = utc_now()
    save_machine_manifest(manifest)
    return 0 if status == "PASS" else 3


def verify_suite(suite: str) -> list[str]:
    manifest = load_machine_manifest()
    blockers = _validate_manifest_binding(manifest)
    suite_result = manifest.get("suites", {}).get(suite)
    if not suite_result:
        return [*blockers, f"{suite}: missing current-machine preflight summary"]
    expected_kind = expected_preflight_kind(suite)
    if not preflight_kind_matches(
        suite, str(suite_result.get("preflight_kind"))
    ):
        blockers.append(
            f"{suite}: preflight kind mismatch "
            f"({suite_result.get('preflight_kind')} != {expected_kind})"
        )
    if suite_result.get("status") != "READY_FOR_FORMAL_RUN":
        blockers.append(f"{suite}: status={suite_result.get('status')}")
    if suite in {"original", "e5"}:
        experiment_profile = E5_PROFILE_ID if suite == "e5" else None
        rows = {row.get("model_id"): row for row in suite_result.get("models", [])}
        for model in TRAINABLE_MODELS:
            row = rows.get(model)
            if not row:
                blockers.append(f"{model}: missing preflight row")
                continue
            expected = preflight_identity(
                model,
                experiment_profile=experiment_profile,
                training_profile=PROFILE_ID,
            )
            mismatches = [
                key for key, value in expected.items() if row.get(key) != value
            ]
            if row.get("status") != "PASS":
                blockers.append(f"{model}: {row.get('status')}")
            if mismatches:
                blockers.append(
                    f"{model}: identity mismatch ({', '.join(mismatches)})"
                )
        eval_rows = {
            row.get("model_id"): row
            for row in suite_result.get("evaluate_only_prechecks", [])
        }
        for model in NONTRAINABLE_MODELS:
            if eval_rows.get(model, {}).get("status") != "PASS":
                blockers.append(f"{model}: evaluate-only precheck not PASS")
    else:
        rows = suite_result.get("models", [])
        expected = st_preflight_identity(suite)
        if len(rows) != 1:
            blockers.append(f"{suite}: expected one exact preflight artifact")
        else:
            row = rows[0]
            mismatches = [
                key for key, value in expected.items() if row.get(key) != value
            ]
            if row.get("status") != "PASS":
                blockers.append(f"{suite}: {row.get('status')}")
            if mismatches:
                blockers.append(
                    f"{suite}: identity mismatch ({', '.join(mismatches)})"
                )
    return blockers


def original_skip_msgnet_preflight_blockers(
    rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    blockers: list[str] = []
    rows_by_model: dict[str, Mapping[str, Any]] = {}
    duplicates: set[str] = set()
    for row in rows:
        model = str(row.get("model_id"))
        if model in rows_by_model:
            duplicates.add(model)
        rows_by_model[model] = row
    blockers.extend(f"{model}: duplicate preflight row" for model in sorted(duplicates))
    unexpected = sorted(set(rows_by_model) - set(TRAINABLE_MODELS))
    blockers.extend(f"{model}: unexpected preflight row" for model in unexpected)
    for model in TRAINABLE_MODELS:
        row = rows_by_model.get(model)
        if row is None:
            blockers.append(f"{model}: missing preflight row")
            continue
        expected_status = (
            "FAIL_OOM" if model == ORIGINAL_SKIP_MSGNET_MODEL else "PASS"
        )
        actual_status = str(row.get("status"))
        if actual_status != expected_status:
            blockers.append(
                f"{model}: status={actual_status}; expected {expected_status}"
            )
        identity_mismatches = row.get("identity_mismatches", [])
        if identity_mismatches:
            blockers.append(
                f"{model}: identity mismatch ({', '.join(identity_mismatches)})"
            )
    return blockers


def msgnet_has_fail_oom_evidence(row: Mapping[str, Any]) -> bool:
    if row.get("model_id") != ORIGINAL_SKIP_MSGNET_MODEL:
        return False
    if row.get("status") != "FAIL_OOM":
        return False
    evidence = " ".join(
        str(row.get(field, ""))
        for field in ("error_type", "error_message", "traceback")
    ).lower()
    return "outofmemory" in evidence or "out of memory" in evidence


def verify_original_skip_msgnet_suite() -> list[str]:
    manifest = load_machine_manifest()
    blockers = _validate_manifest_binding(manifest)
    suite_result = manifest.get("suites", {}).get("original")
    if not suite_result:
        return [
            *blockers,
            "original: missing current-machine preflight summary",
        ]
    if not preflight_kind_matches(
        "original", str(suite_result.get("preflight_kind"))
    ):
        blockers.append(
            "original: preflight kind mismatch "
            f"({suite_result.get('preflight_kind')} != "
            f"{expected_preflight_kind('original')})"
        )

    summary_rows = {
        row.get("model_id"): row for row in suite_result.get("models", [])
    }
    artifact_rows: list[dict[str, Any]] = []
    for model in TRAINABLE_MODELS:
        artifact_row = _artifact_row(
            model,
            experiment_profile=None,
            exit_code=None,
            log_path=None,
        )
        artifact_rows.append(artifact_row)
        summary_row = summary_rows.get(model)
        if summary_row is None:
            blockers.append(f"{model}: missing machine-manifest preflight row")
            continue
        for field in ("status", "artifact_path", "artifact_sha256"):
            if summary_row.get(field) != artifact_row.get(field):
                blockers.append(
                    f"{model}: machine-manifest {field} does not match "
                    "current artifact"
                )
    blockers.extend(
        original_skip_msgnet_preflight_blockers(
            suite_result.get("models", [])
        )
    )
    blockers.extend(original_skip_msgnet_preflight_blockers(artifact_rows))
    msgnet_artifact = next(
        (
            row
            for row in artifact_rows
            if row.get("model_id") == ORIGINAL_SKIP_MSGNET_MODEL
        ),
        {},
    )
    if not msgnet_has_fail_oom_evidence(msgnet_artifact):
        blockers.append(
            "msgnet: FAIL_OOM artifact lacks OutOfMemoryError evidence"
        )

    eval_rows = {
        row.get("model_id"): row
        for row in suite_result.get("evaluate_only_prechecks", [])
    }
    current_machine = machine_identity()
    registry = load_registry()
    for model in NONTRAINABLE_MODELS:
        row = eval_rows.get(model)
        if row is None or row.get("status") != "PASS":
            blockers.append(f"{model}: evaluate-only precheck not PASS")
            continue
        source_path = PROJECT_ROOT / str(registry.get(model).source_path)
        expected_source_hash = (
            sha256_file(source_path) if source_path.is_file() else None
        )
        if row.get("source_hash") != expected_source_hash:
            blockers.append(f"{model}: evaluate-only source identity mismatch")
        matched, mismatches = machine_identities_match(row, current_machine)
        if not matched:
            blockers.append(
                f"{model}: evaluate-only machine mismatch "
                f"({', '.join(mismatches)})"
            )
        artifact_path = Path(str(row.get("artifact_path", "")))
        if not artifact_path.is_file():
            blockers.append(f"{model}: evaluate-only artifact missing")
        elif row.get("artifact_sha256") != sha256_file(artifact_path):
            blockers.append(f"{model}: evaluate-only artifact hash mismatch")
    return blockers


def _run_map_rows(experiment_type: str) -> list[dict[str, Any]]:
    payload = json.loads(RUN_MAP_PATH.read_text(encoding="utf-8"))
    return [
        row for row in payload["runs"] if row["experiment_type"] == experiment_type
    ]


def original_skip_msgnet_run_rows(
    rows: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    original_rows = [
        dict(row)
        for row in (
            rows if rows is not None else _run_map_rows("original_benchmark")
        )
    ]
    msgnet_rows = [
        row
        for row in original_rows
        if row.get("model_id") == ORIGINAL_SKIP_MSGNET_MODEL
    ]
    if len(original_rows) != 28:
        raise ValueError(
            f"Original run map must contain exactly 28 rows, got {len(original_rows)}"
        )
    if len(msgnet_rows) != 1 or msgnet_rows[0].get("mode") != "TRAIN":
        raise ValueError("Original run map must contain one trainable msgnet row")
    expected_models = [*NONTRAINABLE_MODELS, *TRAINABLE_MODELS]
    actual_models = [row.get("model_id") for row in original_rows]
    if actual_models != expected_models:
        raise ValueError(
            "Original run map order must be Persistence, MovingAverage, "
            "then all 26 trainable models in registry order"
        )
    for row in original_rows:
        expected_mode = (
            "EVALUATE_ONLY"
            if row.get("model_id") in NONTRAINABLE_MODELS
            else "TRAIN"
        )
        if row.get("mode") != expected_mode:
            raise ValueError(
                f"{row.get('model_id')}: mode={row.get('mode')}; "
                f"expected {expected_mode}"
            )
    selected = [
        row
        for row in original_rows
        if row.get("model_id") != ORIGINAL_SKIP_MSGNET_MODEL
    ]
    if len(selected) != ORIGINAL_SKIP_MSGNET_EXPECTED:
        raise ValueError(
            "MSGNet-only partial suite must contain exactly "
            f"{ORIGINAL_SKIP_MSGNET_EXPECTED} rows"
        )
    return selected


def run_id_collision_blockers(
    rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    blockers = []
    for row in rows:
        run_root = (
            PROJECT_ROOT / str(row["output_root"]) / str(row["new_run_id"])
        )
        if run_root.exists():
            blockers.append(
                f"{row['model_id']}: run-id already exists: {run_root}"
            )
    return blockers


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _expected_artifact_profile(row: Mapping[str, Any]) -> str:
    return "NON_TRAINABLE" if row["mode"] == "EVALUATE_ONLY" else "TRAIN"


def _freeze_mismatches_for_resume(
    recorded: Mapping[str, Any], current: Mapping[str, Any]
) -> list[str]:
    matched, mismatches = freeze_identities_match(recorded, current)
    if not matched:
        return mismatches
    exact_keys = ("training_batch_profile_id", "frozen_identity_hash")
    return [key for key in exact_keys if recorded.get(key) != current.get(key)]


def _execution_receipt_for_run(
    row: Mapping[str, Any],
    *,
    execution_manifest: Mapping[str, Any],
    current_freeze: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    """Find the immutable execution record that binds this run-id to a freeze.

    Older formal artifacts do not contain a per-run freeze receipt.  The
    machine execution manifest was written before those processes were
    launched, so it is the authoritative compatibility receipt for those
    artifacts as well as newly completed runs.
    """

    recorded_freeze = execution_manifest.get("freeze_identity")
    if not isinstance(recorded_freeze, Mapping):
        return False, ["execution_manifest.freeze_identity"]
    freeze_mismatches = _freeze_mismatches_for_resume(
        recorded_freeze, current_freeze
    )
    if freeze_mismatches:
        return False, [f"freeze_identity.{key}" for key in freeze_mismatches]
    for execution in execution_manifest.get("executions", []):
        if not isinstance(execution, Mapping):
            continue
        if execution.get("suite") not in {
            "original",
            ORIGINAL_SKIP_MSGNET_SUITE,
        }:
            continue
        formal_models = execution.get("formal_models", [])
        run_ids = execution.get("run_ids", [])
        if not isinstance(formal_models, list) or not isinstance(run_ids, list):
            continue
        if any(
            model == row["model_id"] and run_id == row["new_run_id"]
            for model, run_id in zip(formal_models, run_ids)
        ):
            return True, []
    return False, ["execution_receipt.model_id_run_id"]


def _artifact_identity_mismatches(
    row: Mapping[str, Any],
    *,
    payloads: Mapping[str, dict[str, Any] | None],
    receipt_found: bool,
    receipt_mismatches: Sequence[str],
    current_freeze: Mapping[str, Any],
    require_completed_artifacts: bool,
) -> list[str]:
    """Return every identity field that prevents a safe fixed-id reuse."""

    expected_profile = _expected_artifact_profile(row)
    expected_formal_training = row["mode"] != "EVALUATE_ONLY"
    expected_protocol_hash = current_freeze["base_benchmark_protocol_hash"]
    expected_batch = {
        "training_batch_profile_id": PROFILE_ID,
        "training_batch_profile_hash": PROFILE_HASH,
        "train_batch_size": 4,
        "val_batch_size": 4,
        "test_batch_size": 4,
        "gradient_accumulation_steps": 1,
        "effective_train_batch_size": 4,
    }
    mismatches: list[str] = []

    def require(
        payload_name: str, field: str, expected: Any
    ) -> None:
        payload = payloads.get(payload_name)
        if not isinstance(payload, Mapping) or payload.get(field) != expected:
            mismatches.append(f"{payload_name}.{field}")

    for payload_name in (
        "effective_config",
        "resolved_config",
        "artifact_manifest",
    ):
        require(payload_name, "model_id", row["model_id"])

    for payload_name in (
        "effective_config",
        "resolved_config",
        "artifact_manifest",
    ):
        require(payload_name, "run_mode", "formal")
        require(payload_name, "artifact_profile", expected_profile)

    require("effective_config", "formal_training", expected_formal_training)
    require("resolved_config", "formal_training", expected_formal_training)
    require("artifact_manifest", "formal_training", expected_formal_training)

    for payload_name in (
        "effective_config",
        "resolved_config",
        "artifact_manifest",
        "protocol_check",
    ):
        require(payload_name, "protocol_hash", expected_protocol_hash)
    for payload_name in (
        "effective_config",
        "resolved_config",
        "protocol_check",
    ):
        for field, expected in expected_batch.items():
            require(payload_name, field, expected)
    for payload_name in ("effective_config", "resolved_config"):
        if expected_formal_training:
            require(payload_name, "seed", int(current_freeze["seed"]))
            require(payload_name, "loss", row["loss_id"])

    if require_completed_artifacts:
        for payload_name in ("model_summary", "prediction_metadata"):
            require(payload_name, "model_id", row["model_id"])
        require(
            "prediction_metadata", "protocol_hash", expected_protocol_hash
        )

    if not receipt_found:
        mismatches.extend(receipt_mismatches)
    return sorted(set(mismatches))


def classify_formal_resume_evidence(
    row: Mapping[str, Any], evidence: Mapping[str, Any]
) -> dict[str, Any]:
    """Make the fail-closed resume decision from gathered read-only evidence."""

    if not evidence.get("run_root_exists"):
        return {"decision": "WILL_RUN", "reason": "run-id directory is absent"}
    identity_mismatches = list(evidence.get("identity_mismatches", []))
    if identity_mismatches:
        return {
            "decision": "BLOCKED_IDENTITY_MISMATCH",
            "reason": ", ".join(identity_mismatches),
        }
    if evidence.get("active_run"):
        return {
            "decision": "BLOCKED_IDENTITY_MISMATCH",
            "reason": "existing run_status is RUNNING with a live worker pid",
        }
    if evidence.get("run_status") != "COMPLETED":
        return {
            "decision": "WILL_RUN",
            "reason": "identity-matched incomplete attempt will be archived",
        }
    missing = list(evidence.get("required_files_missing", []))
    if missing:
        return {
            "decision": "BLOCKED_IDENTITY_MISMATCH",
            "reason": "missing required artifact files: " + ", ".join(missing),
        }
    validation = evidence.get("artifact_validation")
    if not isinstance(validation, Mapping) or validation.get("status") != "PASS":
        return {
            "decision": "BLOCKED_IDENTITY_MISMATCH",
            "reason": "artifact_validation.status is not PASS",
        }
    return {
        "decision": "SKIP_COMPLETED",
        "reason": "COMPLETED artifact validation and identity matched",
    }


def _pid_is_running(pid: Any) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def assess_formal_resume(
    row: Mapping[str, Any],
    *,
    current_freeze: Mapping[str, Any],
    execution_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Assess one fixed run-id directory without modifying it."""

    run_root = PROJECT_ROOT / str(row["output_root"]) / str(row["new_run_id"])
    result: dict[str, Any] = {
        "model_id": row["model_id"],
        "run_id": row["new_run_id"],
        "run_root": str(run_root),
    }
    if not run_root.is_dir():
        result.update(classify_formal_resume_evidence(row, {"run_root_exists": False}))
        return result

    payloads = {
        name: _read_json_object(run_root / f"{name}.json")
        for name in (
            "run_status",
            "effective_config",
            "resolved_config",
            "artifact_manifest",
            "protocol_check",
            "model_summary",
            "prediction_metadata",
        )
    }
    run_status_payload = payloads["run_status"]
    run_status = (
        None
        if run_status_payload is None
        else str(run_status_payload.get("status", ""))
    )
    receipt_found, receipt_mismatches = _execution_receipt_for_run(
        row,
        execution_manifest=execution_manifest,
        current_freeze=current_freeze,
    )
    identity_mismatches = _artifact_identity_mismatches(
        row,
        payloads=payloads,
        receipt_found=receipt_found,
        receipt_mismatches=receipt_mismatches,
        current_freeze=current_freeze,
        require_completed_artifacts=run_status == "COMPLETED",
    )
    status_profile = (
        None
        if run_status_payload is None
        else run_status_payload.get("artifact_profile")
    )
    expected_profile = _expected_artifact_profile(row)
    if run_status == "COMPLETED" and status_profile != expected_profile:
        identity_mismatches.append("run_status.artifact_profile")
    if run_status == "COMPLETED" and run_status_payload.get("exit_code") != 0:
        identity_mismatches.append("run_status.exit_code")

    required_files = PROFILES[expected_profile]
    missing = sorted(name for name in required_files if not (run_root / name).is_file())
    validation: dict[str, Any] | None = None
    if run_status == "COMPLETED" and not missing and not identity_mismatches:
        try:
            validation = validate_run(
                run_root,
                expected_protocol_hash=str(current_freeze["base_benchmark_protocol_hash"]),
                expected_training_batch_profile_id=PROFILE_ID,
                expected_training_batch_profile_hash=PROFILE_HASH,
            )
        except Exception as exc:
            validation = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}

    evidence = {
        "run_root_exists": True,
        "run_status": run_status,
        "identity_mismatches": sorted(set(identity_mismatches)),
        "active_run": run_status == "RUNNING"
        and _pid_is_running(
            payloads["effective_config"].get("formal_worker_pid")
            if isinstance(payloads["effective_config"], Mapping)
            else None
        ),
        "required_files_missing": missing,
        "artifact_validation": validation,
    }
    result.update(classify_formal_resume_evidence(row, evidence))
    result["artifact_validation"] = validation
    result["identity_mismatches"] = evidence["identity_mismatches"]
    return result


def formal_resume_plan(
    rows: Sequence[Mapping[str, Any]],
    *,
    current_freeze: Mapping[str, Any] | None = None,
    execution_manifest: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    freeze = dict(current_freeze or current_freeze_identity())
    manifest = dict(execution_manifest or _load_execution_manifest())
    return [
        assess_formal_resume(
            row,
            current_freeze=freeze,
            execution_manifest=manifest,
        )
        for row in rows
    ]


def _failed_attempt_root(run_root: Path) -> Path:
    root = run_root.parent / ".uniform_batch4_failed_attempts" / run_root.name
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    candidate = root / f"{stamp}_{os.getpid()}"
    sequence = 1
    while candidate.exists():
        candidate = root / f"{stamp}_{os.getpid()}_{sequence}"
        sequence += 1
    return candidate


def archive_incomplete_formal_run(run_root: Path) -> Path:
    """Preserve a failed attempt before reusing its fixed run-id directory."""

    if not run_root.is_dir():
        raise FileNotFoundError(f"Incomplete run directory disappeared: {run_root}")
    archive_root = _failed_attempt_root(run_root)
    workspace_root = PROJECT_ROOT.resolve()
    source_root = run_root.resolve()
    target_root = archive_root.resolve()
    if (
        workspace_root not in source_root.parents
        or workspace_root not in target_root.parents
        or source_root == target_root
    ):
        raise ValueError("Refusing to archive a run outside the project workspace")
    archive_root.parent.mkdir(parents=True, exist_ok=True)
    os.replace(run_root, archive_root)
    write_json(
        archive_root / "resume_archive.json",
        {
            "status": "ARCHIVED_INCOMPLETE_FORMAL_ATTEMPT",
            "original_run_dir": str(run_root),
            "archived_at": utc_now(),
        },
    )
    return archive_root


def _record_formal_failure_artifact(
    row: Mapping[str, Any], *, exit_code: int, log_path: Path
) -> Path:
    """Ensure every non-zero worker exit leaves immutable failure evidence."""

    run_root = PROJECT_ROOT / str(row["output_root"]) / str(row["new_run_id"])
    target_root = run_root if run_root.is_dir() else _failed_attempt_root(run_root)
    target_root.mkdir(parents=True, exist_ok=True)
    status_path = target_root / "run_status.json"
    if not status_path.is_file():
        write_json(
            status_path,
            {
                "status": "FAILED",
                "run_mode": "formal",
                "artifact_profile": "FAILED",
                "started_at": None,
                "finished_at": utc_now(),
                "exit_code": exit_code,
                "failure_stage": "formal_runner",
                "error_type": "FormalWorkerExit",
                "error_message": f"Formal worker exited with code {exit_code}",
            },
        )
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    artifact = target_root / f"formal_runner_failure_{stamp}_{os.getpid()}.json"
    write_json(
        artifact,
        {
            "status": "FAILED",
            "model_id": row["model_id"],
            "run_id": row["new_run_id"],
            "exit_code": exit_code,
            "log_path": str(log_path),
            "recorded_at": utc_now(),
        },
    )
    return artifact


def partial_original_execution_metadata() -> dict[str, Any]:
    return {
        "suite_status": ORIGINAL_SKIP_MSGNET_SUITE_STATUS,
        "completed_expected": ORIGINAL_SKIP_MSGNET_EXPECTED,
        "completed_count": 0,
        "skipped_models": [ORIGINAL_SKIP_MSGNET_MODEL],
        "skip_reason": ORIGINAL_SKIP_MSGNET_REASON,
        "full_original_28_complete": False,
        "e4_aggregation_allowed": False,
        "original_28_readiness_allowed": False,
    }


def _execution_path() -> Path:
    return DOC_ROOT / (
        f"UNIFORM_BATCH4_MACHINE_EXECUTION_{machine_identity()['machine_id']}.json"
    )


def _load_execution_manifest() -> dict[str, Any]:
    path = _execution_path()
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "schema_version": "uniform_batch4_machine_execution_v2",
        "machine_identity": machine_identity(),
        "freeze_identity": current_freeze_identity(),
        "executions": [],
    }


def _tee_process(command: Sequence[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command),
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        return int(process.wait())


def _formal_log_path(suite: str, model: str) -> Path:
    root = PROJECT_ROOT / "custom_models/logs/uniform_bs4/formal" / suite
    root.mkdir(parents=True, exist_ok=True)
    base = root / f"{model}.log"
    if not base.exists():
        return base
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return root / f"{model}_{stamp}_{os.getpid()}.log"


def _print_resume_plan(plan: Sequence[Mapping[str, Any]], *, dry_run: bool) -> None:
    for item in plan:
        model = str(item["model_id"])
        decision = item["decision"]
        if decision == "SKIP_COMPLETED":
            marker = (
                "SKIP_COMPLETED"
                if dry_run
                else "SKIP_COMPLETED_IDENTITY_MATCHED"
            )
            print(f"{marker}: {model}")
        elif decision == "WILL_RUN":
            print(f"WILL_RUN: {model}")
        else:
            print(f"BLOCKED_IDENTITY_MISMATCH: {model} ({item['reason']})")


def run_formal_suite(
    suite: str, python_executable: str, *, dry_run: bool = False
) -> int:
    gate_suite = "e5" if suite == "e5_core" else suite
    resume_plan: list[dict[str, Any]] = []
    if suite == ORIGINAL_SKIP_MSGNET_SUITE:
        blockers = verify_original_skip_msgnet_suite()
        try:
            partial_rows = original_skip_msgnet_run_rows()
        except ValueError as exc:
            blockers.append(str(exc))
            partial_rows = []
        if not blockers:
            resume_plan = formal_resume_plan(partial_rows)
            blockers.extend(
                f"{item['model_id']}: {item['reason']}"
                for item in resume_plan
                if item["decision"] == "BLOCKED_IDENTITY_MISMATCH"
            )
    else:
        blockers = verify_suite(gate_suite)
        partial_rows = []
    if resume_plan:
        _print_resume_plan(resume_plan, dry_run=dry_run)
    if blockers:
        print("FORMAL RUN BLOCKED:")
        for blocker in blockers:
            print(f"- {blocker}")
        return 4
    if dry_run:
        print("DRY_RUN_PLAN_COMPLETE: no training was started")
        return 0
    execution = {
        "suite": suite,
        "start_time": utc_now(),
        "preflight_summary": str(machine_manifest_paths()[0]),
        "formal_models": [],
        "run_ids": [],
        "output_roots": [],
        "exit_codes": {},
        "status": "RUNNING",
    }
    if suite == ORIGINAL_SKIP_MSGNET_SUITE:
        execution.update(partial_original_execution_metadata())
        execution["resume_plan"] = resume_plan
        execution["skipped_completed"] = [
            item["model_id"]
            for item in resume_plan
            if item["decision"] == "SKIP_COMPLETED"
        ]
        execution["completed_count"] = len(execution["skipped_completed"])
    manifest = _load_execution_manifest()
    manifest["executions"].append(execution)
    write_json(_execution_path(), manifest)
    if suite in {"original", "e5_core", ORIGINAL_SKIP_MSGNET_SUITE}:
        experiment_type = (
            "original_benchmark"
            if suite in {"original", ORIGINAL_SKIP_MSGNET_SUITE}
            else "e5_common_loss"
        )
        rows = (
            partial_rows
            if suite == ORIGINAL_SKIP_MSGNET_SUITE
            else _run_map_rows(experiment_type)
        )
        for row in rows:
            model = row["model_id"]
            plan_item = next(
                (
                    item
                    for item in resume_plan
                    if item["model_id"] == model
                    and item["run_id"] == row["new_run_id"]
                ),
                None,
            )
            if plan_item is not None:
                if plan_item["decision"] == "SKIP_COMPLETED":
                    continue
                run_root = Path(plan_item["run_root"])
                if run_root.is_dir():
                    try:
                        archive_root = archive_incomplete_formal_run(run_root)
                    except OSError as exc:
                        execution["status"] = "FAILED_STOPPED"
                        execution["end_time"] = utc_now()
                        execution["archive_error"] = {
                            "model_id": model,
                            "run_root": str(run_root),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                        write_json(_execution_path(), manifest)
                        print(f"FORMAL RUN BLOCKED: unable to archive {model}: {exc}")
                        return 4
                    execution.setdefault("archived_incomplete_attempts", {})[
                        model
                    ] = str(archive_root)
                    write_json(_execution_path(), manifest)
            command_name = (
                "evaluate-only" if row["mode"] == "EVALUATE_ONLY" else "train"
            )
            command = [
                python_executable,
                str(PROJECT_ROOT / "custom_models/src/benchmark_v2/run_benchmark.py"),
                command_name,
                "--model",
                model,
                "--training-profile",
                PROFILE_ID,
                "--output-root",
                str(PROJECT_ROOT / row["output_root"]),
                "--run-id",
                row["new_run_id"],
                "--device",
                "cuda",
            ]
            if suite == "e5_core":
                command.extend(["--experiment-profile", E5_PROFILE_ID])
            if row["mode"] != "EVALUATE_ONLY":
                command.extend(["--preflight-root", str(PREFLIGHT_ROOT)])
            log_path = _formal_log_path(suite, model)
            start = utc_now()
            code = _tee_process(command, log_path)
            execution["formal_models"].append(model)
            execution["run_ids"].append(row["new_run_id"])
            execution["output_roots"].append(row["output_root"])
            execution["exit_codes"][model] = {
                "exit_code": code,
                "start_time": start,
                "end_time": utc_now(),
                "log_path": str(log_path),
            }
            if suite == ORIGINAL_SKIP_MSGNET_SUITE and code == 0:
                execution["completed_count"] += 1
            write_json(_execution_path(), manifest)
            if code != 0:
                failure_artifact = _record_formal_failure_artifact(
                    row, exit_code=code, log_path=log_path
                )
                execution["status"] = "FAILED_STOPPED"
                execution["end_time"] = utc_now()
                execution["failure_artifact"] = str(failure_artifact)
                write_json(_execution_path(), manifest)
                return code
    else:
        is_a8 = suite == "a8"
        model = "st_mgprompt_a8" if is_a8 else "st_mgprompt_full"
        run_id = (
            "component_ablation_a8_bs4_seed2026"
            if is_a8
            else "full_fixed_dual_keep_msmgdwu_bs4_seed2026"
        )
        command = [
            python_executable,
            str(PROJECT_ROOT / "custom_models/src/st_mgprompt/run_st_mgprompt.py"),
            "--full",
            "--training-profile",
            PROFILE_ID,
            "--run-id",
            run_id,
            "--output-root",
            str(PROJECT_ROOT / "custom_models/results/st_mgprompt_uniform_bs4"),
            "--amp",
        ]
        if is_a8:
            command.extend(["--component-ablation", "A8"])
        else:
            command.append("--canonical-full")
        if platform.system() == "Windows":
            command.append("--windows-safe-mode")
        log_path = _formal_log_path(suite, model)
        start = utc_now()
        code = _tee_process(command, log_path)
        execution["formal_models"].append(model)
        execution["run_ids"].append(run_id)
        execution["output_roots"].append(
            "custom_models/results/st_mgprompt_uniform_bs4"
        )
        execution["exit_codes"][model] = {
            "exit_code": code,
            "start_time": start,
            "end_time": utc_now(),
            "log_path": str(log_path),
        }
        if code != 0:
            execution["status"] = "FAILED_STOPPED"
            execution["end_time"] = utc_now()
            write_json(_execution_path(), manifest)
            return code
    execution["status"] = (
        "COMPLETED_EXPECTED_27"
        if suite == ORIGINAL_SKIP_MSGNET_SUITE
        else "COMPLETED"
    )
    execution["end_time"] = utc_now()
    execution["output_roots"] = sorted(set(execution["output_roots"]))
    write_json(_execution_path(), manifest)
    return 0


def _run_completed(row: Mapping[str, Any]) -> bool:
    run_root = PROJECT_ROOT / row["output_root"] / row["new_run_id"]
    if not run_root.is_dir():
        return False
    for status_path in run_root.rglob("run_status.json"):
        try:
            status = json.loads(status_path.read_text(encoding="utf-8")).get(
                "status"
            )
        except (OSError, json.JSONDecodeError):
            continue
        if status in COMPLETED_STATUSES:
            return True
    for complete_path in run_root.rglob("train_complete.json"):
        try:
            status = json.loads(complete_path.read_text(encoding="utf-8")).get(
                "status"
            )
        except (OSError, json.JSONDecodeError):
            continue
        if status in COMPLETED_STATUSES:
            return True
    return False


def e5_finalize_blockers(
    *, base28_complete: bool, a8_complete: bool
) -> list[str]:
    blockers = []
    if not base28_complete:
        blockers.append("BASE28_NOT_FROZEN")
    if not a8_complete:
        blockers.append("BATCH4_A8_NOT_COMPLETE")
    return blockers


def finalize_e5(python_executable: str) -> int:
    original_rows = _run_map_rows("original_benchmark")
    base28_complete = all(_run_completed(row) for row in original_rows)
    try:
        from benchmark_v2.experiments.e5_common_loss.a8_reference import (
            validate_a8_reference,
        )

        a8_complete = (
            validate_a8_reference(training_profile=PROFILE_ID).get("status")
            == "VALID"
        )
    except Exception:
        a8_complete = False
    blockers = e5_finalize_blockers(
        base28_complete=base28_complete, a8_complete=a8_complete
    )
    if blockers:
        print("E5 FINALIZE BLOCKED:")
        for blocker in blockers:
            print(f"- {blocker}")
        return 5
    e5_root = (
        PROJECT_ROOT
        / "custom_models/results/benchmark_v2_uniform_bs4"
        / "common_loss_architecture_seed2026"
    )
    commands = [
        [
            python_executable,
            str(PROJECT_ROOT / "custom_models/src/benchmark_v2/run_benchmark.py"),
            "e5-reference-a8",
            "--training-profile",
            PROFILE_ID,
            "--output-path",
            str(DOC_ROOT / "E5_BATCH4_A8_REFERENCE.json"),
        ],
        [
            python_executable,
            str(PROJECT_ROOT / "custom_models/src/benchmark_v2/run_benchmark.py"),
            "e5-readiness",
            "--training-profile",
            PROFILE_ID,
            "--output-root",
            str(e5_root),
            "--report-path",
            str(DOC_ROOT / "E5_BATCH4_READINESS.json"),
        ],
        [
            python_executable,
            str(PROJECT_ROOT / "custom_models/src/benchmark_v2/run_benchmark.py"),
            "e5-aggregate",
            "--training-profile",
            PROFILE_ID,
            "--output-root",
            str(e5_root),
            "--require-complete",
        ],
    ]
    for index, command in enumerate(commands, start=1):
        log_path = _formal_log_path("e5_finalize", f"step_{index}")
        code = _tee_process(command, log_path)
        if code != 0:
            return code
    return 0


def precheck() -> int:
    profile = load_training_profile(PROFILE_ID)
    blockers = []
    if profile.profile_hash != PROFILE_HASH:
        blockers.append("PROFILE_HASH_MISMATCH")
    if not git_identity().get("git_commit"):
        blockers.append(
            "GIT_COMMIT_UNAVAILABLE_SET_UNIFORM_BATCH4_GIT_COMMIT"
        )
    protocol_result = _run_capture(
        [
            sys.executable,
            str(PROJECT_ROOT / "custom_models/src/benchmark_v2/run_benchmark.py"),
            "protocol-check",
        ]
    )
    if protocol_result.returncode != 0:
        blockers.append("BENCHMARK_PROTOCOL_CHECK_FAILED")
    identity = machine_identity()
    if identity.get("gpu_name") is None:
        blockers.append("CUDA_GPU_NOT_AVAILABLE")
    manifest = load_machine_manifest()
    manifest["machine_identity"] = identity
    manifest["freeze_identity"] = current_freeze_identity()
    manifest["TARGET_MACHINE_STATUS"] = (
        _machine_status_from_suites(manifest.get("suites", {}))
        if not blockers
        else "BLOCKED_ON_THIS_MACHINE_NON_OOM"
    )
    manifest["precheck"] = {
        "status": "PASS" if not blockers else "FAIL_NON_OOM",
        "blockers": blockers,
        "protocol_stdout": protocol_result.stdout,
        "protocol_stderr": protocol_result.stderr,
        "completed_at": utc_now(),
    }
    manifest["updated_at"] = utc_now()
    save_machine_manifest(manifest)
    if blockers:
        for blocker in blockers:
            print(f"- {blocker}")
        return 2
    print(json.dumps(identity, ensure_ascii=False, indent=2))
    return 0


def compare_manifests(left_path: str, right_path: str) -> int:
    left = json.loads(Path(left_path).read_text(encoding="utf-8"))
    right = json.loads(Path(right_path).read_text(encoding="utf-8"))
    matched, mismatches = freeze_identities_match(
        left["freeze_identity"], right["freeze_identity"]
    )
    print(
        json.dumps(
            {"matched": matched, "mismatches": mismatches},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if matched else 6


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Machine-bound gate for UNIFORM_BATCH4 formal execution."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("precheck")
    inventory = sub.add_parser("preflight-inventory")
    inventory.add_argument("--suite", choices=["original", "e5"], required=True)
    inventory.add_argument("--python", dest="python_executable", required=True)
    st = sub.add_parser("preflight-st")
    st.add_argument("--suite", choices=["full", "a8"], required=True)
    st.add_argument("--python", dest="python_executable", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument(
        "--suite", choices=["original", "e5", "full", "a8"], required=True
    )
    formal = sub.add_parser("formal-exec")
    formal.add_argument(
        "--suite",
        choices=[
            "original",
            ORIGINAL_SKIP_MSGNET_SUITE,
            "e5_core",
            "full",
            "a8",
        ],
        required=True,
    )
    formal.add_argument("--python", dest="python_executable", required=True)
    formal.add_argument(
        "--dry-run",
        "--plan",
        action="store_true",
        help="Read-only formal execution plan; never launches training.",
    )
    finalize = sub.add_parser("finalize-e5")
    finalize.add_argument("--python", dest="python_executable", required=True)
    compare = sub.add_parser("compare-manifests")
    compare.add_argument("left")
    compare.add_argument("right")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "precheck":
        return precheck()
    if args.command == "preflight-inventory":
        return run_inventory_preflight(args.suite, args.python_executable)
    if args.command == "preflight-st":
        return run_st_preflight(args.suite, args.python_executable)
    if args.command == "verify":
        blockers = verify_suite(args.suite)
        if blockers:
            print("FORMAL RUN BLOCKED:")
            for blocker in blockers:
                print(f"- {blocker}")
            return 4
        print(f"{args.suite}: READY_FOR_FORMAL_RUN")
        return 0
    if args.command == "formal-exec":
        return run_formal_suite(
            args.suite,
            args.python_executable,
            dry_run=args.dry_run,
        )
    if args.command == "finalize-e5":
        return finalize_e5(args.python_executable)
    if args.command == "compare-manifests":
        return compare_manifests(args.left, args.right)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
