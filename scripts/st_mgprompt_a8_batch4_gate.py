from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "custom_models/src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from benchmark_v2.artifacts import atomic_write_json
from st_mgprompt.a8_batch4_contract import (
    A8_DEFINITION, A8_MODEL_ID, A8_OUTPUT_ROOT, A8_REFERENCE_ID,
    A8_REFERENCE_RELATIVE_PATH, A8_RUN_ID, A8_RUN_RELATIVE_PATH,
    A8_SCOPE_ID, A8_VARIANT, EXPECTED_CONFIG, FEATURE_ORDER,
    LOSS_ID, PRECISION_POLICY, TRAINING_PROFILE_ID, graph_identity,
    loss_identity, precision_identity, variant_contract,
)


A8_RUN_ROOT = PROJECT_ROOT / A8_RUN_RELATIVE_PATH
LOCK_PATH = PROJECT_ROOT / "custom_models/logs/uniform_bs4/a8.lock.json"
AUDIT_ROOT = PROJECT_ROOT / "custom_models/logs/uniform_bs4/audit/a8"
PREFLIGHT_RESULT_PATH = AUDIT_ROOT / "preflight/latest.json"
PREFLIGHT_OUTPUT_ROOT = AUDIT_ROOT / "preflight/artifacts"
ARCHIVE_DIRECTORY = "_archived_attempts"
QUARANTINE_DIRECTORY = "_quarantine"
REQUIRED_HORIZONS = (3, 6, 10)


class A8GateError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> tuple[Any | None, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except FileNotFoundError:
        return None, f"MISSING:{path.name}"
    except (OSError, ValueError) as exc:
        return None, f"INVALID:{path.name}:{type(exc).__name__}"


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def validate_contract(project_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root or PROJECT_ROOT)
    graph = graph_identity(root)
    reasons = []
    if graph.get("node_count") != 134 or not graph.get("matrix_names"):
        reasons.append("GRAPH_STRUCTURE_INVALID")
    if len(FEATURE_ORDER) != 16:
        reasons.append("FEATURE_COUNT_INVALID")
    if reasons:
        raise A8GateError(f"A8 contract invalid: {reasons}")
    return {
        "status": "PASS", **variant_contract(), "graph": graph,
        "loss": loss_identity(root), "precision_policy": precision_identity(),
        "expected_config": dict(EXPECTED_CONFIG),
    }


def _metric_validation(run_dir: Path) -> tuple[dict[int, dict[str, Any]], list[str]]:
    payloads, reasons = {}, []
    for horizon in REQUIRED_HORIZONS:
        path = run_dir / f"metrics_eval_h{horizon}.json"
        payload, error = _read_json(path)
        if error or not isinstance(payload, dict):
            reasons.append(error or f"INVALID:{path.name}")
            continue
        payloads[horizon] = payload
        if payload.get("horizon") != horizon:
            reasons.append(f"HORIZON_CONFLICT:{horizon}")
        for key in ("MAE", "RMSE", "R2", "Score"):
            if not _finite(payload.get(key)):
                reasons.append(f"NONFINITE:H{horizon}:{key}")
        count = payload.get("valid_target_count", payload.get("ValidCount"))
        if not _finite(count) or float(count) <= 0:
            reasons.append(f"INVALID_COUNT:H{horizon}")
    return payloads, reasons


def _loadable_checkpoint(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        import torch
        torch.load(path, map_location="cpu", weights_only=False)
        return True
    except Exception:
        return False


def inspect_a8_artifact(output_root: str | Path | None = None, project_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root or PROJECT_ROOT)
    run_dir = Path(output_root) if output_root else root / A8_RUN_RELATIVE_PATH
    if not run_dir.is_dir():
        return {
            "status": "RUN_MISSING", "ready": False,
            "scope_id": A8_SCOPE_ID, "model_id": A8_MODEL_ID,
            "run_id": A8_RUN_ID, "batch_size": 4, "lookback": 144,
            "node_count": 134, "feature_count": 16, "horizon": 10,
            "loss_id": LOSS_ID, "precision": PRECISION_POLICY,
            "formal_training": True, "run_dir": str(run_dir),
            "metrics_complete": False, "checkpoint_loadable": False,
            "explicit_config_conflicts": [], "reasons": ["RUN_MISSING"],
        }
    status, status_error = _read_json(run_dir / "run_status.json")
    config, config_error = _read_json(run_dir / "effective_config.json")
    status = status if isinstance(status, dict) else {}
    config = config if isinstance(config, dict) else {}
    reasons = [value for value in (status_error, config_error) if value]
    aliases = {
        "scope_id": ("scope_id",), "model_id": ("model_id",), "run_id": ("run_id",),
        "component_ablation": ("component_ablation",), "definition": ("definition",),
        "training_profile": ("training_profile", "training_batch_profile_id"),
        "train_batch_size": ("train_batch_size",), "val_batch_size": ("val_batch_size",),
        "test_batch_size": ("test_batch_size",), "gradient_accumulation_steps": ("gradient_accumulation_steps",),
        "seed": ("seed",), "lookback": ("lookback",), "max_pred_len": ("max_pred_len",),
        "loss_function": ("loss_function", "loss_id"), "precision_policy": ("precision_policy", "precision"),
        "formal_training": ("formal_training",),
    }
    conflicts = []
    for key, expected in EXPECTED_CONFIG.items():
        values = [config.get(name) for name in aliases[key] if name in config]
        if values and all(value != expected for value in values):
            conflicts.append(key)
    _, metric_reasons = _metric_validation(run_dir)
    reasons.extend(metric_reasons)
    checkpoint_loadable = _loadable_checkpoint(run_dir / "best_checkpoint.pt")
    if not checkpoint_loadable:
        reasons.append("CHECKPOINT_NOT_LOADABLE")
    if status.get("status") != "COMPLETED" or status.get("exit_code") != 0:
        reasons.append("NOT_COMPLETED")
    if conflicts:
        reasons.append("EXPLICIT_CONFIG_CONFLICT")
    ready = not reasons
    return {
        "status": "READY" if ready else str(status.get("status") or "INCOMPLETE"),
        "ready": ready, "scope_id": A8_SCOPE_ID, "model_id": A8_MODEL_ID,
        "run_id": A8_RUN_ID, "batch_size": 4, "lookback": 144,
        "node_count": 134, "feature_count": 16, "horizon": 10,
        "loss_id": LOSS_ID, "precision": PRECISION_POLICY,
        "formal_training": True, "run_dir": str(run_dir),
        "metrics_complete": not metric_reasons,
        "checkpoint_loadable": checkpoint_loadable,
        "explicit_config_conflicts": conflicts, "reasons": reasons,
    }


def build_readiness() -> dict[str, Any]:
    inspected = inspect_a8_artifact()
    return {"status": "READY" if inspected["ready"] else "NOT_READY", "reference_id": A8_REFERENCE_ID, "artifact": inspected}


def build_freeze_plan() -> dict[str, Any]:
    return {"status": "PASS", "schema_version": "a8_explicit_snapshot_v2", **variant_contract(), "expected_config": dict(EXPECTED_CONFIG), "graph": graph_identity(), "loss": loss_identity(), "precision_policy": precision_identity()}


def build_preflight_plan() -> dict[str, Any]:
    return {"status": "PASS", **variant_contract(), "device": "cuda", "forward_pass": "REQUIRED", "backward_pass": "REQUIRED", "finite": "REQUIRED", "output_shape": [4, 134, 10]}


def read_matching_preflight_pass(path: Path = PREFLIGHT_RESULT_PATH) -> dict[str, Any] | None:
    payload, error = _read_json(path)
    if error or not isinstance(payload, dict):
        return None
    expected = {"status": "PASS", "scope_id": A8_SCOPE_ID, "model_id": A8_MODEL_ID, "run_id": A8_RUN_ID, "batch_size": 4, "precision": PRECISION_POLICY, "forward_pass": True, "backward_pass": True, "finite": True, "output_shape": [4, 134, 10]}
    return payload if all(payload.get(key) == value for key, value in expected.items()) else None


def _preflight_command() -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "custom_models/src/st_mgprompt/run_st_mgprompt.py"),
        "--preflight-full-shape", "--component-ablation", A8_VARIANT,
        "--training-profile", TRAINING_PROFILE_ID, "--run-id", A8_RUN_ID,
        "--output-root", str(PREFLIGHT_OUTPUT_ROOT),
        "--model-input-path", str(PROJECT_ROOT / "dataset/sdwpf_model_input_base.parquet"),
        "--eval-target-path", str(PROJECT_ROOT / "dataset/sdwpf_eval_target.parquet"),
        "--device", "cuda", "--no-amp", "--windows-safe-mode",
    ]


def run_preflight(*, report_path: Path | None = None, child_log_root: Path | None = None,
                  runner: Any = subprocess.run, **_: Any) -> tuple[int, dict[str, Any]]:
    validate_contract()
    child_root = Path(child_log_root or AUDIT_ROOT / "preflight").resolve()
    child_root.mkdir(parents=True, exist_ok=True)
    log_path = child_root / "a8_preflight.log"
    started_at = utc_now()
    with log_path.open("w", encoding="utf-8", newline="") as handle:
        completed = runner(
            _preflight_command(), cwd=str(PROJECT_ROOT),
            env={**os.environ, "PYTHONPATH": str(SOURCE_ROOT)},
            stdout=handle, stderr=subprocess.STDOUT, check=False,
        )
    exit_code = int(completed.returncode)
    child_path = PREFLIGHT_OUTPUT_ROOT / A8_RUN_ID / "STMGPrompt_ComponentAblation/full_shape_smoke_report.json"
    child, error = _read_json(child_path)
    child = child if isinstance(child, dict) else {}
    gradients = child.get("gradient_checks") or {}
    passed = (
        exit_code == 0 and error is None and child.get("status") == "passed"
        and child.get("loss_finite") is True and bool(gradients)
        and all(bool(value) for value in gradients.values())
        and child.get("strict_reload_completed") is True
        and child.get("batch_shape", {}).get("train_x", [None])[0] == 4
    )
    payload = {
        "schema_version": "st_mgprompt_a8_batch4_preflight_result_v2",
        "status": "PASS" if passed else "FAILED", "scope_id": A8_SCOPE_ID,
        "model_id": A8_MODEL_ID, "run_id": A8_RUN_ID, "batch_size": 4,
        "precision": PRECISION_POLICY, "device": "cuda",
        "forward_pass": bool(gradients), "backward_pass": bool(gradients),
        "finite": child.get("loss_finite") is True and all(bool(value) for value in gradients.values()),
        "output_shape": [4, 134, 10], "created_at": started_at,
        "finished_at": utc_now(), "exit_code": exit_code,
        "child_log": str(log_path), "child_report": str(child_path),
    }
    atomic_write_json(PREFLIGHT_RESULT_PATH, payload)
    if report_path:
        atomic_write_json(report_path, payload)
    return (0 if passed else 1), payload


def _formal_command(input_path: str | None, target_path: str | None) -> list[str]:
    command = [
        sys.executable, str(PROJECT_ROOT / "custom_models/src/st_mgprompt/run_st_mgprompt.py"),
        "--component-ablation", A8_VARIANT, "--training-profile", TRAINING_PROFILE_ID,
        "--run-id", A8_RUN_ID, "--output-root", str(PROJECT_ROOT / A8_OUTPUT_ROOT),
        "--model-name", "STMGPrompt_ComponentAblation", "--device", "cuda",
        "--no-amp", "--windows-safe-mode",
    ]
    if input_path:
        command.extend(["--model-input-path", input_path])
    if target_path:
        command.extend(["--eval-target-path", target_path])
    return command


def run(*, input_path: str | None = None, target_path: str | None = None,
        log_root: Path | None = None, runner: Any = subprocess.run, **_: Any) -> tuple[int, dict[str, Any]]:
    validate_contract()
    if read_matching_preflight_pass() is None:
        return 74, {"status": "PREFLIGHT_MISSING", "scope_id": A8_SCOPE_ID, "child_started": False}
    plan = build_plan()
    if plan["action"] == "SKIP_COMPLETED":
        return 0, {"status": "SKIP_COMPLETED", "scope_id": A8_SCOPE_ID, "readiness": build_readiness()}
    if plan["action"] == "BLOCK_EXPLICIT_CONFIG_CONFLICT":
        return 74, {"status": "BLOCK_EXPLICIT_CONFIG_CONFLICT", "plan": plan}
    if plan["action"] == "ARCHIVE_AND_RUN":
        archive_or_quarantine(apply=True)
    logs = Path(log_root or AUDIT_ROOT / "formal").resolve()
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / "a8_batch4_formal.log"
    command = _formal_command(input_path, target_path)
    with log_path.open("w", encoding="utf-8", newline="") as handle:
        completed = runner(command, cwd=str(PROJECT_ROOT),
                           env={**os.environ, "PYTHONPATH": str(SOURCE_ROOT)},
                           stdout=handle, stderr=subprocess.STDOUT, check=False)
    exit_code = int(completed.returncode)
    readiness = build_readiness()
    if exit_code == 0 and readiness["status"] == "READY":
        write_reference()
    return (0 if exit_code == 0 and readiness["status"] == "READY" else exit_code or 74), {
        "status": "COMPLETED" if exit_code == 0 and readiness["status"] == "READY" else "FAILED",
        "scope_id": A8_SCOPE_ID, "run_id": A8_RUN_ID, "exit_code": exit_code,
        "log_path": str(log_path), "readiness": readiness,
    }


def build_plan() -> dict[str, Any]:
    inspected = inspect_a8_artifact()
    action = "SKIP_COMPLETED" if inspected["ready"] else "RUN_MISSING" if inspected["status"] == "RUN_MISSING" else "BLOCK_EXPLICIT_CONFIG_CONFLICT" if inspected["explicit_config_conflicts"] else "ARCHIVE_AND_RUN"
    return {"status": "PASS", "scope_id": A8_SCOPE_ID, "action": action, "artifact": inspected}


def build_inventory() -> dict[str, Any]:
    return {"scope_id": A8_SCOPE_ID, "entries": [inspect_a8_artifact()]}


def archive_or_quarantine(*, kind: str = ARCHIVE_DIRECTORY, apply: bool = False, reason: str = "explicit A8 action") -> dict[str, Any]:
    source = A8_RUN_ROOT
    target = source.parent / kind / f"{source.name}__{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    files = [path for path in source.rglob("*") if path.is_file()] if source.is_dir() else []
    result = {"schema_version": "a8_archive_receipt_v2", "scope_id": A8_SCOPE_ID, "model_id": A8_MODEL_ID, "run_id": A8_RUN_ID, "action": kind, "source": str(source), "target": str(target), "status": "PREVIEW", "started_at": utc_now(), "finished_at": None, "exit_code": None, "file_count": len(files), "total_size_bytes": sum(path.stat().st_size for path in files), "message": reason}
    if apply and source.is_dir():
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, target)
        result.update({"status": "COMPLETED", "finished_at": utc_now(), "exit_code": 0})
    return result


def build_reference_payload() -> dict[str, Any]:
    inspected = inspect_a8_artifact()
    return {"schema_version": "e5_a8_reference_v2", "reference_id": A8_REFERENCE_ID, **{key: inspected[key] for key in ("status", "scope_id", "model_id", "run_id", "batch_size", "lookback", "node_count", "feature_count", "horizon", "loss_id", "precision", "formal_training", "run_dir", "metrics_complete", "checkpoint_loadable")}}


def write_reference(path: Path | None = None) -> dict[str, Any]:
    payload = build_reference_payload()
    if payload["status"] != "READY":
        raise A8GateError("A8 reference cannot be published before readiness.")
    target = path or PROJECT_ROOT / A8_REFERENCE_RELATIVE_PATH
    atomic_write_json(target, payload)
    return payload


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0); return pid > 0
    except OSError:
        return False


def _process_start_time(pid: int) -> float | None:
    try:
        import psutil
        return float(psutil.Process(pid).create_time())
    except Exception:
        return None


def lock_status(path: Path = LOCK_PATH) -> dict[str, Any]:
    if not path.is_file(): return {"status": "ABSENT", "path": str(path)}
    payload, error = _read_json(path)
    required = {"scope_id", "hostname", "pid", "process_start_time", "created_at"}
    if error or not isinstance(payload, dict) or not required.issubset(payload): return {"status": "MALFORMED", "path": str(path)}
    if payload["hostname"] != socket.gethostname() or not _pid_alive(int(payload["pid"])): return {"status": "STALE", "path": str(path), "owner": payload}
    actual = _process_start_time(int(payload["pid"]))
    if actual is not None and abs(actual - float(payload["process_start_time"])) >= 1: return {"status": "STALE", "path": str(path), "owner": payload}
    return {"status": "ACTIVE", "path": str(path), "owner": payload}


def clear_stale_lock(path: Path = LOCK_PATH) -> dict[str, Any]:
    status = lock_status(path)
    if status["status"] == "STALE": path.unlink(); return {"status": "CLEARED", "path": str(path)}
    return status


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate-contract", "freeze-plan", "preflight-plan", "plan", "dry-run", "readiness", "inventory", "lock-status", "write-reference"):
        sub.add_parser(name)
    clear = sub.add_parser("clear-stale-lock"); clear.add_argument("--lock-path", default=str(LOCK_PATH))
    quarantine = sub.add_parser("quarantine-existing"); quarantine.add_argument("--apply", action="store_true")
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--report-path", default=str(PREFLIGHT_RESULT_PATH))
    preflight.add_argument("--child-log-root", default=str(AUDIT_ROOT / "preflight"))
    preflight.add_argument("--source-revision")
    formal = sub.add_parser("run")
    formal.add_argument("--input-path"); formal.add_argument("--target-path")
    formal.add_argument("--log-root", default=str(AUDIT_ROOT / "formal"))
    formal.add_argument("--source-revision")
    readiness = sub.choices["readiness"]
    readiness.add_argument("--report-path")
    readiness.add_argument("--reference-path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate-contract": result = validate_contract()
        elif args.command == "freeze-plan": result = build_freeze_plan()
        elif args.command == "preflight-plan": result = build_preflight_plan()
        elif args.command in {"plan", "dry-run"}: result = build_plan()
        elif args.command == "readiness": result = build_readiness()
        elif args.command == "inventory": result = build_inventory()
        elif args.command == "lock-status": result = lock_status()
        elif args.command == "clear-stale-lock": result = clear_stale_lock(Path(args.lock_path))
        elif args.command == "quarantine-existing": result = archive_or_quarantine(kind=QUARANTINE_DIRECTORY, apply=args.apply)
        elif args.command == "write-reference": result = write_reference()
        elif args.command == "preflight":
            code, result = run_preflight(report_path=Path(args.report_path), child_log_root=Path(args.child_log_root))
            print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)); return code
        elif args.command == "run":
            code, result = run(input_path=args.input_path, target_path=args.target_path, log_root=Path(args.log_root))
            print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)); return code
        else: raise A8GateError(f"Unsupported command: {args.command}")
        if args.command == "readiness":
            if args.report_path: atomic_write_json(Path(args.report_path), result)
            if args.reference_path and result["status"] == "READY": write_reference(Path(args.reference_path))
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 4 if args.command == "readiness" and result.get("status") != "READY" else 0
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error_type": type(exc).__name__, "error_message": str(exc)}, ensure_ascii=False)); return 2


if __name__ == "__main__": raise SystemExit(main())
